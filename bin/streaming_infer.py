import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from types import MethodType

from transformers import AutoProcessor

from qwen_asr import Qwen3ASRModel

from data import DualChannelConvDataset
from infer_utils import speaker_cer, special_token_f1_sequence
from inference import (
        _build_dialogue_messages,
        _build_dialogue_text_prompt,
        init_dialogue_streaming_state,
        streaming_transcribe_dialogue,
        finish_streaming_transcribe_dialogue
    ) 


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Simple resample to 16k if needed."""
    if sr == 16000:
        return wav.astype(np.float32, copy=False)

    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))

    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)

    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)

    return np.interp(x_new, x_old, wav).astype(np.float32)


def normalize_text_for_eval(text: str) -> str:
    """
    根据你的 target 格式做一点清洗。
    例如:
    language Japanese<asr_text><speaker_B>...
    """
    if text is None:
        return ""

    text = str(text).strip()

    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]

    # 防止有些输出带类似前缀
    if "text=" in text and "<speaker_" in text:
        idx = text.find("<speaker_")
        text = text[idx:]

    return text.strip()


def make_json_serializable(obj):
    """
    防止 numpy 类型不能 json.dump。
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_metrics(pred_text: str, ref_text: str):
    pred_text = normalize_text_for_eval(pred_text)
    ref_text = normalize_text_for_eval(ref_text)

    cer_res = speaker_cer(pred_text, ref_text)
    f1_res = special_token_f1_sequence(pred_text, ref_text)

    return {
        "speaker_cer": cer_res,
        "special_token_f1_sequence": f1_res,
    }


def run_streaming_case(
    asr: Qwen3ASRModel,
    wav16k_a: np.ndarray,
    wav16k_b: np.ndarray,
    step_ms: int,
    query: str,
) -> str:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))

    print(f"\n===== streaming step = {step_ms} ms =====")

    state = asr.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=2,
        context=query,
    )
    
    pos = 0
    call_id = 0

    while pos < wav16k_a.shape[0] and pos < wav16k_b.shape[0]:
        seg_a = wav16k_a[pos: pos + step]
        seg_b = wav16k_b[pos: pos + step]
        pos += seg_a.shape[0]
        call_id += 1

        asr.streaming_transcribe(seg_a, seg_b, state)

        print(
            f"[call {call_id:03d}] "
            f"language={state.language!r} "
            f"text={state.text!r}"
        )

    asr.finish_streaming_transcribe(state)

    print(f"[final] language={state.language!r} text={state.text!r}")

    return state.text


def update_summary(summary, step_ms: int, metrics: dict):
    """
    累积 corpus-level 统计。
    这里 speaker CER 用 edits/ref_chars 做 micro 统计。
    special sequence F1 用 tp_lcs/fp/fn 累积。
    """
    step_key = str(step_ms)

    if step_key not in summary:
        summary[step_key] = {
            "num_samples": 0,
            "speaker": {
                "speaker_A": {"edits": 0, "ref_chars": 0},
                "speaker_B": {"edits": 0, "ref_chars": 0},
                "micro_avg": {"edits": 0, "ref_chars": 0},
            },
            "special_sequence": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
            },
        }

    item = summary[step_key]
    item["num_samples"] += 1

    cer = metrics["speaker_cer"]

    for spk in ["speaker_A", "speaker_B", "micro_avg"]:
        if spk in cer:
            item["speaker"][spk]["edits"] += int(cer[spk].get("edits", 0))
            item["speaker"][spk]["ref_chars"] += int(cer[spk].get("ref_chars", 0))

    f1 = metrics["special_token_f1_sequence"]

    # 兼容你之前那个函数的字段名
    tp = f1.get("tp_lcs", f1.get("tp", 0))
    fp = f1.get("fp", 0)
    fn = f1.get("fn", 0)

    item["special_sequence"]["tp"] += int(tp)
    item["special_sequence"]["fp"] += int(fp)
    item["special_sequence"]["fn"] += int(fn)


def finalize_summary(summary):
    """
    把累计的 edits/ref_chars/tp/fp/fn 转成 CER / P / R / F1。
    """
    for step_key, item in summary.items():
        for spk, stat in item["speaker"].items():
            edits = stat["edits"]
            ref_chars = stat["ref_chars"]
            stat["cer"] = edits / max(ref_chars, 1)

        s = item["special_sequence"]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        s["precision"] = precision
        s["recall"] = recall
        s["f1"] = f1

    return summary


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Streaming Evaluation")

    p.add_argument("--data_dir", type=str, default="train.jsonl")
    p.add_argument(
        "--audio_root_a",
        type=str,
        default="/n/work6/yizhang/Moris/zoom2025/audios/A_gd",
    )
    p.add_argument(
        "--audio_root_b",
        type=str,
        default="/n/work6/yizhang/Moris/zoom2025/audios/B_gd",
    )

    p.add_argument(
        "--model_path",
        type=str,
        default="/n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_pre/qwen3-asr-sft-l3/checkpoint-12000",
    )

    p.add_argument(
        "--output_jsonl",
        type=str,
        default="streaming_eval_results.jsonl",
    )

    p.add_argument(
        "--steps_ms",
        type=int,
        nargs="+",
        default=[200, 500, 1000],
    )

    p.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="For debugging. -1 means use all samples.",
    )

    return p.parse_args()


def main() -> None:
    data_args = parse_args()

    ASR_MODEL_PATH = data_args.model_path

    asr = Qwen3ASRModel.LLM(
        model=ASR_MODEL_PATH,
        gpu_memory_utilization=0.8,
        max_new_tokens=256,
    )
    
    asr._build_messages = MethodType(_build_dialogue_messages, asr)
    asr._build_text_prompt = MethodType(_build_dialogue_text_prompt, asr)
    asr.init_streaming_state = MethodType(init_dialogue_streaming_state, asr)
    asr.streaming_transcribe = MethodType(streaming_transcribe_dialogue, asr)
    asr.finish_streaming_transcribe = MethodType(finish_streaming_transcribe_dialogue, asr)

    query = """You are a streaming dialogue transcriber.

Transcribe the speech from Speaker A and Speaker B.

Use special tokens to represent dialogue events:

<ts> : turn switch
<te> : turn end
<bc> : backchannel
<pause> : speaker pause
<silence> : conversation silence

Output the transcript in chronological order."""

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        fix_mistral_regex=True,
    )

    asr.processor = processor

    annotation_paths = [str(path) for path in Path(data_args.data_dir).glob("*.jsonl")]

    dataset = DualChannelConvDataset(
        annotation_paths=annotation_paths,
        processor=processor,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        query=query,
    )

    output_path = Path(data_args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {}

    with output_path.open("w", encoding="utf-8") as fout:
        for sample_idx, data in enumerate(dataset):
            if data_args.max_samples > 0 and sample_idx >= data_args.max_samples:
                break

            print(f"\n\n========== sample {sample_idx} ==========")

            # 这里原来是 np.concat，建议改成 np.concatenate
            wav_a = data["audios"][0]
            wav_b = data["audios"][1]
            wav16k_a = _resample_to_16k(wav_a, 16000)
            wav16k_b = _resample_to_16k(wav_b, 16000)
            # sf.write("./debug.wav", wav16k, samplerate=16000)
            ref_text = normalize_text_for_eval(data["target"])
            
            # 尽量保留一些样本 id 信息，方便之后定位 bad case
            sample_id = data.get("id", sample_idx)
            audio_path_a = data.get("audio_path_a", None)
            audio_path_b = data.get("audio_path_b", None)

            for step_ms in data_args.steps_ms:
                pred_text = run_streaming_case(
                    asr=asr,
                    wav16k_a=wav16k_a,
                    wav16k_b=wav16k_b,
                    step_ms=step_ms,
                    query=query,
                )

                pred_text = normalize_text_for_eval(pred_text)

                metrics = compute_metrics(
                    pred_text=pred_text,
                    ref_text=ref_text,
                )

                update_summary(summary, step_ms, metrics)

                result = {
                    "sample_idx": sample_idx,
                    "sample_id": sample_id,
                    "step_ms": step_ms,
                    "audio_path_a": audio_path_a,
                    "audio_path_b": audio_path_b,
                    "pred_text": pred_text,
                    "ref_text": ref_text,
                    "metrics": metrics,
                }

                result = make_json_serializable(result)

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                spk_micro_cer = metrics["speaker_cer"]["micro_avg"]["cer"]
                event_f1 = metrics["special_token_f1_sequence"]["f1"]

                print(
                    f"[metrics] step={step_ms}ms "
                    f"speaker_micro_CER={spk_micro_cer:.4f} "
                    f"special_seq_F1={event_f1:.4f}"
                )

    summary = finalize_summary(summary)

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_serializable(summary),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved jsonl results to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()