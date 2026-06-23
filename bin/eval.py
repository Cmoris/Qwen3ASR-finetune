import os
import json
import argparse
from typing import Tuple
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, DistributedSampler

from data import DualChannelConvDataset, DataCollatorForDualChannelQwen3ASRFinetuning
from infer_utils import speaker_cer, special_token_f1_sequence
from qwen_asr import Qwen3ASRModel

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

def _print_result(title: str, results) -> None:
    print(f"\n===== {title} =====")
    for i, r in enumerate(results):
        print(f"[sample {i}] language={r.language!r}")
        print(f"[sample {i}] text={r.text!r}")
        if r.time_stamps is not None and len(r.time_stamps) > 0:
            head = r.time_stamps[0]
            tail = r.time_stamps[-1]
            print(f"[sample {i}] ts_first: {head.text!r} {head.start_time}->{head.end_time} s")
            print(f"[sample {i}] ts_last : {tail.text!r} {tail.start_time}->{tail.end_time} s")

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
        "--batch_size",
        type=int,
        default=4,
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

    return p.parse_args()

def rank_output_path(base_path: str | Path, shard_id: int, num_shards: int) -> Path:
    base_path = Path(base_path)

    if num_shards <= 1:
        return base_path

    return base_path.with_name(
        f"{base_path.stem}.rank{shard_id}{base_path.suffix}"
    )
    
def compute_metrics(pred_text: str, ref_text: str):
    pred_text = normalize_text_for_eval(pred_text)
    ref_text = normalize_text_for_eval(ref_text)

    cer_res = speaker_cer(pred_text, ref_text)
    f1_res = special_token_f1_sequence(pred_text, ref_text)

    return {
        "speaker_cer": cer_res,
        "special_token_f1_sequence": f1_res,
    }
    
def update_summary(summary, metrics: dict):
    """
    累积 corpus-level 统计。
    这里 speaker CER 用 edits/ref_chars 做 micro 统计。
    special sequence F1 用 tp_lcs/fp/fn 累积。
    """

    summary = {
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

    item = summary
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

def main() -> None:
    data_args = parse_args()
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{rank}"
    ASR_MODEL_PATH = data_args.model_path
    
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        max_inference_batch_size=32,
        max_new_tokens=256,
        device_map=device
    )
    
    query = """You are a streaming dialogue transcriber.

Transcribe the speech from Speaker A and Speaker B.

Use special tokens to represent dialogue events:

<ts> : turn switch
<te> : turn end
<bc> : backchannel
<pause> : speaker pause
<silence> : conversation silence

Output the transcript in chronological order."""
    
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    annotation_paths = [str(path) for path in Path(data_args.data_dir).glob("*.jsonl")]

    dataset = DualChannelConvDataset(
        annotation_paths=annotation_paths,
        processor=processor,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        query=query,
    )

    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    collator = DataCollatorForDualChannelQwen3ASRFinetuning(processor)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=data_args.batch_size,
        sampler=sampler,
        num_workers=16,
        collate_fn=collator,
        pin_memory=True,
    )
    
    output_path = rank_output_path(
        data_args.output_jsonl,
        shard_id=rank,
        num_shards=world_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    
    with torch.no_grad(), output_path.open("w", encoding="utf-8") as fout:
        for batch_idx, batch in enumerate(dataloader):

            meta = {
                "target_texts": batch.pop("target_texts"),
                "prefix_texts": batch.pop("prefix_texts"),
                "audio_path_a": batch.pop("audio_path_a"),
                "audio_path_b": batch.pop("audio_path_b"),
            }
            
            batch = batch["prefix_inputs"]

            batch = batch.to(model.device).to(model.dtype)

            outputs = model.generate(
                **batch,
                max_new_tokens=256,
            )
            
            input_len = batch["input_ids"].size(1)

            if hasattr(outputs, "sequences"):
                sequences = outputs.sequences
            else:
                sequences = outputs

            gen_ids = sequences[:, input_len:]

            pred_texts = processor.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i, pred in enumerate(pred_texts):
                ref_text = normalize_text_for_eval(meta["target_texts"][i])
                pred_text = normalize_text_for_eval(pred)
                
                metrics = compute_metrics(
                    pred_text=pred,
                    ref_text=ref_text,
                )

                update_summary(summary, metrics)
                
                result = {
                    "sample_idx": int(batch_idx*data_args.batch_size + i),
                    "pred_text": pred_text,
                    "pred_raw": pred,
                    "ref_text": ref_text,
                    "audio_path_a": meta["audio_path_a"][i],
                    "audio_path_b": meta["audio_path_b"][i],
                    "metrics": metrics,
                }
                
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

if __name__ == "__main__":
    main()
