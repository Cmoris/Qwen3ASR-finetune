import os
import json
import argparse
from pathlib import Path
import numpy as np
from types import MethodType

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GenerationConfig

from data import DualChannelConvDataset, DataCollatorForDualChannelQwen3ASRFinetuning
from data.collator import build_dual_channel_position_ids
from infer_utils import speaker_cer, special_token_f1_sequence
from qwen_asr import Qwen3ASRModel
from inference import dual_channel_forward

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

    for token in [
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
    ]:
        text = text.replace(token, "")

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
    p = argparse.ArgumentParser("Qwen3-ASR Non-streaming Evaluation")

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
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)

    p.add_argument("--use_pos_emb", action='store_true')
    p.add_argument("--use_channel_emb", action='store_true')

    p.add_argument(
        "--model_path",
        type=str,
        default="/n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_pre/qwen3-asr-sft-l3/checkpoint-12000",
    )

    p.add_argument(
        "--output_jsonl",
        type=str,
        default="eval_results.jsonl",
    )

    return p.parse_args()

def expand_annotation_paths(data_dir: str) -> list[str]:
    path = Path(data_dir)
    if path.is_file():
        if path.suffix not in {".jsonl", ".json"}:
            raise ValueError(f"Unsupported annotation file: {path}")
        return [str(path)]
    if path.is_dir():
        paths = sorted(str(p) for p in path.glob("*.jsonl"))
        if not paths:
            raise ValueError(f"No .jsonl files found under: {path}")
        return paths
    raise FileNotFoundError(f"data_dir does not exist: {path}")

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
    
def new_summary_item():
    return {
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

def update_summary(summary: dict, metrics: dict):
    """
    累积 corpus-level 统计。
    这里 speaker CER 用 edits/ref_chars 做 micro 统计。
    special sequence F1 用 tp_lcs/fp/fn 累积。
    """

    item = summary.setdefault("overall", new_summary_item())
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

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_eval_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no `.thinker.forward`.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._eval_forward_patched = True

def move_batch_to_device(batch, device, dtype):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        out[key] = value
    return out

def get_model_device_dtype(model):
    try:
        param = next(model.parameters())
    except StopIteration:
        return torch.device("cpu"), torch.float32
    return param.device, param.dtype

def audio_paths_from_audio_list(audio_list):
    paths_a, paths_b = [], []
    for audio_info in audio_list:
        a_info = audio_info.get("A")
        b_info = audio_info.get("B")
        paths_a.append(str(a_info["audio"]) if a_info is not None else None)
        paths_b.append(str(b_info["audio"]) if b_info is not None else None)
    return paths_a, paths_b


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
    if torch.cuda.is_available():
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "cpu"
    ASR_MODEL_PATH = data_args.model_path
    
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # attn_implementation="flash_attention_2",
        max_inference_batch_size=32,
        max_new_tokens=data_args.max_new_tokens,
        device_map=device
    )
    
    query = """You are a dialogue transcriber.

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
    model.eval()

    if data_args.use_pos_emb or data_args.use_channel_emb:
        model.thinker.forward = MethodType(dual_channel_forward, model.thinker)
        
        if data_args.use_channel_emb and not hasattr(model.thinker, "audio_channel_embed"):
            raise RuntimeError(
                "use_channel_emb=True requires model.thinker.audio_channel_embed. "
                "Load a checkpoint that contains this module or run without --use_channel_emb."
            )
        
    patch_outer_forward(model)

    model.generation_config = GenerationConfig.from_model_config(model.config)

    annotation_paths = expand_annotation_paths(data_args.data_dir)

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
    collator = DataCollatorForDualChannelQwen3ASRFinetuning(processor=processor,
                                                            use_channel_emb=data_args.use_channel_emb,
                                                            use_pos_emb=data_args.use_pos_emb)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=data_args.batch_size,
        sampler=sampler,
        num_workers=data_args.num_workers,
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
                "audio_list": batch.pop("audio_list"),
            }
            
            batch = batch["prefix_inputs"]

            model_device, model_dtype = get_model_device_dtype(model)
            batch = move_batch_to_device(batch, model_device, model_dtype)

            outputs = model.generate(
                **batch,
                max_new_tokens=data_args.max_new_tokens,
            )
            
            input_len = batch["input_ids"].size(1)

            if hasattr(outputs, "sequences"):
                sequences = outputs.sequences
            else:
                sequences = outputs

            gen_ids = sequences[:, input_len:]

            pred_texts = processor.batch_decode(
                gen_ids,
                skip_special_tokens=False,
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
                    "prefix_text": meta["prefix_texts"][i],
                    "metrics": metrics,
                }
                
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    summary = finalize_summary(summary)

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_serializable(summary),
            f,
            ensure_ascii=False,
            indent=2,
        )
        
    print(f"Saved jsonl results to: {output_path}")
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()
