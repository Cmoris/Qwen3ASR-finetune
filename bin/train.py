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
        "--output_dir",
        type=str,
        default="results",
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
    
    output_dir = Path(data_args.output_dir) / Path(data_args.model_path).parent.stem
    output_path = output_dir / "eval.jsonl"
    breakpoint()
    output_path = rank_output_path(
        output_path,
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
                
    summary = finalize_summary(summary)

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_serializable(summary),
            f,
            ensure_ascii=False,
            indent=2,
        )

if __name__ == "__main__":
    main()
yizhang@ampc07:~/Moris/Qwen3ASR/bin$ cd scripts/
yizhang@ampc07:~/Moris/Qwen3ASR/bin/scripts$ ls
eval.sh  lora.sh  streaming_infer_mm.sh  streaming_infer.sh  train.sh
yizhang@ampc07:~/Moris/Qwen3ASR/bin/scripts$ cd ..
yizhang@ampc07:~/Moris/Qwen3ASR/bin$ ls
constants.py  eval.py    infer_utils.py              __pycache__   scripts                streaming_infer.py  vllm_streaming.py
data          inference  merge_streaming_results.py  results_eval  streaming_infer_mm.py  train.py
yizhang@ampc07:~/Moris/Qwen3ASR/bin$ cat train.py 
import argparse
import os
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
from types import MethodType

import torch
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model

from data.dataset import DualChannelConvDataset
from data.streaming_dataset import DualChannelConvStreamingDataset
from data.collator import DataCollatorForDualChannelQwen3ASRFinetuning

from inference import dual_channel_forward
from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
                       SPEAKER_TOKENS)



def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

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
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control
    
def maybe_enable_lora(model, args_cli):
    if not args_cli.lora_enable:
        return model

    target_modules = [
        x.strip() for x in args_cli.lora_target_modules.split(",") if x.strip()
    ]
    lora_config = LoraConfig(
        r=args_cli.lora_r,
        lora_alpha=args_cli.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args_cli.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--data_dir", type=str, default="train.jsonl")
    p.add_argument("--validate_dir", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")
    p.add_argument("--audio_root_a", type=str, default="/n/work6/yizhang/Moris/zoom2025/audios/A_gd")
    p.add_argument("--audio_root_b", type=str, default="/n/work6/yizhang/Moris/zoom2025/audios/B_gd")
    

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--report_to", type=str, default=None)
    # LoRA / PEFT
    p.add_argument("--lora_enable", type=bool, default=False)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to apply LoRA to.",
    )
    
    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument(
        "--data_version",
        type=str,
        choices=["nonstreaming", "streaming"],
        default="nonstreaming"
    )

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()

def make_dialogue_module(processor,
                        data_args,
                        query) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.data_dir is not None:
        if data_args.data_version == "nonstreaming":
            train_dataset = DualChannelConvDataset(
                annotation_paths=[str(path) for path in Path(data_args.data_dir).glob("*.jsonl")],
                processor=processor,
                audio_root_a=data_args.audio_root_a,
                audio_root_b=data_args.audio_root_b,
                query=query
            )
        elif data_args.data_version == "streaming":
            train_dataset = DualChannelConvStreamingDataset(
                annotation_paths=[str(path) for path in Path(data_args.data_dir).glob("*.jsonl")],
                processor=processor,
                audio_root_a=data_args.audio_root_a,
                audio_root_b=data_args.audio_root_b,
                query=query,
                sample_strategy="event",
                prefix_time_strategy="all"
            )
        else:
            raise ValueError("Invalid data_args.data_version")
    else:
        raise ValueError("data_args.data_path is None")
    
    if data_args.validate_dir is not None:
        if data_args.data_version == "nonstreaming":
            validate_dataset = DualChannelConvDataset(
                annotation_paths=[str(path) for path in Path(data_args.validate_dir).glob("*.jsonl")],
                processor=processor,
                audio_root_a=data_args.audio_root_a,
                audio_root_b=data_args.audio_root_b,
                query=query
            )
        elif data_args.data_version == "streaming":
            validate_dataset = DualChannelConvStreamingDataset(
                annotation_paths=[str(path) for path in Path(data_args.validate_dir).glob("*.jsonl")],
                processor=processor,
                audio_root_a=data_args.audio_root_a,
                audio_root_b=data_args.audio_root_b,
                query=query,
                sample_strategy="event",
                prefix_time_strategy="all"
            )
        else:
            raise ValueError("Invalid data_args.data_version")
    else:
        validate_dataset = None
    
    data_collator = DataCollatorForDualChannelQwen3ASRFinetuning(processor)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=validate_dataset,
                data_collator=data_collator)


def main():
    args_cli = parse_args()

    if not args_cli.data_dir:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    new_tokens = [
        TE_TOKEN,
        TS_TOKEN,
        BC_TOKEN,
        PAUSE_TOKEN,
        SILENCE_TOKEN,
        SPEAKER_TOKENS["A"][0],
        SPEAKER_TOKENS["A"][1],
        SPEAKER_TOKENS["B"][0],
        SPEAKER_TOKENS["B"][1],
    ]
    processor.tokenizer.add_tokens(new_tokens, special_tokens=False)
    
    query = """You are a streaming dialogue transcriber.

        Transcribe the speech from Speaker A and Speaker B.

        Use special tokens to represent dialogue events:

        <ts> : turn switch
        <te> : turn end
        <bc> : backchannel
        <pause> : speaker pause
        <silence> : conversation silence

        Output the transcript in chronological order."""
    
    hidden_size = model.thinker.config.text_config.hidden_size
    model.thinker.audio_channel_embed = torch.nn.Embedding(2, hidden_size).to(
        next(model.thinker.parameters()).device
    )
    model.thinker.forward = MethodType(dual_channel_forward, model.thinker)
    patch_outer_forward(model)
    
    model = maybe_enable_lora(model, args_cli)
    
    model.generation_config = GenerationConfig.from_model_config(model.config)

    data_module = make_dialogue_module(processor, args_cli, query)
    
    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.validate_dir),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=args_cli.report_to,
            deepspeed=args_cli.deepspeed
    )
    
    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        **data_module,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)],
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    trainer.save_model(args_cli.output_dir)
    processor.save_pretrained(args_cli.output_dir)

if __name__ == "__main__":
    main()