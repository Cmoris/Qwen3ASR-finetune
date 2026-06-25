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
        modules_to_save=["audio_channel_embed"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def freeze_audio_tower(model):
    """
    Freeze Qwen3-ASR audio tower / audio encoder parameters.

    This function is intentionally a bit defensive because different
    Qwen3-ASR wrappers may use slightly different attribute names.
    """

    candidate_names = [
        "audio_tower",
    ]

    frozen = []

    # 常见情况：audio module 在 model.thinker 下面
    if hasattr(model.base_model, "thinker"):
        for name in candidate_names:
            if hasattr(model.base_model.thinker, name):
                module = getattr(model.base_model.thinker, name)
                for p in module.parameters():
                    p.requires_grad = False
                frozen.append(f"thinker.{name}")

    if len(frozen) == 0:
        print("[warn] No audio tower module found to freeze.")
        print("[warn] Please check model.named_modules() for the correct audio module name.")
    else:
        print("[info] Frozen audio modules:")
        for name in frozen:
            print(f"  - {name}")

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
    p.add_argument("--freeze_audio_tower", type=bool, default=False)
    
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
    p.add_argument("--use_pos_emb", type=bool, default=False)
    p.add_argument("--use_channel_emb", type=bool, default=False)

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
                prefix_time_strategy="all",
                max_audio_context_secs=60.0
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
                prefix_time_strategy="all",
                max_audio_context_secs=60.0
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
    
    if args_cli.use_channel_emb:
        hidden_size = model.thinker.config.text_config.hidden_size
        model.thinker.audio_channel_embed = torch.nn.Embedding(2, hidden_size).to(
            next(model.thinker.parameters()).device
        )
        
        model.thinker.forward = MethodType(dual_channel_forward, model.thinker)
        
    patch_outer_forward(model)
    
    model = maybe_enable_lora(model, args_cli)
    
    if args_cli.freeze_audio_tower:
        model = freeze_audio_tower(model)
        model.print_trainable_parameters()

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