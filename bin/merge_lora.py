# merge_lora_qwen3asr.py
import argparse
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

from transformers import GenerationConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--lora_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()

def sanitize_generation_config(model):
    if not hasattr(model, "generation_config") or model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    gc = model.generation_config

    # greedy / beam search 时这些 sampling 参数应该为空
    if getattr(gc, "do_sample", False) is False:
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
        gc.typical_p = None
        gc.epsilon_cutoff = None
        gc.eta_cutoff = None

    return model

def main():
    args = parse_args()

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 1. 先加载原始 Qwen3-ASR
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="cpu",   # 合并建议先放 CPU，省显存
    )

    model = asr_wrapper.model
    processor = asr_wrapper.processor

    # 2. 尽量从 LoRA checkpoint 加载 tokenizer / processor
    #    因为你训练时 add_tokens 了特殊 token
    try:
        processor = asr_wrapper.processor.__class__.from_pretrained(args.lora_path)
        print(f"[info] Loaded processor from {args.lora_path}")
    except Exception as e:
        print(f"[warn] Failed to load processor from lora_path, use base processor instead: {e}")


    # 4. 加载 LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        args.lora_path,
        is_trainable=False,
    )

    # 5. 合并 LoRA 权重到 base model
    print("[info] Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    # 6. 保存合并后的完整模型
    print(f"[info] Saving merged model to {args.save_path}")
    merged_model = sanitize_generation_config(merged_model)
    merged_model.save_pretrained(
        args.save_path,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    processor.save_pretrained(args.save_path)

    print("[done] merged model saved.")


if __name__ == "__main__":
    main()