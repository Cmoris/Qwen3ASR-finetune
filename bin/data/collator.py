from dataclasses import dataclass
from typing import Any, Dict, List

import torch

def find_contiguous_spans(mask: torch.Tensor):
    """
    mask: [L] bool
    return: list of (start, end), end exclusive
    """
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return []

    spans = []
    start = idx[0].item()
    prev = idx[0].item()

    for x in idx[1:].tolist():
        if x == prev + 1:
            prev = x
        else:
            spans.append((start, prev + 1))
            start = x
            prev = x

    spans.append((start, prev + 1))
    return spans


def build_dual_channel_position_ids(
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    audio_pad_id: int,
    *,
    audio_start_id: int | None = None,
    audio_end_id: int | None = None,
    num_channels: int = 2,
    include_audio_boundary_tokens: bool = True,
):
    """
    Build shared-time position_ids for dual-channel audio.

    input_ids: [B, L]
    attention_mask: [B, L]
    audio_pad_id: token id of <|audio_pad|>, e.g. 151676

    Returns:
        position_ids: [B, L]
    """

    device = input_ids.device
    B, L = input_ids.shape

    position_ids = torch.zeros_like(input_ids, dtype=torch.long)

    for b in range(B):
        valid = attention_mask[b].bool()
        audio_pad_mask = (input_ids[b] == audio_pad_id) & valid

        pad_spans = find_contiguous_spans(audio_pad_mask)

        # Optional: extend each audio pad span to include audio start/end tokens.
        # For your example:
        #   151669 <|audio_pad|> ... <|audio_pad|> 151670
        # becomes one complete audio block.
        audio_blocks = []
        for s, e in pad_spans:
            block_s, block_e = s, e

            if include_audio_boundary_tokens:
                if audio_start_id is not None and s - 1 >= 0:
                    if input_ids[b, s - 1].item() == audio_start_id:
                        block_s = s - 1

                if audio_end_id is not None and e < L:
                    if input_ids[b, e].item() == audio_end_id:
                        block_e = e + 1

            audio_blocks.append((block_s, block_e))

        cur_pos = 0
        ptr = 0
        i = 0

        while i < len(audio_blocks):
            group = audio_blocks[i : i + num_channels]
            group_start = min(s for s, _ in group)
            group_end = max(e for _, e in group)

            # 1. Normal text before this audio group
            for t in range(ptr, group_start):
                if valid[t]:
                    position_ids[b, t] = cur_pos
                    cur_pos += 1
                else:
                    position_ids[b, t] = 1

            # 2. Shared-time positions for A/B audio blocks
            max_block_len = max(e - s for s, e in group)

            for s, e in group:
                block_len = e - s
                position_ids[b, s:e] = (
                    cur_pos + torch.arange(block_len, device=device)
                )

            # Audio group consumes time only once, not num_channels times
            cur_pos += max_block_len

            ptr = group_end
            i += num_channels

        # 3. Text after the last audio group
        for t in range(ptr, L):
            if valid[t]:
                position_ids[b, t] = cur_pos
                cur_pos += 1
            else:
                position_ids[b, t] = 1

    return position_ids

@dataclass
class DataCollatorForDualChannelQwen3ASRFinetuning:
    processor: Any
    eos_text: str = "<|im_end|>"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        audio_lists = [f["audio_list"] for f in features]

        # 每条样本可以有多个 audio，batch 内展开
        audios = []
        for f in features:
            audios.extend(f["audios"])

        # 和官方逻辑一致：full = prefix + target + eos
        full_texts = [
            pfx + tgt + self.eos_text
            for pfx, tgt in zip(prefix_texts, targets)
        ]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        
        
        # input_ids = full_inputs['input_ids']
        # seq_length = input_ids.size(1)
        # audio_pad_id = self.processor.tokenizer.encode("<|audio_pad|>")[0]
        # breakpoint()
        # audio_pad_id_idx = torch.where((input_ids == audio_pad_id)[0])[0]
        
        # position_ids = torch.arange(seq_length, device=input_ids.device)
        # position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        # position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        audio_pad_id = self.processor.tokenizer.encode("<|audio_pad|>")[0]
        audio_start_id = self.processor.tokenizer.encode("<|audio_start|>")[0]
        audio_end_id = self.processor.tokenizer.encode("<|audio_end|>")[0]

        position_ids = build_dual_channel_position_ids(
            input_ids=full_inputs["input_ids"],
            attention_mask=full_inputs["attention_mask"],
            audio_pad_id=audio_pad_id,
            audio_start_id=audio_start_id,
            audio_end_id=audio_end_id,
            num_channels=2,
            include_audio_boundary_tokens=True,
        )
        
        audio_nums = full_inputs['feature_attention_mask'].size(0)
        audio_channel_ids = torch.tensor([0,1]*(audio_nums//2), dtype=torch.long)
        
        full_inputs["position_ids"] = position_ids
        full_inputs["audio_channel_ids"] = audio_channel_ids
        
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()

        labels = full_inputs["input_ids"].clone()

        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        full_inputs["prefix_inputs"] = prefix_inputs
        full_inputs["prefix_texts"] = prefix_texts
        full_inputs["target_texts"] = targets
        
        full_inputs["audio_path_a"] = [getattr(a[0]["A"], "audio", None) for a in audio_lists]
        full_inputs["audio_path_b"] = [getattr(a[0]["B"], "audio", None) for a in audio_lists]
        
        return full_inputs
    