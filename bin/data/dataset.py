import json
import random
from pathlib import Path
from typing import Optional
import numpy as np

import soundfile as sf
import librosa

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, logging
import sys
sys.path.append("../")
from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE, DEFAULT_CONTEXT_LENGTH)

# logger = logging.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_audio(ele: dict):
    path = ele["audio"]

    # 1. 读取音频，不走 torchcodec
    wav, orig_sr = sf.read(path, dtype="float32")

    # stereo / multi-channel -> mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    audio_duration = len(wav) / orig_sr

    audio_start = ele.get("audio_start", None)
    audio_end = ele.get("audio_end", None)

    if audio_start is None:
        audio_start = 0.0
    if audio_end is None:
        audio_end = audio_duration

    # 防止越界
    audio_start = max(0.0, float(audio_start))
    audio_end = min(float(audio_end), audio_duration)

    if audio_end <= audio_start:
        # 避免空音频
        audio_end = min(audio_start + 1.0 / orig_sr, audio_duration)

    # 2. 先在原采样率下裁剪
    start_sample = int(round(audio_start * orig_sr))
    end_sample = int(round(audio_end * orig_sr))
    clip = wav[start_sample:end_sample]

    # 3. 重采样到 DEFAULT_SAMPLE_RATE
    if orig_sr != DEFAULT_SAMPLE_RATE:
        clip = librosa.resample(
            clip,
            orig_sr=orig_sr,
            target_sr=DEFAULT_SAMPLE_RATE,
        )
        audio_sr = DEFAULT_SAMPLE_RATE
    else:
        audio_sr = orig_sr

    # 4. 构造 clip_pts，对应重采样后的每个 sample 的原始时间戳
    nframes = len(clip)
    clip_pts = audio_start + np.arange(nframes) / DEFAULT_SAMPLE_RATE

    clip = torch.from_numpy(clip).float()

    return clip, clip_pts, audio_sr


def safe_chunk(wav: torch.Tensor, start: int, end: int, chunk_samples: int) -> torch.Tensor:
    chunk = wav[start:end]
    if len(chunk) < chunk_samples:
        pad = make_dummy_audio(chunk_samples - len(chunk))  
        chunk = torch.cat([chunk, pad])
    return chunk

def make_dummy_audio(num_samples: int, noise_scale: float = 1e-4) -> torch.Tensor:
    """用极小噪声代替静音，避免被 feature extractor 当成 padding 截断"""
    return torch.randn(num_samples) * noise_scale

def build_conversation(
    audio_list,
    query="",
    sr=16000,
):
    conversation = []
    audio_inputs = []

    for chunk_info in audio_list:
        a_info = chunk_info["A"]
        b_info = chunk_info["B"]

        if a_info is not None:
            start_time = a_info["audio_start"]
            end_time = a_info["audio_end"]
        else:
            start_time = b_info["audio_start"]
            end_time = b_info["audio_end"]
            
        length = int((end_time - start_time) * sr)

        if a_info is not None:
            chunk_a, _, _ = read_audio(a_info)
        else:
            chunk_a = make_dummy_audio(length)

        if b_info is not None:
            chunk_b, _, _ = read_audio(b_info)
        else:
            chunk_b = make_dummy_audio(length)
            
        total_samples = max(len(chunk_a), len(chunk_b))
        if len(chunk_a) < total_samples:
            chunk_a = torch.nn.functional.pad(chunk_a, (0, total_samples - len(chunk_a)))
        if len(chunk_b) < total_samples:
            chunk_b = torch.nn.functional.pad(chunk_b, (0, total_samples - len(chunk_b)))

        cur_uttrs = []

        for u in chunk_info["utterances"]:
            if u["end"] < start_time:
                continue
            if u["start"] > end_time:
                continue

            speaker = u["speaker"]
            text = u["text"].strip()
            if len(text) == 0:
                continue

            suffix = ""
            if u.get("is_turn_taking", False):
                suffix += TS_TOKEN
            if u.get("is_back_channel", False):
                suffix += BC_TOKEN
            if u.get("is_pause", False):
                suffix += PAUSE_TOKEN
            if u.get("is_silence", False):
                suffix += SILENCE_TOKEN
            if u.get("is_turn_ending", False):
                suffix += TE_TOKEN

            spk_tag = "speaker_A" if speaker == "A" else "speaker_B"
            cur_uttrs.append(f"<{spk_tag}>{text}</{spk_tag}>{suffix}")

        
        
        assistant_text = "".join(cur_uttrs)

        user_content = [
            {"type": "audio", "audio": None},
            {"type": "audio", "audio": None},
        ]

        audio_inputs.append(chunk_a)
        audio_inputs.append(chunk_b)
        conversation.append({"role": "system", "content": query or ""})
        conversation.append({"role": "user", "content": user_content})
        conversation.append({"role": "assistant", "content": assistant_text})

    return conversation, audio_inputs



class DualChannelConvDataset(Dataset):
    # SYSTEM_PROMPT = (
    #     "You are a real-time dual-channel meeting transcriber. "
    #     "For each audio chunk you receive, extend the running transcript "
    #     "using [A]/[B] speaker tags with <ts> (turn-start) and <te> (turn-end) tokens."
    # )

    # QUERY = (
    #     "Transcribe the conversation between the two speakers in real time. "
    #     "Use [A] and [B] tags, <ts> and <te> tokens."
    # )
    SYSTEM_PROMPT = ""

    QUERY = ""

    def __init__(
        self,
        annotation_paths: list[str],
        processor: Optional[AutoProcessor],
        audio_root_a: str,
        audio_root_b: str,
        sample_rate:  int   = DEFAULT_SAMPLE_RATE,
        query:        Optional[str] = None,
    ):
        super().__init__()

        self.processor      = processor
        self.audio_root_a   = Path(audio_root_a)
        self.audio_root_b   = Path(audio_root_b)
        self.sr             = sample_rate
        self.query          = query or self.QUERY
        # ── build seek-based handle list (same as LiveCC) ────────────────────
        self.handles: list[tuple[str, int]] = []
        for ap in annotation_paths:
            ap = str(ap)
            if ap.endswith(".jsonl"):
                # last line stores seek indices
                seeks = json.loads(_read_last_line(ap))
                self.handles.extend([(ap, sk) for sk in seeks])
                # logger.warning(f"Loaded {ap} ({len(seeks)} samples)")
            elif ap.endswith(".json"):
                # single-record JSON; seek=0 sentinel handled in load_record
                self.handles.append((ap, -1))
                logger.warning(f"Loaded single-record {ap}")
            else:
                raise ValueError(f"Unsupported annotation format: {ap}")

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def load_record(self, index: int) -> dict:
        path, seek = self.handles[index]
        if seek == -1:                          # single .json file
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        with open(path, encoding="utf-8") as f:
            f.seek(seek)
            return json.loads(f.readline())
        
    def _resolve_audio(self, ele: dict) -> list[dict]:
        uttr = ele['content'][0]["utterances"]
        conv_id = ele['content'][1]["id"]
        audio_list = []
        
        t_A_time = [[u["start"], u["end"]]  for u in uttr if u["speaker"] == 'A']
        t_B_time = [[u["start"], u["end"]]  for u in uttr if u["speaker"] == 'B']
        t_A_time.sort()
        t_B_time.sort()
        
        speakers = set([u["speaker"] for u in uttr])
        
        if 'A' in speakers and 'B' in speakers:
            t_A_start, t_A_end = t_A_time[0][0], t_A_time[-1][-1]
            t_B_start, t_B_end = t_B_time[0][0], t_B_time[-1][-1]
            t_start = min(t_A_start, t_B_start)
            t_end = max(t_A_end, t_B_end)
            audio_dict = {
                "A": {"audio":self.audio_root_a / f"{conv_id}_a.wav", "audio_start": t_start, "audio_end": t_end},
                "B": {"audio":self.audio_root_b / f"{conv_id}_b.wav", "audio_start": t_start, "audio_end": t_end},
                "utterances": uttr,
            }
        elif 'A' in speakers:
            t_A_start, t_A_end = t_A_time[0][0], t_A_time[-1][-1]
            t_start, t_end = t_A_start, t_A_end
            audio_dict = {
                "A": {"audio":self.audio_root_a / f"{conv_id}_a.wav", "audio_start": t_start, "audio_end": t_end},
                "B": None,
                "utterances": uttr,
            }
        else:
            t_B_start, t_B_end = t_B_time[0][0], t_B_time[-1][-1]
            t_start, t_end = t_B_start, t_B_end
            audio_dict = {
                "A": None,
                "B": {"audio":self.audio_root_b / f"{conv_id}_b.wav", "audio_start": t_start, "audio_end": t_end},
                "utterances": uttr,
            }
            
        audio_list.append(audio_dict)
                
        return audio_list

    # ── Core item builder ────────────────────────────────────────────────────

    def getitem(self, index: int) -> dict:
        record = self.load_record(index)
        audio_list = self._resolve_audio(record[0])

        conversation, audio_inputs = build_conversation(
            audio_list,
            query=self.query,
            sr=self.sr,
        )

        # 只保留 user 部分作为 prefix
        target_text = "language Japanese<asr_text>"+conversation[-1]["content"]

        prefix_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return {
            "prompt": self.query,
            "prefix_text": prefix_text,
            "target": target_text,
            "audios": [a.numpy() for a in audio_inputs],
            "audio_list": audio_list
        }

    # ── Public Dataset API ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.handles)

    def __getitem__(self, index: int) -> dict:
        return self.getitem(index)
        # max_tries = 10
        # for _ in range(max_tries):
        #     try:
        #         return self.getitem(index)
        #     except Exception as e:
        #         logger.warning(f"Failed {_}-th try to get item {index}: {e}")
        #         index = random.randint(0, self.__len__() - 1)
        #         logger.warning(f"Retrying to get item {index}")
        # raise Exception(f"Failed to get item after {max_tries} retries")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: read last line efficiently (mirror LiveCC's readlastline)
# ─────────────────────────────────────────────────────────────────────────────

def _read_last_line(path: str, buf: int = 4096) -> str:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        pos, last = size, b""
        while pos > 0:
            read_sz = min(buf, pos)
            pos -= read_sz
            f.seek(pos)
            chunk = f.read(read_sz)
            lines = (chunk + last).split(b"\n")
            last  = lines[0]
            non_empty = [l for l in lines[1:] if l.strip()]
            if non_empty:
                return non_empty[-1].decode("utf-8")
    return last.decode("utf-8")


def record_display(record: dict):
    print("=== Utterances ===")
    for u in record[0]["content"][0]["utterances"]:
        flag = " ← TURN-TAKING" if u["is_turn_taking"] else ""
        print(f"  [{u['speaker']}] {u['start']:.2f}-{u['end']:.2f}  {u['text']}{flag}")

    print("\n=== Stream (first 20 events) ===")
    for ev in record[1]["content"][0]["text_stream"]:
        print(f"  {ev['start']:.3f} {ev['end']:.3f}  [{ev['speaker']}]  {ev['token']:6s}  ({ev['kind']})")
    print("\n=== Training sequence ===")

# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    from collator import DataCollatorForDualChannelQwen3ASRFinetuning
    from torch.utils.data import DataLoader
    import tqdm
    from qwen_asr import Qwen3ASRModel
    
    def setup_logger(log_file: str) -> logging.Logger:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("train")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # 避免重复调用时重复添加 handler
        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 输出到文件
        file_handler = logging.FileHandler(
            log_file,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger
    
    logger = setup_logger("debug.log")
    
    dir = "/ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l10_conv_train_with_backchannel"

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
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
    audio_token_id = processor("<|AUDIO|>").input_ids[0]
    
    query = """You are a streaming dialogue transcriber.

Transcribe the speech from Speaker A and Speaker B.

Use special tokens to represent dialogue events:

<ts> : turn switch
<te> : turn end
<bc> : backchannel
<pause> : speaker pause
<silence> : conversation silence

Output the transcript in chronological order."""
    
    ds = DualChannelConvDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        processor=processor,
        audio_root_a="/ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd",
        audio_root_b="/ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd",
        query=query
    )
    print(f"Dataset length: {len(ds)}")
    collator = DataCollatorForDualChannelQwen3ASRFinetuning(processor)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collator)
    max_size = 0
    for batch in tqdm.tqdm(loader):
        
        if batch['input_features'].size(2) > max_size:
            max_size = batch['input_features'].size(2)
        if batch['input_features'].size(2) > 2000:
            print(max_size)
            breakpoint()
        logger.info(f"num input features: {batch['input_features'].size()}")
        logger.info(f"attention mask size: {batch["attention_mask"].size()}")
        logger.info(f"feature attention mask size: {batch["feature_attention_mask"].size()}")
        logger.info(f"input ids size:{batch["input_ids"].size()}")
        logger.info(f"labels size: {batch['labels'].size()}")
        input_ids = batch["input_ids"]
        input_ids = processor.batch_decode(input_ids, skip_special_tokens=False)
        # logging.info(f"input_ids : {input_ids}")
        
        labels = batch['labels'].masked_fill(batch['labels'] == -100, 0)
        labels = processor.batch_decode(labels, skip_special_tokens=False)
        # logging.info(f"labels : {labels}")
    print(max_size)
