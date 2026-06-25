import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

import sys
sys.path.append("../")

from constants import (
    TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
    SPEAKER_TOKENS, STREAMING_CONT,
    DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE, DEFAULT_CONTEXT_LENGTH,
)


def make_dummy_audio(num_samples: int, noise_scale: float = 1e-4) -> torch.Tensor:
    return torch.randn(num_samples) * noise_scale


def read_audio_segment(path, audio_start, audio_end, target_sr=16000):
    wav, orig_sr = sf.read(str(path), dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    audio_duration = len(wav) / orig_sr

    audio_start = max(0.0, float(audio_start))
    audio_end = min(float(audio_end), audio_duration)

    if audio_end <= audio_start:
        audio_end = min(audio_start + 1.0 / orig_sr, audio_duration)

    start_sample = int(round(audio_start * orig_sr))
    end_sample = int(round(audio_end * orig_sr))

    clip = wav[start_sample:end_sample]

    if orig_sr != target_sr:
        clip = librosa.resample(
            clip,
            orig_sr=orig_sr,
            target_sr=target_sr,
        )

    return torch.from_numpy(clip).float()


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
            last = lines[0]

            non_empty = [l for l in lines[1:] if l.strip()]
            if non_empty:
                return non_empty[-1].decode("utf-8")

    return last.decode("utf-8")


def extract_user_parts(record):
    """
    record:
    [
      {"role": "user", ...},
      {"role": "assistant", ...}
    ]
    """
    user_msg = record[0]
    assistant_msg = record[1]

    utterances = user_msg["content"][0]["utterances"]
    conv_id = user_msg["content"][1]["id"]

    text_stream = assistant_msg["content"][0]["text_stream"]

    return conv_id, utterances, text_stream


def get_dialog_time_range(utterances, text_stream=None):
    starts = [u["start"] for u in utterances]
    ends = [u["end"] for u in utterances]

    if text_stream is not None:
        starts += [x["start"] for x in text_stream]
        ends += [x["end"] for x in text_stream]

    return min(starts), max(ends)


def build_cutoff_times(
    t_start: float,
    t_end: float,
    chunk_secs: float,
    include_event_times: bool = True,
    text_stream: Optional[list] = None,
):
    times = []

    cur = t_start + chunk_secs
    while cur < t_end:
        times.append(round(cur, 3))
        cur += chunk_secs

    times.append(round(t_end, 3))

    if include_event_times and text_stream is not None:
        for x in text_stream:
            tok = x["token"]
            kind = x.get("kind", "")
            if tok.startswith("<") or kind in {"ts", "te", "pause", "silence", "bc"}:
                times.append(round(float(x["end"]), 3))

    times = sorted(set(t for t in times if t_start < t <= t_end))
    return times


def build_event_suffix_from_utterance(u):
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

    return suffix


def collect_asr_prefix_for_utterance(
    text_stream,
    utterance,
    cutoff_time: float,
):
    """
    从 text_stream 里取出属于当前 utterance 的、cutoff 之前的 ASR token。
    事件 token 不在这里处理。
    """
    speaker = utterance["speaker"]
    u_start = float(utterance["start"])
    u_end = float(utterance["end"])

    toks = []

    for x in text_stream:
        if x.get("speaker") != speaker:
            continue

        kind = x.get("kind", "asr")
        token = x["token"]

        # 这里只收 ASR / overlap 文本 token，不收事件 token
        if kind not in {"asr", "overlap"}:
            continue
        if token.startswith("<"):
            continue

        xs = float(x["start"])
        xe = float(x["end"])

        # token 属于这个 utterance 的时间范围，并且已经出现在 cutoff 前
        if xs >= u_start and xe <= min(u_end, cutoff_time):
            toks.append(token)

    return "".join(toks)


def build_parallel_utterance_prefix_target(
    utterances,
    text_stream,
    cutoff_time: float,
    add_language_prefix: bool = True,
    include_empty_speaker: bool = False,
):
    """
    A/B 并行输出，但事件 token 放在 speaker tag 外面。

    输出形式:
      language Japanese<asr_text>
      <speaker_A>...</speaker_A><te>
      <speaker_B>...</speaker_B><pause>
      <speaker_B>...</speaker_B><te>
    """

    speaker_pieces = {
        "A": [],
        "B": [],
    }

    # 按 utterance 原本时间排序；但最后输出时仍然 A track / B track 分开
    utts = sorted(
        utterances,
        key=lambda u: (float(u["start"]), float(u["end"]))
    )

    for u in utts:
        speaker = u["speaker"]
        if speaker not in {"A", "B"}:
            continue

        u_start = float(u["start"])
        u_end = float(u["end"])

        # 这个 utterance 还没开始，不输出
        if u_start > cutoff_time:
            continue

        # 已完成 utterance：可以直接用 utterance text，也可以用 text_stream 重建
        if u_end <= cutoff_time:
            text = u.get("text", "").strip()

            # 如果你更信任 forced alignment 后的 text_stream，也可以换成下面这个：
            # text = collect_asr_prefix_for_utterance(text_stream, u, cutoff_time)

            if text:
                spk_tag = "speaker_A" if speaker == "A" else "speaker_B"
                speaker_pieces[speaker].append(
                    f"<{spk_tag}>{text}</{spk_tag}>"
                )

            # 事件 token 放在 speaker tag 外面
            suffix = build_event_suffix_from_utterance(u)
            if suffix:
                speaker_pieces[speaker].append(suffix)

        else:
            # 未完成 utterance：只输出当前 cutoff 前已经出现的 ASR prefix，不输出事件
            text = collect_asr_prefix_for_utterance(
                text_stream=text_stream,
                utterance=u,
                cutoff_time=cutoff_time,
            )

            if text:
                spk_tag = "speaker_A" if speaker == "A" else "speaker_B"
                speaker_pieces[speaker].append(
                    f"<{spk_tag}>{text}</{spk_tag}>"
                )

    pieces = []

    if add_language_prefix:
        pieces.append("language Japanese<asr_text>")

    if include_empty_speaker or speaker_pieces["A"]:
        pieces.extend(speaker_pieces["A"])

    if include_empty_speaker or speaker_pieces["B"]:
        pieces.extend(speaker_pieces["B"])

    return "".join(pieces)


class IncrementalDualChannelConvDataset(Dataset):
    SYSTEM_PROMPT = ""
    QUERY = ""

    def __init__(
        self,
        annotation_paths: list[str],
        processor: Optional[AutoProcessor],
        audio_root_a: str,
        audio_root_b: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        query: Optional[str] = None,
        chunk_secs: float = 1.0,
        min_audio_secs: float = 0.5,
        max_context_secs: Optional[float] = None,
        include_event_times: bool = True,
        target_mode: str = "cumulative",
    ):
        super().__init__()

        assert target_mode in {"cumulative", "delta"}

        self.processor = processor
        self.audio_root_a = Path(audio_root_a)
        self.audio_root_b = Path(audio_root_b)
        self.sr = sample_rate
        self.query = query or self.QUERY

        self.chunk_secs = chunk_secs
        self.min_audio_secs = min_audio_secs
        self.max_context_secs = max_context_secs
        self.include_event_times = include_event_times
        self.target_mode = target_mode

        self.record_handles = []
        for ap in annotation_paths:
            ap = str(ap)

            if ap.endswith(".jsonl"):
                seeks = json.loads(_read_last_line(ap))
                self.record_handles.extend([(ap, sk) for sk in seeks])
            elif ap.endswith(".json"):
                self.record_handles.append((ap, -1))
            else:
                raise ValueError(f"Unsupported annotation format: {ap}")

        # 这里把 record 展开成多个 prefix sample
        self.samples = []
        self._build_samples()

    def load_record_by_handle(self, handle):
        path, seek = handle

        if seek == -1:
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        with open(path, encoding="utf-8") as f:
            f.seek(seek)
            return json.loads(f.readline())

    def _build_samples(self):
        for rec_idx, handle in enumerate(self.record_handles):
            record = self.load_record_by_handle(handle)
            conv_id, utterances, text_stream = extract_user_parts(record)

            t_start, t_end = get_dialog_time_range(utterances, text_stream)

            cutoff_times = build_cutoff_times(
                t_start=t_start,
                t_end=t_end,
                chunk_secs=self.chunk_secs,
                include_event_times=self.include_event_times,
                text_stream=text_stream,
            )

            prev_cutoff = t_start

            for cutoff in cutoff_times:
                if cutoff - t_start < self.min_audio_secs:
                    continue

                self.samples.append({
                    "rec_idx": rec_idx,
                    "cutoff": cutoff,
                    "prev_cutoff": prev_cutoff,
                })

                prev_cutoff = cutoff

    def __len__(self):
        return len(self.samples)

    def _resolve_audio_paths(self, conv_id):
        path_a = self.audio_root_a / f"{conv_id}_a.wav"
        path_b = self.audio_root_b / f"{conv_id}_b.wav"
        return path_a, path_b

    def _build_audio_prefix(self, conv_id, t_start, cutoff):
        path_a, path_b = self._resolve_audio_paths(conv_id)

        if self.max_context_secs is None:
            audio_start = t_start
        else:
            audio_start = max(t_start, cutoff - self.max_context_secs)

        audio_end = cutoff

        chunk_a = read_audio_segment(
            path_a,
            audio_start=audio_start,
            audio_end=audio_end,
            target_sr=self.sr,
        )

        chunk_b = read_audio_segment(
            path_b,
            audio_start=audio_start,
            audio_end=audio_end,
            target_sr=self.sr,
        )

        total_samples = max(len(chunk_a), len(chunk_b))

        if len(chunk_a) < total_samples:
            chunk_a = torch.nn.functional.pad(
                chunk_a,
                (0, total_samples - len(chunk_a)),
            )

        if len(chunk_b) < total_samples:
            chunk_b = torch.nn.functional.pad(
                chunk_b,
                (0, total_samples - len(chunk_b)),
            )

        return chunk_a, chunk_b, audio_start, audio_end

    def _build_conversation_prefix(self):
        conversation = [
            {"role": "system", "content": self.query or ""},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": None},
                    {"type": "audio", "audio": None},
                ],
            },
        ]

        prefix_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prefix_text

    def __getitem__(self, index):
        sample = self.samples[index]

        handle = self.record_handles[sample["rec_idx"]]
        record = self.load_record_by_handle(handle)

        conv_id, utterances, text_stream = extract_user_parts(record)
        t_start, t_end = get_dialog_time_range(utterances, text_stream)

        cutoff = sample["cutoff"]
        prev_cutoff = sample["prev_cutoff"]

        chunk_a, chunk_b, audio_start, audio_end = self._build_audio_prefix(
            conv_id=conv_id,
            t_start=t_start,
            cutoff=cutoff,
        )

        if self.target_mode == "cumulative":
            target_text = build_parallel_utterance_prefix_target(
                utterances=utterances,
                text_stream=text_stream,
                cutoff_time=cutoff,
                add_language_prefix=True,
            )
        else:
            # delta 模式：只输出上一个 cutoff 到当前 cutoff 之间新增的 token
            delta_stream = [
                x for x in text_stream
                if prev_cutoff < float(x["end"]) <= cutoff
            ]

            target_text = build_parallel_utterance_prefix_target(
                utterances=utterances,
                text_stream=delta_stream,
                cutoff_time=cutoff,
                add_language_prefix=True,
            )

        prefix_text = self._build_conversation_prefix()

        return {
            "prompt": self.query,
            "prefix_text": prefix_text,
            "target": target_text,
            "audios": [
                chunk_a.numpy(),
                chunk_b.numpy(),
            ],
            "conv_id": conv_id,
            "cutoff": cutoff,
            "audio_start": audio_start,
            "audio_end": audio_end,
        }
    
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
    
    dir = "/ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l3_conv_train_with_backchannel"

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
    
    ds = IncrementalDualChannelConvDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        processor=processor,
        audio_root_a="/ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd",
        audio_root_b="/ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd",
        query=query,
        chunk_secs=1,
        min_audio_secs=0.5,
    )
    print(f"Dataset length: {len(ds)}")
    collator = DataCollatorForDualChannelQwen3ASRFinetuning(processor)
    loader = DataLoader(ds, batch_size=4, num_workers=16, shuffle=False, collate_fn=collator)
    max_size = 0
    for batch in tqdm.tqdm(loader):
        
        if batch['input_features'].size(2) > max_size:
            max_size = batch['input_features'].size(2)
        if batch['input_features'].size(2) > 3000:
            print(max_size)
            breakpoint()
        logger.info(f"target_texts: {batch['target_texts']}")
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
