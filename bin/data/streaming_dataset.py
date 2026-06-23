import json
import random
from pathlib import Path
from typing import Optional, Any, Literal
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


def make_dummy_audio(num_samples: int, noise_scale: float = 1e-4) -> torch.Tensor:
    """用极小噪声代替静音，避免被 feature extractor 当成 padding 截断"""
    return torch.randn(num_samples) * noise_scale


# ─────────────────────────────────────────────────────────────────────────────
# Incremental / prefix-streaming helpers for Qwen3-ASR style SFT
# ─────────────────────────────────────────────────────────────────────────────

NON_SPEECH_TOKEN = "<non_speech>"

_KIND_TO_EVENT_TOKEN = {
    "ts": TS_TOKEN,
    "te": TE_TOKEN,
    "bc": BC_TOKEN,
    "pause": PAUSE_TOKEN,
    "silence": SILENCE_TOKEN,
}


def _speaker_open(speaker: str) -> str:
    # SPEAKER_TOKENS is expected to be like {"A": ("<speaker_A>", "</speaker_A>"), ...}
    if speaker in SPEAKER_TOKENS:
        return SPEAKER_TOKENS[speaker][0]
    return f"<speaker_{speaker}>"


def _speaker_close(speaker: str) -> str:
    if speaker in SPEAKER_TOKENS:
        return SPEAKER_TOKENS[speaker][1]
    return f"</speaker_{speaker}>"


def get_record_utterances(record: list | dict) -> list[dict]:
    """Support your current record format: [user_turn, assistant_turn]."""
    user_turn = record[0] if isinstance(record, list) else record
    return user_turn["content"][0]["utterances"]


def get_record_conv_id(record: list | dict) -> str:
    user_turn = record[0] if isinstance(record, list) else record
    return str(user_turn["content"][1]["id"])


def get_record_text_stream(record: list | dict) -> list[dict]:
    """Read assistant text_stream from [user_turn, assistant_turn]."""
    if not isinstance(record, list) or len(record) < 2:
        return []
    return record[1]["content"][0].get("text_stream", [])


def get_dialog_time_range(utterances: list[dict], text_stream: Optional[list[dict]] = None) -> tuple[float, float]:
    """Use utterance time as the stable audio range; text_stream may contain zero-length events."""
    times = []
    for u in utterances:
        times.append((float(u["start"]), float(u["end"])))
    if not times and text_stream:
        for x in text_stream:
            times.append((float(x["start"]), float(x["end"])))
    if not times:
        raise ValueError("empty utterances/text_stream; cannot determine audio range")
    return min(s for s, _ in times), max(e for _, e in times)


def _event_suffix_from_utterance(u: dict) -> str:
    """
    从 utterance-level flags 构造事件 token。
    顺序可以按你原来的定义调整。
    """
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


def _speaker_span(speaker: str, text: str) -> str:
    if speaker == "A":
        return f"{SPEAKER_TOKENS['A'][0]}{text}{SPEAKER_TOKENS['A'][1]}"
    elif speaker == "B":
        return f"{SPEAKER_TOKENS['B'][0]}{text}{SPEAKER_TOKENS['B'][1]}"
    else:
        raise ValueError(f"Unknown speaker: {speaker}")


def _partial_text_by_interval(
    text: str,
    utt_start: float,
    utt_end: float,
    interval_start: float,
    interval_end: float,
) -> str:
    """
    从一个 utterance 中截取 [interval_start, interval_end] 对应的部分文本。
    用 utterance 内部均匀时间近似切字符。
    """
    text = text.strip()
    if not text:
        return ""

    # 没有重叠
    if interval_end <= utt_start or interval_start >= utt_end:
        return ""

    duration = max(utt_end - utt_start, 1e-6)
    n = len(text)

    left = max(interval_start, utt_start)
    right = min(interval_end, utt_end)

    left_ratio = (left - utt_start) / duration
    right_ratio = (right - utt_start) / duration

    char_start = int(np.floor(left_ratio * n))
    char_end = int(np.ceil(right_ratio * n))

    char_start = max(0, min(n, char_start))
    char_end = max(0, min(n, char_end))

    if char_end <= char_start:
        return ""

    return text[char_start:char_end]


def build_prefix_target_from_utterances(
    utterances: list[dict],
    t_now: float,
    *,
    window_start: Optional[float] = None,
    allow_partial_utterance: bool = True,
    add_event_only_when_utterance_finished: bool = True,
    keep_complete_sentence: bool = False,
) -> str:
    """
    构造 Qwen3-ASR 风格 prefix target。

    核心特点：
    1. 按 utterance 级别排序，不按 char-level text_stream 排序。
    2. utterance 内保持完整顺序，不会 A/B 字符交错。
    3. overlap 时，以 utterance 为原子单位输出：
       <speaker_A>完整/部分A句子</speaker_A><speaker_B>完整/部分B句子</speaker_B>
    4. 事件 token 只在 utterance 完成后输出，避免半句话后面直接 <ts>/<te>。
    """

    parts = []

    if window_start is None:
        window_start = float("-inf")

    # 关键：按 utterance 起点排序，而不是按 char token end 排序
    utts = sorted(
        utterances,
        key=lambda u: (
            float(u["start"]),
            float(u["end"]),
            u.get("speaker", ""),
        ),
    )

    for u in utts:
        u_start = float(u["start"])
        u_end = float(u["end"])

        # 这个 utterance 还没开始
        if u_start > t_now:
            continue

        if u_end <= window_start:
            continue

        speaker = u["speaker"]
        full_text = u.get("text", "").strip()

        if not full_text:
            continue

        is_finished = u_end <= t_now

        if u_start >= window_start and is_finished:
            text = full_text

        else:
            if keep_complete_sentence:
                # 严格完整句子模式：
                # 只输出完整落在 [window_start, t_now] 内的 utterance
                continue

            if not allow_partial_utterance:
                continue

            text = _partial_text_by_interval(
                full_text,
                utt_start=u_start,
                utt_end=u_end,
                interval_start=window_start,
                interval_end=t_now,
            )

        if not text:
            continue

        parts.append(_speaker_span(speaker, text))
        
        # 事件 token 发生在 utterance 结束处。
        # 只有 utterance end 在当前窗口内，才输出事件。
        event_time_in_window = window_start < u_end <= t_now

        if event_time_in_window:
            if is_finished or not add_event_only_when_utterance_finished:
                parts.append(_event_suffix_from_utterance(u))

    return "".join(parts)


def build_prefix_sample_times(
    text_stream,
    start_time,
    end_time,
    *,
    chunk_secs=DEFAULT_CHUNK_SECS,
    sample_strategy: Literal["chunk", "event"] = "chunk",
    max_prefix_samples: Optional[int] = None,
):
    """
    Build prefix sampling times for incremental Qwen3-ASR-style training.

    sample_strategy:
        "chunk":
            Sample at fixed time intervals:
                start + chunk_secs, start + 2 * chunk_secs, ...

        "event":
            Sample only when output text may change.
            Namely, use token/event end times from text_stream.

    max_prefix_samples:
        If not None, uniformly subsample the candidate times.
    """
    if end_time <= start_time:
        return [end_time]
    
    if sample_strategy == "chunk":
        times = []
        t = start_time + chunk_secs

        while t < end_time:
            times.append(round(t, 3))
            t += chunk_secs

        times.append(round(end_time, 3))

    elif sample_strategy == "event":
        times = sorted({
            round(float(item["end"]), 3)
            for item in text_stream
            if start_time < float(item["end"]) <= end_time
        })

        # 保证最后一个 prefix 一定覆盖完整音频
        if not times or abs(times[-1] - end_time) > 1e-3:
            times.append(round(end_time, 3))

    else:
        raise ValueError(
            f"Unknown sample_strategy={sample_strategy!r}. "
            f"Expected 'chunk' or 'event'."
        )

    # 去重 + 范围过滤
    times = sorted({
        t for t in times
        if start_time < t <= end_time
    })

    if max_prefix_samples is not None and len(times) > max_prefix_samples:
        # 均匀下采样，而不是随机采样，保证覆盖整段对话
        idxs = np.linspace(
            0,
            len(times) - 1,
            max_prefix_samples,
        ).round().astype(int)

        times = [times[i] for i in idxs]

        # 再去重一次，防止 round 后 index 重复
        times = sorted(set(times))

        # 保证最后一个时间点还在
        if abs(times[-1] - end_time) > 1e-3:
            times[-1] = round(end_time, 3)

    return times


def build_prefix_conversation(audio_list, query="", sr=16000):
    """
    Same spirit as build_conversation(), but for SFT prefix mode:
    only system+user are put in prefix_text; assistant target is returned separately.
    """
    MIN_AUDIO_SECS = 0.5
    min_samples = int(MIN_AUDIO_SECS * sr)
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

        if a_info is not None:
            chunk_a, _, _ = read_audio(a_info)
            chunk_a = pad_audio_to_min_len(chunk_a, min_samples=min_samples)
        else:
            length = max(1, int((end_time - start_time) * sr))
            chunk_a = make_dummy_audio(length)

        if b_info is not None:
            chunk_b, _, _ = read_audio(b_info)
            chunk_b = pad_audio_to_min_len(chunk_b, min_samples=min_samples)
        else:
            length = max(1, int((end_time - start_time) * sr))
            chunk_b = make_dummy_audio(length)

        user_content = [
            {"type": "audio", "audio": None},
            {"type": "audio", "audio": None},
        ]

        audio_inputs.append(chunk_a)
        audio_inputs.append(chunk_b)
        conversation.append({"role": "system", "content": query or ""})
        conversation.append({"role": "user", "content": user_content})

    return conversation, audio_inputs

def pad_audio_to_min_len(wav: torch.Tensor, min_samples: int) -> torch.Tensor:
    if wav.numel() >= min_samples:
        return wav
    pad = make_dummy_audio(min_samples - wav.numel()).to(wav.device)
    return torch.cat([wav, pad], dim=0)


class DualChannelConvStreamingDataset(Dataset):
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
        chunk_secs: float = DEFAULT_CHUNK_SECS,
        prefix_time_strategy: Literal["random", "last", "all"] = "random",
        sample_strategy: Literal["chunk", "event"] = "chunk",
        max_prefix_samples_per_record: Optional[int] = None,
        close_unfinished_speaker: bool = False,
        non_speech_token: str = NON_SPEECH_TOKEN,
        max_audio_context_secs: float=30.0
    ):
        super().__init__()

        self.processor      = processor
        self.audio_root_a   = Path(audio_root_a)
        self.audio_root_b   = Path(audio_root_b)
        self.sr             = sample_rate
        self.query          = query or self.QUERY
        self.chunk_secs     = chunk_secs
        self.prefix_time_strategy = prefix_time_strategy
        self.sample_strategy      = sample_strategy
        self.max_prefix_samples_per_record = max_prefix_samples_per_record
        self.close_unfinished_speaker = close_unfinished_speaker
        self.non_speech_token = non_speech_token
        self.max_audio_context_secs = max_audio_context_secs
        # ── special token ids for label masking  ──────────────
        
        (
            self.im_start_id,
            self.assistant_id,
            self.newline_id,
            self.im_end_id,
        ) = processor.tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids

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

        # Optional full expansion: one dataset item = one prefix time.
        # Default random keeps the original seek handle structure and samples one prefix per record per epoch.
        self.prefix_handles: Optional[list[tuple[str, int, float]]] = None
        if self.prefix_time_strategy == "all":
            self.prefix_handles = []
            for path, seek in self.handles:
                rec = self._load_record_from_handle(path, seek)
                utts = get_record_utterances(rec)
                stream = get_record_text_stream(rec)
                t_start, t_end = get_dialog_time_range(utts, stream)
                times = build_prefix_sample_times(
                    stream,
                    t_start,
                    t_end,
                    chunk_secs=self.chunk_secs,
                    sample_strategy=self.sample_strategy,
                    max_prefix_samples=self.max_prefix_samples_per_record,
                )
                
                self.prefix_handles.extend((path, seek, t) for t in times)

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def _load_record_from_handle(self, path: str, seek: int) -> dict:
        if seek == -1:                          # single .json file
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        with open(path, encoding="utf-8") as f:
            f.seek(seek)
            return json.loads(f.readline())

    def load_record(self, index: int) -> dict:
        if self.prefix_handles is not None:
            path, seek, _ = self.prefix_handles[index]
        else:
            path, seek = self.handles[index]
        return self._load_record_from_handle(path, seek)
    
    def _get_audio_window_start(
        self,
        *,
        t_start: float,
        t_now: float,
    ) -> float:
        if t_now <= t_start:
            t_now = t_start + 1.0 / self.sr

        return max(t_start, t_now - self.max_audio_context_secs)

    def _resolve_audio_prefix(
        self,
        ele: dict,
        *,
        t_start: float,
        t_now: float,
    ) -> list[dict]:
        uttr = ele["content"][0]["utterances"]
        conv_id = ele["content"][1]["id"]
        speakers = set(u["speaker"] for u in uttr)

        if t_now <= t_start:
            t_now = t_start + 1.0 / self.sr

        audio_start = self._get_audio_window_start(
            t_start=t_start,
            t_now=t_now,
        )

        audio_dict = {
            "A": {
                "audio": self.audio_root_a / f"{conv_id}_a.wav",
                "audio_start": audio_start,
                "audio_end": t_now,
            } if "A" in speakers else None,
            "B": {
                "audio": self.audio_root_b / f"{conv_id}_b.wav",
                "audio_start": audio_start,
                "audio_end": t_now,
            } if "B" in speakers else None,
            "utterances": uttr,
        }
        return [audio_dict]

    def _select_prefix_time(self, index: int, sample_times: list[float]) -> float:
        """Pick which prefix time this __getitem__ should return."""
        if self.prefix_handles is not None:
            return float(self.prefix_handles[index][2])  # here record is actually dataset index
        if not sample_times:
            raise ValueError("no prefix sample times")
        if self.prefix_time_strategy == "last":
            return float(sample_times[-1])
        # default: dynamic random prefix per epoch; keeps original dataset length.
        return float(random.choice(sample_times))
    
    def getitem(self, index: int) -> dict:
        record = self.load_record(index)
        user_turn = record[0]
        utterances = get_record_utterances(record)
        text_stream = get_record_text_stream(record)
        t_start, t_end = get_dialog_time_range(utterances, text_stream)
        
        sample_times = build_prefix_sample_times(
            text_stream,
            t_start,
            t_end,
            chunk_secs=self.chunk_secs,
            sample_strategy=self.sample_strategy,
            max_prefix_samples=self.max_prefix_samples_per_record,
        )

        t_now = self._select_prefix_time(index=index, sample_times=sample_times)

        utterances = record[0]["content"][0]["utterances"]

        audio_window_start = self._get_audio_window_start(
            t_start=t_start,
            t_now=t_now,
        )

        target_body = build_prefix_target_from_utterances(
            utterances=utterances,
            t_now=t_now,
            window_start=audio_window_start,
            allow_partial_utterance=True,
            keep_complete_sentence=False,
        )

        # If this fixed chunk introduces no new transcript/event, append one action token.
        # This makes the sample explicitly teach "no new speech output now" without accumulating
        # infinite <non_speech><non_speech>... in the transcript history.

        audio_list = self._resolve_audio_prefix(
            user_turn,
            t_start=t_start,
            t_now=t_now,
        )

        conversation, audio_inputs = build_prefix_conversation(
            audio_list,
            query=self.query,
            sr=self.sr,
        )

        target_text = "language Japanese<asr_text>" + target_body
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
            "audio_start": t_start,
            "audio_end": t_now,
            "target_body": target_body,
            "audio_list": audio_list,
        }

    def __len__(self) -> int:
        if self.prefix_handles is not None:
            return len(self.prefix_handles)
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
        NON_SPEECH_TOKEN,
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
    
    ds = DualChannelConvStreamingDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        processor=processor,
        audio_root_a="/ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd",
        audio_root_b="/ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd",
        query=query,
        sample_strategy="event",
        prefix_time_strategy="all",
        max_audio_context_secs=60.0
    )
    print(f"Dataset length: {len(ds)}")
    collator = DataCollatorForDualChannelQwen3ASRFinetuning(processor)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)
    max_size = 0
    for batch in tqdm.tqdm(loader):
        if batch['input_features'].size(2) > max_size:
            max_size = batch['input_features'].size(2)
        if batch['input_features'].size(2) > 3000:
            print(max_size)
            breakpoint()
        logger.info(f"target_texts: {batch['target_texts']}")
        logger.info(f"num input features: {batch['input_features'].size()}")
        logger.info(f"attention mask size: {batch['attention_mask'].size()}")
        logger.info(f"feature attention mask size: {batch['feature_attention_mask'].size()}")
        logger.info(f"input ids size:{batch['input_ids'].size()}")
        logger.info(f"labels size: {batch['labels'].size()}")
        input_ids = batch["input_ids"]
        input_ids = processor.batch_decode(input_ids, skip_special_tokens=False)
        # logging.info(f"input_ids : {input_ids}")
        
        labels = batch['labels'].masked_fill(batch['labels'] == -100, 0)
        labels = processor.batch_decode(labels, skip_special_tokens=False)
        # logging.info(f"labels : {labels}")
    print(max_size)
