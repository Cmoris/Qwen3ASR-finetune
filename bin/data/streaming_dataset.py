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
    DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE
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
    utterance_end: Optional[float] = None,
):
    """
    从 text_stream 里取出属于当前 utterance 的、cutoff 之前的 ASR token。
    事件 token 不在这里处理。
    """
    speaker = utterance["speaker"]
    u_start = float(utterance["start"])
    u_end = float(utterance["end"])
    if utterance_end is not None:
        u_end = min(u_end, float(utterance_end))

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
    按 utterance 的起始时间输出，事件 token 放在 speaker tag 外面。

    两个 speaker 发生重叠时，较早开始的 utterance 截止到后开始的
    utterance 的 start。被截断的文本必须从带时间戳的 text_stream 重建，
    这样不会把较早 speaker 的重叠文本输出到较晚 speaker 之后。

    输出形式:
      language Japanese<asr_text>
      <speaker_A>...</speaker_A><te>
      <speaker_B>...</speaker_B><pause>
      <speaker_B>...</speaker_B><te>
    """

    pieces = []
    if add_language_prefix:
        pieces.append("language Japanese<asr_text>")

    # Python 的排序是稳定的；start/end 相同时保留标注中的原始顺序。
    utts = sorted(
        utterances,
        key=lambda u: (float(u["start"]), float(u["end"]))
    )

    for index, u in enumerate(utts):
        speaker = u["speaker"]
        if speaker not in {"A", "B"}:
            continue

        u_start = float(u["start"])
        u_end = float(u["end"])

        # 这个 utterance 还没开始，不输出
        if u_start > cutoff_time:
            continue

        # 如果另一位 speaker 在当前 utterance 结束前开始说话，丢弃当前
        # utterance 从该时刻起的重叠文本。后开始的 utterance 会在后续
        # iteration 中按时间顺序正常输出。
        visible_end = u_end
        for later in utts[index + 1:]:
            later_start = float(later["start"])
            if later_start >= u_end:
                break
            if later.get("speaker") != speaker and later_start > u_start:
                visible_end = later_start
                break

        was_truncated_by_overlap = visible_end < u_end
        utterance_completed = u_end <= cutoff_time

        if utterance_completed and not was_truncated_by_overlap:
            text = u.get("text", "").strip()

        else:
            # 未完成或被重叠截断的 utterance 只能从时间对齐结果重建。
            text = collect_asr_prefix_for_utterance(
                text_stream=text_stream,
                utterance=u,
                cutoff_time=cutoff_time,
                utterance_end=visible_end,
            )

        if text or include_empty_speaker:
            spk_tag = "speaker_A" if speaker == "A" else "speaker_B"
            pieces.append(f"<{spk_tag}>{text}</{spk_tag}>")

        # 事件描述的是原 utterance 的结束状态，只有原 utterance 完整进入
        # 当前 prefix 后才输出。
        if utterance_completed:
            suffix = build_event_suffix_from_utterance(u)
            if suffix:
                pieces.append(suffix)

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


def save_dataset_samples_for_inspection(
    dataset: Dataset,
    output_dir: str | Path,
    num_samples: int = 10,
    start_index: int = 0,
    save_stereo: bool = True,
) -> list[Path]:
    """
    保存若干数据集输出，便于人工检查音频与 target 是否匹配。

    每个样本保存为独立目录，其中包含：
      - ``speaker_A.wav``：A 通道
      - ``speaker_B.wav``：B 通道
      - ``stereo_AB.wav``：可选；左声道 A、右声道 B
      - ``target.txt``：对应的 target text
      - ``metadata.json``：样本索引、会话 ID 和时间范围

    Args:
        dataset: ``IncrementalDualChannelConvDataset`` 或具有相同输出格式的数据集。
        output_dir: 检查结果的保存目录。
        num_samples: 从 ``start_index`` 开始保存的样本数。
        start_index: 第一个待保存的 dataset index。
        save_stereo: 是否额外保存便于双耳监听的双声道音频。

    Returns:
        实际创建的样本目录列表。
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if start_index > len(dataset):
        raise IndexError(
            f"start_index {start_index} exceeds dataset length {len(dataset)}"
        )
    sample_rate = getattr(dataset, "sr", None)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(
            f"dataset.sr must be a positive integer, got {sample_rate!r}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    end_index = min(start_index + num_samples, len(dataset))
    saved_dirs = []

    for index in range(start_index, end_index):
        sample = dataset[index]
        audios = sample.get("audios")
        if not isinstance(audios, (list, tuple)) or len(audios) != 2:
            raise ValueError(
                f"sample {index} must contain two audio channels in 'audios'"
            )

        channels = []
        for speaker, audio in zip(("A", "B"), audios):
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().float().numpy()
            else:
                audio = np.asarray(audio, dtype=np.float32)

            audio = np.squeeze(audio)
            if audio.ndim != 1:
                raise ValueError(
                    f"sample {index} speaker {speaker} audio must be mono, "
                    f"got shape {audio.shape}"
                )
            if audio.size == 0:
                raise ValueError(
                    f"sample {index} speaker {speaker} audio is empty"
                )
            if not np.isfinite(audio).all():
                raise ValueError(
                    f"sample {index} speaker {speaker} audio contains NaN or Inf"
                )
            channels.append(audio.astype(np.float32, copy=False))

        target = sample.get("target")
        if not isinstance(target, str):
            raise TypeError(
                f"sample {index} 'target' must be str, got {type(target).__name__}"
            )

        conv_id = str(sample.get("conv_id", "unknown"))
        safe_conv_id = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in conv_id
        )
        sample_dir = output_dir / (
            f"sample_{index:06d}_{safe_conv_id}_"
            f"{float(sample.get('audio_start', 0.0)):.3f}-"
            f"{float(sample.get('audio_end', 0.0)):.3f}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        max_length = max(len(channels[0]), len(channels[1]))
        padded_channels = [
            np.pad(audio, (0, max_length - len(audio)))
            for audio in channels
        ]

        sf.write(sample_dir / "speaker_A.wav", padded_channels[0], sample_rate)
        sf.write(sample_dir / "speaker_B.wav", padded_channels[1], sample_rate)
        if save_stereo:
            stereo = np.stack(padded_channels, axis=1)
            sf.write(sample_dir / "stereo_AB.wav", stereo, sample_rate)

        (sample_dir / "target.txt").write_text(target, encoding="utf-8")

        metadata = {
            "dataset_index": index,
            "conv_id": conv_id,
            "cutoff": sample.get("cutoff"),
            "audio_start": sample.get("audio_start"),
            "audio_end": sample.get("audio_end"),
            "sample_rate": sample_rate,
            "num_samples": max_length,
            "duration_seconds": max_length / sample_rate,
            "target": target,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        saved_dirs.append(sample_dir)

    return saved_dirs


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

    saved_dirs = save_dataset_samples_for_inspection(
        dataset=ds,
        output_dir="debug_streaming_samples",
        num_samples=10,
        start_index=0,
    )
    print(saved_dirs)

    print(f"Dataset length: {len(ds)}")
    collator = DataCollatorForDualChannelQwen3ASRFinetuning(processor=processor,
                                                            use_channel_emb=False,
                                                            use_pos_emb=False)
    loader = DataLoader(ds, batch_size=1, num_workers=16, shuffle=False, collate_fn=collator)
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
