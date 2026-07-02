from qwen_asr import Qwen3ASRModel
from dataset import DualChannelConvDataset
import soundfile as sf
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.append("../")
from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE, DEFAULT_CONTEXT_LENGTH)


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Simple resample to 16k if needed."""
    if sr == 16000:
        return wav.astype(np.float32, copy=False)

    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))

    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)

    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)

    return np.interp(x_new, x_old, wav).astype(np.float32)

query = """You are a streaming dialogue transcriber.

        Transcribe the speech from Speaker A and Speaker B.

        Use special tokens to represent dialogue events:

        <ts> : turn switch
        <te> : turn end
        <bc> : backchannel
        <pause> : speaker pause
        <silence> : conversation silence

        Output the transcript in chronological order."""

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
dataset = DualChannelConvDataset(
    annotation_paths=[str(path) for path in Path("/n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_test_with_backchannel").glob("*.jsonl")],
    processor=processor,
    audio_root_a="/n/work6/yizhang/Moris/zoom2025/audios/A_gd",
    audio_root_b="/n/work6/yizhang/Moris/zoom2025/audios/B_gd",
    query=query
)

data = dataset[84]
wav_a = data["audios"][0]
wav_b = data["audios"][1]
wav16k_a = _resample_to_16k(wav_a, 16000)
wav16k_b = _resample_to_16k(wav_b, 16000)
sf.write("./debug_a.wav", wav16k_a, samplerate=16000)
sf.write("./debug_b.wav", wav16k_b, samplerate=16000)