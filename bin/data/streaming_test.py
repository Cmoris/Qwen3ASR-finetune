import os
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _to_numpy_audio(x):
    """
    Convert torch.Tensor / np.ndarray / list to float32 numpy mono audio.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    elif isinstance(x, np.ndarray):
        x = x.astype(np.float32)
    else:
        x = np.asarray(x, dtype=np.float32)

    if x.ndim > 1:
        x = x.reshape(-1)

    return x.astype(np.float32)


def save_debug_streaming_sample(
    debug_dir,
    index,
    target_text,
    audio_a,
    audio_b,
    sample_rate=16000,
    conv_id=None,
    cutoff=None,
    audio_start=None,
    audio_end=None,
    prefix_text=None,
    extra_meta=None,
    save_stereo=True,
):
    """
    Save one streaming training sample for debugging.

    Files saved:
      debug_dir/
        sample_xxxxxx/
          A.wav
          B.wav
          AB_stereo.wav
          target.txt
          prefix.txt
          meta.json
    """

    debug_dir = Path(debug_dir)
    name = f"sample_{index:06d}"

    if conv_id is not None:
        name += f"_id-{conv_id}"

    if cutoff is not None:
        name += f"_t-{float(cutoff):.3f}"

    out_dir = debug_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_a = _to_numpy_audio(audio_a)
    audio_b = _to_numpy_audio(audio_b)

    # pad to same length for easier comparison
    max_len = max(len(audio_a), len(audio_b))

    if len(audio_a) < max_len:
        audio_a = np.pad(audio_a, (0, max_len - len(audio_a)))

    if len(audio_b) < max_len:
        audio_b = np.pad(audio_b, (0, max_len - len(audio_b)))

    sf.write(out_dir / "A.wav", audio_a, sample_rate)
    sf.write(out_dir / "B.wav", audio_b, sample_rate)

    if save_stereo:
        stereo = np.stack([audio_a, audio_b], axis=1)
        sf.write(out_dir / "AB_stereo.wav", stereo, sample_rate)

    with open(out_dir / "target.txt", "w", encoding="utf-8") as f:
        f.write(target_text)

    if prefix_text is not None:
        with open(out_dir / "prefix.txt", "w", encoding="utf-8") as f:
            f.write(prefix_text)

    meta = {
        "index": index,
        "conv_id": conv_id,
        "cutoff": cutoff,
        "audio_start": audio_start,
        "audio_end": audio_end,
        "duration_sec": max_len / sample_rate,
        "num_samples": max_len,
        "sample_rate": sample_rate,
        "target_text": target_text,
    }

    if extra_meta is not None:
        meta.update(extra_meta)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return str(out_dir)