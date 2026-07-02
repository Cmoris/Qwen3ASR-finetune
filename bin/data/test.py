import os
import random
from pathlib import Path
import soundfile as sf
import numpy as np
import torch


def maybe_save_debug_audio(
    *,
    index: int,
    audio_inputs: list,
    audio_list: list[dict],
    save_dir: str | Path = "debug_audios",
    sample_rate: int = 16000,
    prob: float = 0.005,
    max_save: int = 100,
    enabled: bool = True,
    target_body: str | None = None
):
    """
    在 Dataset.__getitem__ 里随机保存少量音频，用于检查流式窗口是否正确。

    Args:
        index:
            当前 dataset index。
        audio_inputs:
            build_prefix_conversation 返回的 audio_inputs。
            你的代码里顺序是 [A_audio, B_audio]。
        audio_list:
            _resolve_audio_prefix 返回的 audio_list。
            用于读取 audio_start/audio_end/conv_id 等信息。
        save_dir:
            保存目录。
        sample_rate:
            保存 wav 的采样率。
        prob:
            每个样本保存的概率。
        max_save:
            最多保存多少组，避免训练时疯狂写磁盘。
        enabled:
            是否启用。
    """
    if not enabled:
        return

    if random.random() > prob:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 用目录里的文件数控制最多保存多少组
    existing = list(save_dir.glob("*.wav"))
    if len(existing) >= max_save * 2:
        return

    if not audio_list:
        return

    chunk_info = audio_list[0]

    a_info = chunk_info.get("A")
    b_info = chunk_info.get("B")

    # 你的 audio_inputs 顺序是 A, B
    if len(audio_inputs) < 2:
        return

    start = None
    end = None
    conv_id = "unknown"

    if a_info is not None:
        start = a_info.get("audio_start")
        end = a_info.get("audio_end")
        conv_id = Path(str(a_info.get("audio", "unknown"))).stem.replace("_a", "")
    elif b_info is not None:
        start = b_info.get("audio_start")
        end = b_info.get("audio_end")
        conv_id = Path(str(b_info.get("audio", "unknown"))).stem.replace("_b", "")

    if start is None:
        start = -1
    if end is None:
        end = -1

    def to_numpy_audio(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        elif isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        else:
            x = np.asarray(x, dtype=np.float32)

        # soundfile 需要一维或二维 ndarray
        if x.ndim > 1:
            x = x.squeeze()

        return x

    a_audio = to_numpy_audio(audio_inputs[0])
    b_audio = to_numpy_audio(audio_inputs[1])

    prefix = f"idx{index}_conv{conv_id}_{start:.2f}-{end:.2f}"

    sf.write(save_dir / f"{prefix}_A.wav", a_audio, sample_rate)
    sf.write(save_dir / f"{prefix}_B.wav", b_audio, sample_rate)

    if target_body is not None:
        with open(save_dir / f"{prefix}.txt", "w", encoding="utf-8") as f:
            f.write(target_body)