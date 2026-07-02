import argparse
import numpy as np
import soundfile as sf
from pathlib import Path

from transformers import AutoProcessor

from qwen_asr import Qwen3ASRModel

from data import DualChannelConvDataset
from infer_utils import speaker_cer, special_token_f1_sequence


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Simple resample to 16k if needed (uses linear interpolation; good enough for a test)."""
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


def run_streaming_case(asr: Qwen3ASRModel, wav16k: np.ndarray, step_ms: int, query: str) -> None:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))

    print(f"\n===== streaming step = {step_ms} ms =====")
    
    state = asr.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=2.0,
        context=query
    )

    pos = 0
    call_id = 0
    while pos < wav16k.shape[0]:
        seg = wav16k[pos : pos + step]
        pos += seg.shape[0]
        call_id += 1
        asr.streaming_transcribe(seg, state)
        print(f"[call {call_id:03d}] language={state.language!r} text={state.text!r}")

    asr.finish_streaming_transcribe(state)
    print(f"[final] language={state.language!r} text={state.text!r}")

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--data_dir", type=str, default="train.jsonl")
    p.add_argument("--audio_root_a", type=str, default="/n/work6/yizhang/Moris/zoom2025/audios/A_gd")
    p.add_argument("--audio_root_b", type=str, default="/n/work6/yizhang/Moris/zoom2025/audios/B_gd")
    
    p.add_argument("--model_path", type=str, default="/n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_pre/qwen3-asr-sft-l3/checkpoint-12000")

    return p.parse_args()
    
def main() -> None:
    data_args = parse_args()
    # Streaming is vLLM-only and no forced aligner supported.
    
    ASR_MODEL_PATH = data_args.model_path
    
    asr = Qwen3ASRModel.LLM(
        model=ASR_MODEL_PATH,
        gpu_memory_utilization=0.8,
        max_new_tokens=32, # set a small value for streaming
    )
    
    query = """You are a streaming dialogue transcriber.

        Transcribe the speech from Speaker A and Speaker B.

        Use special tokens to represent dialogue events:

        <ts> : turn switch
        <te> : turn end
        <bc> : backchannel
        <pause> : speaker pause
        <silence> : conversation silence

        Output the transcript in chronological order."""
        
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B", fix_mistral_regex=True)
    asr.processor = processor
    dataset = DualChannelConvDataset(
        annotation_paths=[str(path) for path in Path(data_args.data_dir).glob("*.jsonl")],
        processor=processor,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        query=query
    )
    
    for data in dataset:
        wav = np.concat(data["audios"])
        wav16k = _resample_to_16k(wav, 16000)
        
        target = data["target"]

        for step_ms in [200, 500, 1000]:
            run_streaming_case(asr, wav16k, step_ms, query=query)
            print(f"[target] {target}")


if __name__ == "__main__":
    main()
