from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np

from qwen_asr.inference.utils import (
    SAMPLE_RATE,
    normalize_language_name,
    parse_asr_output,
    validate_language,
)

@dataclass
class DialogueStreamingState:
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int

    chunk_id: int

    buffer_a: np.ndarray
    buffer_b: np.ndarray

    audio_accum_a: np.ndarray
    audio_accum_b: np.ndarray

    prompt_raw: str
    context: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str
    
import re

def strip_repeated_asr_preamble(text: str) -> str:
    """
    清理模型生成中重复出现的:
      language Japanese<asr_text>
      language XXX<asr_text>
      <asr_text>

    目标：返回纯 transcript。
    """
    if text is None:
        return ""

    text = str(text)

    # 去掉 chat/end token
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")

    # 如果出现多个 language XXX<asr_text>，通常表示模型从头重答。
    # 对 streaming 状态而言，更安全的是取最后一次 preamble 后面的内容。
    pattern = r"language\s+[A-Za-z]+<asr_text>"

    matches = list(re.finditer(pattern, text))
    if matches:
        last = matches[-1]
        text = text[last.end():]

    # 兜底：如果还有裸 <asr_text>
    if "<asr_text>" in text:
        text = text.split("<asr_text>")[-1]

    return text.strip()

def _build_dialogue_messages(self, context: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": context or ""},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": None},
                {"type": "audio", "audio": None},
            ],
        },
    ]


def _build_dialogue_text_prompt(
    self,
    context: str,
    force_language: Optional[str],
) -> str:
    msgs = self._build_messages(context=context)

    base = self.processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    if force_language:
        base = base + f"language {force_language}<asr_text>"

    return base

def init_dialogue_streaming_state(
    self,
    context: str = "",
    language: Optional[str] = None,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    chunk_size_sec: float = 2.0,
) -> DialogueStreamingState:
    if self.backend != "vllm":
        raise ValueError("Dialogue streaming ASR is supported only for vLLM backend.")
    if chunk_size_sec is None or float(chunk_size_sec) <= 0:
        raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

    force_language = None
    if language is not None and str(language).strip() != "":
        ln = normalize_language_name(str(language))
        validate_language(ln)
        force_language = ln

    chunk_size_samples = int(round(float(chunk_size_sec) * SAMPLE_RATE))
    chunk_size_samples = max(1, chunk_size_samples)

    prompt_raw = self._build_text_prompt(
        context=context,
        force_language=force_language,
    )

    return DialogueStreamingState(
        unfixed_chunk_num=int(unfixed_chunk_num),
        unfixed_token_num=int(unfixed_token_num),
        chunk_size_sec=float(chunk_size_sec),
        chunk_size_samples=int(chunk_size_samples),
        chunk_id=0,

        buffer_a=np.zeros((0,), dtype=np.float32),
        buffer_b=np.zeros((0,), dtype=np.float32),

        audio_accum_a=np.zeros((0,), dtype=np.float32),
        audio_accum_b=np.zeros((0,), dtype=np.float32),

        prompt_raw=prompt_raw,
        context=context or "",
        force_language=force_language,

        language="",
        text="",
        _raw_decoded="",
    )
    
def _to_float32_pcm_1d(x: np.ndarray) -> np.ndarray:
    if x is None:
        raise ValueError("pcm input must not be None.")

    x = np.asarray(x)

    if x.ndim != 1:
        x = x.reshape(-1)

    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32, copy=False)

    return x


def streaming_transcribe_dialogue(
    self,
    pcm16k_a: np.ndarray,
    pcm16k_b: np.ndarray,
    state: DialogueStreamingState,
) -> DialogueStreamingState:
    if self.backend != "vllm":
        raise ValueError("streaming_transcribe_dialogue() is supported only for vLLM backend.")
    if state is None:
        raise ValueError("state must not be None. Call init_dialogue_streaming_state() first.")

    x_a = _to_float32_pcm_1d(pcm16k_a)
    x_b = _to_float32_pcm_1d(pcm16k_b)

    if x_a.shape[0] != x_b.shape[0]:
        raise ValueError(
            f"pcm16k_a and pcm16k_b must have the same length for synchronized dialogue streaming, "
            f"got {x_a.shape[0]} and {x_b.shape[0]}."
        )

    if x_a.shape[0] > 0:
        state.buffer_a = np.concatenate([state.buffer_a, x_a], axis=0)
        state.buffer_b = np.concatenate([state.buffer_b, x_b], axis=0)

    while (
        state.buffer_a.shape[0] >= state.chunk_size_samples
        and state.buffer_b.shape[0] >= state.chunk_size_samples
    ):
        chunk_a = state.buffer_a[: state.chunk_size_samples]
        chunk_b = state.buffer_b[: state.chunk_size_samples]

        state.buffer_a = state.buffer_a[state.chunk_size_samples :]
        state.buffer_b = state.buffer_b[state.chunk_size_samples :]

        if state.audio_accum_a.shape[0] == 0:
            state.audio_accum_a = chunk_a
            state.audio_accum_b = chunk_b
        else:
            state.audio_accum_a = np.concatenate([state.audio_accum_a, chunk_a], axis=0)
            state.audio_accum_b = np.concatenate([state.audio_accum_b, chunk_b], axis=0)

        # Build prefix with rollback strategy
        prefix = ""
        if state.chunk_id < state.unfixed_chunk_num:
            prefix = ""
        else:
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            k = int(state.unfixed_token_num)
            while True:
                end_idx = max(0, len(cur_ids) - k)
                prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                if '\ufffd' not in prefix:
                    break
                else:
                    if end_idx == 0:
                        prefix = ""
                        break
                    k += 1
                    
        prompt = state.prompt_raw + prefix

        inp = {
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [
                    state.audio_accum_a,
                    state.audio_accum_b,
                ]
            },
        }

        outputs = self.model.generate(
            [inp],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        
        gen_text = outputs[0].outputs[0].text

        state._raw_decoded = prefix + gen_text

        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt
        
        state.chunk_id += 1

    return state

def finish_streaming_transcribe_dialogue(
    self,
    state: DialogueStreamingState,
) -> DialogueStreamingState:
    if self.backend != "vllm":
        raise ValueError("finish_streaming_transcribe_dialogue() is supported only for vLLM backend.")
    if state is None:
        raise ValueError("state must not be None.")

    if state.buffer_a is None or state.buffer_a.shape[0] == 0:
        return state

    len_a = state.buffer_a.shape[0]
    len_b = state.buffer_b.shape[0]
    max_len = max(len_a, len_b)

    tail_a = state.buffer_a
    tail_b = state.buffer_b

    if len_a < max_len:
        tail_a = np.pad(tail_a, (0, max_len - len_a))
    if len_b < max_len:
        tail_b = np.pad(tail_b, (0, max_len - len_b))

    state.buffer_a = np.zeros((0,), dtype=np.float32)
    state.buffer_b = np.zeros((0,), dtype=np.float32)

    if state.audio_accum_a.shape[0] == 0:
        state.audio_accum_a = tail_a
        state.audio_accum_b = tail_b
    else:
        state.audio_accum_a = np.concatenate([state.audio_accum_a, tail_a], axis=0)
        state.audio_accum_b = np.concatenate([state.audio_accum_b, tail_b], axis=0)

    prefix = ""
    if state.chunk_id < state.unfixed_chunk_num:
        prefix = ""
    else:
        cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
        end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
        prefix = self.processor.tokenizer.decode(cur_ids[:end_idx])
        
    prompt = state.prompt_raw + prefix

    inp = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": [
                state.audio_accum_a,
                state.audio_accum_b,
            ]
        },
    }

    outputs = self.model.generate(
        [inp],
        sampling_params=self.sampling_params,
        use_tqdm=False,
    )

    gen_text = outputs[0].outputs[0].text

    state._raw_decoded = prefix + gen_text

    lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    state.language = lang
    state.text = txt
    
    state.chunk_id += 1

    return state