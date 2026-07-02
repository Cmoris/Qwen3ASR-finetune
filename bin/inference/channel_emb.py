import torch
import torch.nn as nn

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerCausalLMOutputWithPast, _get_feat_extract_output_lengths


def dual_channel_forward(
    self,
    input_ids=None,
    input_features=None,
    attention_mask=None,
    feature_attention_mask=None,
    audio_feature_lengths=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    rope_deltas=None,
    labels=None,
    use_cache=None,
    cache_position=None,
    audio_channel_ids=None,   # 新增
    **kwargs,
):
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # 1. audio encoder
    if input_features is not None:
        audio_features = self.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)

        # 2. add channel embedding
        if audio_channel_ids is not None:
            if feature_attention_mask is not None:
                feature_lens = torch.sum(feature_attention_mask, dim=1)
            else:
                feature_lens = audio_feature_lengths

            # 注意：这里需要用模型文件里的 _get_feat_extract_output_lengths
            audio_out_lens = _get_feat_extract_output_lengths(feature_lens)
            
            channel_ids_per_audio_token = torch.repeat_interleave(
                audio_channel_ids.to(audio_features.device),
                audio_out_lens.to(audio_features.device),
            )

            ch_embeds = self.audio_channel_embed(
                channel_ids_per_audio_token
            ).to(audio_features.dtype)

            audio_features = audio_features + ch_embeds
            
        # 3. scatter back into LLM inputs_embeds
        audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    else:
        audio_feature_lengths = None

    # 4. position_ids：如果外部没传，就沿用原逻辑
    if attention_mask is not None and position_ids is None:
        if (
            cache_position is None
            or (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
        ):
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = self.get_rope_index(attention_mask)
            rope_deltas = rope_deltas - delta0
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length = input_ids.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.get_text_config().vocab_size,
        )

    return Qwen3ASRThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )

