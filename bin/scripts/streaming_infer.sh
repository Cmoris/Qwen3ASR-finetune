# export CUDA_VISIBLE_DEVICES=4,5,6,7

# python example_qwen3_asr_vllm_streaming.py \
#   --model_path /n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_pre/qwen3-asr-sft-l3/checkpoint-12000 \
#   --data_dir /n/work6/yizhang/Moris/zoom2025/pretrain_labels/l3_conv_test_with_backchannel \
#   --audio_root_a "/n/work6/yizhang/Moris/zoom2025/audios/A_all" \
#   --audio_root_b "/n/work6/yizhang/Moris/zoom2025/audios/B_all" \
# #   --output_dir 
export CUDA_VISIBLE_DEVICES=1

python streaming_infer.py \
  --data_dir /n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_test_with_backchannel \
  --model_path /n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_finetune/qwen3-asr-sft-l3/checkpoint-9100 \
  --output_jsonl results/streaming_eval_finetune_results_test_l3_c2.jsonl \
  --audio_root_a "/n/work6/yizhang/Moris/zoom2025/audios/A_gd" \
  --audio_root_b "/n/work6/yizhang/Moris/zoom2025/audios/B_gd" \
  --steps_ms 1000 2000 3000
