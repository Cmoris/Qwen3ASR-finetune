export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 eval.py \
  --model_path /n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_finetune/qwen3-asr-sft-l3/checkpoint-9100 \
  --data_dir /n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_test_with_backchannel \
  --audio_root_a "/n/work6/yizhang/Moris/zoom2025/audios/A_gd" \
  --audio_root_b "/n/work6/yizhang/Moris/zoom2025/audios/B_gd" \
  --output_jsonl /home/yizhang/Moris/Qwen3ASR/bin/results_eval/eval.jsonl \
  --batch_size 4 \