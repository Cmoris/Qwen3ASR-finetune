export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 eval.py \
  --model_path /ctd/Works/m-wu/Models/StreamingSpeechLLM_with_pos/ASR_CONV_finetune/qwen3-asr-sft-l3/checkpoint-17000 \
  --data_dir /ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l3_conv_test_with_backchannel \
  --audio_root_a "/ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd" \
  --audio_root_b "/ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd" \
  --use_pos_emb True \
  --use_channel_emb True \
  --output_jsonl ./results_eval/eval.jsonl \
  --batch_size 4 \