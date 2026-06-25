export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 train.py \
  --deepspeed ./scripts/zero2.json \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --data_version "streaming" \
  --data_dir /ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l10_conv_train_with_backchannel \
  --audio_root_a "/ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd" \
  --audio_root_b "/ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd" \
  --output_dir /ctd/Works/m-wu/Models/StreamingSpeechLLM_with_pos/ASR_CONV_finetune/qwen3-asr-sft-streaming-l10 \
  --batch_size 8 \
  --grad_acc 2 \
  --lr 2e-5 \
  --epochs 4 \
  --save_steps 1000 \
  --report_to tensorboard \
  --resume 1
