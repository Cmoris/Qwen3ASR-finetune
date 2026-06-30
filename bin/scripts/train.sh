export CUDA_VISIBLE_DEVICES=0,1
torchrun --master_port=29500  --nproc_per_node=2 train.py \
  --deepspeed ./scripts/zero2.json \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --data_version "streaming" \
  --data_dir /ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l3_conv_train_with_backchannel \
  --audio_root_a "/ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd" \
  --audio_root_b "/ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd" \
  --output_dir /ctd/Works/m-wu/Models/StreamingSpeechLLM/ASR_CONV_finetune/qwen3-asr-sft-l3-chunk1s-streaming \
  --chunk_secs 1 \
  --batch_size 1 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 4 \
  --save_steps 1000 \
  --report_to tensorboard \
  --resume 1
