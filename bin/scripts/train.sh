export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py \
  --deepspeed /home/yizhang/Moris/StreamingSpeechLLM/bin/baseline/scripts/zero2.json \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --data_version "nonstreaming" \
  --data_dir /n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_train_with_backchannel \
  --audio_root_a "/n/work6/yizhang/Moris/zoom2025/audios/A_gd" \
  --audio_root_b "/n/work6/yizhang/Moris/zoom2025/audios/B_gd" \
  --output_dir /n/work6/yizhang/Moris/Models/StreamingSpeechLLM_with_pos_v2/ASR_CONV_finetune/qwen3-asr-sft-l3 \
  --batch_size 4 \
  --grad_acc 2 \
  --lr 2e-5 \
  --epochs 4 \
  --save_steps 1000 \
  --report_to tensorboard \
  --resume 1
