export CUDA_VISIBLE_DEVICES=0,1

OUTDIR=./results_eval/qwen3-asr-sft-l3_overlap

torchrun --nproc_per_node=2 eval.py \
  --model_path /ctd/Works/m-wu/Models/StreamingSpeechLLM/ASR_CONV_finetune/qwen3-asr-sft-l3-overlap/checkpoint-142000 \
  --data_dir /ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l3_conv_test_with_backchannel_overlap \
  --audio_root_a "/ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd" \
  --audio_root_b "/ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd" \
  --output_jsonl ${OUTDIR}/eval.jsonl \
  --batch_size 4 \

python merge_nonstreaming_results.py \
  --input_glob "${OUTDIR}/eval.rank*.jsonl" \
  --output_jsonl results_eval/qwen3-asr-sft-l3-overlap/eval.merged.jsonl