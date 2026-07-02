#!/usr/bin/env bash
set -e
GPUS=(1 2)
NUM_GPUS=${#GPUS[@]}
OUTDIR=results_no_context
OUTFILE=streaming_eval.jsonl

for RANK in $(seq 0 $((NUM_GPUS - 1))); do
  GPU_ID=${GPUS[$RANK]}

  CUDA_VISIBLE_DEVICES=${GPU_ID} python streaming_infer_mm.py \
    --data_dir /n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_test_with_backchannel \
    --audio_root_a /n/work6/yizhang/Moris/zoom2025/audios/A_gd \
    --audio_root_b /n/work6/yizhang/Moris/zoom2025/audios/B_gd \
    --model_path /n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_CONV_finetune/qwen3-asr-sft-l3/checkpoint-9100 \
    --output_jsonl ${OUTDIR}/${OUTFILE} \
    --steps_ms 200 500 1000 2000 \
    --num_shards ${NUM_GPUS} \
    --shard_id ${RANK} \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.8 \
    > ${OUTDIR}/rank${RANK}.log 2>&1 &
done

wait

python merge_streaming_results.py \
  --input_glob "${OUTDIR}/streaming_eval.rank*.jsonl" \
  --output_jsonl "${OUTDIR}/streaming_eval.merged.jsonl"

rm "${OUTDIR}/streaming_eval.rank*.jsonl"