#!/bin/bash
# Serve the planner model with vLLM behind an OpenAI-compatible API. Any vision-language model that vLLM supports works;
# the framework only needs /v1/chat/completions with image inputs and JSON-schema structured output.
#   MODEL=Qwen/Qwen3.8-27B GPUS=0,1 TP=2 ./scripts/serve_vlm_vllm.sh
#   MODEL=RedHatAI/Qwen3.8-27B-INT4 GPUS=0 TP=1 NAME=Qwen3.8-27B-INT4 ./scripts/serve_vlm_vllm.sh   (one 24 GB GPU)
# Notes from the reference setup (RTX 3090, Ampere): dynamic FP8 quantisation fails (it also quantises the vision
# tower); use a pre-quantised INT4 checkpoint or bf16 with more GPUs. FlashInfer JIT needs a matching nvcc; the
# prebuilt FlashAttention backend and a bf16 KV cache avoid it. MAXIMG bounds images per prompt (2 current + 2k exemplars).
set -u
MODEL=${MODEL:?set MODEL}; NAME=${NAME:-$MODEL}; GPUS=${GPUS:-0}; TP=${TP:-1}; QUANT=${QUANT:-none}; PORT=${PORT:-8100}
MAXLEN=${MAXLEN:-8192}; UTIL=${UTIL:-0.90}; MAXIMG=${MAXIMG:-12}; MAXSEQS=${MAXSEQS:-8}
ATTN=${ATTN:-FLASH_ATTN}; KVDTYPE=${KVDTYPE:-auto}
Q=(); [ "$QUANT" != none ] && Q=(--quantization "$QUANT")
export CUDA_VISIBLE_DEVICES=$GPUS VLLM_LOGGING_LEVEL=INFO VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve "$MODEL" --tensor-parallel-size "$TP" "${Q[@]}" --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" --limit-mm-per-prompt "{\"image\": $MAXIMG}" \
  --mm-processor-kwargs '{"max_pixels": 409600, "min_pixels": 12544}' --max-num-seqs "$MAXSEQS" \
  --attention-backend "$ATTN" --kv-cache-dtype "$KVDTYPE" ${REASONING_PARSER:+--reasoning-parser $REASONING_PARSER} \
  --port "$PORT" --host 127.0.0.1 --served-model-name "$NAME" ${EXTRA:-}
