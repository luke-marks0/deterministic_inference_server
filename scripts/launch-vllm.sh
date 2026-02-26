#!/bin/sh
set -eu

set -- \
  --model "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype "${VLLM_DTYPE:-auto}" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --pipeline-parallel-size "${VLLM_PIPELINE_PARALLEL_SIZE:-1}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
  --attention-backend "${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
  --host "${VLLM_HOST:-0.0.0.0}" \
  --port "${VLLM_CONTAINER_PORT:-8000}"

if [ "${VLLM_TRUST_REMOTE_CODE:-0}" = "1" ]; then
  set -- "$@" --trust-remote-code
fi

if [ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}" = "1" ]; then
  set -- "$@" --enable-auto-tool-choice
fi

if [ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]; then
  set -- "$@" --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
fi

if [ -n "${VLLM_CPU_OFFLOAD_GB:-}" ]; then
  set -- "$@" --cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}"
fi

if [ -n "${VLLM_TOOL_CALL_PARSER:-}" ]; then
  set -- "$@" --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
fi

if [ -n "${VLLM_REASONING_PARSER:-}" ]; then
  set -- "$@" --reasoning-parser "${VLLM_REASONING_PARSER}"
fi

if [ -n "${VLLM_QUANTIZATION:-}" ]; then
  set -- "$@" --quantization "${VLLM_QUANTIZATION}"
fi

exec vllm serve "$@"
