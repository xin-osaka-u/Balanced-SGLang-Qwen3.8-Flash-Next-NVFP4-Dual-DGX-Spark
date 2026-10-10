#!/usr/bin/env bash
# Start Qwen3.8-Flash-Next NVFP4 across two DGX Sparks (TP=2) with the patched SGLang.
# Run on rank 0. Starts rank 1 over ssh. No autostart on purpose: see docs/operations.md.
#
#   cp scripts/config.env.example scripts/config.env && $EDITOR scripts/config.env
#   scripts/launch_tp2.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/config.env"
mkdir -p "$LOG_DIR"

# Flags are identical on both ranks except --node-rank / --host / --port.
SGLANG_ARGS=(
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME"
  --quantization modelopt_fp4
  --kv-cache-dtype bfloat16                    # lossless choice: no FP8 KV (docs/decisions.md)
  --page-size 64
  --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64
  --ple-offload-embedding --ple-offload-backend file   # 23.8 GiB PLE table per rank on NVMe, not in UMA
  --mem-fraction-static 0.75
  --max-running-requests 4 --cuda-graph-max-bs-decode 4
  --max-total-tokens 380928                    # freed memory kept as headroom, not a bigger KV pool
  --context-length 200000
  --nnodes 2 --dist-init-addr "$MASTER_ADDR:$MASTER_PORT" --tp-size 2
  --speculative-algorithm NEXTN --speculative-adaptive          # depth adapts per request: 1/3/5/7 steps
  --speculative-num-steps 3                                     # starting depth. Adaptive needs patch 0009;
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4   # without it, drop --speculative-adaptive (fixed 3/1/4)
  --speculative-token-map "$HOT_TOKEN_MAP"
  --trust-remote-code
)

COMMON_ENV="export PATH=/usr/local/cuda/bin:\$PATH PYTHONUNBUFFERED=1 MAX_JOBS=4 \
NCCL_NET_GDR_LEVEL=0 NCCL_LOCAL_REGISTER=0 NCCL_GRAPH_REGISTER=0 NCCL_GIN_ENABLE=0 \
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_TC=106 NCCL_IB_SL=3 \
NCCL_CROSS_NIC=1 NCCL_IB_DISABLE=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300 NCCL_DEBUG=WARN \
SGLANG_QWEN4_PLE_REUSE=1"
# MAX_JOBS=4: the first start after a flashinfer upgrade JIT-compiles kernels while the model
# already occupies ~90 GB of unified memory; 20 parallel nvcc jobs OOM-killed the scheduler.
# SGLANG_QWEN4_PLE_REUSE=1: reuse the verified on-disk PLE table (patches 0006-0008).

echo "stopping old instances..."
pkill -f "sglang.launch_server" 2>/dev/null || true
ssh -o BatchMode=yes "$WORKER_HOST" 'pkill -f "sglang.launch_server" 2>/dev/null || true'
sleep 3
if [ "$DROP_CACHES" = 1 ]; then
  sync && sudo sysctl -q -w vm.drop_caches=3 || true
  ssh "$WORKER_HOST" 'sync && sudo sysctl -q -w vm.drop_caches=3' || true
fi

echo "starting rank 0..."
(
  eval "$COMMON_ENV"
  export NCCL_IB_HCA="$MASTER_IB_HCA" NCCL_SOCKET_IFNAME="$MASTER_IFNAME" \
         GLOO_SOCKET_IFNAME="$MASTER_IFNAME" TP_SOCKET_IFNAME="$MASTER_IFNAME"
  setsid nohup "$PYTHON_BIN" -m sglang.launch_server "${SGLANG_ARGS[@]}" \
    --node-rank 0 --host 0.0.0.0 --port "$API_PORT" \
    > "$LOG_DIR/rank0.log" 2>&1 < /dev/null &
)

echo "starting rank 1 on $WORKER_HOST..."
ARGS_Q=$(printf '%q ' "${SGLANG_ARGS[@]}")
ssh -o BatchMode=yes "$WORKER_HOST" "bash -c '
  mkdir -p $LOG_DIR; $COMMON_ENV
  export NCCL_IB_HCA=\"$WORKER_IB_HCA\" NCCL_SOCKET_IFNAME=\"$WORKER_IFNAME\" \
         GLOO_SOCKET_IFNAME=\"$WORKER_IFNAME\" TP_SOCKET_IFNAME=\"$WORKER_IFNAME\"
  setsid nohup $PYTHON_BIN -m sglang.launch_server $ARGS_Q --node-rank 1 \
    > $LOG_DIR/rank1.log 2>&1 < /dev/null &
'"

echo "launched. Health: curl -s localhost:$API_PORT/health  (first boot ~9 min, reuse boots ~8 min)"
