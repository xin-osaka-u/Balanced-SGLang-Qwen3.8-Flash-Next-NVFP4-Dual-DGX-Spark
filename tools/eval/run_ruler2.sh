#!/usr/bin/env bash
# RULER2 at 32k context, thinking off, T=0. A long-context tripwire for changes that
# touch attention, KV layout or block selection (e.g. the QSA ring in patch 0009).
# 12 subtasks x 8 samples = 96 prompts of ~31k tokens; ~50-60 min at 4-way concurrency.
# One sample is 12.5 % of a subtask, so only differences larger than that mean anything.
#
# Thinking is switched off by nt_proxy.py (listens on :8001). Start it first:
#   python nt_proxy.py &
# sgl-eval's ruler2 data generator calls a bare `python`, so the venv that holds
# sgl-eval must be first on PATH.
#
# usage: run_ruler2.sh [out_dir]      (needs `pip install sgl-eval`)
set -eu
OUT=${1:-./ruler2_$(date +%Y%m%d-%H%M)}
MODEL=${MODEL:-qwen3.8-flash-next}
TOK=${TOKENIZER:?set TOKENIZER=/path/to/model/dir}
mkdir -p "$OUT"
PATH="$(dirname "$(command -v sgl-eval)"):$PATH" sgl-eval run ruler2 \
  --base-url http://127.0.0.1:8001/v1 --model "$MODEL" \
  --ruler2-seq-len 32768 --ruler2-dataset-size 8 --ruler2-tokenizer "$TOK" \
  --num-threads 4 --max-tokens 256 --temperature 0 --out-dir "$OUT" 2>&1 | tee "$OUT/run.log"
