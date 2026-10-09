#!/usr/bin/env bash
# GSM8K with the checkpoint author's protocol, so the score is comparable to the
# published reference (97.27 %, BF16 band 97.12-97.50):
#   sgl-eval, full 1319 questions, thinking on (model default), T=0.6, top-p 0.95,
#   max_tokens 8192, seed 0. We used 4 threads (= --max-running-requests).
# Takes ~100 min on two Sparks. Then: python gsm8k_token_stats.py OUT/output-rs0.jsonl
#
# usage: run_gsm8k.sh [base_url] [out_dir]     (needs `pip install sgl-eval` in some venv)
set -eu
BASE=${1:-http://127.0.0.1:8000}
OUT=${2:-./gsm8k_$(date +%Y%m%d-%H%M)}
MODEL=${MODEL:-qwen3.8-flash-next}
mkdir -p "$OUT"
sgl-eval ping --base-url "$BASE/v1" --model "$MODEL"
sgl-eval run gsm8k --base-url "$BASE/v1" --model "$MODEL" --num-threads 4 \
  --temperature 0.6 --top-p 0.95 --max-tokens 8192 --seed 0 --out-dir "$OUT" 2>&1 | tee "$OUT/run.log"
