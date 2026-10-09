#!/usr/bin/env bash
# One "measurement window" after a (re)start: wait for health, warm up, then
# speed (det_bench) and the CN/JP lossless gate. Compare the result dir against
# results/ and against your own previous window, never against someone else's box.
#
# usage: run_window.sh <tag> [base_url] [out_root]
set -u
TAG=$1; BASE=${2:-http://127.0.0.1:8000}; OUT=${3:-./windows}/$TAG
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python3}
mkdir -p "$OUT"
for i in $(seq 1 150); do
  curl -s -m 3 "$BASE/health" >/dev/null 2>&1 && { echo "healthy after $((i * 10)) s"; break; }
  [ "$i" = 150 ] && { echo "not healthy after 25 min"; exit 2; }
  sleep 10
done
sleep 20
curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"你好"}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null
$PY "$HERE/det_bench.py" "$BASE" 3 2>&1 | tee "$OUT/det_bench.txt"
$PY "$HERE/lossless_test.py" "$BASE" "$OUT" "$TAG" 2>&1 | tee "$OUT/lossless.txt"
