#!/usr/bin/env bash
# Throughput / TTFT / temperature under different ACTIVE caps, both nodes capped together.
# Run on the master node as the service user (uses sudo locally and over ssh to spark2). Only after maintenance.
#   ./cap_bench.sh "none 2200 1800"
# Each cap runs ~/tune_20261008/run_window.sh cap_<x> (det_bench + lossless + profile) and samples power/temp.
set -u
CAPS=${1:-"none 2200 1800"}
RW=../../tools/bench/run_window.sh
OUT=./results; mkdir -p "$OUT"
setcap() {  # none|MHz
  if [ "$1" = none ]; then sudo nvidia-smi -rgc >/dev/null; ssh spark2 sudo nvidia-smi -rgc >/dev/null
  else sudo nvidia-smi -lgc 300,"$1" >/dev/null; ssh spark2 sudo nvidia-smi -lgc 300,"$1" >/dev/null; fi
}
trap 'setcap none; echo "(both nodes reset to -rgc)"' EXIT
for c in $CAPS; do
  tag=cap_$c; echo "===== cap $c -> $tag"; setcap "$c"
  for h in spark1 spark2; do ssh $h "nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu --format=csv,noheader" | sed "s/^/$h before: /"; done
  ( while true; do for h in spark1 spark2; do echo "$h,$(date +%s),$(ssh $h nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"; done; sleep 5; done ) >> "$OUT/${tag}_samples.csv" &
  SAMP=$!
  bash "$RW" "$tag"
  kill $SAMP 2>/dev/null
  for h in spark1 spark2; do awk -F, -v h=$h '$1==h{p+=$4;t+=$5;n++; if($5>mt)mt=$5} END{if(n)printf "%s during %s: avg power %.1f W, avg temp %.1f C, peak temp %d C (%d samples)\n",h,tag,p/n,t/n,mt,n}' tag=$tag "$OUT/${tag}_samples.csv"; done
done
