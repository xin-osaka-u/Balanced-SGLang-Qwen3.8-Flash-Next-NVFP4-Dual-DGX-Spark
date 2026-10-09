#!/usr/bin/env bash
# Step 0 of the clock-cap plan: with the model loaded and NO requests, does a lower clock cap
# actually reduce idle power and temperature on this node? Run on each node after the power-drain
# maintenance (spark2 must be unlatched first). Needs sudo for -lgc/-rgc. Restores -rgc on any exit.
#   sudo ./step0_idle_cap_test.sh [idle_clocks=210,1000] [seconds_per_phase=120]
set -u
CAP=${1:-210,1000}; SEC=${2:-120}
OUT=${STEP0_OUT:-./results}; mkdir -p "$OUT"
F="$OUT/step0_$(hostname)_$(date +%Y%m%d_%H%M%S).csv"
echo "phase,ts,clock_mhz,power_w,temp_c,util_pct,pstate" > "$F"
trap 'nvidia-smi -rgc >/dev/null 2>&1; echo "(clocks reset)"' EXIT
phase() {  # name seconds
  local name=$1 n=$2 i=0 busy=0
  local sp=0 st=0 sc=0 cnt=0
  while [ $i -lt "$n" ]; do
    local l; l=$(nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu,utilization.gpu,pstate --format=csv,noheader,nounits)
    echo "$name,$(date +%s),${l// /}" >> "$F"
    IFS=, read -r c p t u ps <<<"${l// /}"
    [ "${u:-0}" -gt 5 ] && busy=$((busy+1))
    sp=$(echo "$sp + $p" | bc -l); st=$((st + ${t%.*})); sc=$((sc + c)); cnt=$((cnt+1))
    sleep 2; i=$((i+2))
  done
  printf "%-10s %4ds  avg clock %5d MHz  avg power %6.2f W  avg temp %4.1f C  busy samples %d/%d\n" \
    "$name" "$n" $((sc/cnt)) "$(echo "$sp / $cnt" | bc -l)" "$(echo "$st / $cnt" | bc -l)" "$busy" "$cnt"
  [ $busy -gt 0 ] && echo "   WARNING: requests ran during '$name'; numbers are not idle numbers"
}
echo "node $(hostname), cap $CAP, ${SEC}s per phase, log $F"
echo "A: uncapped idle"; nvidia-smi -rgc >/dev/null; phase uncapped "$SEC"
echo "B: capped idle (-lgc $CAP)"; nvidia-smi -lgc "$CAP" | tail -1; phase capped "$SEC"
echo "C: back to -rgc"; nvidia-smi -rgc | tail -1; phase reset 30
echo "verdict: compare 'capped' vs 'uncapped' power/temp. No drop -> stop the clock-cap plan and find what holds P0."
