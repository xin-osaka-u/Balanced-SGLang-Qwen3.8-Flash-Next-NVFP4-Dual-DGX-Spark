"""Per-question token and latency distribution of an sgl-eval GSM8K run.

Raw tok/s says little about how long a task takes. This prints how many tokens
(thinking included) a question needs, how much of the budget runaway thinking
burns, and tokens per *correct* answer.

usage: python gsm8k_token_stats.py <out_dir>/output-rs0.jsonl
"""
import collections
import json
import statistics as st
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]


def q(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p * len(a)))]


def desc(name, rs):
    if not rs:
        print(f"{name:18s} n=0")
        return
    t = [r["num_generated_tokens"] for r in rs]
    d = [r["generation_end_time"] - r["generation_start_time"] for r in rs]
    print(f"{name:18s} n={len(t):4d} mean={st.mean(t):6.0f} median={st.median(t):5.0f} "
          f"p90={q(t, .9):5d} p99={q(t, .99):5d} | time mean={st.mean(d):5.1f}s "
          f"median={st.median(d):5.1f}s")


ok = [r for r in rows if r.get("symbolic_correct")]
trunc = [r for r in rows if r["finish_reason"] != "stop"]
desc("all", rows)
desc("correct", ok)
desc("wrong", [r for r in rows if not r.get("symbolic_correct")])
desc("truncated", trunc)
desc("all minus truncated", [r for r in rows if r["finish_reason"] == "stop"])

edges = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 2000), (2000, 8192)]
b = collections.Counter()
for r in rows:
    t = r["num_generated_tokens"]
    if r["finish_reason"] != "stop":
        b["truncated"] += 1
        continue
    for lo, hi in edges:
        if lo <= t < hi:
            b[f"{lo}-{hi - 1}"] += 1
            break
print("buckets:", ", ".join(f"{k}: {v}" for k, v in b.items()))
tot = sum(r["num_generated_tokens"] for r in rows)
tt = sum(r["num_generated_tokens"] for r in trunc)
print(f"accuracy {len(ok)}/{len(rows)} = {100 * len(ok) / len(rows):.2f}%")
print(f"truncated {len(trunc)} ({100 * len(trunc) / len(rows):.2f}% of questions, "
      f"{100 * tt / tot:.1f}% of all generated tokens)")
print(f"tokens per correct answer: {tot / max(1, len(ok)):.0f}")
