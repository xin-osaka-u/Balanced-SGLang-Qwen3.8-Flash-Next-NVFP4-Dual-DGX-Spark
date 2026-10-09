"""Deterministic decode benchmark: same prompts at T=0, N repeats, decode-only tok/s.

Usage: python det_bench.py [base_url] [repeats]
Identical outputs each repeat, so the spread across repeats is system noise only.
"""
import json
import sys
import time
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000") + "/v1/chat/completions"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PROMPTS = {
    "receipt": "把下面小票整理成JSON（store,date,items[{name,qty,price}],total）：\n"
               "全家便利店 2026-09-30 19:42\n饭团 2×150=300\n乌龙茶 1×138\n炸鸡 1×220\n合计 658日元",
    "essay": "用中文写一篇约500字的家庭理财小贴士，分5点。",
    "jp": "次の内容を日本語で丁寧なお知らせ文に書き直してください：10月10日に小学校の運動会、雨天時は12日に延期、弁当と水筒を持参。",
}


def run(prompt):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; toks = 0; text = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                toks = d["usage"]["completion_tokens"]
            for c in d.get("choices", []):
                piece = c.get("delta", {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    text.append(piece)
    t1 = time.time()
    return toks, (toks - 1) / (t1 - t_first), (t_first - t0) * 1000, hash("".join(text))


for name, p in PROMPTS.items():
    res = [run(p) for _ in range(N)]
    rates = [r[1] for r in res]
    same = len({r[3] for r in res}) == 1
    print(f"{name:8s} tokens={res[0][0]:4d} identical={same} tok/s=" + " ".join(f"{x:5.1f}" for x in rates)
          + f"  mean={sum(rates)/len(rates):5.1f} spread={max(rates)-min(rates):4.1f} ttft_ms={sum(r[2] for r in res)/N:.0f}")
