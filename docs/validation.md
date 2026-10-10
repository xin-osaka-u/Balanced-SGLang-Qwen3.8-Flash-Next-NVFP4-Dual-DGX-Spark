# Validation gates

Run these after every change: engine upgrade, patch, flag, firmware or OTA update. A change that fails a gate does not ship, however much faster it is.

## Gate 1: CN/JP lossless suite (always, ~3 min)

```bash
python tools/bench/lossless_test.py http://127.0.0.1:8000 ./out <tag>
python tools/bench/cmp.py ./out <old_tag> <tag>
```

The suite uses five fictional household documents, with thinking off and T=0:
- a Chinese lease
- a Chinese monthly expense list
- a Japanese invoice
- a Japanese school notice
- a Japanese property disclosure, which is summarized free-form and not scored

The four structured cases are scored field by field against hand-written ground truth. The pass mark is **36/36** (13 + 7 + 7 + 9).

These cases are chosen to catch damage that headline benchmarks hide:
- numbers copied wrongly (金额 / 円)
- dates converted wrongly between formats
- Japanese → Chinese item names
- a subtotal that has to be computed rather than copied

**Replace them with your own documents.** The point is to cover your traffic. Use made-up content, not real personal data.

**Comparing outputs.** In our runs, the structured cases have produced identical bytes across boots. Free-form prose at T=0 does not, even with speculative decoding off. That comes from this engine's own non-determinism, so judge prose by meaning, not by bytes.

## Gate 2: speed and acceptance (always, ~5 min)

```bash
tools/bench/run_window.sh <tag>      # waits for health, warms up, runs det_bench and gate 1
```

- `det_bench.py` reports tok/s, TTFT, and whether the 3 repeats were identical.
- Compare against your own previous window on the same boxes. A 3-run mean on the JSON prompt is stable to about ±3%; prose varies more.
- Also check the acceptance length in the server log (`accept len`). With adaptive depth: 5–8 on the JSON prompt, about 2.2 on prose. With fixed 3/1/4: about 4.0 and 2.2–3.5.
- With adaptive depth, the first JSON run after a quiet period is slower (80–87 tok/s against 112). Look at runs 2 and 3.

## Gate 3: GSM8K with the author's protocol (after engine/kernel changes, ~100 min)

```bash
tools/eval/run_gsm8k.sh http://127.0.0.1:8000 ./gsm8k_<tag>
python tools/eval/gsm8k_token_stats.py ./gsm8k_<tag>/output-rs0.jsonl
```

Pass when:
- the score is inside the checkpoint author's BF16 band (97.12–97.50 on their hardware; we got 97.42), **and**
- the truncation rate and mean tokens per question have not grown.

A change that makes the model "think longer" is a regression even if the score holds. GSM8K is saturated, so treat it as a tripwire, not a ranking.

Run all 1319 questions. We tried the first 300 as a shortcut: two runs of the same configuration scored 96.67% and 97.67%, which is wider than the effect we were looking for.

## Gate 4: non-thinking snapshot (after big upgrades, ~2.5 h)

`tools/eval/run_snapshot.sh` runs IFEval, mgsm zh/ja, TruthfulQA mc2 en/zh and MMLU-Pro (25 per subject), with thinking off through `tools/eval/nt_proxy.py`.

Compare against [results/nonthinking_snapshot.txt](../results/nonthinking_snapshot.txt) with the same lm-eval version and the same extractor. A drop of more than about 2 points on IFEval or mgsm is worth investigating.

## Gate 5: PLE table integrity (after the first reuse boot, and after any crash)

```bash
python tools/ple/full_compare.py <ple_table_rank.bin> <model_dir> <rank> 2   # on each node, ~40 s
HOSTS="spark1 spark2" PLE_DIR=... tools/ple/verify_ple_reuse.sh --hash
```

Pass when:
- all shards are byte-identical to the checkpoint,
- the mapping is read-only (`r--s`), and
- the table's mtime does not advance while serving.

## Gate 6: RULER2 at 32k (after changes to attention, KV layout or block selection, ~1 h)

```bash
python tools/eval/nt_proxy.py &
TOKENIZER=/path/to/model tools/eval/run_ruler2.sh ./ruler2_<tag>
```

Gates 1–3 use short sequences. This model's sparse attention only starts *selecting* blocks beyond 2,048 tokens, so a bug in that path is invisible to them. Compare subtask by subtask against [results/](../results): with 8 samples per subtask, one sample is 12.5%, and our three reference runs differ by exactly that on 3 of 12 subtasks.

For patch 0009 specifically, also run the two offline checks in [tools/qsa_ring](../tools/qsa_ring).

## What we deliberately do not use as gates

- **Other people's tok/s tables.** Different prompts, languages, thinking settings and boxes.
- **Sampled (T>0) chat prompts for A/B.** Too noisy: our early "family prompts" A/B could not tell configurations apart.
- **Token-by-token equality on prose.** See gate 1.
