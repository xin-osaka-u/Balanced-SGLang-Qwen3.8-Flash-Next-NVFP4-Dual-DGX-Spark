# Balanced-SGLang-Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark

_A balanced recipe: Qwen3.8-Flash-Next (NVFP4) on two DGX Sparks with SGLang TP=2._

This recipe serves Qwen3.8-Flash-Next for personal and household use. It optimizes for four things, in this order:

1. **Correct.** We add no loss beyond the published checkpoint. Accuracy is verified, not assumed.
2. **Stable.** Built to run for weeks, and to detect its own failure modes.
3. **Cheap to keep.** Restarts barely touch the SSD, and the GPU drops to half power when idle.
4. **Fast enough.** About 112 tok/s on short structured output, 37–38 tok/s on Chinese and Japanese prose, and 0.2–0.3 s to the first token.

It is not a tok/s leaderboard entry. Other recipes are faster (see [community](#other-recipes)), mostly by quantizing further. If you want a tool that a family can trust to read contracts, bills and school notices, read on.

> **中文简介**
> 这是一套在两台 DGX Spark 上用 SGLang 运行 Qwen3.8-Flash-Next 的均衡方案，面向个人和家庭部署，不追求跑分。
> - **精度**：在官方 NVFP4 权重之外不再做任何量化，KV 用 BF16。GSM8K 全量 1319 题得 97.35%，落在权重作者给出的 BF16 范围内；中日文结构化抽取 36/36。
> - **稳定**：能自己发现 GB10 锁频这类硬件故障；踩过的坑和排查方法都写在文档里。
> - **省**：PLE 表放在 SSD 上并跨重启复用，每次重启的写入从约 28 GB 降到 2–3 GB；空闲时 GPU 功耗减半，推理时限频到 2200 MHz，功耗再降约四成、速度只慢约 4%。
> - **够快**：短 JSON 约 112 tok/s，中日文长文 37–38 tok/s，首字 0.2–0.3 秒。草稿深度按内容自动调整：结构化输出多猜几个 token，长文少猜。
>
> 这个仓库不是让你照抄命令的。它记录了每个选择背后的理由、实测数据和放弃的方案，方便你（或你的 AI 助手）按自己的硬件和用途做判断。
>
> **日本語の概要**
> 2台の DGX Spark 上で SGLang を使い、Qwen3.8-Flash-Next を動かすためのバランス重視のレシピです。個人・家庭での実運用を想定しており、ベンチマークの数字は追いかけていません。
> - **精度**：公開されている NVFP4 チェックポイント以上の量子化は一切行わず、KV キャッシュは BF16 です。GSM8K（全1319問）で 97.35%、作者が示す BF16 の範囲内に収まっています。日中の構造化抽出テストは 36/36 です。
> - **安定性**：GB10 のクロック固着のようなハードウェア障害を自動で検知します。実際に遭遇した障害と見つけ方はドキュメントにまとめています。
> - **維持コスト**：PLE テーブルを NVMe に置いて再起動をまたいで再利用するため、再起動ごとの SSD 書き込みは約 28 GB から 2〜3 GB に減りました。アイドル時の GPU 消費電力は半分になります。推論中もクロックを 2200 MHz に抑えており、速度は約 4% 落ちますが、消費電力は約 4 割下がります。
> - **速度**：短い JSON で約 112 tok/s、日本語・中国語の長文で 37〜38 tok/s、最初のトークンまで 0.2〜0.3 秒です。ドラフトの深さは内容に応じて自動で変わり、構造化出力では深く、文章では浅くなります。日常使いには十分な速さです。
>
> コマンドをそのままコピーするためのリポジトリではありません。それぞれの選択の理由、実測データ、採用しなかった案を記録しています。ご自身のハードウェアや用途に合わせた判断（AI アシスタントに任せる場合も含めて）の材料にしてください。

## At a glance

| | |
|---|---|
| Model | Qwen3.8-Flash-Next, stock ModelOpt NVFP4 checkpoint. Experts NVFP4, PLE FP8, all else BF16. **Nothing requantized.** |
| Engine | SGLang `main@7be5e3473c` plus [9 patches](patches/) (2 are upstream PRs) |
| Topology | 2× DGX Spark, TP=2 over a direct ConnectX-7 RoCE link |
| KV cache | **BF16**, 380,928 tokens, 4 seats, 200K context; about 43 GB kept free as headroom |
| Speculative decoding | NEXTN (MTP) with adaptive depth (1–7 draft steps, up to 8 tokens) and a 65,536-token draft vocab |
| Speed, single stream | about 112 tok/s short CN JSON · 37 CN prose · 38 JP prose · TTFT 190–285 ms (clocks capped at 2200 MHz) |
| GSM8K (author protocol, all 1319) | **97.35%** (97.42% at fixed depth; one question apart). The author reports 97.27% on the same weights (BF16 band 97.12–97.50). 85% of questions finish in ≤ 400 tokens. |
| RULER2 at 32k context | 0.717, within single-sample noise of the stock engine layout (0.727) |
| CN/JP structured-extraction gate | **36/36** on every configuration we shipped |
| Restart cost | ~8 min, **2–3 GB** of SSD writes per node (was 27–29 GB) |
| GPU power per node (optional governor) | idle 11–12 W → 5.5–6.3 W · decoding 19–24 W → 12.5 W at the 2200 MHz cap |

All numbers and their conditions: [docs/measurements.md](docs/measurements.md).

## What we did, and what we deliberately did not do

**Did:**
- Kept the 23.8 GiB/rank PLE table on NVMe instead of in unified memory. This freed 24 GB and made decode *faster*.
- Made that table survive restarts: verified, mapped read-only, written once.
- Ported vLLM's sm_121 skinny BF16 GEMM to SGLang.
- Made the reduced draft vocabulary work at TP=2.
- Fixed the sparse-attention index ring so drafts can go deeper than 4 tokens, then let the engine choose the depth per request: +50% on structured output, prose unchanged.
- Trimmed host settings that were wearing the SSD for nothing.
- Wrote a clock governor that caps idle and active clocks and detects the GB10 clock latch.

**Did not:**
- Use FP8 KV, FP8 dense layers, INT4 experts, fewer active experts, a 4-bit lm_head or an NVFP4 PLE.
- Use other people's tok/s tables as a target.
- Set up autostart.
- Install off-channel drivers or kernels.

Every one of these has a reason in [docs/decisions.md](docs/decisions.md).

## Measure tasks, not just tok/s

Thinking-on answers vary a lot in length, so raw tok/s says little about how long a *task* takes. On GSM8K with thinking on:
- The median question needs 166 tokens, about 7 s.
- 0.9% of questions run away to the 8,192-token limit and burn 18% of all tokens.
- Wrong answers average 8–9× the tokens of correct ones.
- Overall that is about 420 tokens per *correct* answer.

`tools/eval/gsm8k_token_stats.py` prints this for your own runs. A change that makes the model think longer is a regression, even when tok/s goes up.

## Quick start (if your setup matches ours)

```bash
# 1. Engine
git clone https://github.com/sgl-project/sglang.git && cd sglang && git checkout 7be5e3473c
git am ../Balanced-SGLang-Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark/patches/*.patch && pip install -e "python[all]"   # fresh venv, both nodes

# 2. Draft vocabulary (once)
python tools/hot_tokens/build_hot_tokens.py --tokenizer $MODEL/tokenizer.json --out hot_tokens_65536.pt

# 3. Host settings (optional, recommended): see system/README.md

# 4. Start, then run the gates
cp scripts/config.env.example scripts/config.env   # edit paths, IPs, HCAs
scripts/launch_tp2.sh
tools/bench/run_window.sh first-boot
```

Then read [docs/validation.md](docs/validation.md) and [docs/operations.md](docs/operations.md). If your setup differs (one Spark, other languages, agent concurrency), start with [AGENTS.md](AGENTS.md).

## Repository map

| Path | What |
|---|---|
| [docs/](docs/) | Decisions, validation gates, operations and failure modes, measurements, dead ends |
| [patches/](patches/) | The exact SGLang patch series we run, with upstream status |
| [scripts/](scripts/) | Two-node launcher and config template |
| [tools/bench](tools/bench) | `det_bench.py` (T=0 speed), `lossless_test.py` + `cmp.py` (CN/JP gate), `run_window.sh` |
| [tools/eval](tools/eval) | GSM8K (author protocol), token statistics, RULER2 at 32k, non-thinking snapshot, thinking-off proxy |
| [tools/qsa_ring](tools/qsa_ring) | Offline checks for the index-ring patch |
| [tools/hot_tokens](tools/hot_tokens) | Draft-vocabulary builder |
| [tools/ple](tools/ple) | PLE table byte-exact compare and reuse check |
| [ops/gpu-governor](ops/gpu-governor) | Idle and active clock caps plus clock-latch alarm (optional) |
| [system/](system/) | sysctl and apt settings, logging notes |
| [results/](results/) | Raw GSM8K and RULER2 metrics, snapshot summary |
| [AGENTS.md](AGENTS.md) | Guide for AI assistants adapting this recipe |

## Honest limitations

- **Short track record on this exact config.** Adaptive depth went live on 2026-10-10, the rest on 2026-10-08. Earlier configurations ran for multi-day stretches. The worker had one kernel panic on 2026-10-06 whose cause we don't know (see [operations](docs/operations.md)).
- **Adaptive depth is tuned for single-stream structured output.** On 4-way concurrent reasoning (GSM8K) it was about 10% slower than fixed depth, one run each. The non-thinking snapshot (IFEval, mgsm, TruthfulQA, MMLU-Pro) was not repeated on it.
- **The governor is young.** It has run in production since 2026-10-09. Its first latch check false-alarmed and was replaced.
- **Built for our traffic:** 4 seats and mostly single-stream household use. We have no aggregate-throughput or tool-calling numbers.
- **Pinned to an older SGLang.** It predates the Qwen4 decoder refactor (sglang#42478). Patch 0001 may no longer be needed upstream; we haven't re-checked. Patch 0009 is not upstreamed and touches a Triton kernel, so it needs the most care on a rebase.
- **The production draft vocabulary includes our private traffic.** The public builder uses word frequencies only. Its table shares 94.2% of ids with ours, but we haven't measured its acceptance length. Measure it on your own prompts.

## Other recipes

They make different trade-offs, and several are faster. Pick by what you value.
- [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks): vLLM, FP8 KV by default, 1M context, 8 seats.
- [myllmbox/qwen38-flash-next-cluster-recipe](https://github.com/myllmbox/qwen38-flash-next-cluster-recipe): vLLM, INT4-AutoRound (5 experts) by default, BF16 KV, sampled deep drafts, high concurrency.
- [beastllama/dgx-spark-qwen38-flash-next-recipe](https://github.com/beastllama/dgx-spark-qwen38-flash-next-recipe): SGLang TP=2 field notes.
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker): multi-model vLLM images for Spark.

## Upstream work from this recipe

- [sglang#43249](https://github.com/sgl-project/sglang/pull/43249): `--speculative-token-map` with a vocab-sharded lm_head (TP>1). Fixes [#42397](https://github.com/sgl-project/sglang/issues/42397).
- [sglang#43250](https://github.com/sgl-project/sglang/pull/43250): sm_121 skinny BF16 GEMM for decode projections and lm_head.
- [TP=2 PLE reuse report on sglang#42400](https://github.com/sgl-project/sglang/pull/42400#issuecomment-6071960442)

## Credits and license

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The kernel and plan table come from vLLM (vllm-project/vllm#59605 and #59632). The PLE sampling check follows shantanugoel/ple_reuse.py. The SGLang team built the engine and file-backed PLE. Model weights are not included.
