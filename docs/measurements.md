# Measurements

Every number here comes from our own two boxes, together with the conditions it was measured under. Compare your own numbers against your own earlier runs on the same box. Comparing against other people's boxes, or other people's tables, tells you much less.

**Hardware and software**

| Item | Value |
|---|---|
| Machines | 2× DGX Spark (GB10, sm_121, 128 GB unified memory each), DGX OS 7.2.3 |
| Kernel / driver | 7.0.0-1019-nvidia, driver 580.178.04 |
| Interconnect | ConnectX-7 direct link, RoCE v2 (two rails per port), NCCL 2.29.7 with 16 channels (default) |
| Engine | SGLang `main@7be5e3473c` plus [patches](../patches) |
| Python stack | torch 2.13.0+cu130, flashinfer 0.7.0.post1, nvidia-cutlass-dsl 4.8.0, quack-kernels 0.6.5 |
| Model | Qwen3.8-Flash-Next NVFP4 (ModelOpt). Experts NVFP4, PLE table FP8, everything else BF16. MTP head BF16. |
| Serving config | TP=2, BF16 KV, NEXTN with adaptive depth (since 2026-10-10; fixed 3/1/4 before), 65,536-id hot-token draft head, 4 seats, 200K context, 380,928 KV tokens |
| GPU clocks | Since 2026-10-09: capped at 2200 MHz while serving and 1000 MHz when idle ([governor](../ops/gpu-governor)). Sections below say which state they were measured in. |

## Speed: single stream, thinking off, T=0

**Current production** (adaptive depth, 2200 MHz cap; two boots on 2026-10-10):

| | Fixed 3/1/4 + ring fix, same cap | **Production (adaptive)** |
|---|---|---|
| Short JSON | 72.4–72.7 (first run after boot 65, then 76–77) | **about 112** warmed up (87 / 112 / 111 and 81 / 112 / 112) |
| CN prose | 34.2–35.5 | 36.8–37.7 |
| JP prose | 35.1–35.6 | 37.6–38.0 |
| TTFT (short prompts) | 170–210 ms | 190–285 ms |
| CN/JP gate, end-to-end tok/s on the four structured cases | 54–65 | 65–93 |

- The first structured request after a quiet period runs at 80–87 tok/s while the controller's acceptance average warms up.
- Acceptance length: 5.7 on structured output (the cap is 8), 2.2 on prose. On prose the controller falls back to short drafts, which is why prose does not get slower.
- A whole verify cycle takes about 59 ms on prose and about 190 ms on deep structured drafts.

**History: before the cap and before adaptive depth** (uncapped clocks, fixed 3/1/4). Measured with `tools/bench/det_bench.py`: 3 runs per prompt and 3 server boots for the tuned column. The prompts are:
- **Short JSON**: a short Chinese receipt turned into JSON, about 90 tokens of output.
- **CN prose**: a 500-character Chinese essay.
- **JP prose**: a Japanese notice rewrite.

| | Before patches 0002–0005 and the hot-token map | Production |
|---|---|---|
| Short JSON | 64.2 tok/s | **77.8** (75.9 / 77.2 / 80.3) |
| CN prose | 31.4 | **38.4** (37.3 / 38.2 / 39.7) |
| JP prose | 34.7 | **39.0** (36.2 / 38.4 / 42.4) |
| TTFT (short prompts) | 180–270 ms | 180–270 ms |

- **Spread between runs.** Long prose at T=0 does not come out identical from run to run. This engine has the same non-determinism with speculative decoding off, so the speed numbers for prose vary with the text. The JSON task was identical across all runs and all boots.
- **With MTP off:** 24–26 tok/s.
- **MTP acceptance** with 4 draft tokens:

  | Task | Acceptance length | Acceptance rate |
  |---|---|---|
  | Short JSON | 4.00 | 1.00 |
  | Chinese prose | 2.6–3.5 | — |
  | GSM8K, thinking on, 4-way concurrency | 2.67 | 0.56 |

## QSA pending ring: stock vs fixed layout (patch 0009)

Offline replay of verify windows with the real slot builders (`tools/qsa_ring/ring_test_sim.py`), counting compressed keys that do not equal the mean of their true members:

| Layout | Draft tokens | Wrong keys |
|---|---|---|
| Stock, 1 group | 4 | 1558 / 1924 (81%) |
| Upstream draft PR sizing, 2 groups | 8 | 1646 / 3690 |
| Patch 0009: 2 groups | 4 | 0 |
| Patch 0009: 3 groups | 6 or 8 | 0 |
| Patch 0009: 4 / 5 groups | 12 / 16 | 0 |

The CUDA-graph Triton kernel matches the host builders at 1, 2, 3 and 5 groups (`tools/qsa_ring/ring_test_gpu.py`).

**In service** the fix changed nothing we could measure. Same code, three layouts, RULER2 at 32k context (8 samples per task, thinking off, T=0):

| | Stock layout (1 group) | Ring fix, fixed depth (2 groups) | Production (3 groups, adaptive) |
|---|---|---|---|
| mk_niah basic / easy / medium / hard | 1.0 / 0.93 / 0.375 / 0.50 | 1.0 / 0.93 / 0.375 / 0.375 | 1.0 / 0.93 / 0.50 / 0.375 |
| mv_niah basic / easy / medium / hard | 1.0 / 0.29 / 0.13 / 0.75 | 1.0 / 0.29 / 0.13 / 0.64 | 1.0 / 0.29 / 0.13 / 0.76 |
| qa basic / easy / medium / hard | 1.0 / 1.0 / 0.875 / 0.875 | 1.0 / 1.0 / 0.875 / 0.875 | 1.0 / 1.0 / 0.75 / 0.875 |
| Total | 0.727 | 0.707 | 0.717 |

- 9 of the 12 subtasks are identical across the three. The rest differ by one sample (12.5% of a task), in no consistent direction.
- So the aliasing in the stock layout is real, but its effect on block selection is below what this test can resolve. We keep the fix because it is correct and it is what makes deeper drafts possible.
- mv_niah easy/medium are low on all three. That is the model finding only some of several values, not a deployment effect.
- Speed with the ring fix alone is unchanged, and the JSON prompt is byte-identical to the stock layout.

## PLE placement: `file` vs `pinned`

Same commit and flags, before patches 0002–0008:

| | pinned | **file** |
|---|---|---|
| Free memory after startup (rank 0) | 18.9 GB | **43.0 GB** |
| Short JSON decode | 49.1 tok/s | **65.5** |
| CN / JP prose | 28.8 / 33.8 | 32.1–33.4 / 35.4–35.8 |
| TTFT (short prompts) | 167–254 ms | 185–450 ms |

Moving the 23.8 GiB/rank table out of unified memory made decode *faster* and freed 24 GB. Our unverified guess at the cause: there is less memory reclaim pressure while the CPU launches kernels. TTFT got slightly worse on prompts it had not seen before.

## Boot and SSD writes per node

| | Boot to healthy | SSD written |
|---|---|---|
| Without reuse (delete the table and rebuild) | 8.5–12 min | ~27–29 GB |
| First boot with reuse (builds the table and sidecar) | 514 s | ~29 GB |
| Reuse boot | **468 s** | **2.7 GB** (rank 0) / 1.8 GB (rank 1): logs and JIT cache |

- **Byte-exact check after reuse:** 64/64 shards identical to the checkpoint on both ranks (`tools/ple/full_compare.py`). Table hashes stayed unchanged after serving requests.
- **Idle SSD writes per node:** about 2–6 GB/day after the changes in [system/](../system). Most of that comes from other services on the box.
- **SSD wear at the time of writing:** 1.68 TB written, 0% life used, 0 media errors.

## Memory (steady state, model loaded)

- `free` shows 35–39 GB available, of which about 33 GB is page cache (mostly the PLE table).
- Swap stays static: no swap-out under load.
- Adaptive depth adds about 1.3 GB of GPU memory.
- After a long evaluation night the scheduler's Python heap had grown to 12–13 GB per node and available memory fell to 26–31 GB. It was flat when idle. A torch profiler run inside our benchmark window was one contributor; the window script in this repo does not run one.
- Weights take 43.3 GB on rank 0 (main model).

## Power and clocks (GPU rail only, from `nvidia-smi`; wall power not measured)

| State | Clock | GPU power | GPU temp |
|---|---|---|---|
| Model loaded, idle, uncapped | 2405 MHz | 11–12 W | ~50 °C |
| Model loaded, idle, `-lgc 210,1000` | ~980 MHz | 5.7–6.3 W | ~47 °C |
| No model, idle | 208 MHz (P8) | ~4.5 W | — |
| Full load (15 s fp16 4096² matmul) | 2377–2493 MHz | 82–84 W | 63–66 °C |
| Decoding, uncapped | ~2.4 GHz | 19–24 W | 62–63 °C peak |
| Decoding, capped at 2200 MHz (production) | 2200 MHz | 12.5–12.7 W | 56–61 °C peak |
| Decoding, capped at 1800 MHz | 1800 MHz | 9.6–9.8 W | 53–58 °C peak |
| **Latched GB10 (fault)**, under the same load | **611 MHz** | **5.5 W** | — |

## Accuracy

**GSM8K**, using the checkpoint author's protocol (`tools/eval/run_gsm8k.sh`): all 1319 questions, thinking on, T=0.6, top-p 0.95, max_tokens 8192, seed 0, 4 threads. The run took 101 minutes.

| | Fixed 3/1/4, stock ring (2026-10-09) | Fixed 3/1/4, ring fix | **Production: adaptive** (2026-10-10) | Checkpoint author (same weights, B200) |
|---|---|---|---|---|
| Accuracy | 97.42% (1285/1319) | 97.42% (1285/1319) | **97.35%** (1284/1319) | 97.27% (BF16 band 97.12–97.50) |
| Truncated at 8192 | 0.91% (12) | 0.61% (8) | 0.91% (12) | 1.14% |
| Mean tokens per question | 411 | 386 | 408 | ~432 |
| Tokens per correct answer | 422 | 396 | 419 | — |
| Median time per question | 7.3 s | 6.0 s | 6.6 s | — |

- The three runs agree to within one question. The first two share 28 of their 34 wrong answers; the rest are sampling at T=0.6.
- Two 300-question subsets of the same configuration scored 96.67% and 97.67%. A subset cannot tell these configurations apart. Run all 1319.
- The time row mixes two effects. The first run started right after a boot with a cold page cache and uncapped clocks; the other two ran warm under the 2200 MHz cap. The fair pair is the last two: adaptive depth is about 10% slower on this 4-way reasoning workload.

Per-question distribution for the first run (`tools/eval/gsm8k_token_stats.py`); the other two have the same shape:

| | n | Mean tokens | Median | p90 | p99 | Mean (median) time |
|---|---|---|---|---|---|---|
| All | 1319 | 411 | 166 | 589 | 5970 | 17.9 s (7.3 s) |
| Correct | 1285 | 340 | 165 | 505 | 4297 | 14.7 s (7.2 s) |
| Wrong | 34 | 3110 | 1730 | 8192 | 8192 | 138 s (81 s) |
| Truncated | 12 | 8192 | — | — | — | 366 s |

Token-count buckets:

| Tokens | Questions | Share |
|---|---|---|
| < 100 | 223 | 17% |
| 100–199 | 567 | 43% |
| 200–399 | 327 | 25% |
| 400–799 | 101 | 8% |
| 800–1999 | 49 | 4% |
| 2000–8191 | 40 | 3% |
| Truncated | 12 | 0.9% |

- 85% of questions finish in ≤ 400 tokens, typically in about 7 s.
- The 12 runaway generations are 0.9% of questions but **18% of all generated tokens**.
- 422 tokens per *correct* answer, which is about 12.7 correct answers per minute at 4-way concurrency.
- The 34 errors are mostly GSM8K label, rounding and units issues. Eight of them are thinking loops cut off by the token limit. We saw no arithmetic-error pattern.

**Non-thinking snapshot** (`tools/eval/run_snapshot.sh`, lm-eval 0.4.13, T=0), taken on the 2026-10-09 config and not repeated with adaptive depth. There is no same-weights reference score for these. Use them as regression baselines, not as leaderboard numbers.

| Task | n | Score |
|---|---|---|
| IFEval, prompt-level strict / loose | 541 | 87.1 / 90.0 |
| IFEval, instruction-level strict / loose | 541 | 91.1 / 93.4 |
| mgsm zh / ja (flexible-extract) | 250 / 250 | 82.0 / 80.0 |
| TruthfulQA mc2, en / zh | 817 / 788 | 66.7 / 56.2 |
| MMLU-Pro, 25 per subject, 5-shot CoT | 350 | 82.0 |

- **mgsm:** some "misses" are answer-extraction noise. For example, the model answers "160 minutes" and then adds "2 h 40 min", and the extractor takes the last number. Compare runs with the same extractor.
- **MMLU-Pro by subject:** math 96, physics 100, engineering 64, law 68.

**RULER2 at 32k** (`tools/eval/run_ruler2.sh`): 0.717 in production. The three-layout table is in the QSA ring section above.

**CN/JP lossless gate** (`tools/bench/lossless_test.py`): 36/36 on every configuration we shipped. The four structured cases score 13/13, 7/7, 7/7 and 9/9.
