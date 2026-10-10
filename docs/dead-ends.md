# Dead ends (measured, so you don't have to)

| Tried | Result | Conclusion |
|---|---|---|
| `NCCL_MIN/MAX_NCHANNELS=4` (from some Spark guides) | Single-stream decode 56 → 42.8 tok/s | Keep the default of 16 channels |
| Newer cuBLAS (13.4.2.4, 13.8.1.7) preloaded under torch 2.13 | Picked the same `cutlass_80_wmma` kernels for all 36 decode shapes: 12,740 / 12,904 / 12,933 µs total vs 13.1.1.3. 13.8 was slower on a few small shapes. | The sm_121 notes cover NVFP4/MXFP8 large GEMMs, not small-M BF16. Use the skinny GEMM instead. |
| Deeper drafts on stock SGLang (NEXTN 7/8, adaptive) | Startup refused: QSA requires `num_draft_tokens <= compress_ratio (4)` | Needs patch 0009. This was a dead end until we wrote it. |
| A *fixed* deeper draft (7 steps / 8 tokens) | Not shipped. From measured per-step cost (4–5 ms) and acceptance lengths, we projected +32% on JSON, −16% on prose and −11% on reasoning. | Use adaptive depth, which got +50% on JSON with prose unchanged |
| GSM8K on the first 300 questions as a quick gate | 96.67% and 97.67% from two runs of one configuration | Too noisy to see a one-point effect. Run all 1319. |
| Active clock cap at 1800 MHz | About 10% slower on prose than uncapped (2200 MHz: about 4%), for 3 W less than 2200 MHz | 2200 MHz is the better trade for us |
| PLE file backend with `SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=16` (default 8) | No TTFT improvement | Keep 8 |
| `--disable-flashinfer-autotune` | 64.5 vs 65.5 tok/s, within noise | No effect |
| `--boundary-reduction` (resolves to `rs+rsv` for this model) | Profile shows only AllReduce at TP=2 with batch ≤ 4; it falls back as documented | Nothing to gain at our batch sizes |
| A/B with sampled "family" chat prompts | Differences between configs drowned in sampling noise | Use T=0 fixed prompts (det_bench) plus scored gates instead |
| `pinned` PLE backend (table in unified memory) | 18.9 GB free, 49 tok/s on JSON | `file` is better on both counts (43 GB free, 65.5 tok/s) |
| Pipeline parallel (PP=2) instead of TP=2 | We ran it during bring-up and moved to TP=2 | Not re-measured on the current stack |

Not tried on purpose, because they break the "no loss beyond the checkpoint" rule:
- FP8 KV
- FP8 dense layers
- INT4 experts
- Fewer active experts
- A 4-bit lm_head
- NVFP4 PLE

Other recipes measure these; see the community links in the [README](../README.md).
