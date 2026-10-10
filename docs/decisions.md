# Decisions, and why

Each entry gives the choice, the reason, what we measured or rejected, and when you might choose differently. This file matters more than the scripts. Your hardware, traffic and tolerance for loss are different from ours, so use the reasoning here, not our numbers.

Our constraints, in priority order:
1. **No loss beyond the published checkpoint.** We don't requantize, don't prune experts, and use no FP8 KV. Every change has to pass the CN/JP gate ([validation.md](validation.md)).
2. **Stable for weeks** on two desktop boxes that also run other things.
3. **Cheap to keep:** low SSD wear and low idle power.
4. **Fast enough** for a household's structured-extraction and chat traffic: one request at a time most of the time, at most four at once.

---

### Weights: the stock NVFP4 checkpoint, unchanged

The published ModelOpt NVFP4 checkpoint stores the routed experts in NVFP4, the PLE table in FP8, and everything else in BF16 (attention, shared experts, GDN, the MTP head and the lm_head).

We don't requantize anything else. Some faster community recipes do: INT4 experts, 5-of-10 active experts, a 4-bit lm_head, an NVFP4 PLE table, FP8 dense layers. They publish quality checks, but on 200-question subsets with about ±3 points of noise and no reference run, so a loss of a few points cannot be seen. For a tool that a family relies on to read contracts and bills, we would rather know.

*When to choose differently:* if a scoreboard number matters most to you and you can run your own A/B with enough samples to see a few points, test those quantizations against your own tasks.

### KV cache: BF16

- FP8 KV would roughly double the KV pool. We don't need that: 4 seats × 200K context fit in 380,928 BF16 tokens.
- We haven't measured FP8 KV quality ourselves. We chose not to take the risk.
- A community recipe reported that FP8 KV cost about 0.3 accepted draft tokens per step on this model. So it is not free on speed either.

### KV size: `--max-total-tokens 380928`, `--max-running-requests 4`

- Memory freed by the PLE file backend stays free as headroom, about 43 GB after startup. It is not turned into a bigger KV pool.
- On unified memory, headroom matters. The GPU, the page cache that holds the PLE table, JIT compiles and other services all share the same 128 GB.
- Our early instability (OOM hangs during bring-up) came from running without headroom.
- Four seats matches our traffic. If you serve agents, measure aggregate throughput and choose for yourself.

### PLE table on NVMe: `--ple-offload-backend file`

The 23.8 GiB/rank n-gram (PLE) table moves out of unified memory into a file-backed mapping.

| | Free memory after startup | Short-JSON decode |
|---|---|---|
| `pinned` | 18.9 GB | 49 tok/s |
| `file` | 43.0 GB | 65.5 tok/s |

TTFT on prompts the server had not seen got slightly worse: 185–450 ms vs 167–254 ms. Full table in [measurements.md](measurements.md).

### PLE table reuse across restarts (patches 0006–0008)

Without reuse, each boot rewrote 25.6 GB per node, about 27–29 GB of SSD writes per restart. The upstream advice was to delete the file before every boot, because rewriting a populated file in place was far slower than writing a fresh sparse one.

Our version:
1. Write the table once.
2. `msync` + `fsync`, then write a sidecar fingerprint (config hashes, shard names/sizes/mtimes, shape, dtype, TP size and rank).
3. On a later boot, map the table **read-only** and check every shard against the checkpoint before using it.

Why each piece:
- **Sidecar after `fsync`:** a crash in the middle of the first write leaves no sidecar, so the next boot rebuilds instead of trusting a half-written table.
- **Read-only mapping:** with a writable mapping, GPU page pinning kept marking pages dirty. The file's mtime advanced for hours and the kernel wrote back data that had not changed.
- **No cross-node check needed:** each rank owns its own half of the rows and checks only that half.

Result: reuse boots in 468 s and writes about 2.7 GB (logs and JIT cache).

### Speculative decoding: NEXTN with adaptive depth (patch 0009)

The model's MTP head drafts tokens and the target model verifies every one, so output quality does not depend on draft depth. Depth only changes speed.

**The old cap.** Stock SGLang refuses more than 4 draft tokens on this model. The sparse attention (QSA) keeps a per-request "pending ring" of index keys, one compression group of 4 slots. A verify window wider than 4 would write two positions into the same slot.

**A bug hiding behind the cap.** Even at 4 draft tokens, a verify window that does not start on a multiple of 4 overwrites keys that the open group has already committed. That group's compressed key is then the average of a shifted window. In an offline replay with the real slot builders, the stock layout got 81% of the compressed keys closed inside a verify window wrong. It only affects block *selection*, and only once a sequence exceeds the indexer's 2,048-token budget.

**Patch 0009** sizes the ring to `N = ceil((draft_tokens + 3) / 4)` groups: 2 groups for 4 draft tokens, 3 for 8. The upstream draft PR [sglang#40143](https://github.com/sgl-project/sglang/pull/40143) has the same idea with one group fewer, which still aliases in our replay. `SGLANG_QSA_RING_GROUPS=1` restores the stock layout for A/B runs.

**Why adaptive and not simply deeper.** A draft step is not free: about 4–5 ms each, against a 40 ms plain decode step. Structured output accepts nearly every draft token. Prose and reasoning accept about 2. From our per-step timings, a fixed depth of 8 would have gained about 32% on JSON and *lost* about 16% on prose, so we never shipped it. `--speculative-adaptive` picks 1, 3, 5 or 7 draft steps from recent acceptance:

| Single stream, thinking off, T=0 | Fixed 3/1/4 | Adaptive (up to 8 draft tokens) |
|---|---|---|
| Short JSON | 72–76 tok/s | **112** once warmed up (80–87 on the first request) |
| CN / JP prose | 34–36 / 35–36 | 37–38 / 38 |
| Acceptance length, structured / prose | 4.0 (the cap) / 2.2 | 5.7 / 2.2 |

**What it costs:**
- About 1.3 GB more GPU memory (per-draft Mamba state and four runtime states).
- 4-way concurrent reasoning got about 10% slower than fixed depth: on GSM8K the median question took 6.6 s against 6.0 s. One run each, so treat it as a direction, not a number.

**Accuracy** is unchanged on every gate: full GSM8K 97.35% vs 97.42% (one question), RULER2 32k within single-sample noise, 36/36 on the CN/JP gate, and the JSON prompt byte-identical. See [measurements.md](measurements.md).

*When to choose differently:* if your traffic is mostly concurrent reasoning or agents, keep patch 0009 for the ring fix and run fixed 3/1/4 (drop `--speculative-adaptive`).

### Reduced draft vocabulary: `--speculative-token-map` (65,536 ids)

- On every draft step, the draft head otherwise multiplies by the full lm_head: 124,160 rows per rank at TP=2. Restricting it to tokens we actually produce makes the step cheaper.
- The target model still verifies every token, so output does not change. A token outside the table simply can't be drafted.
- Upstream crashed at TP>1 (it indexed the vocab-sharded head with global ids). Our fix is in patch 0002 and upstream PR [sglang#43249](https://github.com/sgl-project/sglang/pull/43249).
- Build the table per rank, so both shards are balanced: [tools/hot_tokens](../tools/hot_tokens). Ours mixes zh/ja/en word frequencies with our own traffic. The public script uses word frequencies only.
- **If most of your traffic is not CJK**, rebuild the table for your languages and check acceptance length before and after.

### sm_121 skinny BF16 GEMM (patches 0002–0005)

**The problem:** on GB10, cuBLAS serves decode-sized BF16 GEMMs (M ≤ 16) with SM80 WMMA kernels at 125–205 of 273 GB/s. In our decode profile these were the largest single cost: about 69% of GPU time was BF16 GEMM/GEMV.

**The port:** vLLM's CuTe-DSL skinny kernel and its SM121 plan table, plus rows we measured for SGLang's fused shapes. Only shapes with a measured plan use it; everything else stays on cuBLAS.

**Measured gains:**
- Isolated large projections: 1.3–2.3× faster.
- Isolated lm_head at M=1: 1.43× faster.
- Rank-0 share of time in cuBLAS WMMA kernels: 25% → 6%.
- In service, overlap with other CUDA streams roughly halves the kernel-level gain.
- End to end, together with the token map: +21% JSON, +22% CN prose, +12% JP prose.

**Numerics:** BF16 with FP32 accumulation, the same precision class as cuBLAS. The reduction order is different, so results are not guaranteed bit-identical. In isolated tests, 99.9–100% of outputs were identical. Upstream PR: [sglang#43250](https://github.com/sgl-project/sglang/pull/43250).

### NCCL: leave the defaults

`NCCL_MIN/MAX_NCHANNELS=4`, from some Spark guides, dropped single-stream decode from 56 to 42.8 tok/s. We stay on the default 16 channels.

The other `NCCL_*` settings in the launcher are needed for RoCE without GPUDirect on this platform:
- `NET_GDR_LEVEL=0`
- no buffer registration
- RoCE v2 with traffic class 106 / service level 3
- `CROSS_NIC=1` across both rails

### Compile kernels before the model loads; `MAX_JOBS=4`

The first start after a flashinfer upgrade JIT-compiles MoE kernels with about 20 parallel nvcc/cicc jobs. If the model already holds about 90 GB of unified memory at that point, the scheduler gets OOM-killed.

So after any engine upgrade:
1. Compile and warm kernels with no model loaded.
2. Then start serving.
3. Always cap `MAX_JOBS=4`.

### Thinking on or off per task, and a token budget

- Structured extraction (receipts, bills, notices) runs with thinking off. It is faster, and the CN/JP gate passes 36/36 in this mode.
- Reasoning runs with thinking on.
- On GSM8K, 0.9% of questions ran into the 8,192-token limit, and those 12 consumed 18% of all generated tokens. Runaway thinking is also a strong signal of a wrong answer: wrong answers averaged 3,110 tokens, correct ones 340.
- A practical `max_tokens` of 2–4K cuts that tail. The cost is that a few hard questions that would eventually be answered correctly get cut off instead.

### Active clock cap: 2200 MHz

While serving, the governor holds the GPU clock at 2200 MHz instead of the full 2.4–2.5 GHz. We measured about 4% less speed for about 40% less GPU power under decode (19–24 W → 12.5 W per node) and a few degrees lower temperature. An 1800 MHz cap saved more but cost about 10% on prose. Table in [ops/gpu-governor](../ops/gpu-governor/README.md).

*When to choose differently:* if you want the last 4%, leave `GOV_ACTIVE_CLOCKS` empty.

### No autostart

Neither the model server nor the dashboards start at boot.

- **Restarts are deliberate.** A two-node start needs both nodes healthy and the link up.
- **Lower risk at boot.** A crash-looping 120 GB process on a unified-memory box is the worst way to find out something changed.
- **Every restart gets checked.** After each restart we run a measurement window ([validation.md](validation.md)).

*When to choose differently:* unattended boxes. Use a systemd unit with a start delay, a health check, and `Restart=no`.

### Updates: NVIDIA's OTA channel only, rarely

We install only what DGX OS pushes officially. No off-channel kernels, drivers, CUDA or firmware. Each update is followed by the kernel warm-up described above and the full gate.
