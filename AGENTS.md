# For AI agents adapting this recipe

You are probably helping someone deploy a model on their own hardware. Don't copy this repository verbatim. Read it for the reasoning, then rebuild the parts that apply to your user's boxes, model and traffic.

## Read in this order
1. [docs/decisions.md](docs/decisions.md): each choice, the measurement behind it, and when to choose differently.
2. [docs/validation.md](docs/validation.md): the gates that decide whether a change ships.
3. [docs/operations.md](docs/operations.md): failure modes we hit, and how we detected them.
4. [docs/dead-ends.md](docs/dead-ends.md): things that were measured and did not help.
5. [patches/README.md](patches/README.md): what each patch changes, and its upstream status.

## Invariants of this recipe
Ask your user before changing any of these. They are policy, not tuning:
- **No loss beyond the published checkpoint:**
  - BF16 KV.
  - No requantization, expert pruning, FP8 dense layers or 4-bit heads.
  - Speculative decoding only in forms where the target model verifies every token.
- **Every change passes gate 1** (structured CN/JP extraction, field-level exact) **before it ships.** After engine changes, it must also pass gate 3 (GSM8K, author protocol, within the BF16 band, with no growth in tokens per question).
- **Compare against the same boxes' recorded baseline**, not against other people's numbers.
- **Prefer tuning to rolling back,** but ask the human before rolling back production.
- **Warm or compile kernels with no model loaded,** and cap `MAX_JOBS=4` on unified-memory boxes.
- **Updates only through the vendor's official channel** (DGX OS OTA), and rarely.

## Adapting to different hardware or traffic
- **One Spark (TP=1):** the PLE table at TP=1 is 47.7 GiB.
  - The `file` backend and reuse matter even more there.
  - The skinny GEMM plan table includes TP=1 rows from vLLM.
  - The hot-token head needs no slicing at TP=1.
- **Different languages:** rebuild the hot-token table (`tools/hot_tokens/build_hot_tokens.py`). Check `accept len` in the server log before and after on the user's real prompts.
- **More concurrency (agents):** raise `--max-running-requests` and `--cuda-graph-max-bs-decode` together. The skinny GEMM only plans M ≤ 16, so larger batches fall back to cuBLAS automatically. Re-measure aggregate throughput and keep about 30–40 GB of unified memory free.
- **Newer SGLang:** apply the patches one at a time and run the gates after each.
  - Check first whether [sglang#43249](https://github.com/sgl-project/sglang/pull/43249) and [#43250](https://github.com/sgl-project/sglang/pull/43250) are merged, and whether 0001 is still needed.
- **Other GB10 boxes:** after every power event, check for the clock latch: a 15 s fp16 matmul should reach about 80 W.

## Things that look like wins but are not
See [docs/dead-ends.md](docs/dead-ends.md). In short:
- Fewer NCCL channels.
- Newer cuBLAS.
- Deeper drafts on QSA.
- A bigger PLE RSS budget.
- A/B tests on sampled chat prompts.
