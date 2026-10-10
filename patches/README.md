# Patches

These patches are exactly what runs in our production setup (0001–0008 since 2026-10-08, 0009 since 2026-10-10). Apply them in order on SGLang `main@7be5e3473c` (2026-10-03):

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 7be5e3473c
git am /path/to/Balanced-SGLang-Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark/patches/*.patch
pip install -e "python[all]"        # in a fresh venv; we used torch 2.13.0+cu130, flashinfer 0.7.0.post1
```

| # | What | Why | Upstream status |
|---|---|---|---|
| 0001 | **QSA cute fallback**: when a QSA call falls outside the sm_121 KDA kernel's contract, it falls back to the `flash_attn.cute` varlen kernel instead of raising, and k/v are cast to q's dtype. **pynccl**: `parallel_state` collectives use pynccl even when it is in the "disabled" state outside CUDA graphs. | Both were carried over from our bring-up on 2026-09-09 `main`, where the server would not run without them. **We have not re-checked whether current `main` still needs them**, so drop them first when you rebase. | Not upstreamed |
| 0002–0005 | **sm_121 skinny BF16 GEMM**: vLLM's CuTe-DSL kernel plus plan table, hooked into the unquantized linears and the lm_head. Plans are compiled before CUDA graph capture. 0002 also contains the **TP-aware hot-token draft head**. | cuBLAS on GB10 runs decode-sized BF16 GEMMs with SM80 WMMA kernels at 125–205 of 273 GB/s. See [docs/decisions.md](../docs/decisions.md). | Cleaned-up versions: [sglang#43250](https://github.com/sgl-project/sglang/pull/43250) (GEMM) and [sglang#43249](https://github.com/sgl-project/sglang/pull/43249) (hot-token head). The PR version pads short rank blocks with distinct cold ids; the production version repeats the last id. With a balanced table (ours is 32768/32768) the two are identical. |
| 0006–0008 | **PLE static reuse**: after the first full write, `msync`+`fsync`, then write a sidecar fingerprint. Later boots that match the fingerprint map the table **read-only** and check every shard against the checkpoint (sampled 4 KiB windows) before use. An engine-only change (SGLang version, loader code) verifies and keeps the table; a model-side change rebuilds it. Enabled with `SGLANG_QWEN4_PLE_REUSE=1`. | Each boot otherwise rewrote 25.6 GB per node, about 27–29 GB of SSD writes per restart. A writable mapping also kept pages dirty for hours after load. | Related: [sglang#42400](https://github.com/sgl-project/sglang/pull/42400) (single-node copy skip); [our TP=2 report](https://github.com/sgl-project/sglang/pull/42400#issuecomment-6071960442) |
| 0009 | **QSA multi-group pending ring**: the per-request index-key ring holds `ceil((draft_tokens + 3) / 4)` compression groups instead of one. The same slot formula is used in the host builders, the KV pool and the CUDA-graph Triton kernel. `SGLANG_QSA_RING_GROUPS=1` restores the stock layout. | Stock SGLang rejects more than 4 draft tokens on this model, and at 4 it overwrites committed keys whenever a verify window is not aligned to the group ([docs/decisions.md](../docs/decisions.md)). With the patch, `--speculative-adaptive` starts and the ring is correct at every depth. | Not upstreamed. Same idea as the draft [sglang#40143](https://github.com/sgl-project/sglang/pull/40143), with one more group; that PR's sizing still aliases in our replay. |

The sampled-window check in 0006 follows `shantanugoel/ple_reuse.py` (Apache-2.0).

**When you rebase onto newer SGLang**, apply them one at a time and run the gates in [docs/validation.md](../docs/validation.md) after each one. If you only have time for one patch, 0006–0008 save the most SSD wear, and 0002–0005 give the most speed on prose. 0009 with `--speculative-adaptive` gives the most on structured output. After rebasing 0009, run [tools/qsa_ring](../tools/qsa_ring) before starting a server.
