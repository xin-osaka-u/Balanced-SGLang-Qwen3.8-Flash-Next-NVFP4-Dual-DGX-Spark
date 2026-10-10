# QSA pending-ring checks (patch 0009)

Both scripts need the patched SGLang importable. Neither needs a running server.

- `ring_test_sim.py` (CPU): replays speculative verify windows with the real slot builders and
  counts compressed index keys that do not equal the mean of their true members. It shows the
  stock one-group layout aliasing, and the patched sizing `N = ceil((draft_tokens + 3) / 4)` clean.
- `ring_test_gpu.py` (CUDA, a few tens of MB): checks that the CUDA-graph Triton row-metadata
  kernel produces the same slots as the host builders at 1, 2, 3 and 5 groups.

Run both after rebasing patch 0009. A wrong kernel passes every one-group test and silently
corrupts the index keys at more groups, so these two are the only offline guard.
