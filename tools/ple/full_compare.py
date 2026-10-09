"""Byte-exact comparison of this rank's on-disk PLE table against the checkpoint's n-gram shards (reads only).
usage: full_compare.py <table.bin> <model_dir> <tp_rank> <tp_size>"""
import sys, os, re, json, struct, time, numpy as np, torch
from safetensors import safe_open
table_path, model, rank, tp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ROW = 160
rows_total = os.path.getsize(table_path) // ROW
tab = np.memmap(table_path, dtype=np.uint8, mode="r", shape=(rows_total, ROW))
shards = {}  # idx -> (file, key, nrows)
for n in sorted(os.listdir(model)):
    if not n.endswith(".safetensors"): continue
    with open(os.path.join(model, n), "rb") as f:
        L = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(L))
    for k, v in h.items():
        m = re.search(r"ngram_embedding\.shard_(\d+)\.weight$", k)
        if m: shards[int(m.group(1))] = (n, k, v["shape"][0])
idx = sorted(shards); per_rank = len(idx) // tp
mine = idx[rank * per_rank:(rank + 1) * per_rank]
print(f"{len(idx)} shards in checkpoint, rank {rank}/{tp} owns shards {mine[0]}..{mine[-1]}, table rows {rows_total}")
t0 = time.time(); ok = bad = 0; off = 0; opened = {}
for i in mine:
    n, k, nr = shards[i]
    f = opened.get(n) or opened.setdefault(n, safe_open(os.path.join(model, n), "pt", device="cpu"))
    src = f.get_tensor(k).view(torch.uint8).numpy()
    dst = np.asarray(tab[off:off + nr])
    same = dst.shape == src.shape and np.array_equal(dst, src)
    if same: ok += 1
    else:
        bad += 1
        nbad = int((dst != src).sum()) if dst.shape == src.shape else -1
        print(f"  MISMATCH shard_{i} rows {off}-{off+nr}: {nbad} differing bytes")
    off += nr
print(f"compared {ok+bad} shards ({off} rows, {off*ROW/2**30:.1f} GiB) in {time.time()-t0:.0f}s: {ok} identical, {bad} mismatched, rows covered {off}/{rows_total}")
print("RESULT:", "TABLE IDENTICAL TO CHECKPOINT" if bad == 0 and off == rows_total else "TABLE DIFFERS")
