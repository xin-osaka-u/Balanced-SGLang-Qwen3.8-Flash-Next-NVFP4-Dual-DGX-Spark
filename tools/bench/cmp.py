"""Compare two tagged output sets. Usage: python cmp.py <dir> <tagA> <tagB>"""
import sys
d, a, b = sys.argv[1:4]
for k in ("cn_lease", "cn_bills", "jp_invoice", "jp_school", "jp_summary"):
    x = open(f"{d}/{a}_{k}.txt").read(); y = open(f"{d}/{b}_{k}.txt").read()
    if x == y:
        print(f"{a} vs {b} {k:11s} IDENTICAL ({len(x)} chars)"); continue
    i = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    print(f"{a} vs {b} {k:11s} differ at char {i}/{max(len(x), len(y))}")
    print(f"    {a}: {x[max(0, i-40):i+40]!r}\n    {b}: {y[max(0, i-40):i+40]!r}")
