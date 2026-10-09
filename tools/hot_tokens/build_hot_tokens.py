"""Build a reduced ("hot") draft vocabulary for NEXTN/MTP: --speculative-token-map.

The MTP draft head multiplies by the full lm_head (248k rows) on every draft step.
Restricting it to the tokens you actually generate makes each draft step cheaper
without changing what the target model accepts: the target still verifies every
token, so output quality is unaffected. A token outside the table simply cannot
be drafted (that step falls back to the target's own token).

score(token) = sum of wordfreq frequencies of the words it appears in (zh, ja, en)
             + structural tokens (digits, JSON punctuation, whitespace)
             + optional: your own corpus (--corpus), weighted heavily
Special tokens are always kept. Under TP the lm_head is vocab-sharded, so we keep
the top PER_RANK ids inside each rank's shard; that keeps the shards balanced
(no padding) and the sliced draft head the same size on every rank.

We built our production table with a private traffic corpus on top of this. The
public version below (wordfreq only) shares 94.2% of its ids with our production
table; it is what we recommend starting from; measure
acceptance length on your own prompts before and after.

usage: python build_hot_tokens.py --tokenizer MODEL/tokenizer.json --out hot.pt
         [--per-rank 32768] [--tp 2] [--lm-head-rows 248320] [--corpus my.txt]
"""
import argparse
import collections
import json

import torch
from tokenizers import Tokenizer
from wordfreq import get_frequency_dict

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-rank", type=int, default=32768)
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--lm-head-rows", type=int, default=248320,
                help="lm_head rows in the checkpoint (text_config.vocab_size, padded)")
ap.add_argument("--corpus", help="optional text file of your own prompts/outputs")
ap.add_argument("--corpus-weight", type=float, default=5.0)
a = ap.parse_args()

tok = Tokenizer.from_file(a.tokenizer)
V = tok.get_vocab_size(with_added_tokens=True)
specials = [t["id"] for t in json.load(open(a.tokenizer))["added_tokens"]]
PART = a.lm_head_rows // a.tp
score = collections.defaultdict(float)


def add_words(lang, n, variants):
    d = get_frequency_dict(lang)
    for w, f in sorted(d.items(), key=lambda kv: -kv[1])[:n]:
        for v in variants(w):
            for i in tok.encode(v, add_special_tokens=False).ids:
                score[i] += f


add_words("zh", 150000, lambda w: [w])
add_words("ja", 150000, lambda w: [w])
add_words("en", 100000, lambda w: [w, " " + w, " " + w.capitalize()])

if a.corpus:
    cnt = collections.Counter(
        tok.encode(open(a.corpus, errors="ignore").read(), add_special_tokens=False).ids)
    tot = sum(cnt.values()) or 1
    for i, c in cnt.items():
        score[i] += a.corpus_weight * c / tot

structural = list("0123456789{}[]:,\"'()<>=+-*/\\|_#@%&!?;.\n\t ") + \
    [" " * k for k in (2, 4, 8)] + ["\n\n", "```", "json", "```json"]
for s in structural:
    for i in tok.encode(s, add_special_tokens=False).ids:
        score[i] += 1.0
for i in specials:
    score[i] += 100.0

hot = []
for r in range(a.tp):
    lo, hi = r * PART, min((r + 1) * PART, V)
    cand = sorted(range(lo, hi), key=lambda i: -score.get(i, 0.0))
    keep = sorted(cand[: a.per_rank])
    print(f"rank {r}: ids {lo}-{hi - 1}, kept {len(keep)}, "
          f"with nonzero score {sum(1 for i in keep if score.get(i, 0) > 0)}")
    hot += keep
torch.save(hot, a.out)
print("saved", a.out, len(hot))
