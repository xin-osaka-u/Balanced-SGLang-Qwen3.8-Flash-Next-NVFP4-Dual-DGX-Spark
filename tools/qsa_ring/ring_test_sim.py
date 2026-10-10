"""Offline correctness check for the QSA pending index-K ring under speculative verify windows.

Replays the real store/compress order of a speculative-paged forward with the real
slot builders from ``sglang.srt.layers.attention.qsa.metadata``:
  1. every position of the verify window is scattered into the ring,
  2. every row whose length completes a group averages that group's members from the ring,
  3. a random prefix of the window is accepted; the rest is re-generated next cycle.
A compressed key is correct when it equals the mean of the member keys the model
produced for those positions *in that forward*. Positions that are later rejected are
re-generated with new keys and their group is re-compressed by the row that closes it.
"""
import os, sys, random
import torch
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots, build_pending_ring_slots, pending_ring_slot,
)

RATIO = 4


def groups_needed(draft, ratio=RATIO):
    return 1 if draft <= 1 else -(-(draft + ratio - 1) // ratio)


def simulate(draft, num_groups, cycles=400, seed=0, req_slots=(1, 2, 3)):
    rng = random.Random(seed)
    span = RATIO * num_groups
    ring = torch.zeros(( (max(req_slots) + 1) * span, 1), dtype=torch.float64)  # 1-dim "key"
    errors = 0; checks = 0
    for req in req_slots:
        seq_len = 0
        pre = rng.randint(1, 11)   # plain-decode cycles before speculation starts
        key_of = {}                      # committed position -> key
        compressed = {}                  # group index -> key written last
        gen_counter = 0
        # pre-commit seq_len tokens one by one through plain decode (window 1)
        for cyc in range(cycles):
            width = draft if cyc >= pre else 1
            base = seq_len
            positions = torch.arange(base, base + width, dtype=torch.long)
            # fresh keys for every window position (a rejected position gets a new key next time)
            window_keys = {}
            for p in positions.tolist():
                gen_counter += 1
                window_keys[p] = float(req * 1e6 + gen_counter)
            reqs = torch.full((width,), req, dtype=torch.long)
            lengths = positions + 1
            # 1. scatter the window into the ring (verify: is_extend=False)
            slots = build_pending_ring_slots(
                token_to_batch_idx=torch.arange(width), req_pool_indices=reqs,
                sequence_lengths=lengths, logical_positions=positions,
                compress_ratio=RATIO, is_extend=False, num_groups=num_groups)
            for s, p in zip(slots.tolist(), positions.tolist()):
                ring[s, 0] = window_keys[p]
            # 2. rows that complete a group compress it from the ring
            for row, p in enumerate(positions.tolist()):
                if (p + 1) % RATIO != 0:
                    continue
                locs = build_group_ring_slots(
                    req_pool_indices=reqs, group_end_positions=torch.tensor([p]),
                    sequence_ids=torch.tensor([row]), compress_ratio=RATIO,
                    num_groups=num_groups)[0]
                got = ring[locs, 0].mean().item()
                members = list(range(p - RATIO + 1, p + 1))
                truth = sum(window_keys[m] if m in window_keys else key_of[m] for m in members) / RATIO
                checks += 1
                if abs(got - truth) > 1e-6:
                    errors += 1
                compressed[p // RATIO] = got
            # 3. accept a random prefix (at least the first token)
            accepted = rng.randint(1, width)
            for p in positions.tolist()[:accepted]:
                key_of[p] = window_keys[p]
            seq_len = base + accepted
        # final: every fully committed group must hold the mean of its committed keys
        for g in range(seq_len // RATIO):
            members = range(g * RATIO, g * RATIO + RATIO)
            if all(m in key_of for m in members) and g in compressed:
                checks += 1
                truth = sum(key_of[m] for m in members) / RATIO
                if abs(compressed[g] - truth) > 1e-6:
                    errors += 1
    return errors, checks


if __name__ == "__main__":
    bad = 0
    for draft, num_groups, expect_ok in [
        (1, 1, True),            # plain decode, historical layout
        (4, 1, False),           # today's production: W4 with one group aliases
        (4, 2, True),            # ring fix at W4
        (8, 2, False),           # upstream PR formula ceil(D/ratio) is too small
        (8, 3, True),            # corrected formula
        (6, 3, True),
        (12, 4, True),
        (16, 5, True),
    ]:
        errors, checks = simulate(draft, num_groups)
        ok = errors == 0
        verdict = "OK" if ok == expect_ok else "UNEXPECTED"
        if verdict != "OK": bad += 1
        print(f"draft={draft:2d} groups={num_groups} (formula says {groups_needed(draft)}): "
              f"{errors:4d} wrong of {checks:5d} compressed keys -> {'correct' if ok else 'ALIASED'}  [{verdict}]")
    # slot arithmetic sanity: N=1 reproduces the historical formula
    pos = torch.arange(0, 64); req = torch.full_like(pos, 3)
    assert torch.equal(pending_ring_slot(req, pos, compress_ratio=4, num_groups=1), req * 4 + pos % 4)
    print("N=1 slot formula matches the historical layout")
    print("ALL OK" if bad == 0 else f"{bad} UNEXPECTED RESULTS")
    sys.exit(bad)
