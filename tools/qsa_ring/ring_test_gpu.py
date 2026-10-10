"""GPU check: the CUDA-graph Triton row-metadata kernel must produce the same pending-ring
slots and group member slots as the host builders, for decode rows (MODE 0) and
target-verify fan-out rows (MODE 1), at ring group counts 1, 2, 3 and 5."""
import sys, torch
from types import SimpleNamespace
from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata
from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata, build_group_ring_slots, build_pending_ring_slots)

device = "cuda"; ratio, full_page, max_pages = 4, 64, 129
bad = 0
for num_groups in (1, 2, 3, 5):
    pool = SimpleNamespace(qsa_compress_ratio=ratio, qsa_compressed_page_size=full_page // ratio,
                           qsa_block_topk=512, qsa_num_groups=num_groups)
    rows_t = torch.arange(8, dtype=torch.int32, device=device)[:, None]
    cols = torch.arange(max_pages * full_page, dtype=torch.int32, device=device)[None, :]
    req_to_token = ((rows_t * max_pages + cols // full_page) * full_page + cols % full_page).contiguous()
    for mode, extend_len, seq_lens_list in [(0, 0, [5, 64, 127, 130, 1000, 3]), (1, 8, [5, 64, 127, 130, 1000, 3]), (1, 4, [9, 10, 11, 12])]:
        bs = len(seq_lens_list); num_rows = bs * max(1, extend_len)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int32, device=device)  # slot 0 never allocated
        z = lambda *shape, dt=torch.int32: torch.zeros(shape, dtype=dt, device=device)
        indexer = QSAIndexerMetadata(
            sequence_lengths=z(num_rows), token_to_batch_idx=torch.arange(num_rows, dtype=torch.int32, device=device),
            token_slot_table=z(num_rows, 1), out_cache_loc=z(num_rows, dt=torch.int64), token_to_kv_pool=pool,
            compress_ratio=ratio, block_topk=512, req_pool_indices=z(num_rows), is_cuda_graph=True,
            graph_write_locs=z(num_rows), graph_compressed_page_table=z(num_rows, max_pages),
            graph_compressed_lengths=z(num_rows), graph_prefix_lengths=z(num_rows),
            decode_logical_positions=z(num_rows), pending_ring_slots=z(num_rows, dt=torch.int64),
            graph_ring_group_locs=z(num_rows, ratio))
        metadata = SimpleNamespace(indexer_metadata=indexer, sequence_lengths=indexer.sequence_lengths,
                                   row_req_pool_indices=z(num_rows))
        launch_graph_metadata(mode=mode, bs=bs, num_rows=num_rows, seq_lens=seq_lens, req_pool_indices=req_pool_indices,
                              extend_lens=None, extend_len=extend_len, num_padding=0, metadata=metadata,
                              req_to_token=req_to_token, pool=pool, seq_offset=1 if mode == 0 else 0)
        torch.cuda.synchronize()
        # host reference from the row layout the kernel wrote (row lengths and row requests)
        row_lens = metadata.sequence_lengths.long(); row_reqs = metadata.row_req_pool_indices.long()
        current = (row_lens - 1).clamp_min(0)
        ref_slots = build_pending_ring_slots(token_to_batch_idx=torch.arange(num_rows, device=device),
                                             req_pool_indices=row_reqs, sequence_lengths=row_lens,
                                             logical_positions=current, compress_ratio=ratio, is_extend=False,
                                             num_groups=num_groups)
        ref_groups = build_group_ring_slots(req_pool_indices=row_reqs, group_end_positions=current,
                                            sequence_ids=torch.arange(num_rows, device=device), compress_ratio=ratio,
                                            num_groups=num_groups)
        ok1 = torch.equal(indexer.pending_ring_slots.long(), ref_slots)
        ok2 = torch.equal(indexer.graph_ring_group_locs.long(), ref_groups)
        # layout sanity for verify fan-out: row k of request r has length base + k + 1
        if mode == 1:
            exp = (seq_lens.long()[:, None] + torch.arange(extend_len, device=device)[None, :] + 1).flatten()
            ok3 = torch.equal(row_lens, exp)
        else:
            ok3 = torch.equal(row_lens, seq_lens.long() + 1)
        span = ratio * num_groups
        in_range = bool(((indexer.pending_ring_slots.long() >= row_reqs * span) & (indexer.pending_ring_slots.long() < (row_reqs + 1) * span)).all())
        print(f"groups={num_groups} mode={mode} extend_len={extend_len}: slots {'ok' if ok1 else 'MISMATCH'}, "
              f"group locs {'ok' if ok2 else 'MISMATCH'}, layout {'ok' if ok3 else 'MISMATCH'}, in-range {in_range}")
        bad += (not ok1) + (not ok2) + (not ok3) + (not in_range)
print("ALL OK" if bad == 0 else f"{bad} FAILURES"); sys.exit(bad)
