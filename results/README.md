# Raw results

- `gsm8k_metrics.json`: sgl-eval metrics for the full GSM8K run (author protocol), production config, 2026-10-09.
- `gsm8k_metrics_ringfix_fixed4.json`, `gsm8k_metrics_adaptive.json`: the same protocol on 2026-10-10, with patch 0009 at fixed 3/1/4 and with adaptive depth (current production).
- `ruler2_32k_stock_ring.json`, `ruler2_32k_ringfix_fixed4.json`, `ruler2_32k_adaptive.json`: RULER2 at 32k on the same code with 1, 2 and 3 ring groups.
- `nonthinking_snapshot.txt`: lm-eval 0.4.13 summary of the non-thinking snapshot, the 2026-10-09 config, same night. Not re-run on the adaptive config.

Per-question outputs are not included (size). Re-create them with `tools/eval/run_gsm8k.sh`.
