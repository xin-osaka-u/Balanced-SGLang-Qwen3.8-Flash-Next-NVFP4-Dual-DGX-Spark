# Operations: running it for weeks

## Starting and restarting

```bash
scripts/launch_tp2.sh          # on rank 0; starts rank 1 over ssh
curl -s localhost:8000/health  # first boot ~9 min, reuse boots ~8 min
tools/bench/run_window.sh boot-$(date +%m%d)   # gates 1 + 2 after every restart
```

- **Restarts are the main source of SSD wear.** Without PLE reuse, each one writes about 27–29 GB per node. With reuse, about 2–3 GB. Don't restart on a schedule.
- **The first boot after an engine or flashinfer upgrade JIT-compiles kernels.** Do it with the model *not* loaded and with `MAX_JOBS=4` (see [decisions.md](decisions.md)).
- **After a power loss**, check the following before starting:
  1. The direct link is up on both nodes. `ip -br link` shows NO-CARRIER on one side while the other box is still off or booting.
  2. GPU clocks are healthy. See the latch check below.
  3. `journalctl -b -1 -e` and `/var/lib/systemd/pstore` for what happened.

## Failure modes we have actually hit

| Symptom | Cause | How we found it | Fix |
|---|---|---|---|
| Everything about 1.5× slower; rank 0 spends half its time waiting in NCCL | **GB10 clock latch** on rank 1: stuck at 611 MHz / 5.5 W *under load*, no throttle reason reported, and `nvidia-smi -rgc/-lgc/-rmc` do nothing | Per-rank profile showed every rank-1 kernel 1.5× slower; then `nvidia-smi` under load | Shut down both boxes, unplug for a few minutes, power on. Afterwards: 2377 MHz / 84 W under load. **Check after every boot:** a 15 s fp16 matmul should reach about 80 W and 2.2 GHz or more. The governor below alarms on this automatically. |
| Weight loading stalls at 97% for more than 10 min, with about 39 MB/s of random IO | Rewriting an already-populated PLE file in place (without the reuse patches) | IO pressure in `/proc/pressure/io`, major faults | Use the reuse patches. Without them, delete `ple_table_*.bin` before each boot. |
| Scheduler OOM-killed on the first start after an upgrade | flashinfer JIT with ~20 parallel nvcc jobs while the model held ~90 GB | `dmesg`, journal | Warm kernels with no model loaded; `MAX_JOBS=4` |
| Server dies at start with a device-side assert in `init_lm_head` | `--speculative-token-map` at TP>1 on unpatched SGLang | Traceback | Patch 0002 / [sglang#43249](https://github.com/sgl-project/sglang/pull/43249) |
| Startup fails with deeper drafts (7/8 draft tokens) | QSA requires `num_draft_tokens <= 4` | Startup error | Stay at NEXTN 3/1/4 |
| Worker kernel panic (`BUG mm/slub.c`, freeing a network skb in softirq), once | Unknown. At the time the box was also copying from a USB HDD. | `/var/lib/systemd/pstore` | Rebooted. Not reproduced since; we are watching for a second occurrence. |

## Idle power: GPU clock governor (optional)

[ops/gpu-governor](../ops/gpu-governor) is a small stdlib-only service.

**What it does:**
- After 300 s with no GPU activity, it caps clocks on both nodes (`nvidia-smi -lgc 210,1000`), and releases them immediately on activity.
- It also raises an alarm if a node can't reach 30 W while busy, which is exactly the latch above.

**Verified so far:**
- Capped idle draws 5.7–6.3 W instead of 11–12 W per node, and runs about 3 °C cooler.
- A dry run on both nodes worked: synchronized idle and active switching, wake on `POST /wake`, release on exit.

**Not yet verified:** it has not run in production yet. Start it with `GOV_DRY_RUN=1` and watch at least one of your traffic cycles before letting it touch clocks.

The saving is about 0.1 kWh/day per node (about 6 W for about 16 idle hours a day). It is small, but the latch alarm alone justifies running it.

## SSD wear budget

Measure:

```bash
sudo smartctl -A /dev/nvme0 | grep -E "Units Written|Percentage Used|Unsafe|Media"
```

| Source of writes | Per node |
|---|---|
| Qwen restart without reuse | 27–29 GB |
| Qwen restart with reuse | 2–3 GB |
| hourly DGX Dashboard update check (before our apt fix) | ~7 GB/day |
| journald, before trimming chatty services | ~1.7 GB/day |
| our idle total now | 2–6 GB/day (mostly other apps) |

At these rates a 4 TB drive's rated endurance is many years away. The point is to avoid paying 30 GB for nothing on every restart.

## Logs worth keeping

- **One persistent journal** (the DGX OS default under `/var/log/journal`). We turned rsyslog off because it duplicated the same lines.
- **The server logs** (`$LOG_DIR/rank0.log`, `rank1.log`). The `Decode batch` lines include `accept len` and `gen throughput`, the quickest health signal you have.
