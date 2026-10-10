# gpu-governor: dynamic GPU clock cap for the dual DGX Spark (GB10) SGLang cluster

Status: **running in production on both nodes since 2026-10-09**, with `GOV_IDLE_CLOCKS=210,1000`
and `GOV_ACTIVE_CLOCKS=300,2200`. Deploy in dry-run mode first (the env files ship with
`GOV_DRY_RUN=1` and no active cap). Our node names below are `spark1` (master, rank 0) and `spark2`
(worker, rank 1); addresses are the direct-link IPs.

## What it does
- Model stays loaded, SGLang untouched. Only `nvidia-smi -lgc/-rgc` moves.
- Idle (no GPU activity for `GOV_IDLE_AFTER_S`, default 300 s) -> `-lgc GOV_IDLE_CLOCKS` (default `210,1000`).
- Any activity (`utilization.gpu >= 5 %`, or `POST /wake`) -> `-rgc` immediately. Optional thermal cap
  while active via `GOV_ACTIVE_CLOCKS=300,2200` (leave empty unless the cap benchmark says it pays).
- Hysteresis: release is instant, re-cap only after a full idle period; the cap is applied once, not every poll.
- Master (spark1) decides and pushes the mode to the worker (spark2) every 2 s; both nodes switch together.
- Fail-safe: SIGTERM/exception/`nvidia-smi` failures/worker losing the master for 90 s -> `-rgc`.
  The unit also has `ExecStopPost=nvidia-smi -rgc`.
- Self-check / latch detector: when active and busy (util >= 50 % for 3 samples) the clock must reach
  `GOV_BUSY_MIN_CLOCK_MHZ` (1500), else ALARM in the journal, `/status`, `/health` (503) and `GOV_ALERT_CMD`.
  A latched GB10 (spark2 on 2026-10-06: 611 MHz, 5.5 W) trips this. The first version checked power
  (`GOV_POWER_OK_W=30`) and false-alarmed on normal decode, which draws only about 14 W; the power
  check is now off by default (`0`).
- Logs every 10 s and on every transition to `/var/log/gpu-governor/samples.csv` (rotated, 20 MB x 5):
  mode, clock, power, temp, util, pstate, event reasons, idle seconds, alarm.
- `GOV_DRY_RUN=1` observes and logs decisions without touching clocks (the env files ship with it on).

## Files
| file | purpose |
|---|---|
| `gpu_governor.py` | the service, stdlib only, both roles |
| `gpu-governor.service` | systemd unit (root, `/opt/gpu-governor`, `/etc/gpu-governor.env`) |
| `gpu-governor.env.master/.worker` | per-node config; set the same `GOV_TOKEN` on both |
| `install.sh` | `sudo ./install.sh master|worker [--start]` on each node; idempotent |
| `step0_idle_cap_test.sh` | step 0: idle power/temp uncapped vs capped, one node, needs sudo |
| `cap_bench.sh` | throughput/TTFT/temp under active caps on both nodes, uses `run_window.sh` |
| `test_governor.py` | offline state-machine test (`python3 test_governor.py` -> `ALL OK`) |
| `dry/` | 2026-10-08 dry-run logs from both nodes |

## HTTP (bind to the 192.168.200.x link only)
- `GET /status` JSON (mode, sample, peers, alarm); `GET /health` 200/503.
- `POST /wake` (master; header `X-Gov-Token`) releases clocks and resets the idle timer. A proxy can call
  it before forwarding a request to hide the ramp-up.
- `POST /state` (worker; only from `GOV_MASTER`'s IP) is the heartbeat the master sends.

## Order of operations after maintenance
0. Both nodes: `sudo ./step0_idle_cap_test.sh` with no traffic. Capped idle must draw less power / run
   cooler than uncapped idle. If not, stop here and find what keeps P0.
1. `./cap_bench.sh "none 2200 1800"` from spark1 (both nodes capped together): decode tok/s, TTFT, peak temp.
   Decide `GOV_ACTIVE_CLOCKS` and confirm `GOV_IDLE_CLOCKS` is accepted by the driver. Our result is below.
2. Install with `GOV_DRY_RUN=1` on both, watch `/status` and the CSV through at least one of your regular traffic cycles.
3. Flip `GOV_DRY_RUN=0`, `systemctl restart gpu-governor` on both. Check a request after >5 min idle:
   `/status` goes idle -> active, TTFT acceptable, power rises under load (no ALARM).

## Cap benchmark (2026-10-09, both nodes capped together, fixed 3/1/4 drafts, one boot)
| Active cap | Short JSON | CN prose | JP prose | GPU power while decoding (rank 0 / rank 1) | Peak temp |
|---|---|---|---|---|---|
| none | 78.4 tok/s | 41.7 | 41.7 | 19.1 / 24.0 W | 63 / 62 °C |
| 2200 MHz | 75.5 | 40.1 | 39.7 | 12.5 / 12.7 W | 61 / 56 °C |
| 1800 MHz | 74.8 | 37.3 | 37.6 | 9.6 / 9.8 W | 58 / 53 °C |

We chose 2200: about 4% slower for about 40% less GPU power under decode. Prose numbers vary by a few
percent between runs, so treat the speed column as approximate. Use the wrapper without the torch
profiler for this benchmark: a profiler run left several GB of heap in the SGLang scheduler until restart.

## Known traffic pattern (2026-10-08 master log, 20 h)
126 prefills; median gap 18 s inside bursts, p90 25 min; ~83 % of the day is idle gaps longer than 5 min,
with an hourly request around :51. So the cap would be on ~16 h/day and released about once an hour.

## Dry-run result (2026-10-08 21:39, no clocks touched)
Master on spark1 with `GOV_IDLE_AFTER_S=20`, worker on spark2: idle after 20 s on both, `/wake` released
both within one poll, wrong token refused, shutdown pushed release to the worker. (Dry-run logs not included.)
