# gpu-governor: dynamic GPU clock cap for the dual DGX Spark (GB10) SGLang cluster

Status: step 0 verified (capped idle 5.7-6.3 W vs 11-12 W per node) and dry-run tested on both nodes;
**not yet running in production**. Deploy in dry-run mode first. Our node names below are `spark1`
(master, rank 0) and `spark2` (worker, rank 1); addresses are the direct-link IPs.

## What it does
- Model stays loaded, SGLang untouched. Only `nvidia-smi -lgc/-rgc` moves.
- Idle (no GPU activity for `GOV_IDLE_AFTER_S`, default 300 s) -> `-lgc GOV_IDLE_CLOCKS` (default `210,1000`).
- Any activity (`utilization.gpu >= 5 %`, or `POST /wake`) -> `-rgc` immediately. Optional thermal cap
  while active via `GOV_ACTIVE_CLOCKS=300,2200` (leave empty unless the cap benchmark says it pays).
- Hysteresis: release is instant, re-cap only after a full idle period; the cap is applied once, not every poll.
- Master (spark1) decides and pushes the mode to the worker (spark2) every 2 s; both nodes switch together.
- Fail-safe: SIGTERM/exception/`nvidia-smi` failures/worker losing the master for 90 s -> `-rgc`.
  The unit also has `ExecStopPost=nvidia-smi -rgc`.
- Self-check / latch detector: when active and busy (util >= 50 % for 3 samples) power must reach
  `GOV_POWER_OK_W` (30 W), else ALARM in the journal, `/status`, `/health` (503) and `GOV_ALERT_CMD`.
  A latched GB10 (spark2 on 2026-10-06: 611 MHz, 5.5 W) trips exactly this.
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
   Decide `GOV_ACTIVE_CLOCKS` (probably empty) and confirm `GOV_IDLE_CLOCKS` is accepted by the driver.
2. Install with `GOV_DRY_RUN=1` on both, watch `/status` and the CSV through at least one of your regular traffic cycles.
3. Flip `GOV_DRY_RUN=0`, `systemctl restart gpu-governor` on both. Check a request after >5 min idle:
   `/status` goes idle -> active, TTFT acceptable, power rises under load (no ALARM).

## Known traffic pattern (2026-10-08 master log, 20 h)
126 prefills; median gap 18 s inside bursts, p90 25 min; ~83 % of the day is idle gaps longer than 5 min,
with an hourly request around :51. So the cap would be on ~16 h/day and released about once an hour.

## Dry-run result (2026-10-08 21:39, no clocks touched)
Master on spark1 with `GOV_IDLE_AFTER_S=20`, worker on spark2: idle after 20 s on both, `/wake` released
both within one poll, wrong token refused, shutdown pushed release to the worker. (Dry-run logs not included.)
