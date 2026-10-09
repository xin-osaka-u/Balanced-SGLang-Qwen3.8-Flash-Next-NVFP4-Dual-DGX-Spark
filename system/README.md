# Host settings (DGX OS 7.x, both nodes)

Small, reversible changes. Each one is here because we measured a problem first.

| File | Install to | Why | Revert |
|---|---|---|---|
| `99-gb10-stability.conf` | `/etc/sysctl.d/` then `sysctl --system` | Unified memory: keep 512 MB free for the kernel and swap less. During early bring-up the box hung when the model plus JIT compiles exhausted UMA. | delete the file |
| `99-no-disk-pkgcache` | `/etc/apt/apt.conf.d/` | The DGX Dashboard checks for updates every hour (hardcoded). Each check rewrote ~126 MB of apt `pkgcache.bin`/`srcpkgcache.bin`, about **7 GB/day per node** of pure SSD wear. With this file the caches stay in memory: an 8-min window that included a check went from ~290 MB to 8–20 MB written. | delete the file |

Logging. Not shipped as files, because they depend on what else runs on your boxes:

- **Keep one persistent journal.** It lives in `/var/log/journal` (the DGX OS default). Use `journalctl -b -1 -e` after a crash or power loss, and the kernel panic record is in `/var/lib/systemd/pstore`.
- **We disabled rsyslog** with `systemctl disable --now rsyslog.service syslog.socket`. It wrote the same lines a second time to `/var/log/syslog`.
- **Find chatty services before guessing.** On our boxes about 100 KB was written per log line, so one line every 5 s (about 17,000 a day) already costs about 1.7 GB/day. Use `journalctl --since -1h -o cat | sort | uniq -c | sort -rn | head`. For services that don't need debug output, add a drop-in with `LogLevelMax=notice`.
- **Check the dashboard's SMART polling interval.** If it polls `smartctl` every 60 s, each poll writes log lines; we moved ours to 600 s.

After all of this, idle writes on our nodes are about 2–6 GB/day. Much of that comes from other services we run on the same boxes. Measure with:

```bash
awk '$3=="nvme0n1"{print $10*512/1e6 " MB written since boot"}' /proc/diskstats
```

Firmware: we install only what NVIDIA pushes through the official DGX OS OTA channel, and rarely. No off-channel drivers, kernels or CUDA.
