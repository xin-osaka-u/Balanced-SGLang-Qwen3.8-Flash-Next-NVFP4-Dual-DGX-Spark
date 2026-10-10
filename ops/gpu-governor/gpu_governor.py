#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dynamic GPU clock governor for a two-node DGX Spark (GB10) SGLang TP=2 cluster.

Goal: lower GPU temperature and idle power without touching the inference
engine. The model stays loaded, SGLang keeps running; only the clock *cap*
moves:

  idle   (no requests for GOV_IDLE_AFTER_S)  ->  nvidia-smi -lgc GOV_IDLE_CLOCKS
  active (any GPU activity or a /wake call)  ->  nvidia-smi -rgc
                                                  (or -lgc GOV_ACTIVE_CLOCKS, a
                                                  thermal cap, if configured)

One process per node, same program, two roles:

  master  samples its own GPU, decides idle/active, applies locally and pushes
          the decision to every peer in GOV_PEERS every poll (heartbeat).
          Exposes POST /wake so a proxy can release clocks before forwarding
          a request, and GET /status.
  worker  applies whatever the master says (only from GOV_MASTER's address),
          samples its own GPU for the log and the power self-check, and has a
          watchdog: no heartbeat for GOV_WATCHDOG_S -> nvidia-smi -rgc.

Fail-safe everywhere: any exit path (SIGTERM, exception, watchdog, failed
nvidia-smi call) ends with the clocks released. Deploy the systemd unit with
ExecStopPost=nvidia-smi -rgc as a second belt.

Self-check (also a hardware clock-latch detector): when active and the GPU is
busy (utilization >= GOV_BUSY_UTIL_PCT for GOV_BUSY_SAMPLES consecutive
samples), clocks.gr must reach GOV_BUSY_MIN_CLOCK_MHZ (and power.draw GOV_POWER_OK_W when set); otherwise an ALARM is logged,
GOV_ALERT_CMD is run (message in $GOV_ALERT_MSG) and /status shows it. A GB10
whose EC latched the clock at ~600 MHz shows exactly this signature (busy, a
few watts) and nvidia-smi cannot fix it; only a power drain can.

Stdlib only (Python 3.10+). GOV_DRY_RUN=1 samples and logs decisions but never
calls nvidia-smi -lgc/-rgc; use it to observe before trusting it with clocks.
"""
from __future__ import annotations

import csv
import json
import logging
import logging.handlers
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

log = logging.getLogger("gpu-governor")

NVIDIA_SMI = os.environ.get("GOV_NVIDIA_SMI", "/usr/bin/nvidia-smi")
QUERY_FIELDS = "clocks.gr,power.draw,temperature.gpu,utilization.gpu,pstate,clocks_event_reasons.active"


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    role: str = "master"                 # master | worker
    listen: str = "127.0.0.1:9870"       # ip:port for the HTTP API (bind to the cluster link, not 0.0.0.0)
    peers: List[str] = field(default_factory=list)  # master: ["192.168.200.2:9870"]
    master: str = ""                     # worker: master's IP; state posts from other IPs are refused
    token: str = ""                      # optional shared secret (header X-Gov-Token)
    poll_s: float = 2.0
    idle_after_s: float = 300.0          # idle this long -> cap
    active_util_pct: int = 5             # utilization >= this is activity
    idle_clocks: str = "210,1000"        # nvidia-smi -lgc argument when idle
    active_clocks: str = ""              # "" -> -rgc; e.g. "300,2200" for a thermal cap
    watchdog_s: float = 90.0             # worker: release if the master is silent this long
    power_ok_w: float = 0.0              # >0: a busy GPU must draw at least this (0 = off; decode draws only ~14 W)
    busy_min_clock_mhz: int = 1500       # >0: a busy, uncapped-or-active GPU must clock at least this (latch: ~611)
    busy_util_pct: int = 50
    busy_samples: int = 3
    alarm_repeat_s: float = 3600.0       # re-run the alert command at most this often
    alert_cmd: str = ""                  # shell command, message in $GOV_ALERT_MSG
    csv_path: str = ""                   # sample log; "" disables
    csv_every_s: float = 10.0
    dry_run: bool = False
    apply_fail_limit: int = 3            # consecutive nvidia-smi failures before fail-safe

    @classmethod
    def from_env(cls, env=os.environ) -> "Config":
        c = cls()
        g = lambda k, d: env.get("GOV_" + k, d)  # noqa: E731
        c.role = g("ROLE", c.role).strip().lower()
        c.listen = g("LISTEN", c.listen).strip()
        c.peers = [p.strip() for p in g("PEERS", "").split(",") if p.strip()]
        c.master = g("MASTER", "").strip()
        c.token = g("TOKEN", "").strip()
        c.poll_s = float(g("POLL_S", c.poll_s))
        c.idle_after_s = float(g("IDLE_AFTER_S", c.idle_after_s))
        c.active_util_pct = int(g("ACTIVE_UTIL_PCT", c.active_util_pct))
        c.idle_clocks = g("IDLE_CLOCKS", c.idle_clocks).strip()
        c.active_clocks = g("ACTIVE_CLOCKS", c.active_clocks).strip()
        c.watchdog_s = float(g("WATCHDOG_S", c.watchdog_s))
        c.power_ok_w = float(g("POWER_OK_W", c.power_ok_w))
        c.busy_min_clock_mhz = int(g("BUSY_MIN_CLOCK_MHZ", c.busy_min_clock_mhz))
        c.busy_util_pct = int(g("BUSY_UTIL_PCT", c.busy_util_pct))
        c.busy_samples = int(g("BUSY_SAMPLES", c.busy_samples))
        c.alarm_repeat_s = float(g("ALARM_REPEAT_S", c.alarm_repeat_s))
        c.alert_cmd = g("ALERT_CMD", "")
        c.csv_path = g("LOG_CSV", "").strip()
        c.csv_every_s = float(g("CSV_EVERY_S", c.csv_every_s))
        c.dry_run = g("DRY_RUN", "0").strip() not in ("", "0", "false", "no")
        c.apply_fail_limit = int(g("APPLY_FAIL_LIMIT", c.apply_fail_limit))
        if c.role not in ("master", "worker"):
            raise SystemExit(f"GOV_ROLE must be master or worker, got {c.role!r}")
        if c.role == "worker" and not c.master:
            raise SystemExit("worker needs GOV_MASTER=<master ip>")
        for clocks in (c.idle_clocks, c.active_clocks):
            if clocks and not _valid_clocks(clocks):
                raise SystemExit(f"bad clocks spec {clocks!r}, want min,max in MHz")
        return c


def _valid_clocks(s: str) -> bool:
    parts = s.split(",")
    return len(parts) == 2 and all(p.strip().isdigit() for p in parts) and int(parts[0]) <= int(parts[1])


# --------------------------------------------------------------------------- GPU access
@dataclass
class Sample:
    ts: float
    clock_mhz: int
    power_w: float
    temp_c: float
    util_pct: int
    pstate: str
    event_reasons: str
    ok: bool = True
    error: str = ""

    @staticmethod
    def failed(msg: str) -> "Sample":
        return Sample(time.time(), -1, -1.0, -1.0, -1, "?", "?", ok=False, error=msg)


def _num(s: str, cast, default):
    try:
        return cast(s.strip())
    except (ValueError, AttributeError):
        return default


def nvidia_sample() -> Sample:
    try:
        out = subprocess.run(
            [NVIDIA_SMI, f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return Sample.failed(f"nvidia-smi: {e}")
    if out.returncode != 0:
        return Sample.failed(f"nvidia-smi rc={out.returncode}: {out.stderr.strip()[:200]}")
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")] if out.stdout.strip() else []
    if len(parts) < 6:
        return Sample.failed(f"unexpected nvidia-smi output: {out.stdout.strip()[:200]}")
    return Sample(
        ts=time.time(),
        clock_mhz=_num(parts[0], int, -1),
        power_w=_num(parts[1], float, -1.0),
        temp_c=_num(parts[2], float, -1.0),
        util_pct=_num(parts[3], int, -1),
        pstate=parts[4],
        event_reasons=parts[5],
    )


def nvidia_apply(args: List[str], dry_run: bool) -> bool:
    """Run nvidia-smi with the given clock arguments; True on success."""
    cmd = [NVIDIA_SMI] + args
    if dry_run:
        log.info("DRY RUN: would run %s", " ".join(shlex.quote(c) for c in cmd))
        return True
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.error("%s failed: %s", " ".join(cmd), e)
        return False
    msg = (out.stdout + out.stderr).strip().replace("\n", " | ")[:300]
    if out.returncode != 0:
        log.error("%s rc=%d: %s", " ".join(cmd), out.returncode, msg)
        return False
    log.info("%s: %s", " ".join(args), msg or "ok")
    return True


# --------------------------------------------------------------------------- governor core
class Governor:
    """State machine shared by both roles. ``sampler``/``applier`` are injectable for tests."""

    def __init__(self, cfg: Config, sampler: Callable[[], Sample] = nvidia_sample,
                 applier: Callable[[List[str], bool], bool] = nvidia_apply, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.sampler = sampler
        self.applier = applier
        self.now = clock
        self.lock = threading.RLock()
        self.mode = "released"           # released | active | idle  (what we last applied; released = -rgc done)
        self.applied_ok = True
        self.apply_failures = 0
        self.last_activity = clock()
        self.last_sample: Optional[Sample] = None
        self.last_master_contact = clock() if cfg.role == "worker" else None
        self.master_mode = ""            # worker: what master last asked for
        self.peer_status: Dict[str, dict] = {}
        self.busy_streak = 0
        self.released_at = clock()
        self.alarm = ""                  # latest alarm text, "" when clear
        self.alarm_at = 0.0
        self.transitions = 0
        self.started = clock()
        self._csv_last = 0.0
        self._csv = None
        if cfg.csv_path:
            os.makedirs(os.path.dirname(cfg.csv_path) or ".", exist_ok=True)
            self._csv = logging.getLogger("gpu-governor.csv")
            self._csv.propagate = False
            h = logging.handlers.RotatingFileHandler(cfg.csv_path, maxBytes=20 << 20, backupCount=5)
            h.setFormatter(logging.Formatter("%(message)s"))
            self._csv.addHandler(h)
            self._csv.setLevel(logging.INFO)
            if os.path.getsize(cfg.csv_path) == 0:
                self._csv.info("ts,iso,role,mode,clock_mhz,power_w,temp_c,util_pct,pstate,event_reasons,idle_s,alarm")

    # ---- clock control
    def _apply(self, mode: str) -> None:
        """Set the clocks for ``mode`` (idle/active/released)."""
        if mode == "idle":
            args = ["-lgc", self.cfg.idle_clocks]
        elif mode == "active" and self.cfg.active_clocks:
            args = ["-lgc", self.cfg.active_clocks]
        else:
            args, mode = ["-rgc"], ("active" if mode == "active" else "released")
        ok = self.applier(args, self.cfg.dry_run)
        with self.lock:
            prev = self.mode
            if ok:
                self.apply_failures = 0
                self.applied_ok = True
                if self.alarm.startswith("nvidia-smi"):
                    self.alarm = ""
                self.mode = mode
                if mode != "idle":
                    self.released_at = self.now()
                    self.busy_streak = 0
                if prev != mode:
                    self.transitions += 1
                    log.info("mode %s -> %s", prev, mode)
                    self._csv_row(force=True)
            else:
                self.apply_failures += 1
                self.applied_ok = False
                if self.apply_failures >= self.cfg.apply_fail_limit:
                    self._raise_alarm(f"nvidia-smi {' '.join(args)} keeps failing; fail-safe, clocks released")
                    if args != ["-rgc"]:
                        self.applier(["-rgc"], self.cfg.dry_run)
                        self.mode = "released"

    def release(self, why: str) -> None:
        log.info("release clocks (%s)", why)
        self._apply("released")

    def set_mode(self, mode: str, why: str = "") -> None:
        with self.lock:
            cur = self.mode
        if mode == "active" and cur in ("active", "released") and not self.cfg.active_clocks:
            return  # already unlocked
        if mode == cur:
            return
        log.info("set %s (%s)", mode, why)
        self._apply(mode)

    # ---- sampling + decisions
    def poll(self) -> Sample:
        s = self.sampler()
        with self.lock:
            self.last_sample = s
            now = self.now()
            if s.ok and s.util_pct >= self.cfg.active_util_pct:
                self.last_activity = now
            if not s.ok:
                log.warning("sample failed: %s", s.error)
        self._self_check(s)
        if self.cfg.role == "master":
            self._decide()
        else:
            self._watchdog()
        self._csv_row()
        return s

    def note_activity(self, source: str) -> None:
        with self.lock:
            self.last_activity = self.now()
        if self.cfg.role == "master":
            self.set_mode("active", f"wake from {source}")

    def idle_seconds(self) -> float:
        with self.lock:
            return self.now() - self.last_activity

    def _decide(self) -> None:
        idle = self.idle_seconds()
        with self.lock:
            mode = self.mode
        if mode != "active" and idle < self.cfg.poll_s * 1.5:
            self.set_mode("active", "activity")  # no-op when released and no active cap
        elif mode != "idle" and idle >= self.cfg.idle_after_s:
            self.set_mode("idle", f"idle {idle:.0f}s")

    def _watchdog(self) -> None:
        with self.lock:
            silent = self.now() - (self.last_master_contact or 0)
            mode = self.mode
        if silent > self.cfg.watchdog_s and mode == "idle":
            self._raise_alarm(f"no heartbeat from master for {silent:.0f}s; released clocks")
            self.release("watchdog")

    def _self_check(self, s: Sample) -> None:
        """A busy GPU must run at a real clock (and, if configured, draw real power) once the cap is released."""
        if not s.ok:
            return
        with self.lock:
            busy = s.util_pct >= self.cfg.busy_util_pct
            self.busy_streak = self.busy_streak + 1 if busy else 0
            settled = self.mode != "idle" and (self.now() - self.released_at) >= 2 * self.cfg.poll_s
            low_clock = self.cfg.busy_min_clock_mhz > 0 and 0 <= s.clock_mhz < self.cfg.busy_min_clock_mhz
            low_power = self.cfg.power_ok_w > 0 and 0 <= s.power_w < self.cfg.power_ok_w
            trip = settled and self.busy_streak >= self.cfg.busy_samples and (low_clock or low_power)
            clear = settled and busy and not (low_clock or low_power) and self.alarm.startswith("GPU busy")
        if trip:
            self._raise_alarm(f"GPU busy ({s.util_pct}% util) but clock {s.clock_mhz} MHz (< {self.cfg.busy_min_clock_mhz}) / {s.power_w:.1f} W: "
                              "cap not released or EC clock latch; power-drain the node if nvidia-smi -rgc does not help")
        elif clear:
            with self.lock:
                self.alarm = ""
            log.info("self-check: %d MHz / %.1f W under load, alarm cleared", s.clock_mhz, s.power_w)

    def _raise_alarm(self, msg: str) -> None:
        with self.lock:
            repeat = (self.now() - self.alarm_at) >= self.cfg.alarm_repeat_s or self.alarm != msg
            self.alarm = msg
            if not repeat:
                return
            self.alarm_at = self.now()
        log.error("ALARM: %s", msg)
        if self.cfg.alert_cmd:
            env = dict(os.environ, GOV_ALERT_MSG=f"[{os.uname().nodename} gpu-governor] {msg}")
            try:
                subprocess.Popen(self.cfg.alert_cmd, shell=True, env=env, stdin=subprocess.DEVNULL)
            except OSError as e:
                log.error("alert command failed: %s", e)

    # ---- worker side of the protocol
    def receive_master(self, body: dict) -> dict:
        with self.lock:
            self.last_master_contact = self.now()
            self.master_mode = body.get("mode", "")
            if self.alarm.startswith("no heartbeat"):
                self.alarm = ""
                log.info("master heartbeat is back, alarm cleared")
            if body.get("mode") == "active" and body.get("activity"):
                self.last_activity = self.now()
        want = body.get("mode")
        if want in ("idle", "active"):
            self.set_mode(want, "master")
        return self.status()

    # ---- reporting
    def status(self) -> dict:
        with self.lock:
            s = self.last_sample
            return {
                "host": os.uname().nodename,
                "role": self.cfg.role,
                "mode": self.mode,
                "applied_ok": self.applied_ok,
                "dry_run": self.cfg.dry_run,
                "idle_s": round(self.now() - self.last_activity, 1),
                "idle_after_s": self.cfg.idle_after_s,
                "idle_clocks": self.cfg.idle_clocks,
                "active_clocks": self.cfg.active_clocks or "-rgc",
                "sample": asdict(s) if s else None,
                "master_mode": self.master_mode,
                "master_silent_s": round(self.now() - self.last_master_contact, 1) if self.last_master_contact else None,
                "peers": self.peer_status,
                "alarm": self.alarm,
                "transitions": self.transitions,
                "uptime_s": round(self.now() - self.started),
            }

    def _csv_row(self, force: bool = False) -> None:
        if not self._csv:
            return
        with self.lock:
            now = self.now()
            if not force and now - self._csv_last < self.cfg.csv_every_s:
                return
            self._csv_last = now
            s = self.last_sample
            row = [f"{now:.0f}", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)), self.cfg.role, self.mode]
            if s and s.ok:
                row += [s.clock_mhz, f"{s.power_w:.2f}", f"{s.temp_c:.0f}", s.util_pct, s.pstate, s.event_reasons]
            else:
                row += ["", "", "", "", "", ""]
            row += [f"{now - self.last_activity:.0f}", self.alarm.replace(",", ";")[:80]]
        self._csv.info(",".join(str(x) for x in row))


# --------------------------------------------------------------------------- HTTP
def _make_handler(gov: Governor):
    cfg = gov.cfg

    class H(BaseHTTPRequestHandler):
        server_version = "gpu-governor/1"

        def log_message(self, fmt, *args):  # quiet
            log.debug("http %s " + fmt, self.client_address[0], *args)

        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            return not cfg.token or self.headers.get("X-Gov-Token", "") == cfg.token

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 65536:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return {}

        def do_GET(self):
            if self.path.startswith("/status"):
                return self._send(200, gov.status())
            if self.path.startswith("/health"):
                st = gov.status()
                ok = st["applied_ok"] and not st["alarm"]
                return self._send(200 if ok else 503, {"ok": ok, "alarm": st["alarm"]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed():
                return self._send(403, {"error": "bad token"})
            body = self._body()
            if self.path.startswith("/wake"):
                if cfg.role != "master":
                    return self._send(200, {"ignored": True, "reason": "worker; wake the master", "status": gov.status()})
                gov.note_activity(f"http {self.client_address[0]}")
                return self._send(200, gov.status())
            if self.path.startswith("/state"):
                if cfg.role != "worker":
                    return self._send(409, {"error": "master does not take state"})
                if self.client_address[0] != cfg.master:
                    return self._send(403, {"error": f"state only accepted from {cfg.master}"})
                return self._send(200, gov.receive_master(body))
            self._send(404, {"error": "not found"})

    return H


def start_http(gov: Governor) -> ThreadingHTTPServer:
    host, _, port = gov.cfg.listen.rpartition(":")
    srv = ThreadingHTTPServer((host or "127.0.0.1", int(port)), _make_handler(gov))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="http", daemon=True).start()
    log.info("listening on %s", gov.cfg.listen)
    return srv


def push_to_peers(gov: Governor, timeout: float = 2.0) -> None:
    """Master: send the current mode to every peer and remember their replies."""
    with gov.lock:
        body = {"mode": "idle" if gov.mode == "idle" else "active",
                "activity": gov.idle_seconds() < gov.cfg.poll_s * 1.5,
                "ts": gov.now(), "from": os.uname().nodename}
    data = json.dumps(body).encode()
    for peer in gov.cfg.peers:
        req = urlrequest.Request(f"http://{peer}/state", data=data, method="POST",
                                 headers={"Content-Type": "application/json", "X-Gov-Token": gov.cfg.token})
        try:
            with urlrequest.urlopen(req, timeout=timeout) as r:
                st = json.loads(r.read() or b"{}")
            st["reachable"] = True
            st["seen"] = gov.now()
            with gov.lock:
                gov.peer_status[peer] = st
            if st.get("alarm"):
                log.warning("peer %s alarm: %s", peer, st["alarm"])
            if st.get("mode") not in (body["mode"], "released") and not (body["mode"] == "active" and st.get("mode") == "released"):
                log.warning("peer %s is in mode %s, asked %s", peer, st.get("mode"), body["mode"])
        except (urlerror.URLError, OSError, ValueError) as e:
            with gov.lock:
                prev = gov.peer_status.get(peer, {})
                gov.peer_status[peer] = {"reachable": False, "error": str(e)[:120], "seen": prev.get("seen")}
            log.warning("peer %s unreachable: %s", peer, str(e)[:120])


# --------------------------------------------------------------------------- main loop
def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=os.environ.get("GOV_LOGLEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cfg = Config.from_env()
    log.info("config: %s", json.dumps({k: v for k, v in asdict(cfg).items() if k != "token"}))
    if not cfg.dry_run and os.geteuid() != 0:
        log.error("nvidia-smi -lgc needs root; run as root or set GOV_DRY_RUN=1")
        return 2
    gov = Governor(cfg)
    stop = threading.Event()

    def _sig(signum, _frame):
        log.info("signal %s, stopping", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    srv = start_http(gov)
    gov.release("startup")  # known state first: unlocked
    try:
        while not stop.is_set():
            t0 = time.time()
            try:
                gov.poll()
                if cfg.role == "master":
                    push_to_peers(gov)
            except Exception:  # noqa: BLE001
                log.exception("poll failed")
            stop.wait(max(0.2, cfg.poll_s - (time.time() - t0)))
    finally:
        try:
            if cfg.role == "master":
                with gov.lock:
                    gov.mode = "released"
                push_to_peers(gov)   # tell workers to unlock too
        except Exception:  # noqa: BLE001
            log.exception("final peer push failed")
        gov.release("shutdown")
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
