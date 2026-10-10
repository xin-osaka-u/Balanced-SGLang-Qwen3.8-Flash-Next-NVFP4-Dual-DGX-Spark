"""Offline test of the governor state machine with a fake GPU and a fake clock. No root, no nvidia-smi."""
import logging, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
import gpu_governor as G

logging.basicConfig(level=logging.WARNING)


class Fake:
    def __init__(self):
        self.t = 1000.0; self.util = 0; self.power = 11.0; self.clock = 2405
        self.calls = []; self.fail = False
    def now(self): return self.t
    def sample(self):
        return G.Sample(self.t, self.clock, self.power, 50.0, self.util, "P0", "0x0")
    def apply(self, args, dry):
        self.calls.append(" ".join(args))
        if self.fail: return False
        if args[0] == "-lgc": self.clock = int(args[1].split(",")[1])
        else: self.clock = 2405
        return True
    def run(self, gov, seconds, util=None, power=None):
        if util is not None: self.util = util
        if power is not None: self.power = power
        for _ in range(int(seconds / gov.cfg.poll_s)):
            self.t += gov.cfg.poll_s; gov.poll()


def master(**kw):
    cfg = G.Config(role="master", poll_s=2, idle_after_s=300, idle_clocks="210,1000", **kw)
    f = Fake(); gov = G.Governor(cfg, sampler=f.sample, applier=f.apply, clock=f.now)
    gov.release("startup"); return f, gov


# 1. idle -> cap after 300 s, activity -> release immediately
f, gov = master()
assert f.calls == ["-rgc"]
f.run(gov, 298); assert gov.mode == "released", gov.mode
f.run(gov, 4); assert gov.mode == "idle" and f.calls[-1] == "-lgc 210,1000", (gov.mode, f.calls)
f.run(gov, 600); assert f.calls.count("-lgc 210,1000") == 1, "must not re-apply every poll"
f.run(gov, 2, util=30); assert gov.mode == "active" and f.calls[-1] == "-rgc"
f.run(gov, 100, util=0); assert gov.mode == "active", "hysteresis: stays active until idle_after"
f.run(gov, 300); assert gov.mode == "idle"
print("1 idle/active cycle ok")

# 2. /wake releases and resets the idle timer
gov.note_activity("test"); assert gov.mode == "active" and gov.idle_seconds() == 0
print("2 wake ok")

# 3. optional thermal cap while active
f2, gov2 = master(active_clocks="300,2200")
assert f2.calls == ["-rgc"]
f2.run(gov2, 2, util=50); assert gov2.mode == "active" and f2.calls[-1] == "-lgc 300,2200"
f2.run(gov2, 302, util=0); assert gov2.mode == "idle" and f2.calls[-1] == "-lgc 210,1000"
f2.run(gov2, 2, util=50); assert f2.calls[-1] == "-lgc 300,2200"
print("3 thermal cap ok")

# 4. self-check: busy but low power -> alarm; power back -> cleared
f3, gov3 = master(power_ok_w=30, busy_util_pct=50, busy_samples=3)
alerts = []
gov3._raise_alarm_orig = gov3._raise_alarm
f3.run(gov3, 10, util=90, power=12.0)
assert gov3.alarm.startswith("GPU busy"), gov3.alarm
f3.run(gov3, 4, util=90, power=75.0)
assert gov3.alarm == "", gov3.alarm
print("4 self-check ok")

# 4b. clock-based latch detector: decode-like load (14 W, 2190 MHz) is fine; 611 MHz while busy alarms; recovery clears
f3b, gov3b = master(busy_util_pct=50, busy_samples=3, busy_min_clock_mhz=1500)
f3b.run(gov3b, 10, util=70, power=14.0); assert gov3b.alarm == "", gov3b.alarm
f3b.clock = 611; f3b.run(gov3b, 10, util=90, power=5.5); assert gov3b.alarm.startswith("GPU busy"), gov3b.alarm
f3b.clock = 2190; f3b.run(gov3b, 4, util=90, power=14.0); assert gov3b.alarm == "", gov3b.alarm
print("4b clock latch ok")

# 5. nvidia-smi failures -> fail-safe after 3, alarm set, clears on success
f4, gov4 = master(apply_fail_limit=3)
f4.run(gov4, 302)  # idle -> tries -lgc
assert gov4.mode == "idle"
f4.fail = True
f4.run(gov4, 2, util=50)  # release fails 1
f4.run(gov4, 302, util=0)  # back to idle? mode stayed idle since release failed; keep failing on next transitions
# force repeated failures through set_mode
for _ in range(3): gov4.set_mode("active", "t"); gov4.mode = "idle"
assert gov4.alarm.startswith("nvidia-smi"), gov4.alarm
f4.fail = False
gov4.set_mode("active", "t"); assert gov4.alarm == "" and gov4.applied_ok
print("5 fail-safe ok")

# 6. worker: follows master, watchdog releases after silence, heartbeat clears alarm
cfg = G.Config(role="worker", master="10.0.0.1", poll_s=2, watchdog_s=90, idle_clocks="210,1000")
f5 = Fake(); w = G.Governor(cfg, sampler=f5.sample, applier=f5.apply, clock=f5.now); w.release("startup")
w.receive_master({"mode": "idle"}); assert w.mode == "idle" and f5.calls[-1] == "-lgc 210,1000"
w.receive_master({"mode": "idle"}); assert f5.calls.count("-lgc 210,1000") == 1
f5.run(w, 88); assert w.mode == "idle"
f5.run(w, 4); assert w.mode == "released" and w.alarm.startswith("no heartbeat"), (w.mode, w.alarm)
w.receive_master({"mode": "active", "activity": True}); assert w.alarm == "" and w.mode == "released"
w.receive_master({"mode": "idle"}); assert w.mode == "idle"
print("6 worker/watchdog ok")

# 7. config validation
import os
try:
    G.Config.from_env({"GOV_IDLE_CLOCKS": "abc"}); raise AssertionError("should reject")
except SystemExit: pass
try:
    G.Config.from_env({"GOV_ROLE": "worker"}); raise AssertionError("worker needs master")
except SystemExit: pass
c = G.Config.from_env({"GOV_ROLE": "master", "GOV_PEERS": "192.168.200.2:9870, ", "GOV_DRY_RUN": "1"})
assert c.peers == ["192.168.200.2:9870"] and c.dry_run
print("7 config ok")
print("ALL OK")
