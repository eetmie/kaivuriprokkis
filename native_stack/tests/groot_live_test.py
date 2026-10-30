"""groot_live.py without hardware or GPU: raw-output gate, RTC prefix/feeds, frame history, supervisor stops and
cleanup, observation coherence, partial service start, pad drain before arming. In-memory fakes only.

    python tests/groot_live_test.py native_stack
"""
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

source = Path(sys.argv[1])
sys.path.insert(0, str(source / "tools"))
import groot_live as gl  # noqa: E402

CONTRACT = {"embodiment": "new_embodiment", "clip_state": True,
            "state": {"joints": {"min": [-50.0, 44.0, -120.0], "max": [-10.0, 124.0, 19.0]}},
            "action": {"valves": {"min": [-0.89, -1.0, -1.0, -1.0], "max": [0.79, 0.547, 1.0, 1.0]}}}
H, D = 40, 132


def raises(fn, text):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        assert text in str(e), (text, e)
        return
    raise AssertionError(f"no error, expected {text!r}")


# ── policy boundary: raw output before clipping ──────────────────────────────

def policy(model_out, rtc_max=0):
    pol = object.__new__(gl.Policy)
    pol.boundary, pol.horizon, pol.width, pol.state_width = gl.Boundary(CONTRACT), H, D, 132
    pol.rtc, pol.rtc_max = ({"max_delay": rtc_max} if rtc_max else None), rtc_max
    pol.frames, pol.history, pol.lag_s = 1, None, 0.0
    pol.b = SimpleNamespace(preprocess=lambda f: np.zeros((3, 4, 4), np.float32))
    pol.engines = SimpleNamespace(run=None)
    pol.g17 = SimpleNamespace(encode_frames=lambda *a: [], infer=lambda *a, **k: model_out)
    return pol


frames, deg, noise = [np.zeros((4, 4, 3), np.uint8)] * 2, np.zeros(3, np.float32), np.zeros((1, H, D), np.float32)
ok = np.zeros((1, H, D), np.float32)
act, model = policy(ok).infer(frames, deg, noise)
assert act.shape == (H, 4) and np.isfinite(act).all()
for bad in (np.inf, -np.inf, np.nan):
    out = ok.copy()
    out[0, 5, 1] = bad
    raises(lambda: policy(out).infer(frames, deg, noise), "non-finite")
raises(lambda: policy(np.zeros((1, H, 4), np.float32)).infer(frames, deg, noise), "shape")
raises(lambda: policy(ok).infer(frames, deg, noise, rtc_delay=3, rtc_prefix=np.zeros((3, 4))), "no RTC")

# ── frame history (video_delta_indices [-k, 0]) ─────────────────────────────

assert gl.frame_slots({"frames": 1, "video_delta_indices": [0]}) == (1, 0.0)
assert gl.frame_slots({"frames": 2, "video_delta_indices": [-30, 0], "kaivuri": {"fps": 30}}) == (2, 1.0)
raises(lambda: gl.frame_slots({"frames": 2, "video_delta_indices": [-30, 0]}), "kaivuri.fps")
for bad in ([0, -30], [-30, -15], [-30, -15, 0]):
    raises(lambda: gl.frame_slots({"frames": len(bad), "video_delta_indices": bad, "kaivuri": {"fps": 30}}),
           "handles [0] and [-k, 0]")

MS = 1_000_000
hist = gl.History(1.0, 0.15)
assert hist.pick(0) is None
for k in range(0, 1300, 250):                                       # one encode every 250 ms
    hist.push(k * MS, f"e{k}")
hist.push(1250 * MS, "again")                                       # same frame twice: first encode kept
assert hist.pick(2000 * MS) == ("e1000", 1.0)
assert hist.pick(2100 * MS)[0] == "e1000" and hist.pick(2200 * MS)[0] == "e1250"
assert hist.pick(1300 * MS)[0] == "e250"
assert hist.pick(2500 * MS) is None                                 # nearest is 1.25 s old: outside 150 ms
hist.push(3000 * MS, "e3000")
assert [e for _, e in hist.entries] == ["e3000"]                    # everything older can never be picked


def two_frame_policy():
    pol = policy(ok)
    pol.frames, pol.lag_s = 2, 1.0
    pol.history = gl.History(1.0, 0.15)
    pol.b = SimpleNamespace(preprocess=lambda f: np.full((3, 4, 4), f[0, 0, 0], np.float32))
    pol.slots = []
    pol.g17 = SimpleNamespace(encode_frames=lambda b, run, pv: [int(pv[0, 0, 0, 0])],
                              infer=lambda b, run, slots, *a, **k: pol.slots.append(slots) or ok)
    return pol


def cams(v):
    return [np.full((4, 4, 3), v, np.uint8)] * 2


pol = two_frame_policy()
raises(lambda: pol.infer(cams(1), deg, noise), "capture time")
for k, ms in enumerate(range(0, 800, 100)):                         # start-up: stored, no plan
    raises(lambda: pol.infer(cams(k), deg, noise, capture_ns=ms * MS), "s old yet")
assert pol.slots == [] and len(pol.history.entries) == 8
t = {}
pol.infer(cams(50), deg, noise, t, capture_ns=1020 * MS)
assert pol.slots == [[[0], [50]]] and abs(t["history_age"] - 1020) < 1e-6   # oldest first: past, now
pol.remember(cams(60), 1100 * MS)                                    # manual mode keeps encoding
pol.infer(cams(70), deg, noise, capture_ns=2090 * MS)
assert pol.slots[-1] == [[60], [70]]
pol.infer(cams(3), deg, noise, past_rgb=cams(9))                     # parity: the fixture's past frames
assert pol.slots[-1] == [[9], [3]]
pol1 = policy(ok)
pol1.remember(cams(1), 0)                                            # one-frame bundles keep no history
assert pol1.history is None

# Valve <-> model units round-trip, and a clipped command maps to the bound.
b = gl.Boundary(CONTRACT)
u = np.array([[0.0, 0.0, 0.0, 0.0], [0.5, 0.3, -0.7, 0.9]], np.float32)
assert np.allclose(b.actions(b.model_actions(u)[None]), u, atol=1e-6)
assert np.isclose(b.actions(b.model_actions(np.array([[0, 1.0, 0, 0]]))[None])[0, 1], 0.547)

# ── RTC feeds and the committed prefix ───────────────────────────────────────

pol = policy(ok, rtc_max=12)
f1, f2 = pol.rtc_feeds(5, np.ones((5, 4), np.float32)), pol.rtc_feeds(5, np.ones((5, 4), np.float32))
assert f1["rtc_mask"] is not f2["rtc_mask"]                        # fresh arrays: Engines.run re-uploads
assert f1["rtc_mask"][0, :5].all() and not f1["rtc_mask"][0, 5:].any()
assert (f1["prefix"][0, :5, :4] == 1).all() and not f1["prefix"][0, 5:].any() and not f1["prefix"][0, :, 4:].any()
raises(lambda: pol.rtc_feeds(13, np.ones((13, 4))), "outside")
raises(lambda: pol.rtc_feeds(4, np.full((4, 4), np.nan)), "bad RTC prefix")

fps = 30.0
rows = np.zeros((H, 6), np.float32)
rows[:, :4] = np.linspace(-0.5, 0.5, H)[:, None] * [1, 0.5, -1, 0.25]
t0_old = 1_000_000_000
shift = 8.6
t0_new = t0_old + int(shift / fps * 1e9)
d, prefix, pid = gl.committed_rows(pol, (rows, t0_old, 7), t0_new, 10, fps)
assert d == 10 and pid == 7
want = gl.interp_rows(rows[:, :4], shift + np.arange(10))
assert np.allclose(b.actions(np.pad(prefix, ((0, H - 10), (0, D - 4)))[None])[:10], want, atol=1e-5)
d, _, _ = gl.committed_rows(pol, (rows, t0_old, 7), t0_old + int(33.5 / fps * 1e9), 10, fps)
assert d == 6, d                                                   # rows 33.5..38.5 lie inside the plan
assert gl.committed_rows(pol, None, t0_new, 10, fps)[0] == 0

delay = gl.RtcDelay(12, fps, 40)
assert delay.want() == 12
for ms in (250, 290, 270):
    delay.add(ms)
assert delay.want() == 10                                          # ceil((290 + 40) * 30 / 1000)
delay.add(900)
assert delay.want() == 12

# ── supervisor ───────────────────────────────────────────────────────────────


class FakeClient:
    instances = []
    state = "armed"

    def __init__(self, path):
        self.sends, self.sequence, self.closed = [], 0, False
        FakeClient.instances.append(self)

    def send(self, op, steps=(), **kw):
        self.sequence += 1
        self.sends.append((op, kw.get("flags")))

    def status(self):
        return {"state": FakeClient.state, "reason": "test", "applied_command_sequence": -1,
                "applied_host_monotonic_ns": 0, "applied_action": [0] * 6}

    def arm(self, pump=False):
        if FakeClient.state == "fault":
            raise RuntimeError("arm refused")

    def close(self):
        self.closed = True


client_mod = SimpleNamespace(Client=FakeClient, DEADMAN=1, PUMP=4, DRIVE=1, DISARM=2)


def supervisor(**kw):
    FakeClient.instances.clear()
    return gl.Supervisor(client_mod, "sock", fps, **kw)


# No first plan: bounded wait, then the run stops with kind "plan" and disarms.
FakeClient.state = "armed"
sup = supervisor(first_plan_s=0.1, pump=True)
sup.start()
sup.join(timeout=2)
assert not sup.is_alive() and sup.stop_kind == "plan", sup.stop_reason
c = FakeClient.instances[0]
assert c.sends[0] == (1, 5) and c.sends[-1][0] == 2 and c.closed   # pump held only inside the wait

# Arming fails before start(): shutdown still disarms and closes; join is never called.
FakeClient.state = "disarmed"
sup = supervisor()
FakeClient.state = "fault"
raises(sup.arm, "not disarmed")
assert sup.ident is None
sup.shutdown()
sup.shutdown()
c = FakeClient.instances[0]
assert [s[0] for s in c.sends] == [2] and c.closed

# Controller fault while armed -> kind "controller".
FakeClient.state = "fault"
sup = supervisor(first_plan_s=5)
sup.start()
sup.join(timeout=2)
assert sup.stop_kind == "controller"

# Handovers: a plan that continues the running one reports no jump; a different one does.
FakeClient.state = "armed"
sup = supervisor(first_plan_s=5, max_obs_age_s=10)
now = time.monotonic_ns()
assert sup.offer(rows, now, now)[0]
sup.start()
time.sleep(0.15)
t1 = now + int(3 / fps * 1e9)                                       # due before the send
cont = np.zeros_like(rows)
cont[:-3] = rows[3:]
cont[-3:] = rows[-1]
assert sup.offer(cont, t1, now + 1)[0]
time.sleep(0.15)
jump = cont.copy()
jump[:, 1] += 0.8
assert sup.offer(jump, t1, now + 2)[0]
time.sleep(0.15)
assert sup.offer(rows, t1, now + 2) == (False, "not newer than the running plan", None)
# A mode or prompt switch: the first plan after it follows the centred hold, so it is no handover
# (comparing it with the plan from before the switch reported 0.6-1.05 "jumps" live).
for switch in (sup.drop_plan, lambda: sup.set_mode("model")):
    switch()
    t2 = time.monotonic_ns()
    assert sup.offer(jump, t2, t2)[0]
    time.sleep(0.15)
assert sup.is_alive(), sup.stop_reason
sup.stop("done")
sup.join(timeout=2)
hand = [r for r in sup.dispatch if "jump" in r]
assert len(hand) == 2 and max(hand[0]["jump"]) < 1e-5 and abs(max(hand[1]["jump"]) - 0.8) < 1e-4, hand
assert len(sup.dispatch) == 5 and sup.running_plan()[2] == 5

# ── observation coherence ────────────────────────────────────────────────────


class Reader:
    def __init__(self):
        self.seq, self.offset_ns = 10, 0

    def read(self):
        self.seq += 1
        m = SimpleNamespace(sequence=self.seq, host_monotonic_ns=time.monotonic_ns() - self.offset_ns,
                            width=640, height=480, ir_index=2, pixel_bytes=640 * 480)
        return m, memoryview(bytearray(640 * 480))


class Status:
    imu_offset_ns = 0

    def status(self):
        return {"imu_healthy": True, "fk_valid": True,
                "imu_host_monotonic_ns": time.monotonic_ns() - self.imu_offset_ns}


args = SimpleNamespace(max_frame_age_ms=200, max_camera_skew_ms=50, max_state_skew_ms=100, max_imu_age_ms=50)
r1, r2, st = Reader(), Reader(), Status()
obs = gl.Observer([r1, r2], ["a-ir2.frame", "b-ir2.frame"], st, args)
frames, _, obs_ns, skew, src = obs()
assert len(frames) == 2 and skew < 50 and len(src["cameras"]) == 2
r2.offset_ns = 150_000_000                                          # both fresh, 150 ms apart
raises(obs, "camera skew")
r2.offset_ns = 0
st.imu_offset_ns = -20_000_000
raises(obs, "IMU from the future")
st.imu_offset_ns = 0
r1.seq = 0
raises(obs, "camera restarted")
obs()                                                               # recovers on the next snapshot
obs.ir = [1, 2]
raises(obs, "wrong camera stream")

# ── partial service start keeps ownership ───────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    build, runtime = Path(tmp) / "build", Path(tmp) / "run"
    build.mkdir()
    runtime.mkdir()
    exe = build / "kaivuri-control"
    exe.write_text("#!/bin/sh\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    a = SimpleNamespace(build=str(build), profile="p", safety="s", control_hz=200, serial="/dev/null",
                        live=False, enable_slew=False)
    procs = []
    raises(lambda: gl.start_services(a, runtime, procs), "did not start")
    assert len(procs) == 1
    gl.stop_services(procs)
    assert procs[0].poll() is not None

# ── pad: an overrun queued before arming is drained, one after it still stops ──

with tempfile.TemporaryDirectory() as tmp:
    fifo = Path(tmp) / "pad"
    os.mkfifo(fifo)
    pad = gl.Pad(str(fifo))
    writer = os.open(fifo, os.O_WRONLY)
    dropped = gl._EVENT.pack(0, 0, gl.EV_SYN, gl.SYN_DROPPED, 0)
    os.write(writer, dropped * 3)
    pad.drain()
    assert pad.poll() == []
    os.write(writer, dropped)
    raises(pad.poll, "overrun")
    os.close(writer)
    pad.close()

assert len(set(gl.EXIT.values())) == len(gl.EXIT) and gl.EXIT["finished"] == 0
print("groot_live tests passed", flush=True)
os._exit(0)
