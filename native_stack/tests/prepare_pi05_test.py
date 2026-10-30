"""CPU test of prepare_pi05.py's Mode C unpacking: what it accepts, and the references it refuses.

    python tests/prepare_pi05_test.py native_stack
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

source = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / "tools"))
import prepare_pi05 as pp  # noqa: E402

H, W, D, ROWS = 50, 32, 4, 15
INFO = {"action_horizon": H, "action_dim": W, "kaivuri": {"rtc": {"row_time": True, "max_prefix": 20}}}


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected ValueError with {text!r}")


def fixture(path, index=0, rows=ROWS, pad=0.0, anchor_shift=0.0, source_fixture=0):
    rng = np.random.default_rng(index)
    anchor = rng.standard_normal((rows, D)).astype(np.float32)
    clean = rng.standard_normal((H, W)).astype(np.float32)
    clean[:rows, :D] = anchor + anchor_shift
    clean[:rows, D:] = pad
    np.savez(path / f"rtc_fixture_{index}.npz", rows=np.int64(rows), dims=np.int64(D), anchor=anchor,
             clean=clean, off=rng.standard_normal((H, W)).astype(np.float32),
             noise=rng.standard_normal((H, W)).astype(np.float32), source_fixture=np.int64(source_fixture))


def case(**kw):
    tmp = tempfile.TemporaryDirectory()
    bundle, out = Path(tmp.name) / "b", Path(tmp.name) / "out"
    (bundle / "rtc").mkdir(parents=True)
    out.mkdir()
    fixture(bundle / "rtc", **kw)
    return tmp, bundle, out


# A reference from the fixed oracle: prefix = anchor, padding zero.
tmp, bundle, out = case()
entries = pp.mode_c_fixtures(bundle, out, INFO, {"fixture_0"})
assert [e["directory"] for e in entries] == ["rtc_0"] and entries[0]["rows"] == ROWS
assert entries[0]["source"] == "fixture_0" and entries[0]["file"] == "rtc/rtc_fixture_0.npz"
assert sorted(entries[0]["files"]) == ["anchor.npy", "clean.npy", "noise.npy", "off.npy"]
got = np.load(out / "rtc_0" / "anchor.npy")
assert got.dtype == np.float32 and got.shape == (ROWS, D)
json.dumps(entries)                                         # goes into manifest.json as is

# Exporters before the zero-padding fix left noise on the prefix's padding.
tmp, bundle, out = case(pad=0.25)
raises(lambda: pp.mode_c_fixtures(bundle, out, INFO, {"fixture_0"}), "prefix padding")
# A reference whose prefix is not its anchor.
tmp, bundle, out = case(anchor_shift=1e-3)
raises(lambda: pp.mode_c_fixtures(bundle, out, INFO, {"fixture_0"}), "is not its anchor")
# More rows than the bundle's trained maximum.
tmp, bundle, out = case(rows=21)
raises(lambda: pp.mode_c_fixtures(bundle, out, INFO, {"fixture_0"}), "shape")
# A source fixture that was not prepared.
tmp, bundle, out = case(source_fixture=3)
raises(lambda: pp.mode_c_fixtures(bundle, out, INFO, {"fixture_0"}), "was not prepared")
# A Mode C bundle without references.
tmp = tempfile.TemporaryDirectory()
(Path(tmp.name) / "rtc").mkdir()
raises(lambda: pp.mode_c_fixtures(Path(tmp.name), Path(tmp.name), INFO, set()), "no RTC fixtures")

print("prepare_pi05 tests passed", flush=True)
