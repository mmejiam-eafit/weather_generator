"""The same with fixture archives driven by a latent AR(1) process (three settings), to see
whether a well-behaved fixture can pass the gate with the current sampler. None does
(`docs/open_questions.md` §5.9).

Run from the repo root: `poetry run python notebooks/probes/fixture_gate_ar1.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import sys, tempfile
from datetime import timedelta
from pathlib import Path
import numpy as np
import conftest as cf
import weather.model as wm
import logging
logging.disable(logging.CRITICAL)

def write_ar(data_dir, rng, phi=0.8, noise=0.2, gain=0.3):
    H, W = cf.TIER1_WAVE_SHAPE
    rows = np.linspace(0.0, 1.0, H)[:, None]; cols = np.linspace(0.0, 1.0, W)[None, :]
    pattern = 1.0 + 0.5 * np.cos(np.pi * rows) * np.cos(np.pi * cols)
    names = ("sigwh", "wsh", "wsp", "wsd", "was", "wad")
    factor = {v: rng.standard_normal() for v in names}
    for i in range(cf.TIER1_N_TIMES):
        t = cf.TIER1_T0 + timedelta(hours=3 * i)
        for v in names:
            factor[v] = phi * factor[v] + np.sqrt(1 - phi**2) * rng.standard_normal()
        z = {v: pattern * factor[v] + noise * rng.standard_normal((H, W)) for v in names}
        sigwh = 2.0 * np.exp(gain * z["sigwh"]); wsh = 1.5 * np.exp(gain * z["wsh"]); wsp = 8.0 + z["wsp"]
        wsd = (np.round((90.0 + 40.0 * z["wsd"]) / 22.5).astype(np.int64) % 16 + 1)
        for field in (sigwh, wsh, wsp):
            field[cf.TIER1_LAND_CELL] = np.nan
        wsd[cf.TIER1_LAND_CELL] = -1
        if i % 4 == 0:
            wsh[cf.TIER1_ABSENT_CELL] = np.nan; wsp[cf.TIER1_ABSENT_CELL] = np.nan; wsd[cf.TIER1_ABSENT_CELL] = -1
        cf._write_snapshot(data_dir, t, "sigwh", sigwh.astype(np.float16))
        cf._write_snapshot(data_dir, t, "wsh", wsh.astype(np.float16))
        cf._write_snapshot(data_dir, t, "wsp", wsp.astype(np.float16))
        cf._write_snapshot(data_dir, t, "wsd", wsd.astype(np.int8))
        was = 6.0 * np.exp(gain * z["was"]); was[cf.TIER1_LAND_CELL] = np.nan
        cf._write_snapshot(data_dir, t, "was", was.astype(np.float16))
        wad_raw = (np.round((200.0 + 40.0 * z["wad"]) / 22.5).astype(np.int64) % 16 + 1).astype(np.int8)
        cf._write_snapshot(data_dir, t, "wad", wad_raw)
        cf._write_snapshot(data_dir, t, "wad_come_from", (((wad_raw.astype(np.int32) - 1 + 8) % 16) + 1).astype(np.int8))

captured = {}
orig = wm._require_fidelity
def spy(f):
    captured.clear(); captured.update(f); orig(f)
wm._require_fidelity = spy
for label, kw in [("phi0.8", {}), ("phi0.6", {"phi": 0.6}), ("phi0.9 noise0.1", {"phi": 0.9, "noise": 0.1})]:
    results = []
    for seed in range(6):
        with tempfile.TemporaryDirectory() as d:
            write_ar(Path(d), np.random.default_rng(0), **kw)
            try:
                cf.small_tier1_model(seed).fit(Path(d)); results.append("pass")
            except ValueError:
                fails = []
                for g, dg in captured.items():
                    if not dg.passed:
                        v = dg.variance[g]; m = dg.memory[g]; t = dg.time_of_day[g]
                        what = []
                        if not v.consistent: what.append(f"var{v.ratio:.2f}")
                        if not m.consistent.all(): what.append("mem" + ",".join(str(int(l)) for l, ok in zip(m.lags, m.consistent) if not ok))
                        if not t.consistent: what.append("tod")
                        fails.append(f"{g}:{'/'.join(what)}")
                results.append("FAIL[" + " ".join(fails) + "]")
    print(label); [print("  seed", i, r) for i, r in enumerate(results)]
