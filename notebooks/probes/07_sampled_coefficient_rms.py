"""Per-coefficient std of the real core, its DMD-span projection and the sampler, and the
share beyond the copula's 4.75 saturation (`docs/model_design.md` §5.6.1). Executes the
top of `07_production_sampler_memory.py`.

Run from the repo root: `poetry run python notebooks/probes/07_sampled_coefficient_rms.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import numpy as np
SP = str(RESULTS)  # cache directory; was the session scratchpad
src = open(HERE / "07_production_sampler_memory.py").read()
exec(src.split("# --- per-channel")[0])
steps = np.arange(T, dtype=np.float64)
idx = np.arange(200, T, 10)  # 200 snapshots past the deterministic transient
Phi = dmd.Phi.astype(np.complex128)
a = mode_amplitudes(dmd, ou, _channel_path(ou, steps, np.random.default_rng(0)), steps)[:, idx]
syn = (Phi @ a).real.T  # (n, N)
train_span = (Phi @ (dmd.b0[:, None] * dmd.Lambda[:, None] ** steps[idx][None, :] + eta[:, idx])).real.T
real = np.asarray(X[idx], dtype=np.float64)
for name, v in (("real Gaussianized core", real), ("its DMD-span projection", train_span), ("sampler", syn)):
    sd = v.std(0)
    print(f"{name:26s} per-coefficient std quantiles [5,50,95,99.9]%: {np.round(np.percentile(sd, [5, 50, 95, 99.9]), 3)}"
          f"   |z|>4.75: {np.mean(np.abs(v) > 4.75):.4f}")
