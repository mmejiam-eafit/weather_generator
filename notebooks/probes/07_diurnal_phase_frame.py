"""24 h and 12 h harmonics of the standardized anomaly in physical space (Tucker-40%
reconstruction): amplitude, and whether the phase is locked to UTC or to local solar time
(`docs/open_questions.md` §7.2). Reads the cached Tucker npz and era-2 ocean mask; caches
the harmonic maps next to them.

Run from the repo root: `poetry run python notebooks/probes/07_diurnal_phase_frame.py`.

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
from scipy.ndimage import binary_dilation
SP = str(RESULTS)  # cache directory; was the session scratchpad
R = str(RESULTS)
npz = np.load(f"{R}/anomalies_wsh_r248x576_checkpointed300_202401-202410_tucker.npz")
core = npz["core"]; F_lat = npz["factor_lat"].astype(np.float64); F_lon = npz["factor_lon"].astype(np.float64)
ocean = np.load(f"{R}/era2_ocean_mask.npy").astype(bool)
print("core", core.shape, core.dtype, "F", F_lat.shape, F_lon.shape, "ocean", ocean.shape, ocean.mean())
T = core.shape[0]
t_hours = np.arange(T) * 3.0
lons = np.arange(1440) * 0.25
coastal = ocean & binary_dilation(~ocean, iterations=8)  # within ~2 deg of land
open_ocean = ocean & ~binary_dilation(~ocean, iterations=40)  # > ~10 deg from land
for period in (24.0, 12.0):
    w = np.exp(-2j * np.pi * t_hours / period)
    a = np.tensordot(w, core.astype(np.float64), axes=(0, 0)) * (2.0 / T)  # (248, 576) complex
    A = F_lat @ a @ F_lon.T  # (621, 1440): x(t) ≈ Re(A* e^{i w t})... amplitude |A|
    amp = np.abs(A)
    # x(t) = Re(conj(A) e^{iωt})?  With a = 2/T Σ x e^{-iωt}, x ≈ Re(a e^{iωt}) -> peak at ωt = -arg(a)
    t_peak_utc = (-np.angle(A) / (2 * np.pi) * period) % period
    t_peak_local = (t_peak_utc + lons[None, :] / 15.0) % period
    def concentration(tp, mask):
        ang = 2 * np.pi * tp[mask] / period
        wts = amp[mask]
        z = np.sum(wts * np.exp(1j * ang)) / wts.sum()
        return np.abs(z), (np.angle(z) / (2 * np.pi) * period) % period
    print(f"=== {period:.0f} h harmonic of the standardized anomaly (Tucker-40% reconstruction; units = cell std)")
    for name, m in (("all ocean", ocean), ("coastal (<~2deg)", coastal), ("open ocean (>~10deg)", open_ocean)):
        Ru, pu = concentration(t_peak_utc, m)
        Rl, pl = concentration(t_peak_local, m)
        print(f"  {name:22s} n={m.sum():7d}  median amp {np.median(amp[m]):.3f}  p90 {np.percentile(amp[m],90):.3f}"
              f"  | phase coherence UTC R={Ru:.2f} (peak {pu:4.1f}h UTC)  local R={Rl:.2f} (peak {pl:4.1f}h LST)")
    np.save(f"{SP}/diurnal_A_{int(period)}h.npy", A)
# latitude-band breakdown of the 24h amplitude (era-2 lats assumed -77.5..77.5)
A = np.load(f"{SP}/diurnal_A_24h.npy"); amp = np.abs(A)
lats = -77.5 + 0.25 * np.arange(621)
for lo, hi in ((-78, -50), (-50, -20), (-20, 20), (20, 50), (50, 78)):
    band = (lats >= lo) & (lats < hi)
    m = ocean & band[:, None]
    print(f"  lat [{lo:4d},{hi:4d}): median 24h amp {np.median(amp[m]):.3f}")
