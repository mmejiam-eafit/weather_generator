"""Where the sampler's energy goes: |Lambda|, the deterministic term, per-pair log-magnitude
statistics, and per-mode weights (per mode, sampled and training amplitudes match).
Executes parts of `07_production_sampler_memory.py`.

Run from the repo root: `poetry run python notebooks/probes/07_sampler_energy.py`.

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
from scipy.stats import skew
SP = str(RESULTS)  # cache directory; was the session scratchpad
src = open(HERE / "07_production_sampler_memory.py").read()
exec(src.split("# --- per-channel")[0])
exec(src.split("# --- data EOFs (top 20) and real PCs")[1].split("lags2 = ")[0])

steps = np.arange(T, dtype=np.float64)
det = dmd.b0[:, None] * dmd.Lambda[:, None] ** steps[None, :]
absL = np.abs(dmd.Lambda)
print(f"|Lambda|: max {absL.max():.6f}, n>1: {(absL > 1).sum()}, quantiles {np.round(np.quantile(absL, [0, .5, .9, 1]), 4)}")
print(f"deterministic energy |b0 Λ^t|^2 summed over modes: t=0 {np.sum(np.abs(det[:, 0])**2):.3e}, "
      f"t=T/2 {np.sum(np.abs(det[:, T//2])**2):.3e}, t=T-1 {np.sum(np.abs(det[:, -1])**2):.3e}")
# training reconstruction in the top-20 EOF subspace
recon = (EtPhi @ (det + eta)).real.T
print(f"training reconstruction energy / real (top-20 subspace): "
      f"{np.sum((recon - recon.mean(0))**2) / np.sum(pcs**2):.3f}")
# per-pair magnitude statistics: training vs lognormal model
pair = ~real
logm = np.log(np.abs(eta[ou.mode[pair]]) + 1e-12)
train_e2 = np.mean(np.abs(eta[ou.mode[pair]])**2, 1)
P = ou.innovation_cov / (1 - np.outer(ou.phi, ou.phi))
var_c = np.diag(P)[pair]
model_e2 = np.exp(2 * ou.log_magnitude_mean[pair] + 2 * var_c)
ratio = model_e2 / train_e2
print(f"pairs: Var(log|η|) quantiles {np.round(np.quantile(logm.var(1), [0, .5, .9, 1]), 3)} (Rayleigh: 0.411)")
print(f"       skew(log|η|) quantiles {np.round(np.quantile(skew(logm, axis=1), [0, .5, .9, 1]), 2)}")
print(f"       min log|η| per pair, quantiles {np.round(np.quantile(logm.min(1), [0, .5, 1]), 2)}")
print(f"       model E|η|^2 / training E|η|^2 quantiles {np.round(np.quantile(ratio, [0, .5, .9, 1]), 2)}")
print(f"       stationary channel var vs training var(log|η|): median ratio {np.median(var_c / logm.var(1)):.3f}")
# real channels
re_c = eta[ou.mode[real]].real
print(f"real channels: model var / training var: {np.round(np.diag(P)[real] / re_c.var(1), 3)}")
# synthetic energy by time segment
rng = np.random.default_rng(0)
a = mode_amplitudes(dmd, ou, _channel_path(ou, steps, rng), steps)
zs = (EtPhi @ a).real.T
for lo, hi in ((0, 100), (100, 1000), (1000, T)):
    print(f"synthetic top-20 energy per step, t in [{lo},{hi}): {np.mean(np.sum(zs[lo:hi]**2, 1)):.3e} "
          f"(real {np.mean(np.sum(pcs[lo:hi]**2, 1)):.3e})")
# which modes carry the synthetic energy
contrib_syn = np.mean(np.abs(a)**2, 1) * np.sum(np.abs(EtPhi)**2, 0)
contrib_real = np.mean(np.abs(det + eta)**2, 1) * np.sum(np.abs(EtPhi)**2, 0)
top = np.argsort(contrib_syn)[::-1][:6]
for k in top:
    kind = "pair" if k in ou.mode[pair] or k in ou.partner[pair] else "real"
    print(f"  mode {k:3d} ({kind}) |Λ|={absL[k]:.4f} period={2*np.pi/abs(np.angle(dmd.Lambda[k]))*3 if np.angle(dmd.Lambda[k]) != 0 else np.inf:7.1f}h"
          f"  synthetic weight {contrib_syn[k]:.3e}  training weight {contrib_real[k]:.3e}")
