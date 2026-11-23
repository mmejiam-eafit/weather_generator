"""The production mode OU against its own channels (median phi 0.605; channel ACF vs phi^k)
and the production sampler against the data in the top-20 EOF subspace (ACF, energy
ratio ~17x there). Uses the cached task-07 DMD; replaces `_residual_amplitudes` with a
chunked version to fit in memory (no longer needed since `measured_amplitudes`).
Needs `07_leading_pcs_by_utc_hour.py` first.

Run from the repo root: `poetry run python notebooks/probes/07_production_sampler_memory.py`.

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
import train.dmd as dmd_mod
from train.dmd import DMDFit, fit_mode_ou, mode_amplitudes
from train.dmd import channel_path as _channel_path  # moved from weather.simulate

SP = str(RESULTS)  # cache directory; was the session scratchpad
R = str(RESULTS)
core = np.load(f"{R}/wsh_r248x576_gaussianized_core.npy", mmap_mode="r")
T = core.shape[0]; N = core.shape[1] * core.shape[2]
X = core.reshape(T, N)
d = np.load(f"{R}/wsh_r248x576_dmd_r100.npz")
dmd = DMDFit(Phi=d["Phi"], Lambda=d["Lambda"], b0=d["b0"], lam_cont=d["lam_cont"],
             dt=float(d["dt"]), variant=str(d["variant"]))
pinv_phi = np.linalg.pinv(dmd.Phi.astype(np.complex128))
CH = 128

def chunked_residual_amplitudes(state, fit):
    measured = np.empty((fit.Phi.shape[1], T), dtype=np.complex128)
    for s in range(0, T, CH):
        measured[:, s:s + CH] = pinv_phi @ np.asarray(X[s:s + CH], dtype=np.float64).T
    steps = np.arange(T, dtype=np.float64)
    return measured - fit.b0[:, None] * fit.Lambda[:, None] ** steps[None, :]

dmd_mod._residual_amplitudes = chunked_residual_amplitudes  # memory only; same formula
eta = chunked_residual_amplitudes(None, dmd)
ou = fit_mode_ou(np.zeros((T, 1)), dmd)  # state unused by the chunked replacement
real = ou.partner < 0
print(f"channels: {real.sum()} real, {(~real).sum()} pairs; phi quantiles "
      f"{np.round(np.quantile(ou.phi, [0, .1, .5, .9, 1]), 4)}; clipped at 0.9999: {(ou.phi >= 0.9999).sum()}")

# --- per-channel: training-channel ACF vs phi^k
chan = np.concatenate([eta[ou.mode[real]].real,
                       np.log(np.abs(eta[ou.mode[~real]]) + 1e-12) - ou.log_magnitude_mean[~real, None]])
lags = np.array([1, 2, 4, 8, 16, 32, 64])
labels = ["3h", "6h", "12h", "1d", "2d", "4d", "8d"]
c = chan - chan.mean(1, keepdims=True)
acf = np.stack([np.sum(c[:, l:] * c[:, :-l], 1) / np.sum(c * c, 1) * T / (T - l) for l in lags], 1)
model = ou.phi[:, None] ** lags[None, :]
print("per-channel ACF, median over channels (empirical | phi^k):")
print("   " + "  ".join(f"{lab}: {e:.3f}|{m:.3f}" for lab, e, m in zip(labels, np.median(acf, 0), np.median(model, 0))))
print("   fraction of channels with empirical ACF > phi^k + 0.1: " +
      "  ".join(f"{lab}: {f:.2f}" for lab, f in zip(labels, np.mean(acf > model + 0.1, 0))))

# --- data EOFs (top 20) and real PCs
mean = np.zeros(N)
for s in range(0, T, CH): mean += np.asarray(X[s:s + CH], dtype=np.float64).sum(0)
mean /= T
pcs = np.load(f"{SP}/wsh_pcs400.npy")[:, :20]
E = np.zeros((N, 20))
for s in range(0, T, CH): E += (np.asarray(X[s:s + CH], dtype=np.float64) - mean).T @ pcs[s:s + CH]
E /= np.sum(pcs**2, 0)
EtPhi = E.T @ dmd.Phi.astype(np.complex128)  # (20, r_dmd)

def acf_energy(z, lags):
    z = z - z.mean(0); v = np.sum(z * z)
    return np.array([np.sum(z[l:] * z[:-l]) / v * len(z) / (len(z) - l) for l in lags])

lags2 = [1, 2, 4, 8, 16, 32, 64, 128]; lab2 = labels + ["16d"]
real_acf = acf_energy(pcs, lags2)
delta_n = np.arange(T, dtype=np.float64)
syn_acfs, syn_energy = [], []
for seed in range(5):
    rng = np.random.default_rng(seed)
    path = _channel_path(ou, delta_n, rng)
    a = mode_amplitudes(dmd, ou, path, delta_n)
    zs = (EtPhi @ a).real.T
    syn_acfs.append(acf_energy(zs, lags2)); syn_energy.append(np.sum((zs - zs.mean(0))**2) / np.sum(pcs**2))
syn_acfs = np.array(syn_acfs)
print("top-20 data-EOF subspace, energy-weighted ACF: real | production sampler (5 seeds, min-max)")
for lab, r, lo, hi in zip(lab2, real_acf, syn_acfs.min(0), syn_acfs.max(0)):
    print(f"   {lab:>4s}: {r:.3f} | [{lo:.3f}, {hi:.3f}]")
print(f"   sampler energy in this subspace / real: {np.round(syn_energy, 3)}")
