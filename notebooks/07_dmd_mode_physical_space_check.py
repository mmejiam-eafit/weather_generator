"""Task 07: fit real DMD modes on the Gaussianized core and project them
back to physical (lat, lon) space through the Tucker factors.

Direct empirical follow-up to the "core looks like noise, but a DMD mode
is a linear combination across every core index -- project that
combination back through the (smooth) Tucker factors and it becomes a
coherent spatial pattern again" claim (§11,
`tasks/07-tucker-dmd-rank-selection.md`). Same fitting procedure as
`07_dmd_rank_sweep.py` (train/holdout split, `GaussianCopula` fit on train
only, `dmd_fit(..., variant="exact")`), but this time actually reshaping
and visualizing `Phi`'s columns instead of just reading off aggregate
error/stability numbers.

A DMD mode `Phi[:, k]` lives in the *core's* flattened state space
(`r_lat*r_lon`,) -- reshape it back to `(r_lat, r_lon)` (same row-major
layout `dmd_fit` used to flatten `core[t]` in the first place) and project
through `factor_lat @ mode @ factor_lon.T`, exactly the same operation
used everywhere else in this task to go from core to physical space.
Complex-valued (`Phi` is complex, `factor_lat`/`factor_lon` are real) --
plots the real part, the physically meaningful "spatial shape" of that
mode's oscillation.

Picks the top few modes by `|b0|` (initial amplitude -- how much this mode
actually contributes to the fit), skipping the redundant conjugate partner
of a pair already shown (real dynamics come in conjugate pairs with
identical spatial shape, mirrored oscillation direction).

Run directly: `poetry run python notebooks/07_dmd_mode_physical_space_check.py`.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from core.config import SNAPSHOT_CADENCE_SECONDS
from train.copula import GaussianCopula
from train.dmd import dmd_fit

REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSS_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

TRAIN_FRACTION = 0.8
N_BINS = 500
SEED = 42
R_DMD = 100  # the decided production rank (tasks/07 §10)
N_MODES_TO_SHOW = 4


def _wrap(core: np.ndarray) -> xr.Dataset:
    return xr.Dataset({"core": (("time", "core_lat", "core_lon"), core)})


def _unwrap(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["core"].values)


def main() -> None:
    z = np.load(GAUSS_DIR / "anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz")
    core = z["core"].astype(np.float64)
    factor_lat, factor_lon = z["factor_lat"], z["factor_lon"]
    n_times, r_lat, r_lon = core.shape
    n_train = int(round(n_times * TRAIN_FRACTION))
    print(f"core shape {core.shape}, n_train={n_train}")

    core_train = core[:n_train]
    copula = GaussianCopula(n_bins=N_BINS)
    copula.fit(_wrap(core_train))
    gauss_train = _unwrap(copula.transform(_wrap(core_train)))

    rng = np.random.default_rng(SEED)
    fit = dmd_fit(gauss_train, r_dmd=R_DMD, dt=SNAPSHOT_CADENCE_SECONDS, rng=rng, variant="exact")
    print(f"Fitted DMD: r_dmd={R_DMD}, Phi shape={fit.Phi.shape}")

    ocean_mask = np.load(GAUSS_DIR / "era2_ocean_mask.npy")
    land = ~ocean_mask

    order = np.argsort(-np.abs(fit.b0))
    selected: list[int] = []
    for k in order:
        if len(selected) >= N_MODES_TO_SHOW:
            break
        # skip if this mode's conjugate partner is already selected
        is_dup = any(
            np.isclose(fit.lam_cont[k], np.conj(fit.lam_cont[j]), atol=1e-8) and np.isclose(fit.b0[k], np.conj(fit.b0[j]), atol=1e-6)
            for j in selected
        )
        if not is_dup:
            selected.append(int(k))

    fig, axes = plt.subplots(1, N_MODES_TO_SHOW, figsize=(5 * N_MODES_TO_SHOW, 5), constrained_layout=True)
    for ax, k in zip(axes, selected):
        mode_core = fit.Phi[:, k].reshape(r_lat, r_lon)
        pattern = np.real(factor_lat @ mode_core @ factor_lon.T)
        pattern = np.where(land, np.nan, pattern)

        omega = float(np.imag(fit.lam_cont[k]))  # rad/s
        decay_rate = float(np.real(fit.lam_cont[k]))  # 1/s
        if abs(omega) > 1e-12:
            period_days = (2 * np.pi / abs(omega)) / 86400.0
            period_str = f"period={period_days:.1f}d"
        else:
            period_str = "non-oscillatory"
        if decay_rate < -1e-12:
            decay_str = f"decay tau={-1.0 / decay_rate / 86400.0:.1f}d"
        elif decay_rate > 1e-12:
            decay_str = f"GROWING tau={1.0 / decay_rate / 86400.0:.1f}d"
        else:
            decay_str = "neutral"

        vmax = float(np.nanpercentile(np.abs(pattern), 99))
        im = ax.imshow(pattern, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"mode {k} (|b0|={np.abs(fit.b0[k]):.2f})\n{period_str}, {decay_str}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("DMD modes (r_dmd=100, 20% Tucker core), projected back to physical space")

    out_path = GAUSS_DIR / "dmd_modes_physical_space.png"
    fig.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")


if __name__ == "__main__":
    main()
