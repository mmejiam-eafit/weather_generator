"""Task 07: animate each DMD mode's own time-varying contribution, in
physical space, across the full 9-month window.

`07_dmd_mode_physical_space_check.py` showed each mode's fixed spatial
SHAPE once. The "time-step-ness" lives entirely in that mode's amplitude
`a_k(t)`, a single complex number per timestep -- animating
`Re(a_k(t) * Phi[:, k])`, projected through the Tucker factors, shows how
one mode's own contribution to the field evolves.

Deliberately NOT `b0 * Lambda**t` extrapolated forward from t=0 of
training -- these four modes all decay with tau of 2-3 days, so that
naive forward projection underflows to numerical zero within the first
few weeks and would show nothing for the other ~8.5 months (the same
underflow issue `07_dmd_rank_sweep.py` hit and worked around). Instead,
`a_k(t) = (Phi^+ x(t))_k` is computed from the ACTUAL measured
(Gaussianized) core at every sampled timestep -- the real, data-driven
amplitude trajectory, not a one-shot extrapolation.

Same 548-frame sampling as the other timelapses (00/12 UTC daily,
2024-01-01 to 2024-10-01). No VM needed here -- the 20% core is only
~313MB, fits in memory whole, so all 548 (indeed all 2192) timesteps'
amplitudes are computed from data already local.

Run directly: `poetry run python notebooks/07_dmd_mode_timelapse_gif.py`.
"""

import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

from core.config import SNAPSHOT_CADENCE_SECONDS
from train.copula import GaussianCopula
from train.dmd import dmd_fit

REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSS_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
WINDOW_START = datetime(2024, 1, 1)

TRAIN_FRACTION = 0.8
N_BINS = 500
SEED = 42
R_DMD = 100
# The exact 4 modes shown in dmd_modes_physical_space.png, hardcoded so
# this animates precisely those, not a possibly-different re-selection.
SELECTED_MODES = [84, 98, 92, 99]


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

    core_train = core[:n_train]
    copula = GaussianCopula(n_bins=N_BINS)
    copula.fit(_wrap(core_train))

    rng = np.random.default_rng(SEED)
    fit = dmd_fit(_unwrap(copula.transform(_wrap(core_train))), r_dmd=R_DMD, dt=SNAPSHOT_CADENCE_SECONDS, rng=rng, variant="exact")
    print(f"Fitted DMD: r_dmd={R_DMD}, Phi shape={fit.Phi.shape}")

    # Gaussianize and project ALL 2192 timesteps at once (cheap; core fits
    # whole in memory) -- transform() only requires the fitted per-cell
    # LUTs, doesn't care whether a timestep was in the training split.
    gauss_full = _unwrap(copula.transform(_wrap(core)))
    phi_pinv = np.linalg.pinv(fit.Phi)  # (r_dmd, n_state)
    a_full = phi_pinv @ gauss_full.reshape(n_times, r_lat * r_lon).T  # (r_dmd, n_times)
    print(f"Projected amplitudes: a_full shape={a_full.shape}")

    time_indices = np.array([i for i in range(n_times) if i % 8 in (0, 4)])
    n_frames = len(time_indices)
    print(f"n_frames={n_frames}")

    ocean_mask = np.load(GAUSS_DIR / "era2_ocean_mask.npy")
    land = ~ocean_mask

    def contribution(k: int, t: int) -> np.ndarray:
        mode_core = (a_full[k, t] * fit.Phi[:, k]).reshape(r_lat, r_lon)
        pattern = np.real(factor_lat @ mode_core @ factor_lon.T)
        return np.where(land, np.nan, pattern)

    sample_i = list(range(0, n_frames, 9))
    vmaxes = {}
    for k in SELECTED_MODES:
        sample_vals = [contribution(k, int(time_indices[i])) for i in sample_i]
        vmaxes[k] = float(np.nanpercentile(np.abs(sample_vals), 99))
    print("per-mode robust vmax:", vmaxes)

    fig, axes = plt.subplots(1, len(SELECTED_MODES), figsize=(5 * len(SELECTED_MODES), 5), constrained_layout=True)
    ims = []
    for ax, k in zip(axes, SELECTED_MODES):
        im = ax.imshow(
            contribution(k, int(time_indices[0])), origin="lower", cmap="RdBu_r",
            vmin=-vmaxes[k], vmax=vmaxes[k], aspect="auto",
        )
        ax.set_title(f"mode {k}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
        ims.append(im)

    suptitle = fig.suptitle("")

    def update(i: int):
        t = int(time_indices[i])
        date = WINDOW_START + timedelta(hours=3 * t)
        for im, k in zip(ims, SELECTED_MODES):
            im.set_data(contribution(k, t))
        suptitle.set_text(f"DMD mode contributions, r_dmd={R_DMD} -- {date:%Y-%m-%d %H:%M} UTC, frame {i + 1}/{n_frames}")
        return [*ims, suptitle]

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_path = GAUSS_DIR / "dmd_mode_timelapse.gif"
    ani.save(out_path, writer=PillowWriter(fps=10), dpi=45)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    notebooks_copy = NOTEBOOKS_DIR / out_path.name
    shutil.copy2(out_path, notebooks_copy)
    print(f"Copied to {notebooks_copy}")


if __name__ == "__main__":
    main()
