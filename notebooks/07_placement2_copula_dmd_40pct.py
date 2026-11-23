"""Task 07 final rank decision: fit the copula and DMD on the 40% Tucker
core -- the production fit, not an exploratory diagnostic.

Unlike `07_dmd_mode_physical_space_check.py` / `07_dmd_rank_sweep.py`
(which use an 80/20 train/holdout split purely to measure predictive
rank-selection diagnostics), this script fits both the `GaussianCopula`
and DMD on the FULL 2192-timestep core -- no split. Validation for this
fit happens separately, against real held-out data, not via an internal
holdout of this same window.

Inputs: `anomalies_wsh_r248x576_checkpointed300_202401-202410_tucker.npz`
(the 40% rank Tucker core + factors, recovered from the VM run that hit
its 11h wall-clock cap at iter=170/300, err_pct=23.2962% -- see
`tasks/07-tucker-dmd-rank-selection.md` §5). `r_dmd=100` is the
production DMD rank decided in that same task doc.

Run directly: `poetry run python notebooks/07_placement2_copula_dmd_40pct.py`.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import xarray as xr

from core.config import SNAPSHOT_CADENCE_SECONDS
from train.copula import GaussianCopula
from train.dmd import dmd_fit

REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSS_DIR = REPO_ROOT / "results" / "task07" / "gaussianize"

TUCKER_NPZ = GAUSS_DIR / "anomalies_wsh_r248x576_checkpointed300_202401-202410_tucker.npz"
RANK_TAG = "r248x576"
N_BINS = 500
SEED = 42
R_DMD = 100


def _wrap(core: np.ndarray) -> xr.Dataset:
    return xr.Dataset({"core": (("time", "core_lat", "core_lon"), core)})


def _unwrap(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds["core"].values)


def main() -> None:
    z = np.load(TUCKER_NPZ)
    core = z["core"].astype(np.float64)
    n_times, r_lat, r_lon = core.shape
    print(f"core shape {core.shape} (full window, no train/holdout split)")

    print("Fitting GaussianCopula on the full core...")
    t0 = time.time()
    copula = GaussianCopula(n_bins=N_BINS)
    copula.fit(_wrap(core))
    gauss_full = _unwrap(copula.transform(_wrap(core)))
    print(f"  done in {time.time() - t0:.1f}s")

    copula_path = GAUSS_DIR / f"wsh_{RANK_TAG}_copula.joblib"
    joblib.dump(copula, copula_path)
    print(f"Saved {copula_path}")

    gauss_path = GAUSS_DIR / f"wsh_{RANK_TAG}_gaussianized_core.npy"
    np.save(gauss_path, gauss_full)
    print(f"Saved {gauss_path}")

    print(f"Fitting DMD (r_dmd={R_DMD}) on the full Gaussianized core...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    fit = dmd_fit(gauss_full, r_dmd=R_DMD, dt=SNAPSHOT_CADENCE_SECONDS, rng=rng, variant="exact")
    print(f"  done in {time.time() - t0:.1f}s, Phi shape={fit.Phi.shape}")

    dmd_path = GAUSS_DIR / f"wsh_{RANK_TAG}_dmd_r{R_DMD}.npz"
    np.savez_compressed(
        dmd_path,
        Phi=fit.Phi,
        Lambda=fit.Lambda,
        b0=fit.b0,
        lam_cont=fit.lam_cont,
        dt=fit.dt,
        variant=fit.variant,
    )
    print(f"Saved {dmd_path}")


if __name__ == "__main__":
    main()
