"""Generative fidelity stage on the production wsh 40% Gaussianized core.

Runs the training pipeline's §5.6.3 stage (`train.fidelity.diagnose_fidelity`,
`docs/model_design.md` §5.6.3) on task 07's production core and DMD: the
cached `wsh_r248x576_gaussianized_core.npy` (era 2, 2024-01-01 to
2024-10-01, T = 2192, 3-hourly from 00Z) and `wsh_r248x576_dmd_r100.npz`,
with the mode OU refit by the production `fit_mode_ou`. Prints
`fidelity_summary`; the result is recorded in §5.6.3, "First application".

Memory: capped at `MEMORY_CAP_BYTES` (8 GiB) with `RLIMIT_AS`, task 07's
technique (`notebooks/07_dmd_rank_cv_check.py`, task 07 §4), so an overrun
fails as a clean `MemoryError` instead of pushing the workstation into the
OOM killer. The script refuses to start unless `MemAvailable` covers the cap
plus `MEMORY_MARGIN_BYTES`. Budget, about 5.5 GB resident at the peak
(inside `group_pcs`), leaving room for the address space BLAS threads and
allocator arenas reserve beyond it:

- the core, loaded once (not memory-mapped, which would count against
  `RLIMIT_AS` again): 2.5 GB float64; the state is a reshape view of it;
- `group_pcs`'s float64 copy of it: 2.5 GB;
- ``pinv(Φ)``: 0.23 GB; ``Re Φ`` and ``Im Φ`` in float64: 0.27 GB;
- per-coefficient sampled chunks: 165,888 x 64 x 8 B, about 85 MB.

Run directly: `poetry run python notebooks/07_generative_fidelity_wsh_core.py`.
"""

import resource
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from train.dmd import DMDFit, fit_mode_ou
from train.fidelity import diagnose_fidelity, fidelity_summary

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RESULTS: Final[Path] = REPO_ROOT / "results" / "task07" / "gaussianize"
CORE: Final[Path] = RESULTS / "wsh_r248x576_gaussianized_core.npy"
DMD: Final[Path] = RESULTS / "wsh_r248x576_dmd_r100.npz"
TENSOR: Final[str] = "logE_wsh"
T_START: Final[np.datetime64] = np.datetime64("2024-01-01T00:00")
STEP: Final[np.timedelta64] = np.timedelta64(3, "h")
LAGS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64, 128)
PCS: Final[int] = 20
REPLICATES: Final[int] = 100
SEED: Final[int] = 0
MEMORY_CAP_BYTES: Final[int] = 8 * 1024**3
MEMORY_MARGIN_BYTES: Final[int] = 1 * 1024**3


def _available_bytes() -> int:
    """``MemAvailable`` from /proc/meminfo, in bytes."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _apply_memory_cap() -> None:
    """Refuse to start without headroom, then cap this process's address space."""
    available = _available_bytes()
    if available < MEMORY_CAP_BYTES + MEMORY_MARGIN_BYTES:
        raise SystemExit(
            f"only {available / 1024**3:.1f} GiB available; this run needs its "
            f"{MEMORY_CAP_BYTES / 1024**3:.0f} GiB cap plus "
            f"{MEMORY_MARGIN_BYTES / 1024**3:.0f} GiB of margin"
        )
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))


def _load_dmd() -> DMDFit:
    """The cached task-07 production DMD as a `DMDFit`."""
    data = np.load(DMD)
    return DMDFit(
        Phi=data["Phi"],
        Lambda=data["Lambda"],
        b0=data["b0"],
        lam_cont=data["lam_cont"],
        dt=float(data["dt"]),
        variant=str(data["variant"]),
    )


def main() -> None:
    _apply_memory_cap()
    core = np.load(CORE)
    state = core.reshape(core.shape[0], -1)
    dmd = _load_dmd()
    ou = fit_mode_ou(state, dmd)
    times = T_START + STEP * np.arange(state.shape[0])
    diagnostics = diagnose_fidelity(
        state,
        {TENSOR: state.shape[1]},
        dmd,
        ou,
        times=times,
        lags=list(LAGS),
        ranks={TENSOR: PCS},
        n_replicates=REPLICATES,
        rng=np.random.default_rng(SEED),
    )
    print(fidelity_summary(diagnostics))
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"\npeak resident memory {peak_gib:.1f} GiB (cap {MEMORY_CAP_BYTES / 1024**3:.0f} GiB)")


if __name__ == "__main__":
    main()
