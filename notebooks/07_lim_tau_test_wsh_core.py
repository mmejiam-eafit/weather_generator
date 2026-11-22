"""Group dynamics diagnostics on the production wsh 40% Gaussianized core.

Runs the training pipeline's diagnostic stage (`train.linear_inverse.
diagnose_group`, `docs/model_design.md` §5.6.2) on task 07's production core:
a linear inverse model at a 3 h training lag, the tau test at 3-48 h, and
bootstrap envelopes from the fitted model (see `docs/correlated_variables.md`
§4.4, "First application").

Single member only: no other variable's core exists yet, so this measures
goodness of fit, not cross-variable coupling.

Run directly: `poetry run python notebooks/07_lim_tau_test_wsh_core.py`
(~3 min, ~8 GB peak memory).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from train.linear_inverse import diagnose_group, diagnostics_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE = REPO_ROOT / "results" / "task07" / "gaussianize" / "wsh_r248x576_gaussianized_core.npy"
DT = 3 * 3600.0
TAU_TEST_LAGS = [1, 2, 4, 8, 16]
RANKS = (20, 100)
N_BOOTSTRAP = 100
SEED = 0


def main() -> None:
    core = np.asarray(np.load(CORE, mmap_mode="r"))
    rng = np.random.default_rng(SEED)
    for rank in RANKS:
        diagnostics = diagnose_group(
            {"wsh": core},
            ranks={"wsh": rank},
            dt=DT,
            test_lags=TAU_TEST_LAGS,
            n_bootstrap=N_BOOTSTRAP,
            rng=rng,
        )
        print(diagnostics_summary(diagnostics), end="\n\n")


if __name__ == "__main__":
    main()
