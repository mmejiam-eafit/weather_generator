"""How `diagnose_group`'s bootstrap bands behave at 10 vs 100 replicates on the coupled
OU process from `tests/train/test_linear_inverse.py`. Backs the min-max envelope choice
(`docs/model_design.md` §5.6.2, "Bands").

Run from the repo root: `poetry run python notebooks/probes/07_lim_bootstrap_envelope.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import sys
sys.path.insert(0, str(REPO / "tests" / "train"))
import numpy as np
import test_linear_inverse as t
from train.linear_inverse import diagnose_group, diagnostics_summary

z = t._simulate_ou(t.COUPLED_L, t.COUPLED_Q, np.random.default_rng(0))
for n_boot in (10, 100):
    d = diagnose_group({"a": z[:, :1], "b": z[:, 1:2]}, {"a": 1, "b": 1}, t.DT, [1, 2, 4], n_boot,
                       np.random.default_rng(0))
    print(f"--- n_bootstrap={n_boot}")
    print(diagnostics_summary(d))
