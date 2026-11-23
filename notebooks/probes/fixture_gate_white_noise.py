"""Fits the tier-1 test fixture (white noise) under the generative fidelity gate for model
seeds 0-3 and prints the gate's summary: every seed fails (`docs/open_questions.md` §5.9).

Run from the repo root: `poetry run python notebooks/probes/fixture_gate_white_noise.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import sys, tempfile
from pathlib import Path
import numpy as np
from conftest import write_tier1_archive, small_tier1_model
from train.fidelity import fidelity_summary
import weather.model as wm

orig = wm._require_fidelity
captured = {}
def spy(fid):
    captured.update(fid)
    orig(fid)
wm._require_fidelity = spy
for seed in (0, 1, 2, 3):
    with tempfile.TemporaryDirectory() as d:
        write_tier1_archive(Path(d), np.random.default_rng(0))
        captured.clear()
        try:
            small_tier1_model(seed).fit(Path(d))
            status = "PASSED"
        except ValueError as e:
            status = "FAILED"
        print(f"model seed {seed}: {status}")
        if seed == 0:
            for g, dg in captured.items():
                print(fidelity_summary(dg))
