"""Lag-1 correlation between consecutive 3 h queries a month past the training window.

Compares the sampler design rejected on 2026-11-22 (every query time jumps
independently from one stationary draw at `t_train_start`) with the chained
design in production (`train.dmd.channel_path`), on one channel with
``φ = 0.9`` at 3 h (about 28 h e-folding). The jump design gives about -0.006,
the chained design about 0.90, the process's own value 0.900
(`docs/model_design.md` §5.8, "Rejected").

The jump design no longer exists in `src/`; `_jump_channels` below restates
it from `train.dmd.ou_transition` and `train.linear_inverse.covariance_factor`.

Run from the repo root: `poetry run python notebooks/probes/sampler_jump_vs_chain.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from train.dmd import ModeOUFit, channel_path, ou_stationary_cov, ou_transition
from train.linear_inverse import covariance_factor

PHI = 0.9
DT = 3 * 3600.0
FIRST_STEP = 240  # 30 days after t_train_start, where phi**dn is about 0
N_QUERIES = 40
N_SEEDS = 2000


def _jump_channels(ou: ModeOUFit, delta_n: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The rejected design: one stationary draw at Δn = 0, each time drawn from it independently."""
    c0 = covariance_factor(ou_stationary_cov(ou)) @ rng.standard_normal(ou.phi.size)
    decay, noise_cov = ou_transition(ou, delta_n)
    noise = covariance_factor(noise_cov) @ rng.standard_normal((delta_n.size, ou.phi.size, 1))
    result: np.ndarray = (decay * c0 + noise[..., 0]).T
    return result


def _lag1(paths: np.ndarray) -> float:
    """Lag-1 correlation pooled over realizations, ``(n_seeds, n_t)``, about the ensemble mean."""
    x = paths - paths.mean(axis=0)
    return float(
        np.sum(x[:, :-1] * x[:, 1:]) / np.sqrt(np.sum(x[:, :-1] ** 2) * np.sum(x[:, 1:] ** 2))
    )


def main() -> None:
    ou = ModeOUFit(
        mode=np.array([0], dtype=np.int64),
        partner=np.array([-1], dtype=np.int64),
        phi=np.array([PHI], dtype=np.float64),
        innovation_cov=np.array([[1.0 - PHI**2]], dtype=np.float64),
        log_magnitude_mean=np.zeros(1, dtype=np.float64),
        phase_offset=np.zeros(1, dtype=np.float64),
        dt=DT,
    )
    delta_n = FIRST_STEP + np.arange(N_QUERIES, dtype=np.float64)
    jump = np.array(
        [_jump_channels(ou, delta_n, np.random.default_rng(s))[0] for s in range(N_SEEDS)]
    )
    chain = np.array(
        [channel_path(ou, delta_n, np.random.default_rng(s))[0] for s in range(N_SEEDS)]
    )
    print(
        f"lag-1 correlation between consecutive 3 h queries: jump design {_lag1(jump):.3f}, "
        f"chained {_lag1(chain):.3f}, process {PHI:.3f}"
    )


if __name__ == "__main__":
    main()
