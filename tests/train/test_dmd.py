"""Tests for train.dmd (see docs/model_design.md §5.6, §5.6.1)."""

from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose

from train.dmd import DMDFit, dmd_fit, fit_mode_ou


@pytest.mark.parametrize("variant", ["exact", "bop", "hankel"])
def test_analytic_traveling_wave(variant: Literal["exact", "bop", "hankel"]) -> None:
    """Synthetic A cos(kx - ωt) is recovered as a complex-conjugate pair with correct ω.

    (see docs/model_design.md §5.6; all three variants must agree on the
    physics — a real traveling wave's DMD spectrum contains a mode at
    ``+iω`` and its conjugate at ``-iω``.)
    """
    rng = np.random.default_rng(0)
    n_times, n_x = 200, 64
    k, omega = 0.3, 0.7
    x = np.arange(n_x)
    t = np.arange(n_times) * 0.1
    core = np.cos(k * x[None, :] - omega * t[:, None])[:, :, None].astype(np.float64)

    fit = dmd_fit(core, r_dmd=10, dt=0.1, rng=rng, variant=variant, bop_num_trials=20)
    recovered_omegas = np.abs(fit.lam_cont.imag)

    best = np.argmin(np.abs(recovered_omegas - omega))
    assert np.abs(recovered_omegas[best] - omega) < 0.02

    dominant = fit.Lambda[best]
    assert np.min(np.abs(fit.Lambda - np.conj(dominant))) < 1e-6


def test_bop_enforces_conjugate_pairs() -> None:
    """BOP-DMD's eig_constraints={"conjugate_pairs"} yields exact conjugate eigenvalue pairs.

    (see docs/model_design.md §5.6, §9 gotcha #8 — with variant="bop", pairing
    is enforced at fit time via PyDMD's `eig_constraints`, unlike "exact"/
    "hankel" where the caller must preserve pairing manually at simulation
    time.)
    """
    rng = np.random.default_rng(11)
    n_times, n_x = 150, 32
    k, omega = 0.2, 1.1
    x = np.arange(n_x)
    t = np.arange(n_times) * 0.1
    core = np.cos(k * x[None, :] - omega * t[:, None])[:, :, None].astype(np.float64)

    fit = dmd_fit(core, r_dmd=6, dt=0.1, rng=rng, variant="bop", bop_num_trials=20)

    oscillatory = fit.Lambda[np.abs(fit.Lambda.imag) > 1e-6]
    assert oscillatory.size > 0
    for lam in oscillatory:
        assert np.any(np.abs(fit.Lambda - np.conj(lam)) < 1e-6)


def test_pydmd_snapshot_orientation() -> None:
    """core.reshape(T, N_state).T feeds PyDMD (N_state, T) snapshots, not the reverse.

    (see docs/model_design.md §9 gotcha #22.) Uses an asymmetric shape
    (T != N_state) and r_dmd == N_state: a silent transpose swap would feed
    PyDMD a (T, N_state)-shaped matrix instead, producing a differently
    shaped (and wrong) `Phi` that fails the shape assertion immediately,
    and — even were shapes coincidentally compatible — would not reproduce
    the initial condition exactly.
    """
    rng = np.random.default_rng(7)
    n_times, r_ocean, r_var = 50, 5, 2  # N_state = 10, T = 50 — asymmetric on purpose
    n_state = r_ocean * r_var
    core = rng.normal(size=(n_times, r_ocean, r_var))

    fit = dmd_fit(core, r_dmd=n_state, dt=1.0, rng=rng, variant="exact")

    assert fit.Phi.shape == (n_state, n_state)
    x0 = core.reshape(n_times, n_state)[0]
    reconstructed = (fit.Phi @ fit.b0).real
    assert_allclose(reconstructed, x0, atol=1e-5)


def test_mode_ou_parameter_recovery() -> None:
    """A synthetic OU trace with known (theta, sigma) is recovered within 5%.

    (see docs/model_design.md §5.6.1; replaces the deleted `test_ou.py`'s
    equivalent case.) `fit_mode_ou` reads its residual amplitude
    ``η_k(t) = a_k(t) - â_k(t)`` off of `core` via `Phi`/`b0`/`Lambda`, so a
    single-mode `DMDFit` with `Phi = [[1]]` and `b0 = [0]` makes
    ``η_k(t) == core[t, 0, 0]`` exactly, letting a hand-generated discrete
    AR(1)/OU trace with known parameters drive the recovery directly.
    """
    rng = np.random.default_rng(0)
    theta_true = 0.1
    sigma_true = 1.0
    dt = 1.0
    phi_true = np.exp(-theta_true * dt)
    stationary_variance = sigma_true**2 / (2.0 * theta_true)
    innovation_variance = sigma_true**2 * (1.0 - phi_true**2) / (2.0 * theta_true)

    def complex_normal(variance: float, rng: np.random.Generator) -> complex:
        std = np.sqrt(variance / 2.0)
        return complex(rng.normal(0.0, std), rng.normal(0.0, std))

    n_steps = 50_000
    trace = np.empty(n_steps, dtype=np.complex128)
    trace[0] = complex_normal(stationary_variance, rng)
    for step in range(1, n_steps):
        trace[step] = phi_true * trace[step - 1] + complex_normal(innovation_variance, rng)

    core = trace.reshape(n_steps, 1, 1)
    dmd = DMDFit(
        Phi=np.array([[1.0 + 0.0j]], dtype=np.complex64),
        Lambda=np.array([1.0 + 0.0j], dtype=np.complex128),
        b0=np.array([0.0 + 0.0j], dtype=np.complex128),
        lam_cont=np.array([0.0 + 0.0j], dtype=np.complex128),
        dt=dt,
        variant="exact",
    )

    fit = fit_mode_ou(core, dmd)

    assert abs(fit.theta[0] - theta_true) / theta_true < 0.05
    assert abs(fit.sigma[0] - sigma_true) / sigma_true < 0.05
