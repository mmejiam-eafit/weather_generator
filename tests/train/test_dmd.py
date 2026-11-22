"""Tests for train.dmd (see docs/model_design.md §5.6, §5.6.1)."""

from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose

from train.dmd import (
    DMDFit,
    ModeOUFit,
    conjugate_pairs,
    dmd_fit,
    fit_mode_ou,
    group_state,
    mode_amplitudes,
    ou_stationary_cov,
    ou_transition,
    split_group_state,
)


def _traveling_wave(n_times: int, dt: float, omega: float, k: float = 0.3) -> np.ndarray:
    """``cos(k x − ω t)`` sampled as a ``(T, n_x)`` state."""
    x = np.arange(64)
    t = np.arange(n_times) * dt
    result: np.ndarray = np.cos(k * x[None, :] - omega * t[:, None]).astype(np.float64)
    return result


def _identity_dmd(n_modes: int, dt: float) -> DMDFit:
    """A DMD whose modes are the state axes and whose deterministic part is zero.

    Makes the residual ``η_k(t)`` equal the state's k-th column exactly, so a
    hand-generated process drives `fit_mode_ou` directly.
    """
    return DMDFit(
        Phi=np.eye(n_modes, dtype=np.complex64),
        Lambda=np.ones(n_modes, dtype=np.complex128),
        b0=np.zeros(n_modes, dtype=np.complex128),
        lam_cont=np.zeros(n_modes, dtype=np.complex128),
        dt=dt,
        variant="exact",
    )


@pytest.mark.parametrize("variant", ["exact", "bop", "hankel"])
def test_analytic_traveling_wave(variant: Literal["exact", "bop", "hankel"]) -> None:
    """Synthetic A cos(kx - ωt) is recovered as a complex-conjugate pair with correct ω.

    (see docs/model_design.md §5.6; all three variants must agree on the
    physics — a real traveling wave's DMD spectrum contains a mode at
    ``+iω`` and its conjugate at ``-iω``.)
    """
    rng = np.random.default_rng(0)
    omega = 0.7
    state = _traveling_wave(200, 0.1, omega)

    fit = dmd_fit(state, r_dmd=10, dt=0.1, rng=rng, variant=variant, bop_num_trials=20)
    recovered_omegas = np.abs(fit.lam_cont.imag)

    best = np.argmin(np.abs(recovered_omegas - omega))
    assert np.abs(recovered_omegas[best] - omega) < 0.02

    dominant = fit.Lambda[best]
    assert np.min(np.abs(fit.Lambda - np.conj(dominant))) < 1e-6


def test_bop_enforces_conjugate_pairs() -> None:
    """BOP-DMD's eig_constraints={"conjugate_pairs"} yields exact conjugate eigenvalue pairs.

    (see docs/model_design.md §5.6, §9 gotcha 8 — with variant="bop", pairing
    is enforced at fit time via PyDMD's `eig_constraints`.)
    """
    rng = np.random.default_rng(11)
    state = _traveling_wave(150, 0.1, 1.1, k=0.2)[:, :32]

    fit = dmd_fit(state, r_dmd=6, dt=0.1, rng=rng, variant="bop", bop_num_trials=20)

    oscillatory = fit.Lambda[np.abs(fit.Lambda.imag) > 1e-6]
    assert oscillatory.size > 0
    for lam in oscillatory:
        assert np.any(np.abs(fit.Lambda - np.conj(lam)) < 1e-6)


def test_pydmd_snapshot_orientation() -> None:
    """A (T, n_G) state feeds PyDMD (n_G, T) snapshots, not the reverse.

    (see docs/model_design.md §9 gotcha 22.) Uses an asymmetric shape
    (T != n_G) and r_dmd == n_G: a silent transpose swap would produce a
    differently shaped (and wrong) `Phi`, and would not reproduce the
    initial condition exactly.
    """
    rng = np.random.default_rng(7)
    n_times, n_state = 50, 10
    state = rng.normal(size=(n_times, n_state))

    fit = dmd_fit(state, r_dmd=n_state, dt=1.0, rng=rng, variant="exact")

    assert fit.Phi.shape == (n_state, n_state)
    assert_allclose((fit.Phi @ fit.b0).real, state[0], atol=1e-5)


def test_dmd_fit_flattens_a_three_dimensional_core() -> None:
    """One tensor's (T, r_lat, r_lon) core fits the same as its flattened state (§5.6)."""
    rng = np.random.default_rng(3)
    core = rng.normal(size=(40, 3, 4))

    from_core = dmd_fit(core, r_dmd=6, dt=1.0, rng=rng)
    from_state = dmd_fit(core.reshape(40, 12), r_dmd=6, dt=1.0, rng=rng)

    assert_allclose(from_core.Lambda, from_state.Lambda)
    assert_allclose(from_core.Phi, from_state.Phi)


def test_split_group_state_inverts_group_state() -> None:
    """Members concatenated by `group_state` come back unchanged from `split_group_state`."""
    rng = np.random.default_rng(4)
    cores = {"logE_wsh": rng.normal(size=(9, 2, 3)), "u_wind": rng.normal(size=(9, 4, 1))}
    members = ["u_wind", "logE_wsh"]

    state = group_state(cores, members)
    split = split_group_state(state, members, {m: cores[m].shape[1:] for m in members})

    assert state.shape == (9, 10)
    for member in members:
        assert_allclose(split[member], cores[member])


def test_conjugate_pairs_partitions_real_modes_pairs_and_orphans() -> None:
    """Exact conjugates pair up; real eigenvalues and an unpaired oscillation are real (§5.6.1)."""
    Lambda = np.array([0.9, 0.8 + 0.1j, 0.5, 0.8 - 0.1j, 0.3 + 0.2j], dtype=np.complex128)

    real_idx, rep_idx, partner_idx = conjugate_pairs(Lambda)

    assert list(real_idx) == [0, 2, 4]
    assert list(rep_idx) == [1]
    assert list(partner_idx) == [3]


def test_fit_mode_ou_recovers_correlated_real_channels() -> None:
    """Two real AR(1) channels with correlated innovations: φ and Q are recovered (§5.6.1).

    The off-diagonal of Q is the within-group correlated noise that
    independent per-mode draws would discard.
    """
    rng = np.random.default_rng(0)
    phi_true = np.array([0.9, 0.5])
    q_true = np.array([[1.0, 0.6], [0.6, 1.0]])
    factor = np.linalg.cholesky(q_true)
    n_steps = 50_000
    trace = np.zeros((n_steps, 2), dtype=np.float64)
    noise = rng.standard_normal((n_steps, 2)) @ factor.T
    for step in range(1, n_steps):
        trace[step] = phi_true * trace[step - 1] + noise[step]

    fit = fit_mode_ou(trace, _identity_dmd(2, dt=1.0))

    assert list(fit.partner) == [-1, -1]
    order = np.argsort(fit.mode)
    assert_allclose(fit.phi[order], phi_true, atol=0.01)
    assert_allclose(fit.innovation_cov[np.ix_(order, order)], q_true, atol=0.03)


def test_mode_ou_theta_and_sigma_follow_the_continuous_time_formulas() -> None:
    """`theta` and `sigma` are −log(φ)/dt and sqrt(Q_cc 2θ / (1 − φ²)) (§5.6.1)."""
    fit = ModeOUFit(
        mode=np.array([0], dtype=np.int64),
        partner=np.array([-1], dtype=np.int64),
        phi=np.array([np.exp(-0.1 * 3.0)]),
        innovation_cov=np.array([[0.5]]),
        log_magnitude_mean=np.zeros(1),
        phase_offset=np.zeros(1),
        dt=3.0,
    )

    assert_allclose(fit.theta, [0.1])
    assert_allclose(fit.sigma, np.sqrt(0.5 * 2 * 0.1 / (1 - np.exp(-0.6))))


def test_fit_mode_ou_sends_an_alternating_channel_to_the_fast_floor() -> None:
    """A negative AR(1) coefficient is clipped to 1e-4, not folded to |φ| (§5.6.1)."""
    rng = np.random.default_rng(1)
    trace = np.zeros((20_000, 1), dtype=np.float64)
    for step in range(1, trace.shape[0]):
        trace[step] = -0.8 * trace[step - 1] + rng.standard_normal()

    fit = fit_mode_ou(trace, _identity_dmd(1, dt=1.0))

    assert_allclose(fit.phi, [1e-4])


def test_mode_amplitudes_of_conjugate_pairs_reconstruct_a_real_field() -> None:
    """Pair channels give exactly conjugate amplitudes, so Φ a is real (§5.6.1, gotcha 8).

    Independent per-mode draws made task 07's reconstruction about as
    imaginary as real; the pair construction must leave only rounding.
    Integer steps only: a real mode with a negative eigenvalue has a complex
    fractional power, a separate effect `mode_amplitudes` documents.
    """
    rng = np.random.default_rng(2)
    state = _traveling_wave(300, 0.1, 0.7) + 0.05 * rng.standard_normal((300, 64))
    dmd = dmd_fit(state, r_dmd=6, dt=0.1, rng=rng)
    ou = fit_mode_ou(state, dmd)
    channels = rng.standard_normal((ou.phi.size, 5))

    field = dmd.Phi.astype(np.complex128) @ mode_amplitudes(
        dmd, ou, channels, np.array([0.0, 1.0, 10.0, 299.0, 400.0])
    )

    assert np.any(ou.partner >= 0)
    assert np.linalg.norm(field.imag) / np.linalg.norm(field.real) < 1e-10


def test_ou_transition_leaves_the_stationary_covariance_invariant() -> None:
    """φ^n P φ^n + C(n) = P at fractional and integer n; C(1) = Q, C(0) = 0 (§5.6.1)."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(3, 3))
    ou = ModeOUFit(
        mode=np.arange(3, dtype=np.int64),
        partner=np.full(3, -1, dtype=np.int64),
        phi=np.array([0.95, 0.6, 0.2]),
        innovation_cov=a @ a.T,
        log_magnitude_mean=np.zeros(3),
        phase_offset=np.zeros(3),
        dt=1.0,
    )
    stationary = ou_stationary_cov(ou)
    n_steps = np.array([0.0, 0.5, 1.0, 3.7])

    decay, noise_cov = ou_transition(ou, n_steps)

    for i in range(n_steps.size):
        propagated = decay[i][:, None] * stationary * decay[i][None, :] + noise_cov[i]
        assert_allclose(propagated, stationary, rtol=1e-10)
    assert_allclose(noise_cov[0], 0.0, atol=1e-15)
    assert_allclose(noise_cov[2], ou.innovation_cov, rtol=1e-12)
