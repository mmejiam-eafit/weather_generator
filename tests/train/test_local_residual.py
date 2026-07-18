"""Tests for train.local_residual (see docs/model_design.md §3.10, §5.7)."""

import numpy as np
from numpy.testing import assert_allclose

from train.local_residual import LocalResidualFit, coherent_reconstruction, fit_local_residual
from train.tucker import tucker_fit, tucker_inverse, tucker_transform


def test_variance_ratio_diagnostic() -> None:
    """Synthetic Z with known coherent + local variances; recovered variance_ratio matches target.

    Builds a deterministic coherent signal with known sample variance and
    superimposes i.i.d. local noise sized so that
    ``Var(local) / (Var(local) + Var(coherent))`` hits a chosen target ratio
    per variable. `fit_local_residual` should recover that target (within
    Monte Carlo tolerance) as `variance_ratio`.
    """
    rng = np.random.default_rng(0)
    n_times, n_ocean, n_var = 20_000, 1, 3
    target_ratios = np.array([0.1, 0.4, 0.8], dtype=np.float64)

    t = np.arange(n_times, dtype=np.float64)
    # Amplitude sqrt(2) over many (non-resonant) periods gives Var ≈ 1.0.
    coherent_1d = np.sqrt(2.0) * np.sin(2 * np.pi * t / 137.0)
    var_coherent = coherent_1d.var()
    Z_coherent = np.tile(coherent_1d[:, None, None], (1, n_ocean, n_var))

    sigma_local = np.sqrt(target_ratios / (1.0 - target_ratios) * var_coherent)
    local = rng.normal(0.0, sigma_local[None, None, :], size=(n_times, n_ocean, n_var))
    Z = Z_coherent + local

    fit = fit_local_residual(Z, Z_coherent, dt=1.0)

    assert_allclose(fit.variance_ratio[0], target_ratios, atol=0.03)


def test_active_mask_below_floor_zeroed() -> None:
    """Cells with variance ratio below the floor get theta == sigma == phi == 0.0 exactly."""
    n_times = 5_000
    dt = 1.0
    variance_floor = 0.02

    t = np.arange(n_times, dtype=np.float64)
    coherent_1d = np.sqrt(2.0) * np.sin(2 * np.pi * t / 97.0)
    Z_coherent = np.tile(coherent_1d[:, None, None], (1, 1, 2))

    rng = np.random.default_rng(1)
    # var index 0: no residual at all -> variance_ratio exactly 0, below floor.
    # var index 1: strong local noise -> variance_ratio well above floor.
    local = np.zeros((n_times, 1, 2), dtype=np.float64)
    local[:, 0, 1] = rng.normal(0.0, 2.0, size=n_times)
    Z = Z_coherent + local

    fit = fit_local_residual(Z, Z_coherent, dt=dt, variance_floor=variance_floor)

    assert not fit.active_mask[0, 0]
    assert fit.theta[0, 0] == 0.0
    assert fit.sigma[0, 0] == 0.0
    assert fit.phi[0, 0] == 0.0

    assert fit.active_mask[0, 1]
    assert fit.theta[0, 1] != 0.0
    assert fit.sigma[0, 1] != 0.0


def test_vectorized_ar1_matches_per_cell_loop() -> None:
    """The vectorized fit matches a naive per-cell scalar AR(1)/OU fit (ground-truth oracle).

    The oracle below is written only inside this test, deliberately
    duplicating §5.7's formula but applied one cell/variable at a time with
    an explicit Python loop, to confirm the vectorized broadcasting over
    `(N_ocean, V)` is wired correctly (no axis mixups, no cross-cell
    contamination).
    """
    rng = np.random.default_rng(2)
    n_times, n_ocean, n_var = 200, 6, 3
    dt = 1.0
    variance_floor = 0.02

    Z_coherent = rng.normal(size=(n_times, n_ocean, n_var))
    Z = Z_coherent + rng.normal(scale=0.5, size=(n_times, n_ocean, n_var))

    fit = fit_local_residual(Z, Z_coherent, dt=dt, variance_floor=variance_floor)

    R = Z - Z_coherent
    theta_loop = np.empty((n_ocean, n_var), dtype=np.float32)
    sigma_loop = np.empty((n_ocean, n_var), dtype=np.float32)
    phi_loop = np.empty((n_ocean, n_var), dtype=np.float32)
    ratio_loop = np.empty((n_ocean, n_var), dtype=np.float32)
    active_loop = np.empty((n_ocean, n_var), dtype=np.bool_)

    for i in range(n_ocean):
        for j in range(n_var):
            r = R[:, i, j]
            r_prev, r_next = r[:-1], r[1:]
            num = np.sum(r_prev * r_next)
            den = np.sum(r_prev * r_prev) + 1e-30
            phi_ij = np.float32(np.clip(num / den, -0.9999, 0.9999))
            theta_ij = -np.log(np.clip(phi_ij, 1e-4, 0.9999)) / dt

            # Uncentered second moment, matching train.dmd.fit_mode_ou's
            # np.mean(np.abs(innovation)**2, ...) pattern -- NOT `.var()`,
            # which would incorrectly re-center around the innovation's
            # sample mean instead of treating it as zero-mean by assumption.
            innovation_ij = r_next - phi_ij * r_prev
            innovation_var_ij = np.mean(innovation_ij**2)
            var_r = r.var()
            var_z = Z[:, i, j].var() + 1e-30
            sigma_ij = np.float32(
                np.sqrt(innovation_var_ij * 2.0 * theta_ij / (1.0 - phi_ij**2 + 1e-30))
            )
            ratio_ij = np.float32(var_r / var_z)
            active_ij = bool(ratio_ij > variance_floor)

            theta_loop[i, j] = theta_ij if active_ij else 0.0
            sigma_loop[i, j] = sigma_ij if active_ij else 0.0
            phi_loop[i, j] = phi_ij if active_ij else 0.0
            ratio_loop[i, j] = ratio_ij
            active_loop[i, j] = active_ij

    assert_allclose(fit.theta, theta_loop, rtol=1e-4, atol=1e-6)
    assert_allclose(fit.sigma, sigma_loop, rtol=1e-4, atol=1e-6)
    assert_allclose(fit.phi, phi_loop, rtol=1e-4, atol=1e-6)
    assert_allclose(fit.variance_ratio, ratio_loop, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(fit.active_mask, active_loop)


def test_equilibrium_variance_matches() -> None:
    """Equilibrium draws N(0, sigma^2 / (2*theta)) reproduce the residual's per-cell variance.

    Uses i.i.d. (near-memoryless, phi ≈ 0) local noise so the discrete-time
    OU's equilibrium variance formula cleanly matches the residual's raw
    sample variance.
    """
    rng = np.random.default_rng(3)
    n_times, n_ocean, n_var = 8_000, 4, 2
    dt = 1.0

    Z_coherent = rng.normal(size=(n_times, n_ocean, n_var))
    sigma_true = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64).reshape(n_ocean, 1)
    local = rng.normal(
        0.0, np.broadcast_to(sigma_true, (n_ocean, n_var)), size=(n_times, n_ocean, n_var)
    )
    Z = Z_coherent + local

    fit = fit_local_residual(Z, Z_coherent, dt=dt)
    assert np.all(fit.active_mask)

    R = Z - Z_coherent
    empirical_var = R.var(axis=0)
    equilibrium_var = fit.sigma.astype(np.float64) ** 2 / (2.0 * fit.theta.astype(np.float64))

    assert_allclose(equilibrium_var, empirical_var, rtol=0.1)

    n_draws = 200_000
    draws = rng.normal(0.0, np.sqrt(equilibrium_var), size=(n_draws, n_ocean, n_var))
    sampled_var = draws.var(axis=0)

    assert_allclose(sampled_var, empirical_var, rtol=0.1)


def test_sigma_recovers_truth_under_realistic_autocorrelation() -> None:
    """`sigma` is recovered correctly when phi is meaningfully nonzero, not just near 0.

    Guards the 2026-07-18 fix: `sigma` must be computed from the AR(1)
    *innovation* variance, not R's raw marginal (stationary) variance. These
    coincide only when `phi ≈ 0`, which is exactly the regime
    `test_equilibrium_variance_matches` (above) uses — so that test alone
    would not have caught a regression back to the marginal-variance formula.
    This test injects a genuine discrete OU trace with known `true_sigma` at
    `phi ≈ 0.95` (a realistic, meaningfully-autocorrelated cell) and checks
    the recovered `sigma` against the known truth directly. The old
    marginal-variance formula would recover `sigma` inflated by
    `1 / sqrt(1 - phi**2) ≈ 3.2x` here — numerically confirmed before this
    fix — which `rtol=0.1` below would not tolerate.
    """
    rng = np.random.default_rng(42)
    true_theta = 0.5
    true_sigma = 2.0
    dt = 0.1
    true_phi = np.exp(-true_theta * dt)
    innovation_std = true_sigma * np.sqrt((1.0 - true_phi**2) / (2.0 * true_theta))

    n_times = 200_000
    r = np.empty(n_times, dtype=np.float64)
    r[0] = rng.normal(0.0, true_sigma / np.sqrt(2.0 * true_theta))
    innovations = rng.normal(0.0, innovation_std, size=n_times - 1)
    for t in range(1, n_times):
        r[t] = true_phi * r[t - 1] + innovations[t - 1]

    Z_coherent = np.zeros((n_times, 1, 1), dtype=np.float64)
    Z = r.reshape(n_times, 1, 1) + Z_coherent

    fit = fit_local_residual(Z, Z_coherent, dt=dt)

    assert_allclose(fit.phi[0, 0], true_phi, rtol=0.02)
    assert_allclose(fit.theta[0, 0], true_theta, rtol=0.1)
    assert_allclose(fit.sigma[0, 0], true_sigma, rtol=0.1)


def test_negative_phi_alternation_gives_fast_decorrelation() -> None:
    """A residual that alternates sign every step yields phi ≈ -1 and a *large* theta.

    Guards the 2026-07-17 fix to docs/model_design.md §5.7: `theta` must
    clip `phi` itself (not `|phi|`) before the log. A perfectly alternating
    residual has phi = -1 exactly (before the [-0.9999, 0.9999] clip) —
    negative phi means alternation/anti-correlation, i.e. the residual
    decorrelates in a single step and should get a *large* theta (clipped up
    to the fast-decorrelation floor of 1e-4). The old buggy `|phi|` formula
    would have instead read `|-0.9999| == 0.9999` as "near +1" (slow
    reversion) and produced a *tiny* theta — the opposite conclusion.
    """
    n_times = 200
    dt = 1.0
    amplitude = 2.0
    sign_pattern = np.array([(-1.0) ** step for step in range(n_times)], dtype=np.float64)
    R_1d = amplitude * sign_pattern
    Z_coherent = np.zeros((n_times, 1, 1), dtype=np.float64)
    Z = R_1d.reshape(n_times, 1, 1) + Z_coherent

    fit = fit_local_residual(Z, Z_coherent, dt=dt)

    assert fit.phi[0, 0] < -0.99

    expected_theta = -np.log(1e-4) / dt  # phi clipped to the 1e-4 floor before the log
    assert_allclose(fit.theta[0, 0], expected_theta, rtol=1e-5)

    # The old |phi| bug would have produced theta = -log(0.9999) / dt ≈ 1e-4 (tiny).
    buggy_theta = -np.log(np.clip(np.abs(fit.phi[0, 0]), 1e-4, 0.9999)) / dt
    assert fit.theta[0, 0] > 100.0 * buggy_theta


def test_coherent_reconstruction_matches_tucker_inverse() -> None:
    """`coherent_reconstruction` is a thin wrapper reproducing `tucker_inverse` exactly."""
    rng = np.random.default_rng(4)
    n_times, n_ocean, n_var = 15, 20, 3
    Z = rng.normal(size=(n_times, n_ocean, n_var))

    tucker = tucker_fit(Z, r_ocean=6, r_var=2, rng=rng)
    core_deterministic = tucker_transform(Z, tucker)

    Z_coherent = coherent_reconstruction(core_deterministic, tucker)
    expected = tucker_inverse(core_deterministic, tucker)

    assert Z_coherent.shape == (n_times, n_ocean, n_var)
    assert_allclose(Z_coherent, expected, rtol=1e-12)


def test_local_residual_fit_returns_local_residual_fit_instance() -> None:
    """`fit_local_residual` returns a populated `LocalResidualFit` with the documented shapes."""
    rng = np.random.default_rng(5)
    n_times, n_ocean, n_var = 50, 4, 2
    dt = 10800.0  # 3 hours, this archive's snapshot cadence

    Z_coherent = rng.normal(size=(n_times, n_ocean, n_var))
    Z = Z_coherent + rng.normal(scale=0.3, size=(n_times, n_ocean, n_var))

    fit = fit_local_residual(Z, Z_coherent, dt=dt)

    assert isinstance(fit, LocalResidualFit)
    for field in (fit.theta, fit.sigma, fit.phi, fit.variance_ratio):
        assert field.shape == (n_ocean, n_var)
        assert field.dtype == np.float32
    assert fit.active_mask.shape == (n_ocean, n_var)
    assert fit.active_mask.dtype == np.bool_
    assert fit.dt == dt
