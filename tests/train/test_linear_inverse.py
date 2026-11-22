"""Tests for train.linear_inverse (see docs/correlated_variables.md §4.4)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm, solve_continuous_lyapunov

from train.linear_inverse import (
    GroupDiagnostics,
    GroupPCs,
    diagnose_group,
    diagnostics_summary,
    fit_lim,
    group_coupling_report,
    group_pcs,
    simulate_lim,
    tau_test,
)

DT = 3 * 3600.0
DAY = 86400.0
N_STEPS = 50_000


def _simulate_ou(L: np.ndarray, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact discretization of ``dz = L z dt + dW``, ``<dW dW^T> = Q dt``, sampled every DT."""
    n = L.shape[0]
    G = expm(L * DT)
    C0 = solve_continuous_lyapunov(L, -Q)
    innovation_factor = np.linalg.cholesky(C0 - G @ C0 @ G.T)
    z = np.empty((N_STEPS, n), dtype=np.float64)
    z[0] = np.linalg.cholesky(C0) @ rng.standard_normal(n)
    for t in range(1, N_STEPS):
        z[t] = G @ z[t - 1] + innovation_factor @ rng.standard_normal(n)
    return z


def _exact_innovation_cov(L: np.ndarray, Q: np.ndarray) -> np.ndarray:
    G = expm(L * DT)
    C0 = solve_continuous_lyapunov(L, -Q)
    result: np.ndarray = C0 - G @ C0 @ G.T
    return result


# Member "a" drives member "b" (L[1, 0] != 0, L[0, 1] == 0); noise correlated at 0.5.
COUPLED_L = np.array([[-1.0, 0.0], [2.0, -0.8]], dtype=np.float64) / DAY
COUPLED_Q = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64) / DAY
UNCOUPLED_L = np.array([[-1.0, 0.0], [0.0, -0.8]], dtype=np.float64) / DAY
UNCOUPLED_Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64) / DAY


@pytest.fixture(scope="module")
def coupled_ou() -> np.ndarray:
    return _simulate_ou(COUPLED_L, COUPLED_Q, np.random.default_rng(0))


@pytest.fixture(scope="module")
def uncoupled_ou() -> np.ndarray:
    return _simulate_ou(UNCOUPLED_L, UNCOUPLED_Q, np.random.default_rng(0))


def _two_member_group(z: np.ndarray) -> GroupPCs:
    return GroupPCs(
        pcs=z,
        blocks={"a": slice(0, 1), "b": slice(1, 2)},
        explained_variance={"a": 1.0, "b": 1.0},
    )


def test_fit_lim_recovers_operator_and_noise_covariance(coupled_ou: np.ndarray) -> None:
    """L and Q of a known multivariate OU process are recovered from its samples."""
    fit = fit_lim(coupled_ou, lag=1, dt=DT)
    assert_allclose(fit.L * DAY, COUPLED_L * DAY, atol=0.1)
    assert_allclose(fit.Q * DAY, COUPLED_Q * DAY, atol=0.1)
    assert fit.q_min_eigenvalue > 0.0
    assert fit.logm_imag_fraction < 1e-8
    assert fit.lag == 1
    assert fit.dt == DT


def test_fit_lim_innovation_cov_matches_exact_discretization(coupled_ou: np.ndarray) -> None:
    """The residual covariance estimates the exact one-step innovation covariance."""
    fit = fit_lim(coupled_ou, lag=1, dt=DT)
    exact = _exact_innovation_cov(COUPLED_L, COUPLED_Q)
    assert_allclose(fit.innovation_cov, exact, rtol=0.05, atol=0.002)


def test_fit_lim_rejects_nonpositive_lag(coupled_ou: np.ndarray) -> None:
    with pytest.raises(ValueError, match="lag must be >= 1"):
        fit_lim(coupled_ou, lag=0, dt=DT)


def test_tau_test_linear_markov_process_passes(coupled_ou: np.ndarray) -> None:
    """For a true OU process, L is lag-invariant and forecast errors match the model's."""
    result = tau_test(coupled_ou, training_lag=1, test_lags=[1, 2, 4, 8], dt=DT)
    assert result.training_lag == 1
    assert result.lags.tolist() == [1, 2, 4, 8]
    assert result.operator_drift[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(result.operator_drift < 0.1)
    assert np.all(result.propagator_error < 0.05)
    assert_allclose(result.error_variance_ratio, 1.0, atol=0.05)


def test_tau_test_non_markov_process_drifts() -> None:
    """An AR(2) series observed alone is not first-order Markov: L drifts with lag.

    Roots 0.7 and 0.5 give rho(1) = 0.889 and rho(2) = 0.717, so the lag-2
    operator differs from the lag-1 operator by ~40%, and the lag-1 model
    under-predicts the lag-2 forecast error by ~30%.
    """
    rng = np.random.default_rng(0)
    x = np.zeros(N_STEPS, dtype=np.float64)
    noise = rng.standard_normal(N_STEPS)
    for t in range(2, N_STEPS):
        x[t] = 1.2 * x[t - 1] - 0.35 * x[t - 2] + noise[t]
    result = tau_test(x[:, None], training_lag=1, test_lags=[2, 4], dt=DT)
    assert result.operator_drift[0] > 0.25
    assert result.error_variance_ratio[0] > 1.15


def test_group_pcs_blocks_and_explained_variance() -> None:
    """Exact-rank member keeps all its variance; truncated member keeps a fraction."""
    rng = np.random.default_rng(0)
    n_times = 200
    low_rank = np.einsum(
        "tk,kij->tij", rng.standard_normal((n_times, 2)), rng.standard_normal((2, 3, 4))
    )
    full_rank = rng.standard_normal((n_times, 2, 5))

    group = group_pcs({"a": low_rank, "b": full_rank}, ranks={"a": 2, "b": 3})

    assert group.pcs.shape == (n_times, 5)
    assert group.blocks == {"a": slice(0, 2), "b": slice(2, 5)}
    assert group.explained_variance["a"] == pytest.approx(1.0, abs=1e-10)
    assert 0.0 < group.explained_variance["b"] < 1.0
    assert_allclose(group.pcs.mean(axis=0), 0.0, atol=1e-10)

    centered_b = full_rank.reshape(n_times, -1) - full_rank.reshape(n_times, -1).mean(axis=0)
    retained = np.sum(group.pcs[:, group.blocks["b"]] ** 2) / np.sum(centered_b**2)
    assert retained == pytest.approx(group.explained_variance["b"], rel=1e-10)


def test_group_pcs_rejects_non_finite_core() -> None:
    core = np.ones((10, 2, 2), dtype=np.float64)
    core[3, 0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        group_pcs({"a": core}, ranks={"a": 1})


def test_group_pcs_rejects_rank_out_of_range() -> None:
    """Rank is capped by min(T, n_features): 4 features here, so 5 is too many."""
    core = np.random.default_rng(0).standard_normal((10, 2, 2))
    with pytest.raises(ValueError, match=r"rank for 'a' must be in \[1, 4\]"):
        group_pcs({"a": core}, ranks={"a": 5})


@pytest.mark.parametrize(
    "shape",
    [(20, 5, 10), (300, 1, 5)],
    ids=["more_features_than_snapshots", "more_snapshots_than_features"],
)
def test_group_pcs_matches_svd_on_either_route(shape: tuple[int, int, int]) -> None:
    """Both eigendecomposition routes give the SVD's principal components, up to sign."""
    core = np.random.default_rng(0).standard_normal(shape)
    rank = 3
    group = group_pcs({"a": core}, ranks={"a": rank})

    x = core.reshape(shape[0], -1)
    x = x - x.mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    expected = u[:, :rank] * s[:rank]
    assert_allclose(np.abs(group.pcs), np.abs(expected), atol=1e-10)
    assert group.explained_variance["a"] == pytest.approx(
        float(np.sum(s[:rank] ** 2) / np.sum(s**2)), rel=1e-12
    )


def test_group_coupling_report_detects_one_way_coupling(coupled_ou: np.ndarray) -> None:
    """a -> b coupling shows up as information flowing into b only; correlated noise as MI.

    Exact Granger value for b: its own-past-only regression residual is
    ``C0_bb - C1_bb**2 / C0_bb`` with ``C1 = G C0``; the joint residual is the
    exact innovation variance. a is autonomous, so its value is exactly 0.
    """
    report = group_coupling_report(_two_member_group(coupled_ou), lag=1)

    G = expm(COUPLED_L * DT)
    C0 = solve_continuous_lyapunov(COUPLED_L, -COUPLED_Q)
    exact = _exact_innovation_cov(COUPLED_L, COUPLED_Q)
    C1_bb = (G @ C0)[1, 1]
    expected_granger_b = 0.5 * np.log((C0[1, 1] - C1_bb**2 / C0[1, 1]) / exact[1, 1])

    assert expected_granger_b > 0.04
    assert report.granger_nats["b"] == pytest.approx(expected_granger_b, abs=0.005)
    assert report.granger_nats["a"] == pytest.approx(0.0, abs=0.002)
    assert report.r2_joint["b"] > report.r2_independent["b"]
    assert report.r2_joint["a"] == pytest.approx(report.r2_independent["a"], abs=0.002)

    rho = exact[0, 1] / np.sqrt(exact[0, 0] * exact[1, 1])
    expected_mi = -0.5 * np.log(1.0 - rho**2)
    assert report.noise_mutual_information_nats == pytest.approx(expected_mi, abs=0.01)
    assert report.noise_pairwise_mutual_information_nats[("a", "b")] == pytest.approx(
        report.noise_mutual_information_nats, abs=1e-12
    )


def test_group_coupling_report_uncoupled_group_loses_nothing(uncoupled_ou: np.ndarray) -> None:
    """Independent members: no Granger information, no noise mutual information."""
    report = group_coupling_report(_two_member_group(uncoupled_ou), lag=1)
    assert report.granger_nats["a"] == pytest.approx(0.0, abs=0.002)
    assert report.granger_nats["b"] == pytest.approx(0.0, abs=0.002)
    assert report.noise_mutual_information_nats == pytest.approx(0.0, abs=0.002)


def test_simulate_lim_draws_correlated_noise(coupled_ou: np.ndarray) -> None:
    """Refitting on simulated output recovers the fit's propagator and correlated innovations."""
    fit = fit_lim(coupled_ou, lag=1, dt=DT)
    simulated = simulate_lim(fit, coupled_ou[0], N_STEPS, np.random.default_rng(1))
    refit = fit_lim(simulated, lag=1, dt=DT)
    assert simulated.shape == coupled_ou.shape
    assert_allclose(simulated[0], coupled_ou[0])
    assert_allclose(refit.G, fit.G, atol=0.01)
    assert_allclose(refit.innovation_cov, fit.innovation_cov, rtol=0.05, atol=0.002)


def test_simulate_lim_seed_reproducibility(coupled_ou: np.ndarray) -> None:
    """Same seed -> byte-identical trajectory."""
    fit = fit_lim(coupled_ou, lag=1, dt=DT)
    first = simulate_lim(fit, coupled_ou[0], 100, np.random.default_rng(7))
    second = simulate_lim(fit, coupled_ou[0], 100, np.random.default_rng(7))
    assert np.array_equal(first, second)


def _ar2_series(rng: np.random.Generator) -> np.ndarray:
    """AR(2) with roots 0.7 and 0.5: not first-order Markov when observed alone."""
    x = np.zeros(N_STEPS, dtype=np.float64)
    noise = rng.standard_normal(N_STEPS)
    for t in range(2, N_STEPS):
        x[t] = 1.2 * x[t - 1] - 0.35 * x[t - 2] + noise[t]
    return x


# The stage's specified bootstrap size (model_design.md §5.6.2). Smaller counts
# make the envelope too narrow for "a correct model passes" to hold.
DIAGNOSE_BOOTSTRAP = 100
DIAGNOSE_STEPS = 10_000


def _diagnose_pair(z: np.ndarray, seed: int = 0) -> GroupDiagnostics:
    return diagnose_group(
        {"a": z[:DIAGNOSE_STEPS, :1], "b": z[:DIAGNOSE_STEPS, 1:2]},
        ranks={"a": 1, "b": 1},
        dt=DT,
        test_lags=[1, 2, 4],
        n_bootstrap=DIAGNOSE_BOOTSTRAP,
        rng=np.random.default_rng(seed),
    )


@pytest.fixture(scope="module")
def coupled_diagnostics(coupled_ou: np.ndarray) -> GroupDiagnostics:
    return _diagnose_pair(coupled_ou)


def test_diagnose_group_coupled_members(coupled_diagnostics: GroupDiagnostics) -> None:
    """OU pair: passes the tau test; a -> b coupling and correlated noise are significant."""
    d = coupled_diagnostics
    assert d.members == ("a", "b")
    assert d.drift_upper.shape == (3,)
    assert d.ratio_band.shape == (2, 3)
    assert d.tau_consistent.tolist() == [True, True, True]
    assert d.coupling_significant == {"a": False, "b": True}
    assert d.noise_coupling_significant is True
    assert d.coupling.granger_nats["b"] > d.granger_null_upper["b"]


def test_diagnose_group_uncoupled_members_not_significant(uncoupled_ou: np.ndarray) -> None:
    """Independent pair: neither coupling metric clears its independence null band."""
    d = _diagnose_pair(uncoupled_ou)
    assert d.coupling_significant == {"a": False, "b": False}
    assert d.noise_coupling_significant is False


def test_diagnose_group_single_member_non_markov_fails_tau_test() -> None:
    """A lone AR(2) member fails the tau test off the fit lag; no coupling null is run."""
    x = _ar2_series(np.random.default_rng(0))[:DIAGNOSE_STEPS]
    d = diagnose_group(
        {"x": x[:, None]},
        ranks={"x": 1},
        dt=DT,
        test_lags=[1, 2, 4],
        n_bootstrap=DIAGNOSE_BOOTSTRAP,
        rng=np.random.default_rng(0),
    )
    assert d.tau_consistent.tolist() == [True, False, False]
    assert np.isnan(d.granger_null_upper["x"])
    assert np.isnan(d.noise_mi_null_upper)
    assert d.coupling_significant == {"x": False}
    assert d.noise_coupling_significant is False


def test_diagnose_group_warns_when_state_too_large_for_window() -> None:
    core = np.random.default_rng(0).standard_normal((50, 3, 3))
    with pytest.warns(UserWarning, match="T >= 10 n"):
        diagnose_group(
            {"a": core},
            ranks={"a": 9},
            dt=DT,
            test_lags=[2],
            n_bootstrap=2,
            rng=np.random.default_rng(0),
        )


def test_diagnose_group_rejects_zero_bootstrap(coupled_ou: np.ndarray) -> None:
    with pytest.raises(ValueError, match="n_bootstrap"):
        diagnose_group(
            {"a": coupled_ou[:, :1]},
            ranks={"a": 1},
            dt=DT,
            test_lags=[2],
            n_bootstrap=0,
            rng=np.random.default_rng(0),
        )


def test_diagnose_group_seed_reproducibility(coupled_ou: np.ndarray) -> None:
    """Same seed -> identical bootstrap bands."""
    first = _diagnose_pair(coupled_ou[:2000], seed=3)
    second = _diagnose_pair(coupled_ou[:2000], seed=3)
    assert np.array_equal(first.drift_upper, second.drift_upper)
    assert np.array_equal(first.ratio_band, second.ratio_band)
    assert first.granger_null_upper == second.granger_null_upper


def test_diagnostics_summary_reports_tau_rows_and_coupling(
    coupled_diagnostics: GroupDiagnostics,
) -> None:
    summary = diagnostics_summary(coupled_diagnostics)
    assert summary.startswith("group a, b (n = 2 PCs)")
    assert "coupling into b" in summary
    assert "noise coupling" in summary
    tau_rows = [line for line in summary.splitlines() if line.rstrip().endswith(("fit", "yes"))]
    assert len(tau_rows) == 3
