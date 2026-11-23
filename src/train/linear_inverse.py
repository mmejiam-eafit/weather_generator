"""Linear inverse model (LIM) fit, tau test, and coupling diagnostics for variable groups.

(see docs/model_design.md §5.6.2, docs/correlated_variables.md §4.4, §7)

A correlated-variable group's joint DMD regression (§4.2) is a Linear Inverse
Model (Penland & Sardeshmukh 1995) estimated in a truncated basis. This module
fits that LIM in per-member principal-component coordinates, tests its
linear-Markov assumption (the tau test), estimates the group's correlated
innovation covariance, and measures the information that independent
per-variable fits discard.
"""

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import expm, logm


@dataclass(frozen=True, slots=True)
class GroupPCs:
    """Concatenated per-member principal components of a group's cores (see §4.4)."""

    pcs: np.ndarray  # (T, n) float64, time-centred, member blocks side by side
    blocks: dict[str, slice]  # member -> its columns in `pcs`
    explained_variance: dict[str, float]  # member -> fraction of its core variance retained


@dataclass(frozen=True, slots=True)
class LIMFit:
    """Fitted linear inverse model ``dz = L z dt + dW`` (see §4.4)."""

    G: np.ndarray  # (n, n) least-squares propagator over `lag` snapshots
    L: np.ndarray  # (n, n) continuous-time operator, logm(G) / (lag * dt), per second
    Q: np.ndarray  # (n, n) noise covariance rate, -(L C0 + C0 L^T) (fluctuation-dissipation)
    C0: np.ndarray  # (n, n) contemporaneous covariance
    innovation_cov: np.ndarray  # (n, n) covariance of the residuals z[t+lag] - G z[t]
    lag: int  # training lag tau_0, in snapshots
    dt: float  # seconds between snapshots
    q_min_eigenvalue: float  # < 0 => the linear-Markov model is inconsistent at this lag
    logm_imag_fraction: float  # > 0 => G has eigenvalues on the negative real axis


@dataclass(frozen=True, slots=True)
class TauTestResult:
    """Lag-invariance diagnostics of a LIM fit (see §4.4, Penland & Sardeshmukh 1995)."""

    training_lag: int
    lags: np.ndarray  # (k,) int, test lags in snapshots
    operator_drift: np.ndarray  # (k,) ||L(tau) - L(tau_0)||_F / ||L(tau_0)||_F
    propagator_error: np.ndarray  # (k,) ||expm(L tau) - G_obs(tau)||_F / ||G_obs(tau)||_F
    error_variance_ratio: np.ndarray  # (k,) observed / LIM-predicted forecast error variance


@dataclass(frozen=True, slots=True)
class GroupCouplingReport:
    """Information an independent per-member fit loses relative to the joint fit (see §4.4)."""

    lag: int
    r2_joint: dict[str, float]  # member -> one-step R^2 using the whole group's present
    r2_independent: dict[str, float]  # member -> one-step R^2 using only its own present
    granger_nats: dict[str, float]  # member -> transfer entropy into it from the other members
    noise_mutual_information_nats: float  # multi-information of the joint innovations
    noise_pairwise_mutual_information_nats: dict[tuple[str, str], float]


def _center(z: np.ndarray) -> np.ndarray:
    """Remove the time mean of each column."""
    centered: np.ndarray = z - z.mean(axis=0)
    return centered


def _regress(z: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares propagator ``z[t+lag] ≈ G z[t]`` and its residuals.

    Parameters
    ----------
    z : (T, n) float64
    lag : int

    Returns
    -------
    G : (n, n)
    residuals : (T - lag, n)
    """
    past, future = z[:-lag], z[lag:]
    coef = np.linalg.lstsq(past, future, rcond=None)[0]  # future ≈ past @ coef
    residuals: np.ndarray = future - past @ coef
    return coef.T, residuals


def _second_moment(x: np.ndarray) -> np.ndarray:
    """``x.T @ x / len(x)``: the covariance of rows already centred (or residuals)."""
    result: np.ndarray = x.T @ x / x.shape[0]
    return result


def _logdet(m: np.ndarray) -> float:
    """Log-determinant of a symmetric matrix; NaN unless positive definite."""
    sign, value = np.linalg.slogdet(m)
    return float(value) if sign > 0 else float("nan")


def group_pcs(cores: dict[str, np.ndarray], ranks: dict[str, int]) -> GroupPCs:
    """Reduce each group member's core to its leading principal components and concatenate.

    (see docs/correlated_variables.md §4.4)

    Each member keeps its own block of columns, so downstream metrics stay
    attributable to a variable. See `_leading_pcs` for the decomposition.

    Parameters
    ----------
    cores : dict[str, np.ndarray]
        Member name -> its Gaussianized Tucker core, canonical time-first
        ``(T, r_lat, r_lon)`` (any trailing shape is vectorized). All members
        must share `T`.
    ranks : dict[str, int]
        Member name -> number of principal components to keep,
        ``1 <= r <= min(T, n_features)``.

    Returns
    -------
    GroupPCs

    Raises
    ------
    ValueError
        If a core contains non-finite values or a rank is out of range.
    """
    columns: list[np.ndarray] = []
    blocks: dict[str, slice] = {}
    explained: dict[str, float] = {}
    start = 0
    for name, core in cores.items():
        rank = ranks[name]
        x = core.reshape(core.shape[0], -1).astype(np.float64)
        max_rank = min(x.shape)
        if not 1 <= rank <= max_rank:
            raise ValueError(f"rank for {name!r} must be in [1, {max_rank}], got {rank}")
        if not np.isfinite(x).all():
            raise ValueError(f"core {name!r} contains non-finite values")
        x -= x.mean(axis=0)
        pcs, explained[name] = _leading_pcs(x, rank)
        columns.append(pcs)
        blocks[name] = slice(start, start + rank)
        start += rank
    return GroupPCs(
        pcs=np.concatenate(columns, axis=1), blocks=blocks, explained_variance=explained
    )


def _leading_pcs(x: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    """Leading principal-component time series of a time-centred ``(T, n_features)`` matrix.

    Eigendecomposes whichever second-moment matrix is smaller: the
    ``(T, T)`` Gram matrix (method of snapshots) when features outnumber
    snapshots, as for a production Tucker core (2192 x 142,848), otherwise
    the ``(n_features, n_features)`` covariance. Both give the same PCs up
    to sign; picking the wrong one is the difference between ~40 MB and
    ~20 GB for a long, low-dimensional series.

    Returns
    -------
    pcs : (T, rank) float64
    explained : float
        Fraction of ``sum(x**2)`` the kept components retain.
    """
    n_times, n_features = x.shape
    if n_features < n_times:
        eigvals, eofs = np.linalg.eigh(x.T @ x)  # ascending
        top = np.argsort(eigvals)[::-1][:rank]
        pcs: np.ndarray = x @ eofs[:, top]
    else:
        eigvals, eigvecs = np.linalg.eigh(x @ x.T)  # ascending
        top = np.argsort(eigvals)[::-1][:rank]
        pcs = eigvecs[:, top] * np.sqrt(np.clip(eigvals[top], 0.0, None))
    # vdot, not sum(x**2): no second (T, n_features) temporary (3.9 GB per production core).
    explained = float(np.clip(eigvals[top], 0.0, None).sum() / np.vdot(x, x))
    return pcs, explained


def fit_lim(z: np.ndarray, lag: int, dt: float) -> LIMFit:
    """Fit a linear inverse model to a multivariate time series.

    (see docs/correlated_variables.md §4.4)

    ``G = C(lag) C(0)^-1`` by least squares, ``L = logm(G) / (lag * dt)``,
    and ``Q = -(L C0 + C0 L^T)`` from fluctuation-dissipation. Columns are
    time-centred first.

    Parameters
    ----------
    z : (T, n) float64
        E.g. `GroupPCs.pcs`.
    lag : int
        Training lag tau_0, in snapshots (>= 1).
    dt : float
        Seconds between snapshots.

    Returns
    -------
    LIMFit
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    z = _center(z)
    G, residuals = _regress(z, lag)
    C0 = _second_moment(z)
    L_complex = np.asarray(logm(G)) / (lag * dt)
    logm_imag_fraction = float(np.linalg.norm(L_complex.imag) / np.linalg.norm(L_complex))
    L = np.ascontiguousarray(L_complex.real, dtype=np.float64)
    Q = -(L @ C0 + C0 @ L.T)
    Q = 0.5 * (Q + Q.T)
    return LIMFit(
        G=G,
        L=L,
        Q=Q,
        C0=C0,
        innovation_cov=_second_moment(residuals),
        lag=lag,
        dt=dt,
        q_min_eigenvalue=float(np.linalg.eigvalsh(Q).min()),
        logm_imag_fraction=logm_imag_fraction,
    )


def tau_test(z: np.ndarray, training_lag: int, test_lags: list[int], dt: float) -> TauTestResult:
    """Check that a LIM fit at `training_lag` holds at other lags (the tau test).

    (see docs/correlated_variables.md §4.4, Penland & Sardeshmukh 1995)

    For a true multivariate OU process, ``L`` estimated at any lag is the
    same and ``G_obs(tau) = expm(L tau)``. Deviations measure how far the
    data depart from the linear-Markov assumption. Judge magnitudes against
    a parametric-bootstrap band (rerun on `simulate_lim` output of the same
    length), not an absolute threshold.

    Parameters
    ----------
    z : (T, n) float64
    training_lag : int
        Lag the reference LIM is fit at, in snapshots.
    test_lags : list[int]
        Lags to test, in snapshots.
    dt : float
        Seconds between snapshots.

    Returns
    -------
    TauTestResult
    """
    z = _center(z)
    fit = fit_lim(z, training_lag, dt)
    norm_L = np.linalg.norm(fit.L)
    drift: list[float] = []
    prop_error: list[float] = []
    variance_ratio: list[float] = []
    for lag in test_lags:
        G_obs, _ = _regress(z, lag)
        L_lag = np.asarray(logm(G_obs)).real / (lag * dt)
        drift.append(float(np.linalg.norm(L_lag - fit.L) / norm_L))

        G_model = np.asarray(expm(fit.L * (lag * dt)))
        prop_error.append(float(np.linalg.norm(G_model - G_obs) / np.linalg.norm(G_obs)))

        forecast_errors = z[lag:] - z[:-lag] @ G_model.T
        observed = float(np.mean(np.sum(forecast_errors**2, axis=1)))
        predicted = float(np.trace(fit.C0 - G_model @ fit.C0 @ G_model.T))
        variance_ratio.append(observed / predicted)

    return TauTestResult(
        training_lag=training_lag,
        lags=np.asarray(test_lags, dtype=np.int64),
        operator_drift=np.asarray(drift, dtype=np.float64),
        propagator_error=np.asarray(prop_error, dtype=np.float64),
        error_variance_ratio=np.asarray(variance_ratio, dtype=np.float64),
    )


def group_coupling_report(group: GroupPCs, lag: int = 1) -> GroupCouplingReport:
    """Measure what fitting each group member independently would lose.

    (see docs/correlated_variables.md §4.4)

    Compares the joint one-step regression (every member's future on the
    whole group's present) against independent per-member regressions (each
    member's future on its own present only):

    - deterministic coupling, per member: ``0.5 * ln(det Σ_ind / det Σ_joint)``,
      half of Geweke's (1982) Granger measure, which for Gaussian processes
      equals the transfer entropy into the member (Barnett et al. 2009).
      Non-negative, since the independent regression is nested in the joint.
    - stochastic coupling: the multi-information of the joint innovations
      across member blocks, ``0.5 * (sum_v ln det Σ_vv - ln det Σ)``, i.e.
      what independent per-member noise draws discard; plus every pairwise
      version.

    Parameters
    ----------
    group : GroupPCs
    lag : int
        Regression lag, in snapshots.

    Returns
    -------
    GroupCouplingReport
    """
    z = _center(group.pcs)
    _, joint_residuals = _regress(z, lag)
    sigma_joint = _second_moment(joint_residuals)
    future = z[lag:]

    r2_joint: dict[str, float] = {}
    r2_independent: dict[str, float] = {}
    granger: dict[str, float] = {}
    for name, block in group.blocks.items():
        _, own_residuals = _regress(z[:, block], lag)
        future_power = float(np.sum(future[:, block] ** 2))
        r2_joint[name] = 1.0 - float(np.sum(joint_residuals[:, block] ** 2)) / future_power
        r2_independent[name] = 1.0 - float(np.sum(own_residuals**2)) / future_power
        granger[name] = 0.5 * (
            _logdet(_second_moment(own_residuals)) - _logdet(sigma_joint[block, block])
        )

    block_logdets = {
        name: _logdet(sigma_joint[block, block]) for name, block in group.blocks.items()
    }
    noise_mi = 0.5 * (sum(block_logdets.values()) - _logdet(sigma_joint))
    pairwise: dict[tuple[str, str], float] = {}
    for a, b in combinations(group.blocks, 2):
        idx = np.r_[group.blocks[a], group.blocks[b]]
        pairwise[(a, b)] = 0.5 * (
            block_logdets[a] + block_logdets[b] - _logdet(sigma_joint[np.ix_(idx, idx)])
        )

    return GroupCouplingReport(
        lag=lag,
        r2_joint=r2_joint,
        r2_independent=r2_independent,
        granger_nats=granger,
        noise_mutual_information_nats=noise_mi,
        noise_pairwise_mutual_information_nats=pairwise,
    )


def simulate_lim(fit: LIMFit, z0: np.ndarray, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a fitted LIM forward with jointly correlated innovations.

    (see docs/correlated_variables.md §4.4 — reference implementation of
    within-group correlated noise, and the generator for tau-test bootstrap
    baselines)

    ``z[t+1] = G z[t] + eps[t]``, ``eps ~ N(0, innovation_cov)``, stepping at
    the fit's training-lag cadence. Noise is drawn as one joint vector per
    step, never independently per member.

    Parameters
    ----------
    fit : LIMFit
    z0 : (n,) float64
        Initial state (first row of the output).
    n_steps : int
        Number of rows returned, including `z0`.
    rng : np.random.Generator

    Returns
    -------
    (n_steps, n) float64
    """
    return _simulate_linear(fit.G, fit.innovation_cov, z0, n_steps, rng)


def covariance_factor(cov: np.ndarray) -> np.ndarray:
    """A square root ``F`` with ``F @ F.T = cov``, for drawing correlated Gaussian noise.

    (see docs/correlated_variables.md §4.4)

    Symmetric eigendecomposition rather than Cholesky: estimated innovation
    covariances are often singular or slightly indefinite from rounding,
    which Cholesky rejects. Negative eigenvalues are clipped to zero, the
    nearest positive semidefinite matrix in that eigenbasis. Accepts a stack
    ``(..., n, n)`` and factors each matrix.

    Parameters
    ----------
    cov : (..., n, n) float64, symmetric

    Returns
    -------
    (..., n, n) float64
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    factor: np.ndarray = eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))[..., None, :]
    return factor


def _simulate_linear(
    G: np.ndarray,
    innovation_cov: np.ndarray,
    z0: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``z[t+1] = G z[t] + eps[t]``, ``eps ~ N(0, innovation_cov)``; rows include `z0`."""
    noise_factor = covariance_factor(innovation_cov)
    n = z0.shape[0]
    # All innovations in one draw: the bootstrap bands run hundreds of these
    # simulations, and a per-step rng call dominated the loop's cost.
    noise = rng.standard_normal((n_steps, n)) @ noise_factor.T
    out = np.empty((n_steps, n), dtype=np.float64)
    out[0] = z0
    for i in range(1, n_steps):
        out[i] = G @ out[i - 1] + noise[i]
    return out


@dataclass(frozen=True, slots=True)
class GroupDiagnostics:
    """Training-pipeline stage output for one variable group (see model_design.md §5.6.2).

    Bands are the envelope (min / max) of ``n`` parametric-bootstrap
    replicates as long as the training window: tau-test bands from the
    fitted LIM itself (the null "the linear-Markov model is right"),
    coupling bands from a block-diagonal LIM fit member by member (the null
    "members are independent").

    Why the envelope, not interpolated percentiles: with tens of replicates
    a 5th/95th percentile is mostly interpolation noise, and its nominal
    10% false-alarm rate compounds across lags (a perfect model fails at
    least one of four lags about half the time). If the real statistic and
    the replicates are exchangeable under the null, the real one falls
    outside the two-sided envelope with probability ``2 / (n + 1)`` and
    above the one-sided maximum with ``1 / (n + 1)`` — about 3% per lag at
    ``n = 100``. Caveat: replicates come from the *fitted* model, not the
    true one, so exchangeability is approximate. One flagged lag is weak
    evidence; flags across consecutive lags, or values far outside the
    band, are the signal.
    """

    members: tuple[str, ...]
    explained_variance: dict[str, float]  # member -> core variance retained by its PCs
    lim: LIMFit
    tau: TauTestResult
    drift_upper: np.ndarray  # (k,) bootstrap max of operator drift under the fitted LIM
    ratio_band: np.ndarray  # (2, k) bootstrap min / max of the error-variance ratio
    tau_consistent: np.ndarray  # (k,) bool, drift and ratio inside bands (True at the fit lag)
    coupling: GroupCouplingReport
    granger_null_upper: dict[str, float]  # member -> bootstrap max under independence
    noise_mi_null_upper: float  # bootstrap max under independence; NaN for one member
    coupling_significant: dict[str, bool]  # member -> Granger above its null band
    noise_coupling_significant: bool  # joint-innovation MI above its null band


def diagnose_group(
    cores: dict[str, np.ndarray],
    ranks: dict[str, int],
    dt: float,
    test_lags: list[int],
    n_bootstrap: int,
    rng: np.random.Generator,
    training_lag: int = 1,
) -> GroupDiagnostics:
    """Run the full LIM diagnostic stage on one group's Gaussianized cores.

    (see docs/model_design.md §5.6.2, docs/correlated_variables.md §4.4)

    Reduces each member's core to principal components, fits a LIM at
    `training_lag`, runs the tau test at `test_lags`, measures coupling
    information, and judges each against a parametric-bootstrap band of
    `n_bootstrap` replicates. A single-member group gets a tau test only:
    its coupling information is zero by construction and no null is run.
    Records and judges; never alters or rejects a fit.

    Parameters
    ----------
    cores : dict[str, np.ndarray]
        Member name -> Gaussianized Tucker core ``(T, r_lat, r_lon)``.
    ranks : dict[str, int]
        Member name -> principal components kept (see `group_pcs`).
    dt : float
        Seconds between snapshots.
    test_lags : list[int]
        Tau-test lags, in snapshots.
    n_bootstrap : int
        Bootstrap replicates per band (>= 1).
    rng : np.random.Generator
    training_lag : int
        LIM training lag, in snapshots.

    Returns
    -------
    GroupDiagnostics
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    group = group_pcs(cores, ranks)
    z = _center(group.pcs)
    _warn_if_undersampled(*z.shape)

    lim = fit_lim(z, training_lag, dt)
    tau = tau_test(z, training_lag, test_lags, dt)
    drift_upper, ratio_band = _tau_bootstrap_bands(lim, z, test_lags, n_bootstrap, rng)
    inside_bands = (
        (tau.operator_drift <= drift_upper)
        & (tau.error_variance_ratio >= ratio_band[0])
        & (tau.error_variance_ratio <= ratio_band[1])
    )
    # At the fit lag the model reproduces its own regression by construction,
    # and its bootstrap band is so narrow that rounding alone can land outside
    # it: that row is the fit, not a test.
    tau_consistent = inside_bands | (tau.lags == training_lag)

    coupling = group_coupling_report(group, training_lag)
    members = tuple(group.blocks)
    if len(members) > 1:
        granger_null_upper, noise_mi_null_upper = _coupling_null_upper(
            group, z, training_lag, n_bootstrap, rng
        )
    else:
        granger_null_upper = {members[0]: float("nan")}
        noise_mi_null_upper = float("nan")

    return GroupDiagnostics(
        members=members,
        explained_variance=group.explained_variance,
        lim=lim,
        tau=tau,
        drift_upper=drift_upper,
        ratio_band=ratio_band,
        tau_consistent=tau_consistent,
        coupling=coupling,
        granger_null_upper=granger_null_upper,
        noise_mi_null_upper=noise_mi_null_upper,
        coupling_significant={
            m: bool(coupling.granger_nats[m] > granger_null_upper[m]) for m in members
        },
        noise_coupling_significant=bool(
            coupling.noise_mutual_information_nats > noise_mi_null_upper
        ),
    )


def _warn_if_undersampled(n_times: int, n_state: int) -> None:
    """Warn when the group state is too large for its covariances to be well estimated."""
    if n_times < 10 * n_state:
        warnings.warn(
            f"T={n_times} snapshots for a {n_state}-dimensional group state; LIM covariances "
            "need T >= 10 n to be well estimated (correlated_variables.md §4.4)",
            stacklevel=3,
        )


def _tau_bootstrap_bands(
    lim: LIMFit,
    z: np.ndarray,
    test_lags: list[int],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Tau-test bands under the null "the fitted LIM is the true process".

    Reruns `tau_test` on `n_bootstrap` series simulated from `lim`, each as
    long as `z`, so the bands include the finite-sample estimation noise of
    a window this length.

    Returns
    -------
    drift_upper : (k,) maximum operator drift over the replicates
    ratio_band : (2, k) minimum and maximum error-variance ratio (see
        `GroupDiagnostics` for why the envelope)
    """
    replicates = [
        tau_test(simulate_lim(lim, z[0], z.shape[0], rng), lim.lag, test_lags, lim.dt)
        for _ in range(n_bootstrap)
    ]
    drift = np.array([r.operator_drift for r in replicates], dtype=np.float64)
    ratio = np.array([r.error_variance_ratio for r in replicates], dtype=np.float64)
    return drift.max(axis=0), np.stack([ratio.min(axis=0), ratio.max(axis=0)])


def _coupling_null_upper(
    group: GroupPCs, z: np.ndarray, lag: int, n_bootstrap: int, rng: np.random.Generator
) -> tuple[dict[str, float], float]:
    """Bootstrap maxima of coupling information when members evolve independently.

    Fits each member's own LIM, assembles them into a block-diagonal
    propagator and innovation covariance (no cross-member dynamics, no
    cross-member noise), simulates `n_bootstrap` series as long as `z`, and
    reruns `group_coupling_report` on each. Finite-sample estimates of
    Granger and noise MI are biased upward; this band is that bias.
    """
    n_times, n_state = z.shape
    G_null = np.zeros((n_state, n_state), dtype=np.float64)
    cov_null = np.zeros((n_state, n_state), dtype=np.float64)
    for block in group.blocks.values():
        G_block, residuals = _regress(z[:, block], lag)
        G_null[block, block] = G_block
        cov_null[block, block] = _second_moment(residuals)

    granger: dict[str, list[float]] = {name: [] for name in group.blocks}
    noise_mi: list[float] = []
    for _ in range(n_bootstrap):
        simulated = _simulate_linear(G_null, cov_null, z[0], n_times, rng)
        report = group_coupling_report(
            GroupPCs(simulated, group.blocks, group.explained_variance), lag
        )
        for name in group.blocks:
            granger[name].append(report.granger_nats[name])
        noise_mi.append(report.noise_mutual_information_nats)

    return (
        {name: max(values) for name, values in granger.items()},
        max(noise_mi),
    )


def diagnostics_summary(diagnostics: GroupDiagnostics) -> str:
    """Human-readable report of one group's diagnostics, for logs and notebooks.

    (see docs/model_design.md §5.6.2)

    Parameters
    ----------
    diagnostics : GroupDiagnostics

    Returns
    -------
    str
        Multi-line summary.
    """
    d = diagnostics
    hours = d.lim.dt / 3600.0
    lines = [f"group {', '.join(d.members)} (n = {d.lim.G.shape[0]} PCs)"]
    lines += [f"  variance retained  {m}: {d.explained_variance[m]:.1%}" for m in d.members]
    lines.append(
        f"  LIM at {d.lim.lag * hours:g} h: Q min eigenvalue {d.lim.q_min_eigenvalue:.2e}, "
        f"logm imaginary fraction {d.lim.logm_imag_fraction:.1e}"
    )
    lines.append("  tau test   lag    drift (boot max)   ratio [boot min, max]    ok")
    for i, lag in enumerate(d.tau.lags):
        verdict = "fit" if lag == d.tau.training_lag else ("yes" if d.tau_consistent[i] else "NO")
        lines.append(
            f"           {lag * hours:>4g} h  {d.tau.operator_drift[i]:.3f} ({d.drift_upper[i]:.3f})"
            f"    {d.tau.error_variance_ratio[i]:.3f} "
            f"[{d.ratio_band[0, i]:.3f}, {d.ratio_band[1, i]:.3f}]"
            f"    {verdict}"
        )
    if len(d.members) > 1:
        for m in d.members:
            lines.append(
                f"  coupling into {m}: {d.coupling.granger_nats[m]:.4f} nats "
                f"(null max {d.granger_null_upper[m]:.4f}), "
                f"R2 joint {d.coupling.r2_joint[m]:.3f} vs own {d.coupling.r2_independent[m]:.3f}"
                f"{'  significant' if d.coupling_significant[m] else ''}"
            )
        lines.append(
            f"  noise coupling: {d.coupling.noise_mutual_information_nats:.4f} nats "
            f"(null max {d.noise_mi_null_upper:.4f})"
            f"{'  significant' if d.noise_coupling_significant else ''}"
        )
    return "\n".join(lines)
