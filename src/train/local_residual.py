"""Per-cell local residual Ornstein-Uhlenbeck field (see docs/model_design.md §3.10, §5.7)."""

from dataclasses import dataclass

import numpy as np

from train.tucker import TuckerFit, tucker_inverse


@dataclass(frozen=True, slots=True)
class LocalResidualFit:
    """Fitted per-cell, per-variable local residual OU parameters (see docs/model_design.md §5.7).

    One independent Ornstein-Uhlenbeck process per ocean cell per variable —
    the field-valued analog of `train.dmd.ModeOUFit`, fit to whatever the
    coherent (Tucker + DMD + mode-OU) reconstruction misses.
    """

    theta: np.ndarray  # (N_ocean, V) float32 — mean-reversion rate
    sigma: np.ndarray  # (N_ocean, V) float32 — diffusion coefficient
    phi: np.ndarray  # (N_ocean, V) float32 — discrete AR(1) coefficient, exp(-theta * dt)
    variance_ratio: np.ndarray  # (N_ocean, V) float32 — Var(R)/Var(Z), diagnostic
    active_mask: np.ndarray  # (N_ocean, V) bool — True where variance_ratio > floor
    dt: float  # seconds between snapshots


def coherent_reconstruction(core_deterministic: np.ndarray, tucker: TuckerFit) -> np.ndarray:
    """Compute `Z_coherent(s, v, t)` from the fitted DMD's deterministic reconstruction.

    (see docs/model_design.md §3.10, §5.7)

    A one-line wrapper around `train.tucker.tucker_inverse` — it exists as
    its own named function purely for call-site clarity (notebook §8.6 Cell
    8, `weather/model.py` task 08), not because the logic differs from
    `tucker_inverse` itself.

    **Semantics of `core_deterministic` matter.** This must be the DMD's
    *deterministic* trajectory ``b0_k · exp(lam_cont_k · t)`` propagated
    through the Tucker core — i.e. `â_k(t)` in `train.dmd.fit_mode_ou`'s
    docstring — with *no* mode-OU stochastic wobble `η_k(t)` added in. The
    residual that `fit_local_residual` fits (`R = Z - Z_coherent`) is only
    meaningful if `Z_coherent` excludes that wobble; a caller (task 08) that
    accidentally includes `η_k(t)` here would leak coherent-layer stochastic
    variance into the local-residual fit and double-count it at sample time.

    Deferred in v3 (docs/model_design.md §5.7): Tucker is now per tensor, so
    this reconstructs one tensor's grid; `fit_local_residual` below still
    has v2's raveled ``(T, N_ocean, V)`` layout and is not wired into
    `WeatherModel`.

    Parameters
    ----------
    core_deterministic : (T, r_lat, r_lon) float64
        DMD forward's deterministic reconstruction, with no OU term added.
    tucker : TuckerFit
        The same Tucker fit `core_deterministic` was expressed in.

    Returns
    -------
    Z_coherent : (T, H, W) float64
    """
    return tucker_inverse(core_deterministic, tucker)


def fit_local_residual(
    Z: np.ndarray,
    Z_coherent: np.ndarray,
    dt: float,
    variance_floor: float = 0.02,
) -> LocalResidualFit:
    """Fit independent per-cell OU to the residual R = Z - Z_coherent.

    (see docs/model_design.md §3.10, §5.7)

    Vectorized AR(1) fit across all cells and variables simultaneously — no
    loop, no joblib (§9 gotcha #10: the local-residual fit is embarrassingly
    parallel per cell, and here that parallelism is expressed as a single
    vectorized numpy reduction over the time axis rather than an explicit
    loop or `joblib.Parallel`).

    Parameters
    ----------
    Z : (T, N_ocean, V) float64
        Gaussianized anomalies.
    Z_coherent : (T, N_ocean, V) float64
        Coherent-layer reconstruction, e.g. the output of
        `coherent_reconstruction`. Must be the *deterministic* DMD
        trajectory (no mode-OU wobble) — see `coherent_reconstruction`'s
        docstring.
    dt : float
        Seconds between snapshots.
    variance_floor : float
        Cells with ``Var(R) / Var(Z) <= variance_floor`` have their OU
        parameters zeroed to save simulation work (§3.10).

    Returns
    -------
    LocalResidualFit
    """
    R = Z - Z_coherent  # (T, N_ocean, V)
    R_prev = R[:-1]  # (T-1, N_ocean, V)
    R_next = R[1:]  # (T-1, N_ocean, V)

    # AR(1) coefficient per cell per variable
    num = np.sum(R_prev * R_next, axis=0)  # (N_ocean, V)
    den = np.sum(R_prev * R_prev, axis=0) + 1e-30
    phi = np.clip(num / den, -0.9999, 0.9999).astype(np.float32)

    # Continuous-time reversion rate: phi = exp(-theta * dt)
    # Clip phi itself (not |phi|) to [1e-4, 0.9999]: negative phi indicates
    # alternation (anti-correlation), which we treat as fast decorrelation by
    # pushing it to the clip floor, not by discarding its sign via abs() (the
    # same distinction §5.6.1's mode-OU theta makes, there via Re(phi) since
    # that phi is complex; here phi is already real, so plain clip suffices).
    theta = -np.log(np.clip(phi, 1e-4, 0.9999)) / dt

    # Diffusion coefficient from the AR(1) *innovation* variance, not R's raw
    # marginal (stationary) variance. These are not interchangeable: at
    # stationarity Var(R) = innovation_variance / (1 - phi**2), so using
    # Var(R) directly in the formula below double-counts the (1 - phi**2)
    # factor and inflates sigma by 1/sqrt(1 - phi**2) -- negligible near
    # phi=0 but a 3x+ overestimate at phi=0.95, verified numerically. This is
    # the same distinction `train.dmd.fit_mode_ou` already makes for the
    # complex mode-level OU fit; §5.7's local-residual formula previously
    # used raw Var(R) here, which was wrong for exactly the same reason.
    innovation = R_next - phi[None, :, :] * R_prev
    innovation_variance = np.mean(innovation**2, axis=0)  # (N_ocean, V)
    var_R = R.var(axis=0)  # (N_ocean, V) -- only used for variance_ratio below
    var_Z = Z.var(axis=0) + 1e-30
    # Continuous-time diffusion: innovation_variance = sigma^2 * (1 - phi^2) / (2 theta)
    sigma = np.sqrt(innovation_variance * 2 * theta / (1 - phi**2 + 1e-30)).astype(np.float32)

    variance_ratio = (var_R / var_Z).astype(np.float32)
    active_mask = variance_ratio > variance_floor

    # Zero out inactive cells to save simulation work
    theta = np.where(active_mask, theta, 0.0).astype(np.float32)
    sigma = np.where(active_mask, sigma, 0.0).astype(np.float32)
    phi = np.where(active_mask, phi, 0.0).astype(np.float32)

    return LocalResidualFit(
        theta=theta,
        sigma=sigma,
        phi=phi,
        variance_ratio=variance_ratio,
        active_mask=active_mask,
        dt=dt,
    )
