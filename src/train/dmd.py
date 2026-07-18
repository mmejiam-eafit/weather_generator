"""Dynamic Mode Decomposition and mode-level OU fit on the Tucker core.

(see docs/model_design.md §5.6, §5.6.1)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydmd import BOPDMD, DMD, HankelDMD


@dataclass(frozen=True, slots=True)
class DMDFit:
    """Fitted DMD modes and eigenvalues (see docs/model_design.md §5.6)."""

    Phi: np.ndarray  # (N_state, r_dmd) complex64
    Lambda: np.ndarray  # (r_dmd,) complex128 — discrete-time eigenvalues
    b0: np.ndarray  # (r_dmd,) complex128 — initial amplitudes
    lam_cont: np.ndarray  # (r_dmd,) complex128 — continuous-time eigenvalues
    dt: float  # seconds between snapshots
    variant: str  # "exact", "bop", or "hankel" — provenance


def dmd_fit(
    core: np.ndarray,
    r_dmd: int,
    dt: float,
    rng: np.random.Generator,
    variant: Literal["exact", "bop", "hankel"] = "exact",
    bop_num_trials: int = 100,
    bop_trial_size: float = 0.5,
    hankel_delays: int = 20,
) -> DMDFit:
    """Fit DMD on the Tucker core's flattened snapshots via PyDMD.

    (see docs/model_design.md §5.6)

    Three variants, all sharing the same `DMDFit` shape:

    - ``"exact"`` — plain exact DMD (Tu et al. 2014). Fast; the default for
      repeated rank sweeps during notebook iteration.
    - ``"bop"`` — bagging, optimized DMD (Sashidhar & Kutz 2022) with a
      complex-conjugate-pair eigenvalue constraint. ~100x slower (bags
      `bop_num_trials` models) but gives bagged uncertainty quantification
      and *automatically* enforces conjugate eigenvalue pairs — reserved for
      the one final production fit, once ranks are settled.
    - ``"hankel"`` — time-delay embedded DMD (Brunton et al. 2017). Use if
      per-cell temporal spectra undershoot at high frequency.

    Non-real eigenvalues arise naturally in conjugate pairs for a real
    reduced operator; traveling waves appear as such a pair with
    ``Im(lam_cont)`` set by the propagation frequency. With ``variant=
    "bop"``, `eig_constraints` enforces this pairing at fit time. With
    ``variant="exact"`` or ``"hankel"``, the pairing is present in the fit
    but is *not* enforced here — it is the caller's responsibility (at
    simulation time, e.g. `weather.simulate`) to force
    ``η_{k'}(t) = conj(η_k(t))`` at every step. See §5.6, §9 gotcha #8.

    Parameters
    ----------
    core : (T, r_ocean, r_var) float64
        Tucker-compressed snapshots, e.g. the output of `train.tucker.
        tucker_transform`.
    r_dmd : int
        DMD truncation rank.
    dt : float
        Seconds between snapshots (3600 * 3 for this archive).
    rng : np.random.Generator
        Source of randomness. Unused by the current PyDMD variants (kept
        for interface symmetry with the rest of `train/` and in case a
        future variant needs it).
    variant : {"exact", "bop", "hankel"}
        Which PyDMD estimator to use. Default ``"exact"``.
    bop_num_trials : int
        Number of bagged trials for ``variant="bop"``. Ignored otherwise.
    bop_trial_size : float
        Fraction of snapshots per bagged trial for ``variant="bop"``.
        Ignored otherwise.
    hankel_delays : int
        Number of time-delay embeddings for ``variant="hankel"``. Ignored
        otherwise.

    Returns
    -------
    DMDFit
    """
    n_times, r_ocean, r_var = core.shape
    n_state = r_ocean * r_var
    snapshots = core.reshape(n_times, n_state).T  # PyDMD expects (n_features, n_snapshots)

    if variant == "exact":
        model = DMD(svd_rank=r_dmd)
        model.fit(snapshots)
        Lambda = np.asarray(model.eigs, dtype=np.complex128)
        Phi = np.asarray(model.modes, dtype=np.complex64)
        b0 = np.asarray(model.amplitudes, dtype=np.complex128)
    elif variant == "bop":
        t_grid = np.arange(n_times, dtype=np.float64) * dt
        bop_model = BOPDMD(
            svd_rank=r_dmd,
            num_trials=bop_num_trials,
            trial_size=bop_trial_size,
            eig_constraints={"conjugate_pairs"},
        )
        bop_model.fit(snapshots, t=t_grid)
        # BOPDMD's .eigs are already continuous-time (unlike "exact"/"hankel"
        # below, which are discrete-time) — exponentiate to land on the same
        # discrete-time Lambda convention before the shared lam_cont step.
        Lambda = np.exp(np.asarray(bop_model.eigs, dtype=np.complex128) * dt)
        Phi = np.asarray(bop_model.modes, dtype=np.complex64)
        b0 = np.asarray(bop_model.amplitudes, dtype=np.complex128)
    elif variant == "hankel":
        hankel_model = HankelDMD(svd_rank=r_dmd, d=hankel_delays)
        hankel_model.fit(snapshots)
        Lambda = np.asarray(hankel_model.eigs, dtype=np.complex128)
        Phi = np.asarray(hankel_model.modes, dtype=np.complex64)
        b0 = np.asarray(hankel_model.amplitudes, dtype=np.complex128)
    else:
        raise ValueError(f"Unknown DMD variant: {variant!r}")

    lam_cont = (np.log(Lambda) / dt).astype(np.complex128)

    return DMDFit(
        Phi=Phi,
        Lambda=Lambda,
        b0=b0,
        lam_cont=lam_cont,
        dt=dt,
        variant=variant,
    )


@dataclass(frozen=True, slots=True)
class ModeOUFit:
    """Fitted per-mode Ornstein-Uhlenbeck parameters (see docs/model_design.md §5.6.1)."""

    theta: np.ndarray  # (r_dmd,) float64
    sigma: np.ndarray  # (r_dmd,) float64
    phi: np.ndarray  # (r_dmd,) complex128 — discrete AR(1) coefficient
    dt: float


def fit_mode_ou(core: np.ndarray, dmd: DMDFit) -> ModeOUFit:
    """Fit per-mode OU parameters to the DMD residual amplitudes.

    (see docs/model_design.md §5.6.1)

    For each DMD mode ``k``, the measured complex amplitude
    ``a_k(t) = (Phi⁺ · x_t)_k`` is compared against the deterministic DMD
    prediction ``â_k(t) = b0_k · exp(lam_cont_k · t)``; the residual
    ``η_k(t) = a_k(t) − â_k(t)`` is the stochastic wobble around the DMD's
    deterministic trajectory. This residual is modeled as a mean-reverting
    complex Ornstein-Uhlenbeck process, recovered here from its exact
    discrete-time AR(1) transition statistics.

    Complex-conjugate pairs: this function fits each mode independently and
    does not itself enforce or rely on conjugate-pair structure between
    modes. When `dmd` was fit with ``variant="bop"``, `dmd.Phi`/`dmd.Lambda`
    already come as conjugate pairs (via PyDMD's `eig_constraints`); with
    ``variant="exact"`` or ``"hankel"``, the pairing is present in the
    spectrum but preserving it at simulation time — forcing
    ``η_{k'}(t) = conj(η_k(t))`` at every step — is the caller's
    responsibility (`weather.simulate`, a later task), not this function's.

    Parameters
    ----------
    core : (T, r_ocean, r_var) float64
        The same Tucker core `dmd` was fit on.
    dmd : DMDFit
        Fitted DMD modes/eigenvalues, e.g. the output of `dmd_fit`.

    Returns
    -------
    ModeOUFit
    """
    n_times = core.shape[0]
    snapshots = core.reshape(n_times, -1).T  # (N_state, T)
    a_t = np.linalg.pinv(dmd.Phi) @ snapshots  # (r_dmd, T) — measured amplitude a_k(t)

    tt = np.arange(n_times)
    a_det = dmd.b0[:, None] * (dmd.Lambda[:, None] ** tt[None, :])  # â_k(t)
    eta = a_t - a_det  # (r_dmd, T) — residual η_k(t)

    eta_t = eta[:, :-1]
    eta_t1 = eta[:, 1:]

    numerator = np.sum(np.conj(eta_t) * eta_t1, axis=1)
    denominator = np.sum(np.abs(eta_t) ** 2, axis=1)
    phi = numerator / denominator

    theta = -np.log(np.clip(phi.real, 1e-4, 0.9999)) / dmd.dt

    # "Var(eta_k)" in the source formula is the AR(1) innovation variance
    # (variance of the residual after removing the phi*eta_t term), not the
    # marginal variance of eta_k itself: Var(eta) = sigma^2 / (2 theta) at
    # stationarity, which would double-count the (1 - |phi|^2) factor below.
    innovation = eta_t1 - phi[:, None] * eta_t
    innovation_variance = np.mean(np.abs(innovation) ** 2, axis=1)
    sigma = np.sqrt(innovation_variance * 2.0 * theta / (1.0 - np.abs(phi) ** 2))

    return ModeOUFit(
        theta=theta.astype(np.float64),
        sigma=sigma.astype(np.float64),
        phi=phi.astype(np.complex128),
        dt=dmd.dt,
    )
