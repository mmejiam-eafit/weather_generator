"""Dynamic Mode Decomposition and mode-level OU on a variable group's joint state.

(see docs/model_design.md §5.6, §5.6.1)

A group's state is its members' Gaussianized Tucker cores, flattened and
concatenated (`group_state`). DMD fits the deterministic linear dynamics;
`fit_mode_ou` fits the stochastic residual as one real AR(1) channel per real
mode and per conjugate pair, with one innovation covariance for the group.
`ou_transition`, `ou_stationary_cov`, `channel_path` and `mode_amplitudes` are
the closed-form pieces the sampler (`weather.simulate`) and the generative
fidelity stage (`train.fidelity`) evolve.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from pydmd import BOPDMD, DMD, HankelDMD

from train.linear_inverse import covariance_factor

CONJUGATE_PAIR_TOL: Final[float] = 1e-6  # |Λ_k' − conj(Λ_k)| below this pairs two modes
_PHI_CLIP: Final[tuple[float, float]] = (1e-4, 0.9999)
_LOG_MAGNITUDE_FLOOR: Final[float] = 1e-12  # guards log(0); negligible against real magnitudes


def group_state(cores: Mapping[str, np.ndarray], members: Sequence[str]) -> np.ndarray:
    """A group's joint DMD state: each member's core flattened, members side by side.

    (see docs/model_design.md §5.6)

    Parameters
    ----------
    cores : Mapping[str, np.ndarray]
        Tensor name -> Gaussianized core ``(T, r_lat, r_lon)``; all share `T`.
    members : Sequence[str]
        The group's tensors, in block order.

    Returns
    -------
    (T, n_G) float64, ``n_G`` the sum of the members' ``r_lat * r_lon``.
    """
    blocks = [cores[m].reshape(cores[m].shape[0], -1) for m in members]
    return np.concatenate(blocks, axis=1).astype(np.float64)


def split_group_state(
    state: np.ndarray, members: Sequence[str], core_shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """Invert `group_state`: cut a joint state into per-member cores (see §5.8 step 2).

    Parameters
    ----------
    state : (n_t, n_G)
    members : Sequence[str]
        Block order used by `group_state`.
    core_shapes : Mapping[str, tuple[int, ...]]
        Member -> its core's per-snapshot shape ``(r_lat, r_lon)``.

    Returns
    -------
    dict[str, np.ndarray]
        Member -> ``(n_t, r_lat, r_lon)``.
    """
    cores: dict[str, np.ndarray] = {}
    start = 0
    for member in members:
        shape = core_shapes[member]
        size = int(np.prod(shape))
        cores[member] = state[:, start : start + size].reshape(state.shape[0], *shape)
        start += size
    return cores


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
    state: np.ndarray,
    r_dmd: int,
    dt: float,
    rng: np.random.Generator,
    variant: Literal["exact", "bop", "hankel"] = "exact",
    bop_num_trials: int = 100,
    bop_trial_size: float = 0.5,
    hankel_delays: int = 20,
) -> DMDFit:
    """Fit DMD on a variable group's joint state via PyDMD.

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
    but is *not* enforced here: `fit_mode_ou` matches pairs by eigenvalue
    and drives each pair with one stochastic channel. See §5.6.1, §9
    gotcha 8.

    Parameters
    ----------
    state : (T, n_G) float64
        A group's joint state (`group_state`). Any trailing shape is
        flattened, so one tensor's ``(T, r_lat, r_lon)`` core also works.
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
    n_times = state.shape[0]
    snapshots = state.reshape(n_times, -1).T  # PyDMD expects (n_features, n_snapshots)

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
    """Stochastic channels of a group's DMD residual (see docs/model_design.md §5.6.1).

    One real AR(1) channel per real mode (its residual's real part) and per
    conjugate pair (the representative's de-meaned log magnitude, with a
    deterministic phase), all sharing one innovation covariance. A pair's
    partner is never a channel of its own: `mode_amplitudes` sets it to the
    conjugate, so the reconstruction is real by construction.
    """

    mode: np.ndarray  # (n_c,) int64 — the DMD mode each channel drives
    partner: np.ndarray  # (n_c,) int64 — that mode's conjugate partner; -1 for a real channel
    phi: np.ndarray  # (n_c,) float64 — AR(1) coefficient at dt, clipped to [1e-4, 0.9999]
    innovation_cov: np.ndarray  # (n_c, n_c) float64 — covariance of one-step innovations
    log_magnitude_mean: np.ndarray  # (n_c,) float64 — 0 for real channels
    phase_offset: np.ndarray  # (n_c,) float64 — phase at step 0; 0 for real channels
    dt: float  # seconds between snapshots

    @property
    def theta(self) -> np.ndarray:
        """Continuous-time mean-reversion rate per channel, ``−log(φ) / dt`` (§5.6.1)."""
        result: np.ndarray = -np.log(self.phi) / self.dt
        return result

    @property
    def sigma(self) -> np.ndarray:
        """Continuous-time diffusion per channel, ``sqrt(Q_cc · 2θ / (1 − φ²))`` (§5.6.1)."""
        variance = np.diag(self.innovation_cov) * 2.0 * self.theta / (1.0 - self.phi**2)
        result: np.ndarray = np.sqrt(variance)
        return result


def conjugate_pairs(
    Lambda: np.ndarray, tol: float = CONJUGATE_PAIR_TOL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition DMD modes into real modes and conjugate pairs (see §5.6.1).

    A real reduced operator has eigenvalues that are exactly real or exactly
    conjugate up to eigensolver rounding, so a tight tolerance recovers the
    pairing unambiguously. An oscillatory mode whose partner the truncation
    dropped is returned as real (task 07's `find_conjugate_pairs`).

    Parameters
    ----------
    Lambda : (r_dmd,) complex
    tol : float

    Returns
    -------
    real_idx : (n_real,) int64
    rep_idx : (n_pairs,) int64 — one representative per pair
    partner_idx : (n_pairs,) int64 — `rep_idx[i]`'s conjugate partner
    """
    used = np.zeros(Lambda.shape[0], dtype=bool)
    real_idx: list[int] = []
    rep_idx: list[int] = []
    partner_idx: list[int] = []
    for k in range(Lambda.shape[0]):
        if used[k]:
            continue
        used[k] = True
        if abs(Lambda[k].imag) < tol:
            real_idx.append(k)
            continue
        candidates = np.flatnonzero(~used)
        distances = np.abs(Lambda[candidates] - np.conj(Lambda[k]))
        if candidates.size and distances.min() < tol:
            partner = int(candidates[np.argmin(distances)])
            used[partner] = True
            rep_idx.append(k)
            partner_idx.append(partner)
        else:
            real_idx.append(k)
    return (
        np.array(real_idx, dtype=np.int64),
        np.array(rep_idx, dtype=np.int64),
        np.array(partner_idx, dtype=np.int64),
    )


def measured_amplitudes(state: np.ndarray, dmd: DMDFit) -> np.ndarray:
    """Least-squares mode amplitudes ``a(t) = Φ⁺ x_t`` of a group state (see §5.6.1).

    Two real products against the real and imaginary parts of ``Φ⁺``
    instead of one complex product: numpy would otherwise upcast the whole
    ``(T, n_G)`` state to complex128, a temporary twice the state's size
    (about 8 GB for one production core). A float64 state is read in place,
    never copied.

    Parameters
    ----------
    state : (T, n_G) float64
        A group state (any trailing shape is flattened).
    dmd : DMDFit

    Returns
    -------
    (r_dmd, T) complex128
    """
    snapshots = state.reshape(state.shape[0], -1)
    pinv_phi = np.linalg.pinv(dmd.Phi.astype(np.complex128))  # (r_dmd, n_G)
    real_part = snapshots @ pinv_phi.real.T
    imag_part = snapshots @ pinv_phi.imag.T
    result: np.ndarray = (real_part + 1j * imag_part).T
    return result


def _residual_amplitudes(state: np.ndarray, dmd: DMDFit) -> np.ndarray:
    """``η_k(t) = (Φ⁺ x_t)_k − b0_k Λ_k^t`` over the training snapshots, ``(r_dmd, T)``."""
    measured = measured_amplitudes(state, dmd)
    steps = np.arange(state.shape[0], dtype=np.float64)
    deterministic = dmd.b0[:, None] * dmd.Lambda[:, None] ** steps[None, :]
    result: np.ndarray = measured - deterministic
    return result


def fit_mode_ou(state: np.ndarray, dmd: DMDFit) -> ModeOUFit:
    """Fit the group's stochastic channels to the DMD residual amplitudes.

    (see docs/model_design.md §5.6.1)

    Real modes contribute ``Re η_k``; each conjugate pair contributes its
    representative's de-meaned ``log|η_k|``, with the phase anchored at the
    last training step and advancing at ``arg Λ_k`` (task 07's validated
    procedure, docs/open_questions.md §5.2). Each channel gets a clipped
    AR(1) coefficient; the one-step innovations of all channels share one
    covariance, which carries the within-group correlated noise.

    Why clip ``φ`` rather than ``|φ|``: a negative coefficient is step-to-step
    alternation, not slow reversion, and the floor treats it as noise-like.
    The clipped value is also what the sampler raises to fractional powers.

    Parameters
    ----------
    state : (T, n_G) float64
        The joint state `dmd` was fit on (any trailing shape is flattened).
    dmd : DMDFit

    Returns
    -------
    ModeOUFit
    """
    eta = _residual_amplitudes(state, dmd)
    real_idx, rep_idx, partner_idx = conjugate_pairs(dmd.Lambda)

    log_magnitude = np.log(np.abs(eta[rep_idx]) + _LOG_MAGNITUDE_FLOOR)
    log_magnitude_mean = log_magnitude.mean(axis=1)
    channels = np.concatenate(
        [eta[real_idx].real, log_magnitude - log_magnitude_mean[:, None]], axis=0
    )

    past, future = channels[:, :-1], channels[:, 1:]
    numerator = np.sum(past * future, axis=1)
    denominator = np.sum(past**2, axis=1)
    raw_phi = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    phi = np.clip(raw_phi, *_PHI_CLIP)
    innovations = future - phi[:, None] * past
    innovation_cov = innovations @ innovations.T / innovations.shape[1]

    n_real = real_idx.size
    last_step = eta.shape[1] - 1
    pair_phase = np.angle(eta[rep_idx, -1]) - last_step * np.angle(dmd.Lambda[rep_idx])
    return ModeOUFit(
        mode=np.concatenate([real_idx, rep_idx]).astype(np.int64),
        partner=np.concatenate([np.full(n_real, -1, dtype=np.int64), partner_idx]),
        phi=phi.astype(np.float64),
        innovation_cov=innovation_cov.astype(np.float64),
        log_magnitude_mean=np.concatenate([np.zeros(n_real, dtype=np.float64), log_magnitude_mean]),
        phase_offset=np.concatenate([np.zeros(n_real, dtype=np.float64), pair_phase]),
        dt=dmd.dt,
    )


def ou_stationary_cov(ou: ModeOUFit) -> np.ndarray:
    """Stationary channel covariance ``P_ab = Q_ab / (1 − φ_a φ_b)`` (see §5.6.1).

    Parameters
    ----------
    ou : ModeOUFit

    Returns
    -------
    (n_c, n_c) float64
    """
    result: np.ndarray = ou.innovation_cov / (1.0 - np.outer(ou.phi, ou.phi))
    return result


def ou_transition(ou: ModeOUFit, n_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form channel transition over ``n_steps · dt`` (see §5.6.1).

    ``c(t + n dt) = φ^n ⊙ c(t) + w`` with
    ``w ~ N(0, C(n))``, ``C(n)_ab = Q_ab (1 − (φ_a φ_b)^n) / (1 − φ_a φ_b)``.
    Defined for fractional ``n ≥ 0`` because ``φ`` is clipped positive. The
    stationary covariance is invariant under it.

    Parameters
    ----------
    ou : ModeOUFit
    n_steps : (n_t,) float, ``>= 0``

    Returns
    -------
    decay : (n_t, n_c) float64 — ``φ^n``
    noise_cov : (n_t, n_c, n_c) float64 — ``C(n)``
    """
    n = np.asarray(n_steps, dtype=np.float64).reshape(-1)
    decay = ou.phi[None, :] ** n[:, None]
    phi_products = np.outer(ou.phi, ou.phi)
    noise_cov = (
        ou.innovation_cov[None]
        * (1.0 - phi_products[None] ** n[:, None, None])
        / (1.0 - phi_products)[None]
    )
    return decay, noise_cov


def _stationary_draw(ou: ModeOUFit, rng: np.random.Generator) -> np.ndarray:
    """One draw of a group's channels from their joint stationary law, ``(n_c,)``."""
    result: np.ndarray = covariance_factor(ou_stationary_cov(ou)) @ rng.standard_normal(ou.phi.size)
    return result


def channel_path(ou: ModeOUFit, delta_n: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One realization of a group's channels at increasing times, ``(n_c, n_t)`` (§5.8 step 1).

    A stationary draw at the first time, then the closed-form transition
    across each gap to the next time. The channels are a Markov process, so
    this gives the exact joint law at every requested time, regular grid or
    not. Each distinct gap is factorized once: a regular grid needs one
    factorization, and its one-step noise covariance is the innovation
    covariance itself. The sampler (`weather.simulate`) and the generative
    fidelity stage (`train.fidelity`, §5.6.3) share this function, so the
    stage judges exactly the law the sampler draws from.

    Parameters
    ----------
    ou : ModeOUFit
    delta_n : (n_t,) float, strictly increasing
        Steps since the training window's first snapshot.
    rng : np.random.Generator

    Returns
    -------
    (n_c, n_t) float64
    """
    path = np.empty((ou.phi.size, delta_n.size), dtype=np.float64)
    path[:, 0] = _stationary_draw(ou, rng)
    if delta_n.size == 1:
        return path
    gaps, gap_index = np.unique(np.diff(delta_n), return_inverse=True)
    decay, noise_cov = ou_transition(ou, gaps)
    factors = covariance_factor(noise_cov)
    standard_normal = rng.standard_normal((delta_n.size - 1, ou.phi.size))
    for step in range(1, delta_n.size):
        k = gap_index[step - 1]
        path[:, step] = decay[k] * path[:, step - 1] + factors[k] @ standard_normal[step - 1]
    return path


def mode_amplitudes(
    dmd: DMDFit, ou: ModeOUFit, channels: np.ndarray, delta_n: np.ndarray
) -> np.ndarray:
    """Complex DMD mode amplitudes from channel values (see §5.6.1, §5.8 step 1).

    ``a_k = b0_k Λ_k^Δn + η_k``, with ``η_k`` the channel value for a real
    mode and ``exp(c + log_magnitude_mean) · exp(i(phase_offset + Δn arg Λ_k))``
    for a pair's representative; the partner gets the exact conjugate.

    Caveat: a real mode with a negative eigenvalue (it alternates sign every
    step) has no real fractional power, so ``Λ^Δn`` is complex at fractional
    ``Δn`` and the field's real part keeps only its projection. Such modes
    have ``|Λ| ≪ 1`` in practice and vanish within a few steps of the
    training window's start.

    Parameters
    ----------
    dmd : DMDFit
    ou : ModeOUFit
    channels : (n_c, n_t) float64
    delta_n : (n_t,) float64
        Steps since the training window's first snapshot (fractional).

    Returns
    -------
    (r_dmd, n_t) complex128
    """
    delta_n = np.asarray(delta_n, dtype=np.float64)
    amplitudes = dmd.b0[:, None] * dmd.Lambda[:, None] ** delta_n[None, :]
    real = ou.partner < 0
    amplitudes[ou.mode[real]] += channels[real]

    pair = ~real
    magnitude = np.exp(channels[pair] + ou.log_magnitude_mean[pair, None])
    frequency = np.angle(dmd.Lambda[ou.mode[pair]])
    phase = ou.phase_offset[pair, None] + frequency[:, None] * delta_n[None, :]
    eta = magnitude * np.exp(1j * phase)
    amplitudes[ou.mode[pair]] += eta
    amplitudes[ou.partner[pair]] += np.conj(eta)
    result: np.ndarray = amplitudes.astype(np.complex128)
    return result
