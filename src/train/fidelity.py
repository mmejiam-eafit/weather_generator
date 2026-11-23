"""Generative fidelity: do samples from a group's fitted dynamics resemble its training cores?

(see docs/model_design.md §5.6.3)

The training-pipeline stage that gates a fit. A group's fitted DMD and mode
OU are simulated exactly as the sampler simulates them (`train.dmd.
channel_path`, then `train.dmd.mode_amplitudes`, §5.8 step 1), in replicates
as long as the training window. Each member's replicates are compared with
the Gaussianized core the model was fit on, in core space:

- **variance**: mean per-snapshot core energy;
- **memory**: autocorrelation of the member's leading principal components;
- **time of day**: a daily cycle left in those components.

A check fails only when the training statistic lies outside the replicates'
(or permutations') envelope **and** differs by more than a practical
tolerance. The envelope alone would fire on real but harmless deviations;
the tolerance alone would ignore sampling noise, about ±20% on the energy of
slow multi-day components over one year. The tolerances are judgement, not
derived (§5.6.3); the production `wsh` errors exceed each by a wide margin.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from train.copula import SATURATION_Z
from train.dmd import DMDFit, ModeOUFit, channel_path, measured_amplitudes, mode_amplitudes
from train.linear_inverse import group_pcs

VARIANCE_RATIO_TOLERANCE: Final[float] = 1.5  # about 22% in standard deviation
ACF_TOLERANCE: Final[float] = 0.1
TIME_OF_DAY_TOLERANCE: Final[float] = 0.05  # share of the leading PCs' variance
RMS_PERCENTILES: Final[tuple[float, ...]] = (5.0, 50.0, 95.0, 99.9)
_COEFFICIENT_SNAPSHOTS: Final[int] = 300  # time subsample for per-coefficient statistics
_CHUNK: Final[int] = 64  # snapshots materialized at once


@dataclass(frozen=True, slots=True)
class VarianceCheck:
    """Sampled versus training core energy for one member (see §5.6.3)."""

    training_energy: float  # mean over snapshots of ||x_t||^2
    replicate_energy: np.ndarray  # (n,) the same for each replicate
    span_fraction: float  # training energy the DMD reconstruction with Φ⁺x reproduces
    rms_quantiles: np.ndarray  # (4,) sampled per-coefficient RMS at RMS_PERCENTILES
    saturated_training: float  # fraction of training values beyond SATURATION_Z
    saturated_sampled: float  # the same for the first replicate
    consistent: bool

    @property
    def ratio(self) -> float:
        """Mean replicate energy over training energy; 1 for a faithful model."""
        return float(self.replicate_energy.mean() / self.training_energy)


@dataclass(frozen=True, slots=True)
class MemoryCheck:
    """Autocorrelation of a member's leading PCs, training versus replicates (see §5.6.3)."""

    lags: np.ndarray  # (k,) int64, snapshots
    training: np.ndarray  # (k,)
    band: np.ndarray  # (2, k) replicate minimum / maximum
    replicate_mean: np.ndarray  # (k,)
    consistent: np.ndarray  # (k,) bool


@dataclass(frozen=True, slots=True)
class TimeOfDayCheck:
    """Residual daily cycle in a member's leading PCs (see §5.6.3, §5.4)."""

    share: float  # variance share explained by the mean at each time of day
    null_mean: float  # the same with labels permuted within each day: mean
    null_max: float  # and maximum over the permutations
    consistent: bool


@dataclass(frozen=True, slots=True)
class FidelityDiagnostics:
    """Generative fidelity stage output for one variable group (see §5.6.3)."""

    members: tuple[str, ...]
    n_replicates: int
    dt: float  # seconds between snapshots
    variance: dict[str, VarianceCheck]
    memory: dict[str, MemoryCheck]
    time_of_day: dict[str, TimeOfDayCheck]

    @property
    def passed(self) -> bool:
        """True when every member passes every check."""
        return all(
            self.variance[m].consistent
            and bool(self.memory[m].consistent.all())
            and self.time_of_day[m].consistent
            for m in self.members
        )


def simulate_amplitudes(
    dmd: DMDFit, ou: ModeOUFit, n_steps: int, rng: np.random.Generator
) -> np.ndarray:
    """One realization of a group's complex mode amplitudes on its training grid.

    (see docs/model_design.md §5.6.3, §5.8 step 1)

    The sampler's own law: `channel_path` from a stationary draw, then
    `mode_amplitudes`, at ``Δn = 0, 1, ..., n_steps − 1``, the training
    window's snapshots.

    Parameters
    ----------
    dmd : DMDFit
    ou : ModeOUFit
    n_steps : int
    rng : np.random.Generator

    Returns
    -------
    (r_dmd, n_steps) complex128
    """
    delta_n = np.arange(n_steps, dtype=np.float64)
    return mode_amplitudes(dmd, ou, channel_path(ou, delta_n, rng), delta_n)


def energy_grams(phi_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two ``(r, r)`` Gram matrices `member_energy` needs, ``(Φᴴ Φ, Φᵀ Φ)``.

    Parameters
    ----------
    phi_block : (n_m, r_dmd) complex
        The rows of ``Φ`` belonging to one member.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Both ``(r_dmd, r_dmd)`` complex128.
    """
    phi = phi_block.astype(np.complex128)
    return phi.conj().T @ phi, phi.T @ phi


def member_energy(grams: tuple[np.ndarray, np.ndarray], amplitudes: np.ndarray) -> np.ndarray:
    """Per-snapshot energy ``‖Re(Φ_m a_t)‖²`` without materializing the state.

    (see docs/model_design.md §5.6.3)

    With ``x = Re(Φ a) = (Φ a + conj(Φ a)) / 2``,
    ``‖x‖² = ½ Re(aᴴ Φᴴ Φ a) + ½ Re(aᵀ Φᵀ Φ a)``: two ``(r, r)`` products per
    snapshot instead of an ``(n_m, T)`` reconstruction. The cross terms
    between modes are exactly what the training data's cancellation lives
    in, so this is the energy of the field, not the sum of per-mode energies.

    Parameters
    ----------
    grams : tuple[np.ndarray, np.ndarray]
        `energy_grams` of the member's rows of ``Φ``.
    amplitudes : (r_dmd, n_t) complex

    Returns
    -------
    (n_t,) float64
    """
    gram_h, gram_t = grams
    hermitian = np.einsum("kt,kt->t", amplitudes.conj(), gram_h @ amplitudes).real
    plain = np.einsum("kt,kt->t", amplitudes, gram_t @ amplitudes).real
    result: np.ndarray = 0.5 * (hermitian + plain)
    return result


def subspace_acf(z: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Energy-weighted autocorrelation of a multivariate series (see §5.6.3).

    Each column is centred. The value at lag ``l`` is the mean lagged inner
    product ``Σ_t z_t · z_{t+l} / (T − l)`` over the mean energy
    ``Σ_t ‖z_t‖² / T``, so components with more variance count more.

    Parameters
    ----------
    z : (T, k) float64
    lags : Sequence[int]
        Each in ``[1, T)``.

    Returns
    -------
    (len(lags),) float64
    """
    centered = z - z.mean(axis=0)
    n_times = centered.shape[0]
    energy = np.sum(centered * centered) / n_times
    return np.array(
        [np.sum(centered[lag:] * centered[:-lag]) / (n_times - lag) / energy for lag in lags],
        dtype=np.float64,
    )


def time_of_day_share(
    pcs: np.ndarray, times: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Variance share of a series explained by its mean at each time of day, with a null.

    (see docs/model_design.md §5.6.3, §5.4)

    The null permutes the time-of-day labels within each UTC day: it keeps
    the day-to-day variability and removes only a cycle locked to the
    clock. With few snapshots per label the share is biased upward even
    without a cycle, which the null's mean measures.

    Parameters
    ----------
    pcs : (T, k) float64
    times : (T,) datetime64
        UTC snapshot times.
    n_permutations : int
    rng : np.random.Generator

    Returns
    -------
    share, null_mean, null_max : float
    """
    days = times.astype("datetime64[D]")
    seconds = (times - days).astype("timedelta64[s]").astype(np.int64)
    label = np.unique(seconds, return_inverse=True)[1]
    day = np.unique(days, return_inverse=True)[1]
    centered = pcs - pcs.mean(axis=0)
    total = float(np.sum(centered * centered))

    by_day = np.argsort(day, kind="stable")
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        # Rows grouped by day, shuffled within it; position j of `shuffled`
        # and of `by_day` always falls in the same day.
        shuffled = np.lexsort((rng.random(day.size), day))
        permuted = np.empty_like(label)
        permuted[by_day] = label[shuffled]
        null[i] = _label_mean_energy(centered, permuted) / total
    share = _label_mean_energy(centered, label) / total
    return share, float(null.mean()), float(null.max())


def _label_mean_energy(z: np.ndarray, label: np.ndarray) -> float:
    """``Σ_labels n_l ‖mean_l‖²``: the energy of each row replaced by its label's mean."""
    counts = np.bincount(label)
    sums = np.zeros((counts.size, z.shape[1]), dtype=np.float64)
    np.add.at(sums, label, z)
    return float(np.sum(sums**2 / counts[:, None]))


def diagnose_fidelity(
    state: np.ndarray,
    member_sizes: Mapping[str, int],
    dmd: DMDFit,
    ou: ModeOUFit,
    times: np.ndarray,
    lags: Sequence[int],
    ranks: Mapping[str, int],
    n_replicates: int,
    rng: np.random.Generator,
) -> FidelityDiagnostics:
    """Run the generative fidelity stage on one group (see docs/model_design.md §5.6.3).

    Simulates `n_replicates` realizations of the fitted dynamics, each as
    long as the training window, and judges each member's variance, memory
    and residual daily cycle against its training core. Records and judges;
    the caller decides what a failure means (`WeatherModel` refuses it).

    Parameters
    ----------
    state : (T, n_G) float64
        The group state `dmd` and `ou` were fit on (`train.dmd.group_state`).
        Read in place: member blocks are views.
    member_sizes : Mapping[str, int]
        Member -> its number of core coefficients, in the state's block order.
    dmd : DMDFit
    ou : ModeOUFit
    times : (T,) datetime64
        UTC snapshot times of the training window.
    lags : Sequence[int]
        Memory-check lags, in snapshots; each ``>= 1`` and below ``T / 2``.
    ranks : Mapping[str, int]
        Member -> leading principal components the memory and time-of-day
        checks read (see `train.linear_inverse.group_pcs`).
    n_replicates : int
        Simulated replicates and time-of-day permutations (>= 1).
    rng : np.random.Generator

    Returns
    -------
    FidelityDiagnostics

    Raises
    ------
    ValueError
        If a lag is outside ``[1, T / 2)``, `n_replicates` is below 1, or
        `member_sizes` does not add up to the state's width.
    """
    n_times = state.shape[0]
    snapshots = state.reshape(n_times, -1)
    _check_arguments(lags, n_times, n_replicates, member_sizes, snapshots.shape[1])
    blocks = _member_blocks(member_sizes)
    members = tuple(blocks)

    pcs = group_pcs({m: snapshots[:, blocks[m]] for m in members}, dict(ranks))
    member_pcs = {m: pcs.pcs[:, pcs.blocks[m]] for m in members}
    grams = {m: energy_grams(dmd.Phi[blocks[m]]) for m in members}
    projection = {
        m: _eofs(snapshots[:, blocks[m]], member_pcs[m]).T
        @ dmd.Phi[blocks[m]].astype(np.complex128)
        for m in members
    }

    energies = {m: np.empty(n_replicates, dtype=np.float64) for m in members}
    acfs = {m: np.empty((n_replicates, len(lags)), dtype=np.float64) for m in members}
    sampled_stats: dict[str, tuple[np.ndarray, float]] = {}
    for i in range(n_replicates):
        amplitudes = simulate_amplitudes(dmd, ou, n_times, rng)
        for m in members:
            energies[m][i] = member_energy(grams[m], amplitudes).mean()
            acfs[m][i] = subspace_acf((projection[m] @ amplitudes).real.T, lags)
            if i == 0:
                sampled_stats[m] = _sampled_coefficient_stats(dmd.Phi[blocks[m]], amplitudes)

    measured = measured_amplitudes(snapshots, dmd)
    variance: dict[str, VarianceCheck] = {}
    memory: dict[str, MemoryCheck] = {}
    time_of_day: dict[str, TimeOfDayCheck] = {}
    for m in members:
        training_energy, saturated_training = _training_energy_stats(snapshots[:, blocks[m]])
        variance[m] = _variance_check(
            training_energy,
            energies[m],
            span_fraction=float(member_energy(grams[m], measured).mean() / training_energy),
            rms_quantiles=sampled_stats[m][0],
            saturated_training=saturated_training,
            saturated_sampled=sampled_stats[m][1],
        )
        memory[m] = _memory_check(lags, subspace_acf(member_pcs[m], lags), acfs[m])
        time_of_day[m] = _time_of_day_check(
            *time_of_day_share(member_pcs[m], times, n_replicates, rng)
        )

    return FidelityDiagnostics(
        members=members,
        n_replicates=n_replicates,
        dt=dmd.dt,
        variance=variance,
        memory=memory,
        time_of_day=time_of_day,
    )


def _check_arguments(
    lags: Sequence[int],
    n_times: int,
    n_replicates: int,
    member_sizes: Mapping[str, int],
    state_width: int,
) -> None:
    """Reject arguments `diagnose_fidelity` cannot judge with."""
    if not lags or min(lags) < 1 or 2 * max(lags) >= n_times:
        raise ValueError(
            f"fidelity lags must be >= 1 and below T / 2 = {n_times / 2:g} snapshots, "
            f"got {list(lags)} (docs/model_design.md §5.6.3)"
        )
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    if sum(member_sizes.values()) != state_width:
        raise ValueError(
            f"member sizes {dict(member_sizes)} add up to {sum(member_sizes.values())}, "
            f"but the group state has {state_width} columns"
        )


def _member_blocks(member_sizes: Mapping[str, int]) -> dict[str, slice]:
    """Member -> its columns of the group state, in block order."""
    blocks: dict[str, slice] = {}
    start = 0
    for member, size in member_sizes.items():
        blocks[member] = slice(start, start + size)
        start += size
    return blocks


def _eofs(x: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """Unit-norm EOFs ``(n_m, k)`` behind time-centred `pcs` of `x`.

    ``E = xᵀ pcs / ‖pcs‖²`` per column. The PCs have zero time mean, so
    ``xᵀ pcs`` equals ``x_centredᵀ pcs`` and `x` needs no centred copy.
    """
    result: np.ndarray = (x.T @ pcs) / np.sum(pcs**2, axis=0)
    return result


def _training_energy_stats(x: np.ndarray) -> tuple[float, float]:
    """Mean per-snapshot energy of a ``(T, n_m)`` block, and its fraction beyond SATURATION_Z."""
    sum_sq = 0.0
    n_saturated = 0
    for start in range(0, x.shape[0], _CHUNK):
        chunk = x[start : start + _CHUNK]
        sum_sq += float(np.sum(chunk * chunk))
        n_saturated += int(np.count_nonzero(np.abs(chunk) > SATURATION_Z))
    return sum_sq / x.shape[0], n_saturated / x.size


def _sampled_coefficient_stats(
    phi_block: np.ndarray, amplitudes: np.ndarray
) -> tuple[np.ndarray, float]:
    """Per-coefficient RMS quantiles and saturated fraction of one sampled realization.

    Materializes ``Re(Φ_m a_t)`` on about `_COEFFICIENT_SNAPSHOTS` evenly
    spaced snapshots, `_CHUNK` at a time, as two real products so the
    complex field is never formed.
    """
    n_steps = amplitudes.shape[1]
    steps = np.unique(
        np.linspace(0, n_steps - 1, min(n_steps, _COEFFICIENT_SNAPSHOTS)).astype(np.int64)
    )
    phi_real = phi_block.real.astype(np.float64)
    phi_imag = phi_block.imag.astype(np.float64)
    sum_sq = np.zeros(phi_block.shape[0], dtype=np.float64)
    n_saturated = 0
    for start in range(0, steps.size, _CHUNK):
        a = amplitudes[:, steps[start : start + _CHUNK]]
        x = phi_real @ a.real - phi_imag @ a.imag
        sum_sq += np.sum(x * x, axis=1)
        n_saturated += int(np.count_nonzero(np.abs(x) > SATURATION_Z))
    rms = np.sqrt(sum_sq / steps.size)
    quantiles = np.percentile(rms, RMS_PERCENTILES).astype(np.float64)
    return quantiles, n_saturated / (phi_block.shape[0] * steps.size)


def _variance_check(
    training_energy: float,
    replicate_energy: np.ndarray,
    span_fraction: float,
    rms_quantiles: np.ndarray,
    saturated_training: float,
    saturated_sampled: float,
) -> VarianceCheck:
    """Judge the energy: outside the replicates' envelope and beyond the tolerance fails."""
    outside = not replicate_energy.min() <= training_energy <= replicate_energy.max()
    ratio = replicate_energy.mean() / training_energy
    beyond = not 1.0 / VARIANCE_RATIO_TOLERANCE <= ratio <= VARIANCE_RATIO_TOLERANCE
    return VarianceCheck(
        training_energy=training_energy,
        replicate_energy=replicate_energy,
        span_fraction=span_fraction,
        rms_quantiles=rms_quantiles,
        saturated_training=saturated_training,
        saturated_sampled=saturated_sampled,
        consistent=not (outside and beyond),
    )


def _memory_check(lags: Sequence[int], training: np.ndarray, replicates: np.ndarray) -> MemoryCheck:
    """Judge each lag: outside the replicates' envelope and beyond the tolerance fails."""
    band = np.stack([replicates.min(axis=0), replicates.max(axis=0)])
    replicate_mean = replicates.mean(axis=0)
    outside = (training < band[0]) | (training > band[1])
    beyond = np.abs(training - replicate_mean) > ACF_TOLERANCE
    return MemoryCheck(
        lags=np.asarray(lags, dtype=np.int64),
        training=training,
        band=band,
        replicate_mean=replicate_mean,
        consistent=~(outside & beyond),
    )


def _time_of_day_check(share: float, null_mean: float, null_max: float) -> TimeOfDayCheck:
    """Judge the daily cycle: above the permutation maximum and the tolerance fails."""
    return TimeOfDayCheck(
        share=share,
        null_mean=null_mean,
        null_max=null_max,
        consistent=not (share > null_max and share - null_mean > TIME_OF_DAY_TOLERANCE),
    )


def fidelity_summary(diagnostics: FidelityDiagnostics) -> str:
    """Human-readable report of one group's fidelity stage, for logs, errors and reports.

    (see docs/model_design.md §5.6.3)

    Parameters
    ----------
    diagnostics : FidelityDiagnostics

    Returns
    -------
    str
        Multi-line summary.
    """
    d = diagnostics
    hours = d.dt / 3600.0
    verdict = "passed" if d.passed else "FAILED"
    lines = [
        f"group {', '.join(d.members)}: generative fidelity {verdict} "
        f"({d.n_replicates} replicates)"
    ]
    for m in d.members:
        v, mem, tod = d.variance[m], d.memory[m], d.time_of_day[m]
        rms = " / ".join(f"{q:.2f}" for q in v.rms_quantiles)
        lines += [
            f"  {m}",
            f"    variance     sampled/training energy {v.ratio:.2f} "
            f"(replicates {v.replicate_energy.min() / v.training_energy:.2f}"
            f"-{v.replicate_energy.max() / v.training_energy:.2f})"
            f"  {'ok' if v.consistent else 'FAIL'}",
            f"                 DMD span carries {v.span_fraction:.1%} of the core energy",
            f"                 sampled per-coefficient RMS at p5/p50/p95/p99.9: {rms}",
            f"                 beyond |z| = {SATURATION_Z:.2f}: sampled {v.saturated_sampled:.2%}, "
            f"training {v.saturated_training:.2%}",
            "    memory       lag     training   replicates [min, max]   ok",
        ]
        for i, lag in enumerate(mem.lags):
            lines.append(
                f"                 {lag * hours:>5g} h  {mem.training[i]:.3f}      "
                f"[{mem.band[0, i]:.3f}, {mem.band[1, i]:.3f}]        "
                f"{'yes' if mem.consistent[i] else 'NO'}"
            )
        lines.append(
            f"    time of day  share {tod.share:.3f} (permutations: mean {tod.null_mean:.3f}, "
            f"max {tod.null_max:.3f})  {'ok' if tod.consistent else 'FAIL'}"
        )
    return "\n".join(lines)
