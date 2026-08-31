"""Land-mask derivation and per-cell NaN sentinel policy for wave-grid
variables (see docs/variable_taxonomy.md §3.3.1, tasks/17-partition-nan-
sentinel-policy.md).

Notebook/src-helper scope only per task 17 -- NOT wired into
src/weather/model.py's `_mask_and_stack` or src/train/ingest.py's existing
mask logic. Promotion to production is deliberately deferred pending this
policy's empirical validation (task 17 §4).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SentinelMaskResult:
    """Result of `apply_partition_sentinel_policy` (see module docstring)."""

    filled: np.ndarray  # (lat, lon, time), every NaN -> 0.0 (land) or sentinel_value (ocean)
    tucker_mask: np.ndarray  # (lat, lon, time) float32 {0,1}: 0=land, 1=ocean(incl. sentinel)
    n_ocean_nan: int  # diagnostic: ocean-cell/timestep NaNs zero-filled as valid


@dataclass(frozen=True, slots=True)
class PeriodSentinelEstimate:
    """Result of `period_sentinel_from_archive` (see its docstring)."""

    conditional_median: float  # (a): median wsp within the bottom-quantile wsh bin
    extrapolated_intercept: float  # (b): linear wsp-vs-wsh fit in that bin, evaluated at wsh=0
    agreement: float  # |a - b|; small = consistent limit, large = shape instability


def land_mask_from_sigwh(sigwh: np.ndarray, *, valid_frac: float = 1.0) -> np.ndarray:
    """Derive an authoritative wave-grid land mask from `sigwh`'s own NaN pattern.

    Parameters
    ----------
    sigwh : (lat, lon, time) float
        Total significant wave height. Unlike the partition triplets
        (wsh/wsp/wsd, etc.), `sigwh` is an aggregate quantity with no
        partition-absence NaN -- it is NaN only over land/ice
        (variable_taxonomy.md §3.1), making its own NaN pattern a usable,
        variable-independent land-mask proxy for the whole wave grid.
    valid_frac : float
        Minimum fraction of timesteps a cell must be non-NaN to count as
        ocean. Strict all-timesteps by default (`1.0`), matching
        production's existing convention (`src/weather/model.py`'s
        `_mask_and_stack`, `docs/tucker_decomposition.md` §4.2's canonical
        formula) -- NOT notebooks 00/07's 0.90 fractional tolerance, which
        is for per-variable reporting, not land-mask ground truth. `sigwh`
        should not need a tolerance to avoid collapsing to near-nothing; if
        it does, that is itself a finding to record (task 17 §4), not
        silently absorbed by loosening `valid_frac`.

    Returns
    -------
    (lat, lon) bool
        `True` = ocean.
    """
    valid = ~np.isnan(sigwh)
    result: np.ndarray = valid.mean(axis=-1) >= valid_frac
    return result


def apply_partition_sentinel_policy(
    X: np.ndarray, land_mask: np.ndarray, *, sentinel_value: float = 0.0
) -> SentinelMaskResult:
    """Fill NaN and build a Tucker mask that distinguishes land from partition-absence.

    Ocean-cell NaN -> filled with `sentinel_value` + marked VALID (a
    genuine "no partition" observation, per task 17's confirmed sentinel
    decision). Land-cell NaN -> zero-filled + marked INVALID (imputed by
    masked HOOI, unchanged from today's behavior).

    Parameters
    ----------
    X : (lat, lon, time) float
        A single variable's raw values.
    land_mask : (lat, lon) bool
        MUST come from an independent source (`land_mask_from_sigwh` or
        equivalent), never from `X`'s own NaN pattern -- that conflation is
        exactly the bug this function fixes.
    sentinel_value : float
        The value to fill at ocean-cell NaN. Height fields pass the
        default `0.0` (H=4*sqrt(m0) -> 0 is the clean, physically correct
        limit, variable_taxonomy.md §2.2). Period fields must pass
        `period_sentinel_from_archive(...)`'s chosen scalar instead -- no
        physical small-value limit exists for T_m01=m0/m1's 0/0
        indeterminate form.

    Returns
    -------
    SentinelMaskResult
    """
    nan_mask = np.isnan(X)
    ocean_3d = np.broadcast_to(land_mask[:, :, None], X.shape)

    filled = np.where(nan_mask & ocean_3d, sentinel_value, X)
    filled = np.where(nan_mask & ~ocean_3d, 0.0, filled)

    tucker_mask = ocean_3d.astype(np.float32)
    n_ocean_nan = int(np.sum(nan_mask & ocean_3d))

    return SentinelMaskResult(filled=filled, tucker_mask=tucker_mask, n_ocean_nan=n_ocean_nan)


def height_log_energy(H: np.ndarray, *, floor_m: float = 0.01) -> np.ndarray:
    """log E := log(max(H, floor_m)**2) -- a floor-before-log fix for
    `variable_taxonomy.md` §3.4's log-transform convention, which never
    specified an epsilon.

    Applies uniformly, not just at sentinel cells: any real `H=0` (a rare
    dead-calm observation) would hit the same `log(0) = -inf` problem this
    guards against.

    Parameters
    ----------
    H : array
        Physical wave height, sentinel-filled (see
        `apply_partition_sentinel_policy`) or real.
    floor_m : float
        Fixed physical constant (default `0.01`, 1 cm) -- negligible
        relative to real signal, not archive-derived (deliberately avoids
        a full-archive percentile study as a blocking prerequisite).

    Returns
    -------
    array, same shape as `H`, finite everywhere.
    """
    result: np.ndarray = np.log(np.maximum(H, floor_m) ** 2)
    return result


def ocean_nan_report(
    X: np.ndarray,
    land_mask: np.ndarray,
    valid_time_fracs: tuple[float, ...] = (1.0, 0.9, 0.7),
) -> dict[str, object]:
    """NaN fraction and per-threshold always-valid-cell fraction, computed
    only over `land_mask`'s ocean cells -- the reusable version of this
    session's disposable one-off scripts.

    Parameters
    ----------
    X : (lat, lon, time) float
    land_mask : (lat, lon) bool
        `True` = ocean. See `land_mask_from_sigwh`.
    valid_time_fracs : tuple[float, ...]
        Thresholds to report the always-valid-cell fraction at.

    Returns
    -------
    dict with keys `n_ocean_cells`, `ocean_nan_fraction`,
    `always_valid_cell_fraction` (dict keyed by threshold).
    """
    n_ocean_cells = int(land_mask.sum())
    if n_ocean_cells == 0:
        raise ValueError("land_mask has no ocean cells")

    valid = ~np.isnan(X)
    ocean_valid = valid[land_mask]  # (n_ocean_cells, time)

    nan_fraction = float(1.0 - ocean_valid.mean())
    per_cell_valid_frac = ocean_valid.mean(axis=-1)
    always_valid_frac = {f: float((per_cell_valid_frac >= f).mean()) for f in valid_time_fracs}

    return {
        "n_ocean_cells": n_ocean_cells,
        "ocean_nan_fraction": nan_fraction,
        "always_valid_cell_fraction": always_valid_frac,
    }


def partition_companion_backfill_check(
    primary: np.ndarray, companion: np.ndarray, land_mask: np.ndarray
) -> dict[str, object]:
    """Check whether `companion` (e.g. wsd) looks genuinely absent or
    stale/backfilled at ocean cells where `primary` (e.g. wsh) is NaN.

    At each ocean cell, compares `companion`'s variance during
    `primary`-NaN runs against its variance during `primary`-valid periods.
    A companion that stays suspiciously constant while `primary` is NaN
    (variance collapses relative to the valid-period baseline) suggests the
    archive backfills it with a stale value rather than leaving it
    genuinely undefined -- diagnostic only, per task 17 §1a; the pipeline
    itself doesn't branch on the answer.

    Parameters
    ----------
    primary : (lat, lon, time) float
    companion : (lat, lon, time) float
    land_mask : (lat, lon) bool
        `True` = ocean.

    Returns
    -------
    dict with keys `n_primary_nan_ocean`, `companion_also_nan_fraction`,
    `mean_companion_std_when_primary_nan`,
    `mean_companion_std_when_primary_valid`.
    """
    primary_nan = np.isnan(primary)
    companion_nan = np.isnan(companion)

    mask_primary_nan_ocean = primary_nan & land_mask[:, :, None]
    n_primary_nan = int(mask_primary_nan_ocean.sum())
    if n_primary_nan == 0:
        return {
            "n_primary_nan_ocean": 0,
            "companion_also_nan_fraction": float("nan"),
            "mean_companion_std_when_primary_nan": float("nan"),
            "mean_companion_std_when_primary_valid": float("nan"),
        }

    companion_also_nan_fraction = float(companion_nan[mask_primary_nan_ocean].mean())

    std_when_nan: list[float] = []
    std_when_valid: list[float] = []
    for i, j in np.argwhere(land_mask):
        p_nan = primary_nan[i, j]
        c = companion[i, j]
        c_finite = ~companion_nan[i, j]
        nan_vals = c[p_nan & c_finite]
        valid_vals = c[~p_nan & c_finite]
        if nan_vals.size >= 2:
            std_when_nan.append(float(np.std(nan_vals)))
        if valid_vals.size >= 2:
            std_when_valid.append(float(np.std(valid_vals)))

    return {
        "n_primary_nan_ocean": n_primary_nan,
        "companion_also_nan_fraction": companion_also_nan_fraction,
        "mean_companion_std_when_primary_nan": (
            float(np.mean(std_when_nan)) if std_when_nan else float("nan")
        ),
        "mean_companion_std_when_primary_valid": (
            float(np.mean(std_when_valid)) if std_when_valid else float("nan")
        ),
    }


def period_sentinel_from_archive(
    wsh_real: np.ndarray, wsp_real: np.ndarray, *, small_height_quantile: float = 0.05
) -> PeriodSentinelEstimate:
    """Estimate lim_{H->0+} T_m01(H) from real, non-sentinel paired
    (wsh, wsp) observations, restricted to the bottom `small_height_quantile`
    of nonzero `wsh` (the "partition present but faint" regime) -- NOT an
    unconditional archive median.

    Rationale (task 17 §1a): `T_m01 = m0/m1` is a 0/0 indeterminate form
    as a partition's energy vanishes, but not an arbitrary one. If a
    fading partition keeps a roughly fixed spectral shape while only its
    amplitude shrinks -- `m0(eps) = eps*M0`, `m1(eps) = eps*M1` for
    shape-only constants `M0, M1` as `eps -> 0+` -- then `T_m01 -> M0/M1`:
    a well-defined, shape-determined limit (an L'Hopital-style
    cancellation along the amplitude parameter), estimable from real
    near-vanishing partitions rather than typical-strength ones.

    Parameters
    ----------
    wsh_real, wsp_real : 1D float arrays
        Real (non-sentinel, non-NaN), paired height/period observations
        from a representative window.
    small_height_quantile : float
        Fraction of the smallest nonzero `wsh` values defining the
        "faint partition" regime the limit is estimated from.

    Returns
    -------
    PeriodSentinelEstimate
        `conditional_median` and `extrapolated_intercept` are two
        independent estimators of the same limit; a large `agreement`
        gap signals the fading-spectral-shape assumption may not hold and
        should be reported, not silently resolved.
    """
    finite = np.isfinite(wsh_real) & np.isfinite(wsp_real) & (wsh_real > 0)
    h = wsh_real[finite]
    p = wsp_real[finite]
    if h.size < 2:
        raise ValueError("need at least 2 finite, positive-height paired (wsh, wsp) observations")

    threshold = np.quantile(h, small_height_quantile)
    low = h <= threshold
    if low.sum() < 2:
        raise ValueError(
            f"small_height_quantile={small_height_quantile} bin has fewer than 2 "
            "observations; widen the quantile"
        )

    conditional_median = float(np.median(p[low]))
    # Center h before fitting: the low-height bin sits far from 0 in absolute
    # terms (e.g. h in [0.3, 1.0]), which makes an uncentered Vandermonde
    # matrix ill-conditioned for a degree-1 polyfit even with millions of
    # points -- a classic numpy gotcha, not evidence the fit is meaningless.
    h_mean = float(h[low].mean())
    slope, intercept_centered = np.polyfit(h[low] - h_mean, p[low], deg=1)
    extrapolated_intercept = float(intercept_centered - slope * h_mean)

    return PeriodSentinelEstimate(
        conditional_median=conditional_median,
        extrapolated_intercept=extrapolated_intercept,
        agreement=abs(conditional_median - extrapolated_intercept),
    )
