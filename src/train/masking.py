"""Land-mask derivation, partition NaN sentinel policy and log transforms for
wave-grid variables (see docs/model_design.md §5.1, §5.3.1;
docs/variable_taxonomy.md §3.3.1, §3.4; tasks/17-partition-nan-sentinel-policy.md).

Array-level helpers (`land_mask_from_sigwh`, `apply_partition_sentinel_policy`,
`height_log_energy`, `period_sentinel_from_archive`, diagnostics) plus the two
preprocessing transformers built on them, `PartitionSentinelFiller` and
`LogTransform`. Task 17 validated the policy (`docs/open_questions.md` §5.6),
which is what promoted it from notebook scope into the v3 pipeline.

All field arrays here are canonical `(time, lat, lon)` -- time is axis 0,
matching `core.config.CANONICAL_DIM_TEMPLATE` and everything these helpers
compose with (`train.ingest.load_training_window`'s output, the
`ClimatologyRemover`/`PerCellStandardizer` chain, and `train.tucker`, which
decomposes `modes=[1, 2]` and leaves time full-rank). Derived *spatial* masks
stay `(lat, lon)`.
"""

from dataclasses import dataclass
from typing import Final, Self

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from core.config import check_canonical

LAND_MASK_VARIABLE: Final[str] = "sigwh"


@dataclass(frozen=True, slots=True)
class SentinelMaskResult:
    """Result of `apply_partition_sentinel_policy` (see module docstring)."""

    filled: np.ndarray  # (time, lat, lon), every NaN -> 0.0 (land) or sentinel_value (ocean)
    tucker_mask: np.ndarray  # (time, lat, lon) float32 {0,1}: 0=land, 1=ocean(incl. sentinel)
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
    sigwh : (time, lat, lon) float
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
    result: np.ndarray = valid.mean(axis=0) >= valid_frac
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
    X : (time, lat, lon) float
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
    ocean_3d = np.broadcast_to(land_mask[None, :, :], X.shape)

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
    X : (time, lat, lon) float
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
    ocean_valid = valid[:, land_mask]  # (time, n_ocean_cells)

    nan_fraction = float(1.0 - ocean_valid.mean())
    per_cell_valid_frac = ocean_valid.mean(axis=0)
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
    primary : (time, lat, lon) float
    companion : (time, lat, lon) float
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

    mask_primary_nan_ocean = primary_nan & land_mask[None, :, :]
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
        p_nan = primary_nan[:, i, j]
        c = companion[:, i, j]
        c_finite = ~companion_nan[:, i, j]
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


def _fill_ocean_nan(values: np.ndarray, ocean_mask: np.ndarray, fill: float) -> np.ndarray:
    """Replace NaN at ocean cells with `fill`; land NaN stays NaN."""
    result: np.ndarray = np.where(np.isnan(values) & ocean_mask[None, :, :], fill, values)
    return result


class PartitionSentinelFiller(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Fill partition-absence NaN at ocean cells; leave land NaN (see model_design.md §5.3.1).

    The wind-sea and primary-swell partitions are NaN at genuine ocean cells
    whenever that partition is absent (variable_taxonomy.md §3.3.1). This
    stage turns those into valid observations -- height ``0.0``, period the
    partition's zero-height limit, direction a ``0°`` placeholder -- and
    leaves land NaN untouched for the later NaN-aware stages and the
    Tucker mask. It runs first in the Pipeline, so nothing downstream ever
    sees partition-absence NaN.

    The land mask comes from `sigwh` alone, never from a partition's own
    NaN (that conflation is the bug task 17 fixed), so `sigwh` must be in
    the fitted Dataset. The direction placeholder is harmless: with zero
    height, ``(u, v) = (0, 0)`` whatever the angle (§5.3).
    """

    def __init__(self, partitions: dict[str, tuple[str, str]], floor_m: float = 0.01) -> None:
        # partitions: height variable -> (period variable, direction variable).
        # floor_m: heights at or below this decode back to 0.0 on inversion
        # (must match LogTransform's floor). sklearn convention: verbatim.
        self.partitions = partitions
        self.floor_m = floor_m

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        """Derive the land mask from `sigwh` and each partition's period sentinel.

        The period sentinel is `period_sentinel_from_archive`'s
        ``extrapolated_intercept`` at its default ``small_height_quantile``:
        task 17 chose it because it stabilizes as the faint-partition bin
        widens while the conditional median drifts toward the archive median
        (tasks/17-partition-nan-sentinel-policy.md, ``T_SENTINEL_S``). Both
        estimates and their agreement are kept for inspection.

        Parameters
        ----------
        X : xr.Dataset
            Raw training window containing `sigwh` and every partition's
            height, period and direction.
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `sigwh` or a partition variable is missing from `X`.
        """
        check_canonical(X, context="PartitionSentinelFiller.fit")
        required = {LAND_MASK_VARIABLE}
        for height, (period, direction) in self.partitions.items():
            required |= {height, period, direction}
        missing = sorted(required - {str(name) for name in X.data_vars})
        if missing:
            raise ValueError(
                f"PartitionSentinelFiller needs {missing} (the land mask comes from "
                f"{LAND_MASK_VARIABLE!r}, docs/model_design.md §5.1)"
            )

        self.ocean_mask_ = land_mask_from_sigwh(X[LAND_MASK_VARIABLE].values)
        self.period_sentinel_estimates_: dict[str, PeriodSentinelEstimate] = {}
        self.period_sentinels_: dict[str, float] = {}
        for height, (period, _direction) in self.partitions.items():
            ocean = np.broadcast_to(self.ocean_mask_[None, :, :], X[height].shape)
            estimate = period_sentinel_from_archive(
                X[height].values[ocean], X[period].values[ocean]
            )
            self.period_sentinel_estimates_[period] = estimate
            self.period_sentinels_[period] = estimate.extrapolated_intercept
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Fill each partition's ocean-cell NaN; land NaN passes through.

        Parameters
        ----------
        X : xr.Dataset

        Returns
        -------
        xr.Dataset
            `X` with partition heights, periods and directions finite at
            every ocean cell.
        """
        check_is_fitted(self, "ocean_mask_")
        check_canonical(X, context="PartitionSentinelFiller.transform")
        result = X.copy(deep=False)
        for height, (period, direction) in self.partitions.items():
            fills = {height: 0.0, period: self.period_sentinels_[period], direction: 0.0}
            for name, fill in fills.items():
                da = result[name]
                result[name] = (da.dims, _fill_ocean_nan(da.values, self.ocean_mask_, fill))
        return result

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        """Decode heights at or below `floor_m` back to ``0.0`` (partition absent).

        `LogTransform` floors every height at `floor_m`, so a sentinel
        ``0.0`` comes back from its inverse as ``floor_m`` up to rounding in
        ``exp(log(floor_m**2) / 2)``; the relative tolerance absorbs that.
        Periods and directions pass through unchanged: where a partition is
        absent they are physically undefined, and there is no inverse rule
        for them yet (model_design.md §5.8, "Open in v3").

        Parameters
        ----------
        X : xr.Dataset

        Returns
        -------
        xr.Dataset
        """
        check_is_fitted(self, "ocean_mask_")
        result = X.copy(deep=False)
        threshold = self.floor_m * (1.0 + 1e-6)
        for height in self.partitions:
            if height not in result.data_vars:
                continue
            da = result[height]
            values = da.values
            result[height] = (da.dims, np.where(values <= threshold, 0.0, values))
        return result


class LogTransform(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Log-energy heights and log periods (see model_design.md §5.3.1).

    ``logE = log(max(H, floor_m)**2)`` (`height_log_energy`) and
    ``T = log(period)``, each renamed to its tensor name
    (variable_taxonomy.md §3.4, §5). The floor applies everywhere, so a real
    dead-calm ``H = 0`` stays finite as well as the sentinel zeros. NaN
    (land) propagates through both directions.
    """

    def __init__(
        self,
        heights: dict[str, str],
        periods: dict[str, str],
        floor_m: float = 0.01,
    ) -> None:
        # heights / periods: source variable -> tensor name, e.g.
        # {"wsh": "logE_wsh"} / {"wsp": "T_wsp"}. sklearn convention: verbatim.
        self.heights = heights
        self.periods = periods
        self.floor_m = floor_m

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        """Record the source -> tensor name maps.

        Parameters
        ----------
        X : xr.Dataset
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If a listed source variable is missing from `X`.
        """
        missing = sorted(
            (set(self.heights) | set(self.periods)) - {str(name) for name in X.data_vars}
        )
        if missing:
            raise ValueError(f"LogTransform: variables {missing} not in the Dataset")
        self.heights_ = dict(self.heights)
        self.periods_ = dict(self.periods)
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Replace each height with its ``logE`` tensor and each period with ``T``.

        Parameters
        ----------
        X : xr.Dataset

        Returns
        -------
        xr.Dataset
        """
        check_is_fitted(self, "heights_")
        result = X.copy(deep=False)
        for source, tensor in self.heights_.items():
            da = result[source]
            result = result.drop_vars(source)
            result[tensor] = (da.dims, height_log_energy(da.values, floor_m=self.floor_m))
        for source, tensor in self.periods_.items():
            da = result[source]
            result = result.drop_vars(source)
            result[tensor] = (da.dims, np.log(da.values))
        return result

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        """Map ``logE`` back to height (``exp(logE / 2)``) and ``T`` back to period.

        Parameters
        ----------
        X : xr.Dataset

        Returns
        -------
        xr.Dataset
        """
        check_is_fitted(self, "heights_")
        result = X.copy(deep=False)
        for source, tensor in self.heights_.items():
            if tensor not in result.data_vars:
                continue
            da = result[tensor]
            result = result.drop_vars(tensor)
            result[source] = (da.dims, np.exp(da.values / 2.0))
        for source, tensor in self.periods_.items():
            if tensor not in result.data_vars:
                continue
            da = result[tensor]
            result = result.drop_vars(tensor)
            result[source] = (da.dims, np.exp(da.values))
        return result
