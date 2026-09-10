"""Per-cell empirical-CDF Gaussian copula (see docs/model_design.md §3.5, §5.4)."""

import warnings
from typing import Final, Self

import numpy as np
import xarray as xr
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from core.config import check_canonical

_EPS: Final[float] = 1e-6


def _searchsorted_per_column(sorted_matrix: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Vectorized per-column ``searchsorted``, avoiding a Python loop over cells.

    (see docs/model_design.md §5.4, §9 gotcha 10 on avoiding per-cell loops)

    Each column of `sorted_matrix` is an independent ascending reference
    array (unrelated to other columns' value ranges). Adding a per-column
    offset large enough to separate columns' numeric ranges lets a single
    global `np.searchsorted` stand in for ``n_cells`` independent ones.

    Parameters
    ----------
    sorted_matrix : (n_bins, n_cells)
        Each column ascending. An all-NaN column (e.g. a land cell) is
        given a harmless monotonic filler so it cannot corrupt the global
        sort; its results are meaningless and expected to be masked by the
        caller using the original NaN pattern.
    queries : (n_times, n_cells)

    Returns
    -------
    (n_times, n_cells) int, per-column ``searchsorted(..., side="left")`` index
    in ``[0, n_bins]``.
    """
    n_bins, n_cells = sorted_matrix.shape
    n_times = queries.shape[0]

    finite_matrix = np.where(np.isfinite(sorted_matrix), sorted_matrix, 0.0)
    finite_queries = np.where(np.isfinite(queries), queries, 0.0)
    combined_min = min(finite_matrix.min(), finite_queries.min())
    combined_max = max(finite_matrix.max(), finite_queries.max())
    span = combined_max - combined_min + 1.0

    safe_matrix = sorted_matrix.copy()
    nan_columns = np.all(np.isnan(sorted_matrix), axis=0)
    if nan_columns.any():
        filler = np.linspace(combined_min, combined_max, n_bins)
        safe_matrix[:, nan_columns] = filler[:, None]

    offsets = np.arange(n_cells, dtype=np.float64) * span
    flat_reference = (safe_matrix + offsets[None, :]).ravel(order="F")
    flat_queries = (np.nan_to_num(queries, nan=combined_min) + offsets[None, :]).ravel(order="F")

    global_idx: np.ndarray = np.searchsorted(flat_reference, flat_queries)
    column_ids = np.repeat(np.arange(n_cells), n_times)
    local_idx = global_idx - column_ids * n_bins
    result: np.ndarray = local_idx.reshape(n_cells, n_times).T
    return result


def _value_to_uniform(
    values: np.ndarray, bin_values: np.ndarray, bin_probs: np.ndarray
) -> np.ndarray:
    """Invert a per-cell quantile LUT: map raw values to their approximate probability.

    (see docs/model_design.md §5.4)

    Parameters
    ----------
    values : (n_times, n_cells)
    bin_values : (n_bins, n_cells)
        Per-cell ascending quantile values.
    bin_probs : (n_bins,)
        Shared probability at each bin.

    Returns
    -------
    (n_times, n_cells) float64 in ``(0, 1)``; NaN where `values` or the
    cell's LUT is NaN.
    """
    n_bins = bin_values.shape[0]
    idx = _searchsorted_per_column(bin_values, values)
    idx_lower = np.clip(idx - 1, 0, n_bins - 2)
    idx_upper = idx_lower + 1

    lower_val = np.take_along_axis(bin_values, idx_lower, axis=0)
    upper_val = np.take_along_axis(bin_values, idx_upper, axis=0)
    lower_prob = bin_probs[idx_lower]
    upper_prob = bin_probs[idx_upper]

    span = upper_val - lower_val
    frac = np.where(span > 0, (values - lower_val) / np.where(span > 0, span, 1.0), 0.0)
    probs = lower_prob + frac * (upper_prob - lower_prob)
    return np.where(np.isnan(values) | np.isnan(lower_val) | np.isnan(upper_val), np.nan, probs)


def _uniform_to_value(
    probs: np.ndarray, bin_values: np.ndarray, bin_probs: np.ndarray
) -> np.ndarray:
    """Look up a per-cell quantile LUT: map a probability back to a raw value.

    (see docs/model_design.md §5.4)

    Parameters
    ----------
    probs : (n_times, n_cells)
    bin_values : (n_bins, n_cells)
        Per-cell ascending quantile values.
    bin_probs : (n_bins,)
        Shared probability at each bin (identical across cells, so this
        direction needs no per-column search).

    Returns
    -------
    (n_times, n_cells) float64; NaN where `probs` or the cell's LUT is NaN.
    """
    n_bins = bin_values.shape[0]
    idx = np.searchsorted(bin_probs, probs.ravel()).reshape(probs.shape)
    idx_lower = np.clip(idx - 1, 0, n_bins - 2)
    idx_upper = idx_lower + 1

    lower_val = np.take_along_axis(bin_values, idx_lower, axis=0)
    upper_val = np.take_along_axis(bin_values, idx_upper, axis=0)
    lower_prob = bin_probs[idx_lower]
    upper_prob = bin_probs[idx_upper]

    span = upper_prob - lower_prob
    frac = (probs - lower_prob) / np.where(span > 0, span, 1.0)
    values = lower_val + frac * (upper_val - lower_val)
    return np.where(np.isnan(probs) | np.isnan(lower_val) | np.isnan(upper_val), np.nan, values)


class GaussianCopula(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Per-cell empirical-CDF Gaussian copula (see docs/model_design.md §3.5, §5.4).

    Forward (`transform`): rank each cell's value against its fitted
    quantile-binned empirical CDF, then apply Φ⁻¹ to reach a standard
    normal. Inverse (`inverse_transform`): apply Φ, then look the
    resulting uniform value up in the same binned CDF. This handles
    positivity/boundedness implicitly (inversion never produces values
    outside the training range) and preserves each cell's marginal shape.

    Known limitation (§3.5): a Gaussian copula has zero joint tail
    dependence, so simulated storm co-occurrences across cells will be
    too rare and too mild. Documented, not patched here — see the
    escalation path in §3.4.
    """

    def __init__(self, n_bins: int = 500) -> None:
        self.n_bins = n_bins

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        """Fit a per-cell quantile-binned empirical CDF for every variable.

        Parameters
        ----------
        X : xr.Dataset
            Standardized anomalies, e.g. the output of `train.standardize.
            PerCellStandardizer.transform`.
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self
        """
        check_canonical(X, context="GaussianCopula.fit")
        self.bin_probs_ = (np.arange(self.n_bins, dtype=np.float64) + 0.5) / self.n_bins
        self.sorted_anomalies_binned_: dict[str, np.ndarray] = {}
        for name, da in X.data_vars.items():
            variable = str(name)
            flat = da.values.reshape(da.shape[0], -1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                binned = np.nanquantile(flat, self.bin_probs_, axis=0)
            self.sorted_anomalies_binned_[variable] = binned.reshape(
                self.n_bins, *da.shape[1:]
            ).astype(np.float32)
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Rank-transform each variable to a standard normal via its fitted LUT.

        Parameters
        ----------
        X : xr.Dataset
            Standardized anomalies sharing `fit`'s variables.

        Returns
        -------
        xr.Dataset
            `X` with each variable replaced by its Gaussianized value.
        """
        check_is_fitted(self, "sorted_anomalies_binned_")
        check_canonical(X, context="GaussianCopula.transform")
        result = X.copy()
        for variable, binned in self.sorted_anomalies_binned_.items():
            if variable not in result.data_vars:
                continue
            da = result[variable]
            values = da.values.reshape(da.shape[0], -1).astype(np.float64)
            binned_flat = binned.reshape(self.n_bins, -1).astype(np.float64)
            probs = _value_to_uniform(values, binned_flat, self.bin_probs_)
            probs = np.clip(probs, _EPS, 1.0 - _EPS)
            z = norm.ppf(probs)
            result[variable] = (da.dims, z.reshape(da.shape))
        return result

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        """Invert `transform`: apply Φ, then look up the binned empirical CDF.

        Parameters
        ----------
        X : xr.Dataset
            Gaussianized values produced by `transform` (or by DMD/OU sampling).

        Returns
        -------
        xr.Dataset
            `X` with each variable replaced by its physical-anomaly value.
        """
        check_is_fitted(self, "sorted_anomalies_binned_")
        check_canonical(X, context="GaussianCopula.inverse_transform")
        result = X.copy()
        for variable, binned in self.sorted_anomalies_binned_.items():
            if variable not in result.data_vars:
                continue
            da = result[variable]
            z = da.values.reshape(da.shape[0], -1).astype(np.float64)
            probs = norm.cdf(z)
            binned_flat = binned.reshape(self.n_bins, -1).astype(np.float64)
            values = _uniform_to_value(probs, binned_flat, self.bin_probs_)
            result[variable] = (da.dims, values.reshape(da.shape))
        return result
