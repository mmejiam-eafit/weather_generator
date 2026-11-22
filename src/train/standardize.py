"""Divide each cell's anomaly by its temporal standard deviation (see docs/model_design.md §5.4)."""

import warnings
from typing import Final, Self

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from core.config import check_canonical

_SIGMA_FLOOR: Final[float] = 1e-6  # below this, treat the divisor as 1 (no rescale)


def _floored_divisor(sigma: np.ndarray) -> np.ndarray:
    """Sigma to actually divide/multiply by: 1.0 in place of a near-zero fitted value.

    A near-constant cell (e.g. intermittently ice-covered, so its anomaly is
    ~0 for almost the entire fit window) can get a `sigma_cell_` that's
    numerically near-zero; dividing by it blows a tiny, real anomaly up to
    an enormous one. Flooring the divisor to 1 instead passes that (already
    small) anomaly through unscaled -- no new information invented, and
    `inverse_transform` uses the same floored value so the round trip stays
    exact. NaN (land) is untouched: `nan < eps` is `False` in numpy, so the
    `else` branch keeps NaN.

    Deliberately NOT applied to `self.sigma_cell_` itself:
    `weather.model._log_trend_significance` consults the true fitted sigma
    to flag exactly these low-variance/high-trend cells, and flooring the
    stored attribute would defeat that diagnostic.
    """
    return np.where(sigma < _SIGMA_FLOOR, 1.0, sigma)


class PerCellStandardizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Divide each cell's anomaly by its per-cell temporal standard deviation.

    (see docs/model_design.md §5.4)

    Runs after `train.climatology.ClimatologyRemover`, whose output
    (anomalies) is this transformer's input. NaN cells (land) get a NaN
    standard deviation and remain NaN through `transform`, consistent with
    the NaN discipline in §2.5: never coerced to zero. A near-zero (but
    non-NaN) `sigma_cell_` is floored to 1 at the point of division/
    multiplication only -- see `_floored_divisor` -- so `sigma_cell_` itself
    always holds the true fitted value.
    """

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        """Compute each variable's per-cell temporal standard deviation.

        Parameters
        ----------
        X : xr.Dataset
            Anomalies, e.g. the output of `train.climatology.
            ClimatologyRemover.transform`.
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self
        """
        check_canonical(X, context="PerCellStandardizer.fit")
        # An all-NaN cell (land) has zero valid observations; np.nanstd warns
        # "Degrees of freedom <= 0" and correctly returns NaN. Expected, not a bug.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            self.sigma_cell_: dict[str, np.ndarray] = {
                str(name): np.nanstd(da.values, axis=0) for name, da in X.data_vars.items()
            }
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Divide each variable by its fitted per-cell standard deviation.

        Parameters
        ----------
        X : xr.Dataset
            Anomalies sharing `fit`'s variables.

        Returns
        -------
        xr.Dataset
            `X` with each variable divided by `sigma_cell_`.
        """
        check_is_fitted(self, "sigma_cell_")
        check_canonical(X, context="PerCellStandardizer.transform")
        result = X.copy()
        for variable, sigma in self.sigma_cell_.items():
            if variable in result.data_vars:
                result[variable] = result[variable] / _floored_divisor(sigma)
        return result

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        """Multiply each variable by its fitted per-cell standard deviation.

        Parameters
        ----------
        X : xr.Dataset
            Standardized anomalies produced by `transform`.

        Returns
        -------
        xr.Dataset
            `X` with each variable multiplied by `sigma_cell_`.
        """
        check_is_fitted(self, "sigma_cell_")
        check_canonical(X, context="PerCellStandardizer.inverse_transform")
        result = X.copy()
        for variable, sigma in self.sigma_cell_.items():
            if variable in result.data_vars:
                result[variable] = result[variable] * _floored_divisor(sigma)
        return result
