"""Divide each cell's anomaly by its temporal standard deviation (see docs/model_design.md §5.4)."""

import warnings
from typing import Self

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from core.config import check_canonical


class PerCellStandardizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Divide each cell's anomaly by its per-cell temporal standard deviation.

    (see docs/model_design.md §5.4)

    Runs after `train.climatology.ClimatologyRemover`, whose output
    (anomalies) is this transformer's input. NaN cells (land) get a NaN
    standard deviation and remain NaN through `transform`, consistent with
    the NaN discipline in §2.5: never coerced to zero.
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
                result[variable] = result[variable] / sigma
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
                result[variable] = result[variable] * sigma
        return result
