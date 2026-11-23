"""Remove per-cell harmonic climatology with linear trend (see docs/model_design.md §3.9, §5.4)."""

from datetime import datetime, timezone
from typing import Final, Self

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from core.config import check_canonical

# Default diurnal-harmonic variables when `diurnal_variables` is unset (see §5.4).
# Waves have no diurnal cycle at 0.25° global resolution, so wave-grid scalars
# are deliberately excluded.
_DEFAULT_DIURNAL_VARIABLES: Final[tuple[str, ...]] = ("sst", "u_wind", "v_wind")

_OMEGA_YEARLY: Final[float] = 2.0 * np.pi / 365.25  # rad/day
_OMEGA_DIURNAL: Final[float] = 2.0 * np.pi  # rad/day (one full cycle per day)
_DAYS_PER_YEAR: Final[float] = 365.25


def _harmonic_design_matrix(
    t_days: np.ndarray, include_diurnal: bool, delta_tau_years: np.ndarray | None = None
) -> np.ndarray:
    """Build the harmonic (+ optional linear trend) regression design matrix (see §3.9, §5.4).

    Columns: intercept, [linear trend], annual cos/sin, semi-annual cos/sin,
    and (if `include_diurnal`) diurnal cos/sin. The trend column, when
    present, is inserted immediately after the intercept and uses
    `delta_tau_years` — a *separate* time origin from `t_days` (see
    `ClimatologyRemover.fit` for why `t_ref_` must differ from
    `t_train_start_`).

    Parameters
    ----------
    t_days : np.ndarray
        Continuous days since the harmonic time origin (`t_train_start_`).
    include_diurnal : bool
        Whether to append the diurnal cos/sin pair.
    delta_tau_years : np.ndarray | None
        Year-normalized time since `t_ref_`, i.e. `(t - t_ref) / 365.25 days`.
        If `None`, the trend column is omitted entirely.
    """
    columns = [np.ones_like(t_days)]
    if delta_tau_years is not None:
        columns.append(delta_tau_years)
    columns.extend(
        [
            np.cos(_OMEGA_YEARLY * t_days),
            np.sin(_OMEGA_YEARLY * t_days),
            np.cos(2.0 * _OMEGA_YEARLY * t_days),
            np.sin(2.0 * _OMEGA_YEARLY * t_days),
        ]
    )
    if include_diurnal:
        columns.append(np.cos(_OMEGA_DIURNAL * t_days))
        columns.append(np.sin(_OMEGA_DIURNAL * t_days))
    return np.stack(columns, axis=1)  # (T, n_terms)


def _days_since_epoch(times: np.ndarray, epoch: datetime) -> np.ndarray:
    """Convert UTC timestamps to continuous days since `epoch` (see §5.4).

    Continuous, not reset at year boundaries — periodicity comes from the
    harmonic cosines, not from day-of-year bookkeeping.
    """
    epoch64 = np.datetime64(epoch.replace(tzinfo=None), "s")
    seconds = (times.astype("datetime64[s]") - epoch64).astype("timedelta64[s]").astype(np.float64)
    return seconds / 86400.0


class ClimatologyRemover(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Remove a per-cell harmonic seasonal (and optionally diurnal/trend) climatology.

    (see docs/model_design.md §3.9, §5.4)

    Fits, per grid cell and per variable, an OLS regression of intercept +
    optional linear trend + annual/semi-annual harmonics + optional diurnal
    harmonics against a continuous days-since-epoch time axis, then
    subtracts it in `transform` and adds it back in `inverse_transform`.
    Cells that are NaN at every training timestep (land) receive NaN
    coefficients, which is harmless: they are excluded later by the ocean
    mask.

    The harmonic terms and the linear trend term deliberately use **two
    different time origins**. Harmonic terms use `t_train_start_` (the
    first training timestamp) via `_days_since_epoch` — this is an
    arbitrary-but-fixed origin, since harmonic coefficients are unaffected
    by the choice of origin. The trend term instead uses `t_ref_`, the
    training window's *midpoint*, because `Δτ(t) = (t − t_ref) / 1 year`
    only decorrelates the recovered `c_lin(s)` from the intercept `c₀(s)`
    when `t_ref` is centered in the training window (§3.9). Reusing
    `t_train_start_` for the trend column would recorrelate `c_lin` with
    `c₀`, defeating the point of adding a separate trend term.
    """

    def __init__(
        self,
        diurnal_variables: list[str] | None = None,
        include_trend: bool = True,
        trend_significance_threshold: float = 0.1,
    ) -> None:
        self.diurnal_variables = diurnal_variables
        self.include_trend = include_trend
        self.trend_significance_threshold = trend_significance_threshold

    def _resolve_diurnal_variables(self) -> tuple[str, ...]:
        if self.diurnal_variables is None:
            return _DEFAULT_DIURNAL_VARIABLES
        return tuple(self.diurnal_variables)

    def _delta_tau_years(self, times: np.ndarray) -> np.ndarray | None:
        """Year-normalized time since `t_ref_`, or `None` if trend is disabled."""
        if not self.include_trend:
            return None
        return _days_since_epoch(times, self.t_ref_) / _DAYS_PER_YEAR

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        """Fit per-cell harmonic (+ optional linear trend) coefficients for every variable in `X`.

        Parameters
        ----------
        X : xr.Dataset
            Training data, dims ``(time, <grid>_lat, <grid>_lon)`` per variable.
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self
        """
        check_canonical(X, context="ClimatologyRemover.fit")
        first_time = X["time"].values[0]
        last_time = X["time"].values[-1]
        self.t_train_start_ = first_time.astype("datetime64[s]").item().replace(tzinfo=timezone.utc)
        t_train_end = last_time.astype("datetime64[s]").item().replace(tzinfo=timezone.utc)
        # Second, distinct time origin (§3.9): the trend term centers on the
        # training window's midpoint, not its start, to decorrelate c_lin
        # from c0. The harmonic terms above are unaffected and keep using
        # t_train_start_.
        self.t_ref_: datetime = self.t_train_start_ + (t_train_end - self.t_train_start_) / 2

        t_days = _days_since_epoch(X["time"].values, self.t_train_start_)
        delta_tau_years = self._delta_tau_years(X["time"].values)
        diurnal_variables = self._resolve_diurnal_variables()

        self.coeffs_: dict[str, np.ndarray] = {}
        self.c_lin_: dict[str, np.ndarray] = {}
        for name, da in X.data_vars.items():
            variable = str(name)
            design = _harmonic_design_matrix(t_days, variable in diurnal_variables, delta_tau_years)
            spatial_shape = da.shape[1:]
            y_flat = da.values.reshape(da.shape[0], -1)

            valid_rows = ~np.all(np.isnan(y_flat), axis=1)
            coeffs_flat = np.full((design.shape[1], y_flat.shape[1]), np.nan, dtype=np.float64)
            if valid_rows.any():
                fitted, *_ = np.linalg.lstsq(design[valid_rows], y_flat[valid_rows], rcond=None)
                coeffs_flat = fitted
            self.coeffs_[variable] = coeffs_flat.reshape(design.shape[1], *spatial_shape)
            if self.include_trend:
                # Trend column sits immediately after the intercept (index 1).
                self.c_lin_[variable] = self.coeffs_[variable][1].copy()
        return self

    def _climatology(self, X: xr.Dataset) -> dict[str, np.ndarray]:
        check_is_fitted(self, "coeffs_")
        t_days = _days_since_epoch(X["time"].values, self.t_train_start_)
        delta_tau_years = self._delta_tau_years(X["time"].values)
        diurnal_variables = self._resolve_diurnal_variables()
        climatology: dict[str, np.ndarray] = {}
        for variable, coeffs in self.coeffs_.items():
            if variable not in X.data_vars:
                continue
            design = _harmonic_design_matrix(t_days, variable in diurnal_variables, delta_tau_years)
            climatology[variable] = np.einsum("tc,c...->t...", design, coeffs)
        return climatology

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Subtract each variable's fitted harmonic climatology.

        Parameters
        ----------
        X : xr.Dataset
            Data sharing `fit`'s variables, at the same or a different time window.

        Returns
        -------
        xr.Dataset
            `X` with each variable replaced by its anomaly (value minus climatology).
        """
        check_canonical(X, context="ClimatologyRemover.transform")
        result = X.copy()
        for variable, mu in self._climatology(X).items():
            result[variable] = result[variable] - mu
        return result

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        """Add each variable's fitted harmonic climatology back to its anomaly.

        Parameters
        ----------
        X : xr.Dataset
            Anomalies produced by `transform`.

        Returns
        -------
        xr.Dataset
            `X` with each variable's climatology added back.
        """
        check_canonical(X, context="ClimatologyRemover.inverse_transform")
        result = X.copy()
        for variable, mu in self._climatology(X).items():
            result[variable] = result[variable] + mu
        return result
