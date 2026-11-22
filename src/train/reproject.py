"""Reproject wind- and ocean-grid variables onto the wave grid.

(see docs/model_design.md §2.6, §3.6, §5.2)
"""

from collections.abc import Hashable
from typing import Self

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from core.config import grid_coordinates


def pad_lon(field: np.ndarray, pad: int = 2) -> np.ndarray:
    """Periodically pad the last axis (longitude) by `pad` columns on each side.

    (see docs/model_design.md §2.6)

    Both grids span ``[0°, 359.75° E]`` with no dateline duplication;
    ``col=1439`` and ``col=0`` are adjacent on the sphere but at opposite
    ends of the array. Any longitude interpolation must pad first or it
    produces NaN across that boundary.
    """
    return np.concatenate([field[..., -pad:], field, field[..., :pad]], axis=-1)


def lonlat_interpolator(
    field: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray
) -> RegularGridInterpolator:
    """Build a lon-wraparound-safe bilinear interpolator for a ``(batch, lat, lon)`` field.

    (see docs/model_design.md §2.6)

    Shared by `_bilinear_to_grid` (query at a regular target grid) and
    `weather.simulate` (query at scattered ``(lat, lon)`` points) — both
    need the same padding-then-`RegularGridInterpolator` setup. NaN at a
    source grid corner propagates naturally to any interpolated point
    whose stencil includes it (see §2.5 NaN discipline); no explicit
    masking is needed here.

    Parameters
    ----------
    field : (batch, n_src_lat, n_src_lon)
        E.g. a time series of 2D fields, or several variables' fields at
        one time — `batch` is just whatever leading axis should ride
        along unchanged.
    src_lats, src_lons : np.ndarray

    Returns
    -------
    RegularGridInterpolator
        Call with an ``(n_points, 2)`` array of ``(lat, lon)`` pairs;
        returns ``(n_points, batch)``.
    """
    padded = pad_lon(field, pad=2)
    values = np.moveaxis(padded, 0, -1)  # (n_src_lat, n_src_lon + 4, batch)
    src_lons_padded = np.concatenate([src_lons[-2:] - 360.0, src_lons, src_lons[:2] + 360.0])

    return RegularGridInterpolator(
        (src_lats, src_lons_padded),
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _bilinear_to_grid(
    field: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a ``(time, lat, lon)`` field onto a new lat/lon grid.

    (see docs/model_design.md §2.6, §5.2)
    """
    interpolator = lonlat_interpolator(field, src_lats, src_lons)

    target_lat_grid, target_lon_grid = np.meshgrid(target_lats, target_lons, indexing="ij")
    query_points = np.stack([target_lat_grid.ravel(), target_lon_grid.ravel()], axis=-1)
    interpolated = interpolator(query_points)  # (n_target_lat * n_target_lon, n_times)
    interpolated = interpolated.reshape(*target_lat_grid.shape, -1)
    return np.moveaxis(interpolated, -1, 0)  # (n_times, n_target_lat, n_target_lon)


def _same_coordinates(source: np.ndarray, target: np.ndarray) -> bool:
    """True when two 1D coordinate arrays describe the same grid lines (to 1e-9 degrees)."""
    return source.shape == target.shape and bool(np.allclose(source, target, rtol=0.0, atol=1e-9))


class Reprojector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Reproject non-target-grid variables onto the target grid via bilinear interpolation.

    (see docs/model_design.md §2.6, §3.6, §5.2)

    Stateless beyond recording the target grid's coordinates: nothing is
    learned from data, but this is kept as a transformer so it composes
    into the preprocessing Pipeline. Must run *after*
    `train.directional.ComingFromToUV` / `train.directional.GoingToToUV`
    so that directional fields are interpolated as ``(u, v)``, not as
    angles (see §9 gotcha 4).

    `inverse_transform` is the identity, not a true inverse: reprojection
    is lossy (information at the source grid's native resolution cannot
    be recovered), and nothing downstream needs it recovered — sampling
    reconstructs fields at the target grid's resolution directly (see
    §4). The identity exists so `sklearn.pipeline.Pipeline.
    inverse_transform` is available at all: sklearn only exposes it when
    *every* step implements it, even as a no-op.
    """

    def __init__(self, target_grid: str = "wave") -> None:
        self.target_grid = target_grid

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        """Record the target grid's coordinates.

        Uses `X`'s own ``<target_grid>_lat``/``<target_grid>_lon``
        coordinates if present (so a small test patch reprojects onto the
        same small patch), falling back to the canonical `core.config`
        grid otherwise.

        Parameters
        ----------
        X : xr.Dataset
            Training data, e.g. the output of `train.directional.
            ComingFromToUV.transform` / `train.directional.GoingToToUV.transform`.
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self
        """
        lat_dim = f"{self.target_grid}_lat"
        lon_dim = f"{self.target_grid}_lon"
        if lat_dim in X.coords and lon_dim in X.coords:
            self.target_lats_ = X.coords[lat_dim].values.astype(np.float64)
            self.target_lons_ = X.coords[lon_dim].values.astype(np.float64)
        else:
            self.target_lats_, self.target_lons_ = grid_coordinates(self.target_grid)
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Reproject every variable not already on the target grid.

        Parameters
        ----------
        X : xr.Dataset
            Data with each variable on its native grid.

        Returns
        -------
        xr.Dataset
            All variables on the target grid's ``(time, <target_grid>_lat,
            <target_grid>_lon)`` coordinates.
        """
        check_is_fitted(self, "target_lats_")
        target_lat_dim = f"{self.target_grid}_lat"
        target_lon_dim = f"{self.target_grid}_lon"

        reprojected_vars: dict[str, tuple[tuple[Hashable, ...], np.ndarray]] = {}
        for name, da in X.data_vars.items():
            lat_dim = next((str(d) for d in da.dims if str(d).endswith("_lat")), None)
            lon_dim = next((str(d) for d in da.dims if str(d).endswith("_lon")), None)
            if lat_dim is None or lon_dim is None:
                reprojected_vars[str(name)] = (da.dims, da.values)
                continue
            if lat_dim == target_lat_dim and lon_dim == target_lon_dim:
                reprojected_vars[str(name)] = (da.dims, da.values)
                continue

            src_lats = da.coords[lat_dim].values.astype(np.float64)
            src_lons = da.coords[lon_dim].values.astype(np.float64)
            if _same_coordinates(src_lats, self.target_lats_) and _same_coordinates(
                src_lons, self.target_lons_
            ):
                # Wind shares the wave grid under its own dim names (§2.2, §5.2).
                # Interpolating it anyway is not a no-op: scipy's linear stencil
                # carries a zero-weight neighbour, so one land NaN would spread
                # to the ocean cells south and west of it.
                reprojected_vars[str(name)] = (("time", target_lat_dim, target_lon_dim), da.values)
                continue
            values = _bilinear_to_grid(
                da.values, src_lats, src_lons, self.target_lats_, self.target_lons_
            )
            reprojected_vars[str(name)] = (("time", target_lat_dim, target_lon_dim), values)

        return xr.Dataset(
            reprojected_vars,
            coords={
                "time": X.coords["time"].values,
                target_lat_dim: self.target_lats_,
                target_lon_dim: self.target_lons_,
            },
        )

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        """Identity — see the class docstring for why this isn't a true inverse."""
        return X
