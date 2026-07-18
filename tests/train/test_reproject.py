"""Tests for train.reproject (see docs/model_design.md §2.6, §3.6, §5.2)."""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose
from sklearn.base import clone

from train.reproject import Reprojector, _bilinear_to_grid, pad_lon


def test_pad_lon_wraps_periodically() -> None:
    """Padding prepends the last columns and appends the first columns (§2.6)."""
    field = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    padded = pad_lon(field, pad=2)
    assert_allclose(padded, [[4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]])


def test_bilinear_to_grid_constant_field_preserved() -> None:
    """Interpolating a spatially constant field returns that same constant everywhere."""
    src_lats = np.array([-1.0, 0.0, 1.0, 2.0])
    src_lons = np.array([0.0, 90.0, 180.0, 270.0])
    field = np.full((3, 4, 4), 5.0)  # (n_times, n_lat, n_lon)

    target_lats = np.array([-0.5, 0.5])
    target_lons = np.array([45.0, 135.0])
    result = _bilinear_to_grid(field, src_lats, src_lons, target_lats, target_lons)

    assert result.shape == (3, 2, 2)
    assert_allclose(result, 5.0)


def test_bilinear_to_grid_lon_wraparound() -> None:
    """Querying near the 359.75°→0° boundary interpolates across it, not to NaN."""
    src_lats = np.array([0.0, 1.0])
    src_lons = np.array([0.0, 90.0, 180.0, 270.0])
    values_by_lon = np.cos(np.radians(src_lons))  # periodic: value at 0 == value at 360
    field = np.broadcast_to(values_by_lon, (1, 2, 4)).copy()

    target_lats = np.array([0.5])
    target_lons = np.array([350.0])
    result = _bilinear_to_grid(field, src_lats, src_lons, target_lats, target_lons)

    # Linear interpolation between lon=270 (value 0.0) and the wrapped lon=360 (value 1.0),
    # 80/90 of the way across.
    expected = (350.0 - 270.0) / 90.0
    assert not np.isnan(result).any()
    assert_allclose(result[0, 0, 0], expected, atol=1e-10)


def test_bilinear_to_grid_nan_propagates() -> None:
    """A NaN source corner contaminates nearby interpolated points, not distant ones."""
    src_lats = np.array([-1.0, 0.0, 1.0, 2.0])
    src_lons = np.array([0.0, 90.0, 180.0, 270.0])
    field = np.full((1, 4, 4), 5.0)
    field[0, 1, 1] = np.nan  # corner at (lat=0, lon=90)

    target_lats = np.array([0.1, 1.5])
    target_lons = np.array([91.0, 200.0])
    result = _bilinear_to_grid(field, src_lats, src_lons, target_lats, target_lons)

    assert np.isnan(result[0, 0, 0])  # stencil includes the NaN corner (lat=0, lon=90)
    assert not np.isnan(result[0, 1, 1])  # stencil is (lat 1-2, lon 180-270): far from it


def test_reprojector_fit_uses_x_coordinates_when_present() -> None:
    """fit() targets X's own <grid>_lat/<grid>_lon coords, not the canonical full grid."""
    ds = xr.Dataset(
        {"sigwh": (("time", "wave_lat", "wave_lon"), np.zeros((2, 3, 4)))},
        coords={
            "time": [0, 1],
            "wave_lat": [-75.0, -74.75, -74.5],
            "wave_lon": [0.0, 0.25, 0.5, 0.75],
        },
    )
    reprojector = Reprojector().fit(ds)
    assert_allclose(reprojector.target_lats_, [-75.0, -74.75, -74.5])
    assert_allclose(reprojector.target_lons_, [0.0, 0.25, 0.5, 0.75])


def test_reprojector_passthrough_target_grid_variable() -> None:
    """A variable already on the target grid is returned unchanged."""
    values = np.arange(24.0).reshape(2, 3, 4)
    ds = xr.Dataset(
        {"sigwh": (("time", "wave_lat", "wave_lon"), values)},
        coords={
            "time": [0, 1],
            "wave_lat": [-75.0, -74.75, -74.5],
            "wave_lon": [0.0, 0.25, 0.5, 0.75],
        },
    )
    result = Reprojector().fit_transform(ds)
    assert result["sigwh"].dims == ("time", "wave_lat", "wave_lon")
    assert_allclose(result["sigwh"].values, values)


def test_reprojector_reprojects_non_target_grid_variable() -> None:
    """A wind-grid variable is bilinearly interpolated onto the wave-grid target."""
    ds = xr.Dataset(
        {
            "sigwh": (("time", "wave_lat", "wave_lon"), np.zeros((2, 2, 2))),
            "u_wind": (("time", "wind_lat", "wind_lon"), np.full((2, 4, 4), 7.5)),
        },
        coords={
            "time": [0, 1],
            "wave_lat": [-0.5, 0.5],
            "wave_lon": [45.0, 135.0],
            "wind_lat": [-1.0, 0.0, 1.0, 2.0],
            "wind_lon": [0.0, 90.0, 180.0, 270.0],
        },
    )
    result = Reprojector().fit_transform(ds)

    assert result["u_wind"].dims == ("time", "wave_lat", "wave_lon")
    assert result["u_wind"].shape == (2, 2, 2)
    assert_allclose(result["u_wind"].values, 7.5)
    assert_allclose(result["wave_lat"].values, [-0.5, 0.5])
    assert_allclose(result["wave_lon"].values, [45.0, 135.0])


def test_reprojector_inverse_transform_is_identity() -> None:
    """inverse_transform is a no-op: reprojection is lossy and has no true inverse."""
    ds = xr.Dataset(
        {"sigwh": (("time", "wave_lat", "wave_lon"), np.arange(8.0).reshape(2, 2, 2))},
        coords={"time": [0, 1], "wave_lat": [-0.5, 0.5], "wave_lon": [45.0, 135.0]},
    )
    reprojector = Reprojector().fit(ds)
    result = reprojector.inverse_transform(ds)
    assert result is ds


def test_reprojector_sklearn_clone() -> None:
    """clone() produces a fresh unfitted copy with matching params."""
    reprojector = Reprojector(target_grid="wave")
    cloned = clone(reprojector)
    assert isinstance(cloned, Reprojector)
    assert cloned.target_grid == "wave"
    assert not hasattr(cloned, "target_lats_")
