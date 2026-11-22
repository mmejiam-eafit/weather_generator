"""Tests for train.standardize (see docs/model_design.md §5.4)."""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose
from sklearn.base import clone

from train.standardize import PerCellStandardizer


def _dataset(values: np.ndarray) -> xr.Dataset:
    n_times, n_lat, n_lon = values.shape
    return xr.Dataset(
        {"wsh": (("time", "wave_lat", "wave_lon"), values)},
        coords={
            "time": np.arange(n_times),
            "wave_lat": np.arange(n_lat, dtype=np.float64),
            "wave_lon": np.arange(n_lon, dtype=np.float64),
        },
    )


def test_standardizer_divides_by_per_cell_std() -> None:
    """transform() divides each cell by its own fitted temporal std."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.0, scale=3.0, size=(200, 2, 2))
    ds = _dataset(values)

    standardizer = PerCellStandardizer().fit(ds)
    result = standardizer.transform(ds)

    expected_sigma = np.std(values, axis=0)
    assert_allclose(standardizer.sigma_cell_["wsh"], expected_sigma)
    assert_allclose(result["wsh"].values, values / expected_sigma)


def test_standardizer_inverse_transform_roundtrip() -> None:
    """fit -> transform -> inverse_transform reproduces the original values."""
    rng = np.random.default_rng(1)
    values = rng.normal(loc=5.0, scale=2.0, size=(50, 3, 3))
    ds = _dataset(values)

    standardizer = PerCellStandardizer()
    standardized = standardizer.fit_transform(ds)
    recovered = standardizer.inverse_transform(standardized)

    assert_allclose(recovered["wsh"].values, values, rtol=1e-10)


def test_standardizer_unit_variance_after_transform() -> None:
    """After transform(), each cell's standard deviation is ~1."""
    rng = np.random.default_rng(2)
    values = rng.normal(loc=0.0, scale=7.0, size=(500, 2, 2))
    ds = _dataset(values)

    result = PerCellStandardizer().fit_transform(ds)
    assert_allclose(np.std(result["wsh"].values, axis=0), 1.0, atol=1e-10)


def test_standardizer_land_cell_stays_nan() -> None:
    """An all-NaN cell (land) is never coerced to zero; it stays NaN."""
    values = np.random.default_rng(3).normal(size=(30, 2, 2))
    values[:, 0, 0] = np.nan
    ds = _dataset(values)

    result = PerCellStandardizer().fit_transform(ds)
    assert np.isnan(result["wsh"].values[:, 0, 0]).all()
    assert not np.isnan(result["wsh"].values[:, 1, 1]).any()


def test_standardizer_sklearn_clone() -> None:
    """clone() produces a fresh unfitted copy."""
    standardizer = PerCellStandardizer()
    cloned = clone(standardizer)
    assert isinstance(cloned, PerCellStandardizer)
    assert not hasattr(cloned, "sigma_cell_")


def test_standardizer_near_constant_cell_does_not_blow_up() -> None:
    """A cell with a tiny (near-zero) fitted std passes its real anomaly through unscaled."""
    rng = np.random.default_rng(4)
    values = rng.normal(loc=0.0, scale=3.0, size=(200, 1, 2))
    values[:, 0, 0] = 5.0 + rng.normal(scale=1e-10, size=200)  # near-constant cell
    ds = _dataset(values)

    standardizer = PerCellStandardizer().fit(ds)
    assert standardizer.sigma_cell_["wsh"][0, 0] < 1e-6  # true fitted sigma stays tiny
    result = standardizer.transform(ds)

    assert np.all(np.isfinite(result["wsh"].values[:, 0, 0]))
    assert_allclose(result["wsh"].values[:, 0, 0], values[:, 0, 0])


def test_standardizer_exactly_constant_cell_no_nan_or_inf() -> None:
    """A cell with std exactly 0.0 transforms to its own (finite) constant value."""
    rng = np.random.default_rng(5)
    values = rng.normal(loc=0.0, scale=2.0, size=(50, 1, 2))
    values[:, 0, 0] = 7.0  # exactly constant
    ds = _dataset(values)

    standardizer = PerCellStandardizer().fit(ds)
    assert standardizer.sigma_cell_["wsh"][0, 0] == 0.0
    result = standardizer.transform(ds)

    assert np.all(np.isfinite(result["wsh"].values[:, 0, 0]))
    assert_allclose(result["wsh"].values[:, 0, 0], 7.0)


def test_standardizer_inverse_transform_roundtrip_near_zero_sigma() -> None:
    """fit -> transform -> inverse_transform recovers the original values at a near-constant cell."""
    rng = np.random.default_rng(6)
    values = rng.normal(loc=0.0, scale=4.0, size=(100, 1, 2))
    values[:, 0, 0] = 5.0 + rng.normal(scale=1e-10, size=100)
    ds = _dataset(values)

    standardizer = PerCellStandardizer()
    standardized = standardizer.fit_transform(ds)
    recovered = standardizer.inverse_transform(standardized)

    assert_allclose(recovered["wsh"].values, values, rtol=1e-6)
