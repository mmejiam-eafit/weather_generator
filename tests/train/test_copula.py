"""Tests for train.copula (see docs/model_design.md §3.5, §5.4)."""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose
from scipy import stats
from sklearn.base import clone

from train.copula import GaussianCopula


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


def test_forward_inverse_roundtrip() -> None:
    """Cell-wise round trip (value -> Gaussian -> value) preserves original values.

    Some quantization error is expected and acceptable: the LUT holds only
    `n_bins` quantiles, so recovery is exact only at those quantiles and
    linearly interpolated elsewhere. Error concentrates in sparse tail
    regions of the (skewed) exponential distribution used here; the bulk
    of the distribution should recover tightly.
    """
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=2.0, size=(3000, 2, 2))  # skewed, like Hs
    ds = _dataset(values)

    copula = GaussianCopula(n_bins=300)
    z = copula.fit_transform(ds)
    recovered = copula.inverse_transform(z)

    error = np.abs(recovered["wsh"].values - values)
    assert np.median(error) < 0.02
    assert np.quantile(error, 0.95) < 0.5


def test_gaussian_output_distribution() -> None:
    """After transform(), each cell's distribution is standard normal (KS test)."""
    rng = np.random.default_rng(1)
    values = rng.exponential(scale=2.0, size=(3000, 1, 1))
    ds = _dataset(values)

    z = GaussianCopula(n_bins=500).fit_transform(ds)
    sample = z["wsh"].values[:, 0, 0]

    ks_stat, p_value = stats.kstest(sample, "norm")
    assert p_value > 0.01, f"KS test rejected normality: stat={ks_stat}, p={p_value}"


def test_bounded_variable_stays_bounded_after_inverse() -> None:
    """A non-negative variable's recovered values never go negative (positivity preserved)."""
    rng = np.random.default_rng(2)
    values = rng.exponential(scale=1.5, size=(1500, 1, 1))
    ds = _dataset(values)

    copula = GaussianCopula(n_bins=200).fit(ds)
    # Simulate DMD/OU emitting standard normal samples, including extreme tails.
    z_samples = xr.Dataset(
        {"wsh": (("time", "wave_lat", "wave_lon"), rng.normal(size=(500, 1, 1)) * 3.0)},
        coords={
            "time": np.arange(500),
            "wave_lat": ds.coords["wave_lat"],
            "wave_lon": ds.coords["wave_lon"],
        },
    )
    recovered = copula.inverse_transform(z_samples)

    assert np.nanmin(recovered["wsh"].values) >= 0.0 - 1e-9


def test_land_cell_stays_nan_through_forward_and_inverse() -> None:
    """An all-NaN cell (land) never contaminates other cells and stays NaN."""
    rng = np.random.default_rng(3)
    values = rng.exponential(scale=2.0, size=(2000, 2, 2))
    values[:, 0, 0] = np.nan
    ds = _dataset(values)

    copula = GaussianCopula(n_bins=100)
    z = copula.fit_transform(ds)
    recovered = copula.inverse_transform(z)

    assert np.isnan(z["wsh"].values[:, 0, 0]).all()
    assert np.isnan(recovered["wsh"].values[:, 0, 0]).all()
    assert not np.isnan(z["wsh"].values[:, 1, 1]).any()
    assert not np.isnan(recovered["wsh"].values[:, 1, 1]).any()
    error = np.abs(recovered["wsh"].values[:, 1, 1] - values[:, 1, 1])
    assert np.median(error) < 0.05


def test_multiple_cells_independent() -> None:
    """Different cells with different marginals are each Gaussianized independently."""
    rng = np.random.default_rng(4)
    low = rng.exponential(scale=1.0, size=(1000, 1, 1))
    high = rng.exponential(scale=10.0, size=(1000, 1, 1))
    values = np.concatenate([low, high], axis=2)  # (1000, 1, 2)
    ds = _dataset(values)

    z = GaussianCopula(n_bins=300).fit_transform(ds)
    for cell in range(2):
        sample = z["wsh"].values[:, 0, cell]
        _, p_value = stats.kstest(sample, "norm")
        assert p_value > 0.01


def test_sklearn_clone() -> None:
    """clone() produces a fresh unfitted copy with matching params."""
    copula = GaussianCopula(n_bins=250)
    cloned = clone(copula)
    assert isinstance(cloned, GaussianCopula)
    assert cloned.n_bins == 250
    assert not hasattr(cloned, "sorted_anomalies_binned_")
