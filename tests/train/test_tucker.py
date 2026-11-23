"""Tests for train.tucker (see docs/model_design.md §5.5)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from train.tucker import TuckerFit, fill_land, tucker_fit, tucker_inverse, tucker_transform

_SHAPE = (40, 12, 16)  # (T, H, W)
_RANK = (3, 4)


def _planted(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """An exactly rank-(3, 4) spatial tensor with a land block, land NaN."""
    n_times, n_lat, n_lon = _SHAPE
    lat_true = rng.normal(size=(n_lat, _RANK[0]))
    lon_true = rng.normal(size=(n_lon, _RANK[1]))
    core_true = rng.normal(size=(n_times, *_RANK))
    x = lat_true @ core_true @ lon_true.T
    ocean_mask = np.ones((n_lat, n_lon), dtype=bool)
    ocean_mask[:3, :4] = False
    x[:, ~ocean_mask] = np.nan
    return x, ocean_mask


def test_tucker_fit_recovers_planted_low_rank_tensor_through_land_mask() -> None:
    """A rank-r tensor is reconstructed at ocean cells; shapes and dtypes follow §5.5."""
    rng = np.random.default_rng(0)
    x, ocean_mask = _planted(rng)

    fit, core = tucker_fit(fill_land(x, ocean_mask), ocean_mask, _RANK, rng)
    recon = tucker_inverse(core, fit)

    assert isinstance(fit, TuckerFit)
    assert fit.factor_lat.shape == (_SHAPE[1], _RANK[0])
    assert fit.factor_lon.shape == (_SHAPE[2], _RANK[1])
    assert fit.factor_lat.dtype == np.float64
    assert core.shape == (_SHAPE[0], *_RANK)
    ocean_error = np.linalg.norm(recon[:, ocean_mask] - x[:, ocean_mask]) / np.linalg.norm(
        x[:, ocean_mask]
    )
    assert ocean_error < 1e-3
    assert fit.reconstruction_error < 1e-3


def test_tucker_fit_reconstruction_error_is_over_ocean_cells_only() -> None:
    """`reconstruction_error` equals the relative error of the returned core at ocean cells."""
    rng = np.random.default_rng(1)
    x, ocean_mask = _planted(rng)
    x[:, ocean_mask] += rng.normal(scale=0.5, size=x[:, ocean_mask].shape)

    fit, core = tucker_fit(fill_land(x, ocean_mask), ocean_mask, _RANK, rng)
    recon = tucker_inverse(core, fit)
    expected = np.linalg.norm(recon[:, ocean_mask] - x[:, ocean_mask]) / np.linalg.norm(
        x[:, ocean_mask]
    )

    assert_allclose(fit.reconstruction_error, expected, rtol=1e-6)


def test_tucker_fit_stops_before_n_iter_max_once_converged() -> None:
    """An exactly low-rank tensor converges well before the iteration cap."""
    rng = np.random.default_rng(2)
    x, ocean_mask = _planted(rng)

    fit, _ = tucker_fit(fill_land(x, ocean_mask), ocean_mask, _RANK, rng, n_iter_max=300)

    assert 0 < fit.n_iter < 300


def test_tucker_fit_same_seed_same_fit() -> None:
    """Two fits from identically seeded generators are identical (§ randomness discipline)."""
    x, ocean_mask = _planted(np.random.default_rng(3))
    filled = fill_land(x, ocean_mask)

    fit_a, core_a = tucker_fit(filled, ocean_mask, _RANK, np.random.default_rng(7))
    fit_b, core_b = tucker_fit(filled, ocean_mask, _RANK, np.random.default_rng(7))

    assert_allclose(core_a, core_b)
    assert_allclose(fit_a.factor_lat, fit_b.factor_lat)
    assert fit_a.n_iter == fit_b.n_iter


def test_fill_land_zero_fills_land_and_keeps_ocean_values() -> None:
    """Land becomes 0.0 (float32); ocean values pass through."""
    x, ocean_mask = _planted(np.random.default_rng(4))

    filled = fill_land(x, ocean_mask)

    assert filled.dtype == np.float32
    assert np.all(filled[:, ~ocean_mask] == 0.0)
    assert_allclose(filled[:, ocean_mask], x[:, ocean_mask], rtol=1e-6)


def test_fill_land_rejects_nan_at_an_ocean_cell() -> None:
    """An unfilled ocean NaN is an upstream failure, not something to impute (§9 gotcha 24)."""
    x, ocean_mask = _planted(np.random.default_rng(5))
    x[7, 6, 9] = np.nan

    with pytest.raises(ValueError, match="1 ocean cells hold NaN"):
        fill_land(x, ocean_mask)


def test_tucker_inverse_subgrid_matches_the_full_grid_slice() -> None:
    """Evaluating rows x cols equals slicing the full reconstruction (§5.8 step 3)."""
    rng = np.random.default_rng(6)
    fit = TuckerFit(
        factor_lat=rng.normal(size=(12, 3)),
        factor_lon=rng.normal(size=(16, 4)),
        reconstruction_error=0.0,
        n_iter=0,
    )
    core = rng.normal(size=(5, 3, 4))
    rows = np.array([2, 3, 9])
    cols = np.array([15, 0, 7])

    subgrid = tucker_inverse(core, fit, rows=rows, cols=cols)

    assert_allclose(subgrid, tucker_inverse(core, fit)[:, rows][:, :, cols], rtol=1e-12)


def test_tucker_transform_recovers_a_core_in_the_factor_span() -> None:
    """With orthonormal fitted factors, projecting a reconstruction returns its core.

    The ALS runs on the float32 input §5.5 specifies, so the factors are
    orthonormal to float32 precision, which sets the tolerance.
    """
    rng = np.random.default_rng(8)
    x, ocean_mask = _planted(rng)
    fit, _ = tucker_fit(fill_land(x, ocean_mask), ocean_mask, _RANK, rng)
    core = rng.normal(size=(6, *_RANK))

    recovered = tucker_transform(tucker_inverse(core, fit), fit)

    assert_allclose(recovered, core, atol=1e-5)
