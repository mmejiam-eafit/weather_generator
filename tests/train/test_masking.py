"""Tests for train.masking (see docs/variable_taxonomy.md §3.3.1,
tasks/17-partition-nan-sentinel-policy.md)."""

from typing import cast

import numpy as np
from numpy.testing import assert_allclose

from train.masking import (
    apply_partition_sentinel_policy,
    height_log_energy,
    land_mask_from_sigwh,
    ocean_nan_report,
    partition_companion_backfill_check,
    period_sentinel_from_archive,
)


def test_land_mask_from_sigwh_strict() -> None:
    """A cell counts as ocean only if non-NaN at every timestep (default valid_frac=1.0)."""
    sigwh = np.zeros((4, 3, 3))  # (time, lat, lon)
    sigwh[:, 0, 0] = np.nan  # land: NaN every timestep
    sigwh[2, 1, 1] = np.nan  # ocean, but NaN at one timestep -- still land under strict mode

    mask = land_mask_from_sigwh(sigwh)

    assert mask.shape == (3, 3)
    assert not mask[0, 0]
    assert not mask[1, 1]
    assert mask[0, 1] and mask[2, 2]


def test_land_mask_from_sigwh_tolerant() -> None:
    """A looser valid_frac lets an occasionally-NaN cell still count as ocean."""
    sigwh = np.zeros((10, 1, 1))  # (time, lat, lon)
    sigwh[0, 0, 0] = np.nan  # 90% valid

    assert not land_mask_from_sigwh(sigwh, valid_frac=1.0)[0, 0]
    assert land_mask_from_sigwh(sigwh, valid_frac=0.9)[0, 0]


def test_apply_partition_sentinel_policy_ocean_nan_is_valid() -> None:
    """The one behavior that differs from today's `valid = ~isnan(X)` notebook
    pattern: an ocean cell that was NaN in X must be VALID (1) in tucker_mask,
    not imputed like land."""
    X = np.full((3, 2, 2), 1.5)  # (time, lat, lon)
    X[1, 0, 0] = np.nan  # ocean cell, partition-absent at one timestep
    land_mask = np.array([[True, True], [False, True]])  # (1,0) is land

    result = apply_partition_sentinel_policy(X, land_mask, sentinel_value=0.0)

    assert result.tucker_mask[1, 0, 0] == 1.0
    assert result.filled[1, 0, 0] == 0.0
    assert result.n_ocean_nan == 1


def test_apply_partition_sentinel_policy_land_stays_invalid() -> None:
    """Land cells keep today's treatment: zero-filled and marked INVALID (imputed)."""
    X = np.full((3, 2, 2), 1.5)  # (time, lat, lon)
    X[:, 1, 0] = np.nan  # land cell, NaN every timestep
    land_mask = np.array([[True, True], [False, True]])

    result = apply_partition_sentinel_policy(X, land_mask)

    assert np.all(result.tucker_mask[:, 1, 0] == 0.0)
    assert np.all(result.filled[:, 1, 0] == 0.0)
    assert result.n_ocean_nan == 0


def test_apply_partition_sentinel_policy_real_values_untouched() -> None:
    """Non-NaN values, ocean or land, pass through unchanged."""
    X = np.full((3, 2, 2), 2.5)  # (time, lat, lon)
    land_mask = np.ones((2, 2), dtype=bool)

    result = apply_partition_sentinel_policy(X, land_mask)

    assert_allclose(result.filled, X)
    assert np.all(result.tucker_mask == 1.0)


def test_apply_partition_sentinel_policy_custom_sentinel() -> None:
    """A non-default sentinel_value (e.g. a derived period placeholder) fills ocean NaN."""
    X = np.array([[[np.nan]], [[5.0]], [[6.0]]])  # (time=3, lat=1, lon=1)
    land_mask = np.array([[True]])

    result = apply_partition_sentinel_policy(X, land_mask, sentinel_value=7.5)

    assert result.filled[0, 0, 0] == 7.5


def test_height_log_energy_floors_zero() -> None:
    """log E at H=0 is finite, not -inf/NaN, and equals log(floor_m**2)."""
    result = height_log_energy(np.array([0.0]), floor_m=0.01)

    assert np.isfinite(result).all()
    assert_allclose(result, np.log(np.array([0.01**2])))


def test_height_log_energy_real_values_unchanged() -> None:
    """Well above the floor, height_log_energy matches plain log(H**2)."""
    H = np.array([1.0, 2.5, 10.0])

    result = height_log_energy(H, floor_m=0.01)

    assert_allclose(result, np.log(H**2))


def test_ocean_nan_report_restricted_to_ocean() -> None:
    """NaN fraction and always-valid fraction are computed only over ocean cells."""
    X = np.zeros((4, 2, 1))  # (time, lat, lon)
    X[:, 0, 0] = np.nan  # land, all NaN -- must not count toward "ocean NaN"
    X[0, 1, 0] = np.nan  # ocean, one NaN out of 4 timesteps
    land_mask = np.array([[False], [True]])

    report = ocean_nan_report(X, land_mask, valid_time_fracs=(1.0, 0.5))

    assert report["n_ocean_cells"] == 1
    assert_allclose(cast(float, report["ocean_nan_fraction"]), 0.25)
    always_valid_frac = cast("dict[float, float]", report["always_valid_cell_fraction"])
    assert always_valid_frac[1.0] == 0.0
    assert always_valid_frac[0.5] == 1.0


def test_partition_companion_backfill_check_no_primary_nan() -> None:
    """No primary NaN at ocean cells -> nan-filled, non-crashing result."""
    primary = np.ones((3, 2, 2))  # (time, lat, lon)
    companion = np.ones((3, 2, 2))
    land_mask = np.ones((2, 2), dtype=bool)

    result = partition_companion_backfill_check(primary, companion, land_mask)

    assert result["n_primary_nan_ocean"] == 0
    assert np.isnan(cast(float, result["companion_also_nan_fraction"]))


def test_partition_companion_backfill_check_detects_stale_companion() -> None:
    """A companion that stays constant during primary-NaN runs (stale/backfilled)
    shows near-zero std there, vs. real variation when primary is valid."""
    rng = np.random.default_rng(0)
    T = 40
    primary = rng.normal(size=(T, 1, 1))  # (time, lat, lon)
    companion = rng.normal(size=(T, 1, 1))

    nan_idx = slice(0, 20)
    primary[nan_idx, 0, 0] = np.nan
    companion[nan_idx, 0, 0] = 99.0  # stale filler, constant during the gap
    land_mask = np.array([[True]])

    result = partition_companion_backfill_check(primary, companion, land_mask)

    assert result["n_primary_nan_ocean"] == 20
    assert result["companion_also_nan_fraction"] == 0.0
    assert cast(float, result["mean_companion_std_when_primary_nan"]) < 1e-9
    assert cast(float, result["mean_companion_std_when_primary_valid"]) > 0.1


def test_period_sentinel_from_archive_recovers_known_intercept() -> None:
    """A synthetic linear low-height trend is recovered by both estimators."""
    rng = np.random.default_rng(1)
    h = rng.uniform(0.01, 1.0, size=2000)
    true_intercept = 4.0
    p = true_intercept + 0.5 * h + rng.normal(scale=1e-3, size=h.shape)

    estimate = period_sentinel_from_archive(h, p, small_height_quantile=0.05)

    assert_allclose(estimate.extrapolated_intercept, true_intercept, atol=0.05)
    assert_allclose(estimate.conditional_median, true_intercept, atol=0.1)
    assert estimate.agreement < 0.1


def test_period_sentinel_from_archive_ignores_non_finite() -> None:
    """NaN/non-positive entries passed in by mistake are dropped, not propagated."""
    h = np.array([np.nan, 0.0, -1.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p = np.array([1.0, 2.0, 3.0, 5.0, 5.1, 5.2, 5.3, 5.4])

    estimate = period_sentinel_from_archive(h, p, small_height_quantile=0.5)

    assert np.isfinite(estimate.conditional_median)
    assert np.isfinite(estimate.extrapolated_intercept)
