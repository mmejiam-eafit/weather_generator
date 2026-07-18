"""Tests for train.climatology (see docs/model_design.md §5.4)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

from train.climatology import ClimatologyRemover

_T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _harmonic_dataset(n_times: int, n_lat: int, n_lon: int, diurnal: bool = False) -> xr.Dataset:
    times = np.array(
        [
            np.datetime64((_T0 + timedelta(hours=3 * i)).replace(tzinfo=None), "s")
            for i in range(n_times)
        ]
    )
    t_days = np.arange(n_times) * 0.125  # 3h cadence
    omega_y = 2.0 * np.pi / 365.25
    seasonal = 2.0 * np.cos(omega_y * t_days) + 1.5 * np.sin(omega_y * t_days) + 10.0
    if diurnal:
        seasonal = seasonal + 0.5 * np.cos(2.0 * np.pi * t_days)
    field = np.broadcast_to(seasonal[:, None, None], (n_times, n_lat, n_lon)).copy()
    return xr.Dataset(
        {"was": (("time", "wind_lat", "wind_lon"), field)},
        coords={
            "time": times,
            "wind_lat": np.arange(n_lat, dtype=np.float64),
            "wind_lon": np.arange(n_lon, dtype=np.float64),
        },
    )


def test_harmonic_roundtrip() -> None:
    """fit -> transform -> inverse_transform reproduces training data at fit points."""
    ds = _harmonic_dataset(n_times=40, n_lat=2, n_lon=3)
    remover = ClimatologyRemover()
    anomalies = remover.fit_transform(ds)
    recovered = remover.inverse_transform(anomalies)

    assert_allclose(recovered["was"].values, ds["was"].values, atol=1e-8)
    assert_allclose(anomalies["was"].values, 0.0, atol=1e-8)  # purely seasonal signal, no residual


def test_diurnal_variable_fits_diurnal_term() -> None:
    """A variable named in diurnal_variables gets the diurnal harmonic fitted too."""
    ds = _harmonic_dataset(n_times=80, n_lat=2, n_lon=2, diurnal=True)
    remover = ClimatologyRemover(diurnal_variables=["was"])
    anomalies = remover.fit_transform(ds)

    # intercept + trend (include_trend defaults True) + 2 seasonal pairs + 1 diurnal pair
    assert remover.coeffs_["was"].shape[0] == 8
    assert_allclose(anomalies["was"].values, 0.0, atol=1e-6)


def test_default_diurnal_variables_excludes_wave_scalars() -> None:
    """Without an explicit diurnal_variables, a wave-grid scalar gets no diurnal term."""
    ds = _harmonic_dataset(n_times=40, n_lat=2, n_lon=2)
    ds = ds.rename({"was": "sigwh"})
    remover = ClimatologyRemover().fit(ds)
    assert (
        remover.coeffs_["sigwh"].shape[0] == 6
    )  # intercept + trend + 2 seasonal pairs, no diurnal


def test_land_cell_gets_nan_coefficients_without_corrupting_others() -> None:
    """An all-NaN cell (land) gets NaN coefficients; other cells are unaffected."""
    ds = _harmonic_dataset(n_times=40, n_lat=2, n_lon=2)
    ds["was"].values[:, 0, 0] = np.nan

    remover = ClimatologyRemover().fit(ds)
    assert np.isnan(remover.coeffs_["was"][:, 0, 0]).all()
    assert not np.isnan(remover.coeffs_["was"][:, 1, 1]).any()


def test_missing_snapshot_row_excluded_from_fit() -> None:
    """A fully-missing snapshot (e.g. from an ingest gap) doesn't corrupt the fit.

    Trend is disabled here: this fixture spans only ~5 days, over which
    Δτ(t) is nearly collinear with the annual sin term (both are close to
    linear across a window this short relative to the 365.25-day period),
    so the design matrix is ill-conditioned and the min-norm lstsq split
    between those two columns becomes numerically sensitive to which row is
    dropped -- a fixture-window artifact, not a trend-logic bug. This test
    is about harmonic-fit robustness to a missing row, so it isolates that
    with include_trend=False; trend recovery is covered separately by
    test_trend_recovery on a realistic multi-year window.
    """
    ds = _harmonic_dataset(n_times=40, n_lat=2, n_lon=2)
    clean_coeffs = ClimatologyRemover(include_trend=False).fit(ds).coeffs_["was"]

    ds["was"].values[5, :, :] = np.nan
    coeffs_with_gap = ClimatologyRemover(include_trend=False).fit(ds).coeffs_["was"]

    assert_allclose(coeffs_with_gap, clean_coeffs, atol=1e-6)


def test_reference_epoch_is_first_training_timestamp() -> None:
    """t_train_start_ is set from X's first timestamp, not reset per call."""
    ds = _harmonic_dataset(n_times=10, n_lat=1, n_lon=1)
    remover = ClimatologyRemover().fit(ds)
    assert remover.t_train_start_ == _T0


def _trend_dataset(n_years: float, c0_true: np.ndarray, c_lin_true: np.ndarray) -> xr.Dataset:
    """Synthetic dataset with a known per-cell intercept, linear trend, and seasonal cycle.

    `c0_true` and `c_lin_true` share shape ``(n_lat, n_lon)``; `c_lin_true` is
    in "units per year", matching how `ClimatologyRemover` normalizes
    `Δτ(t) = (t - t_ref) / 365.25 days`. Uses a daily cadence starting at
    `_T0` so that `t_ref_` (the training window's midpoint, as computed by
    `ClimatologyRemover.fit`) falls exactly at day index `(n_days - 1) / 2`.
    """
    n_lat, n_lon = c0_true.shape
    n_days = int(round(n_years * 365.25))
    times = np.array(
        [np.datetime64((_T0 + timedelta(days=i)).replace(tzinfo=None), "s") for i in range(n_days)]
    )
    t_days = np.arange(n_days, dtype=np.float64)
    t_ref_days = (n_days - 1) / 2.0  # midpoint of [0, n_days - 1] -> matches fit()'s t_ref_
    delta_tau = (t_days - t_ref_days) / 365.25
    omega_y = 2.0 * np.pi / 365.25
    seasonal = 2.0 * np.cos(omega_y * t_days) + 1.5 * np.sin(omega_y * t_days)

    field = (
        c0_true[None, :, :]
        + c_lin_true[None, :, :] * delta_tau[:, None, None]
        + seasonal[:, None, None]
    )
    return xr.Dataset(
        {"was": (("time", "wind_lat", "wind_lon"), field)},
        coords={
            "time": times,
            "wind_lat": np.arange(n_lat, dtype=np.float64),
            "wind_lon": np.arange(n_lon, dtype=np.float64),
        },
    )


def test_trend_recovery() -> None:
    """A known per-cell linear trend (e.g. 0.05/year) is recovered within 5% relative error."""
    c0_true = np.full((2, 3), 10.0, dtype=np.float64)
    c_lin_true = np.array([[0.03, 0.05, 0.08], [-0.04, 0.06, 0.02]], dtype=np.float64)
    ds = _trend_dataset(n_years=4.0, c0_true=c0_true, c_lin_true=c_lin_true)

    remover = ClimatologyRemover().fit(ds)

    assert_allclose(remover.c_lin_["was"], c_lin_true, rtol=0.05, atol=1e-6)


def test_trend_uncorrelated_with_c0() -> None:
    """Centering the trend term at t_ref_ (midpoint) decorrelates c_lin from c0 across cells.

    `c0_true` varies only along lat and `c_lin_true` only along lon, so their
    true cross-cell correlation is exactly zero (see docs/model_design.md
    §3.9: centering at the midpoint is what makes `c_lin` uncorrelated with
    `c0`). A midpoint-centered fit should recover that near-zero
    correlation.

    To show the *alternative* is measurably worse, reconstruct what the
    intercept would have been had the trend column instead been anchored at
    `t_train_start_`. This is an exact algebraic reparametrization -- for
    `y = c0 + c_lin * (t - t_ref) / 1yr`, rewriting in terms of
    `(t - t_train_start) / 1yr = (t - t_ref)/1yr + offset_years` gives
    `y = (c0 - offset_years * c_lin) + c_lin * (t - t_train_start)/1yr`, i.e.
    `c0_offcenter = c0_centered - offset_years * c_lin` -- so no separate
    fit is needed to compute it. The off-center intercept is then strongly
    (spuriously) correlated with `c_lin` even though the true generating
    process has zero correlation between them.
    """
    n_lat, n_lon = 6, 7
    lat_idx = np.arange(n_lat, dtype=np.float64)
    lon_idx = np.arange(n_lon, dtype=np.float64) - (n_lon - 1) / 2.0
    c0_true = 10.0 + 0.01 * lat_idx[:, None] + 0.0 * lon_idx[None, :]
    c_lin_true = 0.1 * lon_idx[None, :] + 0.0 * lat_idx[:, None]
    assert abs(np.corrcoef(c0_true.ravel(), c_lin_true.ravel())[0, 1]) < 1e-10

    ds = _trend_dataset(n_years=4.0, c0_true=c0_true, c_lin_true=c_lin_true)
    remover = ClimatologyRemover().fit(ds)

    c0_hat = remover.coeffs_["was"][0].reshape(-1)
    c_lin_hat = remover.c_lin_["was"].reshape(-1)
    corr_centered = np.corrcoef(c0_hat, c_lin_hat)[0, 1]

    offset_years = (remover.t_ref_ - remover.t_train_start_).total_seconds() / (365.25 * 86400.0)
    c0_hat_offcenter = c0_hat - offset_years * c_lin_hat
    corr_offcenter = np.corrcoef(c0_hat_offcenter, c_lin_hat)[0, 1]

    assert abs(corr_centered) < 1e-6
    assert abs(corr_offcenter) > 0.5
    assert abs(corr_offcenter) > abs(corr_centered) + 0.5  # measurably worse off-center


def test_t_ref_is_training_window_midpoint() -> None:
    """t_ref_ is the exact midpoint between the first and last training timestamps."""
    n_times = 40
    ds = _harmonic_dataset(n_times=n_times, n_lat=1, n_lon=1)
    remover = ClimatologyRemover().fit(ds)

    expected_start = _T0
    expected_end = _T0 + timedelta(hours=3 * (n_times - 1))
    expected_midpoint = expected_start + (expected_end - expected_start) / 2

    assert remover.t_ref_ == expected_midpoint
    assert remover.t_ref_ != remover.t_train_start_


def test_include_trend_false_drops_column_and_roundtrips() -> None:
    """include_trend=False drops the Δτ column entirely; transform/inverse_transform still round-trip."""
    ds = _harmonic_dataset(n_times=40, n_lat=2, n_lon=2)
    remover_with_trend = ClimatologyRemover(include_trend=True).fit(ds)
    remover_without_trend = ClimatologyRemover(include_trend=False).fit(ds)

    assert (
        remover_without_trend.coeffs_["was"].shape[0]
        == remover_with_trend.coeffs_["was"].shape[0] - 1
    )
    assert remover_without_trend.c_lin_ == {}

    anomalies = remover_without_trend.transform(ds)
    recovered = remover_without_trend.inverse_transform(anomalies)
    assert_allclose(recovered["was"].values, ds["was"].values, atol=1e-8)
    assert_allclose(anomalies["was"].values, 0.0, atol=1e-8)
