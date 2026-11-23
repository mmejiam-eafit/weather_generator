"""Tests for weather.validate (see docs/model_design.md §6)."""

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from numpy.testing import assert_allclose

from weather.model import WeatherModel
from weather.simulate import simulate_field
from weather.validate import (
    _empirical_semivariogram,
    _haversine_km,
    _load_real_physical,
    _random_ocean_cells,
    extreme_statistics_check,
    full_report,
    held_out_normality_check,
    joint_density_check,
    qq_plot_check,
    spectra_check,
    throughput_benchmark,
    variogram_check,
)


def _comparison_data(model: WeatherModel, data_dir: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Real (physical, archive-native) and simulated data over the model's training window."""
    real = _load_real_physical(model, data_dir, model.t_start, model.t_end)
    simulated = simulate_field(model, model.t_start, model.t_end, variables=list(model.variables))
    return real, simulated


def test_haversine_km_equatorial_quarter_circle() -> None:
    """A 90-degree longitude separation on the equator is a quarter of Earth's circumference."""
    distance = _haversine_km(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([90.0]))
    expected = np.pi / 2.0 * 6371.0
    assert_allclose(distance, expected, rtol=1e-6)


def test_haversine_km_zero_for_same_point() -> None:
    """Distance from a point to itself is exactly zero."""
    distance = _haversine_km(np.array([12.3]), np.array([45.6]), np.array([12.3]), np.array([45.6]))
    assert_allclose(distance, 0.0, atol=1e-9)


def test_empirical_semivariogram_zero_for_constant_field() -> None:
    """A spatially constant field has zero semivariance in every populated bin."""
    lats = np.array([0.0, 0.0, 1.0, 1.0])
    lons = np.array([0.0, 1.0, 0.0, 1.0])
    values = np.full(4, 3.5)
    bin_edges = np.linspace(0.0, 500.0, 6)

    result = _empirical_semivariogram(values, lats, lons, bin_edges)
    populated = ~np.isnan(result)
    assert populated.any()
    assert_allclose(result[populated], 0.0, atol=1e-9)


def test_random_ocean_cells_are_all_within_the_ocean_mask(fitted_tier1_model: WeatherModel) -> None:
    """Every returned (lat_idx, lon_idx) pair is True in the fitted ocean mask."""
    rng = np.random.default_rng(0)
    cells = _random_ocean_cells(fitted_tier1_model, n_cells=5, rng=rng)
    for lat_idx, lon_idx in cells:
        assert fitted_tier1_model.ocean_mask_[lat_idx, lon_idx]


def test_random_ocean_cells_caps_at_available_count(fitted_tier1_model: WeatherModel) -> None:
    """Requesting more cells than exist returns every ocean cell exactly once, no error."""
    rng = np.random.default_rng(0)
    n_ocean = int(fitted_tier1_model.ocean_mask_.sum())
    cells = _random_ocean_cells(fitted_tier1_model, n_cells=n_ocean + 1000, rng=rng)
    assert cells.shape[0] == n_ocean
    assert len({tuple(cell) for cell in cells}) == n_ocean


def test_load_real_physical_has_archive_native_variables_and_wave_grid_shape(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """The real-data helper recovers archive-native names on the common wave grid."""
    model = fitted_tier1_model
    real = _load_real_physical(model, tier1_archive, model.t_start, model.t_end)

    assert set(real.data_vars) == {"sigwh", "wsh", "wsp", "wsd", "was", "wad_come_from"}
    assert real["sigwh"].dims == ("time", "wave_lat", "wave_lon")
    assert real["sigwh"].shape == (60, 6, 5)


def test_qq_plot_check_returns_figure(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """qq_plot_check produces a Figure without raising, for several variables."""
    model = fitted_tier1_model
    real, simulated = _comparison_data(model, tier1_archive)

    fig = qq_plot_check(model, real, simulated, variables=["sigwh", "wsp"], n_cells=5)
    assert isinstance(fig, Figure)


def test_spectra_check_returns_figure(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """spectra_check produces a Figure without raising."""
    model = fitted_tier1_model
    real, simulated = _comparison_data(model, tier1_archive)

    fig = spectra_check(model, real, simulated, variables=["sigwh"], n_cells=5)
    assert isinstance(fig, Figure)


def test_variogram_check_returns_figure(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """variogram_check produces a Figure without raising, using a small cell subsample."""
    model = fitted_tier1_model
    real, simulated = _comparison_data(model, tier1_archive)

    fig = variogram_check(
        model, real, simulated, variable="sigwh", n_bins=5, n_sample_cells=10, n_snapshots=3
    )
    assert isinstance(fig, Figure)


def test_joint_density_check_returns_figure(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """joint_density_check produces a Figure without raising."""
    model = fitted_tier1_model
    real, simulated = _comparison_data(model, tier1_archive)

    fig = joint_density_check(model, real, simulated, var_a="sigwh", var_b="was")
    assert isinstance(fig, Figure)


def test_extreme_statistics_check_returns_figure(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """extreme_statistics_check produces a Figure without raising."""
    model = fitted_tier1_model
    real, simulated = _comparison_data(model, tier1_archive)

    fig = extreme_statistics_check(model, real, simulated, variable="sigwh", n_cells=5)
    assert isinstance(fig, Figure)


def test_throughput_benchmark_returns_positive_timings_per_n(
    fitted_tier1_model: WeatherModel,
) -> None:
    """throughput_benchmark reports one positive microseconds-per-point figure per N."""
    result = throughput_benchmark(fitted_tier1_model, ns=(1, 50), seed=0)
    assert set(result) == {1, 50}
    for value in result.values():
        assert value > 0.0


def test_held_out_normality_check_pvalues_within_unit_interval(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """One p-value per sampled core coefficient per tensor, each in [0, 1] (§6 check 7)."""
    model = fitted_tier1_model
    p_values = held_out_normality_check(
        model, tier1_archive, model.t_start, model.t_end, n_coefficients=3
    )
    assert set(p_values) == set(model.tensor_names_)
    for values in p_values.values():
        assert values.shape == (3,)
        finite = values[~np.isnan(values)]
        assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_held_out_normality_check_training_cores_are_near_normal(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """On the training window itself the copula makes each coefficient close to N(0, 1).

    Not every p-value is large: held-out projection (`tucker_transform`)
    differs slightly from the ALS core near land, and the LUT has 50 bins.
    """
    model = fitted_tier1_model
    p_values = held_out_normality_check(
        model, tier1_archive, model.t_start, model.t_end, n_coefficients=4
    )
    assert np.median(np.concatenate(list(p_values.values()))) > 0.05


def test_full_report_writes_a_pdf(
    tier1_archive: Path, fitted_tier1_model: WeatherModel, tmp_path: Path
) -> None:
    """full_report assembles every check into a readable, non-empty PDF file."""
    model = fitted_tier1_model
    output = tmp_path / "report.pdf"

    full_report(model, tier1_archive, model.t_start, model.t_end, output=output, seed=0)

    assert output.exists()
    assert output.read_bytes().startswith(b"%PDF")
