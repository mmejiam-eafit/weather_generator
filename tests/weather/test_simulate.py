"""Tests for weather.simulate (see docs/model_design.md §5.8, §8.4)."""

from datetime import timedelta

import numpy as np
import pytest
from numpy.testing import assert_allclose

from weather.model import WeatherModel
from weather.simulate import sample, sample_batch, simulate_field


def _mid_grid_point(model: WeatherModel) -> tuple[float, float]:
    reprojector = model.preprocess_.named_steps["reproject"]
    lats, lons = reprojector.target_lats_, reprojector.target_lons_
    return float(lats[len(lats) // 2]), float(lons[len(lons) // 2])


def test_batch_matches_single_point(fitted_tier1_model: WeatherModel) -> None:
    """sample_batch([lat],[lon],[t],var)[0] == sample(lat,lon,t,var) given the same seed."""
    model = fitted_tier1_model
    lat, lon = _mid_grid_point(model)
    t = model.t_start + timedelta(hours=6)

    batch_value = sample_batch(
        model,
        lats=np.array([lat]),
        lons=np.array([lon]),
        times=np.array([t.replace(tzinfo=None)], dtype="datetime64[s]"),
        variables="sigwh",
        seed=42,
    )[0]
    single_value = sample(model, lat=lat, lon=lon, t=t, variable="sigwh", seed=42)

    assert_allclose(batch_value, single_value)


def test_seed_reproducibility(fitted_tier1_model: WeatherModel) -> None:
    """Same seed, same query -> byte-exact output."""
    model = fitted_tier1_model
    lat, lon = _mid_grid_point(model)
    t = model.t_start + timedelta(hours=9)

    first = sample(model, lat=lat, lon=lon, t=t, variable="sigwh", seed=7)
    second = sample(model, lat=lat, lon=lon, t=t, variable="sigwh", seed=7)

    assert first == second


def test_different_seeds_differ(fitted_tier1_model: WeatherModel) -> None:
    """Different seeds produce different (non-degenerate) samples."""
    model = fitted_tier1_model
    lat, lon = _mid_grid_point(model)
    t = model.t_start + timedelta(hours=12)

    values = [sample(model, lat=lat, lon=lon, t=t, variable="sigwh", seed=s) for s in range(10)]
    assert len(set(values)) > 1


def test_sample_batch_multiple_variables_shape(fitted_tier1_model: WeatherModel) -> None:
    """Requesting multiple variables returns shape (N, len(variables))."""
    model = fitted_tier1_model
    lat, lon = _mid_grid_point(model)
    times = np.array(
        [(model.t_start + timedelta(hours=3 * i)).replace(tzinfo=None) for i in range(4)],
        dtype="datetime64[s]",
    )

    result = sample_batch(
        model,
        lats=np.full(4, lat),
        lons=np.full(4, lon),
        times=times,
        variables=["sigwh", "wsp"],
        seed=1,
    )
    assert result.shape == (4, 2)


def test_sample_batch_points_at_same_time_share_field(fitted_tier1_model: WeatherModel) -> None:
    """Two points at the same query time are drawn from the same underlying field.

    Sanity check for the shared-per-unique-time OU/DMD draw: querying the
    same (lat, lon, t) twice within one batch call must give identical
    values, since both entries fall in the same unique-time group.
    """
    model = fitted_tier1_model
    lat, lon = _mid_grid_point(model)
    t = (model.t_start + timedelta(hours=15)).replace(tzinfo=None)

    result = sample_batch(
        model,
        lats=np.array([lat, lat]),
        lons=np.array([lon, lon]),
        times=np.array([t, t], dtype="datetime64[s]"),
        variables="sigwh",
        seed=3,
    )
    assert_allclose(result[0], result[1])


def test_simulate_field_shape_and_variables(fitted_tier1_model: WeatherModel) -> None:
    """simulate_field returns the requested variables at 3h cadence over the window."""
    model = fitted_tier1_model
    t_start = model.t_start
    t_end = t_start + timedelta(hours=24)

    result = simulate_field(model, t_start, t_end, variables=["sigwh", "was"], seed=5)

    assert set(result.data_vars) == {"sigwh", "was"}
    assert result["sigwh"].shape[0] == 8  # 24h / 3h cadence
    assert result["sigwh"].dims == ("time", "wave_lat", "wave_lon")


def test_simulate_field_seed_reproducibility(fitted_tier1_model: WeatherModel) -> None:
    """Same seed, same window -> byte-exact field trajectory."""
    model = fitted_tier1_model
    t_start = model.t_start
    t_end = t_start + timedelta(hours=12)

    first = simulate_field(model, t_start, t_end, variables=["sigwh"], seed=11)
    second = simulate_field(model, t_start, t_end, variables=["sigwh"], seed=11)

    assert_allclose(first["sigwh"].values, second["sigwh"].values)


def test_simulate_field_requires_utc(fitted_tier1_model: WeatherModel) -> None:
    """A naive t_start raises ValueError rather than silently attaching UTC."""
    model = fitted_tier1_model
    with pytest.raises(ValueError, match="UTC-aware"):
        simulate_field(
            model,
            model.t_start.replace(tzinfo=None),
            model.t_end,
            variables=["sigwh"],
        )


def _time64(model: WeatherModel, hours: int) -> np.datetime64:
    return np.datetime64((model.t_start + timedelta(hours=hours)).replace(tzinfo=None), "s")


def test_sample_batch_interpolates_heights_in_log_energy_space(
    fitted_tier1_model: WeatherModel,
) -> None:
    """Halfway between two grid nodes, H is their geometric mean (§5.8 steps 5–6).

    The sampler interpolates logE = log(H²) and exponentiates afterwards, so
    the midpoint is exp of the mean log, not the arithmetic mean of heights.
    All three points share one query time, hence one field.
    """
    model = fitted_tier1_model
    reprojector = model.preprocess_.named_steps["reproject"]
    lat = float(reprojector.target_lats_[3])
    lon_a, lon_b = (float(v) for v in reprojector.target_lons_[2:4])
    t = _time64(model, 6)

    node_a, node_b, midpoint = sample_batch(
        model,
        lats=np.full(3, lat),
        lons=np.array([lon_a, lon_b, 0.5 * (lon_a + lon_b)]),
        times=np.full(3, t),
        variables="sigwh",
        seed=4,
    )

    assert_allclose(midpoint, np.sqrt(node_a * node_b), rtol=1e-9)


def test_sample_batch_physical_ranges(fitted_tier1_model: WeatherModel) -> None:
    """Heights and speeds are non-negative, directions in [0, 360) (§6 check 13)."""
    model = fitted_tier1_model
    reprojector = model.preprocess_.named_steps["reproject"]
    rng = np.random.default_rng(0)
    n = 40
    lats = rng.uniform(reprojector.target_lats_[1], reprojector.target_lats_[-1], size=n)
    lons = rng.uniform(reprojector.target_lons_[1], reprojector.target_lons_[-1], size=n)
    times = np.array([_time64(model, 3 * int(k)) for k in rng.integers(0, 20, size=n)])

    values = sample_batch(model, lats, lons, times, ["sigwh", "wsh", "was", "wsd"], seed=2)

    assert np.all(np.isfinite(values))
    assert np.all(values[:, :3] >= 0.0)
    assert np.all((values[:, 3] >= 0.0) & (values[:, 3] < 360.0))


def test_simulate_field_is_nan_on_land_and_finite_on_ocean(
    fitted_tier1_model: WeatherModel,
) -> None:
    """The ALS's imputed land values never reach the output (§5.8 step 3)."""
    model = fitted_tier1_model
    field = simulate_field(
        model, model.t_start, model.t_start + timedelta(hours=12), ["sigwh", "wsh"], seed=0
    )
    ocean = model.ocean_mask_

    for variable in ("sigwh", "wsh"):
        values = field[variable].values
        assert np.all(np.isnan(values[:, ~ocean]))
        assert np.all(np.isfinite(values[:, ocean]))


def test_sample_batch_rejects_times_before_the_training_window(
    fitted_tier1_model: WeatherModel,
) -> None:
    """The model extrapolates forward only; earlier times raise (§5.8)."""
    model = fitted_tier1_model
    lat, lon = _mid_grid_point(model)

    with pytest.raises(ValueError, match="before the training window"):
        sample_batch(
            model,
            np.array([lat]),
            np.array([lon]),
            np.array([_time64(model, -3)]),
            "sigwh",
            seed=0,
        )


def test_sample_batch_rejects_an_unmodeled_variable(fitted_tier1_model: WeatherModel) -> None:
    """`wad` is not modeled (wad_come_from is the source of truth, §9 gotcha 5a)."""
    model = fitted_tier1_model
    lat, lon = _mid_grid_point(model)

    with pytest.raises(ValueError, match="not modeled"):
        sample_batch(model, np.array([lat]), np.array([lon]), np.array([_time64(model, 0)]), "wad")
