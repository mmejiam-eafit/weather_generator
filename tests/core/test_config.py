"""Tests for core.config (see docs/model_design.md §2.2, §2.8)."""

from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from core.config import (
    GRID_LON_STEP_DEG,
    OCEAN_GRID_LAT_STEP_DEG,
    OCEAN_GRID_SHAPE,
    SNAPSHOT_CADENCE_SECONDS,
    VARIABLE_GRID,
    WAVE_GRID_LAT_AT_ROW0,
    WAVE_GRID_LAT_STEP_DEG,
    WAVE_GRID_SHAPE,
    WIND_GRID_LAT_AT_ROW0,
    WIND_GRID_SHAPE,
    Settings,
    grid_coordinates,
)


def test_grid_shapes_match_archive() -> None:
    """Shapes must match the documented (rows, cols) per grid (§2.2)."""
    assert WAVE_GRID_SHAPE == (621, 1440)
    assert WIND_GRID_SHAPE == (721, 1440)
    assert OCEAN_GRID_SHAPE == (1121, 2880)


def test_row0_latitudes_and_steps() -> None:
    """Row-0 latitudes and resolution steps per the empirically verified §2.7 convention."""
    assert WAVE_GRID_LAT_AT_ROW0 == -75.0
    assert WIND_GRID_LAT_AT_ROW0 == -90.0
    assert WAVE_GRID_LAT_STEP_DEG == 0.25
    assert OCEAN_GRID_LAT_STEP_DEG == 0.125
    assert GRID_LON_STEP_DEG == 0.25


def test_snapshot_cadence_is_three_hours() -> None:
    """The archive's 3-hourly cadence in seconds (§5)."""
    assert SNAPSHOT_CADENCE_SECONDS == 10800.0


def test_settings_defaults() -> None:
    """Settings default to ./data and ./models with documented hyperparameter defaults."""
    settings = Settings()
    assert settings.data_dir == Path("./data")
    assert settings.model_dir == Path("./models")
    assert settings.default_r_ocean == 300
    assert settings.default_r_var == 6
    assert settings.default_r_dmd == 100
    assert settings.copula_lut_bins == 500


def test_settings_env_override() -> None:
    """WGEN_-prefixed environment variables override Settings defaults."""
    import os

    os.environ["WGEN_DATA_DIR"] = "/mnt/archive"
    try:
        settings = Settings()
        assert settings.data_dir == Path("/mnt/archive")
    finally:
        del os.environ["WGEN_DATA_DIR"]


def test_variable_grid_covers_all_variables() -> None:
    """Every archive variable from the §10 table maps to one of the three grids."""
    assert VARIABLE_GRID["sigwh"] == "wave"
    assert VARIABLE_GRID["wsd"] == "wave"
    assert VARIABLE_GRID["was"] == "wind"
    assert VARIABLE_GRID["wad"] == "wind"
    assert VARIABLE_GRID["sst"] == "ocean"
    assert VARIABLE_GRID["ocd"] == "ocean"


def test_grid_coordinates_wave() -> None:
    """Wave grid: row 0 = -75°, 0.25° steps, 621 rows x 1440 cols (§2.2)."""
    lats, lons = grid_coordinates("wave")
    assert lats.shape == (621,)
    assert lons.shape == (1440,)
    assert_allclose(lats[0], -75.0)
    assert_allclose(lats[-1], -75.0 + 620 * 0.25)
    assert_allclose(lons[0], 0.0)
    assert_allclose(lons[-1], 359.75)


def test_grid_coordinates_wind() -> None:
    """Wind grid: row 0 = -90°, 0.25° steps, 721 rows x 1440 cols (§2.2)."""
    lats, lons = grid_coordinates("wind")
    assert lats.shape == (721,)
    assert_allclose(lats[0], -90.0)
    assert_allclose(lats[-1], 90.0)


def test_grid_coordinates_ocean_raises() -> None:
    """Ocean grid orientation is unverified; requesting it raises (§2.7)."""
    with pytest.raises(ValueError, match="unverified"):
        grid_coordinates("ocean")
