"""Tests for core.config (see docs/model_design.md §2.2, §2.8)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from core.config import (
    GRID_LON_STEP_DEG,
    OCEAN_GRID_LAT_STEP_DEG,
    OCEAN_GRID_SHAPE,
    SNAPSHOT_CADENCE_SECONDS,
    TIME_AXIS,
    VARIABLE_GRID,
    WAVE_GRID_LAT_AT_ROW0,
    WAVE_GRID_LAT_STEP_DEG,
    WAVE_GRID_SHAPE,
    WIND_GRID_LAT_AT_ROW0,
    WIND_GRID_SHAPE,
    Settings,
    canonical_dims,
    check_canonical,
    grid_coordinates,
)


def test_grid_shapes_match_archive() -> None:
    """Shapes must match the documented (rows, cols) per grid (§2.2)."""
    assert WAVE_GRID_SHAPE == (721, 1440)
    assert WIND_GRID_SHAPE == (721, 1440)
    assert OCEAN_GRID_SHAPE == (1121, 2880)


def test_row0_latitudes_and_steps() -> None:
    """Row-0 latitudes and resolution steps per the empirically verified §2.7 convention."""
    assert WAVE_GRID_LAT_AT_ROW0 == -90.0
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
    """Wave grid: row 0 = -90°, 0.25° steps, 721 rows x 1440 cols (§2.2, era 3)."""
    lats, lons = grid_coordinates("wave")
    assert lats.shape == (721,)
    assert lons.shape == (1440,)
    assert_allclose(lats[0], -90.0)
    assert_allclose(lats[-1], -90.0 + 720 * 0.25)
    assert_allclose(lons[0], 0.0)
    assert_allclose(lons[-1], 359.75)


def test_grid_coordinates_wind() -> None:
    """Wind grid: row 0 = -90°, 0.25° steps, 721 rows x 1440 cols (§2.2)."""
    lats, lons = grid_coordinates("wind")
    assert lats.shape == (721,)
    assert_allclose(lats[0], -90.0)
    assert_allclose(lats[-1], 90.0)


def test_grid_coordinates_ocean_raises() -> None:
    """Ocean grid orientation is confirmed (open_questions.md §1.1), but the
    grid is non-linear (Mercator/tripolar) with no 2D coordinate support
    yet; requesting it still raises."""
    with pytest.raises(ValueError, match="not a linear"):
        grid_coordinates("ocean")


def _dataset(dims: tuple[str, str, str], shape: tuple[int, int, int]) -> xr.Dataset:
    return xr.Dataset({"sigwh": (dims, np.zeros(shape))})


def test_canonical_dims_names_the_grid() -> None:
    """`canonical_dims` spells the per-grid dim tuple (see CANONICAL_DIM_TEMPLATE)."""
    assert canonical_dims("wave") == ("time", "wave_lat", "wave_lon")
    assert canonical_dims("wind") == ("time", "wind_lat", "wind_lon")
    assert TIME_AXIS == 0


def test_check_canonical_accepts_time_first() -> None:
    """The production order passes unchanged."""
    check_canonical(_dataset(("time", "wave_lat", "wave_lon"), (4, 3, 5)))


def test_check_canonical_rejects_time_last() -> None:
    """`(lat, lon, time)` must raise, not silently reduce over the wrong axis.

    This is the regression guard for the whole canonical-order convention:
    the transformers infer the time axis positionally, so before this check
    a transposed Dataset produced a wrongly-shaped `sigma_cell_`/LUT with no
    error at all (see `core.config.CANONICAL_DIM_TEMPLATE`).
    """
    transposed = _dataset(("wave_lat", "wave_lon", "time"), (3, 5, 4))
    with pytest.raises(ValueError, match="expected canonical"):
        check_canonical(transposed)


def test_check_canonical_reports_variable_and_context() -> None:
    """The error names the offending variable and the caller."""
    transposed = _dataset(("wave_lat", "wave_lon", "time"), (3, 5, 4))
    with pytest.raises(ValueError, match="'sigwh'.* in GaussianCopula.fit"):
        check_canonical(transposed, context="GaussianCopula.fit")


def test_check_canonical_rejects_non_3d() -> None:
    """A 2D variable is not a field array and must not slip through."""
    flat = xr.Dataset({"sigwh": (("wave_lat", "wave_lon"), np.zeros((3, 5)))})
    with pytest.raises(ValueError, match="expected canonical"):
        check_canonical(flat)
