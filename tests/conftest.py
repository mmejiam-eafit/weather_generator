"""Shared pytest fixtures: small synthetic archives (see AGENTS.md testing protocol)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from core.config import grid_coordinates
from core.io import snapshot_path
from weather.model import WeatherModel

_N_TIMES = 4
_N_LAT = 3
_N_LON = 5

# Tier-1 end-to-end fixture (see docs/model_design.md §3.8, used by tests/weather/).
# Both patches: 6 rows x 5 columns from the canonical row 0 (-90.00 .. -88.75,
# 0.00 .. 1.00). Wind shares the wave grid's lines, as in the real archive (§2.2),
# so `Reprojector` passes it through instead of interpolating.
TIER1_T0: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc)
TIER1_WAVE_SHAPE = (6, 5)
TIER1_WIND_SHAPE = (6, 5)
TIER1_N_TIMES = 60
# `wad_come_from`, not `wad`: wad_come_from is the source of truth for wind
# direction (see docs/model_design.md §3.7, §9 gotcha 5a). `wad` is still
# written to disk by write_tier1_archive below (as its true 180°-shifted
# counterpart) purely so train.ingest's wad/wad_come_from consistency check
# (§5.1) has something to validate — it is not in this modeled-variable list.
TIER1_VARIABLES = ["sigwh", "wsh", "wsp", "wsd", "was", "wad_come_from"]
# The seven tier-1 tensors those variables become (docs/model_design.md §3.8, §10).
TIER1_TENSORS = {"logE_sigwh", "logE_wsh", "T_wsp", "u_wsea", "v_wsea", "u_wind", "v_wind"}
# Land: sigwh, every wave field and `was` NaN at every snapshot, so the land
# mask excludes it. Absent partition: wsh/wsp NaN and wsd the -1 missing code at an
# ocean cell every fourth snapshot -- must stay ocean (task 17, §5.1).
TIER1_LAND_CELL = (0, 0)
TIER1_ABSENT_CELL = (2, 3)


def _write_snapshot(data_dir: Path, t: datetime, variable: str, array: np.ndarray) -> None:
    path = snapshot_path(data_dir, t, variable)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def write_tier1_archive(data_dir: Path, rng: np.random.Generator) -> None:
    """Write a small synthetic tier-1 archive covering wave and wind grids.

    Includes one land cell and one ocean cell whose wind-sea partition is
    intermittently absent (`TIER1_LAND_CELL`, `TIER1_ABSENT_CELL`).
    """
    for i in range(TIER1_N_TIMES):
        t = TIER1_T0 + timedelta(hours=3 * i)
        sigwh = rng.exponential(2.0, size=TIER1_WAVE_SHAPE)
        wsh = rng.exponential(1.5, size=TIER1_WAVE_SHAPE)
        wsp = rng.normal(8.0, 1.0, size=TIER1_WAVE_SHAPE)
        wsd = rng.integers(1, 17, size=TIER1_WAVE_SHAPE)
        for field in (sigwh, wsh, wsp):
            field[TIER1_LAND_CELL] = np.nan
        wsd[TIER1_LAND_CELL] = -1
        if i % 4 == 0:
            wsh[TIER1_ABSENT_CELL] = np.nan
            wsp[TIER1_ABSENT_CELL] = np.nan
            wsd[TIER1_ABSENT_CELL] = -1
        _write_snapshot(data_dir, t, "sigwh", sigwh.astype(np.float16))
        _write_snapshot(data_dir, t, "wsh", wsh.astype(np.float16))
        _write_snapshot(data_dir, t, "wsp", wsp.astype(np.float16))
        _write_snapshot(data_dir, t, "wsd", wsd.astype(np.int8))
        was = rng.exponential(6.0, size=TIER1_WIND_SHAPE)
        was[TIER1_LAND_CELL] = np.nan
        _write_snapshot(data_dir, t, "was", was.astype(np.float16))
        wad_raw = rng.integers(1, 17, size=TIER1_WIND_SHAPE).astype(np.int8)
        _write_snapshot(data_dir, t, "wad", wad_raw)
        # wad_come_from is wad's true 180deg-shifted counterpart (both origin_code=1,
        # 16-point compass codes, so +180deg is exactly +8 steps mod 16) — must stay
        # exactly consistent or train.ingest's consistency check (§5.1) will fail.
        wad_come_from_raw = (((wad_raw.astype(np.int32) - 1 + 8) % 16) + 1).astype(np.int8)
        _write_snapshot(data_dir, t, "wad_come_from", wad_come_from_raw)


def small_tier1_model(seed: int = 0, **overrides: Any) -> WeatherModel:
    """An unfitted WeatherModel sized for `tier1_archive` (Tucker 40% of 6 x 5 is 2 x 2)."""
    params: dict[str, Any] = {
        "t_start": TIER1_T0,
        "t_end": TIER1_T0 + timedelta(hours=3 * TIER1_N_TIMES),
        "variables": TIER1_VARIABLES,
        "r_dmd": 8,
        "copula_lut_bins": 50,
        "lim_rank": 3,
        "tau_test_lags": (1, 2, 4),
        "n_bootstrap": 2,  # bands are not under test here; the bootstrap dominates fit time
        "random_state": seed,
    }
    params.update(overrides)
    return WeatherModel(**params)


def fit_small_tier1_model(data_dir: Path, seed: int = 0) -> WeatherModel:
    """Fit `small_tier1_model` on a tier1_archive."""
    model = small_tier1_model(seed)
    model.fit(data_dir)
    return model


@pytest.fixture
def tier1_archive(tmp_path: Path) -> Path:
    """A small synthetic tier-1 archive (wave+wind grids) rooted at tmp_path."""
    write_tier1_archive(tmp_path, np.random.default_rng(0))
    return tmp_path


@pytest.fixture
def fitted_tier1_model(tier1_archive: Path) -> WeatherModel:
    """A WeatherModel fit on the `tier1_archive` fixture."""
    return fit_small_tier1_model(tier1_archive)


@pytest.fixture
def small_wave_dataset() -> xr.Dataset:
    """A tiny synthetic wave-grid dataset: sigwh (scalar) and (wsh, wsd) (directional pair).

    Shaped like the output of `train.ingest.load_training_window` on the
    wave grid (see docs/model_design.md §5.1, §5.3): direction fields are
    already decoded to degrees, not raw compass codes.
    """
    rng = np.random.default_rng(0)
    lats, lons = grid_coordinates("wave")
    dims = ("time", "wave_lat", "wave_lon")
    shape = (_N_TIMES, _N_LAT, _N_LON)
    return xr.Dataset(
        {
            "sigwh": (dims, rng.uniform(0.5, 5.0, size=shape)),
            "wsh": (dims, rng.uniform(0.5, 5.0, size=shape)),
            "wsd": (dims, rng.uniform(0.0, 360.0, size=shape)),
        },
        coords={
            "time": np.arange(_N_TIMES),
            "wave_lat": lats[:_N_LAT],
            "wave_lon": lons[:_N_LON],
        },
    )


@pytest.fixture
def small_wind_dataset() -> xr.Dataset:
    """A tiny synthetic wind-grid dataset: (was, wad), a directional pair.

    Shaped like the output of `train.ingest.load_training_window` on the
    wind grid (see docs/model_design.md §5.1, §5.3).
    """
    rng = np.random.default_rng(1)
    lats, lons = grid_coordinates("wind")
    dims = ("time", "wind_lat", "wind_lon")
    shape = (_N_TIMES, _N_LAT, _N_LON)
    return xr.Dataset(
        {
            "was": (dims, rng.uniform(1.0, 20.0, size=shape)),
            "wad": (dims, rng.uniform(0.0, 360.0, size=shape)),
        },
        coords={
            "time": np.arange(_N_TIMES),
            "wind_lat": lats[:_N_LAT],
            "wind_lon": lons[:_N_LON],
        },
    )
