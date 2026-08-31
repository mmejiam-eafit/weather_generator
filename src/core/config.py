"""Grid constants and environment-tunable settings (see docs/model_design.md §2.2, §2.8).

Grid constants describe the three raw archive grids (wave, wind, ocean) and
are fixed facts about the on-disk data, not tunable parameters. ``Settings``
holds paths and default hyperparameters that a caller may override via
``WGEN_``-prefixed environment variables.

**These constants describe era 3** (2025-10-27 onward, current and
open-ended -- see `core.archive_eras.ARCHIVE_ERAS`), the era all new/future
data and production training use. The real archive is *not* one uniform
shape/cadence across its full history: era 1 (2022-2023) uses a coarser
grid at 6-hourly cadence, and era 2 (2023-10-02 to 2025-10-26) used a
different wave grid (shape ``(621, 1440)``, ``lat0=-75.0``) before the era 3
cutover (see docs/model_design.md §2.2.1 and
tasks/16-archive-grid-resolution-drift.md). ``WIND_GRID_SHAPE``,
``OCEAN_GRID_SHAPE``, and ``SNAPSHOT_CADENCE_SECONDS`` are identical across
era 2 and era 3, so they needed no change; only the wave-grid constants
below track era 3 specifically.

**Known, accepted tradeoff:** `grid_coordinates` below (and
`train.ingest._coordinates_for_shape`, its only caller) has no per-era
awareness -- loading an era 1 or era 2 window still labels that window's
wave-grid latitudes using *these* (era 3) constants via corner-slicing. For
era 2 this mislabels the true ``[-75, 80]`` range as ``[-90, 65]`` (off by a
constant 15°); era 1 uses a coarser, differently-stepped grid entirely, so
its mislabeling is worse and non-uniform. This is deliberate, not a bug:
era 3 is open-ended and is what all new/future data and production training
use, so its accuracy is what matters going forward -- era 1/era 2 were never
correctly labeled for eras other than their own under the old constants
either; this only changes which era is "the accurate one by construction."
See `core.archive_eras.era_for_timestamp` for era-aware shape/cadence facts;
new date-aware code should go through it instead of importing these flat
constants directly.
"""

from pathlib import Path
from typing import Final

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict

WAVE_GRID_SHAPE: Final[tuple[int, int]] = (721, 1440)  # era 3 -- see module docstring
WIND_GRID_SHAPE: Final[tuple[int, int]] = (721, 1440)  # era 2 & 3 -- see module docstring
OCEAN_GRID_SHAPE: Final[tuple[int, int]] = (1121, 2880)  # era 2 & 3 -- see module docstring

WAVE_GRID_LAT_AT_ROW0: Final[float] = -90.0
WIND_GRID_LAT_AT_ROW0: Final[float] = -90.0
# OCEAN_GRID_LAT_AT_ROW0 is intentionally unset -- but not because the
# orientation is unknown. docs/open_questions.md §1.1 confirms it directly
# (row 0 = south, column 0 = 0°E increasing eastward, matching the wave/wind
# convention exactly), via a real RTOFS NetCDF Latitude/Longitude lookup
# remapped through hycom_satocean_idx.npy. What blocks this constant is a
# different problem: the ocean grid's row spacing is non-uniform in degrees
# (Mercator projection south of ~47°N; ~0.10°/row near -65°, ~0.125°/row near
# -50..-27°) and the region north of ~47°N is not zonal at all (tripolar
# patch) -- so it cannot be expressed as `lat0 + row * step` the way the
# wave/wind grids are. Setting a single lat0 here would be actively wrong,
# not merely unverified. Real per-cell 2D coordinate data is required
# instead; see the derivation in docs/open_questions.md §1.1.
#
# WAVE_GRID_LAT_AT_ROW0 is era 3's empirically verified value (identical to
# WIND_GRID_LAT_AT_ROW0's convention — see
# notebooks/verify_era1_era3_grid_coordinates.ipynb, 1.0000 land-mask
# agreement). `core.archive_eras.ArchiveEra` intentionally has no lat0/step
# fields of its own — whether they also hold for era 1/era 2 is a known,
# accepted gap (see the module docstring's tradeoff note above).
# `grid_coordinates` below is used for every era's shape-slicing today --
# correctly for era 3, mislabeled for era 1/era 2.

WAVE_GRID_LAT_STEP_DEG: Final[float] = 0.25
WIND_GRID_LAT_STEP_DEG: Final[float] = 0.25
OCEAN_GRID_LAT_STEP_DEG: Final[float] = 0.125
GRID_LON_STEP_DEG: Final[float] = 0.25  # wave and wind; ocean is 0.125

# Era 2 and era 3 (identical) -- see module docstring. `core.archive_eras.
# era_for_timestamp(t).cadence_seconds` is the era-aware equivalent;
# `core.io.enumerate_snapshots` and `weather.model.WeatherModel._fit_impl`
# both already use it instead of this constant. Kept for external consumers
# (e.g. notebooks) pinned to the current cadence.
SNAPSHOT_CADENCE_SECONDS: Final[float] = 3.0 * 3600.0  # Δt = 3h (see docs/model_design.md §5)

# Which grid each archive variable's magnitude lives on (see docs/model_design.md
# §2.2, §10). `wad_come_from` is the source of truth for wind direction (§3.7,
# §9 gotcha 5a) — `core.io.load_snapshot` loads it like any other direction field.
VARIABLE_GRID: Final[dict[str, str]] = {
    "sigwh": "wave",
    "wsh": "wave",
    "wsp": "wave",
    "wsd": "wave",
    "pwh": "wave",
    "pwp": "wave",
    "pwd": "wave",
    "pswh": "wave",
    "pswp": "wave",
    "pswd": "wave",
    "was": "wind",
    "wad": "wind",
    "wad_come_from": "wind",
    "sst": "ocean",
    "ocs": "ocean",
    "ocd": "ocean",
}


def grid_coordinates(grid: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the 1D (lats, lons) coordinate arrays for a named grid.

    (see docs/model_design.md §2.2)

    Uses era 3's (current/production) lat0/lon-step convention for the wave
    grid, and the era2/era3-shared convention for the wind grid --
    unconditionally, for every archive era. Era 1/era 2 wave-grid windows are
    therefore mislabeled if loaded (an accepted tradeoff, not a bug -- see
    the module docstring and tasks/16-archive-grid-resolution-drift.md).

    Parameters
    ----------
    grid : str
        One of ``"wave"``, ``"wind"``. ``"ocean"`` is not yet supported --
        not because its orientation is unknown (docs/open_questions.md §1.1
        confirms row 0 = south, column 0 = 0°E increasing eastward), but
        because it is not a linear `lat0 + row * step` grid: row spacing is
        non-uniform (Mercator) south of ~47°N and non-zonal (tripolar patch)
        north of it. Real per-cell 2D coordinate data is required; see
        `OCEAN_GRID_LAT_AT_ROW0`'s comment above.

    Returns
    -------
    lats : np.ndarray, float64, row 0 = south per §2.2.
    lons : np.ndarray, float64, spanning ``[0°, 359.75° E]``.

    Raises
    ------
    ValueError
        If `grid` is ``"ocean"`` or otherwise unrecognized.
    """
    if grid == "wave":
        n_rows, n_cols = WAVE_GRID_SHAPE
        lat0, lat_step = WAVE_GRID_LAT_AT_ROW0, WAVE_GRID_LAT_STEP_DEG
    elif grid == "wind":
        n_rows, n_cols = WIND_GRID_SHAPE
        lat0, lat_step = WIND_GRID_LAT_AT_ROW0, WIND_GRID_LAT_STEP_DEG
    else:
        raise ValueError(
            f"grid {grid!r} is not supported; the ocean grid's orientation is "
            "confirmed (see docs/open_questions.md §1.1) but it is not a "
            "linear lat0+step grid (Mercator south of ~47°N, tripolar patch "
            "north of it) -- real per-cell 2D coordinate data is required "
            "and not yet implemented."
        )
    lats = lat0 + np.arange(n_rows, dtype=np.float64) * lat_step
    lons = np.arange(n_cols, dtype=np.float64) * GRID_LON_STEP_DEG
    return lats, lons


class Settings(BaseSettings):
    """Env-tunable paths and default hyperparameters (see docs/model_design.md §2.8).

    All fields are overridable via environment variables prefixed ``WGEN_``,
    e.g. ``WGEN_DATA_DIR=/mnt/archive``.
    """

    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")

    default_r_ocean: int = 300
    default_r_var: int = 6
    default_r_dmd: int = 100
    copula_lut_bins: int = 500

    model_config = SettingsConfigDict(env_prefix="WGEN_")
