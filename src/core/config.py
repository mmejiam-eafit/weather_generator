"""Grid constants and environment-tunable settings (see docs/model_design.md §2.2, §2.8).

Grid constants describe the three raw archive grids (wave, wind, ocean) and
are fixed facts about the on-disk data, not tunable parameters. ``Settings``
holds paths and default hyperparameters that a caller may override via
``WGEN_``-prefixed environment variables.
"""

from pathlib import Path
from typing import Final

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict

WAVE_GRID_SHAPE: Final[tuple[int, int]] = (621, 1440)
WIND_GRID_SHAPE: Final[tuple[int, int]] = (721, 1440)
OCEAN_GRID_SHAPE: Final[tuple[int, int]] = (1121, 2880)

WAVE_GRID_LAT_AT_ROW0: Final[float] = -75.0
WIND_GRID_LAT_AT_ROW0: Final[float] = -90.0
# OCEAN_GRID_LAT_AT_ROW0 is intentionally unset: the ocean grid's row/lat
# orientation has not been empirically verified (see docs/model_design.md
# §2.7). Do not assume it matches the wave/wind convention by analogy — set
# this constant only after running the correlation study described there.

WAVE_GRID_LAT_STEP_DEG: Final[float] = 0.25
WIND_GRID_LAT_STEP_DEG: Final[float] = 0.25
OCEAN_GRID_LAT_STEP_DEG: Final[float] = 0.125
GRID_LON_STEP_DEG: Final[float] = 0.25  # wave and wind; ocean is 0.125

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

    Parameters
    ----------
    grid : str
        One of ``"wave"``, ``"wind"``. ``"ocean"`` is not yet supported: its
        row/latitude orientation is unverified (see §2.7).

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
            "unverified (see docs/model_design.md §2.7)."
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
