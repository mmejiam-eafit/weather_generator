"""Archive path construction and low-level I/O (see docs/model_design.md §2.1, §2.3, §5.1).

Reused by every downstream module that touches the raw archive or persists
model artifacts. Do not reimplement path construction, compass-code
decoding, or snapshot loading elsewhere — import from here.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import numpy as np

from core.archive_eras import era_for_timestamp

# Continuous fields stored as float16 on disk (see docs/model_design.md §2.3, §10).
_CONTINUOUS_VARIABLES: Final[frozenset[str]] = frozenset(
    {"sigwh", "wsh", "wsp", "pwh", "pwp", "pswh", "pswp", "was", "sst", "ocs"}
)

# Direction fields stored as int8 16-point compass codes, keyed to the code that
# represents North (see docs/model_design.md §2.3). `wad` is still origin-code
# decodable but unused downstream (`wad_come_from` is the source of truth for
# wind direction — see task 02). `ocd`'s `going_to` convention is provisional
# pending §2.7 ocean-grid re-verification, but the *decode* (origin code `1`,
# same as `wad`) is settled regardless of that open question.
_DIRECTION_ORIGIN_CODES: Final[dict[str, int]] = {
    "wsd": 0,
    "pwd": 0,
    "pswd": 0,
    "wad": 1,
    "wad_come_from": 1,
    "ocd": 1,
}


def require_utc(t: datetime, name: str) -> None:
    """Raise ValueError if `t` is not UTC-aware (see docs/model_design.md §2.8)."""
    if t.tzinfo is None or t.utcoffset() != timezone.utc.utcoffset(t):
        raise ValueError(f"{name} must be UTC-aware, got {t!r}")


def snapshot_path(data_dir: Path, t: datetime, variable: str) -> Path:
    """Construct the on-disk path for one variable's snapshot (see docs/model_design.md §2.1).

    Parameters
    ----------
    data_dir : Path
        Root of the archive, e.g. ``./data``.
    t : datetime
        UTC-aware snapshot timestamp. Must fall on a 3-hour boundary.
    variable : str
        Variable name, e.g. ``"sigwh"``.

    Returns
    -------
    Path
        ``data_dir / "<YYYY>/<MM>/<DD>/<HH>" / "<variable>.npy"``.

    Raises
    ------
    ValueError
        If `t` is not UTC-aware.
    """
    require_utc(t, "t")
    return data_dir / f"{t:%Y/%m/%d/%H}" / f"{variable}.npy"


def enumerate_snapshots(t_start: datetime, t_end: datetime) -> Iterator[datetime]:
    """Yield UTC snapshot timestamps at each era's native cadence over ``[t_start, t_end)``.

    (see docs/model_design.md §2.2, §5.1)

    Cadence is looked up per-timestamp via `core.archive_eras.era_for_timestamp`
    (6-hourly in era 1, 3-hourly in eras 2/3), so a window entirely within
    era 1 no longer enumerates the systematically-absent 03/09/15/21Z slots.
    A window spanning more than one era still enumerates cleanly here (mixed
    cadence) -- this function stays a pure, dumb generator; rejecting
    cross-era windows is `train.ingest.load_training_window`'s job, not
    this function's (see tasks/16-archive-grid-resolution-drift.md).

    Parameters
    ----------
    t_start : datetime
        UTC-aware inclusive start.
    t_end : datetime
        UTC-aware exclusive end.

    Yields
    ------
    datetime
        Each snapshot timestamp, spaced by its era's native cadence, starting
        at `t_start`.

    Raises
    ------
    ValueError
        If `t_start` or `t_end` is not UTC-aware, or if `t_start` is before
        the earliest known archive era.
    """
    require_utc(t_start, "t_start")
    require_utc(t_end, "t_end")
    t = t_start
    while t < t_end:
        yield t
        t += timedelta(seconds=era_for_timestamp(t).cadence_seconds)


def decode_compass_code(codes: np.ndarray, origin_code: int) -> np.ndarray:
    """Decode int8 16-point compass codes to degrees (see docs/model_design.md §2.3).

    Parameters
    ----------
    codes : (M, N) int8
        Compass codes as loaded from disk.
    origin_code : int
        Value of `codes` that represents North (0°). Typically 0 or 1.

    Returns
    -------
    degrees : (M, N) float64
        Native on-disk convention for that field (coming_from or going_to —
        see §2.3); convention handling is the caller's responsibility, not
        this function's. CW from North, in ``[0, 360)``.
    """
    return ((codes.astype(np.int32) - origin_code) * 22.5).astype(np.float64) % 360.0


def load_snapshot(data_dir: Path, t: datetime, variable: str) -> np.ndarray:
    """Load and decode one variable's raw snapshot array (see docs/model_design.md §2.3, §5.1).

    Continuous fields are upcast from ``float16`` to ``float64``. Direction
    fields with a verified compass origin are decoded to degrees via
    `decode_compass_code`. `float16` and `int8` never leave this function.

    Parameters
    ----------
    data_dir : Path
        Root of the archive.
    t : datetime
        UTC-aware snapshot timestamp.
    variable : str
        Variable name, e.g. ``"sigwh"`` or ``"wad"``.

    Returns
    -------
    np.ndarray
        float64 array: physical units for continuous fields, degrees for
        direction fields.

    Raises
    ------
    ValueError
        If `variable` is neither a known direction field (in
        `_DIRECTION_ORIGIN_CODES`) nor a known continuous field (in
        `_CONTINUOUS_VARIABLES`). Verify its on-disk dtype and compass
        origin (if directional) per §2.3/§2.7 before loading it.
    """
    path = snapshot_path(data_dir, t, variable)
    raw: np.ndarray = np.load(path)
    if variable in _DIRECTION_ORIGIN_CODES:
        return decode_compass_code(raw, _DIRECTION_ORIGIN_CODES[variable])
    if variable in _CONTINUOUS_VARIABLES:
        return raw.astype(np.float64)
    raise ValueError(
        f"unknown decoding convention for variable {variable!r}; verify its on-disk "
        "dtype and compass origin (if directional) per docs/model_design.md §2.3/§2.7 "
        "before loading it."
    )


def load_raw(data_dir: Path, t: datetime, variable: str) -> np.ndarray:
    """Load one variable's raw on-disk array with no decoding (see docs/model_design.md §5.1).

    Parameters
    ----------
    data_dir : Path
        Root of the archive.
    t : datetime
        UTC-aware snapshot timestamp.
    variable : str
        Variable name, e.g. ``"sigwh"`` or ``"wad"``.

    Returns
    -------
    np.ndarray
        Untouched on-disk dtype — int8 for direction fields, float16 for
        continuous fields. Callers decode/upcast themselves.
    """
    raw: np.ndarray = np.load(snapshot_path(data_dir, t, variable))
    return raw
