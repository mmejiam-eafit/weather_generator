"""Load a training window from the raw archive (see docs/model_design.md §2.5, §5.1)."""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from core.config import VARIABLE_GRID, grid_coordinates
from core.io import decode_compass_code, enumerate_snapshots, load_raw, load_snapshot, snapshot_path

_ABORT_GAP_SNAPSHOTS = 8  # 24 h at 3 h cadence (see §5.1)
_WARN_GAP_SNAPSHOTS = 2  # 6 h at 3 h cadence


def _check_wad_wad_come_from_consistency(data_dir: Path, t: datetime) -> None:
    """Guard against upstream archive convention drift (§3.7, §5.1)."""
    wad = decode_compass_code(load_raw(data_dir, t, "wad"), origin_code=1)
    wad_from = decode_compass_code(load_raw(data_dir, t, "wad_come_from"), origin_code=1)
    diff = np.abs(((wad - wad_from - 180.0 + 180.0) % 360.0) - 180.0)  # min circular distance
    max_dev = float(np.nanmax(diff))
    if max_dev > 5.0:  # tolerance for 22.5° discretization + rounding
        raise ValueError(
            f"wad and wad_come_from do not differ by 180° at {t.isoformat()}. "
            f"Max deviation: {max_dev:.2f}°. Archive convention may have changed."
        )
    print(
        f"wad/wad_come_from consistency check passed at {t.isoformat()}: "
        f"max deviation {max_dev:.2f}°",
        file=sys.stderr,
    )


def _find_wad_pair_snapshot(
    data_dir: Path, times: list[datetime], *, reverse: bool = False
) -> datetime | None:
    """Locate a snapshot in `times` with both `wad` and `wad_come_from` on disk (§5.1).

    Parameters
    ----------
    data_dir : Path
        Root of the archive.
    times : list[datetime]
        Candidate snapshot timestamps, e.g. the training window's full
        snapshot list from `core.io.enumerate_snapshots`.
    reverse : bool
        If False (default), return the earliest qualifying snapshot. If
        True, return the latest.

    Returns
    -------
    datetime | None
        The qualifying timestamp, or `None` if no snapshot in `times` has
        both files present (e.g. a synthetic test archive that never wrote
        them) — in that case `_check_wad_wad_come_from_consistency` has
        nothing to check and is skipped rather than failing on a missing
        file.
    """
    candidates = reversed(times) if reverse else times
    for t in candidates:
        if (
            snapshot_path(data_dir, t, "wad").exists()
            and snapshot_path(data_dir, t, "wad_come_from").exists()
        ):
            return t
    return None


def _coordinates_for_shape(grid: str, n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice the canonical grid coordinates down to an on-disk array's actual shape.

    The on-disk shape is trusted over the canonical `core.config` constants:
    it is assumed to be the low-index corner (rows/cols starting at 0) of the
    canonical grid, which holds both for the real archive (exact match) and
    for small synthetic test patches.
    """
    lats, lons = grid_coordinates(grid)
    if n_rows > lats.shape[0] or n_cols > lons.shape[0]:
        raise ValueError(
            f"on-disk {grid} grid shape {(n_rows, n_cols)} exceeds the canonical "
            f"shape {(lats.shape[0], lons.shape[0])} (see docs/model_design.md §2.2)."
        )
    return lats[:n_rows], lons[:n_cols]


def load_training_window(
    data_dir: Path,
    t_start: datetime,
    t_end: datetime,
    variables: list[str],
) -> xr.Dataset:
    """Load raw archive files into an xarray.Dataset over ``(time, lat, lon)`` per grid.

    (see docs/model_design.md §5.1)

    Each variable is loaded on its native grid (see §2.2) via `core.io.
    load_snapshot`, which upcasts continuous fields to float64 and decodes
    direction fields to degrees. A missing snapshot file is logged to
    stderr and filled with NaN for that variable. A contiguous run of 2-7
    missing snapshots (6-21 h) triggers a warning but continues; a run of 8
    or more (24 h+) aborts, since the OU fit assumes near-continuous
    sampling.

    Before the per-variable load loop begins, `_check_wad_wad_come_from_
    consistency` runs unconditionally on the earliest window snapshot with
    both `wad` and `wad_come_from` present, and again on the latest such
    snapshot, to guard against upstream archive convention drift (§3.7,
    §5.1). This runs regardless of whether `"wad"`/`"wad_come_from"` are in
    `variables` — it is an archive-integrity guard, not a modeled-variable
    concern. If neither field is present anywhere in the window (e.g. a
    synthetic archive that never wrote them), the check is silently
    skipped.

    Parameters
    ----------
    data_dir : Path
        Root of the archive, e.g. ``./data``.
    t_start : datetime
        UTC-aware inclusive start of the training window.
    t_end : datetime
        UTC-aware exclusive end of the training window.
    variables : list[str]
        Variable names to load, e.g. ``["sigwh", "wsh", "wsp", "wsd", "was", "wad"]``.

    Returns
    -------
    xr.Dataset
        One DataArray per variable, dims ``("time", "<grid>_lat", "<grid>_lon")``
        where ``<grid>`` is the variable's native grid (``"wave"`` or ``"wind"``),
        plus a shared ``"time"`` coordinate.

    Raises
    ------
    ValueError
        If `t_start`/`t_end` is not UTC-aware, if a variable is not in the
        §10 variable table, if a variable's grid is not yet supported
        (e.g. an ocean-grid variable before §2.7 verification), or if `wad`
        and `wad_come_from` do not differ by 180° at the first or last
        checked snapshot (§5.1: archive convention drift).
    RuntimeError
        If a variable has a contiguous gap of 24 hours or more, or if every
        snapshot for a variable is missing across the whole window.
    """
    times = list(enumerate_snapshots(t_start, t_end))
    n_times = len(times)

    first_wad_t = _find_wad_pair_snapshot(data_dir, times)
    if first_wad_t is not None:
        _check_wad_wad_come_from_consistency(data_dir, first_wad_t)
        last_wad_t = _find_wad_pair_snapshot(data_dir, times, reverse=True)
        if last_wad_t is not None and last_wad_t != first_wad_t:
            _check_wad_wad_come_from_consistency(data_dir, last_wad_t)

    data_vars: dict[str, xr.DataArray] = {}

    for variable in variables:
        try:
            grid = VARIABLE_GRID[variable]
        except KeyError:
            raise ValueError(
                f"unknown variable {variable!r}; see docs/model_design.md §10 for "
                "the variable table."
            ) from None

        values: np.ndarray | None = None
        contiguous_gap = 0
        for i, t in enumerate(times):
            try:
                snapshot = load_snapshot(data_dir, t, variable)
            except FileNotFoundError:
                print(f"missing snapshot: {variable} at {t.isoformat()}", file=sys.stderr)
                contiguous_gap += 1
                if contiguous_gap == _WARN_GAP_SNAPSHOTS:
                    warnings.warn(
                        f"{variable} has a gap of {contiguous_gap * 3}h+ ending near "
                        f"{t.isoformat()}",
                        stacklevel=2,
                    )
                if contiguous_gap >= _ABORT_GAP_SNAPSHOTS:
                    raise RuntimeError(
                        f"{variable} has a contiguous gap of {contiguous_gap * 3}h "
                        f"ending at {t.isoformat()}; aborting (see docs/model_design.md §5.1)."
                    ) from None
                continue

            contiguous_gap = 0
            if values is None:
                values = np.full((n_times, *snapshot.shape), np.nan, dtype=np.float64)
            values[i] = snapshot

        if values is None:
            raise RuntimeError(
                f"no snapshots found for variable {variable!r} in "
                f"[{t_start.isoformat()}, {t_end.isoformat()})."
            )

        lats, lons = _coordinates_for_shape(grid, values.shape[1], values.shape[2])
        data_vars[variable] = xr.DataArray(
            values,
            dims=("time", f"{grid}_lat", f"{grid}_lon"),
            coords={f"{grid}_lat": lats, f"{grid}_lon": lons},
        )

    # All snapshot timestamps were already validated as UTC-aware by
    # enumerate_snapshots; np.datetime64 has no timezone of its own, so the
    # conversion below intentionally drops tzinfo (UTC is implied throughout
    # this codebase, see §2.8) rather than signaling a real ambiguity.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="no explicit representation of timezones")
        time_coord = np.array(times, dtype="datetime64[s]")
    return xr.Dataset(data_vars, coords={"time": time_coord})
