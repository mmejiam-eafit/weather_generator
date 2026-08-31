"""Archive era registry: which grid shapes and snapshot cadence apply on which
dates (see docs/model_design.md §2.2, §5.1, and
tasks/16-archive-grid-resolution-drift.md).

The raw archive is not one uniform grid/cadence across its full date range —
it has undergone two confirmed, clean, one-time shape/cadence transitions
(see `ARCHIVE_ERAS`). `core.config`'s flat `WAVE_GRID_SHAPE`/
`WAVE_GRID_LAT_AT_ROW0` describe era 3 specifically — current, open-ended,
and the era production training targets; era 2's wave grid differs (see
`ARCHIVE_ERAS` below). `WIND_GRID_SHAPE`/`OCEAN_GRID_SHAPE`/
`SNAPSHOT_CADENCE_SECONDS` are identical across era 2 and era 3, so they
carry no such caveat. Kept for backward compatibility
(`core.config.grid_coordinates`, `notebooks/verify_size_of_matrices.ipynb`).
New date-aware code should go through `era_for_timestamp` here instead of
importing those flat constants.

Deliberately has zero intra-`core` dependencies (duplicates a tiny UTC check
rather than importing `core.io.require_utc`), so `core.io` can import from
this module (`enumerate_snapshots`'s per-era cadence) without a circular import.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final


def _require_utc(t: datetime) -> None:
    """Raise ValueError if `t` is not UTC-aware.

    Duplicated from `core.io.require_utc` — see module docstring for why.
    """
    if t.tzinfo is None or t.utcoffset() != timezone.utc.utcoffset(t):
        raise ValueError(f"t must be UTC-aware, got {t!r}")


@dataclass(frozen=True, slots=True)
class ArchiveEra:
    """Fixed grid-shape/cadence facts for one contiguous stretch of the archive.

    Established by direct binary-search inspection of the real archive (see
    tasks/16-archive-grid-resolution-drift.md), not from documentation.

    `ocean_grid_shapes` is keyed per-variable (``"sst"``, ``"ocs"``, ``"ocd"``)
    rather than a single shape, because era 1's ocean grid is internally
    inconsistent: `sst` differs from `ocs`/`ocd`. Ocean shapes are recorded
    here for future use only — `core.config.grid_coordinates` still does not
    support `"ocean"`, not because orientation is unknown (confirmed,
    docs/open_questions.md §1.1) but because the grid is non-linear
    (Mercator/tripolar) and has no 2D per-cell coordinate support yet;
    nothing in this module wires ocean loading through.

    `lat0`/lon-step coordinate convention is deliberately NOT modeled here:
    only *shape* was empirically confirmed for era 1 and era 3 via land-mask
    correlation (era 2's convention was already verified separately). Era 3's
    finding is wired into `core.config.grid_coordinates` (it now uses era 3's
    convention for every era); era 1's is not — see
    `tasks/16-archive-grid-resolution-drift.md` and
    `notebooks/verify_era1_era3_grid_coordinates.ipynb` for the accepted gap
    this leaves for era 1/era 2 windows.
    """

    name: str
    start: datetime  # UTC-aware, inclusive
    end: datetime | None  # UTC-aware, exclusive; None = open-ended (still current)
    wave_grid_shape: tuple[int, int]
    wind_grid_shape: tuple[int, int]
    ocean_grid_shapes: dict[str, tuple[int, int]]
    cadence_seconds: float

    def __post_init__(self) -> None:
        _require_utc(self.start)
        if self.end is not None:
            _require_utc(self.end)

    def contains(self, t: datetime) -> bool:
        """True if `t` falls in this era's half-open ``[start, end)`` interval."""
        if t < self.start:
            return False
        return self.end is None or t < self.end


# Eras are ordered and contiguous by construction: era[i].end == era[i+1].start
# for every i but the last, whose end is None (open). See
# test_archive_eras_registry_is_ordered_and_contiguous for the enforced guard.
#
# The era1/era2 seam deliberately sets era1.end = era2.start = 2023-10-02T00:00Z,
# one 3h slot *after* era1's last real (non-empty) snapshot at
# 2023-10-01T18:00Z. That 21:00Z slot is one of era1's systematically-vacant
# 03/09/15/21 slots (confirmed alternating-vacancy pattern across all of
# era1) — classifying it as still-era1 keeps the registry gap-free rather
# than leaving an orphan timestamp unclassified. The era2/era3 seam has no
# such gap: era2's last real snapshot (2025-10-26T21:00Z) is exactly 3h
# before era3's start.
ARCHIVE_ERAS: Final[tuple[ArchiveEra, ...]] = (
    ArchiveEra(
        name="era1",
        start=datetime(2022, 1, 1, tzinfo=timezone.utc),
        end=datetime(2023, 10, 2, tzinfo=timezone.utc),
        wave_grid_shape=(181, 360),
        wind_grid_shape=(181, 360),
        ocean_grid_shapes={"sst": (180, 360), "ocs": (1265, 2880), "ocd": (1265, 2880)},
        cadence_seconds=6 * 3600.0,  # 6-hourly: 00/06/12/18Z only; 03/09/15/21Z structurally empty
    ),
    ArchiveEra(
        name="era2",
        start=datetime(2023, 10, 2, tzinfo=timezone.utc),
        end=datetime(2025, 10, 27, tzinfo=timezone.utc),
        wave_grid_shape=(621, 1440),
        wind_grid_shape=(721, 1440),
        ocean_grid_shapes={"sst": (1121, 2880), "ocs": (1121, 2880), "ocd": (1121, 2880)},
        cadence_seconds=3 * 3600.0,
    ),
    ArchiveEra(
        name="era3",
        start=datetime(2025, 10, 27, tzinfo=timezone.utc),
        # Open-ended: no later era is currently known. The archive's last real
        # snapshot as of this writing is 2026-07-20T00:00Z, but that is a
        # *data availability* fact, not an era boundary -- do not confuse the
        # two. If a future shape/cadence change is discovered, add era4 here
        # and set this to era4.start.
        end=None,
        # Numerically == wind's shape, AND confirmed physically identical:
        # lat0=-90.0, 0.25° steps, same as WIND_GRID_LAT_AT_ROW0/STEP_DEG --
        # see notebooks/verify_era1_era3_grid_coordinates.ipynb (1.0000
        # same-date land-mask agreement vs. 0.5643 different-date control).
        # Wired into core.config.grid_coordinates via WAVE_GRID_LAT_AT_ROW0.
        wave_grid_shape=(721, 1440),
        wind_grid_shape=(721, 1440),
        ocean_grid_shapes={"sst": (1121, 2880), "ocs": (1121, 2880), "ocd": (1121, 2880)},
        # Presumed same as era2; NOT independently reconfirmed for era3
        # specifically (see tasks/16-archive-grid-resolution-drift.md, "deferred").
        cadence_seconds=3 * 3600.0,
    ),
)


def era_for_timestamp(t: datetime) -> ArchiveEra:
    """Return the `ArchiveEra` containing UTC timestamp `t`.

    This is the single date-dispatch point in the codebase: every other
    module that needs a grid shape or cadence should call this and read the
    returned era's attributes, rather than re-implementing a date check.

    Parameters
    ----------
    t : datetime
        UTC-aware timestamp.

    Returns
    -------
    ArchiveEra
        The era whose ``[start, end)`` interval contains `t`. Era 3 is
        open-ended (``end=None``): any `t` on or after its start resolves to
        era 3, including dates far beyond the archive's currently-known last
        snapshot -- there is no known era 4 to distinguish it from, so this
        is the least-wrong available answer, not a silently-guessed one.

    Raises
    ------
    ValueError
        If `t` is not UTC-aware, or if `t` is before the earliest known
        archive era (before ``ARCHIVE_ERAS[0].start``, i.e. before the
        archive begins).
    """
    _require_utc(t)
    for era in ARCHIVE_ERAS:
        if era.contains(t):
            return era
    raise ValueError(
        f"{t.isoformat()} is before the earliest known archive era "
        f"({ARCHIVE_ERAS[0].name}, starting {ARCHIVE_ERAS[0].start.isoformat()}); no "
        "grid-shape/cadence facts are known this far back. See docs/model_design.md "
        "§2.2 and tasks/16-archive-grid-resolution-drift.md."
    )
