"""Tests for core.archive_eras (see docs/model_design.md §2.2, tasks/16-archive-grid-resolution-drift.md)."""

from datetime import datetime, timedelta, timezone

import pytest

from core.archive_eras import ARCHIVE_ERAS, era_for_timestamp


def test_era_for_timestamp_era1_start() -> None:
    """The earliest known timestamp resolves to era1."""
    era = era_for_timestamp(datetime(2022, 1, 1, tzinfo=timezone.utc))
    assert era.name == "era1"
    assert era.wave_grid_shape == (181, 360)
    assert era.wind_grid_shape == (181, 360)
    assert era.cadence_seconds == 6 * 3600.0


def test_era_for_timestamp_era1_ocean_shapes_differ_by_variable() -> None:
    """era1's ocean grid is internally inconsistent: sst differs from ocs/ocd."""
    era = era_for_timestamp(datetime(2022, 6, 15, tzinfo=timezone.utc))
    assert era.ocean_grid_shapes["sst"] == (180, 360)
    assert era.ocean_grid_shapes["ocs"] == (1265, 2880)
    assert era.ocean_grid_shapes["ocd"] == (1265, 2880)
    assert era.ocean_grid_shapes["sst"] != era.ocean_grid_shapes["ocs"]


def test_era_for_timestamp_era1_last_real_snapshot() -> None:
    """2023-10-01T18:00Z (era1's last real snapshot) still resolves to era1."""
    era = era_for_timestamp(datetime(2023, 10, 1, 18, tzinfo=timezone.utc))
    assert era.name == "era1"


def test_era_for_timestamp_era1_era2_seam() -> None:
    """era1/era2 seam: one microsecond before the seam is still era1, the seam itself is era2."""
    seam = datetime(2023, 10, 2, tzinfo=timezone.utc)
    just_before = seam - timedelta(microseconds=1)
    assert era_for_timestamp(just_before).name == "era1"
    assert era_for_timestamp(seam).name == "era2"


def test_era_for_timestamp_era2_last_real_snapshot() -> None:
    """2025-10-26T21:00Z (era2's last real snapshot) still resolves to era2."""
    era = era_for_timestamp(datetime(2025, 10, 26, 21, tzinfo=timezone.utc))
    assert era.name == "era2"
    assert era.wave_grid_shape == (621, 1440)
    assert era.wind_grid_shape == (721, 1440)
    assert era.cadence_seconds == 3 * 3600.0


def test_era_for_timestamp_era2_era3_seam() -> None:
    """era2/era3 seam: one microsecond before is era2, the seam itself is era3."""
    seam = datetime(2025, 10, 27, tzinfo=timezone.utc)
    just_before = seam - timedelta(microseconds=1)
    assert era_for_timestamp(just_before).name == "era2"
    assert era_for_timestamp(seam).name == "era3"


def test_era_for_timestamp_era3_matches_wind_shape_but_is_independent() -> None:
    """era3's wave grid is numerically equal to the wind grid's shape (a separate fact)."""
    era = era_for_timestamp(datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert era.name == "era3"
    assert era.wave_grid_shape == (721, 1440)
    assert era.wind_grid_shape == (721, 1440)
    assert era.wave_grid_shape == era.wind_grid_shape


def test_era_for_timestamp_era3_is_open_ended_into_the_far_future() -> None:
    """era3 has no known successor: a timestamp far beyond the archive's known end still resolves."""
    era = era_for_timestamp(datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert era.name == "era3"


def test_era_for_timestamp_before_earliest_era_raises() -> None:
    """A timestamp before era1's start has no known grid/cadence facts."""
    with pytest.raises(ValueError, match="before the earliest known archive era"):
        era_for_timestamp(datetime(2021, 12, 31, tzinfo=timezone.utc))


def test_era_for_timestamp_naive_datetime_raises() -> None:
    """A naive (non-UTC-aware) datetime is rejected."""
    with pytest.raises(ValueError, match="UTC-aware"):
        era_for_timestamp(datetime(2024, 1, 1))  # noqa: DTZ001


def test_archive_eras_registry_is_ordered_and_contiguous() -> None:
    """Guard against future registry edits introducing gaps or overlaps."""
    assert len(ARCHIVE_ERAS) >= 1
    for i in range(len(ARCHIVE_ERAS) - 1):
        current, following = ARCHIVE_ERAS[i], ARCHIVE_ERAS[i + 1]
        assert current.end is not None, f"{current.name} must have a defined end (not the last era)"
        assert (
            current.end == following.start
        ), f"{current.name}.end must equal {following.name}.start for a gap-free registry"
    assert ARCHIVE_ERAS[-1].end is None, "only the last era may be open-ended"
