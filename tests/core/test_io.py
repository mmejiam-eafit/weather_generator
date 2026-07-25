"""Tests for core.io (see docs/model_design.md §2.1, §2.3, §5.1)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from core.io import (
    decode_compass_code,
    enumerate_snapshots,
    load_raw,
    load_snapshot,
    snapshot_path,
)
from train.ingest import _check_wad_wad_come_from_consistency


def test_snapshot_path_zero_padded() -> None:
    """Month/day/hour zero-padding is always present, even for single-digit values."""
    t = datetime(2025, 1, 2, 3, tzinfo=timezone.utc)
    path = snapshot_path(Path("./data"), t, "sigwh")
    assert path == Path("./data/2025/01/02/03/sigwh.npy")


def test_snapshot_path_utc_timezone_required() -> None:
    """A naive datetime raises ValueError rather than silently attaching UTC."""
    naive = datetime(2025, 1, 2, 3)
    with pytest.raises(ValueError, match="UTC-aware"):
        snapshot_path(Path("./data"), naive, "sigwh")


def test_enumerate_snapshots_three_hour_cadence() -> None:
    """Yields timestamps 3 hours apart over [t_start, t_end)."""
    t_start = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    t_end = datetime(2025, 1, 1, 9, tzinfo=timezone.utc)
    times = list(enumerate_snapshots(t_start, t_end))
    assert times == [
        datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 3, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 6, tzinfo=timezone.utc),
    ]


def test_enumerate_snapshots_utc_timezone_required() -> None:
    """Naive t_start or t_end raises ValueError."""
    with pytest.raises(ValueError, match="UTC-aware"):
        list(enumerate_snapshots(datetime(2025, 1, 1), datetime(2025, 1, 2, tzinfo=timezone.utc)))


def test_enumerate_snapshots_six_hour_cadence_in_era1() -> None:
    """A window entirely within era1 (2022-2023) yields only the 6-hourly slots that exist.

    (see tasks/16-archive-grid-resolution-drift.md — era1's 03/09/15/21Z
    hours are structurally empty across the whole era, not random gaps.)
    """
    t_start = datetime(2022, 6, 1, 0, tzinfo=timezone.utc)
    t_end = datetime(2022, 6, 1, 19, tzinfo=timezone.utc)
    times = list(enumerate_snapshots(t_start, t_end))
    assert times == [
        datetime(2022, 6, 1, 0, tzinfo=timezone.utc),
        datetime(2022, 6, 1, 6, tzinfo=timezone.utc),
        datetime(2022, 6, 1, 12, tzinfo=timezone.utc),
        datetime(2022, 6, 1, 18, tzinfo=timezone.utc),
    ]


def test_enumerate_snapshots_mixed_cadence_across_era_boundary_not_rejected_here() -> None:
    """enumerate_snapshots itself does not reject a cross-era window (that's ingest's job);
    it enumerates with each timestamp's own era cadence, so the step size varies at the seam.
    """
    t_start = datetime(2023, 10, 1, 12, tzinfo=timezone.utc)  # era1, 6-hourly
    t_end = datetime(2023, 10, 2, 6, tzinfo=timezone.utc)  # era2, 3-hourly
    times = list(enumerate_snapshots(t_start, t_end))
    # era1 steps of 6h until crossing the 2023-10-02T00:00Z seam, then era2's 3h steps.
    assert times == [
        datetime(2023, 10, 1, 12, tzinfo=timezone.utc),
        datetime(2023, 10, 1, 18, tzinfo=timezone.utc),
        datetime(2023, 10, 2, 0, tzinfo=timezone.utc),
        datetime(2023, 10, 2, 3, tzinfo=timezone.utc),
    ]
    deltas = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    assert deltas == [timedelta(hours=6), timedelta(hours=6), timedelta(hours=3)]


def test_wsd_compass_codes() -> None:
    """wsd/pwd use origin code 0 = North (see §2.3)."""
    assert_allclose(decode_compass_code(np.array([[0]], dtype=np.int8), origin_code=0), [[0.0]])
    assert_allclose(decode_compass_code(np.array([[4]], dtype=np.int8), origin_code=0), [[90.0]])


def test_wad_compass_codes() -> None:
    """wad uses origin code 1 = North (see §2.3)."""
    assert_allclose(decode_compass_code(np.array([[1]], dtype=np.int8), origin_code=1), [[0.0]])
    assert_allclose(decode_compass_code(np.array([[5]], dtype=np.int8), origin_code=1), [[90.0]])


def test_decode_compass_code_wraps_to_0_360() -> None:
    """Decoded degrees stay within [0, 360) even for codes below the origin."""
    degrees = decode_compass_code(np.array([[0]], dtype=np.int8), origin_code=1)
    assert_allclose(degrees, [[337.5]])


def test_load_snapshot_continuous_upcasts_to_float64(tmp_path: Path) -> None:
    """A continuous variable is loaded and upcast from float16 to float64."""
    t = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    snap_dir = tmp_path / "2025/01/01/00"
    snap_dir.mkdir(parents=True)
    raw = np.array([[1.5, 2.5]], dtype=np.float16)
    np.save(snap_dir / "sigwh.npy", raw)

    loaded = load_snapshot(tmp_path, t, "sigwh")
    assert loaded.dtype == np.float64
    assert_allclose(loaded, [[1.5, 2.5]])


def test_load_snapshot_direction_decodes_to_degrees(tmp_path: Path) -> None:
    """A direction variable is loaded and decoded to degrees, not raw codes."""
    t = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    snap_dir = tmp_path / "2025/01/01/00"
    snap_dir.mkdir(parents=True)
    raw = np.array([[1, 5]], dtype=np.int8)
    np.save(snap_dir / "wad.npy", raw)

    loaded = load_snapshot(tmp_path, t, "wad")
    assert loaded.dtype == np.float64
    assert_allclose(loaded, [[0.0, 90.0]])


def test_load_snapshot_unverified_variable_raises(tmp_path: Path) -> None:
    """Loading a variable with no known decoding convention raises."""
    t = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    snap_dir = tmp_path / "2025/01/01/00"
    snap_dir.mkdir(parents=True)
    np.save(snap_dir / "bogus.npy", np.array([[0]], dtype=np.int8))

    with pytest.raises(ValueError, match="unknown decoding convention"):
        load_snapshot(tmp_path, t, "bogus")


def test_wad_come_from_compass_codes_match_wad_formula() -> None:
    """wad_come_from decodes with the same origin code as wad (origin 1, §2.3)."""
    assert_allclose(decode_compass_code(np.array([[1]], dtype=np.int8), origin_code=1), [[0.0]])
    assert_allclose(decode_compass_code(np.array([[5]], dtype=np.int8), origin_code=1), [[90.0]])


def test_ocd_compass_codes_match_wad_formula() -> None:
    """ocd decodes with the same origin code as wad (origin 1, §2.3) — no correction."""
    assert_allclose(decode_compass_code(np.array([[1]], dtype=np.int8), origin_code=1), [[0.0]])
    assert_allclose(decode_compass_code(np.array([[5]], dtype=np.int8), origin_code=1), [[90.0]])


def test_load_snapshot_wad_come_from_decodes_to_degrees(tmp_path: Path) -> None:
    """wad_come_from is loaded and decoded to degrees, not raw codes (§2.3)."""
    t = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    snap_dir = tmp_path / "2025/01/01/00"
    snap_dir.mkdir(parents=True)
    raw = np.array([[1, 5]], dtype=np.int8)
    np.save(snap_dir / "wad_come_from.npy", raw)

    loaded = load_snapshot(tmp_path, t, "wad_come_from")
    assert loaded.dtype == np.float64
    assert_allclose(loaded, [[0.0, 90.0]])
    assert np.all((loaded >= 0.0) & (loaded < 360.0))


def test_load_snapshot_ocd_decodes_to_degrees(tmp_path: Path) -> None:
    """ocd is loaded and decoded to degrees, not raw codes (§2.3)."""
    t = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    snap_dir = tmp_path / "2025/01/01/00"
    snap_dir.mkdir(parents=True)
    raw = np.array([[1, 5]], dtype=np.int8)
    np.save(snap_dir / "ocd.npy", raw)

    loaded = load_snapshot(tmp_path, t, "ocd")
    assert loaded.dtype == np.float64
    assert_allclose(loaded, [[0.0, 90.0]])
    assert np.all((loaded >= 0.0) & (loaded < 360.0))


def test_load_raw_direction_field_no_upcast(tmp_path: Path) -> None:
    """load_raw returns the untouched int8 array for a direction field, no decoding."""
    t = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    snap_dir = tmp_path / "2025/01/01/00"
    snap_dir.mkdir(parents=True)
    raw = np.array([[1, 5]], dtype=np.int8)
    np.save(snap_dir / "wad.npy", raw)

    loaded = load_raw(tmp_path, t, "wad")
    assert loaded.dtype == np.int8
    assert_allclose(loaded, [[1, 5]])


def test_load_raw_continuous_field_no_upcast(tmp_path: Path) -> None:
    """load_raw returns the untouched float16 array for a continuous field, no upcasting."""
    t = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    snap_dir = tmp_path / "2025/01/01/00"
    snap_dir.mkdir(parents=True)
    raw = np.array([[1.5, 2.5]], dtype=np.float16)
    np.save(snap_dir / "sigwh.npy", raw)

    loaded = load_raw(tmp_path, t, "sigwh")
    assert loaded.dtype == np.float16
    assert_allclose(loaded, [[1.5, 2.5]])


def test_wad_and_wad_come_from_consistency_check(tmp_path: Path) -> None:
    """A wad/wad_come_from pair exactly 180 deg apart passes; any other offset raises (§5.1).

    `_check_wad_wad_come_from_consistency` physically lives in `train.ingest`
    (it needs `core.io.load_raw` + `core.io.decode_compass_code` directly,
    not `load_snapshot`), but is exercised here per docs/model_design.md
    §12.5's naming of this test.
    """
    t = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
    snap_dir = tmp_path / "2025/01/01/00"
    snap_dir.mkdir(parents=True)

    # wad codes 1, 5 -> 0 deg, 90 deg. wad_come_from codes 9, 13 -> 180 deg,
    # 270 deg. Both cells differ by exactly 180 deg: a consistent pair.
    np.save(snap_dir / "wad.npy", np.array([[1, 5]], dtype=np.int8))
    np.save(snap_dir / "wad_come_from.npy", np.array([[9, 13]], dtype=np.int8))
    _check_wad_wad_come_from_consistency(tmp_path, t)  # must not raise

    # Now wad_come_from matches wad exactly (0 deg offset instead of 180 deg).
    np.save(snap_dir / "wad_come_from.npy", np.array([[1, 5]], dtype=np.int8))
    with pytest.raises(ValueError, match="do not differ by 180"):
        _check_wad_wad_come_from_consistency(tmp_path, t)
