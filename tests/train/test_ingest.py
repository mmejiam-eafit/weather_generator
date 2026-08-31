"""Tests for train.ingest (see docs/model_design.md §2.5, §5.1)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from core.io import snapshot_path
from train.ingest import load_training_window

_T0 = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)


def _write_snapshot(
    data_dir: Path, t: datetime, variable: str, array: np.ndarray, dtype: type
) -> None:
    path = snapshot_path(data_dir, t, variable)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(dtype))


def _write_uniform_window(
    data_dir: Path,
    variable: str,
    n_times: int,
    shape: tuple[int, int],
    dtype: type,
    value: float,
    skip_indices: frozenset[int] = frozenset(),
    t0: datetime = _T0,
    step_hours: int = 3,
) -> None:
    for i in range(n_times):
        if i in skip_indices:
            continue
        t = t0 + timedelta(hours=step_hours * i)
        _write_snapshot(data_dir, t, variable, np.full(shape, value), dtype)


def test_load_training_window_continuous_variable(tmp_path: Path) -> None:
    """A wave-grid continuous variable loads with the right shape, dims, and values."""
    _write_uniform_window(tmp_path, "sigwh", n_times=4, shape=(5, 8), dtype=np.float16, value=2.5)
    ds = load_training_window(tmp_path, _T0, _T0 + timedelta(hours=12), ["sigwh"])

    assert ds["sigwh"].dims == ("time", "wave_lat", "wave_lon")
    assert ds["sigwh"].shape == (4, 5, 8)
    assert_allclose(ds["sigwh"].values, np.full((4, 5, 8), 2.5))
    assert_allclose(ds["wave_lat"].values, [-90.0, -89.75, -89.5, -89.25, -89.0])
    assert_allclose(ds["wave_lon"].values, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])


def test_load_training_window_direction_variable_decoded(tmp_path: Path) -> None:
    """A wind-grid direction variable (wad) is decoded to degrees, origin code 1."""
    codes = np.array([[1, 5, 9]], dtype=np.int8)
    _write_snapshot(tmp_path, _T0, "wad", codes, np.int8)
    ds = load_training_window(tmp_path, _T0, _T0 + timedelta(hours=3), ["wad"])

    assert ds["wad"].dims == ("time", "wind_lat", "wind_lon")
    assert_allclose(ds["wad"].values[0], [[0.0, 90.0, 180.0]])


def test_load_training_window_missing_snapshot_fills_nan(tmp_path: Path) -> None:
    """A single missing snapshot is logged to stderr and filled with NaN, not aborted."""
    _write_uniform_window(
        tmp_path,
        "sigwh",
        n_times=3,
        shape=(2, 2),
        dtype=np.float16,
        value=1.0,
        skip_indices=frozenset({1}),
    )
    ds = load_training_window(tmp_path, _T0, _T0 + timedelta(hours=9), ["sigwh"])

    assert np.isnan(ds["sigwh"].values[1]).all()
    assert not np.isnan(ds["sigwh"].values[0]).any()
    assert not np.isnan(ds["sigwh"].values[2]).any()


def test_load_training_window_moderate_gap_warns(tmp_path: Path) -> None:
    """A contiguous gap of 6h (2 missing snapshots) warns but does not raise."""
    _write_uniform_window(
        tmp_path,
        "sigwh",
        n_times=4,
        shape=(2, 2),
        dtype=np.float16,
        value=1.0,
        skip_indices=frozenset({1, 2}),
    )
    with pytest.warns(UserWarning, match="gap"):
        ds = load_training_window(tmp_path, _T0, _T0 + timedelta(hours=12), ["sigwh"])
    assert np.isnan(ds["sigwh"].values[1]).all()
    assert np.isnan(ds["sigwh"].values[2]).all()


def test_load_training_window_long_gap_aborts(tmp_path: Path) -> None:
    """A contiguous gap of 24h+ (8 missing snapshots) aborts with RuntimeError."""
    _write_uniform_window(
        tmp_path,
        "sigwh",
        n_times=10,
        shape=(2, 2),
        dtype=np.float16,
        value=1.0,
        skip_indices=frozenset(range(1, 9)),
    )
    with pytest.raises(RuntimeError, match="contiguous gap"):
        load_training_window(tmp_path, _T0, _T0 + timedelta(hours=30), ["sigwh"])


def test_load_training_window_no_snapshots_raises(tmp_path: Path) -> None:
    """A variable with zero snapshots present anywhere in the window raises RuntimeError."""
    with pytest.raises(RuntimeError, match="no snapshots found"):
        load_training_window(tmp_path, _T0, _T0 + timedelta(hours=9), ["sigwh"])


def test_load_training_window_unknown_variable_raises(tmp_path: Path) -> None:
    """A variable absent from the §10 table raises ValueError."""
    with pytest.raises(ValueError, match="unknown variable"):
        load_training_window(tmp_path, _T0, _T0 + timedelta(hours=3), ["not_a_real_variable"])


def test_load_training_window_ocean_variable_raises(tmp_path: Path) -> None:
    """An ocean-grid variable raises: orientation is confirmed
    (open_questions.md §1.1) but the grid is non-linear (Mercator/tripolar)
    with no 2D coordinate support yet."""
    _write_snapshot(tmp_path, _T0, "sst", np.full((2, 2), 15.0), np.float16)
    with pytest.raises(ValueError, match="not a linear"):
        load_training_window(tmp_path, _T0, _T0 + timedelta(hours=3), ["sst"])


def test_load_training_window_multiple_variables_different_grids(tmp_path: Path) -> None:
    """Wave- and wind-grid variables coexist in one Dataset with independent dims."""
    _write_uniform_window(tmp_path, "sigwh", n_times=2, shape=(3, 4), dtype=np.float16, value=1.0)
    _write_uniform_window(tmp_path, "was", n_times=2, shape=(5, 6), dtype=np.float16, value=2.0)
    ds = load_training_window(tmp_path, _T0, _T0 + timedelta(hours=6), ["sigwh", "was"])

    assert ds["sigwh"].shape == (2, 3, 4)
    assert ds["was"].shape == (2, 5, 6)
    assert "wave_lat" in ds.coords and "wind_lat" in ds.coords


def test_load_training_window_first_snapshot_wad_consistency_failure_raises(
    tmp_path: Path,
) -> None:
    """A wad/wad_come_from mismatch at the first snapshot aborts before the load loop (§5.1).

    Neither `wad` nor `wad_come_from` is in `variables`, and no `sigwh`
    snapshot is ever written — proving the check runs unconditionally and
    before the per-variable loop reaches `sigwh`.
    """
    mismatched = np.array([[1, 5]], dtype=np.int8)  # both decode to the same 0/90 deg
    _write_snapshot(tmp_path, _T0, "wad", mismatched, np.int8)
    _write_snapshot(tmp_path, _T0, "wad_come_from", mismatched, np.int8)

    with pytest.raises(ValueError, match="do not differ by 180"):
        load_training_window(tmp_path, _T0, _T0 + timedelta(hours=3), ["sigwh"])


def test_load_training_window_last_snapshot_wad_consistency_failure_raises(
    tmp_path: Path,
) -> None:
    """A wad/wad_come_from mismatch only at the LAST snapshot still aborts the whole load.

    Every snapshot but the last has a correctly-related pair; only the last
    snapshot's `wad_come_from` is corrupted to match `wad` instead of
    opposing it. If the check only ran once at the start, this window would
    load without error — proving it also runs on the last snapshot.
    """
    n_times = 3
    good_wad = np.array([[1, 5]], dtype=np.int8)  # 0 deg, 90 deg
    good_wad_come_from = np.array([[9, 13]], dtype=np.int8)  # 180 deg, 270 deg -> consistent

    for i in range(n_times):
        t = _T0 + timedelta(hours=3 * i)
        _write_snapshot(tmp_path, t, "wad", good_wad, np.int8)
        if i == n_times - 1:
            _write_snapshot(tmp_path, t, "wad_come_from", good_wad, np.int8)  # corrupted
        else:
            _write_snapshot(tmp_path, t, "wad_come_from", good_wad_come_from, np.int8)
    _write_uniform_window(
        tmp_path, "sigwh", n_times=n_times, shape=(1, 2), dtype=np.float16, value=1.0
    )

    with pytest.raises(ValueError, match="do not differ by 180"):
        load_training_window(tmp_path, _T0, _T0 + timedelta(hours=3 * n_times), ["sigwh"])


def test_load_training_window_wad_consistency_check_passes_silently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A correctly-related wad/wad_come_from pair across the window loads cleanly (§5.1).

    Success is logged to stderr (via the informational print pattern this
    file already uses for gap handling) but no warning or exception fires.
    """
    n_times = 3
    wad = np.array([[1, 5]], dtype=np.int8)  # 0 deg, 90 deg
    wad_come_from = np.array([[9, 13]], dtype=np.int8)  # 180 deg, 270 deg -> consistent

    for i in range(n_times):
        t = _T0 + timedelta(hours=3 * i)
        _write_snapshot(tmp_path, t, "wad", wad, np.int8)
        _write_snapshot(tmp_path, t, "wad_come_from", wad_come_from, np.int8)
    _write_uniform_window(
        tmp_path, "sigwh", n_times=n_times, shape=(1, 2), dtype=np.float16, value=1.0
    )

    ds = load_training_window(tmp_path, _T0, _T0 + timedelta(hours=3 * n_times), ["sigwh"])

    assert ds["sigwh"].shape == (n_times, 1, 2)
    assert "consistency check passed" in capsys.readouterr().err


def test_load_training_window_wad_consistency_check_ignores_land_cells(tmp_path: Path) -> None:
    """Land cells with identical wad/wad_come_from fill codes don't trip the
    check; only ocean cells (masked via `was`'s NaN pattern) are checked
    (tasks/15-wad-consistency-check-land-mask-artifact.md).

    Reproduces the real-archive false positive found 2026-08-14: without
    masking, this window's land cells alone (deviation 180° from the
    expected 180° relationship) would raise, even though the ocean cells
    are cleanly 180° apart.
    """
    was = np.array([[np.nan, 5.0, np.nan, 3.0]], dtype=np.float16)
    wad = np.array([[1, 1, 1, 5]], dtype=np.int8)  # land fill code (1) repeated at idx 0/2
    # land: same fill code as wad (idx 0/2); ocean: 180 deg apart (idx 1/3)
    wad_come_from = np.array([[1, 9, 1, 13]], dtype=np.int8)

    _write_snapshot(tmp_path, _T0, "was", was, np.float16)
    _write_snapshot(tmp_path, _T0, "wad", wad, np.int8)
    _write_snapshot(tmp_path, _T0, "wad_come_from", wad_come_from, np.int8)

    ds = load_training_window(tmp_path, _T0, _T0 + timedelta(hours=3), ["was"])

    assert ds["was"].shape == (1, 1, 4)


def test_load_training_window_wad_consistency_check_still_detects_ocean_drift(
    tmp_path: Path,
) -> None:
    """A genuine mismatch at an ocean cell still raises even with land-cell masking.

    Confirms the tasks/15 fix doesn't make the check vacuous.
    """
    was = np.array([[np.nan, 5.0]], dtype=np.float16)  # idx 0 land, idx 1 ocean
    wad = np.array([[1, 1]], dtype=np.int8)
    wad_come_from = np.array([[1, 1]], dtype=np.int8)  # idx 1 ocean: same code -> not 180 apart

    _write_snapshot(tmp_path, _T0, "was", was, np.float16)
    _write_snapshot(tmp_path, _T0, "wad", wad, np.int8)
    _write_snapshot(tmp_path, _T0, "wad_come_from", wad_come_from, np.int8)

    with pytest.raises(ValueError, match="do not differ by 180"):
        load_training_window(tmp_path, _T0, _T0 + timedelta(hours=3), ["was"])


# --- Archive era drift (see tasks/16-archive-grid-resolution-drift.md) ---

_ERA1_T0 = datetime(2022, 6, 1, 0, tzinfo=timezone.utc)


def test_load_training_window_era1_window_loads_with_era1_cadence(tmp_path: Path) -> None:
    """A window entirely within era1 (2022-2023) loads at era1's 6-hourly cadence, no gaps logged."""
    _write_uniform_window(
        tmp_path,
        "sigwh",
        n_times=3,
        shape=(5, 8),
        dtype=np.float16,
        value=2.5,
        t0=_ERA1_T0,
        step_hours=6,
    )
    ds = load_training_window(tmp_path, _ERA1_T0, _ERA1_T0 + timedelta(hours=18), ["sigwh"])

    assert ds["sigwh"].shape == (3, 5, 8)
    assert not np.isnan(ds["sigwh"].values).any()
    assert list(ds["time"].values) == [
        np.datetime64(_ERA1_T0.replace(tzinfo=None) + timedelta(hours=6 * i), "s") for i in range(3)
    ]


def test_load_training_window_spans_era_boundary_raises_before_file_io(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A window spanning the era1/era2 seam is rejected immediately, before any file I/O.

    No files are written at all — if the rejection happened anywhere other
    than up-front (e.g. deep in the per-variable load loop), this would
    instead fail with "no snapshots found" or log "missing snapshot" lines.
    """
    t_start = datetime(2023, 10, 1, 18, tzinfo=timezone.utc)  # era1's last real snapshot
    t_end = datetime(2023, 10, 2, 3, tzinfo=timezone.utc)  # era2

    with pytest.raises(ValueError, match="spans more than one archive era"):
        load_training_window(tmp_path, t_start, t_end, ["sigwh"])

    captured = capsys.readouterr()
    assert "missing snapshot" not in captured.err


def test_load_training_window_shape_changes_mid_window_raises(tmp_path: Path) -> None:
    """A variable whose on-disk shape changes mid-window (within a single era) raises clearly."""
    _write_snapshot(tmp_path, _T0, "sigwh", np.full((5, 8), 1.0), np.float16)
    _write_snapshot(tmp_path, _T0 + timedelta(hours=3), "sigwh", np.full((5, 7), 1.0), np.float16)

    with pytest.raises(ValueError, match="inconsistent with this window's earlier"):
        load_training_window(tmp_path, _T0, _T0 + timedelta(hours=6), ["sigwh"])


def test_load_training_window_shape_exceeds_era_canonical_raises(tmp_path: Path) -> None:
    """A variable shape exceeding its era's canonical bound raises clearly.

    era1's canonical wave shape is (181, 360); this writes a (300, 500)
    snapshot, which exceeds it on both axes.
    """
    _write_snapshot(tmp_path, _ERA1_T0, "sigwh", np.full((300, 500), 1.0), np.float16)

    with pytest.raises(ValueError, match="exceeding the era1 canonical"):
        load_training_window(tmp_path, _ERA1_T0, _ERA1_T0 + timedelta(hours=6), ["sigwh"])
