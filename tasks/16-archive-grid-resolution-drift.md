# Task 16 — Archive grid-resolution/cadence drift: era registry + fixes

**Depends on:** 01, 03, 08 (touches `core/io.py`, `train/ingest.py`,
`weather/model.py`, all already-completed tasks). **Blocks:** 09, 10, 11 —
whoever implements sampling/validation/notebooks needs to pick a
single-era training window and derive `dt` from `core.archive_eras`, not a
flat constant.

## How this was found

Discovered while adapting `notebooks/verify_size_of_matrices.ipynb` (task
15's sibling, unrelated) to check a random snapshot from the real archive at
`data/` (a symlink to real NOAA GFS-Wave/WAVEWATCH III + GFS data) instead of
a synthetic one. The file size didn't match the documented shape. Three
parallel investigation agents then confirmed, by direct binary-search
inspection of real files, that `src/core/config.py`'s `WAVE_GRID_SHAPE`,
`WIND_GRID_SHAPE`, `OCEAN_GRID_SHAPE`, `SNAPSHOT_CADENCE_SECONDS` — treated
everywhere in the v1→v2 migration (tasks 00–15) as flat, always-true
constants — only actually hold for a ~2-year sub-window of the archive. The
whole migration built and tested exclusively against synthetic fixtures with
one fixed shape and never touched the real archive, so this went undetected
until now.

## The three archive eras (confirmed facts)

| Era | Dates (UTC) | Wave shape | Wind shape | Ocean shape | Cadence |
|---|---|---|---|---|---|
| **era1** | 2022-01-01 → 2023-10-01T18:00Z (last real snapshot) | (181,360) | (181,360) | `sst`=(180,360) **≠** `ocs`/`ocd`=(1265,2880) — internally inconsistent | 6-hourly (00/06/12/18Z only; 03/09/15/21Z dirs exist but are structurally empty across the whole era, confirmed, not random gaps) |
| **era2** | 2023-10-02T00:00Z → 2025-10-26T21:00Z (last real snapshot) | (621,1440) — matches pre-existing `WAVE_GRID_SHAPE` | (721,1440) — matches pre-existing `WIND_GRID_SHAPE` | (1121,2880), consistent — matches pre-existing `OCEAN_GRID_SHAPE` | 3-hourly — matches pre-existing `SNAPSHOT_CADENCE_SECONDS` |
| **era3** | 2025-10-27T00:00Z → archive end (2026-07-20T00:00Z, confirmed matching `docs/model_design.md`'s documented cutoff) | (721,1440) — now numerically equal to wind's shape | (721,1440) | (1121,2880) | 3-hourly (presumed same as era2, not independently reconfirmed) |

All three transitions are confirmed clean, synchronized, one-time cutovers
(no patchy/interleaved periods) — verified by binary-search inspection on
both sides of each boundary. Additional finding: the archive's last ~7
snapshots (2026-07-19T03:00Z–2026-07-20T00:00Z) are missing all 3 ocean
variables (`sst`/`ocs`/`ocd`) — 13/16 files present per snapshot. Lower
priority since ocean-grid loading is unsupported regardless of era
(`core.config.grid_coordinates("ocean")` raises unconditionally, §2.7).

**Coordinate convention (`lat0`/lon-step), separately verified**, via
`notebooks/verify_era1_era3_grid_coordinates.ipynb` (land/ocean NaN-mask
correlation against era 2's known-correct grid — the same general technique
`docs/model_design.md` §2.7 describes for the still-open ocean-orientation
question):
- **era1**: `lat0=-90.0`, `1.0°` steps, `lon0=0.0` — a coarser *global* grid,
  decisively confirmed (97.6% land-mask agreement at this offset vs. 86.8%
  at the next-best candidate, dropping off sharply beyond that). This is
  **not** era 2's regional wave-grid convention (`lat0=-75.0`) — it matches
  the wind grid's convention, just 4x coarser.
- **era3**: the wave grid is a literal copy of the wind grid — confirmed via
  same-timestamp cell-by-cell land-mask agreement of 1.0000 (vs. 0.5643 for
  a different-date control, which rules out coincidence from static
  coastline geometry alone). `lat0=-90.0`, `0.25°` steps.

## What was implemented

1. **New module `src/core/archive_eras.py`** — `ArchiveEra` dataclass (name,
   start/end, wave/wind shape, per-variable ocean shapes, cadence) +
   `ARCHIVE_ERAS: tuple[ArchiveEra, ...]` (the three eras above) +
   `era_for_timestamp(t) -> ArchiveEra`, the single date-dispatch point in
   the codebase. New file rather than extending `core/config.py`:
   `config.py`'s established character is flat constants with zero logic; an
   era registry is a lookup function over an ordered sequence of facts,
   which belongs in its own module. Deliberately has zero intra-`core`
   dependencies (duplicates a 3-line `_require_utc` rather than importing
   `core.io.require_utc`) so `core.io` can import from it without a circular
   import.
2. **`core/io.py`**: `enumerate_snapshots` is now era-aware — steps by
   `era_for_timestamp(t).cadence_seconds` instead of a hardcoded 3 hours. A
   window entirely within era 1 now correctly enumerates only the 6-hourly
   slots that actually exist, instead of enumerating every 3-hourly slot and
   logging spammy false "missing snapshot" lines for the systematically-
   vacant off-hours. Does not itself reject cross-era windows (mixed
   cadence enumerates fine here) — that's `load_training_window`'s job.
3. **`train/ingest.py`**:
   - `_resolve_window_era(times)` — called right after `times`/`n_times` are
     computed, before any file I/O (including before the existing
     `_check_wad_wad_come_from_consistency` guard). Compares
     `era_for_timestamp(times[0])` vs `era_for_timestamp(times[-1])`; raises
     `ValueError` naming both eras and the transition boundary if they
     differ.
   - `_validate_snapshot_shape(...)` — called in the per-variable load loop
     after each successful `load_snapshot`. Two checks: shape doesn't exceed
     the window's era's canonical shape (mirrors `_coordinates_for_shape`'s
     existing tolerance for small synthetic test patches), and shape matches
     the window's own first-loaded shape for that variable. Replaces what
     was previously an unhandled, low-context numpy broadcast `ValueError`
     with a clear, diagnostic one. Skipped for `grid == "ocean"`
     (unsupported regardless of era).
4. **`weather/model.py`**: `_fit_impl` now derives `dt` for `dmd_fit`/
   `fit_local_residual` from `core.archive_eras.era_for_timestamp(self.t_start)
   .cadence_seconds`, not the flat `SNAPSHOT_CADENCE_SECONDS` import. This
   fixes a real, previously-silent correctness bug: training on an
   era-1-only window would otherwise have used the wrong `dt` (3h assumed,
   6h actual) in the DMD eigenvalue / OU theta time-constant math. Safe to
   derive from `self.t_start` alone because `load_training_window` has
   already enforced single-era-ness by the time this runs.
5. **`core/config.py`**: all four existing constants (`WAVE_GRID_SHAPE`,
   `WIND_GRID_SHAPE`, `OCEAN_GRID_SHAPE`, `SNAPSHOT_CADENCE_SECONDS`) kept
   unchanged in value — they remain the sole basis for `grid_coordinates()`
   and have real external consumers
   (`notebooks/verify_size_of_matrices.ipynb` uses `OCEAN_GRID_SHAPE`).
   Comments updated to note each is era 2's value specifically, pointing at
   `core.archive_eras.ARCHIVE_ERAS` as the authoritative per-era source. No
   test changes needed — `tests/core/test_config.py`'s existing assertions
   are era-2 facts, still correctly true.
6. **New notebook `notebooks/verify_era1_era3_grid_coordinates.ipynb`** —
   the empirical coordinate-convention study described above, executed
   against the real archive, not placeholder output.
7. **`docs/model_design.md`** — §1's "2022-2025, 4 years, ~11,700 snapshots
   at 3h cadence" claim corrected (it spans all three eras); new §2.2.1
   tabulating the eras and cross-referencing `core.archive_eras.ARCHIVE_ERAS`
   as the source of truth rather than duplicating numbers that could drift;
   §5.1 describes the new guards; §6 corrected to match; §8.6's notebook
   pseudocode flagged (its example `T_TRAIN_START`/`T_TRAIN_END` now spans
   eras and would be rejected — flagged, not silently fixed, since the real
   production window is still an open decision); gotcha 13 flagged (its
   `t_train_start=2022-01-01` example is now illegal on its own); new
   gotcha 15 summarizing the eras.

## Deliberately NOT done (explicit scope boundaries)

- **Regridding/resampling to unify eras into one trainable multi-resolution
  window.** A bigger, separate architecture decision requiring explicit
  user sign-off — this task's cross-era rejection is the *opposite* of
  that: it makes mixed-era windows fail loudly rather than silently
  misbehave or require a resampling design nobody's approved yet.
- **Wiring era1/era3's verified coordinate convention back into
  `core.archive_eras`/`core.config.grid_coordinates`.** The verification
  notebook (item 6 above) empirically confirmed both conventions, but
  `ArchiveEra` intentionally has no `lat0`/lon-step fields yet, and
  `grid_coordinates()` still returns era 2's coordinates unconditionally
  for every era. A real era-1 or era-3 window will now **load successfully**
  (shape-validated, single-era, no crash) but its `wave_lat`/`wave_lon`/
  `wind_lat`/`wind_lon` coordinate *labels* are only correct for era 2.
  Wiring the notebook's findings into the loader is a natural, well-scoped
  follow-up, deliberately left as one — not silently resolved here to avoid
  scope creep from "detect and reject the problem" into "also add a second,
  less-tested new grid convention" in the same change.
- **Ocean-grid loading support.** `core.config.grid_coordinates("ocean")`
  continues to raise unconditionally, unrelated to era. The era registry
  records ocean facts (including era 1's per-variable irregularity) for
  future use only.
- **Independently reconfirming era 3's cadence.** Recorded as `3*3600.0`
  with an inline comment flagging it as presumed-same-as-era-2, not
  independently reconfirmed. Cheap future follow-up: spot-check a handful
  of era-3 timestamps for actual 3h spacing.
- **Fixing `train/ingest.py`'s gap-warning hour-multiplier for non-3h
  eras.** Now that `enumerate_snapshots` is cadence-aware, the existing
  gap-warning messages' hardcoded `contiguous_gap * 3` hour-count is wrong
  for era-1 windows (should be `* 6`) — a message-text accuracy issue only
  (the gap thresholds still fire at the correct snapshot *counts*, this
  doesn't change *when* gaps are caught, only what the printed hour figure
  says). Not fixed here.
- **`notebooks/test_pipeline.ipynb`.** Its `T_START=2023-01-01`/
  `T_END=2025-01-01` also spans era1→era2 and would break under the new
  cross-era rejection — but this notebook is already broken for larger,
  unrelated reasons: it imports `train.dmd.exact_dmd` (renamed `dmd_fit` in
  task 06), `train.ou.fit_ou` (module deleted in task 06), constructs
  `WeatherModel.from_components(..., ou=ou, ...)` (renamed `mode_ou=`, now
  also needs `local_residual=`, in task 08), and calls
  `weather.full_report`/`sample_batch` (tasks 09/10, not yet implemented).
  It's a stale pre-v2-migration notebook; a date-range edit alone wouldn't
  make it runnable. Rewriting it end-to-end is already `tasks/11-notebooks.md`'s
  scope (⬜, not started) — this task does not touch it, but whoever
  picks up task 11 must choose a single-era window and derive `dt` via
  `core.archive_eras.era_for_timestamp`, not the flat constant.
- **Choosing the final production training window.** This task makes the
  *code* correctly detect/reject/handle eras and corrects the *false*
  "one uniform 4-year window" documentation claim; it does not pick the
  new number. `docs/model_design.md` §1 explicitly flags this as still open.

## Tests

- `tests/core/test_archive_eras.py` (new) — era boundary dispatch at exact
  seam timestamps (both sides of era1/era2 and era2/era3), era1's
  per-variable ocean shape irregularity, before-earliest-era raises, era3
  open-ended resolves (not raises), naive-datetime raises, registry
  ordered/contiguous.
- `tests/core/test_io.py` — `enumerate_snapshots` yields 6-hourly within
  era1; does not raise across an era boundary (rejection is ingest's job)
  and correctly varies step size across the seam.
- `tests/train/test_ingest.py` — era-1 window loads correctly at 6h
  cadence; cross-era window raises before any file I/O (asserted via
  `capsys`, nothing in stderr); same-variable shape changing mid-window
  raises; a shape exceeding its era's canonical bound raises. All
  pre-existing tests in this file (anchored at era-2 dates) pass unmodified.
- `tests/weather/test_model.py` — monkeypatches `era_for_timestamp` and the
  `dmd_fit`/`fit_local_residual` call sites to capture the `dt` actually
  passed, fits a small tier-1 model, asserts it came from the era registry.

## Acceptance criteria

- `poetry run pytest -q` passes with no regressions.
- `poetry run mypy --strict` (or the repo's `mypy src tests` CI invocation)
  is clean.
- `load_training_window(Path("data"), datetime(2022,6,1,tzinfo=UTC),
  datetime(2022,6,2,tzinfo=UTC), ["sigwh"])` succeeds against the real
  archive (era-1 window, 6-hourly).
- `load_training_window(Path("data"), datetime(2023,10,1,18,tzinfo=UTC),
  datetime(2023,10,2,3,tzinfo=UTC), ["sigwh"])` raises the new
  "spans more than one archive era" `ValueError`, not a numpy broadcast
  error.
- `notebooks/verify_era1_era3_grid_coordinates.ipynb` runs top-to-bottom
  against the real archive and states both verdicts with real evidence.
