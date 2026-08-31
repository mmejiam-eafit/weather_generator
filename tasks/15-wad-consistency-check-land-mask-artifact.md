# Task 15 — Fix `wad`/`wad_come_from` consistency check's land-cell false positive

**Depends on:** — · **Blocks:** 00–14 (all of them — every notebook calls
`train.ingest.load_training_window`, which runs this check unconditionally,
regardless of which variables are actually requested)
**Source:** Discovered 2026-08-14 during first real-archive validation after
the data disk was reconnected. Not one of `docs/open_questions.md`'s 15
tracked design questions — this is a code bug, not an open design question.
**Primary source files:** `src/train/ingest.py`
(`_check_wad_wad_come_from_consistency`), `tests/train/test_ingest.py`

**Why this is numbered 15 but listed first:** filed as task 15 per
instruction (append, don't renumber 00–14 and their notebooks/cross-refs).
But it is a hard execution blocker, not one more open question to
investigate whenever convenient — until it lands, **none** of tasks 00–14
can run their notebooks against real data at all, so it should be picked up
before them despite the number.

## 1. What problem this will fix

`load_training_window` unconditionally calls
`_check_wad_wad_come_from_consistency` on the first and last window
snapshots that have both `wad` and `wad_come_from` on disk — regardless of
whether the caller's `variables` list even includes them. On real era-3
data this check currently always fails:

```
ValueError: wad and wad_come_from do not differ by 180° at 2025-11-01T00:00:00+00:00. Max deviation: 180.00°.
```

Root cause, confirmed by direct investigation against the real archive
(2025-11-01, 2025-12-15, 2026-03-01, 2026-04-30 all checked): this is a
**land-cell fill-value artifact, not a real archive-convention violation**.

- Over ocean cells only (masked via `was`'s NaN pattern — same wind grid as
  `wad`/`wad_come_from`), the deviation from 180° is **exactly 0.0°** on
  every date checked.
- Over land cells, `wad` and `wad_come_from` store an identical raw fill
  code (e.g. raw `16` in both, observed directly), so they decode to the
  *same* degrees value instead of differing — a spurious 180° "deviation."
- The check's current `np.nanmax` runs over the whole grid, land included,
  so it is guaranteed to fail on every real snapshot: `was==NaN` land
  fraction was 44.8% in the sample checked, exactly matching the
  `wad==wad_come_from` fraction.

This is a real bug in the guard itself, not evidence of upstream archive
drift — §1.3's finding (`wad`/`wad_come_from` are a confirmed 180°-apart
pair, `wad_come_from` is the source of truth) still holds; this task only
fixes how it's *checked*.

## 2. Steps to fix it

1. In `_check_wad_wad_come_from_consistency`, build an ocean mask the same
   way `notebooks/tbd/01_understand_tucker_decomposition.ipynb` and the new
   `notebooks/*.ipynb` do (finite-valued cells of a same-grid continuous
   variable — `was` is the natural choice here, already loaded on the wind
   grid). Restrict the `nanmax` deviation check to that mask instead of the
   whole grid.
2. Decide how the mask gets into the function: either load `was` at the
   same timestamp inside the guard (one extra file read, only at the two
   checked snapshots), or pass in an already-decoded ocean mask from the
   caller. Prefer the former — keeps the guard self-contained and it's only
   evaluated twice per `load_training_window` call, not once per snapshot.
3. Add a regression test in `tests/train/test_ingest.py`: a synthetic
   window where land cells have identical `wad`/`wad_come_from` fill codes
   (previously would false-positive) alongside ocean cells that are cleanly
   180° apart — confirm the fixed check passes. Keep (or adapt) an existing
   test for the real drift case: ocean cells that are *not* 180° apart
   should still raise.
4. Re-run `load_training_window` against a real short era-3 window
   end-to-end (reproduction command used during discovery:
   `load_training_window(Path("data"), datetime(2025,11,1,tzinfo=timezone.utc), datetime(2025,11,1,6,tzinfo=timezone.utc), ["sigwh","wsh"])`)
   and confirm it no longer raises.
5. Run the full test suite to confirm no regressions.

## 3. Notebook(s) needed

None. This is a direct fix to `src/train/ingest.py` plus a unit test, not
an open design question needing an exploratory notebook. (The diagnostic
work — ocean-mask-conditioned deviation check across four dates — was
already done ad hoc via `poetry run python` during discovery and doesn't
need to be preserved as a notebook.)

## 4. Conclusion

**Fixed 2026-08-14.** `_check_wad_wad_come_from_consistency` in
`src/train/ingest.py` now masks to ocean cells (via `was`'s NaN pattern,
loaded at the same timestamp) before computing the deviation-from-180°
check, falling back to the whole grid if `was` isn't present (e.g. a
synthetic test archive). Two regression tests added to
`tests/train/test_ingest.py`:
`test_load_training_window_wad_consistency_check_ignores_land_cells`
(land-fill-code false positive no longer raises) and
`test_load_training_window_wad_consistency_check_still_detects_ocean_drift`
(a genuine ocean-cell mismatch still raises — masking isn't vacuous). Full
test suite: 152 passed (was 150; +2 new).

Confirmed directly against the real archive: `load_training_window` now
loads real era-3 windows cleanly, e.g. 2025-11-01 (`sigwh`, `wsh`) and
2026-02-01 (`was`, `wad`, `wad_come_from`) — both report `max deviation
0.00° (ocean cells only)`. §1.3's finding stands unchanged (`wad`/
`wad_come_from` are a confirmed 180°-apart pair; `wad_come_from` is the
source of truth) — only the guard's implementation was wrong, not the
underlying archive convention.

Tasks 00–14 are now unblocked and can proceed to actually run their
notebooks against real data.
