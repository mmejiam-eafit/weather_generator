# Task 11 — Notebooks

**Depends on:** 08, 09, 10. **Blocks:** 13.

## Current state

`notebooks/verify_direction_convention.ipynb` exists and is complete — it
empirically established the §2.3/§2.4 convention table. Check whether
`notebooks/test_pipeline.ipynb` currently exists; if so, it reflects v1's
single-transformer, no-trend, no-local-residual pipeline and needs the
rewrite below. `notebooks/verify_ocean_grid_orientation.ipynb` does not
exist yet.

## Changes

### 11a. New: `notebooks/verify_ocean_grid_orientation.ipynb`

Per §2.7's TASK paragraph — this **gates** enabling `sst`/`ocs`/`ocd`
(tier 2/3, §3.8) and is a hard prerequisite for
`tests/core/test_grid_ocean.py::test_orientation` (§12.6: "do not write this
test (or any ocean-grid pipeline code) before that notebook exists and
passes"). Implement:

- Correlate `sst`'s NaN pattern against a Natural Earth land polygon
  rasterized at 0.125°, across all four `(row 0 south vs. north) ×
  (rows increase vs. decrease)` candidate orientations.
- **≥30 arbitrary days**, not a single snapshot — this supersedes the
  lightweight, single-snapshot, self-derived orientation check used in
  `verify_direction_convention.ipynb`'s ocean-grid section (§2.3, §2.4,
  §2.7: that check is good directional evidence but "not a substitute for
  this notebook").
- Confirm the correct candidate shows **≥85%** agreement and every
  alternative **≤60%**.
- Independently cross-check with the known warm-pool location (western
  Pacific).
- On success, report the concrete `OCEAN_GRID_LAT_AT_ROW0`/
  `OCEAN_GRID_LAT_STEP_DEG` values — these are currently unset in
  `src/core/config.py` (see the comment there, line 21-24) precisely because
  this notebook hasn't run yet. Hardcoding those two constants into
  `core/config.py` is a one-line follow-up once this notebook passes; it is
  not part of this task itself (keep the notebook and the config edit as
  separably-reviewable changes).

### 11b. Rewrite `notebooks/test_pipeline.ipynb` per §8.6

Ten cells, exact structure:

1. **Setup** — imports, `SEED`, all four window boundaries:
   `T_TRAIN_START` = 2022-01-01, `T_TRAIN_END` = 2026-01-01 (exclusive),
   `T_METRICS_START`/`T_METRICS_END` = 2026 H1 window,
   `T_EXTRAP_START`/`T_EXTRAP_END` = extrapolation window.
2. **Ingest** — `load_training_window(data_dir, T_TRAIN_START, T_TRAIN_END,
   variables)` with the tier-1 `variables` list
   (`["sigwh", "wsh", "wsp", "was", "wsd", "wad_come_from"]` — `wad_come_from`,
   not `wad`; task 08's note applies here too).
3. **Build + `fit_transform`** the preprocessing Pipeline (task 08's
   construction), with `Pipeline(memory=Path("./cache"))` for iteration
   caching; print the `trend_significant_cells` fraction.
4. **Inspect `Z`** — QQ plots at random cells (should be `N(0,1)`).
5. **Tucker** at 2+ candidate ranks (e.g. 200 vs 400); print
   `reconstruction_error` for each. **First sub-cell prints
   `partial_tucker.__doc__`** per gotcha #21 — confirm the tuple-unpacking
   pattern before relying on it.
6. **DMD** — `variant="exact"` explicitly, with a comment explaining why:
   this cell reruns on every rank sweep, and BOP-DMD's ~100-model bagging
   makes that painfully slow during iteration. Switch to `"bop"` only in the
   final production-run cell (below), not here. Plot eigenvalues.
7. **Mode OU** — `fit_mode_ou(core, dmd)`.
8. **Local residual** — `coherent_reconstruction` → `fit_local_residual`;
   print `active_mask` fraction; plot the variance-ratio map.
9. **Assemble** — `WeatherModel.from_components(...)`; `model.save(...)`.
10. **Validation** — `full_report(...)` over both windows (§6.A held-out
    metrics, §6.B extrapolation checks, §6.D hurricane-season checks),
    output to `./reports/v0.1.0_validation.pdf`.

Add a final markdown + code cell for the **"Production runs" one-liner**
(§8.6) — `WeatherModel(..., dmd_variant="bop", ...)`, the one place `"bop"`
belongs, with an inline comment pointing back to why every upstream cell
uses `"exact"`. Note `t_end` is exclusive (§8.6 "Important" — last training
snapshot is `t_end - 3h`).

## Acceptance criteria

- `verify_ocean_grid_orientation.ipynb` runs end-to-end against the real
  archive and reports pass/fail against the ≥85%/≤60% thresholds.
- `test_pipeline.ipynb` runs end-to-end
  (`poetry run jupyter nbconvert --to notebook --execute`) on at least a
  small synthetic/subset archive without error, producing a saved model
  directory and a validation PDF.
