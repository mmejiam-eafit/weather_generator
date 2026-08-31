# Task 10 — Validation: hurricane-season checks + pipeline step-name fixes

**Depends on:** 08, 09. **Blocks:** 11, 12.

## Current state

`src/weather/validate.py` implements checks 1-8ish today: `qq_plot_check`,
`spectra_check`, `variogram_check`, `joint_density_check`,
`extreme_statistics_check`, `throughput_benchmark`,
`held_out_normality_check`, plus `full_report`. Three call sites reference
the now-deleted `"directional"` pipeline step by name:
`_load_real_physical` (lines 58-61) and `joint_density_check` (lines
506-511, which reads `directional.pairs_` to look up a variable's
`u_name`/`v_name`).

## Changes — `src/weather/validate.py`

1. **Fix the `named_steps["directional"]` references** to use both new
   steps:
   - `_load_real_physical`: chain `coming_from_to_uv.transform` then
     `going_to_to_uv.transform`, then `reproject.transform`; invert in
     either order afterward (§3.7: disjoint pair sets, so composition order
     doesn't matter for the inverse either) — e.g.
     `going_to_to_uv.inverse_transform` then
     `coming_from_to_uv.inverse_transform`.
   - `joint_density_check` (~line 506-511): when looking up `u_name`/
     `v_name` for a given variable pair, search
     `coming_from_to_uv.pairs_ + going_to_to_uv.pairs_` (concatenated)
     instead of a single `directional.pairs_`.
2. **Add the §6.D storm-signature detector** as a new function, shared by
   both new checks:
   ```python
   def detect_storm_signatures(
       was: xr.DataArray, sigwh: xr.DataArray,
       u_wind: xr.DataArray, v_wind: xr.DataArray,
       lat_bounds: tuple[float, float] = (8.0, 35.0),
       lon_bounds: tuple[float, float] = (-100.0, -15.0),
       wind_threshold_ms: float = 17.0,
   ) -> ...:  # per-snapshot candidate feature list/mask — see §6.D
   ```
   Three required criteria (§6.D), restricted to the tropical/subtropical
   North Atlantic box given by the default bounds above:
   - A contiguous patch where `was >= wind_threshold_ms` co-located with
     elevated `sigwh` ("high energy density").
   - A coherent cyclonic (positive, Northern Hemisphere) vorticity maximum
     from `(u_wind, v_wind)` — `ζ = ∂v/∂x − ∂u/∂y`, finite-differenced on the
     wave grid — co-located with the patch.
   - Tracked across consecutive 3h snapshots: patch/vorticity-max centroid
     persists ≥24h (≥8 consecutive qualifying snapshots) and translates at
     ~10-30 km/h between snapshots (use the file's existing `_haversine_km`
     between consecutive centroids, divided by 3h).
3. **Check 18 — `hurricane_season_reproduction_check`** (§6.D.18): for each
   Aug-Oct window in 2022, 2023, 2024, 2025 (all in-sample), run the
   detector on real archive data (via `load_training_window`, the same
   loader `_load_real_physical` already uses) to get a per-season baseline
   (feature count, peak intensity, typical translation speed); run the
   detector on multi-seed `simulate_field` output over the same windows;
   compare the simulated-across-seeds distribution to the real per-season
   baseline via a statistical test (e.g. KS test or bootstrap CI overlap).
   **Match statistics across the four seasons, not individual storms**
   (§1's stated non-goal).
4. **Check 19 — `hurricane_season_extrapolation_structural_check`**
   (§6.D.19): simulate the actual 2026 Aug-Oct extrapolation window; run the
   same detector; assert **at least one** qualifying feature was detected —
   a capability check, not a statistical-match check (no ground truth exists
   for this window).
5. Wire both into `full_report` (§6, notebook Cell 10) alongside the
   existing checks — add a PDF page for each, following this file's
   existing `_text_page`/figure pattern.

## Tests — `tests/weather/test_validate.py`

- Update any existing test referencing `"directional"` named step to the new
  two-step names.
- Add unit tests for `detect_storm_signatures` on a synthetic snapshot
  sequence: a hand-constructed cyclone-like feature (rotating wind field +
  elevated wave height, translating at a plausible speed) must be detected;
  a synthetic "just windy, no rotation" field must **not** be detected —
  this second case is what proves the vorticity criterion is doing real
  work, not just the wind-speed threshold alone.

## Acceptance criteria

- `grep -n 'named_steps\["directional"\]' src/weather/validate.py` returns
  nothing.
- `detect_storm_signatures` fires on the synthetic cyclone fixture and does
  not fire on the synthetic "windy but not rotating" fixture.
- `full_report` runs both new checks without error on a small synthetic
  archive covering at least one Aug-Oct window.
