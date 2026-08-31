# Task 12 — Integration tests + import-boundary enforcement

**Depends on:** 10. **Blocks:** 13.

## Current state

No `tests/test_import_boundaries.py`, no `tests/weather/test_integration.py`
exist yet.

## Changes

### 12a. New: `tests/test_import_boundaries.py`

AST-based enforcement of §8.3's dependency rules:

- `core/` imports only stdlib, numpy, xarray, pydantic-settings.
- `train/` imports from `core` plus numpy/scipy/scikit-learn/xarray/
  tensorly/pydmd; **never** imports from `weather`.
- `weather/` imports from `core` and `train` plus numpy/scipy/
  scikit-learn/xarray/matplotlib.

Single required test per §12.5:
`test_train_does_not_import_weather` — walk `src/train/*.py`'s AST
`Import`/`ImportFrom` nodes, fail if any references `weather`.

### 12b. New: `tests/weather/test_integration.py` (marked `@pytest.mark.integration`)

All six listed in §12.7, each against a synthetic archive fixture (not the
real 2022-2025 data — these must run in CI):

- `test_end_to_end_marginal_fit` — train on 3 synthetic months, simulate 1
  overlapping month, assert per-cell mean/std within 10% of held-out truth.
- `test_extrapolation_no_drift` — train on 3 months, extrapolate 6 months
  past training end (no ground truth), assert per-cell simulated mean over
  the extrapolation window is within 3σ of the fitted climatology's seasonal
  expectation. Guards against `Re(lam_cont) > 0` runaway DMD modes producing
  unbounded growth on long extrapolation.
- `test_extrapolation_output_validity` — same setup; assert all outputs
  finite, directional variables in `[0, 360)`, magnitude variables `≥ 0`
  (§6.B.13 in test form).
- `test_two_seed_stability` — two seeds; assert per-cell mean-of-difference
  has zero-mean Gaussian distribution (paired samples, §6.B.14 in test
  form).
- `test_hurricane_season_reproduction` (§6.D.18 in test form) — synthetic
  archive covering ≥1 full Aug-Oct window; train, simulate the same window
  multi-seed, assert `detect_storm_signatures`' (task 10) feature-count/
  intensity/translation-speed distribution on simulated output is
  statistically consistent with the same detector run on the real
  synthetic-archive input. **The synthetic fixture must contain an actually
  planted storm-like feature** (rotating wind + elevated wave height,
  translating at a plausible speed) or this test is vacuous — build the
  fixture the same way task 10's `detect_storm_signatures` unit test does.
- `test_hurricane_season_extrapolation_structural` (§6.D.19 in test form) —
  same synthetic setup, extrapolate past training end through a synthetic
  Aug-Oct window, assert at least one storm-signature-qualifying feature is
  detected.

## Acceptance criteria

- `poetry run pytest -m integration` passes with all six integration tests
  plus the import-boundary test green.
- `poetry run pytest --cov=src` reports ≥85% coverage (§12.4) once tasks
  01-12 are all complete — if short, identify the specific gap before task
  13 closes out (don't pad coverage with low-value tests to hit the number).
