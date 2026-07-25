# v3 Implementation Handoff — Physical-Consistency Filtering

**Target reader:** LLM agent working directly in the `weather-generator` repository.

**Scope:** implement post-hoc physical-consistency filtering. This is a single, contained feature addition; the rest of v2 is unchanged.

**Effort estimate:** ~1 day of focused work — 1 new module (~250 lines), 1 API-surface edit, ~10 new tests, minor docs update.

---

## 1. Context

v2 is implemented and running in the repo. The trained model produces samples that are statistically calibrated (per-cell marginals and cross-cell correlations match training data) but not guaranteed to be individually physically consistent. Sample outputs may violate:

- The wave-energy budget: `sigwh² ≈ wsh² + pswh²` (approximate energy conservation for the sea-state decomposition).
- Fetch-limited wave-wind coupling: in enclosed seas, wave height should scale with wind speed by Pierson-Moskowitz.
- Peak-period plausibility: `pwp` should fall in a physically meaningful range.

For most consumers this doesn't matter — statistical calibration is what they need. For a small subset (consumers feeding samples into downstream physics-based simulators, or ones whose acceptance tests check physical relationships), per-sample validity is required.

**v3 adds one thing:** a rejection-sampling wrapper around `sample_batch` that discards samples violating documented constraints and resamples. Opt-in via a new public function; the existing `sample_batch` behavior is unchanged.

**What v3 is not:**
- Not a change to training. The fitted model is identical to v2.
- Not physics-informed modeling. The constraints are checked post-hoc, not enforced during fitting.
- Not a change to `WeatherModel`, the preprocessing Pipeline, or any of Tucker/DMD/OU.
- Not a change to the data-loading or ingest pipeline.

If a change to any of the above is being proposed while implementing v3, stop and confirm before proceeding.

---

## 2. Scope of changes

### New files
- `src/weather/physics.py` — the constraint definitions and rejection-sampling entry point. See §4.
- `tests/weather/test_physics.py` — new tests. See §7.

### Modified files
- `src/weather/__init__.py` — add re-exports for the new public API. See §5.
- `docs/model_design.md` — bump version tag from v2 to v3; add a paragraph to the "What changed" section; ensure §3.15 and §5.9 (which already describe the filtering in the design doc) match the implementation. Confirm they do; if drift is found, update the doc to match implementation, not the reverse.
- `CHANGELOG.md` — add v0.2.0 entry describing physical-consistency filtering.

### Files that must NOT change
- Anything under `src/core/`.
- Anything under `src/train/`.
- `src/weather/model.py` (`WeatherModel` class).
- `src/weather/simulate.py` (existing `sample`, `sample_batch`, `simulate_field`).
- `src/weather/validate.py`.
- `pyproject.toml` runtime dependencies — no new deps needed.
- `notebooks/test_pipeline.ipynb`.

If any of these seems to need modification during implementation, that's a signal that the design has been misread. Stop and re-check the design doc's §5.9 for the intended integration point.

---

## 3. Design contract (from the design doc)

Summary of the intended behavior, sourced from `docs/model_design.md` §3.15 and §5.9. Re-read those sections in full before starting; this is a compressed version for reference.

**Contract:**
- Public function `sample_batch_with_physics(model, lats, lons, times, variables, constraints, max_retries, seed) -> (values, accepted)`.
- For each query point independently, draw a sample from `sample_batch` and evaluate the constraints. If all pass, accept. If any fails, resample with a new seed derived from the original.
- After `max_retries` attempts, return the last sample regardless and set `accepted[i] = False`.
- Log per-constraint rejection rates at completion (not per-attempt — that would be a firehose).
- Default `max_retries = 10`.
- The `constraints` parameter is a `PhysicsConstraints` frozen dataclass with tunable tolerances.

**What "physics" means here:** three specific constraints, defined in §4. Not general physical validity. Anything not in the list is out of scope for v3.

---

## 4. New module: `src/weather/physics.py`

### 4.1 Module structure

```python
"""Post-hoc physical-consistency filtering for weather samples.

See docs/model_design.md §3.15, §5.9 for design rationale.
Not called during training. Applied only at query time via rejection sampling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from weather.model import WeatherModel
from weather.simulate import sample_batch

# --- constants ------------------------------------------------------------
_G = 9.80665                          # m/s²
_PIERSON_MOSKOWITZ_COEFF = 0.21       # dimensionless, from PM 1964

# --- public dataclass -----------------------------------------------------

@dataclass(frozen=True, slots=True)
class PhysicsConstraints:
    """Tunable tolerances for post-hoc physical-consistency filtering.

    See model_design.md §5.9 for the constraint definitions.
    """
    energy_budget_atol_m2: float = 0.5      # m² tolerance on |sigwh² - wsh² - pswh²|
    fetch_scaling_atol_m: float = 1.5       # m tolerance on |wsh - 0.21·was²/g|
    peak_period_min_s: float = 2.0          # lower bound on pwp
    peak_period_max_s: float = 25.0         # upper bound on pwp
    disabled: frozenset[str] = field(default_factory=frozenset)
    # ^ names of constraints to disable, e.g. frozenset({"fetch_scaling"}) if the
    #   fetch-limited mask isn't available yet.


# --- private: constraint evaluators --------------------------------------

def _check_energy_budget(values, variables, atol_m2): ...
def _check_fetch_scaling(values, variables, lats, lons, atol_m): ...
def _check_peak_period(values, variables, min_s, max_s): ...


# --- public: entry points ------------------------------------------------

def check_constraints(
    values: np.ndarray,          # (N, V) sample values in physical units
    variables: list[str],
    lats: np.ndarray,            # (N,)
    lons: np.ndarray,            # (N,)
    constraints: PhysicsConstraints,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate all enabled constraints on a batch of samples.

    Returns
    -------
    all_pass : (N,) bool
        True where all enabled constraints pass for that query point.
    failures : dict[str, (N,) bool]
        Per-constraint failure mask. True where that specific constraint fails.
        Keys: "energy_budget", "fetch_scaling", "peak_period".
        Disabled constraints do not appear in the dict.
    """
    ...


def sample_batch_with_physics(
    model: WeatherModel,
    lats: np.ndarray,            # (N,)
    lons: np.ndarray,            # (N,)
    times: np.ndarray,           # (N,) datetime64[s]
    variables: list[str] | str,
    constraints: PhysicsConstraints,
    max_retries: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample with rejection until physical constraints are satisfied.

    See model_design.md §5.9.

    Returns
    -------
    values : (N, V) float64
        Sampled values. For query points where max_retries was exceeded,
        the last-attempt values are returned (see accepted).
    accepted : (N,) bool
        True where the returned sample satisfies all enabled constraints.
        False where max_retries was exceeded.
    """
    ...
```

### 4.2 Constraint 1: wave-energy budget

**Physical basis:** significant wave height for a mixed sea state is the RMS of wind-sea and swell heights: `sigwh² = wsh² + pswh² + higher_swell_terms`. For tier 1 (wind-sea + primary swell only), the higher-swell contribution is small.

**Check:**
```
|sigwh² - wsh² - pswh²| < energy_budget_atol_m²
```

**Implementation:**
```python
def _check_energy_budget(values, variables, atol_m2):
    """Evaluate the energy-budget constraint per query point.

    Returns (N,) bool mask. True where constraint FAILS.
    """
    idx_sigwh = variables.index("sigwh")
    if "wsh" not in variables or "pswh" not in variables:
        return np.zeros(values.shape[0], dtype=bool)   # not enough vars to check
    idx_wsh = variables.index("wsh")
    idx_pswh = variables.index("pswh")

    lhs = values[:, idx_sigwh] ** 2
    rhs = values[:, idx_wsh] ** 2 + values[:, idx_pswh] ** 2
    return np.abs(lhs - rhs) > atol_m2
```

**Guard:** if any of `sigwh`, `wsh`, `pswh` is absent from `variables`, the constraint is silently skipped (returns all-false). This is intentional — the check is meaningful only when all three are being sampled together. Do not raise.

**Default tolerance `0.5 m²`:** derived from real archive data; verified in `test_energy_budget_constraint_accepts_real_data`. Do not change without also re-running that test.

### 4.3 Constraint 2: fetch-limited Pierson-Moskowitz

**Physical basis:** in enclosed seas or short-fetch coastal regions, wave height in equilibrium with local wind follows the Pierson-Moskowitz spectrum for a fully-developed sea: `H_s = 0.21 · U² / g`. Only applies where fetch is genuinely limited; in the open ocean, waves can be swell-dominated or arriving from distant weather systems.

**Check:**
```
if is_fetch_limited(lat, lon):
    require |wsh - 0.21 · was² / g| < fetch_scaling_atol_m
```

**Fetch-limited mask.** For v3, use a hardcoded conservative approximation: mark a query cell as fetch-limited only if it's in one of these enclosed seas (bounding boxes):

```python
_FETCH_LIMITED_REGIONS = [
    # (lat_min, lat_max, lon_min, lon_max, name)
    (30.0,  46.0,   0.0,  36.0, "mediterranean"),
    (12.0,  30.0,  32.0,  43.0, "red_sea"),
    (24.0,  30.0,  48.0,  56.0, "persian_gulf"),
    (40.0,  47.0,  27.0,  42.0, "black_sea"),
    (53.0,  66.0,   9.0,  30.0, "baltic_sea"),
]

def _is_fetch_limited(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return (N,) bool mask. True where fetch is considered limited."""
    mask = np.zeros(lats.shape, dtype=bool)
    lons_normalized = lons % 360.0
    for lat_min, lat_max, lon_min, lon_max, _ in _FETCH_LIMITED_REGIONS:
        in_region = (
            (lats >= lat_min) & (lats <= lat_max)
            & (lons_normalized >= lon_min) & (lons_normalized <= lon_max)
        )
        mask |= in_region
    return mask
```

**Implementation:**
```python
def _check_fetch_scaling(values, variables, lats, lons, atol_m):
    """Fetch-limited PM constraint. Returns (N,) bool mask of failures."""
    if "wsh" not in variables or "was" not in variables:
        return np.zeros(values.shape[0], dtype=bool)
    idx_wsh = variables.index("wsh")
    idx_was = variables.index("was")

    fetch_mask = _is_fetch_limited(lats, lons)
    wsh_predicted = _PIERSON_MOSKOWITZ_COEFF * values[:, idx_was] ** 2 / _G
    violation = np.abs(values[:, idx_wsh] - wsh_predicted) > atol_m
    return fetch_mask & violation
```

**Guard:** for open-ocean queries (outside the fetch-limited regions), the constraint is silently satisfied.

**Default tolerance `1.5 m`:** generous. PM is an equilibrium relation and real data has transient conditions. Verified in `test_fetch_scaling_constraint_accepts_real_data`.

**Extension point:** the hardcoded region list is v3-appropriate but not production-quality. A future v4 could compute a proper fetch mask from bathymetry or coastline distance. Leave a comment in the module noting this.

### 4.4 Constraint 3: peak-period sanity

**Physical basis:** peak wave period `pwp` outside `[2, 25]` seconds is physically implausible. Very short periods (<2s) are capillary/ripple regime, not gravity waves. Very long periods (>25s) don't occur naturally in normal oceanography.

**Check:**
```
require peak_period_min_s < pwp < peak_period_max_s
```

**Implementation:**
```python
def _check_peak_period(values, variables, min_s, max_s):
    """Peak-period range constraint. Returns (N,) bool mask of failures."""
    if "pwp" not in variables:
        return np.zeros(values.shape[0], dtype=bool)
    idx_pwp = variables.index("pwp")

    pwp_vals = values[:, idx_pwp]
    return (pwp_vals < min_s) | (pwp_vals > max_s)
```

**Guard:** if `pwp` is not sampled, constraint is skipped.

**Defaults `2 s` and `25 s`:** hard physics limits, not sample statistics. Do not tighten without physical justification.

### 4.5 Composition in `check_constraints`

```python
def check_constraints(values, variables, lats, lons, constraints):
    failures = {}
    if "energy_budget" not in constraints.disabled:
        failures["energy_budget"] = _check_energy_budget(
            values, variables, constraints.energy_budget_atol_m2
        )
    if "fetch_scaling" not in constraints.disabled:
        failures["fetch_scaling"] = _check_fetch_scaling(
            values, variables, lats, lons, constraints.fetch_scaling_atol_m
        )
    if "peak_period" not in constraints.disabled:
        failures["peak_period"] = _check_peak_period(
            values, variables,
            constraints.peak_period_min_s, constraints.peak_period_max_s,
        )

    if not failures:
        return np.ones(values.shape[0], dtype=bool), failures

    any_failure = np.any(np.stack(list(failures.values())), axis=0)
    return ~any_failure, failures
```

### 4.6 Rejection sampling in `sample_batch_with_physics`

```python
def sample_batch_with_physics(
    model, lats, lons, times, variables, constraints,
    max_retries=10, seed=None,
):
    import logging
    logger = logging.getLogger(__name__)

    if isinstance(variables, str):
        variables = [variables]

    N = len(lats)
    V = len(variables)
    values = np.full((N, V), np.nan, dtype=np.float64)
    accepted = np.zeros(N, dtype=bool)
    pending = np.ones(N, dtype=bool)

    # Per-constraint failure counters for diagnostics.
    total_failures: dict[str, int] = {}
    total_attempts = 0

    rng = np.random.default_rng(seed)

    for attempt in range(max_retries + 1):
        if not pending.any():
            break
        idx = np.where(pending)[0]
        attempt_seed = int(rng.integers(0, 2**31 - 1))
        attempt_values = sample_batch(
            model, lats[idx], lons[idx], times[idx], variables,
            seed=attempt_seed,
        )                                                 # (n_pending, V)

        all_pass, failures = check_constraints(
            attempt_values, variables,
            lats[idx], lons[idx],
            constraints,
        )

        # Accumulate diagnostics
        total_attempts += len(idx)
        for name, mask in failures.items():
            total_failures[name] = total_failures.get(name, 0) + int(mask.sum())

        # Accept the passing ones
        accepted_this_attempt = idx[all_pass]
        values[accepted_this_attempt] = attempt_values[all_pass]
        accepted[accepted_this_attempt] = True
        pending[accepted_this_attempt] = False

        # For the remaining pending ones, keep the last-attempt values as fallback
        # in case max_retries is exceeded.
        rejected_this_attempt = idx[~all_pass]
        values[rejected_this_attempt] = attempt_values[~all_pass]

    # Log per-constraint rejection rates.
    if total_attempts > 0:
        for name, count in total_failures.items():
            rate = count / total_attempts
            logger.info(
                "physics constraint '%s' failure rate: %.1f%% "
                "(%d failures / %d attempts)",
                name, rate * 100, count, total_attempts,
            )

    n_unaccepted = int((~accepted).sum())
    if n_unaccepted > 0:
        logger.warning(
            "%d/%d query points exceeded max_retries=%d and returned unaccepted",
            n_unaccepted, N, max_retries,
        )

    return values, accepted
```

**Notes on the implementation:**

- Seed derivation is done via a `Generator` split from the caller's seed. This ensures reproducibility: the same top-level seed produces the same sequence of retry seeds. Don't just increment the seed by 1 per retry — that produces correlated samples across retries.
- Pending mask tracks which points still need a valid sample. Once a point is accepted, it's frozen; retries only run on the remaining pending points.
- Rejected samples' *last* values are returned as fallback if max_retries is exceeded. This means callers get *something* even on failure, and the `accepted` mask tells them whether to trust each result.
- Per-constraint failure counts are logged as `INFO`. Overall unaccepted count is logged as `WARNING` if any exist.

---

## 5. Public API integration

### 5.1 Edit `src/weather/__init__.py`

Add the imports and update `__all__`:

```python
"""weather — public API for the stochastic weather generator."""
from weather.model import WeatherModel
from weather.simulate import sample, sample_batch, simulate_field
from weather.physics import (
    PhysicsConstraints,
    check_constraints,
    sample_batch_with_physics,
)
from weather.validate import full_report

__all__ = [
    "WeatherModel",
    "sample", "sample_batch", "simulate_field",
    "PhysicsConstraints", "check_constraints", "sample_batch_with_physics",
    "full_report",
]
```

### 5.2 No changes to `WeatherModel`

`WeatherModel` does not gain any new methods, attributes, or `__init__` parameters. The physical-consistency machinery lives entirely outside the model class. This is intentional — the model is a container of fitted state and shouldn't grow when new post-processing options are added.

### 5.3 No changes to `sample_batch`

`sample_batch` remains exactly as it is in v2. `sample_batch_with_physics` is a wrapper that calls it internally.

---

## 6. Package dependency rules

- `weather/physics.py` may import from `numpy`, `xarray`, stdlib, and from `weather.model` and `weather.simulate`.
- It must NOT import from `train/` — that would violate the layering rules (see design doc §8.3).
- It must NOT import from `core/` — physics-consistency filtering is a public-API concern, not a low-level primitive.

Confirm by running the import-boundary test after implementation:
```
pytest tests/test_import_boundaries.py
```

---

## 7. Tests

### 7.1 Test file: `tests/weather/test_physics.py`

Ten required tests. Each is a specific pytest case with a well-defined assertion. Do not add tests beyond these unless a bug or edge case is found during implementation.

**Constraint acceptance on real data** (run against the archive at `./data`; mark as `@pytest.mark.integration` since they need real data):

```python
@pytest.mark.integration
def test_energy_budget_constraint_accepts_real_data():
    """>99% of ocean cells in 2026-H1 satisfy sigwh² ≈ wsh² + pswh² at 0.5 m² tol."""
    # Load a sample of 2026-H1 snapshots.
    # Compute |sigwh² - wsh² - pswh²| per ocean cell.
    # Assert >99% of ocean cells are within 0.5 m².
    # If the assertion fails, the tolerance is too tight or the constraint
    # is misspecified. Do NOT relax the tolerance to make the test pass;
    # instead flag it for review.

@pytest.mark.integration
def test_fetch_scaling_constraint_accepts_real_data():
    """>90% of cells in Mediterranean/Baltic/etc. satisfy PM at 1.5 m tol."""

@pytest.mark.integration
def test_peak_period_constraint_accepts_real_data():
    """100% of real pwp values in 2026-H1 fall in [2, 25] s."""
```

**Rejection-sampling behavior** (unit tests, no real data needed):

```python
def test_check_constraints_returns_correct_masks():
    """Synthetic sample known to violate energy_budget by 1.0 m² is caught."""
    values = np.array([[4.0, 2.0, 2.0, 1.0]])   # sigwh, wsh, pswh, was
    variables = ["sigwh", "wsh", "pswh", "was"]
    lats = np.array([0.0])
    lons = np.array([0.0])
    constraints = PhysicsConstraints()
    all_pass, failures = check_constraints(values, variables, lats, lons, constraints)
    # sigwh² = 16, wsh² + pswh² = 8, diff = 8 > 0.5 tol → fail
    assert not all_pass[0]
    assert failures["energy_budget"][0]

def test_disabled_constraints_are_skipped():
    """A constraint in `disabled` doesn't appear in the failures dict."""
    constraints = PhysicsConstraints(disabled=frozenset({"energy_budget"}))
    # Same synthetic values as above.
    all_pass, failures = check_constraints(values, variables, lats, lons, constraints)
    assert all_pass[0]                             # passes because check is disabled
    assert "energy_budget" not in failures

def test_missing_variables_skip_constraint():
    """If `sigwh` isn't in the sample, energy_budget check is skipped."""
    values = np.array([[2.0, 1.0]])                # only wsh, pswh
    variables = ["wsh", "pswh"]
    lats = np.array([0.0])
    lons = np.array([0.0])
    constraints = PhysicsConstraints()
    all_pass, failures = check_constraints(values, variables, lats, lons, constraints)
    assert all_pass[0]
    # energy_budget key present but all False (no failure detected):
    assert not failures["energy_budget"][0]

def test_fetch_limited_mask_open_ocean_passes():
    """PM check does not fail at a mid-Pacific query point."""
    lats = np.array([0.0])
    lons = np.array([-140.0 % 360.0])
    # Values that would VIOLATE PM if enforced:
    values = np.array([[5.0, 3.0]])                # wsh=5m at was=3m/s (impossible in equilibrium)
    variables = ["wsh", "was"]
    constraints = PhysicsConstraints()
    all_pass, failures = check_constraints(values, variables, lats, lons, constraints)
    assert all_pass[0]                             # passes because open ocean is not fetch-limited

def test_fetch_limited_mask_mediterranean_fails():
    """PM check fails at a Mediterranean query point with implausible values."""
    lats = np.array([40.0])
    lons = np.array([15.0])
    # wsh way above PM prediction:
    values = np.array([[5.0, 3.0]])
    variables = ["wsh", "was"]
    constraints = PhysicsConstraints()
    all_pass, failures = check_constraints(values, variables, lats, lons, constraints)
    assert not all_pass[0]
    assert failures["fetch_scaling"][0]
```

**Rejection loop behavior** (unit tests, mock `sample_batch`):

```python
def test_sample_batch_with_physics_returns_valid_samples(mocker):
    """After enough retries, all returned samples pass constraints."""
    # Mock sample_batch to alternate valid/invalid outputs, verify that
    # the final returned samples all satisfy check_constraints.

def test_sample_batch_with_physics_max_retries_exceeded(mocker):
    """Points that never satisfy constraints return accepted=False."""
    # Mock sample_batch to always return invalid values.
    # Assert accepted is all-False and values are the last-attempt values.

def test_sample_batch_with_physics_seed_reproducibility(mocker):
    """Same top-level seed produces byte-exact output on re-run."""
    # Run twice with the same seed, assert values and accepted are identical.

def test_sample_batch_with_physics_seed_produces_uncorrelated_retries(mocker):
    """Retry samples aren't just seed+1; they're properly split from the RNG."""
    # This one is subtle. Track the seeds passed to sample_batch across retries
    # and confirm they're not sequential.
```

### 7.2 Existing tests must continue to pass

Confirm the full existing suite runs green after v3 additions:
```
poetry run pytest
```

If any existing test fails, that indicates unintended coupling — v3 was supposed to be purely additive. Investigate before proceeding.

---

## 8. Documentation updates

### 8.1 `docs/model_design.md`

The design doc already contains §3.15 (rationale) and §5.9 (algorithm spec) covering physical-consistency filtering. Two edits:

1. **Bump version.** Change the title from "Design Handoff (v2)" to "Design Handoff (v3)".
2. **Update the "What changed" section.** Rewrite it as "What changed vs. v2" listing only:
   - Post-hoc physical-consistency filtering added as opt-in via `sample_batch_with_physics`. New module `weather/physics.py`. Model itself is unchanged.

The v1→v2 history that's currently in that section can be dropped — anyone reading v3 has already seen v2 shipped.

3. **Cross-check.** Read §3.15 and §5.9 as implemented, and verify the design doc matches. If any drift is found:
   - If the implementation is correct and the doc is wrong: fix the doc.
   - If the doc is correct and the implementation is wrong: fix the implementation.

Do not fix drift silently; note it in the PR description.

### 8.2 `CHANGELOG.md`

Add a v0.2.0 entry:

```markdown
## [0.2.0] - <today's date>

### Added
- Post-hoc physical-consistency filtering (`weather.physics`):
  - `PhysicsConstraints` dataclass with tunable tolerances.
  - `check_constraints` for evaluating constraints on a batch.
  - `sample_batch_with_physics` for rejection sampling.
- Three constraints: wave-energy budget, fetch-limited Pierson-Moskowitz,
  peak-period range.

### Changed
- Public API in `weather.__init__` now re-exports the new physics module.
- `WeatherModel` is unchanged.

### Not changed
- Training pipeline is byte-exact identical to v0.1.0. Models trained on v0.1.0
  work with v0.2.0 without modification.
```

---

## 9. Acceptance criteria

The implementation is complete when all of the following are true:

1. `src/weather/physics.py` exists and matches the spec in §4.
2. `src/weather/__init__.py` re-exports `PhysicsConstraints`, `check_constraints`, `sample_batch_with_physics`.
3. All ten tests in `tests/weather/test_physics.py` pass.
4. The full existing test suite continues to pass (`poetry run pytest`).
5. `black --check src/ tests/` is clean.
6. `isort --check-only src/ tests/` is clean.
7. `mypy --strict src/` is clean.
8. `docs/model_design.md` version bumped to v3, "What changed" section rewritten.
9. `CHANGELOG.md` has the v0.2.0 entry.
10. The import-boundary test passes (physics doesn't leak into train/ or core/).

No new files beyond `physics.py` and `test_physics.py`. No changes to training code, model class, or existing simulate/validate functions.

---

## 10. Diagnostic outputs to verify manually

After implementation, run the following one-liner in the notebook to sanity-check the module works end-to-end:

```python
from weather import WeatherModel, PhysicsConstraints, sample_batch_with_physics
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

model = WeatherModel.load(Path("./models/v0.1.0"))
lats = np.array([40.0, 0.0, 55.0])       # Mediterranean, Pacific, Baltic
lons = np.array([15.0, -140.0 % 360.0, 20.0])
times = np.array(["2026-08-01T00:00", "2026-08-01T00:00", "2026-08-01T00:00"],
                 dtype="datetime64[s]")

values, accepted = sample_batch_with_physics(
    model, lats, lons, times,
    variables=["sigwh", "wsh", "pswh", "was", "pwp"],
    constraints=PhysicsConstraints(),
    seed=42,
)

print(f"Accepted: {accepted}")
print(f"Values shape: {values.shape}")     # (3, 5)
```

Expected: all three should be accepted within the default max_retries=10. If not, log the constraint diagnostics and check whether the fitted model produces physically-plausible marginals at those cells. This is not a bug in `physics.py` — it's a signal that the model needs tighter fitting for those regions.

---

## 11. Non-goals for v3

Explicitly out of scope for this ticket. Do not implement any of these, and if the design doc seems to require them, flag it and stop:

- **Physics-informed training.** The DMD/OU fits are not modified to satisfy physical constraints. That would be v4+.
- **Data-source cascade.** No routing of queries to archive vs. model based on date. `sample_batch_with_physics` always samples from the model.
- **Projection onto constraint manifolds.** If a sample violates a constraint, we resample. We do not try to project it onto the closest valid sample.
- **Cross-cell physical consistency.** Constraints are per-query-point. A storm moving across cells at implausible speed will pass all three constraints and be accepted.
- **Bathymetric fetch masks.** The hardcoded region list is v3-appropriate. Computing fetch from bathymetry is deferred.
- **Additional constraints beyond the three in §4.** If a physical relationship seems missing, note it for v4; do not add it to v3.
- **Change to `WeatherModel` API surface.** The rejection sampling machinery lives outside the model class.
- **Change to training pipeline or model artifacts.** Existing trained models must work with v3.

---

## 12. If something is ambiguous

The design doc `docs/model_design.md` §3.15 and §5.9 are the source of truth for intent. If this ticket and the design doc disagree, the design doc wins for design questions; for pure implementation choices (variable names, docstring style, log message format) match the surrounding codebase's conventions.

If the design doc is silent on something, apply the "make it match v2's style" heuristic — look at `src/train/local_residual.py` for a similar pure-function+dataclass example, and follow that pattern.

If a real ambiguity is found, stop and ask the human before making a load-bearing decision.
