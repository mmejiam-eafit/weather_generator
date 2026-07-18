# Stochastic Weather Generator — Design Handoff (v2)

**Package:** `weather-generator` (Python, three sibling packages under `src/`: `core`, `train`, `weather`).

**Audience:** coding agent bootstrapping the repo from scratch, or picking up work in progress. This document is self-contained — every convention, formula, and restriction the agent needs is stated here, not by reference.

**One-line summary:** train a stochastic model on the raw 3-hourly `.npy` archive of NOAA GFS-Wave / WAVEWATCH III + GFS wind fields (2022–2025), and provide plausible synthetic samples of any variable at any continuous `(lat, lon, datetime)` when real data isn't available. Validate on 2026.

**Approach chosen:** hierarchical decomposition — **linear-trend + harmonic climatology** → **coherent large-scale modes** via Tucker compression + DMD + OU on mode amplitudes → **per-cell local residuals** via independent OU fields. Per-cell Gaussian-copula marginals wrap the whole thing. Orchestrated by a scikit-learn `BaseEstimator + TransformerMixin` wrapper. See §3 for rationale and rejected alternatives.

**Language:** Python 3.14 only. Batched-first API design (§1, §8). If a downstream consumer eventually shows sub-millisecond point queries are needed and cannot be batched, a C++ query library can be added; do not build one preemptively.

**Training entry point:** a Jupyter notebook (`notebooks/test_pipeline.ipynb`), not a CLI. Training is a one-shot exploratory activity that benefits from cell-level iteration on Tucker/DMD/OU hyperparameters against a cached preprocessing output.

**Training data:** 2022-01-01 to 2025-12-31 (4 years, ~11,700 snapshots at 3h cadence).
**Validation (numerical metrics):** 2026-01-01T00:00Z to 2026-07-19T21:00Z (~200 days, ~1,600 snapshots). Held-out real data used to compute the validation battery in §6.
**Extrapolation window:** 2026-07-20T00:00Z onward — the archive currently ends here. The model must produce plausible synthetic weather for these dates without real data available for comparison. Assessed by consistency and stability checks rather than pointwise metrics; see §6.13.

---

## What changed vs. v1

Read this section if you're already familiar with v1. Otherwise skip to §1.

1. **Adopt TensorLy for Tucker.** `train/tucker.py` is now a thin wrapper around `tensorly.decomposition.partial_tucker` rather than a hand-rolled HOSVD. See §3.11 and §5.5.
2. **Adopt PyDMD for DMD.** `train/dmd.py` is now a thin wrapper around `pydmd.DMD` (with `pydmd.BOPDMD` as an alternate variant for uncertainty quantification and automatic complex-conjugate handling). See §3.11 and §5.6. The escalation paths — Hankel-DMD, BOPDMD — become one-line class swaps.
3. **Linear trend added to climatology.** `μ(s, t)` now includes a `c_lin(s) · (t − t_ref)` term to catch monotone drift (SST warming, storm track shifts). See §3.9 and §5.4.
4. **Hierarchical decomposition of the anomaly.** After Gaussianization, `Z(s, t)` is now decomposed into a *coherent* term (Tucker + DMD + OU on modes) plus a *local residual* term (independent per-cell OU fields). The v1 design captured only the coherent term, which produced samples that were too spatially smooth. See §3.10 and §5.7.
5. **New pure-function module.** `train/local_residual.py` fits and evolves the per-cell OU field. See §5.7.
6. **Notation cleanup.** OU parameters are now subscripted by scope: `θ_mode`, `σ_mode` for DMD modes; `θ_local`, `σ_local` for per-cell residual. `σ_std` is the standardization scale.
7. **Directional conventions are mixed in the archive, and the pipeline reflects that explicitly.** Previous versions assumed all directions were "coming from" and used a single `DirectionalDecomposer`. In reality `wad` and `ocd` are stored as `going_to`, while `wsd` and `wad_come_from` are confirmed `coming_from`; `pwd`/`pswd` are *assumed* `coming_from` pending domain-expert confirmation (see §2.3, §2.4 — the trade-wind/current climatology check used to confirm the other four fields isn't valid ground truth for wave partitions that can be swell-decoupled from local wind). v2 uses **two atomic transformers** — `ComingFromToUV` and `GoingToToUV` — with disjoint pair lists, so the pipeline definition itself shows which convention each variable uses. See §3.7 and §5.3. `wad_come_from` is the source of truth for wind; a runtime `wad`/`wad_come_from` consistency check guards against archive drift (§5.1). This entire convention table was established empirically in `notebooks/verify_direction_convention.ipynb`, a **single-snapshot check** — re-run it against more snapshots/seasons before treating any of it as final, especially the ocean-grid-dependent `ocd` result (§2.7 is still unverified).
8. **Training split.** Explicitly 2022–2025 for training; 2026-01-01 through 2026-07-19T21:00Z for held-out validation metrics; the anchoring approach (§6, §5.8) removes the need for a separate extrapolation window. The archive is truncated at 2026-07-20T00:00Z.

Everything else — grid conventions, direction encoding, NaN policy, path layout, Poetry/black/isort tooling — carries over unchanged.

---

## Table of contents

- §1 — Goal and API sketch
- §2 — Data facts (grids, dtypes, decoding, conventions, NaN policy, tooling)
- §3 — Design decisions and rationale
- §4 — Architecture overview
- §5 — Algorithm, step by step
- §6 — Validation battery
- §7 — Persisted model artifacts
- §8 — Repo bootstrap (layout, packaging, API surface, notebook workflow)
- §9 — Gotchas and implementation notes
- §10 — Variable table
- §11 — References
- §12 — What "done" looks like

---

## 1. Goal

Provide a Python package with a scikit-learn-compatible model class and a batched-first sampling API. The convenience single-point call exists for exploration; the batched call is what consumers should reach for at scale.

```python
from weather import WeatherModel, sample, sample_batch, simulate_field
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# --- Load a trained model ------------------------------------------------
model = WeatherModel.load(Path("./models/v1"))

# --- Single-point convenience (wraps sample_batch internally) ------------
value: float = sample(
    model,
    lat=35.5, lon=-40.2,
    t=datetime(2026, 6, 15, 9, tzinfo=timezone.utc),
    variable="sigwh",
    seed=42,
)

# --- Batched — the fast path ---------------------------------------------
values: np.ndarray = sample_batch(
    model,
    lats=np.array([...]),                       # (N,)
    lons=np.array([...]),                       # (N,)
    times=np.array([...], dtype="datetime64[s]"),
    variables=["sigwh", "was", "wsp"],
    seed=42,
)                                               # (N, 3)

# --- Full-field simulation on the native wave grid -----------------------
import xarray as xr
ds: xr.Dataset = simulate_field(
    model,
    t_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
    t_end=datetime(2026, 7, 1, tzinfo=timezone.utc),
    variables=["sigwh", "was"],
    seed=42,
)
```

The **sampling operations are free functions in `weather.simulate`**, not methods on `WeatherModel`. Pure functions over objects when the object doesn't need to mutate; `WeatherModel` is a passive container of fitted state.

`WeatherModel` itself is a scikit-learn transformer — its sklearn-visible surface is `fit(X)`, `transform(X)`, `fit_transform(X)` — plus custom `save`/`load` and a `from_components` alternate constructor. See §8.5 for the full signature.

**Correctness criteria:**

1. **Marginals match.** Per-cell empirical distributions of samples reproduce the training data's per-cell empirical distributions. Bounded variables (Hs ≥ 0) stay bounded; skewed ones stay skewed.
2. **Space-time covariance is preserved at *both* long and short lags.** Samples at nearby `(lat, lon, t)` are correlated the way real weather is at all resolvable scales — storms look like storms *and* individual cells have realistic local variability. This is a stronger criterion than v1 stated; the local residual layer (§3.10) exists to satisfy it.
3. **Propagation is preserved.** Swell trains and stormfronts advect at physically sensible speeds and don't dissolve prematurely.
4. **Cross-variable coupling is preserved.** Wind speed and wave height covary correctly; wind direction and wave direction covary correctly.
5. **Batched throughput is fast enough for realistic consumers.** Target ~10 µs per point when batched to 10⁴+ points. Single-point unbatched calls are ~100 µs–1 ms.
6. **Held-out year 2026 statistics match training statistics** within the ranges specified in §6. This is the acceptance test — if it fails, the model isn't ready to ship, no matter how good the training-window fit looks.

**Non-goals.** Not trying to forecast better than GFS. Not modeling climate change explicitly (though the trend term in §5.4 catches short-horizon drift). Not matching individual historical storms — matching the *statistics* of storms.

**Why 3-hourly, not daily.** Some point-query consumers (maritime routing in particular) reason about weather that changes on ~6–12 h timescales — storm passages, diurnal wind ramps near coasts. A daily-averaged model can't produce that resolution.

---

## 2. Data facts

Everything in this section is load-bearing. Any deviation propagates silently through the model and shows up as bad samples that pass unit tests but fail validation.

### 2.1 File layout

Raw 3-hourly archive is mirrored locally at:

```
./data/<YYYY>/<MM>/<DD>/<HH>/<variable>.npy
```

with `HH ∈ {00, 03, 06, 09, 12, 15, 18, 21}` (zero-padded, UTC). All paths resolve relative to a configurable `data_dir` argument (defaulting to `./data`) held in `core.config.Settings`. Never hardcode absolute paths.

Path construction in Python (lives in `core.io`):

```python
from pathlib import Path
from datetime import datetime, timezone

def snapshot_path(data_dir: Path, t: datetime, variable: str) -> Path:
    if t.tzinfo is None or t.tzinfo.utcoffset(t) != timezone.utc.utcoffset(t):
        raise ValueError(f"datetime must be UTC-aware, got {t!r}")
    return data_dir / f"{t:%Y/%m/%d/%H}" / f"{variable}.npy"
```

The `f"{t:%Y/%m/%d/%H}"` idiom produces zero-padded output on every locale — do not refactor to manual string concatenation without keeping the padding.

### 2.2 Three grids

The archive is not one uniform grid. Variables live on three:

| Grid       | Shape (rows × cols) | Resolution | Lat range      | Lon range        | Row 0 | Variables |
|------------|---------------------|------------|----------------|------------------|-------|-----------|
| **Wave**   | 621 × 1440          | 0.25°      | [−75°, +80°]   | [0°, 359.75° E]  | south | `sigwh`, `wsh`, `wsp`, `wsd`, `pwh`, `pwp`, `pwd`, `pswh`, `pswp`, `pswd` |
| **Wind**   | 721 × 1440          | 0.25°      | [−90°, +90°]   | [0°, 359.75° E]  | south | `was`, `wad`, `wad_come_from` |
| **Ocean**  | 1121 × 2880         | 0.125°     | (verify §2.7)  | [0°, 359.875° E] | (verify §2.7) | `sst`, `ocs`, `ocd` |

Row indexing (**both wave and wind grids — this is the non-standard part**):

```
wave grid:  lat = -75.0 + row * 0.25    row ∈ [0, 620]
wind grid:  lat = -90.0 + row * 0.25    row ∈ [0, 720]
```

Longitude for both: `lon = col * 0.25`, `col ∈ [0, 1439]`, spanning `[0°, 359.75° E]`.

**Row 0 = south is critical.** Most gridded earth-science datasets follow the opposite convention. See §2.7 for how this was established empirically and how to re-verify it for the ocean grid.

### 2.3 On-disk dtypes and decoding

Two dtype conventions in the archive:

- **Continuous fields** (`sigwh`, `wsh`, `wsp`, `pwh`, `pwp`, `pswh`, `pswp`, `was`, `sst`, `ocs`): stored as `float16` on disk.
- **Direction fields** (`wsd`, `pwd`, `pswd`, `wad`, `wad_come_from`, `ocd`): stored as `int8` 16-point-compass codes (22.5° steps), not raw degrees.

**Immediately on load, upcast everything to `float64`.** `float16` overflows in arithmetic at surprisingly small values (`(300.0)²` overflows).

**Compass code decoding.** Each field has two orthogonal pieces of metadata that must both be respected: the **origin code** (which int8 value represents 0°) and the **native convention** (whether the decoded angle means "the wind/wave is coming *from* this direction" or "going *to* this direction"). Both differ per variable. The full table below — and every convention determination in it — comes from an empirical check against known wind/current climatology (trade winds, Southern Ocean westerlies, Antarctic Circumpolar Current, equatorial currents) in `notebooks/verify_direction_convention.ipynb`. That check used **one snapshot**; re-run it against more snapshots/seasons before treating any row as final, particularly `ocd` (see below):

| Field            | Code range  | Origin code (North = 0°) | Decode formula            | Native convention                                |
|------------------|-------------|--------------------------|---------------------------|--------------------------------------------------|
| `wsd`            | 0–15        | `0`                      | `code * 22.5`             | `coming_from`                                    |
| `pwd`            | 0–15        | `0`                      | `code * 22.5`             | `coming_from` (assumed, pending expert)          |
| `pswd`           | 0–15 assumed| `0`                      | `code * 22.5`             | `coming_from` (assumed, pending expert)          |
| `wad`            | 1–16        | `1`                      | `(code - 1) * 22.5`       | `going_to`                                       |
| `wad_come_from`  | 1–16        | `1`                      | `(code - 1) * 22.5`       | `coming_from`                                    |
| `ocd`            | 1–16        | `1` (same as `wad`)      | `(code - 1) * 22.5`       | `going_to` (pending §2.7 ocean-grid verification) |

**The archive does not use a single uniform convention.** Wave directions are coming-from; ocean current direction is going-to; wind is stored twice (once as going-to in `wad`, once as coming-from in `wad_come_from`). Both origin code and convention must be verified per variable — do not generalize from one field to another. Note that `wad` and `ocd` need *no* origin-code correction relative to each other — `ocd` decodes correctly with the exact same `origin_code=1` already assigned to `wad`; the only difference between the two fields is the convention label, not the decode.

**`ocd`'s `going_to` finding is conditional on the ocean grid's orientation, which is still unverified (§2.7).** The check that established it used a lightweight, self-derived equator/hemisphere determination (via `sst`'s zonal-mean peak and open-ocean fraction at 40-60°), not the full Natural-Earth land-correlation study §2.7 calls for. Treat `ocd`'s convention as provisional until `tests/core/test_grid_ocean.py::test_orientation` (§12) passes — the same caveat `pwd`/`pswd` carry for a different reason applies here too. There is also no `ocd_come_from` sibling field on disk (unlike `wad`), so the fix for `ocd` is necessarily a convention flag in `GoingToToUV`, not a field swap.

Canonical decoder (lives in `core.io`):

```python
def decode_compass_code(codes: np.ndarray, origin_code: int) -> np.ndarray:
    """Decode int8 16-point compass codes to degrees (float64, [0, 360)).

    Returns raw decoded angle in the archive's native convention for that field —
    either "coming from" or "going to". The convention is a separate concern from
    decoding and is handled downstream by the ComingFromToUV or GoingToToUV
    transformers (see §5.3).
    """
    return ((codes.astype(np.int32) - origin_code) * 22.5).astype(np.float64) % 360.0
```

The decoder is convention-agnostic on purpose. Convention handling lives in the preprocessing Pipeline (§3.7, §5.3) so that adding a directional variable requires explicitly choosing which transformer it belongs to — no silent defaulting.

Test cases (must pass):
- `decode_compass_code(np.array([[0]], dtype=np.int8), origin_code=0)` → `[[0.0]]` (North).
- `decode_compass_code(np.array([[4]], dtype=np.int8), origin_code=0)` → `[[90.0]]` (East).
- `decode_compass_code(np.array([[1]], dtype=np.int8), origin_code=1)` → `[[0.0]]` (North, `wad` convention).
- `decode_compass_code(np.array([[5]], dtype=np.int8), origin_code=1)` → `[[90.0]]` (East, `wad` convention).

### 2.4 Direction conventions — mixed across the archive

**All decoded angles are CW from geographic North.** The interpretation of the angle — whether it means "coming from" or "going to" — differs by variable per the table in §2.3.

- **Coming-from** (0° = wind/wave arriving from the North, i.e. moving south):
  - `wsd`, `pwd`, `pswd`, `wad_come_from`
- **Going-to** (0° = wind/wave moving toward the North):
  - `wad`, `ocd`

**The pipeline preserves this per-variable convention explicitly.** Rather than silently normalizing everything to one convention at the decoder, two distinct transformer classes (`ComingFromToUV` and `GoingToToUV`, §5.3) each handle one convention. Which transformer a variable belongs to is stated in the Pipeline construction (§5.3, §8.6), making the convention visible in the pipeline definition itself. Adding a new directional variable requires an explicit choice of transformer — no default.

**`wad_come_from` is the source of truth for wind direction.** `wad` and `wad_come_from` encode the same physical information; using `wad_come_from` avoids one flip in the pipeline. `wad` is not read by any transformer, but at first-snapshot load time a consistency check (§5.1) reads both and verifies `wad − wad_come_from ≡ 180° (mod 360°)` per cell, guarding against upstream archive drift. If the check fails, training aborts.

**`(u, v)` decomposition — the two equations.** Both target the same `(u, v)` semantics: `u` eastward positive, `v` northward positive. They differ only in that "going-to" is the direction the wind vector points, while "coming-from" is 180° away from it.

For **coming-from** input:
```
theta_going_to = (theta_coming_from + 180.0) mod 360.0
theta_rad      = theta_going_to * pi / 180.0
u              = speed * sin(theta_rad)      # eastward
v              = speed * cos(theta_rad)      # northward
```

For **going-to** input:
```
theta_rad      = theta_going_to * pi / 180.0
u              = speed * sin(theta_rad)      # eastward
v              = speed * cos(theta_rad)      # northward
```

**Cardinal-direction verification test cases.** Each transformer has its own four-case table; both must pass in isolation. The expected `(u, v)` is *always* the vector the wind is blowing along, regardless of which convention the input used.

Coming-from → `(u, v)`:

| θ_from | expected `(u, v)`   | Meaning                     |
|--------|---------------------|-----------------------------|
| 0°     | `( 0, -speed )`     | Wind from N → blowing south |
| 90°    | `(-speed,  0 )`     | Wind from E → blowing west  |
| 180°   | `( 0, +speed )`     | Wind from S → blowing north |
| 270°   | `(+speed,  0 )`     | Wind from W → blowing east  |

Going-to → `(u, v)`:

| θ_to   | expected `(u, v)`   | Meaning                      |
|--------|---------------------|------------------------------|
| 0°     | `( 0, +speed )`     | Wind moving N (going north)  |
| 90°    | `(+speed,  0 )`     | Wind moving E (going east)   |
| 180°   | `( 0, -speed )`     | Wind moving S (going south)  |
| 270°   | `(-speed,  0 )`     | Wind moving W (going west)   |

Both tables are load-bearing — each transformer must pass its own set independently. The two sets flip the sign because the input angle represents opposite physical directions.

**Inverse (reconstruction at sample time).** Each transformer inverts using its own convention:

```
speed         = sqrt(u² + v²)
theta_going_to = atan2(u, v) mod 360.0     # arg order (sin, cos), NOT (y, x)

# ComingFromToUV.inverse_transform emits:
theta_coming_from = (theta_going_to + 180) mod 360.0

# GoingToToUV.inverse_transform emits:
theta_going_to  (as computed above)
```

The `atan2(u, v)` arg order (sin first, cos second) is the meteorological convention and is deliberate for both transformers.

Downstream consumers read each variable in its native archive convention. If a consumer needs `wad_come_from` in going-to form, it applies `(θ + 180) mod 360` at its own boundary.

### 2.5 NaN policy

- **Wave grid** — `float` fields carry NaN for land and persistent ice cover. Authoritative land mask for the wave grid.
- **Wind grid** — has *no* NaN. Winds are defined over land in the GFS product. Mask via a Natural Earth land polygon rasterized to the wind grid.
- **Ocean grid** — `sst` almost certainly carries NaN for land; `ocs`/`ocd` behavior needs verification.
- **Direction fields** — `int8`, no NaN. Masking comes from the paired float field.

**Every accumulation guarded with `np.isnan`.** NaN in → NaN through. Never coerce NaN to zero.

For the stochastic model: fit only on ocean cells. Build one boolean `ocean_mask` per grid (see §5.1), model over the ravelled ocean-cell vector, project back to `(H, W)` for output. Ocean cells must be non-NaN in *all* modeled variables at *all* training snapshots.

### 2.6 Longitude wraparound

Both grids span `[0°, 359.75° E]` — no dateline duplication. `col=1439` and `col=0` are adjacent on the sphere but at opposite ends of the array. Pad periodically before any interpolation:

```python
def pad_lon(field: np.ndarray, pad: int = 2) -> np.ndarray:
    return np.concatenate([field[..., -pad:], field, field[..., :pad]], axis=-1)
```

### 2.7 How grid orientation was established (and how to verify it)

The row-0-is-south convention was originally documented as the opposite and turned out to be wrong. The correction came from an empirical study:

- **Wave grid.** Correlate the real NaN pattern in `sigwh` against a rasterized coastline from Natural Earth, across multiple days and every plausible row offset and row order. The correct convention appears as a sharp single peak in agreement (~92.6% match). Verified visually.
- **Wind grid.** Has no NaN mask. Confirmed by visual landmark displacement matching the wave grid, and by winter storm wind maxima landing at plausible North Pacific / North Atlantic cyclone latitudes.

**The ocean grid orientation has not been verified. Do it before including SST or currents.** Method: correlate `sst` NaN pattern against a Natural Earth land polygon rasterized at 0.125°, across (row 0 south vs. north) × (rows increase vs. decrease). Correct convention should show ≥ 85% agreement; alternatives ≤ 60%. Independently confirm with warm-pool location.

Encode result in `core/config.py` as `OCEAN_GRID_LAT_AT_ROW0` and `OCEAN_GRID_LAT_STEP_DEG`.

**TASK — do this before writing any ocean-grid-dependent logic.** Add `notebooks/verify_ocean_grid_orientation.ipynb` implementing the method above (Natural Earth correlation, all four `(row 0 south/north) × (increasing/decreasing)` candidates, ≥30 arbitrary days per the original methodology, warm-pool cross-check) and confirm it passes the ≥85%/≤60% thresholds before implementing `Reprojector`'s ocean-grid path, `ClimatologyRemover`/`GaussianCopula` for `sst`, or `GoingToToUV`'s `ocd` pair. This supersedes the lightweight, single-snapshot, self-derived orientation check used in `notebooks/verify_direction_convention.ipynb` to establish `ocd`'s `going_to` convention (§2.3, §2.4) — that check is good evidence the convention finding is directionally right (see the reasoning there: both physical tests used are robust even to a hemisphere-sign error), but it is not a substitute for this notebook, and `ocd`'s convention should be treated as provisional until this notebook exists and passes. Once it does, encode the result in `core/config.py` and close out `tests/core/test_grid_ocean.py::test_orientation` (§12).

### 2.8 Python conventions for this repo

**Runtime dependencies:** `numpy ≥ 2.0`, `scipy ≥ 1.14`, `scikit-learn ≥ 1.5`, `xarray ≥ 2024.10`, `pandas ≥ 2.2`, **`tensorly ≥ 0.9`** (new in v2), **`pydmd ≥ 2024`** (new in v2), `pydantic-settings ≥ 2.0`.

**Dev dependencies:** `pytest ≥ 8`, `pytest-mock`, `pytest-cov`, `pytest-xdist`, `matplotlib ≥ 3.9`, `black`, `isort`, `mypy ≥ 1.13`, `ipykernel`, `jupyter`.

**Verify Python 3.14 wheels for `tensorly` and `pydmd` at project start.** Both are pure-Python or nearly so and should work, but 3.14 is new enough that not every scientific package has published 3.14-classified wheels. If either package fails to install on 3.14, options are (a) pin to Python 3.13, (b) build from source, (c) wait for maintainers. Do the `poetry install` check before writing any code that depends on either package.

**Packaging:** Poetry. `pyproject.toml` uses `[tool.poetry]` sections; `poetry.lock` committed. Do not switch to hatchling/PDM/uv/setup.py.

**Formatting and linting:** `black` (line 100, target `py314`), `isort` (profile `black`), `mypy --strict`. Do not use `ruff`.

**Coding conventions:**

- Type hints on every function signature. `mypy --strict` in CI.
- **PEP 649 (deferred annotations by default) is active in Python 3.14.** `from __future__ import annotations` is redundant; omit it.
- Docstrings on every public function and class. Numpy-style. Cite handoff sections: `"""... (see model_design.md §5.4)."""`.
- Naming (PEP 8): `snake_case`, `PascalCase` classes, `SCREAMING_SNAKE_CASE` module constants, `_` prefix for private.
- No mutable default arguments.
- Explicit `dtype=` at every array construction.
- No `numpy.float16` past the I/O boundary.
- `@dataclass(frozen=True, slots=True)` for structured data.
- `pathlib.Path` everywhere.
- All timestamps `datetime` with `tzinfo=timezone.utc`. Naive datetimes raise `ValueError`.
- RNG discipline: `rng: np.random.Generator | None` or `seed: int | None`. sklearn-compatible components take `random_state` in `__init__`, materialize `_rng` in `fit` via `sklearn.utils.check_random_state`. Never touch global `np.random`.
- Parallelism via `joblib.Parallel(n_jobs=-1, backend="loky")`. joblib is already a sklearn dep.

**Prefer functions over classes.** Standalone functions default. Classes only when (a) substantial shared state across method calls (`WeatherModel`) or (b) external contract (sklearn transformers).

**Testing:** `pytest` + `pytest-mock`. One test file per source module. Coverage ≥ 85% on `src/`.

**CI:** GitHub Actions, Python 3.14, ubuntu-latest. Steps: `black --check`, `isort --check-only`, `mypy --strict src/`, `pytest --cov`.

---

## 3. Design decisions and rationale

### 3.1 Train on raw 3h data, not daily averages

Point queries at arbitrary `datetime` need sub-daily variability. Use the raw archive.

### 3.2 Rejected: full 3D Gaussian process / kriging

Would require designing a `(lat, lon, time, variable)` covariance kernel; inference scales poorly; doesn't propagate advective phenomena. Reconsider only via SPDE-GMRF (Lindgren–Rue–Lindström 2011) if hierarchical DMD-OU shows unfixable calibration problems.

### 3.3 Rejected: plain real EOF + AR

Real-valued EOFs represent traveling waves as pairs of standing modes in quadrature. AR on the resulting PCs works but has no principled way to distinguish coherent propagation from noise. DMD subsumes this cleanly.

### 3.4 Chosen: Tucker → DMD → OU on modes

For the *coherent* layer of the hierarchical decomposition (§3.10):

- **Tucker (HOSVD)** compresses spatial and variable modes.
- **DMD** on the compressed core extracts the linear deterministic skeleton. Each mode carries its own complex eigenvalue encoding growth rate and frequency.
- **OU on residual mode amplitudes** models stochastic wobble around the deterministic skeleton. Closed-form fit and simulation.

### 3.5 Non-Gaussian marginals via Gaussian copula

DMD and OU are linear-Gaussian; weather variables are skewed and bounded. Interpose an empirical-CDF Gaussian copula per cell:

- **Forward** (training): rank per cell → uniform → `Φ⁻¹` → standard normal.
- **Inverse** (sampling): standard normal → `Φ` → uniform → empirical CDF lookup → original units.

Handles positivity/boundedness implicitly. Known limitation: zero joint tail dependence. Fix later per §3.13 escalation.

### 3.6 Cross-grid handling: reproject to wave grid

Three grids of different sizes need reconciliation. Reproject wind and ocean grids onto the wave grid (bilinear, clipping to `lat ∈ [−75°, +80°]`) as v1. Simpler than fitting three separate models with cross-grid coupling at OU level (§3.13 escalation).

### 3.7 Directional variables via `(u, v)` — two transformers, convention in the pipeline

Never model `θ` directly. Decompose to `(u, v)`, model each as a real field, recombine at sample time via `speed = hypot(u, v)`, `θ = atan2(u, v) mod 360°`.

The archive stores directions in **mixed conventions** (§2.4): most fields are coming-from but `wad` and `ocd` are going-to. Rather than smuggle a per-variable convention flag through a single decoder, the pipeline uses **two atomic transformer classes** that differ only in how they interpret their input angles:

- `ComingFromToUV` — for `(speed, θ_from)` pairs (`wsd`, `pwd`, `pswd`, `wad_come_from`).
- `GoingToToUV` — for `(speed, θ_to)` pairs (`ocd`; `wad` is unused).

Both run before `Reprojector` (§5.2). They act on disjoint variable sets and can be composed in either order. Each has its own `inverse_transform` that reconstructs the variable in its original archive convention.

**Why two transformers instead of one decoder that normalizes to a canonical internal convention:**

- The pipeline definition itself shows which variables use which convention. Reading the `Pipeline([...])` line tells you `wad_come_from` is coming-from and `ocd` is going-to. No config dictionary to look up, no decode-time flip that might be forgotten.
- Adding a new directional variable requires an explicit choice of transformer. There is no default, so an agent cannot silently omit the convention decision.
- Sign errors in one transformer cannot hide behind the other; each has its own four-case cardinal-direction test.
- `inverse_transform` naturally reconstructs each variable in its native archive convention, which is what downstream consumers of the archive already expect.

`wad` is not used by any transformer — `wad_come_from` is the source of truth for wind direction. A runtime consistency check on the first loaded snapshot (§5.1) reads both and asserts they differ by 180° mod 360°, guarding against upstream archive convention drift.

### 3.8 Which variables to model — tiered

**Tier 1 (v0.1.0):** `sigwh`, `wsh`, `wsp`, `was`; `(u_wind, v_wind)`, `(u_wsea, v_wsea)`. Six modeled fields after (u, v) expansion.

**Tier 2 (v0.2.0):** peak wave `(pwh, pwp, pwd)` — adds `(u_peak, v_peak)`; `sst`. Requires `notebooks/verify_ocean_grid_orientation.ipynb` (§2.7) to exist and pass before `sst` is wired into the pipeline.

**Tier 3 (v0.3.0+):** primary swell; ocean currents. `ocd`/`ocs` likewise require `notebooks/verify_ocean_grid_orientation.ipynb` (§2.7) to have passed — do not enable `GoingToToUV`'s `ocd` pair before it does.

`wad` ignored throughout — `wad_come_from` is the source of truth for wind direction (§2.4, §3.7).

### 3.9 Climatology is quasi-stationary — linear trend term

**Change from v1.** v1 assumed a purely harmonic climatology, treating the mean as stationary across the training window. Over 4 years (2022–2025), this is not quite right:

- Regional SST warming is real (Mediterranean, Northeast Pacific, Arctic all show ~0.02–0.05 °C/year trends).
- North Atlantic storm track has migrated poleward over the last few decades.
- 2023–2024 were strong El Niño years; the anomaly statistics of a 2022–2025 window are not stationary in a strict sense.

v2 adds a linear trend term to the per-cell climatology:

```
μ(s, t) = c₀(s) + c_lin(s) · Δτ(t)
       + c₁(s)·cos(ω_y·t) + c₂(s)·sin(ω_y·t)
       + c₃(s)·cos(2·ω_y·t) + c₄(s)·sin(2·ω_y·t)
       [+ c₅(s)·cos(ω_d·t) + c₆(s)·sin(ω_d·t)]     # diurnal for SST + wind only
```

where `Δτ(t) = (t − t_ref) / (1 year)` and `t_ref` is the **midpoint** of the training window (mid-2023 for a 2022–2025 fit). Using the midpoint makes `c_lin` uncorrelated with `c₀`, so both stay interpretable.

**Diagnostic:** the fit emits a per-cell `|c_lin| / σ_std` ratio. Cells where this exceeds ~0.1 have meaningful trend; if fewer than ~5% of cells exceed it, the trend term is essentially fitting noise and `include_trend=False` is cleaner. Report the fraction as a training log line.

**Escalation** to interannual index regressors (ENSO, NAO) or rolling climatology deferred to §3.13.

### 3.10 Hierarchical decomposition — the coherent/local split

**Change from v1.** v1 modeled the Gaussianized anomaly `Z(s, t)` with Tucker + DMD + OU on ~100 mode amplitudes. At sample time, the value at any `(s, t)` was a deterministic function of those 100 amplitudes. Consequence: samples were too spatially smooth — realistic at basin scale, unrealistically clean at cell scale. Real weather has fine-grained texture the coherent modes can't represent.

v2 splits `Z(s, t)` into two additive layers:

```
Z(s, t) = Z_coherent(s, t) + Z_local(s, t)
```

- **`Z_coherent(s, t)`** — captures basin-scale coherent structure, including propagation, cross-basin correlations, and the storm-passage temporal spectrum. Modeled with Tucker + DMD + OU on ~100 mode amplitudes as before. All the physics of "a storm crossing the basin" lives here.
- **`Z_local(s, t)`** — captures per-cell temporal variability the coherent modes miss. Modeled as a *field* of independent OU processes:
  ```
  dZ_local(s, t) = -θ_local(s) · Z_local(s, t) dt + σ_local(s) · dW(s, t)
  ```
  where `θ_local, σ_local` are 2D spatial fields (one value per ocean cell per variable) and `dW(s, t)` are independent Brownian increments per cell. Analogous to the mode-level OU but "OU everywhere in space" instead of "OU per mode."

At fit time, `Z_local` is estimated as the residual `Z − Z_coherent`, and per-cell OU parameters are fit by vectorized AR(1) across all cells simultaneously. At sample time, the local field contributes an independent per-cell noise draw to each query point.

**Cost:** ~`2 × N_ocean × V × 4 bytes` = ~13 MB for tier 1. Sampling adds one vectorized OU draw per query — sub-ms at 10⁴ points.

**Cross-cell independence is a simplifying assumption.** Real residuals likely have short-range spatial correlation (sub-basin eddies span multiple cells). For v0.1.0 the independence assumption is defensible; if the short-lag variogram check in §6 shows this matters, escalate to a Gaussian-Markov-Random-Field local layer (SPDE-GMRF, per §3.13).

**Diagnostic:** at fit time, compute per-cell `Var(Z_local) / Var(Z)`. Regions with high ratios (coastal, ice-edge, western boundary currents) are exactly where fine-scale variability matters. Regions with low ratios (open-ocean interior) barely need the local layer at all. Cells where the ratio is below `local_residual_variance_floor` (default 0.02) get their local OU parameters zeroed to save simulation work.

### 3.11 Library adoption — TensorLy for Tucker, PyDMD for DMD

**Change from v1.** v1 hand-rolled both Tucker/HOSVD and exact DMD. v2 replaces these with thin wrappers around `tensorly.decomposition.partial_tucker` and `pydmd.DMD`. Rationale:

- Both libraries are maintained by the primary research groups in their respective areas (TensorLy's Kolda-Bader lineage; PyDMD's Brunton-Kutz group).
- Both are well-tested against edge cases we would otherwise have to discover ourselves.
- **PyDMD unlocks the escalation paths for free.** Hankel-DMD is a one-line class swap (`pydmd.HankelDMD`); Bagging Optimized DMD with automatic uncertainty quantification is `pydmd.BOPDMD` with `eig_constraints={"conjugate_pairs"}` — which also removes the manual complex-conjugate-pair handling that was gotcha #8 in v1.
- **TensorLy's `partial_tucker`** natively handles the "compress modes 1 and 2, keep mode 0 (time) full-rank" pattern we need. Its HOOI initialization is slightly better than pure HOSVD for the same rank.

The wrappers still add value:

- We return frozen `TuckerFit`/`DMDFit` dataclasses containing only numpy arrays, not tensorly/pydmd class instances. Artifacts serialize cleanly to `.npz` and don't break when tensorly/pydmd versions drift.
- We handle ocean-cell masking, dtype coercion, and NaN guards around the calls.
- We keep the algorithmic knobs surfaceable at the `WeatherModel` level (rank choices, DMD variant, etc.).

See §5.5 and §5.6 for implementation sketches.

### 3.12 sklearn hybrid: transformers where they fit, pure functions where they don't

Unchanged from v1. Preprocessing chain (directional → reproject → climatology → standardize → Gaussianize) are sklearn `BaseEstimator + TransformerMixin` classes; decomposition + stochastic models (Tucker, DMD, mode-OU, local-OU) are pure functions returning frozen dataclasses. `WeatherModel` glues both kinds together with the sklearn transformer surface at the top level.

### 3.13 Documented escalation paths

If validation (§6) shows the base model is insufficient:

- **High-frequency variance under-represented in per-cell spectra** → swap `pydmd.DMD` for `pydmd.HankelDMD`. One-line change in `train/dmd.py`.
- **Extremes and joint tail dependence too weak** → replace Gaussian copula with t-copula (single extra df parameter) or splice GPD tails per cell (Coles 2001).
- **Cross-variable coupling too weak** → raise `r_var` Tucker rank, or add explicit Linear Model of Coregionalization.
- **Short-lag spatial correlation in residuals missing** → replace independent per-cell OU with SPDE-GMRF local layer (Lindgren–Rue–Lindström 2011). Real research-level change; do only if validation forces it.
- **Interannual climate variability biases the climatology** → add index regressors (ENSO NINO3.4, NAO) to the climatology model, or switch to rolling-window climatology re-fit annually.
- **DMD modes miss slow (multi-year) structure** → try `pydmd.BOPDMD` with `eig_constraints={"imag"}` to constrain modes to be purely oscillatory, forcing slow structure into the climatology where it belongs.
- **Uncertainty quantification on modes needed** → use `pydmd.BOPDMD` regardless. Bagged fit gives per-mode confidence intervals.
- **In-sample hurricane-season reproduction check (§6.A.18) looks weak** → escalate to a true leave-one-year-out retrain: fit on 3 of the 4 training years, hold out the fourth's Aug-Oct entirely, repeat the same storm-signature detection purely out-of-sample. 4× the fit cost (one retrain per held-out year), so only do this if the cheaper in-sample check raises a real concern — don't run it by default for v0.1.0.

### 3.14 Three-package split: `core`, `train`, `weather`

Unchanged from v1. `core/` — shared primitives; `train/` — training pipeline (transformers + pure functions); `weather/` — public API. `weather/` depends on `core` and `train`; `train/` depends on `core` only; `core/` has no internal deps. Consumers import only from `weather/`.

---

## 4. Architecture overview

Three deterministic layers plus two stochastic layers.

```
                          TRAINING (offline, notebook-driven)
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │   ./data/YYYY/MM/DD/HH/*.npy      (2022–2025 for training)          │
   │              │                                                     │
   │              ▼                                                     │
   │   [ingest]   load_training_window(data_dir, t_start, t_end)        │
   │              │                                                     │
   │              ▼                                                     │
   │   ┌────────── sklearn Pipeline (preprocessing) ──────────┐        │
   │   │  ComingFromToUV, GoingToToUV      (train/directional.py) │     │
   │   │  Reprojector               (train/reproject.py)       │        │
   │   │  ClimatologyRemover        (train/climatology.py)     │        │
   │   │      — includes linear trend term (§3.9)              │        │
   │   │  PerCellStandardizer       (train/standardize.py)     │        │
   │   │  GaussianCopula            (train/copula.py)          │        │
   │   └───────────┬───────────────────────────────────────────┘        │
   │               │  Gaussianized anomalies Z (T × N_ocean × V)         │
   │               ▼                                                    │
   │   ┌────────── Coherent decomposition (§5.5–5.6) ──────────┐       │
   │   │  tucker_fit   → TuckerFit                              │       │
   │   │      via tensorly.decomposition.partial_tucker         │       │
   │   │  dmd_fit      → DMDFit                                 │       │
   │   │      via pydmd.DMD (or BOPDMD variant)                 │       │
   │   │  fit_mode_ou  → ModeOUFit  (per-mode θ, σ, φ)          │       │
   │   └───────────┬────────────────────────────────────────────┘       │
   │               │  Z_coherent(s, t)                                  │
   │               ▼                                                    │
   │   ┌────────── Local residual (§5.7, new in v2) ───────────┐       │
   │   │  R(s, t) = Z(s, t) − Z_coherent(s, t)                  │       │
   │   │  fit_local_residual  → LocalResidualFit                │       │
   │   │      per-cell (θ_local, σ_local, φ_local) field        │       │
   │   └───────────┬────────────────────────────────────────────┘       │
   │               │                                                    │
   │               ▼                                                    │
   │   WeatherModel.from_components(pipeline, tucker, dmd,              │
   │                                mode_ou, local_residual, ...)      │
   │       .save(model_dir)                                             │
   └────────────────────────────────────────────────────────────────────┘

                         QUERY (online, batched preferred)
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │   weather.simulate.sample_batch(model, lats, lons, times, vars)    │
   │              │                                                     │
   │              ▼                                                     │
   │   Evolve mode OU + DMD to requested times (closed-form)            │
   │              │                                                     │
   │              ▼                                                     │
   │   Reconstruct compressed state → Tucker inverse → Z_coherent(s, t) │
   │              │                                                     │
   │              ▼                                                     │
   │   Draw Z_local(s, t) from per-cell OU (independent, closed-form)   │
   │              │                                                     │
   │              ▼                                                     │
   │   Z(s, t) = Z_coherent(s, t) + Z_local(s, t)                       │
   │              │                                                     │
   │              ▼                                                     │
   │   Bilinear interpolate at (lat, lon) [padded for lon wraparound]   │
   │              │                                                     │
   │              ▼                                                     │
   │   preprocess.inverse_transform  → invert copula, standardization,  │
   │                                    add climatology, recombine u/v  │
   │              │                                                     │
   │              ▼                                                     │
   │   values[]                                                         │
   └────────────────────────────────────────────────────────────────────┘
```

**Key architectural properties:**

- The preprocessing chain is one sklearn `Pipeline`. Its `inverse_transform` is the same object as its `transform` — no duplicated inverse logic.
- Sampling is composed of two independent stochastic contributions: coherent (via ~100 mode-OU) and local (via `N_ocean × V` cell-OU). Both use closed-form OU transitions — no discrete stepping needed for point queries.
- `WeatherModel.transform(X)` runs *only* the preprocessing Pipeline and returns Gaussianized anomalies at native grid resolution. Not `Z_coherent`, not `Z_local`. Consumers who want those get them from the fitted attributes directly.

---

## 5. Algorithm, step by step

Notation used throughout (naming disambiguated in v2):

- `T` — training snapshots (~11,700 for 2022–2025 at 3h cadence).
- `H, W = 621, 1440` — wave grid dimensions.
- `N_ocean` — ocean cells after masking; expect ~60% of `H·W` ≈ 5.4 × 10⁵.
- `V` — jointly modeled variables. Tier 1: ~6 after (u, v) expansion.
- `r_ocean`, `r_var`, `r_dmd` — truncation ranks.
- `Δt = 10800 s` (3 hours).
- `t_ref` — climatology reference epoch (midpoint of training window).
- `σ_std(s, v)` — per-cell standardization scale.
- `σ_climatology(s, v)` = `σ_std(s, v)` (alias for clarity).
- `θ_mode(k)`, `σ_mode(k)`, `φ_mode(k)` — OU parameters per DMD mode.
- `θ_local(s, v)`, `σ_local(s, v)`, `φ_local(s, v)` — OU parameters per ocean cell per variable.

### 5.1 Ingest

Lives in `core/io.py` and `train/ingest.py`.

```python
def snapshot_path(data_dir: Path, t: datetime, variable: str) -> Path: ...

def enumerate_snapshots(data_dir: Path, t_start: datetime, t_end: datetime):
    t = t_start
    while t < t_end:
        yield t, snapshot_path(data_dir, t)
        t += timedelta(hours=3)

def load_training_window(
    data_dir: Path,
    t_start: datetime,
    t_end: datetime,
    variables: list[str],
) -> xr.Dataset:
    """Load raw archive files into an xarray.Dataset over (time, lat, lon, variable)."""
    ...
```

Each `.npy` upcast to `float64` immediately. int8 direction fields decoded via `decode_compass_code` (§2.3).

**Runtime consistency check on first snapshot.** Before beginning the training window load, `load_training_window` reads `wad` and `wad_come_from` from a single snapshot (the earliest snapshot with both present) and verifies they differ by 180° mod 360° per cell:

```python
def _check_wad_wad_come_from_consistency(data_dir: Path, t: datetime) -> None:
    """Guard against upstream archive convention drift (§3.7, §5.1)."""
    wad = decode_compass_code(load_raw(data_dir, t, "wad"), origin_code=1)
    wad_from = decode_compass_code(load_raw(data_dir, t, "wad_come_from"), origin_code=1)
    diff = np.abs(((wad - wad_from - 180.0 + 180.0) % 360.0) - 180.0)   # min circular distance
    max_dev = float(np.nanmax(diff))
    if max_dev > 5.0:   # tolerance for 22.5° discretization + rounding
        raise ValueError(
            f"wad and wad_come_from do not differ by 180° at {t.isoformat()}. "
            f"Max deviation: {max_dev:.2f}°. Archive convention may have changed."
        )
```

Log the max deviation on success. Rerun on the last snapshot too — if the archive convention changed *within* the training window, the second check catches it.

**Implementation prerequisite: `core/io.py` currently refuses to load `wad_come_from` and `ocd` at all.** `_DIRECTION_ORIGIN_CODES` deliberately omits both (per the v1 design, back when `wad_come_from` was believed to be a redundant duplicate and `ocd`'s convention was unverified), so `load_snapshot` raises `ValueError` for either today. `load_raw` above is *not* `core.io.load_snapshot` — it's a lower-level raw-array read that must be added (or `load_snapshot`'s allowlist extended) before any of §3.7/§5.3's pipeline wiring can run. Add `"wad_come_from": 1` and `"ocd": 1` to `_DIRECTION_ORIGIN_CODES` as part of implementing this section, not as an afterthought.

**Gap handling:** missing file → log to stderr, mark NaN. Gap of 24 h+ → abort. Gap 6–24 h → warn.

**Memory:** 4 years × 6 vars at `float64` at global 0.25° is ~240 GB. Use `np.memmap` or chunk time-blocks (30 days at a time).

**Ocean mask construction:** for each grid, intersection of `not isnan` across all training snapshots and modeled variables on that grid. Wind grid needs external Natural Earth land polygon rasterized to `(721, 1440)`. Store as `.ocean_mask_` on `WeatherModel` — bool `(621, 1440)`.

### 5.2 Reprojection (`Reprojector` transformer)

Wind (721 × 1440) and ocean (1121 × 2880) fields resampled to the wave grid (621 × 1440, `lat ∈ [−75°, +80°]`).

```python
class Reprojector(BaseEstimator, TransformerMixin):
    """Reproject wind/ocean grids onto the wave grid (§3.6, §5.2)."""
    def __init__(self, target_grid: str = "wave") -> None:
        self.target_grid = target_grid
    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        return self
    def transform(self, X: xr.Dataset) -> xr.Dataset: ...
```

**Directional variables must be reprojected in `(u, v)` space, not as `θ`.** Handled by chaining `ComingFromToUV` and `GoingToToUV` (both, §5.3) *before* `Reprojector`.

### 5.3 Directional decomposition — two transformers

For each `(speed, direction)` pair, replace with `(u, v)`. Two transformer classes handle the two archive conventions; they run *before* `Reprojector` so reprojection sees linear-safe `(u, v)` fields.

**`ComingFromToUV`** — for pairs where the direction is stored as "coming from":

```python
class ComingFromToUV(BaseEstimator, TransformerMixin):
    """Decompose (speed, θ_from) pairs into (u, v) components.

    θ_from is meteorological "coming from", CW from N (§2.4).
    """
    def __init__(
        self,
        pairs: list[tuple[str, str, str, str]],
    ) -> None:
        # Each pair: (speed_var, direction_var, u_out_name, v_out_name).
        self.pairs = pairs

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        # For each pair: (θ_from + 180°) → θ_to → (u, v)
        ...

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        # For each pair: (u, v) → θ_to → (θ_to - 180°) mod 360° → θ_from
        ...
```

**`GoingToToUV`** — for pairs where the direction is stored as "going to":

```python
class GoingToToUV(BaseEstimator, TransformerMixin):
    """Decompose (speed, θ_to) pairs into (u, v) components.

    θ_to is going-to, CW from N (§2.4).
    """
    def __init__(
        self,
        pairs: list[tuple[str, str, str, str]],
    ) -> None:
        self.pairs = pairs

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        # For each pair: θ_to → (u, v) directly
        ...

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        # For each pair: (u, v) → atan2(u, v) mod 360° = θ_to
        ...
```

**Canonical pair assignment for v0.1.0 (tier 1) and beyond:**

`ComingFromToUV`:
- `(wsh, wsd, u_wsea, v_wsea)` — tier 1
- `(was, wad_come_from, u_wind, v_wind)` — tier 1. Source of truth for wind; `wad` is not used.
- `(pwh, pwd, u_peak, v_peak)` — tier 2 (pending expert verification of `pwd` convention).
- `(pswh, pswd, u_pswell, v_pswell)` — tier 3 (pending expert verification of `pswd` convention).

`GoingToToUV`:
- `(ocs, ocd, u_current, v_current)` — tier 3.

Non-directional variables (`sigwh`, `wsp`, `pwp`, `pswp`, `sst`) pass through both transformers unchanged.

**Both transformers preserve non-modified variables in the output Dataset** — they only add `(u, v)` columns and remove the paired `(speed, direction)` columns for pairs they own. Untouched variables (both non-directional ones and pairs owned by the other transformer) pass through as-is. This makes composition in either order commutative for disjoint pair sets.

**`inverse_transform` reconstructs each variable in its native archive convention.** `ComingFromToUV` produces `wad_come_from` back on inversion (not `wad`); `GoingToToUV` produces `wad`-style angles for `ocd`. Downstream consumers get outputs matching the input archive convention.

**Pipeline placement** (from §5, notebook Cell 3):

```python
preprocess = Pipeline([
    ("coming_from_to_uv", ComingFromToUV(pairs=[
        ("wsh", "wsd", "u_wsea", "v_wsea"),
        ("was", "wad_come_from", "u_wind", "v_wind"),
    ])),
    ("going_to_to_uv", GoingToToUV(pairs=[
        # empty for tier 1; populated when tier 3 (ocean currents) enabled
    ])),
    ("reproject",   Reprojector(target_grid="wave")),
    ("climatology", ClimatologyRemover(diurnal_variables=["sst"], include_trend=True)),
    ("standardize", PerCellStandardizer()),
    ("gaussianize", GaussianCopula(n_bins=500)),
])
```

Even when `GoingToToUV` has an empty pair list (tier 1), the transformer is present in the Pipeline as a scaffold. This makes it obvious to any reader that going-to variables *would* live there, and enabling tier 3 is a matter of adding entries — not adding a new pipeline step.

### 5.4 Climatology and Gaussianization

Three transformers, run consecutively in the Pipeline.

**`ClimatologyRemover`** fits per ocean cell per variable:

```
μ(s, t) = c₀(s) + c_lin(s) · Δτ(t)
       + c₁(s)·cos(ω_y·t) + c₂(s)·sin(ω_y·t)
       + c₃(s)·cos(2·ω_y·t) + c₄(s)·sin(2·ω_y·t)
       [+ c₅(s)·cos(ω_d·t) + c₆(s)·sin(ω_d·t)]     # diurnal
```

where:

- `Δτ(t) = (t − t_ref) / 1_year`, with `t_ref = (t_train_start + t_train_end) / 2`. Normalizing by 1 year gives `c_lin` interpretable units of "change per year." Centering at `t_ref` makes `c_lin` uncorrelated with `c₀`.
- `ω_y = 2π / 365.25 days`, `ω_d = 2π / 1 day`.
- Diurnal terms (`c₅, c₆`) included only for `sst`, `u_wind`, `v_wind`. Waves don't have a diurnal cycle at 0.25° global resolution.

Fit by vectorized OLS. Fitted attributes:

- `self.coeffs_` : `(n_terms, N_ocean, V)` float64
- `self.t_ref_` : datetime (UTC)
- `self.trend_significance_ratio_` : `(N_ocean, V)` float32 — per-cell `|c_lin| / σ_std`. Diagnostic.

`__init__` parameters:

```python
def __init__(
    self,
    diurnal_variables: list[str] | None = None,   # default: ["sst", "u_wind", "v_wind"]
    include_trend: bool = True,
    trend_significance_threshold: float = 0.1,
) -> None: ...
```

If `include_trend=False`, the design matrix drops the `Δτ` column. Log the fraction of cells where `trend_significance_ratio > trend_significance_threshold`; if it's under ~1%, consider dropping the trend term for a cleaner model.

**`PerCellStandardizer`** divides anomaly by per-cell temporal std:

```
A(s, t) = Ỹ(s, t) − μ(s, t)          # anomaly
A_std(s, t) = A(s, t) / σ_std(s)      # standardized anomaly
```

Fitted attribute: `self.sigma_std_` : `(N_ocean, V)` float32.

**`GaussianCopula`** learns per-cell empirical CDFs:

```
Z(s, t) = Φ⁻¹(F_s(A_std(s, t)))
```

where `F_s` is the per-cell empirical CDF, stored as a quantile-binned lookup table.

```python
class GaussianCopula(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins: int = 500) -> None: ...
    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        self.sorted_anomalies_binned_ = ...    # (n_bins, N_ocean, V) float32
        return self
    def transform(self, X: xr.Dataset) -> xr.Dataset: ...
    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset: ...
```

**LUT storage:** full sorted values are `T × N_ocean × V × 8 bytes`. For 4 years × 5.4 × 10⁵ ocean cells × 6 vars ≈ 350 GB — infeasible. Downsample to `n_bins=500` quantile bins per cell → ~6 GB, marginal-fidelity loss negligible.

### 5.5 Tucker (HOSVD) — TensorLy wrapper

Lives in `train/tucker.py`. Not a transformer.

```python
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot

tl.set_backend("numpy")   # explicit — TensorLy supports jax/torch backends


@dataclass(frozen=True, slots=True)
class TuckerFit:
    U_ocean: np.ndarray    # (N_ocean, r_ocean) float64
    U_var:   np.ndarray    # (V, r_var)         float64
    r_ocean: int
    r_var:   int
    reconstruction_error: float   # relative Frobenius error, diagnostic


def tucker_fit(
    Z: np.ndarray,                     # (T, N_ocean, V) Gaussianized anomalies
    r_ocean: int,
    r_var: int,
    rng: np.random.Generator,
    n_iter_max: int = 100,             # 0 → pure HOSVD, >0 → HOOI refinement
    tol: float = 1e-4,
) -> TuckerFit:
    """Tucker-2 compression via partial_tucker (§3.11, §5.5).

    Compresses modes 1 (ocean cells) and 2 (variables), keeps mode 0 (time) full-rank.
    """
    (_core, [U_ocean, U_var]), _errors = partial_tucker(
        Z,
        rank=[r_ocean, r_var],
        modes=[1, 2],
        init="svd",
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=rng,
    )
    # Reconstruction error diagnostic
    Z_recon = multi_mode_dot(
        multi_mode_dot(Z, [U_ocean.T, U_var.T], modes=[1, 2]),
        [U_ocean, U_var],
        modes=[1, 2],
    )
    err = float(np.linalg.norm(Z_recon - Z) / np.linalg.norm(Z))
    return TuckerFit(
        U_ocean=U_ocean.astype(np.float64),
        U_var=U_var.astype(np.float64),
        r_ocean=r_ocean,
        r_var=r_var,
        reconstruction_error=err,
    )


def tucker_transform(Z: np.ndarray, fit: TuckerFit) -> np.ndarray:
    """Compress Z → core, shape (T, r_ocean, r_var)."""
    return multi_mode_dot(Z, [fit.U_ocean.T, fit.U_var.T], modes=[1, 2])


def tucker_inverse(core: np.ndarray, fit: TuckerFit) -> np.ndarray:
    """Uncompress core → Z, shape (T, N_ocean, V)."""
    return multi_mode_dot(core, [fit.U_ocean, fit.U_var], modes=[1, 2])
```

**Rank selection:**
- `r_ocean`: start 200. Increase until `reconstruction_error < 0.05`. Expect 300–500 for global 0.25°.
- `r_var`: start `min(V, 6)`. Cross-variable structure is low-dimensional.

Verify the exact tuple-unpacking pattern for `partial_tucker` in the installed tensorly version — the docstring above shows the current expected return shape but tensorly has changed its return signature in the past. First cell of the notebook should print `partial_tucker.__doc__` to confirm.

### 5.6 DMD — PyDMD wrapper

Lives in `train/dmd.py`. Not a transformer.

```python
from pydmd import DMD, BOPDMD, HankelDMD
from typing import Literal


@dataclass(frozen=True, slots=True)
class DMDFit:
    Phi: np.ndarray         # (N_state, r_dmd) complex64
    Lambda: np.ndarray      # (r_dmd,)         complex128
    b0: np.ndarray          # (r_dmd,)         complex128
    lam_cont: np.ndarray    # (r_dmd,)         complex128
    dt: float
    variant: str            # "exact", "bop", "hankel" — for provenance


def dmd_fit(
    core: np.ndarray,                  # (T, r_ocean, r_var) from tucker_transform
    r_dmd: int,
    dt: float,
    rng: np.random.Generator,
    variant: Literal["exact", "bop", "hankel"] = "exact",
    bop_num_trials: int = 100,
    bop_trial_size: float = 0.5,
    hankel_delays: int = 20,
) -> DMDFit:
    """DMD via PyDMD (Tu et al. 2014; Sashidhar & Kutz 2022 for BOP).

    variant="exact" — plain exact DMD.
    variant="bop"   — bagging + optimized DMD with conjugate-pair constraint.
                      Automatically ensures real reconstructions.
    variant="hankel"— time-delay embedded DMD (Brunton et al. 2017).
                      Use if per-cell temporal spectra undershoot at high freq.
    """
    T, r_o, r_v = core.shape
    N_state = r_o * r_v
    snapshots = core.reshape(T, N_state).T    # PyDMD expects (n_features, n_snapshots)

    if variant == "exact":
        model = DMD(svd_rank=r_dmd)
        model.fit(snapshots)
        Lambda = np.asarray(model.eigs, dtype=np.complex128)
        Phi = np.asarray(model.modes, dtype=np.complex64)
        b0 = np.asarray(model.amplitudes, dtype=np.complex128)
    elif variant == "bop":
        t_grid = np.arange(T) * dt
        model = BOPDMD(
            svd_rank=r_dmd,
            num_trials=bop_num_trials,
            trial_size=bop_trial_size,
            eig_constraints={"conjugate_pairs"},   # ensures real reconstruction (§3.11)
        )
        model.fit(snapshots, t=t_grid)
        Lambda = np.exp(np.asarray(model.eigs, dtype=np.complex128) * dt)
        Phi = np.asarray(model.modes, dtype=np.complex64)
        b0 = np.asarray(model.amplitudes, dtype=np.complex128)
    elif variant == "hankel":
        model = HankelDMD(svd_rank=r_dmd, d=hankel_delays)
        model.fit(snapshots)
        Lambda = np.asarray(model.eigs, dtype=np.complex128)
        Phi = np.asarray(model.modes, dtype=np.complex64)
        b0 = np.asarray(model.amplitudes, dtype=np.complex128)
    else:
        raise ValueError(f"Unknown DMD variant: {variant!r}")

    lam_cont = (np.log(Lambda) / dt).astype(np.complex128)
    return DMDFit(
        Phi=Phi, Lambda=Lambda, b0=b0, lam_cont=lam_cont, dt=dt, variant=variant,
    )
```

**Rank `r_dmd`:** start 50, extend to 100–200 for a global model. Diagnostic: plot `Lambda` on the complex plane. Retain modes on or near the unit circle plus the leading decaying continuum.

**Complex-conjugate pair handling.** With `variant="bop"`, the `eig_constraints={"conjugate_pairs"}` option enforces this at fit time — no manual handling needed. With `variant="exact"` or `"hankel"`, the pairs are still present in the fit but the sampler must force `η_{k'}(t) = conj(η_k(t))` at every step (see §5.6.1 below and §9 gotcha #8). **Default to `variant="exact"` during notebook iteration** (§8.6 Cells 5-8 re-fit DMD on every Tucker/DMD rank sweep; BOP-DMD's ~100-model bagging makes that 100× slower per sweep, which is painful across a rank grid). **Switch to `variant="bop"` only for the final production fit**, once ranks are settled — the conjugate-pair constraint and bagged UQ are worth the one-time 100× cost there, but not during exploration.

#### 5.6.1 Mode-level OU on residual amplitudes

For each DMD mode `k`:

```
a_k(t) = k-th component of (Φ⁺ · x_t)           # measured complex amplitude
â_k(t) = b0_k · exp(lam_cont_k · t)              # deterministic prediction
η_k(t) = a_k(t) − â_k(t)                         # residual
```

Fit AR(1) → OU:

```
φ_mode(k) = Σ_t conj(η_k(t)) · η_k(t+Δt)  /  Σ_t |η_k(t)|²
θ_mode(k) = −log(clip(Re(φ_mode(k)), 1e-4, 0.9999)) / Δt
σ_mode(k)² = Var(η_k) · 2·θ_mode(k) / (1 − |φ_mode(k)|²)     # continuous-time diffusion
```

**`θ_mode(k)` clips `Re(φ_mode(k))`, not `|φ_mode(k)|`.** These are not
interchangeable: a negative real part means the mode's residual alternates
sign step-to-step (anti-correlation), which is physically distinct from slow
mean-reversion even if `|φ|` happens to be large in that case. Clipping the
*real part* pushes negative-real-part modes to the fast-decorrelation end of
the range (`θ` large) rather than assigning them whatever slow decay `|φ|`
alone would imply — deliberately treating alternation as noise-like, not
as a real slow oscillation. Using `|φ|` here would recover the wrong `θ` for
any mode with `Re(φ) < 0`.

Lives in `train/dmd.py` alongside `dmd_fit`. Returns a `ModeOUFit` frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class ModeOUFit:
    theta: np.ndarray       # (r_dmd,) float64
    sigma: np.ndarray       # (r_dmd,) float64
    phi:   np.ndarray       # (r_dmd,) complex128
    dt:    float
```

### 5.7 Local residual OU field (new in v2)

Lives in `train/local_residual.py`. This is the field-valued analog of the mode-level OU — one independent OU process per ocean cell per variable, fit to the coherent-reconstruction residual.

```python
@dataclass(frozen=True, slots=True)
class LocalResidualFit:
    theta:            np.ndarray   # (N_ocean, V) float32
    sigma:            np.ndarray   # (N_ocean, V) float32
    phi:              np.ndarray   # (N_ocean, V) float32
    variance_ratio:   np.ndarray   # (N_ocean, V) float32 — Var(R)/Var(Z), diagnostic
    active_mask:      np.ndarray   # (N_ocean, V) bool — True where variance_ratio > floor
    dt:               float


def coherent_reconstruction(
    core_deterministic: np.ndarray,   # (T, r_ocean, r_var) from DMD forward
    tucker: TuckerFit,
) -> np.ndarray:
    """Compute Z_coherent(s, v, t) from the fitted DMD's deterministic reconstruction."""
    return tucker_inverse(core_deterministic, tucker)


def fit_local_residual(
    Z: np.ndarray,                    # (T, N_ocean, V) Gaussianized anomalies
    Z_coherent: np.ndarray,           # (T, N_ocean, V) from coherent_reconstruction
    dt: float,
    variance_floor: float = 0.02,
) -> LocalResidualFit:
    """Fit independent per-cell OU to the residual R = Z − Z_coherent (§3.10, §5.7).

    Vectorized AR(1) fit across all cells simultaneously — no loop, no joblib.
    """
    R = Z - Z_coherent                                          # (T, N_ocean, V)
    R_prev = R[:-1]                                             # (T-1, N_ocean, V)
    R_next = R[1:]                                              # (T-1, N_ocean, V)

    # AR(1) coefficient per cell per variable
    num = np.sum(R_prev * R_next, axis=0)                        # (N_ocean, V)
    den = np.sum(R_prev * R_prev, axis=0) + 1e-30
    phi = np.clip(num / den, -0.9999, 0.9999).astype(np.float32)

    # Continuous-time reversion rate: phi = exp(-theta * dt)
    # Clip phi itself (not |phi|) to [1e-4, 0.9999]: negative phi indicates
    # alternation (anti-correlation), which we treat as fast decorrelation by
    # pushing it to the clip floor, not by discarding its sign via abs() (the
    # same distinction §5.6.1's mode-OU theta makes, there via Re(phi) since
    # that phi is complex; here phi is already real, so plain clip suffices).
    theta = -np.log(np.clip(phi, 1e-4, 0.9999)) / dt

    # Diffusion coefficient from the AR(1) *innovation* variance, not R's raw
    # marginal (stationary) variance — these are not interchangeable. At
    # stationarity Var(R) = innovation_variance / (1 - phi**2), so using
    # Var(R) directly here double-counts the (1 - phi**2) factor and inflates
    # sigma by 1/sqrt(1 - phi**2) — negligible near phi=0, a 3x+ overestimate
    # at phi=0.95 (verified numerically 2026-07-18). This is the same
    # distinction §5.6.1's mode-OU sigma already makes for the complex
    # mode-level fit ("Var(eta_k) in the source formula is the AR(1)
    # innovation variance... not the marginal variance"); this section
    # previously used raw Var(R), which was wrong for the identical reason.
    innovation = R_next - phi[None, :, :] * R_prev
    innovation_variance = np.mean(innovation**2, axis=0)          # (N_ocean, V)
    var_R = R.var(axis=0)          # (N_ocean, V) — used only for variance_ratio below
    var_Z = Z.var(axis=0) + 1e-30
    # Continuous-time diffusion: innovation_variance = sigma^2 (1 - phi^2) / (2 theta)
    sigma = np.sqrt(
        innovation_variance * 2 * theta / (1 - phi**2 + 1e-30)
    ).astype(np.float32)

    variance_ratio = (var_R / var_Z).astype(np.float32)
    active_mask = variance_ratio > variance_floor

    # Zero out inactive cells to save simulation work
    theta = np.where(active_mask, theta, 0.0).astype(np.float32)
    sigma = np.where(active_mask, sigma, 0.0).astype(np.float32)
    phi = np.where(active_mask, phi, 0.0).astype(np.float32)

    return LocalResidualFit(
        theta=theta, sigma=sigma, phi=phi,
        variance_ratio=variance_ratio, active_mask=active_mask, dt=dt,
    )
```

**Diagnostic to log at fit time:** fraction of cells where `active_mask` is True, per variable. If it's below ~30% for a variable, the coherent modes are already doing most of the job for that variable and the local residual is essentially noise. If it's above ~80%, the coherent modes are missing important structure and rank choices should be revisited.

### 5.8 Sampling and continuous-time query

Free functions in `weather/simulate.py`. They read from the fitted `WeatherModel`.

**Continuous-time point query — the `sample_batch` fast path.** For each query time `t_q`, use the OU closed-form transition rather than a full grid simulation:

**Step 1: Evolve mode-level OU to query time.**
```
Δn = (t_q − t_train_start).total_seconds() / dt      # can be fractional
For each DMD mode k:
    â_k(t_q) = b0_k · exp(lam_cont_k · (t_q − t_train_start))
    η_k(t_q) = φ_mode(k)^Δn · η_k(t_ref)
             + N_ℂ(0, σ_mode(k)² · (1 − |φ_mode(k)|^(2·Δn)) / (2·θ_mode(k)))
    a_k(t_q) = â_k(t_q) + η_k(t_q)
```

For `variant="exact"` or `"hankel"` DMD, after drawing `η_k` for the first mode of each conjugate pair, force `η_{k'}(t_q) = conj(η_k(t_q))` for its partner. For `variant="bop"` this is automatic.

**Step 2: Reconstruct coherent field on the grid.**
```
x(t_q) = Re(Σ_k a_k(t_q) · Φ_k)                      # (N_state,)
core(t_q) = x(t_q).reshape(r_ocean, r_var)           # (r_ocean, r_var)
Z_coherent(·, ·, t_q) = tucker_inverse(core(t_q)[None, ...], model.tucker_)[0]
                                                      # (N_ocean, V)
```

**Step 3: Draw local residual at each query point.**
```
For each query point (s_q, v_q):
    s = nearest ocean cell to (lat_q, lon_q)
    Z_local(s_q, v_q, t_q) = φ_local(s, v)^Δn · Z_local(s, v, t_ref)
                           + N(0, σ_local(s, v)² · (1 − φ_local(s, v)^(2·Δn)) / (2·θ_local(s, v)))
```

If no previous state `Z_local(s, v, t_ref)` is available (isolated query, no trajectory), draw from equilibrium: `N(0, σ_local² / (2·θ_local))`.

**Step 4: Sum, interpolate, invert.**
```
Z(s_q, v_q, t_q) = Z_coherent(s_q, v_q, t_q) + Z_local(s_q, v_q, t_q)
value(lat_q, lon_q, v_q, t_q) = bilinear_interp(Z, lat_q, lon_q)
                                  # with lon-wraparound padding per §2.6
result = model.preprocess_.inverse_transform(value)
```

Wrap in a small xarray Dataset containing just the queried points so `inverse_transform` handles Gaussianization inversion, standardization inversion, climatology addition, and `(u, v) → (speed, direction)` recombination.

**Compute cost:** step 1 is `O(N_queries · r_dmd)`, step 2 is `O(N_ocean · r_dmd)` per unique `t_q` (batchable), step 3 is `O(N_queries)`, step 4 is `O(N_queries)`. For `N_queries = 10⁴` and `r_dmd = 100`, total ~10 µs per query batched — matches criterion §1.5.

**Full-field simulation on the 3h grid** (`simulate_field`) uses the same steps but with query times `t_q ∈ {t_start, t_start + Δt, ...}` on the full ocean-cell grid. The local residual is generated as `N_ocean` independent AR(1) trajectories, again vectorized.

---

## 6. Validation

**Training:** 2022-01-01T00:00Z → 2025-12-31T21:00Z.
**Held-out metrics window:** 2026-01-01T00:00Z → 2026-07-19T21:00Z. Real archive data available; used for numerical validation with ground-truth comparison.
**Extrapolation window:** 2026-07-20T00:00Z onward. Archive ends here; no ground-truth comparison possible. Assessed by internal-consistency and stability criteria only.

The validation battery splits accordingly into two parts. `weather.validate.full_report` runs both and assembles them into a PDF:

```python
full_report(
    model,
    data_dir=Path("./data"),
    metrics_start="2026-01-01T00:00Z",
    metrics_end="2026-07-19T21:00Z",       # last snapshot with real data
    extrapolation_start="2026-07-20T00:00Z",
    extrapolation_end="2026-12-31T21:00Z",  # or wherever the router needs to reach
    output=Path("./reports/v0.1.0_validation.pdf"),
)
```

### 6.A Held-out metrics (2026-01-01 → 2026-07-19)

Real data available. All must pass to consider v0.1.0 ready to ship.

1. **Per-cell marginal QQ.** 30 randomly-chosen ocean cells, per variable, real 2026-H1 vs. simulated 2026-H1. Should be near-diagonal.
2. **Per-cell temporal spectra (Welch PSD).** Same cells, log-log plot. Must match on the `[3h, 30d]` frequency band. Systematic under-representation at high freq → switch DMD variant to `"hankel"` (§3.13).
3. **Long-range empirical variogram** at a fixed snapshot, averaged across snapshots. Real vs. simulated must agree within ±10% at ranges `[100 km, 5000 km]`.
4. **Short-lag variogram** at ranges `[25 km, 200 km]`. Direct test that the local residual layer (§3.10) is doing its job. Real weather has non-negligible variance at 25 km (one grid cell); model samples should too. If short-lag variance is systematically below real, lower `local_residual_variance_floor` or escalate to SPDE-GMRF (§3.13).
5. **Joint density at a single cell.** `(sigwh, was)` and `(u_wind, v_wind)` scatter plots. Correct correlation structure in both magnitude and sign.
6. **Extreme statistics.** 99th percentile of `sigwh` per cell for the metrics window, real vs. simulated. Systematic under-estimate → escalate copula per §3.13. **Note:** the metrics window covers only H1 (winter and spring — no Southern Hemisphere winter storms, no Northern Hemisphere hurricane season peak). This generic percentile check doesn't target hurricane season specifically — §6.D's checks 18-19 do that directly.
7. **Held-out `.transform` residual distribution.** Apply `model.transform(metrics_window_data)` (runs only preprocessing) and KS-test per-cell values against `N(0, 1)`. Systematic non-normality on 2026-H1 → the copula or climatology overfit the 2022–2025 window.
8. **Trend-term diagnostic.** Report `trend_significance_ratio` distribution across ocean cells per variable. Verify median above 0.02 (trend term doing real work) but under 0.5 (trend not overwhelming seasonal cycle). Spatial map for regions where the ratio is high (Mediterranean SST, Arctic SST).
9. **Climatology extrapolation check.** Compare `μ_predicted(2026-04-15)` (from the fitted climatology, extrapolated 4 months past `t_train_end`) against `μ_estimated_from_2026-H1_data_alone`. If the trend term improved the extrapolation (lower RMSE than a no-trend fit), keep `include_trend=True`. If not, drop it.
10. **Local residual variance ratio map.** Report `active_mask` fraction per variable and per basin. If a variable's active mask is < 30% globally, question whether the residual layer is worth its cost for that variable.

### 6.B Extrapolation-only checks (2026-07-20 onward)

No ground truth available. These are internal-consistency and stability checks — the model is generating plausible samples for dates the archive doesn't cover, which is the actual use case. All must pass.

11. **Marginal drift.** Sample the extrapolation window at each cell and per variable; compare per-cell marginals against those of the held-out metrics window (§6.A.1). Marginals should differ only in ways attributable to (a) climatology's seasonal cycle (H2 is the mirror season of H1 in each hemisphere), (b) the climatology trend term (small, ~months of drift), and (c) sampling noise. **Systematic drift** — e.g. simulated `sigwh` grows unboundedly over months of extrapolation, or wind speeds decay toward zero — indicates either DMD modes with non-physical growth (`Re(lam_cont) > 0`) or OU parameter numerical instability. Diagnose by plotting per-cell mean and std over the full extrapolation window; expect seasonal cycle plus mild trend, not monotone drift.
12. **Deterministic-only sanity check.** Run `simulate_field` with the OU noise disabled (an internal flag; not a public option). What's left is `μ(s, t) + Re(Σ_k â_k(t) Φ_k)` — pure climatology + deterministic DMD skeleton, no stochasticity. Verify this looks like weather (seasonal cycle, gradient toward the equator, storm tracks in the right latitude bands) — not like noise or a monotone field. Failure here means the DMD deterministic skeleton has captured something non-physical.
13. **Reproduction test — the actual use case.** For a routing use case scenario spanning the extrapolation window, invoke `sample_batch(model, ...)` at typical query densities and confirm: (a) no NaN in output over ocean cells, (b) no negative Hs / wind speed, (c) direction values in `[0, 360)`, (d) per-cell temporal spectra at the extrapolation window have the same shape as the metrics window (§6.A.2) — the *characteristics* of the weather are the same, even if the specific realization isn't verifiable. This is the "does it produce plausible weather for the routing engine" check.
14. **Two-seed stability.** Generate two full-basin trajectories over the extrapolation window with two different seeds. Compute per-cell mean and std across the two realizations. The difference between the two per-cell means should be Gaussian around zero (paired samples from the same distribution); a systematic bias indicates the sampler is seed-dependent in a way it shouldn't be.
15. **Round-trip through climatology.** For a small subset of extrapolation-window query times, verify `model.preprocess_.inverse_transform(model.preprocess_.transform(model.simulate_field(...))) ≈ model.simulate_field(...)` within numerical tolerance. Catches pickle-load errors in the preprocessing chain that only manifest on unseen dates.

### 6.C Performance and consumer acceptance

Applied on both windows.

16. **Batched throughput benchmark.** Time `sample_batch` at N = {1, 100, 10 000, 1 000 000}. Report per-point µs. Confirm the batched fast path meets §1 criterion 5.
17. **End-to-end consumer test.** For each real consumer (routing engine, wave-climate summary): run the consumer on real 2026-H1 weather and on synthetic 2026-H1 weather, compare output distributions (§6.A domain). Then run on synthetic 2026-H2 weather, verify the consumer's output distribution is *self-consistent* with H1 (allowing for seasonal differences) — no crashes, no NaN, no out-of-range values (§6.B domain). This is the acceptance test that actually matters.

### 6.D North Atlantic hurricane season — a targeted check

None of 1-17 specifically exercises tropical cyclones. The held-out metrics window (2026 H1, Jan-Jul) misses North Atlantic hurricane season (peaks Aug-Oct) entirely — it falls in the extrapolation-only window, §6.B, where there's no ground truth. For a maritime-routing consumer this is arguably the highest-stakes season the model needs to get right, so it gets its own pair of checks rather than relying on the generic ones above to catch a problem here incidentally.

**Storm-signature detector (shared by both checks below).** A candidate tropical cyclone, per 3h snapshot, is a contiguous patch of ocean cells in the tropical/subtropical North Atlantic (~8-35°N, 15-100°W) where:
- `was` ≥ 17 m/s (tropical-storm force) over a minimum contiguous area, co-located with elevated `sigwh` (wave energy ∝ Hs², so this is the "high energy density" criterion).
- A coherent cyclonic (positive, NH) vorticity maximum, computed from `(u_wind, v_wind)`, co-located with the patch — the rotational structure that distinguishes an actual cyclone from "it's just windy here," and the specific structure plain EOF+AR can't represent but DMD's complex-conjugate modes should (§3.3's rejection rationale — this check directly stress-tests that choice).
- Tracked across consecutive snapshots, the patch/vorticity-max centroid persists ≥24h and translates at a physically plausible speed (~10-30 km/h) — the "moving front" criterion, distinguishing a genuine storm from a flickering isolated extreme value.

18. **Historical hurricane-season reproduction (in-sample, real ground truth).** For each Aug-Oct in 2022, 2023, 2024, 2025 (all inside the training window — this is an in-sample fit-quality check, not an out-of-sample generalization test; see the leave-one-year-out escalation in §3.13 if it looks weak): run the storm-signature detector on the real archive data to get a per-season baseline (feature count, peak intensity, typical translation speed). Simulate the same windows (multiple seeds) and run the same detector on the output. The simulated distribution across seeds should be statistically consistent with the real per-season baseline — matching the *statistics* of storms across the four seasons, not reproducing individual historical storms (that's explicitly a non-goal, §1). A model that can't even reproduce known hurricane seasons in-sample has no chance of extrapolating them correctly to 2026.
19. **Extrapolation structural check (2026 Aug-Oct, no ground truth).** Simulate the actual production extrapolation window and run the same storm-signature detector. There's nothing real to compare against, so this isn't a statistical-match check like #18 — it's a structural capability check: does at least one detector-qualifying feature (high energy density + coherent rotation + persistent translation) emerge at all during the season when the real North Atlantic reliably produces them? A model that produces zero qualifying features here, despite passing #18 in-sample, indicates the coherent DMD modes responsible for cyclone-like structure aren't extrapolating past the training window — investigate `Re(lam_cont)` for the responsible modes before shipping.

---

## 7. Persisted model artifacts

Directory layout — one dir per model version:

```
./models/<version>/
├── metadata.json          # variable list, grid extents, cadence, training window,
│                          # ranks, DMD variant, git commit hash,
│                          # sklearn / numpy / scipy / tensorly / pydmd versions
├── ocean_mask.npy         # (621, 1440) bool
├── preprocess.pkl         # fitted sklearn Pipeline
├── tucker.npz             # U_ocean, U_var, reconstruction_error
├── dmd.npz                # Phi, Lambda, b0, lam_cont, dt, variant
├── mode_ou.npz            # theta_mode, sigma_mode, phi_mode
└── local_residual.npz     # theta_local, sigma_local, phi_local,
                           # variance_ratio, active_mask
```

**Why pickle for the preprocessing Pipeline.** sklearn transformers pickle cleanly; joblib.dump is the sklearn-native persistence path. The pipeline state (climatology coefficients + copula LUT) is a few GB, manageable.

**Why `.npz` for the decomposition stages.** Pure array bundles, no class structure. Numpy-native, faster to load, cross-version-stable.

Dtype policy: `float32` for large arrays (`Phi`, `U_ocean`, `theta_local`, `sigma_local`), `float64` for small ones (mode OU parameters, DMD eigenvalues). Complex: `complex64` for `Phi`, `complex128` for `Lambda`, `b0`, `lam_cont`, `phi_mode`.

Ballpark total size for a 2022–2025 tier-1 model: 6–8 GB, dominated by `preprocess.pkl`.

---

## 8. Repo bootstrap

### 8.1 Directory layout

```
weather_generator/
├── AGENTS.md
├── pyproject.toml
├── poetry.lock
├── README.md
├── CHANGELOG.md
├── docs/
│   └── model_design.md               # this document
├── notebooks/
│   ├── test_pipeline.ipynb           # training entry point (§8.6)
│   ├── verify_direction_convention.ipynb    # coming_from/going_to check (§2.3, §2.4) — done
│   └── verify_ocean_grid_orientation.ipynb  # TASK: §2.7 land-correlation study — not started;
│                                             # required before sst/ocs/ocd (Tier 2/3, §3.8)
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                 # Grid constants (Final), Settings
│   │   └── io.py                     # snapshot_path, load_snapshot,
│   │                                 # decode_compass_code
│   ├── train/
│   │   ├── __init__.py
│   │   ├── ingest.py                 # load_training_window (xarray) + consistency check (§5.1)
│   │   ├── directional.py            # ComingFromToUV, GoingToToUV transformers (§5.3)
│   │   ├── reproject.py              # Reprojector
│   │   ├── climatology.py            # ClimatologyRemover (with trend, §5.4)
│   │   ├── standardize.py            # PerCellStandardizer
│   │   ├── copula.py                 # GaussianCopula
│   │   ├── tucker.py                 # tucker_fit + inverse via tensorly (§5.5)
│   │   ├── dmd.py                    # dmd_fit + mode-OU via pydmd (§5.6)
│   │   └── local_residual.py         # NEW: fit_local_residual (§5.7)
│   └── weather/
│       ├── __init__.py               # Public API surface (§8.4)
│       ├── model.py                  # WeatherModel(BaseEstimator, TransformerMixin)
│       ├── simulate.py               # sample, sample_batch, simulate_field
│       └── validate.py               # full_report + per-check helpers
├── tests/
│   ├── conftest.py
│   ├── core/
│   ├── train/
│   │   └── test_local_residual.py    # NEW in v2. Plus test_directional.py,
│   │                                 # test_pipeline_composition.py
│   └── weather/
└── .github/
    └── workflows/
        └── ci.yml
```

`data/` and `models/` are `.gitignore`d.

### 8.2 `pyproject.toml` (Poetry)

```toml
[tool.poetry]
name = "weather-generator"
version = "0.1.0"
description = "Stochastic weather generator for point queries at arbitrary (lat, lon, datetime)"
readme = "README.md"
packages = [
    { include = "core", from = "src" },
    { include = "train", from = "src" },
    { include = "weather", from = "src" },
]

[tool.poetry.dependencies]
python = "^3.14"
numpy = ">=2.0"
scipy = ">=1.14"
scikit-learn = ">=1.5"
xarray = ">=2024.10"
pandas = ">=2.2"
tensorly = ">=0.9"           # NEW in v2 (§5.5)
pydmd = ">=2024"             # NEW in v2 (§5.6)
pydantic-settings = ">=2.0"

[tool.poetry.group.dev.dependencies]
pytest = ">=8"
pytest-mock = "*"
pytest-cov = "*"
pytest-xdist = "*"
matplotlib = ">=3.9"
black = "*"
isort = "*"
mypy = ">=1.13"
ipykernel = "*"
jupyter = "*"

[tool.black]
line-length = 100
target-version = ["py314"]

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.14"
strict = true
warn_unreachable = true
mypy_path = "src"
namespace_packages = true

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]
markers = [
    "integration: opt-in integration tests (require real data)",
]

[build-system]
requires = ["poetry-core>=1.9.0"]
build-backend = "poetry.core.masonry.api"
```

**Verify tensorly and pydmd wheels for Python 3.14 exist before any implementation work.** Run `poetry install` and check for compilation warnings or fallback-to-source-build behavior. If either package doesn't have 3.14 wheels, log an issue and decide whether to (a) pin Python to 3.13, (b) build from source in CI, or (c) wait for maintainers.

### 8.3 Package dependency rules

- **`core/`** imports only stdlib, numpy, xarray, pydantic-settings.
- **`train/`** imports from `core`, plus numpy, scipy, scikit-learn, xarray, **tensorly, pydmd**. Does not import from `weather`.
- **`weather/`** imports from `core` and `train`, plus numpy, scipy, scikit-learn, xarray, matplotlib.

Enforce with `tests/test_import_boundaries.py`.

### 8.4 Public API surface

```python
# weather/__init__.py
from weather.model import WeatherModel
from weather.simulate import sample, sample_batch, simulate_field
from weather.validate import full_report

__all__ = [
    "WeatherModel",
    "sample", "sample_batch", "simulate_field",
    "full_report",
]
```

Grid constants come from `core.config`:

```python
from core.config import WAVE_GRID_SHAPE, WAVE_GRID_LAT_AT_ROW0
```

### 8.5 The `WeatherModel` class

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self, Literal

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from train.tucker import TuckerFit
from train.dmd import DMDFit, ModeOUFit
from train.local_residual import LocalResidualFit


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    trained_at: datetime
    t_train_start: datetime
    t_train_end: datetime
    t_ref: datetime                          # climatology reference epoch (mid-window)
    variables: tuple[str, ...]
    r_ocean: int
    r_var: int
    r_dmd: int
    dmd_variant: str
    include_trend: bool
    include_local_residual: bool
    git_commit: str
    sklearn_version: str
    numpy_version: str
    tensorly_version: str
    pydmd_version: str


class WeatherModel(BaseEstimator, TransformerMixin):
    """Hierarchical stochastic weather generator.

    Architecture: climatology (linear + harmonic) + coherent modes (Tucker + DMD + OU)
    + per-cell local residual OU field. sklearn-compatible transformer surface.

    Sampling is NOT a method on this class. Use free functions in weather.simulate:
        sample(model, ...)
        sample_batch(model, ...)
        simulate_field(model, ...)

    Fitted attributes (trailing underscore per sklearn convention):
        preprocess_       : sklearn.pipeline.Pipeline (directional → gaussian copula)
        tucker_           : TuckerFit
        dmd_              : DMDFit
        mode_ou_          : ModeOUFit
        local_residual_   : LocalResidualFit  (None if include_local_residual=False)
        ocean_mask_       : np.ndarray, shape (621, 1440), bool
        metadata_         : ModelMetadata
    """

    def __init__(
        self,
        t_start: datetime,
        t_end: datetime,
        variables: list[str],
        r_ocean: int = 300,
        r_var: int = 6,
        r_dmd: int = 100,
        dmd_variant: Literal["exact", "bop", "hankel"] = "exact",
        copula_lut_bins: int = 500,
        diurnal_variables: list[str] | None = None,
        include_trend: bool = True,
        trend_significance_threshold: float = 0.1,
        include_local_residual: bool = True,
        local_residual_variance_floor: float = 0.02,
        random_state: int | None = None,
    ) -> None:
        # sklearn convention: __init__ only stores parameters, no validation.
        self.t_start = t_start
        self.t_end = t_end
        self.variables = variables
        self.r_ocean = r_ocean
        self.r_var = r_var
        self.r_dmd = r_dmd
        self.dmd_variant = dmd_variant
        self.copula_lut_bins = copula_lut_bins
        self.diurnal_variables = diurnal_variables
        self.include_trend = include_trend
        self.trend_significance_threshold = trend_significance_threshold
        self.include_local_residual = include_local_residual
        self.local_residual_variance_floor = local_residual_variance_floor
        self.random_state = random_state

    # --- sklearn transformer surface ---------------------------------------
    def fit(self, X: Path, y: None = None) -> Self: ...
    def transform(self, X: Path) -> xr.Dataset:
        """Apply preprocessing chain only. Returns Gaussianized anomalies."""
    def fit_transform(self, X: Path, y: None = None) -> xr.Dataset: ...

    # --- Alternate constructor for notebook assembly -----------------------
    @classmethod
    def from_components(
        cls,
        *,
        preprocess: Pipeline,
        tucker: TuckerFit,
        dmd: DMDFit,
        mode_ou: ModeOUFit,
        local_residual: LocalResidualFit | None,
        ocean_mask: np.ndarray,
        metadata: ModelMetadata,
    ) -> Self: ...

    # --- Persistence -------------------------------------------------------
    def save(self, model_dir: Path) -> None: ...
    @classmethod
    def load(cls, model_dir: Path) -> Self: ...

    # --- Introspection -----------------------------------------------------
    def ocean_mask(self) -> np.ndarray:
        return self.ocean_mask_
```

Default `dmd_variant="exact"` — fast enough for repeated Tucker/DMD rank sweeps during notebook iteration (§8.6). Switch to `dmd_variant="bop"` only for the final production fit, once ranks are settled: `eig_constraints={"conjugate_pairs"}` then handles the complex-conjugate pair problem automatically and adds bagged uncertainty quantification, at ~100× the fit cost of `"exact"` — worth paying once, not on every sweep.

### 8.6 Notebook workflow (`notebooks/test_pipeline.ipynb`)

Training is a notebook activity. Structure so that expensive stages run once and cheap stages iterate.

```
Cell 1  — Setup: imports, paths, SEED, training + validation windows.
          T_TRAIN_START      = datetime(2022, 1,  1,  tzinfo=timezone.utc)
          T_TRAIN_END        = datetime(2026, 1,  1,  tzinfo=timezone.utc)   # exclusive
          T_METRICS_START    = datetime(2026, 1,  1,  tzinfo=timezone.utc)
          T_METRICS_END      = datetime(2026, 7, 19, 21, tzinfo=timezone.utc) # last real snap
          T_EXTRAP_START     = datetime(2026, 7, 20, 0,  tzinfo=timezone.utc) # first synth-only
          T_EXTRAP_END       = datetime(2026, 12, 31, 21, tzinfo=timezone.utc) # or wherever
                                                                              # the router needs

Cell 2  — Ingest training data (2022–2025 only, NOT including 2026-H1).
          X_raw = load_training_window(data_dir, T_TRAIN_START, T_TRAIN_END, variables)

Cell 3  — Build and fit the preprocessing Pipeline (§5.2–5.4).
          preprocess = Pipeline([
              ("coming_from_to_uv", ComingFromToUV(pairs=[
                  ("wsh", "wsd", "u_wsea", "v_wsea"),
                  ("was", "wad_come_from", "u_wind", "v_wind"),
              ])),
              ("going_to_to_uv", GoingToToUV(pairs=[
                  # empty for tier 1; tier 3 (ocean currents) adds:
                  # ("ocs", "ocd", "u_current", "v_current"),
              ])),
              ("reproject",   Reprojector(target_grid="wave")),
              ("climatology", ClimatologyRemover(
                  diurnal_variables=["sst"],
                  include_trend=True,
              )),
              ("standardize", PerCellStandardizer()),
              ("gaussianize", GaussianCopula(n_bins=500)),
          ], memory=Path("./cache"))          # cache for iteration
          Z = preprocess.fit_transform(X_raw)              # slow: ~30 min
          # inspect climatology diagnostics:
          climatology = preprocess.named_steps["climatology"]
          print(f"trend-significant cells: "
                f"{(climatology.trend_significance_ratio_ > 0.1).mean():.1%}")

Cell 4  — Inspect Z: QQ plots at random cells (should be N(0,1)).

Cell 5  — Tucker at various ranks.
          tucker_200 = tucker_fit(Z.to_numpy(), r_ocean=200, r_var=6, rng=rng)
          tucker_400 = tucker_fit(Z.to_numpy(), r_ocean=400, r_var=6, rng=rng)
          print(f"200: err={tucker_200.reconstruction_error:.4f}")
          print(f"400: err={tucker_400.reconstruction_error:.4f}")

Cell 6  — DMD at chosen Tucker. `variant="exact"` here — this cell reruns on
          every rank sweep, and BOP-DMD's ~100-model bagging makes that
          painfully slow during iteration. Refit with `variant="bop"` only for
          the final production model (see "Production runs" below); the
          WeatherModel assembled in Cell 9 below is for validation feedback
          during iteration, not the model that ships.
          core = tucker_transform(Z.to_numpy(), tucker_400)
          dmd = dmd_fit(core, r_dmd=100, dt=DT, rng=rng, variant="exact")
          plot_eigenvalues(dmd)

Cell 7  — Mode OU fit against the chosen DMD.
          mode_ou = fit_mode_ou(core, dmd)

Cell 8  — Coherent reconstruction, then local residual fit (NEW in v2).
          Z_coherent = coherent_reconstruction(core, tucker_400, dmd, mode_ou)
          local = fit_local_residual(Z.to_numpy(), Z_coherent, dt=DT,
                                      variance_floor=0.02)
          print(f"active local cells: {local.active_mask.mean():.1%}")
          plot_variance_ratio_map(local)

Cell 9  — Assemble WeatherModel; save.
          model = WeatherModel.from_components(
              preprocess=preprocess,
              tucker=tucker_400, dmd=dmd, mode_ou=mode_ou,
              local_residual=local,
              ocean_mask=ocean_mask, metadata=metadata,
          )
          model.save(Path("./models/v0.1.0"))

Cell 10 — Validation: metrics on 2026-H1 (real ground truth),
          extrapolation checks on 2026-H2 (no ground truth).
          report = full_report(
              model, data_dir=data_dir,
              metrics_start=T_METRICS_START,
              metrics_end=T_METRICS_END,
              extrapolation_start=T_EXTRAP_START,
              extrapolation_end=T_EXTRAP_END,
              output=Path("./reports/v0.1.0_validation.pdf"),
          )
```

**Production runs** (after the notebook workflow has settled) use the "just train the whole thing" one-liner. This is the one place `dmd_variant="bop"` belongs — everywhere upstream in the iteration workflow (§8.6 Cell 6) uses `"exact"`:

```python
model = WeatherModel(
    t_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
    t_end=datetime(2026, 1, 1, tzinfo=timezone.utc),     # exclusive; last training snap
                                                          # is 2025-12-31T21:00Z
    variables=["sigwh", "wsh", "wsp", "was", "wad", "wsd"],
    r_ocean=300, r_var=6, r_dmd=100,
    dmd_variant="bop",  # final fit only — see §5.6, §8.5
    include_trend=True, include_local_residual=True,
    random_state=42,
)
model.fit(Path("./data"))
model.save(Path("./models/v0.1.0"))
```

**Important:** `t_end` is *exclusive* — the last training snapshot is `t_end - 3h`. For a training window ending at 2025-12-31T21:00Z, pass `t_end=datetime(2026, 1, 1, tzinfo=timezone.utc)`. This matches the `enumerate_snapshots(..., t_start, t_end)` convention in §5.1.

The 2026-H1 data (2026-01-01 to 2026-07-19T21:00Z) is **not** used for training — it's reserved entirely for the numerical validation battery. If the archive gets more data appended later, the training window can be extended; the metrics window then slides forward accordingly.

**Pipeline caching.** `Pipeline(memory=Path("./cache"))` transparently caches each transformer's output. Turn caching *off* (delete the cache dir) before production runs — the fitted `preprocess.pkl` shouldn't carry references to a disk cache that other machines don't have.

---

## 9. Gotchas and implementation notes

Ordered by expected pain if missed.

1. **Row 0 = south for both wave and wind grids.** Test explicitly: place a marker at 51.5° N, 0° E and confirm it lands on London in `sigwh`'s NaN mask, not in the Southern Ocean.

2. **Ocean grid orientation is unverified.** Do the correlation study in §2.7 before enabling `sst`/`ocs`/`ocd`. Do not assume by analogy.

3. **Longitude wraparound at 359.75° → 0°.** Any interpolation across this boundary produces NaN without periodic padding. Enforce at the code level.

4. **Directional interpolation must go through `(u, v)`.** Enforce by running **both** `ComingFromToUV` and `GoingToToUV` **before** `Reprojector` in the Pipeline. Every directional variable must belong to exactly one of the two transformers; nothing should reach `Reprojector` as a raw angle.

5. **Compass code origins AND native conventions both differ per field.**
   - Origins: `wsd`/`pwd`/`pswd` use code 0 = North (`origin_code=0`); `wad`/`wad_come_from`/`ocd` use code 1 = North (`origin_code=1`).
   - Conventions: `wsd`/`pwd`/`pswd`/`wad_come_from` are `coming_from`; `wad`/`ocd` are `going_to`.
   Both are handled by keeping the decoder convention-agnostic (§2.3) and pushing convention handling into the two transformer classes (§3.7, §5.3). Never combine origin and convention decisions in the same function — separation is what prevents entangled bugs.

5a. **`wad` is not read by the pipeline; `wad_come_from` is the source of truth.** Adding a pair `("was", "wad", ...)` to `GoingToToUV` would produce mathematically valid samples but bypass the point of using `wad_come_from` (to avoid a flip that might drift). The runtime `wad`/`wad_come_from` consistency check (§5.1) exists to catch upstream drift; do not remove it as "unnecessary." If it ever fails, that's the signal — not a false alarm — that the archive convention changed.

6. **`float16` precision on disk is much coarser than `float64` in memory.** `sigwh` at 8 m has ~0.008 m precision. Not a bug.

7. **Land mask consistency.** Use the *intersection* of `not isnan` across all modeled variables at all training snapshots.

8. **Complex-conjugate DMD pair handling — depends on DMD variant.** With `dmd_variant="bop"`, `eig_constraints={"conjugate_pairs"}` handles this automatically. With `"exact"` or `"hankel"`, the sampler must force `η_{k'}(t) = conj(η_k(t))` at every step. Add an assertion: `assert np.abs(Y_sim.imag).max() < 1e-10` after every simulation step in debug builds. **v2 defaults to `"exact"`** (fast enough for repeated rank sweeps during notebook iteration, §8.6) **so this manual handling is live during iteration** — don't skip it just because `"bop"` makes it unnecessary in the final fit. Switch to `"bop"` for the production model, which sidesteps this class of bug entirely at ~100× the fit cost — a cost worth paying once ranks are settled, not on every sweep.

9. **Wraparound at year boundaries in the climatology.** Compute `t_since_epoch_days` continuously from `t_ref` (the training-window midpoint); don't reset at Jan 1. The cosines handle periodicity.

10. **Parallelism goes on modes, seeds, and cells for the local residual — never on grid cells for the coherent layer.** Coherent-layer grid cells couple through Tucker reconstruction. But the local residual fit is embarrassingly parallel per cell (in fact vectorized in one numpy op — no joblib needed). Use `joblib.Parallel` for the mode-level OU across independent modes if it ever becomes a bottleneck.

11. **`float16` in arithmetic paths.** Never. Cast to `float64` at the I/O boundary.

12. **Time zones — always UTC.** Naive datetimes raise `ValueError` at the API boundary.

13. **Reference epochs are TWO datetimes, not one.**
    - `t_train_start = 2022-01-01T00:00Z` — beginning of training window; used by DMD/OU as the time origin for `Λ^{Δn}` and OU closed-form evolution.
    - `t_ref = 2024-01-01T10:30Z` — midpoint of the 2022-01-01 → 2026-01-01 training window, used as the climatology's linear trend origin (`Δτ = (t − t_ref) / 1_year`).
    Compute `t_ref` at fit time as `t_train_start + (t_train_end - t_train_start) / 2`. Do not hardcode; if the training window changes, `t_ref` should shift with it. Both stored in `metadata.json`. All temporal computations at query time reference the correct one — climatology uses `t_ref`, DMD/OU use `t_train_start`. Getting these swapped produces a systematic bias that's hard to spot in QQ plots but shows up loudly in the trend-term diagnostic (§6.A.8).

14. **Physical consistency across variables is NOT enforced.** `wsh ≤ sigwh` etc. can be violated. The Gaussian copula preserves correlations but not physical relationships. Defer projection layer until validation shows it's needed.

15. **Nested BLAS threading pitfall.** `sample_batch` must not spawn its own workers. Honor `OMP_NUM_THREADS` etc. set by the caller.

16. **sklearn `clone()` requires `__init__` transparency.** Every parameter passed to `__init__` must be stored on `self` under the same name, unmodified. `sklearn.base.clone(model)` uses `get_params()` (which reads those attributes) to create a fresh unfitted copy. Do validation lazily in `fit`.

17. **Pickle version drift.** `preprocess.pkl` is a pickled sklearn Pipeline. `metadata.json` records tensorly, pydmd, and sklearn versions used at training. `WeatherModel.load` warns if any differ from the runtime — a mismatch on tensorly or pydmd almost always breaks loading.

18. **Local residual assumes cell-independence.** If short-lag variogram check (§6.4) shows short-range spatial correlation in residuals matters, escalate to SPDE-GMRF (§3.13). For v0.1.0 the independence assumption is defensible but should be diagnosed explicitly.

19. **Local residual can over-fit near ice edges.** Cells near the seasonally-varying ice edge alternate between "valid ocean" and "ice" across the training window. The intersection ocean mask (§5.1) should exclude these, but any that slip through will produce spuriously large `σ_local` (the residual there is not really OU-distributed). Diagnostic: plot `σ_local` and check for a suspiciously bright rim around Antarctic/Arctic ice edges.

20. **Trend term can fit noise on short windows.** With 4 years of training data, the linear trend estimator is noisy in the presence of interannual variability (ENSO, NAO). The `trend_significance_ratio_` diagnostic in `ClimatologyRemover` reports where this is happening. If the median ratio is above ~0.5 in specific regions, the trend is likely aliasing interannual variability into a fake secular trend; escalate to index-regressed climatology (§3.13).

21. **`partial_tucker` return signature has drifted historically.** TensorLy has changed the exact tuple shape returned by `partial_tucker` between minor versions. First cell of the notebook: `print(partial_tucker.__doc__)` and confirm the unpacking pattern. Add a version pin (`tensorly>=0.9,<0.10`) once you've validated a specific version.

22. **PyDMD snapshot orientation is `(n_features, n_snapshots)`.** Opposite of what we treat internally. One `.T` at the boundary in `dmd_fit`. Getting this wrong produces plausible-looking eigenvalues on the wrong basis.

---

## 10. Variable table

Directions decoded per §2.3 (origin code) then handled by the appropriate transformer per §5.3 (native convention). The "Modeled as" column shows the final internal representation after preprocessing.

| File            | Meaning                            | On-disk dtype | Grid  | Origin | Native conv.   | Transformer         | Modeled as                                 | Tier |
|-----------------|------------------------------------|---------------|-------|--------|----------------|---------------------|--------------------------------------------|------|
| `sigwh.npy`     | Significant wave height            | float16       | wave  | —      | —              | (pass-through)      | scalar                                     | 1    |
| `wsh.npy`       | Wind-sea wave height               | float16       | wave  | —      | —              | (pass-through)      | scalar                                     | 1    |
| `wsp.npy`       | Wind-sea period                    | float16       | wave  | —      | —              | (pass-through)      | scalar                                     | 1    |
| `wsd.npy`       | Wind-sea direction                 | int8 (0–15)   | wave  | 0      | `coming_from`  | `ComingFromToUV`    | paired with `wsh` → `(u_wsea, v_wsea)`     | 1    |
| `was.npy`       | Wind speed at surface              | float16       | wind  | —      | —              | (pass-through)      | scalar                                     | 1    |
| `wad_come_from.npy` | Wind direction, coming from    | int8 (1–16)   | wind  | 1      | `coming_from`  | `ComingFromToUV`    | paired with `was` → `(u_wind, v_wind)`     | 1    |
| `wad.npy`       | Wind direction, going to           | int8 (1–16)   | wind  | 1      | `going_to`     | (consistency check only, §5.1) | **not modeled** — raw codes are `wad_come_from` shifted 180° (0% cell-identical; *not* a duplicate); same wind vector, opposite convention | 1    |
| `pwh.npy`       | Peak wave height                   | float16       | wave  | —      | —              | (pass-through)      | scalar                                     | 2    |
| `pwp.npy`       | Peak wave period                   | float16       | wave  | —      | —              | (pass-through)      | scalar                                     | 2    |
| `pwd.npy`       | Peak wave direction                | int8 (0–15)   | wave  | 0      | `coming_from` *(pending expert confirm)* | `ComingFromToUV` | paired with `pwh` → `(u_peak, v_peak)` | 2    |
| `sst.npy`       | Sea surface temperature            | float16       | ocean | —      | —              | (pass-through, verify units K/°C) | scalar                              | 2    |
| `pswh.npy`      | Primary swell height               | float16       | wave  | —      | —              | (pass-through)      | scalar                                     | 3    |
| `pswp.npy`      | Primary swell period               | float16       | wave  | —      | —              | (pass-through)      | scalar                                     | 3    |
| `pswd.npy`      | Primary swell direction            | int8 (0–15) assumed | wave | 0 assumed | `coming_from` *(pending expert confirm)* | `ComingFromToUV` | paired with `pswh` → `(u_pswell, v_pswell)` | 3 |
| `ocs.npy`       | Ocean current speed                | float16       | ocean | —      | —              | (pass-through)      | scalar                                     | 3    |
| `ocd.npy`       | Ocean current direction            | int8 (1–16)   | ocean | 1 (same as `wad`) | `going_to` *(pending §2.7 ocean-grid verification)* | `GoingToToUV` | paired with `ocs` → `(u_current, v_current)`; no `ocd_come_from` sibling exists | 3    |

---

## 11. References

Ordered by usefulness for this project.

**DMD:**
- Kutz, Brunton, Brunton, Proctor (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.* SIAM. Book + code.
- Tu, Rowley, Luchtenburg, Brunton, Kutz (2014). "On dynamic mode decomposition: theory and applications." *J. Computational Dynamics.*
- Sashidhar & Kutz (2022). "Bagging, optimized dynamic mode decomposition." *Phil. Trans. R. Soc. A.* BOP-DMD with UQ.
- Brunton, Brunton, Proctor, Kaiser, Kutz (2017). "Chaos as an intermittently forced linear system." *Nature Communications.* Hankel-DMD / HAVOK.
- Ichinaga et al. (2024). "PyDMD: A Python package for robust dynamic mode decomposition." *JMLR.*

**Tensor decompositions:**
- Kolda & Bader (2009). "Tensor decompositions and applications." *SIAM Review.* Tucker/HOSVD reference.
- Kossaifi, Panagakis, Anandkumar, Pantic (2019). "TensorLy: tensor learning in Python." *JMLR.*

**Stochastic weather generators:**
- Wilks & Wilby (1999). "The weather generation game." *Progress in Physical Geography.*
- Ailliot, Allard, Monbet, Naveau (2015). "Stochastic weather generators: an overview." *J. Soc. Fr. Stat.*
- Rasmussen (2013). "Multisite precipitation generation using a latent autoregressive model." *WRR.*

**Alternatives for escalation:**
- Lindgren, Rue, Lindström (2011). "An explicit link between Gaussian fields and Gaussian Markov random fields: the SPDE approach." *JRSS-B.* Alternative if DMD-OU has unfixable calibration problems; also the framework for SPDE-GMRF local layer.
- Coles (2001). *An Introduction to Statistical Modeling of Extreme Values.* GPD-tail escalation.

**Directional statistics:**
- Mardia & Jupp (2000). *Directional Statistics.*

**sklearn:**
- Buitinck et al. (2013). "API design for machine learning software: experiences from the scikit-learn project." arXiv:1309.0238.

**Randomized SVD:**
- Halko, Martinsson, Tropp (2011). "Finding structure with randomness." *SIAM Review.*

---

## 12. What "done" looks like

A tagged v0.1.0 release providing:

1. `poetry install` produces a working environment on Python 3.14. `poetry run pytest` passes.
2. A working `notebooks/test_pipeline.ipynb` that produces a fitted `WeatherModel` on 2022–2025 data and a validation PDF against 2026 held-out data end-to-end.
3. The public API from §8.4, with docstrings and full type hints.
4. Green CI on Python 3.14: `black --check src/ tests/`, `isort --check-only src/ tests/`, `mypy --strict src/`, `pytest --cov=src` (coverage ≥ 85%).
5. Unit test coverage:
   - `tests/core/test_io.py::test_wsd_compass_codes` — known code → degree pairs (§2.3).
   - `tests/core/test_io.py::test_wad_compass_codes` — 1-indexed origin.
   - `tests/train/test_directional.py::test_cardinal_signs` (ComingFromToUV) — four `θ_from ∈ {0°, 90°, 180°, 270°}` cases from §2.4 (coming-from table). **Load-bearing.**
   - `tests/train/test_directional.py::test_uv_roundtrip` (ComingFromToUV) — `(speed, θ_from) → (u, v) → (speed', θ_from')` reproduces input within `1e-10`.
   - `tests/train/test_directional.py::test_inverse_transform_returns_native_convention` (ComingFromToUV) — inversion reconstructs `θ_from` (not `θ_to`), matching archive.
   - `tests/train/test_directional.py::test_pass_through_non_paired_vars` (ComingFromToUV) — non-directional variables and pairs owned by GoingToToUV pass through untouched.
   - `tests/train/test_directional.py::test_cardinal_signs` (GoingToToUV) — four `θ_to ∈ {0°, 90°, 180°, 270°}` cases from §2.4 (going-to table). **Load-bearing.**
   - `tests/train/test_directional.py::test_uv_roundtrip` (GoingToToUV).
   - `tests/train/test_directional.py::test_inverse_transform_returns_native_convention` (GoingToToUV) — inversion reconstructs `θ_to`.
   - `tests/train/test_directional.py::test_pass_through_non_paired_vars` (GoingToToUV).
   - `tests/core/test_io.py::test_wad_and_wad_come_from_consistency_check` — the 180° consistency assertion on a real snapshot.
   - `tests/train/test_pipeline_composition.py::test_transformers_commute_on_disjoint_pairs` — `ComingFromToUV(...) → GoingToToUV(...)` and `GoingToToUV(...) → ComingFromToUV(...)` produce identical output when their pair lists are disjoint (they always should be).
   - `tests/core/test_io.py::test_snapshot_path_zero_padded`.
   - `tests/core/test_io.py::test_utc_timezone_required` — naive datetime raises `ValueError`.
   - `tests/train/test_climatology.py::test_harmonic_roundtrip` — fit → eval reproduces training data at fit points.
   - `tests/train/test_climatology.py::test_trend_recovery` — synthetic linear trend recovered within 5%.
   - `tests/train/test_climatology.py::test_trend_uncorrelated_with_c0` — verify that centering at `t_ref` decorrelates `c_lin` and `c₀`.
   - `tests/train/test_copula.py::test_forward_inverse_roundtrip`.
   - `tests/train/test_copula.py::test_gaussian_output_distribution` — KS test after forward transform.
   - `tests/train/test_tucker.py::test_tucker_fit_roundtrip` — analytic rank-`r` tensor recovered exactly at rank `r`.
   - `tests/train/test_tucker.py::test_tensorly_return_shape` — confirms the tuple-unpacking pattern in the installed tensorly version; fails loudly on drift.
   - `tests/train/test_dmd.py::test_analytic_traveling_wave` — synthetic `A cos(kx - ωt)` recovered as complex-conjugate pair with correct `ω`, tested against each variant (`"exact"`, `"bop"`, `"hankel"`).
   - `tests/train/test_dmd.py::test_bop_enforces_conjugate_pairs` — output modes come as conjugate pairs after BOP-DMD with `eig_constraints={"conjugate_pairs"}`.
   - `tests/train/test_dmd.py::test_pydmd_snapshot_orientation` — verifies that `.T` is applied in the right place.
   - `tests/train/test_dmd.py::test_mode_ou_parameter_recovery` — synthetic OU trace with known `(θ, σ)` recovered within 5%.
   - **`tests/train/test_local_residual.py::test_variance_ratio_diagnostic`** — synthetic Z with known coherent + local components; recovered `variance_ratio` matches expected.
   - **`tests/train/test_local_residual.py::test_active_mask_below_floor_zeroed`** — cells with variance ratio below floor get zeroed OU parameters.
   - **`tests/train/test_local_residual.py::test_vectorized_ar1_matches_per_cell_loop`** — the vectorized fit produces identical results to a per-cell scalar OU fit.
   - **`tests/train/test_local_residual.py::test_equilibrium_variance_matches`** — sampling from the fitted OU at equilibrium reproduces the residual's per-cell variance.
   - `tests/weather/test_model.py::test_fit_save_load_sample` — end-to-end pipeline on a small synthetic archive.
   - `tests/weather/test_model.py::test_transform_inverse_roundtrip`.
   - `tests/weather/test_model.py::test_from_components_matches_fit`.
   - `tests/weather/test_model.py::test_sklearn_clone` — `sklearn.base.clone(model)` produces a fresh unfitted copy.
   - `tests/weather/test_simulate.py::test_batch_matches_single_point` — `sample_batch(model, [lat], [lon], [t], var)[0] == sample(model, lat, lon, t, var)` given the same seed.
   - `tests/weather/test_simulate.py::test_seed_reproducibility`.
   - **`tests/weather/test_simulate.py::test_short_lag_variogram_nonzero`** — samples show non-zero variance at 25 km lag (short-lag validation criterion §6.4).
   - `tests/test_import_boundaries.py::test_train_does_not_import_weather` — AST-based enforcement of §8.3.
6. Grid convention verification tests, required only if the corresponding tier is enabled:
   - `tests/core/test_grid_wave.py::test_row0_is_south`.
   - `tests/core/test_grid_ocean.py::test_orientation` — required before enabling `sst`/`ocs`/`ocd`. Evidence comes from `notebooks/verify_ocean_grid_orientation.ipynb` (§2.7 TASK — not yet started); do not write this test (or any ocean-grid pipeline code) before that notebook exists and passes its ≥85%/≤60% thresholds.
   - `tests/train/test_directional.py::test_pwd_convention_expert_verified` — required before enabling `pwd` (tier 2).
   - `tests/train/test_directional.py::test_pswd_convention_expert_verified` — required before enabling `pswd` (tier 3).
7. Integration tests (marked `@pytest.mark.integration`):
   - `tests/weather/test_integration.py::test_end_to_end_marginal_fit` — trains on 3 months of synthetic archive data, simulates 1 month of overlapping period, asserts per-cell mean and std within 10% of held-out truth.
   - `tests/weather/test_integration.py::test_extrapolation_no_drift` — trains on 3 months, extrapolates 6 months past the training end (no ground truth), asserts per-cell simulated mean over the extrapolation window is within 3σ of the seasonal expectation from the fitted climatology. Guards against runaway DMD modes (`Re(lam_cont) > 0`) that would produce unbounded growth on long extrapolation.
   - `tests/weather/test_integration.py::test_extrapolation_output_validity` — trains on 3 months, extrapolates 6 months, asserts all outputs are finite, all directional variables in `[0, 360)`, all magnitude variables ≥ 0. §6.B.13 in test form.
   - `tests/weather/test_integration.py::test_two_seed_stability` — same as above but two seeds; asserts per-cell mean of the difference has zero-mean Gaussian distribution (paired samples). §6.B.14 in test form.
   - **`tests/weather/test_integration.py::test_hurricane_season_reproduction`** — NEW, §6.D.18. Trains on a synthetic archive covering ≥1 full Aug-Oct window, simulates the same window, asserts the storm-signature detector's feature count/intensity/translation-speed distribution on simulated output is statistically consistent with the same detector run on the real synthetic-archive input.
   - **`tests/weather/test_integration.py::test_hurricane_season_extrapolation_structural`** — NEW, §6.D.19. Same synthetic setup, extrapolates past training end through a synthetic Aug-Oct window, asserts at least one storm-signature-qualifying feature is detected — a capability check, not a statistical-match check.
8. Documentation:
   - `README.md` — install, minimal example, link to `docs/`.
   - `docs/model_design.md` — this document.
9. `CHANGELOG.md` following Keep a Changelog format, with v0.1.0 entry describing the initial release, and a note that this design is v2 of the handoff (v1 preserved in git history for reference).
