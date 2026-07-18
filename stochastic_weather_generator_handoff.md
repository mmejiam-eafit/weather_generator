# Stochastic Weather Generator — Design Handoff

**Package:** new standalone Python package (working name `weather-synth` — rename freely).

**Audience:** coding agent bootstrapping the repo from scratch. This document is self-contained: every convention, formula, and restriction the agent needs is stated here, not by reference.

**One-line summary:** train a stochastic model on a raw 3-hourly `.npy` archive of NOAA GFS-Wave / WAVEWATCH III + GFS wind fields, and provide plausible synthetic samples of any variable at any continuous `(lat, lon, datetime)` when real data isn't available.

**Approach chosen:** Tucker compression → Dynamic Mode Decomposition (DMD) → Ornstein–Uhlenbeck (OU) processes on residual mode amplitudes, with per-cell Gaussian-copula marginals and harmonic climatology removal. See §3 for rationale and rejected alternatives.

**Language:** Python only, 3.11+. Batched-first API design (§1, §8). If a downstream consumer eventually shows sub-millisecond point queries are needed and cannot be batched, a C++ query library can be added; do not build one preemptively.

---

## Table of contents

- §1 — Goal and API sketch
- §2 — Data facts (grids, dtypes, decoding, conventions, NaN policy)
- §3 — Design decisions and rationale
- §4 — Architecture overview
- §5 — Algorithm, step by step
- §6 — Validation battery
- §7 — Persisted model artifacts
- §8 — Repo bootstrap (layout, packaging, API surface, CLI)
- §9 — Gotchas and implementation notes
- §10 — Variable table
- §11 — References
- §12 — What "done" looks like

---

## 1. Goal

Provide a Python package with a batched-first API. The convenience single-point call exists for exploration and notebook use; the batched call is what consumers should reach for at scale.

```python
from weather_synth import WeatherModel
import numpy as np
from datetime import datetime, timezone

model = WeatherModel.load("./models/v1")

# Single-point convenience — internally wraps sample_batch.
value: float = model.sample(
    lat=35.5, lon=-40.2,
    t=datetime(2025, 6, 15, 9, tzinfo=timezone.utc),
    variable="sigwh",
    seed=42,
)

# Batched — the actually-fast path.
values: np.ndarray = model.sample_batch(
    lats=np.array([...]),                      # (N,)
    lons=np.array([...]),                      # (N,)
    times=np.array([...], dtype="datetime64[s]"),
    variables=["sigwh", "was", "wsp"],
    seed=42,
)                                              # (N, 3)

# Full-field simulation on the native wave grid.
import xarray as xr
ds: xr.Dataset = model.simulate_field(
    t_start=datetime(2025, 6, 1, tzinfo=timezone.utc),
    t_end=datetime(2025, 7, 1, tzinfo=timezone.utc),
    variables=["sigwh", "was"],
    seed=42,
)
```

**Correctness criteria:**

1. **Marginals match.** Per-cell empirical distributions of samples reproduce the training data's per-cell empirical distributions. Bounded variables (Hs ≥ 0) stay bounded; skewed ones stay skewed.
2. **Space-time covariance is preserved.** Samples at nearby `(lat, lon, t)` are correlated the way real weather is. Storms look like storms.
3. **Propagation is preserved.** Swell trains and stormfronts advect at physically sensible speeds and don't dissolve prematurely.
4. **Cross-variable coupling is preserved.** Wind speed and wave height covary correctly; wind direction and wave direction covary correctly.
5. **Batched throughput is fast enough for realistic consumers.** Target ~10 µs per point when batched to 10⁴+ points (one BLAS call). Single-point unbatched calls are ~100 µs–1 ms — acceptable for interactive use, not for tight inner loops. Consumers with tight loops should batch. If a real consumer profiles the query path and finds batching insufficient, that's the signal to add C++ — not before.

**Non-goals.** Not trying to forecast better than GFS. Not modelling climate change or long-term trends (assume stationarity within the training window). Not matching individual historical storms — matching the *statistics* of storms.

**Why 3-hourly, not daily.** Some point-query consumers (maritime routing in particular) reason about weather that changes on ~6–12 h timescales — storm passages, diurnal wind ramps near coasts. A daily-averaged model can't produce that resolution. Train on the raw 3h archive.

---

## 2. Data facts

Everything in this section is load-bearing. Any deviation propagates silently through the model and shows up as bad samples that pass unit tests but fail validation.

### 2.1 File layout

Raw 3-hourly archive is mirrored locally at:

```
./data/<YYYY>/<MM>/<DD>/<HH>/<variable>.npy
```

with `HH ∈ {00, 03, 06, 09, 12, 15, 18, 21}` (zero-padded, UTC). All paths must resolve relative to a configurable `--data-dir` CLI argument, defaulting to `./data`. Never hardcode absolute paths.

Path construction in Python:

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

**Row 0 = south is critical.** Most gridded earth-science datasets follow the opposite convention (row 0 = north pole, latitude decreasing with row). These grids do not. Any code that geo-references a grid must apply the formulas above. See §2.7 for how this was established empirically and how to re-verify it for the ocean grid.

### 2.3 On-disk dtypes and decoding

Two dtype conventions in the archive:

- **Continuous fields** (`sigwh`, `wsh`, `wsp`, `pwh`, `pwp`, `pswh`, `pswp`, `was`, `sst`, `ocs`): stored as `float16` on disk.
- **Direction fields** (`wsd`, `pwd`, `pswd`, `wad`, `wad_come_from`, `ocd`): stored as `int8` 16-point-compass codes (22.5° steps), not raw degrees.

**Immediately on load, upcast everything to `float64`.** Reasons: `float16` overflows in arithmetic at surprisingly small values (`(10.0)² = 100` is fine, but `(100.0)²` overflows), and `int8` degrees don't exist as a data type. `float16` and `int8` are storage formats, never compute formats in this pipeline.

**Compass code decoding (values differ per field — this trips people up):**

| Field                | Observed code range | Origin (North = 0°) | Decode formula                  |
|----------------------|---------------------|---------------------|---------------------------------|
| `wsd`                | 0 – 15              | code `0`            | `degrees = code * 22.5`         |
| `pwd`                | 0 – 15              | code `0`            | `degrees = code * 22.5`         |
| `pswd` (unverified)  | 0 – 15 (assumed)    | code `0` (assumed)  | `degrees = code * 22.5` (verify) |
| `wad`                | 1 – 16              | code `1`            | `degrees = (code - 1) * 22.5`   |
| `wad_come_from`      | (ignored — see §2.4)| —                   | —                               |
| `ocd` (unverified)   | (verify)            | (verify)            | (verify)                        |

Canonical implementation:

```python
def decode_compass_code(codes: np.ndarray, origin_code: int) -> np.ndarray:
    """Decode int8 16-point compass codes to degrees (float64, [0, 360)).

    Parameters
    ----------
    codes : (M, N) int8
        Compass codes as loaded from disk.
    origin_code : int
        Value of `codes` that represents North (0°). Typically 0 or 1.

    Returns
    -------
    degrees : (M, N) float64
        Meteorological "coming from" convention, CW from North, in [0, 360).
    """
    return ((codes.astype(np.int32) - origin_code) * 22.5).astype(np.float64) % 360.0
```

Test cases (must pass):
- `decode_compass_code(np.array([[0]], dtype=np.int8), origin_code=0)` → `[[0.0]]` (North).
- `decode_compass_code(np.array([[4]], dtype=np.int8), origin_code=0)` → `[[90.0]]` (East).
- `decode_compass_code(np.array([[1]], dtype=np.int8), origin_code=1)` → `[[0.0]]` (North, `wad` convention).
- `decode_compass_code(np.array([[5]], dtype=np.int8), origin_code=1)` → `[[90.0]]` (East, `wad` convention).

### 2.4 Direction convention

**All decoded directions are meteorological "coming from", measured clockwise from geographic North.**

- `0°` = wind/waves arriving *from* the North (blowing/moving *toward* the South).
- `90°` = arriving from the East (moving toward the West).
- `180°` = arriving from the South.
- `270°` = arriving from the West.

This is the standard GFS convention, applied after decoding the compass code.

The `wad_come_from` variable appears to duplicate `wad` semantically and is undocumented upstream. **Ignore it.** Verify this assumption early — if the two ever disagree for the same timestep, escalate before proceeding.

For any modeling that requires `(u, v)` vector components (necessary for stochastic modeling of vector fields; angle-space interpolation is broken at the 0°/360° discontinuity):

```
theta_going_to = (theta_coming_from + 180.0) mod 360.0
theta_rad      = theta_going_to * pi / 180.0
u = speed * sin(theta_rad)      # eastward component  (positive east, negative west)
v = speed * cos(theta_rad)      # northward component (positive north, negative south)
```

**Verification test cases (must pass — this catches sign flips before they propagate):**

| θ_from | expected `(u, v)`         | Meaning                       |
|--------|---------------------------|-------------------------------|
| 0°     | `( 0, -speed )`           | Wind from N → blowing south   |
| 90°    | `(-speed,  0 )`           | Wind from E → blowing west    |
| 180°   | `( 0, +speed )`           | Wind from S → blowing north   |
| 270°   | `(+speed,  0 )`           | Wind from W → blowing east    |

Inverse at query time:

```
speed         = sqrt(u² + v²)
theta_going_to = atan2(u, v) mod 360.0     # note arg order: (sin, cos), not (y, x)
theta_coming_from = (theta_going_to + 180) mod 360.0
```

The `atan2(u, v)` arg order (sin first, cos second) is the meteorological convention and is deliberate. `atan2(v, u)` would give the standard mathematical angle CCW from East, which is *not* what any consumer expects. Do not swap.

### 2.5 NaN policy

- **Wave grid** — `float` fields (`sigwh`, `wsh`, `wsp`, `pwh`, `pwp`, `pswh`, `pswp`) carry NaN for land and persistent ice cover. This is the authoritative land mask for the wave grid.
- **Wind grid** — has *no* NaN. Winds are defined over land in the GFS product. Cells over land carry real wind values, not sentinels. Do not model wind over land in the stochastic generator — mask via a Natural Earth land polygon rasterized to the wind grid.
- **Ocean grid** — `sst` almost certainly carries NaN for land; `ocs`/`ocd` behavior needs verification.
- **Direction fields** — `int8`, no NaN. Masking always comes from the paired float field: mask a direction only where its paired magnitude is NaN or below a physical noise floor (e.g. `wsh < 0.05 m`).

**Every accumulation guarded with `np.isnan`.** NaN in → NaN through. Never coerce a NaN to zero: a real 0 (calm sea) is a physically meaningful value that must not be confused with land.

For the stochastic model: fit only on ocean cells. Build one boolean `ocean_mask` per grid (see §5.1), model over the ravelled ocean-cell vector, project back to `(H, W)` for output. Ocean cells must be non-NaN in *all* modelled variables at *all* training snapshots — an ocean cell that becomes ice-covered mid-training is not a stable-ocean cell.

### 2.6 Longitude wraparound

Both grids span `[0°, 359.75° E]` — no polar cap padding, no dateline duplication. `col=1439` and `col=0` are adjacent on the sphere but at opposite ends of the array. Bilinear interpolation across this boundary produces NaN unless the array is padded periodically first:

```python
def pad_lon(field: np.ndarray, pad: int = 2) -> np.ndarray:
    """Periodically pad the last axis by `pad` columns on each side."""
    return np.concatenate([field[..., -pad:], field, field[..., :pad]], axis=-1)
```

Any operation that interpolates in longitude (spatial reprojection, bilinear query at arbitrary `lon`) must first pad. Any operation that decomposes/reconstructs the full spatial field (Tucker, DMD) should treat the padded columns as diagnostic-only — don't count them toward reconstruction error.

### 2.7 How grid orientation was established (and how to verify it)

The row-0-is-south convention was originally documented as the opposite (row 0 = north, standard for most gridded datasets) and turned out to be wrong. The correction came from an empirical study:

- **Wave grid.** Correlate the real NaN pattern in `sigwh` (which reliably marks land + persistent ice) against a rasterized coastline from Natural Earth, across multiple days and every plausible row offset and row order. The correct convention appears as a sharp single peak in agreement (~92.6% match); every other option is much worse. Verified visually — continents align with coastlines exactly at that offset.
- **Wind grid.** Has no NaN mask to correlate against. Confirmed two ways: (a) visual landmark displacement matches the wave grid's convention; (b) winter storm wind maxima in `was` land at latitudes consistent with real North Pacific / North Atlantic cyclones only under the south-first convention (the opposite convention places them in implausible Southern Ocean latitudes).

**The ocean grid orientation has not been verified. Do it before including SST or currents in the model.** Method:

1. Pick 30 arbitrary days from the archive.
2. Load `sst.npy` for each.
3. For every combination of (row 0 at south vs. north) × (rows increase vs. decrease), correlate the NaN pattern in `sst` against a Natural Earth land polygon rasterized at 0.125° resolution.
4. The correct convention should show ≥ 85% agreement; alternatives should be ≤ 60%.
5. Independently confirm with `sst` warm-pool location — the Western Pacific warm pool should land at the equator ± 10°, not at 60°S.

Encode the result in `config.py` as `OCEAN_GRID_LAT_AT_ROW0` and `OCEAN_GRID_LAT_STEP_DEG` (sign encodes direction). Add the verification as an integration test that runs once when SST is enabled and fails loudly if the on-disk convention ever changes upstream.

### 2.8 Python conventions for this repo

**Runtime dependencies:** `numpy ≥ 2.0`, `scipy ≥ 1.14`, `scikit-learn ≥ 1.5`, `xarray ≥ 2024.10`, `pandas ≥ 2.2`.

**Dev dependencies:** `pytest ≥ 8`, `pytest-cov`, `pytest-xdist`, `matplotlib ≥ 3.9`, `ruff ≥ 0.7`, `mypy ≥ 1.13`.

**Packaging:** `pyproject.toml` with `hatchling` backend, `src/` layout. Version from git tags via `hatch-vcs`.

**Coding conventions:**

- Type hints on every function signature (public and private). `from __future__ import annotations` at the top of every module. `mypy --strict` in CI, treat warnings as errors.
- Docstrings on every public function and class. Numpy-style. Enforced via `ruff` rule set `D`.
- Naming (PEP 8): `snake_case` for functions/methods/variables/modules, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for module-level constants, `_` prefix for private names.
- No mutable default arguments. Ever. Ruff catches this.
- Explicit dtype at every `np.zeros` / `np.empty` / `np.array` — never rely on numpy defaults. This one bit us elsewhere.
- No `numpy.float16` past the I/O boundary. It overflows silently at modest values.
- No hidden global state. Pass configuration as arguments; no module-level mutable globals.
- `@dataclass(frozen=True, slots=True)` for structured data over `dict`. Frozen so equality/hashing work; slots to catch typos.
- `pathlib.Path` everywhere; never string paths.
- All timestamps `datetime` with `tzinfo=timezone.utc`. Naive datetimes raise `ValueError` immediately at the API boundary.
- RNG discipline: every function that generates randomness takes `rng: np.random.Generator | None = None` or a `seed: int | None = None`. Never touch the global `np.random`.
- Parallelism: rely on BLAS threading for numeric loops. When an explicit CPU-bound loop over independent items is unavoidable, use `joblib.Parallel(n_jobs=-1, backend="loky")`. No `multiprocessing.Pool`, no bare `concurrent.futures` for numeric work.
- No implicit numeric conversions. All casts explicit via `.astype(...)` or `np.asarray(..., dtype=...)`.

**Testing:** `pytest`, one test file per source module (`test_<module>.py`), fixtures in `conftest.py`. Coverage target ≥ 85% for `src/weather_synth/`, 100% for the public API. Use `pytest-xdist` for parallel execution.

**CI:** single GitHub Actions workflow, matrix over Python 3.11 / 3.12 / 3.13 on ubuntu-latest. Steps: `ruff check`, `ruff format --check`, `mypy --strict src/`, `pytest --cov`. That's the whole gate.

---

## 3. Design decisions and rationale

### 3.1 Train on raw 3h data, not daily averages

Point queries at arbitrary `datetime` need sub-daily variability. A daily energy average collapses it by design. Use the raw archive.

### 3.2 Rejected: full 3D Gaussian process / kriging

Would require designing a `(lat, lon, time, variable)` covariance kernel; inference cost scales poorly with data volume; doesn't propagate advective phenomena. Reconsider only via SPDE-GMRF (Lindgren–Rue–Lindström 2011) if DMD-OU shows unfixable calibration problems.

### 3.3 Rejected: plain real EOF + AR

Real-valued EOFs represent traveling waves as pairs of standing modes in quadrature, obscuring propagation. AR on the resulting PCs technically works but has no principled way to distinguish coherent propagation from noise. DMD subsumes this cleanly (§3.4).

### 3.4 Chosen: Tucker → DMD → OU

- **Tucker (HOSVD)** compresses the spatial modes and the variable mode. Treating lat, lon, and variable as separate tensor modes gives interpretable factor matrices and avoids one huge SVD.
- **DMD** on the compressed core extracts the linear deterministic skeleton. Each mode carries its own eigenvalue `λ_k = e^{(σ_k + iω_k)Δt}`, encoding growth rate and frequency. Traveling swell appears as a complex-conjugate mode pair with `Im(λ)` set by its propagation frequency, without special-casing.
- **OU on residual amplitudes** models stochastic wobble around the deterministic skeleton. Closed-form fit, closed-form simulation, closed-form transition density — enables cheap continuous-time query without a full grid simulation.

Escalation paths if the base model underperforms in validation (§6):

- Insufficient high-frequency variance → **Hankel-DMD** (time-delay embed the state before DMD). Cheap to add.
- Insufficient joint tail dependence → **t-copula** (single extra df parameter) or **GPD tail splicing** per cell (Coles 2001).
- Insufficient cross-variable coupling → raise `r_var` truncation, or add explicit **Linear Model of Coregionalization**.

### 3.5 Non-Gaussian marginals via Gaussian copula

DMD and OU are linear-Gaussian; weather variables (Hs, wind speed, wave heights) are skewed and bounded. Interpose an empirical-CDF Gaussian copula per cell:

- **Forward** (training): rank each cell's time series → uniform on `(0,1)` → apply `Φ⁻¹` → standard normal. DMD/OU operate on the Gaussianized field.
- **Inverse** (sampling): DMD/OU emit standard normal → apply `Φ` → uniform → look up in the empirical CDF → original units.

This handles positivity and boundedness implicitly (the inverse never produces values outside the training range) and preserves per-cell shape.

Known limitation: Gaussian copulas have zero joint tail dependence. Simulated storm co-occurrences across cells will be too rare and too mild for a bulk-Gaussian model. Fix later (see §3.4 escalation) once validation quantifies how bad this is.

### 3.6 Cross-grid handling

Three grids of different sizes cannot be jointly decomposed without reconciliation. Two options:

- **Option A (preferred for v1):** reproject wind and ocean grids onto the wave grid (bilinear, clipping to `lat ∈ [−75°, +80°]`). Loses wind information at high latitudes, but those are outside routing range anyway. Ocean → wave downsampling (0.125° → 0.25°) is straightforward.
- **Option B (later):** fit three separate Tucker + DMD + OU models, one per grid. Couple at the OU level via joint innovations. More flexible; harder to fit and validate.

Recommend A for v1. Design the reprojection as a swappable layer.

### 3.7 Directional variables always via `(u, v)`

Never model `θ` directly. For every `(speed, direction)` pair in the archive:

1. Decode direction from int8 compass code (§2.3).
2. Convert "coming from" → "going to" and decompose to `(u, v)` per §2.4.
3. Model `(u, v)` as ordinary real fields alongside the scalars.
4. At sample time, `speed = hypot(u, v)`, `θ_to = atan2(u, v) mod 360°`, `θ_from = (θ_to + 180°) mod 360°`.

For directions without a natural paired magnitude, use `(cos θ, sin θ)` on the unit circle with a unit-norm constraint at reconstruction. In this archive every direction has a paired magnitude, so this branch is unlikely to fire.

### 3.8 Which variables to model — tiered

Ship in tiers, each adding real value on its own:

**Tier 1 (v0.1.0):** `sigwh`, `wsh`, `wsp` (wave-grid scalars); `was` (wind speed); derived `(u_wind, v_wind)` and `(u_wsea, v_wsea)`. Six modelled fields after (u, v) expansion. Covers most maritime routing needs.

**Tier 2 (v0.2.0):** peak wave `(pwh, pwp, pwd)` — adds `(u_peak, v_peak)`; `sst`. Requires the ocean-grid convention verification (§2.7).

**Tier 3 (v0.3.0+):** primary swell `(pswh, pswp, pswd)`; ocean currents `(ocs, ocd)`. Requires `pswd`/`ocd` compass convention verification. Primary swell may prove redundant with the `sqrt(sigwh² - wsh²)` residual — decide based on tier 1's residual QQ.

`wad_come_from` is ignored throughout (§2.4).

---

## 4. Architecture overview

```
                          TRAINING (offline)
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │   ./data/YYYY/MM/DD/HH/*.npy                                       │
   │              │                                                     │
   │              ▼                                                     │
   │   [ingest]  Read archive, decode int8 → degrees, upcast to float64 │
   │              │                                                     │
   │              ▼                                                     │
   │   [reproject] Wind/ocean grids → wave grid (bilinear, u/v-aware)   │
   │              │                                                     │
   │              ▼                                                     │
   │   [directional] (speed, θ_from) → (u, v)                           │
   │              │                                                     │
   │              ▼                                                     │
   │   [climatology] Fit harmonic mean per cell per variable; subtract  │
   │              │                                                     │
   │              ▼                                                     │
   │   [standardize] Divide by per-cell temporal std                    │
   │              │                                                     │
   │              ▼                                                     │
   │   [Gaussianize] Empirical CDF → Φ⁻¹ per cell; store inverse LUT    │
   │              │                                                     │
   │              ▼                                                     │
   │   [Tucker] HOSVD on ocean-cell mode + variable mode                │
   │              │                                                     │
   │              ▼                                                     │
   │   [DMD] Exact DMD on core time series                              │
   │              │                                                     │
   │              ▼                                                     │
   │   [OU fit] (θ, σ) per DMD mode from residual amplitudes            │
   │              │                                                     │
   │              ▼                                                     │
   │   Model artifact dir: climatology, LUT, U_ocean, U_var,            │
   │                       Φ, Λ, b₀, (θ, σ) per mode                    │
   └────────────────────────────────────────────────────────────────────┘

                         QUERY (online, Python — batched preferred)
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │   (lats[], lons[], times[], variables[])                           │
   │              │                                                     │
   │              ▼                                                     │
   │   Evolve OU + DMD to requested times (closed-form)                 │
   │              │                                                     │
   │              ▼                                                     │
   │   Reconstruct compressed state → Tucker uncompress → Z(H, W, V)    │
   │              │                                                     │
   │              ▼                                                     │
   │   Bilinear interpolate at (lat, lon) [padded for lon wraparound]   │
   │              │                                                     │
   │              ▼                                                     │
   │   Invert Gaussianization (LUT) + standardization + climatology     │
   │              │                                                     │
   │              ▼                                                     │
   │   Convert (u, v) → (speed, direction) if directional               │
   │              │                                                     │
   │              ▼                                                     │
   │   values[]                                                         │
   └────────────────────────────────────────────────────────────────────┘
```

---

## 5. Algorithm, step by step

Notation:
- `T` — training snapshots (~2920 per year at 3h cadence).
- `H, W = 621, 1440` — wave grid dimensions. All modeling on the wave grid post-reprojection.
- `N_ocean` — ocean cells after masking; expect roughly 60% of `H·W`, so ~5.4 × 10⁵.
- `V` — jointly modelled variables. Tier 1: ~6 after (u, v) expansion.
- `r_ocean`, `r_var`, `r_dmd` — truncation ranks.
- `Δt = 10800 s` (3 hours).

### 5.1 Ingest

```python
def enumerate_snapshots(data_dir, t_start, t_end):
    t = t_start
    while t < t_end:
        yield t, snapshot_path(data_dir, t)   # see §2.1
        t += timedelta(hours=3)
```

Load each `.npy`, upcast to `float64` immediately (`arr.astype(np.float64)`), decode int8 direction fields via `decode_compass_code` (§2.3).

**Gap handling:** missing file → log to stderr, mark that snapshot NaN for that variable. Contiguous gap of 24 h+ (8 timesteps) → abort with a clear error; the OU fit assumes near-continuous sampling. Gap 6–24 h → warn but continue.

**Memory:** one year of the full-state tensor at `float64` is ~40 GB per variable at global 0.25°. Use `np.memmap` for on-disk backing, or chunk time-blocks (e.g. 30 days at a time) and stream through Tucker's mode-2 SVD.

**Ocean mask construction:** for each grid, take the intersection of `not isnan` across all training snapshots and all modelled variables on that grid. Add a Natural Earth land polygon rasterized to the wind grid (since wind has no intrinsic mask). Store as `ocean_mask_wave.npy` — a `(621, 1440)` bool array. Model over `mask.sum()` ocean cells; use `mask.nonzero()` to scatter back to 2D at output time.

### 5.2 Reproject wind and ocean grids to wave grid

Wind (721 × 1440) and ocean (1121 × 2880) fields must be resampled to the wave grid (621 × 1440, `lat ∈ [−75°, +80°]`):

```python
from scipy.interpolate import RegularGridInterpolator

def to_wave_grid(field, src_lats, src_lons, wave_lats, wave_lons):
    field_padded = pad_lon(field, pad=2)                    # §2.6
    src_lons_padded = np.concatenate([
        src_lons[-2:] - 360.0, src_lons, src_lons[:2] + 360.0
    ])
    interp = RegularGridInterpolator(
        (src_lats, src_lons_padded), field_padded,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    LAT, LON = np.meshgrid(wave_lats, wave_lons, indexing="ij")
    return interp((LAT, LON))
```

**Directional variables must be reprojected in `(u, v)` space, not as `θ`.** Angle-space bilinear interpolation across the 0°/360° discontinuity produces garbage — a wind of `359°` and one of `1°` will average to `180°`, the exact opposite of correct. Decompose to `(u, v)` first (§2.4), reproject each component independently, recombine at query time if needed.

### 5.3 Directional decomposition

For each `(speed, direction)` pair, replace with `(u, v)`. Formulas per §2.4. Pairs in the archive:

- `(was, wad)` → `(u_wind, v_wind)`
- `(wsh, wsd)` → `(u_wsea, v_wsea)`
- `(pwh, pwd)` → `(u_peak, v_peak)`
- `(pswh, pswd)` → `(u_pswell, v_pswell)`
- `(ocs, ocd)` → `(u_current, v_current)`

Non-directional variables (`sigwh`, `wsp`, `pwp`, `pswp`, `sst`) pass through unchanged.

### 5.4 Climatology and Gaussianization

Fit per ocean cell per variable:

```
μ(t) = c₀
     + c₁·cos(ω_y·t) + c₂·sin(ω_y·t)
     + c₃·cos(2·ω_y·t) + c₄·sin(2·ω_y·t)
     [+ c₅·cos(ω_d·t) + c₆·sin(ω_d·t)]      # diurnal, SST and wind only
```

with `ω_y = 2π/365.25 days⁻¹` and `ω_d = 2π/day⁻¹`. Fit by vectorized OLS; store coefficients as `(n_terms, N_ocean, V)`.

**Diurnal terms only for `sst`, `u_wind`, `v_wind`.** Waves don't have a diurnal cycle at 0.25° global resolution.

**Reference epoch matters.** Store `t_train_start` as an ISO 8601 UTC string in metadata. At query time, compute `t_since_epoch_days = (t_query - t_train_start).total_seconds() / 86400.0` and pass to `cos(ω_y · t_since_epoch_days)`. Do not reset day-of-year at Jan 1 — periodicity comes from the cosines, not from bookkeeping.

Standardize per cell: divide by per-cell temporal std of the anomaly, store `σ_cell`.

Gaussianize per cell:

```python
from scipy.stats import norm, rankdata

def to_gaussian(anom):
    T = anom.shape[0]
    ranks = rankdata(anom, axis=0)
    u = (ranks - 0.5) / T
    z = norm.ppf(u)
    sorted_vals = np.sort(anom, axis=0)     # for inverse LUT
    return z, sorted_vals
```

**LUT storage:** full sorted values are `T × N_ocean × V × 8 bytes`. For 2 years × 5.4 × 10⁵ ocean cells × 10 vars ≈ 250 GB — infeasible. Downsample to 500 quantile bins per cell → ~5 GB, marginal-fidelity loss negligible.

### 5.5 Tucker (HOSVD) compression

Modes: `(time, ocean_cell, variable)`. Compress the last two, keep time full-rank (DMD needs it).

```python
from sklearn.utils.extmath import randomized_svd

def hosvd_2mode(Z, r_ocean, r_var, rng):
    # Z: (T, N_ocean, V) after Gaussianization
    T, N, V = Z.shape
    # mode-2 (ocean_cell) randomized SVD
    Z2 = Z.transpose(1, 0, 2).reshape(N, T * V)
    U_ocean, _, _ = randomized_svd(Z2, n_components=r_ocean, random_state=rng)
    # mode-3 (variable) SVD — small
    Z3 = Z.transpose(2, 0, 1).reshape(V, T * N)
    U_var, _, _ = np.linalg.svd(Z3, full_matrices=False)
    U_var = U_var[:, :r_var]
    core = np.einsum("tnv,nk,vm->tkm", Z, U_ocean, U_var)
    return core, U_ocean, U_var
```

**Rank selection:**
- `r_ocean`: start at 200. Increase until reconstruction error drops below 5% of `‖Z‖_F`. Expect ~300–500 for global 0.25°.
- `r_var`: start at `min(V, 6)`. Cross-variable structure is low-dimensional.

Sanity check: reconstruct `Z_recon = np.einsum("tkm,nk,vm->tnv", core, U_ocean, U_var)` and measure Frobenius error against `Z`.

### 5.6 Exact DMD on the compressed core

Flatten core to snapshots `x_t ∈ ℝ^{r_ocean · r_var}`:

```python
def exact_dmd(snapshots, r_dmd, dt, rng):
    # snapshots: (T, N_state), real
    X = snapshots[:-1].T                              # (N_state, T-1)
    Xp = snapshots[1:].T
    U, S, Vt = randomized_svd(X, n_components=r_dmd, random_state=rng)
    V = Vt.T
    A_tilde = U.conj().T @ Xp @ V @ np.diag(1.0 / S)  # (r_dmd, r_dmd)
    eigvals, W = np.linalg.eig(A_tilde)
    Phi = Xp @ V @ np.diag(1.0 / S) @ W               # (N_state, r_dmd), complex
    b0 = np.linalg.pinv(Phi) @ snapshots[0]
    lam_cont = np.log(eigvals) / dt
    return Phi, eigvals, lam_cont, b0
```

**Rank `r_dmd`:** start 50, extend to 100–200 for a global model. Diagnostic: plot `eigvals` on the complex plane. Retain modes on or near the unit circle (physically persistent oscillators) plus the leading decaying continuum. Discard modes with `|eigvals|` far outside `[0.5, 1.05]`.

**Report only physically meaningful frequencies:** filter out modes where `|Im(lam_cont)| < 1e-8` when reporting periods — the "period" `2π/Im(λ)` blows up spuriously for near-real eigenvalues. These modes are legitimate mode-shapes (typically slow drift), just not oscillators.

**Complex-conjugate pair handling.** Non-real DMD eigenvalues come in conjugate pairs. Amplitudes `a_k, a_{k'}` must remain conjugates for the reconstructed field to be real. Handling options:

- Fit OU on the complex amplitude of one member; at simulation time force `η_{k'} = conj(η_k)`.
- Or use real-Schur DMD from the start (Kutz et al. 2016 has the algorithm).

Either works; pick one and enforce it. Test: `np.abs(Y_sim.imag).max() < 1e-10` after every simulation step.

### 5.7 OU fit on residual amplitudes

For each mode `k`:

```
a_k(t) = (Φ⁺ · x_t)_k              # measured amplitude, complex
â_k(t) = b0_k · Λ_kk^t              # deterministic DMD prediction
η_k(t) = a_k(t) - â_k(t)            # residual to be modelled stochastically
```

AR(1) → OU parameters:

```
φ_k    = Σ_t conj(η_k(t)) · η_k(t+Δt)  /  Σ_t |η_k(t)|²
θ_k    = -log(clip(Re(φ_k), 1e-4, 0.9999)) / Δt
σ_k²   = Var(η_k) · 2·θ_k / (1 - |φ_k|²)     # continuous-time diffusion
```

Store `(θ_k, σ_k, φ_k)` per mode.

### 5.8 Simulation and continuous-time query

Discrete simulation on the 3h grid:

```
Initialize η_k(0) from equilibrium N(0, σ_k²/(2·θ_k)) — or from observed state if seeding a specific date.
For t = Δt, 2Δt, ...:
    for each mode k:
        â_k(t) = b0_k · Λ_kk^{t/Δt}
        ξ_k    = complex Gaussian ~ N(0, σ_k²(1 - |φ_k|²)/(2·θ_k))
        η_k(t) = φ_k · η_k(t-Δt) + ξ_k
        a_k(t) = â_k(t) + η_k(t)
    x(t)   = Re( Σ_k a_k(t) · Φ_k )                # compressed state, (N_state,)
    Z(t)   = np.einsum("km,nk,vm->nv", ...)         # Tucker uncompress → (N_ocean, V)
    Z2D(t) = scatter to (H, W, V) via ocean_mask.nonzero()
    U(t)   = Φ_stdnorm(Z2D(t))                      # Gaussian CDF
    anom   = empirical_cdf_inverse(U(t))            # per-cell LUT
    Y(t)   = climatology(t) + σ_cell * anom
    Y(t)   = combine_uv_to_speed_dir(Y(t))          # for directional variables
```

**Continuous-time point query at arbitrary `(lat, lon, t_q)`:**

For consumers that need a single point (or a small batch) at arbitrary times without a full trajectory, use the closed-form OU transition — no discrete grid simulation needed:

```
Δn        = (t_q - t_train_start) / Δt              # can be fractional
η_k(t_q)  = φ_k^Δn · η_k(t_start) + N(0, σ_k²·(1 - |φ_k|^{2·Δn}) / (2·θ_k))
â_k(t_q)  = b0_k · Λ_kk^Δn
```

Reconstruct `Z2D(t_q)` as above, bilinearly interpolate at `(lat_q, lon_q)` (with lon-wraparound padding per §2.6), invert copula + climatology, done.

This is what `sample_batch` uses — one OU evolution to each query time (or to the bracketing snapshots for interpolation), no full grid ever materialized. It's why batched queries are `O(N · r_dmd)`, not `O(H · W · N)`.

---

## 6. Validation

Split: train on `[t_start, t_train_end)`, validate on `[t_train_end, t_end)`. Recommend 2 years training, 1 year validation. Every check emits a plot or scalar in the PDF report from `weather-synth-validate`.

Required checks (all must pass to consider v0.1.0 ready to ship):

1. **Per-cell marginal QQ.** 30 randomly-chosen ocean cells, per variable, real-vs-simulated. Should be near-diagonal. Systematic deviation → copula LUT is under-resolved (raise `n_bins`) or climatology is wrong.
2. **Per-cell temporal spectra (Welch PSD).** Same cells, log-log plot. Must match on the `[3h, 30d]` frequency band. Expect some fall-off at the highest frequencies from the linear-Gaussian assumption. If the fall-off is more than 2× at the storm-passage band (~4–8 days), add Hankel-DMD.
3. **Empirical variogram** at a fixed snapshot, averaged across snapshots. Real vs. simulated must agree within ±10% at ranges `[100 km, 5000 km]`.
4. **Joint density at a single cell.** `(sigwh, was)` and `(u_wind, v_wind)` scatter plots. Must show the correct correlation structure (both magnitude and sign).
5. **Extreme statistics.** 99th percentile of `sigwh` per cell, per season, real vs. simulated. Systematic under-estimate → escalate copula per §3.4.
6. **Batched throughput benchmark.** Time `sample_batch` at N = {1, 100, 10 000, 1 000 000}. Report per-point µs. Confirm the batched fast path is on the order documented in §1 correctness criterion 5.
7. **End-to-end consumer test.** For each real consumer (routing engine, wave-climate summary, etc.), run the consumer on real weather and on synthetic weather for the same held-out date range, compare output distributions. This is the acceptance test that actually matters; the earlier checks are diagnostics.

Emit results as a PDF report per `weather-synth-validate --output report.pdf` (§8.4).

---

## 7. Persisted model artifacts

Directory layout — one dir per model version, mirroring `data/` layout convention:

```
./models/<version>/
├── metadata.json          # variable list, grid extents, cadence, training window,
│                          # ranks, git commit hash, code version
├── ocean_mask.npy         # (621, 1440) bool — wave-grid ocean mask
├── climatology.npz        # coeffs (n_terms, N_ocean, V), reference epoch UTC ISO string
├── standardization.npz    # sigma_cell (N_ocean, V)
├── copula_lut.npz         # sorted_anomalies_binned (n_bins, N_ocean, V), n_bins=500
├── tucker.npz             # U_ocean (N_ocean, r_ocean), U_var (V, r_var)
├── dmd.npz                # Phi (N_state, r_dmd) complex64,
│                          # Lambda (r_dmd,) complex128,
│                          # b0 (r_dmd,) complex128
└── ou.npz                 # theta (r_dmd,) float64, sigma (r_dmd,) float64,
                           # phi (r_dmd,) complex128
```

All arrays via `numpy.savez` / `numpy.savez_compressed`. Rationale for `.npy`/`.npz` over HDF5/Zarr:
- Matches the raw archive's format (no extra dependencies).
- Trivially loadable by any future C++ query library via cnpy without conversion.
- Numpy-native, no separate schema to maintain.

Dtype policy: `float32` where precision allows, `float64` where numerical stability demands it (SVD/eig outputs, OU parameters). Complex: `complex64` for large arrays (`Phi`), `complex128` for small ones (`Lambda`, `b0`, OU `phi`).

Ballpark total size for a 2-year global model: 5–10 GB, dominated by `copula_lut.npz`.

---

## 8. Repo bootstrap

### 8.1 Directory layout

`src/` layout — keeps tests running against the installed package, catches missing package-data issues that a flat layout hides.

```
weather-synth/
├── pyproject.toml                       # packaging + tool config
├── README.md                            # install, minimal example, links to docs/
├── LICENSE
├── AGENTS.md                            # this repo's own conventions (subset of §2.8)
├── CHANGELOG.md                         # Keep a Changelog format
├── .gitignore                           # includes data/, models/, .venv/, __pycache__/
├── .github/
│   └── workflows/
│       └── ci.yml                       # ruff + mypy + pytest matrix
├── docs/
│   ├── model_design.md                  # this handoff document, promoted
│   ├── conventions.md                   # §2 (grids, decoding, directions, NaN, wraparound)
│   └── validation.md                    # §6, plus how to read the PDF report
├── src/
│   └── weather_synth/
│       ├── __init__.py                  # public API surface (see §8.3)
│       ├── _version.py                  # managed by hatch-vcs from git tags
│       ├── config.py                    # SCREAMING_SNAKE_CASE grid constants
│       ├── paths.py                     # snapshot_path, enumerate_snapshots
│       ├── decode.py                    # decode_compass_code, float16 upcast
│       ├── directional.py               # (speed, θ_from) ⇄ (u, v)
│       ├── reproject.py                 # wind/ocean → wave grid, lon wraparound padding
│       ├── ingest.py                    # load a training window into a big xarray Dataset
│       ├── climatology.py               # harmonic fit + eval
│       ├── copula.py                    # per-cell empirical CDF forward + inverse
│       ├── tucker.py                    # HOSVD compress + reconstruct
│       ├── dmd.py                       # exact DMD fit + reconstruct
│       ├── ou.py                        # fit + closed-form continuous-time simulation
│       ├── model.py                     # WeatherModel: fit(), save(), load(), sample*()
│       ├── simulate.py                  # full-field simulation on the grid
│       ├── validate.py                  # QQ, spectra, variogram, extreme stats, report
│       └── cli.py                       # argparse entry points
├── tests/
│   ├── conftest.py                      # fixtures: tiny synthetic archive, small trained model
│   ├── test_paths.py
│   ├── test_decode.py
│   ├── test_directional.py
│   ├── test_reproject.py
│   ├── test_ingest.py
│   ├── test_climatology.py
│   ├── test_copula.py
│   ├── test_tucker.py
│   ├── test_dmd.py
│   ├── test_ou.py
│   ├── test_model.py
│   ├── test_simulate.py
│   └── test_validate.py
└── scripts/
    ├── train.py                         # thin wrapper around cli.train
    ├── simulate.py                      # thin wrapper around cli.simulate
    └── validate.py                      # thin wrapper around cli.validate
```

Sample data and trained models never live in git. `data/` and `models/` are `.gitignore`'d. For CI, `conftest.py` synthesizes tiny archives on the fly.

### 8.2 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "weather-synth"
description = "Stochastic weather generator for point queries at arbitrary (lat, lon, datetime)"
readme = "README.md"
requires-python = ">=3.11"
license = { file = "LICENSE" }
dynamic = ["version"]
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "scikit-learn>=1.5",
    "xarray>=2024.10",
    "pandas>=2.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov",
    "pytest-xdist",
    "matplotlib>=3.9",
    "ruff>=0.7",
    "mypy>=1.13",
]

[project.scripts]
weather-synth-train    = "weather_synth.cli:train_main"
weather-synth-simulate = "weather_synth.cli:simulate_main"
weather-synth-validate = "weather_synth.cli:validate_main"

[tool.hatch.version]
source = "vcs"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "D", "UP", "B", "SIM", "RUF"]
ignore = ["D100", "D104"]     # module/package docstring — too noisy for internal files

[tool.mypy]
python_version = "3.11"
strict = true
warn_unreachable = true

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]
```

### 8.3 Public API surface

Exported from `weather_synth/__init__.py`:

```python
from weather_synth.model import WeatherModel
from weather_synth.config import (
    WAVE_GRID_SHAPE, WIND_GRID_SHAPE, OCEAN_GRID_SHAPE,
    WAVE_GRID_LAT_AT_ROW0, WIND_GRID_LAT_AT_ROW0,
    GRID_LON_STEP_DEG, WAVE_GRID_LAT_STEP_DEG, OCEAN_GRID_LAT_STEP_DEG,
)

__all__ = [
    "WeatherModel",
    "WAVE_GRID_SHAPE", "WIND_GRID_SHAPE", "OCEAN_GRID_SHAPE",
    "WAVE_GRID_LAT_AT_ROW0", "WIND_GRID_LAT_AT_ROW0",
    "GRID_LON_STEP_DEG", "WAVE_GRID_LAT_STEP_DEG", "OCEAN_GRID_LAT_STEP_DEG",
]
```

Everything else is implementation detail. Consumers should touch nothing prefixed with `_`.

### 8.4 CLI contract

```bash
# training
weather-synth-train \
    --data-dir   ./data \
    --model-dir  ./models/v1 \
    --t-start    2023-01-01T00:00Z \
    --t-end      2025-01-01T00:00Z \
    --variables  sigwh wsh wsp was wad wsd \
    --r-ocean    300 \
    --r-var      6 \
    --r-dmd      100

# bulk simulation to disk (produces a directory tree matching ./data's layout)
weather-synth-simulate \
    --model-dir   ./models/v1 \
    --output-dir  ./synthetic \
    --t-start     2025-01-01T00:00Z \
    --t-end       2025-04-01T00:00Z \
    --variables   sigwh wsh wsp was wad wsd \
    --seed        42

# validation — emits a PDF report
weather-synth-validate \
    --model-dir      ./models/v1 \
    --held-out-dir   ./data \
    --held-out-start 2025-01-01T00:00Z \
    --held-out-end   2025-04-01T00:00Z \
    --output         ./reports/v1_validation.pdf
```

All timestamps ISO 8601 with explicit UTC marker (`Z` or `+00:00`). The argparse validator must reject naive strings immediately.

### 8.5 The `WeatherModel` class

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np
import xarray as xr


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    trained_at: datetime
    t_train_start: datetime
    t_train_end: datetime
    variables: tuple[str, ...]
    r_ocean: int
    r_var: int
    r_dmd: int
    git_commit: str


class WeatherModel:
    """Stochastic weather generator producing samples at continuous (lat, lon, datetime)."""

    # --- construction ---
    @classmethod
    def fit(
        cls,
        data_dir: Path,
        t_start: datetime,
        t_end: datetime,
        variables: list[str],
        r_ocean: int = 300,
        r_var: int = 6,
        r_dmd: int = 100,
        rng: np.random.Generator | None = None,
    ) -> Self: ...

    @classmethod
    def load(cls, model_dir: Path) -> Self: ...

    def save(self, model_dir: Path) -> None: ...

    # --- query ---
    def sample(
        self,
        lat: float,
        lon: float,
        t: datetime,
        variable: str,
        seed: int | None = None,
    ) -> float:
        """Single-point convenience call. Internally wraps sample_batch."""

    def sample_batch(
        self,
        lats: np.ndarray,           # (N,)
        lons: np.ndarray,           # (N,)
        times: np.ndarray,          # (N,) datetime64[s]
        variables: list[str] | str,
        seed: int | None = None,
    ) -> np.ndarray:                # (N,) if single variable else (N, V)
        """Fast path. Consumers with tight loops should batch through this."""

    def simulate_field(
        self,
        t_start: datetime,
        t_end: datetime,
        variables: list[str],
        seed: int | None = None,
    ) -> xr.Dataset:
        """Full-field trajectory on the native wave grid at 3h cadence."""

    # --- introspection ---
    @property
    def metadata(self) -> ModelMetadata: ...

    @property
    def variables(self) -> tuple[str, ...]: ...

    def ocean_mask(self) -> np.ndarray:
        """Boolean (621, 1440) — True where the model can produce samples."""
```

Single-point `sample()` packs into length-1 arrays and calls `sample_batch`. One code path to test, one to profile.

---

## 9. Gotchas and implementation notes

Ordered by expected pain if missed.

1. **Row 0 = south for both wave and wind grids.** Most gridded earth-science datasets follow the opposite convention; assumed defaults will fail silently. Test explicitly: place a marker at a known coastline point (e.g. 51.5° N, 0° E — the Greenwich meridian at London) and confirm the marker lands on London in `sigwh`'s NaN mask, not in the Southern Ocean.

2. **Ocean grid orientation is unverified.** Do the correlation study in §2.7 before enabling `sst`/`ocs`/`ocd`. Do not assume by analogy — the wave and wind grids came from WAVEWATCH III and GFS respectively; SST/currents likely come from a different NOAA product with its own convention.

3. **Longitude wraparound at 359.75° → 0°.** Any interpolation across this boundary produces NaN without periodic padding. Enforce padding at the code level: no function that takes an `arr[..., lon]` array and returns interpolated values without going through `pad_lon` first.

4. **Directional interpolation must go through `(u, v)`.** Angle-space bilinear interpolation across the 0°/360° discontinuity is broken. Enforce at the code level: no function that returns "an interpolated angle" without first going through the `(u, v)` decomposition.

5. **Compass code origins differ per field.** `wsd`/`pwd`/`pswd` use code 0 = North; `wad` uses code 1 = North. The single-function API `decode_compass_code(codes, origin_code)` avoids per-caller confusion. Unit-test with both origins on known code→degree pairs.

6. **`float16` precision on disk is much coarser than `float64` in memory.** `sigwh` at 8 m Hs has ~0.008 m precision on disk. Fine for modeling, but will show up as visible discretization in QQ-plots at low quantiles. Not a bug.

7. **Land mask consistency across variables.** Use the *intersection* of `not isnan` across all modeled variables at all training snapshots. An ocean cell that becomes ice-covered mid-training is not a stable-ocean cell and should be masked out.

8. **Complex-conjugate DMD pair handling.** If you fit OU on `Re(η)` and `Im(η)` independently instead of as one complex process, you can produce non-conjugate pairs at sample time → non-real reconstructed fields. Add an assertion: `assert np.abs(Y_sim.imag).max() < 1e-10` after every simulation step in debug builds.

9. **Wraparound at year boundaries in the climatology.** Compute `t_since_epoch_days` as a continuous quantity from `t_train_start`; don't reset to day-of-year at Jan 1. The cosines handle periodicity; bookkeeping should not.

10. **Parallelism goes on modes and seeds, not on grid cells.** Grid cells couple through Tucker reconstruction. Modes (independent OU processes) and Monte Carlo seeds (independent replications) are trivially parallel. Use `joblib.Parallel(n_jobs=-1, backend="loky")` for CPU-bound loops over these axes. For linear algebra, let BLAS threading do the work — a well-batched `np.linalg` call already parallelizes internally, and wrapping it in a Python-level loop usually makes things slower via GIL contention and BLAS-thread oversubscription.

11. **`float16` in arithmetic paths.** `np.load()` returns `float16` correctly; `.astype(np.float64)` upcasts. **Never** use `numpy.float16` in arithmetic — it silently overflows on multiplication of moderate values (`np.float16(300) * np.float16(300)` returns `inf`). Cast at the I/O boundary, once.

12. **Time zones — always UTC.** Every archive timestamp is UTC. All datetimes in the public API must be `datetime` with `tzinfo=timezone.utc`. Naive datetimes raise `ValueError` at the API boundary — don't silently attach UTC.

13. **Reference epoch.** Store `t_train_start` as an ISO 8601 UTC string in `metadata.json`. All temporal computations at query time use `(t_query - t_train_start).total_seconds() / (3 * 3600)` to get snapshot index. Log a warning when a query is more than ~5 years outside the training window — the DMD extrapolation gets progressively less reliable.

14. **Physical consistency across variables is NOT enforced.** After sampling `sigwh`, `wsh`, `pwh`, the results may not satisfy `wsh ≤ sigwh` or `pwh ≤ sigwh`, and derived quantities like `swell_residual = sqrt(max(sigwh² - wsh², 0))` may be inconsistent with sampled `pwh, pwd`. The Gaussian copula preserves correlations but not physical relationships. If consumers depend on hard physical constraints (wave steepness within physical bounds, energy budget closure), add a post-projection step or a rejection loop. For v1, document the limitation and defer.

15. **Nested BLAS threading pitfall.** When a consumer calls `sample_batch` from inside their own `joblib.Parallel` or `multiprocessing.Pool`, inner BLAS threads multiply by outer process count. 16 workers × 16 BLAS threads → 256-way contention on a 16-core box. `WeatherModel.sample_batch` should not spawn its own workers, and it should honor `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` set by the caller. Document in the README: consumers running many parallel workers should set `OMP_NUM_THREADS=1` in each worker's environment before importing numpy.

---

## 10. Variable table

Directions all in **meteorological "coming from" convention, CW from N**, after decoding per §2.3. Verify all "assumed" or "unverified" rows before enabling in the model.

| File            | Meaning                                | On-disk dtype | Grid  | Decode                          | Modeled as                                | Tier |
|-----------------|----------------------------------------|---------------|-------|---------------------------------|--------------------------------------------|------|
| `sigwh.npy`     | Significant wave height                | float16       | wave  | float upcast                    | scalar                                     | 1    |
| `wsh.npy`       | Wind-sea wave height                   | float16       | wave  | float upcast                    | scalar                                     | 1    |
| `wsp.npy`       | Wind-sea period                        | float16       | wave  | float upcast                    | scalar                                     | 1    |
| `wsd.npy`       | Wind-sea direction (coming from)       | int8 (0–15)   | wave  | `deg = code * 22.5`             | paired with `wsh` → `(u, v)`               | 1    |
| `was.npy`       | Wind speed at surface                  | float16       | wind  | float upcast                    | scalar                                     | 1    |
| `wad.npy`       | Wind direction (coming from)           | int8 (1–16)   | wind  | `deg = (code - 1) * 22.5`       | paired with `was` → `(u, v)`               | 1    |
| `pwh.npy`       | Peak wave height                       | float16       | wave  | float upcast                    | scalar                                     | 2    |
| `pwp.npy`       | Peak wave period                       | float16       | wave  | float upcast                    | scalar                                     | 2    |
| `pwd.npy`       | Peak wave direction (coming from)      | int8 (0–15)   | wave  | `deg = code * 22.5`             | paired with `pwh` → `(u, v)`               | 2    |
| `sst.npy`       | Sea surface temperature                | float16       | ocean | float upcast (verify units K/°C)| scalar                                     | 2    |
| `pswh.npy`      | Primary swell height                   | float16       | wave  | float upcast                    | scalar                                     | 3    |
| `pswp.npy`      | Primary swell period                   | float16       | wave  | float upcast                    | scalar                                     | 3    |
| `pswd.npy`      | Primary swell direction (coming from)  | int8 (assumed 0–15) | wave | assume `deg = code * 22.5`, verify | paired with `pswh` → `(u, v)`         | 3    |
| `ocs.npy`       | Ocean current speed                    | float16       | ocean | float upcast                    | scalar                                     | 3    |
| `ocd.npy`       | Ocean current direction (coming from)  | int8 (verify) | ocean | verify formula                  | paired with `ocs` → `(u, v)`               | 3    |
| `wad_come_from.npy` | Undocumented, duplicates `wad`     | int8          | wind  | n/a                             | **IGNORED**                                | —    |

---

## 11. References

Ordered by usefulness for this project. Read the first two before writing any decomposition code.

- Kutz, Brunton, Brunton, Proctor (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.* SIAM. Book + code. Primary DMD reference.
- Tu, Rowley, Luchtenburg, Brunton, Kutz (2014). "On dynamic mode decomposition: theory and applications." *J. Computational Dynamics.*
- Kolda & Bader (2009). "Tensor decompositions and applications." *SIAM Review.* Tucker/HOSVD reference.
- Brunton, Brunton, Proctor, Kaiser, Kutz (2017). "Chaos as an intermittently forced linear system." *Nature Communications.* Hankel-DMD; escalation path if base model undershoots high-frequency variance.
- Sashidhar & Kutz (2022). "Bagging, optimized dynamic mode decomposition." *Phil. Trans. R. Soc. A.* Bagged DMD with UQ.
- Wilks & Wilby (1999). "The weather generation game." *Progress in Physical Geography.* Survey of stochastic weather generators.
- Ailliot, Allard, Monbet, Naveau (2015). "Stochastic weather generators: an overview." *J. Soc. Fr. Stat.*
- Lindgren, Rue, Lindström (2011). "An explicit link between Gaussian fields and Gaussian Markov random fields: the SPDE approach." *JRSS-B.* Alternative if DMD-OU has unfixable calibration problems.
- Coles (2001). *An Introduction to Statistical Modeling of Extreme Values.* For GPD-tail escalation of the Gaussian copula.
- Mardia & Jupp (2000). *Directional Statistics.* Circular means, distributions on the sphere.
- Halko, Martinsson, Tropp (2011). "Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions." *SIAM Review.* Randomized SVD, which `sklearn.utils.extmath.randomized_svd` implements.

---

## 12. What "done" looks like

A tagged v0.1.0 release providing:

1. Pip-installable package (`pip install weather-synth` from the built wheel; `pip install -e .[dev]` for local development).
2. The three CLI entry points from §8.4: `weather-synth-train`, `weather-synth-simulate`, `weather-synth-validate`.
3. The `WeatherModel` public API from §8.5, with full docstrings and type hints.
4. Green CI on Python 3.11 / 3.12 / 3.13: `ruff check`, `ruff format --check`, `mypy --strict src/`, `pytest --cov=weather_synth` (coverage ≥ 85%).
5. Unit test coverage — each is a specific pytest case, not an abstract requirement:
   - `test_decode.py::test_wsd_compass_codes` — known code → degree pairs (§2.3).
   - `test_decode.py::test_wad_compass_codes` — same, with the 1-indexed origin.
   - `test_directional.py::test_uv_cardinal_signs` — the four `θ_from ∈ {0°, 90°, 180°, 270°}` cases from §2.4. **Load-bearing** — catches sign flips before they propagate.
   - `test_directional.py::test_uv_roundtrip` — `(speed, θ) → (u, v) → (speed', θ')` reproduces input within `1e-10`.
   - `test_paths.py::test_snapshot_path_zero_padded` — month/day/hour zero-padding always present.
   - `test_paths.py::test_utc_timezone_required` — naive datetime raises `ValueError`.
   - `test_climatology.py::test_harmonic_roundtrip` — fit → eval reproduces training data at fit points.
   - `test_copula.py::test_forward_inverse_roundtrip` — cell-wise round-trip preserves original values.
   - `test_copula.py::test_gaussian_output_distribution` — after forward transform, per-cell distribution is standard normal (KS test).
   - `test_tucker.py::test_hosvd_roundtrip` — analytic rank-`r` tensor recovered exactly at rank `r`.
   - `test_dmd.py::test_analytic_traveling_wave` — synthetic `A cos(kx - ωt)` recovered as complex-conjugate pair with correct `ω`.
   - `test_ou.py::test_parameter_recovery` — synthetic OU trace with known `(θ, σ)` recovered within 5%.
   - `test_model.py::test_fit_save_load_sample` — end-to-end pipeline on a small synthetic archive.
   - `test_model.py::test_batch_matches_single_point` — `sample_batch([lat], [lon], [t], var)[0] == sample(lat, lon, t, var)` given the same seed.
   - `test_model.py::test_seed_reproducibility` — same seed, same query → byte-exact output.
6. Grid convention verification tests, required only if the corresponding tier is enabled:
   - `test_integration_wave_grid.py::test_row0_is_south` — coastline correlation, per §2.7.
   - `test_integration_ocean_grid.py::test_orientation` — required before enabling `sst`/`ocs`/`ocd`.
   - `test_integration_pswd.py::test_convention` — required before enabling `pswh`/`pswp`/`pswd`.
7. An integration test (`test_integration.py::test_end_to_end_marginal_fit`) that trains on 3 months of synthetic archive data, simulates 1 month, and asserts per-cell mean and std within 10% of held-out truth.
8. Documentation:
   - `README.md` — install, minimal example, link to `docs/`.
   - `docs/model_design.md` — this handoff document, promoted.
   - `docs/conventions.md` — §2 (grids, direction, path conventions), for consumers who need to interpret raw archive data themselves.
   - `docs/validation.md` — the validation battery (§6), including how to interpret each plot in the emitted PDF report.
9. `CHANGELOG.md` following Keep a Changelog format, with v0.1.0 entry describing the initial release.
