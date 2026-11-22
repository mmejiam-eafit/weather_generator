# Stochastic Weather Generator — Design Handoff (v3)

**Package:** `weather-generator` (Python, three sibling packages under `src/`: `core`, `train`, `weather`).

**Audience:** coding agent bootstrapping the repo from scratch, or picking up work in progress. This document is self-contained — every convention, formula, and restriction the agent needs is stated here, not by reference.

**One-line summary:** train a stochastic model on the raw 3-hourly `.npy` archive produced by the [`noaa_etl`](./noaa_etl/) pipeline — NOAA GFS-Wave GRIB2 (wave *and* wind fields together, one product) plus RTOFS/HYCOM (sea surface temperature and ocean currents) — and provide plausible synthetic samples of any variable at any continuous `(lat, lon, datetime)` when real data isn't available. v3 trains on 12 months of archive era 3 and validates on the real data that follows. **`docs/noaa_etl/` is the unequivocal source of truth for this archive's origin and transformations** (see `docs/noaa_etl/pipeline.md`); every data fact in §2 below is stated to be consistent with it.

**Approach chosen:** hierarchical decomposition, one independent chain per modeled field — **linear-trend + harmonic climatology** on log-energy / log-period / `(u, v)` fields → **per-variable masked Tucker** compression of the standardized anomaly → **Gaussian copula on each Tucker core** → **DMD + OU on mode amplitudes** per variable group. A training-time diagnostics stage (linear inverse model, tau test, coupling information) is recorded with the model. The per-cell local-residual layer is deferred in v3 (§3.10, §5.7). Orchestrated by a scikit-learn `BaseEstimator + TransformerMixin` wrapper. See §3 for rationale and rejected alternatives.

**Language:** Python 3.14 only. Batched-first API design (§1, §8). If a downstream consumer eventually shows sub-millisecond point queries are needed and cannot be batched, a C++ query library can be added; do not build one preemptively.

**Training entry point:** a Jupyter notebook (`notebooks/test_pipeline.ipynb`), not a CLI. Training is a one-shot exploratory activity that benefits from cell-level iteration on Tucker/DMD/OU hyperparameters against a cached preprocessing output.

**Training data (v3):** `2025-10-27T00:00Z` to `2026-10-27T00:00Z` (exclusive): 12 months of archive era 3 (721 × 1440, 3-hourly, §2.2.1), `T = 2920` snapshots. A full annual cycle, so the climatology's annual and semi-annual harmonics are constrained (§3.9). The archive spans three grid/cadence eras and `train.ingest.load_training_window` rejects any window crossing an era boundary. Era 3 is the canonical grid; training on a longer era-3 window is to be revisited once era 3 reaches about 18 months of history (around 2027-04-27).
**Validation (numerical metrics):** `2026-10-27T00:00Z` onward: the real data after the training window, about 25 days when v3 was written (archive downloaded through 2026-11-21) and growing as data arrives. Held-out real data used to compute the validation battery in §6. Earlier versions validated on 2026-01-01 → 2026-07-19, which now falls inside the training window.
**Extrapolation window:** past the last real snapshot. The model must produce plausible synthetic weather there without real data to compare against, assessed by consistency and stability checks rather than pointwise metrics (§6.B).

---

## What changed in v3

Read this section if you know v2. v3 brings the handoff in line with the decisions recorded since v2 in `docs/open_questions.md`, the `docs/` topic series and the task files.

1. **One independent tensor per modeled field** (`variable_taxonomy.md` §4–§5). Heights become log energy (`logE_* = log(max(H, 1 cm)²)`), periods become `log T` (`T_*`), directions become speed- or height-weighted `(u, v)`. No variable axis exists anywhere: 17 tensors in total (§10). §3.8, §5.3.
2. **Per-variable masked Tucker** on each `(T, lat, lon)` tensor, compressing only the spatial modes, through the vendored `train/masked_tucker.partial_tucker_lowmem` (task 07 §5 fixed two tensorly bugs). Replaces v2's joint ocean-cell × variable Tucker and its `r_ocean`/`r_var` ranks. Rank: 40% of each spatial dimension (task 07). §5.5.
3. **Gaussianization after Tucker** (placement ii, DECIDED in `open_questions.md` §4.1). The copula is fit on each Tucker core's coefficients, not on grid cells, and leaves the preprocessing Pipeline. §3.5, §5.5.1.
4. **Land mask from `sigwh`, sentinel fill for absent partitions** (task 17, `open_questions.md` §5.6). §5.1, §5.3.1.
5. **DMD and mode-level OU per variable group**: one group per tensor by default; joint groups are a Linear Inverse Model (`correlated_variables.md` §4). `r_dmd = 100` (task 07). The mode OU drives one real channel per real mode and per conjugate pair (task 07's validated pair procedure), with one joint innovation covariance per group. §5.6, §5.6.1.
6. **Group dynamics diagnostics stage** (tau test, coupling information). §5.6.2.
7. **Local residual deferred.** Not part of the first era-3 model (`include_local_residual=False`). §3.10, §5.7.
8. **Sampling interpolates in linear-safe space.** Per-cell inverses on grid cells, then bilinear interpolation of `(u, v)`, `logE` and `T`, then pointwise `exp` / `hypot` / `atan2`. v2 interpolated physical values, direction angles included. §5.8.
9. **Training and validation windows** moved to era 3 (above).

## What changed in v2 (vs. v1)

Read this section if you're already familiar with v1. Otherwise skip to §1.

1. **Adopt TensorLy for Tucker.** `train/tucker.py` is now a thin wrapper around `tensorly.decomposition.partial_tucker` rather than a hand-rolled HOSVD. See §3.11 and §5.5.
2. **Adopt PyDMD for DMD.** `train/dmd.py` is now a thin wrapper around `pydmd.DMD` (with `pydmd.BOPDMD` as an alternate variant for uncertainty quantification and automatic complex-conjugate handling). See §3.11 and §5.6. The escalation paths — Hankel-DMD, BOPDMD — become one-line class swaps.
3. **Linear trend added to climatology.** `μ(s, t)` now includes a `c_lin(s) · (t − t_ref)` term to catch monotone drift (SST warming, storm track shifts). See §3.9 and §5.4.
4. **Hierarchical decomposition of the anomaly.** After Gaussianization, `Z(s, t)` is now decomposed into a *coherent* term (Tucker + DMD + OU on modes) plus a *local residual* term (independent per-cell OU fields). The v1 design captured only the coherent term, which produced samples that were too spatially smooth. See §3.10 and §5.7.
5. **New pure-function module.** `train/local_residual.py` fits and evolves the per-cell OU field. See §5.7.
6. **Notation cleanup.** OU parameters are now subscripted by scope: `θ_mode`, `σ_mode` for DMD modes; `θ_local`, `σ_local` for per-cell residual. `σ_std` is the standardization scale.
7. **Directional conventions are mixed in the archive, and the pipeline reflects that explicitly.** Previous versions assumed all directions were "coming from" and used a single `DirectionalDecomposer`. In reality `wad` and `ocd` are stored as `going_to`, while `wsd`, `pwd`, `pswd`, and `wad_come_from` are all `coming_from`. This is no longer just an empirical inference: `docs/noaa_etl/noaa_transform.md` (the ETL's own documentation, authoritative for how these fields are produced) cites NOAA's own references — [NCEP GRIB2 Code Table 4.2-0-2](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-2.shtml) defines the raw `WDIR` GRIB field as "wind direction (from which blowing)", and [NDBC's measurement glossary](https://www.ndbc.noaa.gov/faq/measdes.shtml) defines the wave-direction parameters underlying `pwd`/`wsd`/`pswd` the same way — while ocean current direction is the one NDBC exception, reported as the direction the current flows *toward*, confirming `ocd`'s `going_to` convention independently of grid orientation. `pwd` and `pswd` are therefore **confirmed**, not merely assumed pending domain-expert review as a prior version of this doc stated (see §2.3, §2.4). v2 uses **two atomic transformers** — `ComingFromToUV` and `GoingToToUV` — with disjoint pair lists, so the pipeline definition itself shows which convention each variable uses. See §3.7 and §5.3. `wad_come_from` is the source of truth for wind; a runtime `wad`/`wad_come_from` consistency check guards against archive drift (§5.1) — though per the ETL, `wad` is deterministically derived from `wad_come_from` by a `±180°` transform at transform time, so this check is a defense against upstream *pipeline* changes, not archive noise. The convention table was originally established empirically in `notebooks/verify_direction_convention.ipynb`, a single-snapshot check; it is now corroborated by the ETL's documented sourcing, so it no longer needs to be treated as provisional for `wsd`/`pwd`/`pswd`/`wad`/`wad_come_from`/`ocd`'s "coming from vs. going to" question. The 16-point compass *code-to-degree* mapping itself, however, was wrong in a prior version of this doc — see the corrected §2.3.
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

### 2.2 Two grids, not three — "wave" and "wind" are the same product

**A prior version of this doc described three grids (wave, wind, ocean), treating "wave" and "wind" as physically separate sources needing cross-grid reprojection. That's wrong and is corrected here.** Per `docs/noaa_etl/pipeline.md` and `docs/noaa_etl/noaa_transform.md`, all wave *and* wind variables (`sigwh`, `wsh`, `wsp`, `wsd`, `pwh`, `pwp`, `pwd`, `pswh`, `pswp`, `pswd`, `was`, `wad`, `wad_come_from` — 13 in total) are decoded from **one GFS Wave GRIB2 file per cycle/forecast-hour** (`gfswave.t{cycle}z.global.0p25.f{fff}.grib2`); GFS Wave outputs the surface wind fields it uses to force the wave model alongside the wave-state fields, all on the identical 0.25° grid. There is no separate "wind model" or "wind grid" in the archive's origin — only the ocean (HYCOM/RTOFS) variables come from a genuinely different source and grid. This matters architecturally: §3.6/§5.2's `Reprojector` has no work to do for wind (it's already on the wave grid) — only ocean needs reprojecting.

| Grid            | Shape (rows × cols) | Resolution | Lat range      | Lon range        | Row 0 | Variables | Source |
|------------------|---------------------|------------|----------------|------------------|-------|-----------|--------|
| **GFS Wave** (wave + wind) | 721 × 1440 | 0.25°      | [−90°, +90°]   | [0°, 359.75° E]  | south | `sigwh`, `wsh`, `wsp`, `wsd`, `pwh`, `pwp`, `pwd`, `pswh`, `pswp`, `pswd`, `was`, `wad`, `wad_come_from` | GFS Wave GRIB2 (NOAA NOMADS) |
| **Ocean**        | 1121 × 2880         | 0.125°     | (verify §2.7)  | [0°, 359.875° E] | (verify §2.7) | `sst`, `ocs`, `ocd` | RTOFS/HYCOM (NOAA NOMADS), remapped from a native tripolar mesh — see §2.7 |

Row indexing for the GFS Wave grid (**the non-standard part**):

```
lat = -90.0 + row * 0.25    row ∈ [0, 720]
```

This matches NOAA's published grid definition for `gfswave.global.0p25` (`Ni=1440`, `Nj=721`, 0.25° spacing, `La1=90°`, `Lo1=0°`, `La2=-90°`, `Lo2=359.75°`), confirmed in `docs/noaa_etl/noaa_transform.md`. As delivered by `pygrib`, row 0 is the north pole and row index increases southward; the ETL's `flip_latitude` step reverses this for every GFS variable before it's written to `.npy`, which is why row 0 = south (−90°) in the *stored* archive.

Longitude: `lon = col * 0.25`, `col ∈ [0, 1439]`, spanning `[0°, 359.75° E]` (eastward, `[0°, 360°)` convention, not `[-180°, 180°]`).

**Row 0 = south is critical.** Most gridded earth-science datasets follow the opposite convention. This doc previously described establishing it purely via an empirical land-mask correlation study (§2.7's method below); it's now also directly confirmed by the ETL's documented `flip_latitude` behavior, which is the authoritative account of how the stored orientation came to be. See §2.7 for the empirical cross-check and how to extend it to the ocean grid.

#### 2.2.1 The table above now describes era 3 (current) — the archive has three eras

The table above's wave-grid row, and `core.config`'s wave-grid constants (`WAVE_GRID_SHAPE`, `WAVE_GRID_LAT_AT_ROW0`), describe era 3 (2025-10-27 onward, current and open-ended — the era production training targets), not era 2. `core.config`'s wind/ocean/cadence constants were already correct for era 3 — they're numerically identical across era 2 and era 3, so only the wave-grid constants needed updating. Direct binary-search inspection of the real archive (`data/`, a symlink) found two further, confirmed, clean, one-time transitions before era 3. **`core.archive_eras.ARCHIVE_ERAS` is the authoritative, executable source of truth for the table below** — do not hand-copy these numbers elsewhere; import and use `core.archive_eras.era_for_timestamp` instead.

| Era | Dates (UTC) | Wave shape | Wind shape | Ocean shape | Cadence |
|---|---|---|---|---|---|
| **era1** | 2022-01-01 → 2023-10-01T18:00Z | (181,360) | (181,360) | `sst`=(180,360) **≠** `ocs`/`ocd`=(1265,2880) — internally inconsistent | 6-hourly (00/06/12/18Z only; 03/09/15/21Z structurally empty, confirmed across the whole era) |
| **era2** | 2023-10-02T00:00Z → 2025-10-26T21:00Z | (621,1440) — the table above | (721,1440) — the table above | (1121,2880), consistent — the table above | 3-hourly — §5.1's documented default |
| **era3** | 2025-10-27T00:00Z → open-ended (downloaded through 2026-11-21) | (721,1440) — see below | (721,1440) | (1121,2880) | 3-hourly |

All three transitions are confirmed clean, synchronized, one-time cutovers (no patchy/interleaved periods).

**Coordinate convention (`lat0`/step), not just shape** — empirically verified in `notebooks/verify_era1_era3_grid_coordinates.ipynb` (land/ocean mask correlation against era 2's known-correct grid, the same general technique §2.7 describes for the ocean grid):
- **era1**: `lat0=-90.0`, `1.0°` steps, `lon0=0.0` — a coarser *global* grid (181 rows × 1.0° spans exactly −90°..+90°), decisively confirmed (97.6% land-mask agreement at this offset vs. the next-best candidate at 86.8%). This is **not** era 2's regional wave-grid convention (`lat0=-75.0`) — it matches the wind grid's convention, just 4x coarser.
- **era3**: the wave grid is a **literal copy of the wind grid** (confirmed via same-timestamp cell-by-cell land-mask agreement of 1.0000, vs. 0.5643 for a different-date control) — `lat0=-90.0`, `0.25°` steps, i.e. `WIND_GRID_LAT_AT_ROW0`/`WIND_GRID_LAT_STEP_DEG`, not the era-2 wave grid's.

Era 3's finding is now wired into `core.config` — `WAVE_GRID_SHAPE`/`WAVE_GRID_LAT_AT_ROW0` encode it, so `core.config.grid_coordinates` returns era 3's convention for the wave grid unconditionally, for every era. This is correct for era 3 (current/production) but a known, accepted tradeoff for era 1/era 2: `train.ingest.load_training_window` validates *shape* per era and rejects cross-era windows, but an era 1 or era 2 window's wave-grid coordinate *labels* are wrong if loaded (era 1's convention remains entirely unwired). See `core/config.py`'s module docstring for the exact mislabeling and rationale.

### 2.3 On-disk dtypes and decoding

Two dtype conventions in the archive:

- **Continuous fields** (`sigwh`, `wsh`, `wsp`, `pwh`, `pwp`, `pswh`, `pswp`, `was`, `sst`, `ocs`): stored as `float16` on disk. **Units, per `docs/noaa_etl/noaa_transform.md`:** `was` and `ocs` (speeds) are in **knots** (`ms_to_knots`, factor `1.943845249222`, applied at transform time — the raw GRIB/HYCOM values are m/s); wave heights (`sigwh`, `wsh`, `pwh`, `pswh`) are in meters (no unit conversion applied); periods (`wsp`, `pwp`, `pswp`) are in seconds (no conversion); `sst` unit needs verification (§10) — the ETL applies no explicit conversion, so it is presumably the HYCOM native unit.
- **Direction fields** (`wsd`, `pwd`, `pswd`, `wad`, `wad_come_from`, `ocd`): stored as `int8` 16-point-compass codes (22.5° steps), not raw degrees.

**Immediately on load, upcast everything to `float64`.** `float16` overflows in arithmetic at surprisingly small values (`(300.0)²` overflows).

**Compass code decoding — corrected against `docs/noaa_etl/noaa_transform.md`.** A prior version of this section described two different origin codes (`0` for `wsd`/`pwd`/`pswd`, `1` for `wad`/`wad_come_from`/`ocd`) with two different decode formulas. **That table was wrong.** The ETL's own documentation of `calculate_16degree` — the function that actually produces every direction field in the archive — shows all six direction fields share **one** encoding: sectors `1–16`, each spanning 22.5°, centered on `sector * 22.5°`, with **sector 16 = North** (wrapping the 0°/360° boundary), not sector 0 or sector 1. Concretely: sector 1 = 22.5° (NNE), sector 4 = 90° (E), sector 8 = 180° (S), sector 12 = 270° (W), sector 16 = 0°/360° (N). There is no per-field origin code — every direction field decodes the same way.

What genuinely differs per field is the **native convention** (whether the decoded angle means "coming *from*" or "going *to*" this direction) — that part of the prior table is directionally correct and is now further corroborated (not just by the single-snapshot empirical check in `notebooks/verify_direction_convention.ipynb`) by NOAA's own documentation, cited in `docs/noaa_etl/noaa_transform.md`: NCEP GRIB2 Code Table 4.2-0-2 defines raw `wdir` as "wind direction (from which blowing)", and NDBC's glossary defines the wave-direction parameters underlying `pwd`/`wsd`/`pswd` the same way, while ocean current direction is NDBC's one documented exception (direction flowing *toward*). `pwd` and `pswd` are therefore confirmed `coming_from`, not merely assumed pending expert review:

| Field            | Code range | Decode formula (uniform) | Native convention                    |
|------------------|------------|---------------------------|----------------------------------------|
| `wsd`            | 1–16       | `(code % 16) * 22.5`      | `coming_from` (confirmed — NCEP/NDBC) |
| `pwd`            | 1–16       | `(code % 16) * 22.5`      | `coming_from` (confirmed — NCEP/NDBC) |
| `pswd`           | 1–16       | `(code % 16) * 22.5`      | `coming_from` (confirmed — NCEP/NDBC) |
| `wad`            | 1–16       | `(code % 16) * 22.5`      | `going_to` (deliberate ±180° flip of `wad_come_from`, applied by the ETL for consistency with `ocd`'s convention when overlaying wind/current arrows) |
| `wad_come_from`  | 1–16       | `(code % 16) * 22.5`      | `coming_from` (confirmed — raw `wdir`, unconverted) |
| `ocd`            | 1–16       | `(code % 16) * 22.5`      | `going_to` (confirmed — NDBC ocean-current convention; derived directly from `atan2(u, v)` on HYCOM's native velocity vector — this determination does **not** depend on the ocean grid's row/col-to-lat/lon orientation, which is a separate, still-unverified concern, §2.7) |

**Missing/invalid sentinel: `-1`.** `-1` is a valid in-range `int8` value in this store, used by the ETL as the explicit missing-value marker for every direction field — it is **not** a real sector and must be filtered out (via the paired float field's NaN mask, §2.5) before decoding, not passed through the formula above. Feeding `-1` through `(code % 16) * 22.5` silently produces a plausible-looking but meaningless angle (`337.5°`) instead of erroring.

**The archive does not use a single uniform *convention*, even though it now uses a single uniform *code-to-degree formula*.** Wave directions are coming-from; ocean current direction is going-to; wind is stored twice (once as going-to in `wad`, once as coming-from in `wad_come_from`). Convention must still be chosen per variable — do not generalize the convention from one field to another, even though the decode formula itself no longer varies.

Canonical decoder (lives in `core.io`):

```python
def decode_compass_code(codes: np.ndarray) -> np.ndarray:
    """Decode int8 16-point compass codes to degrees (float64, [0, 360)).

    Every direction field in the archive (wsd, pwd, pswd, wad, wad_come_from, ocd)
    uses the identical sector encoding: codes 1-16, sector 16 = North (§2.3). -1 is
    a missing-value sentinel and must be masked out by the caller before calling this
    function, not passed through it.

    Returns raw decoded angle in the archive's native convention for that field —
    either "coming from" or "going to". The convention is a separate concern from
    decoding and is handled downstream by the ComingFromToUV or GoingToToUV
    transformers (see §5.3).
    """
    return ((codes.astype(np.int32) % 16) * 22.5).astype(np.float64) % 360.0
```

The decoder is convention-agnostic on purpose. Convention handling lives in the preprocessing Pipeline (§3.7, §5.3) so that adding a directional variable requires explicitly choosing which transformer it belongs to — no silent defaulting.

Test cases (must pass):
- `decode_compass_code(np.array([[16]], dtype=np.int8))` → `[[0.0]]` (North).
- `decode_compass_code(np.array([[4]], dtype=np.int8))` → `[[90.0]]` (East).
- `decode_compass_code(np.array([[8]], dtype=np.int8))` → `[[180.0]]` (South).
- `decode_compass_code(np.array([[12]], dtype=np.int8))` → `[[270.0]]` (West).

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

- **Wave-state variables** (`sigwh`, `wsh`, `wsp`, `wsd`, `pwh`, `pwp`, `pwd`, `pswh`, `pswp`, `pswd`) — `float` fields carry NaN for land and persistent ice cover. Authoritative land mask for the GFS Wave grid (§2.2).
- **Wind variables** (`was`, `wad`, `wad_come_from`) — from the same GFS Wave grid/product (§2.2), but have *no* NaN. Winds are defined over land in the GFS product. Mask via a Natural Earth land polygon rasterized to the grid.
- **Ocean grid** — `sst` almost certainly carries NaN for land; `ocs`/`ocd` behavior needs verification.
- **Direction fields** — `int8`, no float NaN, but **do** carry an explicit missing-value sentinel, `-1` (§2.3) — a valid in-range `int8` value, not decodable via the compass formula. Filter `-1` explicitly before decoding; don't rely solely on the paired float field's mask, since the two are not guaranteed to be pixel-identical in provenance even though they're masked together in practice for the stochastic model (§2.5 below, ocean-mask intersection).

**Every accumulation guarded with `np.isnan`.** NaN in → NaN through. Never coerce NaN to zero.

For the stochastic model: fit only on ocean cells. Build one boolean `ocean_mask` per grid (see §5.1), model over the ravelled ocean-cell vector, project back to `(H, W)` for output. Ocean cells must be non-NaN in *all* modeled variables at *all* training snapshots.

### 2.6 Longitude wraparound

Both grids span `[0°, 359.75° E]` — no dateline duplication. `col=1439` and `col=0` are adjacent on the sphere but at opposite ends of the array. Pad periodically before any interpolation:

```python
def pad_lon(field: np.ndarray, pad: int = 2) -> np.ndarray:
    return np.concatenate([field[..., -pad:], field, field[..., :pad]], axis=-1)
```

### 2.7 How grid orientation was established (and how to verify it)

The row-0-is-south convention was originally documented as the opposite and turned out to be wrong. The correction is now doubly confirmed: `docs/noaa_etl/noaa_transform.md` documents that raw `pygrib` output has row 0 at the north pole and that the ETL's `flip_latitude` step reverses this for every GFS Wave variable (both wave-state and wind fields, since they're one product per §2.2) before writing to `.npy` — this is the authoritative account. It's also independently corroborated by this repo's own empirical study:

- **GFS Wave grid (wave-state fields).** Correlate the real NaN pattern in `sigwh` against a rasterized coastline from Natural Earth, across multiple days and every plausible row offset and row order. The correct convention appears as a sharp single peak in agreement (~92.6% match). Verified visually.
- **GFS Wave grid (wind fields).** No NaN mask of their own to correlate against (§2.5). Confirmed by visual landmark displacement matching the wave-state fields on the same grid, and by winter storm wind maxima landing at plausible North Pacific / North Atlantic cyclone latitudes.

**The ocean grid's geographic orientation (row/col → lat/lon) has not been verified. Do it before including SST or currents.** Per `docs/noaa_etl/noaa_transform.md`, this genuinely cannot be derived from the codebase alone: HYCOM/RTOFS's native grid is **tripolar** (a real pole at the South Pole plus two displaced poles over land in the Northern Hemisphere, avoiding a coordinate singularity over open Arctic ocean), so rows/columns in the raw decoded array don't correspond to a uniform `lat = f(row)` formula the way the GFS grid does. The ETL's `hycom_index` step resamples this onto the uniform `(1121, 2880)` output grid using a precomputed nearest-neighbor lookup table (`hycom_satocean_idx.npy`, generated *outside* this codebase), and this repo contains only the index table, not the geographic (lat/lon-in-degrees) definition of the grid it targets — so `[0, 0]`'s real-world coordinates are opaque from the ETL alone and must be established empirically, same as this doc already prescribed. Method: correlate `sst` NaN pattern against a Natural Earth land polygon rasterized at 0.125°, across (row 0 south vs. north) × (rows increase vs. decrease). Correct convention should show ≥ 85% agreement; alternatives ≤ 60%. Independently confirm with warm-pool location.

**This is a narrower open question than a prior version of this doc implied.** `ocd`'s `going_to` *directional convention* is now confirmed independently of this grid-orientation question (§2.3): `docs/noaa_etl/noaa_transform.md` shows `ocd` is derived directly from `atan2(u, v)` on HYCOM's native `(u, v)` velocity components — the tripolar-to-regular remap (`hycom_index`) only rearranges which array cell a value lives at, it does not rotate the velocity vector itself, so the angle's *meaning* ("going to") holds regardless of how `[row, col]` maps to `(lat, lon)`. What genuinely still depends on this section's unresolved orientation study is purely spatial: which real-world `(lat, lon)` a given `sst`/`ocs`/`ocd` cell represents, needed for masking, reprojection onto the wave grid, and any lat/lon-dependent climatology term.

Encode result in `core/config.py` as `OCEAN_GRID_LAT_AT_ROW0` and `OCEAN_GRID_LAT_STEP_DEG`.

**TASK — do this before writing any ocean-grid-dependent logic.** Add `notebooks/verify_ocean_grid_orientation.ipynb` implementing the method above (Natural Earth correlation, all four `(row 0 south/north) × (increasing/decreasing)` candidates, ≥30 arbitrary days per the original methodology, warm-pool cross-check) and confirm it passes the ≥85%/≤60% thresholds before implementing `Reprojector`'s ocean-grid path or `ClimatologyRemover`/`GaussianCopula` for `sst`/`ocs`/`ocd` (anything that needs to know which lat/lon a cell is). `GoingToToUV`'s `ocd` pair no longer needs to wait on this notebook specifically for its *convention* (that's settled, §2.3) — but it still needs an ocean-cell mask with correct lat/lon labels to be meaningfully composed with the rest of the pipeline, so in practice `ocd` stays gated on this notebook regardless. Once it passes, encode the result in `core/config.py` and close out `tests/core/test_grid_ocean.py::test_orientation` (§12).

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

- **Tucker** compresses each modeled tensor's own spatial modes (lat, lon), leaving time at full rank so DMD sees one state per snapshot (`tucker_decomposition.md` §4.3). One independent masked fit per tensor; there is no variable mode (§3.8, §5.5).
- **DMD** on the Gaussianized core(s) of a variable group extracts the linear deterministic skeleton. Each mode carries its own complex eigenvalue encoding growth rate and frequency. By default every tensor is its own group; a joint group concatenates its members' cores (`correlated_variables.md` §4).
- **OU on residual mode amplitudes** models stochastic wobble around the deterministic skeleton. Closed-form fit and simulation.

### 3.5 Non-Gaussian marginals via Gaussian copula — on the Tucker core (placement ii)

DMD and OU are linear-Gaussian; weather variables are skewed and bounded. Interpose an empirical-CDF Gaussian copula, per coefficient of each tensor's Tucker core:

- **Forward** (training): rank each core coefficient over time → uniform → `Φ⁻¹` → standard normal.
- **Inverse** (sampling): standard normal → `Φ` → uniform → empirical CDF lookup → core coefficient.

**Placement (ii) is DECIDED** (`open_questions.md` §4.1, task 00, task 07 §5); it supersedes v2's per-grid-cell copula before Tucker. The two placements gave statistically indistinguishable Tucker error at equal rank, so the choice is architectural: Tucker, a linear tool, works in physically grounded log-energy anomaly space, and Gaussianization sits exactly where it is needed, right before DMD's Gaussian-innovation assumption. Known limitation unchanged: zero joint tail dependence (`open_questions.md` §4.2). Fix later per §3.13 escalation.

### 3.6 Cross-grid handling: reproject the ocean grid onto the wave grid

**Corrected from a prior version of this doc, which described reprojecting both "wind" and "ocean" grids onto the "wave" grid, treating wind as a third, separately-gridded source.** Per §2.2, wind and wave-state fields come from the same GFS Wave product on the identical grid — there is nothing to reproject between them, only a pass-through. The only genuine cross-grid reconciliation needed is **ocean (HYCOM/RTOFS, 1121×2880, 0.125°) → wave/wind (721×1440, 0.25°)**. Reproject the ocean grid onto the wave grid (bilinear, clipping to `lat ∈ [−75°, +80°]`) as v1. Simpler than fitting a separate ocean model with cross-grid coupling at OU level (§3.13 escalation).

### 3.7 Directional variables via `(u, v)` — two transformers, convention in the pipeline

Never model `θ` directly. Decompose to `(u, v)`, model each as a real field, recombine at sample time via `speed = hypot(u, v)`, `θ = atan2(u, v) mod 360°`.

The archive stores directions in **mixed conventions** (§2.4): most fields are coming-from but `wad` and `ocd` are going-to. Rather than smuggle a per-variable convention flag through a single decoder, the pipeline uses **two atomic transformer classes** that differ only in how they interpret their input angles:

- `ComingFromToUV` — for `(speed, θ_from)` pairs (`wsd`, `pwd`, `pswd`, `wad_come_from`).
- `GoingToToUV` — for `(speed, θ_to)` pairs (`ocd`; `wad` is unused).

Both run before `Reprojector` (§5.2). They act on disjoint variable sets and can be composed in either order. Each has its own `inverse_transform` that reconstructs the variable in its original archive convention. For the wind-sea and primary-swell triplets (class C) the height is also kept as its own `logE` tensor, so the transformer must not drop it (§5.3).

**Why two transformers instead of one decoder that normalizes to a canonical internal convention:**

- The pipeline definition itself shows which variables use which convention. Reading the `Pipeline([...])` line tells you `wad_come_from` is coming-from and `ocd` is going-to. No config dictionary to look up, no decode-time flip that might be forgotten.
- Adding a new directional variable requires an explicit choice of transformer. There is no default, so an agent cannot silently omit the convention decision.
- Sign errors in one transformer cannot hide behind the other; each has its own four-case cardinal-direction test.
- `inverse_transform` naturally reconstructs each variable in its native archive convention, which is what downstream consumers of the archive already expect.

`wad` is not used by any transformer — `wad_come_from` is the source of truth for wind direction. A runtime consistency check on the first loaded snapshot (§5.1) reads both and asserts they differ by 180° mod 360°, guarding against upstream archive convention drift.

### 3.8 Which variables to model — tiered

v3 models the tensors of `variable_taxonomy.md` §5, one independent tensor each (full list with source fields in §10). The tiers keep v2's raw-field grouping:

- **Tier 1 (v0.1.0)**, 7 tensors: `logE_sigwh`; `u_wind`, `v_wind` (from `was`, `wad_come_from`); `logE_wsh`, `T_wsp`, `u_wsea`, `v_wsea` (wind-sea triplet).
- **Tier 2 (v0.2.0)**, 4 tensors: `T_pwp`, `u_pwave`, `v_pwave` (peak wave; `pwh` ≡ `sigwh`, so no `logE` tensor of its own); `sst`, blocked until per-cell ocean-grid coordinates exist (task 16).
- **Tier 3 (v0.3.0+)**, 6 tensors: `logE_pswh`, `T_pswp`, `u_pswell`, `v_pswell` (primary swell); `u_current`, `v_current`, blocked like `sst`.

`wad` ignored throughout — `wad_come_from` is the source of truth for wind direction (§2.4, §3.7). `sigwh` is always loaded, even when not modeled, because it defines the land mask (§5.1).

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

**v3 caveat.** On v3's 12-month window the linear trend is only weakly separable from the annual harmonic and from interannual variability, so `trend_significance_ratio_` decides whether to keep it; the default stays `include_trend=True` pending task 11 (climatology quasi-stationarity).

**Escalation** to interannual index regressors (ENSO, NAO) or rolling climatology deferred to §3.13.

### 3.10 Hierarchical decomposition — the coherent/local split

**Deferred in v3.** The first era-3 model is coherent layer only (`include_local_residual=False`). The local layer has never been fit on real data (task 07 validated only the mode-level OU; task 10 is open), and its definition below assumes v2's per-cell Gaussianized field, which placement (ii) no longer produces. Where it would live under placement (ii) is recorded as open in `open_questions.md` §6.2. The design below is kept for when validation shows the missing small-scale variance matters; at 40% Tucker rank about 23% of the standardized anomaly's norm is outside the coherent layer (task 07 §12).

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
- **TensorLy's `partial_tucker`** natively handles the "compress modes 1 and 2, keep mode 0 (time) full-rank" pattern we need. Its HOOI initialization is slightly better than pure HOSVD for the same rank. **v3 calls a vendored copy**, `train/masked_tucker.partial_tucker_lowmem`: tensorly 0.9.0's masked ALS materializes several full-size temporaries per iteration and silently ignores `svd="randomized_svd"` after initialization. The fix gave about 1.7× per iteration and half the memory (task 07 §5).

The wrappers still add value:

- We return frozen `TuckerFit`/`DMDFit` dataclasses containing only numpy arrays, not tensorly/pydmd class instances. Artifacts serialize cleanly to `.npz` and don't break when tensorly/pydmd versions drift.
- We handle land masking (passed to the masked ALS), dtype coercion, and NaN guards around the calls.
- We keep the algorithmic knobs surfaceable at the `WeatherModel` level (rank choices, DMD variant, etc.).

See §5.5 and §5.6 for implementation sketches.

### 3.12 sklearn hybrid: transformers where they fit, pure functions where they don't

The preprocessing chain (partition sentinel fill → directional → log transform → reproject → climatology → standardize) is sklearn `BaseEstimator + TransformerMixin` classes in one `Pipeline`. Decomposition and stochastic models (Tucker, DMD, mode-OU, diagnostics) are pure functions returning frozen dataclasses. The core copula is still the `GaussianCopula` transformer class, but applied per Tucker core outside the Pipeline (§5.5.1). `WeatherModel` glues them together with the sklearn transformer surface at the top level.

### 3.13 Documented escalation paths

If validation (§6) shows the base model is insufficient:

- **High-frequency variance under-represented in per-cell spectra** → swap `pydmd.DMD` for `pydmd.HankelDMD`. One-line change in `train/dmd.py`.
- **Extremes and joint tail dependence too weak** → replace Gaussian copula with t-copula (single extra df parameter) or splice GPD tails per cell (Coles 2001).
- **Cross-variable coupling too weak** → form a joint variable group (`correlated_variables.md` §4) and check that its coupling information clears the independence null (§5.6.2); an explicit Linear Model of Coregionalization remains a further option.
- **Tau test fails** (§5.6.2; the production `wsh` core already does from 6 h) → longer DMD training lag, `pydmd.HankelDMD` (time-delay embedding), or modeling the fast 3 h component explicitly as noise. None tested yet.
- **Short-lag spatial correlation in residuals missing** → replace independent per-cell OU with SPDE-GMRF local layer (Lindgren–Rue–Lindström 2011). Real research-level change; do only if validation forces it.
- **Interannual climate variability biases the climatology** → add index regressors (ENSO NINO3.4, NAO) to the climatology model, or switch to rolling-window climatology re-fit annually.
- **DMD modes miss slow (multi-year) structure** → try `pydmd.BOPDMD` with `eig_constraints={"imag"}` to constrain modes to be purely oscillatory, forcing slow structure into the climatology where it belongs.
- **Uncertainty quantification on modes needed** → use `pydmd.BOPDMD` regardless. Bagged fit gives per-mode confidence intervals.
- **In-sample hurricane-season reproduction check (§6.A.18) looks weak** → escalate to a true leave-one-year-out retrain: fit on 3 of the 4 training years, hold out the fourth's Aug-Oct entirely, repeat the same storm-signature detection purely out-of-sample. 4× the fit cost (one retrain per held-out year), so only do this if the cheaper in-sample check raises a real concern — don't run it by default for v0.1.0. Needs several training years; impossible on v3's 12-month era-3 window.

### 3.14 Three-package split: `core`, `train`, `weather`

Unchanged from v1. `core/` — shared primitives; `train/` — training pipeline (transformers + pure functions); `weather/` — public API. `weather/` depends on `core` and `train`; `train/` depends on `core` only; `core/` has no internal deps. Consumers import only from `weather/`.

---

## 4. Architecture overview

Deterministic preprocessing per field, per-tensor compression, per-group stochastic dynamics.

```
                          TRAINING (offline, notebook-driven)
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │   ./data/YYYY/MM/DD/HH/*.npy   (v3: 12 months of era 3)            │
   │              │                                                     │
   │              ▼                                                     │
   │   [ingest]   load_training_window(data_dir, t_start, t_end)        │
   │              │                                                     │
   │              ▼                                                     │
   │   ┌── sklearn Pipeline (preprocessing, §5.2–5.4) ────────────┐     │
   │   │ PartitionSentinelFiller  land mask from sigwh; fills     │     │
   │   │                  absent wind-sea / swell partitions      │     │
   │   │ ComingFromToUV, GoingToToUV  (u, v); keeps class-C       │     │
   │   │                              height                      │     │
   │   │ LogTransform             logE_* heights, T_* periods     │     │
   │   │ Reprojector, ClimatologyRemover, PerCellStandardizer     │     │
   │   └──────────┬───────────────────────────────────────────────┘     │
   │              │  standardized anomaly, one (T, lat, lon) per tensor │
   │              ▼                                                     │
   │   ┌── Per-tensor compression (§5.5) ─────────────────────────┐     │
   │   │ tucker_fit (masked, partial_tucker_lowmem) → TuckerFit   │     │
   │   │ GaussianCopula on the core → Gaussianized core           │     │
   │   │     (T, r_lat, r_lon), N(0, 1) marginals                 │     │
   │   └──────────┬───────────────────────────────────────────────┘     │
   │              │  cores concatenated per variable group              │
   │              ▼                                                     │
   │   ┌── Per-group dynamics (§5.6) ─────────────────────────────┐     │
   │   │ dmd_fit → DMDFit;  fit_mode_ou → ModeOUFit               │     │
   │   │ diagnose_group → GroupDiagnostics (§5.6.2), recorded     │     │
   │   │     only; never alters or gates the fit                  │     │
   │   └──────────┬───────────────────────────────────────────────┘     │
   │              │                                                     │
   │              ▼                                                     │
   │   WeatherModel.from_components(preprocess, tucker, core_copula,    │
   │                                dmd, mode_ou, group_diagnostics,    │
   │                                ocean_mask, metadata).save(dir)     │
   └────────────────────────────────────────────────────────────────────┘

                         QUERY (online, batched preferred)
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │   weather.simulate.sample_batch(model, lats, lons, times, vars)    │
   │              │                                                     │
   │              ▼                                                     │
   │   Evolve each group's DMD + mode OU to the query times             │
   │   (closed form; conjugate pairs exact; correlated noise within     │
   │   joint groups)                                                    │
   │              │                                                     │
   │              ▼                                                     │
   │   Split the joint state into per-tensor Gaussianized cores         │
   │              │                                                     │
   │              ▼                                                     │
   │   Copula inverse → core;  Tucker inverse on the cells needed       │
   │              │                                                     │
   │              ▼                                                     │
   │   Per-cell inverses: standardization, climatology, reprojection    │
   │              │                                                     │
   │              ▼                                                     │
   │   Bilinear interpolate (u, v), logE, T at (lat, lon)               │
   │   [padded for lon wraparound]                                      │
   │              │                                                     │
   │              ▼                                                     │
   │   Pointwise: H = exp(logE / 2), period = exp(T),                   │
   │              (u, v) → speed, direction in native convention        │
   │              │                                                     │
   │              ▼                                                     │
   │   values[]                                                         │
   └────────────────────────────────────────────────────────────────────┘
```

**Key architectural properties:**

- The preprocessing chain is one sklearn `Pipeline`. Its `inverse_transform` is the same object as its `transform` — no duplicated inverse logic. It ends at the standardized anomaly; the core copula is applied per tensor after Tucker (§5.5.1).
- Every modeled field is its own `(T, lat, lon)` tensor from preprocessing through Tucker and the copula. The only place fields meet is a variable group's joint DMD state, built from already-compressed core coefficients (`correlated_variables.md` §4).
- Sampling in v3 is the coherent layer only: mode-level OU per group, closed-form transitions, no discrete stepping needed for point queries. The local residual is deferred (§3.10).
- Per-cell statistics (climatology, standardization) are inverted on grid cells before interpolation, and interpolation happens on linear-safe fields only (§5.8).
- `WeatherModel.transform(X)` runs *only* the preprocessing Pipeline and returns standardized anomalies per tensor at native grid resolution. Not cores, not Gaussianized values.
- Group dynamics diagnostics (§5.6.2) are a training stage whose results are stored on the model and logged, but never feed sampling. A failed tau test or significant coupling is input to the next modeling decision, not an error.

---

## 5. Algorithm, step by step

Notation used throughout (v3):

- `T` — training snapshots (2920 for v3's 12-month era-3 window at 3 h).
- `H, W = 721, 1440` — wave grid dimensions (era 3, current/production).
- `v` — one modeled tensor (`logE_wsh`, `u_wind`, …; §10). Every tensor has the same grid shape `(T, H, W)`.
- `M` — the `(H, W)` ocean mask from `sigwh` (§5.1); land cells are `False`.
- `r_lat, r_lon` — a tensor's Tucker ranks; default 40% of `H, W`, i.e. `(288, 576)`.
- `G` — a variable group (a set of tensors fit by one DMD); `n_G` its joint state size (the sum of its members' `r_lat · r_lon`).
- `r_dmd` — DMD truncation rank per group.
- `Δt = 10800 s` (3 hours).
- `t_ref` — climatology reference epoch (midpoint of training window).
- `σ_std(s, v)` — per-cell standardization scale.
- `θ_mode(k)`, `σ_mode(k)`, `φ_mode(k)` — OU parameters per DMD mode.
- `θ_local`, `σ_local`, `φ_local` — local-residual OU parameters (deferred in v3, §5.7).

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
    wad = decode_compass_code(load_raw(data_dir, t, "wad"))
    wad_from = decode_compass_code(load_raw(data_dir, t, "wad_come_from"))
    diff = np.abs(((wad - wad_from - 180.0 + 180.0) % 360.0) - 180.0)   # min circular distance
    max_dev = float(np.nanmax(diff))
    if max_dev > 5.0:   # tolerance for 22.5° discretization + rounding
        raise ValueError(
            f"wad and wad_come_from do not differ by 180° at {t.isoformat()}. "
            f"Max deviation: {max_dev:.2f}°. Archive convention may have changed."
        )
```

Log the max deviation on success. Rerun on the last snapshot too — if the archive convention changed *within* the training window, the second check catches it. Per `docs/noaa_etl/noaa_transform.md`, `wad` is deterministically derived from `wad_come_from` by a `±180°` transform at ETL transform time, so in the current pipeline this check should always pass; it exists as a guard against a future change to that ETL transform, not as evidence the archive is currently noisy.

**Archive-era consistency check, before any file I/O at all (§2.2.1).** The real archive's grid shape and snapshot cadence are not constant over its full history (§2.2.1, `core.archive_eras.ARCHIVE_ERAS`). Before the wad/wad_come_from check above even runs, `load_training_window` resolves `era_for_timestamp(times[0])` and `era_for_timestamp(times[-1])` and raises `ValueError` immediately if they differ — a cross-era window cannot be loaded into one consistent `Dataset` (shape and/or cadence would differ mid-array). Within a confirmed single-era window, each loaded snapshot's shape is further checked against that era's canonical shape and against the window's own first-loaded shape, so an unexpected shape change (corruption, or an undocumented additional era boundary) fails with a clear diagnostic rather than an opaque numpy broadcast error. `core.io.enumerate_snapshots` is itself era-aware (6-hourly within era 1, 3-hourly within eras 2/3) — it does not reject cross-era windows itself (that stays `load_training_window`'s job), it just enumerates each timestamp at its own era's native cadence.

**Implementation prerequisite: `core/io.py` currently refuses to load `wad_come_from` and `ocd` at all.** `_DIRECTION_FIELDS` (an allowlist of decodable direction-field names — no per-field origin code needed now that §2.3 established one uniform decode formula) deliberately omits both (per the v1 design, back when `wad_come_from` was believed to be a redundant duplicate and `ocd`'s convention was unverified), so `load_snapshot` raises `ValueError` for either today. `load_raw` above is *not* `core.io.load_snapshot` — it's a lower-level raw-array read that must be added (or `load_snapshot`'s allowlist extended) before any of §3.7/§5.3's pipeline wiring can run. Add `"wad_come_from"` and `"ocd"` to `_DIRECTION_FIELDS` as part of implementing this section, not as an afterthought.

**Gap handling:** missing file → log to stderr, mark NaN. Gap of 24 h+ → abort. Gap 6–24 h → warn.

**Memory:** one tensor over v3's window is 2920 × 721 × 1440 values: about 12 GB at `float32`, 24 GB at `float64`. Tier 1's seven tensors do not fit in memory together on a 31 GB workstation. Process one tensor at a time through Tucker (§5.5), and use `np.memmap` or 30-day time blocks for the preprocessing pass.

**Ocean mask construction (v3, task 17):** derived from `sigwh`'s own NaN pattern alone: a cell is ocean only if `sigwh` is non-NaN at every training snapshot (`train.masking.land_mask_from_sigwh`, `valid_frac=1.0`). `sigwh` is an aggregate with no partition-absence NaN, so its NaN means land or ice. The partition fields (`wsh`, `pswh` and their periods and directions) are also NaN at genuine ocean cells where that partition is absent, so their own NaN must never define the mask (§5.3.1, `variable_taxonomy.md` §3.3.1). Wind shares the wave grid (§2.2) and the same mask. Stored as `ocean_mask_` on `WeatherModel`: bool `(721, 1440)`.

### 5.2 Reprojection (`Reprojector` transformer)

**Corrected (§3.6): wind needs no reprojection.** Wind (`was`, `wad`/`wad_come_from`) is already on the wave grid — same GFS Wave product, same 721 × 1440 array (§2.2) — so `Reprojector` is a no-op pass-through for wind variables. Only ocean (1121 × 2880, HYCOM/RTOFS) fields are actually resampled, onto the 721 × 1440 wave/wind grid (`lat ∈ [−90°, +90°]`).

```python
class Reprojector(BaseEstimator, TransformerMixin):
    """Reproject the ocean grid onto the wave grid (§3.6, §5.2).

    Wind and wave-state variables share the wave grid already (§2.2) and pass
    through unchanged; only ocean variables (sst, u_current, v_current) are
    actually resampled.
    """
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

**Canonical pair assignment (v3 tensor names, `variable_taxonomy.md` §5):**

`ComingFromToUV`:
- `(wsh, wsd, u_wsea, v_wsea)` — tier 1. Height kept (class C, below).
- `(was, wad_come_from, u_wind, v_wind)` — tier 1. Source of truth for wind; `wad` is not used.
- `(pwh, pwd, u_pwave, v_pwave)` — tier 2. `pwh` is not kept: it is `sigwh` itself (`open_questions.md` §1.4), whose `logE_sigwh` is the authoritative height. `pwd`'s `coming_from` convention is confirmed (NCEP/NDBC, §2.3).
- `(pswh, pswd, u_pswell, v_pswell)` — tier 3. Height kept. `pswd`'s `coming_from` convention is confirmed (NCEP/NDBC, §2.3).

`GoingToToUV`:
- `(ocs, ocd, u_current, v_current)` — tier 3, blocked by task 16.

`u = magnitude · sin θ_to`, `v = magnitude · cos θ_to`, with the pair's own magnitude (`was`, or the partition height). Non-directional variables (`sigwh`, `wsp`, `pwp`, `pswp`, `sst`) pass through both transformers unchanged.

**Class C keeps its height (v3).** For `wsh` and `pswh`, the height must stay in the Dataset as well as producing `(u, v)`, because it becomes its own `logE` tensor (§5.3.1). Both transformers take a `keep_speed: tuple[str, ...] = ()` parameter naming the magnitudes to retain; the inverse then takes the magnitude from the kept variable and only the direction from `(u, v)`, avoiding the `hypot` reconstruction bias (`open_questions.md` §2.3). Otherwise the transformers add `(u, v)` and remove the `(speed, direction)` pair, and leave every other variable as-is, so composition in either order is commutative for disjoint pair sets.

**`inverse_transform` reconstructs each variable in its native archive convention.** `ComingFromToUV` produces `wad_come_from` back on inversion (not `wad`); `GoingToToUV` produces `ocd` as going-to. Downstream consumers get outputs matching the input archive convention.

**Pipeline placement** (tier 1; §8.6 Cell 3):

```python
preprocess = Pipeline([
    ("sentinel", PartitionSentinelFiller(partitions={"wsh": ("wsp", "wsd")})),
    ("coming_from_to_uv", ComingFromToUV(
        pairs=[
            ("wsh", "wsd", "u_wsea", "v_wsea"),
            ("was", "wad_come_from", "u_wind", "v_wind"),
        ],
        keep_speed=("wsh",),
    )),
    ("going_to_to_uv", GoingToToUV(pairs=[])),   # tier 3 adds ("ocs", "ocd", ...)
    ("log", LogTransform(
        heights={"sigwh": "logE_sigwh", "wsh": "logE_wsh"},
        periods={"wsp": "T_wsp"},
        floor_m=0.01,
    )),
    ("reproject",   Reprojector(target_grid="wave")),
    ("climatology", ClimatologyRemover(diurnal_variables=None, include_trend=True)),
    ("standardize", PerCellStandardizer()),
])
```

Even when `GoingToToUV` has an empty pair list (tiers 1–2), the transformer is present in the Pipeline as a scaffold. This makes it obvious to any reader that going-to variables *would* live there, and enabling tier 3 is a matter of adding entries — not adding a new pipeline step.

#### 5.3.1 Partition sentinel fill and log transforms

Two transformers, both new in v3. They implement `variable_taxonomy.md` §3.3.1 and §3.4 and task 17's validated policy (`open_questions.md` §5.6). Both live in `train/masking.py`, next to the array-level helpers they wrap (`land_mask_from_sigwh`, `apply_partition_sentinel_policy`, `height_log_energy`, `period_sentinel_from_archive`); that module stops being helper-only.

**`PartitionSentinelFiller(partitions: dict[str, tuple[str, str]])`** maps each sometimes-absent partition's height to its `(period, direction)` fields: `{"wsh": ("wsp", "wsd"), "pswh": ("pswp", "pswd")}`. It runs first, so later stages never see partition-absence NaN.

- `fit`: the ocean mask from `sigwh` (§5.1), and one period sentinel per partition from `train.masking.period_sentinel_from_archive` on the window's real, paired faint-partition observations. Stored, with its two estimators and their agreement, as fitted attributes.
- `transform`: at ocean cells only, a NaN height becomes `0.0` (the physically correct limit, `H = 4√m₀ → 0`), a NaN period becomes the partition's sentinel, and a NaN direction becomes `0°`. The direction is a placeholder: with zero height, `(u, v)` is `(0, 0)` whatever the angle. Land NaN is left as NaN.
- `inverse_transform`: a reconstructed height below `floor_m` becomes `0.0` (partition absent). Periods and directions pass through; see the open point in §5.8.

**`LogTransform(heights: dict[str, str], periods: dict[str, str], floor_m: float = 0.01)`** maps source variable → tensor name.

- `transform`: `logE = log(max(H, floor_m)²)` (`train.masking.height_log_energy`); `T = log(period)`. The 1 cm floor applies everywhere, not only at sentinel cells, so a real dead-calm `H = 0` stays finite too.
- `inverse_transform`: `H = exp(logE / 2)`, `period = exp(T)`.

Directional `(u, v)` come before the log step because they use the raw height as magnitude. Land cells stay NaN through climatology and standardization (NaN-aware), and are zero-filled and masked only at the Tucker boundary (§5.5).

### 5.4 Climatology and standardization

Two transformers, the last stages of the Pipeline. v2's third stage, the per-cell `GaussianCopula`, moved after Tucker (§3.5, §5.5.1).

**`ClimatologyRemover`** fits per ocean cell per tensor:

```
μ(s, t) = c₀(s) + c_lin(s) · Δτ(t)
       + c₁(s)·cos(ω_y·t) + c₂(s)·sin(ω_y·t)
       + c₃(s)·cos(2·ω_y·t) + c₄(s)·sin(2·ω_y·t)
       [+ c₅(s)·cos(ω_d·t) + c₆(s)·sin(ω_d·t)]     # diurnal
```

where:

- `Δτ(t) = (t − t_ref) / 1_year`, with `t_ref = (t_train_start + t_train_end) / 2`. Normalizing by 1 year gives `c_lin` interpretable units of "change per year." Centering at `t_ref` makes `c_lin` uncorrelated with `c₀`.
- `ω_y = 2π / 365.25 days`, `ω_d = 2π / 1 day`.
- Diurnal terms (`c₅, c₆`) only for the tensors named in `diurnal_variables` (`sst`, `u_wind`, `v_wind` are the candidates). Waves don't have a diurnal cycle at 0.25° global resolution (`open_questions.md` §7.2).

Fit by vectorized OLS, NaN-aware so land cells stay NaN. Fitted attributes, per tensor: the coefficient fields `(n_terms, H, W)`, `t_ref_`, and `c_lin_` for the trend diagnostic.

`__init__` parameters:

```python
def __init__(
    self,
    diurnal_variables: list[str] | None = None,
    include_trend: bool = True,
    trend_significance_threshold: float = 0.1,
) -> None: ...
```

If `include_trend=False`, the design matrix drops the `Δτ` column. Log the fraction of cells where `|c_lin| / σ_std > trend_significance_threshold`; if it's under ~1%, consider dropping the trend term (and see §3.9's v3 caveat).

**`PerCellStandardizer`** divides the anomaly by its per-cell temporal standard deviation:

```
A(s, t) = Ỹ(s, t) − μ(s, t)          # anomaly
A_std(s, t) = A(s, t) / σ_std(s)      # standardized anomaly
```

Fitted attribute `sigma_cell_`: one `(H, W)` field per tensor. Near-zero-variance cells floor the *divisor* only, never `sigma_cell_` itself, which the trend diagnostic needs unmodified. Placement (ii) exposed this: v2's copula used to absorb the blow-up silently (task 00).

`A_std` per tensor is what Tucker consumes (§5.5) and what `WeatherModel.transform` returns.

### 5.5 Per-tensor masked Tucker

Lives in `train/tucker.py`, a fit wrapper over `train/masked_tucker.partial_tucker_lowmem` (§3.11). Not a transformer. Specified by `tucker_decomposition.md` §2.8 and §4.

**Input, per tensor `v`:** the standardized anomaly `A_std` as a `(T, H, W)` float32 array with land cells filled `0.0` (a neutral placeholder, never NaN and never a large sentinel), plus the mask `M` broadcast over time as an int8 `(T, H, W)` array: 1 = ocean, 0 = land. The masked ALS imputes land from the current reconstruction each iteration, so the placeholder's value never enters the fit.

```python
@dataclass(frozen=True, slots=True)
class TuckerFit:
    factor_lat: np.ndarray        # (H, r_lat) float64
    factor_lon: np.ndarray        # (W, r_lon) float64
    reconstruction_error: float   # relative Frobenius error over ocean cells
    n_iter: int                   # ALS iterations run


def tucker_fit(
    x: np.ndarray,                 # (T, H, W) float32, land filled 0.0
    ocean_mask: np.ndarray,        # (H, W) bool
    rank: tuple[int, int],         # (r_lat, r_lon)
    rng: np.random.Generator,
    n_iter_max: int = 300,
    tol: float = 1e-5,
    checkpoint_every: int = 12,
    early_stop_tol: float = 1e-5,
) -> tuple[TuckerFit, np.ndarray]:
    """Masked Tucker on the spatial modes; returns the fit and its (T, r_lat, r_lon) core."""


def fill_land(field: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """(T, H, W) with land NaN → float32, land 0.0; raises if an ocean cell is NaN."""


def tucker_transform(x: np.ndarray, fit: TuckerFit) -> np.ndarray:
    """x ×_lat factor_latᵀ ×_lon factor_lonᵀ → (T, r_lat, r_lon); x from fill_land."""


def tucker_inverse(
    core: np.ndarray,
    fit: TuckerFit,
    rows: np.ndarray | None = None,       # factor_lat rows to evaluate; None = all
    cols: np.ndarray | None = None,       # factor_lon rows to evaluate; None = all
) -> np.ndarray:
    """core ×_lat factor_lat[rows] ×_lon factor_lon[cols] → (T, len(rows), len(cols))."""
```

`fill_land` is the Tucker boundary of §9 gotcha 24, for training and held-out data alike. A NaN at an ocean cell means an upstream stage failed to fill something (the partition sentinel, or a reprojection stencil touching land), so it raises rather than letting the masked ALS impute an ocean value silently. `tucker_inverse` with `rows`/`cols` evaluates a sub-grid for the sampler (§5.8) at a cost proportional to its size.

**Algorithm.** Masked HOOI with time left at full rank (`modes=[1, 2]`): randomized-SVD initialization, then ALS in chunks of `checkpoint_every` iterations, each chunk resuming from the previous `(core, factors)`. Stop at `n_iter_max`, or when the reconstruction error improves by less than `early_stop_tol` from one chunk to the next. These are the settings of task 07's production run (`svd="randomized_svd"`, `svd_mask_repeats=0`). `tucker_fit` returns the ALS's own core, which is consistent with the imputed land values, and that core is what the copula and DMD consume. `tucker_transform` is for held-out data.

**Rank.** `(r_lat, r_lon) = round(tucker_rank_fraction · (H, W))`, default 0.4 → `(288, 576)`. Task 07 decided 40% for `wsh` on era 2 (23.30% reconstruction error, against 27.47% at 30% and 32.90% at 20%). That was a pragmatic, budget-bound choice, not a swept optimum (`open_questions.md` §5.3). v3 applies the same fraction to every tensor until per-tensor evidence exists; ranks need not match across tensors (`tucker_decomposition.md` §4.1).

**Cost — plan compute before training.** Task 07's 40% run on era 2 (a 7.8 GB tensor) took about 300 s per iteration and reached effective convergence at iteration 170 in 11 h, on an 8-vCPU, 64 GB VM. Its 30% run peaked near 30 GB. v3's tensor is 12.1 GB (`2920 × 721 × 1440` at float32), so expect roughly 1.5× the per-iteration time and memory: about 15–20 h and 45–50 GB per tensor. That is beyond a 31 GB workstation, so Tucker runs on a VM with checkpointing (the `notebooks/07_placement2_tucker_on_anomalies.py` pattern). Tier 1's seven tensors are about a week of serial VM time, or parallel across VMs.

#### 5.5.1 Gaussian copula on the Tucker core

Per tensor, a `GaussianCopula(n_bins=copula_lut_bins)` (§3.5; default 500 bins) is fit on that tensor's core. Each `(r_lat, r_lon)` coefficient gets its own empirical CDF over time, and the transform maps every coefficient to a standard normal:

```python
core_ds = xr.Dataset({v: (("time", "core_lat", "core_lon"), core)})   # passes check_canonical
copula = GaussianCopula(n_bins=500).fit(core_ds)
gaussianized_core = copula.transform(core_ds)[v].values               # (T, r_lat, r_lon)
```

The copula is not a Pipeline stage. `WeatherModel` keeps one per tensor (`core_copula_`) and inverts it at sampling before the Tucker inverse. Lookup-table size: `500 × 288 × 576 × 4 B ≈ 330 MB` per tensor. Task 07 fit the production `wsh` copula on the full 40% core in 110 s.

### 5.6 DMD — PyDMD wrapper

Lives in `train/dmd.py`. Not a transformer.

**Input: a variable group's joint state.** For group `G`, flatten each member's Gaussianized core to `(T, r_lat · r_lon)` and concatenate the members along the feature axis (`correlated_variables.md` §4.2): `(T, n_G)`. A single-member group is one tensor's core, `n_G = 288 · 576 = 165,888` at default rank. Snapshot pairs never cross an era boundary, which a single-era training window guarantees. `train.dmd.group_state(cores, members)` builds this state and `split_group_state` inverts it at sampling. `dmd_fit` flattens any trailing shape, so a single `(T, r_lat, r_lon)` core is also accepted.

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
    state: np.ndarray,                 # (T, n_G) — a group's joint state (above)
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
    T = state.shape[0]
    snapshots = state.T                       # PyDMD expects (n_features, n_snapshots)

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

**Rank `r_dmd = 100`** for a single-member group: DECIDED in task 07 (`open_questions.md` §5.3), at the knee between the spectral and marginal generative-fidelity optima. A surrogate test and 8-fold walk-forward CV both showed predictive accuracy is almost rank-insensitive between 5 and 200. A joint group's `r_dmd^G ≤ min(n_G, T − 1)` is chosen per group. Diagnostic: plot `Lambda` on the complex plane.

**Multi-step validity is in question.** At 3 h, the production `wsh` core fails the tau test from 6 h onward (§5.6.2): it stays predictable longer than a 3 h linear-Markov fit implies. Treat DMD's multi-step behaviour as unvalidated until a longer training lag or `variant="hankel"` has been tested against the same check.

**Complex-conjugate pair handling.** With `variant="bop"`, the `eig_constraints={"conjugate_pairs"}` option enforces this at fit time — no manual handling needed. With `variant="exact"` or `"hankel"`, the pairs are still present in the fit but the sampler must force `η_{k'}(t) = conj(η_k(t))` at every step (see §5.6.1 below and §9 gotcha #8). **Default to `variant="exact"` during notebook iteration** (§8.6 Cell 7 re-fits DMD on every rank sweep; BOP-DMD's ~100-model bagging makes that 100× slower per sweep, which is painful across a rank grid). **Switch to `variant="bop"` only for the final production fit**, once ranks are settled — the conjugate-pair constraint and bagged UQ are worth the one-time 100× cost there, but not during exploration.

#### 5.6.1 Mode-level OU on residual amplitudes

Fit per group on the same joint state DMD used (`fit_mode_ou(state, dmd)`). For each DMD mode `k`:

```
a_k(t) = k-th component of (Φ⁺ · x_t)           # measured complex amplitude
â_k(t) = b0_k · Λ_k^t                            # deterministic prediction
η_k(t) = a_k(t) − â_k(t)                         # residual
```

**One real stochastic channel per real mode and per conjugate pair (v3).** Exact DMD on real data returns modes that are real or come in exact complex-conjugate pairs, and a real field needs `η_k'(t) = conj(η_k(t))` exactly, not merely in distribution (`open_questions.md` §5.2). A pair is therefore one stochastic driver. v3 ports the procedure task 07 validated (`notebooks/07_dmd_ou_generative_check.py`), which took the reconstruction's imaginary-to-real ratio from about 1 to machine precision:

- **Real mode `k`:** channel `c(t) = Re η_k(t)`.
- **Conjugate pair `(k, k')`:** channel `c(t) = log|η_k(t)| − mean_t log|η_k(t)|`. Only the magnitude is stochastic, in log space because it is positive with a nonzero mean. The phase is deterministic, `arg η_k(t) = phase_offset_k + t · arg Λ_k`, anchored at the measured phase of the last training step. The partner is the conjugate, so the pair is exactly conjugate by construction.
- **Pairing:** `k'` is a pair partner when `|Λ_k' − conj(Λ_k)| < 1e-6`. An oscillatory mode whose partner the truncation dropped is treated as a real mode, and the imaginary part it contributes to the reconstruction is discarded.

**AR(1) per channel, one joint innovation covariance per group.** Channels are real, so:

```
φ_c = clip( Σ_t c(t) c(t+1) / Σ_t c(t)²,  1e-4, 0.9999 )
θ_c = −log(φ_c) / Δt
ε_c(t) = c(t+1) − φ_c · c(t)                     # one-step innovation
Q = Cov(ε)                                       # (n_c, n_c), all channels of the group
σ_c² = Q_cc · 2θ_c / (1 − φ_c²)                  # continuous-time diffusion, as in v2
```

`Q`'s off-diagonal is the correlated within-group noise of §5.6.2 and `correlated_variables.md` §4.4. In a joint group it carries the cross-member coupling of the innovations. In a single-member group it carries the correlation between that tensor's own modes, which independent per-mode draws discarded.

**Clip `φ` itself, not `|φ|`.** A negative AR(1) coefficient means the channel alternates sign step to step, which is physically distinct from slow mean reversion. Clipping pushes it to the fast-decorrelation floor (`θ` large), deliberately treating alternation as noise-like. The clipped `φ` is also the one the sampler uses, so fractional powers `φ^Δn` stay defined.

**Transitions** (used by the sampler, §5.8). Elementwise over channel pairs `(a, b)`, for any real `n ≥ 0`:

```
c(t + nΔt) = φ^n ⊙ c(t) + w,   w ~ N(0, C(n)),   C(n)_ab = Q_ab · (1 − (φ_a φ_b)^n) / (1 − φ_a φ_b)
P_ab = Q_ab / (1 − φ_a φ_b)                      # stationary covariance
```

`C(n)` and `P` are positive semidefinite (Schur product of `Q` with a positive semidefinite kernel), and `P` is invariant under the transition.

**Real channels draw real noise at the full innovation variance.** Task 07's script drew complex noise for real modes and kept the real part, which halves their innovation variance. v3 does not copy that.

Lives in `train/dmd.py` alongside `dmd_fit`, with `ou_transition`, `ou_stationary_cov` and `mode_amplitudes` (channels → complex mode amplitudes, §5.8 step 1). Returns a `ModeOUFit` frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class ModeOUFit:
    mode: np.ndarray                # (n_c,) int64 — the DMD mode each channel drives
    partner: np.ndarray             # (n_c,) int64 — that mode's conjugate partner; -1 = real channel
    phi: np.ndarray                 # (n_c,) float64 — clipped AR(1) coefficient at dt
    innovation_cov: np.ndarray      # (n_c, n_c) float64 — Q
    log_magnitude_mean: np.ndarray  # (n_c,) float64 — 0 for real channels
    phase_offset: np.ndarray        # (n_c,) float64 — 0 for real channels
    dt: float
    # theta and sigma are derived properties (formulas above)
```

v2's `ModeOUFit(theta, sigma, phi, dt)` held one complex OU per mode with independent noise; `phi` was complex.

#### 5.6.2 Group dynamics diagnostics: linear inverse model, tau test, coupling information

Lives in `train/linear_inverse.py`. The math, interpretation and literature
are in `correlated_variables.md` §4.4 and §7; this section fixes where the
stage sits and what it records.

**What it answers.** DMD plus mode-level OU (§5.6, §5.6.1) is a linear
regression of each Δt step on the previous one, with Gaussian noise. Read
that way it is a Linear Inverse Model (LIM), which comes with checks this
pipeline lacked. The stage runs once per variable group and records two
answers on the model:

1. **Goodness of fit.** Does a first-order linear model with Gaussian noise
   hold at Δt? This is the tau test: the operator fitted at 3 h must
   predict the 6–48 h behaviour.
2. **Information lost.** What would fitting the group's members
   independently throw away? This is Granger information (transfer entropy)
   into each member and the mutual information of the joint innovations,
   in nats per step.

**Where.** Immediately after the cores DMD will consume exist, on the same
Gaussianized cores: after each tensor's own Tucker fit (§5.5) and the copula
on its core (§5.5.1).

**Groups.** The `groups` hyperparameter maps a group name to its member
variables. `None` means one single-member group per modeled variable,
which gives a tau test only, since coupling is zero by construction.
Candidate multi-member groups are in `correlated_variables.md` §4.1.

```python
for name, members in groups.items():
    group_diagnostics[name] = diagnose_group(
        cores={v: gaussianized_core[v] for v in members},  # (T, r_lat, r_lon) each
        ranks={v: lim_rank for v in members},
        dt=window_era.cadence_seconds,
        test_lags=list(tau_test_lags),
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
```

**Defaults.**

| Parameter | Default | Reason |
|---|---|---|
| `lim_rank` | 20 PCs per member | A full-covariance fit needs T ≥ 10 n, where n is the total PCs in the group; the stage warns when that fails. |
| `tau_test_lags` | `(1, 2, 4, 8, 16)` | 3 h to 48 h. |
| `n_bootstrap` | 100 | Sets the envelope's coverage, below. |
| training lag | 1 step | The lag DMD itself uses. |

**Output.** A `GroupDiagnostics` per group with these fields:

- `explained_variance`: core variance retained by each member's PCs (truncation loss)
- `lim`: the `LIMFit` (operator, noise covariance, consistency flags)
- `tau`, `drift_upper`, `ratio_band`, `tau_consistent`: the tau test, its bands and per-lag verdicts
- `coupling`, `granger_null_upper`, `noise_mi_null_upper`, `coupling_significant`, `noise_coupling_significant`: coupling information against an independence null

`diagnostics_summary` renders one group as text for the log and the
notebook.

**Bands.** Each band is the envelope (minimum to maximum) of `n_bootstrap`
parametric-bootstrap replicates as long as the training window:

- **Tau test**: replicates simulated from the fitted LIM, the null "the
  linear-Markov model is right".
- **Coupling**: replicates from a block-diagonal LIM fit member by member,
  the null "members are independent".

Under exchangeability, a correct model lands outside a two-sided envelope
with probability 2/(n+1), about 3% per lag at n = 100, combining the
two-sided ratio check with the one-sided drift check. Interpolated
5th/95th percentiles were rejected: they are noisy at this replicate count,
and their nominal 10% false-alarm rate compounds across lags. One flagged
lag is weak evidence. Flags across consecutive lags, or values far outside
the band, are the signal. The training lag itself is the fit, not a test,
and always reads as consistent.

**Never gates or alters the fit.** The production `wsh` core already fails
the tau test from 6 h onward (`correlated_variables.md` §4.4, "First
application"), so a hard gate would block every fit. The diagnostics are
evidence for the decisions they inform: the training lag, `variant="hankel"`
(§5.6), and group membership (`open_questions.md` §3.2). They also
complement the held-out checks of §6. They are in-sample, and they measure
the regression's assumptions rather than the samples.

**Correlated noise.** Within a group, stochastic forcing is one joint
Gaussian draw with the group's innovation covariance
(`correlated_variables.md` §4.4). The sampler draws it in DMD channel
coordinates, from the covariance `fit_mode_ou` estimates (§5.6.1);
`simulate_lim` is the reference implementation of the same idea in PC
coordinates. This stage estimates the LIM's own covariance and measures
what independent draws would lose; it does not feed the sampler.

**Cost.** Dominated by one principal-component decomposition per member:
the T × T Gram matrix, under a minute at production scale. The bootstrap
adds two times `n_bootstrap` simulations in the small PC space.

### 5.7 Local residual OU field (deferred in v3)

**Not part of the first era-3 model** (`include_local_residual=False`, §3.10). The specification below is v2's, kept for reinstatement. Its shapes use v2's raveled `(N_ocean, V)` layout and its `Z` is v2's per-cell Gaussianized field. Before reinstating it: refit per tensor on native `(T, H, W)` arrays (`ornstein_uhlenbeck_processes.md` §4.1), decide which space it lives in under placement (ii) (`open_questions.md` §6.2), and settle its variance floor (task 10).

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

Free functions in `weather/simulate.py`. They read from the fitted `WeatherModel` and never mutate it.

**Continuous-time point query — the `sample_batch` fast path.** For each unique query time `t_q`, use the closed-form OU transition rather than a grid simulation.

**Step 1: Evolve each group's channels to `t_q`, then build its mode amplitudes** (§5.6.1).
```
Δn = (t_q − t_train_start).total_seconds() / dt      # can be fractional
c_0 ~ N(0, P)                                         # one stationary draw per group and call, at t_train_start
c(t_q) = φ^Δn ⊙ c_0 + w,   w ~ N(0, C(Δn))           # one correlated draw per group and unique time
real channel (mode k):    η_k = c
pair channel (k, k'):     η_k = exp(c + log_magnitude_mean) · exp(i · (phase_offset + Δn · arg Λ_k)),  η_k' = conj(η_k)
a_k(t_q) = b0_k · Λ_k^Δn + η_k
```
Every unique time jumps from the same reference draw with its own transition noise, so each time has the correct marginal, but points at distinct times are not one trajectory. Use `simulate_field` when a mutually consistent trajectory matters. The noise is one correlated draw per group (`Q` of §5.6.1), never independent per mode or per member (§5.6.2). `variant="bop"` already returns exact pairs; the same channel construction applies.

**Step 2: Joint state → per-tensor Gaussianized cores.**
```
x_G(t_q) = Re(Φ_G · a(t_q))                            # (n_G,)
split x_G by member blocks; reshape each to (r_lat, r_lon)
```

**Step 3: Copula inverse, then Tucker inverse on the cells needed.** `core_copula_[v].inverse_transform` maps the Gaussianized core back to the anomaly-space core. The Tucker inverse then gives the standardized anomaly. For point queries, evaluate only the rows and columns of the queries' 2 × 2 bilinear stencils: `A_std[rows, cols] = factor_lat[rows] · core · factor_lon[cols]ᵀ` (`tucker_inverse(core, fit, rows, cols)`). A full-grid inverse costs about 0.7 GFLOP per tensor per time; a single point's stencil costs about 0.3 MFLOP. Land cells are set to NaN here: the Tucker inverse also reconstructs the imputed land values, which are not data.

**Step 4: Per-cell inverses on those cells.** Standardization, then climatology (evaluated at `t_q` against `t_ref`), then reprojection, which is the identity for wave-grid fields. These use per-cell statistics, so they must be inverted on grid cells, before any interpolation (§4). They are the fitted Pipeline's own steps from `reproject` onward (`preprocess_[reproject:]`), so the sampler duplicates no inverse logic. Those transformers hold full-grid fields and index positionally, so the sampler runs them on a full-grid array that is NaN outside the evaluated cells. That costs one elementwise pass over the grid per unique time, far less than a Tucker inverse; unique times are processed in small chunks to bound memory.

**Step 5: Bilinear interpolation in linear-safe space.** Interpolate the tensors (`u`, `v`, `logE`, `T`) at `(lat, lon)`, with longitude wraparound padding (§2.6). Query longitudes are taken modulo 360°. A stencil touching land gives NaN.

**Step 6: Pointwise nonlinear inverses at the query point.** `H = exp(logE / 2)` (below the 1 cm floor → `0.0`, partition absent; §5.3.1), `period = exp(T)`, and `(u, v) → (speed, direction)` in the native convention, with the class-C magnitude from `logE`. These are the Pipeline's steps before `reproject` (`preprocess_[:reproject]`), applied to the interpolated values.

**Why this order.** Interpolating after the nonlinear inverses would interpolate direction angles across the 0°/360° seam and average heights in physical rather than log space. v2's sampler did both.

**Open in v3.** Where the sampled height falls below the floor (partition absent), that partition's direction and period are physically undefined. They are returned as reconstructed, and `variable_taxonomy.md` §3.3.1 has no inverse rule yet.

**Compute cost:** step 1 is `O(n_c³)` per group and unique time (one factorization of `C(Δn)`, `n_c ≤ r_dmd`). Steps 2–3 are `O(n_G · r_dmd)` plus the stencil evaluation. Step 4 is one elementwise grid pass per unique time. Steps 5–6 are `O(N_queries)`. Point queries never pay the full-grid Tucker inverse.

**Full-field simulation on the 3h grid** (`simulate_field`) uses the same steps over the full grid at `t_q ∈ {t_start, t_start + Δt, ...}`. It draws `c` from the stationary `P` at `t_start` and chains the one-step transition (`C(1) = Q`) instead of jumping from a reference time, so the trajectory is mutually consistent. Its steps 5–6 are grid-wise, with no interpolation: the whole fitted Pipeline's `inverse_transform`.

---

## 6. Validation

**Training (v3):** 2025-10-27T00:00Z → 2026-10-27T00:00Z, 12 months of era 3 (§1).
**Held-out metrics window:** 2026-10-27T00:00Z → the last real snapshot (2026-11-21 when v3 was written; it grows as data arrives). Real archive data, used for numerical validation with ground-truth comparison.
**Extrapolation window:** past the last real snapshot. No ground-truth comparison possible. Assessed by internal-consistency and stability criteria only.

The validation battery splits accordingly into two parts. `weather.validate.full_report` runs both and assembles them into a PDF:

```python
full_report(
    model,
    data_dir=Path("./data"),
    metrics_start="2026-10-27T00:00Z",
    metrics_end="2026-11-20T21:00Z",       # last snapshot with real data
    extrapolation_start="2026-11-21T00:00Z",
    extrapolation_end="2027-04-30T21:00Z",  # or wherever the router needs to reach
    output=Path("./reports/v0.1.0_validation.pdf"),
)
```

### 6.A Held-out metrics (2026-10-27 onward)

Real data available. All must pass to consider v0.1.0 ready to ship.

1. **Per-cell marginal QQ.** 30 randomly-chosen ocean cells, per variable, real 2026-H1 vs. simulated 2026-H1. Should be near-diagonal.
2. **Per-cell temporal spectra (Welch PSD).** Same cells, log-log plot. Must match on the `[3h, 30d]` frequency band. Systematic under-representation at high freq → switch DMD variant to `"hankel"` (§3.13).
3. **Long-range empirical variogram** at a fixed snapshot, averaged across snapshots. Real vs. simulated must agree within ±10% at ranges `[100 km, 5000 km]`.
4. **Short-lag variogram** at ranges `[25 km, 200 km]`. Direct test that the local residual layer (§3.10) is doing its job. Real weather has non-negligible variance at 25 km (one grid cell); model samples should too. If short-lag variance is systematically below real, lower `local_residual_variance_floor` or escalate to SPDE-GMRF (§3.13).
5. **Joint density at a single cell.** `(sigwh, was)` and `(u_wind, v_wind)` scatter plots. Correct correlation structure in both magnitude and sign.
6. **Extreme statistics.** 99th percentile of `sigwh` per cell for the metrics window, real vs. simulated. Systematic under-estimate → escalate copula per §3.13. **Note:** the metrics window covers only H1 (winter and spring — no Southern Hemisphere winter storms, no Northern Hemisphere hurricane season peak). This generic percentile check doesn't target hurricane season specifically — §6.D's checks 18-19 do that directly.
7. **Held-out Gaussianized-core distribution.** In v3 `model.transform` returns standardized anomalies, which are not expected to be `N(0, 1)`; the Gaussianized space is the Tucker core (§5.5.1). Apply `model.transform(metrics_window_data)`, project each tensor onto its Tucker factors (`tucker_transform`), apply its core copula, and KS-test each core coefficient against `N(0, 1)`. Systematic non-normality on the validation window → the copula or climatology overfit the training window.
8. **Trend-term diagnostic.** Report `trend_significance_ratio` distribution across ocean cells per variable. Verify median above 0.02 (trend term doing real work) but under 0.5 (trend not overwhelming seasonal cycle). Spatial map for regions where the ratio is high (Mediterranean SST, Arctic SST).
9. **Climatology extrapolation check.** Compare `μ_predicted(2026-04-15)` (from the fitted climatology, extrapolated 4 months past `t_train_end`) against `μ_estimated_from_2026-H1_data_alone`. If the trend term improved the extrapolation (lower RMSE than a no-trend fit), keep `include_trend=True`. If not, drop it.
10. **Local residual variance ratio map.** Report `active_mask` fraction per variable and per basin. If a variable's active mask is < 30% globally, question whether the residual layer is worth its cost for that variable.

### 6.B Extrapolation-only checks (past the last real snapshot)

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
- `was` ≥ 34 knots (tropical-storm force) over a minimum contiguous area, co-located with elevated `sigwh` (wave energy ∝ Hs², so this is the "high energy density" criterion). **Units corrected:** per `docs/noaa_etl/noaa_transform.md`, `was` is stored in **knots**, not m/s — the transform chain applies `ms_to_knots` (factor `1.943845249222`) to the raw GRIB `ws` field. A prior version of this doc used a `17 m/s` threshold directly against `was`'s stored (knots) values, which would silently misclassify ordinary wind as tropical-storm force; the standard NHC tropical-storm threshold of 17.2 m/s converts to ≈33.4 kt, rounded to the conventional 34 kt cutoff used here.
- A coherent cyclonic (positive, NH) vorticity maximum, computed from `(u_wind, v_wind)`, co-located with the patch — the rotational structure that distinguishes an actual cyclone from "it's just windy here," and the specific structure plain EOF+AR can't represent but DMD's complex-conjugate modes should (§3.3's rejection rationale — this check directly stress-tests that choice).
- Tracked across consecutive snapshots, the patch/vorticity-max centroid persists ≥24h and translates at a physically plausible speed (~10-30 km/h) — the "moving front" criterion, distinguishing a genuine storm from a flickering isolated extreme value.

18. **Historical hurricane-season reproduction (in-sample, real ground truth).** For each Aug-Oct in 2022, 2023, 2024, 2025 (all inside the training window — this is an in-sample fit-quality check, not an out-of-sample generalization test; see the leave-one-year-out escalation in §3.13 if it looks weak): run the storm-signature detector on the real archive data to get a per-season baseline (feature count, peak intensity, typical translation speed). Simulate the same windows (multiple seeds) and run the same detector on the output. The simulated distribution across seeds should be statistically consistent with the real per-season baseline — matching the *statistics* of storms across the four seasons, not reproducing individual historical storms (that's explicitly a non-goal, §1). A model that can't even reproduce known hurricane seasons in-sample has no chance of extrapolating them correctly to 2026.
19. **Extrapolation structural check (2026 Aug-Oct, no ground truth).** Simulate the actual production extrapolation window and run the same storm-signature detector. There's nothing real to compare against, so this isn't a statistical-match check like #18 — it's a structural capability check: does at least one detector-qualifying feature (high energy density + coherent rotation + persistent translation) emerge at all during the season when the real North Atlantic reliably produces them? A model that produces zero qualifying features here, despite passing #18 in-sample, indicates the coherent DMD modes responsible for cyclone-like structure aren't extrapolating past the training window — investigate `Re(lam_cont)` for the responsible modes before shipping.

---

## 7. Persisted model artifacts

Directory layout — one dir per model version:

```
./models/<version>/
├── metadata.json            # windows, raw variables, tensors, groups, Tucker ranks, r_dmd,
│                            # DMD variant, git commit hash,
│                            # sklearn / numpy / scipy / tensorly / pydmd versions
├── ocean_mask.npy           # (721, 1440) bool, from sigwh (§5.1)
├── preprocess.pkl           # fitted sklearn Pipeline (§5.3–§5.4), ends at the standardized anomaly
├── tucker/<tensor>.npz      # factor_lat, factor_lon, reconstruction_error, n_iter
├── core_copula/<tensor>.npz # bin_probs, sorted_core_binned (n_bins, r_lat, r_lon)
├── dmd/<group>.npz          # Phi, Lambda, b0, lam_cont, dt, variant, members, member sizes
├── mode_ou/<group>.npz      # mode, partner, phi, innovation_cov, log_magnitude_mean,
│                            # phase_offset, dt (§5.6.1)
├── group_diagnostics.json   # per group (§5.6.2): members, variance retained, tau-test
│                            # lags / drift / ratio / bands / verdicts, coupling nats,
│                            # null bands and significance flags
└── lim/<group>.npz          # per group: G, L, Q, C0, innovation_cov, lag, dt
```

No `local_residual.npz` in v3 (§5.7).

**Why pickle for the preprocessing Pipeline.** sklearn transformers pickle cleanly; joblib.dump is the sklearn-native persistence path.

**Why `.npz` for everything after it.** Pure array bundles, no class structure. Numpy-native, faster to load, cross-version-stable. The core copulas are stored as their fitted arrays, not as pickled `GaussianCopula` objects, and rebuilt on load.

Dtype policy: `float32` for large arrays (`Phi`, the copula lookup tables, the Tucker factors on disk), `float64` for small ones (mode OU parameters, DMD eigenvalues, LIM matrices). Complex: `complex64` for `Phi`, `complex128` for `Lambda`, `b0`, `lam_cont`, `phi_mode`.

Ballpark total size for a tier-1 model (7 tensors): 3–4 GB. Per tensor that is about 330 MB of copula lookup table, about 130 MB of `Phi` (single-member groups), and about 100 MB of climatology and standardization fields in `preprocess.pkl`.

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

from train.copula import GaussianCopula
from train.dmd import DMDFit, ModeOUFit
from train.linear_inverse import GroupDiagnostics
from train.tucker import TuckerFit


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    trained_at: datetime
    t_train_start: datetime
    t_train_end: datetime
    t_ref: datetime                          # climatology reference epoch (mid-window)
    variables: tuple[str, ...]               # raw archive fields loaded
    tensors: tuple[str, ...]                 # modeled tensors (§10)
    groups: dict[str, tuple[str, ...]]       # group name -> member tensors
    tucker_ranks: dict[str, tuple[int, int]] # tensor -> (r_lat, r_lon)
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

    Architecture (v3): per-tensor climatology + standardization, per-tensor
    masked Tucker, Gaussian copula on each core, DMD + mode OU per variable
    group. sklearn-compatible transformer surface.

    Sampling is NOT a method on this class. Use free functions in weather.simulate:
        sample(model, ...)
        sample_batch(model, ...)
        simulate_field(model, ...)

    Fitted attributes (trailing underscore per sklearn convention):
        preprocess_       : sklearn.pipeline.Pipeline (sentinel fill → standardize)
        ocean_mask_       : np.ndarray, shape (721, 1440), bool
        tensor_names_     : tuple[str, ...]
        groups_           : dict[str, tuple[str, ...]]  (resolved from `groups`)
        tucker_           : dict[str, TuckerFit]        (tensor -> fit)
        core_copula_      : dict[str, GaussianCopula]   (tensor -> fitted copula)
        dmd_              : dict[str, DMDFit]           (group -> fit)
        mode_ou_          : dict[str, ModeOUFit]        (group -> fit)
        group_diagnostics_: dict[str, GroupDiagnostics] (group -> §5.6.2 stage output)
        local_residual_   : None in v3 (§5.7)
        metadata_         : ModelMetadata
    """

    def __init__(
        self,
        t_start: datetime,
        t_end: datetime,
        variables: list[str],                        # raw archive fields (§10)
        tucker_rank_fraction: float = 0.4,           # of (H, W), per tensor (§5.5)
        tucker_max_iter: int = 300,
        tucker_tol: float = 1e-5,
        r_dmd: int = 100,
        dmd_variant: Literal["exact", "bop", "hankel"] = "exact",
        copula_lut_bins: int = 500,
        diurnal_variables: list[str] | None = None,
        include_trend: bool = True,
        trend_significance_threshold: float = 0.1,
        include_local_residual: bool = False,        # deferred in v3 (§5.7)
        local_residual_variance_floor: float = 0.02,
        groups: dict[str, list[str]] | None = None,  # None: one group per tensor (§5.6.2)
        lim_rank: int = 20,
        tau_test_lags: tuple[int, ...] = (1, 2, 4, 8, 16),
        n_bootstrap: int = 100,
        random_state: int | None = None,
    ) -> None:
        # sklearn convention: __init__ only stores parameters, no validation.
        ...  # one `self.<name> = <name>` per parameter, verbatim

    # --- sklearn transformer surface ---------------------------------------
    def fit(self, X: Path, y: None = None) -> Self: ...
    def transform(self, X: Path) -> xr.Dataset:
        """Apply preprocessing chain only. Returns standardized anomalies per tensor."""
    def fit_transform(self, X: Path, y: None = None) -> xr.Dataset: ...

    # --- Alternate constructor for notebook assembly -----------------------
    @classmethod
    def from_components(
        cls,
        *,
        preprocess: Pipeline,
        tucker: dict[str, TuckerFit],
        core_copula: dict[str, GaussianCopula],
        dmd: dict[str, DMDFit],
        mode_ou: dict[str, ModeOUFit],
        group_diagnostics: dict[str, GroupDiagnostics],
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

`variables` names raw archive fields. The Pipeline derives the tensors from them (§5.3), and `groups` refers to tensor names. Including `local_residual_variance_floor` keeps the parameter surface stable for when the local residual returns; it is unused in v3.

**`fit` versus the notebook.** `fit(X)` runs the whole chain in-process and is the path for small windows and tests. A production era-3 window needs about 15–20 h of VM time per tensor for Tucker alone (§5.5). Production therefore uses the notebook workflow, with checkpointed Tucker per tensor on a VM, then `from_components` (§8.6).

Default `dmd_variant="exact"` — fast enough for repeated rank sweeps during notebook iteration (§8.6). Switch to `dmd_variant="bop"` only for the final production fit, once ranks are settled: `eig_constraints={"conjugate_pairs"}` then handles the complex-conjugate pair problem automatically and adds bagged uncertainty quantification, at ~100× the fit cost of `"exact"` — worth paying once, not on every sweep.

### 8.6 Notebook workflow (`notebooks/test_pipeline.ipynb`)

Training is a notebook activity. Structure so that expensive stages run once and cheap stages iterate. The notebook is currently parked in `notebooks/tbd/`; rewriting it to this cell structure is part of the v3 implementation.

```
Cell 1  — Setup: imports, paths, SEED, windows (§1).
          T_TRAIN_START = datetime(2025, 10, 27, tzinfo=timezone.utc)
          T_TRAIN_END   = datetime(2026, 10, 27, tzinfo=timezone.utc)   # exclusive
          T_VALID_START = T_TRAIN_END
          T_VALID_END   = <last real snapshot + 3 h>                     # grows with the archive
          VARIABLES = ["sigwh", "wsh", "wsp", "wsd", "was", "wad_come_from"]   # tier 1
          GROUPS = None                                                  # one group per tensor

Cell 2  — Ingest the training window (one era only; §5.1).
          X_raw = load_training_window(data_dir, T_TRAIN_START, T_TRAIN_END, VARIABLES)

Cell 3  — Build and fit the preprocessing Pipeline (§5.3–§5.4).
          preprocess = Pipeline([...], memory=Path("./cache"))   # §5.3 placement
          A_std = preprocess.fit_transform(X_raw)                # one tensor per variable
          ocean_mask = preprocess.named_steps["sentinel"].ocean_mask_
          # log: trend-significant cell fraction; period sentinels and their agreement

Cell 4  — Inspect A_std: unit scale per cell, land NaN where the mask says,
          no NaN at ocean cells (the sentinel fill worked).

Cell 5  — Tucker per tensor (§5.5). Expensive: 15–20 h per tensor on a VM.
          Run notebooks/07_placement2_tucker_on_anomalies.py-style checkpointed jobs,
          then load each TuckerFit and its core here:
          tucker[v], core[v] = load_checkpointed_tucker(v)

Cell 6  — Copula per core (§5.5.1); QQ plots of the Gaussianized cores.
          core_copula[v] = GaussianCopula(n_bins=500).fit(wrap(core[v]))
          gauss_core[v] = core_copula[v].transform(wrap(core[v]))

Cell 7  — DMD + mode OU per group (§5.6). `variant="exact"` here — this cell reruns
          on every rank sweep; BOP-DMD's ~100-model bagging is for the final fit.
          state[g] = concat(gauss_core[v].reshape(T, -1) for v in members(g))
          dmd[g] = dmd_fit(state[g], r_dmd=100, dt=DT, rng=rng, variant="exact")
          mode_ou[g] = fit_mode_ou(state[g], dmd[g])
          plot_eigenvalues(dmd[g])

Cell 7b — Group dynamics diagnostics (§5.6.2): tau test + coupling, per group.
          group_diagnostics = {
              name: diagnose_group({v: gauss_core[v] for v in members},
                                   ranks={v: 20 for v in members}, dt=DT,
                                   test_lags=[1, 2, 4, 8, 16],
                                   n_bootstrap=100, rng=rng)
              for name, members in groups.items()
          }
          for diag in group_diagnostics.values():
              print(diagnostics_summary(diag))
          # NO across consecutive lags → revisit the training lag or try
          # variant="hankel" before trusting DMD's multi-step behaviour.
          # Coupling not significant → the group gains nothing from a joint fit.

Cell 8  — Assemble WeatherModel; save.
          model = WeatherModel.from_components(
              preprocess=preprocess, tucker=tucker, core_copula=core_copula,
              dmd=dmd, mode_ou=mode_ou, group_diagnostics=group_diagnostics,
              ocean_mask=ocean_mask, metadata=metadata,
          )
          model.save(Path("./models/v0.1.0"))

Cell 9  — Validation (§6) on T_VALID_START → T_VALID_END, extrapolation checks after.
          report = full_report(model, data_dir=data_dir,
                               metrics_start=T_VALID_START, metrics_end=T_VALID_END,
                               extrapolation_start=T_VALID_END,
                               extrapolation_end=..., output=Path("./reports/v0.1.0.pdf"))
```

v2's Cell 8 (local residual) is gone in v3 (§5.7).

**Production runs.** v3's production path is the notebook above with `dmd_variant="bop"` in Cell 7 once ranks are settled. The in-process one-liner is for small windows and tests:

```python
model = WeatherModel(
    t_start=datetime(2025, 10, 27, tzinfo=timezone.utc),
    t_end=datetime(2026, 10, 27, tzinfo=timezone.utc),   # exclusive; last training snap
                                                         # is 2026-10-26T21:00Z
    variables=["sigwh", "wsh", "wsp", "wsd", "was", "wad_come_from"],
    tucker_rank_fraction=0.4, r_dmd=100,
    dmd_variant="bop",  # final fit only — see §5.6, §8.5
    random_state=42,
)
model.fit(Path("./data"))
model.save(Path("./models/v0.1.0"))
```

**Important:** `t_end` is *exclusive* — the last training snapshot is `t_end - 3h`. This matches the `enumerate_snapshots(..., t_start, t_end)` convention in §5.1.

Data from 2026-10-27 onward is **not** used for training — it's reserved for the numerical validation battery. As the archive grows, the validation window grows with it. A longer training window is to be revisited at the era-3 18-month checkpoint (§1).

**Pipeline caching.** `Pipeline(memory=Path("./cache"))` transparently caches each transformer's output. Turn caching *off* (delete the cache dir) before production runs — the fitted `preprocess.pkl` shouldn't carry references to a disk cache that other machines don't have.

---

## 9. Gotchas and implementation notes

Ordered by expected pain if missed.

1. **Row 0 = south for both wave and wind grids.** Test explicitly: place a marker at 51.5° N, 0° E and confirm it lands on London in `sigwh`'s NaN mask, not in the Southern Ocean.

2. **Ocean grid orientation is unverified.** Do the correlation study in §2.7 before enabling `sst`/`ocs`/`ocd`. Do not assume by analogy.

3. **Longitude wraparound at 359.75° → 0°.** Any interpolation across this boundary produces NaN without periodic padding. Enforce at the code level.

4. **Directional interpolation must go through `(u, v)`.** Enforce by running **both** `ComingFromToUV` and `GoingToToUV` **before** `Reprojector` in the Pipeline. Every directional variable must belong to exactly one of the two transformers; nothing should reach `Reprojector` as a raw angle.

5. **All direction fields share one decode formula; only native convention differs per field.** A prior version of this doc claimed two different origin codes (`wsd`/`pwd`/`pswd` = code 0 North; `wad`/`wad_come_from`/`ocd` = code 1 North) — wrong; corrected in §2.3 against `docs/noaa_etl/noaa_transform.md`'s `calculate_16degree`. All six fields decode as `(code % 16) * 22.5`, codes 1–16, sector 16 = North. What genuinely differs per field is only the convention: `wsd`/`pwd`/`pswd`/`wad_come_from` are `coming_from`; `wad`/`ocd` are `going_to`. This is handled by keeping the decoder convention-agnostic (§2.3) and pushing convention handling into the two transformer classes (§3.7, §5.3). Also remember the `-1` missing-value sentinel (§2.3) — mask it out before decoding, in every field, not just the ones you happen to test first.

5a. **`wad` is not read by the pipeline; `wad_come_from` is the source of truth.** Adding a pair `("was", "wad", ...)` to `GoingToToUV` would produce mathematically valid samples but bypass the point of using `wad_come_from` (to avoid a flip that might drift). The runtime `wad`/`wad_come_from` consistency check (§5.1) exists to catch upstream drift; do not remove it as "unnecessary." If it ever fails, that's the signal — not a false alarm — that the archive convention changed.

6. **`float16` precision on disk is much coarser than `float64` in memory.** `sigwh` at 8 m has ~0.008 m precision. Not a bug.

7. **Land mask comes from `sigwh` only** (task 17, §5.1). A partition variable's NaN (`wsh`, `pswh`, their periods and directions) also fires where that partition is absent at a genuine ocean cell. Intersecting every modeled variable's NaN, as v2 did, deletes large areas of real ocean.

8. **Complex-conjugate DMD pairs are one stochastic driver.** A real field needs `η_k'(t) = conj(η_k(t))` exactly. Independent per-mode draws make the reconstruction about as imaginary as real (task 07). v3 builds this into the fit: `fit_mode_ou` gives each pair one log-magnitude channel with a deterministic phase, and `mode_amplitudes` sets the partner to the conjugate (§5.6.1, §5.8). The sampler's real part is then exact, not a discard of a large imaginary part. With `dmd_variant="bop"`, `eig_constraints={"conjugate_pairs"}` also enforces exact pairs at fit time; with `"exact"` or `"hankel"`, pairs are matched by eigenvalue within `1e-6`, and a mode whose partner the truncation dropped is treated as real.

9. **Wraparound at year boundaries in the climatology.** Compute `t_since_epoch_days` continuously from `t_ref` (the training-window midpoint); don't reset at Jan 1. The cosines handle periodicity.

10. **Parallelism goes on tensors, modes and seeds — never on grid cells for the coherent layer.** Coherent-layer grid cells couple through Tucker reconstruction. The per-tensor Tucker fits are independent and parallelize across machines (§5.5). Use `joblib.Parallel` for the mode-level OU across independent modes if it ever becomes a bottleneck.

11. **`float16` in arithmetic paths.** Never. Cast to `float64` at the I/O boundary.

12. **Time zones — always UTC.** Naive datetimes raise `ValueError` at the API boundary.

13. **Reference epochs are TWO datetimes, not one.**
    - `t_train_start = 2025-10-27T00:00Z` — beginning of the v3 training window; used by DMD/OU as the time origin for `Λ^{Δn}` and OU closed-form evolution.
    - `t_ref = 2026-04-27T12:00Z` — midpoint of the 2025-10-27 → 2026-10-27 training window, used as the climatology's linear trend origin (`Δτ = (t − t_ref) / 1_year`).
    Compute `t_ref` at fit time as `t_train_start + (t_train_end - t_train_start) / 2`. Do not hardcode; if the training window changes, `t_ref` should shift with it. Both stored in `metadata.json`. All temporal computations at query time reference the correct one — climatology uses `t_ref`, DMD/OU use `t_train_start`. Getting these swapped produces a systematic bias that's hard to spot in QQ plots but shows up loudly in the trend-term diagnostic (§6.A.8).

14. **Physical consistency across variables is NOT enforced.** `wsh ≤ sigwh` etc. can be violated. The Gaussian copula preserves correlations but not physical relationships. Defer projection layer until validation shows it's needed.

15. **The archive has three grid/cadence eras — a training window must stay within one.** See §2.2.1 for the full table (`core.archive_eras.ARCHIVE_ERAS` is the authoritative source). `load_training_window` now rejects any `[t_start, t_end)` spanning more than one era, and `WeatherModel._fit_impl` derives `dt` for `dmd_fit`/`fit_local_residual` from `core.archive_eras.era_for_timestamp(t_start)` rather than a flat constant — training on an era-1-only window with the old flat `SNAPSHOT_CADENCE_SECONDS` would otherwise have silently used the wrong `dt` (3h assumed, 6h actual) in the DMD eigenvalue / OU theta math. `core.config.grid_coordinates` now returns era 3's `lat0`/lon-step for the wave grid unconditionally, for every era (era 1's and era 3's coordinate *labels* were empirically verified in `notebooks/verify_era1_era3_grid_coordinates.ipynb`; era 3's is now wired in, era 1's deliberately is not). Loading an era 1 or era 2 window still mislabels that window's wave-grid latitudes as a result — an accepted tradeoff, not a bug; see `core/config.py`'s module docstring and `tasks/16-archive-grid-resolution-drift.md`.

15. **Nested BLAS threading pitfall.** `sample_batch` must not spawn its own workers. Honor `OMP_NUM_THREADS` etc. set by the caller.

16. **sklearn `clone()` requires `__init__` transparency.** Every parameter passed to `__init__` must be stored on `self` under the same name, unmodified. `sklearn.base.clone(model)` uses `get_params()` (which reads those attributes) to create a fresh unfitted copy. Do validation lazily in `fit`.

17. **Pickle version drift.** `preprocess.pkl` is a pickled sklearn Pipeline. `metadata.json` records tensorly, pydmd, and sklearn versions used at training. `WeatherModel.load` warns if any differ from the runtime — a mismatch on tensorly or pydmd almost always breaks loading.

18. **(Deferred in v3, §5.7.) Local residual assumes cell-independence.** If short-lag variogram check (§6.4) shows short-range spatial correlation in residuals matters, escalate to SPDE-GMRF (§3.13). For v0.1.0 the independence assumption is defensible but should be diagnosed explicitly.

19. **(Deferred in v3, §5.7.) Local residual can over-fit near ice edges.** Cells near the seasonally-varying ice edge alternate between "valid ocean" and "ice" across the training window. The intersection ocean mask (§5.1) should exclude these, but any that slip through will produce spuriously large `σ_local` (the residual there is not really OU-distributed). Diagnostic: plot `σ_local` and check for a suspiciously bright rim around Antarctic/Arctic ice edges.

20. **Trend term can fit noise on short windows.** v3 trains on only 12 months, where this is sharper than in v2's 4-year framing (§3.9 caveat). With a few years of training data, the linear trend estimator is noisy in the presence of interannual variability (ENSO, NAO). The `trend_significance_ratio_` diagnostic in `ClimatologyRemover` reports where this is happening. If the median ratio is above ~0.5 in specific regions, the trend is likely aliasing interannual variability into a fake secular trend; escalate to index-regressed climatology (§3.13).

21. **Don't call tensorly's `partial_tucker` for the masked fit.** v3 uses the vendored `train/masked_tucker.partial_tucker_lowmem`, which fixes tensorly 0.9.0's per-iteration memory blow-up and its ignored `svd=` argument (task 07 §5). It is pinned to the tensorly 0.9 internals it was vendored from (`tensorly>=0.9,<0.10` in `pyproject.toml`); re-verify `tests/train/test_masked_tucker.py` against any tensorly upgrade.

22. **PyDMD snapshot orientation is `(n_features, n_snapshots)`.** Opposite of what we treat internally. One `.T` at the boundary in `dmd_fit`. Getting this wrong produces plausible-looking eigenvalues on the wrong basis.

23. **Interpolate only linear-safe fields, after the per-cell inverses** (§5.8). Interpolating physical values after the full inverse averages direction angles across the 0°/360° seam and heights in physical rather than log space. v2's sampler did both.

24. **Land cells are NaN until the Tucker boundary, then `0.0` plus mask.** Never feed NaN to the masked ALS: it poisons the initial norm. Never use a large sentinel either: the placeholder must be neutral (`tucker_decomposition.md` §4.2). And never confuse the land placeholder with the partition sentinel (§5.3.1), which marks a real "no partition" observation at an ocean cell.

25. **The 1 cm log floor is applied everywhere, not only at sentinel cells.** A real `H = 0` would otherwise give `log 0 = −∞`. At sampling, heights reconstructed below the floor decode to `0.0` (§5.3.1).

26. **The production `wsh` core fails the tau test at 3 h** (§5.6.2). DMD's one-step fit is internally consistent, but its multi-step predictions are not validated. Don't read good one-step CV (task 07) as evidence for multi-day behaviour.

---

## 10. Variable table

Directions decoded per §2.3 (one uniform formula, `(code % 16) * 22.5`) then handled by the transformer for their native convention (§5.3). The "Tensors" column is the internal representation after preprocessing (`variable_taxonomy.md` §5): 17 independent tensors in total. `wave` and `wind` are the same GFS Wave grid (§2.2).

| File | Meaning | On-disk dtype | Units | Source grid | Native conv. | Transformers | Tensors | Tier |
|---|---|---|---|---|---|---|---|---|
| `sigwh.npy` | Significant wave height | float16 | m | GFS Wave | — | `LogTransform` | `logE_sigwh`; also defines the land mask (§5.1) | 1 |
| `wsh.npy` | Wind-sea wave height | float16 | m | GFS Wave | — | `PartitionSentinelFiller`, `ComingFromToUV` (kept), `LogTransform` | `logE_wsh`; magnitude of `(u_wsea, v_wsea)` | 1 |
| `wsp.npy` | Wind-sea mean period (`T_m01`) | float16 | s | GFS Wave | — | `PartitionSentinelFiller`, `LogTransform` | `T_wsp` | 1 |
| `wsd.npy` | Wind-sea direction | int8 (1–16) | — | GFS Wave | `coming_from` (confirmed — NCEP/NDBC) | `ComingFromToUV` | `u_wsea`, `v_wsea` | 1 |
| `was.npy` | Wind speed at surface | float16 | **knots** | GFS Wave | — | `ComingFromToUV` | magnitude of `(u_wind, v_wind)` | 1 |
| `wad_come_from.npy` | Wind direction, coming from | int8 (1–16) | — | GFS Wave | `coming_from` (confirmed — raw `wdir`) | `ComingFromToUV` | `u_wind`, `v_wind` (knots) | 1 |
| `wad.npy` | Wind direction, going to | int8 (1–16) | — | GFS Wave | `going_to` (deliberate ±180° flip of `wad_come_from`) | consistency check only (§5.1) | **not modeled** | — |
| `pwh.npy` | Peak wave height (≡ `sigwh`, `open_questions.md` §1.4) | float16 | m | GFS Wave | — | `ComingFromToUV` (not kept) | magnitude of `(u_pwave, v_pwave)` only; no tensor of its own | 2 |
| `pwp.npy` | Peak wave mean period | float16 | s | GFS Wave | — | `LogTransform` | `T_pwp` | 2 |
| `pwd.npy` | Peak wave direction | int8 (1–16) | — | GFS Wave | `coming_from` (confirmed — NCEP/NDBC) | `ComingFromToUV` | `u_pwave`, `v_pwave` | 2 |
| `sst.npy` | Sea surface temperature | float16 | verify K/°C (no conversion applied by the ETL) | Ocean (HYCOM/RTOFS) | — | `Reprojector` | `sst` (no log); blocked by task 16 | 2 |
| `pswh.npy` | Primary swell height | float16 | m | GFS Wave | — | `PartitionSentinelFiller`, `ComingFromToUV` (kept), `LogTransform` | `logE_pswh`; magnitude of `(u_pswell, v_pswell)` | 3 |
| `pswp.npy` | Primary swell mean period | float16 | s | GFS Wave | — | `PartitionSentinelFiller`, `LogTransform` | `T_pswp` | 3 |
| `pswd.npy` | Primary swell direction | int8 (1–16) | — | GFS Wave | `coming_from` (confirmed — NCEP/NDBC) | `ComingFromToUV` | `u_pswell`, `v_pswell` | 3 |
| `ocs.npy` | Ocean current speed | float16 | **knots** | Ocean (HYCOM/RTOFS) | — | `GoingToToUV`, `Reprojector` | magnitude of `(u_current, v_current)` | 3 |
| `ocd.npy` | Ocean current direction | int8 (1–16) | — | Ocean (HYCOM/RTOFS) | `going_to` (confirmed — NDBC ocean-current convention, §2.3) | `GoingToToUV`, `Reprojector` | `u_current`, `v_current`; blocked by task 16 | 3 |

Tensor count by tier: 7 (tier 1), +4 (tier 2), +6 (tier 3) = 17.

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
2. A working `notebooks/test_pipeline.ipynb` that produces a fitted `WeatherModel` on the era-3 training window (§1) and a validation PDF against the held-out data that follows it, end-to-end.
3. The public API from §8.4, with docstrings and full type hints.
4. Green CI on Python 3.14: `black --check src/ tests/`, `isort --check-only src/ tests/`, `mypy --strict src/`, `pytest --cov=src` (coverage ≥ 85%).
5. Unit test coverage:
   - `tests/core/test_io.py::test_direction_decode_uniform_formula` — known code → degree pairs for all six direction fields against the single `(code % 16) * 22.5` formula (§2.3); confirms `wsd`/`pwd`/`pswd`/`wad`/`wad_come_from`/`ocd` all decode identically.
   - `tests/core/test_io.py::test_direction_decode_missing_sentinel` — `-1` raises/is masked, not silently decoded to `337.5°` (§2.3).
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
   - `pwd`/`pswd` convention is now confirmed against NOAA's own NCEP GRIB2 Code Table 4.2-0-2 and NDBC documentation (§2.3, via `docs/noaa_etl/noaa_transform.md`), so the `test_pwd_convention_expert_verified` / `test_pswd_convention_expert_verified` gating tests a prior version of this doc required are no longer blocking — write ordinary regression tests (e.g. `tests/train/test_directional.py::test_pwd_pswd_cardinal_signs`, reusing §2.4's coming-from cardinal-direction cases) instead of a "pending confirmation" gate.
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
