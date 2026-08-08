# `noaa_transform` — Raw-to-Standardized Variable Transformation

**Module:** [src/task/noaa_transform.py](../src/task/noaa_transform.py)

## Task Description

Reads the raw GRIB2 (GFS Wave) and binary HYCOM archive (RTOFS) files produced by the
ingestion stage, decodes them, applies unit conversions and grid/format
standardization, and writes out one small `.npy` array per physical variable. This
turns heterogeneous, model-native binary formats into a uniform set of NumPy arrays
that the rest of the system (and downstream consumers) can read without needing
GRIB/HYCOM-specific decoders.

Processing for each file type is delegated to a per-model transform function:
- GFS: `gfs_transform_file` ([src/util/gfs/gfs_transform.py](../src/util/gfs/gfs_transform.py))
- RTOFS/HYCOM: `hycom_transform_file` ([src/util/hycom/hycom_transform.py](../src/util/hycom/hycom_transform.py))

Both run under a shared concurrency-bounded task pool, and a JSON summary of
success/failure per file is uploaded to S3 at the end.

## I/O

**Inputs (parameters):** `bucket`, `download_date`, `max_concurrent_tasks` (default 5).

**Inputs (S3, from ingestion stage):**
- `raw/{yyyy}/{mm}/{dd}/gfs/**` — GFS Wave GRIB2 files, listed via `list_files_s3`.
- `raw/{yyyy}/{mm}/{dd}/rtofs/**` — RTOFS `.a.tgz`/`.b` pairs; `clean_suffixes()`
  strips the `.a.tgz`/`.b` extensions and de-duplicates so each base file is
  transformed once (the `.a.tgz` and `.b` are read together as a pair).

**Outputs (S3 artifacts):**
- GFS: `transform/{yyyy}/{mm}/{dd}/gfs/{cycle}/{variable}-f{fff}.npy` — 13 variables
  per file (see table below).
- RTOFS: `transform/{yyyy}/{mm}/{dd}/rtofs/{variable}-{n00|f{hh}|n-{hh}}.npy` — 3
  variables per file (`sst`, `ocs`, `ocd`).
- Summary artifact: `summary/{yyyy}/{mm}/{dd}/transform_summary.json`.

### GFS variable map (short name → stored name)

| GRIB `shortName` | Stored as | Meaning | Transform chain |
|---|---|---|---|
| `ws` | `was` | Wind speed | flip lat, m/s→knots, round 2dp |
| `wdir` | `wad_come_from` | Wind direction (16-pt), **raw GRIB encoding, unconverted** despite the name | flip lat, 16-sector bucket |
| `shww` | `wsh` | Wind-wave significant height | flip lat, round 2dp |
| `perpw` | `pwp` | Primary wave period | flip lat, round 2dp |
| `dirpw` | `pwd` | Primary wave direction (16-pt), raw GRIB encoding | flip lat, 16-sector bucket |
| `wdir` | `wad` | Wind direction (16-pt), converted to **"coming from"** convention | flip lat, coming-from + 16-sector |
| `mpww` | `wsp` | Wind-wave mean period | flip lat, round 2dp |
| `wvdir` | `wsd` | Wind-wave direction (16-pt) | flip lat, 16-sector bucket |
| `shts` | `pswh` | Primary swell significant height | flip lat, round 2dp |
| `swdir` | `pswd` | Primary swell direction (16-pt) | flip lat, 16-sector bucket |
| `mpts` | `pswp` | Primary swell mean period | flip lat, round 2dp |
| `swh` | `sigwh` | Significant wave height (combined) | flip lat, round 2dp |
| `swh` | `pwh` | Significant/primary wave height (duplicate of `sigwh`, kept for naming compatibility) | flip lat, round 2dp |

### HYCOM variable map

| Source field(s) | Stored as | Meaning | Transform chain |
|---|---|---|---|
| `temp` (layer 0) | `sst` | Sea surface temperature | HYCOM grid remap |
| `u-vel.`, `v-vel.` (layer 0) → magnitude | `ocs` | Ocean current speed | remap, m/s→knots, round 2dp |
| `u-vel.`, `v-vel.` (layer 0) → angle | `ocd` | Ocean current direction (16-pt) | remap, 16-sector bucket |

## Storage & Grid Conventions

These conventions govern how to interpret a raw index `[i, j]` (or `[layer, i, j]`)
in any of the `.npy` artifacts produced by this stage.

### Array indexing → lat/lon (GFS)

- GFS Wave `.npy` arrays are 2-D, indexed `[lat_row, lon_col]`, shape `(721, 1440)`,
  following NOAA's standard published grid definition for `gfswave.global.0p25`
  products (`Ni=1440`, `Nj=721`, 0.25° spacing, `La1=90°`, `Lo1=0°`,
  `La2=-90°`, `Lo2=359.75°`). This is the documented NOAA/NOMADS grid spec, not a
  value independently verified against a bundled sample file in this repo (no GRIB
  fixtures are checked in) — treat it as authoritative but re-confirm against NOAA's
  grid definition docs if exact alignment is critical.
- As delivered by `pygrib`, row 0 is the **north pole (90°N)** and row index
  increases southward. **`flip_latitude` reverses this** for every GFS variable, so
  in the *stored* `.npy` array:
  - `[0, 0]` = **lat −90° (South Pole), lon 0°E**
  - increasing row index → increasing latitude (south → north)
  - increasing column index → increasing longitude, **eastward, in `[0°, 360°)`**
    (not the `[-180°, 180°]` convention)
  - Formula: `lat = -90 + i * 0.25`, `lon = j * 0.25` for row `i`, column `j`.

### Array indexing → lat/lon (HYCOM / RTOFS)

- HYCOM-derived `.npy` arrays (`sst`, `ocs`, `ocd`) are 2-D, shape `(1121, 2880)`,
  produced by `hycom_index` remapping the native HYCOM tripolar grid through the
  precomputed lookup table `hycom_satocean_idx.npy` (shape `(1121, 2880, 2)`, storing
  `(src_lat_index, src_lon_index)` per output cell).
- **Important caveat:** unlike GFS, this remap does **not** go through
  `flip_latitude` — the row/column orientation of the output is whatever was baked
  into `hycom_satocean_idx.npy` when it was generated. This repository only contains
  the index lookup table, not the geographic (lat/lon-in-degrees) definition of the
  output grid it targets, so the exact real-world coordinates of `[0, 0]` **cannot be
  derived from this codebase alone**. The array name (`satocean`) implies it aligns
  to an external satellite/ocean reference grid shared with the consuming weather
  service — confirm the authoritative grid definition there before assuming exact
  bounds.

#### Why a remap is needed: HYCOM's native grid is tripolar

RTOFS/HYCOM does **not** decode into a plain regular lat/lon grid — `read_hycom_fields`
(in [hycom_transform.py](../src/util/hycom/hycom_transform.py)) reads `lon_size`/`lat_size`
straight out of the `.b` metadata file and reshapes the raw `.a` float32 buffer into a
`(lat_size, lon_size)` array using **whatever native grid HYCOM was run on** — for the
global configuration this repo targets, that grid is a **tripolar** mesh:

- One true pole at the **South Pole**, using an ordinary regular lat/lon mesh there.
- The Northern Hemisphere's singularity at 90°N is replaced by **two** displaced poles
  placed over land (avoiding a coordinate singularity over open Arctic ocean) — hence
  "tri-polar" (1 real + 2 displaced poles), not "bipolar."
- The practical consequence: rows/columns in the array HYCOM hands back do **not**
  correspond to a uniform `lat = f(row)`, `lon = f(col)` formula the way the GFS grid
  does. A row near the Arctic in the native array can trace a curved path across real
  latitude/longitude, since it's warped around the two displaced poles.

`hycom_index` (in [grid_utils.py](../src/util/geo/grid_utils.py)) exists specifically to
undo this: `HYCOM_IDX[y, x] = (src_lat_index, src_lon_index)` is a precomputed
nearest-neighbor lookup, generated once (outside this codebase) by comparing the native
tripolar mesh's real-world lat/lon to a target regular grid. Applying it
(`grid_arr[lat_idx, lon_idx]` then reshape) resamples the native, geometrically-warped
HYCOM array onto the uniform `(1121, 2880)` output grid used for every stored HYCOM
variable — this is the step that makes HYCOM data indexable the same simple way as GFS
data, without needing a tripolar-aware consumer downstream.

Concrete evidence from inspecting `hycom_satocean_idx.npy` (`shape (1121, 2880, 2)`,
`dtype int16`):

- `lat_idx` (channel 0) ranges `0–3297`, `lon_idx` (channel 1) ranges `0–4498` — bounds
  that closely match RTOFS/HYCOM's published global 1/12° native grid dimensions,
  confirming `HYCOM_IDX` really does index into the full native tripolar mesh rather
  than an already-regular grid.
- Both indices increase **monotonically but non-linearly** across output rows/columns
  (e.g. sampled `lat_idx` along column 0 goes `0, 425, 425, 425, 425, …` near the south
  edge and `0, 2847, 2848, 2848, 2848, …` near the north edge) — the non-uniform step
  size between output rows is itself a footprint of resampling a warped (tripolar)
  source mesh onto an evenly-spaced target grid; a simple regular source grid would
  produce a constant step instead.
- The recurring `0` index at the very edge of each sampled row/column is most likely a
  fill/sentinel value for output cells with no valid corresponding source cell (e.g.
  land mask or grid edge), not a real coordinate — treat `lat_idx == 0` /
  `lon_idx == 0` at output-grid edges with suspicion rather than as a literal
  south-pole/prime-meridian reference.

This repo does not encode the exact displaced-pole latitudes/placement of the native
grid (that's a property of the upstream HYCOM/RTOFS grid definition, generated
elsewhere), but it does contain and apply the full mechanism (`HYCOM_IDX` +
`hycom_index`) that resolves the tripolar geometry into a usable, uniform output.

#### How `hycom_transform_file` turns this into the S3 artifacts

1. Download `{s3_base}.a.tgz` and `{s3_base}.b` from the `raw/` prefix, extract the
   single member of the tarball to get the raw `.a` bytes.
2. `read_hycom_fields` decodes `temp`, `u-vel.`, `v-vel.` (layer 0 only) from the
   **native tripolar-shaped** array using the dimensions declared in the `.b` file.
3. `compute_magnitude_and_y_based_angle(u, v)` computes ocean current speed/direction
   **on the native grid**, before any remap — `ocs`/`ocd` are still tripolar-shaped at
   this point.
4. `VARIABLE_TRANSFORMATION` drives the rest, per variable:
   - `sst`: `hycom_index` only (tripolar → uniform `(1121, 2880)` remap).
   - `ocs`: `hycom_index` → `ms_to_knots` → `round_2`.
   - `ocd`: `hycom_index` → `calculate_16degree` (16-sector bucket).
   So the tripolar-to-uniform remap (`hycom_index`) is always the **first**
   transformation applied — every later step (unit conversion, rounding, direction
   bucketing) operates on the already-regularized grid.
5. Each resulting array is uploaded via `npy_stream_async` to
   `transform/{yyyy}/{mm}/{dd}/rtofs/{variable}-{n00|f{hh}|n-{hh}}.npy`.

### Direction field encoding

All direction variables are stored as `np.int8` **16-point compass sectors**, not
degrees. Two independent conventions apply and must not be conflated:

1. **"Coming from" vs. "going to":** confirmed against NOAA's authoritative
   references — [NCEP GRIB2 Code Table 4.2-0-2](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-2.shtml)
   defines the raw `WDIR` parameter explicitly as **"Wind Direction (from which
   blowing)"**, and [NDBC's measurement glossary](https://www.ndbc.noaa.gov/faq/measdes.shtml)
   defines wave-direction parameters (`MWD`/`SwD`/`WWD`, the buoy analogs of
   `dirpw`/`swdir`/`wvdir`) the same way: *"the direction from which the waves
   ... are coming."* Ocean **current** direction is the one exception — NDBC
   defines it as *"the direction the current is flowing **toward**."* So:

   | Stored variable | Source GRIB field | Conversion applied? | Resulting convention |
   |---|---|---|---|
   | `wad_come_from` | `wdir` | No — `calculate_16degree` only | "Coming from" (raw `wdir` is already this way per NCEP Table 4.2-0-2 — name is accurate) |
   | `pwd`, `wsd`, `pswd` | `dirpw`, `wvdir`, `swdir` | No — `calculate_16degree` only | "Coming from" (same NOAA/NDBC convention as wind direction) |
   | `wad` | `wdir` | Yes — `vectorize_16degree` applies `transform_directions_to_coming_from_vectorized` (`±180°`) on top of the already-"coming from" raw value | "Going to" — the one field in this set that's flipped relative to the meteorological standard, likely provided so wind arrows can be drawn using the same "toward" convention as `ocd` |
   | `ocd` (HYCOM) | derived from `u-vel.`/`v-vel.` | N/A — derived directly from the velocity vector's bearing, never passed through the conversion | "Going to" — matches NDBC's ocean-current convention (current direction is reported as where it flows toward, not where it originates), since `atan2(u, v)` inherently yields the vector's own heading |

   So despite initial appearances, `wad_come_from` **is** correctly named, and
   `wad` is the one that deviates from the plain-GRIB convention (by design,
   presumably for consistency with `ocd`'s "going to" convention when overlaying
   wind and current arrows on the same map).

2. **16-sector bucketing (`calculate_16degree`):** continuous degrees `[0°, 360°]`
   are mapped to integer sectors `1–16`, each spanning 22.5°, centered on
   `sector * 22.5°`, with **sector 16 = North** (wrapping across the 0°/360°
   boundary) rather than sector 1:

   | Sector | Center | Compass | Sector | Center | Compass |
   |---|---|---|---|---|---|
   | 1 | 22.5° | NNE | 9 | 202.5° | SSW |
   | 2 | 45.0° | NE | 10 | 225.0° | SW |
   | 3 | 67.5° | ENE | 11 | 247.5° | WSW |
   | 4 | 90.0° | E | 12 | 270.0° | W |
   | 5 | 112.5° | ESE | 13 | 292.5° | WNW |
   | 6 | 135.0° | SE | 14 | 315.0° | NW |
   | 7 | 157.5° | SSE | 15 | 337.5° | NNW |
   | 8 | 180.0° | S | 16 | 0.0°/360.0° | N |

   - **Missing/NaN sentinel:** `-1` (a valid, in-range value for the `np.int8`
     store type — must be filtered out by consumers, not mistaken for a real sector).

## Physics Involved

- **Latitude convention:** GFS grids are flipped (`flip_latitude`) so downstream
  consumers get a consistent north-to-south or south-to-north row order.
- **Unit conversion:** wind/current speeds are converted from m/s to knots
  (`ms_to_knots`, factor `1.943845249222`).
- **Directional convention:** Meteorological direction fields (`wdir`) encode where
  wind is blowing *toward*; `vectorize_16degree` first converts this to the
  "coming from" convention (`dir + 180° mod 360`) before bucketing into 16 compass
  sectors of 22.5° each (`calculate_16degree`).
- **Vector decomposition (HYCOM):** ocean current speed/direction are derived from
  the raw `u-vel.`/`v-vel.` (east/north) components:
  - Speed: $\text{magnitude} = \sqrt{u^2 + v^2}$ (`np.hypot`)
  - Direction (from north, clockwise): $\theta = (\deg(\operatorname{atan2}(u, v)) + 360) \bmod 360$,
    undefined (`NaN`) when $u = v = 0$.
- **Grid remapping:** HYCOM's native curvilinear/tripolar grid is remapped onto a
  fixed output grid using a precomputed nearest-neighbor lookup table
  (`hycom_index`, backed by `hycom_satocean_idx.npy`, shape `(1121, 2880, 2)`)
  mapping each output cell to a `(lat_idx, lon_idx)` in the source grid.
- **Binary decoding (HYCOM):** `.a` files are raw big-endian `float32` layers padded
  to 4096-element boundaries per layer; layer offsets/sizes are read from the
  companion `.b` text metadata file. Sentinel fill values (`> 2**99`) are replaced
  with `NaN`.

## Place in Pipeline

**Stage 2** of the ETL, run after `noaa_ingestion` and before `noaa_update_data`.
Consumes `raw/` and produces `transform/`. See [pipeline.md](./pipeline.md).

## Extra Information

- GFS decoding uses `pygrib` to open GRIB2 byte streams and select fields by
  `shortName`; masked/missing values are filled with `NaN`.
- HYCOM decoding manually parses the `.b` text header (locating variable line
  offsets) and slices the `.a` binary payload accordingly; the `.a.tgz` is a
  single-member gzip tarball extracted in memory.
- Both transform paths stream results out via `npy_stream_async` +
  `multipart_stream_to_s3` (chunked `.npy` serialization, no full-file buffering).
- The HYCOM path explicitly calls `gc.collect()` after freeing large intermediate
  arrays, since raw multi-layer HYCOM fields can be large.
- Forecast/timestamp suffix (`f{fff}` for GFS, `n00`/`f{hh}`/`n-{hh}` for RTOFS) is
  parsed from the source file name and re-used in the transformed file name so the
  update stage can resolve it back to an absolute datetime.
