# Task 16 — Implement real per-cell ocean-grid coordinates

**Depends on:** — · **Blocks:** loading `sst`, `ocs`, or `ocd` through
`load_training_window` at all, and therefore any rank-selection or other
notebook work on those three variables (none of tasks 00–15 need them
today, but task 07's per-variable rank sweep does once it reaches `sst`)
**Source:** Discovered 2026-08-16 while extending task 07's rank-selection
notebook to `sst` (the other Class A scalar per `variable_taxonomy.md`
§3.1). Not one of `docs/open_questions.md`'s 15 tracked design questions —
this is a missing implementation, not an open design question. Directly
related to `open_questions.md` §1.1, which resolved *what* the ocean
grid's real coordinates are but never wired that answer into loadable code.
**Primary source files:** `src/core/config.py` (`grid_coordinates`,
`OCEAN_GRID_LAT_AT_ROW0`), `src/train/ingest.py`

## 1. What problem this will fix

`grid_coordinates("ocean")` raises `ValueError` unconditionally, and
`load_training_window` propagates that immediately for any request
touching `sst`, `ocs`, or `ocd` — those three variables cannot be loaded
through the standard pipeline at all right now:

```
ValueError: grid 'ocean' is not supported; the ocean grid's orientation is
confirmed (see docs/open_questions.md §1.1) but it is not a linear
lat0+step grid (Mercator south of ~47°N, tripolar patch north of it) --
real per-cell 2D coordinate data is required and not yet implemented.
```

This isn't an unknown-orientation problem — `open_questions.md` §1.1
already did the hard work: it fetched a real RTOFS NetCDF diagnostic file
(`rtofs.20240127/rtofs_glo_2ds_f000_diag.nc`, public/unauthenticated S3),
which carries CF-convention `Latitude(Y,X)`/`Longitude(Y,X)` 2D coordinate
arrays exactly matching `hycom_satocean_idx.npy`'s index bounds. The remap
`real_lat = Latitude[lat_idx, lon_idx]`, `real_lon = Longitude[lat_idx,
lon_idx] % 360` gives an exact geo-reference for every cell of the
`(1121, 2880)` ocean output grid. That derivation exists in the docs; it
was never turned into code.

## 2. Steps to fix it

1. Obtain (or re-derive, per `open_questions.md` §1.1's method) the full
   `(1121, 2880, 2)` per-cell `(lat, lon)` array via the
   `hycom_satocean_idx.npy` -> RTOFS `Latitude`/`Longitude` remap, and
   decide where it's stored/cached (a checked-in array is likely too
   large; consider a build step or a cached download).
2. Extend `core.config` (or a new module, since this is 2D coordinate
   data, not a `lat0 + step` formula like the other two grids) with a
   function that returns real per-cell `(lat, lon)` for the ocean grid —
   can't reuse `grid_coordinates`'s 1D-array contract as-is.
3. Wire this into whatever `train.ingest`/`core.io` path currently expects
   1D `lat`/`lon` coordinates for the wave/wind grids — the ocean grid's
   2D coordinate arrays need different `xr.Dataset` handling (a simple
   `sel(lat=slice(...), lon=slice(...))` only works for axis-aligned 1D
   coordinates; Mercator + tripolar means an ocean-grid crop is a 2D
   index-space crop, not a lat/lon-space one, at least north of ~47°N).
4. Add tests confirming `sst`/`ocs`/`ocd` load correctly with real
   coordinates attached, and that a spot-checked cell's `(lat, lon)`
   matches `open_questions.md` §1.1's own worked examples (e.g. row 0 near
   -65.00°, row 560 spanning 0.00°-359.84°).

## 3. Notebook(s) needed

None currently planned — this is `src/` implementation work. Once done,
task 07's rank-selection notebook can be extended to `sst` (Class A, per
`variable_taxonomy.md` §3.1) and, if a wind/current joint-group task needs
it, `ocs`/`ocd` (Class B).

## 4. Conclusion

_(Blank. Fill in once implemented and verified against real archive data.)_
