# `noaa_update_data` — Canonical Time-Indexed Publication

**Module:** [src/task/noaa_update_data.py](../src/task/noaa_update_data.py)

## Task Description

Publishes the best available transformed forecast/analysis file for each variable and
target datetime into a canonical, time-indexed `process/` tree — the form consumed by
downstream services. It does not recompute or re-derive data; it resolves *which*
already-transformed file corresponds to a given absolute UTC timestamp and copies it
into place (S3 server-side copy, no re-download/re-upload of content).

Two resolution strategies are used:
- **GFS** — for each of 13 tracked variables and every forecast hour in the requested
  span, `update_gfs` searches backwards (up to 12 days, cycles `18→12→06→00`) for a
  transformed file whose forecast hour matches the target time relative to that
  cycle's base time.
- **RTOFS/HYCOM** — for every transformed file already present under the day's
  `transform/.../rtofs` prefix, `update_hycom` parses the `f{hh}` / `n00` / `n-{hh}`
  suffix in the filename to compute the absolute target datetime directly.

A JSON summary of every update attempt is written to S3.

## I/O

**Inputs (parameters):** `bucket`, `download_date` (the model cycle's base date,
midnight UTC), `forecast_days_download` (default 8), `max_concurrent_tasks` (default 10).

**Inputs (S3, from transform stage):**
- `transform/{yyyy}/{mm}/{dd}/gfs/{cycle}/{variable}-f{fff}.npy` (searched across up to
  12 preceding days/cycles per variable/target-hour combination).
- `transform/{yyyy}/{mm}/{dd}/rtofs/*.npy` (listed directly for the given day).

**Outputs (S3 artifacts):**
- `process/{yyyy}/{mm}/{dd}/{hh}/{variable}.npy` — one file per variable per hour,
  the canonical dataset served to consumers.
- Summary artifact: `summary/{yyyy}/{mm}/{dd}/update_summary.json`.
- Indirect output (triggered from [main.py](../src/main.py) after this stage
  completes): an HTTP `POST {WEATHER_URL}/weather/sync` notifying the weather
  service that new `process/` data is available.

**Tracked GFS variables (`GFS_VARIABLE`):** `was`, `wad_come_from`, `wsh`, `pwp`,
`pwd`, `wad`, `wsp`, `wsd`, `pswh`, `pswd`, `pswp`, `sigwh`, `pwh` — matching the
`variable_store_name` values produced by `noaa_transform`.

## Physics Involved

No new physical computation occurs here; the logic encodes the **operational
semantics of model cycles**:

- GFS cycle-to-target-hour resolution: forecast hour
  $f = \lfloor (\text{target\_dt} - \text{base\_dt}) / 3600s \rfloor$, computed by
  `compute_f_hour`, searched over candidate cycles `{18, 12, 6, 0}` per day, walking
  backward up to 12 days to find the freshest cycle that actually forecasts the
  requested hour (`f > 0` and file exists).
- RTOFS filename semantics: `f{hh}` = forecast +hh hours from `target_dt`, `n00` =
  same time as `target_dt` (nowcast), `n-{hh}` = hindcast −hh hours from `target_dt`.

## Place in Pipeline

**Stage 3 (final)** of the ETL. Consumes `transform/` output produced by
`noaa_transform` and produces the canonical `process/` tree. After this stage,
[main.py](../src/main.py) calls the weather service's sync endpoint. See
[pipeline.md](./pipeline.md).

## Extra Information

- Backtracking (`gfs_find_forecast_with_backtracking`) allows the pipeline to keep
  serving the best available forecast for a target hour even if the most recent
  ingestion cycle is missing or incomplete — it prefers the most recent complete
  cycle rather than failing outright.
- `max_dt` (today at 00:00 UTC) bounds the backward search so it never looks into
  cycles that haven't happened yet.
- Copies use `copy_file_s3` (S3 `copy_object` with metadata replaced), so this stage
  is comparatively cheap — no data is re-read into the process, only S3-side copies.
