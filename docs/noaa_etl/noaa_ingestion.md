# `noaa_ingestion` — Raw Data Ingestion

**Module:** [src/task/noaa_ingestion.py](../src/task/noaa_ingestion.py)

## Task Description

Downloads the raw NOAA model output files needed to build a full forecast cycle and
streams them straight into S3, without ever writing to local disk. It covers two model
families:

- **GFS Wave** (`gfswave`) — global 0.25° gridded wave model GRIB2 files, one per
  synoptic cycle (`00`, `06`, `12`, `18` Z) and forecast hour.
- **RTOFS / HYCOM** (`rtofs`) — global ocean forecast archive files (`.a.tgz` + `.b`
  pairs), including forward forecast hours, a "now" (`n00`) file, and a 24-hour
  hindcast window (`n-01` … `n-24`).

All downloads run concurrently (bounded by a semaphore) using a streaming HTTP client
with retry/backoff, and each file is piped directly into an S3 multipart upload. At the
end, a JSON summary of every download's outcome (success/failure, timestamp, error) is
written to S3.

## I/O

**Inputs (parameters):**
- `download_date` — target UTC date for the model cycle being ingested.
- `forecast_days_download` — number of forecast days to request (1–8, default 8).
- `max_concurrent_tasks` — concurrency cap for simultaneous downloads (default 10).
- `bucket`, `s3_client`, `httpx_client` — destination bucket and async clients.

**External inputs (HTTP sources, NOAA NOMADS):**
- GFS Wave: `https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{YYYYMMDD}/{cycle}/wave/gridded/gfswave.t{cycle}z.global.0p25.f{fff}.grib2`
- RTOFS: `https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/rtofs.{YYYYMMDD}/rtofs_glo.t00z.{f{hh}|n{-hh}|n00}.archs.{a.tgz|b}`

**Authentication / access requirements:**
- NOMADS (`nomads.ncep.noaa.gov`) is a **public, unauthenticated** NOAA data server —
  every request in this module is a plain anonymous `GET`/`HEAD` (see
  `httpx_stream_with_retry` in [src/util/request.py](../src/util/request.py)); no API
  key, token, or credential is sent or required, and none is configured anywhere in
  [src/core/config.py](../src/core/config.py).
- There is **no rate-limit key or allowlisting** in this codebase for NOMADS — NOAA
  publishes these products for free public access, but does ask consumers to follow
  its [fair-use/usage guidance](https://nomads.ncep.noaa.gov) (reasonable concurrency,
  retry with backoff on failures) — reflected here by the bounded semaphore and
  `httpx_stream_with_retry`'s exponential backoff.
- The **only** credentials involved in this module are for the **destination S3
  bucket**, and those are never embedded in the code: `aioboto3.Session()` in
  [src/main.py](../src/main.py) resolves AWS credentials from the standard boto3
  chain (environment variables, shared config/credentials file, or — in the deployed
  ECS/Fargate task per [template.yaml](../template.yaml) — the task's IAM role).
- Practical implication: this stage can run from any network with outbound HTTPS
  access to NOAA's public servers; no secrets need to be provisioned for the download
  side, only for the S3 upload side.

**Outputs (S3 artifacts):**
- Raw files: `raw/{yyyy}/{mm}/{dd}/gfs/{cycle}/gfswave.t{cycle}z.global.0p25.f{fff}.grib2`
- Raw files: `raw/{yyyy}/{mm}/{dd}/rtofs/rtofs_glo.t00z.{...}.{a.tgz|b}`
- Summary artifact: `summary/{yyyy}/{mm}/{dd}/download_summary.json` — a JSON array of
  `{url, s3_key, status, timestamp, error?}` records for every file attempted.

## Physics Involved

This module performs no numerical/physical computation itself; its "physics" is
encoded in the **scheduling** of which files to request, which mirrors the underlying
model's operational cadence:

- GFS runs 4 cycles/day (00/06/12/18Z); forecast hour resolution follows
  `get_rfs_forecast_hours` (1-hour steps up to 120h, 3-hour steps beyond, see
  [gfs_utils.md](./noaa_transform.md#extra-information) for the formula).
- RTOFS is a single daily cycle (`t00z`) that provides both a forecast horizon
  (`f01`…) and a rolling hindcast/nowcast window (`n-24`…`n00`), reflecting how
  ocean reanalysis products blend past assimilated states with forward forecasts.

## Place in Pipeline

This is **stage 1** of the ETL, invoked first by [src/main.py](../src/main.py). It
populates the `raw/` prefix in S3, which is the sole input to the transform stage
(`noaa_transform`). See [pipeline.md](./pipeline.md) for the full flow.

## Extra Information

- Streaming download/upload is implemented via `httpx_stream_with_retry` (HEAD-probe +
  exponential backoff before streaming) and `multipart_stream_to_s3` (chunked S3
  multipart upload), both in [src/util](../src/util/).
- Concurrency is bounded with `asyncio.Semaphore(max_concurrent_tasks)`; all downloads
  run as a single `asyncio.gather`.
- Forecast hour list is produced by `get_rfs_forecast_hours` from
  [src/util/gfs/gfs_utils.py](../src/util/gfs/gfs_utils.py).
- Dependencies: `httpx` (async HTTP/2 client), `aiobotocore`/`aioboto3` (async S3).
