# Task 18 — Cloud migration of experiments

**Depends on:** `18-cloud-migration-of-experiments_subtask-azure-cli-setup.md`
(this machine has never authenticated to Azure before — must be done
first) · **Blocks:** the full-grid extension of task 07 (full
`(721,1440)` rank-fraction sweep), and any future task that needs
full-resolution compute (task 08's masked-HOOI convergence study is a
likely next case, given its own land-region concern is the same one that
made task 07's full-grid attempt slow)
**Source:** Discovered 2026-08-30 while extending task 07 to the full
`(721,1440)` wave grid (the crop-based sweep in
`notebooks/07_tucker_dmd_rank_selection.ipynb` used a small, deliberately
homogeneous 280×230 region — see task 00's own crop-selection notes — which
may understate the rank the full global domain actually needs). Not one of
`docs/open_questions.md`'s tracked design questions — this is an execution
constraint, not an open design question.
**Primary source files:** `notebooks/07_fullgrid_rank_sweep_randominit.py`,
`notebooks/07_fullgrid_rank_sweep_svdinit.py`, `tasks/07-tucker-dmd-rank-selection.md`
**How:** see `tasks/18-cloud-migration-of-experiments_plan.md` (this file
is the what/why; that file is the how)

## 1. What problem this addresses

Running a masked Tucker decomposition (`tensorly.decomposition.partial_tucker`,
numpy/BLAS backend, CPU-only) against the full `(721, 1440)` wave grid hit a
real, demonstrated wall on the local development machine (31GB RAM, ~6-7
usable BLAS threads) — **not a memory wall** (peak RSS stayed in the 3-12GB
range throughout every full-grid attempt) **but a wall-clock wall**: three
separate attempts at a real rank candidate — random-init at `n_iter_max=300`
(~84 minutes), SVD-init at the same uncapped default (~24 minutes), and a
deliberately bounded SVD-init run (~16 minutes) — were all killed before
producing a usable result. This is slower than the same total tensor size
on a small homogeneous crop by roughly two orders of magnitude in
wall-clock terms, consistent with `docs/open_questions.md` §5.4's standing
concern that masked HOOI converges slowly (or poorly) over large
contiguous unobserved regions — the full grid's continents, vs. the crop's
~0% land.

This blocks two things concretely:
- **Task 07's own full-grid rank recommendation.** The current recommendation
  (`(r_lat=80, r_lon=160)`, `tasks/07-tucker-dmd-rank-selection.md` §4) is
  derived entirely from a small, homogeneous 280×230 crop and is explicitly
  flagged there as hardware-bounded, not evidence that error plateaus. A
  real full-grid rank curve is still missing.
- **Task 08** (masked-HOOI convergence under large contiguous land regions,
  `tasks/08-masked-hooi-convergence.md`) has never been run at all, and is
  very likely to hit the same wall-clock problem for the same reason
  (masked-HOOI convergence over the same continental landmasses), since it
  studies convergence sensitivity specifically at imputed land cells.

## 2. Why the cloud, not just "more RAM"

Memory was never the bottleneck (see above), so simply buying/renting more
RAM would not fix this. What's actually needed is more (or faster) CPU
throughput to shrink the per-ALS-iteration cost, and — just as importantly
— the ability to let a multi-hour job run unattended without tying up the
local machine that's also running the interactive development session.
Both are cloud-shaped needs, not local-hardware-upgrade-shaped ones.

A cost comparison across GCP, AWS, Azure, Alibaba Cloud, Oracle Cloud, and
Google Colab was done for this specific workload profile (CPU-only,
8vCPU/32GB class, intermittent multi-hour bursts, tolerant of interruption).
Conclusions that inform this task's scope:
- **Google Colab is not viable** — every tier caps CPU-only runtimes at 2
  vCPUs regardless of price paid, which would badly bottleneck this
  workload's BLAS-threaded (~6-7 core) compute.
- **Azure Spot VMs** were chosen as the cheapest viable option for this
  usage pattern (~$1.85–$7.68/month depending on actual hours used),
  narrowly ahead of Oracle Cloud's preemptible pricing and clearly ahead of
  AWS/GCP/Alibaba spot pricing — and the user already holds an Azure
  account, so no new-account overhead.
- NOAA's own GFS-Wave archive is confirmed freely available (verified
  directly against the live public buckets, not just documentation) on
  both AWS (`s3://noaa-gfs-bdp-pds`) and Azure
  (`noaagfs.blob.core.windows.net/gfs`), with retention going back well
  past this project's training window. **This task deliberately does not
  use that path** — re-deriving the archive from raw GRIB2 would require
  standing up the `noaa-etl` ETL (`gfs_transform.py`), which isn't even
  available in this repo (confirmed absent from the local filesystem during
  task 17's work) or checked out anywhere on this machine. Instead, this
  task uploads the already-processed local archive directly to Azure Blob
  Storage — more storage cost (negligibly so, well under $1/month at this
  data scale), zero new ETL work.

## 3. Scope

**In scope:**
- Provisioning a repeatable Azure VM + Blob Storage setup for running
  existing (or paused) local experiments against real archive data.
- Migrating and completing the specific stalled full-grid rank-sweep
  experiment that motivated this task (see the plan doc for exact
  candidates and order).
- A lightweight per-run resource log (wall time, peak RSS/CPU, config,
  approximate cost) recorded going forward, so future "is a bigger
  rank/window worth it" decisions are evidence-based rather than
  guessed — this session's three killed full-grid attempts were exactly
  this kind of avoidable guess.

**Out of scope:**
- Any change to `src/weather/model.py` or `src/train/ingest.py`'s actual
  execution environment. This is a notebook/experiment-runner concern only,
  mirroring the same blast-radius discipline already established for task
  17 — these are notebooks that justify a pipeline change, not the change
  itself.
- Building a persistent or managed cloud pipeline, CI, or scheduled
  compute. This is on-demand, start/stop-as-needed VM usage for
  exploratory work, not infrastructure-as-a-standing-service.
- Migrating tasks that don't need full-grid-scale compute — most of the
  15 open-question tasks (`tasks/README.md`) run comfortably on the local
  machine today and have no reason to move.

## 4. Conclusion

_(Blank. Fill in once the plan in `tasks/18-cloud-migration-of-experiments_plan.md`
has been executed against a real Azure VM — what ran, what it cost, and
whether the full-grid rank curve changes task 07's or task 08's
conclusions.)_
