# Task 18 Plan — execution order and cloud setup

**What/why:** see `tasks/18-cloud-migration-of-experiments.md`. This file
is the how: which stalled experiments to run first, in what order, and the
concrete Azure steps to run them.

## 1. Ranking this session's full-grid attempts by time spent vs. results obtained

All rows below are the same underlying investigation (task 07's full-grid
follow-on, `wsh`, `height_log_energy`, `land_mask_from_sigwh`, 12-day
window `T=96`, full `(721,1440)` grid, no crop) — the only experiment this
session that hit multi-hour-scale local wall-clock cost.

| Run | Config | Wall time | Outcome |
|---|---|---|---|
| 1 | rank (144,288), `init="random"`, `n_iter_max=300` | ~84 min, killed | **0 results** |
| 2 | rank (10,10), `init="svd"`, `n_iter_max=300` (uncapped default) | ~24 min, killed | **0 results** |
| 3 | rank (144,288), `init="svd"`, `n_iter_max=30` (bounded) | ~16 min, killed (moved to cloud instead) | **0 results** |
| 4 | rank (10,10), `init="random"`, `n_iter_max=300` | ~10-12 min, completed | 1 result (err=0.8869) |
| 5 | rank (10,10), `init="svd"`, `n_iter_max=5/10/20` (3 separate fresh runs) | ~569.6s + 659.2s + 756.4s ≈ 33 min total | 3 results (0.7107 / 0.7068 / 0.7039) |

**The clear worst case is the actual rank this task set out to answer**:
rank (144,288) — 20% of the full grid's dimensions, the size originally
requested to compare against the crop-based sweep — has consumed **~100
minutes of combined local wall-clock time across two attempts (runs 1 and
3) and produced zero results.** Run 2 (also zero results) shows the
problem isn't rank-size-specific either — even the tiny (10,10) rank
failed to converge inside 300 iterations without SVD init being paired
with an explicit, small `n_iter_max`.

**Methodological note to carry into the cloud runs:** run 5's three
separate fits each re-paid the ~500s one-time SVD-initialization cost from
scratch just to sample a different `n_iter_max`. `partial_tucker` already
returns a `rec_errors` list tracking error at *every* iteration of a
single run — the same iteration-vs-error curve should be read off **one**
run with a generously-sized `n_iter_max`, not reconstructed by re-running
the fit multiple times at different caps. This alone would have cut run
5's ~33 minutes to roughly one run's worth (~13-15 min).

## 2. Priority order for the cloud

1. **First (cheap, do before spending cloud time on the expensive case):
   the crop+short-window control experiment**, proposed but never run
   this session. Rerun the existing 280×230 crop (same one
   `07_tucker_dmd_rank_selection.ipynb` already uses) restricted to the
   same 12-day / `T=96` window the full-grid attempts used, at a small
   rank like (10,10). This isolates whether the full grid's much higher
   error at matched rank fractions (§ this task's parent doc) is a genuine
   spatial-heterogeneity effect or an artifact of the short time window —
   needed to correctly interpret step 2's results, and cheap enough
   (small crop) to run either locally or as the very first cloud job, to
   validate the VM/Blob setup cheaply before committing to the expensive
   sweep.

2. **Second (the actual "longest running, least results" experiment,
   now resumed): the full symmetric fraction-of-full-rank sweep on the
   full `(721,1440)` grid.** Use fractions symmetric across both spatial
   axes (the original doubling sequence `(5,10)→(80,160)` drifted badly
   off "same fraction of full rank" by its top end — 80/280=28.6% vs.
   160/230=69.6% on the crop — don't repeat that mistake here):

   | fraction | r_lat (of 721) | r_lon (of 1440) |
   |---|---|---|
   | 5%  | 36  | 72  |
   | 10% | 72  | 144 |
   | 20% | 144 | 288 |
   | 40% | 288 | 576 |
   | 60% | 432 | 864 |
   | 80% | 577 | 1152 |

   Run each with `init="svd"` (proven far better than random this
   session — run 4 vs. run 5 above) and a single fit per rank with
   `n_iter_max` generous enough to see the plateau clearly (start at 40,
   double if `rec_errors` hasn't visibly flattened by the end — don't
   guess a cap blind the way run 3 did). Record the full `rec_errors`
   curve, not just the final error, per the methodological note above.

3. **Third (only after 1 and 2 land): extend the time window back toward
   the crop sweep's full 6 months.** Validate the rank/iteration budget on
   the cheap short window first — don't iterate on configuration against
   the expensive long window while the approach is still being debugged.

## 3. Concrete Azure setup

**Step 0 (prerequisite, do first):** complete
`tasks/18-cloud-migration-of-experiments_subtask-azure-cli-setup.md` — this
machine had no authenticated `az` session at the start of this task, and
the first account tried (a university-managed "Azure for Students Starter"
tenant) turned out to disallow both `Microsoft.Compute` and
`Microsoft.Storage` at the tenant-policy level. The subtask now uses a
second, personal account (`mmejiam9206@outlook.com`, "Azure subscription
1") instead. Everything below assumes `az account show` reports that
account.

**VM size note (2026-08-31): downsized from the original D8s_v5 (8 vCPU)
target to D4s_v5 (4 vCPU/16GB), and from Spot to on-demand.** The personal
account is a Free Trial subscription (`quotaId: FreeTrial_2014-09-01`)
with a default regional quota of 4 total vCPUs / 3 Spot vCPUs in `eastus`
— below 8 either way, and Azure blocks self-service quota increases for
Free Trial subscriptions until upgraded to Pay-As-You-Go (a portal action
only the account owner can do; see the subtask's Conclusion for the full
finding). D4s_v5 on-demand fits the current quota with no waiting; D2s_v5
Spot (2 vCPU, fits the 3-vCPU Spot cap) was considered and rejected —
local runs this session used ~6-7 active BLAS threads, so 2 vCPUs would be
a step *down* from what's already available locally. Revisit D8s_v5 Spot
if/when the account is upgraded to Pay-As-You-Go and a real quota increase
is obtained.

**One-time provisioning:**
1. Create one Azure Storage Account + Blob container in a chosen region
   (e.g. East US, matching where NOAA's own NODD mirror already lives —
   irrelevant to this task since it uploads the local archive rather than
   pulling from NODD, per the parent doc's scope decision, but keeping the
   same region as a habit costs nothing).
2. `azcopy copy`/`azcopy sync` the local `data/` archive up to that
   container, once.

**Status (2026-08-31): done.** Resource group `rg-weather-generator`,
storage account `wxgenarchive2026` (Standard_LRS, Hot tier, TLS1.2
minimum), container `training-data`, all in `eastus`. Scoped the upload to
the **6-month training window** (`2025/11` through `2026/04`, matching
`T_START`/`T_END` already used throughout `notebooks/07_tucker_dmd_rank_selection.ipynb`
and the `07_fullgrid_rank_sweep_*.py` scripts) rather than the full local
archive — the archive on the external drive
(`/media/mmejia/EAFF-B3D3/bearing_noaa/data`) turned out to be **380GB**
total (2022-2026), well past the "few tens of GB" this task's original
cost estimate assumed; the actual training window is **~87GB**, uploaded
via `azcopy copy --recursive` per month, account-key-signed SAS (a
user-delegation SAS via `--as-user` failed with
`AuthorizationPermissionMismatch` — creating a storage account via the
management plane does not itself grant blob data-plane RBAC). Verified:
86.95GB transferred, 0 failed transfers across all 6 jobs. The full
4+ year archive was **not** uploaded — only re-upload additional months
later if a future experiment needs a window outside Nov 2025 - Apr 2026.

**Per work session:**
3. Create a VM: `Standard_D4s_v5`, Ubuntu Server LTS image, in the
   **same region** as the storage account, on-demand (not Spot — see the
   VM size note above), SSH key auth, NSG rule restricting port 22 to the
   operator's own IP only.
4. On the VM: install `poetry`, `git clone` this repo, `poetry install`,
   then `azcopy copy` the archive down from Blob Storage to the VM's local
   disk. Point `DATA_DIR` at that local path — no code changes needed,
   matching the pattern already used in
   `notebooks/07_fullgrid_rank_sweep_svdinit.py`
   (`Path(__file__).resolve().parents[1] / "data"`).
5. Run each prioritized experiment (§2 above) inside `tmux` (survives SSH
   disconnects), **one rank/config per job** — this both matches the
   local one-at-a-time convention already established this session and
   limits any crash/interruption's blast radius to a single rank's work,
   not the whole sweep (worth keeping even on-demand, not just for spot
   eviction risk). Either invoke the existing standalone scripts directly
   (`notebooks/07_fullgrid_rank_sweep_svdinit.py --r_lat N --r_lon N
   --n_iter_max N`) or wrap the equivalent cells via `papermill` if the
   result needs to land back in `notebooks/07_tucker_dmd_rank_selection.ipynb`
   as real notebook output — no notebook-to-script conversion is actually
   required either way.
6. After each run: append wall time, peak RSS, `rec_errors` curve, and
   final error to a running log (a simple table in this task's own
   Conclusion section, or a small CSV alongside the scripts in
   `notebooks/`) — the discipline this session lacked until these three
   costly, undocumented-until-killed attempts.
7. `scp` results (or the finished notebook) back to the local machine;
   commit from local `git` as usual — no need to configure push
   credentials on the disposable VM.
8. `az vm deallocate` (or `delete`, if the investigation is fully done)
   at the end of every session. A forgotten running on-demand VM is the only
   way this stops being cheap.

## 4. Definition of done

- The full `(721,1440)` rank-fraction sweep (§2, step 2) has real
  `rec_errors`-backed results at all six fraction candidates, recorded in
  `tasks/07-tucker-dmd-rank-selection.md` §4 alongside (not replacing) the
  existing crop-based recommendation.
- The crop+short-window control experiment (§2, step 1) has resolved
  whether the full grid's higher error is a heterogeneity effect or a
  window-length confound, recorded in the same place.
- A per-run resource log exists and is populated for every cloud run made
  under this task.
- The Azure VM used for this work has been deallocated or deleted.

## 5. Session log (2026-08-31 attempt — torn down, not completed)

A first real attempt was made and then deliberately torn down (all
resources deleted) so the approach could be reorganized before continuing.
Recorded here so the next attempt doesn't repeat the same dead ends:

1. **Regional capacity, not quota, blocked `eastus`.** After the PAYG
   upgrade (see the subtask's Conclusion), `Standard_D4s_v5` failed with
   `SkuNotAvailable`/`Capacity Restrictions` in `eastus` — a transient
   regional stock issue, not a quota problem (`Total Regional vCPUs` was
   already 10 post-upgrade). Tried `Standard_E8bds_v5` (memory-optimized,
   local NVMe, 8vCPU/64GB) next — also `SkuNotAvailable` in `eastus`.
2. **Per-family quota resets after a PAYG upgrade.** Post-upgrade,
   `standardDSv5Family` and `standardEBDSv5Family` quotas both reset to 0
   in `eastus` even though `cores` (total regional) rose to 10. Attempted
   `az quota create` for `standardDSv5Family` — failed with
   `ContactSupport` (self-service quota API declines the request outright
   on a subscription this new; would need an actual support ticket).
   `standardEBDSv5Family` quota, by contrast, was **already 10** with no
   request needed — used that family instead of chasing a DSv5 increase.
3. **`eastus2` had both capacity and quota for `Standard_E8bds_v5`.**
   VM created successfully there. This put **compute in `eastus2` while
   the storage account/data stayed in `eastus`** — a real, if small,
   architectural wart (cross-region transfer, not same-region-free) worth
   fixing on the next attempt by provisioning storage in whichever region
   actually has VM capacity, checked first, rather than assuming `eastus`
   throughout.
4. **VM setup succeeded end-to-end**: Ubuntu 24.04, pyenv, Python 3.14.3
   built from source, Poetry, `poetry install` all completed cleanly.
   Repo was `scp`'d over as a tarball (not `git clone`) since substantial
   local work — including this very task's files — was uncommitted.
5. **The data download (Blob → VM, cross-region) failed partway through
   and was not root-caused before teardown.** `azcopy copy` of the 87GB
   window started cleanly (0 failures through ~26% complete) then began
   failing en masse — 14,302 of 23,168 file transfers failed
   (`CompletedWithErrors`), escalating rapidly rather than failing
   uniformly from the start. Reported throughput numbers were
   implausibly high (up to ~17,000 Mb/s) for many small (~2MB) files,
   consistent with either Azure Blob per-account request-rate throttling
   under high `azcopy` concurrency, or a network-side interruption on the
   VM — **not distinguished before the session ended**; the actual
   `RESPONSE`/`ERROR CODE` content behind the failures was never pulled
   from the azcopy job log. **Next attempt should check this log's actual
   error codes first** (`grep -A10 'RESPONSE [0-9]' /tmp/azcopy_download.log`
   on the VM, or the detailed per-job log under `~/.azcopy/`) before
   retrying blindly — if it's throttling, `azcopy`'s `--cap-mbps` or a
   lower `AZCOPY_CONCURRENCY_VALUE` should fix it; if it's a dropped
   connection, a plain retry may suffice.
6. **Torn down completely** at the user's request, mid-diagnosis, to
   reorganize the approach rather than continue firefighting one region/
   quota/transfer issue at a time. `az group delete` on both
   `rg-weather-generator` (VM, storage account, all uploaded data) and the
   auto-created `NetworkWatcherRG` — confirmed both fully deleted.
   Nothing Azure-side survives from this attempt; the next attempt starts
   from provisioning again.
