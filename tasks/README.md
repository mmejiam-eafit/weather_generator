# Open-question resolution tasks

This index tracks experiments that answer the 15 currently OPEN /
PARTIALLY ANSWERED / PARTIALLY MOOT questions in `docs/open_questions.md`
(§2.2, §2.4, §3.2, §3.4/§3.5, §4.1, §4.2, §5.3, §5.4, §5.5, §6.1, §7.1,
§7.2, §8.1, §8.2, §8.3). Each task produces a reproducible, notebook-backed
experiment against real archive data, with a place to record what was
found.

**This replaces the previous v1→v2 code-migration task list**, which
predated the current `docs/` series (`variable_taxonomy.md`,
`tucker_decomposition.md`, ..., `correlated_variables.md`,
`open_questions.md`) and referenced a since-superseded
`docs/model_design.md`. That list is not discarded — it's moved to
[`tasks/tbd/`](tbd/v1_v2_migration_index.md), deprioritized rather than
deleted, since it may still inform an eventual implementation plan once
these open questions are resolved.

**Status column** (same legend as the previous index):

| Emoji | Meaning |
|---|---|
| ⬜ | Todo — not started (no gitmoji equivalent; neutral placeholder) |
| 🚧 | In progress (`:construction:`) |
| ✅ | Done (`:white_check_mark:`) |
| 🔒 | Blocked (no gitmoji equivalent; repurposed `:lock:` for "can't proceed") |
| ♻️ | Refactored (`:recycle:`) — done, but the approach changed materially from this file's spec; check the task file's notes before trusting it verbatim |
| ⚰️ | Dropped (`:coffin:`) — deliberately not doing this; reason belongs in the task file, not just this table |

**✅ Task 15 (fixed 2026-08-14)** — `load_training_window` previously
crashed on all real era-3 data due to a land-cell false positive in its
`wad`/`wad_come_from` consistency guard (found on first real-archive
validation after the data disk was reconnected). It's filed as task **15**
(not renumbered to 00, to avoid churning 14 files' names and
cross-references) but is listed first here since it blocked every task
below it. Now fixed — tasks 00–14 are unblocked and can run their
notebooks against real data.

| # | Status | Task | Depends on | Key files |
|---|---|---|---|---|
| 15 | ✅ | Fixed `wad`/`wad_come_from` check's land-mask false positive | — | `15-wad-consistency-check-land-mask-artifact.md` |
| 00 | ✅ | Gaussianization placement (`open_questions.md` §4.1) — decided (i, before Tucker); i-vs-ii left as deferred follow-on | 07 (soft, deferred) | `00-gaussianization-placement.md`, `notebooks/00_gaussianization_placement.ipynb` |
| 01 | ⬜ | Copula family / tail dependence (§4.2) | — | `01-copula-family-tail-dependence.md`, `notebooks/01_copula_family_tail_dependence.ipynb` |
| 02 | ⬜ | Rejection sampling vs. copula fix (§8.3) | 01 | `02-rejection-sampling-vs-copula-fix.md`, *shares* `notebooks/01_copula_family_tail_dependence.ipynb` |
| 03 | ⬜ | Direction preservation under Gaussianization (§2.4) | 00 (soft) | `03-direction-gaussianization-angle-preservation.md`, `notebooks/03_direction_gaussianization_angle_preservation.ipynb` |
| 04 | ⬜ | Joint-group selection (§3.2) | — | `04-joint-group-selection.md`, `notebooks/04_joint_group_selection.ipynb` |
| 05 | ⬜ | Triplet coupling lost to independent Gaussianization (§2.2) | 04 | `05-triplet-coupling-gaussianization.md`, *shares* `notebooks/04_joint_group_selection.ipynb` |
| 06 | ⬜ | Joint multi-channel Tucker vs. downstream, `{u,v,logE,F}` (§3.4/§3.5) | — | `06-joint-channel-tucker-vs-downstream.md`, `notebooks/06_joint_channel_tucker_vs_downstream.ipynb` |
| 07 | ✅ | Tucker/DMD rank selection (§5.3) | 00 (soft), 17 (soft) | `07-tucker-dmd-rank-selection.md`, `notebooks/07_tucker_dmd_rank_selection.ipynb` |
| 08 | ⬜ | Masked-HOOI convergence (§5.4) | — | `08-masked-hooi-convergence.md`, `notebooks/08_masked_hooi_convergence.ipynb` |
| 09 | ⬜ | Growing DMD modes (§5.5) | — | `09-growing-dmd-modes.md`, `notebooks/09_growing_dmd_modes.ipynb` |
| 10 | ⬜ | OU variance-ratio floor (§6.1) | — | `10-ou-variance-floor-validation.md`, `notebooks/10_ou_variance_floor_sensitivity.ipynb` |
| 11 | ⬜ | Climatology quasi-stationarity (§7.1) | — | `11-climatology-quasi-stationarity.md`, `notebooks/11_climatology_trend_window_sensitivity.ipynb` |
| 12 | ⬜ | Directional-consistency tolerance, $\delta_{dir}$ (§8.1) | — | `12-directional-consistency-tolerance.md`, `notebooks/12_directional_consistency_tolerance.ipynb` |
| 13 | ⬜ | Diurnal-signal monitoring trigger (§7.2) | — | `13-diurnal-signal-monitoring-trigger.md`, `notebooks/13_diurnal_signal_monitoring_trigger.ipynb` |
| 14 | ⬜ | Cross-cell/cross-time consistency: build-or-defer scoping (§8.2) | — | `14-spatiotemporal-consistency-scoping.md`, `notebooks/14_spatiotemporal_consistency_scoping.ipynb` |
| 16 | ⬜ | Implement real per-cell ocean-grid coordinates (blocks `sst`/`ocs`/`ocd` loading) | — | `16-ocean-grid-coordinates.md` |
| 17 | ✅ | Partition-absence NaN vs. land/ice NaN sentinel policy (new §5.6) — validated, notebook/`src`-helper scope only | 07 (soft), 08 (soft) | `17-partition-nan-sentinel-policy.md`, `notebooks/17_partition_nan_sentinel_policy.ipynb` |
| 18 | ⬜ | Cloud migration of experiments (full-grid rank sweep hit a wall-clock wall locally, not memory) | — | `18-cloud-migration-of-experiments.md`, `18-cloud-migration-of-experiments_plan.md` |

Tasks 01/02 and 04/05 deliberately share one notebook each — `open_questions.md`
itself states these pairs are the same underlying experiment viewed from
two angles (§4.2/§8.3), or that one question's unresolved residual *is*
the other (§2.2's gap is §3.2). Writing two separate experiments would
duplicate work for no benefit.

Update the status cell as work proceeds. Each task file's own **Conclusion**
section is where the actual finding goes — this table only tracks whether
that section has been filled in, not what it says.

## Data-availability note

**Update, 2026-08-14:** the archive disk is reconnected and mounted at
`./data`. Verified directly: `Settings().data_dir` needs no override, all
16 variables' dtypes/shapes match `core.config`/`core.io` expectations, and
a full snapshot-by-snapshot scan of 2025-11-01 → 2026-05-01 (the superset
of every task's chosen window) found zero missing variables and zero
contiguous gaps. The archive itself goes empty after 2026-08-05T00:00Z (no
task's window reaches that far). Task 15 (above) is now fixed and verified
against real data, so `load_training_window` loads real era-3 windows
cleanly.

Every new notebook uses real, current package APIs (`core.io`,
`core.config`, `train.ingest`, and the relevant `train.*` transformers) and
was checked for syntax validity and import resolution against this repo's
Poetry environment at reorg time. Each notebook's starter-code cells
(imports, `load_training_window` call, mask construction) should now run
without further changes; the experiment-specific analysis cells past that
point are the actual work each task still requires — none of them have
been run end-to-end yet.
