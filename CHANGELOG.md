# Changelog

All notable changes to this project are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **2026-08-20 — Closed task 07 (`tasks/07-tucker-dmd-rank-selection.md`,
  `docs/open_questions.md` §5.3): full Tucker rank sweep on real
  `wsh`/`wsp` under task 17's corrected mask finds no elbow — memory, not
  the error curve, is the binding constraint.** Extended the rank sweep
  to the full candidate set `(5,10),(10,20),(20,40),(40,80),(80,160)` for
  `sigwh`, `wsh`, and `wsp` on the same 280×230/`T=1448` crop; all three
  error curves decrease monotonically at every step with no plateau
  (`sigwh` 9.64%→0.43%, `wsh` 63.16%→23.81%, `wsp` 14.37%→5.84%). Measured
  peak memory per fit empirically, isolated via an `RLIMIT_AS`-capped
  subprocess (so an over-budget attempt fails as a clean `MemoryError`
  rather than the OS OOM-killer taking down an unrelated process): ~12.25–
  12.27GB, essentially rank-independent from `(40,80)` through `(80,160)`
  for both `wsh` and `wsp` — dominated by fixed per-ALS-iteration tensor
  unfolding cost, not target-rank core size. `(80,160)` is recommended as
  a hardware-bounded default (this machine: 31GB total, ~18GB available),
  not because error stopped improving there — re-evaluate on any machine
  with materially more memory. `notebooks/07_tucker_dmd_rank_selection.ipynb`,
  `tasks/07-tucker-dmd-rank-selection.md` §4, `docs/open_questions.md`
  §5.3, and `tasks/README.md` updated with the real results.

- **2026-08-20 — Filed and validated task 17 (`tasks/17-partition-nan-
  sentinel-policy.md`, `docs/open_questions.md` §5.6): a sentinel policy
  distinguishing partition-absence NaN from land/ice NaN in the wave
  partition triplets.** Root cause of task 07's `wsh` pathology
  (masked HOOI imputing ~90-99% of the domain, error increasing with
  rank): no document or code anywhere distinguished land/ice NaN from a
  genuine ocean cell where a specific partition (e.g. wind-sea) is simply
  absent, and there was no authoritative, variable-independent land mask
  — every mask was inferred from a variable's own NaN pattern, exactly
  the thing that conflates the two causes. New `src/train/masking.py`
  (notebook/helper-scoped only, `src/weather/model.py` and
  `src/train/ingest.py` untouched) derives an authoritative land mask
  from `sigwh` (NaN only over land) and applies a sentinel policy: at a
  genuine ocean cell, height gets sentinel `H=0` (marked VALID — the
  clean physical limit as $m_0\to0$), with a numerical floor before
  `log E:=log(H^2)` (`H_FLOOR_M=0.01`); period has no analogous clean
  limit ($T_{m01}=m_0/m_1$ is $0/0$ as the partition vanishes) but is not
  arbitrary either — `period_sentinel_from_archive` extrapolates the
  limit from real near-vanishing partitions (an L'Hopital-style
  argument), landing on `T_SENTINEL_S≈3.84s`. A real numerical bug was
  found and fixed en route: the initial extrapolation's `np.polyfit` was
  ill-conditioned on uncentered height values (`RankWarning` even at
  18M+ points); fixed by centering before fitting. Categorized all 8
  remaining wave-grid variables against the real archive: `wsd`, `pwd`,
  `pwp`, `pswd` carry zero ocean-cell NaN; `wsh`/`wsp` (36.5%) and
  `pswh`/`pswp` (1.5%) do. Found, and left as an open data-provenance
  question, that `wsd` is never NaN even where `wsh` is and is not a
  stale filler (its variance is *higher*, not collapsed, during
  `wsh`-NaN periods) — plausibly `wad`-derived. Empirical validation:
  rerunning `wsh`'s and `wsp`'s Tucker rank sweeps under the corrected
  mask fixes the original pathology outright — coverage 7-11%→**100%**,
  reconstruction error now **decreases** with rank instead of increasing.
  Land mask persisted at `notebooks/17_wave_grid_land_mask.npy`.
  `docs/variable_taxonomy.md` (new §3.3.1, pointers from §3.4/§4/§6) and
  `tasks/README.md` updated accordingly. Production wiring deliberately
  deferred to a separate, not-yet-approved follow-on task.

- **2026-08-19 — Decided Gaussianization placement (`docs/open_questions.md`
  §4.1): after physical transformations and climatology removal, before
  Tucker decomposition (placement i).** This closes the task as a design
  decision rather than a proven result — the "i vs. ii (Gaussianize the
  Tucker core, before DMD)" comparison stays genuinely open, deferred to a
  follow-on task once `tasks/07-tucker-dmd-rank-selection.md` has a
  defensible rank, or to a more targeted literature search, whichever comes
  first. The decision rests on two grounds: it is the natural consequence of
  how the physical fields are already processed (fit the marginal transform
  on the same climatology-removed, physically-transformed residual scale
  that Tucker/DMD subsequently compress), and it matches this project's own
  held-out result from 2026-08-16 below, where placements (i) and (ii) were
  nearly indistinguishable once the DMD-rank confound was controlled for.
  An external cross-check (four independent multi-agent literature/
  experiment reviews, run outside this repo) was also considered: 6 of 8
  distinct answers favored placement (i), 1 favored (ii), 1 refused to pick
  a default; none of the bundle's own code experiments actually ran the
  real Tucker/DMD stack (most failed with missing `numpy`/`scipy`, the rest
  were toy 1-D/2-D analogues), so it was read as theoretical framing, not
  independent empirical confirmation — it did not by itself decide this,
  though it did reinforce that placement (iii) (Gaussianize after DMD
  directly) is essentially never a sound default, consistent with this
  project's own finding that (iii) is the one that fails badly at
  plausible ranks. `docs/open_questions.md` §4.1 (status → DECIDED),
  `tasks/00-gaussianization-placement.md` (§4 Conclusion), and
  `tasks/README.md` (status → ✅) updated accordingly.

- **2026-08-16 — Extended `notebooks/07_tucker_dmd_rank_selection.ipynb`
  into a real rank-selection experiment; filed
  `tasks/16-ocean-grid-coordinates.md` for a missing-implementation gap
  found along the way.** The notebook was a stub (unrun, with a real
  `tensorly==0.9.0` unpacking bug and a `dmd_fit` input-order bug, both
  fixed) targeting `wsh` over 6 months at full 721x1440 resolution — never
  actually runnable (would need ~6TB for one array). Rebuilt on task 00's
  crop pattern with `sigwh` instead of `wsh` (`wsh` goes NaN constantly
  outside local wind-driven seas; over 6 months that left as little as
  7-11% "ocean" under a 90%-valid-cell tolerance on either of task 00's
  crops, causing masked HOOI to impute nearly the whole domain and produce
  a reconstruction-error-vs-rank curve that got *worse* with rank).
  `sigwh` goes NaN only over land; a maximal-rectangle search over a large
  Pacific candidate box found a real, fully NaN-free 280x230 crop (~10x
  task 00's), used for an extended Tucker rank sweep `(5,10)` to `(80,160)`
  and a DMD rank sweep `(5,10,20,40)` with `cond(dmd.Phi)` (task 00's
  diagnostic) alongside the original `max|λ|` check. Findings written into
  the notebook as markdown: Tucker's reconstruction error never plateaus
  in the tested range on either crop size, which (cross-validated across
  two very differently-sized crops) looks like a real property of
  `sigwh`'s spatial variability rather than a rank-ceiling artifact —
  training-error elbow-hunting alone can't produce a rank recommendation
  for this variable; a held-out generalization check per rank (task 00's
  pattern) is the next real step. DMD shows a genuine conditioning cliff
  between `r_dmd=10` and `r_dmd=20` (~15x jump in `cond(Phi)`) invisible to
  the `max|λ|<1` check alone (no growing modes anywhere tested) —
  `r_dmd=10` is the candidate default. Comment style also brought in line
  with the session's established convention (inline comments ≤2 lines,
  longer rationale in markdown cells, shared `fit_tucker`/
  `dmd_rank_diagnostics` functions instead of inline sweep bodies).
  Separately: extending this to `sst` (the other Class A scalar,
  `variable_taxonomy.md` §3.1) found `sst`/`ocs`/`ocd` cannot be loaded at
  all — `core.config.grid_coordinates("ocean")` raises unconditionally;
  the ocean grid's real per-cell coordinates were derived in
  `open_questions.md` §1.1 but never wired into loadable code. Filed as
  `tasks/16-ocean-grid-coordinates.md`, noted in `open_questions.md` §1.1
  and `tasks/README.md`.

- **2026-08-16 — Gaussianization-placement comparison found confounded by
  Tucker/DMD rank choice; `docs/open_questions.md` §4.1 and §5.3, and
  `tasks/00-gaussianization-placement.md`, updated accordingly.**
  `notebooks/00_gaussianization_placement.ipynb` was extended with a
  held-out test period (fit on a 30-day `sigwh` window, evaluate on the
  next 10 days) plus a physical-space inversion/QQ-plot/99th-percentile
  comparison for all three placements. The held-out run initially looked
  like clean evidence for Gaussianizing before any decomposition:
  placement (iii) (DMD residual) reconstructed the held-out 99th
  percentile with error +255.89, vs. -0.02 for placement (i) and +0.15 for
  placement (ii). A follow-up sensitivity sweep (new cells in the same
  notebook) traced that failure to `dmd.Phi`'s condition number
  (~1.2e6 at `r_lat=10, r_lon=20, r_dmd=20`) rather than to placement
  (iii) itself: dropping `r_dmd` to 5 on identical data cut the condition
  number ~5.5x and the held-out error ~100x; a 32-dimensional Tucker core
  (`r_lat=4, r_lon=8, r_dmd=5`) made the held-out reconstruction
  essentially exact (-0.05, better than placement (i)). Widening the
  spatial crop alone, holding rank fixed, helped far less (~2.5x on
  conditioning) — some, not most, of the effect is the "narrow slice of
  state space" a DMD/Koopman framing predicts. **Consequence: task 00's
  placement comparison cannot be trusted at an unjustified rank** —
  `docs/open_questions.md` §4.1 now records a new dependency on §5.3
  (Tucker/DMD rank selection), §5.3 gains a candidate cheap diagnostic
  (`np.linalg.cond(dmd.Phi)`, requires no held-out data), and
  `tasks/00-gaussianization-placement.md` records this as its (still
  open) interim conclusion, with a new soft dependency on task 07. Also
  fixed while investigating: `ocean_mask`'s "zero NaN across the whole
  window" criterion collapsed to 0.1% ocean once the training window grew
  from 14 to 30 days (`wsh` routinely goes NaN during calm moments, not
  just over land) — replaced with the same per-cell NaN-fraction tolerance
  already used for `CROP` selection, applied consistently to
  `train_eval_mask`/`eval_mask` and every downstream ground-truth
  comparison so zero-filled tolerance-cell timesteps can't leak into
  "true" values.

- **2026-08-15 — `notebooks/00_gaussianization_placement.ipynb` now crops to a
  representative ocean region instead of the full 721x1440 grid.**
  `GaussianCopula.fit`/`.transform` (placement i) internally flatten
  `(lat,lon)` into a `(n_bins, n_cells)` array regardless of caller dtype —
  at full grid scale that's ~1.04M cells and pushed peak RSS past 5GB even
  after the prior session's float32/dedup-copy fixes. Rewriting
  `src/train/copula.py` itself was explicitly ruled out of scope (real
  production code — it's also the Gaussianization step in
  `weather.model.WeatherModel.fit`, confirmed via `src/weather/model.py:246`
  — deserves its own dedicated task once there's empirical evidence to
  justify a specific change, not a same-session rewrite). A rectangular crop
  sidesteps the problem with zero flattening of any kind: it's just a
  smaller `(lat',lon',time)` tensor. Two candidate crops added as swappable
  constants (`CROP_10PCT_NAN` = 56x114 cells, `CROP_30PCT_NAN` = 70x144
  cells), both in the southeastern tropical Pacific (persistent trade
  winds) — found by searching for the largest rectangle under a per-cell
  NaN-tolerance threshold; naively avoiding only *true* land instead landed
  in swell-dominated high-southern latitudes where `wsh` is NaN up to 96%
  of the time. Both sizes clear a ~5x-per-axis compression floor against
  the notebook's fixed `R_LAT=10`/`R_LON=20` Tucker ranks, so placement
  (ii)'s core-coefficient comparison isn't trivially avoiding real
  compression. Verified end-to-end (setup + placements i/ii) against real
  archive data for both crops: peak RSS ~1.1GB (down from 5GB+), no crash.
  `docs/gaussian_copula_and_marginals.md` §0/§4's "never flatten `(lat,lon)`"
  rule and its explicit callout of `train/copula.py`'s current violation
  remain unaddressed by this change — deliberately deferred to a
  separately-scoped task once the crop-based results are in.

- **2026-08-14 — Fixed `tasks/15-wad-consistency-check-land-mask-artifact.md`:
  `_check_wad_wad_come_from_consistency` (`src/train/ingest.py`) now masks
  to ocean cells (via `was`'s NaN pattern, same wind grid) before computing
  the deviation-from-180° check, instead of the whole grid. Land cells were
  storing an identical raw fill code in both `wad` and `wad_come_from`,
  which decoded to the same degrees value and always tripped the previous
  whole-grid check, regardless of any real archive drift. Falls back to an
  unmasked check if `was` isn't present at the checked timestamp (e.g. a
  synthetic test archive). Two regression tests added to
  `tests/train/test_ingest.py` — one confirming the land-fill-code false
  positive no longer raises, one confirming a genuine ocean-cell mismatch
  still does. Verified directly against the real archive:
  `load_training_window` now loads real era-3 windows (2025-11-01,
  2026-02-01 checked) cleanly, reporting `max deviation 0.00° (ocean cells
  only)`. Full test suite: 152 passed (was 150). Tasks 00–14 are unblocked.

- **2026-08-14 — Filed `tasks/15-wad-consistency-check-land-mask-artifact.md`:
  `load_training_window` crashes on every real era-3 window.** First
  validation against the real archive since the data disk was reconnected
  found that `train.ingest._check_wad_wad_come_from_consistency` (run
  unconditionally by `load_training_window`, regardless of requested
  variables) always raises on real data:
  `ValueError: wad and wad_come_from do not differ by 180°... Max deviation: 180.00°`.
  Root cause diagnosed directly (checked 2025-11-01, 2025-12-15, 2026-03-01,
  2026-04-30): it's a land-cell fill-value artifact, not archive drift —
  over ocean cells only (masked via `was`'s NaN pattern) the deviation from
  180° is exactly 0.0° on every date checked; land cells store an identical
  raw fill code in both `wad` and `wad_come_from`, so the whole-grid
  `nanmax` the check currently uses always trips. §1.3's finding (`wad`/
  `wad_come_from` are a confirmed 180°-apart pair) is unaffected — only the
  guard's implementation is wrong. This blocks all 13 of tasks 00–14's
  notebooks from running against real data until fixed; see task 15 for the
  fix plan. Separately, confirmed the archive itself is otherwise clean: a
  full scan of 2025-11-01 → 2026-05-01 (superset of every task's window)
  found zero missing variables and zero contiguous gaps; the archive goes
  empty only after 2026-08-05T00:00Z, well past every task's window.

- **2026-08-13 — Fixed stale "ocean grid orientation unverified" wording
  across `src/core/config.py`, `src/core/io.py`, `src/train/ingest.py`,
  `src/core/archive_eras.py`, and the two matching tests
  (`tests/core/test_config.py`, `tests/train/test_ingest.py`).** An audit
  of `src/core/` and `src/train/` against all five ANSWERED
  `docs/open_questions.md` findings (§1.1–§1.5) found real code-level gaps
  only for §1.1 (ocean-grid orientation) — §1.2 (`pwd`/`pswd` convention),
  §1.3 (`wad` decoy), §1.4 (`pwh` duplicate), and §1.5 (mean vs. peak
  period) have no current-code counterpart to fix, since the pipeline
  stages that would misuse them don't exist in `src/` yet. For §1.1: every
  "orientation is unverified" comment/docstring/exception message was
  updated to state plainly that orientation **is** confirmed
  (`open_questions.md` §1.1 — row-0-south, column-0-at-0°E-eastward, via a
  direct RTOFS NetCDF lookup), while being explicit that
  `grid_coordinates("ocean")` still correctly raises for a different,
  unrelated reason: the grid is non-linear (Mercator projection south of
  ~47°N, non-zonal tripolar patch north of it), so `lat0 + row*step` cannot
  represent it — real per-cell 2D coordinate data is required and doesn't
  exist in this codebase yet. That remains a genuine, unaddressed gap (not
  fixed here — it needs a real data-storage decision, not a wording
  change). Full test suite (150 tests) passes.

- **2026-08-13 — Reorganized `tasks/` and `notebooks/` around the 15
  remaining open questions in `docs/open_questions.md`, replacing the
  stale v1→v2 code-migration task list as the active index.** The old
  17-task list (`tasks/00`–`16`) targeted `docs/model_design.md`, since
  superseded by the full `docs/` rewrite and `open_questions.md` — it
  wasn't touching any of the 15 questions still OPEN/PARTIALLY
  ANSWERED/PARTIALLY MOOT (§2.2, §2.4, §3.2, §3.4/§3.5, §4.1, §4.2, §5.3,
  §5.4, §5.5, §6.1, §7.1, §7.2, §8.1, §8.2, §8.3). Moved (not deleted) to
  `tasks/tbd/`; all 11 existing notebooks similarly moved to
  `notebooks/tbd/`. New `tasks/README.md` indexes 15 new tasks (00–14,
  two pairs sharing one experiment each per `open_questions.md`'s own
  cross-references: §4.2/§8.3, §2.2/§3.2), each with a "what question /
  steps / notebook(s) / conclusion" file, backed by 13 new notebooks with
  real starter code (`core`/`train` package APIs, real
  `load_training_window` calls, real mask construction) — checked for
  syntax validity and import resolution, but **not executed against real
  archive data**, since the archive wasn't accessible in this environment
  at write time. Conclusion sections are intentionally left blank pending
  that run.

- **2026-08-11 — Formalized a `log`-transform convention for wave-height
  and wave-period tensors across the documentation series
  (`variable_taxonomy.md` §3.4, cross-referenced from
  `tucker_decomposition.md`, `dynamic_mode_decomposition.md`,
  `gaussian_copula_and_marginals.md`, `harmonic_climatology.md`,
  `ornstein_uhlenbeck_processes.md`, `physical_consistency_constraints.md`,
  `correlated_variables.md`, and `open_questions.md` §§1.5/2.3/2.4/3.4/3.5).**
  Every wave-height field (`sigwh`, `wsh`, `pswh`) now gets an additional,
  independently Tucker-fit tensor $\log E := \log(H^2)$; every period
  field (`wsp`, `pwp`, `pswp`, i.e. $T_{m01}$) is represented as $T :=
  \log T_{m01}$, not raw units. Justified two ways: (1) heights/periods
  follow power-law fetch/duration growth (classical JONSWAP scaling),
  genuinely multiplicative, and `log` linearizes it for every L2-fit
  linear stage in the pipeline (harmonic climatology regression, Tucker's
  HOSVD/HOOI, DMD's snapshot-matrix fit); (2) it resolves a
  previously-open question (`open_questions.md` §2.3): $\log E$'s own
  reconstruction, $H=\exp(\log E/2)$, is now the authoritative magnitude
  source, replacing the biased `hypot(u_dir, v_dir)` — a fix
  `physical_consistency_constraints.md`'s energy-budget/PM checks now
  require explicitly. `sst` and `pwh` (already known degenerate) are
  excluded; whether the same treatment should extend to Class B wind/current
  *speed* (`was`, `ocs`) is left open (`variable_taxonomy.md` §3.2/§6).
  Tensor inventory grew from 15 to 17 (§5's table). `sst` and `sigwh`'s
  tensor *count* is unchanged — the height case is a content transform, not
  an added tensor, since there is no `hypot`-reconstruction problem for a
  Class A scalar.

- **2026-08-11 — Found and flagged an algebraic confound in the §3.4
  joint-channel-Tucker experiment's `E`/`T` diagnostic
  (`docs/correlated_variables.md` §6, `docs/open_questions.md` §3.5).**
  `E = 2log(Hs)` and `T = log(Tm01)` both reduce to expressions in the
  same spectral moment `m_0` with matching-sign coefficients (`E =
  log16+log m_0`, `T = log m_0 - log m_1`), so any observed `E`/`T`
  correlation — including the specific `E − 2T ≈ const` combination §3.4's
  experiment proposed checking — can't be cleanly attributed to genuine
  Miche-breaking-limit steepness coupling versus this shared-moment
  artifact. Proposed a decorrelated basis `{E, F}` (`F := E-T =
  log16+log m_1`, depending only on `m_1`) as a fix, itself flagged as
  unverified, and updated §3.4's experiment design to use it with an
  added null-baseline control.

- **2026-08-11 — Corrected this archive's `*p` fields (`wsp`, `pwp`,
  `pswp`) from "peak period" to "mean period" ($T_{m01}=m_0/m_1$) across
  `docs/variable_taxonomy.md`, `docs/physical_consistency_constraints.md`,
  and `docs/correlated_variables.md`.** Raised as a direct challenge from a
  separate design thread's framing ("ratios of moments, not a sinusoidal
  period"), which turned out to be correct — `variable_taxonomy.md` §2.2
  had it wrong. Confirmed via `noaa-etl`'s source (`wsp`/`pwp`/`pswp` derive
  from GRIB `shortName`s `mpww`/`perpw`/`mpts`) and NCEP's own GRIB2
  parameter table, which names these "Mean period of..." — distinct from
  NOAA's separately-numbered "Peak Wave Period" family this archive never
  reads. Most consequential fix: `correlated_variables.md` §3.3's
  Miche-breaking-limit derivation substitutes a period directly into the
  dispersion relation to get a specific wavelength — using $T_{m01}$
  unmodified there systematically underestimates the bound (conservative,
  not unsafe), and now carries an explicit note that a shape-dependent
  correction factor $T_p\approx\alpha\,T_{m01}$ is needed before applying
  the Miche constant. See `docs/open_questions.md` §1.5 for the full
  evidence trail.

- **2026-08-11 — New open question logged: joint multi-channel Tucker vs.
  downstream joint model for within-partition channel coupling
  (`docs/correlated_variables.md` §6, `docs/open_questions.md` §3.4).**
  Surfaced while reviewing a competing design document
  (`physical_handoff.md`) that proposes stacking one wave partition's
  derived `u, v, E, T` channels into a joint 4-mode Tucker, in tension with
  `tucker_decomposition.md`'s strict per-variable rule. Identified a real,
  structural asymmetry (independent per-channel truncation can discard a
  coupling that's weak-marginal/strong-joint, before any downstream joint
  model sees it) rather than dismissing the proposal outright, and
  specified a bounded, cheap experiment (one partition, rank-matched joint
  vs. independent-plus-CCA/joint-DMD comparison) to resolve it
  empirically. Left open pending that experiment, not decided by
  reasoning alone — same treatment as the still-open Gaussianization-
  placement question (§4.1).

- **2026-08-11 — Corrected `pwh`'s treatment as an independent wave-height
  measurement across `docs/variable_taxonomy.md`,
  `docs/correlated_variables.md`, and `docs/physical_consistency_constraints.md`.**
  `pwh` is a literal duplicate of `sigwh` at the ETL level (confirmed via
  `noaa-etl`'s `gfs_transform.py`, and independently re-confirmed directly
  against NOAA's own GRIB2 parameter table, [4.2-10-0](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-10-0.shtml),
  which defines `DIRPW`/`PERPW` for the "primary wave" but no
  corresponding height field at all — there was never an independent
  field for `pwh` to measure). Most consequentially, this fixed
  `physical_consistency_constraints.md`'s wave-energy-budget constraint,
  which had been specified as
  $\text{sigwh}^2\approx\text{wsh}^2+\text{pwh}^2+\text{pswh}^2$ — a
  formula that, once `pwh ≡ sigwh` is substituted in, double-counts
  `sigwh` against itself and would reject nearly every real sea state.
  Corrected to the physically meaningful two-term
  $\text{sigwh}^2\approx\text{wsh}^2+\text{pswh}^2$. See
  `docs/open_questions.md` §1.4 for the full evidence trail.

- **2026-08-11 — `docs/open_questions.md` §1.1 (ocean-grid orientation)
  resolved by direct geo-referenced lookup, closing what had been the
  archive's last open geometric unknown.** Previously, `sst`/`ocs`/`ocd`'s
  row-0 orientation could only be inferred indirectly (index-table
  structure, provenance archaeology in the `noaa-etl` repo), and remained
  formally open. Downloaded one real RTOFS diagnostic NetCDF file
  (`rtofs.20240127/rtofs_glo_2ds_f000_diag.nc`, from the public
  `s3://noaa-nws-rtofs-pds` bucket) and used its CF-convention 2-D
  `Latitude(Y,X)`/`Longitude(Y,X)` arrays (shape `(3298, 4500)`, matching
  `hycom_satocean_idx.npy`'s bounds exactly) to remap the full
  `(1121, 2880)` index table to real coordinates. Result: **row index
  increases with latitude (row-0-south), column index increases eastward
  from 0°E** — the same convention already confirmed for the wave/wind
  grids, achieved for HYCOM fields without `flip_latitude` because the
  index table's row order was evidently built south-first already. The
  Mercator-projected portion (south of ≈47°N) is zonal per row with
  non-uniform (Mercator-consistent) row spacing; north of 47°N the
  tripolar patch is no longer zonal, so per-cell coordinates there should
  come from the remap directly rather than a formula. This unblocks
  `core/config.py`'s `grid_coordinates` for the ocean grid, and removes the
  last reason `sst`/`ocs`/`ocd` were held at tier 2/3.

- **2026-07-28 — All `tasks/README.md` statuses reset to ⬜ (not done); v1→v2
  migration restarted from an understanding-first pass.** Tasks 00-08 and
  14-16 were previously marked ✅. Reset because prior "done" claims could
  not be trusted at face value: cross-checking `handoff.md` and
  `v3_implementation_handoff.md` against `docs/model_design.md` and the
  actual source showed that stated premises (e.g. "v2 is implemented and
  running," specific design-doc sections existing) did not hold up under
  direct verification. Rather than keep building new work (physical-
  consistency filtering, sampling rewrite, etc.) on top of possibly
  misunderstood ground, the decision was to restart: rebuild a verified
  mental model of the whole pipeline (Tucker → DMD → OU, Gaussianization
  placement, direction conventions) before resuming or trusting any task's
  status.
  - **Kept as-is:** existing code under `src/` and the notebooks under
    `notebooks/` — treated as a refactor target, not discarded ("easier to
    refactor than to start from scratch").
  - **Open decision, not yet made:** where Gaussianization should sit in the
    pipeline relative to Tucker (currently per-cell, pre-Tucker; see
    `handoff.md` §5) — this is the first concrete design question the
    restarted understanding pass needs to resolve before any refactor
    proceeds.
  - Next steps, in order: (1) build shared understanding of pipeline
    concepts, (2) refactor code/tasks to match that understanding, (3) train
    and get real metrics.

### Added

- Repo scaffolding: `pyproject.toml` (Poetry, black, isort, mypy config),
  `.gitignore`, `README.md`.
