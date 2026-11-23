# Changelog

All notable changes to this project are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **2026-11-22 — Probes behind the LIM tau-test stage kept** —
  `notebooks/probes/07_lim_tau_test_first_run.py`,
  `notebooks/probes/07_lim_bootstrap_envelope.py` (new).
  - **First run.** The first tau test on the production `wsh` core, with
    5–95% percentile bands. `notebooks/07_lim_tau_test_wsh_core.py`, which
    runs the pipeline stage, superseded it.
  - **Bootstrap envelope.** `diagnose_group`'s bands at 10 and 100
    replicates on a known coupled process. This backs the choice of
    min–max envelopes over percentiles.

  Both rerun against this tree.

- **2026-11-22 — Probes behind the v3 sampler decisions kept** —
  `notebooks/probes/sampler_jump_vs_chain.py`,
  `notebooks/probes/sampler_per_time_cost.py` (new).
  - **Jump versus chain.** Reproduces the lag-1 correlation that rejected
    the jump design: -0.006 jumping, 0.898 chained, 0.900 for the process.
    It now restates the jump design locally, because the helpers it used
    while the sampler was being rewritten no longer exist.
  - **Per-time cost.** Times the per-cell inverses and the interpolator
    per unique query time at era-3 size. Rerun on 2026-11-22: 105 ms and
    28 ms, against 290 ms and 326 ms when the decision was made; the cost
    depends on machine load.
  - **Dropped.** The fixture smoke and persistence probes fit the tier-1
    fixture, which the fidelity gate now always rejects. Tests and the
    script above cover their results.

- **2026-11-22 — Probe scripts get a home and a rule** — `AGENTS.md`
  (module layout, *Notebook Protocol*). `notebooks/probes/` holds the
  scripts behind recorded results. Every tracked script must run against
  the tree it is committed in; one broken by a later change is fixed or
  deleted, never kept broken.

- **2026-11-22 — The sampler's defects recorded, a generative fidelity gate
  specified, the open questions prioritized** — `docs/model_design.md`
  (summary, §5.4, §5.6, §5.6.1, new §5.6.3, §5.8, §7, §8.5, §8.6, §9 gotcha
  8), `docs/open_questions.md` (new Priorities section, new §4.3, §5.7,
  §5.8, §5.9, reopened §5.2, updated §7.2), `docs/correlated_variables.md`
  §4.4, `AGENTS.md`.
  - **Findings, on task 07's production `wsh` core.**
    - **Variance.** The sampler inflates the variance about 3.8-fold. The
      DMD modes are non-orthogonal, and their training amplitudes cancel;
      independent per-pair phases lose that.
    - **Unmodeled variance.** The 100-mode DMD span carries about 11% of
      the core's variance, and nothing models the rest.
    - **Memory.** Samples forget within about a day; the data keep 0.49
      autocorrelation at 4 days. That slow variability is what fails the
      tau test. A 6 h or 12 h training lag and AR(2)/AR(3) do not fix it.
    - **Daily cycle.** Wind sea keeps a physical, sun-locked daily cycle
      the climatology leaves in.
  - **Specified.** A training stage that simulates the fitted dynamics and
    compares them with the training cores on variance, multi-day memory
    and residual daily cycle. It fails the fit with no bypass (user
    decision).
  - **Prioritized.**
    1. Make the measurement trustworthy.
    2. Choose the space the dynamics are fit in (a new hypothesis: unit
       variance per core coefficient).
    3. Redesign the sampler.
    4. Add diurnal terms for `logE_wsh`.
    5. Land the gate.
    6. Revalidate task 07's `r_dmd`.

- **2026-11-22 — `fastjsonschema` 2.22.0 → 2.22.3 in the lock file** —
  `poetry.lock`. A dev-group transitive dependency (through `nbformat`); the
  lock now also records its `python >= 3.10` requirement. No runtime change.

- **2026-11-22 — Training-time memory scaling recorded as an open question**
  — `docs/open_questions.md` (new §9.1; "out of scope" renumbered to §10).
  A global era-3 fit needs ≥ ~300 GB for in-memory tier-1 preprocessing over
  12 months, and ~45–50 GB per masked-Tucker tensor. No machine available
  today has that. The fitted model is the same whether the work is split by
  cell, tensor or band. How the fit is computed (one large machine, split
  jobs, or out-of-core arrays) is left to be decided together with the
  training and production infrastructure.

- **2026-11-22 — `sample_batch` treats a call's query times as one
  trajectory** — `src/weather/simulate.py`, `tests/weather/test_simulate.py`,
  `docs/model_design.md` §5.8.
  - **Problem.** `sample_batch` jumped every unique query time from one
    stationary draw at `t_train_start`, each with independent noise. Every
    time had the right marginal, but beyond a few decorrelation times
    `φ^Δn ≈ 0`, so consecutive points along a route were independent draws.
    Measured on one channel with `φ = 0.9` at 3 h, a month past
    `t_train_start`: lag-1 correlation −0.006 between consecutive 3-hourly
    queries, against 0.90 for the process. v2's sampler had the same design.
  - **Fix.** Each group's channels are now one Markov chain over the sorted
    unique query times: a stationary draw at the first time, then the
    closed-form transition across each gap (`_channel_path`). That is the
    exact joint law at any set of times, at the same cost; each distinct
    gap is factorized once. The same lag-1 check gives 0.898.
    `simulate_field` uses the same function on its regular grid, so with the
    same seed `sample_batch` at a window's times and grid nodes returns
    `simulate_field`'s values there; a new test checks exactly that, across
    `sample_batch`'s time chunks, and fails against the previous code.
  - **Considered, not done (user decision, 2026-11-22).** Stencil-only
    per-cell inverses (each unique query time still pays about 0.6 s of
    full-grid work at era-3 size), renormalizing the bilinear weights over
    ocean corners at coasts (stencils touching land stay NaN), and chunking
    `simulate_field` over time.

- **2026-11-22 — v3 pipeline implemented: per-tensor Tucker, copula on the
  core, per-group DMD and pair-coherent OU, a v3 sampler** —
  `src/train/tucker.py`, `src/train/dmd.py`, `src/train/linear_inverse.py`,
  `src/train/local_residual.py`, `src/train/ingest.py`, `src/core/logger.py`
  (new), `src/weather/model.py`, `src/weather/simulate.py`,
  `src/weather/validate.py`, `tests/conftest.py`, `tests/train/test_tucker.py`,
  `tests/train/test_dmd.py`, `tests/train/test_ingest.py`,
  `tests/train/test_local_residual.py`, `tests/weather/*`,
  `docs/model_design.md` (§5.5, §5.6, §5.6.1, §5.6.2, §5.8, §7, §9 gotcha 8),
  `AGENTS.md`. Steps 3–6 of the v3 implementation, one change because the
  new `TuckerFit` breaks every caller at once.
  - **Handoff gap filled first.** v3's `ModeOUFit` could hold neither the
    conjugate-pair procedure task 07 validated nor correlated within-group
    noise. §5.6.1 now specifies one real AR(1) channel per real mode and per
    conjugate pair (log magnitude, deterministic phase), one innovation
    covariance per group, and closed-form fractional-step transitions; §5.8
    and §7 follow. Real channels draw real noise at the full variance; task
    07's script halved it by keeping the real part of complex noise.
  - **`train/tucker.py`.** `TuckerFit(factor_lat, factor_lon,
    reconstruction_error, n_iter)`; `tucker_fit` runs
    `partial_tucker_lowmem` in checkpoint chunks with chunk-to-chunk early
    stopping, returns the ALS core, and reports the error over ocean cells
    only. `fill_land` is the Tucker boundary and raises on NaN at an ocean
    cell. `tucker_inverse` takes optional `rows`/`cols` for the sampler's
    sub-grid. Contractions are batched matmuls: a three-operand `einsum`
    would loop naively at production size.
  - **`train/dmd.py`.** `dmd_fit` takes a group's `(T, n_G)` state (any
    trailing shape is flattened, so task 07 scripts that pass a 3-D core
    still run). New `group_state`, `split_group_state`, `conjugate_pairs`,
    `ou_transition`, `ou_stationary_cov`, `mode_amplitudes`; `fit_mode_ou`
    returns the new `ModeOUFit`, with `theta`/`sigma` as derived properties.
  - **`train/linear_inverse.py`.** `covariance_factor` extracted from
    `_simulate_linear` so the sampler reuses it; behaviour unchanged.
  - **`weather/model.py`.** `build_preprocess` derives the §5.3 Pipeline
    from the raw variables (sentinel → `(u, v)` with `keep_speed` → log →
    reproject → climatology → standardize). `fit` runs per-tensor Tucker and
    copula, per-group DMD, mode OU and `diagnose_group`, and logs Tucker
    errors, period sentinels, trend significance and diagnostics verdicts.
    `groups` must cover every tensor exactly once; `include_local_residual=True`
    raises (deferred, §5.7). Hyperparameters, fitted attributes,
    `from_components` and the per-tensor/per-group artifact layout follow
    §7 and §8.5.
  - **`weather/simulate.py`.** Correlated channel draws per group, exact
    conjugate pairs, copula inverse, Tucker inverse on the stencil rows and
    columns only, land masked, the Pipeline's per-cell steps before bilinear
    interpolation of `(u, v)`, `logE` and `T`, its pointwise steps after.
    Longitudes are taken modulo 360. Times before the training window raise:
    decaying DMD terms grow without bound backwards. Requested variables are
    checked before any sampling work. `sample_ou_equilibrium` and
    `simulate_ou_forward` are removed.
  - **`weather/validate.py`.** Real data takes the Pipeline's round trip
    through reprojection; check 7 KS-tests held-out Gaussianized core
    coefficients (§6.A.7); the report adds a §5.6.2 diagnostics page.
  - **Logging.** `core/logger.py` wraps one stdlib-backed structlog logger
    without configuring handlers or structlog globally; the three
    training-path `print` calls moved to it, and the ingest tests read it
    with `caplog`.
  - **Tests.** 230 pass. The tier-1 fixture gains a land cell (NaN `sigwh`
    and `was`) and an intermittently absent wind-sea partition, and puts
    wind on the wave grid's own lines as in era 3. New tests check, among
    others: the ocean mask ignores partition absence; heights halfway
    between nodes are the geometric mean (interpolation in log-energy
    space); land never leaves the sampler; pair amplitudes reconstruct a
    real field to 1e-10; transitions leave the stationary covariance
    invariant; correlated innovations are recovered. The fixture runs 2
    bootstrap replicates, which dominated fit time.
  - **Broken outside the gates.** `notebooks/07_dmd_ou_generative_check.py`
    imports the removed `simulate_ou_forward` and builds the old
    `ModeOUFit`; it stays as the record of task 07's validation.

- **2026-11-22 — `model_design.md` v3: the handoff now specifies the
  decided pipeline, ahead of the `model.py` rewrite** —
  `docs/model_design.md` (header and windows, new "What changed in v3",
  §3, §4, §5.1–§5.8 with new §5.3.1 and §5.5.1, §6 window dates, §7,
  §8.5, §8.6, §9, §10), `docs/open_questions.md` §6.2 (new) and summary
  table. `AGENTS.md` requires updating the handoff before implementing,
  and v2 still described the superseded joint design.
  - **Brought in from decisions already recorded elsewhere.**
    - One independent `(T, lat, lon)` tensor per field, 17 in total, with
      `logE` heights, `log T` periods and weighted `(u, v)`
      (`variable_taxonomy.md`).
    - A per-tensor masked Tucker via `partial_tucker_lowmem` at 40% of each
      spatial dimension (task 07), replacing the joint `r_ocean`/`r_var`
      fit.
    - The copula on each Tucker core (placement ii).
    - The land mask from `sigwh` and the partition sentinel fill (task 17).
    - DMD + mode OU per variable group at `r_dmd = 100`, and the §5.6.2
      diagnostics stage.
  - **New specification.**
    - `PartitionSentinelFiller` and `LogTransform` transformers in
      `train/masking.py`, and a `keep_speed` option on the directional
      transformers so class-C heights survive as `logE` tensors.
    - `TuckerFit` becomes `(factor_lat, factor_lon, reconstruction_error,
      n_iter)`, and `dmd_fit` takes a group's `(T, n_G)` joint state.
    - Per-tensor and per-group artifacts.
    - `WeatherModel` hyperparameters `tucker_rank_fraction`,
      `tucker_max_iter`, `tucker_tol`, `groups`, `lim_rank`,
      `tau_test_lags` and `n_bootstrap`.
    - A sampler order that inverts per-cell statistics on grid cells, then
      interpolates `(u, v)`, `logE` and `T`, then applies `exp`, `hypot`
      and `atan2` pointwise. v2 interpolated direction angles.
  - **Decided with the user (2026-11-22).**
    - Train on 12 months of era 3 (2025-10-27 → 2026-10-27) and validate
      from 2026-10-27 onward.
    - The local residual is deferred for the first era-3 model
      (`include_local_residual=False`); which space it would live in under
      placement (ii) stays open (`open_questions.md` §6.2).
  - **Flagged, not resolved.**
    - Per-tensor Tucker on the 12-month window needs about 15–20 h and
      45–50 GB per tensor, so it must run on a VM.
    - Direction and period are undefined where a sampled partition height
      falls below the 1 cm floor.
    - The 12-month trend term is weakly identified (task 11).
    - The production `wsh` core fails the tau test at 3 h.
  - **Minimal fixes outside the rewritten sections**, because v3 made them
    wrong: §2.2.1's era-3 end date (now open-ended); §6.A check 7, which
    KS-tested `transform` output against N(0, 1) and now tests the held-out
    Gaussianized core coefficients instead; and §12's 2022–2025 training
    window.

- **2026-11-21 — AGENTS.md: conventions ported from the radiologists
  preprocessing pipeline, plus a drift refresh** — `AGENTS.md`.
  - **Ported, adapted to 3.14 and this domain.** Python 3.14 feature table
    (PEP 758 unparenthesized `except` is Black-enforced here; PEP 765
    `finally` suppression banned; no sub-interpreters, free-threaded
    assumptions, t-strings or `compression.zstd`); `collections.abc` and
    PEP 695 alias rules; single-responsibility and explain-non-trivial
    design rules; an Error Handling section; testing philosophy (test
    intent, not implementation; fix the code, not the test); a Commits
    section (`:shortcode: type(scope): subject`, repo-local `exp` type for
    experiments, whole-file staging, no agent attribution, CHANGELOG entry
    in the same commit); a Summary Checklist.
  - **Logging rule, no migration yet.** No new `print` in `src/`; output
    goes through a future `structlog` logger in `core/logger.py`, created
    together with migrating the three existing training-path `print`
    calls (listed as known debt).
  - **Drift fixed.** Task Mapping marks every `src/` module as existing and
    adds `archive_eras.py`, `masked_tucker.py`, `masking.py`;
    `test_pipeline.ipynb` is recorded as parked in `notebooks/tbd/`;
    dependency lists match `pyproject.toml`; module layout covers `docs/`
    topic series, `tasks/`, numbered task notebooks and `azureml/`; a
    document-precedence section; a caveat that `src/` still implements
    Gaussianization placement (i) while (ii) is DECIDED; canonical
    `(time, lat, lon)` dim order and ocean-grid orientation added to the
    coordinate rules; `Settings` example and tool commands corrected.
  - **Not ported.** 120-char lines, Google docstrings, `uv`/`make`, `src.`
    imports, the API/database/clinical layers.

- **2026-11-21 — Linear-inverse-model diagnostics for correlated-variable
  groups as an explicit training-pipeline stage: tau test, correlated
  noise, information lost by independent fits** — `src/train/linear_inverse.py`
  (new; `group_pcs`, `fit_lim`, `tau_test`, `group_coupling_report`,
  `simulate_lim`, and the stage function `diagnose_group` with
  `diagnostics_summary`; `tests/train/test_linear_inverse.py`, 21 tests),
  `docs/model_design.md` §4 diagram and properties, §5.6.2 (new), §7, §8.5,
  §8.6 Cell 7b, `docs/correlated_variables.md` §4.4 (new) and §7 (new,
  related literature), `docs/open_questions.md` §3.2 update,
  `notebooks/07_lim_tau_test_wsh_core.py` (new), `AGENTS.md` (Task Mapping
  row, training-architecture paragraph, module-layout entry).
  - **Pipeline stage.** `diagnose_group` runs once per variable group on the
    Gaussianized cores DMD consumes, judging the tau test against an
    envelope of LIM-simulated replicates and the coupling metrics against
    an independence null (block-diagonal LIM). Results are stored on the
    model (`group_diagnostics_`, persisted as `group_diagnostics.json` +
    `lim_<group>.npz`) and logged, and never alter or gate the fit.
    `WeatherModel` gains `groups`, `lim_rank`, `tau_test_lags` and
    `n_bootstrap` hyperparameters in the handoff; wiring them in is part of
    the pending `model.py` rewrite.
  - **Bands are bootstrap envelopes, not percentiles.** The first draft used
    5th/95th percentiles. With 10 replicates, a true OU process failed its
    own 6 h tau test (ratio 1.003 against [0.995, 1.002]; 100 replicates
    give [0.995, 1.006]), and a nominal 90% band flags a perfect model on
    at least one of four lags about half the time. The envelope's
    false-alarm rate is 2/(n+1), about 3% per lag at the default n = 100.
    The stage tests therefore run at n = 100, on 10,000-step series.
  - **Memory fix.** `group_pcs` always formed the T × T Gram matrix, which
    is 20 GB for a 50,000-step, low-dimensional series and was killed by
    the OOM killer. It now decomposes whichever of the Gram or covariance
    matrix is smaller.
  - **Framework.** A literature search found no Lotka–Volterra model of wave
    energy, but found that §4.2's joint-group DMD is a Linear Inverse Model
    (Penland & Sardeshmukh 1995; POPs, Hasselmann 1988). LIM brings the tau
    test (lag invariance of the fitted operator) and fluctuation–dissipation
    noise covariance.
  - **Specification.** Within-group stochastic forcing is now specified as
    one jointly correlated Gaussian draw, not independent per-mode or
    per-variable draws. The production sampler port is deferred to the
    `weather/simulate.py` rewrite.
  - **Metrics, in nats per step.** Granger/transfer entropy into each member
    (what an independent fit's dynamics lose) and the multi-information of
    the joint innovations (what independent noise loses), plus each
    member's truncation loss.
  - **First result.** Run through `diagnose_group` with 100 replicates, the
    production `wsh` 40% core fails the tau test at every lag from 6 h to
    48 h at both 20 and 100 PCs. Its error-variance ratio of 0.70–0.93,
    against bootstrap envelopes near 1, means the core stays predictable
    longer than a 3 h linear-Markov fit implies, because fast 3 h
    variability masks slower memory. The 100 PCs retain only 11.4% of the
    core's variance. Since this LIM is essentially production DMD's reduced
    operator, the finding applies to the `r_dmd = 100` fit. Candidate
    responses (longer training lag, Hankel DMD) are untested.

- **2026-09-29 — Closed task 07: Tucker rank decided as 40% (248, 576),
  production copula + DMD (`r_dmd=100`) fit on the full 40% core, Azure
  VM infrastructure torn down** — `notebooks/07_placement2_tucker_on_anomalies.py`
  (added `--early_stop_tol`, fractional-scale checkpoint-to-checkpoint
  early stopping), `notebooks/07_placement2_copula_dmd_40pct.py` (new),
  `tasks/07-tucker-dmd-rank-selection.md` §12. Final 40% VM run (rank
  (248,576), 11h cap after an 8h cap proved too short) hit its wall-clock
  cap at iteration 170/300, err_pct=23.2962% — one checkpoint shy of its
  own `1e-4` early-stop threshold, effectively converged. Copula and DMD
  (`r_dmd=100`, the rank decided in §10) both trained on the **full**
  2192-timestep core, no train/holdout split (validation happens
  separately against real data) — copula fit in 110s, DMD fit in 41.1s.
  Decided pragmatically (not via a rigorous knee-point argument like
  `r_dmd`, since Azure credit is now exhausted): 40% checkpoint error
  (23.30%) beats 30% (27.47%) and 20%'s converged baseline (32.90%).
  Deleted the VM, both its disks, NIC, public IP, NSG, and VNET; left the
  raw data archive (`wxgenarchivee2`) and ML workspace resources
  untouched. Also recorded three operational lessons hit for real this
  run: Python's stdout block-buffers when redirected to a file (trust the
  checkpoint CSV, not the log, for live status); `rsync --partial`,
  killed mid-transfer, can rename its own incomplete download over an
  already-good file at the same destination (caused one real 1.17GB
  file corruption, recovered by re-downloading from the VM); and a local
  background process — even one detached with `setsid`+`nohup` — did not
  survive the local IDE crashing, while jobs on the VM (a separate
  machine, run in `tmux`) survived every crash without issue.

- **2026-09-27 — Confirmed Tucker's spatial residual and DMD's temporal
  residual are genuinely different signals, not just different in
  principle: the core doesn't carry the storm signature, and the core
  itself is not a coarsened version of the map** —
  `tasks/07-tucker-dmd-rank-selection.md` §11,
  `notebooks/07_tucker_core_timelapse_gif.py` (new). corr(spatial
  max-\|error\|, core energy/frame-to-frame core change) = 0.19/0.16,
  weak — versus the 0.83-0.94 cross-rank correlation found in the spatial
  error itself (previous entry). Visualized the raw core directly
  (`notebooks/tucker_core_timelapse_20pct.gif`, 548 frames, 50.5MB): no
  continents or basin-scale structure, just noise-like texture with
  energy concentrated toward low mode-indices (surviving SVD ordering,
  not spatial structure) — and it compressed far worse as a GIF than the
  smooth field did, an incidental but consistent confirmation. Net
  finding: whatever spatial detail a given Tucker rank can't represent is
  discarded before the core is ever computed, so DMD/OU structurally
  cannot compensate for it — confirms Tucker rank and `r_dmd` are correctly
  treated as two independent decisions in this task, not one tradeoff.

- **2026-09-27 — Animated true-vs-reconstruction timelapse across all
  three Tucker ranks (20/30/40%), 548 frames over the full 9-month window
  at 00/12 UTC daily** — `notebooks/07_tucker_rank_timelapse_gif.py`
  (new), `notebooks/tucker_rank_timelapse.gif`,
  `tasks/07-tucker-dmd-rank-selection.md` §5. Confirms the static
  two-timestep finding generalizes: the coastline error trace is fixed
  frame to frame, open-ocean error visibly follows storm/eddy boundaries
  as weather evolves. Checked per-frame RMSE across all 548 frames (not
  just the 2 originally-cached timesteps): the 20%>30%>40% ordering holds
  at every frame with no overlap, and each rank's error is stable around
  its own mean (total linear drift over the whole window ~= 1 standard
  deviation of frame-to-frame noise) — confirms this residual is
  genuinely weather-driven, not a leftover seasonal trend. GIF file size
  needed active management: dpi=100 rendered at 345.8MB, dpi=70 at
  175.9MB, settled on dpi=45 (75.2MB, still fully legible) — GIF
  compresses this kind of high-detail, low-frame-to-frame-redundancy
  content poorly regardless of resolution.

- **2026-09-27 — 40% Tucker rank's reconstruction-quality diagnostics
  measured, completing a 3-rank comparison (20%/30%/40%) — more rank keeps
  helping overall, but a small extreme tail of cells resists it** —
  `tasks/07-tucker-dmd-rank-selection.md` §5. 40% costs ~1.6x more per
  iteration than 30% (~201 vs ~126 s/iter) and hit its 3h backstop at
  iteration 50/80 (23.8276%) rather than completing — recovered fine
  thanks to the per-checkpoint save. Across all three ranks: overall RMSE
  and the coastal/open ratio both improve monotonically (0.336→0.284→0.244;
  1.33x→1.29x→1.23x), but the error-distribution-shape trio worsens
  monotonically at every step (percentile-ratio 24.5→27.7→28.3, skewness
  3.01→3.35→3.51, excess kurtosis 10.8→13.6→15.3). Root-caused precisely
  this time: it's not the top quartile falling behind (Q4 actually
  improved *faster* than Q1 going 30%→40%) — it's specifically the single
  most extreme cells (driving the distribution's max) improving much more
  slowly than the bulk's own typical spread (IQR) shrinks, so the "how far
  out is the worst case" ratio keeps growing even as every absolute number
  gets better. Same Gibbs-phenomenon-style mechanism as before, now
  characterized more precisely: not a general transition-zone problem, a
  specific-extreme-tail one.

- **2026-09-27 — 30% Tucker rank's reconstruction-quality diagnostics
  measured: more rank helps the aggregate error, but disagrees with the
  20% baseline on where that help goes** — `tasks/07-tucker-dmd-rank-selection.md`
  §5. Short deliberate re-run (80 iterations, completed naturally this
  time thanks to yesterday's per-checkpoint `.npz` fix) to recover tensors
  the 8h run had lost. Same four measurements as the 20% baseline: overall
  RMSE improved (0.336→0.284) and the coastal ratio barely moved
  (1.33x→1.29x), but every concentration measure got *worse* — half-ratio
  2.52x→2.69x, (max−median)/IQR 24.5→27.7, skewness 3.01→3.35, excess
  kurtosis 10.8→13.6. Root cause: the flattest quartile of open-ocean
  cells improved 19.6% going 20%→30%, the sharpest quartile only 13.7% —
  the extra rank is disproportionately sharpening already-easy regions,
  not closing the gap at genuine spatial transitions. Exactly the
  "disagreement between metrics" case flagged as the interesting one when
  this diagnostic suite was designed. 40% reconstruction-check re-run
  launched immediately after, same pattern, in progress.

- **2026-09-26 — 30% Tucker rank re-run hit its 8h cap without converging
  (iter 220/300, err_pct=27.4651%); found and fixed a gap where the run's
  Tucker factors were never saved** — `tasks/07-tucker-dmd-rank-selection.md`
  §5, `notebooks/07_placement2_tucker_on_anomalies.py`. Power-law fit of
  the error-delta trajectory (`delta ~ iter^-p`) stabilized at p=1.387 as
  more checkpoints accumulated (up from p=1.339 using only the first 60
  iterations) — confirms real, if slow, convergence, but also confirms
  early extrapolation was unreliable: the iter≤60 fit predicted 25.91% at
  iteration 165; actual was 27.55%, 1.64pp worse. Even short of full
  convergence, 27.4651% is a 16.5% relative error reduction over 20%'s
  converged 32.8975%. Separately: `07_placement2_tucker_on_anomalies.py`
  only saved its `(core, factor_lat, factor_lon)` `.npz` bundle after the
  checkpoint loop's normal completion — a process killed early by the
  wall-clock cap never reached that line, so this run's factors were lost
  entirely (only the checkpoint CSV survived), meaning no spatial
  reconstruction check is possible for 30% this round. Fixed going
  forward: the save now happens every checkpoint, overwriting the same
  path, so any future run always leaves a recoverable snapshot regardless
  of how it ends. Per the standing instruction, a cap-hit-without-
  converging outcome means the VM was deallocated immediately and 40% was
  **not** auto-launched (that only happens on a 30% success) — 40% remains
  a deliberate next decision.

- **2026-09-26 — New spatially-resolved reconstruction-error diagnostics
  for Tucker rank comparison, beyond the single aggregate `err_pct`
  number** — `notebooks/07_tucker_reconstruction_visual_check.py` (new),
  `notebooks/07_tucker_checkpoint_timelapse.py` (new),
  `docs/tucker_decomposition.md` §5.1,
  `tasks/07-tucker-dmd-rank-selection.md` §5. Four complementary
  measurements answer *where* reconstruction error concentrates, not just
  how much of it there is: (1) coastal-vs-open-ocean RMSE ratio; (2)
  correlation of `|error|` against the true field's local spatial-gradient
  magnitude, open-ocean only, plus a gradient-magnitude-quartile RMSE
  breakdown; (3) a single-number upper-half/lower-half (sharper 50%
  vs. flatter 50%, by local gradient) RMSE ratio; (4) percentile ratio
  `(P100-P50)/IQR`, skewness, and excess kurtosis of the open-ocean
  `|error|` distribution itself. Measured on `wsh` at 20% rank
  (124,288) against two cached true-field time-slices, extracted from the
  VM's cached tensor via `mmap_mode="r"` (never loading the full ~7.8GB
  array): coastal/open ratio 1.33x, gradient correlation 0.75, half-ratio
  2.52x, percentile-ratio/skewness/excess-kurtosis 24.5/3.0/10.8 — all
  agreeing that error is heavily concentrated at sharp spatial transitions
  (coastlines and open-ocean storm/eddy edges alike), not spread evenly.
  Visualized in `notebooks/tucker_reconstruction_check_20pct.png` (true /
  reconstructed / error maps; needed a percentile-based, not raw-max,
  color scale — a handful of extreme storm pixels otherwise washed out the
  whole plot to near-white) and
  `notebooks/tucker_checkpoint_timelapse_30pct_8hcap.png` (error and
  per-checkpoint error-delta vs. iteration, for reading convergence
  plateaus off a live or finished checkpoint CSV). Reusable against any
  future rank's `.npz` for a direct, apples-to-apples cross-rank
  comparison once 30%/40% converge.

- **2026-09-26 — VM-validated the Tucker cost-cliff fix at 30% rank: real
  ~1.72x speedup confirmed, but not enough to finish in a 3h box; also
  corrected an earlier "zero checkpoints" claim that turned out to be
  wrong** — `tasks/07-tucker-dmd-rank-selection.md` §5. Ran the fixed
  `partial_tucker_lowmem` on the actual Azure VM at (186,432)/30%,
  wall-clock-boxed to 3h via a remote `timeout`. Completed 80/300 requested
  iterations before the cap fired; steady-state pace ~128 s/iter vs. the
  old code's ~220 s/iter at the same rank (~1.72x speedup, real but more
  modest than hoped — the SVD-method fix only cheapens the SVD portion of
  each iteration, not the rank-independent tensor-contraction cost, which
  forms a floor). Error decreased monotonically and smoothly throughout
  (29.32%→27.79%), no memory blowup (~30GB peak of a 58GB cap, no swap).
  While retrieving this data, found the two previously-"killed at ~11h/
  ~11.5h, zero checkpoints" runs (30% and a separate 40% placement-i
  attempt) had actually been making real, steady, monotonically-improving
  progress the whole time (168/200 and 132/200 iterations respectively) —
  the checkpoint CSVs were simply never checked before being stopped and
  before writing that claim down. Corrected in the task doc. Even
  unconverged, 30% rank at iteration 80 already beats 20%'s fully-converged
  error (27.79% vs 32.90%) — a real, decision-relevant result even though
  the 40% rank attempt wasn't reached tonight (per plan, only proceeds
  after a 30% success within its own cap) and Tucker rank remains
  undecided. Also fixed two unrelated VM infra issues found along the way:
  an NSG rule only allowing SSH from a stale IP, and `results/` having
  grown to 19GB on a 95%-full 29GB OS disk (moved to the attached 256GB
  data disk). VM deallocated immediately once the cap fired.

- **2026-09-26 — Diagnosed and fixed the Tucker rank cost cliff — two real
  bugs in how this repo calls tensorly's masked `partial_tucker`, not a
  language/speed problem** — `src/train/masked_tucker.py` (new),
  `tests/train/test_masked_tucker.py` (new),
  `tasks/07-tucker-dmd-rank-selection.md` §5. Read tensorly 0.9.0's
  installed source directly: (1) the per-iteration masked-cell
  re-imputation materializes ~4-5 full-size temporary arrays every
  iteration, rank-independent (~31 GiB at era-2 scale for this line
  alone); (2) the per-iteration factor-update SVD call never propagates
  `method=svd`, so `svd="randomized_svd"` is silently ignored after
  initialization — every iteration actually runs a full economy LAPACK
  SVD instead. Fixed both via a small vendored `partial_tucker_lowmem`
  (in-place mask-aware `copyto`, since this repo's mask is always strictly
  `{0,1}`; propagates the requested SVD method every iteration), now used
  by both rank-sweep notebook scripts in place of tensorly's function
  directly. Verified locally against tensorly's own function for exact
  numerical equivalence (holding the SVD method fixed) and structure
  recovery (once genuinely using randomized SVD); full suite + mypy
  --strict clean. Not yet validated on the VM — Numba/C++ were
  considered and ruled out (the bottleneck is memory + an algorithm bug,
  not Python-loop overhead; the hot path already runs compiled BLAS/
  LAPACK code).

- **2026-09-25 — Task 07: decided the final production `r_dmd = 100`** —
  `docs/open_questions.md` §5.3, `tasks/07-tucker-dmd-rank-selection.md`
  §10, plots in `notebooks/dmd_rank_fidelity_tradeoff.png` and
  `notebooks/dmd_rank_cv_predictive_error.png`
  (`notebooks/07_dmd_rank_summary_plots.py`). The generative check's two
  fidelity metrics disagree on where "best" is (spectral at `r_dmd=150`,
  marginal/KS at `r_dmd=80`); `r_dmd=100` is the knee point between them
  — moving to either neighbor buys a small gain on one metric at a
  disproportionate cost on the other (e.g. `r=80` trades a 12% better KS
  for a 121% worse spectral match). The real-data CV result is flat
  across all three candidates, so it imposes no penalty on this choice.
  Closes the DMD half of task 07's rank-selection question; Tucker's own
  rank remains open.

- **2026-09-25 — Task 07: real-data (no surrogate) 8-fold cross-validation
  confirms the DMD rank finding is not a single-split artifact —
  `docs/open_questions.md` §5.3, `tasks/07-tucker-dmd-rank-selection.md`
  §9.** Ran `notebooks/07_dmd_rank_cv_check.py` (design:
  `dmd_rank_cv_plan.md`, now absorbed and deleted): 8-fold expanding-
  window walk-forward CV (`TimeSeriesSplit`) on the same cached real
  core, no synthetic surrogate at all, with a rolling anchor through each
  fold's entire ~190-step real held-out block (142 anchors/fold) rather
  than the single training-boundary anchor prior checks used.
  Memory-capped at 25 GiB (`RLIMIT_AS`, never triggered). Result:
  one-step predictive accuracy barely moves across the whole
  `r_dmd=5-200` range (under 0.3 percentage points worst-to-best, ~4x
  smaller than the original single-split test's ~1.2-point spread), and
  this near-flatness replicates almost identically across all 8
  independent real windows (fold-to-fold std 0.07-0.13 points). This is
  the direct real-data confirmation of what the phase-randomized
  surrogate test implied via a synthetic null model: rank truly doesn't
  matter much for predictive accuracy here, not an artifact of the one
  split every prior check rested on. `r_dmd` recommendation unchanged,
  now on firmer footing: 50-100.

- **2026-09-24 — Task 07: decisively confirmed the DMD "no elbow" result
  is intrinsic noise, not missing structure or a search bug, and fixed a
  real complex-conjugate-pair gap in OU generation along the way —
  `docs/open_questions.md` §5.2, §5.3, `tasks/07-tucker-dmd-rank-selection.md`
  §8.** Built 20 independent phase-randomized (Prichard-Theiler
  multivariate FFT) surrogates of the Gaussianized core — stationary
  Gaussian processes sharing the real data's exact second-order
  statistics and nothing more — and reran the full DMD rank sweep on each
  (`notebooks/07_dmd_surrogate_rank_check.py`). At every `r_dmd` from 5 to
  200, real held-out error sits inside the surrogate band: the anomaly's
  high effective rank is statistically indistinguishable from correlated
  noise, not recoverable structure a higher rank could capture. Separately,
  a full DMD+OU generative fidelity check
  (`notebooks/07_dmd_ou_generative_check.py`) first exposed a real bug:
  sampling each DMD mode's OU residual independently ignores that exact
  DMD on real data produces algebraically exact conjugate mode pairs,
  making the generated signal comparably imaginary as real
  (`imag_real_ratio_mean` ≈0.93-1.04 at every rank) rather than safely
  real-with-noise. Fixed by modeling only the shared magnitude
  ($|\eta_k(t)|=|\eta_{k'}(t)|$ always, in log space) stochastically per
  conjugate pair, with phase held deterministic at the mode's own
  frequency — automatically conjugate-exact by construction, verified to
  ≈1e-17 (machine precision) at every rank on the real 35,712-dimensional
  core. This also closes `docs/open_questions.md` §5.2's long-open
  "specific procedure" question for conjugate-pair sampling (production
  port to `weather/simulate.py` remains task 9's job). After the fix,
  spectral match keeps improving to `r_dmd≈150` then plateaus, while the
  marginal-distribution match (KS) bottoms out around `r_dmd=50-80` and
  degrades again by `200` — plausibly `cond(Φ)`'s known rise across this
  range destabilizing the generative step numerically, a second,
  independent reason (not yet confirmed) to prefer a moderate rank over
  the top of the tested range. All four scripts ran entirely locally
  against the already-cached core; no VM or Azure budget spent.

- **2026-09-19 — Researched what real physics-based ocean/atmosphere
  forecast models do, to interpret task 07's "no elbow found" DMD result
  — `docs/open_questions.md` §5.3, `tasks/07-tucker-dmd-rank-selection.md`
  §7.** Operational wave models (WAVEWATCH III/SWAN/WAM) solve the
  phase-averaged spectral wave action balance equation, not Navier-Stokes
  — true NS/Boussinesq/RANS models are 10-200x more expensive and
  restricted to local/regional, minutes-to-days use, never basin-scale
  multi-day forecasting, so this project's Tucker/DMD approach is in the
  same statistical/spectral family as real operational practice, not a
  simplification of a "more correct" method. Atmospheric predictability
  has a hard, chaos-imposed ceiling around two weeks (operational wave
  forecasts run 5-7.5 days out); past that horizon, poor skill is
  provably not a model-quality problem, which reframes the ~1287-mode
  requirement found in the previous entry as plausibly real deterministic
  structure up to some horizon plus an irreducibly chaotic remainder no
  rank fixes. Flagged (not actioned) one concrete lead: swell is far more
  predictable than local wind-sea, so separating the two before
  Tucker/DMD — analogous to climatology removal — could concentrate more
  learnable structure into DMD and leave a cleaner residual for OU. Also
  confirmed the DMD+OU split itself matches real operational practice
  (SPPT/SKEB stochastic parameterization; OU-modeled EOF/mode-amplitude
  autocorrelation in stochastic climate models), not a bespoke design
  choice. Full source list (20+ papers/references, grouped by topic) in
  the task doc.

- **2026-09-18 — Task 07's DMD half: ran the first `r_dmd` sweep against a
  real Gaussianized core under placement (ii); no elbow found, but settled
  whether climatology removal is the cause (it isn't) —
  `docs/open_questions.md` §5.3, `tasks/07-tucker-dmd-rank-selection.md`
  §6.** Swept `r_dmd \in \{5,...,200\}` on the 20%-rank era-2 Tucker core
  (`notebooks/07_dmd_rank_sweep.py`, copula and DMD both fit train-only on
  an 80/20 split to avoid leaking the holdout period into the copula's
  LUT): no growing modes anywhere, `cond(Phi)` climbing smoothly
  2.2→227 with no sharp knee, and a corrected one-step-ahead error flat at
  ~99% across the whole range — meaning `docs/open_questions.md` §5.3's
  2026-08-16 finding that `cond(Phi)` tracks held-out quality does not
  clearly hold once DMD's own input has been Gaussianized (that finding
  was established on a raw, non-Gaussianized core). Root-caused the flat
  ~99% error to rank capacity: a POD/SVD of the same core shows only
  31.32% of its variance captured by 200 of 1754 possible modes (90%
  needs ~1287). Getting a trustworthy held-out metric took two failed
  attempts, both silent/plausible-looking wrong numbers rather than
  crashes — worth a look in the task doc if revisiting this sweep.
  Tested directly whether this is an artifact of removing climatology
  before Tucker/DMD (`notebooks/07_dmd_climatology_only_check.py`,
  projecting `ClimatologyRemover`'s fitted signal alone through the same
  Tucker factors): needs only 3 modes for 90% of its own variance, and
  DMD at `r_dmd=6` recovers the exact 365.3-day/182.6-day annual/
  semi-annual periods at 0.07% error — confirming climatology genuinely
  is the low-rank part of the signal, but arguing *against* putting it
  back into DMD's input, since that structure is already captured exactly
  and cheaply by the harmonic fit, and the anomaly's own high rank
  requirement wouldn't shrink just because an unrelated low-rank
  component sits alongside it. **Decision: climatology removal stays
  exactly where it is.** `r_dmd` itself remains undecided — this sweep
  was never memory/cost-bound (unlike Tucker's), so extending past 200 is
  cheap and a natural next step.

- **2026-09-13 — Reversed the Gaussianization-placement decision
  (`docs/open_questions.md` §4.1): placement (ii) — Tucker on
  climatology-removed/standardized anomalies, Gaussianize the Tucker core,
  before DMD — supersedes the 2026-08-19 placement (i) default.** Task 07's
  era-2 cloud rank sweep ran the task-07-derived i-vs-ii comparison that
  both the 2026-08-19 decision and `tasks/00-gaussianization-placement.md`
  had left as deferred follow-on work: same Tucker rank (31,72/5%) under
  each placement gave `err_pct` 50.81% (i) vs. 53.04% (ii) — statistically
  indistinguishable, contradicting the linear-rank-inflation hypothesis
  that motivated re-testing this (per-cell Gaussianization provably
  preserves rank/copula dependence by Sklar's theorem, but not the
  linear/Pearson structure Tucker actually optimizes against). Placement
  (ii) is adopted anyway, on architectural grounds: Tucker stays in
  physically-grounded log-energy anomaly space, and Gaussianization moves
  to sit immediately before DMD, the one stage that actually needs
  Gaussian innovations, rather than broadly upstream of a stage that
  doesn't benefit from it. `docs/open_questions.md` §4.1,
  `docs/gaussian_copula_and_marginals.md` §6.1,
  `tasks/00-gaussianization-placement.md`, `tasks/07-tucker-dmd-rank-
  selection.md` §5, and `tasks/README.md` updated accordingly.

- **2026-09-13 — Fixed a near-zero-variance division blow-up in
  `PerCellStandardizer` (`src/train/standardize.py`), found while running
  the placement (ii) comparison above.** A handful of cells (intermittently
  ice-covered, near-constant through most of the 2-year fit window) got a
  fitted `sigma_cell_` numerically near zero; dividing by it at transform
  time produced values up to `~1e15` (verified: 21 of 500,773 era-2 ocean
  cells, 322 cell-timesteps total). Previously invisible because placement
  (i)'s `GaussianCopula.transform` clips every value into a bounded
  quantile range downstream, silently absorbing the blow-up — placement
  (ii) feeds standardized anomalies straight to Tucker with no such
  clamp, so it's the first place this ever surfaced. Fixed by flooring the
  divisor to `1.0` below a `1e-6` threshold inside `transform`/
  `inverse_transform` only (`_floored_divisor`) — deliberately **not**
  applied to the stored `sigma_cell_` itself, since
  `weather.model._log_trend_significance` separately consults the true
  fitted sigma to flag exactly these low-variance/high-trend cells, and
  flooring the stored attribute would defeat that diagnostic. `NaN` (land)
  cells are unaffected (`nan < eps` is `False`). 3 new tests added to
  `tests/train/test_standardize.py`; full suite (176 tests) and
  `mypy --strict` green.

- **2026-09-13 — Task 07 era-2 rank sweep: 20% rank (124,288) working
  baseline (32.90% err) under the new placement (ii); found a practical
  cost ceiling around 20-30% rank, final production rank deferred
  (`tasks/07-tucker-dmd-rank-selection.md` §5).** On a 9-month,
  multi-season era-2 window (2024-01-01 → 2024-10-01, `T=2192`,
  621×1440 grid) — a materially harder case than the single-season crop
  task 07 concluded on earlier (§4 below) — swept 5% (31,72, 53.04% err)
  and 20% (124,288, 32.90% err) to completion under placement (ii).
  Per-iteration cost scaled mildly between them (2.3x time for a 4x
  per-mode rank increase), but two independent 30-40%-rank attempts (one
  under each placement) were each killed after 11-11.5 hours with **zero**
  completed checkpoints — a real cost cliff, not a memory ceiling this
  time (peak RSS stayed well under the VM's cap throughout), likely
  because `randomized_svd`'s efficiency assumption (target rank small
  relative to the matrix dimension) stops holding around 30-40% of a 1440-
  cell dimension. 20% is recorded as the current working baseline, not a
  final answer; deciding a production rank is deferred pending more
  compute budget (the Azure subscription funding these runs had ~$59.62
  in credit remaining as of 2026-09-13, expiring 2026-09-29) or a
  cheaper/faster Tucker fitting approach able to actually reach 30-40%
  for a fair comparison.

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

### Fixed

- **2026-11-22 — `Reprojector` passes wind through instead of interpolating it
  onto an identical grid** — `src/train/reproject.py`,
  `tests/train/test_reproject.py`. Wind shares the wave grid (§2.2) but is
  loaded under `wind_lat`/`wind_lon` dims, so `Reprojector` bilinearly
  interpolated it onto the same grid lines. That is not a no-op: scipy's
  linear stencil carries a zero-weight neighbour, and `NaN · 0` is NaN, so
  each land NaN in `was` spread to the ocean cells south and west of it. v2's
  intersection ocean mask silently dropped that coastal ring; v3's
  `sigwh`-only mask would instead stop the fit at the Tucker boundary. A
  variable whose coordinates match the target grid to 1e-9° is now relabelled
  and passed through, as §5.2 already specified. Found while building the v3
  test fixtures.

### Added

- **2026-11-22 — Generative fidelity stage gates training; the test suite is
  red until the sampler is redesigned** — `src/train/fidelity.py` (new),
  `src/weather/model.py`, `src/train/dmd.py`, `src/train/copula.py`,
  `src/train/linear_inverse.py`, `src/weather/simulate.py`,
  `src/weather/validate.py`, `tests/conftest.py`,
  `tests/weather/test_model.py`, `notebooks/07_generative_fidelity_wsh_core.py`
  (new). Spec: `docs/model_design.md` §5.6.3.
  - **The stage.** `diagnose_fidelity` simulates each group's fitted DMD and
    mode OU exactly as the sampler does: `channel_path`, which moved here
    from `weather/simulate.py`, then `mode_amplitudes`. Per member it
    judges the variance, the autocorrelation of the leading PCs at
    `fidelity_lags` (3 h to 16 days) and the residual daily cycle against
    replicate envelopes and fixed tolerances. Member energies come from
    `r × r` Gram matrices, so the sampled state is never materialized.
  - **The gate.** `WeatherModel.fit` and `from_components` raise
    `ValueError` with `fidelity_summary`'s text when a check fails; there
    is no bypass. Results are persisted as `fidelity_diagnostics.json` and
    get a page in `full_report`.
  - **New hyperparameters.** `fidelity_lags`, and `fidelity_replicates`
    (100). The plan was to reuse `n_bootstrap`, but the fixture's
    `n_bootstrap = 2` gave meaningless envelopes. A fidelity replicate
    costs one `r × T` simulation; a LIM bootstrap replicate refits the
    LIM.
  - **Memory.**
    - `train.dmd.measured_amplitudes` (new, used by `fit_mode_ou`) computes
      `Φ⁺x` with two real products and no complex upcast of the state.
    - `linear_inverse._leading_pcs` uses `vdot` instead of `sum(x**2)`.
      That removes a full-size temporary, 3.9 GB per production core,
      found when the real-data run hit its memory cap.
    - `train.copula.SATURATION_Z` (new) is the copula's saturation point,
      about 4.753.
  - **Real data.** On task 07's production `wsh` core, under an 8 GiB
    address-space cap (5.3 GiB peak), all three checks fail:
    - variance: 3.84 times the training energy;
    - memory: every lag from 12 h to 16 days;
    - time of day: share 0.148 against a 0.002 null.

    The numbers are in `model_design.md` §5.6.3. The cap was lowered from
    the planned 10 GiB because only 10.1 GiB was free.
  - **Tests: 36 fail, as expected.** Every test that fits the tier-1
    fixture now raises, because the current sampler fails the gate on any
    data tried: white noise, and a latent AR(1) process. The other 195 pass.
    `train/fidelity.py` has no unit tests yet. This is committed
    deliberately in this state so the work is kept. Landing it is
    `docs/open_questions.md` §5.9, priorities 1 and 5.

- **2026-11-22 — v3 preprocessing stages: partition sentinel fill, log
  transform and `keep_speed`** — `src/train/masking.py`,
  `src/train/directional.py`, `tests/train/test_masking.py`,
  `tests/train/test_directional.py`. Steps 1–2 of the v3 implementation
  (`docs/model_design.md` §5.3, §5.3.1). Nothing calls them yet; the
  `WeatherModel` rewrite wires them in.
  - **`PartitionSentinelFiller`.** Fits the land mask from `sigwh` alone
    and each period's zero-height limit (`period_sentinel_from_archive`).
    `transform` fills ocean NaN of an absent partition with height 0, that
    period limit and a 0° direction placeholder; land stays NaN.
    `inverse_transform` decodes heights at the 1 cm floor back to 0.
  - **`LogTransform`.** Heights → `logE_* = log(max(H, floor)²)`, periods
    → `T_* = log(period)`, and the exact inverse.
  - **`keep_speed`** on `ComingFromToUV` and `GoingToToUV`. A listed speed
    (the class-C partition heights) stays alongside its `(u, v)` and the
    inverse takes the magnitude from it instead of `hypot(u, v)`.
  - 13 new tests: the period sentinel recovers a known 3 s limit, land
    stays NaN, the floor decodes back to 0 through the log round trip,
    `clone()` round-trips, and kept speeds survive both directions.

- **2026-11-21 — `structlog` runtime dependency (`>=26.1.0,<27.0.0`)** —
  `pyproject.toml`, `poetry.lock`. Added ahead of use: nothing imports it
  yet. It backs the planned `src/core/logger.py` (AGENTS.md *Logging*),
  which will be created together with migrating the three training-path
  `print(..., file=sys.stderr)` calls. Matches the structlog logger used
  in the radiologists preprocessing pipeline.

- Repo scaffolding: `pyproject.toml` (Poetry, black, isort, mypy config),
  `.gitignore`, `README.md`.
