# Task 07 — Tucker/DMD rank selection

**Depends on:** 00 (soft — ranks are chosen on whichever representation feeds Tucker; a placement change plausibly shifts the elbow), 17 (soft — discovered while extending this task: `wsh`'s original rank sweep was pathological due to a masking bug, now fixed by `train.masking`) · **Blocks:** —
**Open question:** `docs/open_questions.md` §5.3 — OPEN
**Primary source docs:** `docs/tucker_decomposition.md` §6, `docs/dynamic_mode_decomposition.md` §6

## 1. What question this will try to answer

How should Tucker ranks $(r_{lat}, r_{lon}, r_{time})$ and DMD rank
$r_{dmd}$ actually be chosen? Both source documents describe only a
diagnostic (elbow of the reconstruction-error curve for Tucker;
eigenvalue-spectrum/growth-rate plausibility for DMD), not an automatic
stopping rule. This is a per-variable (and per-group, per Task 04) judgment
call — no document proposes a data-driven automatic rank selector, and
this task doesn't build one either; it produces the empirical curves
someone making that judgment call needs.

## 2. Steps to answer it

1. Load one variable (`wsh`) over a longer window (several months) than
   the other tasks use, so the rank sweep has enough data to be meaningful.
   **Update (task 17, 2026-08-20):** the original `wsh` attempt at this
   window found only 7-11% of cells passing even a 90%-valid tolerance,
   because `wsh`'s NaN pattern conflates land with genuine partition
   absence — not a data problem, a masking bug. The notebook worked
   around it by substituting `sigwh` (see its own setup markdown); that
   substitution's results are kept below as a valid baseline, but `wsh`
   itself is now loadable properly via `train.masking.land_mask_from_sigwh`
   + `apply_partition_sentinel_policy` (100% valid coverage, validated in
   `notebooks/17_partition_nan_sentinel_policy.ipynb`) — rerun the full
   rank sweep on real `wsh` (and `wsp`) using that corrected mask before
   treating the `sigwh` proxy as this task's final word.
2. **Tucker rank sweep:** fit masked `partial_tucker` at several
   `(r_lat, r_lon)` candidates (holding time at full resolution, per
   `tucker_decomposition.md` §4.3), and record the relative Frobenius
   reconstruction error at each. Plot error vs. rank and look for the
   elbow.
3. **DMD rank sweep:** fix a Tucker core, then fit `train.dmd.dmd_fit` at
   several `r_dmd` values and record `max|λ|` and the count of modes with
   `|λ|>1` at each. Plausibility here means: does the eigenvalue spectrum
   stay physically reasonable (bounded growth/decay rates) as rank
   increases, not just "does reconstruction error go down."
4. **Decision rule:** choose the smallest rank past which reconstruction
   error stops improving materially (the elbow) for Tucker; choose the
   largest `r_dmd` before spurious growing modes start appearing for DMD.
   Record both curves and the chosen defaults — this task's output is the
   curves and a recommended default, not a single "correct" number, since
   the tradeoff is inherently per-variable.

## 3. Notebook(s) needed

- `notebooks/07_tucker_dmd_rank_selection.ipynb` — loads `wsh` over a
  ~6-month era-3 window, runs the Tucker rank sweep and the DMD rank sweep.

## 4. Conclusion

Run against the real archive (`sigwh`, `wsh`, `wsp`; 2025-11-01 to
2026-05-01, 280×230 crop, `T=1448`). Full rank-candidate set
`(5,10),(10,20),(20,40),(40,80),(80,160)` swept for all three variables
(`wsh`/`wsp` via task 17's corrected mask + sentinel policy —
`apply_partition_sentinel_policy`, `height_log_energy`,
`period_sentinel_from_archive`'s `T_SENTINEL_S=3.8369621352089984`).

**Tucker: no elbow within the feasible range, for any of the three
variables.** Error decreases monotonically and materially at every step,
steepest at the top of the range, not plateauing:

| rank | product | `sigwh` | `wsh` | `wsp` |
|---|---|---|---|---|
| (5,10)   | 50    | 9.64%  | 63.16% | 14.37% |
| (10,20)  | 200   | 4.64%  | 51.37% | 11.92% |
| (20,40)  | 800   | 2.28%  | 40.66% | 9.70%  |
| (40,80)  | 3200  | 1.07%  | 31.79% | 7.73%  |
| (80,160) | 12800 | 0.43%  | 23.81% | 5.84%  |

`wsp` (period, log space) reconstructs substantially better than `wsh`
(height/log-energy) at every rank, consistent with task 17's finding that
`wsh`'s `log E` is ~36.5% sentinel-heavy and bimodal while `wsp`'s field is
smoother.

**Second constraint found, not anticipated by this task's original doc:
peak memory, not the elbow, is what actually bounds the candidate set.**
Measured empirically per rank via an isolated, `RLIMIT_AS`-capped
subprocess (so an over-budget attempt fails as a clean `MemoryError`
instead of the OS OOM-killer taking down an unrelated process, given this
machine's 31GB total / ~18GB available RAM). Peak RSS for a single masked
`partial_tucker` fit on this crop is **~12.25–12.27GB, essentially
constant** from `(40,80)` through `(80,160)`, for both `wsh` and `wsp` —
dominated by fixed per-ALS-iteration unfolding/reconstruction cost of the
full `(280,230,1448)` tensor, not by target-rank core size (the `(80,160)`
core itself is only ~74MB). Running the full 5-rank `wsp` sweep
sequentially in one process also stayed flat at 12.27GB peak — memory does
not accumulate rank-over-rank within a kernel, provided each candidate's
`core`/`factors` are discarded once its error is recorded (not retained
for every candidate simultaneously). `(80,160)` is therefore the largest
candidate this task tested not because a real elbow was found there, but
because it is close to the largest rank this machine can fit at all.

**DMD:** unchanged from the pre-task-17 sweep (fixed Tucker core at
`(20,40)`) — real knee at `cond(Φ)`'s jump between `r_dmd=10` and
`r_dmd=20` (§ above), no growing modes observed up to `r_dmd=40`.

**Rank recommendation:** `(r_lat=80, r_lon=160)` for both `wsh` and `wsp`
— the largest empirically-validated candidate, not a provably optimal one.
This is a hardware-bounded recommendation, not an evidence-that-error-
plateaus one: anyone training on a machine with more available memory
should extend the sweep further (a smaller crop, an out-of-core/chunked
Tucker implementation, or more RAM are the three ways past the ~12.3GB
per-fit ceiling found here) before trusting `(80,160)` as a long-term
default. `r_dmd` recommendation unchanged: smallest `r_dmd` past which
`cond(Φ)` stops improving materially, i.e. `r_dmd≈20` per the existing
sweep, not re-run under the corrected mask (DMD operates on the Tucker
core, not the raw NaN-affected tensor, so task 17's fix doesn't change
that sweep's inputs).

This closes "no rank defined" for `wsh`/`wsp` and changes
`open_questions.md` §5.3's status to reflect both findings (elbow
non-existence within a feasible range, and the new memory-ceiling
finding) — see that section's own update. No follow-on task spawned; the
memory ceiling is recorded here as a hardware constraint for future
re-evaluation, not a bug to fix.

## 5. Era-2 cloud rank sweep + Gaussianization placement decision (2026-09-13)

Revisited as part of the canonical-`(time,lat,lon)` refactor and task 18's
cloud migration, to test rank selection against data actually run through
the real preprocessing chain — every prior sweep above (and every
full-grid sweep in `tasks/18-cloud-migration-of-experiments_plan.md`) had
been measuring Tucker's rank requirement on raw log-energy data, never on
`ClimatologyRemover` + `PerCellStandardizer` (+ `GaussianCopula`, depending
on placement) output as production actually feeds Tucker.

**Data:** era 2 (2023-10-02 → 2025-10-27, wave grid 621×1440), chosen over
era 3 for enough span to fit a real annual harmonic. Transformers fit on a
strided (12-hourly) full 2-year window; Tucker run on a full-resolution
9-month sub-window (2024-01-01 → 2024-10-01, `T=2192`) spanning winter
through fall — i.e. multiple storm/wave regimes, unlike the single-season
crop in §4 above. `modes=[1,2]` (canonical order, time full-rank),
`svd="randomized_svd"`, `svd_mask_repeats=0`, checkpointed ALS resume.

**Gaussianization placement — decided, see `open_questions.md` §4.1 for
the full writeup.** Ran the same rank on both candidate placements:

| Placement | rank | err_pct | s/iter |
|---|---|---|---|
| (i) Gaussianize then Tucker | (31,72) 5% | 50.81% | 65.0 |
| (ii) Tucker on raw standardized anomalies, Gaussianize the core | (31,72) 5% | 53.04% | 64.6 |

Essentially indistinguishable — contradicts the linear-rank-inflation
hypothesis that motivated the comparison (per-cell Gaussianization
provably preserves rank/copula dependence but not linear/Pearson structure;
see `gaussian_copula_and_marginals.md` §2.4). **Placement (ii) adopted as
canonical anyway**, on architectural grounds: Tucker stays in
physically-grounded log-energy anomaly space, and Gaussianization moves to
sit immediately before DMD — the stage that actually needs Gaussian
innovations — rather than broadly upstream of a stage (Tucker) it doesn't
help. Placement (ii) is also what surfaced a real latent bug: a handful of
near-zero-variance (intermittently ice-covered) cells had no floor in
`PerCellStandardizer`, producing ~1e15-magnitude values that placement
(i)'s `GaussianCopula` clipping had been silently absorbing. Fixed in
`src/train/standardize.py` (floors the division only, never the stored
`sigma_cell_`, since `weather.model._log_trend_significance` needs the true
value — see that file's docstring).

**Rank sweep under placement (ii), same 9-month window:**

| rank | % of grid | err_pct | s/iter | total wall time |
|---|---|---|---|---|
| (31,72)   | 5%  | 53.04% | 64.6  | 3.59h |
| (124,288) | 20% | 32.90% (converged) | 151.4 | 8.41h |
| (186,432) | 30% | 27.54% at iter 168/200 (still improving) | ~220 | manually stopped ~11h |

**Correction, 2026-09-26**: the original "zero checkpoints" claim above was
wrong — the checkpoint CSVs were never actually retrieved from the VM
before writing this doc. Re-reading them directly off the VM disk (they
survived the intervening deallocate/start cycles) shows both the 30% run
above **and** the separate (248,576)/40% attempt under placement (i)
(reached iter 132/200 at ~311 s/iter, error 27.47%→26.10%, also
monotonically improving) were making real, steady, monotonically-improving
progress the whole time — neither had actually hung. Both were stopped
manually, apparently without checking the checkpoint file first. There was
no true "zero-progress hang" to explain; the two runs were simply slower
per iteration (~1.5-2x) than the 20% baseline and got killed before
reaching all 200 requested iterations. This doesn't change the diagnosis
below (the memory/SVD-method bugs are real, confirmed by reading tensorly's
source directly, independent of this correction) but it does change the
empirical framing: this was a cost/patience problem, not a hang/instability
problem.

**Rank: not yet decided.** 20% (124,288) is the current working baseline
— a real, trustworthy data point, not a plateau. Deciding a production
rank is deferred, blocked on either more compute budget (the Azure
subscription funding these runs had ~$59.62 in credit remaining as of
2026-09-13, expiring 2026-09-29) or a cheaper/faster Tucker fitting
approach that can actually reach 30-40% for comparison. No further ranks
were attempted this round for cost reasons.

**Cost-cliff root cause diagnosed and fixed, 2026-09-26 — not yet
VM-validated.** Read tensorly 0.9.0's actual installed source (not
assumed) against this repo's exact masked-ALS usage and found two real
issues inside `partial_tucker`'s loop, neither requiring a faster
language (Numba/C++ would not help — the hot path already runs compiled
BLAS/LAPACK code; the actual bottleneck is memory and an algorithm bug,
not raw compute speed):
1. **Memory**: the per-iteration mask re-imputation
   (`tensor = tensor*mask + recon*(1-mask)`) materializes ~4-5 full-size
   temporary arrays every iteration, rank-independent (~31 GiB at era-2
   scale for this line alone) — plausibly why even the working 20% run
   was already close to the ceiling.
2. **Correctness/speed bug**: the per-iteration factor-update SVD call
   never passes `method=svd`, so `svd="randomized_svd"` is silently
   ignored after initialization — every iteration actually runs a full
   economy LAPACK SVD instead, rank-dependent in cost and far more
   expensive than the randomized algorithm that was supposedly requested.

**Fix**: `src/train/masked_tucker.py` (new) — `partial_tucker_lowmem`, a
small vendored patch of tensorly's loop (reusing tensorly's own
`multi_mode_dot`/`unfold`/`svd_interface`) fixing exactly these two
lines: `np.copyto(tensor, recon, where=mask==0)` in place of the naive
blend (this repo's mask is always strictly `{0,1}`, so this is exact, not
approximate), and propagating `method=svd` into the per-iteration SVD
call. `notebooks/07_gaussianized_rank_sweep.py` and
`notebooks/07_placement2_tucker_on_anomalies.py` now import this instead
of tensorly's `partial_tucker` directly (same call signature, drop-in).
Verified locally (`tests/train/test_masked_tucker.py`, 2 tests): the
blend rewrite is numerically exact against tensorly's own function when
holding the SVD method fixed (isolating just that change), and the
`method=svd` fix still recovers a planted low-rank structure through a
mask (<1% error) once genuinely using `svd="randomized_svd"` every
iteration. Full suite (178 tests) + `mypy --strict` both clean.

**VM-validated, 2026-09-26.** Ran the fixed `partial_tucker_lowmem` at 30%
rank (186,432) on the real VM, wall-clock-boxed to 3h via a remote `timeout`
(not just requested politely — enforced so a repeat hang couldn't burn
budget unattended). Deployed by rsyncing the local working tree directly
(no commit/push — both were uncommitted/unpushed at the time); the VM's own
divergent uncommitted edits from an earlier session were `git stash`ed
first, not overwritten, so they're recoverable (`git stash list` on the VM).
Two infra issues found and fixed along the way, unrelated to the Tucker fix
itself: the VM's NSG only allowed SSH from a stale IP (updated to add the
current one, additively); `results/` had grown to 19GB on the 29GB OS disk
(95% full) — moved to the attached 256GB data disk and symlinked back
before running anything.

Result: **80/300 requested iterations completed before the 3h cap fired**
(`EXIT_CODE=124`). Steady-state pace **~128 s/iter** (iter 5→80, computed
off the per-iteration checkpoint deltas), versus the old code's ~220 s/iter
at the same rank — a real, reproducible **~1.72x speedup**, but far more
modest than hoped. Error decreased monotonically and smoothly the whole
time (29.32%→27.79%), with clearly diminishing returns by iter 80 (~16x
slower improvement-per-iteration than the first checkpoint) — normal ALS
convergence behavior, no instability. Peak memory stayed ~30GB (well under
the 58GB cap), no swap — the memory fix is working as intended, no sign of
the thrashing that plausibly explained the original cliff. Extrapolating
the observed steady pace, full convergence at this rank would need roughly
7-11h more, not something to attempt unattended in one more time-boxed
session without a much longer window.

**Why the speedup is modest, not dramatic**: the diagnosis (§ above) is
still correct, but the SVD-method fix only cheapens the SVD portion of each
iteration (theoretically ~3.3x faster for both modes' unfold matrices at
this rank, from the `min(matrix_dim)/target_rank` ratio) — the *other*
rank-independent per-iteration costs (tensor contractions via
`multi_mode_dot` over the full 7.84GB tensor) form a floor that this fix
doesn't touch, which is consistent with an overall ~1.7-2x rather than
~3x+ speedup.

**This partial result is still decision-relevant**: 30% rank at iteration
80 — *not even converged yet* — already beats 20%'s fully-converged result
(27.79% vs 32.90%, a 15.5% relative error reduction). Whatever the true
converged 30% error turns out to be, it's very likely meaningfully better
than 20%. The 40% attempt was not made tonight, per the plan's own ordering
(only proceed to 40% after a 30% success within its cap) — the VM was
deallocated immediately once the 3h cap fired. **Tucker rank is still not
decided** — 20% remains the only *fully converged* data point, but 30% is
now a real, promising, partially-converged candidate rather than an
unexplained failure.

**30% re-attempted with an 8h cap, 2026-09-26 (same evening).** Same rank,
same VM, same deterministic trajectory (verified: error at iterations
5/10/15/20 matches the 3h attempt to 4 decimal places) — just a longer
wall-clock budget, since 3h clearly wasn't enough. Also added an IP to the
VM's SSH allow-list (additive, same pattern as before — the environment's
outbound IP had changed) and pre-staged the 40% run script in advance so it
can launch immediately on a 30% success, per the standing instruction:
30% success → write up, launch 40% (10h cap) immediately on the same VM,
no re-sync/restart needed; 30% hitting its cap → write up, deallocate, do
not attempt 40%. As of iteration 50/300 (~1h51m elapsed), error is at
27.995% and still decelerating smoothly (checkpoint-to-checkpoint delta
dropped from 0.42pp at iteration 10 to 0.05pp at iteration 50 — clearly
flattening, not plateaued yet). See
`notebooks/07_tucker_checkpoint_timelapse.py` for the error/delta-vs-
iteration plot (reusable for any future checkpoint CSV).

**Spatially-resolved reconstruction-error diagnostics, 2026-09-26.** While
the 30% re-run was in progress, built a complementary way to compare ranks
beyond the single aggregate `err_pct` number: *where* does reconstruction
error concentrate, not just how much of it there is. Full methodology
written up in `docs/tucker_decomposition.md` §5.1 (general, reusable
across variables/ranks); tool is
`notebooks/07_tucker_reconstruction_visual_check.py`, applied here to
`wsh` at 20% rank (124,288) against two cached true-field time-slices
(`results/task07/gaussianize/demo_slices.npz`, timesteps 400 and 1600 of
the same 2024-01-01→2024-10-01 window, extracted from the VM's cached
tensor via `mmap_mode="r"` so the ~7.8GB array was never loaded whole).

Four complementary measurements, all agreeing for this baseline:

| Measurement | Value | Reads as |
|---|---|---|
| Coastal (within 3 cells of land) vs. open-ocean RMSE | 1.33x | mild coastline effect |
| corr(\|true-field local gradient\|, \|error\|), open ocean only | 0.748 | strong |
| RMSE(sharper 50% of open-ocean cells) / RMSE(flatter 50%) | 2.52x | error concentrates in the transition half |
| (max − median) / IQR, skewness, excess kurtosis of open-ocean \|error\| | 24.5 / 3.01 / 10.79 | strongly right-skewed, heavy-tailed |

Visuals: `notebooks/tucker_reconstruction_check_20pct.png` (true /
reconstructed / error maps, robust percentile-based color scale — the raw
max/min range made the plot nearly useless, a handful of extreme
storm-event pixels stretched it so far the bulk of the field washed out to
near-white). The error map shows a fine tracery of high error tracing both
the coastline *and* the edges of open-ocean anomaly patches themselves
(storm/eddy boundaries) — the classic signature of a low-rank/smooth-basis
reconstruction: broad slowly-varying patterns are captured well by a
handful of global modes, but sharp spatial transitions need many more
basis functions to resolve, so residual error piles up almost entirely at
those transitions wherever they happen to sit.

**Why this matters for the rank decision**: the aggregate `err_pct` number
alone can't distinguish "higher rank is genuinely resolving sharp
transitions" from "higher rank is just polishing already-easy flat
regions." Once 30% (and 40%, if reached) converge, rerunning this exact
diagnostic against the same cached true-field slices gives four
directly-comparable per-rank numbers. If they all improve together
(half-ratio drops, skewness/kurtosis fall) that's strong evidence more
rank is fixing the real structural limitation; if the aggregate error
drops but these four don't move, that would mean the improvement is
concentrated in the bulk, not the part of the error we actually care about
— a materially different, and less useful, kind of improvement than the
single `err_pct` number alone would suggest.

**30% (186,432), 8h-cap result, 2026-09-26.** Killed by the wall-clock cap
as expected (`EXIT_CODE=124`) at **iteration 220/300, err_pct=27.4651%**
(elapsed 28,086.9s, right at the 8h boundary). Steady-state pace held at
~126 s/iter throughout (consistent with the 3h attempt). A power-law fit
of the per-checkpoint error delta (`delta ~ iter^-p`) was refit as more
data accumulated and the exponent stabilized: p=1.339 (through iter 60) →
1.380 (through iter 165) → **1.387 (final, through iter 220)** — p > 1
throughout, meaning the infinite-horizon tail sum of future improvement
converges (this rank is not runaway-divergent), but early-data fits
underestimated where it would land: the iter≤60 fit predicted 25.91% at
iteration 165, the actual was 27.55% — 1.64pp worse than that early
extrapolation, a real, quantified lesson that power-law extrapolation from
a short early window is unreliable and needs to be re-fit as data
accumulates, not trusted from an early snapshot. Final extrapolated
iter→∞ floor: **~23.93%** (likely still an underestimate given the same
pattern, i.e. treat as a rough lower bound, not a precise prediction).
Plots: `notebooks/tucker_checkpoint_timelapse_30pct_8hcap_final.png`.

Even short of that hypothetical floor, 27.4651% is already a **16.5%
relative error reduction** over 20%'s fully-converged 32.8975%.

**Gap found and fixed for future runs**: `07_placement2_tucker_on_anomalies.py`
only saved the `(core, factor_lat, factor_lon)` `.npz` bundle *after* the
checkpoint loop's normal completion (`while done_iters < total_iters`) —
a process killed early by the wall-clock `timeout` never reaches that
line, so **this 30% run's Tucker factors were never saved**, only the
checkpoint CSV (error/iteration/elapsed time). No spatial reconstruction
check (§5.1-style diagnostics) is possible for 30% this round as a result
— confirmed and accepted rather than restarting (would cost another
~7-8h to redo the ~6h already spent, for a much smaller marginal benefit
than moving on to 40%). Fixed going forward: the `.npz` save now happens
inside the loop, every checkpoint, overwriting the same path each time —
so any future run (starting with 40% below) always has a recoverable
snapshot of the latest factors regardless of how it ends.

VM deallocated immediately once the cap fired. Per the standing
instruction (30% success → launch 40% immediately; 30% hitting its cap
without converging → write up, deallocate, stop), this run hit its cap
without converging, so **40% was not launched** — matches the plan's own
ordering from earlier (40% only follows a 30% success within its own
cap). 40% remains a deliberate next decision, not an automatic one.

**30% spatial reconstruction check, 2026-09-27.** Since the 8h run's
tensors were lost (§ above), re-ran 30% specifically to get a recoverable
snapshot for the §5.1 diagnostics — deliberately short (`--total_iters 80`,
matching the very first 3h attempt's iteration count for continuity),
completed naturally this time (`EXIT_CODE=0`, all 80 iterations,
27.7944% — identical to both earlier partial runs at that iteration,
confirming determinism yet again) rather than being timeout-killed, now
that the per-checkpoint `.npz` save is in place. Same four measurements
as the 20% baseline, same cached true-field slices:

| Measurement | 20% (converged) | 30% (iter 80, unconverged) |
|---|---|---|
| Overall RMSE (2 slices) | 0.3357 | 0.2844 |
| Coastal/open RMSE ratio | 1.33x | 1.29x |
| corr(\|gradient\|, \|error\|), open ocean | 0.748 | 0.765 |
| RMSE(sharper 50%)/RMSE(flatter 50%) | 2.52x | 2.69x |
| (max−median)/IQR | 24.48 | 27.67 |
| Skewness / excess kurtosis | 3.01 / 10.79 | 3.35 / 13.60 |

Visual: `notebooks/tucker_reconstruction_check_30pct.png`.

**This is exactly the "disagreement" case flagged as the interesting one
when this methodology was designed**: overall RMSE dropped (as expected —
more rank can only help the aggregate fit), but every concentration
measure got *worse*, not better — the coastal ratio is the only one that
barely improved. Checking why: Q1 (flattest quartile) RMSE improved
19.6% (0.1679→0.1350) going 20%→30%, while Q4 (sharpest quartile) RMSE
improved only 13.7% (0.5777→0.4988) — the extra rank disproportionately
sharpened already-easy, slowly-varying regions rather than closing the
gap at genuine spatial transitions. In other words: **more rank is making
the reconstruction better everywhere, but proportionally more so exactly
where it was already good, not where it was struggling.** If this pattern
holds at 40% too, it would be real evidence that Tucker rank alone cannot
fix the sharp-transition weakness — it's a structural limitation of a
global smooth-basis decomposition (§ above), not something more rank
straightforwardly buys back, at least not as efficiently as it improves
the easy regions.

**40% spatial reconstruction check, 2026-09-27.** Launched immediately
after 30%'s completed (per instruction, to end up with tensors for all
three ranks). This one hit its own 3h backstop before reaching all 80
iterations (`EXIT_CODE=124`) — 40% costs meaningfully more per iteration
than 30% (~201 s/iter vs ~126 s/iter, roughly 1.6x), confirming rank still
matters for cost even with the fix. Thanks to the per-checkpoint save,
the last completed checkpoint's tensors (**iteration 50, err_pct=23.8276%**)
were still fully recoverable — downloaded before deallocating.

Full three-rank comparison, same four measurements, same cached
true-field slices throughout:

| Measurement | 20% (converged) | 30% (iter 80) | 40% (iter 50) |
|---|---|---|---|
| Overall RMSE (2 slices) | 0.3357 | 0.2844 | 0.2438 |
| Coastal/open RMSE ratio | 1.33x | 1.29x | **1.23x** |
| corr(\|gradient\|, \|error\|), open ocean | 0.748 | 0.765 | 0.741 |
| RMSE(sharper 50%)/RMSE(flatter 50%) | 2.52x | 2.69x | 2.58x |
| (max−median)/IQR | 24.48 | 27.67 | **28.30** |
| Skewness | 3.007 | 3.349 | **3.510** |
| Excess kurtosis | 10.786 | 13.602 | **15.323** |

Visual: `notebooks/tucker_reconstruction_check_40pct.png`.

**Pattern, now with three points instead of two**: overall RMSE and the
coastal ratio both improve *monotonically* with rank (0.336→0.284→0.244;
1.33x→1.29x→1.23x — the coastline problem specifically does get better
with more rank, consistently). The gradient-correlation and half-ratio are
not monotonic across these three ranks (both peak at 30%, dip slightly at
40%) — noisy in this range, no clear trend either way. But the
error-distribution-shape trio (percentile-ratio, skewness, excess
kurtosis) **worsens monotonically at every step**: 24.5→27.7→28.3,
3.01→3.35→3.51, 10.8→13.6→15.3.

Re-examining the Q1-vs-Q4 mechanism that explained this 20%→30%: it does
**not** repeat the same way 30%→40%. Q1 (flattest) improved 10.2%
(0.1350→0.1212) while Q4 (sharpest) improved *more*, 14.6%
(0.4988→0.4261) — the opposite ordering from 20%→30%. Yet the
whole-distribution skew/kurtosis still worsened. Reconciling this: `P100`
(the single most extreme cell) is dropping only slowly (3.47→3.06→2.83)
while `IQR` (the bulk's own typical spread) is dropping much faster
(0.139→0.108→0.098) — the bulk of the domain keeps getting tighter/more
accurate noticeably faster than the single worst cell does, so the ratio
of "how far out the worst case sits" relative to "what's normal" keeps
growing even while every absolute number improves. **This is a different,
more precise mechanism than the quartile-group story suggested**: it is
not really about the top 25% of cells falling behind the bottom 25% —
it's specifically the most extreme handful of cells (driving the max, and
disproportionately driving skewness/kurtosis) that resist improvement
fastest, while everything else, including most of the "sharp" quartile,
keeps improving at a healthy clip.

**Implication for the rank decision**: more Tucker rank keeps helping,
including at genuine spatial transitions in most cases — but a small
number of the very worst cells (likely the sharpest coastlines / most
extreme storm-edge pixels) are structurally hard to close via rank alone,
and that stubborn residual becomes proportionally more visible (not less)
as everything around it improves. This is consistent with, and now more
precisely characterized than, the smooth-global-basis limitation
described in `docs/tucker_decomposition.md` §5.1 — it isn't that rank
"doesn't help" at transitions in general, it's that a small extreme tail
resists rank the way a Gibbs-phenomenon overshoot resists adding more
Fourier terms: better, but slowly, and never fully gone at any finite
rank.

VM deallocated once tensors were safely downloaded.

**Animated true-vs-reconstruction timelapse, 2026-09-27.** Built
`notebooks/07_tucker_rank_timelapse_gif.py` to see the same comparison
animated across the full 9-month window rather than at two fixed
timesteps: 548 frames (00 and 12 UTC every day, `i % 8 in (0, 4)` over the
window's 3-hourly indices — 274 days x 8 samples/day = 2192 = T exactly).
True-field frames extracted from the VM's cached tensor via
`mmap_mode="r"` (never loading the full ~7.8GB array); reconstructions for
all three ranks computed on the fly per frame from the already-local
`.npz` bundles (no VM compute needed for this part). Two rows: true field
+ 20/30/40% reconstructions (shared color scale, so sharpening across rank
is directly visible); a date/per-frame-RMSE info panel + 20/30/40% error
maps (shared color scale across ranks). Output:
`notebooks/tucker_rank_timelapse.gif` (75.2 MB at dpi=45 — an initial
dpi=100 render was 345.8MB, impractically large for a 548-frame GIF with
this much per-frame detail; dpi=70 only got to 175.9MB, dpi=45 was the
point of clearly diminishing file-size returns while staying legible).

What it shows, watched end to end: the coastline trace in every error
panel is fixed frame to frame (land doesn't move), while the open-ocean
error tracery visibly follows storm/eddy boundaries as they form, move,
and dissipate across the season — direct visual confirmation that the
"sharp transition" error characterized statically in §5.1 is a real,
moving, weather-following phenomenon, not a static artifact of the two
timesteps originally checked.

Checked this precisely rather than just eyeballing it: computed RMSE for
all 548 frames, all three ranks (not just the two originally-cached
timesteps). Per-frame RMSE bounces around by a comparable amount to its
own linear drift across the whole 9-month window — i.e. there is no
meaningful systematic trend, just weather-driven noise around a stable
mean:

| Rank | mean RMSE | std | min | max | total linear drift over 548 frames |
|---|---|---|---|---|---|
| 20% | 0.3323 | 0.0083 | 0.3037 | 0.3589 | +0.0082 |
| 30% | 0.2815 | 0.0071 | 0.2571 | 0.3048 | +0.0071 |
| 40% | 0.2418 | 0.0060 | 0.2212 | 0.2617 | +0.0058 |

The 20%>30%>40% RMSE ordering holds at every single frame (min/max ranges
don't overlap across ranks at all), and each rank's own total drift is
about one standard deviation of its own frame-to-frame noise — consistent
with climatology removal having already handled the seasonal cycle (§6
below), so this residual is genuinely weather-driven variability, not a
leftover seasonal trend Tucker happens to be absorbing.

Conceptual note on why this matters beyond visualization: the spatial
residual shown here (true − Tucker reconstruction, at any fixed rank) is
illustrative of the kind of high-spatial-frequency content a low-rank
Tucker fit cannot represent at any single rank — watching it move with
the weather makes concrete why a purely spatial (Tucker) decomposition
needs a *temporal* dynamical model (DMD) plus a stochastic residual model
(OU) downstream: the un-modeled part isn't static clutter, it's real,
evolving physical variability that has to be handled by something. This is
illustrative, not a literal equivalence — DMD and the OU residual model
operate on the Gaussianized Tucker *core*'s time series after this stage,
not on this raw spatial grid directly.

## 6. DMD rank (`r_dmd`) sweep under placement (ii), and the climatology-removal question (2026-09-18)

Closes the DMD half of this task against a real Gaussianized core, using
the one Tucker rank trusted from §5 above (20%, `(r_lat=124, r_lon=288)`,
same 9-month era-2 window). This is the first DMD sweep run against
anything resembling the current pipeline — the only prior sweep (§4, task
07's original notebook) used an 800-dimensional, un-Gaussianized,
single-season crop core `(20,40)`, nothing like this one
(35,712-dimensional). Full design rationale in the plan this session
worked from (three iterations before the methodology was trustworthy —
kept below as a caution for whoever revisits this).

**Method** (`notebooks/07_dmd_rank_sweep.py`): 80/20 train/holdout split
of the core by time index; `GaussianCopula(n_bins=500)` and DMD both fit
on the TRAIN portion only (never on the whole window before splitting —
that would leak the holdout period's own values into the copula's
per-cell LUT, a second leakage channel distinct from the Tucker factors'
own, milder one, since `modes=[1,2]` keeps time full-rank so only the
*spatial basis*, not holdout *dynamics*, was informed by the whole
window). Swept `r_dmd \in \{5,10,20,40,50,80,100,150,200\}`
(`variant="exact"`) — brackets both the only empirical DMD knee this repo
had found before (§4's `r_dmd=10` vs. `20` jump, on a 45x-smaller core)
and `docs/model_design.md:1010`'s literal "start `r_dmd=50`, extend to
100-200" guidance.

**Getting a trustworthy held-out metric took two failed attempts** — kept
here because both failure modes are easy to reproduce by accident and
neither raised an exception (silent, plausible-looking wrong numbers,
exactly the failure class this codebase is generally most exposed to):

1. First attempt: evaluate the deterministic trajectory
   $b_0 \lambda^t$ at holdout timesteps, $t$ continuing the *global* index
   from the training window's own $t=0$ (i.e. $t \approx n_{\text{train}}$
   = 1754 at the start of holdout). Every fitted $|\lambda| < 1$, and
   $0.98^{1754} \approx 4\times10^{-16}$ — below float64 epsilon regardless
   of `r_dmd` or how close to the *start* of holdout the evaluation point
   is (a short lead time barely changes an already-huge exponent). Every
   rank read exactly 100.0000% error — not because any rank was worse than
   any other, but because the deterministic component had already decayed
   to numerical zero by the *end of training itself*.
2. Second attempt fixed the exponent size (small lead times) but kept
   anchoring at the *fit's own* $t=0$, missing that the problem was the
   anchor point, not the exponent — same flat 100.00% result.
3. **Correct fix**: the standard DMD one-step(s)-forward evaluation —
   project the *measured* state at the end of training onto the modes
   ($a_{\text{ref}} = \Phi^+ x_{\text{end of train}}$, the same operation
   `train.dmd.fit_mode_ou` already uses), then advance that *measured*
   amplitude forward by $\lambda^{\text{lead}}$ for small lead times
   (1, 6, 12, 24, 48 steps = 3h to 6 days). This produced real, varying
   numbers (98.5-99.7%) rather than a flat, meaningless one.

**Results.** No growing modes ($|\lambda|>1$) anywhere in
$\{5,...,200\}$. $\operatorname{cond}(\Phi)$ climbs smoothly from 2.2 to
227 — no sharp knee, unlike the 10x jump between `r_dmd=10` and `20` found
on the old crop core. The corrected one-step-ahead error stayed at
98.5-99.7% across the *entire* swept range, essentially flat despite
$\operatorname{cond}(\Phi)$ spanning two orders of magnitude —
**`docs/open_questions.md` §5.3's 2026-08-16 finding that
$\operatorname{cond}(\Phi)$ tracks held-out quality does not clearly hold
here**, at least not within this range; that finding was established
fitting DMD on a raw, non-Gaussianized core (testing placement iii), and
this is the first check of it against a Gaussianized-input (placement ii)
fit.

**Root cause of the flat ~99% error: rank capacity, not fit quality.** A
POD/SVD of the same Gaussianized core shows only 31.32% of its variance
captured by the leading 200 (of 1754 possible) modes — 50% needs ~411,
90% needs ~1287.

**Follow-up test: is this an artifact of removing climatology?**
(`notebooks/07_dmd_climatology_only_check.py`) Projected
`ClimatologyRemover`'s fitted signal *alone* (its 6-term-per-cell
harmonic+trend prediction; `wsh` is not a diurnal variable; zero anomaly,
zero noise; land cells zero-filled to match how the real Tucker fit
treated them, since the fitted coefficients are `NaN` at land by
construction) through the same Tucker factors, then checked its own POD
spectrum and DMD fit:

| | Climatology-only core | Real anomaly core |
|---|---|---|
| Modes for 90% variance | 3 | ~1287 (of 1754) |
| Modes for ~100% variance | 5-6 | (not reached by `r_dmd=200`) |

At `r_dmd=6`, DMD reconstructs the climatology-only signal to 0.07% error
and recovers the *exact* physical periods baked into it — 365.3 days
(annual), 182.6 days (semi-annual), plus a static/trend mode. At
`r_dmd=4` (below the true ~6-dimensional rank), the fit destabilizes: 2
growing modes appear with nonphysical 233-day/3467-day "periods" — a
clean, concrete confirmation that under-choosing `r_dmd` breaks DMD even
for a purely deterministic, low-rank signal, not just a theoretical
concern.

**Conclusion: the mechanism raised (climatology removal takes away the
one genuinely low-rank, DMD-friendly part of the signal) is real, but it
argues against reversing climatology removal, not for it.** That low-rank
structure is already captured exactly and cheaply by
`ClimatologyRemover`'s 6 coefficients per cell; asking DMD to re-derive it
from noisy data would spend modes re-learning something already known
exactly, for no benefit — climatology and the anomaly are separable,
near-orthogonal problems needing vastly different rank (6 vs. 1000+), so
putting them back together wouldn't shrink the anomaly's own requirement,
and could actively mislead rank selection (a low-rank climatology
component masking the hard part's true error at low `r_dmd`).
**Decision: climatology removal stays exactly where it is** (before
Tucker, before DMD) — full reasoning cross-posted to
`docs/open_questions.md` §5.3's 2026-09-18 update, which this section
mirrors. The anomaly's high effective rank is treated as an intrinsic
property of weather-scale variability once the predictable seasonal part
is removed, consistent with DMD only being expected to capture whatever
*is* low-rank/oscillatory, with mode-OU absorbing the rest as stochastic
noise.

**`r_dmd`: still not decided.** Unlike Tucker, this sweep was never
memory- or cost-bound — every fit costs about the same ~30s regardless of
`r_dmd` (confirmed empirically), runs entirely locally with no VM needed,
and produced no growing modes or a clear `cond(\Phi)` knee through 200.
Extending the candidate list (300, 400, ...) to look for an actual elbow
is cheap and a natural next step, not yet done. The §4 crop-based
`r_dmd\approx20` vs. `r_dmd=10` contradiction is superseded by this sweep
at a completely different scale (35,712-dimensional core vs. 800) and
should be treated as moot going forward, not resolved in either
direction. `fit_mode_ou` was explicitly out of scope for this sweep — it
is a downstream fit that happens after `r_dmd` is chosen, not part of
choosing it.

## 7. Physical predictability context (2026-09-19)

§6's "no elbow found through `r_dmd=200`" result prompted a natural
question: how much of the post-climatology anomaly is genuinely learnable
structure a higher-rank DMD could still capture, versus irreducibly
random, OU-appropriate noise? Researched what real physics-based
ocean/atmosphere forecasting models do — full reasoning cross-posted to
`docs/open_questions.md` §5.3's 2026-09-19 update, which this section
mirrors at a glance:

- Operational wave models (WAVEWATCH III/SWAN/WAM) solve the
  phase-averaged spectral wave action balance equation, **not**
  Navier-Stokes — true NS/Boussinesq/RANS ("phase-resolving") models are
  10-200x more expensive and restricted to local/regional, minutes-to-
  days use, never basin-scale multi-day forecasting. This project's
  Tucker/DMD approach is in the same statistical/spectral family as the
  operational one, not a simplification of a "more correct" NS method.
- Atmospheric predictability has a hard, chaos-imposed ceiling around two
  weeks; operational wave forecasts run to 5-7.5 days. Past that horizon,
  poor skill is provably not a model-quality problem. This reframes the
  ~1287-mode requirement (§6) as plausibly a mix of real deterministic
  structure (up to some horizon) plus an irreducibly chaotic remainder no
  amount of rank fixes — not evidence the sweep or pipeline is wrong.
- **Flagged, not-yet-pursued idea**: swell is far more predictable than
  local wind-sea (near-linear dispersive propagation vs. chaotic local
  wind coupling). Separating the two before Tucker/DMD, analogous to
  climatology removal, could concentrate more learnable structure into
  DMD and leave a cleaner residual for OU. Not actioned this round.
- The DMD+OU architecture itself is independently validated: operational
  NWP already uses the same deterministic-dynamics-plus-stochastic-
  residual pattern (SPPT, SKEB), and OU processes specifically model
  temporal autocorrelation of dominant EOF/mode amplitudes in stochastic
  climate models — the same technique as this project's mode-OU stage.

**Further reading (all sources gathered this session, grouped by topic):**

*Wave model governing equations / WW3 background:*
- [WAVEWATCH III Model Description](https://polar.ncep.noaa.gov/waves/wavewatch.shtml) · [FTP directory](https://polar.ncep.noaa.gov/waves/wavewatch/)
- [A Comprehensive Review of Phase-Averaged and Phase-Resolving Wave Models](https://arxiv.org/html/2511.21856v1)
- [CMCC, WW3 model page](https://www.cmcc.it/models/ww3) · [Uniparthenope, WW3 overview](https://meteo.uniparthenope.it/wavewatch-iii-ww3/)
- [NOAA OSTI, "Numerical Developments of the WAVEWATCH III model"](https://vlab.noaa.gov/web/osti-modeling/numerical-developments-of-the-wavewatch-iii-model)
- [Development of a Flexible Coupling Framework for Coastal Inundation Studies](https://arxiv.org/pdf/2003.12652)
- [Assimilation of distributed ocean wave sensors](https://arxiv.org/pdf/2003.04435)
- [A validated coupled 3D hydrodynamic and spectral wind-wave model for the western north Atlantic Ocean](https://arxiv.org/pdf/2505.18803)
- [Role of Surface Gravity Waves in Aquaplanet Ocean Climates](https://arxiv.org/pdf/2106.05032)

*Predictability limits / forecast skill horizon:*
- [ECMWF, "Chaos and Weather Prediction"](https://www.ecmwf.int/en/elibrary/79859-chaos-and-weather-prediction) · [lecture notes PDF](https://www.ecmwf.int/sites/default/files/elibrary/2002/16927-chaos-and-weather-prediction.pdf)
- [ScienceDaily, "Predictability limit: Scientists find bounds of weather forecasting"](https://www.sciencedaily.com/releases/2019/04/190415154722.htm)
- [Krishnamurthy, "Predictability of Weather and Climate," 2019](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019EA000586)
- [Weather Forecasting: What Sets the Forecast Skill Horizon?](https://www.sciencedirect.com/science/article/abs/pii/B9780128117149000024)
- [Testing the Limit of Atmospheric Predictability with a Machine Learning Weather Model](https://arxiv.org/html/2504.20238v1)
- [Week 1: Predictability of Weather and Climate — Chaos and Predictability (course notes)](https://kuiper2000.github.io/chaos_and_predictability/week1/week1.html)
- [Evaluating Ensemble Post-Processing for Wind Power Forecasts](https://arxiv.org/pdf/2009.14127)
- [(untitled arXiv predictability paper)](https://www.arxiv.org/pdf/1904.08766)

*Stochastic parameterization / Ornstein-Uhlenbeck in NWP:*
- [Structure-preserving stochastic parameterization with Ornstein-Uhlenbeck noise](https://arxiv.org/abs/2603.26693)
- [Stochastic parameterization of subgrid-scale processes in coupled ocean-atmosphere systems](https://arxiv.org/pdf/1605.00461)
- [Stochastic Climate Theory and Modelling](https://arxiv.org/pdf/1409.0423)
- [Stochastic Parameterization: Towards a new view of Weather and Climate Models](https://arxiv.org/pdf/1510.08682) · [OSTI copy](https://www.osti.gov/pages/servlets/purl/1376542)
- [Impact of stochastic physics on atmospheric blocking in EC-Earth3](https://wcd.copernicus.org/articles/5/1207/2024/)
- [Strommen et al. 2019, Impact of a Stochastic Parameterization Scheme on Climate Sensitivity in EC-Earth](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JD030732)

*Ocean wave forecast skill / swell vs. wind-sea:*
- [ECMWF, ocean wave forecasting fact sheet](https://www.ecmwf.int/en/about/media-centre/focus/2020/fact-sheet-ocean-wave-forecasting)
- [Skill of Long-Range Forecasts of Ocean Wave Spectra from the Navy ESPC v2 System](https://arxiv.org/pdf/2510.06484)
- [Statistical downscaling of seasonal wave forecasts](https://www.sciencedirect.com/science/article/abs/pii/S1463500318302336)
- [A Foundation Model for the Earth System](https://arxiv.org/pdf/2405.13063)
- [NOAA, Predicting significant wave height](https://repository.library.noaa.gov/view/noaa/55761/noaa_55761_DS1.pdf)
- [NOAA Marine Navigation, Forecasts and Predictions](https://marinenavigation.noaa.gov/forecasts.html)
- [Forecasting: theory and practice](https://arxiv.org/pdf/2012.03854)
- [NWS, Coastal Waters Forecasts With Wave Detail](https://www.weather.gov/marine/wavedetail)
- [AOML/NOAA, Investigating the seasonal predictability of significant wave height](https://www.aoml.noaa.gov/phod/docs/grl54250.pdf)
- [Machine Learning in weakly nonlinear systems: Significant wave heights](https://arxiv.org/pdf/2105.08583)

## 8. Surrogate-data and DMD+OU generative fidelity checks (designed 2026-09-19; run 2026-09-24)

§7's reframing (this is a stochastic generator, not a deterministic
forecaster — DMD should only be expected to extract coherent low-rank
linear dynamics, with mode-OU absorbing the rest) turns §6's "no elbow,
~1287/1754 modes needed" from an ambiguous result into a testable
question: is the unexplained ~69% of variance (a) real recoverable
structure a higher `r_dmd` could still capture, or (b) already
statistically indistinguishable from the correlated noise mode-OU is
designed to model? Two local experiments (no VM, no Azure cost — see
[[project_azure_credit_budget_constraint]]) are designed to answer this
directly, reusing the cached core at
`results/task07/gaussianize/anomalies_wsh_r124x288_checkpointed200_202401-202410_tucker.npz`
and `notebooks/07_dmd_rank_sweep.py`'s existing split/copula/anchoring
conventions. Both stay in Gaussianized-core space (never calling
`GaussianCopula.inverse_transform` or `tucker_inverse`), isolating the
DMD/OU question from the copula's own documented lossiness (clipping to
`z≈±4.75`, LUT extrapolation) — **explicit limitation**: a clean result
here doesn't by itself guarantee no missing structure in physical units
for extremes specifically, since `inverse_transform` is exactly where
extreme-value information is discarded; a physical-space check (e.g. via
`validate.py`'s `spectra_check`) would be needed to close that gap, and is
not part of this design.

**Experiment 1 — phase-randomized surrogate test**
(`notebooks/07_dmd_surrogate_rank_check.py`, not yet written). Build a
Prichard–Theiler multivariate FFT surrogate of the full Gaussianized core:
`rfft` along time, apply **one** random phase per frequency bin
identically across all 35,712 channels (not independently per channel),
force DC/Nyquist phase to 0, `irfft` back. Applying the *same* rotation to
every channel preserves the full complex cross-spectrum exactly at every
frequency, so all auto- and cross-covariance at every lag survives — this
produces the maximum-entropy stationary Gaussian process consistent with
the real data's own second-order statistics. Generate 20 independent
surrogate draws (one is statistically underpowered), re-split each 80/20,
and run the existing per-rank DMD loop (`n_growing`, `max_lambda`,
`phi_cond`, corrected lead-time held-out error) on each, across the same
`R_DMD_CANDIDATES`, plus a POD cumulative-variance curve (`scipy.linalg.
svd`, default `gesdd`) for real vs. surrogate.
**How to read it**: the POD curve is a sanity check on the surrogate code,
not evidence — by Parseval's theorem it's expected to nearly coincide with
the real curve regardless of whether real structure exists, since it's
determined by exactly the cross-spectral quantity the surrogate preserves
by construction. **The load-bearing test is the held-out multi-step DMD
error**, real vs. the 20-surrogate band: a linear predictor is
Bayes-optimal for a Gaussian/linear process, so the surrogate band traces
the covariance-implied error floor. Real error at/inside that floor at
every rank → case (b); real beating the floor at some rank → that gap
quantifies real recoverable structure (case a).

**Result (2026-09-24), decisive — case (b) confirmed.** Ran all 20
surrogates against the full rank sweep
(`results/task07/gaussianize/dmd_surrogate_rank_check_wsh_r124x288_202401-202410.csv`).
POD curves coincide almost exactly (e.g. `r_dmd=200`: real 31.32% vs.
surrogate mean 31.37% [31.30, 31.45]) — the expected sanity check, not
itself evidence. The load-bearing held-out error sits **inside the
surrogate band at every single rank tested**, for both the 1-step and
48-step lead:

| r_dmd | real err_pct_lead1 | surrogate mean [min, max] |
|---|---|---|
| 5   | 99.66% | 99.72% [99.39, 99.95] |
| 200 | 98.49% | 98.65% [98.08, 99.33] |

(lead-48 error is ~100% for both real and every surrogate at every rank —
no gap there either.) **There is no rank from 5 to 200 at which real data
beats the covariance-implied error floor.** The anomaly's high effective
DMD rank is statistically indistinguishable from correlated Gaussian
noise sharing its own second-order statistics — not evidence of
recoverable structure a higher or different `r_dmd` could still capture.
This turns §7's speculative reframing into a tested result, and means
extending the sweep past 200 (flagged in §6 as cheap and open) is no
longer expected to find anything: the ceiling isn't a search artifact.

**Experiment 2 — full DMD+OU generative statistical-fidelity check**
(`notebooks/07_dmd_ou_generative_check.py`, not yet written). Per
`r_dmd` candidate: `fit_mode_ou(gauss_train, dmd_fit(...))`
(`src/train/dmd.py`, already implemented/tested —
`tests/train/test_dmd.py::test_mode_ou_parameter_recovery`); guard against
any candidate with `max|Lambda| >= 1` (today's data has
`max|Lambda| ∈ [0.977, 0.982]` for every rank 5-200, but nothing enforces
this structurally, and an unguarded growing mode would silently blow up
every downstream metric). Seed `eta_0` from the *measured* residual at the
end of training (not `sample_ou_equilibrium`); generate 10 independent
realizations of the holdout period by chaining `simulate_ou_forward(ou,
eta_prev, n_steps=1.0, rng)` **one step at a time** (a single vectorized
multi-step call would draw each horizon's noise independently, giving the
right marginal but the wrong joint law — silently decorrelating the path).
Add back the (numerically negligible, given `n_train≈1754` already
underflows `Lambda**t`) deterministic term, reconstruct
`core_synth = (Phi @ a_synth).real.T.reshape(...)`. Compare the 10
realizations against the single real `gauss_holdout` via: Welch PSD
(vectorized across cells, averaged across realizations *after* computing
per-realization — never concatenated, to avoid a spurious restart artifact
at realization boundaries); mean-centered FFT-based ACF to lag 48; and
`scipy.stats.ks_2samp`'s **D-statistic** (not p-value — pooling ~1.5e8
values gives the p-value unlimited power). Any KS tail mismatch beyond
`|z|≈4.75` is expected to reflect the copula's own saturation, not an OU
failure on extremes.
**How to read it**: metrics still improving noticeably out to `r_dmd=200`
→ case (a); an early plateau that stays flat → case (b), consistent with
Experiment 1's error-floor finding — the two should agree, and a
disagreement between them is itself worth flagging rather than picking
one.

**Bug found and fixed (2026-09-24).** The first real run's
`imag_real_ratio_mean` came back ≈0.93-1.04 at *every* rank — the
generated signal was comparably imaginary as real, not "real plus
negligible noise" (contrast the deterministic reconstruction's ~1e-8 to
1e-16 in Experiment 1). Root cause: exact DMD on real data guarantees
algebraically exact complex-conjugate mode pairs, but `fit_mode_ou` +
`simulate_ou_forward` fit and sample each mode's OU noise
*independently* — nothing enforces that a conjugate pair's two
"independent" draws stay conjugates of each other. This is a known,
documented gap (`docs/open_questions.md` §5.2, both `fit_mode_ou`'s and
`simulate.py`'s own docstrings flag it as deferred to task 9's "coherent +
local sampling rewrite"). Discarding a same-order-of-magnitude imaginary
part via `.real` isn't safe truncation — it throws away real signal
energy, and made the non-monotonic `ks_d_statistic` from that first run
(worse at `r_dmd=150/200` than at `100`) suspect as an artifact rather
than a real finding.

**Fix, decided 2026-09-24 (scoped to this experiment script only, not
`src/` — see `docs/open_questions.md` §5.2's update for the full
writeup).** Rather than mechanically forcing the partner mode to the
conjugate after independent sampling, model the physical quantity that's
actually shared: for a conjugate pair $(k,k')$, $|\eta_k(t)| =
|\eta_{k'}(t)|$ exactly, always, so only *one* real, non-negative
quantity — the magnitude — needs a stochastic model at all. Phase is
purely deterministic, advancing at that mode's own frequency
($\arg(\lambda_k)$), continued from the true measured phase at the end of
training. Sharing one magnitude draw across the pair and negating the
phase for the partner makes them exact conjugates *by construction* — not
a bookkeeping fix, but a genuinely more constrained noise model than
independent complex-residual OU: the stochastic part only ever modulates
the amplitude envelope, never introduces a competing/independent
frequency. Magnitude is fit in **log** space (de-meaned, mean restored at
generation), consistent with this project's convention for
physically-positive quantities (log-energy for wave height). Implementation
reuses `fit_mode_ou`'s exact tested AR(1)/OU estimator formula (factored
out, not re-derived) applied to the log-magnitude series instead of the
raw complex residual; real (non-oscillatory) modes are untouched.
Verified: `imag_real_ratio` dropped from ≈0.5-0.9 (naive, smoke test) to
≈3×10⁻¹⁷-6×10⁻¹⁷ (machine precision, at *every* rank 5-200 on the real
35,712-dimensional core) — and the pairing counts (`n_conjugate_pairs`,
`n_real_modes`) sum to `r_dmd` exactly at every rank, confirming no mode
is silently dropped or double-counted.

**Result (2026-09-24), after the fix**
(`results/task07/gaussianize/dmd_ou_generative_check_wsh_r124x288_202401-202410.csv`):

| r_dmd | log_psd_distance | ks_d_statistic |
|---|---|---|
| 5   | 46.6 | 0.381 |
| 50  | 7.0  | 0.063 |
| 80  | 5.4  | 0.059 |
| 150 | 1.8  | 0.099 |
| 200 | 2.2  | 0.180 |

Spectral match improves steadily out to `r_dmd=150`, then is
flat-to-slightly-worse at `200` — consistent with Experiment 1's ceiling,
no contradiction. The marginal match (KS-D) is **not** monotonic: it
bottoms out around `r_dmd=50-80` and gets *worse* from `100` onward.
Plausible explanation, not separately verified: `cond(Φ)`'s well-known
rise across this range (2.2→227, §6) degrading the numerical stability of
the per-mode magnitude/phase estimates used for generation — a
*different* mechanism from "missing structure," specific to the
generative step rather than the one-step predictive error Experiment 1
measures. If this holds up, it's a second, independent reason (beyond
"higher rank doesn't reveal more structure") to prefer a more modest
`r_dmd`, tentatively in the 50-100 range, rather than the top of the
tested range.

Design fully reviewed (surrogate construction, OU-chaining requirement,
anchor/indexing conventions, ACF mean-centering, KS pooling caveat, and —
after the fix above — the conjugate-pair/log-magnitude procedure all
independently checked) — see plan file history this session for the full
critique. **Implemented and run**: `notebooks/07_dmd_surrogate_rank_check.py`,
`notebooks/07_dmd_ou_generative_check.py`, both entirely locally against
the already-downloaded core — no VM, no Azure budget spent.

## 9. Real-data walk-forward cross-validation (2026-09-25) — confirms almost no rank sensitivity, not an artifact

Both §8 experiments rested on a single train/holdout split. This one asks
the same question a third, independent way — **only real data, no
synthetic surrogate at all** — by checking whether the conclusion
replicates across several genuinely different real held-out windows.
Full design: `dmd_rank_cv_plan.md` (repo root; reviewed by a Plan-agent
pass that caught several concrete cost/correctness issues before
implementation — wrong `TimeSeriesSplit` block-size math, an ambiguous
loop structure that could have cost 9x the compute, a single-anchor-per-
fold sparsity problem — all fixed before running).

**Method** (`notebooks/07_dmd_rank_cv_check.py`): 8-fold expanding-window
CV (`TimeSeriesSplit(n_splits=8, test_size=190)`) on the same cached
core, no surrogate. Per fold: `GaussianCopula` refit train-only, once
(not per rank); `dmd_fit` per `r_dmd`; held-out error evaluated via a
**rolling anchor across the entire ~190-step real test block** (142
anchors per fold via one vectorized projection, not the single training-
boundary anchor `07_dmd_rank_sweep.py` and §8 used), aggregated as each
fold's **median** (robust to the anchors' heavy within-fold
autocorrelation — descriptive within a fold, the actual cross-fold
comparison unit). Memory-capped at 25 GiB (`RLIMIT_AS`, same technique as
§4); ran to completion in ~50 minutes locally, no VM, cap never
triggered. Fold sizes exactly as designed: `n_train` from 672 (fold 0,
~84 days) to 2002 (fold 7, ~250 days), `n_test=190` (~24 days) every
fold.

**Result: real rank sensitivity is far smaller than the single-split
experiments suggested, and the pattern replicates near-identically
across all 8 folds**
(`results/task07/gaussianize/dmd_rank_cv_check_wsh_r124x288_202401-202410.csv`):

| r_dmd | mean 1-step error, 8 folds | std across folds |
|---|---|---|
| 5   | 99.89% | 0.073 |
| 50  | 99.68% | 0.082 |
| 100 | 99.62% | 0.100 |
| 150 | 99.60% | 0.115 |
| 200 | 99.60% | 0.132 |

The full spread from worst (`r=5`) to best (`r=150-200`) is **under 0.3
percentage points** — an order of magnitude smaller than the original
single-split experiment's ~1.2-point spread (99.66% → 98.49%). Per-fold
argmin: the four largest-training folds (`n_train≥1242`) find `r_dmd=200`
literally or effectively tied-best; only the three smallest folds
(`n_train≤1052`) show a slightly earlier optimum (100-150), with `r=200`
marginally worse by ≤0.08 points — plausibly smaller training sets making
the highest ranks a touch less reliable to estimate, not a robust
"200 is bad" signal. 48-step (6-day) error is pinned at ~100% regardless
of rank in every single fold — consistent with everything found before
(no long-horizon deterministic skill survives that far out).

**How this reconciles with §8, not contradicts it.** All three experiments
now agree on the actionable conclusion despite measuring different
things: the surrogate test showed real error is statistically
indistinguishable from noise at every rank; this CV shows essentially the
same thing from a completely independent, real-data-only angle — rank
barely moves true held-out predictive accuracy anywhere from 5 to 200,
replicated near-identically across 8 genuinely different real windows.
The generative check's sharper KS-based degradation past `r_dmd≈100`
measures a *different* thing (marginal-distribution fidelity of full
stochastic generation, not one-step predictive error) and isn't
contradicted here — this experiment does no generation/OU sampling at
all, so it has nothing to say about that finding specifically; it simply
doesn't reproduce anywhere near as sharp a rank effect on the pure
predictive-accuracy axis.

**One statistical caveat, worth flagging explicitly**: the per-fold
POD/variance-explained curve is **not** directly comparable across folds
of different sizes — e.g. fold 0 (`n_train=672`) shows 55.9% variance
explained at `r_dmd=200`, fold 7 (`n_train=2002`) shows only 28.9% at the
same rank. This is a known small-sample inflation effect (fewer total
samples means leading components mechanically explain a larger share of
a smaller total variance), not evidence that rank sufficiency itself
varies by season — the error-based comparison above is the one to trust
for rank selection, not POD averaged across differently-sized folds.

**Conclusion for the r_dmd decision, now on solid multi-method footing**:
this is decisive, real-data confirmation, not a single-split artifact —
one-step predictive accuracy essentially plateaus from the low end of the
tested range, with no real penalty for staying at a moderate rank.
Combined with the generative check's KS-fidelity minimum around
`r_dmd=50-100`, the working recommendation is unchanged and now more
strongly grounded: **pick `r_dmd` somewhere in 50-100** for production —
it costs almost nothing in one-step predictive accuracy relative to
`r_dmd=200` (this experiment), while avoiding the marginal-fidelity
degradation the generative check found past that point. `dmd_rank_cv_plan.md`
can be deleted now that its content is absorbed here.

## 10. Final `r_dmd` decision (2026-09-25): `r_dmd = 100`

Summary plots: `notebooks/07_dmd_rank_summary_plots.py` →
`dmd_rank_fidelity_tradeoff.png`, `dmd_rank_cv_predictive_error.png` (also
copied into `notebooks/` alongside the script, as the evidence backing
this decision).

§9's generative check has two metrics with opposite preferences: spectral
fidelity (`log_psd_distance`) bottoms out at `r_dmd=150`; marginal/KS
fidelity (`ks_d_statistic`) bottoms out at `r_dmd=80`. `r_dmd=100` sits
between them and is a genuine knee point, not just a round compromise —
the tradeoff is asymmetric in its favor:

| | `log_psd_distance` | `ks_d_statistic` | CV mean 1-step error |
|---|---|---|---|
| `r_dmd=80`  | 5.41 | **0.0592** (best) | 99.641% |
| `r_dmd=100` | 2.44 | 0.0662 | 99.620% |
| `r_dmd=150` | **1.82** (best) | 0.0992 | 99.601% |

At `r_dmd=100`, both metrics sit within 12-35% of their own true minimum.
Moving to `r=80` instead buys only an 11.9% improvement in `ks_d` at the
cost of a 2.2x (121%) *worse* `log_psd_distance`; moving to `r=150`
instead buys only a 25.7% improvement in `log_psd_distance` at the cost
of a 49.8% *worse* `ks_d_statistic`. Each neighbor trades a large loss on
one axis for a small gain on the other — `r_dmd=100` is where that
trade stops being worth it in either direction. The real-data CV result
(§9) doesn't discriminate between any of the three (99.60-99.64%, well
inside the ~0.08-0.13 fold-to-fold noise band), so it imposes no penalty
on this choice either.

**Decided: `r_dmd = 100` for the production pipeline**, closing the DMD
half of this task. (Tucker rank remains open — 20% is still a working
baseline, not a proven optimum; see §5's cost-cliff discussion.)

## 11. Does Tucker's spatial residual show up in the core? (2026-09-27)

Follow-up to §5.1's finding that Tucker's spatial reconstruction error
correlates strongly with storm events across independent ranks (r=0.83-0.94)
— does that same signal reach the core DMD actually operates on, or is it
entirely discarded before the core is ever formed? Checked directly on the
20% (converged) core, no VM needed (everything already local):

- **corr(spatial max-\|error\|, `‖core[t]‖`)** = 0.19; **corr(spatial
  max-\|error\|, `‖core[t]-core[t-1]‖`)** = 0.16 — both weak, nowhere near
  the 0.83-0.94 cross-rank correlation in the spatial error itself. The
  core does not carry a strong echo of the storms that dominate Tucker's
  spatial residual.
- **Visualized the raw core directly** (peak-storm day vs. calm day,
  robust percentile color scale — the raw min/max washed the plot out to
  near-uniform, same lesson as §5.1's static check):
  `notebooks/tucker_core_timelapse_20pct.gif` (548 frames, same window as
  the field timelapse, 50.5MB at dpi=45). **The core is not a coarsened
  version of the spatial map** — no continents, no coherent basin-scale
  patches, just structured noise-like texture. What's real: energy
  concentrates toward low mode-indices on both axes (first 10 lat-modes
  carry 37.7% of total energy vs. 0.9% for the last 10; first 10 lon-modes
  24.5% vs. 0.3%) — the surviving SVD-style ordering of Tucker's basis,
  not spatial structure. Peak-storm and calm-day cores look visually
  almost identical, consistent with the weak numeric correlation above.
  (A quick incidental confirmation of "the core is noise-like": it
  compressed far worse as a GIF than the field did — 115.8MB at the same
  dpi=70 that gave the 7-panel field animation only 175.9MB, despite being
  just 2 panels — smooth content compresses well, noise doesn't.)

**Conclusion**: Tucker's spatial residual (§5.1) and DMD's temporal
residual (what OU actually models) are confirmed to be different things
at different pipeline stages, not just different in principle. Whatever
spatial detail a given Tucker rank can't represent is discarded before
the core is ever computed — DMD/OU structurally cannot compensate for it,
because they never see the original spatial grid. This is why Tucker rank
and `r_dmd` remain two independent decisions in this task, not a single
tradeoff.

## 12. Final production run: Tucker 40% rank + copula + DMD (2026-09-29)

Closing this task using the last of the Azure credit ($21.9 remaining,
expiring the same day). Added a checkpoint-to-checkpoint early-stopping
mechanism to `07_placement2_tucker_on_anomalies.py` first
(`--early_stop_tol`, default `1e-5`): stops once a checkpoint's
improvement in reconstruction error, expressed as a **fraction**
(`err_pct/100` — the same scale as the ALS's own internal `tol=1e-5`
passed to `partial_tucker_lowmem`, not the `*100` percent scale
`err_pct` itself is reported in) drops below the threshold. Used
`--early_stop_tol 1e-4` for this run.

**The 40% run itself**, rank (248, 576), `--total_iters 300
--checkpoint_every 5`:

- Launched under an 8h wall-clock cap first (matching the 30% precedent),
  but killed it within minutes of starting, before it had produced any
  checkpoints: 40%'s own earlier short run put its steady-state pace at
  ~199 s/iter vs 30%'s ~126 s/iter (~1.58x slower), which projects to only
  ~140 reachable iterations in 8h, and a power-law fit on that same short
  run's checkpoint deltas (p≈1.32) projected the `1e-4` early-stop
  threshold wouldn't be crossed until ~iter 184 (~10.3h) — an 8h cap was
  very likely to time out before early-stopping ever fired. Relaunched
  immediately with an **11h cap** instead, at negligible sunk cost since
  the 8h attempt hadn't billed for any real compute yet.
- Ran to completion of the 11h `timeout` (exit code 124 — killed, not a
  natural early-stop) at **iteration 170/300, err_pct=23.2962%**. The
  final checkpoint-to-checkpoint delta was 0.0102pp (1.02e-4 fractional)
  — just barely *above* the `1e-4` threshold, meaning the run was on the
  verge of early-stopping naturally and would very likely have done so
  within one more 5-iteration checkpoint. For practical purposes this is
  a converged result, not an arbitrarily-truncated one.
- This iter-170 snapshot supersedes the iter-50 snapshot used for 40%'s
  reconstruction check in § above (that one was a deliberately short
  recovery run; this one is ~3.4x further converged and is the tensor set
  actually carried forward into the copula/DMD fit below).

**Copula + DMD, full core, no train/holdout split.** Unlike the
exploratory 20%-core analysis in §§ above (which used an 80/20 split
purely to measure rank-selection diagnostics), this is the production
fit: `GaussianCopula` and `dmd_fit(r_dmd=100, variant="exact")` (the
rank decided in §10) both trained on the **entire** 2192-timestep core —
validation for this fit happens separately, against real held-out data,
not via an internal split of this same window
(`07_placement2_copula_dmd_40pct.py`). Both steps were fast (142,848
cells vs. the ~894,240-cell full-grid copula fit that takes ~10 min per
`07_gaussianize_fit.py`'s docstring): copula fit in 110.0s, DMD fit in
41.1s, `Phi` shape `(142848, 100)` as expected.

**Artifacts retrieved and verified** (all load cleanly with the expected
shapes), in `results/task07/gaussianize/`:
`anomalies_wsh_r248x576_checkpointed300_202401-202410_tucker.npz` (core
`(2192,248,576)` + both factor matrices), the matching checkpoints CSV,
`wsh_r248x576_copula.joblib`, `wsh_r248x576_gaussianized_core.npy`
(`(2192,248,576)`), and `wsh_r248x576_dmd_r100.npz` (`Phi`, `Lambda`,
`b0`, `lam_cont`, `dt`, `variant`).

**Updated three-rank comparison** (20%/30% unchanged from § above; 40%
now at iter 170 instead of iter 50 — same four measurements, same cached
true-field slices):

| Measurement | 20% (converged) | 30% (iter 220, near-converged) | 40% (iter 170, near-converged) |
|---|---|---|---|
| Overall err_pct (checkpoint) | 32.90% | 27.4651% | **23.2962%** |

(The full coastal/gradient/skew/kurtosis diagnostic re-run against this
iter-170 tensor set was not repeated here — no further VM budget, and the
iter-50 version of that table in § above already established the
direction of every trend clearly enough not to expect it to reverse.)

**Three operational lessons worth recording**, all hit for real during
this run and worth not re-learning next time:

1. Python's stdout is block-buffered (not line-buffered) when redirected
   to a file with no `-u`/`PYTHONUNBUFFERED=1` — the run log showed zero
   bytes for the first ~9.7 hours despite the process making real,
   healthy progress (confirmed via the checkpoint CSV, which is opened,
   written, and closed explicitly on every row and so isn't subject to
   the same buffering). **Trust the checkpoint CSV over the stdout log**
   for any future live status check.
2. `rsync --partial`, if killed mid-transfer while a *different*,
   already-successful transfer to the same destination path is still
   being verified, can rename its own incomplete temp file over the good
   completed file — this silently truncated the 1.17GB tucker `.npz`
   from a valid 1,168,150,207 bytes down to 719,432,349 bytes. Caught via
   `np.load` failing with `BadZipFile`, recovered by restarting the VM
   (source file was untouched) and re-downloading cleanly. **Never run
   two transfers to the same destination path concurrently, and never
   `kill` a `--partial` rsync — let it finish or use a fresh destination
   path.**
3. A local background process launched via the harness's own
   backgrounding (and even one manually detached with `setsid`+`nohup`,
   which normally survives a terminal hangup) did **not** survive the
   local IDE crashing twice — the log stayed empty and the process was
   simply gone, with no signal-related explanation available for why
   `setsid` didn't help. This suggests the local execution sandbox itself
   is tied to the IDE process's lifecycle, not just its controlling
   terminal. The Azure VM, by contrast (a genuinely separate machine, jobs
   run in remote `tmux`), survived every local IDE crash and session
   boundary this task hit without incident. **For any nontrivial compute
   during a long session, prefer running it on the VM over locally, even
   when it doesn't strictly need the VM's resources.**

**Azure teardown.** VM deallocated immediately after each artifact
retrieval was verified locally; final teardown deleted `vm-weather-generator`
(VM), both its disks (`vm-weather-generator_OsDisk_...` and
`disk-weather-generator-data`), its NIC, public IP, NSG, and VNET.
`wxgenarchivee2` (the raw data archive) and the `mlw-weather-generator`
ML workspace and its supporting resources were confirmed untouched —
confirmed via `az resource list` before and after.

**Decided (pragmatically): Tucker rank = 40% `(248, 576)` for the
production pipeline.** Unlike `r_dmd=100` (§10 — a clean knee-point
across many fully-comparable CV-validated ranks), this is a
resource-constrained stopping point, not a rigorously validated optimum:
Azure credit is exhausted and no further VM runs are planned. Within the
evidence actually in hand, 40% is the best available choice — its
checkpoint error (23.30%) is substantially better than 30%'s (27.47%)
and 20%'s fully-converged baseline (32.90%), it was captured at
near-convergence (one checkpoint shy of its own early-stop threshold,
not an arbitrary cutoff), and its copula+DMD fit on the full core
completed cleanly with the production `r_dmd=100`. Both the Tucker rank
and `r_dmd` are now fixed for the production pipeline, closing this
task's two open decisions — with the explicit caveat above on how much
weight the Tucker side of that decision can bear.
