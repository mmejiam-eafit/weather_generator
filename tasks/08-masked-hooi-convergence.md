# Task 08 — Masked-HOOI convergence

**Depends on:** 17 (soft — this task's own mask must exclude only genuine
land/ice, not conflate it with `wsh`'s partition-absence NaN; see below) ·
**Blocks:** —
**Open question:** `docs/open_questions.md` §5.4 — OPEN (mitigation available)
**Primary source docs:** `docs/tucker_decomposition.md` §6

## 1. What question this will try to answer

Is masked HOOI's imputation of land/ice cells guaranteed to converge well?
No — masked HOOI converges monotonically to a stationary point of a
non-convex objective, not provably to the global optimum, and large
contiguous unobserved regions (large landmasses with little interior
coastline to anchor the imputation) are a realistic case for a poor local
completion. This is a genuine numerical limitation. The standard
numerical-optimization mitigation — run from multiple random
initializations and compare reconstruction error/imputed-region stability
across them — hasn't been applied anywhere in this repo; this task applies
it.

## 2. Steps to answer it

1. Load a variable with large contiguous land regions in its grid (`wsh`,
   wave grid) over a representative window. **Update (task 17,
   2026-08-20):** this task specifically studies convergence sensitivity
   at *imputed* (land) cells — so its mask must isolate genuine land/ice,
   not `wsh`'s own NaN pattern, which conflates land with partition
   absence (`docs/open_questions.md` §5.6). Use
   `train.masking.land_mask_from_sigwh` + `apply_partition_sentinel_policy`
   (this task was never run before task 17 landed, so there is no prior
   result to invalidate — just build it correctly from the start).
2. Fit masked `partial_tucker` at a fixed rank from `N_INITS` (≥8)
   different random initializations (different `random_state` per fit,
   same data/mask/rank otherwise).
3. Record, per initialization: the relative Frobenius reconstruction error,
   and the mean/distribution of the *imputed* values at land cells
   specifically (not the reconstruction error, which is computed only over
   masked-in/ocean cells — the imputed land values are what varies freely
   across initializations and is the actual convergence-sensitivity signal).
4. **Decision rule** (per `tucker_decomposition.md` §6's proposed
   mitigation): flag as "initialization-sensitive" if either the
   reconstruction-error spread or the imputed-land-value spread has high
   relative standard deviation across initializations. As a control, repeat
   for a variable/region with a smaller contiguous land region and check
   whether sensitivity correlates with land-region size, as hypothesized.

**Update (task 07, 2026-09-29):** task 07's cost-cliff investigation found
and fixed two real bugs in `tensorly`'s masked-ALS loop — a memory-hungry
mask-blend and a silently-ignored `svd="randomized_svd"` request — now
vendored in `src/train/masked_tucker.py::partial_tucker_lowmem`. That work
validated the fixed implementation converges monotonically and
predictably at real full-grid scale (20/30/40% ranks, all fixed-seed,
multi-hour VM runs), but never tested multi-initialization sensitivity —
every run used a single fixed `random_state`. Task 08's own notebook was
originally scaffolded before any of task 07's work existed (raw `wsh`,
era 3, 1 month, mask derived from `wsh`'s own NaN pattern — the exact
task-17 conflation bug above — and plain unfixed `tensorly.partial_tucker`
with `init="random"`), so it did *not* actually call the fixed
implementation despite this note previously implying it did.

**Redefined 2026-09-29** to genuinely reuse task 07's artifacts: era-2
grid, `wsh`, a 2-month sub-window (2024-01-01 to 2024-03-01) inside the
span task 07's fitted transformers cover; loads `era2_ocean_mask.npy`
(already computed via `land_mask_from_sigwh`, satisfying this step's own
requirement) instead of deriving a mask from NaN; applies the
already-fitted `wsh_climatology.joblib`/`wsh_standardize.joblib`
(`.transform()` only, no re-fit) through the same pipeline order as
`notebooks/07_placement2_tucker_on_anomalies.py`; and now genuinely calls
`partial_tucker_lowmem`. One real wrinkle: that function only supports
`init="svd"` or an `(core, factors)` tuple — `init="random"` is
unsupported by design ("no call site here ever uses it"). The notebook
now builds a genuinely-random `(core, factors)` tuple per seed itself
(uniform-random factors, projected through `multi_mode_dot` for a
correctly-shaped initial core) rather than using an SVD-anchored start,
which stays true to this task's actual question — do different starting
points converge to different local optima. Rank (`[20, 40]`, ~3% of the
era-2 grid) and iteration budget (fixed `n_iter_max=100`, not full
convergence) were both chosen deliberately small for local tractability
across 8 separate fits, not for production realism — this task's
question is about spread across inits, not full convergence, so early
divergence is itself a valid signal. This task remains open — the
notebook is redefined and ready to execute, but has not been run.

## 3. Notebook(s) needed

- `notebooks/08_masked_hooi_convergence.ipynb` — loads `wsh`, fits 8
  random-initialization masked Tucker decompositions at a fixed rank, and
  reports the reconstruction-error and imputed-land-value spread.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — measured spread across initializations, whether it changes
`open_questions.md` §5.4's status tag, and any follow-on task this spawns.)_
