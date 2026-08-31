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

## 3. Notebook(s) needed

- `notebooks/08_masked_hooi_convergence.ipynb` — loads `wsh`, fits 8
  random-initialization masked Tucker decompositions at a fixed rank, and
  reports the reconstruction-error and imputed-land-value spread.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — measured spread across initializations, whether it changes
`open_questions.md` §5.4's status tag, and any follow-on task this spawns.)_
