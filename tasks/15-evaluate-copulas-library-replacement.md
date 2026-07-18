# Task 15 — Evaluate whether the `copulas` package can replace `train/copula.py`

**Depends on:** none (informational/evaluation only). **Blocks:** none. Not
part of the v1→v2 migration dependency chain — this is a standalone
build-vs-buy assessment, separate from [[14-copula-dependency-isolation]]
(which only guards against an *unintentional* drift; this task asks the
opposite question — should we do it *on purpose*).

## What this task is, and isn't

This is a **research/evaluation task, not an implementation task**. The
deliverable is a new notebook, **`notebooks/evaluate_copulas_replacement.
ipynb`**, that runs the checks below live and prints/plots its own
evidence — not prose written from memory. This follows the same pattern
as `notebooks/verify_direction_convention.ipynb` and
`notebooks/verify_ocean_grid_orientation.ipynb`: an executable record that
anyone (a person, or a future agent) can re-run at will — e.g. after a
`copulas` version bump — and get a fresh, trustworthy answer instead of
inheriting a stale claim. **Do not modify `src/train/copula.py` as part of
this task.** If the notebook's verdict is "yes, it can replace it," that's
a separate, explicitly-approved follow-up task — this one only answers the
question.

## Background: what `train.copula.GaussianCopula` actually does

Per `docs/model_design.md` §3.5/§5.4 and the class's own docstring: for
each ocean cell × variable **independently**, it fits an empirical CDF
(a `n_bins`-quantile lookup table) and uses it to rank-transform that
cell's values to a standard normal (`transform`), or invert the process
(`inverse_transform`). Critically, **it does not model any joint/
multivariate dependency structure** — cross-cell and cross-variable
correlation is left entirely to the downstream Tucker + DMD + mode-OU +
local-residual decomposition (task 05-08) operating on the Gaussianized
output. The docstring is explicit that this has zero joint tail
dependence, and that this is a documented, accepted limitation (§3.4's
escalation path), not something this class is meant to fix.

This matters for the evaluation: "copula" in this module's name refers to
the general marginal-to-uniform-to-normal transform technique, not a full
joint copula fit. A library named `copulas` may be solving a materially
different problem (joint dependency modeling) than what this module does
(independent per-cell marginal transform).

**Scope boundary — read this before writing the verdict.** This task
evaluates `copulas` as a drop-in replacement for `train.copula.
GaussianCopula`'s role *in the pipeline as currently architected*. "The
downstream Tucker+DMD+mode-OU+local-residual decomposition already handles
cross-cell/cross-variable structure, so a joint-copula step would be
redundant" is a real observation, but it is an **architecture-fit**
argument, not a **hard technical blocker** — it implicitly assumes the rest
of the pipeline stays fixed. A redesigned pipeline could in principle have
a joint-dependency step feed the decomposition differently, or absorb part
of what Tucker/DMD currently do. Whether such a redesign would be a good
idea is a separate, larger architecture question this task does **not**
answer. Keep this argument clearly labeled as a soft, current-architecture-
dependent consideration in the verdict — never let it stand as a
self-sufficient disqualifier alongside the hard blockers in point 3 below.

## What "replace" would require, concretely

Whatever stands in for `train.copula.GaussianCopula` must, at minimum:

1. Implement the sklearn `BaseEstimator`/`TransformerMixin` surface used
   throughout this pipeline: `fit(X: xr.Dataset)`, `transform(X) ->
   xr.Dataset`, `inverse_transform(X) -> xr.Dataset`, and survive
   `sklearn.base.clone()` (§9 gotcha 16 — see `test_sklearn_clone` in
   `tests/train/test_copula.py` for the exact bar).
2. Fit and transform **independently per cell**, not jointly — see
   Background above and the scope-boundary note on why "it fits a joint
   structure instead, which is redundant with the downstream
   decomposition" is a soft, architecture-fit signal, not by itself proof
   the candidate can't work. Treat it as a flag that motivates checking
   point 3 closely, not as a substitute for checking it.
3. Scale to the real grid, **with the claim actually measured, not
   extrapolated from one data point**: `docs/model_design.md` §7 cites a
   `(621, 1440)` grid — on the order of 10⁵–10⁶ ocean cells × ~6
   variables. `train/copula.py`'s own `_searchsorted_per_column`
   docstring calls out (§9 gotcha 10) that a per-cell Python loop is a
   hard blocker at this scale — the current implementation is fully
   vectorized numpy with no loop over cells. Any candidate must clear the
   same bar. This is the actual hard blocker, if it is one — see the
   "Rigor requirements" section below for how it must be measured.
4. Preserve exact-invertibility semantics: `inverse_transform` must
   recover values via the *same* binned empirical quantile LUT used to
   fit (not a smoothed/parametric marginal approximation, e.g. a fitted
   KDE or Gaussian marginal) — `docs/model_design.md` is explicit that
   inversion should never produce values outside the training range,
   which only holds for a genuine empirical-CDF LUT, not a fitted
   continuous approximation.

## Rigor requirements for scale/performance claims

A prior pass at this notebook extrapolated real-grid-scale timings from a
**single measured data point** (e.g. `time_at_100k_cells / 100k *
300k`) — silently assuming linear scaling without checking whether the
measured data actually supported that assumption, and without any stated
uncertainty on a 2-3 order-of-magnitude extrapolation. Do not repeat that.
Concretely:

- Every extrapolated scale claim must be backed by **at least 4 measured
  sizes spanning at least 2 orders of magnitude**, with a fitted trend
  (e.g. `np.polyfit` on `log(size)` vs. `log(time)` — the slope is the
  empirical complexity exponent), not a single point's rate multiplied up.
- State the fitted exponent explicitly (e.g. "empirically ~O(n^1.05) over
  the tested range") and compare it against the algorithmically expected
  complexity where known (e.g. `train.copula.GaussianCopula`'s global
  `np.searchsorted` over a sorted array of size `n_bins * n_cells` is
  `O(n log n)`; a Python `for` loop instantiating one object per column
  has roughly constant *marginal* per-iteration cost, so linear
  extrapolation is a reasonable — if anything conservative — model for it
  specifically, and the notebook should say so explicitly rather than
  applying the same justification uniformly to every method).
- **`GaussianMultivariate`'s correlation-matrix step must be measured
  separately from its per-column fitting step.** `_get_correlation`
  calls `pandas.DataFrame.corr()` (at least `O(V^2)` for `V` columns)
  followed by `np.linalg.cond` on the result (SVD-based, `O(V^3)`) — this
  is a fundamentally different — and worse — scaling regime than the
  per-column model-selection loop, and the memory-only argument used
  previously (a `V x V` matrix needs `O(V^2)` RAM) understates the
  problem by not addressing its *time* cost at all. Time
  `gm._get_correlation(X)` directly at increasing `V` (a handful of
  points, e.g. 20/50/100/200 columns), fit its scaling exponent the same
  way, and report both the memory bound (already known) and the now-
  measured time bound.
- Any extrapolation beyond the largest tested size must be presented with
  an explicit caveat proportional to how far it reaches (e.g. "tested up
  to 200 columns, extrapolating ~1500x to 300,000 — treat as an order-of-
  magnitude estimate" is honest; a bare number with no such caveat is not).
- Where feasible without exceeding a few minutes of notebook runtime,
  prefer testing a size closer to real scale over trusting a longer
  extrapolation — e.g. push the `GaussianCopula` baseline and the
  `GaussianKDE`-loop benchmark up by another order of magnitude from the
  original pass if runtime allows, to shrink the extrapolation gap rather
  than just caveating it.

## Notebook — `notebooks/evaluate_copulas_replacement.ipynb`

One section per check below, each a markdown cell stating what's being
checked and why, followed by the code cell(s) that actually check it —
mirroring `notebooks/verify_direction_convention.ipynb`'s structure. The
notebook must run top-to-bottom with a fresh kernel (`Restart & Run All`)
with no manual steps and no network access beyond what's already installed
via `poetry`.

1. **API survey.** Import `copulas`, enumerate `copulas.univariate.*` and
   `copulas.multivariate.*` (`inspect.getmembers`/`dir`, plus printing
   each relevant class's docstring). Identify, and print a clear
   conclusion on, which class(es) — if any — fit an *independent,
   per-column, non-parametric empirical marginal*, as opposed to a joint
   multivariate copula (e.g. `GaussianMultivariate`, vine copulas). This
   determines which class(es) get benchmarked in step 2 — it is not
   itself a verdict; per the scope-boundary note above, finding only
   joint-fit classes available is a signal to check scale/time carefully
   (step 2), not a shortcut past checking it.
2. **Scale/performance benchmark.** Per the Rigor requirements above:
   measure at least 4 sizes spanning ≥2 orders of magnitude for each of
   (a) `train.copula.GaussianCopula.fit`/`.transform` (baseline), (b) the
   most relevant `copulas.univariate.*` class from step 1, looped per
   column, and (c) `GaussianMultivariate`'s per-column fitting step and
   its `_get_correlation` step **timed separately**. Fit a log-log trend
   to each, print the empirical scaling exponent, and only then
   extrapolate to real grid scale (`docs/model_design.md` §7:
   ~10⁵–10⁶ ocean cells) — with an explicit caveat on extrapolation
   distance. State plainly, in a markdown cell, whether each is tractable,
   distinguishing "measured" from "extrapolated" claims.
3. **Invertibility check.** Fit the candidate class, transform, invert,
   and plot/assert (inline, with visible output) that round-trip behavior
   matches this repo's semantics: LUT-based near-exact quantile recovery,
   inverted values bounded to the training range. Compare directly against
   `train.copula.GaussianCopula`'s round-trip on the same synthetic data
   (reuse `tests/train/test_copula.py::test_forward_inverse_roundtrip`'s
   fixture construction for a like-for-like comparison).
4. **Dependency footprint.** Run `poetry show copulas --tree` (via
   `!poetry show copulas --tree` or `subprocess`) and display the output —
   relevant context even if steps 1-3 are unfavorable, since a "no"
   verdict on those grounds means this footprint is being paid for
   nothing if `copulas` is later promoted from `dev` to a runtime
   dependency.
5. **Verdict.** Final markdown cell, computed conclusions only (no new
   claims not backed by an above cell's output). Structure it in two
   explicitly separated tiers, per the scope-boundary note above:
   - **Hard blockers** (architecture-independent — true regardless of how
     the rest of the pipeline is designed): scale/time-complexity
     infeasibility of `copulas`' actual algorithms at real grid scale
     (backed by the fitted scaling exponents from step 2, with
     extrapolation distance stated), and any semantic mismatch (e.g. no
     exact empirical-CDF inverse, inversion not bounded to the training
     range).
   - **Soft, current-architecture-dependent considerations** (e.g. "a
     joint-copula step would duplicate what Tucker/DMD already do in the
     pipeline as it exists today") — labeled explicitly as assuming the
     current architecture, not as an independent reason to say No.
   - Yes/No on whether `copulas` can replace `train.copula.GaussianCopula`
     **in the current pipeline**, driven by the hard blockers.
   - If Yes (or "yes with caveats"): what a follow-up implementation task
     would need to change, and any semantics that would differ from the
     current implementation (e.g. loss of exact LUT invertibility).

## Acceptance criteria

- `notebooks/evaluate_copulas_replacement.ipynb` exists, runs top-to-bottom
  from a fresh kernel with no errors, and its final cell states a clear
  Yes/No verdict backed by the evidence produced earlier in the same run.
- Every scale/performance claim is backed by ≥4 measured sizes spanning
  ≥2 orders of magnitude and a fitted scaling exponent — not a single
  point's rate multiplied up. `GaussianMultivariate`'s correlation-matrix
  step has its own separately-measured timing, not just a memory bound.
- The verdict cell keeps hard (architecture-independent) blockers and
  soft (current-architecture-dependent) considerations explicitly
  separate, and does not let a soft consideration stand in as a
  self-sufficient reason for a No verdict.
- `src/train/copula.py` is unmodified by this task regardless of the
  verdict.
- If the verdict is "No," [[14-copula-dependency-isolation]]'s guard
  remains the operative safeguard going forward — no further action
  needed here. If re-run later (e.g. after a `copulas` version bump) and
  the verdict changes, that's exactly the traceability this notebook is
  for — update the verdict cell, don't just note the discrepancy elsewhere.
