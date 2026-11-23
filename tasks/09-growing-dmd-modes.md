# Task 09 — Growing DMD modes

**Depends on:** — · **Blocks:** —
**Open question:** `docs/open_questions.md` §5.5 — OPEN (mitigation available)
**Primary source docs:** `docs/dynamic_mode_decomposition.md` §6

## 1. What question this will try to answer

What should happen if a fitted DMD mode is genuinely growing ($|\lambda_k|
> 1$) rather than just noisy? If DMD's deterministic backbone drives
long-horizon stochastic simulation, a genuinely growing mode diverges
unboundedly. No document specifies detect-and-handle logic. A mitigation
consistent with this project's own established pattern
(`physical_consistency_constraints.md`'s "detect, don't silently absorb"
philosophy) is a reasonable default: clip $|\lambda_k|$ to 1 for modes
exceeding it by a small, plausibly-noise-attributable margin, but flag
(rather than silently clip) any mode whose growth exceeds a larger
threshold. This task produces the evidence needed to set those thresholds,
and to distinguish genuine from noise-attributable growth in the first
place.

## 2. Steps to answer it

1. Load one variable (`wsh`) over a multi-month window and fit a Tucker
   core.
2. **Sweep across DMD rank** ($r_{dmd} \in \{5,10,20,40\}$) on the full
   window and record the count of growing modes and `max|λ|` at each.
3. **Sweep across sliding sub-windows** at a fixed rank, and record the
   same. A mode that grows consistently across both rank choices and
   sub-windows is a stronger candidate for genuine growth (e.g. a real,
   sustained storm intensification captured within the window) than one
   whose apparent growth appears/disappears with rank or window choice
   (more likely noise-attributable).
4. **Decision rule:** use the rank/window-persistence pattern from steps
   2–3 to calibrate the "small margin → clip" vs. "large margin → flag for
   review" thresholds `dynamic_mode_decomposition.md` §6 proposes but
   doesn't quantify.

**Update (task 07, 2026-09-29):** step 2's rank sweep already has real
data, incidentally produced by task 07's own DMD rank selection —
`notebooks/07_dmd_rank_sweep.py` recorded `n_growing` and `max_lambda` at
`r_dmd ∈ {5,10,20,40,50,80,100,150,200}` (a superset of the sweep
proposed above), saved in
`results/task07/gaussianize/dmd_rank_sweep_wsh_r124x288_202401-202410.csv`.
Result: **`n_growing=0` at every single rank tested**, `max_lambda`
consistently in 0.977–0.981 — no genuinely growing mode found anywhere in
that range.

**Redefined 2026-09-29** to reuse task 07's artifacts fully, per the
user's explicit choice: step 2 now reads that CSV directly (no
recomputation) instead of fitting its own core; step 3 (sliding
sub-window sweep) now runs against the actual production 40% Gaussianized
core (`wsh_r248x576_gaussianized_core.npy`) at `r_dmd=100` (the decided
production rank), replacing the from-scratch era-3/4-month Tucker fit and
the originally-planned `r_dmd=20`. No Tucker fit happens in this notebook
at all anymore — both steps consume already-computed artifacts. Flagged
caveat, documented in the notebook itself: since era-2 already shows zero
growing modes at every tested rank, the production-core sliding-window
sweep may also find nothing to calibrate the clip/flag threshold
against — an honest null result, not a bug, and (per the explicit choice
made when redefining this) no era-3 fallback window was kept as a
backup source of a genuine growing-mode example.

## 3. Notebook(s) needed

- `notebooks/09_growing_dmd_modes.ipynb` — reads task 07's existing
  rank-sweep CSV for step 2, and slides sub-windows over the production
  40% Gaussianized core at `r_dmd=100` for step 3. No longer fits its own
  Tucker core.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — measured growth persistence pattern, proposed clip/flag thresholds,
whether it changes `open_questions.md` §5.5's status tag, and any
follow-on task this spawns.)_
