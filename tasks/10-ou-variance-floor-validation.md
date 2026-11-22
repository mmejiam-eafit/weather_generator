# Task 10 — OU variance-ratio floor validation

**Depends on:** — · **Blocks:** —
**Open question:** `docs/open_questions.md` §6.1 — OPEN
**Primary source docs:** `docs/ornstein_uhlenbeck_processes.md` §6

## 1. What question this will try to answer

Is the local-residual variance-ratio floor (`train.local_residual.fit_local_residual`'s
`variance_floor`, default 0.02) a physically sound cutoff, or just a
reasonable-looking default? The floor is grounded in an efficiency
argument (don't bother simulating a cell/variable whose local residual
explains negligible variance), which is sound as far as it goes, but the
actual downstream reconstruction-error cost of its omission has not been
validated end to end — that full validation depends on simulation
infrastructure that does not exist yet anywhere in this repo. This task's
scope is explicitly the floor's **direct, measurable effect** (how many
cells become "active" at a given floor), not that larger, currently
infeasible validation.

**Update (task 07, 2026-09-29):** task 07 did fit mode-level OU noise
(`notebooks/07_dmd_ou_generative_check.py`, using `train.dmd.fit_mode_ou`/
`ModeOUFit`) and validated its generative fidelity across `r_dmd ∈
{5,...,200}` — but that is a different OU model at a different layer of
the pipeline: per-DMD-mode dynamics on the reduced-order residual
`a_k(t) - a_det_k(t)`, with no `variance_floor`, no `active_mask`, and no
per-cell variance-ratio concept anywhere in it. `train.dmd.fit_mode_ou`
has no floor/clip logic at all; the floor this task is about lives only
in `train.local_residual.fit_local_residual`, which task 07 never
touches. Despite both being "OU", there is no code or data overlap — this
task remains fully open. Its own notebook already has a complete,
unexecuted scaffold (`variance_floor` swept over `[0.0, 0.01, 0.02, 0.05,
0.1, 0.2]`).

## 2. Steps to answer it

1. Load one variable (`wsh`) over a representative window, Gaussianize
   (`train.copula.GaussianCopula`) to get `Z`, and fit an independent
   per-variable Tucker + DMD to get the deterministic-only reconstruction
   `Z_coherent` (no mode-OU wobble added, per
   `train.local_residual.coherent_reconstruction`'s docstring — this
   matters, since including the wobble would leak coherent-layer
   stochastic variance into this fit and double-count it).
2. Flatten both to `(T, N_ocean, V=1)` — a valid degenerate case of
   `fit_local_residual`'s signature — since this notebook explores a single
   variable, not the old joint multi-variable design.
3. Sweep `variance_floor` over a range (0.0 to 0.2) and record the
   fraction of ocean cells marked `active_mask` at each, plus the median
   `variance_ratio` across cells.
4. **Decision rule:** report the active-fraction-vs-floor curve as the
   direct evidence for calibrating the floor's efficiency tradeoff (how
   many cells get a local-residual OU process vs. how many are dropped).
   Explicitly flag that this does *not* answer the downstream-error
   question — record that as a separate, still-blocked follow-on
   requiring simulation infrastructure, not something this task can close.

## 3. Notebook(s) needed

- `notebooks/10_ou_variance_floor_sensitivity.ipynb` — loads `wsh`, builds
  `Z`/`Z_coherent` directly via `tensorly` (bypassing `train.tucker`, which
  still assumes the old joint-variable tensor shape), and sweeps
  `variance_floor`.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — the measured active-fraction-vs-floor curve, whether it changes
`open_questions.md` §6.1's status tag, and confirmation that the
downstream-validation follow-on remains explicitly blocked on missing
simulation infrastructure.)_
