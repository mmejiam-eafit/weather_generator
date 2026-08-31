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

## 3. Notebook(s) needed

- `notebooks/09_growing_dmd_modes.ipynb` — loads `wsh` over a 4-month
  era-3 window, fits a Tucker core, and runs both the rank sweep and the
  sliding-sub-window sweep.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — measured growth persistence pattern, proposed clip/flag thresholds,
whether it changes `open_questions.md` §5.5's status tag, and any
follow-on task this spawns.)_
