# Task 05 — Triplet coupling lost to independent Gaussianization

**Depends on:** 04 (`open_questions.md` states §2.2's unresolved residual **is** §3.2) · **Blocks:** —
**Open question:** `docs/open_questions.md` §2.2 — PARTIALLY ANSWERED
**Primary source docs:** `docs/correlated_variables.md` §3–§5, `docs/variable_taxonomy.md` §6

## 1. What question this will try to answer

Does independent per-variable Gaussianization discard the physical
coupling within a height/period/direction triplet (e.g. `wsh`, `wsp`, and
the `wsd`-derived Cartesian pair)? The mechanism that's supposed to answer
this is already specified: `correlated_variables.md` §3–§5's joint DMD over
already-independently-Tucker-compressed coefficients. This is a real
answer, not a deflection — but it isn't a full closure, because *which*
variables belong in a joint group (Task 04/§3.2) is itself unresolved. So
this task is specifically about isolating the **within-triplet** sub-block
of Task 04's fitted joint operator, not re-running a separate experiment.

## 2. Steps to answer it

This task **reuses Task 04's experiment and fitted joint operator
directly** — do not refit a second time.

1. From `notebooks/04_joint_group_selection.ipynb`'s fitted joint DMD
   operator (wind-sea generation group), isolate specifically the
   sub-block relating `u_wsea, v_wsea, T` to each other (the within-triplet
   coupling), separate from the blocks involving `u_wind, v_wind`.
2. Compare this within-triplet sub-block's strength against what an
   independent, per-variable-only DMD fit (no joint state at all) would
   imply — i.e. confirm the coupling the joint fit recovers is genuinely
   invisible to the fully-independent design, not just present in both.
3. **Decision rule:** non-negligible within-triplet coupling recovered by
   the joint fit, that the independent per-variable fits cannot represent
   at all, is direct confirmation that (a) the coupling is real and (b)
   `correlated_variables.md`'s joint-DMD mechanism successfully recovers
   it — closing this question's remaining gap once Task 04 also confirms
   the wind-sea group specifically is a well-chosen (not over- or
   under-drawn) group for this purpose.

## 3. Notebook(s) needed

- *Shares* `notebooks/04_joint_group_selection.ipynb` (the within-triplet
  sub-block isolation is a small additional analysis on that notebook's
  already-fitted joint operator).

## 4. Conclusion

_(Blank. Fill in only after Task 04's notebook has been run against real
archive data — measured within-triplet coupling strength, whether it
changes `open_questions.md` §2.2's status tag from PARTIALLY ANSWERED to
ANSWERED, and any follow-on task this spawns.)_
