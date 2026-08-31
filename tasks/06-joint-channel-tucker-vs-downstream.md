# Task 06 — Joint multi-channel Tucker vs. downstream joint model

**Depends on:** — · **Blocks:** —
**Open question:** `docs/open_questions.md` §3.4 — OPEN (experiment updated to `{E,F}`, see §3.5), §3.5 — ANSWERED (artifact risk confirmed; `{E,F}` fix proposed, unverified)
**Primary source docs:** `docs/correlated_variables.md` §6, `docs/variable_taxonomy.md` §3.4

## 1. What question this will try to answer

Does within-group channel coupling (a single wave partition's derived
`u, v, log E, T`) require a joint multi-channel Tucker (stacking all
channels into one 4-mode tensor and fitting one joint decomposition), or is
it recoverable from independent per-channel Tucker fits plus a downstream
joint model? This was raised by a competing design proposal
(`physical_handoff.md`) and is a real architectural alternative to
`tucker_decomposition.md` §4.1's strict per-variable rule — not something
dismissable by citing that rule alone, since there's a genuine, structural
asymmetry (independent-per-channel truncation can discard a coupling that's
weak-marginal/strong-joint, before any downstream joint model ever sees
it). Separately, §3.5 already found that the naive `{log E, T}` channel
pair is algebraically confounded (both share a `log m₀` term with matching
sign), so the diagnostic must use the decorrelated `{log E, F}` basis
(`F := log E - T`, depending only on `m₁`), not `{log E, T}`.

## 2. Steps to answer it

Exactly `correlated_variables.md` §6's specified experiment, using `F` per
§3.5's fix:

1. **Scope to one partition:** wind-sea (`wsh`/`wsd`/`wsp`) — genuinely
   independent height (unlike `pwh`) and the clearest-motivated steepness
   relationship (§3.3).
2. **Fit (a):** one joint 4-channel Tucker on `{u, v, log E, F}` at a
   modest, fixed rank budget.
3. **Fit (b):** four fully independent per-channel Tuckers on the same
   channels, at a comparable total rank budget, then CCA (or §4.2's joint
   DMD, for the fuller temporal-dynamics check) on the resulting
   coefficient series.
4. **Diagnostic, with a null baseline:** does (a)'s fitted `A_C` contain a
   column indicating genuine `log E`/`F` coupling? Does (b)'s downstream
   joint model recover a comparable version, or a materially weaker one?
   Run **both** against a shuffled/synthetic null baseline (permute one
   channel's time axis) first — this is required, not optional, precisely
   because `{log E, F}` (unlike the discredited `{log E, T}`) removes the
   *mechanical* confound but doesn't itself guarantee a clean null; a
   positive finding must exceed what the null alone produces before it
   counts as evidence of real physics.
5. **Decision rule:** if (b) recovers the coupling about as well as (a),
   the independent per-variable design is safe to keep as the cheaper
   default. If (a) finds a materially stronger relationship the null
   baseline doesn't explain, that's a specific, checkable justification for
   a narrowly scoped exception — restricted to one partition's own already
   physically-related channels, never extended across unrelated variables.

## 3. Notebook(s) needed

- `notebooks/06_joint_channel_tucker_vs_downstream.ipynb` — loads the
  wind-sea triplet, builds all four channels (`u`, `v`, `log E`, `F`), fits
  both (a) and (b), and runs the CCA-with-null-baseline diagnostic.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — measured real vs. null canonical correlation, the decision per step
5, whether it changes `open_questions.md` §3.4's status tag, and any
follow-on task this spawns.)_
