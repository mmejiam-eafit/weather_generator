# Task 17 — Partition-absence NaN vs. land/ice NaN (sentinel policy)

**Depends on:** — (informs 07, 08 — see their updated task docs) · **Blocks:** —
**Open question:** `docs/open_questions.md` §5.6 — new, filed by this task
**Source:** Discovered 2026-08-19/20 while re-running task 07's `wsh` rank
sweep (0-11% of cells passed even a 90%-valid tolerance over 6 months,
forcing the `sigwh`-substitution workaround documented in that notebook's
setup markdown). No document anywhere treats NaN in a partition-energy
field as anything but land/ice (see `docs/model_design.md:308,313,315`,
`docs/variable_taxonomy.md` §4) — this is a real gap, not a settled
decision being revisited.
**Primary source files:** `src/train/masking.py` (new), `docs/variable_taxonomy.md`
§2.2/§3.3/§3.4/§4, `docs/tucker_decomposition.md` §2.8/§4.2

## 1. What problem this will fix

For the three wave-partition triplets (wind-sea `wsh`/`wsd`/`wsp`, peak
`pwh`/`pwd`/`pwp`, primary swell `pswh`/`pswd`/`pswp`), NaN occurs for two
physically different reasons no document or code currently distinguishes:

1. **Land/ice** — genuinely missing data, correctly handled today (masked,
   imputed via masked HOOI).
2. **Partition absence** — a genuine ocean cell where that specific
   partition simply isn't present at that time — a real physical
   "~zero energy" state, not missing data.

There is also **no authoritative, variable-independent land mask anywhere
in the codebase** — `src/weather/model.py`'s `_mask_and_stack` and every
notebook infer land from a variable's own "NaN at any timestep" pattern,
which is exactly what breaks for the partition variables: task 07's
original `wsh` sweep found only 7-11% of cells passed even a 90%-valid-cell
tolerance over 6 months, so masked HOOI ended up imputing ~90% of the
domain, producing a reconstruction error that got *worse* with rank instead
of better. The notebook worked around this by substituting `sigwh`, which
has no partition-absence NaN, instead of solving the underlying issue.
`sigwh`'s own NaN pattern is a usable, independent land-mask proxy for the
whole wave grid, since it's the one wave variable that is NaN *only* over
land/ice.

**Three decisions confirmed, fixed inputs to this task, not to be
re-litigated:**

1. **Sentinel semantics.** At a genuine ocean cell where a partition
   variable is NaN, fill it with a sentinel and mark it **VALID** in the
   Tucker mask — treat "no partition" as a real physical observation
   Tucker/DMD fit directly on, not as missing data for masked HOOI to
   impute. Land cells keep today's treatment unchanged (zero-filled,
   INVALID/imputed).
2. **Blast radius.** Notebook + new reusable `src/train/masking.py` helper
   only, for now. **Not** wired into `src/weather/model.py` or
   `src/train/ingest.py`'s existing mask logic. Production wiring is a
   deliberately deferred follow-on task, contingent on this policy
   validating empirically — these notebooks are the justification for a
   pipeline change, not the change itself.
3. **Log-space companion policy (height and period need *different*
   sentinel values, decided now, not deferred).** Production Tucker input
   is `log E := log(H^2)` and `T := log T_m01` (`variable_taxonomy.md`
   §3.4), and both fields are defined from the same spectral moments
   (§2.2): $H_{m0}=4\sqrt{m_0}$, $T_{m01}=m_0/m_1$.
   - **Height:** $H\to 0$ is the clean, physically correct limit as
     $m_0\to 0$ — the raw sentinel `H=0` (decision 1) is exactly right.
     The only remaining problem is numerical: `log(0)=-inf`. Fixed by
     flooring before logging, uniformly (not just at sentinel cells — a
     genuine calm-sea `H=0` hits the same issue): `log E := log(max(H,
     H_FLOOR_M)**2)`, `H_FLOOR_M = 0.01` (1 cm) — a fixed physical
     constant, negligible relative to real signal, not archive-derived.
   - **Period:** $T_{m01}=m_0/m_1$ is a genuine $0/0$ indeterminate form
     as both moments vanish together — but not an arbitrary one. If a
     fading partition keeps a roughly fixed spectral *shape* while only
     its amplitude shrinks ($m_0(\varepsilon)=\varepsilon M_0$,
     $m_1(\varepsilon)=\varepsilon M_1$ for shape constants $M_0,M_1$),
     then $T_{m01}\to M_0/M_1$: a well-defined, shape-determined limit
     (an L'Hopital-style cancellation along the amplitude parameter),
     estimable from real near-vanishing partitions rather than
     typical-strength ones. `T_SENTINEL_S` is therefore derived — not
     invented — from real, non-sentinel, paired `(wsh, wsp)` observations
     in the bottom quantile of nonzero `wsh`: (a) the conditional median
     of `wsp` there, cross-checked against (b) a linear extrapolation of
     `wsp` vs. `wsh` in that same regime, read off at `wsh=0`. Marked
     **VALID** (not imputed) — a constant fill contributes negligible
     spurious variance to a low-rank fit, and marking it invalid would
     reintroduce height's original ~90%-domain imputation pathology for
     period's own (separate, per-`tucker_decomposition.md` §4.1) Tucker
     fit, since `wsp`'s NaN pattern is presumably ≈ `wsh`'s.
   - **Direction** (`wsd`/`pswd`) needs no independent sentinel.
     `variable_taxonomy.md` already documents the `(u_dir, v_dir)`
     embedding shrinking toward the origin for negligible-energy
     partitions — once `H`'s sentinel is 0, the embedding is `(0,0)`
     regardless of the angle, provided the angle isn't itself `NaN`
     (`0 * NaN = NaN`, not `0`, in IEEE754). Fix: substitute any finite
     placeholder for `wsd` at height-sentinel cells before
     `ComingFromToUV`/`GoingToToUV` runs — a numerical guard, not a
     modeling decision. This makes the `wsd`-vs-`wsh` asymmetry check
     (§2 step 4) purely diagnostic.

## 2. Steps to answer it

1. Derive an authoritative wave-grid land mask from `sigwh`'s own NaN
   pattern (`land_mask_from_sigwh`, strict all-timesteps), over the same
   representative window/crop task 07 already validated
   (`T_START=2025-11-01`, `T_END=2026-05-01`,
   `CROP=(-55.75, 14.0, 211.0, 268.25)`, 280×230, fully `sigwh`-clean).
2. For each remaining wave-grid variable (`wsh`, `wsp`, `wsd`, `pwd`,
   `pwp`, `pswh`, `pswp`, `pswd` — `pwh` excluded, degenerate/=`sigwh` per
   `variable_taxonomy.md` §3.3), compute the ocean-only NaN extent
   (`ocean_nan_report`) against that land mask, one variable at a time,
   foreground only. Categorize into no-ocean-NaN vs. has-ocean-NaN groups
   from the real, re-executed numbers.
3. Check the `wsd`-vs-`wsh` asymmetry (`partition_companion_backfill_check`)
   — is `wsd` genuinely always-defined at ocean cells, or backfilled/stale
   when `wsh` is NaN? Diagnostic only per decision 3 above; record the
   answer regardless.
4. Derive the log-space companion values per decision 3:
   `height_log_energy`'s floor sanity-check (no `-inf`/NaN, floor sits
   clearly below real data), and `period_sentinel_from_archive`'s
   conditional-median/extrapolated-intercept pair with their agreement.
5. **Decision rule / empirical validation.** Rerun task 07's `wsh` Tucker
   rank sweep (the one with the pathological worse-with-rank curve) using
   `apply_partition_sentinel_policy` + `height_log_energy` instead of the
   old mask/raw-log, same crop/ranks as the `sigwh` sweep. Run the
   equivalent `wsp` sweep using the derived period sentinel. A materially
   different (ideally not worse-with-rank) curve is the concrete pass/fail
   signal for whether this policy actually works — and the gate for any
   future production-wiring task.

## 3. Notebook(s) needed

- `notebooks/17_partition_nan_sentinel_policy.ipynb` — real archive,
  one variable per cell, foreground only (per this session's standing
  "don't crash the IDE" instruction). Runs §2's five steps against real
  data, produces the persisted categorization table, the log-space
  companion derivations, and the `wsh`/`wsp` rank-sweep comparison.
- `notebooks/07_tucker_dmd_rank_selection.ipynb` — gets a new setup cell
  importing `train.masking` and a `wsh`-with-corrected-mask rank sweep
  alongside (not replacing) the existing `sigwh` sweep.
- `notebooks/08_masked_hooi_convergence.ipynb` — still assumes `wsh`'s old
  land-only mask; needs a note (and, if already run, a rerun) once this
  task's mask is available.

## 4. Conclusion

**Policy validated against real archive data (2026-08-20), same window/crop
as task 07's `sigwh` sweep.** `notebooks/17_partition_nan_sentinel_policy.ipynb`
ran end-to-end. Results:

**Categorization** (§2 step 2, `train.masking.ocean_nan_report` against the
`sigwh`-derived land mask, not each variable's own NaN pattern):

| Variable | Ocean NaN fraction | Always-valid @ 100%/90%/70% |
|---|---|---|
| `wsh` | 36.5% | 0.0% / 1.3% / 35.8% |
| `wsp` | 36.5% (identical) | 0.0% / 1.3% / 35.8% |
| `wsd` | **0.0%** | 100% / 100% / 100% |
| `pwd` | 0.0% | 100% / 100% / 100% |
| `pwp` | 0.0% | 100% / 100% / 100% |
| `pswh` | 1.5% | 70.7% / 94.5% / 100% |
| `pswp` | 1.5% (identical) | 70.7% / 94.5% / 100% |
| `pswd` | 0.0% | 100% / 100% / 100% |

No-ocean-NaN group: `wsd`, `pwd`, `pwp`, `pswd`. Has-ocean-NaN group:
`wsh`/`wsp` (severe), `pswh`/`pswp` (mild). Height/period pairs go NaN
together exactly, as expected.

**`wsd`-vs-`wsh` asymmetry** (§2 step 3): `wsd` is never NaN when `wsh`
is, and is not a frozen filler — its variance during `wsh`-NaN periods
(std ≈ 74.5°) is *higher*, not collapsed, than during `wsh`-valid periods
(std ≈ 55.3°). Most likely the archive reports some actively-varying
wind-sea direction (plausibly `wad`-derived, which has no NaN at all)
even absent a detected partition — a real, open data-provenance question,
not resolved here, and not one the pipeline needs to resolve given the
embedding's own zero-at-H=0 behavior (§1 decision 3).

**Log-space companions, derived and validated** (§2 step 4):
- `H_FLOOR_M = 0.01` (1cm): `log E` is finite everywhere with this floor;
  real archive data's minimum `log E` (-5.99, i.e. `H≈5cm`) sits
  comfortably above the floor (-9.21), no overlap.
- `T_SENTINEL_S = 3.8369621352089984`s (`extrapolated_intercept` at
  `small_height_quantile=0.05`, the width where the two independent
  estimators agree best — 0.51s gap, vs. up to 3.3s at other widths).
  `extrapolated_intercept` stabilizes near 3.3–3.8s across
  `q=0.05`–`0.30` while `conditional_median` drifts upward toward the
  archive's overall median (6.94s) as the bin widens — the pattern
  expected if the intercept is the true zero-height limit. A real bug
  was found and fixed along the way: the initial `np.polyfit` call was
  ill-conditioned (uncentered `h` values far from zero, `RankWarning`
  even at 18M+ points) — fixed in `train.masking.period_sentinel_from_archive`
  by centering before fitting, standard numerical hygiene, not a change
  to the underlying method.

**Empirical validation — pathology fixed** (§2 step 5): under the
corrected mask, both `wsh` and `wsp` reach **100% valid coverage** (vs.
7–11% before), and reconstruction error now **decreases monotonically
with rank** — `wsh`: 63.2%→51.4%→40.7%; `wsp`: 14.4%→11.9%→9.7% (ranks
`(5,10)→(10,20)→(20,40)`) — the reverse of the original pathology
(4.51%→13.55%→32.76%, increasing). This is the concrete pass/fail signal
this task set out to check, and it passed. Absolute error remains higher
than `sigwh`'s own sweep at the same ranks (9.64%/4.64%/2.28%) — `wsh`'s
~36.5%-sentinel-heavy, bimodal `log E` distribution is plausibly harder
to compress than `sigwh`'s smoother field; a real secondary observation,
not investigated further here (task 07's own rank-selection sweep is the
right place to pick an actual `wsh` rank recommendation now that it's
loadable at all).

**Status: policy validated, notebook/`src`-helper scope only, per the
confirmed blast-radius decision.** `src/weather/model.py` and
`src/train/ingest.py` remain untouched. Production wiring is a separate,
not-yet-approved follow-on task — the empirical gate for proposing it has
now been cleared.
