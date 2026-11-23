# Task 00 — Gaussianization placement

**Depends on:** 07 (soft — only for re-validating i-vs-ii at a defensible
rank; does not block the decision already taken, see §4 Conclusion) ·
**Blocks:** 03, 07 (soft — see their own "Depends on")
**Open question:** `docs/open_questions.md` §4.1 — DECIDED (placement ii,
Tucker on log-energy anomalies then Gaussianize the core, before DMD —
supersedes the 2026-08-19 placement-i decision; see 2026-09-13 update below)
**Primary source docs:** `docs/gaussian_copula_and_marginals.md` §6.1

## 1. What question this will try to answer

Where, along each variable's independent processing chain, should marginal
Gaussianization happen? Three placements are live and unresolved: (i) on
raw per-cell physical anomalies (current default — maximal
interpretability, most LUTs, most expensive); (ii) on that variable's own
Tucker-core coefficients (fewer LUTs, but "per-cell" stops meaning a
physical grid location); (iii) on that variable's DMD mode
amplitudes/residuals (most directly serves the OU-innovation Gaussianity
requirement, but two prior linear compressions have already run on
non-Gaussianized data by then). This is the single most consequential open
question in the whole documentation set — `tucker_decomposition.md`,
`dynamic_mode_decomposition.md`, and `ornstein_uhlenbeck_processes.md` all
assume it's settled upstream and explicitly decline to settle it
themselves. No amount of further documentation-level reasoning closes it;
it needs empirical validation against real training data, once.

## 2. Steps to answer it

1. Load one real variable with a wide, informative dynamic range (`wsh`)
   over a representative window.
2. Fit all three placements on the *same* real data:
   - (i) Gaussianize the raw per-cell anomaly directly.
   - (ii) Fit an independent per-variable Tucker decomposition first
     (`tucker_decomposition.md` §4.1 — `tensorly.decomposition.partial_tucker`,
     masked, per `notebooks/01_understand_tucker_decomposition.ipynb`'s
     convention), then Gaussianize the resulting core coefficients.
   - (iii) Additionally fit DMD (`train.dmd.dmd_fit`) on that Tucker core,
     compute the deterministic-trajectory residual `η_k(t)`, and
     Gaussianize its real/imaginary parts independently.
3. Invert each placement fully back to **physical `wsh` units** — this is
   the comparison that matters, per `gaussian_copula_and_marginals.md`
   §6.1's own framing ("comparing physical-space QQ-plots and reconstructed
   extreme statistics under each placement"):
   - (i) inverts via the copula's own `inverse_transform` alone.
   - (ii)/(iii) additionally need the Tucker (and DMD) inverse before the
     comparison is in physical space.
4. **Decision rule:** for each placement, plot a physical-space QQ-plot of
   reconstructed vs. true `wsh`, and compare a reconstructed extreme
   statistic (e.g. 99th-percentile `wsh`) against the real archive value.
   The placement(s) that best preserve both, weighed against their LUT-count
   cost (§6.1's table: $n_{lat} n_{lon}$ vs. $r_{lat} r_{lon} r_T$ vs.
   $r_{dmd}$) and directness for OU's Gaussian-innovation need, is the
   recommendation this task should produce.

## 3. Notebook(s) needed

- `notebooks/00_gaussianization_placement.ipynb` — loads `sigwh` over a
  30-day era-3 training window plus a 10-day held-out test window, fits all
  three placements, inverts each back to physical units (training-period
  and held-out comparisons, QQ-plots + 99th-percentile recovery), and runs
  a follow-up Tucker/DMD rank-sensitivity sweep (see §4).

## 4. Conclusion

**Decision (2026-08-19): placement (i) — Gaussianize after physical
transformations and climatology removal, before Tucker — is adopted as the
standing default.** This is a deliberate design choice, justified by (a)
being the natural consequence of how the physical fields are already
processed (fit the marginal transform on the same residual scale that
Tucker/DMD subsequently compress, not on an opaque coefficient space), and
(b) matching the held-out result below, where (i) and (ii) were nearly
indistinguishable once the DMD-rank confound was controlled for. It is
*not* a proof that (i) beats (ii) — that comparison is explicitly left as
pending follow-on work (see end of this section), deferred rather than
blocking. Full reasoning and an external-literature cross-check are
recorded in `docs/open_questions.md` §4.1's 2026-08-19 update; the interim
finding that motivated waiting (below) is kept for context.

**Interim finding that preceded the decision — the comparison run was
confounded by an unresolved rank choice, not a final answer on its own.**

The training-period round trip (any monotonic invertible transform matches
its own training data closely by construction) was inconclusive by design,
as flagged directly by the notebook's own held-out section. Held-out
evaluation (fit on 30 days, test on the next 10, `sigwh`, southeastern
tropical Pacific crop) initially looked like a clean result in placement
(i)'s favor: placement (iii) (DMD residual) reconstructed the held-out
99th percentile with error **+255.89**, vs. **-0.02** for placement (i)
and **+0.15** for placement (ii) — all three also failed the KS
calibration test, but (iii)'s failure was catastrophic in a way the others
weren't.

A follow-up sensitivity sweep (same notebook, new cells after the held-out
comparison) traced that catastrophic failure to `dmd.Phi`'s condition
number, not to placement (iii) as a category: at the ranks used
(`r_lat=10, r_lon=20, r_dmd=20`), `cond(Φ) ≈ 1.2e6`. Lowering `r_dmd` to 5
on the identical data dropped `cond(Φ)` ~5.5x and the held-out error
~100x (to +2.36); also shrinking the Tucker core to a 32-dimensional state
(`r_lat=4, r_lon=8, r_dmd=5`) made the held-out reconstruction essentially
exact (**-0.05**, better than placement (i)'s own error). Widening the
spatial crop alone, holding rank fixed, helped much less (~2.5x on
`cond(Φ)`). Full table in `docs/open_questions.md` §5.3's update.

**This means the placement comparison this task exists to run cannot be
trusted at an arbitrary, unjustified rank** — a placement-(iii) failure at
an over-parameterized rank is evidence against that rank, not against that
placement. `docs/open_questions.md` §4.1 has been updated to record this
as a new dependency on §5.3 (Tucker/DMD rank selection), and this task's
header now lists a soft dependency on task 07 accordingly.

**Follow-on (deferred, not blocking):** re-run this task's comparison once
task 07 has a rank-selection approach (or at minimum, once `dmd.Phi`'s
condition number is reported and controlled for at every rank tested) to
check whether placement (ii) is actually equivalent to, or better than,
the (i) default adopted above — or resolve it via a more targeted
literature search instead. The condition-number diagnostic itself
(`np.linalg.cond(dmd.Phi)`, cheap, no held-out data needed) is a candidate
starting point for task 07. Implementation work proceeds with (i) in the
meantime.

**Decision (2026-09-13): superseded — placement (ii) is now canonical.**
This is exactly the follow-on comparison called for above, run by task 07
against real era-2 data (`tasks/07-tucker-dmd-rank-selection.md` §5): (i)
and (ii) at the same Tucker rank (31,72) gave statistically
indistinguishable error (50.81% vs. 53.04%). The empirical near-tie doesn't
favor either placement — the switch to (ii) is an architectural decision
(Tucker stays in physically-grounded log-energy anomaly space;
Gaussianization moves to sit immediately before DMD, the stage that
actually needs Gaussian innovations), not a result of this task's own
QQ-plot/extreme-statistic comparison, which was never re-run under (ii).
Full reasoning in `docs/open_questions.md` §4.1's 2026-09-13 update.
