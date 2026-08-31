# Task 07 — Tucker/DMD rank selection

**Depends on:** 00 (soft — ranks are chosen on whichever representation feeds Tucker; a placement change plausibly shifts the elbow), 17 (soft — discovered while extending this task: `wsh`'s original rank sweep was pathological due to a masking bug, now fixed by `train.masking`) · **Blocks:** —
**Open question:** `docs/open_questions.md` §5.3 — OPEN
**Primary source docs:** `docs/tucker_decomposition.md` §6, `docs/dynamic_mode_decomposition.md` §6

## 1. What question this will try to answer

How should Tucker ranks $(r_{lat}, r_{lon}, r_{time})$ and DMD rank
$r_{dmd}$ actually be chosen? Both source documents describe only a
diagnostic (elbow of the reconstruction-error curve for Tucker;
eigenvalue-spectrum/growth-rate plausibility for DMD), not an automatic
stopping rule. This is a per-variable (and per-group, per Task 04) judgment
call — no document proposes a data-driven automatic rank selector, and
this task doesn't build one either; it produces the empirical curves
someone making that judgment call needs.

## 2. Steps to answer it

1. Load one variable (`wsh`) over a longer window (several months) than
   the other tasks use, so the rank sweep has enough data to be meaningful.
   **Update (task 17, 2026-08-20):** the original `wsh` attempt at this
   window found only 7-11% of cells passing even a 90%-valid tolerance,
   because `wsh`'s NaN pattern conflates land with genuine partition
   absence — not a data problem, a masking bug. The notebook worked
   around it by substituting `sigwh` (see its own setup markdown); that
   substitution's results are kept below as a valid baseline, but `wsh`
   itself is now loadable properly via `train.masking.land_mask_from_sigwh`
   + `apply_partition_sentinel_policy` (100% valid coverage, validated in
   `notebooks/17_partition_nan_sentinel_policy.ipynb`) — rerun the full
   rank sweep on real `wsh` (and `wsp`) using that corrected mask before
   treating the `sigwh` proxy as this task's final word.
2. **Tucker rank sweep:** fit masked `partial_tucker` at several
   `(r_lat, r_lon)` candidates (holding time at full resolution, per
   `tucker_decomposition.md` §4.3), and record the relative Frobenius
   reconstruction error at each. Plot error vs. rank and look for the
   elbow.
3. **DMD rank sweep:** fix a Tucker core, then fit `train.dmd.dmd_fit` at
   several `r_dmd` values and record `max|λ|` and the count of modes with
   `|λ|>1` at each. Plausibility here means: does the eigenvalue spectrum
   stay physically reasonable (bounded growth/decay rates) as rank
   increases, not just "does reconstruction error go down."
4. **Decision rule:** choose the smallest rank past which reconstruction
   error stops improving materially (the elbow) for Tucker; choose the
   largest `r_dmd` before spurious growing modes start appearing for DMD.
   Record both curves and the chosen defaults — this task's output is the
   curves and a recommended default, not a single "correct" number, since
   the tradeoff is inherently per-variable.

## 3. Notebook(s) needed

- `notebooks/07_tucker_dmd_rank_selection.ipynb` — loads `wsh` over a
  ~6-month era-3 window, runs the Tucker rank sweep and the DMD rank sweep.

## 4. Conclusion

Run against the real archive (`sigwh`, `wsh`, `wsp`; 2025-11-01 to
2026-05-01, 280×230 crop, `T=1448`). Full rank-candidate set
`(5,10),(10,20),(20,40),(40,80),(80,160)` swept for all three variables
(`wsh`/`wsp` via task 17's corrected mask + sentinel policy —
`apply_partition_sentinel_policy`, `height_log_energy`,
`period_sentinel_from_archive`'s `T_SENTINEL_S=3.8369621352089984`).

**Tucker: no elbow within the feasible range, for any of the three
variables.** Error decreases monotonically and materially at every step,
steepest at the top of the range, not plateauing:

| rank | product | `sigwh` | `wsh` | `wsp` |
|---|---|---|---|---|
| (5,10)   | 50    | 9.64%  | 63.16% | 14.37% |
| (10,20)  | 200   | 4.64%  | 51.37% | 11.92% |
| (20,40)  | 800   | 2.28%  | 40.66% | 9.70%  |
| (40,80)  | 3200  | 1.07%  | 31.79% | 7.73%  |
| (80,160) | 12800 | 0.43%  | 23.81% | 5.84%  |

`wsp` (period, log space) reconstructs substantially better than `wsh`
(height/log-energy) at every rank, consistent with task 17's finding that
`wsh`'s `log E` is ~36.5% sentinel-heavy and bimodal while `wsp`'s field is
smoother.

**Second constraint found, not anticipated by this task's original doc:
peak memory, not the elbow, is what actually bounds the candidate set.**
Measured empirically per rank via an isolated, `RLIMIT_AS`-capped
subprocess (so an over-budget attempt fails as a clean `MemoryError`
instead of the OS OOM-killer taking down an unrelated process, given this
machine's 31GB total / ~18GB available RAM). Peak RSS for a single masked
`partial_tucker` fit on this crop is **~12.25–12.27GB, essentially
constant** from `(40,80)` through `(80,160)`, for both `wsh` and `wsp` —
dominated by fixed per-ALS-iteration unfolding/reconstruction cost of the
full `(280,230,1448)` tensor, not by target-rank core size (the `(80,160)`
core itself is only ~74MB). Running the full 5-rank `wsp` sweep
sequentially in one process also stayed flat at 12.27GB peak — memory does
not accumulate rank-over-rank within a kernel, provided each candidate's
`core`/`factors` are discarded once its error is recorded (not retained
for every candidate simultaneously). `(80,160)` is therefore the largest
candidate this task tested not because a real elbow was found there, but
because it is close to the largest rank this machine can fit at all.

**DMD:** unchanged from the pre-task-17 sweep (fixed Tucker core at
`(20,40)`) — real knee at `cond(Φ)`'s jump between `r_dmd=10` and
`r_dmd=20` (§ above), no growing modes observed up to `r_dmd=40`.

**Rank recommendation:** `(r_lat=80, r_lon=160)` for both `wsh` and `wsp`
— the largest empirically-validated candidate, not a provably optimal one.
This is a hardware-bounded recommendation, not an evidence-that-error-
plateaus one: anyone training on a machine with more available memory
should extend the sweep further (a smaller crop, an out-of-core/chunked
Tucker implementation, or more RAM are the three ways past the ~12.3GB
per-fit ceiling found here) before trusting `(80,160)` as a long-term
default. `r_dmd` recommendation unchanged: smallest `r_dmd` past which
`cond(Φ)` stops improving materially, i.e. `r_dmd≈20` per the existing
sweep, not re-run under the corrected mask (DMD operates on the Tucker
core, not the raw NaN-affected tensor, so task 17's fix doesn't change
that sweep's inputs).

This closes "no rank defined" for `wsh`/`wsp` and changes
`open_questions.md` §5.3's status to reflect both findings (elbow
non-existence within a feasible range, and the new memory-ceiling
finding) — see that section's own update. No follow-on task spawned; the
memory ceiling is recorded here as a hardware constraint for future
re-evaluation, not a bug to fix.
