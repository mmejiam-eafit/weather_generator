# Task 14 — Cross-cell/cross-time consistency: build-or-defer scoping

**Depends on:** — · **Blocks:** —
**Open question:** `docs/open_questions.md` §8.2 — OPEN, structurally so
**Primary source docs:** `docs/physical_consistency_constraints.md` §5, §6

## 1. What question this will try to answer

Can a per-query-point-independent check ever catch cross-cell or cross-time
physical inconsistency? No, and this is not a gap to be patched
incrementally — it is a structural property of the rejection-sampling
design as specified. A storm's wave field advecting across adjacent cells
at an implausible speed, or a field at $t$ and $t+\Delta t$ that is
mutually inconsistent, would satisfy every per-point constraint
individually and be accepted in full. Fixing this needs a fundamentally
different mechanism (a joint constraint over a spatiotemporal
neighborhood). **This task is therefore not a calibration** (there's no
single threshold to tune, per the source doc's own framing) — it's a
feasibility/scoping study: how often does real data actually exhibit the
kind of violation such a check would need to catch, what would the
cheapest version look like, and is it worth building.

## 2. Steps to answer it

1. Load `wsh` over a storm-active window at 3-hourly cadence.
2. Compute a real, concrete diagnostic of how much there is to catch:
   frame-to-frame displacement speed of the peak-`wsh` cell between
   consecutive snapshots (a simple advection-speed proxy). Real storm
   systems translate at roughly 20–50 km/h; outliers well above that in
   genuinely valid data are informative about how noisy a naive
   single-peak-tracking check would be, separate from the underlying
   question of how often *true* implausible advection occurs.
3. Prototype the cheapest plausible joint check: a local
   finite-difference advection-speed bound over a small spatial
   neighborhood plus one adjacent timestep — extending
   `physical_consistency_constraints.md` §5's existing
   tolerance-calibration method (percentile of real, valid data) to a
   2-point neighborhood instead of one query point.
4. Estimate that prototype's false-positive rate against real, physically
   valid data (using the distribution from step 2 as the calibration
   input), and estimate its added cost: it needs the previous timestep's
   full field at check time, not just the current query point — a real
   architectural change from the current fully-independent per-point
   design.
5. **Output a recommendation, not a parameter:** build now / defer / defer
   indefinitely, weighing the measured violation rate (step 2), the
   prototype's false-positive rate (step 4), and the added state-tracking
   cost against what the three existing per-point constraints already
   cover.

## 3. Notebook(s) needed

- `notebooks/14_spatiotemporal_consistency_scoping.ipynb` — loads `wsh`
  over a two-week storm-active era-3 window and computes the real
  frame-to-frame peak-cell advection-speed statistic (step 2). The
  neighborhood-check prototype (steps 3–4) is the concrete next addition
  once run against real data.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — the measured advection-speed distribution, the prototype's
false-positive rate, and the build/defer/defer-indefinitely recommendation
this task should produce. Explicitly not a calibrated `δ` parameter the
way Task 12's is.)_
