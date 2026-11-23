# Task 13 — Diurnal-signal monitoring trigger

**Depends on:** — · **Blocks:** —
**Open question:** `docs/open_questions.md` §7.2 — OPEN, conditionally (not an active concern)
**Primary source docs:** `docs/harmonic_climatology.md` §6, `docs/noaa_etl/pipeline.md`

## 1. What question this will try to answer

Is the "waves have no diurnal signal" reasoning resolution-dependent?
Yes, explicitly and narrowly: the claim is stated relative to typical
global-model resolution and forcing data (GFS-Wave/WAVEWATCH III-class
products). It should be revisited only if this pipeline is ever pointed at
a higher-resolution regional wave model, where locally wind-driven
short-fetch seas forced by diurnal sea/land-breeze cycles could carry a
weak but real diurnal signature. **This is not an active concern at the
current archive's resolution** — this task is a re-runnable trigger check,
not a hypothesis test to resolve once.

## 2. Steps to answer it

This task's conclusion has an expiration date built in — re-run all four
steps whenever the archive's data source changes, not just once:

1. Re-read `docs/noaa_etl/pipeline.md`/`noaa_ingestion.md` and confirm the
   documented wave source is still GFS-Wave at global (0.25°) resolution.
2. Confirm `harmonic_climatology.md`'s `_DEFAULT_DIURNAL_VARIABLES`
   (`sst`, `u_wind`, `v_wind`) still excludes all wave-grid variables.
3. If both are unchanged, no action needed — but record the check date, not
   just "closed."
4. **Only if a change is found** in step 1 or 2: fit
   `train.climatology.ClimatologyRemover(diurnal_variables=["wsh"])`
   against real archive `wsh` data and compare the recovered diurnal-term
   amplitude against a real annual-term amplitude baseline (computed now,
   step 5 below, precisely so a future re-run has something concrete to
   diff against).
5. Compute the real baseline now regardless of steps 1–2's outcome: fit the
   diurnal-inclusive climatology for `wsh` and record the
   diurnal/annual amplitude ratio. This is expected to be small under
   current conditions — the number itself isn't the finding, it's the
   reference point for whenever this task is re-run.

**Update (task 07, 2026-09-29):** re-checked step 2 incidentally while
auditing task 07's overlap with other open tasks —
`_DEFAULT_DIURNAL_VARIABLES` in `src/train/climatology.py` is unchanged
(`("sst", "u_wind", "v_wind")`) and still excludes all wave-grid
variables, consistent with this task's assumption. This is only a
same-state confirmation, not new work: step 1 (re-reading the ETL docs
for the wave source) wasn't re-verified today, and steps 4–5 (the actual
diurnal/annual amplitude baseline) still haven't been run — this task
remains open per its own re-runnable-trigger design.

## 3. Notebook(s) needed

- `notebooks/13_diurnal_signal_monitoring_trigger.ipynb` — loads `wsh`,
  fits the diurnal-inclusive climatology baseline (step 5), and states
  steps 1–4's narrative check inline as markdown (not automatable — they're
  a documentation/config re-read, not a data computation).

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run — the baseline
diurnal/annual ratio, the check date, and confirmation this conclusion
**expires** and must be re-verified on future runs, unlike the other 14
tasks.)_
