# Task 11 — Climatology quasi-stationarity

**Depends on:** — · **Blocks:** —
**Open question:** `docs/open_questions.md` §7.1 — OPEN
**Primary source docs:** `docs/harmonic_climatology.md` §6

## 1. What question this will try to answer

Is a fixed harmonic shape plus one linear trend, held constant over the
whole fitting window, an adequate model of the climatological background?
Three linked, unresolved facts: (a) real seasonal cycles' amplitude/phase
can drift over long records, which a fixed-shape harmonic design can't
represent; (b) a linear trend is only the best-fit constant rate over the
specific window used, not something that necessarily generalizes; (c)
because the trend coefficient is defined relative to whichever window it's
fit on, two fits over different windows aren't simply
noisier/cleaner estimates of one true number — they can honestly answer
subtly different questions (a short window can have interannual
variability alias into an apparent trend a longer window would average
away). Point (c) is the one directly testable against real data without
new infrastructure.

## 2. Steps to answer it

1. Load one variable with a genuine seasonal *and* diurnal cycle over a
   window long enough to support two overlapping sub-windows (`was`, wind
   grid — see the notebook's own note on why not `sst`: the ocean grid
   isn't currently loadable via `load_training_window`,
   `core.config.grid_coordinates` still raises for `"ocean"` even though
   `open_questions.md` §1.1 has since answered its orientation question;
   `core.config.py` just hasn't been updated to match).
2. Refit `train.climatology.ClimatologyRemover(include_trend=True)`
   independently over two overlapping sub-windows of the same loaded data
   (e.g. the earliest-possible and latest-possible 60%-length sub-windows).
3. Compare the recovered `c_lin_` (trend) coefficient maps between the two
   fits — both the mean magnitude of each and the magnitude of their
   difference.
4. **Decision rule:** if the window-to-window trend difference is
   comparable in size to the trend itself, that's direct, real-data
   evidence for point (c) above — trend coefficients from this design
   should be reported together with their fitting window, not treated as a
   window-independent physical constant, until the (separately unresolved)
   production training-window decision is made.

## 3. Notebook(s) needed

- `notebooks/11_climatology_trend_window_sensitivity.ipynb` — loads `was`
  over a ~5-month era-3 window, refits over two overlapping sub-windows,
  and compares the `c_lin_` maps.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — the measured trend-map difference relative to the trend's own
magnitude, whether it changes `open_questions.md` §7.1's status tag, and
any connection this suggests to the separate training-window decision.)_
