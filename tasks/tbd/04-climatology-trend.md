# Task 04 — Climatology: add linear trend term

**Depends on:** none (independent of the direction-convention chain).
**Blocks:** 08.

## Current state

`src/train/climatology.py`'s `ClimatologyRemover` fits only harmonic terms
(`_harmonic_design_matrix`, line 20-36): intercept + annual cos/sin +
semi-annual cos/sin + optional diurnal cos/sin. No linear trend column.
`fit()` sets `self.t_train_start_` (line 85) as the time origin for
`_days_since_epoch`, used by the harmonic terms — v2 needs a **second,
distinct** epoch (`t_ref`, the training-window **midpoint**) specifically
for the new trend term (§3.9; §9 gotcha 13: "Reference epochs are TWO
datetimes, not one").

## Changes — `src/train/climatology.py`

1. Add `Δτ(t) = (t − t_ref) / 365.25 days` as a new design-matrix column.
   Full formula (§3.9, §5.4):
   ```
   μ(s,t) = c₀(s) + c_lin(s)·Δτ(t)
          + c₁(s)cos(ω_y t) + c₂(s)sin(ω_y t)
          + c₃(s)cos(2ω_y t) + c₄(s)sin(2ω_y t)
          [+ c₅(s)cos(ω_d t) + c₆(s)sin(ω_d t)]
   ```
2. `t_ref` is the training window's **midpoint**:
   `t_ref = t_train_start + (t_train_end - t_train_start) / 2`, computed at
   `fit()` time from `X`'s own time coordinate (first and last timestamps),
   stored as `self.t_ref_` (UTC `datetime`). Centering at the midpoint is
   what decorrelates `c_lin` from `c₀` (§3.9) — the harmonic terms keep
   using `t_train_start_`/days-since-epoch exactly as before (unaffected);
   only the new trend column uses `t_ref_` and year-normalized units. Do not
   reuse `t_train_start_` for the trend column even though it's tempting —
   that would recorrelate `c_lin` with `c₀`, defeating the point.
3. `__init__` gains two new parameters (§5.4):
   ```python
   def __init__(
       self,
       diurnal_variables: list[str] | None = None,
       include_trend: bool = True,
       trend_significance_threshold: float = 0.1,
   ) -> None: ...
   ```
   If `include_trend=False`, drop the `Δτ` column from the design matrix
   entirely — don't just fit it and zero the coefficient.
4. New fitted attribute: `self.c_lin_` — the raw per-cell trend coefficient,
   shape matching the other per-cell coefficient arrays. **Do not** compute
   `trend_significance_ratio_` inside this class: it needs `σ_std`, which
   isn't known here (`PerCellStandardizer` fits later, downstream in the
   Pipeline). Compute it in `WeatherModel._fit_impl` instead (task 08),
   post-hoc, as `|c_lin_| / standardizer.sigma_std_`, once both fitted
   objects exist — this avoids coupling `ClimatologyRemover` to a
   downstream transformer's fitted state.
5. Log (in `WeatherModel`, task 08, using this class's `c_lin_`) the fraction
   of cells with `trend_significance_ratio > trend_significance_threshold`
   (§5.4: "if under ~1%, consider dropping the trend term for a cleaner
   model").

## Tests — `tests/train/test_climatology.py`

- `test_harmonic_roundtrip` (existing — confirm still passes; harmonic terms
  are unaffected by this change).
- `test_trend_recovery` (§12.5) — synthetic per-cell linear trend recovered
  within 5%.
- `test_trend_uncorrelated_with_c0` (§12.5) — fit on synthetic data with a
  known trend + intercept; confirm centering at `t_ref` (the midpoint)
  decorrelates the recovered `c_lin` and `c₀` across cells, e.g. by
  comparing against a deliberately-off-center epoch and showing the
  correlation is measurably worse there.
- Add a case for `include_trend=False`: confirm the design matrix has one
  fewer column and `transform`/`inverse_transform` still round-trip
  correctly without the trend term.

## Acceptance criteria

- With `include_trend=True`, a synthetic dataset with an injected
  `0.05/year` linear trend recovers `c_lin` within 5% relative error.
- `t_ref_` equals the training window's exact midpoint, not `t_train_start_`.
