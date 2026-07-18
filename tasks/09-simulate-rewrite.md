# Task 09 — Sampling rewrite: coherent + local residual

**Depends on:** 08. **Blocks:** 10, 11.

## Current state

**Updated 2026-07-17 after task 06 landed.** `train/ou.py` (and its
`sample_ou_equilibrium`/`simulate_ou_forward`/`_complex_normal`) is gone.
`src/weather/simulate.py` currently has a **verbatim, unmodified port** of
those three functions pasted in as private module-level functions (same
math, only the `OUFit` type hint changed to `ModeOUFit` — identical fields)
— a same-session mechanical patch to keep the module importable, explicitly
*not* task 09 itself. `sample_batch`/`simulate_field` still evolve only the
mode-level OU and reconstruct only `Z_coherent` — no local-residual
contribution, and no complex-conjugate-pair forcing for the `"exact"`/
`"hankel"` DMD variants (§9 gotcha 8 — moot in v1, since it had no variant
concept, but load-bearing now that task 06's default is `"exact"`). This
task's actual work is unchanged by that patch: add the local-residual
evolution step, add conjugate-pair forcing, and — since `_complex_normal`/
`sample_ou_equilibrium`/`simulate_ou_forward` already exist verbatim in the
file now — treat step 1 below ("port `train/ou.py`'s deleted simulation
helpers in as private functions here") as **already done**; start from step
2 (conjugate-pair forcing).

## Changes — `src/weather/simulate.py`

1. **Port `train/ou.py`'s deleted simulation helpers in as private
   functions here** (task 06's decision): `_complex_normal(variance, rng)`
   and the mode-amplitude evolution step, operating on `model.mode_ou_`
   (renamed from `model.ou_`) — no behavior change from v1's
   `simulate_ou_forward`/`sample_ou_equilibrium`.
2. **Force complex-conjugate pairing** after drawing each mode's stochastic
   wobble, when `model.dmd_.variant in ("exact", "hankel")` (§5.8, §9
   gotcha 8): identify conjugate-pair mode indices (adjacent modes with
   `Lambda[k'] ≈ conj(Lambda[k])`) and, having drawn `η_k(t)` for one mode
   of each pair, set `η_{k'}(t) = conj(η_k(t))` rather than drawing both
   independently. Skip this step when `variant == "bop"` (`eig_constraints=
   {"conjugate_pairs"}` already guarantees it at fit time). Add the debug
   assertion from gotcha 8 — `assert np.abs(Y_sim.imag).max() < 1e-10`
   after reconstructing the real field — as a cheap (`O(N_state)` per unique
   query time) always-on check, not gated behind a debug flag unless
   profiling later shows it costs something at the target throughput.
3. **New local-residual evolution step** (§5.8 Step 3), added to both
   `sample_batch` and `simulate_field`:
   ```
   Z_local(s_q,v_q,t_q) = φ_local(s,v)^Δn · Z_local(s,v,t_ref)
                        + N(0, σ_local(s,v)²(1 − φ_local(s,v)^(2Δn)) / (2θ_local(s,v)))
   ```
   Same closed-form OU transition as the mode layer, but **real-valued**
   (not complex) and indexed by ocean cell + variable, not by mode.
   Implement as a small private helper, e.g.
   `_evolve_local_residual(local_residual_fit, prev_state, delta_n, rng)`,
   parallel in structure to the mode-amplitude helper from step 1 but using
   `rng.normal` instead of `_complex_normal`. If `model.local_residual_ is
   None` (task 08's `include_local_residual=False` case), skip this step
   entirely — `Z_local` contributes zero.
   - No previous state available (isolated query, no trajectory) → draw
     from equilibrium `N(0, σ_local²/(2θ_local))`, matching the mode layer's
     existing equilibrium-draw pattern (§5.8).
4. **Sum before inversion**:
   `Z(s_q,v_q,t_q) = Z_coherent(s_q,v_q,t_q) + Z_local(s_q,v_q,t_q)`, *then*
   bilinear-interpolate, *then* `preprocess_.inverse_transform` — same
   ordering as today (inversion on the full grid before spatial
   interpolation, §4), just with the extra additive term folded in first.
5. `_invert_to_physical_grid` needs no structural change here — it already
   just calls `model.preprocess_.inverse_transform`, which now runs both
   `ComingFromToUV.inverse_transform` and `GoingToToUV.inverse_transform` as
   pipeline steps, transparently, as long as task 08's Pipeline is wired
   correctly.
6. `simulate_field`'s step-by-step chaining (currently
   `for i in range(n_steps): eta = simulate_ou_forward(...)`) needs a
   parallel per-cell-vectorized local-residual trajectory alongside the
   existing mode-amplitude trajectory (§5.8: "The local residual is
   generated as `N_ocean` independent AR(1) trajectories, again
   vectorized" — no python-level loop over cells, only over time steps,
   matching the existing mode-trajectory loop's structure).

## Tests — `tests/weather/test_simulate.py`

- `test_batch_matches_single_point` (existing).
- `test_seed_reproducibility` (existing).
- `test_short_lag_variogram_nonzero` (§12.5, new) — samples show non-zero
  variance at 25 km lag; the runtime-facing proof that the local residual
  layer is actually contributing (§6.A.4's short-lag variogram check, now
  testable without a real archive).
- Add: `include_local_residual=False` still runs `sample_batch` without
  crashing (local contribution is identically zero, not an error).
- Add: a case exercising `dmd_variant="exact"` specifically, asserting the
  debug assertion (`Y_sim.imag` near zero) holds — i.e. conjugate-pair
  forcing is actually wired up at the call site, not just documented in
  task 06.

## Acceptance criteria

- `sample_batch`/`simulate_field` produce purely real output (imaginary
  residual `< 1e-10`) for all three `dmd_variant` values.
- Disabling the local residual layer still produces valid samples;
  enabling it measurably increases short-lag (25 km) variance versus
  disabling it, on a synthetic fixture engineered to have nonzero local
  residual variance.
