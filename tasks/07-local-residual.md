# Task 07 — Local residual OU field (new module)

**Depends on:** 05, 06. **Blocks:** 08, 09.

## Current state

Does not exist. v1 modeled only the coherent (Tucker+DMD+mode-OU) layer,
which produced samples that were too spatially smooth (§3.10). v2 adds a
second additive layer for per-cell temporal variability the coherent modes
miss: `Z(s,t) = Z_coherent(s,t) + Z_local(s,t)`.

## Changes — new file `src/train/local_residual.py`

Implement exactly as specified in §5.7 — both the dataclass and both
functions are given in full in the design doc; port them verbatim (the
formulas are already vectorized numpy — no loop, no joblib needed, per the
doc's own note in §9 gotcha 10):

```python
@dataclass(frozen=True, slots=True)
class LocalResidualFit:
    theta:          np.ndarray   # (N_ocean, V) float32
    sigma:          np.ndarray   # (N_ocean, V) float32
    phi:            np.ndarray   # (N_ocean, V) float32
    variance_ratio: np.ndarray   # (N_ocean, V) float32 — Var(R)/Var(Z), diagnostic
    active_mask:    np.ndarray   # (N_ocean, V) bool — True where variance_ratio > floor
    dt:             float


def coherent_reconstruction(
    core_deterministic: np.ndarray,   # (T, r_ocean, r_var) from DMD forward, no OU term
    tucker: TuckerFit,
) -> np.ndarray:
    """Compute Z_coherent(s, v, t) from the fitted DMD's deterministic reconstruction."""
    return tucker_inverse(core_deterministic, tucker)


def fit_local_residual(
    Z: np.ndarray,             # (T, N_ocean, V) Gaussianized anomalies
    Z_coherent: np.ndarray,    # (T, N_ocean, V) from coherent_reconstruction
    dt: float,
    variance_floor: float = 0.02,
) -> LocalResidualFit:
    """Fit independent per-cell OU to the residual R = Z - Z_coherent (§3.10, §5.7).

    Vectorized AR(1) fit across all cells simultaneously — no loop, no joblib.
    """
    ...
```

- `coherent_reconstruction` is a one-line call to `train.tucker.
  tucker_inverse` — it exists as its own named function only for call-site
  clarity in the notebook (§8.6 Cell 8) and in `weather/model.py`, not
  because the logic differs from `tucker_inverse` itself.
- **`Z_coherent` passed in is the *deterministic* reconstruction**
  (`b0_k · exp(lam_cont_k · t)` propagated through Tucker), *not* including
  the mode-OU stochastic wobble — confirm `weather/model.py` (task 08) calls
  this with `core_deterministic` (no `η_k(t)` term added). This matches
  §5.7's framing: "the field-valued analog of the mode-level OU... fit to
  the coherent-**reconstruction** residual."
- `fit_local_residual`'s body — copy §5.7's implementation exactly, including:
  - `R = Z - Z_coherent`; `R_prev = R[:-1]`, `R_next = R[1:]`.
  - `phi = clip(sum(R_prev*R_next, axis=0) / (sum(R_prev**2, axis=0) + 1e-30), -0.9999, 0.9999)`.
  - `theta = -log(clip(phi, 1e-4, 0.9999)) / dt` — **clip `phi` itself, not
    `|phi|`**: `docs/model_design.md` §5.7 previously said `|phi|` here,
    which is wrong (a verifier caught this 2026-07-17 — it's the same
    "clip magnitude instead of signed value" mistake §5.6.1's mode-OU
    formula had and was fixed for; negative `phi` means alternation/
    anti-correlation and should clip to the fast-decorrelation floor, not
    be read via its magnitude). Doc fixed to match; this spec now matches
    the doc.
  - `sigma = sqrt(innovation_variance * 2*theta / (1 - phi**2 + 1e-30))`
    where `innovation = R_next - phi*R_prev` and
    `innovation_variance = mean(innovation**2, axis=0)` — **not**
    `var_R = R.var(axis=0)`. A verifier caught this 2026-07-18 (numerically
    confirmed: at `phi=0.95` the marginal-variance version overestimates
    `sigma` by ~3.24x, matching the predicted `1/sqrt(1-phi**2)` bias
    exactly) — `docs/model_design.md` §5.7 previously used raw `var_R`
    here, the same mistake `train.dmd.fit_mode_ou`'s docstring already
    warns against for the complex mode-level case ("`Var(eta_k)`... is the
    AR(1) innovation variance... not the marginal variance... which would
    double-count the `(1 - |phi|^2)` factor"). `var_R = R.var(axis=0)` is
    still computed, but only feeds `variance_ratio` below, not `sigma`.
  - `variance_ratio = var_R / (Z.var(axis=0) + 1e-30)`;
    `active_mask = variance_ratio > variance_floor`.
  - Zero out `theta`/`sigma`/`phi` where `not active_mask` — this is what
    lets simulation skip inactive cells cheaply (§3.10: "save simulation
    work").
  - Keep the `+1e-30` numerical-stability epsilons exactly as given; they
    guard div-by-zero on cells with degenerate (all-identical) residuals.

## Tests — new file `tests/train/test_local_residual.py`

All four listed explicitly in §12.5:

- `test_variance_ratio_diagnostic` — synthetic `Z` with known coherent +
  local components; recovered `variance_ratio` matches the expected value.
- `test_active_mask_below_floor_zeroed` — cells with variance ratio below
  `variance_floor` get `theta == sigma == phi == 0.0` exactly (not
  approximately) after fitting.
- `test_vectorized_ar1_matches_per_cell_loop` — the vectorized fit produces
  identical results to a naive per-cell scalar OU fit. Write the loop
  version *only inside the test* as a ground-truth oracle — do not add a
  second implementation to the source module.
- `test_equilibrium_variance_matches` — sampling from the fitted OU at
  equilibrium (`N(0, σ²/(2θ))`) reproduces the residual's per-cell empirical
  variance.

## Acceptance criteria

- `fit_local_residual` on a synthetic `Z`/`Z_coherent` pair with a known
  planted per-cell OU residual recovers `(θ, σ)` within 5% (same tolerance
  as task 06's `test_mode_ou_parameter_recovery`).
- Cells with `variance_ratio <= variance_floor` have exactly zeroed OU
  parameters after fitting, not just small ones.
