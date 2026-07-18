# Task 05 — Tucker: replace hand-rolled HOSVD with TensorLy wrapper

**Depends on:** none. **Blocks:** 07, 08.

## Current state

`src/train/tucker.py`'s `hosvd()` hand-rolls two separate SVDs
(`sklearn.utils.extmath.randomized_svd` for the ocean-cell mode, a full
`np.linalg.svd` for the variable mode) and returns a `TuckerFit` with no
reconstruction-error diagnostic. `tucker_transform`/`tucker_inverse` use
`np.einsum` directly.

## Changes — `src/train/tucker.py`

Replace with the TensorLy-based implementation given verbatim in §5.5:

1. `import tensorly as tl; from tensorly.decomposition import
   partial_tucker; from tensorly.tenalg import multi_mode_dot`;
   `tl.set_backend("numpy")` — explicit, since TensorLy also supports
   jax/torch backends and shouldn't be left to pick one implicitly.
2. `TuckerFit` dataclass gains a `reconstruction_error: float` field
   (relative Frobenius error, diagnostic — new in v2).
3. Rename `hosvd` → `tucker_fit`:
   ```python
   def tucker_fit(
       Z: np.ndarray, r_ocean: int, r_var: int, rng: np.random.Generator,
       n_iter_max: int = 100, tol: float = 1e-4,
   ) -> TuckerFit: ...
   ```
   Body:
   ```python
   (_core, [U_ocean, U_var]), _errors = partial_tucker(
       Z, rank=[r_ocean, r_var], modes=[1, 2],
       init="svd", n_iter_max=n_iter_max, tol=tol, random_state=rng,
   )
   ```
   **This exact nested-tuple unpacking pattern was verified against the
   installed `tensorly==0.9.0` in a prior session** — a naive
   `core, factors = result` unpacking raises `AttributeError`, the doc's
   pattern above is the correct one for this pinned version. Compute
   `reconstruction_error` by reconstructing via `multi_mode_dot` in both
   directions and taking `norm(Z_recon - Z) / norm(Z)` (see §5.5's full
   `tucker_fit` body for the exact reconstruction call).
4. `tucker_transform`/`tucker_inverse`: reimplement via
   `multi_mode_dot(Z, [fit.U_ocean.T, fit.U_var.T], modes=[1, 2])` (forward)
   and `multi_mode_dot(core, [fit.U_ocean, fit.U_var], modes=[1, 2])`
   (inverse), replacing the `einsum` calls. Numerically equivalent to the
   old einsum, but keeps the module TensorLy-native throughout (§3.11) so
   there's exactly one tensor-contraction implementation, not two that can
   drift apart.
5. Keep the returned `U_ocean`/`U_var` at `float64` (v1 cast to `float32`;
   §5.5's dataclass docstring specifies `float64` — follow the doc, not v1's
   prior choice. The `float32`-for-storage downcast happens separately, at
   `WeatherModel.save()` time, per §7's artifact dtype policy — task 08).
6. **Gotcha #21** (§9) belongs in the notebook, not runtime code: the first
   cell of `test_pipeline.ipynb` (task 11) should print
   `partial_tucker.__doc__` and confirm the unpacking pattern before relying
   on it against a newer tensorly version. Do not add this as a runtime
   assertion inside `tucker_fit` itself — it would slow down every fit for a
   check that only matters once per tensorly version bump.

## Tests — `tests/train/test_tucker.py`

- `test_hosvd_roundtrip` → rename `test_tucker_fit_roundtrip`: analytic
  rank-`r` tensor recovered exactly at rank `r` (existing test; update the
  call site from `hosvd` to `tucker_fit`).
- `test_tensorly_return_shape` (§12.5, new) — confirms the tuple-unpacking
  pattern against the installed tensorly version; fails loudly on drift
  (the runtime-facing counterpart to gotcha #21's notebook check).

## Acceptance criteria

- `tucker_fit` reproduces `hosvd`'s numerical behavior on the existing
  roundtrip test (same rank-`r` analytic tensor recovered exactly).
- `reconstruction_error` is present and near-zero for a tensor that's
  exactly rank-`(r_ocean, r_var)`.
- `grep -rn "\bhosvd\b" src/ tests/` returns nothing after this task.
