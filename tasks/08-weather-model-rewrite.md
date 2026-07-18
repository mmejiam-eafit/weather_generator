# Task 08 — `WeatherModel` rewrite

**Depends on:** 02, 03, 04, 05, 06, 07. **Blocks:** 09, 10, 11.

## Current state

**Updated 2026-07-17 after tasks 02-06 landed.** `src/weather/model.py`
already reflects the *mechanical* fallout of those tasks (a same-session
compatibility patch, not task 08 itself): it wires the two-transformer
`ComingFromToUV`/`GoingToToUV` scaffold with the correct `wad_come_from`
pair (not `wad`), imports `tucker_fit`/`dmd_fit`/`fit_mode_ou`/`ModeOUFit`
from their new locations, calls `dmd_fit(..., )` with no `variant=` override
(so it silently always uses the function's own `"exact"` default), and
`save`/`load` already persist `TuckerFit.reconstruction_error` and
`DMDFit.variant`. What it does **not** yet have — this task's actual scope —
is everything below: a `dmd_variant` `__init__` param to actually control
that default, `include_trend`/`include_local_residual` params, the local
residual layer entirely (`local_residual_`/`local_residual.npz` don't
exist), `t_ref`/`tensorly_version`/`pydmd_version` on `ModelMetadata`, and
the `mode_ou_`/`local_residual_` attribute names (`self.ou_` is still named
`ou_`, holding a `ModeOUFit` — renaming it to `mode_ou_` is part of this
task, per step 6 below).

**The `except subprocess.CalledProcessError, FileNotFoundError, OSError:`
line (no parens) is not a bug** — an earlier pass here mistakenly called it
invalid Python-2 syntax. It's valid PEP 758 syntax (unparenthesized
multi-exception `except`), supported since this project's pinned Python
3.14; confirmed via `ast.parse`, `mypy --strict`, and passing tests in a
later verification pass. No fix needed; ignore the old claim if you see it
referenced elsewhere.

## Changes — `src/weather/model.py`

1. **`ModelMetadata`** — add per §8.5: `t_ref: datetime`,
   `dmd_variant: str`, `include_trend: bool`, `include_local_residual: bool`,
   `tensorly_version: str`, `pydmd_version: str` (alongside the existing
   `trained_at`, `t_train_start`, `t_train_end`, `variables`, `r_ocean`,
   `r_var`, `r_dmd`, `git_commit`, `sklearn_version`, `numpy_version`).
   Populate `pydmd_version` via `importlib.metadata.version("pydmd")` (pydmd
   has no `__version__` attribute — confirmed in a prior session) and
   `tensorly_version` via `tensorly.__version__` directly (tensorly does
   expose this one).
2. **`WeatherModel.__init__`** — add, with these exact defaults (§8.5):
   ```python
   dmd_variant: Literal["exact", "bop", "hankel"] = "exact",
   include_trend: bool = True,
   trend_significance_threshold: float = 0.1,
   include_local_residual: bool = True,
   local_residual_variance_floor: float = 0.02,
   ```
   Store verbatim on `self`, no validation (sklearn `clone()` transparency —
   §9 gotcha 16).
3. **Pipeline construction** (`_fit_impl`) — replace the single
   `DirectionalDecomposer` step with the two-transformer scaffold from §5.3:
   ```python
   Pipeline([
       ("coming_from_to_uv", ComingFromToUV(pairs=[
           ("wsh", "wsd", "u_wsea", "v_wsea"),
           ("was", "wad_come_from", "u_wind", "v_wind"),
       ])),
       ("going_to_to_uv", GoingToToUV(pairs=[])),   # tier 1: empty; populate for tier 3
       ("reproject", Reprojector(target_grid="wave")),
       ("climatology", ClimatologyRemover(
           diurnal_variables=self.diurnal_variables,
           include_trend=self.include_trend,
           trend_significance_threshold=self.trend_significance_threshold,
       )),
       ("standardize", PerCellStandardizer()),
       ("gaussianize", GaussianCopula(n_bins=self.copula_lut_bins)),
   ])
   ```
   **Tier-1 `self.variables` must be
   `["sigwh", "wsh", "wsp", "was", "wsd", "wad_come_from"]`** — note
   `wad_come_from`, **not** `wad`. (`docs/model_design.md` §8.6's own
   "Production runs" example snippet literally lists `"wad"` in that array —
   that one line contradicts §3.7/§3.8/§5.3's explicit statement that `wad`
   is unused and `wad_come_from` is the source of truth; see
   `tasks/README.md`'s "Known doc bug" note. Do not wire `wad` into any
   transformer here.)
4. **Decomposition chain**, after Gaussianization:
   ```python
   tucker_fit_ = tucker_fit(z, r_ocean=self.r_ocean, r_var=self.r_var, rng=rng)
   core = tucker_transform(z, tucker_fit_)
   dmd_fit_ = dmd_fit(core, r_dmd=self.r_dmd, dt=SNAPSHOT_CADENCE_SECONDS,
                       rng=rng, variant=self.dmd_variant)
   mode_ou_fit = fit_mode_ou(core, dmd_fit_)          # task 06
   ```
   Then, if `self.include_local_residual`:
   ```python
   core_deterministic = ...  # b0 * exp(lam_cont * t) per snapshot, no OU term added
   z_coherent = coherent_reconstruction(core_deterministic, tucker_fit_)   # task 07
   local_residual_fit = fit_local_residual(
       z, z_coherent, dt=SNAPSHOT_CADENCE_SECONDS,
       variance_floor=self.local_residual_variance_floor,
   )
   ```
   else `local_residual_fit = None`.
5. **Trend-significance diagnostic** (task 04's design decision): compute
   `trend_significance_ratio_ = |climatology.c_lin_[v]| / standardize.sigma_cell_[v]`
   per variable `v`, from `preprocess.named_steps["climatology"].c_lin_`
   and `preprocess.named_steps["standardize"].sigma_cell_` after both are
   fit — **note the actual attribute name is `sigma_cell_`, not
   `sigma_std_`** (§5.4's prose uses `sigma_std_`/`σ_std` as an alias, but
   `train/standardize.py`'s real fitted attribute, confirmed by reading the
   source, is `sigma_cell_: dict[str, np.ndarray]`, keyed by variable name
   — the same `dict[str, np.ndarray]`-per-variable shape as `c_lin_`, so a
   direct per-variable elementwise divide is correct; only cells with
   `include_trend=True` have a `c_lin_` entry at all — skip variables
   absent from `c_lin_` when `include_trend=False`). Log the fraction
   exceeding `self.trend_significance_threshold` (§5.4, §6.A.8).
6. **`from_components`** — add `mode_ou: ModeOUFit` and
   `local_residual: LocalResidualFit | None` parameters (replacing the old
   `ou: OUFit` parameter); store as `self.mode_ou_` / `self.local_residual_`.
7. **`save`/`load`** — per §7's artifact layout:
   - `mode_ou.npz`: `theta`, `sigma`, `phi`, `dt` (renamed from `ou.npz`).
   - `local_residual.npz` (new; only written if `include_local_residual`):
     `theta`, `sigma`, `phi`, `variance_ratio`, `active_mask`, `dt`.
   - `dmd.npz` gains a `variant` string field.
   - `tucker.npz` gains `reconstruction_error`.
   - `metadata.json` gains the six new `ModelMetadata` fields from step 1;
     `load()`'s version-mismatch warning (§9 gotcha 17) extends to
     `tensorly_version`/`pydmd_version` alongside the existing sklearn/numpy
     checks — "a mismatch on tensorly or pydmd almost always breaks loading."
   - `load()` must handle `local_residual.npz` being absent (model trained
     with `include_local_residual=False`) by setting
     `self.local_residual_ = None`, not raising.
8. Rename the `self.ou_` attribute (and the `ou`/`ou_fit` local variables in
   `_fit_impl`/`load`/`from_components`) to `self.mode_ou_`, consistent with
   step 6 and with `ModelMetadata`/§8.5's naming.

## Tests — `tests/weather/test_model.py`

- `test_fit_save_load_sample` — end-to-end on a small synthetic archive
  (existing test; update pipeline step names and new fields).
- `test_transform_inverse_roundtrip` (existing).
- `test_from_components_matches_fit` (existing; update signature for
  `mode_ou`/`local_residual` params).
- `test_sklearn_clone` — `sklearn.base.clone(model)` produces a fresh
  unfitted copy; extend to cover the new `__init__` params (§9 gotcha 16 —
  every param must round-trip through `get_params()`/`set_params()`
  unmodified).
- Add: `include_local_residual=False` round-trips through `save`/`load`
  with `local_residual.npz` absent and `model.local_residual_ is None`
  after load.
- Add: `metadata.json` round-trips `dmd_variant`, `include_trend`, `t_ref`
  correctly.

## Acceptance criteria

- `WeatherModel(...).fit(data_dir)` on a small synthetic tier-1 archive
  succeeds end-to-end, producing `mode_ou_`/`tucker_`/`dmd_` with no NaN.
- `save()` then `load()` round-trips every fitted attribute, including when
  `include_local_residual=False`.
- `python -c "import weather.model"` succeeds (already true today).
