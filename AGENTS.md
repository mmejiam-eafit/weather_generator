# AGENTS.md — Source Code Agent Protocol

## Mission

Implement the stochastic weather generator defined in
[`docs/model_design.md`](docs/model_design.md). Train on the raw 3-hourly NOAA
GFS-Wave / WAVEWATCH III + GFS wind archive; produce plausible synthetic
samples of any variable at any continuous `(lat, lon, datetime)`.

All production code lives under `src/` split across three sibling packages:
`core/`, `train/`, `weather/`. Training entry point is
`notebooks/test_pipeline.ipynb`. Tests under `tests/`.

There is **no CLI** — training is notebook-driven, and query is programmatic.

---

## Architecture Reference

The canonical design is in
[`docs/model_design.md`](docs/model_design.md). Read the relevant section
before implementing any module — it is the source of truth for data
conventions, algorithm, and API shape.

**Package dependency rules (enforced in tests):**

- `core/` — no internal deps. Imports only stdlib, numpy, xarray,
  pydantic-settings.
- `train/` — depends on `core`. May import from numpy, scipy, scikit-learn,
  xarray. **Does not** import from `weather`.
- `weather/` — public API. Depends on `core` and `train`. Consumers should
  import only from here.

**Two-layer training architecture:**

- **Preprocessing** (`train/directional.py` — `ComingFromToUV` and
  `GoingToToUV` both live here, §3.7 — plus `train/reproject.py`,
  `train/climatology.py`, `train/standardize.py`, `train/copula.py`) — six
  scikit-learn transformer classes chained via
  `sklearn.pipeline.Pipeline`. Each has a real `fit`, `transform`, and
  `inverse_transform`.
- **Decomposition + stochastic model** (`train/tucker.py`, `train/dmd.py`,
  `train/local_residual.py`) — pure functions taking arrays + hyperparameters,
  returning frozen dataclasses (`TuckerFit`, `DMDFit`, `ModeOUFit`,
  `LocalResidualFit`). Not transformers.

`weather.model.WeatherModel` inherits `BaseEstimator + TransformerMixin` and
glues the two layers together. Sampling operations are **free functions** in
`weather.simulate`, not methods on `WeatherModel`.

See handoff §3.12, §3.14, and §4 for the full rationale.

---

## Task Mapping

Each module maps to a section of the handoff. Always read the section before
implementing.

| Module | Handoff § | Status |
|---|---|---|
| `src/core/config.py` | §2.2, §2.7, §2.8 | ✓ Exists (extend) |
| `src/core/io.py` | §2.1, §2.3, §5.1 | ✓ Exists (extend) |
| `src/train/ingest.py` | §2.5, §5.1 | Not started |
| `src/train/directional.py` | §2.4, §3.7, §5.3 | Not started |
| `src/train/reproject.py` | §2.6, §3.6, §5.2 | Not started |
| `src/train/climatology.py` | §3.9, §5.4 | Not started |
| `src/train/standardize.py` | §5.4 | Not started |
| `src/train/copula.py` | §5.4 | Not started |
| `src/train/tucker.py` | §5.5 | Not started |
| `src/train/dmd.py` | §5.6, §5.6.1 | Not started |
| `src/train/local_residual.py` | §3.10, §5.7 | Not started |
| `src/weather/model.py` | §7, §8.4, §8.5 | Not started |
| `src/weather/simulate.py` | §5.8, §8.4 | ✓ Exists (extend) |
| `src/weather/validate.py` | §6 | ✓ Exists (extend) |
| `notebooks/test_pipeline.ipynb` | §8.6 | Not started |
| `notebooks/verify_ocean_grid_orientation.ipynb` | §2.7 | Not started |

---

## Python Standards (Mandatory)

All code in `src/` must comply with these PEPs.

| PEP | Topic | Rule |
|---|---|---|
| **8** | Style | Line length ≤ 100 (matches `pyproject.toml`, `[tool.black] line-length`). |
| **257** | Docstrings | Every public function, class, and module gets a docstring. Numpy-style. Cite the handoff section: `"""... (see docs/model_design.md §5.4)."""`. |
| **484 / 526** | Type hints | Annotate all function signatures, dataclass fields, and module-level constants. Enforced by `mypy --strict`. |
| **585** | Generic types | Use built-in generics: `list[str]`, `dict[str, np.ndarray]`, `tuple[int, ...]`. Never import `List`, `Dict`, `Tuple` from `typing`. |
| **604** | Union syntax | Use `X \| Y` and `X \| None`, never `Union[X, Y]` or `Optional[X]`. |
| **612** | ParamSpec | Use `ParamSpec` / `Concatenate` when wrapping callables (decorators, higher-order functions). |
| **647** | TypeGuard | Use `TypeGuard` (or `TypeIs` from PEP 742) for narrowing functions instead of bare `isinstance` chains. |
| **695** | Type parameters | Use the `type` statement and PEP 695 generic syntax (`def f[T](x: T) -> T`) — cleaner than `TypeVar` for new code. |
| **696** | Type parameter defaults | Use where sensible for library-facing generics. |
| **649** | Deferred annotations | Default in Python 3.14 — annotations are lazily evaluated. `from __future__ import annotations` is redundant; **omit it**. |

**Minimum Python version:** 3.14 (see `pyproject.toml` `[tool.poetry.dependencies] python = "^3.14"`). CI runs only 3.14 until 3.15 ships.

---

## Code Style Rules (Mandatory)

### Prefer functions over classes

Write standalone functions as the default unit of logic. Use a class only
when:

1. It carries substantial shared state across multiple method calls (like
   `WeatherModel`, which holds fitted preprocessing Pipeline + Tucker + DMD +
   OU artifacts).
2. It must conform to an external contract (sklearn transformers must be
   classes because sklearn's Pipeline calls `.fit()` / `.transform()`).

Dataclasses (`@dataclass(frozen=True, slots=True)`) are the right choice for
structured data.

```python
# Good — pure function returning a frozen dataclass
def dmd_fit(
    core: np.ndarray,
    r_dmd: int,
    dt: float,
    rng: np.random.Generator,
) -> DMDFit:
    """Fit exact DMD, per Tu et al. 2014 (see docs/model_design.md §5.6)."""
    ...

@dataclass(frozen=True, slots=True)
class DMDFit:
    Phi: np.ndarray
    Lambda: np.ndarray
    b0: np.ndarray
    lam_cont: np.ndarray
    dt: float

# Good — sklearn transformer (external contract)
class ClimatologyRemover(BaseEstimator, TransformerMixin):
    """Remove per-cell harmonic climatology (see docs/model_design.md §5.4)."""
    def __init__(self, diurnal_variables: list[str] | None = None) -> None:
        self.diurnal_variables = diurnal_variables
    def fit(self, X: xr.Dataset, y: None = None) -> Self: ...
    def transform(self, X: xr.Dataset) -> xr.Dataset: ...
    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset: ...

# Avoid — class wrapping a single stateless action
class ClimatologyFitter:
    def __init__(self, include_diurnal: bool = False): ...
    def run(self, field, t_days): ...
```

### Sampling is free functions, not methods

The `sample`, `sample_batch`, and `simulate_field` API lives in
`weather/simulate.py` as free functions taking `WeatherModel` as the first
argument. `WeatherModel` is a passive container of fitted state; the
sampling algorithms don't mutate it, so they shouldn't be methods.

```python
# Good
from weather import WeatherModel, sample_batch
model = WeatherModel.load(Path("./models/v1"))
values = sample_batch(model, lats, lons, times, ["sigwh"], seed=42)

# Avoid — do not add these as methods on WeatherModel
values = model.sample_batch(lats, lons, times, ["sigwh"], seed=42)
```

### Small changes over large refactors

- Implement only what the current task requires.
- Do not rename, restructure, or "improve" existing working code unless
  explicitly asked.
- If new code needs a helper that overlaps with existing logic, import the
  existing helper rather than rewriting it.

### Formatting and linting

- **Formatter:** `black`. Configured in `pyproject.toml` (`[tool.black]`).
  Line length 100, target `py314`.
- **Import sort:** `isort` with `profile = "black"`. Configured in
  `pyproject.toml` (`[tool.isort]`).
- **Type checker:** `mypy --strict`. Configured in `pyproject.toml`
  (`[tool.mypy]`).
- Run `black src/ tests/ && isort src/ tests/` before committing (or use
  editor integration).
- Do **not** use `ruff`. The team standardizes on black + isort.

### No unnecessary abstractions

- No generic base classes "for future extension".
- No factory functions wrapping a single constructor.
- No `utils.py` or `helpers.py` catch-alls. Put helpers next to the code that
  uses them, or in a domain-named module.
- No configuration objects for stateless computations — pass the arguments.

---

## Dependency Management (Mandatory)

### Source of truth: `pyproject.toml`, managed by Poetry

- **Always** check `pyproject.toml` before importing a package.
- **Never** install a dependency unless explicitly asked by the user.
- **Never** use `pip install` directly. All dependencies go through
  `poetry add <package>` (or manual edit of `pyproject.toml` followed by
  `poetry lock`).
- **Never** use `conda` or create virtual environments manually. Poetry
  manages the venv.
- Do **not** switch to hatchling, PDM, uv, or setup.py. The team's other
  repos use Poetry — consistency wins.

### Available runtime dependencies (from `pyproject.toml`)

```
numpy>=2.0        scipy>=1.14        scikit-learn>=1.5
xarray>=2024.10   pandas>=2.2        pydantic-settings>=2.0
```

### Available dev dependencies

```
pytest>=8         pytest-mock        pytest-cov         pytest-xdist
matplotlib>=3.9   black              isort              mypy>=1.13
ipykernel         jupyter
```

If a task needs a package not listed above, **stop and ask** — do not add it.
In particular:

- Do **not** add `dask` unless out-of-core computation becomes necessary and
  chunking with `numpy.memmap` has been tried first.
- Do **not** add `torch`, `jax`, or any deep learning framework. This model
  is pure `numpy`/`scipy`/`sklearn`.
- Do **not** add HDF5/Zarr for artifact I/O — `.npz` + pickled sklearn
  Pipeline is intentional per handoff §7.
- Do **not** add `ruff`, `flake8`, or `pyflakes`. Black + isort + mypy is the
  toolchain.

---

## Testing Protocol (Mandatory)

### Framework

- **pytest** is the only test runner. Never `unittest.TestCase`.
- **pytest-mock** (`mocker` fixture) is the mocking library. Prefer it over
  `monkeypatch.setattr` and `unittest.mock.patch`.
- Tests live under `tests/` mirroring `src/`:
  `src/train/dmd.py` → `tests/train/test_dmd.py`.

### Rules

| Rule | Detail |
|---|---|
| Fixtures over setup | Use `@pytest.fixture` for shared state (synthetic archives, small trained models). No `setUp`/`tearDown`. |
| Names | `test_<function_or_class>_<scenario>` — e.g. `test_decode_compass_code_wad_origin_one`, `test_dmd_recovers_traveling_wave`, `test_climatology_remover_inverse_transform`. |
| Mocking | Use `mocker.patch(...)`. Do not use `monkeypatch` or `unittest.mock`. |
| Patches | Only create patches when **explicitly asked**. Default to dependency injection or fixture-based fakes. |
| Assertions | Use plain `assert` for scalars. Use `numpy.testing.assert_allclose(actual, expected, rtol=..., atol=...)` for arrays. Never `assert (a == b).all()` for floats. |
| Randomness | Every random operation takes `rng: np.random.Generator` or `seed: int`. Test fixtures use `np.random.default_rng(0)`. |
| sklearn compliance | Every transformer must be tested for `fit → transform → inverse_transform` round-trip. Use `sklearn.utils.estimator_checks.check_estimator` where it applies; skip its checks that assume `X` is a 2D array (we use xarray Datasets). |
| Coverage | ≥ 85% on `src/`. Public API surface must be 100%. |
| No network | Tests must not read from S3 or make HTTP calls. Synthesize archives in `conftest.py`. |
| No large fixtures | Test archives are small (~10 timesteps, ~50×80 spatial patch). Full-scale runs happen in `tests/weather/test_integration.py`, opt-in via `-m integration`. |

### Example — transformer round-trip

```python
# tests/train/test_directional.py
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose
from train.directional import ComingFromToUV

def test_coming_from_to_uv_inverse_transform(small_wind_dataset):
    """(speed, θ_from) → (u, v) → (speed', θ_from') must reproduce input."""
    transformer = ComingFromToUV(
        pairs=[("was", "wad_come_from", "u_wind", "v_wind")],
    )
    Z = transformer.fit_transform(small_wind_dataset)
    X_recovered = transformer.inverse_transform(Z)
    assert_allclose(
        X_recovered["was"].values,
        small_wind_dataset["was"].values,
        atol=1e-10,
    )
    assert_allclose(
        X_recovered["wad_come_from"].values,
        small_wind_dataset["wad_come_from"].values,
        atol=1e-10,
    )
```

### Example — pure function property test

```python
# tests/train/test_dmd.py
import numpy as np
from train.dmd import dmd_fit

def test_dmd_recovers_traveling_wave_frequency():
    """Synthetic A cos(kx - ωt) yields a mode with the correct ω."""
    rng = np.random.default_rng(0)
    T, N = 200, 64
    k, omega = 0.3, 0.7
    x = np.arange(N)
    t = np.arange(T) * 0.1
    core = np.cos(k * x[None, :] - omega * t[:, None])[:, :, None]  # (T, N, 1)
    fit = dmd_fit(core, r_dmd=10, dt=0.1, rng=rng)
    recovered_omegas = np.abs(fit.lam_cont.imag)
    assert np.min(np.abs(recovered_omegas - omega)) < 0.02
```

---

## Module Layout

Follow this directory structure. Create files only as their tasks require.
Do **not** scaffold empty files ahead of time.

```
weather_generator/
├── AGENTS.md                            # This file
├── pyproject.toml                       # Poetry config + tool config
├── poetry.lock                          # Committed
├── README.md
├── CHANGELOG.md
├── docs/
│   └── model_design.md                  # Full design handoff (source of truth)
├── notebooks/
│   ├── test_pipeline.ipynb              # Training entry point (see handoff §8.6)
│   ├── verify_direction_convention.ipynb      # coming_from/going_to check (§2.3, §2.4)
│   └── verify_ocean_grid_orientation.ipynb    # Ocean grid land-correlation study (§2.7)
├── src/
│   ├── core/                            # Shared primitives — no internal deps
│   │   ├── __init__.py
│   │   ├── config.py                    # Grid constants (Final), Settings
│   │   └── io.py                        # snapshot_path, load_snapshot,
│   │                                    # decode_compass_code
│   ├── train/                           # Training pipeline
│   │   ├── __init__.py
│   │   ├── ingest.py                    # load_training_window (xarray)
│   │   ├── directional.py               # ComingFromToUV, GoingToToUV transformers
│   │   ├── reproject.py                 # Reprojector transformer
│   │   ├── climatology.py               # ClimatologyRemover transformer
│   │   ├── standardize.py               # PerCellStandardizer transformer
│   │   ├── copula.py                    # GaussianCopula transformer
│   │   ├── tucker.py                    # tucker_fit, tucker_transform,
│   │   │                                # tucker_inverse (pure functions)
│   │   ├── dmd.py                       # dmd_fit, fit_mode_ou (pure functions)
│   │   └── local_residual.py            # fit_local_residual (pure function)
│   └── weather/                         # Public API
│       ├── __init__.py                  # Re-exports (see handoff §8.4)
│       ├── model.py                     # WeatherModel(BaseEstimator, TransformerMixin)
│       ├── simulate.py                  # sample, sample_batch, simulate_field
│       └── validate.py                  # full_report + per-check helpers
├── tests/
│   ├── conftest.py                      # Fixtures: synthetic archives, small models
│   ├── core/
│   ├── train/
│   └── weather/
└── .github/
    └── workflows/
        └── ci.yml                       # black + isort + mypy + pytest
```

`data/` and `models/` are `.gitignore`d — sample archives and trained models
never live in git.

---

## Existing Code

Two modules already exist. Extend rather than replace them.

- **`src/core/config.py`** — will hold grid constants, `Settings` for
  env-tunable paths. Extend as new constants are needed (e.g.
  `OCEAN_GRID_LAT_AT_ROW0` after the §2.7 verification).
- **`src/core/io.py`** — will hold `snapshot_path`, `load_snapshot`,
  `decode_compass_code`. Reuse these from every downstream module; do not
  reimplement.
- **`src/weather/simulate.py`** — placeholder. Fill with `sample`,
  `sample_batch`, `simulate_field` per handoff §5.8, §8.4.
- **`src/weather/validate.py`** — placeholder. Fill with `full_report` and
  per-check helpers per handoff §6.

`WeatherModel` in `src/weather/model.py` does not exist yet. Create it per
handoff §8.5.

**Bootstrapping order:** implement modules top-to-bottom in the Task Mapping
table. Each module's tests must pass before starting the next. In
particular:

- `core/` must be green before `train/`.
- All `train/` preprocessing transformers must be green before
  `train/tucker.py` (Tucker consumes their output).
- `train/tucker.py`, `dmd.py`, `local_residual.py` must be green before
  `weather/model.py`.
- `weather/model.py` must be green before `weather/simulate.py` and
  `weather/validate.py`.

---

## Sklearn Integration Rules (Mandatory)

The preprocessing transformers must be sklearn-compatible, and `WeatherModel`
must be sklearn-compatible. Get these details right or `Pipeline` composition
and `clone()` will break.

### For every transformer in `src/train/`

- Inherit `from sklearn.base import BaseEstimator, TransformerMixin`.
- `__init__` stores its parameters *verbatim* on `self` under the same name.
  No validation, no coercion, no side effects. `sklearn.base.clone` reads
  them via `get_params()`.
- Validation happens lazily in `fit`.
- Fitted attributes end in `_` (trailing underscore). `sklearn.utils.
  validation.check_is_fitted(estimator)` looks for these.
- Every transformer implements `fit`, `transform`, and (where meaningful)
  `inverse_transform`. Preprocessing stages all have meaningful inverses.
- `fit_transform` is inherited from `TransformerMixin` — override only if
  there's a computation to reuse between `fit` and `transform` on the same
  `X`.
- Transformers pass `xarray.Dataset` through the Pipeline, not 2D numpy
  arrays. This is unusual but valid: sklearn's `Pipeline` only requires each
  stage's output be a valid input to the next. It does not enforce the
  `(n_samples, n_features)` convention.
- `check_estimator` will fail on our transformers (it assumes 2D arrays).
  That's expected and acceptable. We inherit sklearn's mixins for their
  behavior, not for conformance to the tabular-data contract.

### For `WeatherModel`

- Inherits `BaseEstimator + TransformerMixin`.
- `__init__` stores hyperparameters verbatim. RNG materializes in `fit` via
  `np.random.default_rng(self.random_state)` stored on `self._rng`.
- `fit(X, y=None) -> Self`, `transform(X) -> xr.Dataset`, `fit_transform(X, y=None) -> xr.Dataset`.
- `transform(X)` runs *only* the preprocessing chain — Tucker/DMD/OU are not
  transformers and do not participate in `.transform()`.
- Custom methods: `save(model_dir)`, `@classmethod load(model_dir)`,
  `@classmethod from_components(...)`. See handoff §8.5.

### Persistence

- The fitted preprocessing Pipeline is persisted via `joblib.dump` (pickle
  under the hood) as `preprocess.pkl`.
- Tucker/DMD/OU fits are persisted as `.npz` bundles (pure numpy arrays; no
  class structure needed).
- `metadata.json` records sklearn/numpy/scipy versions used at fit time.
  `WeatherModel.load` warns if the runtime differs.

---

## Numerical & Scientific Safety Rules

These are the domain-specific rules that catch silently-wrong outputs. A
violation may pass every unit test and still produce wrong samples.

### NaN discipline

- **Never coerce NaN to zero.** A NaN means "land / no data / masked"; a real
  zero (calm sea, no swell, no current) is a physically meaningful value
  that must not be confused with land.
- Every reduction over a spatial or temporal axis must use its NaN-aware
  variant (`np.nanmean`, `np.nanstd`, etc.) or explicitly guard with
  `np.isnan`.
- Test explicitly: all-NaN input → NaN output; single-valid-point input →
  correct scalar. See handoff §2.5.

### Dtype discipline

- Every `np.zeros`, `np.empty`, `np.array`, `np.full` must specify `dtype=`
  explicitly.
- `float16` is a **storage-only** dtype. Upcast to `float64` at the I/O
  boundary (`arr.astype(np.float64)`) and never do arithmetic on `float16`
  arrays. `np.float16(300) ** 2` returns `inf`.
- `int8` compass codes are decoded via `core.io.decode_compass_code(codes,
  origin_code)` — see handoff §2.3. Never manually decode inline; the
  origin-code difference between `wsd`/`pwd` (origin 0) and `wad` (origin 1)
  is a common bug source.

### Time and coordinate discipline

- **All timestamps UTC.** Every `datetime` in the public API must have
  `tzinfo=timezone.utc`. Naive datetimes raise `ValueError` at the API
  boundary — do not silently attach UTC.
- **Row 0 = south** for both wave and wind grids. See handoff §2.7 for the
  empirical derivation. Any code that geo-references a grid must apply the
  formulas from §2.2; never assume "row 0 = north" defaults from other
  earth-science tooling.
- **Longitude wraparound** at 359.75° → 0° must be handled by periodic
  padding (`pad_lon`) before any spatial interpolation. See handoff §2.6.
- **Direction convention is mixed across the archive.** Most directional
  fields (`wsd`, `pwd`, `pswd`, `wad_come_from`) are meteorological "coming
  from", CW from N; `wad` and `ocd` are "going to". See handoff §2.4.
  Directional interpolation must always go through `(u, v)` — angle-space
  interpolation across the 0°/360° discontinuity is broken. Enforce by
  running `ComingFromToUV` and `GoingToToUV` **before** `Reprojector` in the
  Pipeline.

### Randomness discipline

- Every function that generates randomness takes `rng: np.random.Generator |
  None = None` or `seed: int | None = None`. Sklearn-compatible components
  take `random_state: int | None` in `__init__` and materialize `_rng =
  np.random.default_rng(self.random_state)` in `fit`.
- **Never touch the global `np.random`.**
- Deterministic replay is a hard requirement: same seed + same query → byte-
  exact output. Tested in
  `tests/weather/test_simulate.py::test_seed_reproducibility`.

### Physical consistency (documented limitation)

- The Gaussian copula preserves per-cell marginals and cross-cell
  correlations but does **not** enforce physical relationships (e.g.
  `wsh ≤ sigwh`, wave steepness bounds). Samples may violate these. Do not
  add post-hoc projection layers to enforce them without explicit
  authorization — the fix belongs in the copula (t-copula or GPD tails, per
  handoff §3.13), not in a downstream patch.

### BLAS threading

- `sample_batch` must not spawn its own workers. Let the underlying BLAS
  thread; honor `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
  `MKL_NUM_THREADS` set by the caller.
- Explicit parallelism (over modes, seeds, or independent snapshots) uses
  `joblib.Parallel(n_jobs=-1, backend="loky")`. joblib is already a sklearn
  dependency; no extra install needed. Never bare `multiprocessing.Pool` or
  `concurrent.futures` for numeric code.

---

## Environment Configuration

Grid constants are `Final` module-level bindings in `core/config.py`:

```python
# src/core/config.py
from typing import Final

WAVE_GRID_SHAPE: Final[tuple[int, int]] = (621, 1440)
WIND_GRID_SHAPE: Final[tuple[int, int]] = (721, 1440)
OCEAN_GRID_SHAPE: Final[tuple[int, int]] = (1121, 2880)

WAVE_GRID_LAT_AT_ROW0: Final[float] = -75.0
WIND_GRID_LAT_AT_ROW0: Final[float] = -90.0
# OCEAN_GRID_LAT_AT_ROW0: see handoff §2.7 — verify empirically before setting.

WAVE_GRID_LAT_STEP_DEG: Final[float] = 0.25
WIND_GRID_LAT_STEP_DEG: Final[float] = 0.25
OCEAN_GRID_LAT_STEP_DEG: Final[float] = 0.125
GRID_LON_STEP_DEG: Final[float] = 0.25   # wave and wind; ocean is 0.125
```

Env-tunable settings (paths, tunable defaults) go in `Settings`:

```python
# src/core/config.py (settings section)
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")

    default_r_ocean: int = 300
    default_r_var: int = 6
    default_r_dmd: int = 100
    copula_lut_bins: int = 500

    model_config = {"env_prefix": "WGEN_"}
```

---

## Notebook Protocol

Training happens in `notebooks/test_pipeline.ipynb`. There is no CLI. See
handoff §8.6 for the intended cell structure.

**Requirements for the notebook:**

- Every cell runnable in order from a fresh kernel restart.
- No hidden mutable state between cells beyond the RNG and `Path` variables.
- Reproducibility: the notebook takes a `SEED` constant at the top; every
  RNG derived from it.
- Caching: use `sklearn.pipeline.Pipeline(memory=Path("./cache"))` during
  iterative development so restarting the kernel doesn't force a full re-run
  of ingest → Gaussianization. Turn caching off (or delete the cache dir)
  before saving a "production" model.
- Convert the notebook to a script only if you find yourself running the
  same cells repeatedly — but even then, keep it under `notebooks/` as a
  `.py` file, not a CLI entry point.

---

## Iteration Protocol

At the end of each implementation task:

1. Summarise files created or changed.
2. List passing / failing tests, with `pytest -v` output for anything new.
3. Report `black --check`, `isort --check-only`, and `mypy --strict src/`
   status — all three must be clean.
4. Identify the next module in the Task Mapping table.
5. Flag any blockers, ambiguities in the handoff, or missing dependencies.

If a task reveals a gap in the handoff, **update the handoff first**
(`docs/model_design.md`), then implement. Do not silently deviate.

---

## What NOT to Do

- Do not scaffold empty files or placeholder modules.
- Do not add features beyond the current task scope.
- Do not add a CLI. Training is notebook-driven; sampling is programmatic.
- Do not add methods `sample`/`sample_batch`/`simulate_field` on
  `WeatherModel` — they're free functions in `weather.simulate`.
- Do not import from `weather` inside `train/` or `core/`. Package
  dependencies flow one way.
- Do not add comments that restate the code (`# increment counter`).
- Do not add error handling for scenarios that cannot occur at the current
  call site.
- Do not create patches unless explicitly asked.
- Do not import from `typing` what Python 3.14 builtins or PEPs provide
  (`list`, `dict`, `tuple`, `type`, `X | Y`, `X | None`).
- Do not add `from __future__ import annotations` — PEP 649 makes
  annotations lazy by default in 3.14.
- Do not use `unittest` for any purpose.
- Do not use `ruff`, `flake8`, or `pyflakes` — this repo uses `black + isort
  + mypy`.
- Do not switch from Poetry to hatchling, PDM, uv, or setup.py.
- Do not install packages. Check `pyproject.toml` first, then ask.
- Do not add a C++ query library. If a real consumer profiles the query path
  and finds batched Python insufficient, revisit — but not before.
- Do not touch the raw archive (`./data/`) from tests. Synthesize.
- Do not swallow NaN. Never `.fillna(0)` on weather variables.
- Do not force Tucker/DMD/OU into sklearn transformers. They're pure
  functions returning frozen dataclasses; that shape is intentional.
- Do not validate inputs in `__init__` of a sklearn-compatible class.
  `sklearn.base.clone` will break. Validate lazily in `fit`.
