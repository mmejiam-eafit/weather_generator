# AGENTS.md — Source Code Agent Protocol

## Mission

Implement the stochastic weather generator defined in
[`docs/model_design.md`](docs/model_design.md). Train on the raw 3-hourly NOAA
GFS-Wave / WAVEWATCH III + GFS wind archive; produce plausible synthetic
samples of any variable at any continuous `(lat, lon, datetime)`.

All production code lives under `src/` split across three sibling packages:
`core/`, `train/`, `weather/`. Tests under `tests/`. Open design questions are
resolved as numbered, notebook-backed tasks under `tasks/` (index:
[`tasks/README.md`](tasks/README.md)).

There is **no CLI** — training is notebook-driven, and query is programmatic.
The intended training entry point, `notebooks/test_pipeline.ipynb` (§8.6), is
currently parked in `notebooks/tbd/` with the deprecated v1→v2 migration set.

---

## Architecture Reference

The canonical design is in
[`docs/model_design.md`](docs/model_design.md). Read the relevant section
before implementing any module — it is the source of truth for data
conventions, algorithm, and API shape, subject to the precedence below.

### Which document wins

When documents disagree, the higher entry wins:

1. [`docs/noaa_etl/`](docs/noaa_etl/) — the archive's origin and
   transformations. `model_design.md` defers to it explicitly.
2. **DECIDED** entries in [`docs/open_questions.md`](docs/open_questions.md)
   and the findings recorded in `tasks/NN-*.md`. They supersede
   `model_design.md` where they conflict (e.g. §4.1, Gaussianization
   placement — see below).
3. [`docs/model_design.md`](docs/model_design.md) — design, algorithm, API.
   The topic documents (`docs/variable_taxonomy.md`,
   `docs/tucker_decomposition.md`, `docs/correlated_variables.md`, …) go
   deeper on one stage each; if one disagrees with `model_design.md`, flag
   it rather than picking a side silently.

The root-level `handoff.md`, `physical_handoff.md`,
`stochastic_weather_generator_handoff.md` and `context.md` are historical
snapshots, not sources of truth.

### Package dependency rules

Dependencies flow one way. No test enforces this yet — hold the line in
review.

- `core/` — no internal deps. Imports only stdlib, numpy, xarray,
  pydantic-settings, structlog.
- `train/` — depends on `core`. May import from numpy, scipy, scikit-learn,
  xarray, tensorly, pydmd. **Does not** import from `weather`.
- `weather/` — public API. Depends on `core` and `train`. Consumers should
  import only from here.

### Two-layer training architecture

- **Preprocessing** (`train/masking.py` — `PartitionSentinelFiller`,
  `LogTransform` — plus `train/directional.py` — `ComingFromToUV` and
  `GoingToToUV`, §3.7 — `train/reproject.py`, `train/climatology.py`,
  `train/standardize.py`) — seven scikit-learn transformer classes chained
  via `sklearn.pipeline.Pipeline` (`weather.model.build_preprocess`, §5.3),
  ending at the standardized anomaly, one `(T, lat, lon)` tensor per field.
  Each has a real `fit`, `transform`, and `inverse_transform`.
- **Decomposition + stochastic model** (`train/tucker.py`, `train/dmd.py`) —
  pure functions taking arrays + hyperparameters, returning frozen
  dataclasses (`TuckerFit`, `DMDFit`, `ModeOUFit`). Not transformers.
  Per tensor: masked Tucker, then a `train/copula.py` `GaussianCopula` fit on
  the Tucker core (placement ii, §5.5.1) — a transformer, but applied per
  core, not as a Pipeline stage. Per variable group: DMD and the mode OU on
  the joint state. `train/local_residual.py` (`LocalResidualFit`) is
  deferred in v3 (§5.7) and not wired in. Alongside it, the **group dynamics
  diagnostics** stage (`train/linear_inverse.py`, §5.6.2): `diagnose_group`
  runs once per variable group on the Gaussianized cores DMD consumes, and
  returns a `GroupDiagnostics`. It holds a tau test of the linear-Markov
  assumption and the coupling information an independent fit loses, each
  judged against a bootstrap envelope. Results are stored on the model and
  logged; they never feed sampling or gate the fit.

`weather.model.WeatherModel` inherits `BaseEstimator + TransformerMixin` and
glues the two layers together. Sampling operations are **free functions** in
`weather.simulate`, not methods on `WeatherModel`.

See handoff §3.12, §3.14, and §4 for the full rationale.

**Gaussianization placement (ii) is implemented.** Tucker runs on the
(log-energy) standardized anomalies and the copula Gaussianizes each Tucker
core before DMD (**DECIDED**, `docs/open_questions.md` §4.1; task 07 fixed
Tucker rank 40% and `r_dmd = 100` under it). Placement (i), the copula as
the last Pipeline stage, is gone from `src/`.

---

## Task Mapping

Each module maps to a section of the design docs. Always read the section
before changing the module. Section numbers without a file name refer to
`docs/model_design.md`.

| Module | Spec | Status |
|---|---|---|
| `src/core/config.py` | §2.2, §2.7, §2.8 | ✓ Exists |
| `src/core/logger.py` | AGENTS.md *Logging* | ✓ Exists |
| `src/core/io.py` | §2.1, §2.3, §5.1 | ✓ Exists |
| `src/core/archive_eras.py` | §2.2.1, §5.1 | ✓ Exists |
| `src/train/ingest.py` | §2.5, §5.1 | ✓ Exists |
| `src/train/directional.py` | §2.4, §3.7, §5.3 | ✓ Exists |
| `src/train/reproject.py` | §2.6, §3.6, §5.2 | ✓ Exists |
| `src/train/climatology.py` | §3.9, §5.4 | ✓ Exists |
| `src/train/standardize.py` | §5.4 | ✓ Exists |
| `src/train/copula.py` | §5.4 | ✓ Exists |
| `src/train/tucker.py` | §5.5 | ✓ Exists |
| `src/train/masked_tucker.py` | §5.5; `tasks/07-tucker-dmd-rank-selection.md` §5 | ✓ Exists — the ALS behind `train/tucker.py` |
| `src/train/masking.py` | §5.1, §5.3.1; `docs/variable_taxonomy.md` §3.3.1; task 17 | ✓ Exists |
| `src/train/dmd.py` | §5.6, §5.6.1 | ✓ Exists |
| `src/train/linear_inverse.py` | §5.6.2; `docs/correlated_variables.md` §4.4 | ✓ Exists |
| `src/train/local_residual.py` | §3.10, §5.7 | ✓ Exists — deferred in v3, not wired into `WeatherModel` |
| `src/weather/model.py` | §7, §8.4, §8.5 | ✓ Exists |
| `src/weather/simulate.py` | §5.8, §8.4 | ✓ Exists |
| `src/weather/validate.py` | §6 | ✓ Exists |
| `notebooks/test_pipeline.ipynb` | §8.6 | Parked in `notebooks/tbd/` |
| `notebooks/verify_ocean_grid_orientation.ipynb` | §2.7 | Superseded — orientation settled by direct lookup (`docs/open_questions.md` §1.1, task 16) |

---

## Python Standards (Mandatory)

All code in `src/` must comply with these PEPs.

| PEP | Topic | Rule |
|---|---|---|
| **8** | Style | Line length ≤ 100 (matches `pyproject.toml`, `[tool.black] line-length`). `snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants; math names (`Phi`, `Z`, `X`) are fine where they mirror the design doc's formulas. |
| **257** | Docstrings | Every public function, class, and module gets a docstring. Numpy-style. Cite the handoff section: `"""... (see docs/model_design.md §5.4)."""`. |
| **484 / 526** | Type hints | Annotate all function signatures, dataclass fields, and module-level constants. Enforced by `mypy --strict`. |
| **585** | Generic types | Use built-in generics: `list[str]`, `dict[str, np.ndarray]`, `tuple[int, ...]`. Never import `List`, `Dict`, `Tuple` from `typing`. Abstract types come from `collections.abc` (`Callable`, `Iterable`, `Iterator`, `Sequence`, `Mapping`), never from `typing`. |
| **604** | Union syntax | Use `X \| Y` and `X \| None`, never `Union[X, Y]` or `Optional[X]`. |
| **612** | ParamSpec | Use `ParamSpec` / `Concatenate` when wrapping callables (decorators, higher-order functions). |
| **647** | TypeGuard | Use `TypeGuard` (or `TypeIs` from PEP 742) for narrowing functions instead of bare `isinstance` chains. |
| **695** | Type parameters | Use the `type` statement for aliases (`type GridShape = tuple[int, int]`) and PEP 695 generic syntax (`def f[T](x: T) -> T`, `class C[T]:`). Never `TypeVar` / `Generic` / `TypeAlias` or a bare alias assignment. |
| **696** | Type parameter defaults | Use where sensible for library-facing generics. |
| **649 / 749** | Deferred annotations | Default in Python 3.14 — annotations are lazily evaluated. `from __future__ import annotations` is redundant; **omit it**. Introspect annotations with `annotationlib.get_annotations()`, not `__annotations__`. |

`typing` imports that remain legitimate (no builtin equivalent): `Final`,
`Literal`, `Self`, `Any`, `cast`, `TypeIs`, `TypeGuard`, `ParamSpec`,
`Concatenate`.

No mutable module-level state. Module-level bindings are constants,
annotated `Final` (as in `core/config.py`).

```python
# Correct
from collections.abc import Callable

type GridShape = tuple[int, int]


def first_valid[T](items: list[T | None]) -> T | None:
    return next((item for item in items if item is not None), None)


def reduce_cells(field: np.ndarray, fn: Callable[[np.ndarray], float]) -> np.ndarray: ...


# Incorrect
from typing import Callable, Optional, TypeVar

T = TypeVar("T")
GridShape = tuple[int, int]


def first_valid(items: list[Optional[T]]) -> Optional[T]: ...
```

### Python 3.14 language and runtime features

**Python version:** `requires-python = ">=3.14,<3.15"` in `pyproject.toml`.
CI runs only 3.14 until 3.15 ships.

| Feature | PEP | Rule |
|---|---|---|
| Deferred annotations, `annotationlib` | 649 / 749 | Use (see table above). |
| `except A, B:` without parentheses | 758 | **Formatter-enforced.** Black (`target-version = ["py314"]`) rewrites `except (A, B):` to `except A, B:` — see `src/weather/model.py`. Parentheses stay required when binding with `as`: `except (A, B) as exc:`. Don't fight the formatter. |
| `return` / `break` / `continue` in `finally` | 765 | **Banned.** It silently swallows the in-flight exception; 3.14 emits a `SyntaxWarning`. |
| Sub-interpreters (`concurrent.interpreters`, `InterpreterPoolExecutor`) | 734 | **Don't use.** numpy and scipy don't support sub-interpreters. Explicit parallelism is `joblib` loky (see *BLAS threading*). |
| Free-threaded (no-GIL) build | 779 | **Don't rely on it.** The numeric stack and its BLAS threading are validated on the default build only. |
| Template strings `t"..."` | 750 | **Don't introduce.** No templating use case here; f-strings suffice. |
| `compression.zstd` | 784 | **Not for artifacts.** `.npz` + joblib pickle is intentional (§7). |

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

### Single responsibility per function

A function does one job. Don't fuse a computation, a validation pass, and an
unrelated side effect (emitting a diagnostic, writing a checkpoint) into one
body — extract the distinct concerns into separately named functions. A
docstring that needs several unrelated paragraphs to describe one function
is a sign it should be split. This complements *No unnecessary
abstractions*: that rule guards against indirection nobody needs; this one
guards against fusing concerns that are genuinely different.

### Explain non-trivial approaches

When code takes a non-obvious route — a workaround for a library bug, a
tolerance chosen from data, a deliberate deviation from the textbook
algorithm or from `docs/model_design.md` — its docstring or an inline
comment must say **why**, and **under what caveats** the choice holds.
Examples already in the tree: `train/masked_tucker.py` (the vendored
tensorly masked-ALS fixes, and why each is safe) and the `wad` consistency
tolerances in `train/ingest.py`. This is the complement of *no comments
that restate the code*: explain reasoning, never mechanics.

### Formatting and linting

- **Formatter:** `black`. Configured in `pyproject.toml` (`[tool.black]`).
  Line length 100, target `py314`.
- **Import sort:** `isort` with `profile = "black"`. Configured in
  `pyproject.toml` (`[tool.isort]`).
- **Type checker:** `mypy`, strict mode (`strict = true` in `[tool.mypy]`).
  CI checks `src` and `tests`.
- Run `poetry run black src tests && poetry run isort src tests` before
  committing (or use editor integration).
- Do **not** use `ruff`. The team standardizes on black + isort.

### No unnecessary abstractions

- No generic base classes "for future extension".
- No factory functions wrapping a single constructor.
- No `utils.py` or `helpers.py` catch-alls. Put helpers next to the code that
  uses them, or in a domain-named module.
- No configuration objects for stateless computations — pass the arguments.

---

## Error Handling

- Catch specific exception types. Never a bare `except:`.
- When translating a low-level failure into a clearer one, chain it:
  `raise ValueError(...) from exc`, so the cause survives. Use `from None`
  only when the original exception adds nothing (as `train/ingest.py` does
  for an unknown variable name's `KeyError`).
- Raise builtin exceptions (`ValueError`, `KeyError`, `FileNotFoundError`)
  whose message names the offending value and, where one exists, the doc
  section that defines the valid ones. No custom exception hierarchy until
  a caller needs to catch one failure type distinctly.
- Never suppress an exception with `return`, `break` or `continue` in a
  `finally` block (PEP 765).
- `warnings.warn` is for conditions the caller should know about but that
  don't invalidate the result (e.g. a long run of missing snapshots in
  `train/ingest.py`).

---

## Logging

- **No new `print` in `src/`.** Diagnostic output goes through one
  `structlog` logger, `from core.logger import logger`, with structured
  key-value pairs instead of an interpolated string:
  `logger.info("wad consistency check passed", time=t.isoformat(), max_dev_deg=max_dev)`.
- Design constraint for `core/logger.py`: `weather` is a library consumed
  programmatically, so it routes through `structlog.stdlib.LoggerFactory`
  and must **not** attach handlers to the root logger at import time, nor
  call `structlog.configure` (both change process-wide state). An
  application may do that; a library that does hijacks its consumer's
  logging. The training notebook or script configures output, e.g.
  `logging.basicConfig(level=logging.INFO)`. Tests read records with
  pytest's `caplog` at `core.logger.LOGGER_NAME`.
- `warnings.warn` stays the tool for caller-actionable conditions (see
  *Error Handling*).
- Logging is on the training path only (ingest checks, period sentinels,
  trend significance, Tucker errors, group diagnostics). The query path
  (`weather/simulate.py`) logs nothing and must stay that way.

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

### Available runtime dependencies (from `pyproject.toml` `[project]`)

```
numpy>=2.0              scipy>=1.14             scikit-learn>=1.5
xarray>=2024.10         pandas>=3.0.3,<4        pydantic-settings>=2.0
tensorly>=0.9,<0.10     pydmd>=2025.8.1,<2026   seaborn>=0.13.2,<0.14
plotly>=6.9,<7          structlog>=26.1,<27
```

### Available dev, test and lint dependencies

PEP 735 `[dependency-groups]`:

```
dev:   matplotlib>=3.9   ipykernel        jupyter
test:  pytest>=8         pytest-mock      pytest-cov      pytest-xdist
lint:  black             isort            mypy>=1.13
```

Poetry's own `[tool.poetry.group.dev.dependencies]` adds `ipykernel`,
`plotly`, `nbformat`.

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
| Isolation | No shared mutable state between tests and no dependence on execution order (`pytest-xdist` runs them in parallel). |
| Names | `test_<function_or_class>_<scenario>` — e.g. `test_decode_compass_code_wad_origin_one`, `test_dmd_recovers_traveling_wave`, `test_climatology_remover_inverse_transform`. |
| Mocking | Use `mocker.patch(...)`. Do not use `monkeypatch` or `unittest.mock`. |
| Patches | Only create patches when **explicitly asked**. Default to dependency injection or fixture-based fakes. |
| Assertions | Use plain `assert` for scalars. Use `numpy.testing.assert_allclose(actual, expected, rtol=..., atol=...)` for arrays. Never `assert (a == b).all()` for floats. |
| Randomness | Every random operation takes `rng: np.random.Generator` or `seed: int`. Test fixtures use `np.random.default_rng(0)`. |
| sklearn compliance | Every transformer must be tested for `fit → transform → inverse_transform` round-trip. Use `sklearn.utils.estimator_checks.check_estimator` where it applies; skip its checks that assume `X` is a 2D array (we use xarray Datasets). |
| Coverage | ≥ 85% on `src/`. Public API surface must be 100%. |
| No network | Tests must not read from S3 or make HTTP calls. Synthesize archives in `conftest.py`. |
| No large fixtures | Test archives are small (~10 timesteps, ~50×80 spatial patch). Full-scale runs belong in `tests/weather/test_integration.py` (not yet written), opt-in via `-m integration` (marker registered in `pyproject.toml`). |

### Test intent, not implementation

A test that only verifies implementation is worse than no test. Its value
comes from confirming the code does what it is *supposed* to do, not how it
currently happens to do it. A test that asserts internal mechanics — a
private helper was called, an incidental call order — passes even when the
result is wrong, and breaks on harmless refactors.

- Assert on outputs, return values, and observable effects. That includes
  the arguments passed to a collaborator when that call *is* the unit's
  contract (e.g. the array handed to a fit function): a fake that returns a
  canned value can't reveal that the wrong data was forwarded.
- The anti-pattern is narrower than "never assert on mocks": it is checking
  that a call merely *happened* (`mock.assert_called_once()` alone proves
  nothing), or asserting mechanics that don't affect the outcome.
- Property and round-trip assertions are this repo's natural form of
  intent: `inverse_transform(transform(X)) ≈ X`, a known frequency
  recovered by DMD, same seed → byte-identical samples.
- Before writing a test, ask: *if the internals were rewritten but the
  behaviour — including what reaches the dependencies — were preserved,
  would this test still pass?* If not, it is coupled to implementation.

```python
# Correct — asserts the observable outcome
def test_standardizer_unit_variance_after_transform(small_wave_dataset: xr.Dataset) -> None:
    Z = PerCellStandardizer().fit_transform(small_wave_dataset)
    assert_allclose(np.nanstd(Z["sigwh"].values, axis=0), 1.0, atol=1e-12)


# Incorrect — proves np.nanstd ran, not that the result is right
def test_standardizer_calls_nanstd(mocker, small_wave_dataset):
    spy = mocker.spy(np, "nanstd")
    PerCellStandardizer().fit(small_wave_dataset)
    spy.assert_called()  # a reduction over the wrong axis would still pass
```

### When a test fails, fix the code — not the test

A test encodes what the code is *supposed* to do, so a failure is a signal
to investigate, not an invitation to edit the test.

- Compare the assertion against the intended behaviour (design doc section,
  task file, docstring, callers) before touching anything.
- Change the test only if the test itself is wrong (outdated or incorrect
  intent). If it encodes the intent correctly, the fix belongs in the code.
- Never loosen a tolerance or rewrite an expected value just to make a test
  pass against the current implementation. When a tolerance genuinely has
  to change, document why (see *Explain non-trivial approaches*) and record
  it in `CHANGELOG.md`.

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
├── CHANGELOG.md                         # Every change, dated (see Commits)
├── README.md
├── pyproject.toml                       # Poetry config + tool config
├── poetry.lock                          # Committed
├── context.md                           # Orientation snapshot — not a source of truth
├── *handoff*.md                         # Historical handoffs — not sources of truth
├── docs/
│   ├── model_design.md                  # Design handoff (v3): design, algorithm, API
│   ├── open_questions.md                # Open / DECIDED design questions (DECIDED wins)
│   ├── <topic>.md                       # Per-stage deep dives: variable_taxonomy,
│   │                                    # harmonic_climatology, tucker_decomposition,
│   │                                    # dynamic_mode_decomposition, ornstein_uhlenbeck_
│   │                                    # processes, gaussian_copula_and_marginals,
│   │                                    # correlated_variables, physical_consistency_
│   │                                    # constraints
│   └── noaa_etl/                        # Archive origin + transforms (data source of truth)
├── tasks/
│   ├── README.md                        # Task index + status legend
│   ├── NN-<slug>.md                     # One open question per task; findings recorded here
│   └── tbd/                             # Deprecated v1→v2 migration task set
├── notebooks/
│   ├── NN_<slug>.ipynb                  # Task NN experiments
│   ├── NN_<slug>.py                     # Task NN long / unattended runs (scripts, not CLIs)
│   └── tbd/                             # Parked: test_pipeline.ipynb, verify_*.ipynb, …
├── azureml/                             # Cloud experiment tooling (task 18)
├── src/
│   ├── core/                            # Shared primitives — no internal deps
│   │   ├── __init__.py
│   │   ├── config.py                    # Grid constants (Final), canonical dims,
│   │   │                                # VARIABLE_GRID, Settings
│   │   ├── io.py                        # snapshot_path, load_snapshot, load_raw,
│   │   │                                # decode_compass_code, require_utc
│   │   ├── archive_eras.py              # ARCHIVE_ERAS, era_for_timestamp
│   │   └── logger.py                    # logger: the one structlog logger
│   ├── train/                           # Training pipeline
│   │   ├── __init__.py
│   │   ├── ingest.py                    # load_training_window (xarray)
│   │   ├── directional.py               # ComingFromToUV, GoingToToUV transformers
│   │   ├── reproject.py                 # pad_lon, lonlat_interpolator, Reprojector
│   │   ├── climatology.py               # ClimatologyRemover transformer
│   │   ├── standardize.py               # PerCellStandardizer transformer
│   │   ├── copula.py                    # GaussianCopula (fit per Tucker core)
│   │   ├── tucker.py                    # fill_land, tucker_fit, tucker_transform,
│   │   │                                # tucker_inverse (per tensor, pure functions)
│   │   ├── masked_tucker.py             # partial_tucker_lowmem (vendored tensorly fix)
│   │   ├── masking.py                   # Land mask, partition sentinel policy,
│   │   │                                # PartitionSentinelFiller, LogTransform
│   │   ├── dmd.py                       # group_state, dmd_fit, fit_mode_ou,
│   │   │                                # ou_transition, mode_amplitudes
│   │   ├── linear_inverse.py            # group_pcs, fit_lim, tau_test, diagnose_group
│   │   └── local_residual.py            # coherent_reconstruction, fit_local_residual
│   └── weather/                         # Public API
│       ├── __init__.py                  # Re-exports (see handoff §8.4)
│       ├── model.py                     # WeatherModel(BaseEstimator, TransformerMixin),
│       │                                # build_preprocess, persistence (§7)
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

`data/`, `models/`, `results/`, `cache/` and `reports/` are `.gitignore`d —
archives, trained models and experiment outputs never live in git.

---

## Existing Code

Every `src/` module in the Task Mapping table exists. Extend rather than
replace.

- **`src/core/config.py`** — era-3 grid constants, `CANONICAL_DIM_TEMPLATE`
  + `check_canonical`, `VARIABLE_GRID`, `Settings`. Extend as new constants
  are needed.
- **`src/core/io.py`** — `snapshot_path`, `load_snapshot`, `load_raw`,
  `decode_compass_code`, `require_utc`. Reuse these from every downstream
  module; do not reimplement.
- **`src/core/archive_eras.py`** — `ARCHIVE_ERAS`, `era_for_timestamp`. New
  date-aware code goes through `era_for_timestamp` rather than the flat
  era-3 constants in `config.py`.
- **`src/weather/`** — `WeatherModel` (§8.5), `sample` / `sample_batch` /
  `simulate_field` (§5.8, §8.4) and `full_report` (§6), re-exported from
  `weather/__init__.py`.

**Dependency order:** a change to a lower layer must be green before work
continues above it. In particular:

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
- **Canonical dim order is `(time, <grid>_lat, <grid>_lon)`** — time is axis
  0 for every multi-timestep field array (`core.config.CANONICAL_DIM_TEMPLATE`).
  The preprocessing transformers infer the time axis positionally, so a
  wrong order computes per-cell statistics over the wrong axis *without
  raising*. Validate with `core.config.check_canonical` at boundaries.
  Raveled arrays keep their documented layouts: Tucker takes one
  `(T, H, W)` tensor at a time, and a group's DMD state is `(T, n_G)`
  (§5.5, §5.6).
- **Row 0 = south** for the wave, wind and ocean grids. See handoff §2.7 for
  the empirical derivation. Any code that geo-references a grid must apply
  the formulas from §2.2; never assume "row 0 = north" defaults from other
  earth-science tooling. The ocean grid's row spacing is non-uniform in
  degrees, so it has no scalar `lat0 + row * step` formula
  (`docs/open_questions.md` §1.1).
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
  dependency; no extra install needed. Never bare `multiprocessing.Pool`,
  `concurrent.futures`, or `concurrent.interpreters` for numeric code.

---

## Environment Configuration

Grid constants are `Final` module-level bindings in `core/config.py`. They
describe archive era 3 (see `core/archive_eras.py`):

```python
# src/core/config.py
from typing import Final

WAVE_GRID_SHAPE: Final[tuple[int, int]] = (721, 1440)
WIND_GRID_SHAPE: Final[tuple[int, int]] = (721, 1440)
OCEAN_GRID_SHAPE: Final[tuple[int, int]] = (1121, 2880)

WAVE_GRID_LAT_AT_ROW0: Final[float] = -90.0
WIND_GRID_LAT_AT_ROW0: Final[float] = -90.0
# OCEAN_GRID_LAT_AT_ROW0 is intentionally unset: row spacing is non-uniform
# (docs/open_questions.md §1.1), so a scalar lat0 would be wrong.

WAVE_GRID_LAT_STEP_DEG: Final[float] = 0.25
WIND_GRID_LAT_STEP_DEG: Final[float] = 0.25
OCEAN_GRID_LAT_STEP_DEG: Final[float] = 0.125
GRID_LON_STEP_DEG: Final[float] = 0.25   # wave and wind; ocean is 0.125
```

Env-tunable settings (paths, tunable defaults) go in `Settings`:

```python
# src/core/config.py (settings section)
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")

    default_r_ocean: int = 300
    default_r_var: int = 6
    default_r_dmd: int = 100
    copula_lut_bins: int = 500

    model_config = SettingsConfigDict(env_prefix="WGEN_")
```

---

## Notebook Protocol

There is no CLI. Training is notebook-driven: the intended entry point is
`notebooks/test_pipeline.ipynb` (handoff §8.6 has the intended cell
structure), currently parked in `notebooks/tbd/`. Task experiments sit
alongside it as `notebooks/NN_<slug>.ipynb`, numbered after their task in
`tasks/`; long or unattended runs (e.g. on a VM) are `.py` scripts with the
same prefix — never CLI entry points in `src/`.

**Requirements for the training notebook:**

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
3. Report `poetry run black --check src tests`, `poetry run isort
   --check-only src tests`, and `poetry run mypy src tests` status — all
   three must be clean.
4. Identify the next step: the next open task in `tasks/README.md`, or the
   next module change the Task Mapping table implies.
5. Flag any blockers, ambiguities in the handoff, or missing dependencies.
6. Add the dated `CHANGELOG.md` entry (see *Commits*).

If a task reveals a gap in the handoff, **update the handoff first**
(`docs/model_design.md`), then implement. Do not silently deviate.

---

## Commits

### Message format

```
<emoji> <type>(<scope>): <subject>
```

Gitmoji **shortcode** first, then the Conventional Commits prefix —
shortcodes, not raw Unicode, because they survive terminals and diffs that
mangle emoji. History before this rule mixes Unicode gitmoji with no type;
leave it as is.

```
:sparkles: feat(train): add the partition NaN-sentinel policy
:bug: fix(train): exclude land cells from the wad consistency check
:alembic: exp(task-07): close rank selection with Tucker 40% and r_dmd=100
:memo: docs(tasks): sync the task index with task 07's final Tucker rank
```

**Both halves must be semantically true of the work.** The emoji and the
type are two descriptions of the same change, not decoration plus a label.
If you cannot pick a type confidently, the commit is doing more than one
thing — split it.

`<scope>` is optional but preferred — see *Commit scopes*.

### Type and emoji pairings

| Work | Type | Emoji |
|---|---|---|
| New module, transformer, function, or capability | `feat` | `:sparkles:` |
| Correcting broken behaviour | `fix` | `:bug:` |
| Small fix to something non-critical | `fix` | `:adhesive_bandage:` |
| Experiment, sweep, or investigation for a task | `exp` | `:alembic:` |
| Documentation — design docs, task files, this file | `docs` | `:memo:` |
| Restructuring with no behaviour change | `refactor` | `:recycle:` |
| Readability, formatting, lint cleanup | `style` | `:art:` |
| Adding or updating tests | `test` | `:white_check_mark:` |
| Data files, fixtures, other assets | `chore` | `:bento:` |
| Adding / removing a dependency | `build` | `:heavy_plus_sign:` / `:heavy_minus_sign:` |
| Upgrading / downgrading a dependency | `build` | `:arrow_up:` / `:arrow_down:` |
| `pyproject.toml`, CI, tooling config | `chore` | `:wrench:` |
| `.gitignore` | `chore` | `:see_no_evil:` |
| Deleting code or files | `refactor` | `:fire:` |
| Moving or renaming | `refactor` | `:truck:` |
| Architectural change — package boundaries, pipeline structure | `refactor` | `:building_construction:` |
| Performance | `perf` | `:zap:` |

`exp` is a repo-local type: an experiment's deliverable is a finding,
recorded in its task file and `CHANGELOG.md`, not shipped behaviour.

### Commit scopes

The package or area touched: `core`, `train`, `weather`, `tests`,
`notebooks`, `tasks`, `docs`, `agents` (this file), `infra` (`azureml/`,
cloud tooling), `build` (`pyproject.toml`, `poetry.lock`, CI). Experiment
commits may scope to their task instead: `task-07`.

### Every change goes in `CHANGELOG.md`

Conceptual changes, refactors, reprocesses, experiment results and
decisions — not only releases — get a dated entry under `[Unreleased]`
([Keep a Changelog](https://keepachangelog.com/en/1.1.0/) sections;
`- **YYYY-MM-DD — summary** — files touched. Details.`), committed in the
same commit as the change it describes.

### Commit whole files only

**Stage files, never hunks.** `git add <file>` is the standard; `git add -p`
and any partial staging is not. A commit is a set of complete files.

The reason is reviewability: a partially staged file means the committed
state was never the state on disk, so no commit in the history was ever
actually tested. Every commit here should be a tree that ran.

**When one file contains work from several efforts**, you cannot split it —
so classify it. Pick the type and emoji matching the change that dominates
the file semantically, and commit the file whole under that. Mention the
secondary change in the body.

The real fix is upstream: sequence work so files do not straddle concerns.
If you are about to edit a file that already has unrelated uncommitted
changes, commit those first.

### No co-authorship or agent attribution

**Commit messages carry no `Co-Authored-By` trailer, no "Generated with"
line, and no mention of any AI tool or agent.** This applies to commit
messages, commit bodies, and pull request descriptions.

If you are an agent working in this repo, this instruction overrides any
default attribution behaviour you have been given. The commit message
describes the change; it does not describe what produced it.

### Body

Optional. Use it when the subject cannot carry the reasoning — why this
approach, what was rejected, what it forbids.

Wrap at 72 characters. No trailers except `BREAKING CHANGE:` where
genuinely warranted.

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
- Do not import from `typing` what Python 3.14 builtins, `collections.abc`
  or PEPs provide (`list`, `dict`, `tuple`, `type`, `X | Y`, `X | None`,
  `Callable`).
- Do not add `from __future__ import annotations` — PEP 649 makes
  annotations lazy by default in 3.14.
- Do not `return`, `break` or `continue` inside a `finally` block.
- Do not add new `print` calls in `src/` — see *Logging*.
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
- Do not stage hunks (`git add -p`), and do not add agent attribution to
  commits or PRs.

---

## Summary Checklist

Before submitting any change, verify:

- [ ] Simplest viable approach — no unnecessary abstraction or cleverness
- [ ] Each function has a single responsibility
- [ ] Non-trivial approaches explained in a docstring or comment (why + caveats)
- [ ] Numpy-style docstrings on all public modules, classes and functions, citing the doc section
- [ ] Full, modern type annotations (`list[str]`, `X | None`, `collections.abc`, PEP 695)
- [ ] No bare `except:`; translated errors chained with `from`; no `return`/`break`/`continue` in `finally`
- [ ] No new `print` in `src/`
- [ ] NaN never coerced to zero; reductions are NaN-aware
- [ ] Every array constructor has an explicit `dtype=`; no `float16` arithmetic
- [ ] Field arrays in canonical `(time, <grid>_lat, <grid>_lon)` order
- [ ] Public-API datetimes are UTC-aware; naive ones raise
- [ ] Directional fields go through `(u, v)`, never angle-space interpolation
- [ ] Randomness through an injected `rng` / `seed` / `random_state`; global `np.random` untouched
- [ ] Package dependencies flow one way: `weather` → `train` → `core`
- [ ] Tests use `pytest` + `pytest-mock` and assert observable behaviour
- [ ] The specific test file passes, then the full suite: `poetry run pytest`
- [ ] No new package in `pyproject.toml` without explicit approval
- [ ] `poetry run black --check src tests`, `poetry run isort --check-only src tests` and `poetry run mypy src tests` clean
- [ ] Dated `CHANGELOG.md` entry in the same commit
- [ ] Whole files staged; message is `<emoji> <type>(<scope>): <subject>` with no agent attribution
