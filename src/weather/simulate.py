"""Free-function sampling API: sample, sample_batch, simulate_field.

(see docs/model_design.md §5.8, §8.4)

Free functions, not methods on `WeatherModel` — the model is a passive
container of fitted state; these algorithms don't mutate it (see AGENTS.md
"Sampling is free functions, not methods").

Order of operations (§5.8): evolve each group's stochastic channels as one
Markov chain over the requested times (one correlated draw per group and
time step, §5.6.1) → complex mode amplitudes, conjugate
pairs exact → per-tensor Gaussianized cores → copula inverse → Tucker
inverse on the cells needed, land NaN → the Pipeline's per-cell inverses on
the grid → bilinear interpolation of the linear-safe tensors → the
Pipeline's pointwise inverses. Interpolating after the nonlinear inverses
would average direction angles across 0°/360° and heights in physical space.
"""

from datetime import datetime
from typing import Final

import numpy as np
import xarray as xr
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from core.io import require_utc
from train.dmd import (
    ModeOUFit,
    mode_amplitudes,
    ou_stationary_cov,
    ou_transition,
    split_group_state,
)
from train.linear_inverse import covariance_factor
from train.reproject import lonlat_interpolator
from train.tucker import tucker_inverse
from weather.model import PER_CELL_FIRST_STEP, WeatherModel, core_dataset

# Unique query times per full-grid pass in `sample_batch`. The per-cell
# inverses run on full-grid arrays (§5.8 step 4), about 8 MB per tensor and
# time at era-3 resolution, so the chunk bounds that memory.
_TIME_CHUNK: Final[int] = 8


def _split_pipeline(model: WeatherModel) -> tuple[Pipeline, Pipeline]:
    """The fitted Pipeline cut into its pointwise and per-cell halves (§5.8 steps 4, 6)."""
    names = [name for name, _ in model.preprocess_.steps]
    index = names.index(PER_CELL_FIRST_STEP)
    return model.preprocess_[:index], model.preprocess_[index:]


def _wave_grid_coordinates(model: WeatherModel) -> tuple[np.ndarray, np.ndarray]:
    """The fitted Reprojector's target grid coordinates."""
    reprojector = model.preprocess_.named_steps[PER_CELL_FIRST_STEP]
    return reprojector.target_lats_, reprojector.target_lons_


def _cadence_seconds(model: WeatherModel) -> float:
    """Seconds between snapshots; every group shares the training window's cadence."""
    return next(iter(model.dmd_.values())).dt


def _delta_n(times64: np.ndarray, t_train_start: datetime, dt: float) -> np.ndarray:
    """Steps since the training window's first snapshot (fractional, §5.8 step 1).

    Raises
    ------
    ValueError
        For a time before `t_train_start`: DMD's decaying deterministic
        terms grow without bound backwards in time, so the model only
        extrapolates forward from its training window.
    """
    t_train_start64 = np.datetime64(t_train_start.replace(tzinfo=None), "s")
    seconds = (times64 - t_train_start64).astype("timedelta64[s]").astype(np.float64)
    if np.any(seconds < 0):
        raise ValueError(
            f"query times before the training window's start {t_train_start.isoformat()} "
            "are not supported (docs/model_design.md §5.8)"
        )
    result: np.ndarray = seconds / dt
    return result


def _stationary_draw(ou: ModeOUFit, rng: np.random.Generator) -> np.ndarray:
    """One draw of a group's channels from their joint stationary law, ``(n_c,)``."""
    result: np.ndarray = covariance_factor(ou_stationary_cov(ou)) @ rng.standard_normal(ou.phi.size)
    return result


def _channel_path(ou: ModeOUFit, delta_n: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One realization of a group's channels at increasing times, ``(n_c, n_t)`` (§5.8 step 1).

    A stationary draw at the first time, then the closed-form transition
    across each gap to the next time. The channels are a Markov process, so
    this gives the exact joint law at every requested time, regular grid or
    not. Each distinct gap is factorized once: a regular grid needs one
    factorization, and its one-step noise covariance is the innovation
    covariance itself.

    Parameters
    ----------
    ou : ModeOUFit
    delta_n : (n_t,) float, strictly increasing
        Steps since the training window's first snapshot.
    rng : np.random.Generator

    Returns
    -------
    (n_c, n_t) float64
    """
    path = np.empty((ou.phi.size, delta_n.size), dtype=np.float64)
    path[:, 0] = _stationary_draw(ou, rng)
    if delta_n.size == 1:
        return path
    gaps, gap_index = np.unique(np.diff(delta_n), return_inverse=True)
    decay, noise_cov = ou_transition(ou, gaps)
    factors = covariance_factor(noise_cov)
    standard_normal = rng.standard_normal((delta_n.size - 1, ou.phi.size))
    for step in range(1, delta_n.size):
        k = gap_index[step - 1]
        path[:, step] = decay[k] * path[:, step - 1] + factors[k] @ standard_normal[step - 1]
    return path


def _gaussianized_cores(
    model: WeatherModel, channels: dict[str, np.ndarray], delta_n: np.ndarray
) -> dict[str, np.ndarray]:
    """Channels → mode amplitudes → joint states → per-tensor cores (§5.8 steps 1–2).

    Returns
    -------
    dict[str, np.ndarray]
        Tensor -> Gaussianized core ``(n_t, r_lat, r_lon)``.
    """
    cores: dict[str, np.ndarray] = {}
    for group, members in model.groups_.items():
        dmd = model.dmd_[group]
        amplitudes = mode_amplitudes(dmd, model.mode_ou_[group], channels[group], delta_n)
        state = (dmd.Phi.astype(np.complex128) @ amplitudes).real.T
        shapes = {
            m: (model.tucker_[m].factor_lat.shape[1], model.tucker_[m].factor_lon.shape[1])
            for m in members
        }
        cores.update(split_group_state(state, members, shapes))
    return cores


def _standardized_anomaly(
    model: WeatherModel,
    gaussianized_cores: dict[str, np.ndarray],
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Copula inverse on each core, then its Tucker inverse on ``rows × cols`` (§5.8 step 3)."""
    anomaly: dict[str, np.ndarray] = {}
    for tensor, gaussianized in gaussianized_cores.items():
        core_ds = model.core_copula_[tensor].inverse_transform(core_dataset(tensor, gaussianized))
        anomaly[tensor] = tucker_inverse(core_ds[tensor].values, model.tucker_[tensor], rows, cols)
    return anomaly


def _grid_dataset(
    model: WeatherModel, fields: dict[str, np.ndarray], times64: np.ndarray
) -> xr.Dataset:
    """Full-grid fields as a canonical Dataset, land set to NaN.

    The Tucker inverse also reconstructs the ALS's imputed land values;
    those are not data, so land is masked before any inverse runs.
    """
    wave_lats, wave_lons = _wave_grid_coordinates(model)
    land = ~model.ocean_mask_
    data_vars = {}
    for tensor, values in fields.items():
        masked = values.copy()
        masked[:, land] = np.nan
        data_vars[tensor] = (("time", "wave_lat", "wave_lon"), masked)
    return xr.Dataset(
        data_vars,
        coords={
            "time": times64.astype("datetime64[s]"),
            "wave_lat": wave_lats,
            "wave_lon": wave_lons,
        },
    )


def _scatter_to_grid(
    blocks: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray, grid_shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Place each ``(n_t, n_rows, n_cols)`` sub-grid on a NaN full grid."""
    fields: dict[str, np.ndarray] = {}
    for tensor, block in blocks.items():
        full = np.full((block.shape[0], *grid_shape), np.nan, dtype=np.float64)
        full[:, rows[:, None], cols[None, :]] = block
        fields[tensor] = full
    return fields


def _stencil_rows_cols(
    lats: np.ndarray, lons: np.ndarray, grid_lats: np.ndarray, grid_lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Grid rows and columns covering every query's 2 × 2 bilinear stencil (§5.8 step 3).

    Matches `lonlat_interpolator`'s stencil choice: the lower corner is the
    last grid line at or below the query (clipped to the second-to-last row
    so a query on the top row uses the top interval), and the column after
    the last one wraps to column 0. A superset is harmless; it only costs
    evaluation.
    """
    n_lat, n_lon = grid_lats.size, grid_lons.size
    row0 = np.clip(np.searchsorted(grid_lats, lats, side="right") - 1, 0, n_lat - 2)
    step = grid_lons[1] - grid_lons[0]
    col0 = np.floor((lons - grid_lons[0]) / step).astype(np.int64) % n_lon
    rows = np.unique(np.concatenate([row0, row0 + 1]))
    cols = np.unique(np.concatenate([col0, (col0 + 1) % n_lon]))
    return rows, cols


def _check_variables(model: WeatherModel, requested: list[str]) -> None:
    """Raise unless every requested name is a modeled archive field.

    The Pipeline's inverses return exactly the raw fields it was fit on
    (`model.variables`), so this is checked before any sampling work.
    """
    missing = sorted(set(requested) - set(model.variables))
    if missing:
        raise ValueError(
            f"variables {missing} are not modeled; available: {sorted(model.variables)} "
            "(docs/model_design.md §10)"
        )


def sample_batch(
    model: WeatherModel,
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    variables: list[str] | str,
    seed: int | None = None,
) -> np.ndarray:
    """Batched point query — the fast path (see docs/model_design.md §5.8).

    All query times belong to one realization: each group's channels are a
    Markov chain over the sorted unique times, so consecutive queries along
    a route carry the model's temporal correlation. At the times and grid
    nodes of a `simulate_field` window, with the same seed, the two
    functions return the same values. Only the stencil rows and columns are
    Tucker-inverted; the per-cell inverses run on the grid before
    interpolation, the pointwise inverses after.

    Parameters
    ----------
    model : WeatherModel
    lats, lons : (N,) float
        Degrees; longitudes are taken modulo 360.
    times : (N,) datetime64[s]
        UTC, at or after the training window's start.
    variables : list[str] | str
        Archive-native variable name(s), e.g. ``"sigwh"`` or ``["sigwh", "was"]``.
    seed : int | None

    Returns
    -------
    np.ndarray
        ``(N,)`` if `variables` is a single string, else ``(N, len(variables))``.
        NaN where a query's stencil touches land or leaves the grid.
    """
    check_is_fitted(model, "preprocess_")
    variable_list = [variables] if isinstance(variables, str) else list(variables)
    _check_variables(model, variable_list)
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.mod(np.asarray(lons, dtype=np.float64), 360.0)
    points = np.stack([lats, lons], axis=-1)

    times64 = np.asarray(times, dtype="datetime64[s]")
    unique_times, inverse = np.unique(times64, return_inverse=True)
    inverse = inverse.reshape(-1)
    delta_n = _delta_n(unique_times, model.metadata_.t_train_start, _cadence_seconds(model))

    rng = np.random.default_rng(seed)
    channels = {g: _channel_path(ou, delta_n, rng) for g, ou in model.mode_ou_.items()}

    pointwise, per_cell = _split_pipeline(model)
    wave_lats, wave_lons = _wave_grid_coordinates(model)
    rows, cols = _stencil_rows_cols(lats, lons, wave_lats, wave_lons)
    tensors = list(model.tensor_names_)
    linear = np.full((lats.size, len(tensors)), np.nan, dtype=np.float64)
    for start in range(0, unique_times.size, _TIME_CHUNK):
        chunk = slice(start, start + _TIME_CHUNK)
        chunk_channels = {g: path[:, chunk] for g, path in channels.items()}
        blocks = _standardized_anomaly(
            model, _gaussianized_cores(model, chunk_channels, delta_n[chunk]), rows, cols
        )
        grid = _grid_dataset(
            model,
            _scatter_to_grid(blocks, rows, cols, model.ocean_mask_.shape),
            unique_times[chunk],
        )
        on_grid = per_cell.inverse_transform(grid)
        for offset, time_index in enumerate(range(start, start + grid.sizes["time"])):
            at_time = inverse == time_index
            fields = np.stack([on_grid[t].values[offset] for t in tensors])
            linear[at_time] = lonlat_interpolator(fields, wave_lats, wave_lons)(points[at_time])

    physical = pointwise.inverse_transform(
        xr.Dataset({t: (("point",), linear[:, i]) for i, t in enumerate(tensors)})
    )
    if isinstance(variables, str):
        result: np.ndarray = physical[variables].values
        return result
    return np.stack([physical[name].values for name in variable_list], axis=-1)


def sample(
    model: WeatherModel,
    lat: float,
    lon: float,
    t: datetime,
    variable: str,
    seed: int | None = None,
) -> float:
    """Single-point convenience call; wraps `sample_batch` (see docs/model_design.md §1, §5.8).

    Parameters
    ----------
    model : WeatherModel
    lat, lon : float
    t : datetime
        UTC-aware.
    variable : str
        Archive-native variable name.
    seed : int | None

    Returns
    -------
    float
    """
    require_utc(t, "t")
    values = sample_batch(
        model,
        lats=np.array([lat], dtype=np.float64),
        lons=np.array([lon], dtype=np.float64),
        times=np.array([t.replace(tzinfo=None)], dtype="datetime64[s]"),
        variables=variable,
        seed=seed,
    )
    return float(values[0])


def simulate_field(
    model: WeatherModel,
    t_start: datetime,
    t_end: datetime,
    variables: list[str],
    seed: int | None = None,
) -> xr.Dataset:
    """Full-field synthetic trajectory on the native wave grid at the snapshot cadence.

    (see docs/model_design.md §5.8)

    The same channel chain as `sample_batch`, on a regular grid of times;
    the whole fitted Pipeline inverts the grid, with no interpolation.

    Parameters
    ----------
    model : WeatherModel
    t_start, t_end : datetime
        UTC-aware; `t_start` inclusive, `t_end` exclusive; `t_start` at or
        after the training window's start.
    variables : list[str]
        Archive-native variable names to keep in the result.
    seed : int | None

    Returns
    -------
    xr.Dataset
        Dims ``(time, wave_lat, wave_lon)``, restricted to `variables`; land NaN.
    """
    check_is_fitted(model, "preprocess_")
    require_utc(t_start, "t_start")
    require_utc(t_end, "t_end")
    _check_variables(model, variables)
    dt = _cadence_seconds(model)
    times64 = np.arange(
        np.datetime64(t_start.replace(tzinfo=None), "s"),
        np.datetime64(t_end.replace(tzinfo=None), "s"),
        np.timedelta64(int(dt), "s"),
        dtype="datetime64[s]",
    )
    if times64.size == 0:
        raise ValueError(f"empty simulation window: [{t_start.isoformat()}, {t_end.isoformat()})")
    delta_n = _delta_n(times64, model.metadata_.t_train_start, dt)

    rng = np.random.default_rng(seed)
    channels = {g: _channel_path(ou, delta_n, rng) for g, ou in model.mode_ou_.items()}
    anomaly = _standardized_anomaly(model, _gaussianized_cores(model, channels, delta_n))
    physical: xr.Dataset = model.preprocess_.inverse_transform(
        _grid_dataset(model, anomaly, times64)
    )
    result: xr.Dataset = physical[variables]
    return result
