"""Free-function sampling API: sample, sample_batch, simulate_field.

(see docs/model_design.md §5.8, §8.4)

Free functions, not methods on `WeatherModel` — the model is a passive
container of fitted state; these algorithms don't mutate it (see AGENTS.md
"Sampling is free functions, not methods").
"""

import warnings
from datetime import datetime
from typing import Final

import numpy as np
import xarray as xr
from sklearn.utils.validation import check_is_fitted

from core.io import require_utc
from train.dmd import ModeOUFit
from train.reproject import lonlat_interpolator
from train.tucker import tucker_inverse
from weather.model import WeatherModel

_REPROJECT_STEP: Final[str] = "reproject"


# --- Mechanical restoration of train/ou.py's deleted simulation helpers ----
# `train/ou.py` was deleted (task 06): its mode-OU *fitting* logic moved into
# `train/dmd.py` (`fit_mode_ou`), but its *simulation/evolution* helpers were
# deliberately not ported anywhere — task 09 ("Sampling rewrite: coherent +
# local") owns that, and needs to add complex-conjugate-pair forcing (§9
# gotcha 8) and a parallel local-residual evolution step (§5.7, §5.8) neither
# of which exist yet. The two functions below are a verbatim, unmodified port
# of `train/ou.py`'s prior implementation (same math, only the type hint
# changed from the deleted `OUFit` to `ModeOUFit` — identical fields), added
# here only so this module and its test suite are importable again in the
# interim. Task 09 must revisit both call sites in `sample_batch`/
# `simulate_field` below to add conjugate-pair forcing and the local-residual
# term — do not treat this restoration as task 09 being done.
def _complex_normal(variance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a circularly-symmetric complex Gaussian sample per element.

    `variance` is the total variance ``E[|Z|**2]``; real and imaginary
    parts are independent with half that variance each.
    """
    std = np.sqrt(variance / 2.0)
    return rng.normal(0.0, std) + 1j * rng.normal(0.0, std)


def sample_ou_equilibrium(ou: ModeOUFit, rng: np.random.Generator) -> np.ndarray:
    """Draw each mode's initial amplitude from its stationary distribution.

    (see docs/model_design.md §5.8: ``η_k(0) ~ N(0, σ_k² / (2θ_k))``)

    Parameters
    ----------
    ou : ModeOUFit
    rng : np.random.Generator

    Returns
    -------
    (r_dmd,) complex128
    """
    variance = ou.sigma**2 / (2.0 * ou.theta)
    return _complex_normal(variance, rng)


def simulate_ou_forward(
    ou: ModeOUFit,
    eta: np.ndarray,
    n_steps: np.ndarray | float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve each OU mode's amplitude forward by ``n_steps * ou.dt`` in closed form.

    (see docs/model_design.md §5.8)

    The same closed-form transition serves both the discrete grid
    simulation (called repeatedly with ``n_steps=1``, chaining forward)
    and the continuous-time point-query fast path (called once with a
    possibly fractional ``n_steps`` directly from a reference state).

    Parameters
    ----------
    ou : ModeOUFit
    eta : np.ndarray
        Current per-mode amplitude. Mode is always axis 0; any further
        axes (e.g. a batch of query times) must match `n_steps`'s further
        axes — e.g. ``eta`` shaped ``(r_dmd, 1)`` against ``n_steps``
        shaped ``(1, n_times)`` broadcasts to ``(r_dmd, n_times)``.
    n_steps : np.ndarray | float
        Elapsed time in units of `ou.dt`; may be fractional and need not
        be 1.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, complex128, the broadcast shape of `eta` and `n_steps`.
    """
    n_steps_arr = np.asarray(n_steps, dtype=np.float64)
    eta_arr = np.asarray(eta)

    # ou.phi/sigma/theta are always 1D (r_dmd,); numpy broadcasting aligns
    # shapes from the *trailing* axis, so a bare (r_dmd,) would collide
    # with n_steps's own trailing (e.g. query-time) axis instead of lining
    # up with eta/n_steps's leading mode axis. Reshape to (r_dmd, 1, 1, ...)
    # to match however many extra axes eta/n_steps carry beyond the mode axis.
    extra_dims = max(eta_arr.ndim, n_steps_arr.ndim, 1) - 1
    mode_shape = (-1,) + (1,) * extra_dims
    phi = ou.phi.reshape(mode_shape)
    sigma = ou.sigma.reshape(mode_shape)
    theta = ou.theta.reshape(mode_shape)

    phi_n = phi**n_steps_arr
    variance = sigma**2 * (1.0 - np.abs(phi) ** (2.0 * n_steps_arr)) / (2.0 * theta)
    noise = _complex_normal(variance, rng)
    result: np.ndarray = phi_n * eta_arr + noise
    return result


def _wave_grid_coordinates(model: WeatherModel) -> tuple[np.ndarray, np.ndarray]:
    """Fetch the fitted Reprojector's target grid coordinates from `model.preprocess_`."""
    reprojector = model.preprocess_.named_steps[_REPROJECT_STEP]
    return reprojector.target_lats_, reprojector.target_lons_


def _invert_to_physical_grid(model: WeatherModel, z: np.ndarray, times64: np.ndarray) -> xr.Dataset:
    """Scatter Gaussianized ocean-cell values to the grid and invert the preprocessing chain.

    (see docs/model_design.md §4 — inversion must happen on the full grid,
    before any spatial interpolation, since climatology/standardization/
    copula are all fit per grid cell.)

    Parameters
    ----------
    model : WeatherModel
    z : (n_times, N_ocean, V) float64
        Gaussianized ocean-cell values, e.g. from `train.tucker.tucker_inverse`.
    times64 : (n_times,) datetime64[s]

    Returns
    -------
    xr.Dataset
        Physical-unit values in archive-native variable names (directional
        pairs recombined to speed/direction by `ComingFromToUV.
        inverse_transform` / `GoingToToUV.inverse_transform`), dims
        ``(time, wave_lat, wave_lon)``.
    """
    n_times = z.shape[0]
    n_lat, n_lon = model.ocean_mask_.shape
    z2d = np.full((n_times, n_lat, n_lon, len(model.variable_names_)), np.nan, dtype=np.float64)
    z2d[:, model.ocean_mask_, :] = z

    wave_lats, wave_lons = _wave_grid_coordinates(model)
    data_vars = {
        name: (("time", "wave_lat", "wave_lon"), z2d[:, :, :, i])
        for i, name in enumerate(model.variable_names_)
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="no explicit representation of timezones")
        time_coord = times64.astype("datetime64[s]")
    ds = xr.Dataset(
        data_vars, coords={"time": time_coord, "wave_lat": wave_lats, "wave_lon": wave_lons}
    )
    result: xr.Dataset = model.preprocess_.inverse_transform(ds)
    return result


def _delta_n(times64: np.ndarray, t_train_start: datetime, dt: float) -> np.ndarray:
    """Elapsed time since `t_train_start`, in units of `dt` (may be fractional)."""
    t_train_start64 = np.datetime64(t_train_start.replace(tzinfo=None), "s")
    seconds = (times64 - t_train_start64).astype("timedelta64[s]").astype(np.float64)
    return seconds / dt


def sample_batch(
    model: WeatherModel,
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    variables: list[str] | str,
    seed: int | None = None,
) -> np.ndarray:
    """Batched point query — the fast path (see docs/model_design.md §5.8).

    One closed-form OU/DMD jump per *unique* query time (from a single
    seeded equilibrium draw, so `seed` fully determines the realization),
    one preprocessing-chain inversion over all unique times together, then
    a bilinear interpolation per time group at the group's own points. See
    §5.8 for why this is a deliberate approximation for batches spanning
    many distinct times (use `simulate_field` when a dense, mutually
    consistent trajectory matters) and why inversion precedes
    interpolation (§4).

    Parameters
    ----------
    model : WeatherModel
    lats, lons : (N,) float
    times : (N,) datetime64[s]
    variables : list[str] | str
        Archive-native variable name(s), e.g. ``"sigwh"`` or ``["sigwh", "was"]``.
    seed : int | None

    Returns
    -------
    np.ndarray
        ``(N,)`` if `variables` is a single string, else ``(N, len(variables))``.
    """
    check_is_fitted(model, "preprocess_")
    if isinstance(variables, str):
        single_variable = True
        variable_list = [variables]
    else:
        single_variable = False
        variable_list = list(variables)

    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    n_points = lats.shape[0]

    rng = np.random.default_rng(seed)
    times64 = np.asarray(times, dtype="datetime64[s]")
    unique_times, inverse = np.unique(times64, return_inverse=True)
    n_unique = unique_times.shape[0]

    delta_n = _delta_n(unique_times, model.metadata_.t_train_start, model.dmd_.dt)

    eta_start = sample_ou_equilibrium(model.mode_ou_, rng)  # (r_dmd,)
    eta_q = simulate_ou_forward(model.mode_ou_, eta_start[:, None], delta_n[None, :], rng)
    a_det = model.dmd_.b0[:, None] * (model.dmd_.Lambda[:, None] ** delta_n[None, :])
    a_q = a_det + eta_q  # (r_dmd, n_unique)

    x_q = (model.dmd_.Phi @ a_q).real  # (N_state, n_unique)
    core_q = x_q.T.reshape(n_unique, model.tucker_.r_ocean, model.tucker_.r_var)
    z_q = tucker_inverse(core_q, model.tucker_)  # (n_unique, N_ocean, V)

    physical = _invert_to_physical_grid(model, z_q, unique_times)
    wave_lats, wave_lons = _wave_grid_coordinates(model)

    results = {name: np.full(n_points, np.nan, dtype=np.float64) for name in variable_list}
    for time_idx in range(n_unique):
        point_mask = inverse == time_idx
        if not point_mask.any():
            continue
        field_slices: list[np.ndarray] = []
        for name in variable_list:
            variable_array = physical.data_vars[name]
            field_slices.append(variable_array.values[time_idx])
        fields = np.stack(field_slices, axis=0)  # (len(variable_list), n_lat, n_lon)
        interpolator = lonlat_interpolator(fields, wave_lats, wave_lons)
        points = np.stack([lats[point_mask], lons[point_mask]], axis=-1)
        interpolated = interpolator(points)  # (n_points_in_group, len(variable_list))
        for v_idx, name in enumerate(variable_list):
            results[name][point_mask] = interpolated[:, v_idx]

    if single_variable:
        return results[variable_list[0]]
    return np.stack([results[name] for name in variable_list], axis=-1)


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
    """Full-field synthetic trajectory on the native wave grid at 3h cadence.

    (see docs/model_design.md §5.8)

    Unlike `sample_batch`, this chains the OU process step by step (each
    step's innovation depends on the previous step's state), giving a
    single mutually consistent trajectory rather than independent jumps
    from a reference time.

    Parameters
    ----------
    model : WeatherModel
    t_start, t_end : datetime
        UTC-aware; `t_start` inclusive, `t_end` exclusive, at 3h cadence.
    variables : list[str]
        Archive-native variable names to keep in the result.
    seed : int | None

    Returns
    -------
    xr.Dataset
        Dims ``(time, wave_lat, wave_lon)``, restricted to `variables`.
    """
    check_is_fitted(model, "preprocess_")
    require_utc(t_start, "t_start")
    require_utc(t_end, "t_end")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="no explicit representation of timezones")
        times64 = np.arange(
            np.datetime64(t_start.replace(tzinfo=None), "s"),
            np.datetime64(t_end.replace(tzinfo=None), "s"),
            np.timedelta64(int(model.dmd_.dt), "s"),
        )
    n_steps = times64.shape[0]
    if n_steps == 0:
        raise ValueError(f"empty simulation window: [{t_start.isoformat()}, {t_end.isoformat()})")

    delta_n_start = _delta_n(times64[:1], model.metadata_.t_train_start, model.dmd_.dt)[0]

    rng = np.random.default_rng(seed)
    r_dmd = model.dmd_.Lambda.shape[0]
    eta = sample_ou_equilibrium(model.mode_ou_, rng)
    a_all = np.empty((r_dmd, n_steps), dtype=np.complex128)
    for i in range(n_steps):
        if i > 0:
            eta = simulate_ou_forward(model.mode_ou_, eta, 1.0, rng)
        a_det = model.dmd_.b0 * (model.dmd_.Lambda ** (delta_n_start + i))
        a_all[:, i] = a_det + eta

    x_all = (model.dmd_.Phi @ a_all).real  # (N_state, n_steps)
    core_all = x_all.T.reshape(n_steps, model.tucker_.r_ocean, model.tucker_.r_var)
    z_all = tucker_inverse(core_all, model.tucker_)  # (n_steps, N_ocean, V)

    physical = _invert_to_physical_grid(model, z_all, times64)
    result: xr.Dataset = physical[variables]
    return result
