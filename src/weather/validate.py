"""Validation battery for a fitted WeatherModel: per-check helpers + full_report.

(see docs/model_design.md §6)

Implements the QQ, spectra, variogram, joint-density, extreme-statistics,
throughput and held-out Gaussianized-core checks, and adds a page with the
§5.6.2 group diagnostics. The end-to-end consumer test is out of scope for
this library: it requires running a consumer-specific downstream application
on real vs. synthetic weather, which cannot be automated generically.
"""

import matplotlib

matplotlib.use("Agg")  # headless: every check here only ever calls savefig, never show()

import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy import signal, stats
from sklearn.utils.validation import check_is_fitted

from core.io import require_utc
from train.fidelity import fidelity_summary
from train.ingest import load_training_window
from train.linear_inverse import diagnostics_summary
from train.tucker import fill_land, tucker_transform
from weather.model import PER_CELL_FIRST_STEP, WeatherModel, core_dataset
from weather.simulate import sample_batch, simulate_field

_EARTH_RADIUS_KM: Final[float] = 6371.0


def _random_ocean_cells(model: WeatherModel, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """`n_cells` random ``(lat_idx, lon_idx)`` pairs from the fitted ocean mask."""
    lat_idx, lon_idx = np.nonzero(model.ocean_mask_)
    n_available = lat_idx.shape[0]
    chosen = rng.choice(n_available, size=min(n_cells, n_available), replace=False)
    return np.stack([lat_idx[chosen], lon_idx[chosen]], axis=1)


def _load_real_physical(
    model: WeatherModel, data_dir: Path, t_start: datetime, t_end: datetime
) -> xr.Dataset:
    """Real archive-native data on the fitted wave grid, in physical units.

    (see docs/model_design.md §4, §6)

    Runs the fitted Pipeline forward through reprojection and back, skipping
    climatology and standardization. The round trip applies what the
    sampler applies to its own output -- absent partitions decode to height
    0, wind is resampled onto the wave grid -- so the result is directly
    comparable to `weather.simulate.simulate_field`'s.
    """
    x_raw = load_training_window(data_dir, t_start, t_end, model.variables)
    names = [name for name, _ in model.preprocess_.steps]
    to_wave_grid = model.preprocess_[: names.index(PER_CELL_FIRST_STEP) + 1]
    result: xr.Dataset = to_wave_grid.inverse_transform(to_wave_grid.transform(x_raw))
    return result


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km between paired ``(lat1, lon1)`` and ``(lat2, lon2)`` arrays."""
    lat1_rad, lon1_rad = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2_rad, lon2_rad = np.deg2rad(lat2), np.deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    result: np.ndarray = 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return result


def _empirical_semivariogram(
    values: np.ndarray, lats: np.ndarray, lons: np.ndarray, bin_edges: np.ndarray
) -> np.ndarray:
    """Mean squared pairwise difference within each distance bin, for one snapshot."""
    lat_a, lat_b = np.meshgrid(lats, lats, indexing="ij")
    lon_a, lon_b = np.meshgrid(lons, lons, indexing="ij")
    distances = _haversine_km(lat_a, lon_a, lat_b, lon_b)
    squared_diff = (values[:, None] - values[None, :]) ** 2

    upper = np.triu_indices(values.shape[0], k=1)
    distances = distances[upper]
    squared_diff = squared_diff[upper]

    bin_idx = np.digitize(distances, bin_edges) - 1
    result = np.full(bin_edges.shape[0] - 1, np.nan)
    for b in range(result.shape[0]):
        in_bin = bin_idx == b
        if in_bin.any():
            result[b] = 0.5 * np.nanmean(squared_diff[in_bin])
    return result


def _text_page(title: str, lines: list[str]) -> Figure:
    """A blank page with monospace text — for scalar-valued checks in the PDF report."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    return fig


def qq_plot_check(
    model: WeatherModel,
    real: xr.Dataset,
    simulated: xr.Dataset,
    variables: list[str],
    n_cells: int = 30,
    n_quantiles: int = 101,
    rng: np.random.Generator | None = None,
) -> Figure:
    """Per-cell marginal QQ plot, real vs. simulated (see docs/model_design.md §6 check 1).

    Parameters
    ----------
    model : WeatherModel
    real, simulated : xr.Dataset
        Archive-native, physical-unit data on the wave grid (dims
        ``(time, wave_lat, wave_lon)``), e.g. from `_load_real_physical`
        and `weather.simulate.simulate_field` respectively.
    variables : list[str]
    n_cells : int
    n_quantiles : int
    rng : np.random.Generator | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(rng)
    cells = _random_ocean_cells(model, n_cells, rng)
    quantile_grid = np.linspace(0.0, 1.0, n_quantiles)

    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 5), squeeze=False)
    for ax, variable in zip(axes[0], variables):
        for lat_idx, lon_idx in cells:
            real_series = real[variable].values[:, lat_idx, lon_idx]
            sim_series = simulated[variable].values[:, lat_idx, lon_idx]
            if np.all(np.isnan(real_series)) or np.all(np.isnan(sim_series)):
                continue
            real_q = np.nanquantile(real_series, quantile_grid)
            sim_q = np.nanquantile(sim_series, quantile_grid)
            ax.plot(real_q, sim_q, ".", alpha=0.3, color="C0", markersize=3)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(variable)
        ax.set_xlabel("real quantile")
        ax.set_ylabel("simulated quantile")
    fig.suptitle("Per-cell marginal QQ (§6 check 1)")
    fig.tight_layout()
    return fig


def spectra_check(
    model: WeatherModel,
    real: xr.Dataset,
    simulated: xr.Dataset,
    variables: list[str],
    n_cells: int = 30,
    rng: np.random.Generator | None = None,
) -> Figure:
    """Per-cell temporal spectra (Welch PSD), real vs. simulated (see docs/model_design.md §6 check 2).

    Parameters
    ----------
    model : WeatherModel
    real, simulated : xr.Dataset
    variables : list[str]
    n_cells : int
    rng : np.random.Generator | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(rng)
    cells = _random_ocean_cells(model, n_cells, rng)
    fs = 1.0 / next(iter(model.dmd_.values())).dt  # every group shares the snapshot cadence

    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 5), squeeze=False)
    for ax, variable in zip(axes[0], variables):
        for lat_idx, lon_idx in cells:
            real_series = real[variable].values[:, lat_idx, lon_idx]
            sim_series = simulated[variable].values[:, lat_idx, lon_idx]
            if np.any(np.isnan(real_series)) or np.any(np.isnan(sim_series)):
                continue
            nperseg = min(real_series.shape[0], 64)
            f_real, psd_real = signal.welch(real_series, fs=fs, nperseg=nperseg)
            f_sim, psd_sim = signal.welch(sim_series, fs=fs, nperseg=nperseg)
            ax.loglog(f_real, psd_real, color="C0", alpha=0.3)
            ax.loglog(f_sim, psd_sim, color="C1", alpha=0.3)
        ax.set_title(variable)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD")
    fig.suptitle("Per-cell temporal spectra: real (blue) vs. simulated (orange) (§6 check 2)")
    fig.tight_layout()
    return fig


def variogram_check(
    model: WeatherModel,
    real: xr.Dataset,
    simulated: xr.Dataset,
    variable: str,
    n_bins: int = 20,
    max_distance_km: float = 5000.0,
    n_sample_cells: int = 1500,
    n_snapshots: int = 5,
    rng: np.random.Generator | None = None,
) -> Figure:
    """Empirical semivariogram, real vs. simulated (see docs/model_design.md §6 check 3).

    Averages the binned semivariance across `n_snapshots` randomly-chosen
    times to reduce noise, per the design doc. Distances are computed only
    among a random subsample of `n_sample_cells` ocean cells (pairwise
    distance is O(cells²); the full ocean grid can be 10^5-10^6 cells).

    Parameters
    ----------
    model : WeatherModel
    real, simulated : xr.Dataset
    variable : str
    n_bins : int
    max_distance_km : float
    n_sample_cells : int
    n_snapshots : int
    rng : np.random.Generator | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(rng)
    cells = _random_ocean_cells(model, n_sample_cells, rng)
    lat_idx, lon_idx = cells[:, 0], cells[:, 1]

    wave_lats = real.coords["wave_lat"].values
    wave_lons = real.coords["wave_lon"].values
    cell_lats = wave_lats[lat_idx]
    cell_lons = wave_lons[lon_idx]

    bin_edges = np.linspace(0.0, max_distance_km, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_times = real[variable].shape[0]
    time_indices = rng.choice(n_times, size=min(n_snapshots, n_times), replace=False)

    real_bins = np.full((time_indices.shape[0], n_bins), np.nan)
    sim_bins = np.full((time_indices.shape[0], n_bins), np.nan)
    for i, t in enumerate(time_indices):
        real_values = real[variable].values[t, lat_idx, lon_idx]
        sim_values = simulated[variable].values[t, lat_idx, lon_idx]
        real_bins[i] = _empirical_semivariogram(real_values, cell_lats, cell_lons, bin_edges)
        sim_bins[i] = _empirical_semivariogram(sim_values, cell_lats, cell_lons, bin_edges)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        real_mean = np.nanmean(real_bins, axis=0)
        sim_mean = np.nanmean(sim_bins, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(bin_centers, real_mean, "o-", label="real")
    ax.plot(bin_centers, sim_mean, "s--", label="simulated")
    ax.set_xlabel("distance (km)")
    ax.set_ylabel(f"semivariance ({variable})")
    ax.set_title(f"Empirical variogram: {variable} (§6 check 3)")
    ax.legend()
    fig.tight_layout()
    return fig


def joint_density_check(
    model: WeatherModel,
    real: xr.Dataset,
    simulated: xr.Dataset,
    var_a: str,
    var_b: str,
    rng: np.random.Generator | None = None,
) -> Figure:
    """Joint density at one ocean cell, real vs. simulated (see docs/model_design.md §6 check 4).

    Parameters
    ----------
    model : WeatherModel
    real, simulated : xr.Dataset
    var_a, var_b : str
    rng : np.random.Generator | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(rng)
    lat_idx, lon_idx = _random_ocean_cells(model, 1, rng)[0]

    real_a = real[var_a].values[:, lat_idx, lon_idx]
    real_b = real[var_b].values[:, lat_idx, lon_idx]
    sim_a = simulated[var_a].values[:, lat_idx, lon_idx]
    sim_b = simulated[var_b].values[:, lat_idx, lon_idx]

    fig, (ax_real, ax_sim) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax_real.scatter(real_a, real_b, s=8, alpha=0.5)
    ax_real.set_title("real")
    ax_sim.scatter(sim_a, sim_b, s=8, alpha=0.5, color="C1")
    ax_sim.set_title("simulated")
    for ax in (ax_real, ax_sim):
        ax.set_xlabel(var_a)
        ax.set_ylabel(var_b)
    fig.suptitle(f"Joint density at one ocean cell: {var_a} vs. {var_b} (§6 check 4)")
    fig.tight_layout()
    return fig


def extreme_statistics_check(
    model: WeatherModel,
    real: xr.Dataset,
    simulated: xr.Dataset,
    variable: str,
    percentile: float = 99.0,
    n_cells: int = 30,
    rng: np.random.Generator | None = None,
) -> Figure:
    """Per-cell extreme percentile, real vs. simulated (see docs/model_design.md §6 check 5).

    Parameters
    ----------
    model : WeatherModel
    real, simulated : xr.Dataset
    variable : str
    percentile : float
    n_cells : int
    rng : np.random.Generator | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(rng)
    cells = _random_ocean_cells(model, n_cells, rng)

    real_extremes = np.full(cells.shape[0], np.nan)
    sim_extremes = np.full(cells.shape[0], np.nan)
    for i, (lat_idx, lon_idx) in enumerate(cells):
        real_series = real[variable].values[:, lat_idx, lon_idx]
        sim_series = simulated[variable].values[:, lat_idx, lon_idx]
        if not np.all(np.isnan(real_series)):
            real_extremes[i] = np.nanpercentile(real_series, percentile)
        if not np.all(np.isnan(sim_series)):
            sim_extremes[i] = np.nanpercentile(sim_series, percentile)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real_extremes, sim_extremes, alpha=0.6)
    finite = np.isfinite(real_extremes) & np.isfinite(sim_extremes)
    if finite.any():
        lo = float(min(real_extremes[finite].min(), sim_extremes[finite].min()))
        hi = float(max(real_extremes[finite].max(), sim_extremes[finite].max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel(f"real p{percentile:g}")
    ax.set_ylabel(f"simulated p{percentile:g}")
    ax.set_title(f"Extreme statistics: {variable} (§6 check 5)")
    fig.tight_layout()
    return fig


def throughput_benchmark(
    model: WeatherModel,
    ns: tuple[int, ...] = (1, 100, 10_000, 1_000_000),
    seed: int | None = None,
) -> dict[int, float]:
    """Time `weather.simulate.sample_batch` at increasing batch sizes (see docs/model_design.md §6 check 6).

    All `N` points in a given batch share a single query time, which is the
    scenario the "batched fast path" (§1 correctness criterion 5 — one BLAS
    call for 10⁴+ points batched *at once*) is meant to speed up: `N`
    points spread across `N` distinct times would instead hit
    `sample_batch`'s one-interpolator-per-unique-time path once per point,
    which is not what this check is measuring. Query points are drawn
    uniformly within the fitted wave grid's lat/lon bounds; some may land on
    land (NaN result), which doesn't affect timing.

    Parameters
    ----------
    model : WeatherModel
    ns : tuple[int, ...]
    seed : int | None

    Returns
    -------
    dict[int, float]
        Microseconds per point, keyed by `N`.
    """
    check_is_fitted(model, "preprocess_")
    rng = np.random.default_rng(seed)
    reprojector = model.preprocess_.named_steps["reproject"]
    lat_lo, lat_hi = float(reprojector.target_lats_.min()), float(reprojector.target_lats_.max())
    lon_lo, lon_hi = float(reprojector.target_lons_.min()), float(reprojector.target_lons_.max())
    query_variable = model.variables[0]
    query_time = np.datetime64(model.t_start.replace(tzinfo=None), "s")

    results: dict[int, float] = {}
    for n in ns:
        lats = rng.uniform(lat_lo, lat_hi, size=n)
        lons = rng.uniform(lon_lo, lon_hi, size=n)
        times = np.full(n, query_time)
        start = time.perf_counter()
        sample_batch(model, lats=lats, lons=lons, times=times, variables=query_variable, seed=seed)
        elapsed = time.perf_counter() - start
        results[n] = elapsed * 1.0e6 / n
    return results


def held_out_normality_check(
    model: WeatherModel,
    held_out_dir: Path,
    held_out_start: datetime,
    held_out_end: datetime,
    n_coefficients: int = 30,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """KS-test p-values of held-out Gaussianized core coefficients (see model_design.md §6 check 7).

    v3 Gaussianizes the Tucker core, not grid cells (§5.5.1), so this
    applies the preprocessing chain (`model.transform`), projects each
    tensor onto its Tucker factors, applies that tensor's core copula, and
    tests `n_coefficients` random coefficient series against N(0, 1).
    Systematic non-normality means the copula or climatology overfit the
    training window. Snapshots with NaN at an ocean cell (an ingest gap)
    are dropped before projecting.

    Parameters
    ----------
    model : WeatherModel
    held_out_dir : Path
    held_out_start, held_out_end : datetime
        UTC-aware.
    n_coefficients : int
    rng : np.random.Generator | None

    Returns
    -------
    dict[str, np.ndarray]
        Tensor -> p-values, one per sampled coefficient (``nan`` when fewer
        than 8 snapshots remain).
    """
    rng = np.random.default_rng(rng)
    anomalies = model.transform(held_out_dir, held_out_start, held_out_end)
    ocean = model.ocean_mask_
    p_values: dict[str, np.ndarray] = {}
    for tensor in model.tensor_names_:
        field = anomalies[tensor].values
        complete = ~np.isnan(field[:, ocean]).any(axis=1)
        core = tucker_transform(fill_land(field[complete], ocean), model.tucker_[tensor])
        gaussianized = model.core_copula_[tensor].transform(core_dataset(tensor, core))
        flat = gaussianized[tensor].values.reshape(core.shape[0], -1)
        chosen = rng.choice(flat.shape[1], size=min(n_coefficients, flat.shape[1]), replace=False)
        p_values[tensor] = np.array(
            [
                stats.kstest(flat[:, j], "norm").pvalue if flat.shape[0] >= 8 else np.nan
                for j in chosen
            ],
            dtype=np.float64,
        )
    return p_values


def full_report(
    model: WeatherModel,
    held_out_dir: Path,
    held_out_start: datetime,
    held_out_end: datetime,
    output: Path,
    seed: int | None = None,
) -> None:
    """Assemble the §6 validation battery into a multi-page PDF (see docs/model_design.md §6).

    Compares real held-out data against a `weather.simulate.simulate_field`
    trajectory over ``[held_out_start, held_out_end)``. Check 8 (end-to-end
    consumer test) is out of scope for this library — see the module
    docstring.

    Parameters
    ----------
    model : WeatherModel
    held_out_dir : Path
    held_out_start, held_out_end : datetime
        UTC-aware.
    output : Path
        Destination PDF path.
    seed : int | None
    """
    check_is_fitted(model, "preprocess_")
    require_utc(held_out_start, "held_out_start")
    require_utc(held_out_end, "held_out_end")
    rng = np.random.default_rng(seed)

    real = _load_real_physical(model, held_out_dir, held_out_start, held_out_end)
    simulated = simulate_field(
        model, held_out_start, held_out_end, variables=list(model.variables), seed=seed
    )
    common_variables = sorted(
        {str(name) for name in real.data_vars} & {str(name) for name in simulated.data_vars}
    )

    coming_from = model.preprocess_.named_steps["coming_from_to_uv"]
    going_to = model.preprocess_.named_steps["going_to_to_uv"]
    real_vectors = going_to.transform(coming_from.transform(real))
    simulated_vectors = going_to.transform(coming_from.transform(simulated))
    vector_pairs = [
        (u_name, v_name)
        for _, _, u_name, v_name in (*coming_from.pairs_, *going_to.pairs_)
        if u_name in real_vectors.data_vars and u_name in simulated_vectors.data_vars
    ]

    figures = [
        qq_plot_check(model, real, simulated, common_variables, rng=rng),
        spectra_check(model, real, simulated, common_variables, rng=rng),
        variogram_check(model, real, simulated, variable=common_variables[0], rng=rng),
    ]
    for var_a, var_b in vector_pairs:
        figures.append(
            joint_density_check(model, real_vectors, simulated_vectors, var_a, var_b, rng=rng)
        )
    extreme_variable = "sigwh" if "sigwh" in common_variables else common_variables[0]
    figures.append(
        extreme_statistics_check(model, real, simulated, variable=extreme_variable, rng=rng)
    )

    throughput = throughput_benchmark(model, seed=seed)
    figures.append(
        _text_page(
            "Batched throughput (§6 check 6)",
            [f"N={n:>10,}: {us_per_point:.2f} us/point" for n, us_per_point in throughput.items()],
        )
    )

    p_values = held_out_normality_check(model, held_out_dir, held_out_start, held_out_end, rng=rng)
    normality_lines = [
        f"{variable}: median p={np.nanmedian(values):.3f}, "
        f"frac p<0.05={np.nanmean(values < 0.05):.2%}"
        for variable, values in p_values.items()
    ]
    figures.append(
        _text_page("Held-out Gaussianized-core normality, KS test (§6 check 7)", normality_lines)
    )
    figures.append(
        _text_page(
            "Group dynamics diagnostics (§5.6.2)",
            [diagnostics_summary(d) for d in model.group_diagnostics_.values()],
        )
    )
    figures.append(
        _text_page(
            "Generative fidelity (§5.6.3)",
            [fidelity_summary(d) for d in model.fidelity_diagnostics_.values()],
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
