"""Per-tensor masked Tucker compression of the spatial modes (see docs/model_design.md §5.5).

Each modeled field is its own ``(T, lat, lon)`` tensor; Tucker compresses the
two spatial modes and leaves time at full rank for DMD. The fit wraps
`train.masked_tucker.partial_tucker_lowmem`, the vendored masked ALS that
fixes tensorly 0.9.0's memory blow-up and ignored ``svd=`` argument (§9
gotcha 21), so land cells are imputed by the ALS instead of entering the fit.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from train.masked_tucker import partial_tucker_lowmem

_SPATIAL_MODES: Final[tuple[int, int]] = (1, 2)
_ERROR_BLOCK_SNAPSHOTS: Final[int] = 64  # time block for the ocean-only error


@dataclass(frozen=True, slots=True)
class TuckerFit:
    """Spatial factor matrices of one tensor's masked Tucker fit (see model_design.md §5.5)."""

    factor_lat: np.ndarray  # (H, r_lat) float64
    factor_lon: np.ndarray  # (W, r_lon) float64
    reconstruction_error: float  # relative Frobenius error over ocean cells
    n_iter: int  # ALS iterations run


def fill_land(field: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """Zero-fill land for the masked ALS, refusing NaN at ocean cells (see §5.5, §9 gotcha 24).

    Land is NaN everywhere upstream; ``0.0`` is the neutral placeholder the
    masked ALS needs (NaN poisons its initial norm, a large sentinel biases
    the SVD initialization). An ocean NaN means an upstream stage failed to
    fill something, so it raises instead of being imputed silently.

    Parameters
    ----------
    field : (T, H, W) float
        One tensor's standardized anomaly, land NaN.
    ocean_mask : (H, W) bool
        True at ocean cells (`train.masking.land_mask_from_sigwh`).

    Returns
    -------
    (T, H, W) float32, land 0.0.

    Raises
    ------
    ValueError
        If any ocean cell is NaN at any snapshot.
    """
    ocean_nan = np.isnan(field) & ocean_mask[None, :, :]
    if ocean_nan.any():
        n_bad = int(ocean_nan.any(axis=0).sum())
        raise ValueError(
            f"{n_bad} ocean cells hold NaN at the Tucker boundary; an upstream stage left "
            "them unfilled (partition sentinel or a reprojection stencil touching land, "
            "docs/model_design.md §5.5)"
        )
    result: np.ndarray = np.where(ocean_mask[None, :, :], field, 0.0).astype(np.float32)
    return result


def tucker_fit(
    x: np.ndarray,
    ocean_mask: np.ndarray,
    rank: tuple[int, int],
    rng: np.random.Generator,
    n_iter_max: int = 300,
    tol: float = 1e-5,
    checkpoint_every: int = 12,
    early_stop_tol: float = 1e-5,
) -> tuple[TuckerFit, np.ndarray]:
    """Masked Tucker on the spatial modes; returns the fit and its ``(T, r_lat, r_lon)`` core.

    (see docs/model_design.md §5.5)

    Randomized-SVD initialization, then ALS in chunks of `checkpoint_every`
    iterations, each resuming from the previous ``(core, factors)``: the
    settings of task 07's production run. Stops at `n_iter_max`, or when the
    ocean-only reconstruction error improves by less than `early_stop_tol`
    from one chunk to the next. The ALS's own core is returned: it is
    consistent with the imputed land values, and it is what the copula and
    DMD consume. Use `tucker_transform` for held-out data instead.

    Parameters
    ----------
    x : (T, H, W) float32
        One tensor, land filled ``0.0`` (`fill_land`).
    ocean_mask : (H, W) bool
    rank : (r_lat, r_lon)
    rng : np.random.Generator
        Seeds the randomized SVD. tensorly 0.9 accepts only an int or a
        legacy ``RandomState``, so one int is drawn from `rng` and reused by
        every chunk, as task 07's script reused its fixed seed.
    n_iter_max : int
        Total ALS iterations across all chunks.
    tol : float
        The ALS's own within-chunk stopping tolerance.
    checkpoint_every : int
        Iterations per chunk.
    early_stop_tol : float
        Minimum chunk-to-chunk improvement in relative error; 0 disables.

    Returns
    -------
    fit : TuckerFit
    core : (T, r_lat, r_lon) float64
    """
    seed = int(rng.integers(2**31 - 1))
    mask = np.broadcast_to(ocean_mask[None, :, :], x.shape).astype(np.int8)
    init: Literal["svd"] | tuple[np.ndarray, list[np.ndarray]] = "svd"
    n_iter = 0
    error = float("nan")
    previous_error = float("nan")
    while n_iter < n_iter_max:
        chunk = min(checkpoint_every, n_iter_max - n_iter)
        (core, factors), errors = partial_tucker_lowmem(
            x,
            rank=list(rank),
            modes=list(_SPATIAL_MODES),
            n_iter_max=chunk,
            init=init,
            tol=tol,
            svd="randomized_svd",
            random_state=seed,
            mask=mask,
            svd_mask_repeats=0,
        )
        init = (core, factors)
        n_iter += len(errors)
        error = _ocean_relative_error(x, core, factors[0], factors[1], ocean_mask)
        if early_stop_tol > 0 and previous_error - error < early_stop_tol:
            break
        # A chunk the ALS's own `tol` cut short has converged within that chunk.
        if len(errors) < chunk:
            break
        previous_error = error

    fit = TuckerFit(
        factor_lat=np.asarray(factors[0], dtype=np.float64),
        factor_lon=np.asarray(factors[1], dtype=np.float64),
        reconstruction_error=error,
        n_iter=n_iter,
    )
    return fit, np.asarray(core, dtype=np.float64)


def _ocean_relative_error(
    x: np.ndarray,
    core: np.ndarray,
    factor_lat: np.ndarray,
    factor_lon: np.ndarray,
    ocean_mask: np.ndarray,
) -> float:
    """Relative Frobenius error over ocean cells, accumulated in time blocks.

    A full reconstruction of a production tensor is another 12 GB, so the
    residual is formed `_ERROR_BLOCK_SNAPSHOTS` snapshots at a time.
    """
    residual_sq = 0.0
    norm_sq = 0.0
    for start in range(0, x.shape[0], _ERROR_BLOCK_SNAPSHOTS):
        stop = start + _ERROR_BLOCK_SNAPSHOTS
        block = x[start:stop][:, ocean_mask].astype(np.float64)
        recon = _expand(core[start:stop], factor_lat, factor_lon)[:, ocean_mask]
        residual_sq += float(np.sum((recon - block) ** 2))
        norm_sq += float(np.sum(block**2))
    return float(np.sqrt(residual_sq / norm_sq)) if norm_sq > 0 else 0.0


def tucker_transform(x: np.ndarray, fit: TuckerFit) -> np.ndarray:
    """Project a tensor onto fitted factors: ``x ×_lat factor_latᵀ ×_lon factor_lonᵀ``.

    (see docs/model_design.md §5.5)

    For held-out data. Land must already be ``0.0`` (`fill_land`): with
    orthonormal factors the projection then weights only ocean values,
    whereas the training core came from the ALS's imputed land values, so
    the two agree only approximately near coasts.

    Parameters
    ----------
    x : (T, H, W)
    fit : TuckerFit

    Returns
    -------
    core : (T, r_lat, r_lon) float64
    """
    core: np.ndarray = fit.factor_lat.T @ x.astype(np.float64) @ fit.factor_lon
    return core


def tucker_inverse(
    core: np.ndarray,
    fit: TuckerFit,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct grid values from a core: ``core ×_lat factor_lat ×_lon factor_lon``.

    (see docs/model_design.md §5.5, §5.8 step 3)

    `rows` / `cols` restrict the reconstruction to a sub-grid, at a cost
    proportional to its size: the sampler evaluates only the rows and
    columns its query stencils touch. The result also covers land cells
    (the ALS's imputation); callers mask them.

    Parameters
    ----------
    core : (T, r_lat, r_lon)
    fit : TuckerFit
    rows : (n_rows,) int | None
        Grid rows to evaluate; all when `None`.
    cols : (n_cols,) int | None
        Grid columns to evaluate; all when `None`.

    Returns
    -------
    (T, n_rows, n_cols) float64
    """
    factor_lat = fit.factor_lat if rows is None else fit.factor_lat[rows]
    factor_lon = fit.factor_lon if cols is None else fit.factor_lon[cols]
    return _expand(core, factor_lat, factor_lon)


def _expand(core: np.ndarray, factor_lat: np.ndarray, factor_lon: np.ndarray) -> np.ndarray:
    """``factor_lat @ core[t] @ factor_lonᵀ`` for every snapshot, as batched BLAS matmuls.

    A three-operand `np.einsum` without path optimization loops over every
    index combination, which is unusable at production size.
    """
    result: np.ndarray = factor_lat @ np.asarray(core, dtype=np.float64) @ factor_lon.T
    return result
