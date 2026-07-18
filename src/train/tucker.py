"""Tucker (HOSVD) compression of ocean-cell and variable modes (see docs/model_design.md §5.5)."""

from dataclasses import dataclass

import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot

tl.set_backend("numpy")  # explicit — TensorLy also supports jax/torch backends


@dataclass(frozen=True, slots=True)
class TuckerFit:
    """Fitted two-mode Tucker (HOSVD) factor matrices (see docs/model_design.md §5.5)."""

    U_ocean: np.ndarray  # (N_ocean, r_ocean) float64
    U_var: np.ndarray  # (V, r_var) float64
    r_ocean: int
    r_var: int
    reconstruction_error: float  # relative Frobenius error, diagnostic


def tucker_fit(
    Z: np.ndarray,
    r_ocean: int,
    r_var: int,
    rng: np.random.Generator,
    n_iter_max: int = 100,
    tol: float = 1e-4,
) -> TuckerFit:
    """Fit a two-mode Tucker decomposition compressing the ocean-cell and variable modes of `Z`.

    (see docs/model_design.md §5.5)

    Thin wrapper around `tensorly.decomposition.partial_tucker`. The time
    mode (0) is left full-rank (DMD needs it); modes 1 (ocean cells) and 2
    (variables) are compressed jointly via Higher Order Orthogonal Iteration,
    initialized from an SVD.

    Parameters
    ----------
    Z : (T, N_ocean, V) float64
        Gaussianized, standardized anomalies. Must be fully finite: ocean
        cells are non-NaN in every modeled variable at every training
        snapshot by construction of the ocean mask.
    r_ocean : int
        Ocean-cell mode truncation rank.
    r_var : int
        Variable mode truncation rank.
    rng : np.random.Generator
        Source of randomness for `partial_tucker`'s SVD-based initialization.
    n_iter_max : int
        Maximum HOOI refinement iterations. 0 yields a pure HOSVD.
    tol : float
        Convergence tolerance on the reconstruction error between HOOI
        iterations.

    Returns
    -------
    TuckerFit
    """
    # This nested-tuple unpacking pattern (`(_core, [U_ocean, U_var]), _errors`)
    # is specific to tensorly==0.9.0's `partial_tucker` return shape; see
    # `test_tensorly_return_shape` and docs/model_design.md §9 gotcha #21.
    (_core, [U_ocean, U_var]), _errors = partial_tucker(
        Z,
        rank=[r_ocean, r_var],
        modes=[1, 2],
        init="svd",
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=rng,
    )

    Z_recon = multi_mode_dot(
        multi_mode_dot(Z, [U_ocean.T, U_var.T], modes=[1, 2]),
        [U_ocean, U_var],
        modes=[1, 2],
    )
    reconstruction_error = float(np.linalg.norm(Z_recon - Z) / np.linalg.norm(Z))

    return TuckerFit(
        U_ocean=U_ocean.astype(np.float64),
        U_var=U_var.astype(np.float64),
        r_ocean=r_ocean,
        r_var=r_var,
        reconstruction_error=reconstruction_error,
    )


def tucker_transform(Z: np.ndarray, fit: TuckerFit) -> np.ndarray:
    """Compress `Z` into the Tucker core (see docs/model_design.md §5.5).

    Parameters
    ----------
    Z : (T, N_ocean, V) float64
    fit : TuckerFit

    Returns
    -------
    core : (T, r_ocean, r_var) float64
    """
    result: np.ndarray = multi_mode_dot(Z, [fit.U_ocean.T, fit.U_var.T], modes=[1, 2])
    return result


def tucker_inverse(core: np.ndarray, fit: TuckerFit) -> np.ndarray:
    """Uncompress a Tucker core back to full ocean-cell/variable space.

    (see docs/model_design.md §5.5)

    Parameters
    ----------
    core : (T, r_ocean, r_var) float64
    fit : TuckerFit

    Returns
    -------
    Z : (T, N_ocean, V) float64
    """
    result: np.ndarray = multi_mode_dot(core, [fit.U_ocean, fit.U_var], modes=[1, 2])
    return result
