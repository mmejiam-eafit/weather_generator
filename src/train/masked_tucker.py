"""Memory-conscious masked Tucker ALS, vendored from tensorly==0.9.0.

Fixes two issues found in `tensorly.decomposition.partial_tucker`'s masked
(land-cell) ALS loop while diagnosing task 07's Tucker rank cost cliff
(`tasks/07-tucker-dmd-rank-selection.md` §5: 20% rank completes in 8.41h,
30%/40% each hung 8-11.5+ hours with zero checkpoints) — see the plan
history for the full diagnosis:

1. **Memory.** Tensorly's per-iteration mask re-imputation,
   ``tensor = tensor * mask + multi_mode_dot(core, factors, ...) * (1 - mask)``,
   materializes ~4-5 full-size temporary arrays every iteration, identically
   at every rank (rank-independent). This repo's mask is always strictly
   ``{0, 1}`` (`train.masking`'s `tucker_mask`), so the blend is a pure
   per-element select, not a weighted average -- ``np.copyto(..., where=)``
   computes the identical result in place, roughly halving this step's
   peak memory.
2. **Correctness/speed.** Tensorly's per-iteration factor update calls
   ``svd_interface(..., n_eigenvecs=rank[index], random_state=random_state)``
   WITHOUT a ``method=`` argument -- unlike its one-time ``init="svd"``
   step, which correctly passes ``method=svd``. Every iteration after
   initialization therefore silently ignores ``svd="randomized_svd"`` and
   falls back to `svd_interface`'s default, a full economy LAPACK SVD
   (``"truncated_svd"``), regardless of what was requested. This is
   rank-*dependent* in cost and asymptotically far more expensive than a
   properly-applied randomized SVD, plausibly the dominant contributor to
   the rank-dependent part of the cost cliff.

Everything else (convergence/error tracking, the ``init=(core, factors)``
checkpoint-resume contract) is unchanged from tensorly's own loop -- this
is a small, line-diffable patch, not a rewrite. Correctness verified in
`tests/train/test_masked_tucker.py` against tensorly's own `partial_tucker`.
"""

from math import sqrt
from typing import Literal

import numpy as np
import tensorly as tl
from tensorly.base import unfold
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg.svd import svd_interface


def partial_tucker_lowmem(
    tensor: np.ndarray,
    rank: list[int],
    modes: list[int],
    n_iter_max: int,
    init: Literal["svd"] | tuple[np.ndarray, list[np.ndarray]],
    tol: float,
    svd: str,
    random_state: int | np.random.Generator | None,
    mask: np.ndarray | None = None,
    svd_mask_repeats: int = 5,
) -> tuple[tuple[np.ndarray, list[np.ndarray]], list[float]]:
    """Masked partial Tucker ALS with the two fixes described in the module docstring.

    Drop-in replacement for `tensorly.decomposition.partial_tucker` for this
    repo's exact usage: ``init`` is either ``"svd"`` (first checkpoint) or an
    already-fitted ``(core, factors)`` pair (resuming from a checkpoint) --
    ``init="random"`` is not supported, since no call site here ever uses it.

    Parameters
    ----------
    tensor : (T, D_lat, D_lon) float32
    rank : [r_lat, r_lon]
    modes : [1, 2] in every call site here (time kept full-rank)
    n_iter_max : int
        ALS iterations for THIS call (one checkpoint chunk).
    init : "svd" or (core, factors)
    tol : float
    svd : str
        e.g. "randomized_svd" -- now actually honored every iteration, not
        just at initialization.
    random_state : int | np.random.Generator | None
    mask : (T, D_lat, D_lon) int8 or None
        1 where observed, 0 where missing (land) -- matches
        `train.masking`'s `tucker_mask` convention exactly.
    svd_mask_repeats : int
        Passed to `svd_interface`'s missing-value imputation at
        initialization only (matches tensorly's own `initialize_tucker`).

    Returns
    -------
    (core, factors), rec_errors -- identical shape/structure to
    `tensorly.decomposition.partial_tucker`'s return value.
    """
    if init == "svd":
        factors = []
        for index, mode in enumerate(modes):
            mask_unfold = None if mask is None else unfold(mask, mode)
            U, _, _ = svd_interface(
                unfold(tensor, mode),
                n_eigenvecs=rank[index],
                method=svd,
                mask=mask_unfold,
                n_iter_mask_imputation=svd_mask_repeats,
                random_state=random_state,
            )
            factors.append(U)
        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)
    else:
        core, factors = init

    # Never mutate the caller's array -- tensorly's own `tensor * mask + ...`
    # creates a new array via `*`/`+`; one explicit copy here preserves that
    # contract while allowing in-place updates for the rest of this function.
    tensor = tensor.copy()
    missing = None if mask is None else mask == 0

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):
        if missing is not None:
            recon = multi_mode_dot(core, factors, modes=modes, transpose=False)
            np.copyto(tensor, recon, where=missing)
            del recon

        for index, mode in enumerate(modes):
            core_approximation = multi_mode_dot(
                tensor, factors, modes=modes, skip=index, transpose=True
            )
            eigenvecs, _, _ = svd_interface(
                unfold(core_approximation, mode),
                n_eigenvecs=rank[index],
                method=svd,  # the fix: tensorly's own loop omits this
                random_state=random_state,
            )
            factors[index] = eigenvecs

        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)

        rec_error = sqrt(abs(norm_tensor**2 - tl.norm(core, 2) ** 2)) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1 and tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            break

    return (core, factors), rec_errors
