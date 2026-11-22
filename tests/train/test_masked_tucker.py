"""Tests for train.masked_tucker -- the vendored, memory-conscious masked ALS.

Verifies both fixes described in `src/train/masked_tucker.py`'s docstring
before ever running this on the real VM: (a) the in-place masking-blend
rewrite is numerically equivalent to tensorly's own `partial_tucker`, not
merely close, and doesn't mutate the caller's input; (b) the `method=svd`
propagation fix means a genuinely-applied `svd="randomized_svd"` still
recovers a planted low-rank structure through the mask.
"""

import numpy as np
from numpy.testing import assert_allclose
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot

from train.masked_tucker import partial_tucker_lowmem


def _synthetic_masked_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """An analytic low-rank (T, n_lat, n_lon) tensor with ~15% cells masked."""
    n_times, n_lat, n_lon, r_lat, r_lon = 50, 30, 40, 6, 8
    lat_true = rng.normal(size=(n_lat, r_lat))
    lon_true = rng.normal(size=(n_lon, r_lon))
    core_true = rng.normal(size=(n_times, r_lat, r_lon))
    x = np.einsum("tab,ia,jb->tij", core_true, lat_true, lon_true).astype(np.float32)
    mask = (rng.random(size=x.shape) > 0.15).astype(np.int8)
    x_observed = np.where(mask == 1, x, 0.0).astype(np.float32)
    return x_observed, mask, r_lat, r_lon


def test_masking_blend_rewrite_matches_tensorly_exactly() -> None:
    """Same `svd` method both sides isolates the blend rewrite as the only difference.

    With `svd="truncated_svd"` explicitly, tensorly's own (buggy)
    per-iteration SVD call already defaults to `"truncated_svd"` regardless
    of what's requested -- so this specific method choice makes tensorly's
    original loop and `partial_tucker_lowmem` do the identical per-mode SVD
    computation. Any difference in output can only come from the masking-
    blend rewrite, which should be numerically exact, not merely close.
    """
    rng = np.random.default_rng(0)
    x, mask, r_lat, r_lon = _synthetic_masked_case(rng)
    x_before = x.copy()

    (core_ref, factors_ref), errors_ref = partial_tucker(
        x,
        rank=[r_lat, r_lon],
        modes=[1, 2],
        init="svd",
        n_iter_max=8,
        tol=1e-6,
        random_state=0,
        mask=mask,
        svd="truncated_svd",
        svd_mask_repeats=0,
    )

    (core_lowmem, factors_lowmem), errors_lowmem = partial_tucker_lowmem(
        x,
        rank=[r_lat, r_lon],
        modes=[1, 2],
        init="svd",
        n_iter_max=8,
        tol=1e-6,
        random_state=0,
        mask=mask,
        svd="truncated_svd",
        svd_mask_repeats=0,
    )

    assert_allclose(core_lowmem, core_ref, rtol=1e-5, atol=1e-6)
    for f_lowmem, f_ref in zip(factors_lowmem, factors_ref):
        assert_allclose(f_lowmem, f_ref, rtol=1e-5, atol=1e-6)
    assert_allclose(errors_lowmem, errors_ref, rtol=1e-5, atol=1e-6)

    # The caller's input array must never be mutated in place.
    assert_allclose(x, x_before)


def test_randomized_svd_method_actually_applied_recovers_structure() -> None:
    """With the `method=svd` propagation fix, `svd="randomized_svd"` still
    recovers the planted low-rank structure through the mask.

    Doesn't assert bit-identical results against tensorly's own (buggy)
    loop here -- that loop silently never applies "randomized_svd" after
    initialization, so an exact match isn't the right check. This instead
    checks the actual thing that matters: masked reconstruction error is
    small, confirming the fix doesn't break convergence.
    """
    rng = np.random.default_rng(1)
    x, mask, r_lat, r_lon = _synthetic_masked_case(rng)

    (core, factors), errors = partial_tucker_lowmem(
        x,
        rank=[r_lat, r_lon],
        modes=[1, 2],
        init="svd",
        n_iter_max=30,
        tol=1e-8,
        random_state=1,
        mask=mask,
        svd="randomized_svd",
        svd_mask_repeats=0,
    )

    recon = multi_mode_dot(core, factors, modes=[1, 2])
    masked_err_pct = 100.0 * float(np.linalg.norm((recon - x) * mask) / np.linalg.norm(x * mask))
    # Not asserting errors[-1] < errors[0]: randomized_svd is stochastic per
    # iteration, so the error trajectory isn't guaranteed monotonic the way
    # exact-SVD-based HOOI's is -- that's expected, not a bug in the fix.
    # Final reconstruction quality is the thing that actually matters here.
    assert masked_err_pct < 1.0
