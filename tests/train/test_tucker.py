"""Tests for train.tucker (see docs/model_design.md §5.5)."""

import numpy as np
from numpy.testing import assert_allclose
from tensorly.decomposition import partial_tucker

from train.tucker import TuckerFit, tucker_fit, tucker_inverse, tucker_transform


def test_tucker_fit_roundtrip() -> None:
    """An analytic rank-r tensor is recovered (near-)exactly at rank r."""
    rng = np.random.default_rng(0)
    n_times, n_ocean, n_var = 60, 40, 3
    r_ocean, r_var = 5, 2

    u_ocean_true = rng.normal(size=(n_ocean, r_ocean))
    u_var_true = rng.normal(size=(n_var, r_var))
    core_true = rng.normal(size=(n_times, r_ocean, r_var))
    Z = np.einsum("tkm,nk,vm->tnv", core_true, u_ocean_true, u_var_true)

    fit = tucker_fit(Z, r_ocean=r_ocean, r_var=r_var, rng=rng)
    core = tucker_transform(Z, fit)
    Z_recon = tucker_inverse(core, fit)

    relative_error = np.linalg.norm(Z_recon - Z) / np.linalg.norm(Z)
    assert relative_error < 1e-4
    assert fit.reconstruction_error < 1e-4


def test_tucker_fit_factor_shapes() -> None:
    """U_ocean and U_var have the requested truncation ranks and dtype float64."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(30, 25, 4))

    fit = tucker_fit(Z, r_ocean=10, r_var=3, rng=rng)

    assert fit.U_ocean.shape == (25, 10)
    assert fit.U_var.shape == (4, 3)
    assert fit.r_ocean == 10
    assert fit.r_var == 3
    assert fit.U_ocean.dtype == np.float64
    assert fit.U_var.dtype == np.float64


def test_tucker_transform_inverse_shapes() -> None:
    """tucker_transform/tucker_inverse round trip through the expected shapes."""
    rng = np.random.default_rng(2)
    n_times = 15
    Z = rng.normal(size=(n_times, 20, 3))

    fit = tucker_fit(Z, r_ocean=6, r_var=2, rng=rng)
    core = tucker_transform(Z, fit)
    assert core.shape == (n_times, 6, 2)

    Z_recon = tucker_inverse(core, fit)
    assert Z_recon.shape == Z.shape


def test_tucker_fit_reconstruction_error_decreases_with_rank() -> None:
    """Reconstruction error at a higher r_ocean is no worse than at a lower one."""
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(80, 60, 4))

    def _relative_error(r_ocean: int) -> float:
        fit = tucker_fit(Z, r_ocean=r_ocean, r_var=3, rng=np.random.default_rng(3))
        Z_recon = tucker_inverse(tucker_transform(Z, fit), fit)
        return float(np.linalg.norm(Z_recon - Z) / np.linalg.norm(Z))

    assert _relative_error(50) <= _relative_error(10) + 1e-9


def test_tucker_fit_reconstruction_error_matches_manual_computation() -> None:
    """The stored `reconstruction_error` matches an independently computed relative error."""
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(25, 18, 3))

    fit = tucker_fit(Z, r_ocean=8, r_var=2, rng=rng)
    Z_recon = tucker_inverse(tucker_transform(Z, fit), fit)
    expected_error = float(np.linalg.norm(Z_recon - Z) / np.linalg.norm(Z))

    assert_allclose(fit.reconstruction_error, expected_error, rtol=1e-8)


def test_tensorly_return_shape() -> None:
    """`partial_tucker`'s return shape matches the pattern `tucker_fit` relies on.

    Guards against `tensorly` changing the nested-tuple structure of
    `partial_tucker`'s return value across versions (see
    docs/model_design.md §9 gotcha #21: this has drifted before). If this
    test fails, `tucker_fit`'s unpacking line —
    ``(_core, [U_ocean, U_var]), _errors = partial_tucker(...)`` — needs
    updating for the newly installed tensorly version.
    """
    rng = np.random.default_rng(4)
    n_times, n_ocean, n_var = 10, 8, 3
    r_ocean, r_var = 4, 2
    Z = rng.normal(size=(n_times, n_ocean, n_var))

    result = partial_tucker(
        Z,
        rank=[r_ocean, r_var],
        modes=[1, 2],
        init="svd",
        n_iter_max=10,
        tol=1e-4,
        random_state=rng,
    )

    assert isinstance(result, tuple) and len(result) == 2, (
        "partial_tucker no longer returns a 2-tuple of "
        "((core, factors), errors) — update tucker_fit's unpacking pattern"
    )
    core_and_factors, errors = result
    assert isinstance(core_and_factors, tuple) and len(core_and_factors) == 2, (
        "partial_tucker's first element is no longer a (core, factors) pair — "
        "update tucker_fit's unpacking pattern"
    )
    core, factors = core_and_factors
    assert isinstance(factors, list) and len(factors) == 2, (
        "partial_tucker's factors are no longer a 2-element list — "
        "update tucker_fit's unpacking pattern"
    )

    U_ocean, U_var = factors
    assert U_ocean.shape == (
        n_ocean,
        r_ocean,
    ), "U_ocean has an unexpected shape from partial_tucker"
    assert U_var.shape == (n_var, r_var), "U_var has an unexpected shape from partial_tucker"
    assert core.shape == (
        n_times,
        r_ocean,
        r_var,
    ), "core has an unexpected shape from partial_tucker"


def test_tucker_fit_returns_tucker_fit_instance() -> None:
    """`tucker_fit` returns a `TuckerFit` with a populated `reconstruction_error`."""
    rng = np.random.default_rng(6)
    Z = rng.normal(size=(12, 10, 3))

    fit = tucker_fit(Z, r_ocean=4, r_var=2, rng=rng)

    assert isinstance(fit, TuckerFit)
    assert isinstance(fit.reconstruction_error, float)
    assert fit.reconstruction_error >= 0.0
