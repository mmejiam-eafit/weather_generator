"""Tests for train.directional (see docs/model_design.md §2.4, §3.7, §5.3)."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from sklearn.base import clone

from train.directional import ComingFromToUV, GoingToToUV

_SPEED_VAL = 10.0
_CARDINAL_THETAS = np.array([0.0, 90.0, 180.0, 270.0])


def _cardinal_dataset(speed_name: str, direction_name: str) -> xr.Dataset:
    """A tiny dataset with `speed_name` = 10 and `direction_name` at the four cardinal angles."""
    return xr.Dataset(
        {
            speed_name: (
                ("time", "wave_lat", "wave_lon"),
                np.full((4, 1, 1), _SPEED_VAL, dtype=np.float64),
            ),
            direction_name: (
                ("time", "wave_lat", "wave_lon"),
                _CARDINAL_THETAS.reshape(4, 1, 1),
            ),
        },
        coords={"time": np.arange(4), "wave_lat": [0.0], "wave_lon": [0.0]},
    )


# ---------------------------------------------------------------------------
# ComingFromToUV
# ---------------------------------------------------------------------------


def test_cardinal_signs_coming_from_to_uv() -> None:
    """theta_from ∈ {0°, 90°, 180°, 270°} must produce the coming-from (u, v) table (§2.4).

    Load-bearing: catches sign flips before they propagate to every
    downstream (u, v) field.
    """
    ds = _cardinal_dataset("wsh", "wsd")
    transformer = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")])

    Z = transformer.fit_transform(ds)

    expected_u = np.array([0.0, -_SPEED_VAL, 0.0, _SPEED_VAL])
    expected_v = np.array([-_SPEED_VAL, 0.0, _SPEED_VAL, 0.0])
    assert_allclose(Z["u_wsea"].values.ravel(), expected_u, atol=1e-10)
    assert_allclose(Z["v_wsea"].values.ravel(), expected_v, atol=1e-10)


def test_uv_roundtrip_coming_from_to_uv() -> None:
    """(speed, theta_from) → (u, v) → (speed', theta_from') reproduces input within 1e-10."""
    rng = np.random.default_rng(42)
    speed = rng.uniform(0.1, 25.0, size=(2, 3, 4))
    theta = rng.uniform(0.0, 360.0, size=(2, 3, 4))
    ds = xr.Dataset(
        {
            "wsh": (("time", "wave_lat", "wave_lon"), speed),
            "wsd": (("time", "wave_lat", "wave_lon"), theta),
        },
        coords={"time": np.arange(2), "wave_lat": np.arange(3.0), "wave_lon": np.arange(4.0)},
    )
    transformer = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")])

    Z = transformer.fit_transform(ds)
    recovered = transformer.inverse_transform(Z)

    assert_allclose(recovered["wsh"].values, speed, atol=1e-10)
    assert_allclose(recovered["wsd"].values, theta, atol=1e-10)


def test_inverse_transform_returns_native_convention_coming_from_to_uv(
    small_wave_dataset: xr.Dataset,
) -> None:
    """inverse_transform reconstructs theta_from, not theta_to (§2.4)."""
    transformer = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")])
    Z = transformer.fit_transform(small_wave_dataset)
    recovered = transformer.inverse_transform(Z)

    assert_allclose(recovered["wsh"].values, small_wave_dataset["wsh"].values, atol=1e-10)
    assert_allclose(recovered["wsd"].values, small_wave_dataset["wsd"].values, atol=1e-10)


def test_pass_through_non_paired_vars_coming_from_to_uv(
    small_wave_dataset: xr.Dataset, small_wind_dataset: xr.Dataset
) -> None:
    """Non-directional vars and pairs owned by GoingToToUV pass through untouched (§3.7, §5.3)."""
    combined = xr.merge([small_wave_dataset, small_wind_dataset])
    transformer = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")])

    Z = transformer.fit_transform(combined)

    assert_allclose(Z["sigwh"].values, combined["sigwh"].values)
    assert_allclose(Z["was"].values, combined["was"].values)
    assert_allclose(Z["wad"].values, combined["wad"].values)
    assert "wsh" not in Z.data_vars
    assert "wsd" not in Z.data_vars
    assert "u_wsea" in Z.data_vars
    assert "v_wsea" in Z.data_vars
    assert "u_wind" not in Z.data_vars
    assert "v_wind" not in Z.data_vars


def test_sklearn_clone_coming_from_to_uv() -> None:
    """clone() produces a fresh unfitted copy (no __init__ side effects to break it)."""
    transformer = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")])
    cloned = clone(transformer)
    assert isinstance(cloned, ComingFromToUV)
    assert not hasattr(cloned, "pairs_")


# ---------------------------------------------------------------------------
# GoingToToUV
# ---------------------------------------------------------------------------


def test_cardinal_signs_going_to_to_uv() -> None:
    """theta_to ∈ {0°, 90°, 180°, 270°} must produce the going-to (u, v) table (§2.4).

    Load-bearing: catches sign flips before they propagate to every
    downstream (u, v) field.
    """
    ds = _cardinal_dataset("was", "wad")
    transformer = GoingToToUV(pairs=[("was", "wad", "u_wind", "v_wind")])

    Z = transformer.fit_transform(ds)

    expected_u = np.array([0.0, _SPEED_VAL, 0.0, -_SPEED_VAL])
    expected_v = np.array([_SPEED_VAL, 0.0, -_SPEED_VAL, 0.0])
    assert_allclose(Z["u_wind"].values.ravel(), expected_u, atol=1e-10)
    assert_allclose(Z["v_wind"].values.ravel(), expected_v, atol=1e-10)


def test_uv_roundtrip_going_to_to_uv() -> None:
    """(speed, theta_to) → (u, v) → (speed', theta_to') reproduces input within 1e-10."""
    rng = np.random.default_rng(43)
    speed = rng.uniform(0.1, 25.0, size=(2, 3, 4))
    theta = rng.uniform(0.0, 360.0, size=(2, 3, 4))
    ds = xr.Dataset(
        {
            "was": (("time", "wind_lat", "wind_lon"), speed),
            "wad": (("time", "wind_lat", "wind_lon"), theta),
        },
        coords={"time": np.arange(2), "wind_lat": np.arange(3.0), "wind_lon": np.arange(4.0)},
    )
    transformer = GoingToToUV(pairs=[("was", "wad", "u_wind", "v_wind")])

    Z = transformer.fit_transform(ds)
    recovered = transformer.inverse_transform(Z)

    assert_allclose(recovered["was"].values, speed, atol=1e-10)
    assert_allclose(recovered["wad"].values, theta, atol=1e-10)


def test_inverse_transform_returns_native_convention_going_to_to_uv(
    small_wind_dataset: xr.Dataset,
) -> None:
    """inverse_transform reconstructs theta_to, not theta_from (§2.4)."""
    transformer = GoingToToUV(pairs=[("was", "wad", "u_wind", "v_wind")])
    Z = transformer.fit_transform(small_wind_dataset)
    recovered = transformer.inverse_transform(Z)

    assert_allclose(recovered["was"].values, small_wind_dataset["was"].values, atol=1e-10)
    assert_allclose(recovered["wad"].values, small_wind_dataset["wad"].values, atol=1e-10)


def test_pass_through_non_paired_vars_going_to_to_uv(
    small_wave_dataset: xr.Dataset, small_wind_dataset: xr.Dataset
) -> None:
    """Non-directional vars and pairs owned by ComingFromToUV pass through untouched (§3.7, §5.3)."""
    combined = xr.merge([small_wave_dataset, small_wind_dataset])
    transformer = GoingToToUV(pairs=[("was", "wad", "u_wind", "v_wind")])

    Z = transformer.fit_transform(combined)

    assert_allclose(Z["sigwh"].values, combined["sigwh"].values)
    assert_allclose(Z["wsh"].values, combined["wsh"].values)
    assert_allclose(Z["wsd"].values, combined["wsd"].values)
    assert "was" not in Z.data_vars
    assert "wad" not in Z.data_vars
    assert "u_wind" in Z.data_vars
    assert "v_wind" in Z.data_vars
    assert "u_wsea" not in Z.data_vars
    assert "v_wsea" not in Z.data_vars


def test_sklearn_clone_going_to_to_uv() -> None:
    """clone() produces a fresh unfitted copy (no __init__ side effects to break it)."""
    transformer = GoingToToUV(pairs=[("was", "wad", "u_wind", "v_wind")])
    cloned = clone(transformer)
    assert isinstance(cloned, GoingToToUV)
    assert not hasattr(cloned, "pairs_")


# ---------------------------------------------------------------------------
# keep_speed (class-C heights survive as their own variable, §5.3)
# ---------------------------------------------------------------------------


def test_coming_from_to_uv_keep_speed_retains_magnitude() -> None:
    """A kept speed stays in the Dataset next to (u, v); its direction is still replaced."""
    ds = _cardinal_dataset("wsh", "wsd")
    transformer = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")], keep_speed=("wsh",))

    Z = transformer.fit_transform(ds)

    assert {"wsh", "u_wsea", "v_wsea"} <= set(Z.data_vars)
    assert "wsd" not in Z.data_vars
    assert_allclose(Z["wsh"].values, ds["wsh"].values)


def test_coming_from_to_uv_keep_speed_inverse_uses_kept_magnitude() -> None:
    """With a kept speed, the inverse takes magnitude from it and only direction from (u, v)."""
    ds = _cardinal_dataset("wsh", "wsd")
    transformer = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")], keep_speed=("wsh",))
    Z = transformer.fit_transform(ds)
    Z["wsh"] = Z["wsh"] * 0.7  # differs from hypot(u, v) = 10 on purpose

    recovered = transformer.inverse_transform(Z)

    assert_allclose(recovered["wsh"].values, 0.7 * _SPEED_VAL)
    assert_allclose(recovered["wsd"].values.ravel(), _CARDINAL_THETAS, atol=1e-10)


def test_going_to_to_uv_keep_speed_inverse_uses_kept_magnitude() -> None:
    """Same contract for the going-to transformer."""
    ds = _cardinal_dataset("ocs", "ocd")
    transformer = GoingToToUV(pairs=[("ocs", "ocd", "u_current", "v_current")], keep_speed=("ocs",))
    Z = transformer.fit_transform(ds)
    Z["ocs"] = Z["ocs"] * 0.5

    recovered = transformer.inverse_transform(Z)

    assert "ocd" not in Z.data_vars
    assert_allclose(recovered["ocs"].values, 0.5 * _SPEED_VAL)
    assert_allclose(recovered["ocd"].values.ravel(), _CARDINAL_THETAS, atol=1e-10)


def test_coming_from_to_uv_keep_speed_rejects_unknown_variable() -> None:
    """keep_speed must name the speed of an owned pair; validated lazily in fit()."""
    ds = _cardinal_dataset("wsh", "wsd")
    transformer = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")], keep_speed=("wsd",))

    with pytest.raises(ValueError, match="keep_speed"):
        transformer.fit(ds)
