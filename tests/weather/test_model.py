"""Tests for weather.model (see docs/model_design.md §7, §8.4, §8.5)."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import TIER1_T0, TIER1_VARIABLES, TIER1_WAVE_SHAPE, fit_small_tier1_model
from numpy.testing import assert_allclose
from sklearn.base import clone

import weather.model as model_module
from core.archive_eras import ArchiveEra
from train.dmd import dmd_fit as _real_dmd_fit
from train.local_residual import fit_local_residual as _real_fit_local_residual
from weather.model import ModelMetadata, WeatherModel


def test_fit_produces_expected_fitted_attributes(fitted_tier1_model: WeatherModel) -> None:
    """fit() populates every documented fitted attribute with sane shapes."""
    model = fitted_tier1_model

    assert model.ocean_mask_.dtype == np.bool_
    assert model.ocean_mask_.shape == TIER1_WAVE_SHAPE
    n_ocean = int(model.ocean_mask_.sum())
    assert n_ocean > 0

    assert set(model.variable_names_) == {"sigwh", "wsp", "u_wsea", "v_wsea", "u_wind", "v_wind"}
    assert model.tucker_.U_ocean.shape == (n_ocean, 10)
    assert model.tucker_.U_var.shape == (len(model.variable_names_), 4)
    assert model.dmd_.Phi.shape == (10 * 4, 8)
    assert model.dmd_.variant == "exact"
    assert model.mode_ou_.theta.shape == (8,)
    assert not np.isnan(model.mode_ou_.theta).any()
    assert not np.isnan(model.mode_ou_.sigma).any()
    assert model.local_residual_ is not None
    assert model.local_residual_.theta.shape == (n_ocean, len(model.variable_names_))
    assert not np.isnan(model.local_residual_.theta).any()
    assert isinstance(model.metadata_, ModelMetadata)
    assert model.metadata_.variables == model.variable_names_
    assert model.metadata_.dmd_variant == "exact"
    assert model.metadata_.include_trend is True
    assert model.metadata_.include_local_residual is True
    assert model.metadata_.t_train_start <= model.metadata_.t_ref <= model.metadata_.t_train_end


def test_fit_include_local_residual_false_skips_local_residual(tier1_archive: Path) -> None:
    """include_local_residual=False leaves local_residual_ set to None after fit()."""
    model = WeatherModel(
        t_start=TIER1_T0,
        t_end=TIER1_T0 + timedelta(hours=3 * 60),
        variables=TIER1_VARIABLES,
        r_ocean=10,
        r_var=4,
        r_dmd=8,
        copula_lut_bins=50,
        include_local_residual=False,
        random_state=0,
    )
    model.fit(tier1_archive)
    assert model.local_residual_ is None
    assert model.metadata_.include_local_residual is False


def test_fit_save_load_sample(tier1_archive: Path, tmp_path: Path) -> None:
    """save() then load() reconstructs an equivalent model, end to end on a synthetic archive."""
    model = fit_small_tier1_model(tier1_archive)
    model_dir = tmp_path / "model"
    model.save(model_dir)
    loaded = WeatherModel.load(model_dir)

    assert_allclose(loaded.ocean_mask_, model.ocean_mask_)
    assert loaded.variable_names_ == model.variable_names_
    # tucker.npz stores U_ocean/U_var as float32 (§7 dtype policy), so a
    # round trip loses float32-level precision even though both arrays are
    # float64 in memory before/after save+load.
    assert_allclose(loaded.tucker_.U_ocean, model.tucker_.U_ocean, rtol=1e-6, atol=1e-8)
    assert_allclose(loaded.tucker_.U_var, model.tucker_.U_var, rtol=1e-6, atol=1e-8)
    assert loaded.tucker_.reconstruction_error == model.tucker_.reconstruction_error
    assert_allclose(loaded.dmd_.Phi, model.dmd_.Phi)
    assert_allclose(loaded.dmd_.Lambda, model.dmd_.Lambda)
    assert loaded.dmd_.variant == model.dmd_.variant
    assert_allclose(loaded.mode_ou_.theta, model.mode_ou_.theta)
    assert_allclose(loaded.mode_ou_.sigma, model.mode_ou_.sigma)
    assert model.local_residual_ is not None
    assert loaded.local_residual_ is not None
    assert_allclose(loaded.local_residual_.theta, model.local_residual_.theta)
    assert_allclose(loaded.local_residual_.sigma, model.local_residual_.sigma)
    assert_allclose(loaded.local_residual_.variance_ratio, model.local_residual_.variance_ratio)
    assert_allclose(loaded.local_residual_.active_mask, model.local_residual_.active_mask)
    assert loaded.metadata_ == model.metadata_


def test_save_load_roundtrip_include_local_residual_false(
    tier1_archive: Path, tmp_path: Path
) -> None:
    """include_local_residual=False round-trips: local_residual.npz absent, local_residual_ None."""
    model = WeatherModel(
        t_start=TIER1_T0,
        t_end=TIER1_T0 + timedelta(hours=3 * 60),
        variables=TIER1_VARIABLES,
        r_ocean=10,
        r_var=4,
        r_dmd=8,
        copula_lut_bins=50,
        include_local_residual=False,
        random_state=0,
    )
    model.fit(tier1_archive)
    assert model.local_residual_ is None

    model_dir = tmp_path / "model"
    model.save(model_dir)
    assert not (model_dir / "local_residual.npz").exists()

    loaded = WeatherModel.load(model_dir)
    assert loaded.local_residual_ is None


def test_metadata_json_roundtrips_new_fields(tier1_archive: Path, tmp_path: Path) -> None:
    """metadata.json round-trips dmd_variant, include_trend, and t_ref correctly."""
    model = WeatherModel(
        t_start=TIER1_T0,
        t_end=TIER1_T0 + timedelta(hours=3 * 60),
        variables=TIER1_VARIABLES,
        r_ocean=10,
        r_var=4,
        r_dmd=8,
        copula_lut_bins=50,
        dmd_variant="exact",
        include_trend=True,
        random_state=0,
    )
    model.fit(tier1_archive)

    model_dir = tmp_path / "model"
    model.save(model_dir)

    metadata_dict = json.loads((model_dir / "metadata.json").read_text())
    assert metadata_dict["dmd_variant"] == "exact"
    assert metadata_dict["include_trend"] is True
    assert metadata_dict["t_ref"] == model.metadata_.t_ref.isoformat()

    loaded = WeatherModel.load(model_dir)
    assert loaded.metadata_.dmd_variant == "exact"
    assert loaded.metadata_.include_trend is True
    assert loaded.metadata_.t_ref == model.metadata_.t_ref


def test_from_components_matches_fit(fitted_tier1_model: WeatherModel) -> None:
    """A model built via from_components from another model's fitted pieces matches it."""
    fitted = fitted_tier1_model

    rebuilt = WeatherModel.from_components(
        preprocess=fitted.preprocess_,
        tucker=fitted.tucker_,
        dmd=fitted.dmd_,
        mode_ou=fitted.mode_ou_,
        local_residual=fitted.local_residual_,
        ocean_mask=fitted.ocean_mask_,
        metadata=fitted.metadata_,
    )

    assert rebuilt.t_start == fitted.t_start
    assert rebuilt.t_end == fitted.t_end
    assert tuple(rebuilt.variables) == fitted.variable_names_
    assert_allclose(rebuilt.ocean_mask_, fitted.ocean_mask_)
    assert_allclose(rebuilt.tucker_.U_ocean, fitted.tucker_.U_ocean)
    assert_allclose(rebuilt.dmd_.Phi, fitted.dmd_.Phi)
    assert_allclose(rebuilt.mode_ou_.theta, fitted.mode_ou_.theta)
    assert rebuilt.local_residual_ is not None
    assert fitted.local_residual_ is not None
    assert_allclose(rebuilt.local_residual_.theta, fitted.local_residual_.theta)


def test_transform_inverse_roundtrip(tier1_archive: Path, fitted_tier1_model: WeatherModel) -> None:
    """preprocess_.inverse_transform(preprocess_.transform(X)) reproduces X on wave-native variables.

    Some quantization error is expected (see train.copula's own round-trip
    tests): the copula LUT holds only `copula_lut_bins` quantiles fit from
    just `TIER1_N_TIMES` samples here, so recovery is exact only at those
    quantiles. Reprojected variables (was/wad_come_from -> u_wind/v_wind) are
    excluded: Reprojector's inverse is deliberately the identity (see its
    docstring), so only wave-native variables round-trip.
    """
    model = fitted_tier1_model

    from train.ingest import load_training_window

    x_raw = load_training_window(tier1_archive, model.t_start, model.t_end, model.variables)
    gaussianized = model.preprocess_.transform(x_raw)
    recovered = model.preprocess_.inverse_transform(gaussianized)

    for variable in ["sigwh", "wsh", "wsp"]:
        error = np.abs(recovered[variable].values - x_raw[variable].values)
        assert np.median(error) < 0.05


def test_transform_defaults_to_training_window(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """transform(X) without explicit bounds uses the training window."""
    result = fitted_tier1_model.transform(tier1_archive)
    assert result["sigwh"].shape[0] == 60


def test_ocean_mask_accessor(fitted_tier1_model: WeatherModel) -> None:
    """ocean_mask() returns the fitted mask."""
    assert_allclose(fitted_tier1_model.ocean_mask(), fitted_tier1_model.ocean_mask_)


def test_sklearn_clone() -> None:
    """clone(model) produces a fresh unfitted copy with matching params.

    Every __init__ param (§9 gotcha 16) must round-trip through
    get_params()/set_params() unmodified — including the new
    dmd_variant/include_trend/trend_significance_threshold/
    include_local_residual/local_residual_variance_floor params.
    """
    model = WeatherModel(
        t_start=TIER1_T0,
        t_end=TIER1_T0 + timedelta(days=1),
        variables=TIER1_VARIABLES,
        r_ocean=10,
        r_var=4,
        r_dmd=8,
        dmd_variant="bop",
        copula_lut_bins=250,
        diurnal_variables=["sst"],
        include_trend=False,
        trend_significance_threshold=0.25,
        include_local_residual=False,
        local_residual_variance_floor=0.05,
        random_state=42,
    )
    cloned = clone(model)
    assert isinstance(cloned, WeatherModel)
    assert cloned.t_start == TIER1_T0
    assert cloned.t_end == TIER1_T0 + timedelta(days=1)
    assert cloned.variables == TIER1_VARIABLES
    assert cloned.r_ocean == 10
    assert cloned.r_var == 4
    assert cloned.r_dmd == 8
    assert cloned.dmd_variant == "bop"
    assert cloned.copula_lut_bins == 250
    assert cloned.diurnal_variables == ["sst"]
    assert cloned.include_trend is False
    assert cloned.trend_significance_threshold == 0.25
    assert cloned.include_local_residual is False
    assert cloned.local_residual_variance_floor == 0.05
    assert cloned.random_state == 42
    assert not hasattr(cloned, "preprocess_")

    original_params = model.get_params()
    cloned_params = cloned.get_params()
    assert original_params == cloned_params


def test_fit_derives_dt_from_archive_era_not_flat_constant(
    tier1_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dt for dmd_fit/fit_local_residual comes from core.archive_eras.era_for_timestamp,
    not a reimported flat SNAPSHOT_CADENCE_SECONDS constant (see
    tasks/16-archive-grid-resolution-drift.md).
    """
    sentinel_cadence = 999.0
    fake_era = ArchiveEra(
        name="fake",
        start=TIER1_T0 - timedelta(days=1),
        end=None,
        wave_grid_shape=TIER1_WAVE_SHAPE,
        wind_grid_shape=TIER1_WAVE_SHAPE,
        ocean_grid_shapes={},
        cadence_seconds=sentinel_cadence,
    )
    monkeypatch.setattr(model_module, "era_for_timestamp", lambda t: fake_era)

    captured_dts: list[float] = []

    def spy_dmd_fit(*args: Any, **kwargs: Any) -> Any:
        captured_dts.append(kwargs["dt"])
        return _real_dmd_fit(*args, **kwargs)

    def spy_fit_local_residual(*args: Any, **kwargs: Any) -> Any:
        captured_dts.append(kwargs["dt"])
        return _real_fit_local_residual(*args, **kwargs)

    monkeypatch.setattr(model_module, "dmd_fit", spy_dmd_fit)
    monkeypatch.setattr(model_module, "fit_local_residual", spy_fit_local_residual)

    fit_small_tier1_model(tier1_archive)

    assert len(captured_dts) == 2
    assert captured_dts == [sentinel_cadence, sentinel_cadence]
