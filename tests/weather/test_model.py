"""Tests for weather.model (see docs/model_design.md §5.1, §7, §8.4, §8.5)."""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    TIER1_ABSENT_CELL,
    TIER1_LAND_CELL,
    TIER1_N_TIMES,
    TIER1_T0,
    TIER1_TENSORS,
    TIER1_VARIABLES,
    TIER1_WAVE_SHAPE,
    fit_small_tier1_model,
    small_tier1_model,
)
from numpy.testing import assert_allclose
from pytest_mock import MockerFixture
from sklearn.base import clone

from core.archive_eras import ArchiveEra
from train.ingest import load_training_window
from train.linear_inverse import GroupDiagnostics
from weather.model import ModelMetadata, WeatherModel, build_preprocess, resolve_groups


def test_fit_produces_expected_fitted_attributes(fitted_tier1_model: WeatherModel) -> None:
    """fit() populates every §8.5 fitted attribute with per-tensor and per-group shapes."""
    model = fitted_tier1_model

    assert set(model.tensor_names_) == TIER1_TENSORS
    assert model.groups_ == {t: (t,) for t in model.tensor_names_}
    for tensor in model.tensor_names_:
        assert model.tucker_[tensor].factor_lat.shape == (TIER1_WAVE_SHAPE[0], 2)
        assert model.tucker_[tensor].factor_lon.shape == (TIER1_WAVE_SHAPE[1], 2)
        assert tensor in model.core_copula_
    for group in model.groups_:
        n_modes = model.dmd_[group].Lambda.size
        assert model.dmd_[group].Phi.shape == (4, n_modes)  # n_G = 2 x 2
        assert model.dmd_[group].variant == "exact"
        assert np.all(np.isfinite(model.mode_ou_[group].innovation_cov))
        assert isinstance(model.group_diagnostics_[group], GroupDiagnostics)
    assert model.local_residual_ is None
    assert isinstance(model.metadata_, ModelMetadata)
    assert set(model.metadata_.tensors) == TIER1_TENSORS
    assert model.metadata_.tucker_ranks == {t: (2, 2) for t in model.tensor_names_}
    assert model.metadata_.include_local_residual is False
    assert model.metadata_.t_train_start <= model.metadata_.t_ref <= model.metadata_.t_train_end


def test_fit_ocean_mask_comes_from_sigwh_only(fitted_tier1_model: WeatherModel) -> None:
    """sigwh's land cell is masked; a cell with an absent partition stays ocean (task 17)."""
    mask = fitted_tier1_model.ocean_mask_

    assert mask.shape == TIER1_WAVE_SHAPE
    assert not mask[TIER1_LAND_CELL]
    assert mask[TIER1_ABSENT_CELL]
    assert int(mask.sum()) == mask.size - 1


def test_fit_include_local_residual_true_raises(tier1_archive: Path) -> None:
    """The local residual is deferred in v3 (§5.7); asking for it fails loudly."""
    with pytest.raises(ValueError, match="deferred"):
        small_tier1_model(include_local_residual=True).fit(tier1_archive)


def test_fit_joint_group_shares_one_dmd_state(tier1_archive: Path) -> None:
    """A two-member group fits one DMD over both members' cores (§5.6)."""
    groups = {"wind": ["u_wind", "v_wind"]}
    groups.update({t: [t] for t in TIER1_TENSORS - {"u_wind", "v_wind"}})

    model = small_tier1_model(groups=groups).fit(tier1_archive)

    assert model.dmd_["wind"].Phi.shape[0] == 8  # two 2 x 2 cores
    assert model.group_diagnostics_["wind"].members == ("u_wind", "v_wind")


def test_resolve_groups_rejects_a_tensor_left_out() -> None:
    """Every tensor needs dynamics, so groups must cover each exactly once (§5.6.2)."""
    with pytest.raises(ValueError, match="missing: \\['b'\\]"):
        resolve_groups({"g": ["a"]}, ("a", "b"))


def test_build_preprocess_tier1_yields_the_seven_tier1_tensors(tier1_archive: Path) -> None:
    """Tier-1 raw variables become the seven tier-1 tensors (§3.8, §10)."""
    x_raw = load_training_window(
        tier1_archive,
        TIER1_T0,
        TIER1_T0 + timedelta(hours=3 * TIER1_N_TIMES),
        TIER1_VARIABLES,
    )

    a_std = build_preprocess(TIER1_VARIABLES).fit_transform(x_raw)

    assert {str(name) for name in a_std.data_vars} == TIER1_TENSORS


def test_transform_returns_standardized_anomalies_per_tensor(
    tier1_archive: Path, fitted_tier1_model: WeatherModel
) -> None:
    """transform() gives one tensor per variable: unit scale at ocean cells, land NaN."""
    a_std = fitted_tier1_model.transform(tier1_archive)
    ocean = fitted_tier1_model.ocean_mask_

    assert {str(name) for name in a_std.data_vars} == TIER1_TENSORS
    for tensor in TIER1_TENSORS:
        values = a_std[tensor].values
        assert values.shape[0] == TIER1_N_TIMES
        assert np.all(np.isfinite(values[:, ocean]))
        assert np.all(np.isnan(values[:, ~ocean]))
        assert_allclose(np.std(values[:, ocean], axis=0), 1.0, rtol=1e-8)


def test_transform_inverse_roundtrip(tier1_archive: Path, fitted_tier1_model: WeatherModel) -> None:
    """preprocess_.inverse_transform(transform(X)) recovers wave-native variables.

    Exact up to rounding where the height is above the 1 cm log floor;
    absent partitions come back as height 0 (§5.3.1). Wind is excluded:
    Reprojector's inverse is the identity (see its docstring).
    """
    model = fitted_tier1_model
    x_raw = load_training_window(tier1_archive, model.t_start, model.t_end, model.variables)

    recovered = model.preprocess_.inverse_transform(model.preprocess_.transform(x_raw))

    for variable, floored in [("sigwh", True), ("wsh", True), ("wsp", False)]:
        raw = x_raw[variable].values
        present = np.isfinite(raw) & (raw > 0.01 if floored else True)
        assert_allclose(recovered[variable].values[present], raw[present], rtol=1e-9)
    absent = np.isnan(x_raw["wsh"].values) & model.ocean_mask_[None, :, :]
    assert absent.any()
    assert np.all(recovered["wsh"].values[absent] == 0.0)


def test_fit_save_load_roundtrip(tier1_archive: Path, tmp_path: Path) -> None:
    """save() then load() reconstructs every component (§7)."""
    model = fit_small_tier1_model(tier1_archive)
    model_dir = tmp_path / "model"
    model.save(model_dir)

    loaded = WeatherModel.load(model_dir)

    assert loaded.metadata_ == model.metadata_
    assert loaded.groups_ == model.groups_
    assert_allclose(loaded.ocean_mask_, model.ocean_mask_)
    for tensor in model.tensor_names_:
        # Factors are float32 on disk (§7 dtype policy).
        assert_allclose(
            loaded.tucker_[tensor].factor_lat, model.tucker_[tensor].factor_lat, rtol=1e-6
        )
        assert loaded.tucker_[tensor].n_iter == model.tucker_[tensor].n_iter
        assert_allclose(
            loaded.core_copula_[tensor].sorted_anomalies_binned_[tensor],
            model.core_copula_[tensor].sorted_anomalies_binned_[tensor],
        )
    for group in model.groups_:
        assert_allclose(loaded.dmd_[group].Phi, model.dmd_[group].Phi)
        assert loaded.dmd_[group].variant == model.dmd_[group].variant
        assert_allclose(loaded.mode_ou_[group].innovation_cov, model.mode_ou_[group].innovation_cov)
        assert_allclose(loaded.mode_ou_[group].partner, model.mode_ou_[group].partner)
        loaded_diag = loaded.group_diagnostics_[group]
        diag = model.group_diagnostics_[group]
        assert_allclose(loaded_diag.lim.G, diag.lim.G)
        assert_allclose(loaded_diag.tau.error_variance_ratio, diag.tau.error_variance_ratio)
        assert_allclose(loaded_diag.tau_consistent, diag.tau_consistent)
    assert loaded.local_residual_ is None


def test_save_writes_the_documented_artifact_layout(
    fitted_tier1_model: WeatherModel, tmp_path: Path
) -> None:
    """One file per tensor and per group under the §7 directory names."""
    model_dir = tmp_path / "model"
    fitted_tier1_model.save(model_dir)

    for name in ("metadata.json", "ocean_mask.npy", "preprocess.pkl", "group_diagnostics.json"):
        assert (model_dir / name).exists()
    for tensor in fitted_tier1_model.tensor_names_:
        assert (model_dir / "tucker" / f"{tensor}.npz").exists()
        assert (model_dir / "core_copula" / f"{tensor}.npz").exists()
    for group in fitted_tier1_model.groups_:
        for sub in ("dmd", "mode_ou", "lim"):
            assert (model_dir / sub / f"{group}.npz").exists()
    assert not (model_dir / "local_residual.npz").exists()
    metadata = json.loads((model_dir / "metadata.json").read_text())
    assert metadata["t_ref"] == fitted_tier1_model.metadata_.t_ref.isoformat()
    assert set(metadata["tensors"]) == TIER1_TENSORS


def test_from_components_matches_fit(fitted_tier1_model: WeatherModel) -> None:
    """A model built via from_components from another model's fitted pieces matches it."""
    fitted = fitted_tier1_model

    rebuilt = WeatherModel.from_components(
        preprocess=fitted.preprocess_,
        tucker=fitted.tucker_,
        core_copula=fitted.core_copula_,
        dmd=fitted.dmd_,
        mode_ou=fitted.mode_ou_,
        group_diagnostics=fitted.group_diagnostics_,
        fidelity_diagnostics=fitted.fidelity_diagnostics_,
        ocean_mask=fitted.ocean_mask_,
        metadata=fitted.metadata_,
    )

    assert rebuilt.t_start == fitted.t_start
    assert rebuilt.t_end == fitted.t_end
    assert rebuilt.variables == TIER1_VARIABLES
    assert rebuilt.tensor_names_ == fitted.tensor_names_
    assert rebuilt.groups_ == fitted.groups_
    assert_allclose(rebuilt.ocean_mask_, fitted.ocean_mask_)


def test_ocean_mask_accessor(fitted_tier1_model: WeatherModel) -> None:
    """ocean_mask() returns the fitted mask."""
    assert_allclose(fitted_tier1_model.ocean_mask(), fitted_tier1_model.ocean_mask_)


def test_sklearn_clone() -> None:
    """clone(model) produces a fresh unfitted copy with every __init__ param unmodified (§9.16)."""
    model = WeatherModel(
        t_start=TIER1_T0,
        t_end=TIER1_T0 + timedelta(days=1),
        variables=TIER1_VARIABLES,
        tucker_rank_fraction=0.3,
        tucker_max_iter=50,
        tucker_tol=1e-4,
        r_dmd=8,
        dmd_variant="bop",
        copula_lut_bins=250,
        diurnal_variables=["sst"],
        include_trend=False,
        trend_significance_threshold=0.25,
        include_local_residual=False,
        local_residual_variance_floor=0.05,
        groups={"wind": ["u_wind", "v_wind"]},
        lim_rank=5,
        tau_test_lags=(1, 2),
        n_bootstrap=7,
        random_state=42,
    )

    cloned = clone(model)

    assert isinstance(cloned, WeatherModel)
    assert cloned.get_params() == model.get_params()
    assert cloned.groups == {"wind": ["u_wind", "v_wind"]}
    assert cloned.tau_test_lags == (1, 2)
    assert not hasattr(cloned, "preprocess_")


def test_fit_derives_dt_from_archive_era_not_flat_constant(
    tier1_archive: Path, mocker: MockerFixture
) -> None:
    """DMD, mode OU and the LIM all use the window era's cadence (tasks/16).

    Patches the era lookup `weather.model` uses (this test predates the v3
    rewrite and kept its patch) with a sentinel cadence no flat constant
    could produce.
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
    mocker.patch("weather.model.era_for_timestamp", return_value=fake_era)

    model = fit_small_tier1_model(tier1_archive)

    for group in model.groups_:
        assert model.dmd_[group].dt == sentinel_cadence
        assert model.mode_ou_[group].dt == sentinel_cadence
        assert model.group_diagnostics_[group].lim.dt == sentinel_cadence
