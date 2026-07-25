"""WeatherModel: sklearn-compatible wrapper gluing preprocessing to Tucker/DMD/OU.

(see docs/model_design.md §7, §8.4, §8.5)
"""

import importlib.metadata
import json
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

import joblib
import numpy as np
import sklearn
import tensorly
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from core.archive_eras import era_for_timestamp
from train.climatology import ClimatologyRemover
from train.copula import GaussianCopula
from train.directional import ComingFromToUV, GoingToToUV
from train.dmd import DMDFit, ModeOUFit, dmd_fit, fit_mode_ou
from train.ingest import load_training_window
from train.local_residual import LocalResidualFit, coherent_reconstruction, fit_local_residual
from train.reproject import Reprojector
from train.standardize import PerCellStandardizer
from train.tucker import TuckerFit, tucker_fit, tucker_transform


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadata recorded at fit time (see docs/model_design.md §7, §8.5)."""

    trained_at: datetime
    t_train_start: datetime
    t_train_end: datetime
    t_ref: datetime  # climatology trend reference epoch (training window midpoint)
    variables: tuple[str, ...]
    r_ocean: int
    r_var: int
    r_dmd: int
    dmd_variant: str
    include_trend: bool
    include_local_residual: bool
    git_commit: str
    sklearn_version: str
    numpy_version: str
    tensorly_version: str
    pydmd_version: str


def _current_git_commit() -> str:
    """Best-effort current git commit hash; ``"unknown"`` outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError, OSError:
        return "unknown"


def _mask_and_stack(gaussianized: xr.Dataset) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Build the ocean mask and ravel Gaussianized variables into ``(T, N_ocean, V)``.

    (see docs/model_design.md §5.1)

    The mask is the intersection of not-NaN across every training
    snapshot and every modeled variable on the (post-reprojection) wave
    grid: a cell that is NaN in any modeled variable at even one
    snapshot (land, or ice-covered mid-training) is excluded.

    Parameters
    ----------
    gaussianized : xr.Dataset
        Output of the fitted preprocessing Pipeline, dims
        ``(time, wave_lat, wave_lon)`` per variable.

    Returns
    -------
    ocean_mask : (n_lat, n_lon) bool
    variable_names : tuple[str, ...]
        `gaussianized`'s variables, in the order stacked into `Z`.
    Z : (T, N_ocean, V) float64
    """
    variable_names = tuple(str(name) for name in gaussianized.data_vars)
    spatial_shape = gaussianized[variable_names[0]].shape[1:]

    ocean_mask = np.ones(spatial_shape, dtype=bool)
    for name in variable_names:
        ocean_mask &= ~np.isnan(gaussianized[name].values).any(axis=0)

    n_times = gaussianized[variable_names[0]].shape[0]
    n_ocean = int(ocean_mask.sum())
    z = np.empty((n_times, n_ocean, len(variable_names)), dtype=np.float64)
    for i, name in enumerate(variable_names):
        z[:, :, i] = gaussianized[name].values[:, ocean_mask]

    return ocean_mask, variable_names, z


def _log_trend_significance(
    climatology: ClimatologyRemover,
    standardize: PerCellStandardizer,
    threshold: float,
) -> None:
    """Log the fraction of cells whose linear trend is significant relative to its noise floor.

    (see docs/model_design.md §3.9, §5.4)

    Per variable, ``ratio = |c_lin_[v]| / sigma_cell_[v]``: the fitted trend
    slope's magnitude relative to that cell's temporal standard deviation.
    `climatology.c_lin_` only has entries when `climatology.include_trend`
    was `True` at fit time, so this is a no-op (nothing logged) when the
    trend term was disabled.

    Parameters
    ----------
    climatology : ClimatologyRemover
        Fitted; consulted for `c_lin_`.
    standardize : PerCellStandardizer
        Fitted; consulted for `sigma_cell_`.
    threshold : float
        Ratio above which a cell's trend is considered significant.
    """
    for variable, c_lin in climatology.c_lin_.items():
        sigma = standardize.sigma_cell_[variable]
        ratio = np.abs(c_lin) / sigma
        exceeds_fraction = float(np.nanmean(ratio > threshold))
        print(
            f"trend significance ({variable!r}): {exceeds_fraction:.1%} of cells exceed "
            f"|c_lin|/sigma_cell > {threshold}",
            file=sys.stderr,
        )


class WeatherModel(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Stochastic weather generator with an sklearn-compatible transformer surface.

    (see docs/model_design.md §8.5)

    sklearn surface (via BaseEstimator + TransformerMixin):
        ``__init__`` carries hyperparameters only.
        ``fit(X, y=None)`` — end-to-end training.
        ``transform(X)`` — runs the preprocessing chain only.
        ``fit_transform(X, y=None)`` — fit and transform in one pass, reusing
            the anomalies computed during fit.
        ``get_params()`` / ``set_params()`` — inherited from BaseEstimator.

    Custom surface: ``save``, ``load``, ``from_components``.

    Fitted attributes (trailing underscore per sklearn convention):
        preprocess_, tucker_, dmd_, mode_ou_, local_residual_, ocean_mask_,
        variable_names_, metadata_

        `local_residual_` is `None` when fit with `include_local_residual=False`.

    Sampling is **not** a method on this class — use the free functions in
    `weather.simulate` (`sample`, `sample_batch`, `simulate_field`).
    """

    def __init__(
        self,
        t_start: datetime,
        t_end: datetime,
        variables: list[str],
        r_ocean: int = 300,
        r_var: int = 6,
        r_dmd: int = 100,
        dmd_variant: Literal["exact", "bop", "hankel"] = "exact",
        copula_lut_bins: int = 500,
        diurnal_variables: list[str] | None = None,
        include_trend: bool = True,
        trend_significance_threshold: float = 0.1,
        include_local_residual: bool = True,
        local_residual_variance_floor: float = 0.02,
        random_state: int | None = None,
    ) -> None:
        # sklearn convention: __init__ only stores parameters, verbatim, no
        # validation or side effects — sklearn.base.clone() relies on this.
        self.t_start = t_start
        self.t_end = t_end
        self.variables = variables
        self.r_ocean = r_ocean
        self.r_var = r_var
        self.r_dmd = r_dmd
        self.dmd_variant = dmd_variant
        self.copula_lut_bins = copula_lut_bins
        self.diurnal_variables = diurnal_variables
        self.include_trend = include_trend
        self.trend_significance_threshold = trend_significance_threshold
        self.include_local_residual = include_local_residual
        self.local_residual_variance_floor = local_residual_variance_floor
        self.random_state = random_state

    def _fit_impl(self, X: Path, y: None = None) -> xr.Dataset:
        """Shared fit logic; returns the Gaussianized anomalies for `fit_transform` to reuse."""
        rng = np.random.default_rng(self.random_state)

        x_raw = load_training_window(X, self.t_start, self.t_end, self.variables)
        # load_training_window already confirmed [t_start, t_end) falls in a single
        # archive era (see train.ingest._resolve_window_era) — safe to derive dt
        # from t_start alone rather than trusting a flat, era-2-only constant
        # (see tasks/16-archive-grid-resolution-drift.md).
        window_era = era_for_timestamp(self.t_start)
        preprocess = Pipeline(
            [
                (
                    "coming_from_to_uv",
                    ComingFromToUV(
                        pairs=[
                            ("wsh", "wsd", "u_wsea", "v_wsea"),
                            ("was", "wad_come_from", "u_wind", "v_wind"),
                        ]
                    ),
                ),
                (
                    # Empty for tier 1; populated when tier 3 (ocean currents,
                    # `ocd`) is enabled (see docs/model_design.md §5.3, §3.8).
                    # `wad` is never wired here — `wad_come_from` is the source
                    # of truth for wind direction (§3.7, §9 gotcha 5a).
                    "going_to_to_uv",
                    GoingToToUV(pairs=[]),
                ),
                ("reproject", Reprojector(target_grid="wave")),
                (
                    "climatology",
                    ClimatologyRemover(
                        diurnal_variables=self.diurnal_variables,
                        include_trend=self.include_trend,
                        trend_significance_threshold=self.trend_significance_threshold,
                    ),
                ),
                ("standardize", PerCellStandardizer()),
                ("gaussianize", GaussianCopula(n_bins=self.copula_lut_bins)),
            ]
        )
        gaussianized: xr.Dataset = preprocess.fit_transform(x_raw)

        _log_trend_significance(
            preprocess.named_steps["climatology"],
            preprocess.named_steps["standardize"],
            self.trend_significance_threshold,
        )

        ocean_mask, variable_names, z = _mask_and_stack(gaussianized)
        tucker_fit_result = tucker_fit(z, r_ocean=self.r_ocean, r_var=self.r_var, rng=rng)
        core = tucker_transform(z, tucker_fit_result)
        dmd_fit_result = dmd_fit(
            core,
            r_dmd=self.r_dmd,
            dt=window_era.cadence_seconds,
            rng=rng,
            variant=self.dmd_variant,
        )
        mode_ou_fit = fit_mode_ou(core, dmd_fit_result)

        local_residual_fit: LocalResidualFit | None = None
        if self.include_local_residual:
            # DMD's deterministic-only reconstruction (b0 * Lambda**t, no
            # mode-OU stochastic wobble added) — matches the pattern in
            # weather.simulate's sample_batch/simulate_field. See
            # train.local_residual.coherent_reconstruction's docstring for
            # why the residual fit below is only meaningful without that
            # wobble folded in.
            tt = np.arange(z.shape[0])
            a_det = dmd_fit_result.b0[:, None] * (dmd_fit_result.Lambda[:, None] ** tt[None, :])
            x_det = (dmd_fit_result.Phi @ a_det).real
            core_deterministic = x_det.T.reshape(z.shape[0], self.r_ocean, self.r_var)
            z_coherent = coherent_reconstruction(core_deterministic, tucker_fit_result)
            local_residual_fit = fit_local_residual(
                z,
                z_coherent,
                dt=window_era.cadence_seconds,
                variance_floor=self.local_residual_variance_floor,
            )

        self.preprocess_ = preprocess
        self.tucker_ = tucker_fit_result
        self.dmd_ = dmd_fit_result
        self.mode_ou_ = mode_ou_fit
        self.local_residual_ = local_residual_fit
        self.ocean_mask_ = ocean_mask
        self.variable_names_ = variable_names
        self.metadata_ = ModelMetadata(
            trained_at=datetime.now(timezone.utc),
            t_train_start=self.t_start,
            t_train_end=self.t_end,
            # Reuse the fitted ClimatologyRemover's own t_ref_ (the training
            # window's midpoint, see train.climatology's docstring) rather
            # than recomputing it independently — avoids two sources of
            # truth for the same value.
            t_ref=preprocess.named_steps["climatology"].t_ref_,
            variables=variable_names,
            r_ocean=self.r_ocean,
            r_var=self.r_var,
            r_dmd=self.r_dmd,
            dmd_variant=self.dmd_variant,
            include_trend=self.include_trend,
            include_local_residual=self.include_local_residual,
            git_commit=_current_git_commit(),
            sklearn_version=sklearn.__version__,
            numpy_version=np.__version__,
            tensorly_version=tensorly.__version__,
            # pydmd exposes no __version__ attribute; importlib.metadata is
            # the only way to recover its installed version.
            pydmd_version=importlib.metadata.version("pydmd"),
        )
        return gaussianized

    def fit(self, X: Path, y: None = None) -> Self:
        """Train the model end-to-end on the archive rooted at `X`.

        Parameters
        ----------
        X : Path
            Root of the data archive (typically ``./data``). Windowed by
            `self.t_start` / `self.t_end`.
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self
        """
        self._fit_impl(X, y)
        return self

    def transform(
        self,
        X: Path,
        t_start: datetime | None = None,
        t_end: datetime | None = None,
    ) -> xr.Dataset:
        """Apply the fitted preprocessing chain to a new data window.

        Parameters
        ----------
        X : Path
            Root of the data archive (the training `data_dir` or a
            differently-rooted held-out mirror with the same layout).
        t_start, t_end : datetime | None
            UTC-aware window bounds; both default to the training window
            (`self.t_start`, `self.t_end`) when omitted. Pass explicit
            bounds to run on a held-out window (see §6 validation check 7).

        Returns
        -------
        xr.Dataset
            Gaussianized standardized anomalies, dims
            ``(time, wave_lat, wave_lon)`` per variable.
        """
        check_is_fitted(self, "preprocess_")
        window_start = self.t_start if t_start is None else t_start
        window_end = self.t_end if t_end is None else t_end
        x_raw = load_training_window(X, window_start, window_end, self.variables)
        result: xr.Dataset = self.preprocess_.transform(x_raw)
        return result

    def fit_transform(self, X: Path, y: None = None) -> xr.Dataset:
        """Fit and return the Gaussianized anomalies in one pass.

        Overridden (rather than relying on `TransformerMixin`'s default
        ``fit(X).transform(X)``) because `fit` already computes this
        Dataset internally — recomputing it would repeat the whole
        preprocessing pass.
        """
        return self._fit_impl(X, y)

    @classmethod
    def from_components(
        cls,
        *,
        preprocess: Pipeline,
        tucker: TuckerFit,
        dmd: DMDFit,
        mode_ou: ModeOUFit,
        local_residual: LocalResidualFit | None,
        ocean_mask: np.ndarray,
        metadata: ModelMetadata,
    ) -> Self:
        """Construct from already-fit components.

        Used by the notebook workflow when Tucker/DMD/mode-OU/local-residual
        hyperparameters were iterated on separately and the chosen versions
        need bundling into a saveable `WeatherModel` without re-running `fit`.

        Parameters
        ----------
        preprocess : Pipeline
            A fitted preprocessing Pipeline (see §5.2-§5.4).
        tucker : TuckerFit
        dmd : DMDFit
        mode_ou : ModeOUFit
        local_residual : LocalResidualFit | None
            `None` when the model was fit with `include_local_residual=False`.
        ocean_mask : np.ndarray
            Boolean, ``(n_lat, n_lon)``.
        metadata : ModelMetadata

        Returns
        -------
        Self
        """
        model = cls(
            t_start=metadata.t_train_start,
            t_end=metadata.t_train_end,
            variables=list(metadata.variables),
            r_ocean=metadata.r_ocean,
            r_var=metadata.r_var,
            r_dmd=metadata.r_dmd,
            # ModelMetadata.dmd_variant is a plain str (always one of the three
            # Literal values in practice, since it's only ever populated from
            # this same __init__ param at fit() time).
            dmd_variant=metadata.dmd_variant,  # type: ignore[arg-type]
            include_trend=metadata.include_trend,
            include_local_residual=metadata.include_local_residual,
        )
        model.preprocess_ = preprocess
        model.tucker_ = tucker
        model.dmd_ = dmd
        model.mode_ou_ = mode_ou
        model.local_residual_ = local_residual
        model.ocean_mask_ = ocean_mask
        model.variable_names_ = metadata.variables
        model.metadata_ = metadata
        return model

    def save(self, model_dir: Path) -> None:
        """Persist fitted state to disk (see docs/model_design.md §7).

        Parameters
        ----------
        model_dir : Path
            Destination directory, created if absent.
        """
        check_is_fitted(self, "preprocess_")
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.preprocess_, model_dir / "preprocess.pkl")
        np.save(model_dir / "ocean_mask.npy", self.ocean_mask_)
        np.savez(
            model_dir / "tucker.npz",
            # float32 on disk per §7's dtype policy (large arrays); TuckerFit
            # itself stays float64 in memory (train.tucker's contract) — load()
            # upcasts back after reading.
            U_ocean=self.tucker_.U_ocean.astype(np.float32),
            U_var=self.tucker_.U_var.astype(np.float32),
            r_ocean=self.tucker_.r_ocean,
            r_var=self.tucker_.r_var,
            reconstruction_error=self.tucker_.reconstruction_error,
        )
        np.savez(
            model_dir / "dmd.npz",
            Phi=self.dmd_.Phi,
            Lambda=self.dmd_.Lambda,
            b0=self.dmd_.b0,
            lam_cont=self.dmd_.lam_cont,
            dt=self.dmd_.dt,
            variant=self.dmd_.variant,
        )
        np.savez(
            model_dir / "mode_ou.npz",
            theta=self.mode_ou_.theta,
            sigma=self.mode_ou_.sigma,
            phi=self.mode_ou_.phi,
            dt=self.mode_ou_.dt,
        )
        if self.local_residual_ is not None:
            np.savez(
                model_dir / "local_residual.npz",
                theta=self.local_residual_.theta,
                sigma=self.local_residual_.sigma,
                phi=self.local_residual_.phi,
                variance_ratio=self.local_residual_.variance_ratio,
                active_mask=self.local_residual_.active_mask,
                dt=self.local_residual_.dt,
            )

        metadata_dict = asdict(self.metadata_)
        metadata_dict["trained_at"] = self.metadata_.trained_at.isoformat()
        metadata_dict["t_train_start"] = self.metadata_.t_train_start.isoformat()
        metadata_dict["t_train_end"] = self.metadata_.t_train_end.isoformat()
        metadata_dict["t_ref"] = self.metadata_.t_ref.isoformat()
        metadata_dict["variables"] = list(self.metadata_.variables)
        (model_dir / "metadata.json").write_text(json.dumps(metadata_dict, indent=2))

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        """Reconstruct from disk. Inverse of `save` (see docs/model_design.md §7).

        Warns if the runtime's sklearn/numpy/tensorly/pydmd versions differ
        from the versions recorded at training time (see §9 gotcha 17) — a
        mismatch on tensorly or pydmd almost always breaks loading.

        Parameters
        ----------
        model_dir : Path

        Returns
        -------
        Self
        """
        metadata_dict = json.loads((model_dir / "metadata.json").read_text())

        if metadata_dict["sklearn_version"] != sklearn.__version__:
            warnings.warn(
                f"model was trained with scikit-learn {metadata_dict['sklearn_version']!r}, "
                f"runtime has {sklearn.__version__!r}; preprocess.pkl may not load correctly.",
                stacklevel=2,
            )
        if metadata_dict["numpy_version"] != np.__version__:
            warnings.warn(
                f"model was trained with numpy {metadata_dict['numpy_version']!r}, "
                f"runtime has {np.__version__!r}.",
                stacklevel=2,
            )
        if metadata_dict["tensorly_version"] != tensorly.__version__:
            warnings.warn(
                f"model was trained with tensorly {metadata_dict['tensorly_version']!r}, "
                f"runtime has {tensorly.__version__!r}; tucker.npz may not be interpreted "
                "correctly.",
                stacklevel=2,
            )
        pydmd_version = importlib.metadata.version("pydmd")
        if metadata_dict["pydmd_version"] != pydmd_version:
            warnings.warn(
                f"model was trained with pydmd {metadata_dict['pydmd_version']!r}, "
                f"runtime has {pydmd_version!r}; dmd.npz may not be interpreted correctly.",
                stacklevel=2,
            )

        metadata = ModelMetadata(
            trained_at=datetime.fromisoformat(metadata_dict["trained_at"]),
            t_train_start=datetime.fromisoformat(metadata_dict["t_train_start"]),
            t_train_end=datetime.fromisoformat(metadata_dict["t_train_end"]),
            t_ref=datetime.fromisoformat(metadata_dict["t_ref"]),
            variables=tuple(metadata_dict["variables"]),
            r_ocean=metadata_dict["r_ocean"],
            r_var=metadata_dict["r_var"],
            r_dmd=metadata_dict["r_dmd"],
            dmd_variant=metadata_dict["dmd_variant"],
            include_trend=metadata_dict["include_trend"],
            include_local_residual=metadata_dict["include_local_residual"],
            git_commit=metadata_dict["git_commit"],
            sklearn_version=metadata_dict["sklearn_version"],
            numpy_version=metadata_dict["numpy_version"],
            tensorly_version=metadata_dict["tensorly_version"],
            pydmd_version=metadata_dict["pydmd_version"],
        )

        preprocess = joblib.load(model_dir / "preprocess.pkl")
        ocean_mask = np.load(model_dir / "ocean_mask.npy")

        with np.load(model_dir / "tucker.npz") as data:
            tucker_fit_result = TuckerFit(
                # Stored as float32 (§7 dtype policy); upcast to match
                # TuckerFit's float64 contract (train.tucker docstring).
                U_ocean=data["U_ocean"].astype(np.float64),
                U_var=data["U_var"].astype(np.float64),
                r_ocean=int(data["r_ocean"]),
                r_var=int(data["r_var"]),
                reconstruction_error=float(data["reconstruction_error"]),
            )
        with np.load(model_dir / "dmd.npz") as data:
            dmd_fit_result = DMDFit(
                Phi=data["Phi"],
                Lambda=data["Lambda"],
                b0=data["b0"],
                lam_cont=data["lam_cont"],
                dt=float(data["dt"]),
                variant=str(data["variant"]),
            )
        with np.load(model_dir / "mode_ou.npz") as data:
            mode_ou_fit = ModeOUFit(
                theta=data["theta"],
                sigma=data["sigma"],
                phi=data["phi"],
                dt=float(data["dt"]),
            )

        local_residual_path = model_dir / "local_residual.npz"
        local_residual_fit: LocalResidualFit | None = None
        if local_residual_path.exists():
            with np.load(local_residual_path) as data:
                local_residual_fit = LocalResidualFit(
                    theta=data["theta"],
                    sigma=data["sigma"],
                    phi=data["phi"],
                    variance_ratio=data["variance_ratio"],
                    active_mask=data["active_mask"],
                    dt=float(data["dt"]),
                )

        return cls.from_components(
            preprocess=preprocess,
            tucker=tucker_fit_result,
            dmd=dmd_fit_result,
            mode_ou=mode_ou_fit,
            local_residual=local_residual_fit,
            ocean_mask=ocean_mask,
            metadata=metadata,
        )

    def ocean_mask(self) -> np.ndarray:
        """Boolean ``(n_lat, n_lon)`` — True where the model can produce samples."""
        check_is_fitted(self, "ocean_mask_")
        return self.ocean_mask_
