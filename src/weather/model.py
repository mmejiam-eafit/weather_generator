"""WeatherModel: sklearn-compatible wrapper gluing preprocessing to Tucker/DMD/OU.

(see docs/model_design.md §7, §8.4, §8.5)

v3 architecture: one preprocessing Pipeline ending at the standardized
anomaly, then per tensor a masked Tucker fit and a Gaussian copula on its
core, then per variable group a DMD, a mode-level OU and the §5.6.2
diagnostics stage.
"""

import importlib.metadata
import json
import subprocess
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, Self

import joblib
import numpy as np
import sklearn
import tensorly
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from core.archive_eras import era_for_timestamp
from core.logger import logger
from train.climatology import ClimatologyRemover
from train.copula import GaussianCopula
from train.directional import ComingFromToUV, GoingToToUV
from train.dmd import DMDFit, ModeOUFit, dmd_fit, fit_mode_ou, group_state
from train.ingest import load_training_window
from train.linear_inverse import (
    GroupCouplingReport,
    GroupDiagnostics,
    LIMFit,
    TauTestResult,
    diagnose_group,
)
from train.masking import LogTransform, PartitionSentinelFiller
from train.reproject import Reprojector
from train.standardize import PerCellStandardizer
from train.tucker import TuckerFit, fill_land, tucker_fit

# Canonical tensor derivation (docs/model_design.md §5.3, §5.3.1, §10). A
# pair, partition or log entry is wired in only when all of its source
# variables are in `WeatherModel.variables`, so one table serves every tier.
PARTITIONS: Final[dict[str, tuple[str, str]]] = {
    "wsh": ("wsp", "wsd"),
    "pswh": ("pswp", "pswd"),
}
COMING_FROM_PAIRS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("wsh", "wsd", "u_wsea", "v_wsea"),
    # `wad_come_from`, never `wad`: the source of truth for wind (§3.7, §9 gotcha 5a).
    ("was", "wad_come_from", "u_wind", "v_wind"),
    ("pwh", "pwd", "u_pwave", "v_pwave"),
    ("pswh", "pswd", "u_pswell", "v_pswell"),
)
GOING_TO_PAIRS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("ocs", "ocd", "u_current", "v_current"),
)
KEPT_SPEEDS: Final[tuple[str, ...]] = ("wsh", "pswh")  # class C: also their own logE tensor
LOG_HEIGHTS: Final[dict[str, str]] = {
    "sigwh": "logE_sigwh",
    "wsh": "logE_wsh",
    "pswh": "logE_pswh",
}
LOG_PERIODS: Final[dict[str, str]] = {"wsp": "T_wsp", "pwp": "T_pwp", "pswp": "T_pswp"}
LOG_FLOOR_M: Final[float] = 0.01

# The first per-cell Pipeline step: steps before it are pointwise (the
# sampler applies them after interpolation), steps from it onward are per
# grid cell (applied before), see §5.8 steps 4 and 6.
PER_CELL_FIRST_STEP: Final[str] = "reproject"

_CORE_DIMS: Final[tuple[str, str, str]] = ("time", "core_lat", "core_lon")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadata recorded at fit time (see docs/model_design.md §7, §8.5)."""

    trained_at: datetime
    t_train_start: datetime
    t_train_end: datetime
    t_ref: datetime  # climatology trend reference epoch (training window midpoint)
    variables: tuple[str, ...]  # raw archive fields loaded
    tensors: tuple[str, ...]  # modeled tensors (§10)
    groups: dict[str, tuple[str, ...]]  # group name -> member tensors
    tucker_ranks: dict[str, tuple[int, int]]  # tensor -> (r_lat, r_lon)
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


def build_preprocess(
    variables: list[str],
    diurnal_variables: list[str] | None = None,
    include_trend: bool = True,
    trend_significance_threshold: float = 0.1,
) -> Pipeline:
    """The v3 preprocessing Pipeline for a set of raw archive fields (see §5.3, §5.3.1, §5.4).

    Order: sentinel fill → ``(u, v)`` decomposition → log transforms →
    reprojection → climatology → standardization. Each table entry above is
    included only when its source variables are all present, so tier 1's
    variables give tier 1's seven tensors. `GoingToToUV` stays in the
    Pipeline even with no pairs, where tier 3's currents will go (§5.3).

    Parameters
    ----------
    variables : list[str]
        Raw archive fields (§10). Must include ``sigwh``, which defines the
        land mask (§5.1).
    diurnal_variables, include_trend, trend_significance_threshold
        Passed to `ClimatologyRemover` (§5.4).

    Returns
    -------
    Pipeline
        Unfitted.
    """
    present = set(variables)
    partitions = {
        height: fields for height, fields in PARTITIONS.items() if {height, *fields} <= present
    }
    coming_from = [pair for pair in COMING_FROM_PAIRS if {pair[0], pair[1]} <= present]
    going_to = [pair for pair in GOING_TO_PAIRS if {pair[0], pair[1]} <= present]
    kept = tuple(speed for speed, *_ in coming_from if speed in KEPT_SPEEDS)
    heights = {source: tensor for source, tensor in LOG_HEIGHTS.items() if source in present}
    periods = {source: tensor for source, tensor in LOG_PERIODS.items() if source in present}
    return Pipeline(
        [
            ("sentinel", PartitionSentinelFiller(partitions=partitions, floor_m=LOG_FLOOR_M)),
            ("coming_from_to_uv", ComingFromToUV(pairs=coming_from, keep_speed=kept)),
            ("going_to_to_uv", GoingToToUV(pairs=going_to)),
            ("log", LogTransform(heights=heights, periods=periods, floor_m=LOG_FLOOR_M)),
            (PER_CELL_FIRST_STEP, Reprojector(target_grid="wave")),
            (
                "climatology",
                ClimatologyRemover(
                    diurnal_variables=diurnal_variables,
                    include_trend=include_trend,
                    trend_significance_threshold=trend_significance_threshold,
                ),
            ),
            ("standardize", PerCellStandardizer()),
        ]
    )


def core_dataset(tensor: str, core: np.ndarray) -> xr.Dataset:
    """Wrap one tensor's ``(T, r_lat, r_lon)`` core so `GaussianCopula` accepts it (§5.5.1).

    The copula checks canonical ``(time, *_lat, *_lon)`` dims; the core's
    coefficient axes are named ``core_lat`` / ``core_lon`` to pass it.
    """
    return xr.Dataset({tensor: (_CORE_DIMS, core)})


def resolve_groups(
    groups: Mapping[str, list[str]] | None, tensors: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    """Variable groups for DMD, one single-member group per tensor by default (§5.6.2).

    Parameters
    ----------
    groups : Mapping[str, list[str]] | None
        Group name -> member tensors, or `None`.
    tensors : tuple[str, ...]
        Every modeled tensor.

    Returns
    -------
    dict[str, tuple[str, ...]]

    Raises
    ------
    ValueError
        If the groups name an unknown tensor, or do not cover every tensor
        exactly once: a tensor outside every group would have no dynamics to
        sample from.
    """
    if groups is None:
        return {tensor: (tensor,) for tensor in tensors}
    resolved = {name: tuple(members) for name, members in groups.items()}
    assigned = [member for members in resolved.values() for member in members]
    unknown = sorted(set(assigned) - set(tensors))
    if unknown:
        raise ValueError(f"groups name unknown tensors {unknown}; tensors are {list(tensors)}")
    repeated = sorted({m for m in assigned if assigned.count(m) > 1})
    missing = sorted(set(tensors) - set(assigned))
    if repeated or missing:
        raise ValueError(
            f"groups must cover every tensor exactly once (docs/model_design.md §5.6.2); "
            f"repeated: {repeated}, missing: {missing}"
        )
    return resolved


def _tucker_rank(fraction: float, grid_shape: tuple[int, ...]) -> tuple[int, int]:
    """``round(fraction · (H, W))``, at least 1 per mode (§5.5)."""
    n_lat, n_lon = grid_shape
    return max(1, round(fraction * n_lat)), max(1, round(fraction * n_lon))


def _log_trend_significance(
    climatology: ClimatologyRemover,
    standardize: PerCellStandardizer,
    threshold: float,
) -> None:
    """Log the fraction of cells whose linear trend is significant relative to its noise floor.

    (see docs/model_design.md §3.9, §5.4)

    Per tensor, ``ratio = |c_lin_[v]| / sigma_cell_[v]``: the fitted trend
    slope's magnitude relative to that cell's temporal standard deviation.
    `climatology.c_lin_` only has entries when the trend term was fitted, so
    nothing is logged when it was disabled.
    """
    for tensor, c_lin in climatology.c_lin_.items():
        ratio = np.abs(c_lin) / standardize.sigma_cell_[tensor]
        logger.info(
            "trend significance",
            tensor=tensor,
            exceeds_fraction=float(np.nanmean(ratio > threshold)),
            threshold=threshold,
        )


def _log_period_sentinels(sentinel: PartitionSentinelFiller) -> None:
    """Log each partition's period sentinel and its two estimators' agreement (§5.3.1)."""
    for period, estimate in sentinel.period_sentinel_estimates_.items():
        logger.info(
            "period sentinel",
            period=period,
            sentinel_s=estimate.extrapolated_intercept,
            conditional_median_s=estimate.conditional_median,
            agreement_s=estimate.agreement,
        )


def _log_group_diagnostics(group: str, diagnostics: GroupDiagnostics) -> None:
    """Log one group's §5.6.2 verdicts: lags outside the tau-test band, coupling flags."""
    hours = diagnostics.lim.dt / 3600.0
    logger.info(
        "group diagnostics",
        group=group,
        members=list(diagnostics.members),
        tau_inconsistent_lags_h=[
            float(lag * hours)
            for lag, ok in zip(diagnostics.tau.lags, diagnostics.tau_consistent)
            if not ok
        ],
        coupling_significant=diagnostics.coupling_significant,
        noise_coupling_significant=diagnostics.noise_coupling_significant,
    )


def _fit_tensor_compression(
    a_std: xr.Dataset,
    ocean_mask: np.ndarray,
    rank_fraction: float,
    max_iter: int,
    tol: float,
    copula_lut_bins: int,
    rng: np.random.Generator,
) -> tuple[dict[str, TuckerFit], dict[str, GaussianCopula], dict[str, np.ndarray]]:
    """Per tensor: masked Tucker, then a copula on its core (§5.5, §5.5.1).

    Returns
    -------
    tucker : tensor -> TuckerFit
    core_copula : tensor -> fitted GaussianCopula
    gaussianized_core : tensor -> ``(T, r_lat, r_lon)``
    """
    rank = _tucker_rank(rank_fraction, ocean_mask.shape)
    tucker: dict[str, TuckerFit] = {}
    core_copula: dict[str, GaussianCopula] = {}
    gaussianized_core: dict[str, np.ndarray] = {}
    for name in a_std.data_vars:
        tensor = str(name)
        x = fill_land(a_std[tensor].values, ocean_mask)
        fit, core = tucker_fit(x, ocean_mask, rank, rng, n_iter_max=max_iter, tol=tol)
        logger.info(
            "tucker fit",
            tensor=tensor,
            rank=list(rank),
            reconstruction_error=fit.reconstruction_error,
            n_iter=fit.n_iter,
        )
        copula = GaussianCopula(n_bins=copula_lut_bins).fit(core_dataset(tensor, core))
        tucker[tensor] = fit
        core_copula[tensor] = copula
        gaussianized_core[tensor] = copula.transform(core_dataset(tensor, core))[tensor].values
    return tucker, core_copula, gaussianized_core


def _fit_group_dynamics(
    gaussianized_core: Mapping[str, np.ndarray],
    groups: Mapping[str, tuple[str, ...]],
    dt: float,
    r_dmd: int,
    dmd_variant: Literal["exact", "bop", "hankel"],
    lim_rank: int,
    tau_test_lags: tuple[int, ...],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, DMDFit], dict[str, ModeOUFit], dict[str, GroupDiagnostics]]:
    """Per group: DMD and mode OU on the joint state, then the §5.6.2 diagnostics.

    A group's DMD rank is ``min(r_dmd, n_G, T − 1)`` (§5.6), and each
    member's LIM rank is capped at its core size and `T`, which small test
    grids otherwise exceed.
    """
    dmd: dict[str, DMDFit] = {}
    mode_ou: dict[str, ModeOUFit] = {}
    diagnostics: dict[str, GroupDiagnostics] = {}
    for group, members in groups.items():
        state = group_state(gaussianized_core, members)
        n_times, n_state = state.shape
        dmd[group] = dmd_fit(
            state,
            r_dmd=min(r_dmd, n_state, n_times - 1),
            dt=dt,
            rng=rng,
            variant=dmd_variant,
        )
        mode_ou[group] = fit_mode_ou(state, dmd[group])
        cores = {member: gaussianized_core[member] for member in members}
        diagnostics[group] = diagnose_group(
            cores=cores,
            ranks={member: min(lim_rank, cores[member][0].size, n_times) for member in members},
            dt=dt,
            test_lags=list(tau_test_lags),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        _log_group_diagnostics(group, diagnostics[group])
    return dmd, mode_ou, diagnostics


class WeatherModel(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Hierarchical stochastic weather generator (see docs/model_design.md §8.5).

    Architecture (v3): per-tensor climatology + standardization, per-tensor
    masked Tucker, Gaussian copula on each core, DMD + mode OU per variable
    group. sklearn-compatible transformer surface.

    sklearn surface (via BaseEstimator + TransformerMixin):
        ``__init__`` carries hyperparameters only.
        ``fit(X, y=None)`` — end-to-end training.
        ``transform(X)`` — runs the preprocessing chain only.
        ``fit_transform(X, y=None)`` — fit and transform in one pass.

    Custom surface: ``save``, ``load``, ``from_components``.

    Fitted attributes (trailing underscore per sklearn convention):
        preprocess_        : Pipeline (sentinel fill → standardize)
        ocean_mask_        : (H, W) bool, from sigwh (§5.1)
        tensor_names_      : tuple[str, ...]
        groups_            : dict[str, tuple[str, ...]]
        tucker_            : dict[str, TuckerFit]        (tensor -> fit)
        core_copula_       : dict[str, GaussianCopula]   (tensor -> fitted copula)
        dmd_               : dict[str, DMDFit]           (group -> fit)
        mode_ou_           : dict[str, ModeOUFit]        (group -> fit)
        group_diagnostics_ : dict[str, GroupDiagnostics] (group -> §5.6.2 output)
        local_residual_    : None in v3 (§5.7)
        metadata_          : ModelMetadata

    Sampling is **not** a method on this class — use the free functions in
    `weather.simulate` (`sample`, `sample_batch`, `simulate_field`).
    """

    def __init__(
        self,
        t_start: datetime,
        t_end: datetime,
        variables: list[str],
        tucker_rank_fraction: float = 0.4,
        tucker_max_iter: int = 300,
        tucker_tol: float = 1e-5,
        r_dmd: int = 100,
        dmd_variant: Literal["exact", "bop", "hankel"] = "exact",
        copula_lut_bins: int = 500,
        diurnal_variables: list[str] | None = None,
        include_trend: bool = True,
        trend_significance_threshold: float = 0.1,
        include_local_residual: bool = False,
        local_residual_variance_floor: float = 0.02,
        groups: dict[str, list[str]] | None = None,
        lim_rank: int = 20,
        tau_test_lags: tuple[int, ...] = (1, 2, 4, 8, 16),
        n_bootstrap: int = 100,
        random_state: int | None = None,
    ) -> None:
        # sklearn convention: __init__ only stores parameters, verbatim, no
        # validation or side effects — sklearn.base.clone() relies on this.
        self.t_start = t_start
        self.t_end = t_end
        self.variables = variables
        self.tucker_rank_fraction = tucker_rank_fraction
        self.tucker_max_iter = tucker_max_iter
        self.tucker_tol = tucker_tol
        self.r_dmd = r_dmd
        self.dmd_variant = dmd_variant
        self.copula_lut_bins = copula_lut_bins
        self.diurnal_variables = diurnal_variables
        self.include_trend = include_trend
        self.trend_significance_threshold = trend_significance_threshold
        self.include_local_residual = include_local_residual
        self.local_residual_variance_floor = local_residual_variance_floor
        self.groups = groups
        self.lim_rank = lim_rank
        self.tau_test_lags = tau_test_lags
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def _fit_impl(self, X: Path) -> xr.Dataset:
        """Shared fit logic; returns the standardized anomalies for `fit_transform`."""
        if self.include_local_residual:
            raise ValueError(
                "include_local_residual=True is not available in v3: the local residual "
                "is deferred (docs/model_design.md §5.7, docs/open_questions.md §6.2)"
            )
        rng = np.random.default_rng(self.random_state)
        x_raw = load_training_window(X, self.t_start, self.t_end, self.variables)
        # load_training_window already confirmed [t_start, t_end) falls in a single
        # archive era, so dt comes from that era's cadence, not a flat constant
        # (tasks/16-archive-grid-resolution-drift.md).
        dt = era_for_timestamp(self.t_start).cadence_seconds

        preprocess = build_preprocess(
            self.variables,
            self.diurnal_variables,
            self.include_trend,
            self.trend_significance_threshold,
        )
        a_std: xr.Dataset = preprocess.fit_transform(x_raw)
        _log_period_sentinels(preprocess.named_steps["sentinel"])
        _log_trend_significance(
            preprocess.named_steps["climatology"],
            preprocess.named_steps["standardize"],
            self.trend_significance_threshold,
        )

        ocean_mask: np.ndarray = preprocess.named_steps["sentinel"].ocean_mask_
        tensors = tuple(str(name) for name in a_std.data_vars)
        groups = resolve_groups(self.groups, tensors)
        tucker, core_copula, gaussianized_core = _fit_tensor_compression(
            a_std,
            ocean_mask,
            self.tucker_rank_fraction,
            self.tucker_max_iter,
            self.tucker_tol,
            self.copula_lut_bins,
            rng,
        )
        dmd, mode_ou, diagnostics = _fit_group_dynamics(
            gaussianized_core,
            groups,
            dt,
            self.r_dmd,
            self.dmd_variant,
            self.lim_rank,
            self.tau_test_lags,
            self.n_bootstrap,
            rng,
        )

        self.preprocess_ = preprocess
        self.ocean_mask_ = ocean_mask
        self.tensor_names_ = tensors
        self.groups_ = groups
        self.tucker_ = tucker
        self.core_copula_ = core_copula
        self.dmd_ = dmd
        self.mode_ou_ = mode_ou
        self.group_diagnostics_ = diagnostics
        self.local_residual_ = None
        self.metadata_ = ModelMetadata(
            trained_at=datetime.now(timezone.utc),
            t_train_start=self.t_start,
            t_train_end=self.t_end,
            # The fitted ClimatologyRemover's own t_ref_ (the training window's
            # midpoint) — one source of truth for the trend epoch (§9 gotcha 13).
            t_ref=preprocess.named_steps["climatology"].t_ref_,
            variables=tuple(self.variables),
            tensors=tensors,
            groups=groups,
            tucker_ranks={
                t: (fit.factor_lat.shape[1], fit.factor_lon.shape[1]) for t, fit in tucker.items()
            },
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
        return a_std

    def fit(self, X: Path, y: None = None) -> Self:
        """Train the model end-to-end on the archive rooted at `X`.

        In-process training is the path for small windows and tests; a
        production era-3 window runs Tucker per tensor on a VM and assembles
        the model with `from_components` (§8.5, §8.6).

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

        Raises
        ------
        ValueError
            If `include_local_residual` is True (deferred in v3), `sigwh` is
            missing from `variables`, or `groups` does not partition the
            tensors.
        """
        self._fit_impl(X)
        return self

    def transform(
        self,
        X: Path,
        t_start: datetime | None = None,
        t_end: datetime | None = None,
    ) -> xr.Dataset:
        """Apply the fitted preprocessing chain to a data window.

        Parameters
        ----------
        X : Path
            Root of the data archive (the training `data_dir` or a
            differently-rooted held-out mirror with the same layout).
        t_start, t_end : datetime | None
            UTC-aware window bounds; both default to the training window.
            Pass explicit bounds for a held-out window (§6 check 7).

        Returns
        -------
        xr.Dataset
            Standardized anomalies, one ``(time, wave_lat, wave_lon)``
            variable per tensor; land NaN. Not cores, not Gaussianized.
        """
        check_is_fitted(self, "preprocess_")
        window_start = self.t_start if t_start is None else t_start
        window_end = self.t_end if t_end is None else t_end
        x_raw = load_training_window(X, window_start, window_end, self.variables)
        result: xr.Dataset = self.preprocess_.transform(x_raw)
        return result

    def fit_transform(self, X: Path, y: None = None) -> xr.Dataset:
        """Fit and return the standardized anomalies in one pass.

        Overridden because `fit` already computes this Dataset; the default
        ``fit(X).transform(X)`` would repeat the whole preprocessing pass.
        """
        return self._fit_impl(X)

    @classmethod
    def from_components(
        cls,
        *,
        preprocess: Pipeline,
        tucker: dict[str, TuckerFit],
        core_copula: dict[str, GaussianCopula],
        dmd: dict[str, DMDFit],
        mode_ou: dict[str, ModeOUFit],
        group_diagnostics: dict[str, GroupDiagnostics],
        ocean_mask: np.ndarray,
        metadata: ModelMetadata,
    ) -> Self:
        """Construct from already-fit components (the §8.6 notebook path).

        Production fits Tucker per tensor on a VM with checkpoints, so the
        pieces are fit separately and bundled here without re-running `fit`.

        Parameters
        ----------
        preprocess : Pipeline
            A fitted preprocessing Pipeline (§5.3–§5.4).
        tucker, core_copula : dict keyed by tensor
        dmd, mode_ou, group_diagnostics : dict keyed by group
        ocean_mask : (H, W) bool
        metadata : ModelMetadata

        Returns
        -------
        Self
        """
        model = cls(
            t_start=metadata.t_train_start,
            t_end=metadata.t_train_end,
            variables=list(metadata.variables),
            r_dmd=metadata.r_dmd,
            # ModelMetadata.dmd_variant is a plain str, only ever populated from
            # this same __init__ parameter at fit time.
            dmd_variant=metadata.dmd_variant,  # type: ignore[arg-type]
            include_trend=metadata.include_trend,
            include_local_residual=metadata.include_local_residual,
            groups={name: list(members) for name, members in metadata.groups.items()},
        )
        model.preprocess_ = preprocess
        model.ocean_mask_ = ocean_mask
        model.tensor_names_ = metadata.tensors
        model.groups_ = dict(metadata.groups)
        model.tucker_ = tucker
        model.core_copula_ = core_copula
        model.dmd_ = dmd
        model.mode_ou_ = mode_ou
        model.group_diagnostics_ = group_diagnostics
        model.local_residual_ = None
        model.metadata_ = metadata
        return model

    def save(self, model_dir: Path) -> None:
        """Persist fitted state in the §7 layout.

        Parameters
        ----------
        model_dir : Path
            Destination directory, created if absent.
        """
        check_is_fitted(self, "preprocess_")
        for sub in ("tucker", "core_copula", "dmd", "mode_ou", "lim"):
            (model_dir / sub).mkdir(parents=True, exist_ok=True)

        joblib.dump(self.preprocess_, model_dir / "preprocess.pkl")
        np.save(model_dir / "ocean_mask.npy", self.ocean_mask_)
        for tensor in self.tensor_names_:
            _save_tucker(model_dir / "tucker" / f"{tensor}.npz", self.tucker_[tensor])
            _save_copula(
                model_dir / "core_copula" / f"{tensor}.npz", tensor, self.core_copula_[tensor]
            )
        records: dict[str, Any] = {}
        for group, members in self.groups_.items():
            sizes = [
                self.tucker_[m].factor_lat.shape[1] * self.tucker_[m].factor_lon.shape[1]
                for m in members
            ]
            _save_dmd(model_dir / "dmd" / f"{group}.npz", self.dmd_[group], members, sizes)
            _save_mode_ou(model_dir / "mode_ou" / f"{group}.npz", self.mode_ou_[group])
            diagnostics = self.group_diagnostics_[group]
            _save_lim(model_dir / "lim" / f"{group}.npz", diagnostics.lim)
            records[group] = _diagnostics_record(diagnostics)
        (model_dir / "group_diagnostics.json").write_text(json.dumps(records, indent=2))
        (model_dir / "metadata.json").write_text(
            json.dumps(_metadata_record(self.metadata_), indent=2)
        )

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        """Reconstruct from disk; the inverse of `save` (see docs/model_design.md §7).

        Warns if the runtime's sklearn/numpy/tensorly/pydmd versions differ
        from those recorded at training time (§9 gotcha 17).

        Parameters
        ----------
        model_dir : Path

        Returns
        -------
        Self
        """
        record = json.loads((model_dir / "metadata.json").read_text())
        _warn_on_version_drift(record)
        metadata = _metadata_from_record(record)
        diagnostics_records = json.loads((model_dir / "group_diagnostics.json").read_text())
        return cls.from_components(
            preprocess=joblib.load(model_dir / "preprocess.pkl"),
            tucker={t: _load_tucker(model_dir / "tucker" / f"{t}.npz") for t in metadata.tensors},
            core_copula={
                t: _load_copula(model_dir / "core_copula" / f"{t}.npz", t) for t in metadata.tensors
            },
            dmd={g: _load_dmd(model_dir / "dmd" / f"{g}.npz") for g in metadata.groups},
            mode_ou={g: _load_mode_ou(model_dir / "mode_ou" / f"{g}.npz") for g in metadata.groups},
            group_diagnostics={
                g: _diagnostics_from_record(
                    diagnostics_records[g], _load_lim(model_dir / "lim" / f"{g}.npz")
                )
                for g in metadata.groups
            },
            ocean_mask=np.load(model_dir / "ocean_mask.npy"),
            metadata=metadata,
        )

    def ocean_mask(self) -> np.ndarray:
        """Boolean ``(H, W)`` — True where the model can produce samples."""
        check_is_fitted(self, "ocean_mask_")
        return self.ocean_mask_


# --- Persistence helpers (§7) ----------------------------------------------


def _metadata_record(metadata: ModelMetadata) -> dict[str, Any]:
    """`ModelMetadata` as JSON-ready values (datetimes as ISO strings)."""
    record = asdict(metadata)
    for key in ("trained_at", "t_train_start", "t_train_end", "t_ref"):
        record[key] = getattr(metadata, key).isoformat()
    return record


def _metadata_from_record(record: dict[str, Any]) -> ModelMetadata:
    """Inverse of `_metadata_record`."""
    return ModelMetadata(
        trained_at=datetime.fromisoformat(record["trained_at"]),
        t_train_start=datetime.fromisoformat(record["t_train_start"]),
        t_train_end=datetime.fromisoformat(record["t_train_end"]),
        t_ref=datetime.fromisoformat(record["t_ref"]),
        variables=tuple(record["variables"]),
        tensors=tuple(record["tensors"]),
        groups={name: tuple(members) for name, members in record["groups"].items()},
        tucker_ranks={t: (int(r[0]), int(r[1])) for t, r in record["tucker_ranks"].items()},
        r_dmd=int(record["r_dmd"]),
        dmd_variant=str(record["dmd_variant"]),
        include_trend=bool(record["include_trend"]),
        include_local_residual=bool(record["include_local_residual"]),
        git_commit=str(record["git_commit"]),
        sklearn_version=str(record["sklearn_version"]),
        numpy_version=str(record["numpy_version"]),
        tensorly_version=str(record["tensorly_version"]),
        pydmd_version=str(record["pydmd_version"]),
    )


def _warn_on_version_drift(record: dict[str, Any]) -> None:
    """Warn for each library whose runtime version differs from the training one (gotcha 17)."""
    runtime = {
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "tensorly": tensorly.__version__,
        "pydmd": importlib.metadata.version("pydmd"),
    }
    for library, version in runtime.items():
        trained = record[f"{library}_version"]
        if trained != version:
            warnings.warn(
                f"model was trained with {library} {trained!r}, runtime has {version!r}; "
                "persisted artifacts may not load or behave identically.",
                stacklevel=3,
            )


def _save_tucker(path: Path, fit: TuckerFit) -> None:
    # Factors float32 on disk (§7 dtype policy); TuckerFit stays float64 in memory.
    np.savez(
        path,
        factor_lat=fit.factor_lat.astype(np.float32),
        factor_lon=fit.factor_lon.astype(np.float32),
        reconstruction_error=fit.reconstruction_error,
        n_iter=fit.n_iter,
    )


def _load_tucker(path: Path) -> TuckerFit:
    with np.load(path) as data:
        return TuckerFit(
            factor_lat=data["factor_lat"].astype(np.float64),
            factor_lon=data["factor_lon"].astype(np.float64),
            reconstruction_error=float(data["reconstruction_error"]),
            n_iter=int(data["n_iter"]),
        )


def _save_copula(path: Path, tensor: str, copula: GaussianCopula) -> None:
    # Fitted arrays, not a pickled object (§7): rebuilt by `_load_copula`.
    np.savez(
        path,
        bin_probs=copula.bin_probs_,
        sorted_core_binned=copula.sorted_anomalies_binned_[tensor],
    )


def _load_copula(path: Path, tensor: str) -> GaussianCopula:
    """Rebuild a fitted `GaussianCopula` from its lookup table."""
    with np.load(path) as data:
        copula = GaussianCopula(n_bins=int(data["bin_probs"].shape[0]))
        copula.bin_probs_ = data["bin_probs"]
        copula.sorted_anomalies_binned_ = {tensor: data["sorted_core_binned"]}
    return copula


def _save_dmd(path: Path, fit: DMDFit, members: tuple[str, ...], sizes: list[int]) -> None:
    np.savez(
        path,
        Phi=fit.Phi,
        Lambda=fit.Lambda,
        b0=fit.b0,
        lam_cont=fit.lam_cont,
        dt=fit.dt,
        variant=fit.variant,
        members=np.array(members, dtype=np.str_),
        member_sizes=np.array(sizes, dtype=np.int64),
    )


def _load_dmd(path: Path) -> DMDFit:
    with np.load(path) as data:
        return DMDFit(
            Phi=data["Phi"],
            Lambda=data["Lambda"],
            b0=data["b0"],
            lam_cont=data["lam_cont"],
            dt=float(data["dt"]),
            variant=str(data["variant"]),
        )


def _save_mode_ou(path: Path, fit: ModeOUFit) -> None:
    np.savez(
        path,
        mode=fit.mode,
        partner=fit.partner,
        phi=fit.phi,
        innovation_cov=fit.innovation_cov,
        log_magnitude_mean=fit.log_magnitude_mean,
        phase_offset=fit.phase_offset,
        dt=fit.dt,
    )


def _load_mode_ou(path: Path) -> ModeOUFit:
    with np.load(path) as data:
        return ModeOUFit(
            mode=data["mode"],
            partner=data["partner"],
            phi=data["phi"],
            innovation_cov=data["innovation_cov"],
            log_magnitude_mean=data["log_magnitude_mean"],
            phase_offset=data["phase_offset"],
            dt=float(data["dt"]),
        )


def _save_lim(path: Path, lim: LIMFit) -> None:
    np.savez(
        path,
        G=lim.G,
        L=lim.L,
        Q=lim.Q,
        C0=lim.C0,
        innovation_cov=lim.innovation_cov,
        lag=lim.lag,
        dt=lim.dt,
        q_min_eigenvalue=lim.q_min_eigenvalue,
        logm_imag_fraction=lim.logm_imag_fraction,
    )


def _load_lim(path: Path) -> LIMFit:
    with np.load(path) as data:
        return LIMFit(
            G=data["G"],
            L=data["L"],
            Q=data["Q"],
            C0=data["C0"],
            innovation_cov=data["innovation_cov"],
            lag=int(data["lag"]),
            dt=float(data["dt"]),
            q_min_eigenvalue=float(data["q_min_eigenvalue"]),
            logm_imag_fraction=float(data["logm_imag_fraction"]),
        )


def _diagnostics_record(d: GroupDiagnostics) -> dict[str, Any]:
    """Everything in `GroupDiagnostics` except the LIM matrices (those go to ``lim/``)."""
    return {
        "members": list(d.members),
        "explained_variance": d.explained_variance,
        "tau": {
            "training_lag": d.tau.training_lag,
            "lags": d.tau.lags.tolist(),
            "operator_drift": d.tau.operator_drift.tolist(),
            "propagator_error": d.tau.propagator_error.tolist(),
            "error_variance_ratio": d.tau.error_variance_ratio.tolist(),
        },
        "drift_upper": d.drift_upper.tolist(),
        "ratio_band": d.ratio_band.tolist(),
        "tau_consistent": d.tau_consistent.tolist(),
        "coupling": {
            "lag": d.coupling.lag,
            "r2_joint": d.coupling.r2_joint,
            "r2_independent": d.coupling.r2_independent,
            "granger_nats": d.coupling.granger_nats,
            "noise_mutual_information_nats": d.coupling.noise_mutual_information_nats,
            # JSON keys must be strings: the member pair becomes a list entry.
            "noise_pairwise_mutual_information_nats": [
                [a, b, value]
                for (a, b), value in d.coupling.noise_pairwise_mutual_information_nats.items()
            ],
        },
        "granger_null_upper": d.granger_null_upper,
        "noise_mi_null_upper": d.noise_mi_null_upper,
        "coupling_significant": d.coupling_significant,
        "noise_coupling_significant": d.noise_coupling_significant,
    }


def _diagnostics_from_record(record: dict[str, Any], lim: LIMFit) -> GroupDiagnostics:
    """Inverse of `_diagnostics_record`, given the group's LIM."""
    tau = record["tau"]
    coupling = record["coupling"]
    return GroupDiagnostics(
        members=tuple(record["members"]),
        explained_variance=dict(record["explained_variance"]),
        lim=lim,
        tau=TauTestResult(
            training_lag=int(tau["training_lag"]),
            lags=np.array(tau["lags"], dtype=np.int64),
            operator_drift=np.array(tau["operator_drift"], dtype=np.float64),
            propagator_error=np.array(tau["propagator_error"], dtype=np.float64),
            error_variance_ratio=np.array(tau["error_variance_ratio"], dtype=np.float64),
        ),
        drift_upper=np.array(record["drift_upper"], dtype=np.float64),
        ratio_band=np.array(record["ratio_band"], dtype=np.float64),
        tau_consistent=np.array(record["tau_consistent"], dtype=bool),
        coupling=GroupCouplingReport(
            lag=int(coupling["lag"]),
            r2_joint=dict(coupling["r2_joint"]),
            r2_independent=dict(coupling["r2_independent"]),
            granger_nats=dict(coupling["granger_nats"]),
            noise_mutual_information_nats=float(coupling["noise_mutual_information_nats"]),
            noise_pairwise_mutual_information_nats={
                (a, b): float(value)
                for a, b, value in coupling["noise_pairwise_mutual_information_nats"]
            },
        ),
        granger_null_upper=dict(record["granger_null_upper"]),
        noise_mi_null_upper=float(record["noise_mi_null_upper"]),
        coupling_significant=dict(record["coupling_significant"]),
        noise_coupling_significant=bool(record["noise_coupling_significant"]),
    )
