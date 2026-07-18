"""Decompose (speed, direction) pairs into (u, v) vector components.

The archive stores directions in mixed conventions (see docs/model_design.md
§2.4): most fields are "coming from" but `wad` and `ocd` are "going to".
Rather than smuggle a per-variable convention flag through a single
decoder, two atomic transformer classes each handle one convention
explicitly (see §3.7, §5.3):

- `ComingFromToUV` — for ``(speed, theta_from)`` pairs.
- `GoingToToUV` — for ``(speed, theta_to)`` pairs.

Both act on disjoint variable sets, run before `train.reproject.Reprojector`
in the preprocessing Pipeline (so reprojection interpolates linear-safe
``(u, v)`` fields instead of angles), and can be composed in either order.
Neither hardcodes a canonical pair list -- the caller states pairs
explicitly at construction. The canonical tier assignment lives in
`weather.model`'s pipeline construction (§5.3).
"""

from typing import Self

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ComingFromToUV(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Decompose (speed, theta_from) pairs into (u, v) components.

    theta_from is meteorological "coming from", CW from N (see
    docs/model_design.md §2.4, §3.7, §5.3).

    Runs before `train.reproject.Reprojector` in the preprocessing
    Pipeline, so that spatial reprojection interpolates linear-safe
    ``(u, v)`` fields instead of angles (broken across the 0°/360°
    discontinuity).
    """

    def __init__(self, pairs: list[tuple[str, str, str, str]]) -> None:
        # Each pair: (speed_var, direction_var, u_out_name, v_out_name).
        # Required, no default (§3.7: "no silent defaulting" -- an agent
        # adding a new directional variable must explicitly choose a
        # transformer). sklearn convention: store verbatim, validate lazily
        # in fit().
        self.pairs = pairs

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        """Record the (speed, direction, u, v) pairs this transformer owns.

        Parameters
        ----------
        X : xr.Dataset
            Training data, e.g. the output of `train.ingest.load_training_window`.
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self
        """
        self.pairs_ = tuple(self.pairs)
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Replace each owned (speed, theta_from) pair with (u, v).

        Parameters
        ----------
        X : xr.Dataset
            Data containing the pairs stated at construction.

        Returns
        -------
        xr.Dataset
            `X` with each ``(speed_var, direction_var)`` pair replaced by
            ``(u_out_name, v_out_name)``. Variables this transformer does
            not own -- both non-directional variables and pairs owned by
            `GoingToToUV` -- pass through unchanged (§3.7, §5.3).
        """
        check_is_fitted(self, "pairs_")
        result = X.copy()
        for speed_var, direction_var, u_out_name, v_out_name in self.pairs_:
            speed = result[speed_var]
            theta_to = (result[direction_var] + 180.0) % 360.0
            theta_to_rad = np.deg2rad(theta_to)
            result[u_out_name] = speed * np.sin(theta_to_rad)
            result[v_out_name] = speed * np.cos(theta_to_rad)
            result = result.drop_vars([speed_var, direction_var])
        return result

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        """Recombine each owned (u, v) pair back into (speed, theta_from).

        Parameters
        ----------
        X : xr.Dataset
            Data containing the ``(u_out_name, v_out_name)`` pairs produced
            by `transform`.

        Returns
        -------
        xr.Dataset
            `X` with each ``(u_out_name, v_out_name)`` pair replaced by
            ``(speed_var, direction_var)``, with `direction_var` restored
            in its native "coming from" convention (§2.4). Non-owned
            variables pass through unchanged.
        """
        check_is_fitted(self, "pairs_")
        result = X.copy()
        for speed_var, direction_var, u_out_name, v_out_name in self.pairs_:
            u = result[u_out_name]
            v = result[v_out_name]
            # arg order (sin, cos) -- i.e. (u, v) -- not (y, x) (§2.4).
            theta_to = np.degrees(np.arctan2(u, v)) % 360.0
            result[speed_var] = np.hypot(u, v)
            result[direction_var] = (theta_to + 180.0) % 360.0
            result = result.drop_vars([u_out_name, v_out_name])
        return result


class GoingToToUV(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Decompose (speed, theta_to) pairs into (u, v) components.

    theta_to is "going to", CW from N (see docs/model_design.md §2.4, §3.7,
    §5.3).

    Runs before `train.reproject.Reprojector` in the preprocessing
    Pipeline, so that spatial reprojection interpolates linear-safe
    ``(u, v)`` fields instead of angles (broken across the 0°/360°
    discontinuity).
    """

    def __init__(self, pairs: list[tuple[str, str, str, str]]) -> None:
        # Each pair: (speed_var, direction_var, u_out_name, v_out_name).
        # Required, no default (§3.7: "no silent defaulting"). sklearn
        # convention: store verbatim, validate lazily in fit().
        self.pairs = pairs

    def fit(self, X: xr.Dataset, y: None = None) -> Self:
        """Record the (speed, direction, u, v) pairs this transformer owns.

        Parameters
        ----------
        X : xr.Dataset
            Training data, e.g. the output of `train.ingest.load_training_window`.
        y : None
            Ignored (sklearn convention).

        Returns
        -------
        self
        """
        self.pairs_ = tuple(self.pairs)
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Replace each owned (speed, theta_to) pair with (u, v).

        Parameters
        ----------
        X : xr.Dataset
            Data containing the pairs stated at construction.

        Returns
        -------
        xr.Dataset
            `X` with each ``(speed_var, direction_var)`` pair replaced by
            ``(u_out_name, v_out_name)``. Variables this transformer does
            not own -- both non-directional variables and pairs owned by
            `ComingFromToUV` -- pass through unchanged (§3.7, §5.3).
        """
        check_is_fitted(self, "pairs_")
        result = X.copy()
        for speed_var, direction_var, u_out_name, v_out_name in self.pairs_:
            speed = result[speed_var]
            theta_to_rad = np.deg2rad(result[direction_var])
            result[u_out_name] = speed * np.sin(theta_to_rad)
            result[v_out_name] = speed * np.cos(theta_to_rad)
            result = result.drop_vars([speed_var, direction_var])
        return result

    def inverse_transform(self, X: xr.Dataset) -> xr.Dataset:
        """Recombine each owned (u, v) pair back into (speed, theta_to).

        Parameters
        ----------
        X : xr.Dataset
            Data containing the ``(u_out_name, v_out_name)`` pairs produced
            by `transform`.

        Returns
        -------
        xr.Dataset
            `X` with each ``(u_out_name, v_out_name)`` pair replaced by
            ``(speed_var, direction_var)``, with `direction_var` restored
            in its native "going to" convention (§2.4) -- no `+180` shift.
            Non-owned variables pass through unchanged.
        """
        check_is_fitted(self, "pairs_")
        result = X.copy()
        for speed_var, direction_var, u_out_name, v_out_name in self.pairs_:
            u = result[u_out_name]
            v = result[v_out_name]
            # arg order (sin, cos) -- i.e. (u, v) -- not (y, x) (§2.4).
            result[speed_var] = np.hypot(u, v)
            result[direction_var] = np.degrees(np.arctan2(u, v)) % 360.0
            result = result.drop_vars([u_out_name, v_out_name])
        return result
