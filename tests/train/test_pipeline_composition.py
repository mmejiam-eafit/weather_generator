"""Cross-transformer composition tests for train.directional (see docs/model_design.md §3.7, §5.3)."""

import xarray as xr
from numpy.testing import assert_allclose

from train.directional import ComingFromToUV, GoingToToUV


def test_transformers_commute_on_disjoint_pairs(
    small_wave_dataset: xr.Dataset, small_wind_dataset: xr.Dataset
) -> None:
    """ComingFromToUV → GoingToToUV and the reverse order produce identical output.

    The two transformers act on disjoint variable sets (§5.3: "This makes
    composition in either order commutative for disjoint pair sets"), so
    either ordering in the preprocessing Pipeline must be equivalent.
    """
    combined = xr.merge([small_wave_dataset, small_wind_dataset])
    coming_from = ComingFromToUV(pairs=[("wsh", "wsd", "u_wsea", "v_wsea")])
    going_to = GoingToToUV(pairs=[("was", "wad", "u_wind", "v_wind")])

    forward = going_to.fit_transform(coming_from.fit_transform(combined))
    backward = coming_from.fit_transform(going_to.fit_transform(combined))

    assert set(forward.data_vars) == set(backward.data_vars)
    for name in forward.data_vars:
        assert_allclose(forward[name].values, backward[name].values, atol=1e-10)
