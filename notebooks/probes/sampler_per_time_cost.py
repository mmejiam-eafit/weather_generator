"""Cost of the sampler's per-cell inverses and interpolator per unique query time at era-3
grid size: about 290 ms + 326 ms (CHANGELOG, 2026-11-22, "Considered, not done").

Run from the repo root: `poetry run python notebooks/probes/sampler_per_time_cost.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import time
import numpy as np, xarray as xr
from sklearn.pipeline import Pipeline
from core.config import grid_coordinates
from train.climatology import ClimatologyRemover
from train.standardize import PerCellStandardizer
from train.reproject import Reprojector, lonlat_interpolator
lats, lons = grid_coordinates("wave")
names = ["logE_sigwh", "logE_wsh", "T_wsp", "u_wsea", "v_wsea", "u_wind", "v_wind"]
rng = np.random.default_rng(0)
def ds(n):
    t = np.datetime64("2025-10-27T00:00:00") + np.arange(n) * np.timedelta64(3, "h")
    return xr.Dataset({v: (("time", "wave_lat", "wave_lon"), rng.standard_normal((n, lats.size, lons.size)).astype(np.float64)) for v in names},
                      coords={"time": t, "wave_lat": lats, "wave_lon": lons})
per_cell = Pipeline([("reproject", Reprojector()), ("climatology", ClimatologyRemover(diurnal_variables=["u_wind", "v_wind"])), ("standardize", PerCellStandardizer())]).fit(ds(6))
grid = ds(8)
t0 = time.perf_counter(); out = per_cell.inverse_transform(grid); t1 = time.perf_counter()
print(f"per-cell inverse, 8 unique times x 7 tensors, full grid: {t1 - t0:.2f} s ({(t1 - t0) / 8 * 1000:.0f} ms per unique time)")
fields = np.stack([out[v].values[0] for v in names])
t0 = time.perf_counter(); f = lonlat_interpolator(fields, lats, lons); v = f(np.array([[10.0, 200.0]])); t1 = time.perf_counter()
print(f"interpolator build + 1 point, one unique time: {(t1 - t0) * 1000:.0f} ms")
