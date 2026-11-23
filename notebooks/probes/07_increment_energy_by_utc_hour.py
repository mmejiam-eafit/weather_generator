"""3 h increment energy and state energy of the whole Gaussianized core, by UTC hour.
Flat to within 0.7%: the full core is dominated by noise-like coefficients.
Reads task 07's cached production core under `results/task07/gaussianize/`. Ran without a memory cap; it loads the 2.5 GB core.

Run from the repo root: `poetry run python notebooks/probes/07_increment_energy_by_utc_hour.py`.

Scratch probe, kept as the evidence behind the result above; not maintained
to repo standards and may change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "task07" / "gaussianize"
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

import numpy as np
CORE = str(RESULTS / "wsh_r248x576_gaussianized_core.npy")
core = np.load(CORE, mmap_mode="r")
print("shape", core.shape, core.dtype)
T = core.shape[0]
hours = (np.arange(T) * 3) % 24
inc = np.zeros(T - 1, dtype=np.float64)
energy = np.zeros(T, dtype=np.float64)
prev = None
for start in range(0, T, 64):
    block = np.asarray(core[start:start + 64], dtype=np.float64).reshape(-1, core.shape[1] * core.shape[2])
    energy[start:start + block.shape[0]] = np.einsum("ij,ij->i", block, block)
    if prev is not None:
        d = block[0] - prev
        inc[start - 1] = d @ d
    d = np.diff(block, axis=0)
    inc[start:start + block.shape[0] - 1] = np.einsum("ij,ij->i", d, d)
    prev = block[-1]
mean_inc = inc.mean()
mean_e = energy.mean()
print("step (UTC h -> h+3)   mean |dz|^2 / overall   mean |z|^2 / overall")
for h in range(0, 24, 3):
    sel = hours[:-1] == h
    sel_e = hours == h
    print(f"  {h:02d} -> {(h + 3) % 24:02d}           {inc[sel].mean() / mean_inc:6.3f}               {energy[sel_e].mean() / mean_e:6.3f}")
# day-to-day robustness: fraction of days where the 00->03 or 21->00 step is the day's largest increment
days = inc[: (T - 1) // 8 * 8].reshape(-1, 8)
arg = days.argmax(axis=1)
print("argmax-step histogram over days (index = start hour/3):", np.bincount(arg, minlength=8))
