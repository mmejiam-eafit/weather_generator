"""Computes the core's 400 leading PCs, caches them as `results/task07/gaussianize/
wsh_pcs400.npy` (other probes read it; run this first), and prints increment and state
energy of the top 5/20/100 PCs by UTC hour.
Reads task 07's cached production core under `results/task07/gaussianize/`. Ran without a memory cap; it loads the 2.5 GB core.

Run from the repo root: `poetry run python notebooks/probes/07_leading_pcs_by_utc_hour.py`.

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
from train.linear_inverse import group_pcs
SP = str(RESULTS)  # cache directory; was the session scratchpad
CORE = str(RESULTS / "wsh_r248x576_gaussianized_core.npy")
core = np.asarray(np.load(CORE, mmap_mode="r"))
g = group_pcs({"wsh": core}, {"wsh": 400})
np.save(f"{SP}/wsh_pcs400.npy", g.pcs)
print("explained by 400:", g.explained_variance)
z = g.pcs
T = z.shape[0]
hours = (np.arange(T) * 3) % 24
for k in (5, 20, 100):
    zk = z[:, :k]
    d = np.diff(zk, axis=0)
    inc = np.sum(d**2, axis=1)
    e = np.sum(zk**2, axis=1)
    print(f"--- top {k} PCs: mean|dz|^2/mean|z|^2 = {inc.mean()/e.mean():.3f}")
    print("   step      |dz|^2 rel   |z|^2 rel")
    for h in range(0, 24, 3):
        print(f"   {h:02d}->{(h+3)%24:02d}     {inc[hours[:-1]==h].mean()/inc.mean():6.3f}      {e[hours==h].mean()/e.mean():6.3f}")
