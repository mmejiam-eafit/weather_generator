"""The cancellation finding: Phi's Gram condition number (4.6e3), near-duplicate modes
(|cos| 0.965), training span energy against the per-mode sum (15x), sampler span energy
(31-34x), and the random-phase test (27x) (`docs/model_design.md` §5.6.1, "Known
defect"). Executes the top of `07_production_sampler_memory.py`.

Run from the repo root: `poetry run python notebooks/probes/07_mode_cancellation.py`.

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
SP = str(RESULTS)  # cache directory; was the session scratchpad
src = open(HERE / "07_production_sampler_memory.py").read()
exec(src.split("# --- per-channel")[0])

Phi = dmd.Phi.astype(np.complex128)
G = Phi.conj().T @ Phi  # Gram
norms = np.sqrt(np.real(np.diag(G)))
C = np.abs(G) / np.outer(norms, norms)
np.fill_diagonal(C, 0)
print(f"mode norms quantiles {np.round(np.quantile(norms, [0, .5, 1]), 3)}; cond(Gram) {np.linalg.cond(G):.2e}")
print(f"max |cos| between distinct modes {C.max():.4f}; pairs of modes with |cos|>0.9 (excluding conjugates): ", end="")
conj_of = {k: int(np.argmin(np.abs(dmd.Lambda - np.conj(dmd.Lambda[k])))) for k in range(len(norms))}
hits = [(i, j, C[i, j]) for i in range(len(norms)) for j in range(i + 1, len(norms)) if C[i, j] > 0.9 and conj_of[i] != j]
print(len(hits), [(i, j, round(c, 3)) for i, j, c in hits[:8]])

def energy(a):  # sum_t ||Re(Phi a_t)||^2 = sum_t 0.5*(a^H G a + Re(a^T Phi^T Phi a))
    H = Phi.T @ Phi
    e = 0.5 * (np.real(np.einsum("it,ij,jt->t", a.conj(), G, a)) + np.real(np.einsum("it,ij,jt->t", a, H, a)))
    return e
steps = np.arange(T, dtype=np.float64)
a_train = dmd.b0[:, None] * dmd.Lambda[:, None] ** steps[None, :] + eta
e_train = energy(a_train)
incoh_train = 0.5 * np.sum(np.abs(a_train)**2 * norms[:, None]**2, 0)  # no cross terms (pair conj terms aside)
real_energy = 0.0
for s in range(0, T, CH): real_energy += np.sum(np.asarray(X[s:s + CH], dtype=np.float64)**2)
print(f"training: ||Re(Phi a)||^2 summed {e_train.sum():.4e} | diagonal-only (no cross-mode terms) {incoh_train.sum():.4e} "
      f"| raw core energy {real_energy:.4e}")
for seed in range(3):
    rng = np.random.default_rng(seed)
    a = mode_amplitudes(dmd, ou, _channel_path(ou, steps, rng), steps)
    e = energy(a)
    print(f"sampler seed {seed}: ||Re(Phi a)||^2 summed {e.sum():.4e} = {e.sum()/e_train.sum():.1f}x training span energy")
# a check of the cancellation hypothesis: training amplitudes with each pair's phase re-randomized per mode (fixed offset)
rng = np.random.default_rng(0)
shuffled = a_train.copy()
pair = ou.partner >= 0
for rep, par in zip(ou.mode[pair], ou.partner[pair]):
    rot = np.exp(1j * rng.uniform(0, 2 * np.pi))
    shuffled[rep] *= rot; shuffled[par] *= np.conj(rot)
print(f"training amplitudes, each pair's phase rotated by a random constant: {energy(shuffled).sum()/e_train.sum():.1f}x")
