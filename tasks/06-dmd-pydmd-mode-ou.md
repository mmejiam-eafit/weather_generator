# Task 06 — DMD: replace hand-rolled exact DMD with PyDMD wrapper; fold mode-OU into `dmd.py`; delete `ou.py`

**Depends on:** none. **Blocks:** 07, 08, 09.

## Current state

`src/train/dmd.py`'s `exact_dmd()` hand-rolls DMD via `randomized_svd` +
`np.linalg.eig`. `src/train/ou.py` is a **separate module** holding
`OUFit`, `fit_ou` (the mode-level AR(1)→OU recovery), and the OU
**simulation** primitives (`sample_ou_equilibrium`, `simulate_ou_forward`,
`_complex_normal`) used both at fit time and by `weather/simulate.py` at
query time.

v2's directory layout (§8.1) lists only `tucker.py` and `dmd.py` under
`train/` — no `ou.py` — and §5.6.1 explicitly says the mode-OU fit "Lives in
`train/dmd.py` alongside `dmd_fit`."

**Decision for this migration** (resolves an ambiguity the design doc leaves
implicit): fold the *fitting* side of `ou.py` into `dmd.py`, renamed
`ModeOUFit`/`fit_mode_ou`. Move the *simulation/evolution* helpers
(`sample_ou_equilibrium`, `simulate_ou_forward`, `_complex_normal`) into
`weather/simulate.py` as private helpers (task 09) — they're runtime
query-path code, not a fitting concern, `weather/simulate.py` is their only
caller, and §5.8's pseudocode presents the evolution formulas inline at the
call site rather than via a shared module. Delete `train/ou.py` entirely;
nothing imports it after task 09 lands.

## Changes — `src/train/dmd.py`

1. `from pydmd import DMD, BOPDMD, HankelDMD`; `from typing import Literal`.
2. `DMDFit` dataclass gains a `variant: str` field (provenance: `"exact"`,
   `"bop"`, or `"hankel"`).
3. Rename `exact_dmd` → `dmd_fit`, full three-branch signature and body per
   §5.6:
   ```python
   def dmd_fit(
       core: np.ndarray, r_dmd: int, dt: float, rng: np.random.Generator,
       variant: Literal["exact", "bop", "hankel"] = "exact",
       bop_num_trials: int = 100, bop_trial_size: float = 0.5, hankel_delays: int = 20,
   ) -> DMDFit: ...
   ```
   - `"exact"`: `DMD(svd_rank=r_dmd)`.
   - `"bop"`: `BOPDMD(svd_rank=r_dmd, num_trials=bop_num_trials,
     trial_size=bop_trial_size, eig_constraints={"conjugate_pairs"})`, fit
     with an explicit `t=np.arange(T)*dt` grid (BOPDMD needs it; `DMD`/
     `HankelDMD` don't). **This branch's `.eigs` are already continuous-time**
     — `Lambda = exp(eigs * dt)` here specifically, unlike the other two
     branches whose `.eigs` are discrete-time and need no `exp()`.
   - `"hankel"`: `HankelDMD(svd_rank=r_dmd, d=hankel_delays)`.
   - PyDMD expects `(n_features, n_snapshots)` — **one `.T` at the boundary**
     (`core.reshape(T, N_state).T`), opposite of this codebase's internal
     `(T, ...)`-first convention (§9 gotcha 22: "Getting this wrong produces
     plausible-looking eigenvalues on the wrong basis"). The existing
     hand-rolled `exact_dmd` already has this orientation right by
     construction (`snapshots = core.reshape(n_times, -1).T`) — preserve
     that exact transpose point when swapping in PyDMD.
   - `lam_cont = log(Lambda) / dt`, computed once after the branch dispatch
     (works uniformly for all three variants since `Lambda` is normalized to
     discrete-time by then).
4. **Default `variant="exact"`, not `"bop"`** (§5.6, §8.5) — fast enough for
   repeated rank sweeps during notebook iteration; `"bop"`'s ~100-model
   bagging is reserved for the one final production fit (§8.6 "Production
   runs").
5. Add `ModeOUFit` dataclass + `fit_mode_ou`, formulas verbatim from §5.6.1:
   ```
   a_k(t) = k-th component of (Φ⁺ · x_t)
   â_k(t) = b0_k · exp(lam_cont_k · t)
   η_k(t) = a_k(t) − â_k(t)
   φ_mode(k) = Σ_t conj(η_k(t))·η_k(t+Δt) / Σ_t |η_k(t)|²
   θ_mode(k) = −log(clip(Re(φ_mode(k)), 1e-4, 0.9999)) / Δt
   σ_mode(k)² = Var(η_k)·2·θ_mode(k) / (1 − |φ_mode(k)|²)
   ```
   **Note `θ_mode(k)` clips `Re(φ_mode(k))`, not `|φ_mode(k)|`** — a
   negative real part (alternation/anti-correlation) is deliberately pushed
   to the fast-decorrelation end rather than assigned whatever slow decay
   `|φ|` alone would imply. `docs/model_design.md` §5.6.1 previously stated
   this with `|φ|`, which is wrong for any mode with `Re(φ) < 0`; fixed
   2026-07-17 to match v1's actual (correct, comment-justified)
   implementation.
   This is a straight port of v1's `dmd_residuals()` (compute `η_k(t)`) +
   `train/ou.py`'s `fit_ou()` (AR(1)→OU recovery) merged into one function
   in `dmd.py`; the numerics are unchanged from v1 (v1's `fit_ou` already
   implements this correctly), only the module location and dataclass name
   (`OUFit` → `ModeOUFit`) change. **Keep** v1's innovation-variance
   correction (`train/ou.py` lines 47-53's comment on why the AR(1)
   innovation variance, not the marginal variance, feeds the `sigma`
   formula) — it's a real correctness note, not v1 cruft to drop.
6. Complex-conjugate pair handling (§5.6, §9 gotcha 8): document in the
   docstring that `variant="bop"` handles this automatically via
   `eig_constraints`; `"exact"`/`"hankel"` require the *caller*
   (`weather/simulate.py`, task 09) to force
   `η_{k'}(t) = conj(η_k(t))` at every simulation step. `dmd_fit` itself does
   nothing extra here — just don't let this requirement get lost crossing
   the module boundary into task 09.

## Delete

- `src/train/ou.py` in full (fitting logic → `dmd.py` above; simulation
  logic → `weather/simulate.py`, task 09).

## Tests

- Delete `tests/train/test_ou.py`; its coverage is absorbed into
  `tests/train/test_dmd.py::test_mode_ou_parameter_recovery`.
- **`tests/train/test_dmd.py`** (§12.5):
  - `test_analytic_traveling_wave` — synthetic `A cos(kx − ωt)` recovered as
    a complex-conjugate pair with correct `ω`, tested against **each**
    variant (`"exact"`, `"bop"`, `"hankel"`).
  - `test_bop_enforces_conjugate_pairs` — BOP-DMD output modes come as
    conjugate pairs after fitting with `eig_constraints={"conjugate_pairs"}`.
  - `test_pydmd_snapshot_orientation` — verifies the `.T` is applied
    correctly (e.g. feed an asymmetric-shape core and confirm no silent
    transpose-mismatch).
  - `test_mode_ou_parameter_recovery` — synthetic OU trace with known
    `(θ, σ)` recovered within 5% (replaces the deleted `test_ou.py`'s
    equivalent case).

## Acceptance criteria

- `dmd_fit(core, r_dmd, dt, rng, variant="exact")` numerically matches the
  old `exact_dmd`'s output on the existing analytic-traveling-wave test.
- All three variants pass `test_analytic_traveling_wave`.
- `grep -rn "train\.ou\|\bOUFit\b" src/ tests/` returns nothing after this
  task (task 09 reintroduces `_complex_normal`/equilibrium-draw logic
  locally in `simulate.py`, under new names — not by re-importing `ou.py`).
