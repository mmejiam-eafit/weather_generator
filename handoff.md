# Session Handoff

**Target reader:** a new agent or person picking up where this conversation left off.

**Scope:** this document covers only what happened in this working session — repo
orientation, some environment setup, and exploratory notebook work culminating in an
open design question about the training pipeline. For general repo orientation, read
[`context.md`](context.md), [`AGENTS.md`](AGENTS.md), and
[`docs/model_design.md`](docs/model_design.md) first — they're authoritative and this
document doesn't repeat their content.

---

## 1. Repo orientation (no code changes)

Reviewed `stochastic_weather_generator_handoff.md` (root, v1 design doc — historical,
superseded by `docs/model_design.md`'s v2) and `v3_implementation_handoff.md` (root,
untracked, proposes post-hoc physical-consistency filtering).

**Finding, confirmed against actual source:** `v3_implementation_handoff.md`'s premises
don't hold up:
- It claims `docs/model_design.md` §3.15/§5.9 already spec the filtering — false; §3 ends
  at §3.14, §5 ends at §5.8 (confirmed by grepping headings).
- It claims "v2 is implemented and running" — false; per `tasks/README.md`, tasks 09–13
  are still ⬜ (sampling rewrite, hurricane-season validation, notebooks, integration
  tests/import boundaries, final docs/release).
- It conflicts with `AGENTS.md`'s "Physical consistency (documented limitation)" section:
  post-hoc projection/rejection layers require explicit authorization, and the fix
  belongs in the copula (t-copula/GPD tails), not a downstream patch.

This matches what `context.md` already flags under "Known open issues" — **don't trust
`v3_implementation_handoff.md` at face value**, and don't implement from it without
re-scoping and explicit authorization.

**Actual next step in the v2 migration (per `tasks/README.md`):** task 09, the sampling
rewrite (`sample_batch`/`simulate_field` currently only exercise the coherent layer;
local-residual noise and DMD conjugate-pair forcing aren't wired in).

## 2. Environment setup

- `data/` is now a symlink to `/Users/mmejiam9206/Documents/weather_data` (that target
  directory exists and has its own `data/06`, `data/07` month subfolders — i.e. the
  archive's `MM/DD/HH/variable.npy` layout, no `YYYY` folder observed yet in what's been
  browsed so far).

## 3. Exploratory notebooks (new, untracked, since committed by the user)

Six notebooks exist under `notebooks/`; this session touched the first two in detail:

- `00_check_data_sources.ipynb` — loads `sst.npy` snapshots and stacks them. Fixed
  `np.vstack` (wrong axis semantics) → `np.stack(ssts, axis=-1)` to get the intended
  `(1121, 2880, 12)` shape (lat, lon, time).
- `01_understand_tucker_decomposition.ipynb` — the bulk of this session's work. Current
  end state:
  - Loads and stacks `sst` the same way as notebook 00, upcasting each snapshot to
    `float64` at load time (the on-disk dtype is `float16`, which both overflows
    silently at moderate values and isn't supported by LAPACK/`linalg` — hence the
    original `TypeError: array type float16 is unsupported in linalg`).
  - `plot_3d_volume(array, ...)` — a reusable Plotly `Volume` helper: downsamples each
    axis to ≤60 points, leaves NaNs in place (dropping them breaks the regular-grid
    assumption `go.Volume` relies on), and **explicitly sets the 3D scene's axis
    ranges from the real data extents**. That last part was a real bug fix: Plotly's
    autorange silently fails on NaN-heavy volumes (~30% land/ice in `sst`) and falls
    back to its default `[-1, 1]` scene, which renders as an apparently-empty cube even
    though the underlying `X`/`Y`/`Z`/`value` arrays are correct — verified this
    against the real archive data directly (not just guessed).
  - Ocean-mask cell: builds `ocean_mask`/`tucker_mask`/`sst_filled`, explicitly modeled
    on `src/weather/model.py`'s `_mask_and_stack` (same "non-NaN across every
    snapshot" construction) but **deliberately diverging** from it — production ravels
    `(lat, lon)` into one `ocean_cell` Tucker mode and drops land cells from the tensor
    entirely; this notebook instead keeps lat/lon as two separate Tucker modes (for
    pedagogical exploration of spatial factors) and passes the mask into TensorLy's
    `partial_tucker(mask=...)`, which iteratively imputes masked entries from the
    low-rank reconstruction. **This was an explicit user choice**, made when asked to
    pick between the two approaches — see §4 below before assuming it should change.
  - Two latent bugs fixed in the `partial_tucker` call, both previously masked by the
    original `LinAlgError: SVD did not converge`:
    - `time_rank`/`latitude_rank`/`longitude_rank` were mapped to the wrong axes
      (`sst_full.shape[0]` is lat, not time). Now `latitude_rank`/`longitude_rank` are
      arbitrary as before; `time_rank = sst_full.shape[-1]` (full rank, matching the
      "keep time full-rank" convention used elsewhere in the design), and `rank=` in
      the `partial_tucker` call is ordered to match `modes=[0, 1, 2]` = (lat, lon,
      time).
    - Return-value unpacking was wrong: `partial_tucker` returns `(core, factors),
      rec_errors` where `factors` is *one list* of 3 arrays, not 3 separate values.
      Fixed to `(matrix_core, factors), errors = partial_tucker(...)` followed by
      `latitude_factors, longitude_factors, time_factors = factors`.
  - A new cell renders `sst_filled` (the masked/imputed array actually passed to
    Tucker) right after the raw `sst_full` render, for visual comparison.

Notebooks `02`–`05` (DMD, Gaussian transform/copulas, DMD-of-Tucker,
Gaussianization-on-DMD) exist but weren't touched this session — worth checking their
state before assuming they're consistent with the pipeline understanding below.

## 4. Design discussion: does `src/train/tucker.py` need a mask?

User asked to add mask support + a mask-building function to `tucker.py`, reasoning
from the notebook finding that masking was "imperative" for the NaN land values.

**Pushed back, and the user agreed to stop and not make the change.** Reasoning: the
notebook's mask requirement was an artifact of its own divergent design choice (§3
above) — keeping the full `(lat, lon)` grid with scattered NaN instead of raveling to
an ocean-cell vector first. `tucker_fit`'s `Z` input is already guaranteed fully finite
by construction (`_mask_and_stack` in `src/weather/model.py` masks and ravels land
cells out *before* `tucker_fit` ever runs — see its docstring at
`src/train/tucker.py:44-46`). Adding mask support to `tucker.py` would either be a
no-op or a real algorithmic change (switching from exact decomposition on a pre-masked
ocean-cell vector to iterative NaN-imputation on a raw grid), which would ripple into
DMD/OU/`WeatherModel` (all assume `N_ocean`-shaped modes) and contradict the
`(time, ocean_cell, variable)` mode structure documented in `docs/model_design.md`
§5.5.

**Status: open, deliberately unresolved.** The user said "let's step back and re-define
the process" — no concrete next step was agreed. If this comes back up, the real
question is probably "should production's NaN-handling strategy change at all," which
is a separate, bigger question from anything the notebook forced.

Separately (informational, not actionable): explained why `tucker.py`'s
`reconstruction_error` re-derives `Z_recon` from the original `Z` via
`multi_mode_dot(Z, [U.T, ...])` → `multi_mode_dot(..., [U, ...])` rather than reusing
`partial_tucker`'s own discarded `_core` return value — it's deliberately exercising
the same round trip `tucker_transform`/`tucker_inverse` use at serving time (no `core`
is ever persisted in `TuckerFit`), and it sidesteps a documented tensorly
version-fragility issue with trusting the returned core's exact semantics
(`tucker.py:63-65`).

## 5. Open thread: should Gaussianization move downstream of Tucker?

**This is the live, unresolved question — the most important thing for whoever
continues this to pick up.**

User is reconsidering whether per-cell Gaussianization (`GaussianCopula`, currently
step 6 of the `preprocess` sklearn `Pipeline` in `src/weather/model.py:217-248`) should
happen before Tucker at all, on the grounds that "the normalization of a weather
variable needs to be done differently."

I produced a mermaid diagram of the **actual current pipeline** (read directly from
`_fit_impl` in `src/weather/model.py`, not the design doc, since the design doc can
drift from implementation) and posted it in-conversation. Current order, for reference:

```
load_training_window
  → ComingFromToUV → GoingToToUV → Reprojector → ClimatologyRemover
  → PerCellStandardizer → GaussianCopula          [[ "preprocess" Pipeline ]]
  → _mask_and_stack (ocean mask, ravel to N_ocean) → Z: (T, N_ocean, V)
  → tucker_fit / tucker_transform → core: (T, r_ocean, r_var)
  → dmd_fit → fit_mode_ou
  → (if include_local_residual) deterministic DMD reconstruction
      → coherent_reconstruction (Tucker-inverse) → fit_local_residual
  → WeatherModel
```

Gaussianization currently operates **per-cell, on the full wave grid**, on
standardized anomalies — i.e. before any spatial compression. Everything downstream
(Tucker's HOSVD, DMD's linear dynamics, both OU fits) assumes it's operating on
already-Gaussian data; that's the explicit rationale in `docs/model_design.md` §3.5
("DMD and OU are linear-Gaussian; weather variables are skewed and bounded... Interpose
an empirical-CDF Gaussian copula per cell").

**Where I left it:** asked the user where they're thinking of moving Gaussianization
to — after Tucker (Gaussianizing the compressed `core` instead of raw grid cells),
after DMD (Gaussianizing residual amplitudes), or somewhere else — since "per-cell"
stops meaning a physical grid cell once you're past Tucker's `ocean_cell` basis. **No
answer was given before this session ended.**

### What the next agent/person needs to know before touching this

- Moving Gaussianization is not a mechanical reordering of Pipeline steps. The whole
  downstream linear-Gaussian machinery (Tucker HOSVD init, DMD eigendecomposition, both
  OU fits) is designed around operating on Gaussian input; moving the copula changes
  what statistical assumptions each stage is actually valid under.
  - `docs/model_design.md` §3.5 and §5.4 are the current source of truth for why
    Gaussianization is where it is and how it's implemented (empirical CDF → Φ⁻¹
    forward, LUT-based inverse at sample time).
  - `_mask_and_stack` computes the ocean mask from **already-Gaussianized** data
    (`~np.isnan(gaussianized[name].values)`) — if Gaussianization moves after masking,
    check whether mask construction needs to move too, or start from
    pre-Gaussianization anomalies instead.
  - Tucker operates jointly across ocean cells *and variables* — Gaussianizing
    post-Tucker would mean Gaussianizing linear combinations of cells/variables
    (Tucker-core entries), not physical quantities. Whether an empirical-CDF-per-mode
    copula is even meaningful there is an open question, not a given.
- This is a training-pipeline architecture question, not a notebook experiment — same
  caution applies as in §4: don't implement changes to `src/weather/model.py` or
  `src/train/*` without the user confirming the actual target design first, especially
  since it also touches `docs/model_design.md`'s documented algorithm (which would need
  updating to match, not the reverse — per this repo's own convention seen in
  `v3_implementation_handoff.md` §8.1).

## 6. Quick pointers for continuing

- Read `context.md` for general v2-migration status (tasks 09–13 outstanding) — don't
  re-derive it, it's already current as of this session.
- Read `AGENTS.md` before touching training code — binding conventions, layering
  rules, and the physical-consistency-limitation policy referenced in §1 and §4 above.
- If picking up the Gaussianization question: get an explicit answer to "where should
  it move to" before writing any code — the mermaid diagram in this document (§5) is
  the shared reference point for that conversation.
- Notebooks `02`–`05` weren't reviewed this session; check their state before assuming
  they reflect the pipeline as currently understood.
