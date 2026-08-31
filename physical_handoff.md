# Weather Generator: Pre-Tucker Transforms and Multi-Channel Decomposition Redesign

## Context for the agent

I am revisiting the stochastic weather generator (`src/weather/`) after a break. The design has shifted since the last implementation pass, and I've made unshared changes to the variable-handling code that you'll need to read before proposing edits. **Before making any changes, read the current state of `src/weather/` and confirm what's already in place vs. what's described below.** Do not assume the codebase matches an earlier design doc.

The user (Miguel) has been iterating on this design in a separate thread. This document captures the design decisions from that thread that need to land in code. There are caveats and constraints the design assistant is not aware of — surface any conflicts you find between this spec and the actual code, and ask before resolving them rather than silently reconciling.

---

## Architectural summary

The pipeline is:

```
raw fields per variable
  → per-variable pre-transforms (physical, monotone)
  → per-variable Tucker decomposition (lat, lon, t [, channel])
  → DMD on the temporal factor A_t
  → OU process per DMD mode
  → Gaussianization of coefficient time series
  → Gaussian copula across all variables' coefficient series
```

**Gaussianization has moved from pre-Tucker (blanket normal-score on raw fields) to after Tucker,** on the coefficient time series. The mild physical pre-transforms (log for magnitudes/periods, u/v for directions) tame the tails enough that Tucker's leading modes reflect typical variability rather than outlier storms, and the post-Tucker gaussianization handles residual non-Gaussianity right before OU/copula (which is where Gaussianity actually matters mechanically).

**Per-variable-group Tucker.** Instead of one monolithic decomposition across all variables, there is one Tucker per variable group (each Class 0 scalar, each Class 3 magnitude+direction group, each Class 4 partitioned wave triplet). Cross-variable coupling is picked up downstream by the copula. Within-group coupling (u/v/E/T for the same partition) is picked up by the joint Tucker's channel factor.

---

## Variable classification

Variables fall into three classes based on how many channels they contribute to a single Tucker decomposition. Each class has its own tensor shape entering Tucker.

### Class 0 — Scalar variables

Single measurement per (lat, lon, t). Example: `sst`.

- **Tensor shape entering Tucker:** `(lat_size, lon_size, t)`
- **Tucker ranks:** `(R_lat, R_lon, R_t)`
- **Factor matrices:** `A_lat (lat × R_lat)`, `A_lon (lon × R_lon)`, `A_t (t × R_t)`
- **Core:** `G` of shape `(R_lat, R_lon, R_t)`
- **Pre-transforms:** none by default; may apply `log` per-variable if the marginal is heavy-tailed and positive-definite.

### Class 3 — Magnitude + direction variables

Two raw inputs (magnitude, direction), decomposed jointly. Examples: `ocs`/`ocd`, `was`/`wad`.

- **Channels:** `(u, v, E)` where
  - `u = M · cos(θ)`
  - `v = M · sin(θ)`
  - `E = 2 · log(M)`
- **Tensor shape entering Tucker:** `(lat_size, lon_size, t, 3)`
- **Tucker ranks:** `(R_lat, R_lon, R_t, R_C)` with `R_C ≤ 3`
- **Factors:** as Class 0 plus `A_C (3 × R_C)` — the channel factor.
- **Open decision for currents (`ocs`/`ocd`):** consider dropping the `E` channel and using just `(u, v)` (2-channel variant). Current magnitudes have narrow dynamic range and are close to Gaussian in u/v already; the `E` channel may not earn its keep. For wind (`was`/`wad`), keep `E` — wind is heavy-tailed. Decide empirically by comparing reconstruction error at a fixed rank budget on 2022–2025 training data.

### Class 4 — Magnitude + direction + period variables

Three raw inputs (magnitude, direction, period), decomposed jointly. Examples: partitioned waves `pwh`/`pwp`/`pwd`, `pswh`/`pswp`/`pswd`, `wsh`/`wsp`/`wsd`.

- **Channels:** `(u, v, E, T)` where
  - `u = Hs · cos(θ)`
  - `v = Hs · sin(θ)`
  - `E = 2 · log(Hs)`
  - `T = log(T_period)`
- **Tensor shape entering Tucker:** `(lat_size, lon_size, t, 4)`
- **Tucker ranks:** `(R_lat, R_lon, R_t, R_C)` with `R_C ≤ 4`
- **Factors:** as Class 0 plus `A_C (4 × R_C)`.

**Why 4 channels and not 4 axes.** The channel mode has *size* 4 (indexed by channel identity: 0=u, 1=v, 2=E, 3=T). It is not four separate axes. This is a single mode-4 tensor of shape `(lat, lon, t, 4)`, not a mode-7 tensor of shape `(lat, lon, u, v, E, T, t)`. `A_C` is a 4×R_C matrix whose columns are the low-rank basis directions of channel coupling — PCA-over-channels embedded inside the spatiotemporal Tucker.

**Why joint decomposition per variable-group and not per-channel.** Grouping preserves the physical coupling among Hs, θ, T within a single partition. Decomposing each channel independently throws this away and forces the downstream copula to reinvent it as generic statistical dependence. With joint decomposition, wave steepness (`E − 2T + const`) becomes a linear direction that Tucker can discover as a rank-1 pattern in the channel factor when it's present.

---

## Pre-transform rules

For each variable, the forward and inverse transforms must be:

**Forward (raw → tensor channels):**
```
u = M · cos(θ)              # meters (waves) or m/s (wind/currents)
v = M · sin(θ)              # same units as u
E = 2 · log(M)              # dimensionless, log-energy
T = log(T_period)           # dimensionless, log-seconds
```

**Inverse (tensor channels → raw):**
```
M         = exp(E / 2)                 # NOT sqrt(u² + v²) — see note below
θ         = atan2(v, u)                # radians; convert to degrees for output
T_period  = exp(T)
```

**Important — magnitude reconstruction.** Do not reconstruct `M` from `sqrt(u² + v²)` after Tucker. The u/v channels have been reduced and reconstructed under an L² loss in u/v space, and `E` has been reduced independently under an L² loss in log-space. The two reconstructions are not consistent, and `sqrt(u² + v²)` will systematically underestimate `M` relative to `exp(E/2)` (Jensen's inequality on the concave sqrt applied to noisy reconstructions). Use `E` as the source of truth for magnitude; use `atan2(v, u)` only for direction. This is why the redundant-looking channel design (u, v, E together) is actually load-bearing: `A_C` gets to combine the two magnitude representations optimally under Tucker's L² objective.

**Direction convention.** Wave direction conventions differ across datasets (going-to vs. coming-from). The existing `ComingFromToUV` and `GoingToToUV` transformer atoms are the correct place to handle this per-variable. Do not reimplement the convention conversion inline in the tensor assembly — call the transformers. Each partitioned wave variable (`pw*`, `pw*`, `sw*`, `ww*`) needs its convention confirmed against the ETL source; do not assume they all match. `was`/`wad` from ERA5 is coming-from; wave directions from GFS-Wave are typically going-to. Verify per source before decomposing.

**Positivity floors:**
- `Hs ≥ Hs_min` (default 0.01 m) before `log(Hs)` to avoid `-inf`.
- `T_period ≥ T_min` (default 1.0 s) before `log(T_period)`.
- Wind speed `was ≥ was_min` (default 0.01 m/s) before `log(was)`.
- Current speed `ocs ≥ ocs_min` (default 0.001 m/s) before `log(ocs)` if the `E` channel is included.
- These floors apply symmetrically on inverse: clip reconstructed magnitudes to the same floor before writing physical output.

---

## `sigwh` (total significant wave height)

`sigwh` satisfies the energy budget `sigwh² ≈ pwh² + pswh² + wsh²` up to model discretization. Two options:

- **Option A:** Model `sigwh` independently as a Class 0 scalar; keep the energy-budget relation as a post-hoc physical filter in `physics.py` (`sample_batch_with_physics`).
- **Option B:** Drop `sigwh` from inputs entirely; reconstruct at sampling time as `sigwh = sqrt(pwh² + pswh² + wsh²)`.

**Recommendation:** Option B. One fewer decomposition to fit, and it removes an internal consistency constraint the copula would otherwise have to enforce via rejection sampling. The reconstruction is deterministic and exact under the energy-partition assumption already baked into WW3.

Confirm with Miguel before removing `sigwh` from the input variable list, since this changes the ETL contract.

---

## Concrete tensor assembly example (Class 4)

```python
# per-partition ETL output, each of shape (lat_size, lon_size, t)
Hs        = raw_Hs                              # significant height per partition
theta_rad = convention_transformer(raw_dir)     # ComingFromToUV / GoingToToUV → radians
T_p       = raw_period

# floors
Hs   = np.maximum(Hs,   Hs_min)
T_p  = np.maximum(T_p,  T_min)

# channels
u = Hs * np.cos(theta_rad)
v = Hs * np.sin(theta_rad)
E = 2.0 * np.log(Hs)
T = np.log(T_p)

# stack into (lat, lon, t, 4)
tensor = np.stack([u, v, E, T], axis=-1)

# Tucker
core, factors = tucker_decomposition(
    tensor,
    ranks=(R_lat, R_lon, R_t, R_C),
)
# factors: [A_lat, A_lon, A_t, A_C]
```

For Class 3 with `(u, v, E)`: same pattern with 3 channels. For Class 3 with only `(u, v)`: same pattern with 2 channels. For Class 0: no stacking, tensor is directly `(lat, lon, t)`.

---

## Suggested code organization

Introduce a small type hierarchy for variables. The three Tucker shapes correspond to three concrete subclasses of a `Variable` base:

- `ScalarVariable` (Class 0): `sst`, optionally `sigwh` if Option A.
- `MagDirVariable` (Class 3): `wind = (was, wad)`, `current = (ocs, ocd)`.
- `MagDirPeriodVariable` (Class 4): `peak_wave = (pwh, pwp, pwd)`, `primary_swell = (pswh, pswp, pswd)`, `wind_sea = (wsh, wsp, wsd)`.

Each subclass knows:

1. **Forward transform:** how to assemble the channel tensor from raw ETL output (including convention conversion via existing transformer atoms and positivity floors).
2. **Inverse transform:** how to invert reconstructed channels back to physical raw fields, with the `M = exp(E/2)` rule and clipping to floors.
3. **Tucker shape/ranks:** which mode count and default rank tuple to use.
4. **Channel metadata:** channel names and units, for logging and validation.

Downstream stages (DMD on `A_t`, OU fit on reduced coefficients, gaussianization, copula) operate polymorphically on the coefficient time series and do not need to know the Class of the variable. Class is a data-assembly concern that becomes invisible after Tucker.

---

## Non-negotiable rules

- **No `cos(θ + T·k·t)` phase terms anywhere.** Period is a spectral moment ratio (or peak-frequency reciprocal), not an oscillation frequency of a monochromatic wave. Any code that treats `wap`/`wsp`/`pwp`/`pswp` as a phase clock is wrong.
- **No `wap` as a wind variable.** Wind has `was` (speed) and `wad` (direction) only. There is no wind period. If any code path or config references a wind period, flag it — it's either a naming collision or an artifact of an earlier misunderstanding.
- **No global gaussianization before Tucker.** Only the physical monotone pre-transforms above (log-magnitude, log-period, u/v). Full gaussianization happens on reduced coefficient series, downstream.
- **No `T / (4·√Hs)` or similar hardcoded steepness normalizations.** Wave steepness must be discoverable by Tucker's `A_C` factor if it's really present in the data; hardcoding it pre-empts the decomposition and only holds for fully-developed wind seas (breaks entirely for swell).
- **No `sqrt(u² + v²)` for magnitude reconstruction.** Use `exp(E/2)`.

---

## Version context

The last shared design was v3, which added `physics.py` with `sample_batch_with_physics` for post-hoc physical-consistency filtering. The changes described here are on top of v3 and touch:

- `src/weather/` variable-handling module (new type hierarchy).
- Tucker call sites (per-variable-group, 3D or 4D depending on Class).
- Any code that assumed a single monolithic Tucker across all variables (should not exist post-refactor).
- Any pre-Tucker gaussianization step (must be removed).
- Round-trip / reconstruction code (must switch to `exp(E/2)` for magnitudes).

The training/eval split is unchanged: 2022–2025 training (analysis), 2026-01-01 through 2026-06-30 held-out (analysis), 2026-07-01 through 2026-07-19 diagnostic-only (forecast product).

---

## Requests to the agent

1. **Read the current state of `src/weather/` first.** Report back what variable-handling code exists, what shape the current Tucker inputs are, and whether any pre-Tucker gaussianization is still present. Do not edit until we've reconciled the current state with this spec.
2. **Flag discrepancies.** If anything in this document conflicts with what Miguel has already implemented locally, ask before resolving. He has made unshared changes.
3. **Do not silently reconcile `sigwh` handling.** Ask which of Option A or Option B has been chosen before removing or keeping it.
4. **Confirm direction conventions per source.** Do not assume all wave partitions use the same convention. Check ETL config or ask.
5. **Preserve the existing transformer atoms** (`ComingFromToUV`, `GoingToToUV`, etc.). Do not reimplement.
