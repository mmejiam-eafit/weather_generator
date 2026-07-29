# Variable Taxonomy

**Status:** foundational document. Every other document in this set (Tucker
decomposition, DMD, Ornstein–Uhlenbeck residuals, Gaussian-copula
transformation, harmonic climatology, physical-consistency constraints)
assumes the field representation defined here.

**Purpose:** define, from first principles, how each physical quantity in
the archive should be represented before any statistical decomposition
touches it. This is a specification of the *correct* representation, not
a description of the current `src/` implementation.

---

## 0. Two errors this document corrects

This document replaces a deleted draft that made two specific errors.
Both are corrected here explicitly, so neither is repeated by a future
author of this document set.

### Error 1 — this document must be prescriptive, not descriptive

The deleted draft documented what the current `src/` code does and framed
that as the definition of correctness. That is backwards. In the author's
own words:

> I DO NOT need them to reflect the current code. It's the opposite. I
> need these documents to define what the process IS in order to CORRECT
> the actual code files to what they should be.

Accordingly, this document defines the correct theory — circular
statistics, the directional wave spectrum, and the resulting variable
classes — and states how it *should* be applied to this archive. It does
not contain a "how the pipeline currently handles this" section that
treats existing code behavior as ground truth. Where a specific,
concrete place in the current code is known to depart from the
prescription below, it is flagged in a sentence or two as a pointer for
whoever does the refactor — never as this document's framing, and never
as its main content.

### Error 2 — never merge variables into one 4D tensor, and never flatten a variable's tensor to 2D

The deleted draft treated (or left room to treat) the archive as one
joint `(lat, lon, time, variable)` object, and left room for
mode-collapsing reshapes during decomposition. Both are wrong. In the
author's own words:

> Don't treat the entirety of the variables as a single 4D tensor of
> size (lat, lon, time, num_variables); instead, treat each of them as a
> single 3D tensor (lat, lon, time)... They should NOT be converted to a
> 2D matrix of (lat * lon, time) nor of (lat, lon * time) or any other
> combination. Flattening has the disadvantage of losing local
> information, so NEVER flatten.

Concretely, and this applies throughout the rest of this document and
every document built on it:

- Every physical field this document defines (a pure scalar, or a
  Cartesian component derived from a magnitude/direction pair) is its
  own **independent** 3D tensor of shape `(lat, lon, time)`. Fields are
  never stacked into a joint 4D `(lat, lon, time, variable)` tensor —
  not even two fields that happen to live on the same grid and share the
  same `(lat, lon)` shape (e.g. two wave-grid fields). Shape compatibility
  is not license to merge.
- Land/ice cells are handled via a boolean `(lat, lon)` mask, built and
  applied exactly as in `notebooks/01_understand_tucker_decomposition.ipynb`
  (`ocean_mask = ~np.isnan(sst_full).any(axis=-1)` over a
  `(lat, lon, time)`-stacked field), combined with mask-aware iterative
  imputation (that notebook's `partial_tucker(..., mask=tucker_mask)`
  pattern). Masked cells are never dropped/raveled out of the tensor into
  some other axis.
- The `(lat, lon, time)` tensor is never reshaped into a 2D matrix for any
  decomposition purpose — not `(lat*lon, time)`, not `(lat, lon*time)`,
  not any other collapsed pairing. Keeping the three modes separate is
  what lets a mode-wise method (Tucker, in particular) find an
  independent low-rank subspace along each one; collapsing two modes into
  one forecloses that and destroys that mode's own local/neighbor
  structure.
- **Narrow, explicitly permitted exception:** a reshape used purely as a
  computational-vectorization trick for an operation that is
  mathematically equivalent to an independent per-`(lat, lon)`-cell
  computation (e.g. a broadcasted elementwise numpy op over the full
  grid, or broadcasting a `(lat, lon)` mask across the `time` axis via
  `mask[:, :, None]`) is *not* what this rule forbids — no modeling
  structure is lost there, because no two modes' cross-structure is ever
  treated as one merged axis. What is forbidden is any point where a
  decomposition's own cross-mode structure treats two or more of
  `{lat, lon, time}`, or two different variables, as a single index.

Section 4 restates this rule specifically for the fields this document
produces; the Results section (5) states what that means for every
downstream stage.

---

## 1. Summary

The archive stores raw variables that fall into exactly three physically
distinct classes:

| Class | Members | DOF per cell | Description |
|---|---|---|---|
| **(A) Pure scalar** | `sigwh`, `sst` | 1 | Magnitude only — no direction, no period |
| **(B) Vector** | `was`+`wad_come_from` (wind); `ocs`+`ocd` (current) | 2 | Magnitude + direction, no period |
| **(C) Triplet** | `wsh`/`wsd`/`wsp` (wind-sea); `pwh`/`pwd`/`pwp` (peak wave); `pswh`/`pswd`/`pswp` (primary swell) | 3 | Magnitude(height) + direction + period, all three derived from one directional spectrum |

The classification is not cosmetic. It determines which quantities may be
treated as ordinary real numbers (heights, periods, sst) and which are
angles that require a circular-statistics treatment (all `*d` fields)
before they may be averaged, interpolated, or fed into any linear
decomposition. Section 2 derives why, Section 3 gives the per-class
recipe, and Sections 4–5 state how the results are represented and
consumed downstream — as independent 3D tensors, never merged or
flattened (Error 2).

---

## 2. Theoretical foundation

### 2.1 Circular statistics: why angles are not a vector space

A compass bearing $\theta$ is a point on the circle group
$$
\mathbb{T} = \mathbb{R}\big/360°\mathbb{Z},
$$
the real line modulo $360°$. $\mathbb{T}$ is a compact abelian group under
addition — but it is **not** a vector space over $\mathbb{R}$, because
scalar multiplication does not descend to the quotient. A vector space
requires that $c\cdot\theta$ be well-defined for every real $c$ and every
$\theta$; on $\mathbb{T}$ it is not, for non-integer $c$.

**Concrete failure — averaging.** Let $\theta = 350°$. As a real-number
representative this could be written $350°$ or, equally validly (same
point on the circle), $-10°$. Halving each representative ($c=\tfrac12$,
the operation an arithmetic mean performs):
$$
\tfrac12(350°) = 175° \qquad\text{vs.}\qquad \tfrac12(-10°) = -5° \equiv 355°.
$$
These differ by $180°$ — diametrically opposite points on the circle —
purely as an artifact of which real-number representative was chosen
before the arithmetic was done. Since an arithmetic mean *is* scalar
multiplication by $1/2$ combined with addition, "the average of two
compass bearings" is ill-defined without extra structure. This is exactly
why naively averaging $350°$ and $10°$ gives $180°$ (arithmetic mean)
instead of the physically correct $0°$/$360°$ — a $180°$ error, the same
magnitude of error this archive's `wad` decoy field happens to exhibit
(§3.2), for an unrelated reason. A field of angles therefore cannot be
treated as a vector space element (no valid linear combinations), has no
non-arbitrary origin (the location of $0°$ is a labeling convention, not
a physical distinguished point) and no rotation-compatible total order.

**Correct treatment — embed the circle in $\mathbb{R}^2$.** The circle
embeds isometrically into the Euclidean plane via the unit-circle map;
composed with a magnitude $r \ge 0$ this gives, in compass convention
(clockwise from geographic North, $\theta$ increasing toward East):
$$
u = r\sin\theta \quad(\text{eastward component}), \qquad
v = r\cos\theta \quad(\text{northward component}).
$$
$(u, v) \in \mathbb{R}^2$ *is* a genuine vector space element: averaging,
linear operators, spatial interpolation, and low-rank decomposition are
all well-defined on it, with no representative-choice ambiguity, because
$(\sin\theta,\cos\theta)$ is a single-valued, $360°$-periodic function of
$\theta$ — not a choice of representative in $\mathbb{R}$.

**The transform's Jacobian.** Write $F(r,\theta) = (u,v) =
(r\sin\theta,\, r\cos\theta)$. Then
$$
J = \frac{\partial(u,v)}{\partial(r,\theta)} =
\begin{pmatrix} \sin\theta & r\cos\theta \\ \cos\theta & -r\sin\theta \end{pmatrix},
\qquad
\det J = -r\sin^2\theta - r\cos^2\theta = -r.
$$
So $|\det J| = r$: the area element transforms as $du\,dv = r\,dr\,d\theta$,
identical in magnitude to the ordinary polar-coordinate Jacobian (the
compass convention is a reflection/rotation of the standard
counterclockwise-from-East convention, which flips the sign of $\det J$
but not its magnitude). This is what will let a later document
change variables between a $(r,\theta)$ density and a $(u,v)$ density
without introducing a spurious Jacobian factor beyond $r$.

**Why `hypot(u, v)` recovers $r$ exactly.** By the Pythagorean identity,
$$
u^2 + v^2 = r^2\sin^2\theta + r^2\cos^2\theta = r^2(\sin^2\theta+\cos^2\theta) = r^2,
$$
so $\sqrt{u^2+v^2} = r$ for every $\theta$ and every $r \ge 0$ — exact,
with no small-angle approximation and no dependence on which quadrant
$\theta$ falls in. The angle itself is recovered (mod $360°$) via
$\theta = \operatorname{atan2}(u, v)$ — note the argument order is
$(u,v) = (\sin\text{-component}, \cos\text{-component})$, not the
$(y,x)$ order `atan2` normally implies, precisely because $u$ was built
from $\sin\theta$ and $v$ from $\cos\theta$ in the compass convention above.

**Going-to vs. coming-from.** A physical velocity vector (wind, current)
points in the direction the fluid is moving *toward*. The embedding above
is the natural one when $\theta$ is already a "going-to" bearing. If a
field is archived in "coming-from" convention instead, it must be rotated
by $180°$ **before** applying the embedding,
$\theta_{\text{to}} = (\theta_{\text{from}} + 180°) \bmod 360°$, or the
resulting $(u,v)$ point exactly antiparallel to the true physical vector.
Getting this rotation backwards, or omitting it, reproduces precisely the
$180°$ failure mode derived above — it is the same error, just introduced
by a convention mismatch rather than by naive angle-averaging.

### 2.2 The directional wave spectrum

A sea state's surface elevation is modeled (to first, linear order) as a
superposition of many independent sinusoidal wave trains with random
phases, spread over frequency $f$ and direction $\theta$ (the direction
*from* which each component travels, by oceanographic convention). Its
statistics are fully captured by the **directional wave spectrum**
$S(f,\theta)$, an energy density such that $S(f,\theta)\,df\,d\theta$ is
the variance of surface elevation carried by components in
$[f, f+df]\times[\theta,\theta+d\theta]$.

**Spectral moments.** Define
$$
m_n = \int_0^\infty\!\!\int_0^{360°} f^n\, S(f,\theta)\, d\theta\, df .
$$
By Wiener–Khinchin, the zeroth moment $m_0$ is exactly the variance of
surface elevation, $m_0 = \operatorname{Var}(\eta)$.

**Significant wave height, $H_s = 4\sqrt{m_0}$.** For a narrow-banded
Gaussian process, individual wave amplitudes are Rayleigh-distributed
with scale parameter tied to $m_0$, and the classical definition of
significant wave height (the mean of the highest one-third of wave
heights, $H_{1/3}$) can be computed as an order statistic of that Rayleigh
law; the well-known result (Longuet-Higgins, 1952) is
$H_{1/3} \approx 4.00\sqrt{m_0}$. Operational wave models (WAVEWATCH III,
and this archive) sidestep the order-statistic derivation and instead
*define* the spectral significant height by convention as
$$
H_{m0} \;:=\; 4\sqrt{m_0},
$$
which is what `sigwh`, `wsh`, `pwh`, and `pswh` report. This is why a
partition's height is fundamentally a functional of the *whole* spectrum
restricted to that partition's support — an $L^2$-type norm, not a
"typical wave" measurement.

**Peak period, $T_p$.** Let $E(f) = \int_0^{360°} S(f,\theta)\,d\theta$ be
the (direction-integrated) 1D frequency spectrum. The peak frequency is
$f_p = \arg\max_f E(f)$, and $T_p = 1/f_p$. (Other period statistics exist
— e.g. mean period $T_{m01} = m_0/m_1$ — but this archive's `*p` fields
are peak periods.)

**Mean/peak direction — and its tie back to §2.1.** The mean direction of
a spectrum (or a partition of one) is *itself* a circular-mean problem,
and is solved with exactly the embedding derived in §2.1. Define the
first directional Fourier moments
$$
a_1 = \frac{1}{m_0}\int\!\!\int \cos\theta\, S(f,\theta)\, d\theta\, df,
\qquad
b_1 = \frac{1}{m_0}\int\!\!\int \sin\theta\, S(f,\theta)\, d\theta\, df .
$$
This is precisely a resultant-vector computation: $(b_1, a_1)$ is the
energy-weighted average of $(\sin\theta,\cos\theta)$ over the spectrum,
exactly the $(u,v)$ embedding of §2.1 but with $r=S(f,\theta)$ as the
weighting instead of a per-cell magnitude. The mean direction is then
$$
\theta_1 = \operatorname{atan2}(b_1, a_1),
$$
and the resultant length $r_1=\sqrt{a_1^2+b_1^2}\in[0,1]$ measures
directional concentration (this is the circular-statistics mean
resultant length; circular variance is $1-r_1$). The directional spread
follows the standard formula (Kuik, van Vledder & Holthuijsen, 1988):
$$
\sigma_\theta = \sqrt{2(1-r_1)} \quad\text{(radians)}.
$$
The archive's `*d` fields (per-partition or bulk peak direction) are a
discretized read-out of this same quantity — never a free, independently
drawn number.

**Consequence for `sigwh`.** Because the spectral partitions
(wind-sea, peak/dominant, primary swell) occupy — by construction of the
partitioning algorithm — disjoint frequency–direction regions of the same
spectrum, variance adds across partitions: $m_{0,\text{total}} =
\sum_p m_{0,p}$. Since $H_{s,p} = 4\sqrt{m_{0,p}} \Rightarrow
H_{s,p}^2 = 16\, m_{0,p}$,
$$
H_{s,\text{total}} = 4\sqrt{m_{0,\text{total}}}
= 4\sqrt{\textstyle\sum_p m_{0,p}}
= \sqrt{\textstyle\sum_p H_{s,p}^2},
$$
i.e. `sigwh` is the RMS (quadrature) combination of the partition heights
— exact if the partitioning is truly disjoint, approximate to the extent
it is not. `sigwh` has no single direction or period of its own precisely
because it is a scalar functional over the *entire* spectrum, integrating
away the angular and frequency information that a single partition
retains.

---

## 3. Guidelines: the A/B/C breakdown

### 3.0 Grid facts (dataset ground truth)

| Grid | Shape (lat × lon) | Resolution | Row 0 latitude | Orientation confirmed? |
|---|---|---|---|---|
| Wave (WAVEWATCH III) | 721 × 1440 | 0.25° | −90.0° (south pole) | Yes |
| Wind (GFS) | 721 × 1440 | 0.25° | −90.0° (south pole) | Yes |
| Ocean | 1121 × 2880 | 0.125° | *unset* | **No** — see §6 |

(Constants: `src/core/config.py`; these are era-3 values, the current and
open-ended archive era — see that module's docstring for the era-1/era-2
caveat, which is a separate, orthogonal concern from this document's
topic.) Directions are archived as `int8` 16-point compass codes
(22.5° steps) and decoded via `core.io.decode_compass_code`, keyed to
which raw code represents North ("origin code"): fields in the
`wsd`/`pwd`/`pswd` family use origin code 0; `wad`/`ocd` use origin code 1.
"Coming-from" and "going-to" differ by exactly 180°, i.e. 8 compass-code
steps. Periods are strictly positive, physically bounded scalars —
never angles, never circular.

### 3.1 Class (A) — pure scalar

| Field | Meaning | Direction? | Period? |
|---|---|---|---|
| `sigwh` | Composite/total significant wave height | No — see §2.2 | No |
| `sst` | Sea surface temperature | No (not an angular quantity at all) | No |

Both are ordinary elements of $\mathbb{R}$ (per cell, per time). No
circular-statistics treatment applies; they pass through as a single
scalar tensor apiece.

### 3.2 Class (B) — vector (magnitude + direction, no period)

These are instantaneous velocity fields, not oscillatory wave partitions
— "period" has no meaning for a mean flow.

| Pair | Grid | Convention | Status |
|---|---|---|---|
| `was` (speed) + `wad_come_from` (direction) | wind | coming_from | **Confirmed correct** — the only valid wind-direction field |
| `was` + `wad` | wind | *looks* coming_from | **Decoy** — confirmed going_to, i.e. exactly 180° off; do not use |
| `ocs` (speed) + `ocd` (direction) | ocean | going_to | Confirmed |

Embedding (§2.1), with the 180° correction applied only where the native
convention is coming-from:
$$
\theta_{\text{to}} =
\begin{cases}
(\theta_{\text{wad\_come\_from}} + 180°) \bmod 360° & \text{wind}\\[2pt]
\theta_{\text{ocd}} & \text{current (already going\_to)}
\end{cases}
$$
$$
u = \text{speed}\cdot\sin\theta_{\text{to}}, \qquad
v = \text{speed}\cdot\cos\theta_{\text{to}}.
$$
This yields `u_wind, v_wind` and `u_current, v_current` — literal
eastward/northward Cartesian velocity components, exactly the quantities
a linear decomposition or spatial interpolation can safely act on. Speed
is recoverable exactly at any point via `hypot(u, v)` (§2.1); it need not
be separately retained.

### 3.3 Class (C) — full triplet (height + direction + period)

One triplet per WAVEWATCH III partition of the same underlying $S(f,\theta)$:

| Triplet | Height | Direction | Period | Direction convention |
|---|---|---|---|---|
| Wind-sea | `wsh` | `wsd` | `wsp` | **Confirmed** coming_from |
| Peak/dominant wave | `pwh` | `pwd` | `pwp` | **Assumed** coming_from — not empirically proven |
| Primary swell | `pswh` | `pswd` | `pswp` | **Assumed** coming_from — not empirically proven |

The `pwd`/`pswd` convention is inferred only by analogy (same origin-code
family, `0`, as the confirmed `wsd`) — it has not been put through the
same empirical trade-wind / westerlies check that settled `wad` vs.
`wad_come_from` and `wsd` (see `notebooks/verify_direction_convention.ipynb`,
which tests `was`/`wad`, `wsh`/`wsd`, and `ocd`, but not `pwd`/`pswd`
directly). Treat this as an open item (§6), not a settled fact.

As derived in §2.2, height/period/direction of one partition are three
different functionals of the *same* $S(f,\theta)$ restricted to that
partition's support — not three independent draws. The representation
should reflect that coupling as far as the tensor design allows:

- **Period** ($T_p$) is an ordinary positive scalar — never circular —
  and stays its own scalar tensor.
- **Height and direction are embedded jointly**, exactly as in §3.2's
  vector case but using the partition's height as the magnitude:
$$
\theta_{\text{to}} = (\theta_{\text{from}} + 180°)\bmod 360°,
\qquad
u_{\text{dir}} = H\sin\theta_{\text{to}}, \qquad
v_{\text{dir}} = H\cos\theta_{\text{to}}.
$$
  This mirrors the energy-weighted $a_1$/$b_1$ construction of §2.2 (a
  height-scaled direction vector shrinks toward the origin exactly when
  a partition carries negligible energy on a given day, which is also
  when its direction estimate is least meaningful — a desirable property,
  not an accident). $H$ is recovered exactly via `hypot(u_dir, v_dir)`
  and need not be separately retained, by the same argument as §3.2.

  *Caveat for later documents:* if $u_{\text{dir}}$ and $v_{\text{dir}}$
  are independently low-rank-approximated (as Error 2 requires — each is
  its own tensor), `hypot` of the two *reconstructions* is only an
  approximation to a directly-decomposed $H$, since low-rank
  approximation does not commute with a nonlinear map like `hypot`. Note
  this as an open question for the Tucker document (§6) rather than
  resolving it here.

Per triplet, this produces **3 tensors**: $T_p$, $u_{\text{dir}}$,
$v_{\text{dir}}$ — replacing the 3 raw archived fields (height, direction,
period) one-for-one in count, but not in kind.

---

## 4. Correct treatment for this dataset's tensors

Every field defined above — each class-(A) scalar, each class-(B)
Cartesian pair, each class-(C) period and Cartesian-direction pair — is
represented as its **own independent 3D tensor** of shape
`(lat, lon, time)`. This is the direct application of Error 2 (§0) to the
variable set defined in §3:

- No two fields are ever stacked into a joint 4D `(lat, lon, time, variable)`
  tensor — including fields that happen to share a grid and shape (e.g.
  `u_wind` and `u_wsea` are both `(721, 1440, T)` but remain two separate
  tensors, never merged along a variable axis).
- Fields on different grids (wave/wind at `(721,1440,T)`; ocean at
  `(1121,2880,T)`) cannot be merged even shape-wise — this is a second,
  independent reason (beyond Error 2 itself) that a single joint tensor is
  not just undesirable but not even well-formed for this dataset.
- Land/ice cells are carried via a boolean `(lat, lon)` mask per grid,
  built the way `notebooks/01_understand_tucker_decomposition.ipynb`
  builds `ocean_mask`, and handled by mask-aware imputation inside
  whatever decomposition consumes the tensor (that notebook's
  `partial_tucker(mask=...)` pattern) — never by raveling `(lat, lon)`
  into a single masked/dropped axis.
- No field's `(lat, lon, time)` tensor is ever reshaped to `(lat*lon, time)`,
  `(lat, lon*time)`, or any other 2D collapse for decomposition purposes.
  Elementwise, per-cell-independent vectorized operations (e.g. the
  broadcasted $u=r\sin\theta$ computation itself, or broadcasting the
  `(lat,lon)` mask across `time`) are the one permitted exception, since
  they merge no cross-mode structure.

**Pointer for the refactor (not this document's framing):** the notebook
cited above explicitly contrasts its own approach with the production
pipeline, which instead ravels `(lat, lon)` into a single `ocean_cell`
Tucker mode and drops land cells from the tensor entirely
(`src/weather/model.py`'s `_mask_and_stack`, per that notebook's inline
comment) — this is the concrete place that needs to change to match §4.

---

## 5. Results: what downstream stages receive

Under this design, every downstream stage — harmonic climatology fitting,
Gaussian-copula transformation, Tucker decomposition, DMD — receives the
same kind of object repeatedly: **one independent `(lat, lon, time)`
tensor per field**, never a joint 4D stack, never flattened. The full
inventory, applying §3's rules to the archive's raw variables:

| Source class | Raw fields | Tensors produced | Count |
|---|---|---|---|
| (A) | `sigwh` | `sigwh` | 1 |
| (A) | `sst` | `sst` | 1 |
| (B) | `was`, `wad_come_from` | `u_wind`, `v_wind` | 2 |
| (B) | `ocs`, `ocd` | `u_current`, `v_current` | 2 |
| (C) | `wsh`, `wsd`, `wsp` | `wsp`, `u_wsea`, `v_wsea` | 3 |
| (C) | `pwh`, `pwd`, `pwp` | `pwp`, `u_pwave`, `v_pwave` | 3 |
| (C) | `pswh`, `pswd`, `pswp` | `pswp`, `u_pswell`, `v_pswell` | 3 |
| **Total** | 15 meaningful raw fields (`wad` decoy excluded) | | **15 independent tensors** |

Every one of the 15 tensors has shape `(lat, lon, time)` on its native
grid (wave/wind: `(721,1440,T)`; ocean: `(1121,2880,T)`), shares its
grid's boolean `(lat,lon)` ocean/ice mask, and is otherwise self-contained.
Any cross-field dependence a later stage needs to model (e.g. the
correlation between wind speed and wind-sea height, or the height/period
coupling within one partition flagged in §3.3 and §6) is the job of that
stage's own dependency structure (e.g. a copula over per-cell marginals) —
never a pre-decomposition merge of the tensors themselves. This is the
concrete meaning of "orthogonal to the tensor-stacking question" for this
document: the taxonomy determines *which* and *how many* tensors exist;
Error 2 governs how they are (never) combined; the two compose cleanly
because every one of these 15 objects is already, by construction, a
single `(lat, lon, time)` tensor and nothing more.

---

## 6. Open questions / limitations

- **Ocean-grid orientation is unverified.** `core/config.py` deliberately
  leaves the ocean grid's row-0 latitude unset; whether row 0 is south (as
  confirmed for wave/wind) has not been established by the coastline
  correlation study that would settle it (§2.7-style verification).
  `notebooks/verify_direction_convention.ipynb` uses a cheaper proxy
  (equator-row via zonal-mean `sst` maximum, hemisphere via open-ocean
  fraction at 40–60° from that row) sufficient only to place a few
  direction-diagnostic latitude bands — not a substitute for the full
  verification. Do not assume the ocean grid matches the wave/wind
  row-0=south convention by analogy.
- **`pwd`/`pswd`'s coming_from convention is assumed, not proven.** Only
  `wad`/`wad_come_from` (wind) and `wsd` (wind-sea) have been empirically
  checked against known persistent wind/current patterns (trade winds,
  Southern Ocean westerlies, major equatorial currents). `pwd` and `pswd`
  share `wsd`'s origin code by analogy only; treat as unproven until
  checked the same way.
- **Per-cell independent Gaussianization of a (C)-class triplet risks
  discarding real physical coupling.** §2.2/§3.3 establish that a
  partition's height, period, and direction are correlated functionals of
  one spectrum (e.g. wave-growth/fetch relations link height and period;
  JONSWAP-type peakedness links height and spectral shape). If the
  Gaussian-copula stage marginally Gaussianizes $T_p$, $u_{\text{dir}}$,
  and $v_{\text{dir}}$ independently without an explicit joint dependency
  structure across that triplet, it silently discards this coupling. This
  is a concern for the Gaussian-copula document to resolve, not this one
  — flagged here because it originates in the taxonomy, not the copula
  machinery.
- **Low-rank approximation does not commute with `hypot`.** As noted in
  §3.3, if a partition's height-scaled direction is decomposed as two
  independent tensors $u_{\text{dir}}, v_{\text{dir}}$ (per Error 2, each
  gets its own factors), `hypot` of their reconstructions only
  approximates a directly-decomposed $H$. Whether the Tucker document
  needs $H$ retained as an additional, separately-decomposed tensor for
  reconstruction-accuracy reasons is an open design question left to that
  document.
- **`sigwh`'s RMS relationship to partition heights assumes disjoint
  partitions.** §2.2's derivation is exact only if the wind-sea/peak/swell
  partitioning algorithm produces spectrally non-overlapping bins; to the
  extent it does not, `sigwh = sqrt(wsh² + pwh² + pswh²)`-style
  reconstructions are approximate.
- **`wad` is confirmed to be a decoy.** It decodes as a plausible
  coming_from field but is actually going_to, exactly 180° from
  `wad_come_from` in every cell at the checked snapshot. It should never
  be wired into any pipeline; `wad_come_from` is the only correct
  wind-direction source.
