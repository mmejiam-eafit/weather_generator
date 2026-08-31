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
| **(C) Triplet** | `wsh`/`wsd`/`wsp` (wind-sea); `pwh`\*/`pwd`/`pwp` (peak wave); `pswh`/`pswd`/`pswp` (primary swell) | 3 (2 for peak wave — see \*) | Magnitude(height) + direction + period, all three derived from one directional spectrum |

\* **`pwh` is not an independent measurement — see §3.3.** It is a literal
byte-for-byte duplicate of `sigwh` at the ETL level, confirmed both in
`noaa-etl`'s own transform table and in NOAA's official GRIB2 parameter
definitions (there is no "primary wave height" field to begin with — only
`DIRPW`/`PERPW`). The "peak wave" row genuinely contributes 2 independent
degrees of freedom (direction, period), not 3.

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
which is what `sigwh`, `wsh`, and `pswh` report — each computed from its
own partition's (or the combined spectrum's) $m_0$. `pwh` is the one
exception: it reports `sigwh`'s value verbatim rather than an independent
$H_{m0}$ computed over a "peak wave" partition's own support, because no
such NOAA-defined field exists to compute it from (§3.3 explains why).
This is why a partition's height is fundamentally a functional of the
*whole* spectrum restricted to that partition's support — an $L^2$-type
norm, not a "typical wave" measurement — for every field in this list
except `pwh`.

**Correction: mean period, $T_{m01}$ — not peak period.** An earlier
version of this section claimed this archive's `*p` fields are peak
periods, $T_p = 1/f_p$ where $f_p = \arg\max_f E(f)$ is the single
frequency at which the direction-integrated spectrum $E(f) =
\int_0^{360°}S(f,\theta)\,d\theta$ is largest. **That is wrong**, checked
directly against source and NOAA documentation rather than assumed from
field naming. `noaa-etl`'s `gfs_transform.py` derives this archive's three
period fields from GRIB `shortName`s `mpww` → `wsp`, `perpw` → `pwp`,
`mpts` → `pswp`. Cross-checked against NCEP's own
[GRIB2–GRIB1 parameter conversion table](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/GRIB2_parmeter_conversion_table.html),
these are explicitly named **"Mean period of wind waves"** (GRIB2
10-0-6), **"Primary wave mean period"** (10-0-11), and **"Mean period of
total swell"** respectively — distinct, separately-numbered parameters
from NOAA's own "Peak Wave Period"/"Peak Period of Wind Waves"/"Peak
Period of Total Swell" family, which this archive's ETL never reads at
all. The physical quantity these mean-period fields report, confirmed
against [WAVEWATCH III's own user manual](https://polar.ncep.noaa.gov/waves/wavewatch/manual.v5.16.pdf)
and standard oceanographic convention, is

$$T_{m01} = \frac{m_0}{m_1},$$

the ratio of the zeroth and first spectral moments — an energy-weighted
*mean* frequency across the whole partition's spectrum, inverted. This is
a materially different statistic from $T_p = 1/f_p$: $T_{m01}$ is pulled
toward wherever the bulk of spectral energy sits, while $T_p$ tracks only
the single largest bin. The two coincide only in the idealized limit of an
infinitely narrow spectral peak; for a broad or multi-modal spectrum
(routine for a combined or swell-dominated sea state) they can diverge
meaningfully. Every downstream use of "period" in this document series
should be read as $T_{m01}$, not $T_p$ — this has concrete consequences
for `physical_consistency_constraints.md` §2.3 and `correlated_variables.md`
§3.3, both corrected alongside this document.

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

**Open question, not resolved here: should a `log`-magnitude tensor be
added alongside `u, v` for this class too, mirroring §3.3's resolution for
Class C heights?** §3.3 below adds an explicit `log`-transformed magnitude
tensor for wave heights, both to tame Tucker/DMD's sensitivity to a wide,
right-skewed dynamic range and to serve as an authoritative,
`hypot`-reconstruction-bias-free magnitude source. Wind speed (`was`) is
plausibly heavy-tailed enough for the same argument to apply; current
speed (`ocs`) plausibly is not — it has a much narrower physical dynamic
range, and whether the same treatment earns its keep here is genuinely
unclear rather than assumed. Left open (§6) pending the same kind of
archive-specific check §3.3's resolution was based on.

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

**Correction: `pwh` is not a genuine height measurement of a "peak wave"
partition — it is `sigwh` restated.** This is not an approximation or a
naming quirk; it is a hard identity at the ETL level. Two independent
lines of evidence confirm it:

1. **Source-code level.** `noaa-etl`'s `gfs_transform.py` contains two
   separate `VARIABLE_TRANSFORMATION` entries that both read GRIB
   `shortName: "swh"` and apply the *identical* transform chain
   (`[flip_latitude, round_2]`) — one producing `sigwh`, the other `pwh`.
   They are the same array on disk, not two measurements that happen to
   agree.
2. **NOAA-documentation level (checked independently of the code, per the
   user's request to verify this against the authoritative source rather
   than trust the ETL alone).** NCEP's own GRIB2 parameter table
   ([Table 4.2-10-0, "Oceanographic — Waves"](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-10-0.shtml))
   defines `DIRPW` ("Primary Wave Direction") and `PERPW` ("Primary Wave
   Mean Period") as real, independent parameters — but defines **no
   corresponding height parameter** anywhere in the table. Every other
   height in this archive traces to a real NOAA height field (`HTSGW` →
   `sigwh`, `WVHGT` → wind-sea height, numbered swell-partition heights →
   `pswh`); `pwh` cannot, because NOAA never defined a "primary wave
   height" field for it to come from. Reusing `sigwh`/`HTSGW`'s value for
   `pwh` was very likely a naming-compatibility stand-in for a field that
   structurally does not exist upstream, not an ETL bug or a coincidence.

**Consequence for this taxonomy.** The "peak wave" row of the Class (C)
table above is not a genuine 3-DOF triplet the way wind-sea and primary
swell are. `pwd` and `pwp` are independent, real fields (`DIRPW`,
`PERPW`); `pwh` carries zero information beyond what `sigwh` (Class A)
already provides. Any downstream stage that treats `pwh` as an independent
observable — e.g. summing it in quadrature against `wsh`/`pswh` as though
it were a third, disjoint partition height (§6, and see
`docs/physical_consistency_constraints.md`'s energy-budget constraint,
corrected alongside this document) — is double-counting `sigwh`. The
correct treatment is to keep `pwd`/`pwp` as the genuinely independent
members of this group, and treat `pwh` as redundant with the Class (A)
`sigwh` tensor that already exists — not as this triplet's own height.

As derived in §2.2, height/period/direction of one partition are three
different functionals of the *same* $S(f,\theta)$ restricted to that
partition's support — not three independent draws. The representation
should reflect that coupling as far as the tensor design allows:

- **Period is `log`-transformed before it becomes a tensor: $T := \log
  T_{m01}$, not raw $T_{m01}$.** $T_{m01}$ is an ordinary positive
  scalar — never circular — but is left in raw units nowhere in this
  design; see the new subsection below for why, and for the matching
  height treatment.
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
  not an accident, and one a plain unit-direction vector would lose — so
  this embedding is kept magnitude-scaled, not switched to a unit vector,
  despite the redundancy the next point introduces).

  **Resolved (was an open caveat for the Tucker document): a fourth
  tensor, $\log E := 2\log H = \log(H^2)$, is the authoritative source for
  reconstructing $H$ — not `hypot(u_dir, v_dir)`.** The caveat this
  section previously left open — that `hypot` of two independently
  low-rank-approximated reconstructions is only an approximation to a
  directly-decomposed $H$, since low-rank approximation does not commute
  with a nonlinear map like `hypot` — is resolved by retaining $\log E$ as
  its own, separately Tucker-fit tensor, and reconstructing $H =
  \exp(\log E/2)$ from *its own* compressed reconstruction rather than
  from `hypot`'s biased combination of two others. See the new subsection
  below for the full justification (why `log`, not raw $H$) and for why
  this is worth the resulting redundancy (`hypot(u_dir,v_dir)` and
  $\exp(\log E/2)$ now both estimate $H$, and are expected to disagree
  slightly after independent low-rank compression — by design, use the
  $\log E$-derived value).

Per triplet, this produces **4 tensors**: $\log E$, $T$ ($=\log T_{m01}$),
$u_{\text{dir}}$, $v_{\text{dir}}$ — one more than the 3 raw archived
fields (height, direction, period), the added tensor being the price of
an unbiased magnitude-reconstruction path. For the peak-wave triplet
specifically, no separate $\log E$ tensor is created: `pwh` is `sigwh`
restated (above), so the Class-A `sigwh` tensor's own $\log E$ (added
alongside it, per the subsection below) already serves as the
authoritative height source for reconstructing `pwh` too — creating one
would just be `sigwh`'s $\log E$ under a different name.

### 3.3.1 Partition-absence NaN is not land/ice NaN

**Finding (task 17, 2026-08-20), real archive data, `sigwh`-derived land
mask, 6-month window, 280×230 South Pacific crop (64,400 ocean cells,
1,448 timesteps):** every prior mention of NaN in this archive's wave
fields (§0, §4 below; `docs/model_design.md:308,313,315`) treats it purely
as a land/ice signal. For the partition triplets, NaN also occurs at
genuine ocean cells whenever that specific partition simply isn't present
— a real physical "no partition" state, not missing data. Categorized
directly against an authoritative, variable-independent land mask (derived
from `sigwh`'s own NaN pattern — `sigwh` is an aggregate, NaN only over
land/ice — via `train.masking.land_mask_from_sigwh`, not from each
variable's own NaN pattern, which conflates the two causes):

| Variable | Ocean NaN fraction | Cells always-valid @ 100%/90%/70% time |
|---|---|---|
| `wsh` | 36.5% | 0.0% / 1.3% / 35.8% |
| `wsp` | 36.5% (identical to `wsh`) | 0.0% / 1.3% / 35.8% |
| `wsd` | **0.0%** | 100% / 100% / 100% |
| `pwd` | 0.0% | 100% / 100% / 100% |
| `pwp` | 0.0% | 100% / 100% / 100% |
| `pswh` | 1.5% | 70.7% / 94.5% / 100% |
| `pswp` | 1.5% (identical to `pswh`) | 70.7% / 94.5% / 100% |
| `pswd` | 0.0% | 100% / 100% / 100% |

(`pwh` excluded — degenerate/`sigwh` restated, §3.3.)

**No-ocean-NaN group: all three direction fields (`wsd`, `pwd`, `pswd`)
and the peak-partition period (`pwp`).** **Has-ocean-NaN group: both
height/period pairs for the two "sometimes-absent" partitions** (wind-sea
`wsh`/`wsp`; primary swell `pswh`/`pswp`, less severely). `wsh`/`wsp`'s
identical fractions, and `pswh`/`pswp`'s identical fractions, confirm
height and period go NaN together exactly, as expected — both are
computed from the same partition's spectral moments (§2.2), and both
vanish together when the partition is absent.

**A genuine asymmetry, checked directly rather than assumed:** `wsd` is
NaN 0% of the time even at cells/timesteps where `wsh` is NaN — but this
is not a stale/backfilled filler value either
(`train.masking.partition_companion_backfill_check`, same window):
`wsd`'s variance during `wsh`-NaN periods (std ≈ 74.5°) is *higher*, not
collapsed toward zero, than during `wsh`-valid periods (std ≈ 55.3°). A
frozen filler would show near-zero variance during the gap; this doesn't.
The archive most likely reports *some* actively-varying wind-sea direction
value (plausibly derived from `wad`, wind direction, which has no NaN at
all — §3.2) even when no wind-sea partition is detected, but confirming
that exact mechanism needs a direct `wsd`-vs-`wad` correlation check, not
yet performed — treat as a real, open data-provenance question, not a
resolved one.

**Sentinel policy** (decided, `tasks/17-partition-nan-sentinel-policy.md`
§1, notebook/`src/train/masking.py`-scoped only — not yet wired into
production): at a has-ocean-NaN cell, height gets sentinel `H=0` (the
clean physical limit as $m_0\to0$, marked VALID); the corresponding
`log E` uses a floored transform (`train.masking.height_log_energy`,
floor `H_FLOOR_M=0.01`, i.e. 1cm — real archive data in this window never
goes below 5cm, so the floor sits comfortably outside the real range, not
overlapping it) rather than `log(0)=-inf`. Period has no analogous clean
limit ($T_{m01}=m_0/m_1$ is $0/0$ as both moments vanish); see §5.6 for
its data-derived placeholder. Direction needs no independent sentinel —
the `(u_dir, v_dir)` embedding (§3.3 above) already shrinks to `(0,0)`
once height's sentinel is 0, regardless of `wsd`'s own value.

See `docs/open_questions.md` §5.6 for the full derivation and the
still-open production-wiring follow-on.

### 3.4 The `log`-transform convention for heights and periods

**Statement.** Every height-type field in this archive (`sigwh`, `wsh`,
`pswh` — Class A/C heights; excluding the degenerate `pwh`, §3.3) is
represented downstream not just via its magnitude-scaled direction
embedding (Class C) but by an additional tensor $\log E := \log(H^2) =
2\log H$. Every period field (`wsp`, `pwp`, `pswp`, i.e. $T_{m01}$, §2.2)
is represented as $T := \log T_{m01}$, not raw $T_{m01}$. `sst` (Class A,
not a wave-energy quantity) is explicitly **excluded** — nothing below
applies to it.

**Floor/epsilon gap (surfaced and resolved by task 17, §3.3.1 above):**
this convention as originally stated never specified what happens at
$H=0$ (`log(0)=-inf`) — latent for any genuine calm-sea observation, and
load-bearing once §3.3.1's sentinel policy deliberately introduces `H=0`
at partition-absent cells. `train.masking.height_log_energy` floors
before logging (`H_FLOOR_M=0.01`); no equivalent floor is needed or
attempted for $T$'s $\log$, since $T_{m01}=m_0/m_1$'s own indeterminacy at
the partition-absent limit means the raw sentinel there is never `0` to
begin with (§5.6).

**Why `log`, not raw units.** Two independent justifications, one
statistical and one physical, both pointing the same direction:

1. **Dynamic range and least-squares leverage.** `sigwh`-class heights
   plausibly span roughly two orders of magnitude across the archive
   (calm partitions a few centimeters, storm partitions several meters;
   $H^2$, closer to the physically fundamental energy quantity, spans
   correspondingly more). Every downstream linear stage that touches this
   tensor before Gaussianization — Tucker's HOSVD/HOOI, harmonic
   climatology's per-cell regression (`harmonic_climatology.md`), DMD's
   snapshot-matrix least-squares fit — is an L2-optimal method whose
   objective is dominated quadratically by the largest-magnitude points in
   the data. Left in raw units, a handful of storm snapshots would
   dominate what each of these fits considers "signal," at the expense of
   correctly representing typical conditions. `log` compresses this range
   before any of those stages ever see the data.
2. **Fetch/duration-limited growth is a power law, and `log` linearizes
   it.** Classical JONSWAP-type fetch-limited growth curves
   (`physical_consistency_constraints.md` §2.2, `correlated_variables.md`
   §2) describe both $H_s$ and the spectrum's characteristic period as
   power laws in fetch $F$ (roughly $H_s\propto F^{1/2}$, period $\propto
   F^{1/3}$) — genuinely multiplicative growth, not additive. `log`
   converts multiplicative growth into additive/linear growth, which is
   the natural coordinate system for any of the linear stages above to
   operate in. This also explains, rather than just observes, why
   period's natural dynamic range is narrower than height's: a smaller
   growth exponent means a smaller multiplicative range for the same
   spread of fetch/duration values encountered archive-wide — so `log`'s
   benefit for period is real but smaller than for height, proportionally
   to that difference in exponent.

Both justifications are independent of, and unaffected by, whether a
group's channels are ever jointly (rather than independently) decomposed
— that is `correlated_variables.md` §6's separate, still-open question;
this `log` convention is a per-variable marginal transform, compatible
with strictly independent per-variable Tucker fits (`tucker_decomposition.md`
§4.1) exactly as much as with a joint one.

**Correction, at the query-time end of the pipeline: reconstruct $H =
\exp(\log E/2)$, never $\sqrt{u_{\text{dir}}^2+v_{\text{dir}}^2}$.** Once
`log E`'s own compressed reconstruction is available, it is the
authoritative source of magnitude — not `hypot` of the two independently
low-rank-approximated direction components, which (a) inherits whatever
bias each of those two *separate* compressions introduces, compounded, and
(b) is a nonlinear function applied *after* two independent lossy
approximations, which does not commute with those approximations having
been optimized in $u,v$-space rather than $H$-space. `sample_batch`/`sample`
(`src/weather/simulate.py`) and any physical-consistency check downstream
(`physical_consistency_constraints.md`) should read $H$ from $\exp(\log
E/2)$, using $u_{\text{dir}}, v_{\text{dir}}$ only for `atan2`-based
direction. Symmetrically, $T_{m01} = \exp(T)$ on reconstruction.

**Where this sits relative to other pipeline stages.** The `log` transform
is a pre-transform on physical-unit fields, applied before harmonic
climatology removal (`harmonic_climatology.md`) — that document's own
worked examples currently list raw `wsp` as a fitted tensor; read those as
$T=\log(\text{wsp})$ under this correction, since the climatology
regression is itself an L2 fit and benefits from the same dynamic-range
argument above. It is applied before Gaussianization under every
placement `gaussian_copula_and_marginals.md` §6.1 considers (i/ii/iii) —
those placements are about *where along the chain* Gaussianization sits,
not whether a physical `log` pre-transform happens first; the two
questions compose independently.

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
  into a single masked/dropped axis. **Caveat (§3.3.1):** for the
  partition triplets, a variable's *own* NaN pattern is not equivalent to
  land/ice — it also fires at genuine ocean cells whose partition is
  simply absent. This bullet's "boolean mask per grid" still describes
  land/ice correctly; deriving that mask from the *right* variable
  (`sigwh`, not the partition variable itself) is what §3.3.1 fixes.
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
| (A) | `sigwh` | $\log E_{\text{sigwh}} := \log(\text{sigwh}^2)$ | 1 |
| (A) | `sst` | `sst` (no `log` transform — §3.4 excludes it) | 1 |
| (B) | `was`, `wad_come_from` | `u_wind`, `v_wind` | 2 |
| (B) | `ocs`, `ocd` | `u_current`, `v_current` | 2 |
| (C) | `wsh`, `wsd`, `wsp` | $\log E_{\text{wsh}}$, $T_{\text{wsp}}(=\log\text{wsp})$, `u_wsea`, `v_wsea` | 4 |
| (C) | `pwh`\*, `pwd`, `pwp` | $T_{\text{pwp}}(=\log\text{pwp})$, `u_pwave`, `v_pwave` | 3 |
| (C) | `pswh`, `pswd`, `pswp` | $\log E_{\text{pswh}}$, $T_{\text{pswp}}(=\log\text{pswp})$, `u_pswell`, `v_pswell` | 4 |
| **Total** | 15 meaningful raw fields (`wad` decoy excluded) | | **17 independent tensors** |

\* `pwh` (≡ `sigwh`, see §3.3) is still the true (uncompressed) magnitude
used to construct `u_pwave`/`v_pwave`'s embedding at tensor-construction
time — that step happens before any Tucker compression, so no
reconstruction-bias issue arises there. `pwh`'s triplet gets no separate
$\log E$ tensor of its own (§3.3/§3.4): at *reconstruction* time, `sigwh`'s
own $\log E_{\text{sigwh}}$ tensor is the authoritative height source for
`pwh` too, not a new one. `pwd` and `pwp` are still real, independent
fields requiring their own tensors — but `u_pwave`/`v_pwave` should not be
read as carrying independent peak-wave-height information.

**Tensor count went from 15 to 17** relative to the previous revision of
this table: `wsh` and `pswh` each gained a dedicated $\log E$ tensor
(§3.3/§3.4); `sigwh`'s own tensor is a `log`-transform of its existing
content, not a new tensor; every period tensor's *content* changed
(`log`-transformed) without changing the *count*.

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
  Gaussian-copula stage marginally Gaussianizes $T_{m01}$, $u_{\text{dir}}$,
  and $v_{\text{dir}}$ independently without an explicit joint dependency
  structure across that triplet, it silently discards this coupling. This
  is a concern for the Gaussian-copula document to resolve, not this one
  — flagged here because it originates in the taxonomy, not the copula
  machinery.
- **Resolved (was open): low-rank approximation does not commute with
  `hypot`.** §3.3/§3.4 now retain $\log E$ as its own, separately Tucker-fit
  tensor and specify $H=\exp(\log E/2)$ as the authoritative
  reconstruction, rather than leaving `hypot(u_dir,v_dir)`'s
  reconstruction-bias unaddressed. What remains genuinely open, not
  resolved by this: whether the same treatment is worth its cost for
  Class B magnitudes (`was`, `ocs` — §3.2's new open note) and whether
  `log E`'s specific compression fidelity (rather than just its
  existence) is adequate at whatever Tucker rank is chosen — a rank-
  selection question for `tucker_decomposition.md`, not this document.
- **`sigwh`'s RMS relationship to partition heights assumes disjoint
  partitions — and, separately, `pwh` must not be one of the terms.**
  §2.2's derivation (`sigwh² ≈ wsh² + pswh² + ...`, summing genuinely
  independent partition heights in quadrature) is exact only if the
  partitioning algorithm produces spectrally non-overlapping bins, which
  is itself an approximation. But there is a second, more basic problem
  fixed by this revision (§3.3): `pwh` is not a third independent term to
  include at all — it is `sigwh` restated (confirmed against both
  `noaa-etl`'s transform table and NOAA's own GRIB2 parameter definitions,
  which have no "primary wave height" field). A relation of the form
  `sigwh = sqrt(wsh² + pwh² + pswh²)` is not merely approximate; it
  double-counts `sigwh` on both sides and should not be used. The
  physically meaningful quadrature relation this archive can actually
  check is `sigwh² ≈ wsh² + pswh²` (wind-sea plus secondary/primary-swell
  partition, whichever `pswh` represents) — still subject to the
  disjointness caveat above, and still missing whatever energy a genuine,
  unarchived "peak wave" partition height would have contributed.
- **`wad` is confirmed to be a decoy.** It decodes as a plausible
  coming_from field but is actually going_to, exactly 180° from
  `wad_come_from` in every cell at the checked snapshot. It should never
  be wired into any pipeline; `wad_come_from` is the only correct
  wind-direction source.
- **Categorized (was open): partition-absence NaN vs. land/ice NaN.**
  §3.3.1 above resolves *which* variables carry ocean-cell NaN and gives a
  sentinel policy for it, empirically checked against real archive data
  (`docs/open_questions.md` §5.6, `tasks/17-partition-nan-sentinel-policy.md`).
  What remains genuinely open: the exact mechanism behind `wsd`'s
  always-defined-but-not-stale behavior at wind-sea-absent cells (§3.3.1),
  and whether/how the policy gets wired into production
  (`src/weather/model.py`, `src/train/ingest.py`) — deliberately deferred,
  not yet approved.
