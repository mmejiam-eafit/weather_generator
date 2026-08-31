# Physical Consistency Constraints

A specification and graduate-level study guide for the post-hoc,
query-point rejection-sampling layer that enforces basic physical
plausibility on samples drawn from the `weather_generator` stochastic
model. This document does not describe existing code — see §0.

---

## 0. Two errors this document corrects

This document is part of a small documentation series (companions include
`docs/variable_taxonomy.md` and `docs/tucker_decomposition.md`) that
replaces an earlier draft set the user deleted because it made two
specific errors. Both are stated here explicitly, as required for every
document in the series, even where — as is mostly the case below — the
second error has little effect on the content.

### Error 1 — this must be prescriptive, not descriptive

The previous draft wrote as though it were documenting an existing module.
It is not. **There is no `src/weather/physics.py` (or any equivalent
consistency-checking module) anywhere in this repository today.** `git
grep` for rejection sampling, energy-budget checks, or mean-period bounds
in `src/` turns up nothing; `AGENTS.md`'s "Physical consistency (documented
limitation)" section explicitly flags this as unimplemented and requires
"explicit authorization" before any post-hoc layer is added.

In the user's own words: *"I DO NOT need them to reflect the current
code. It's the opposite. I need these documents to define what the process
IS in order to CORRECT the actual code files to what they should be."*

Accordingly, everything below — the module boundary, the function
signatures, the dataclasses, the retry budget — is a **specification for a
module to be written**, not a description of one that exists. Treat this
as an assignment brief a student could implement from, not as generated
API docs. Where existing code is referenced (`weather.simulate.sample_batch`,
`WeatherModel`), it is referenced only as the *interface this new layer
must wrap*, not as something this document is describing the internals of.

Note also that a separate file, `v3_implementation_handoff.md`, sits at
the repo root making related claims about a physical-consistency module.
Per `context.md`'s "Known open issues," that handoff cites
`docs/model_design.md` sections that don't exist and overstates how much
of the v2 pipeline is actually implemented. This document does not rely on
it and should not be treated as confirming it.

### Error 2 — the tensor-representation correction (why it barely touches this document)

The user's second, separate correction, stated in their own words: *"Don't
treat the entirety of the variables as a single 4D tensor of size (lat,
lon, time, num_variables); instead, treat each of them as a single 3D
tensor (lat, lon, time)... NEVER flatten them from this 3 dimensional
representation... Flattening has the disadvantage of losing local
information, so NEVER flatten."*

This rule governs how the **training-time** internal representation of
each physical variable is held — one `(lat, lon, time)` tensor per
variable, never stacked into a 4D block and never flattened — and is the
subject of `docs/variable_taxonomy.md` and `docs/tucker_decomposition.md`.

It is stated here for consistency across the series, but it changes almost
nothing about *this* document's content, for a specific reason worth
being explicit about: **rejection sampling as specified here operates
entirely at query time, on scalar, physical-unit values already produced
by `sample_batch`/`sample`/`simulate_field` at specific `(lat, lon, time)`
query points.** It never reconstructs, indexes into, or touches the full
spatial `(lat, lon, time)` training tensors, and it is not a training-time
transformer in the `sklearn.Pipeline` sense. Each accept/reject decision
is made independently, per query point, on a handful of scalars (`sigwh`,
`wsh`, `pwh`, `pswh`, `wsp`, `pwp`, `pswp`, `was`, ...) returned for that
one point. So while the constraint that "never flatten the training
tensor" is real and binding elsewhere in the pipeline, it has no
opportunity to be violated by a layer that only ever sees post-query
scalars — there is no tensor here to flatten or not flatten. This section
exists so a reader auditing the whole series for the error doesn't have to
guess why it isn't discussed further below.

---

## 1. What this is, why it's here

`WeatherModel` samples wave, wind, and ocean-current fields from a fitted
Tucker + DMD + mode-wise Ornstein-Uhlenbeck stochastic model, with a
Gaussian-copula layer (`train/copula.py`) coupling cross-cell and
cross-variable marginals (see the sibling copula document). A Gaussian
copula preserves marginal distributions and pairwise (linear) correlation
structure faithfully, but it has a well-known, structural limitation: it
cannot represent strong **joint tail dependence** — the tendency of
extreme values in genuinely coupled physical quantities to co-occur more
often than a Gaussian dependence structure predicts. Concretely, this
model can and does draw samples where, e.g., the wind-sea component of
significant wave height exceeds what the total combined sea height allows,
or where a small local wind speed is paired with a wind-sea height that
implies a fully-developed gale, or where a mean period falls far outside
any wave regime the model is meant to represent.

None of these are "bugs" in the sense of a coding error; they are the
predictable consequence of fitting a copula whose tail-dependence
structure is looser than the true joint physics. The question this
document answers is: **given that the fitted model will occasionally
produce physically inconsistent draws, what is the theoretically correct
way to filter them, and what exactly should that filter check?**

This document specifies three concrete, checkable physical constraints
(§2.1–2.3) and the sampling-theoretic justification for enforcing them by
rejection rather than by projection (§2.4), then specifies the shape of
the module that should implement this (§4).

---

## 2. Theoretical foundation

### 2.1 Wave-energy budget: $H_{s,\text{total}}^2 \approx \sum_i H_{s,i}^2$

**Correction (post-dated the original draft of this section): `pwh` is not
one of the $H_{s,i}$ terms.** It is confirmed — both at the `noaa-etl`
source level (two `VARIABLE_TRANSFORMATION` entries, `sigwh` and `pwh`,
both reading GRIB `shortName: "swh"` through an identical transform chain)
and independently at NOAA's own documentation level (NCEP GRIB2
[Table 4.2-10-0](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-10-0.shtml)
defines `DIRPW`/`PERPW` for the "primary wave" but no corresponding height
parameter at all) — that `pwh` is `sigwh` restated, not an independently
measured "primary wave" partition height. Every formula, tuple, and
tolerance-calibration step below that previously included `pwh` as a
disjoint quadrature term has been corrected accordingly; see
`docs/variable_taxonomy.md` §3.3 for the full derivation of this finding.

**Setup.** For a stationary, zero-mean sea-surface elevation process
$\eta(t)$, the wave spectrum $S(f)$ is defined so that the $n$-th spectral
moment is

$$m_n = \int_0^\infty f^n\, S(f)\, df,$$

and the zeroth moment is exactly the variance of the surface elevation,
$m_0 = \operatorname{Var}(\eta) = \mathbb{E}[\eta^2]$ (Parseval's theorem
applied to a stationary process). Significant wave height is defined —
originally as the mean of the highest one-third of zero-crossing wave
heights, $H_{1/3}$ — and, under the (excellent, empirically-verified for
wind-generated seas) narrow-band assumption that wave heights follow a
Rayleigh distribution with scale parameter tied to $m_0$, this reduces to
the closed form

$$H_s = 4\sqrt{m_0}.$$

This is the archive's `sigwh` for the combined sea, `wsh` for the wind-sea
partition, and `pswh` for the (secondary) swell partition
(`core/config.py`'s `VARIABLE_GRID` groups these as the wave-grid
variables). The archive also carries a `pwh` field, nominally a fourth
partition height, but it is **not** an independent measurement — see the
note after the superposition relation below — and does not belong in this
list as a partition height.

**Superposition.** A partitioned sea state is, physically, the linear
superposition of surface elevations generated by distinct systems: the
locally wind-forced sea, and one or more swell trains arriving from
distant, already-decayed storms. Write

$$\eta_{\text{total}}(t) = \eta_{\text{ws}}(t) + \sum_{i} \eta_{\text{swell},i}(t).$$

Variance of a sum of random processes is, in general,

$$\operatorname{Var}(\eta_{\text{total}}) = \operatorname{Var}(\eta_{\text{ws}}) + \sum_i \operatorname{Var}(\eta_{\text{swell},i}) + 2\!\!\sum_{\text{pairs } (i,j)}\!\! \operatorname{Cov}(\eta_i, \eta_j).$$

**The load-bearing assumption.** The energy-budget relation used as a
consistency constraint asserts the cross-covariance terms are
negligible — i.e. that the partitions are **statistically independent**
(or at least uncorrelated). This is physically reasonable to first order:
wind-sea and swell are generated by different systems, at different times
and places, arriving from different directions with generally
incommensurate frequency content, so their elevation records behave like
independent random processes over the averaging window used to estimate
the spectrum. Under that assumption, $m_{0,\text{total}} = m_{0,\text{ws}} +
\sum_i m_{0,\text{swell},i}$, and substituting $m_{0,i} = (H_{s,i}/4)^2$
gives, exactly under the independence assumption,

$$H_{s,\text{total}}^2 = H_{s,\text{ws}}^2 + \sum_i H_{s,\text{swell},i}^2.$$

For this archive's variables, that is $\text{sigwh}^2 \approx \text{wsh}^2 + \text{pswh}^2$
— **not** `+ pwh²`. `pwh` must be excluded from this sum: it is a literal
duplicate of `sigwh` at the ETL level, not a third independent partition
height. This is confirmed two ways: (1) `noaa-etl`'s `gfs_transform.py`
derives both `sigwh` and `pwh` from the identical GRIB `shortName: "swh"`
field through an identical transform chain — they are the same array on
disk; (2) independently, NOAA's own GRIB2 parameter table
([4.2-10-0](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-10-0.shtml))
defines `DIRPW`/`PERPW` ("Primary Wave Direction"/"Primary Wave Mean
Period") as real fields but defines **no corresponding height
parameter** — there was never a "primary wave height" field for `pwh` to
independently measure. Including `pwh²` in the sum double-counts `sigwh`
against itself: substituting `pwh ≡ sigwh` into the naive three-term
version gives `sigwh² ≈ wsh² + sigwh² + pswh²`, i.e. `0 ≈ wsh² + pswh²`,
which is false for essentially every real sea state. The two-term relation
above is the one this archive can actually check, and it is correspondingly
weaker evidence than a genuine three-partition budget would be — it only
tests wind-sea/swell energy balance against the total, not a full
partition-by-partition accounting, because this archive does not carry an
independent third partition height to include.

**Why this is an approximate, tolerance-band relationship, not an exact
identity** — be explicit about all three sources of slack (these apply to
the genuine `wsh`/`pswh` terms; they are irrelevant to the excluded `pwh`
term, which is wrong for a structural reason, not an approximation one):

1. **Independence is approximate, not exact.** Any finite-record spectral
   estimate has nonzero sample cross-correlation between partitions even
   when the population cross-correlation is zero. More importantly,
   nonlinear four-wave resonant interactions (Hasselmann 1962) genuinely
   transfer energy between the wind-sea and swell frequency bands over
   time, which is a real, physical coupling mechanism, not just estimation
   noise — the partitions are not *perfectly* independent generative
   processes.
2. **Partitioning-algorithm boundary effects.** WAVEWATCH III's spectral
   partitioning (an inverted-watershed algorithm over the 2-D
   frequency-direction spectrum, Hanson & Phillips 2001) assigns each
   spectral bin to exactly one partition based on local minima in the
   spectrum. Energy near a partition boundary can be ambiguously assigned,
   so $\sum_i m_{0,i}$ computed from the *partitioned* outputs need not sum
   exactly to $m_0$ of the *total* spectrum even before any independence
   assumption is invoked.
3. **Truncation of minor partitions.** This archive retains only the
   wind-sea and one swell partition (`pswh`); a real sea state can carry
   additional, lower-energy swell systems (and, in principle, a genuine
   independent peak/primary-wave partition this archive does not carry at
   all) that are dropped, biasing the two-term sum low relative to the
   true total — on top of, and larger than, the ordinary truncation slack
   a full three-or-more-partition archive would have.

**Consequence for this document's use of the relation.** The constraint
should be checked as a **tolerance band around equality**, e.g.

$$\left| \frac{\text{sigwh}^2 - (\text{wsh}^2 + \text{pswh}^2)}{\text{sigwh}^2} \right| < \varepsilon_{\text{energy}},$$

with $\varepsilon_{\text{energy}}$ an illustrative, empirically-tuned
constant (§5), not a value derived from theory — the theory only tells us
the relation should hold *approximately*, not what the tolerance should
be.

### 2.2 Fetch-limited Pierson-Moskowitz coupling: $H_s \approx C\, U^2/g$

**The fully-developed-sea spectrum.** Pierson & Moskowitz (1964) proposed,
from empirical fits to North Atlantic ship observations, a one-dimensional
spectrum for a **fully developed sea** — one that has been acted on by a
steady wind over unlimited fetch and duration, so that the wave field is
in energetic equilibrium with the wind and no longer growing:

$$S_{\text{PM}}(\omega) = \frac{\alpha g^2}{\omega^5}\exp\!\left[-\beta\left(\frac{\omega_p}{\omega}\right)^4\right],$$

with the Phillips constant $\alpha \approx 8.1\times 10^{-3}$, shape
constant $\beta \approx 0.74$, and peak frequency tied to wind speed by
$\omega_p \approx 0.855\, g / U$ (with $U$ the wind speed at the reference
height used in the original fit, conventionally 19.5 m; different
reference heights rescale the constant below).

**Zeroth-moment integration.** Substitute $u = \beta\,\omega_p^4\,
\omega^{-4}$, so $du = -4\beta\,\omega_p^4\,\omega^{-5}\,d\omega$, giving
$\omega^{-5}\,d\omega = -du / (4\beta\,\omega_p^4)$. Then

$$m_0 = \int_0^\infty S_{\text{PM}}(\omega)\, d\omega
= \alpha g^2 \int_0^\infty \omega^{-5} e^{-\beta(\omega_p/\omega)^4}\, d\omega
= \frac{\alpha g^2}{4\beta\,\omega_p^4}\int_0^\infty e^{-u}\, du
= \frac{\alpha g^2}{4\beta\,\omega_p^4}.$$

Substituting $\omega_p = 0.855\, g/U$:

$$m_0 = \frac{\alpha g^2}{4\beta \cdot 0.855^4\,(g/U)^4} = \frac{\alpha}{4\beta \cdot 0.855^4}\cdot\frac{U^4}{g^2}.$$

Then, using $H_s = 4\sqrt{m_0}$,

$$H_s = 4\sqrt{\frac{\alpha}{4\beta\cdot 0.855^4}}\;\frac{U^2}{g} = C\,\frac{U^2}{g}.$$

Plugging in $\alpha = 0.0081$, $\beta = 0.74$ gives $C \approx 0.29$ at the
19.5 m reference height; converting to a 10 m reference height (as used by
GFS/`was` in this archive) via the standard 1/7-power-law wind profile
brings the commonly cited constant down into the $C \approx 0.21$–$0.24$
range seen across the literature. **Treat $C \approx 0.21$ as the
illustrative textbook value, not as a number this repository should assume
is correct for its own wind reference height and instrument mix** — it
should be re-fit or at least sanity-checked against the archive's own
paired `(was, sigwh)` observations under genuinely fetch-limited
conditions (§5) before being hard-coded as a tolerance-band center.

**Why this relation only applies in fetch-limited seas.** The derivation
above assumes the sea state is in *local equilibrium with the local wind*
— i.e., that essentially all the wave energy present was generated by the
wind blowing at that location. This is the defining condition of a
**fully-developed sea**, and it is only physically realized in practice
under **fetch-limited or duration-limited** growth reaching saturation —
classically, enclosed or semi-enclosed basins (e.g. large lakes, marginal
seas, coastal embayments) where the fetch (the distance over which wind
blows unobstructed) is short enough, and the wind steady enough for long
enough, that the sea state's growth curve (Hasselmann's JONSWAP-style
fetch-limited growth relations, which describe $H_s$ and the spectrum's
characteristic period climbing with $\sqrt{\text{fetch}}$/$\sqrt{\text{duration}}$
up to a plateau — classically stated in terms of $T_p$, though this
archive's own period fields are $T_{m01}$, see §2.3's correction) has
actually saturated at the PM asymptote.

**Why it is not expected to hold in the open ocean.** In the open ocean,
the fetch and duration are effectively unbounded, so a genuinely
fully-developed sea driven purely by the *local* wind is unusual — but
more importantly, a large fraction of the wave energy present at any given
open-ocean point is often **swell**: waves generated by a distant storm
system, which has since moved on or decayed, that has propagated for days
across open water with comparatively little dissipation (deep-water
gravity waves are only weakly damped). The observed local wind at a query
point in the open ocean can therefore be essentially decoupled from the
locally observed wave field — light local wind with a large, long-period
swell arriving from a storm thousands of kilometers away is routine and
physically correct, not an inconsistency. Enforcing $H_s \approx C\,U^2/g$
against the *total* $H_s$ (`sigwh`) at an arbitrary open-ocean point would
therefore incorrectly reject a large fraction of physically valid swell-
dominated samples.

**Consequence.** This constraint should be applied, if at all, only (a)
against the **wind-sea partition height** (`wsh`), which by construction
represents the locally-wind-forced component and is the quantity the PM
relation is actually about, not the swell-inclusive total; and (b) gated
to query points identified as genuinely fetch-limited (enclosed/marginal
seas, or a short-fetch flag from bathymetry/coastline data), not applied
uniformly across the open ocean. This is a substantive scope restriction,
not a detail — see §3 and §6.

### 2.3 Mean-period plausibility

**Correction: this archive's period fields are $T_{m01}$ (mean period,
$m_0/m_1$), not $T_p$ (peak period, $1/f_p$) — an earlier version of this
section conflated the two.** Confirmed directly against `noaa-etl`'s
source (`wsp`/`pwp`/`pswp` derive from GRIB `shortName`s `mpww`/`perpw`/`mpts`)
and NOAA's own parameter naming (`"Mean period of wind waves"`, `"Primary
wave mean period"`, `"Mean period of total swell"` — distinct,
separately-numbered GRIB2 parameters from NOAA's genuinely separate "Peak
Wave Period" family, which this archive never reads); see
`variable_taxonomy.md` §2.2 for the full evidence trail. $T_{m01}$ is an
energy-weighted average frequency across the *entire* partition spectrum,
inverted — not the reciprocal of the single largest spectral bin. The two
coincide only for an infinitely narrow spectral peak and can diverge
meaningfully for a broad or multi-modal one.

**Why the argument below still mostly works despite this.** The
deep-water dispersion relation $\omega^2=gk$ strictly relates *one*
frequency to *one* wavelength — it has no direct opinion about an
energy-weighted average frequency taken across a whole spectrum of
different wavelengths. Applying it to $T_{m01}$ the way the derivation
below applies it is therefore an additional approximation on top of
everything already flagged as approximate in this section: it implicitly
treats the spectrum as narrow enough that its energy-weighted mean
frequency behaves, for dispersion purposes, like a single characteristic
frequency — reasonable for the plausibility-*bound* use this section puts
it to (both $T_p$ and $T_{m01}$ occupy the same broad physical regime, so
a floor/ceiling check calibrated loosely is not sensitive to which one is
used), but not reasonable for anything that needs the *specific* numeric
wavelength implied by a *specific* period value (see
`correlated_variables.md` §3.3, where this distinction is load-bearing,
not just a caveat).

**Deep-water dispersion.** For linear gravity waves in deep water (depth
much greater than wavelength), the dispersion relation is

$$\omega^2 = g k,$$

so period $T = 2\pi/\omega$ and wavelength $\lambda = 2\pi/k$ satisfy $T =
\sqrt{2\pi\lambda/g}$: longer waves have longer periods, with no upper
bound imposed by the relation itself — $T \to \infty$ as $\lambda \to
\infty$ is mathematically permitted by pure deep-water gravity-wave
dispersion.

**The capillary-gravity transition and the short-period floor.** At short
wavelengths, surface tension becomes dynamically significant and the
dispersion relation gains a cubic-in-$k$ correction (Kelvin's
capillary-gravity dispersion relation):

$$\omega^2 = g k + \frac{\tau}{\rho} k^3,$$

with $\tau \approx 0.074\,\text{N/m}$ the air-water surface tension and
$\rho \approx 1000\,\text{kg/m}^3$. Phase speed $c(k) = \omega/k =
\sqrt{g/k + \tau k/\rho}$ has a minimum where $dc/dk = 0$, at

$$k_{\min} = \sqrt{\frac{g\rho}{\tau}} \approx 364\ \text{rad/m} \quad(\lambda_{\min}\approx 1.7\ \text{cm}),$$

with minimum phase speed $c_{\min} = (4g\tau/\rho)^{1/4} \approx
0.23\,\text{m/s}$ and corresponding period $T_{\min} = 2\pi/(c_{\min}
k_{\min}) \approx 0.07\,\text{s}$. Below this scale, waves are
capillary-dominated ripples, not gravity waves, and are governed by
entirely different generation and dissipation physics (viscous damping
dominates; there is no meaningful "significant height"/"mean period"
partition structure).

Be precise about what this means for the "couple of seconds" floor used in
practice: **the true capillary-gravity minimum period (~0.07 s) is much
shorter than the practical floor.** The relevant fact for this archive is
not the exact minimum of the dispersion relation, but that WAVEWATCH
III's partitioned wind-sea/swell fields represent *organized,
wind-forced gravity waves* — the discretized frequency bins and the
partitioning algorithm are built around the gravity-wave regime the model
resolves, not the capillary-gravity ripple regime, and the archive does
not represent ripples as a windsea/swell partition at all. In practice,
wind-sea mean periods (`wsp`) below roughly one to a few seconds indicate
either a numerical/interpolation artifact or a query landing outside the
regime this dataset represents, rather than a physically meaningful very
short-period gravity sea. The "couple of seconds" lower plausibility bound
is therefore a practical floor motivated by, but numerically well above,
the first-principles capillary-gravity transition.

**The long-period ceiling is empirical, not derived.** As shown above,
deep-water dispersion places no first-principles upper bound on gravity-
wave period. The observed climatological ceiling — long-period swell from
major Southern Ocean or North Pacific storms reaching mid-to-high-20s-second
mean periods at the shores they eventually strike, with anything
meaningfully beyond that essentially unobserved in routine wave
climatology — comes from a combination of practical constraints that are
not encoded in the dispersion relation itself: the finite size and
duration of real storm-generating wind fields (fetch/duration-limited
growth again, §2.2, bounds how much energy accumulates at very low
frequencies in the first place), the finite size of ocean basins limiting
how far a swell train can travel before striking a coastline, and
frequency dispersion spreading a swell train's energy over time and space
as it propagates (longer-period components travel faster in deep water,
so an initially compact storm-generated spectrum fans out, and its
lowest-frequency, longest-period tail arrives increasingly attenuated and
eventually indistinguishable from noise). **Be honest that any specific
upper cutoff used operationally (e.g. "25 seconds") is an empirical
plausibility bound drawn from wave-climatology records, not a hard limit
derivable from $\omega^2 = gk$ or any other first-principles relation.**

**Consequence.** The constraint checked is a plausibility interval

$$T_{\min} \lesssim T_{m01} \lesssim T_{\max},$$

illustratively $T_{\min}\approx 2\,\text{s}$, $T_{\max}\approx 25\,\text{s}$,
applied to each period field present (`wsp`, `pwp`, `pswp`) — both bounds
tunable, and the upper bound specifically understood to be a climatological
plausibility statement rather than a physical impossibility statement.

### 2.4 Rejection sampling vs. projection: the sampling-theoretic case

Let $q(x)$ denote the joint distribution, at a fixed query point, over the
handful of scalar quantities the model returns (e.g. $x = (\text{sigwh},
\text{wsh}, \text{pwh}, \text{pswh}, \text{wsp}, \text{pwp}, \text{pswp},
\text{was}, \dots)$ at that point) — this is exactly the distribution
`sample_batch`/`sample` draws from, implicitly, via the fitted
Tucker+DMD+OU+copula machinery. Let $C(x) \in \{0,1\}$ be the indicator
that $x$ satisfies the constraints of §2.1–2.3 (a conjunction of the
energy-budget, PM, and mean-period checks). Assume $Z \equiv \Pr_q(C(x)=1)
> 0$.

**Claim.** The classical acceptance-rejection algorithm — draw $X \sim
q$; if $C(X) = 1$, accept and return $X$; otherwise discard and redraw,
independently, until acceptance — produces a sample distributed *exactly*
as $q(x \mid C(x)=1) = q(x)\,C(x)/Z$.

**Proof.** Let $X^\*$ be the value ultimately returned. For any measurable
set $A$,

$$\Pr(X^\* \in A) = \sum_{n=1}^{\infty} \Pr(\text{first } n-1 \text{ draws rejected})\cdot \Pr(\text{$n$-th draw} \in A \text{ and accepted}).$$

Draws are i.i.d. from $q$, so each has rejection probability $1-Z$
independent of prior draws, and

$$\Pr(\text{draw} \in A,\ C=1) = \int_A q(x)\,C(x)\,dx.$$

Hence

$$\Pr(X^\* \in A) = \left(\int_A q(x)C(x)\,dx\right)\sum_{n=0}^{\infty}(1-Z)^n
= \frac{1}{Z}\int_A q(x)\,C(x)\,dx = \int_A q(x\mid C(x)=1)\,dx.$$

Since this holds for every measurable $A$, $X^\*$ is distributed exactly
as $q(\cdot \mid C=1)$. $\blacksquare$

This is the standard von Neumann acceptance-rejection method specialized
to an indicator acceptance function (rather than the more general
"accept with probability $f(x)/(Mg(x))$" form used when resampling from a
different proposal $g$ to approximate a target $f$) — here the proposal
*is* the target, we are simply conditioning it on a constraint, and the
result is exact for any finite acceptance rate $Z>0$, with expected number
of draws per accepted sample $= 1/Z$.

**Contrast: projection onto the constraint manifold.** An alternative,
tempting approach is to keep every draw but deterministically map invalid
ones onto the nearest point of the constraint set — e.g. rescale `wsh`,
`pswh` so the energy budget holds exactly, or clip `wsp` to
$[T_{\min}, T_{\max}]$. Formally, let $M = \{x : C(x)=1\}$ and define a
projection $\Pi(x) = \arg\min_{x'\in M} d(x,x')$ for some metric $d$, with
$\Pi(x)=x$ when $x\in M$ already.

This is **not sampling from any well-defined probability distribution**,
for a concrete, structural reason: $\Pi$ is many-to-one on the invalid
region $M^c$. An entire open neighborhood of invalid points — everywhere
that maps to the same nearest boundary point — is collapsed onto a single
point of $\partial M$ (a set of strictly lower dimension than the ambient
space whenever $M$ has nonempty interior). The pushforward measure
$\Pi_\#q$ therefore has an atom (or, more generally, a singular component
concentrated on the lower-dimensional boundary $\partial M$) with mass
equal to $\Pr_q(M^c \text{ mapping to that point's preimage})$ — mass that,
under $q$ itself, was spread continuously over an open region, not
concentrated at a single point. This is a genuine artifact: it manufactures
spurious density right at the constraint boundary that has no
correspondence to how likely $q$-draws near the boundary actually are; a
downstream user computing, say, the empirical distribution of `wsp` near
$T_{\max}$ would see an inflated point mass at exactly $T_{\max}$ that
reflects nothing about the physics, only the geometry of the clipping
operation. Contrast this with rejection sampling's output, which is
absolutely continuous wherever $q$ restricted to $M$ is, with no
boundary artifact of any kind, by the proof above.

There is a real cost to rejection sampling — it is not free. When the
acceptance rate $Z$ is low, the expected number of draws per accepted
sample, $1/Z$, grows correspondingly, wasting sampling budget (and, per
`AGENTS.md`'s BLAS-threading rules, that waste is per-point serial work,
not something joblib parallelism a cross query points can hide if the
rate is uniformly low). But this cost is precisely the tradeoff for
exactness: **rejection sampling is the theoretically clean choice because
it is provably exact, at the cost of discarded work; projection is cheap
but samples from nothing in particular.**

---

## 3. Guidelines

State plainly, and do not soften:

- **All three constraints in §2.1–2.3 are magnitude-only.** None of them
  checks direction. This is a real, physically relevant gap: for example,
  a wind-sea partition is, by definition, locally wind-generated, so its
  direction (`wsd`) is physically expected to track the wind direction
  (`wad_come_from`, the archive's source-of-truth wind direction field —
  see `context.md`'s direction-convention findings) within some plausible
  veer/backing angle. Nothing in this specification checks that a sampled
  `wsd` is anywhere near the sampled wind direction at the same point.
  Flag this explicitly as a scope gap and a candidate future extension
  (§6), not something silently covered by the height/period checks above.
- **Pure scalar/vector fields outside the wave partitions are entirely out
  of scope for all three constraints.** `sst` (sea surface temperature),
  `was`/`wad`/`wad_come_from` (wind speed/direction), and `ocs`/`ocd`
  (ocean current speed/direction) are not wave-partition quantities and
  none of the three constraints says anything about them as *outputs* to
  be validated. Note in particular that `was` (wind speed) appears in
  §2.2 only as an **input** used to bound `wsh` — the PM constraint checks
  that wind-sea height is consistent with wind speed, it does not itself
  constrain wind speed. There is currently no physical-consistency check
  of any kind on `sst`, or on `ocs`/`ocd`, in this specification.
- Treat every numeric tolerance in this document (energy-budget slack,
  the PM coefficient $C$, the period bounds $T_{\min}/T_{\max}$) as
  illustrative and tunable, to be calibrated against real archive data
  (§5), not as physical constants to hard-code from this document alone.

---

## 4. Correct treatment / specification

This is a **post-processing layer**, not a training-time or model-fitting
component. It should:

- Wrap whatever sampling function the (corrected) model exposes —
  today that is `weather.simulate.sample_batch` (and its thin wrappers
  `sample`, `simulate_field`), which already return physical-unit arrays
  per `(lat, lon, time)` query point (see the signature in
  `src/weather/simulate.py`). This layer consumes that output; it does
  not replace or reimplement it.
  - **Reconstruction authority, since this layer's correctness depends on
    it:** for every height field involved in §2.1–2.2, the physical-unit
    value `sample_batch` returns must come from $H=\exp(\log E/2)$
    (`variable_taxonomy.md` §3.4's dedicated tensor), never from
    `hypot(u_dir, v_dir)`. The latter combines two *separately*
    low-rank-approximated reconstructions and is a biased estimate of $H$
    for exactly the reason §3.4 resolves — feeding a `hypot`-based value
    into this layer's energy-budget or PM checks would run the arithmetic
    in §2.1–2.2 correctly on a systematically wrong input, not a wrong
    calculation. Likewise, every period field must come from
    $T_{m01}=\exp(T)$, consistent with §2.3's correction. This is a
    requirement on the sampling layer this document wraps, not something
    this layer can itself verify from a bare scalar it receives.
- Operate **independently per query point**, on the physical-unit scalars
  returned for that point. It has no notion of, and does not need, the
  full spatial `(lat, lon, time)` tensor for any variable (this is why
  Error 2, §0, barely touches this document).
- Evaluate the three constraints of §2.1–2.3 (energy budget, gated PM
  coupling, mean-period bounds) per point, independently of every other
  query point in the batch.
- On failure, **redraw the point with a fresh seed derivation** and
  re-check, up to a **bounded retry budget** (a fixed maximum number of
  attempts per point) rather than looping indefinitely — a low
  acceptance rate at some points is itself important diagnostic
  information (§5), not something to mask by unbounded retrying.
- If the retry budget is exhausted without an accepted draw, return an
  explicit non-acceptance flag plus a defined fallback value (e.g. the
  last-drawn value, or a climatological point estimate) — never silently
  substitute a projected/clipped value as if it were a valid draw (§2.4).
- **Not** touch training, the fitted `WeatherModel` state (Tucker, DMD,
  mode-OU, copula), or any preprocessing/decomposition stage. This is
  additive and sits strictly downstream of sampling.

Sketch of the shape this should take (illustrative signatures, following
this repo's existing conventions — free functions over `WeatherModel`,
per `AGENTS.md`'s "sampling is free functions, not methods" rule, and
frozen dataclasses for structured results):

```python
# Illustrative only — nothing below exists yet; this is the specification.

@dataclass(frozen=True, slots=True)
class ConsistencyTolerances:
    energy_budget_rel_tol: float          # epsilon_energy, see 2.1
    pm_coefficient: float                 # C, see 2.2 -- only applied when fetch-limited
    pm_rel_tol: float
    peak_period_min_s: float              # T_min, see 2.3
    peak_period_max_s: float              # T_max, see 2.3

@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    values: np.ndarray            # (N, len(variables)) -- accepted or fallback values
    accepted: np.ndarray          # (N,) bool
    n_attempts: np.ndarray        # (N,) int -- how many draws each point took
    per_constraint_pass: dict[str, np.ndarray]  # diagnostic, see 5

def enforce_physical_consistency(
    model: WeatherModel,
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    variables: list[str],
    tolerances: ConsistencyTolerances,
    max_attempts: int = 8,
    seed: int | None = None,
) -> ConsistencyResult:
    ...
```

---

## 5. Results: tolerance calibration and diagnostics

**Choosing tolerance bands.** Every tolerance in §2 should be set by
checking, against real archive observations (not model samples), what
fraction of genuine `(sigwh, wsh, pswh, wsp, pwp, pswp, was)` tuples
already satisfy each constraint at a series of candidate tolerance widths
(`pwh` excluded from the energy-budget calibration — it is not an
independent quantity; see §2.1). Concretely, for the energy budget:
compute the relative residual
$|\text{sigwh}^2 - (\text{wsh}^2+\text{pswh}^2)| /
\text{sigwh}^2$ across a representative archive sample (stratified by
season and region, since storm activity, and hence partition
cross-correlation, varies systematically with both), and pick
$\varepsilon_{\text{energy}}$ so that some target fraction (e.g. the 95th
or 99th percentile of the real-data residual) is inside the band — a
tolerance so tight that it rejects a meaningful fraction of genuine
archive observations is miscalibrated, not "strict." The same
percentile-of-real-data approach applies to the PM coefficient (fit $C$
and its acceptable band from real `(was, wsh)` pairs at genuinely
fetch-limited archive locations only) and to the mean-period bounds (set
$T_{\min}, T_{\max}$ from the empirical tails of the real `wsp`/`pwp`/`pswp`
distributions, not assumed from the illustrative numbers in §2.3).

**What the rejection loop should return.** Beyond the accepted value
itself, the loop should surface, per query point: an accept/reject flag,
the number of attempts taken (a direct measure of local acceptance rate),
and a fallback value used if the retry budget was exhausted. Critically,
these rejection rates should be retained and reported as **a diagnostic
of the underlying model**, not discarded as mere filtering overhead. A
rejection-rate map broken out by constraint (energy budget vs. PM vs.
mean-period), by season, and by region is direct evidence of *where and
when the copula's tail-dependence structure is worst* — e.g. a
persistently high energy-budget rejection rate in a specific ocean basin
during its storm season would point straight at where the copula most
needs improvement (§6), which is exactly the kind of signal a purely
pass/fail filter would throw away.

---

## 6. Open questions / limitations

- **Direction-blind scope gap (§3).** None of the three constraints
  checks direction consistency — most notably, wind-sea direction (`wsd`)
  tracking wind direction (`wad_come_from`). This is a known, physically
  meaningful gap, not an oversight to be silently accepted; it is a
  candidate future extension once a defensible veer/backing tolerance is
  derived and validated the same way as §5.
- **Cross-cell/cross-time blindness is a structural limitation of any
  per-point-independent check**, not just an omission to patch. Because
  every accept/reject decision in this specification is made independently
  per query point, it **cannot** enforce any relationship that spans
  multiple points — e.g. a storm's wave field advecting across adjacent
  grid cells at an implausible speed, or a wave field at time $t$ and
  $t+\Delta t$ that is mutually inconsistent, would satisfy all three
  per-point constraints at every individual point and still be accepted
  in full. Fixing this would require a fundamentally different
  mechanism (e.g. a joint constraint evaluated over a spatiotemporal
  neighborhood, which rejection sampling in this per-point form cannot
  express) — explicitly out of scope for what is specified here.
- **The rejection-vs-copula-fix decision is live, not settled.** Two
  architectural paths are on the table:

  | | Post-hoc rejection sampling (§4, chosen for now) | Fix the marginal/copula model (t-copula tail dependence, or GPD tails via extreme value theory) |
  |---|---|---|
  | Pro | Theoretically clean (exact conditional samples, §2.4); doesn't touch the fitted model; fast to implement and iterate | Root-cause — addresses *why* inconsistent draws occur; a t-copula's extra tail-dependence parameter (or GPD tails) might also help the related storm-co-occurrence tail-dependence problem noted in the copula document |
  | Con | Symptom-level — does not fix the underlying Gaussian-copula tail-dependence limitation that produces inconsistent draws in the first place; wastes sampling budget when the acceptance rate is low (§2.4); structurally direction-blind and cross-point-blind (above) | Bigger, riskier change to a fitted model; no exact per-sample guarantee analogous to rejection sampling's — a t-copula/GPD refit changes the model's whole joint distribution, and its own tail behavior would itself need validation |

  The team has chosen path (i), post-hoc rejection sampling, **for now** —
  it is fast, safe, and exact given whatever model currently exists. Path
  (ii) remains the more root-cause-oriented alternative and should be
  revisited; this document does not resolve which is ultimately correct
  for the project.
- **Be honest that rejection sampling masks, rather than fixes, a
  poorly-calibrated model.** A high rejection rate at some region/season
  is not primarily a performance problem to be tuned away with a bigger
  retry budget — it is evidence the copula is a poor fit for the true
  joint physics there. This specification's diagnostics (§5) are designed
  precisely so that fact stays visible instead of being buried inside an
  opaque accept/reject filter.
