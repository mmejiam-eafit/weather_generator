# Open Questions Across the Pipeline Documentation

**Provenance.** This document consolidates every open question raised across
the eight sibling study guides in `docs/` (`variable_taxonomy.md`,
`tucker_decomposition.md`, `dynamic_mode_decomposition.md`,
`ornstein_uhlenbeck_processes.md`, `gaussian_copula_and_marginals.md`,
`harmonic_climatology.md`, `physical_consistency_constraints.md`,
`correlated_variables.md`). Each sibling's own "Open questions" section was
read in full to build this list — nothing here is invented independently of
that set, though several items are pushed further than their source
document pushed them, using reasoning available from *reading the set as a
whole* that no single document had access to on its own.

**How to read the status tags:**

- **OPEN** — genuinely unresolved by any document in this set; no answer is
  given here beyond what was already on the table, possibly with a
  practical mitigation noted.
- **ANSWERED** — this consolidation resolves the question, either by
  pointing at a sibling document that already answers it (sometimes without
  that document realizing it was answering a question raised elsewhere), or
  by deriving a new answer directly in this document.
- **PARTIALLY ANSWERED** — a real answer now exists, but it does not fully
  close the question; the residual piece is stated explicitly.

Nothing in this document authorizes a code change by itself — these are
still specification-level documents, not implementations.

---

## 1. Data and convention-level questions

### 1.1 Is the ocean grid's row-0 latitude/orientation south-first, like the wave and wind grids? — **OPEN**

Raised in: `variable_taxonomy.md` §6.

`core/config.py` deliberately leaves the ocean grid's row-0 latitude unset.
Wave and wind grids were confirmed row-0-south via a coastline/land-mask
correlation study; the ocean grid has not had the equivalent study run
against real data. `notebooks/verify_direction_convention.ipynb` uses a
cheaper proxy (zonal-mean `sst` maximum for the equator row, open-ocean
fraction at 40–60° for hemisphere sense) sufficient only to support a few
direction-diagnostic latitude bands, not a full verification.

This cannot be resolved by documentation — it requires running the same
kind of empirical correlation study against real archive data that settled
the wave/wind case. **Remains open**, and no `sst`/`ocs`/`ocd` field should
be treated as geo-referenced until it is.

### 1.2 Is `pwd`/`pswd`'s "coming_from" convention correct? — **OPEN**

Raised in: `variable_taxonomy.md` §6, `correlated_variables.md` §3.

Only `wad`/`wad_come_from` (wind) and `wsd` (wind-sea) have been checked
against known persistent wind/current patterns (trade winds, Southern Ocean
westerlies, major equatorial currents) — a check that works because those
fields track *local*, climatologically-persistent forcing. `pwd` (peak
wave) and `pswd` (primary swell) describe wave partitions that can be swell
arriving from a distant, decoupled storm system, so the same climatology
check is not valid ground truth for them. They share `wsd`'s origin code by
analogy only.

This is an empirical question about the archive's encoding, not a
theoretical one — **remains open** pending a domain-expert check or an
equivalent empirical study (e.g. checking `pwd`/`pswd` against a case where
the wave field is known, from independent synoptic data, to be locally
wind-generated rather than swell-dominated, so the trade-wind-style
consistency check becomes valid again for that specific case).

### 1.3 Is `wad` usable as a wind-direction field? — **ANSWERED** (already resolved; restated here for completeness)

Raised in: `variable_taxonomy.md` §6.

No — `wad` is confirmed to be a decoy: it decodes as a plausible
`coming_from` field but is actually `going_to`, exactly 180° from
`wad_come_from` at every checked cell. `wad_come_from` is the only correct
wind-direction source and should be the only one ever wired into any
pipeline stage. This was already a closed, confirmed finding when it was
written into `variable_taxonomy.md` — it is listed here only so this
document is a complete index, not because it is actually in question.

---

## 2. Representation-level questions (taxonomy, Tucker, Gaussianization)

### 2.1 Does `sigwh² ≈ wsh² + pswh²` require disjoint partitions, and is that assumption safe? — **ANSWERED**

Raised in: `variable_taxonomy.md` §6 ("`sigwh`'s RMS relationship to
partition heights assumes disjoint partitions").

Yes, the identity is exact only if the wind-sea/peak/swell spectral
partitioning is non-overlapping, and to the extent it isn't, the identity
is approximate. This is not actually a loose end — it is precisely the
situation `physical_consistency_constraints.md` §2.2 already characterizes
correctly: that document derives the energy-budget relation as an
approximate, tolerance-banded check (`|sigwh² − wsh² − pswh²| < atol`)
*because* real partitions have some spectral overlap and because
higher-order swell contributions are neglected — it never treats the
relation as an exact identity to begin with. **The two documents' concerns
are the same fact, viewed from two sides**: the taxonomy document flags the
assumption from the decomposition side, and the physical-consistency
document already builds the correct tolerance-based treatment for it from
the constraint side. No further resolution is needed; treat
`physical_consistency_constraints.md` §2.2 as this question's answer.

### 2.2 Does independent per-variable Gaussianization discard the physical coupling within a height/period/direction triplet? — **PARTIALLY ANSWERED**

Raised in: `variable_taxonomy.md` §6 ("flagged here because it originates in
the taxonomy, not the copula machinery... a concern for the Gaussian-copula
document to resolve").

`gaussian_copula_and_marginals.md` does not, in fact, resolve this — its own
open-questions section (§6) is about *where* Gaussianization sits in the
per-variable chain, not about *restoring cross-member coupling within one
triplet*. Those are different questions. The actual answer to *this*
question comes from `correlated_variables.md`, whose whole §3–§5 is a
specification for exactly this: grouping a triplet's members (e.g. `wsh`,
`wsp`, and the `wsd`-derived Cartesian pair) into one joint DMD state built
from their already-independently-Tucker-compressed coefficients, so that
the coupling discarded by independent per-variable, per-cell treatment
upstream (Gaussianization *and* Tucker *and* DMD, all per-variable) is
reintroduced at the one stage designed to model temporal/cross-member
dynamics jointly.

This is a real answer, not a deflection — but it is not a full closure:
`correlated_variables.md` §6 itself notes that **which** variables belong in
a joint group is an open modeling choice (see §3.2 below), so the
mechanism for recovering the coupling is now specified, while its exact
scope (how many groups, how large) is not.

### 2.3 Does low-rank Tucker reconstruction preserve `speed = hypot(u, v)` for a directional field? — **OPEN** (mitigation available)

Raised in: `variable_taxonomy.md` §6 ("Low-rank approximation does not
commute with `hypot`"), explicitly deferred to `tucker_decomposition.md`.

`tucker_decomposition.md` does not, in fact, address this anywhere in its
text (confirmed by reading its full "Open questions and limitations"
section — it covers cross-variable coupling, masked-HOOI convergence, and
rank selection, but never `hypot`/reconstruction-of-derived-magnitude). The
deferral was not honored. **This remains genuinely open**: if a partition's
height-scaled direction is decomposed as two independently-Tucker-compressed
tensors $u_{\text{dir}}, v_{\text{dir}}$, then
$\operatorname{hypot}\!\big(\widehat u_{\text{dir}}, \widehat
v_{\text{dir}}\big)$ (hats denoting the reconstructed, truncated-rank
tensors) is not in general equal to a directly, separately low-rank-compressed
$\widehat H$ — matrix/tensor truncation does not commute with a nonlinear
map like $\sqrt{u^2+v^2}$.

A concrete mitigation exists, though no document has adopted it: **decompose
the partition's height $H$ (or speed) as its own separate scalar tensor**, in
addition to $u,v$, rather than relying solely on reconstructing it via
`hypot`. This trades redundancy (the magnitude is now represented twice —
once directly, once implicitly via $u,v$) for exactness of the physically
important quantity, at the cost of needing some downstream reconciliation
if the two disagree after reconstruction. Whether that redundancy is worth
paying is an empirical/design tradeoff that has not been evaluated —
**status remains open**, now with a named candidate solution.

### 2.4 Does per-cell independent Gaussianization of $u$ and $v$ preserve `speed = hypot(u, v)`? — **OPEN** (distinct from 2.3)

Raised in: `gaussian_copula_and_marginals.md` §3 ("Known open subtlety").

This is a different failure mode from §2.3 above, occurring earlier in the
chain: even before any Tucker compression, $u$ and $v$ are each Gaussianized
via their own *separate* per-cell empirical CDF — two different monotone
nonlinear regradings. The exact identity $speed=\operatorname{hypot}(u,v)$
does not survive into Gaussianized $(Z_u, Z_v)$-space, and there is no
guarantee the reconstructed $\operatorname{hypot}(u,v)$ after inverse
transform matches whatever speed marginal was actually fit for that
partition. `gaussian_copula_and_marginals.md` correctly identifies this as
an open empirical question and does not resolve it, nor does any other
sibling document. **Remains open.** Note it compounds with §2.3: even a
perfectly rank-preserving Tucker/DMD stage would not undo distortion already
introduced at the Gaussianization step, so the two issues would need
independent empirical characterization, not a single combined fix.

---

## 3. Cross-variable coupling

### 3.1 What mechanism captures cross-variable dynamical coupling, given that Tucker and DMD are now strictly per-variable? — **ANSWERED**

Raised in: `tucker_decomposition.md` §4.4/§6, `dynamic_mode_decomposition.md`
§6 (both explicitly declined to propose a mechanism and flagged this as
open).

`correlated_variables.md` answers this directly: identify physically
justified correlated-variable groups (e.g. `{was, wsh, wsp}`,
`{wad_come_from, wsd}`), and for each group, concatenate the members'
*already independently Tucker-compressed* per-variable coefficient
sequences into one joint state before running DMD's snapshot-pair
regression on that group — never touching or re-raveling the original
per-variable spatial `(lat, lon, time)` tensors, so this does not
reintroduce the four-dimensional-stacking mistake the tensor-representation
rule forbids. The physical grounding is the shared wave-action balance
equation (`correlated_variables.md` §2): wind and wind-sea variables are
coupled through a real, shared governing PDE, not merely correlated by
coincidence, which is exactly the kind of structure a joint (rather than
independent) dynamical fit is suited to capture — a jointly-fit linear
operator can represent genuine lead-lag structure (e.g. a rising `was` mode
predicting future `wsh` growth) that two independent per-variable DMD fits
cannot represent at all, since each only sees its own autocorrelation.

### 3.2 Which variables belong in a joint group, and how many groups are needed? — **OPEN**

Raised in: `correlated_variables.md` §6, explicitly described there as
"mirrors, and narrows without closing" the question in §3.1.

No general algorithm for group membership is given anywhere. The tradeoff
is real: growing a group (e.g. adding the peak-wave triplet to capture
$S_{nl}$'s energy transfer beyond the wind-sea partition) increases the
joint state's dimension and hence the DMD rank and data requirements,
risking an underdetermined or overfit joint operator for a fixed training
window; keeping groups small and conservative avoids that risk but
reproduces the independent-fit blind spot for whatever coupling falls
outside the chosen group. **Remains open** — a modeling/hyperparameter
choice to be made empirically (e.g. by checking, per §5 of
`correlated_variables.md`, whether a candidate group's jointly-fit operator
actually shows non-negligible off-diagonal coupling terms; if it doesn't,
the group was drawn too widely for no benefit).

### 3.3 Is the wind-direction/wind-sea-direction alignment equilibrium a stable node, as the scalar energy case is? — **ANSWERED HERE**

Raised in: `correlated_variables.md` §3.2 and §6 ("§3.2's directional-alignment
ODE is illustrative, not fully stability-analyzed... left open").

This can be answered directly from the vector ODE `correlated_variables.md`
§3.2 already writes down:

$$
\dot{\mathbf D} = \kappa\, U\,|\mathbf D|\, \hat{\mathbf U}_{\text{vec}} - c_{ds}\,|\mathbf D|^{1/2}\,\mathbf D,
\qquad \hat{\mathbf U}_{\text{vec}} = \mathbf U_{\text{vec}}/U .
$$

Write $\mathbf D = r(\cos\theta,\sin\theta)$ in polar coordinates relative
to the (quasi-static) wind bearing, so $\theta$ is the angular offset
between wind-sea direction and wind direction and $r = |\mathbf D| = H_s$.
Projecting $\dot{\mathbf D}$ onto the radial and tangential unit vectors
gives, after simplification,

$$
\dot r = \kappa U r\cos\theta - c_{ds}r^{3/2}, \qquad
\dot\theta = -\kappa U \sin\theta .
$$

The angular equation decouples completely from $r$ — it is exactly the
classical overdamped-pendulum / Adler phase equation. Its fixed points are
$\theta=0$ (aligned) and $\theta=\pi$ (anti-aligned); linearizing,
$\dot\theta\approx-\kappa U\,\theta$ near $\theta=0$ (stable, decay rate
$\kappa U$) and $\dot\epsilon\approx+\kappa U\,\epsilon$ near $\theta=\pi+\epsilon$
(unstable). So $\theta=0$ attracts **every** initial condition except the
single unstable equilibrium at exactly $\theta=\pi$ — wind-sea direction
aligns with wind direction from any starting offset, matching the physical
picture $S_{in}$'s directional peaking already motivates.

At $\theta=0$, the radial equation reduces to exactly the scalar energy
ODE of §2.3–§2.4, with fixed point $r^\* = (\kappa U/c_{ds})^2$ and
linearized rate $-\tfrac12 \kappa U$ there (same computation as the scalar
case). Because $\partial\dot\theta/\partial r \equiv 0$ and
$\partial\dot r/\partial\theta|_{\theta=0}=-\kappa U r\sin\theta|_{\theta=0}=0$,
the full 2D Jacobian at $(\theta,r)=(0,r^\*)$ is **diagonal**, with real
eigenvalues $-\kappa U$ (angular) and $-\tfrac12\kappa U$ (radial). Both
negative and real: **the aligned equilibrium is a stable node**, exactly as
the scalar case is — and angular relaxation is exactly twice as fast as
radial relaxation ($|\lambda_\theta| = 2\,|\lambda_r|$), a clean, falsifiable
quantitative prediction of this reduction worth checking against real
fetch-limited growth data if this proposal is ever implemented.

This inherits the same caveat the scalar analysis already carries (§2.4 of
`correlated_variables.md`): it assumes $U$ (and hence the wind bearing) is
quasi-static relative to the alignment relaxation timescale, consistent
with how that document already treats $U$'s own dynamics elsewhere.

---

## 4. Gaussianization placement

### 4.1 Where, along each variable's independent processing chain, should Gaussianization happen? — **OPEN**

Raised in: `gaussian_copula_and_marginals.md` §6.1 (the deepest treatment),
echoed as a dependency in `tucker_decomposition.md`, `dynamic_mode_decomposition.md`,
and `ornstein_uhlenbeck_processes.md` — all four assume this is settled
upstream and explicitly decline to settle it themselves.

Three placements remain live and are laid out with a full pro/con table in
`gaussian_copula_and_marginals.md` §6.1: (i) on raw per-cell physical
anomalies (current default; maximal interpretability, most LUTs, most
expensive); (ii) on that variable's own Tucker-core coefficients (fewer
LUTs, but "per-cell" stops meaning a physical grid location); (iii) on that
variable's DMD mode amplitudes/residuals (most directly serves the
OU-innovation Gaussianity requirement, but by then two prior linear
compressions have already run on non-Gaussianized data, and complex-mode
Gaussianization reintroduces the Class-C direction-handling problem in a
new guise). **This is the single most consequential open question in the
whole documentation set** — it is a live design decision requiring empirical
validation (comparing physical-space QQ-plots and reconstructed extreme
statistics under each placement) that has not been performed by any
document here, and no amount of further documentation-level reasoning
closes it; it needs to be settled empirically, once, against real training
data.

### 4.2 What copula family should be used — Gaussian, or one with tail dependence? — **OPEN** (distinct axis from 4.1)

Raised in: `gaussian_copula_and_marginals.md` §6.2, `physical_consistency_constraints.md`
§6.

Independently of *where* Gaussianization sits (§4.1), there is a second,
orthogonal open question: *what kind* of copula to use at all. The current
design's implicit Gaussian copula has provably zero upper-tail dependence
($\lambda_U=0$ for $\rho<1$), which both documents identify as the root
cause of the physical-consistency violations the rejection-sampling layer
(`physical_consistency_constraints.md`) exists to patch after the fact. A
$t$-copula (tunable $\lambda_U>0$ via one degrees-of-freedom parameter) or
splicing Generalized Pareto Distribution tails onto each cell's empirical
CDF are both named as root-cause alternatives. **Neither document resolves
which to use, nor whether to do so instead of, or in addition to,
rejection sampling.** A $t$-copula/GPD-tail construction could in principle
be placed at any of the three §4.1 locations, so resolving §4.1 does not
resolve this, and vice versa — they are genuinely independent axes of the
same design space, and both remain open.

---

## 5. DMD-specific questions

### 5.1 Should DMD vectorize each Tucker-core time slice into a state vector (classical exact DMD), or use a tensor-native formulation? — **ANSWERED** (recommendation)

Raised in: `dynamic_mode_decomposition.md` §4.2/§6, which explicitly declines
to take a position, calling both "legitimate."

Reading the reasoning already laid out in that document, a default is
defensible: **use the narrow reshape** — vectorize each `(r_lat, r_lon)`
Tucker-core time slice into a single column and proceed with classical
exact DMD. The document's own analysis already establishes this is *not*
the forbidden representational flattening (it reshapes an already-compressed
coefficient grid whose spatial adjacency was captured separately by
Tucker's own factor matrices, not the raw physical grid), so there is no
principled reason tied to this documentation set's core tensor-representation
concern to avoid it. Tensor-native (e.g. higher-order) DMD variants remain
a legitimate escalation path if empirical evidence later shows the
vectorized fit loses meaningful structure — consistent with this repo's
general pattern of an escalation ladder (plain method first, fancier
variant only once a concrete deficiency is observed) rather than a reason
to default to the more complex option up front. **Recommendation, not a
closed empirical fact** — flagged as such.

### 5.2 Should conjugate-paired DMD/OU modes be sampled independently, or forced to satisfy the exact conjugate relationship at every simulated step? — **ANSWERED IN PRINCIPLE**, procedure still open

Raised in: `dynamic_mode_decomposition.md` §2.4/§6, `ornstein_uhlenbeck_processes.md`
§6.

The *principle* is not actually in doubt once stated precisely: a real
physical field requires $\eta_{k'}(t) = \overline{\eta_k(t)}$ exactly, not
merely in distribution — independently sampling two conjugate-paired modes'
stochastic residuals will, with probability 1, produce a complex-valued
reconstructed field, which is unphysical. So the two modes must be treated
as **one shared stochastic driver**: fit once, simulate one member of the
pair, obtain the other by conjugation — not as two independently-fit,
independently-simulated processes. Both sibling documents already lean
toward this conclusion without quite stating it as a decision. **What
remains genuinely open is the specific fitting/simulation procedure** for
that shared-driver treatment (exactly how the single complex OU fit is
parameterized and sampled such that conjugation is exact rather than
approximate) — that implementation detail is left to whatever code
performs sampling.

### 5.3 How should Tucker ranks $(r_{\text{lat}}, r_{\text{lon}}, r_{\text{time}})$ and DMD rank $r_{\text{dmd}}$ be chosen? — **OPEN**

Raised in: `tucker_decomposition.md` §6, `dynamic_mode_decomposition.md` §6
(the same underlying issue, applied to two different decompositions).

Both documents describe only a diagnostic (elbow of the reconstruction-error
curve for Tucker; eigenvalue-spectrum/growth-rate plausibility for DMD), not
an automatic stopping rule. **Remains open** as a per-variable (and, per
§3.2 above, per-group) judgment call to be made and re-checked by whoever
trains the model — no document proposes a data-driven automatic rank
selector, and none is proposed here either, since doing so responsibly
requires empirical cross-validation this documentation-only exercise cannot
perform.

### 5.4 Is masked-HOOI's imputation of land cells guaranteed to converge well? — **OPEN** (mitigation available)

Raised in: `tucker_decomposition.md` §6.

No — masked HOOI converges monotonically to a stationary point of a
non-convex objective, not provably to the global optimum, and large
contiguous unobserved regions (large landmasses with little interior
coastline to anchor the imputation) are a realistic case for a poor local
completion. This is a genuine numerical limitation, not just a theoretical
caveat. **Remains open** as a property of the method, but the standard
numerical-optimization mitigation applies and is worth stating explicitly
since no document does: run masked HOOI from multiple random initializations
and compare reconstruction error/imputed-region stability across them,
flagging variables or regions where the imputation is sensitive to
initialization for closer inspection, rather than trusting a single run
blindly.

### 5.5 What should happen if a fitted DMD mode is genuinely growing ($|\lambda_k| > 1$) rather than just noisy? — **OPEN** (mitigation available)

Raised in: `dynamic_mode_decomposition.md` §6.

If DMD's deterministic backbone drives long-horizon stochastic simulation,
a genuinely growing mode diverges unboundedly. No document specifies
detect-and-handle logic. **Remains open**, but a mitigation consistent with
this documentation set's own established pattern (`physical_consistency_constraints.md`'s
rejection-sampling philosophy) is a reasonable default worth naming: clip
$|\lambda_k|$ to $1$ at simulation time for modes exceeding it by a small,
plausibly-noise-attributable margin, but flag (rather than silently clip)
any mode whose growth exceeds a larger threshold for manual review — the
same "detect, don't silently absorb" philosophy the physical-consistency
document already applies to individual samples, applied here to a fitted
model artifact instead.

---

## 6. Ornstein–Uhlenbeck-specific questions

### 6.1 Is the local-residual variance-ratio floor a physically sound cutoff, or just a reasonable-looking default? — **OPEN**

Raised in: `ornstein_uhlenbeck_processes.md` §6.

The floor is grounded in an efficiency argument (don't bother simulating a
cell/variable whose local residual explains negligible variance), which is
sound as far as it goes, but the actual downstream error its omission
introduces has not been validated end to end — that validation depends on
simulation and hurricane-season-style checks that are not yet in place
anywhere in this repo. **Remains open**; treat any specific numeric
threshold as provisional until such a validation exists.

---

## 7. Harmonic climatology's quasi-stationarity assumptions

### 7.1 Is a fixed harmonic shape plus one linear trend, held constant over the whole fitting window, an adequate model of the climatological background? — **OPEN**

Raised in: `harmonic_climatology.md` §6 (three related bullets, consolidated
here as one question with three facets, since they are the same underlying
assumption viewed from different angles).

Three linked facts, none resolved by any document: (a) real seasonal cycles'
amplitude and phase can themselves drift over long records (e.g. from
decadal ocean-atmosphere oscillations), which a fixed-shape harmonic design
cannot represent — any such drift is folded into either the harmonic
coefficients as a compromise average or the residual as spurious
low-frequency "noise"; (b) a single linear trend is itself only the
best-fit constant rate over the specific window used, and says nothing
about whether that rate generalizes or is itself the locally-linear piece
of something curved; (c) because the trend coefficient is defined relative
to whichever window it is fit on, two fits over different windows of the
same record are not simply noisier/cleaner estimates of one true number —
they can be honest answers to subtly different questions (a short window
can have interannual variability, e.g. an ENSO event, alias into an
apparent trend that a longer window would average away).

**Remains open.** Point (c) is not purely a climatology-document concern —
it connects directly to a broader, still-open decision elsewhere in this
project (which single archive era, or subset of the archive, to actually
train the production model on — a decision this documentation set does not
touch and does not resolve). Any reported trend coefficient should be read
together with the specific window it was estimated over, not treated as a
window-independent physical constant, until that broader training-window
decision is made.

### 7.2 Is the "waves have no diurnal signal" reasoning resolution-dependent? — **OPEN**, conditionally

Raised in: `harmonic_climatology.md` §6.

Yes, explicitly and narrowly: the claim is stated relative to typical
global-model resolution and forcing data (GFS-Wave/WAVEWATCH III-class
products). It should be revisited only if this pipeline is ever pointed at
a higher-resolution regional wave model, where locally wind-driven
short-fetch seas forced by diurnal sea/land-breeze cycles could carry a
weak but real diurnal signature. **Not an active concern at the current
archive's resolution** — listed here for completeness, not because it
needs attention now.

---

## 8. Physical-consistency constraints

### 8.1 Should the physical-consistency constraints check direction, not just magnitude? — **PARTIALLY ANSWERED**

Raised in: `physical_consistency_constraints.md` §6 ("direction-blind scope
gap... a candidate future extension once a defensible veer/backing
tolerance is derived and validated").

`correlated_variables.md` §3.2 (and §3.3 of this document) now supplies
exactly the physical grounding that was missing: the wind/wind-sea
directional-alignment system has a derived, stable equilibrium at zero
angular offset, with an explicit relaxation rate $\kappa U$ shared with the
magnitude-growth dynamics. This licenses a concrete **candidate constraint,
proposed here for the first time**: flag as directionally inconsistent any
query point where the angular offset $|\theta| = |\angle(\mathbf D) -
\angle(\mathbf U_{\text{vec}})|$ between wind-sea direction and wind
direction exceeds a tolerance $\delta_{\text{dir}}$ — **unless** the local
sea state is young/duration-limited (small $r/r^\*$, i.e. wave age not yet
saturated), exactly mirroring the existing energy-budget constraint's
young-sea exception, since §3.3's derivation shows magnitude and direction
relax on the same shared timescale.

This is a real, physically-grounded proposal, not just a restatement of the
gap — but it is not a closed answer: $\delta_{\text{dir}}$ still needs
empirical calibration against real archive data exactly the way the
existing three constraints' tolerances were calibrated
(`physical_consistency_constraints.md` §5), and the "unless young" exception
requires the same recent-wind-history state that §8.4 below notes is
missing from the current per-query-point-independent design. **Status:
the missing mechanism is now specified; the missing calibration and the
missing state-tracking are not.**

### 8.2 Can a per-query-point-independent check ever catch cross-cell or cross-time physical inconsistency? — **OPEN**, and structurally so

Raised in: `physical_consistency_constraints.md` §6.

No, and this is not a gap to be patched incrementally — it is a structural
property of the rejection-sampling design as specified. A storm's wave
field advecting across adjacent cells at an implausible speed, or a field
at $t$ and $t+\Delta t$ that is mutually inconsistent, would satisfy every
per-point constraint individually and be accepted in full. Fixing this
requires a fundamentally different mechanism (a joint constraint evaluated
over a spatiotemporal neighborhood), which no document in this set
specifies. **Remains open and explicitly out of scope** for the
per-point design as currently specified — not a defect to silently accept,
but also not something to be resolved by refining tolerances.

### 8.3 Rejection sampling, or fix the copula? — **OPEN** (same axis as §4.2)

Raised in: `physical_consistency_constraints.md` §6.

Restated here only to make the cross-reference explicit: this is the same
open question as §4.2 above, viewed from the constraints side rather than
the copula side. The team has chosen rejection sampling for now (fast,
safe, exact given whatever model exists); fixing the copula's tail
dependence remains the more root-cause-oriented alternative. **Remains
open** — see §4.2 for the full framing; resolving it requires the same
empirical work either document alone can specify but not perform.

### 8.4 Does a static Pierson-Moskowitz check misfire on physically valid transient ("young") sea states? — **ANSWERED** (proposal exists; not yet incorporated)

Raised in: `correlated_variables.md` §2.5 and §6, addressed *to*
`physical_consistency_constraints.md`.

Yes — `correlated_variables.md` §2.5 derives this directly from its own
reduced ODE: a duration-limited sea genuinely sitting below the PM
equilibrium curve is a physically valid transient, not necessarily an
inconsistent draw, and a static per-point tolerance-band check has no way
to distinguish "transient-but-valid" from "genuinely inconsistent," since it
only examines one instant. Two concrete refinements are proposed there: a
wave-age- or duration-aware tolerance, or a trajectory check against the
closed-form growth curve rather than only its fixed point. **The answer
exists and is specific** — but both refinements require recent-wind-history
state that `physical_consistency_constraints.md`'s current
per-query-point-independent design does not carry, so implementing either
is an architectural extension of that document's scope, not a parameter
tweak. This consolidation treats the *question* as answered (yes, it
misfires, and here is a specific fix) while flagging that the *code-level
integration* of that fix is still to be done by whoever next edits
`physical_consistency_constraints.md`.

---

## 9. Explicitly out of scope (not open questions — scope boundaries stated by design)

These are not unresolved questions so much as places where a sibling
document deliberately drew a boundary. Listed here only to avoid them being
mistaken for open items that need chasing:

- **`correlated_variables.md`'s reduced ODE system is a pedagogical
  simplification**, not a claim that WAVEWATCH III literally integrates a
  two-equation system. It discards the full frequency-direction spectral
  shape, holds the wind-input coefficient fixed rather than wave-age
  dependent, omits $S_{nl}$'s self-interaction, and treats wind's own
  dynamics as trivial relaxation. This is intentional scope-narrowing for
  pedagogical clarity, not an oversight.
- **`physical_consistency_constraints.md`'s three constraints are
  deliberately narrow** (wave-energy budget, fetch-limited
  Pierson-Moskowitz, peak-period bounds) — "physical consistency" here
  means exactly these three checks, not a general physical-validity
  guarantee. Additional constraints are explicitly deferred to a future
  revision, not silently assumed absent.

---

## Summary table

| # | Question | Status |
|---|---|---|
| 1.1 | Ocean-grid orientation | OPEN |
| 1.2 | `pwd`/`pswd` convention | OPEN |
| 1.3 | `wad` usability | ANSWERED (no) |
| 2.1 | `sigwh` disjoint-partition assumption | ANSWERED |
| 2.2 | Triplet coupling lost by independent Gaussianization | PARTIALLY ANSWERED |
| 2.3 | `hypot` vs. low-rank Tucker | OPEN (mitigation proposed) |
| 2.4 | `hypot` vs. independent Gaussianization | OPEN |
| 3.1 | Cross-variable coupling mechanism | ANSWERED |
| 3.2 | Joint-group selection | OPEN |
| 3.3 | Directional-alignment equilibrium stability | ANSWERED HERE |
| 4.1 | Gaussianization placement | OPEN (the central question) |
| 4.2 | Copula family (Gaussian vs. t/GPD) | OPEN |
| 5.1 | DMD state-vectorization | ANSWERED (recommendation) |
| 5.2 | Conjugate-pair sampling | ANSWERED IN PRINCIPLE, procedure open |
| 5.3 | Tucker/DMD rank selection | OPEN |
| 5.4 | Masked-HOOI convergence | OPEN (mitigation proposed) |
| 5.5 | Growing DMD modes | OPEN (mitigation proposed) |
| 6.1 | OU variance-ratio floor | OPEN |
| 7.1 | Climatology quasi-stationarity | OPEN |
| 7.2 | Diurnal reasoning resolution-dependence | OPEN (conditional, inactive) |
| 8.1 | Direction-blind physical constraints | PARTIALLY ANSWERED |
| 8.2 | Cross-cell/cross-time blindness | OPEN (structural) |
| 8.3 | Rejection vs. copula fix | OPEN (= 4.2) |
| 8.4 | Static PM check vs. young-sea transients | ANSWERED (not yet integrated) |
