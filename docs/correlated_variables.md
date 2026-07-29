# Correlated Variables: Dynamical Coupling Behind the Correlations

A graduate-level treatment of *why* several variables in this archive are
correlated: not as an incidental statistical fact to be patched over with a
richer copula, but as the observable shadow of a single governing PDE —
WAVEWATCH III's spectral wave-action balance equation — acting on wind and
wave-state fields simultaneously. This document derives that claim
properly, reduces it to a small, honest, worked ODE system explicitly
analogized to Lotka–Volterra predator–prey dynamics, and turns the result
into two concrete specifications: a sharpened reading of the
fetch-limited Pierson–Moskowitz constraint in `physical_consistency_constraints.md`,
and a joint-compressed-state DMD proposal that answers the open
cross-variable-coupling question left standing by `tucker_decomposition.md`
and `dynamic_mode_decomposition.md`.

---

## 0. Conventions this document follows, and one subtlety this document adds

This is a new document, not a rewrite of a deleted flawed draft — there is
no earlier version to correct. It inherits two conventions already
established by its six sibling documents, and states one additional
subtlety needed because its whole subject is cross-variable structure.

**Convention 1 — prescriptive, not descriptive.** Like every sibling
document, this one specifies what a correctly-designed pipeline *should*
do, not what `src/` currently does. There is no code anywhere in this
repository today that fits coupled cross-variable dynamics, joint DMD
states, or a dynamical (as opposed to static, per-point) physical-
consistency check. Everything from §2 onward — the reduced ODE system, the
joint-DMD state construction, the growth-curve-aware consistency check —
is a specification for something to be built, argued for on first
principles, not a report on existing behavior.

**Convention 2 — tensor representation.** Per `variable_taxonomy.md` and
`tucker_decomposition.md`: every physical variable is its own independent
3D `(lat, lon, time)` tensor. Variables are never stacked into a joint 4D
`(lat, lon, time, variable)` tensor, and no variable's tensor is ever
flattened to 2D for any decomposition purpose. This document does not
re-derive why (the taxonomy and Tucker documents already do, at length);
it only uses the result.

**The one subtlety this document adds — a joint state built from
already-compressed coefficients is not the forbidden 4D stack.** §4
proposes concatenating several *already independently Tucker-compressed*
variables' coefficient sequences into one joint state vector, for the sole
purpose of a joint DMD fit. Read carelessly, "concatenate several
variables into one combined state" can sound exactly like the forbidden
operation Convention 2 rules out. It is not, and the distinction is worth
stating with precision rather than asserting by analogy:

- Each variable in a proposed group still gets its **own, independent**
  Tucker fit, on its **own** raw `(lat, lon, time)` tensor, exactly as
  `tucker_decomposition.md` §4.1 specifies — no shared core, no shared
  factor matrices, nothing about the Tucker stage changes. The output of
  that stage, per variable $v$, is a small core $C^{(v)} \in
  \mathbb{R}^{r_{\text{lat}}^{(v)} \times r_{\text{lon}}^{(v)} \times T}$ —
  tens or hundreds of numbers per timestep, not a `(721,1440)` grid.
- `dynamic_mode_decomposition.md` §4.2 already establishes, for a *single*
  variable, that reaching classical DMD's snapshot-matrix form requires
  vectorizing each timestep's compressed `(r_lat, r_lon)` coefficient grid
  into one column, $x_t^{(v)} = \mathrm{vec}(C^{(v)}_{:,:,t}) \in
  \mathbb{R}^{r_{\text{lat}}^{(v)} r_{\text{lon}}^{(v)}}$ — and that this
  reshape is a *narrower, different* operation than the forbidden raw-grid
  flattening, precisely because the object being reshaped is already a
  compressed coefficient grid whose spatial adjacency has already been
  captured by Tucker's own factor matrices, not the original physical grid.
- §4 below takes exactly that already-permitted per-variable vector
  $x_t^{(v)}$ — never the raw tensor, never anything upstream of the
  variable's own completed Tucker fit — and, for a deliberately chosen,
  small group of variables $\mathcal V = \{v_1,\dots,v_g\}$ believed to be
  dynamically coupled, concatenates their vectors into one joint state
  $x_t^{\mathcal V} = (x_t^{(v_1)\top}, \dots, x_t^{(v_g)\top})^\top$.

The concatenation happens **after** every group member's independent
Tucker fit is already complete, operates on **small, already-compressed**
numeric vectors (total dimension the sum of a handful of small per-variable
ranks, not the sum of `(lat, lon)` grid sizes), and never re-ravels,
re-touches, or merges the original spatial tensors in any way. It is a
deliberate, small-scale, post-compression operation on coefficient
sequences — precisely the "some other mechanism operating on the outputs
of these per-variable Tucker fits" that `tucker_decomposition.md` §4.4
explicitly left open, and precisely the "coupled/multi-variable DMD
variant" that `dynamic_mode_decomposition.md` §6 explicitly flagged as the
legitimate way cross-variable structure could eventually be captured. It
is not, and should never be read as, a backdoor reintroduction of Error 2's
forbidden `(lat, lon, time, variable)` tensor.

---

## 1. Summary

Several variables in this archive are correlated not because two
independent stochastic processes happen to co-vary, but because they are
different **functionals of one shared physical field evolving under one
governing PDE**: the spectral wave-action balance equation that
WAVEWATCH III integrates. Wind speed/direction and wind-sea
height/direction/period are not independently-drawable quantities that a
sufficiently expressive copula could someday learn to correlate correctly
— wind *forces* wave growth through a term that is structurally
multiplicative in the current wind state and the current wave state, wave
steepness *dissipates* wave energy through a term that is nonlinear in the
wave state alone, and nonlinear wave-wave interactions redistribute energy
*within* the spectrum in a way that is cubic in the wave state itself. This
is dynamical coupling — a shared right-hand side, not a shared prior.

§2 derives this claim at two levels: qualitatively, from the real spectral
balance equation and its source terms (§2.1); and rigorously, via a small,
deliberately simplified two-variable ODE system for bulk wind speed and
bulk wind-sea energy (§2.2–§2.4), built by direct structural analogy to
classical Lotka–Volterra predator–prey dynamics, whose fixed point is shown
to reproduce the fetch-limited Pierson–Moskowitz relation from
`physical_consistency_constraints.md` **exactly**, and whose stability is
analyzed honestly (a damped, monotonic approach to equilibrium — a stable
node — in contrast to Lotka–Volterra's neutrally-stable center). §3 gives
two further worked examples grounded in real mechanisms. §4–§5 turn the
result into a concrete specification: identify physically-justified
correlated-variable groups and fit DMD jointly across each group's
already-Tucker-compressed coefficients, which is the first mechanism in
this documentation set actually capable of representing cross-variable
predictive (lead–lag) structure. §6 states honestly what remains open.

---

## 2. Theoretical foundation

### 2.1 The wave-action balance equation: why wind and wave-state variables are coupled, not just correlated

WAVEWATCH III integrates a balance equation for the **wave action density**
$N(f,\theta,\mathbf x,t) := S(f,\theta,\mathbf x,t)/\sigma$, where $S$ is
the directional wave spectrum of `variable_taxonomy.md` §2.2 and
$\sigma = 2\pi f$ is the intrinsic (relative-to-current) radian frequency.
Action, not energy, is the correct conserved density here because a wave
field propagating through a slowly varying current does work against (or
receives work from) that current — a classical geometric-optics/WKB result
— so it is $N$, not $S$ itself, that is materially conserved along a wave
ray in the absence of sources:

$$
\frac{\partial N}{\partial t} + \nabla_{\mathbf x}\cdot(\dot{\mathbf x}N)
+ \frac{\partial}{\partial f}(\dot f N) + \frac{\partial}{\partial\theta}(\dot\theta N)
= \frac{S_{\text{tot}}}{\sigma}, \qquad S_{\text{tot}} = S_{in} + S_{nl} + S_{ds} + \cdots
$$

The left-hand side describes **conservative transport** of existing wave
action through physical space (advection at the group velocity, refracted
by currents and depth), through frequency space ($\dot f$, shifted by
spatial/temporal gradients in depth or current — shoaling and blocking),
and through direction space ($\dot\theta$, refraction bending wave rays
around bathymetry and currents). None of that transport term creates or
destroys energy; it only moves existing wave action around. All generation,
redistribution-within-spectrum, and dissipation physics — the part that
actually couples wind to wave state — lives entirely on the right-hand
side, in $S_{\text{tot}}$, divided by $\sigma$ purely to convert an
energy-flux source into the action-flux units the left side is stated in.

**$S_{in}$ — wind input, and why it is a genuine multiplicative coupling.**
Two classical mechanisms combine here. Phillips' (1957) resonance
mechanism is a *linear*, additive seeding process: turbulent pressure
fluctuations in the airflow occasionally resonate with a still-negligible
sea-surface perturbation, injecting a small amount of energy essentially
independent of how much wave energy is already present — dominant only for
a genuinely calm sea with no established spectrum yet. Miles' (1957)
critical-layer mechanism dominates thereafter: it is a linear
shear-flow instability of the wind profile at the height where the mean
wind speed matches the wave's phase speed, and it draws energy from the
airflow at a rate **proportional to the wave energy already present**,

$$
S_{in}(f,\theta) \;=\; \beta\!\left(\frac{U}{c(f)},\, \theta - \theta_{\text{wind}}\right)\, \sigma \, S(f,\theta),
$$

with $\beta$ a wind-input growth-rate coefficient depending on the local
wave age $U/c(f)$ (wind speed relative to the wave's phase speed — young,
slow waves are pumped harder than a wave train already moving nearly as
fast as the wind) and on the angular offset between the wave component's
direction and the wind direction (peaked at zero offset — waves traveling
with the wind are pumped harder than waves crossing or opposing it). This
is the mathematical origin of the "self-referencing" claim this document
makes precise below: $S_{in}$ is literally a **product** of a
wind-dependent coefficient and the current spectral density itself — the
rate at which wave energy grows depends jointly, multiplicatively, on the
current wind state *and* the current wave state, structurally identical in
kind to Lotka–Volterra's $xy$ predation term (§2.2). The $\beta(\cdot,
\theta-\theta_{\text{wind}})$ directional dependence, peaked at the wind
direction, is also the physical mechanism behind §3.2's worked example
(wind-sea direction tracking wind direction).

**$S_{nl}$ — nonlinear four-wave resonant transfer, and why it is
self-referencing within the spectrum itself.** Hasselmann (1962) showed
that weakly nonlinear interactions among quartets of wave components
satisfying the resonance conditions $\mathbf k_1+\mathbf k_2 = \mathbf
k_3+\mathbf k_4$, $\omega_1+\omega_2=\omega_3+\omega_4$ transfer energy
between frequencies and directions **without net energy input or loss** —
a Boltzmann-like scattering integral, cubic in the spectral density $S$
itself, since it involves triple products of spectral values at the four
resonant wavenumbers. This term is why a growing wind sea's spectral peak
migrates to lower frequencies over time (the well-known JONSWAP
peak-downshift) and is a genuine physical reason that a partition's height,
period, and direction are correlated functionals of one evolving object —
not merely because they are all read off the same instantaneous spectrum
(`variable_taxonomy.md` §2.2's point), but because the spectrum's *own
evolution* actively redistributes energy across the frequency/direction
bins that height, period, and direction are each computed from.

**$S_{ds}$ — whitecapping dissipation, and why it is a nonlinear
self-limiting term.** Once the sea surface's local slope exceeds a
critical steepness, wave crests become unstable and break, dissipating
energy at a rate that grows **faster than linearly** in the spectral
density — commonly parametrized (Hasselmann 1974; Komen et al. 1984, the
WAM Cycle-3 whitecapping closure) as proportional to some power $m\ge1$ of
an overall steepness measure times the spectral density itself,
$S_{ds} \propto -(\hat s)^m\,\sigma\,S(f,\theta)$, with steepness $\hat s$
itself growing with the wave field's own energy. This is the physical
origin of §2.3's saturating dissipation term, and — separately — of §3.3's
static steepness bound: both are different manifestations of the same
underlying breaking mechanism, one integrated in time (energy saturation),
one evaluated instantaneously (a height/period bound).

**The consequence for this archive.** Because $S_{in}$ couples wind state
and wave state multiplicatively, $S_{nl}$ couples the spectrum to itself
cubically, and $S_{ds}$ couples the wave state to its own steepness
nonlinearly, the wave-action balance equation is, term by term, a system of
coupled, nonlinear, **self-referencing** evolution equations relating wind
and wave-state variables. Correlation among this archive's bulk statistics
(`was`, `wsh`, `wsp`, `wsd`, …) is the observable shadow of this shared
governing dynamical system — not an independent statistical coincidence
that a richer joint density, fit with no reference to the underlying
physics, could someday learn on its own. §2.2–§2.4 make this precise with
a small, honestly-simplified worked example; §3 gives further concrete
instances; §4 turns the claim into a specification.

### 2.2 Reference structure: classical Lotka–Volterra

The canonical example of a coupled, self-referencing (each equation
depends nonlinearly on *both* state variables) ODE system is the classical
Lotka–Volterra predator–prey model,

$$
\dot x = \alpha x - \beta xy, \qquad \dot y = \delta xy - \gamma y, \qquad \alpha,\beta,\gamma,\delta > 0,
$$

with $x$ a prey population and $y$ a predator population. Both equations
contain the same bilinear term $xy$ — prey growth is reduced by predation
at a rate proportional to the product of both populations, and predator
growth is fueled by that same product — which is exactly the structural
signature this document is calling "self-referencing coupling": neither
equation is a function of its own state variable alone.

**Fixed points.** Setting $\dot x=\dot y=0$: either $x=0$ (with then
$\dot y=0 \Rightarrow y=0$ too — the extinction fixed point $(0,0)$), or,
for $x,y>0$, dividing through by $y$ and $x$ respectively,

$$
\alpha - \beta y = 0 \;\Rightarrow\; y^* = \frac{\alpha}{\beta}, \qquad
\delta x - \gamma = 0 \;\Rightarrow\; x^* = \frac{\gamma}{\delta}.
$$

**Linearized stability at the interior fixed point.** The Jacobian is

$$
J(x,y) = \begin{pmatrix} \alpha - \beta y & -\beta x \\ \delta y & \delta x - \gamma \end{pmatrix}.
$$

At $(x^*,y^*) = (\gamma/\delta,\, \alpha/\beta)$, both diagonal entries
vanish by construction (they are exactly the fixed-point conditions above),
leaving

$$
J(x^*,y^*) = \begin{pmatrix} 0 & -\beta x^* \\ \delta y^* & 0 \end{pmatrix}
= \begin{pmatrix} 0 & -\beta\gamma/\delta \\ \delta\alpha/\beta & 0 \end{pmatrix}.
$$

The characteristic polynomial of a $2\times2$ matrix with zero trace is
$\lambda^2 = -\det J = -(\beta x^*)(\delta y^*) = -\beta\gamma \cdot
\alpha\delta/(\delta\beta) = -\alpha\gamma$, so

$$
\lambda_{1,2} = \pm i\sqrt{\alpha\gamma} \quad \text{— purely imaginary}.
$$

A linearization with purely imaginary eigenvalues is a **center**:
neighboring trajectories neither spiral in nor spiral out at linear order.
For a genuinely nonlinear system this classification alone would be
inconclusive (a linear center can, in principle, be destroyed by
higher-order terms into a weak spiral) — but Lotka–Volterra is one of the
rare nonlinear systems where the neutral stability is exact, not merely a
linearization artifact, because the system possesses an explicit conserved
quantity. Direct computation (dividing the first equation by $x$, the
second by $y$, and combining) shows

$$
V(x,y) := \delta x - \gamma\ln x + \beta y - \alpha \ln y
$$

satisfies $\dot V \equiv 0$ along every trajectory. $V$'s level sets are
closed curves encircling $(x^*,y^*)$, so Lotka–Volterra's orbits are
genuinely closed, sustained oscillations — a family of periodic solutions
indexed by initial condition, not a single stable equilibrium. This is the
explicit point of contrast §2.4 returns to: **the reduced wind/wave-growth
system below shares Lotka–Volterra's coupled, multiplicative, nonlinear
ODE *structure*, but not its long-time qualitative behavior.**

### 2.3 A reduced wind/wave-growth ODE pair

This section reduces the qualitative physics of §2.1 to the smallest
honest bulk-quantity system that (a) is genuinely coupled and
self-referencing in the Lotka–Volterra sense, and (b) is simple enough to
analyze by hand. **This is a deliberately simplified pedagogical reduction
of the real spectral balance equation, not WAVEWATCH III's actual physics
restated in two lines** — it discards the full frequency–direction
spectral shape, the wave-age dependence of the true wind-input coefficient
$\beta(U/c,\cdot)$, and the entire nonlinear-transfer term $S_{nl}$. §6
returns to exactly what is being discarded.

**State variables.** Let $U$ denote bulk wind speed (this archive's
`was`), and let

$$
E := \mathrm{wsh}^2
$$

denote a bulk wind-sea energy proxy, proportional to the wind-sea
partition's spectral variance $m_{0,\text{ws}}$ up to the fixed constant
$16$ in $H_{m0}=4\sqrt{m_0}$ (`variable_taxonomy.md` §2.2). Using $E$
rather than $H_s = \mathrm{wsh}$ itself as the state variable is
deliberate: $E$, not $H_s$, is the quantity the physical energy balance
(and hence $S_{in}$, $S_{ds}$) is stated in terms of.

**Growth term.** §2.1 established $S_{in}$ is proportional to the current
spectral density and to a wind-dependent coefficient. At the bulk level,
using the standard quadratic wind-stress law $\tau = \rho_{\text{air}} C_D
U^2$ (friction velocity $u_* \propto U$, so the momentum flux driving Miles'
mechanism scales as $U^2$) and holding the wave-age dependence of $\beta$
fixed (a simplification flagged in §6), the growth contribution to $\dot E$
is taken proportional to **both** $U^2$ (available wind forcing) **and**
$E$ itself (existing wave energy) — a direct bulk analogue of
Lotka–Volterra's $xy$ term:

$$
\dot E\big|_{\text{growth}} = \kappa\, U^2\, E, \qquad \kappa > 0.
$$

**Dissipation term.** §2.1's whitecapping mechanism dissipates energy at a
rate that grows faster than linearly in $E$, via a steepness that itself
scales as $\hat s \propto \sqrt{E}$ (a fixed characteristic wavelength
converts height into steepness). Taking dissipation proportional to
steepness times energy, $S_{ds} \propto \hat s \cdot E \propto E^{3/2}$,
gives a nonlinear, **self-limiting** loss term (Hasselmann 1974-style
whitecapping scaling; see also the Shore Protection Manual / JONSWAP
fetch-limited growth curves, whose empirical saturation this term is built
to reproduce):

$$
\dot E\big|_{\text{dissipation}} = -c_{ds}\, E^{3/2}, \qquad c_{ds} > 0.
$$

**The reduced system.** Wind is treated as relaxing toward a synoptic-scale
forcing target $U_\infty$ (representing large-scale pressure-gradient-driven
meteorology, on a timescale $1/\alpha_U$ of order hours to a day) —
essentially undriven by the wave field at leading order, in contrast to
Lotka–Volterra's fully symmetric mutual coupling (a genuine, honestly-stated
asymmetry; see the aside at the end of §2.4):

$$
\boxed{\ \dot U = -\alpha_U\,(U - U_\infty), \qquad \dot E = \kappa\, U^2 E - c_{ds}\, E^{3/2}\ } \tag{2.1}
$$

Both terms of the $E$-equation are nonlinear in $E$ (growth is linear in
$E$ but multiplicative with $U^2$; dissipation is $E^{3/2}$), and the
growth term is jointly a function of both state variables — the same
qualitative "coupled, self-referencing" structure as Lotka–Volterra,
applied to a system with a different physical character (§2.4 makes the
difference precise).

**Fixed point recovers Pierson–Moskowitz exactly.** Under sustained
constant wind ($U_\infty$ fixed), $\dot U = 0 \Rightarrow U^* = U_\infty$.
Setting $\dot E = 0$ at $U=U^*$ and discarding the trivial calm-sea root
$E^*=0$:

$$
\kappa (U^*)^2 E^* = c_{ds}(E^*)^{3/2}
\quad\Longrightarrow\quad
(E^*)^{1/2} = \frac{\kappa}{c_{ds}}(U^*)^2
\quad\Longrightarrow\quad
E^* = \left(\frac{\kappa}{c_{ds}}\right)^2 (U^*)^4 .
$$

Since $H_s^* = \sqrt{E^*}$,

$$
H_s^* = \frac{\kappa}{c_{ds}}\,(U^*)^2 . \tag{2.2}
$$

This is **exactly** the fetch-limited Pierson–Moskowitz relation $H_s
\approx C\,U^2/g$ derived in `physical_consistency_constraints.md` §2.2,
with the identification

$$
\frac{\kappa}{c_{ds}} = \frac{C}{g}.
$$

That is the central result of this section: **the static PM constraint that
document treats as a per-query-point plausibility check is literally the
nontrivial fixed point of this dynamical system under sustained constant
wind.** Only the *ratio* $\kappa/c_{ds}$ is pinned by matching $C$; fully
separating $\kappa$ and $c_{ds}$ individually requires an independent
timescale measurement (e.g. matching the relaxation rate derived in §2.4 to
observed fetch/duration-limited growth rates) — a calibration task, not
resolved here, consistent with `physical_consistency_constraints.md`'s own
insistence that $C$ itself is an illustrative, empirically-tunable constant.

### 2.4 Honest stability analysis: a stable node, not a center

Real fetch/duration-limited wave growth is empirically a **damped,
monotonic, saturating** approach to the Pierson–Moskowitz equilibrium — the
JONSWAP-type growth curves climb with $\sqrt{\text{fetch}}$ or
$\sqrt{\text{duration}}$ and flatten smoothly, with no overshoot or ringing
— in sharp qualitative contrast to Lotka–Volterra's sustained closed-orbit
oscillation. This section shows the reduced system (2.1) reproduces that
contrast honestly, rather than asserting it.

**Jacobian.** Write $\dot U = f(U,E) = -\alpha_U(U-U_\infty)$, $\dot E =
h(U,E) = \kappa U^2 E - c_{ds}E^{3/2}$. Then

$$
J(U,E) = \begin{pmatrix} \partial f/\partial U & \partial f/\partial E \\ \partial h/\partial U & \partial h/\partial E \end{pmatrix}
= \begin{pmatrix} -\alpha_U & 0 \\ 2\kappa U E & \kappa U^2 - \tfrac32 c_{ds}\sqrt{E} \end{pmatrix}.
$$

The **top-right entry is identically zero** — the direct algebraic
signature of $U$'s dynamics not depending on $E$ at all in this leading-order
reduction. This single structural fact, not a numerical coincidence, is
what forecloses a center outcome below.

**Evaluate at the fixed point** $(U^*,E^*) = \big(U_\infty,\, (\kappa
U_\infty^2/c_{ds})^2\big)$ from §2.3. Using $\sqrt{E^*} = (\kappa/c_{ds})
U_\infty^2$ in the bottom-right entry,

$$
\kappa U_\infty^2 - \tfrac32 c_{ds}\sqrt{E^*}
= \kappa U_\infty^2 - \tfrac32 c_{ds}\cdot\frac{\kappa}{c_{ds}}U_\infty^2
= \kappa U_\infty^2 - \tfrac32\kappa U_\infty^2 = -\tfrac12\kappa U_\infty^2 .
$$

$$
J(U^*,E^*) = \begin{pmatrix} -\alpha_U & 0 \\ 2\kappa U_\infty E^* & -\tfrac12\kappa U_\infty^2 \end{pmatrix}.
$$

**Eigenvalues.** $J$ is lower-triangular, so its eigenvalues are simply its
diagonal entries:

$$
\lambda_1 = -\alpha_U, \qquad \lambda_2 = -\tfrac12\kappa U_\infty^2 .
$$

Both are **real and strictly negative** (given $\alpha_U,\kappa,U_\infty >
0$) — a **stable node** (generic case $\lambda_1\ne\lambda_2$), not a
center. Contrast directly with §2.2: Lotka–Volterra's Jacobian at its
interior fixed point has zero *diagonal* and nonzero off-diagonal entries,
giving $\lambda^2 = -\det J < 0$ and hence purely imaginary roots. The
wind/wave system's Jacobian has the reverse pattern — a **zero off-diagonal
entry** (one-way coupling: $U$ unaffected by $E$) and **strictly negative
diagonal entries** (both sub-processes are individually damped) — which
structurally *guarantees* real, negative eigenvalues regardless of the
specific values of $\alpha_U,\kappa,U_\infty$, for exactly the opposite
reason Lotka–Volterra structurally guarantees purely imaginary ones.

**Global confirmation, not just local linearization.** The $E$-equation at
fixed $U=U_\infty$ is a Bernoulli ODE and integrates in closed form.
Substituting $w := E^{-1/2}$ (the standard Bernoulli linearizing
substitution for a $E^{3/2}$ nonlinearity), $\dot w = -\tfrac12 E^{-3/2}\dot
E$ turns $\dot E = \kappa U_\infty^2 E - c_{ds}E^{3/2}$ into the **linear**
ODE

$$
\dot w = -\tfrac12\kappa U_\infty^2\, w + \tfrac12 c_{ds},
$$

with explicit solution $w(t) = w_\infty + (w_0-w_\infty)\,e^{-\kappa
U_\infty^2 t/2}$, $w_\infty = c_{ds}/(\kappa U_\infty^2)$. Since $E(t) =
1/w(t)^2$: starting from any young-sea state ($E_0$ small, $w_0$ large),
$w(t)$ decays **monotonically and exponentially** toward $w_\infty$, so
$E(t)$ climbs **monotonically**, with no overshoot, saturating smoothly to
$E_\infty = 1/w_\infty^2 = (\kappa U_\infty^2/c_{ds})^2 = E^*$ — reproducing
(2.2) and confirming, globally rather than only in the linearization, that
the approach to Pierson–Moskowitz equilibrium is a damped, saturating climb
of exactly the qualitative shape reported in JONSWAP-type fetch/duration
growth curves, not a decaying oscillation.

**On the predator–prey analogy, stated precisely.** The claim this document
makes is about *ODE structure* — mutual (here, one-way but still genuinely
multiplicative and nonlinear), self-referencing coupling terms, the same
mechanism that makes Lotka–Volterra's $xy$ term interesting — **not** about
qualitative long-time behavior, which differs on purpose: Lotka–Volterra is
neutrally stable with sustained closed orbits: the wind/wave-growth
reduction is asymptotically stable with monotonic, damped convergence. Using
the analogy to claim the wave system should oscillate the way predator–prey
populations do would be a real error; using it to explain *why* wind and
wave-state variables are coupled through a multiplicative, state-dependent
rate term rather than drawn independently is exactly the intended and
correct use.

*Aside — a genuinely two-way version.* Real air–sea coupling does have a
second-order feedback path: increased wave steepness raises surface
roughness (Charnock's relation, $z_0 \propto u_*^2/g$), which can locally
increase drag and slightly slow the near-surface wind. Adding a small
feedback term, $\dot U = -\alpha_U(U-U_\infty) - \mu E$ with $\mu \ll
\alpha_U,\kappa U_\infty^2$, makes the top-right Jacobian entry $-\mu$
(small, nonzero) — genuinely bidirectional, as in Lotka–Volterra. The trace
of $J$ remains negative (both diagonal terms are still present and
negative), so the fixed point remains stable; but the determinant increases
with $\mu$ (both terms in $\det J = \tfrac12\alpha_U\kappa U_\infty^2 +
2\mu\kappa U_\infty E^*$ are positive for $\mu>0$), which can, for $\mu$
large enough, push the discriminant $\mathrm{tr}(J)^2-4\det(J)$ negative —
turning the stable node into a **stable spiral** (damped oscillatory
approach, still stable, still qualitatively unlike Lotka–Volterra's
undamped center). This is a genuinely interesting refinement — closer to
some observed wave-growth overshoot phenomena in fully-coupled models —
but working it through rigorously is beyond this document's scope; it is
noted here as an honest acknowledgment of what the one-way reduction (2.1)
leaves out, not developed further (§6).

### 2.5 Consequence: Pierson–Moskowitz is an equilibrium, not a static law

`physical_consistency_constraints.md` §2.2 checks $H_s \approx C\,U^2/g$ as
a **static, per-query-point** tolerance band, gated to fetch-limited
locations. §2.3–§2.4 above show precisely what that check is checking: the
value $E$ (equivalently $H_s$) *converges to* under sustained constant
wind, on a relaxation timescale $1/|\lambda_2| = 2/(\kappa U_\infty^2)$
derived explicitly above — not a value the wave field is required to sit at
instantaneously. This matters specifically **because** the true system is
dynamical:

A rapidly freshening wind — a genuine, physically common event, not an
edge case — has $U_\infty$ jump (or ramp) upward while $E$, obeying (2.1),
has not yet had time ($\sim 1/|\lambda_2|$, i.e. several hours to a day
depending on $U_\infty$) to climb to the *new*, higher equilibrium $E^* =
(\kappa U_\infty^2/c_{ds})^2$. Such a state is a physically valid,
transient, duration-limited "young sea" — by construction, **below** the PM
curve — and it is exactly the kind of state the closed-form solution of
§2.4 describes explicitly as a point partway along the monotonic climb
toward $E_\infty$, not an error.

A static tolerance-band check, evaluated at one instant with no memory of
the wind's recent history, has **no way to distinguish** this
transient-but-physically-valid young-sea state from a genuinely
inconsistent draw (e.g. a copula sample that pairs a given wind speed with
an implausibly large wind-sea height, with no growth-curve trajectory that
could ever have produced it) — both simply look like "$H_s$ below the band
around $C\,U^2/g$" to a check that only looks at one instant. This is a
structural blind spot of the static check, not a tuning issue that a wider
tolerance band could fix without cost (widening the band uniformly would
also admit genuinely inconsistent low-$H_s$/high-$U$ draws that are not
young seas at all).

**Proposed refinement (a specification, not a resolution enacted here).**
Two options, in increasing order of fidelity, follow directly from §2.3–2.4
and are offered for `physical_consistency_constraints.md`'s next editor to
evaluate, not adopted by this document:

1. **A wave-age- or duration-aware tolerance.** Gate the PM tolerance band's
   width (or suspend the check entirely) on an estimated wave age or
   estimated time-since-last-significant-wind-shift at the query point,
   widening the acceptable band for young/rapidly-developing seas and
   tightening it only for seas plausibly at or near saturation.
2. **A trajectory check against this document's growth curve, not just its
   fixed point.** Rather than comparing the sampled $(U, H_s)$ pair only
   against the asymptote $C\,U^2/g$, integrate (2.1)'s closed-form solution
   forward from a recent-wind-history estimate at that point and compare
   the sample against the *predicted instantaneous* $E(t)$ on the growth
   curve, accepting a draw that lies near its own duration-appropriate
   point on the trajectory rather than only near the equilibrium.

Both require inputs (recent wind history, an estimate of fetch/duration)
that the current per-point-independent design of
`physical_consistency_constraints.md` (§2.4 of that document; its own §6
already flags "cross-cell/cross-time blindness" as a structural limitation
of any per-point check) does not have available — implementing either
option is therefore a genuine architectural extension of that document's
scope, not a parameter tweak, and is flagged again as an open item in §6
below rather than resolved here.

---

## 3. Guidelines: further worked correlated-variable examples

### 3.1 Wind speed and wind-sea height/period — recap and variable classes

Already derived rigorously in §2: `was` (Class B magnitude) forces `wsh`
(Class C height) through $S_{in}$'s multiplicative coupling; `wsp` (Class C
period) is not itself part of the reduced energy ODE, but §2.1's $S_{nl}$
term is the physical reason a growing sea's peak period is not independent
of its height — the same nonlinear transfer that saturates the spectrum's
energy also migrates its peak frequency, which is why fetch-limited growth
curves report $H_s$ and $T_p$ climbing together, not as two separately-drawn
quantities.

### 3.2 Wind direction and wind-sea direction

**Mechanism.** §2.1's wind-input coefficient $\beta(U/c,\,\theta -
\theta_{\text{wind}})$ is peaked at zero directional offset: $S_{in}$ pumps
energy preferentially into spectral components already traveling with the
wind, not merely into "more energy at the existing wave direction." A
locally wind-generated sea's mean direction therefore tracks the forcing
wind direction as a *consequence of the same source term* that drives its
height — this is not a separate, independently-modelable directional
coincidence.

**Variable classes.** Wind direction (`wad_come_from`, Class B) and
wind-sea direction (`wsd`, the direction member of the Class C triplet)
are, in the taxonomy's own design, embedded as two entirely unrelated
Cartesian pairs: `u_wind, v_wind` and `u_wsea, v_wsea` — two independent
tensors, fit with two independent Tucker decompositions and two independent
DMD fits under the current per-variable design, with no mechanism to
represent that one is dynamically forcing the other's alignment.

**An illustrative (not fully stability-analyzed) coupled picture.** Using
`variable_taxonomy.md` §3.2–§3.3's embeddings, let $\mathbf U_{\text{vec}} =
U(\sin\theta_{\text{wind,to}}, \cos\theta_{\text{wind,to}})$ be the wind's
own Cartesian vector and $\mathbf D := (u_{\text{wsea}}, v_{\text{wsea}}) =
H_s(\sin\theta_{\text{wsd,to}}, \cos\theta_{\text{wsd,to}})$ the height-scaled
wind-sea direction vector already defined in the taxonomy. A natural vector
generalization of (2.1)'s scalar growth/dissipation split is

$$
\dot{\mathbf D} = \kappa\, U\,|\mathbf D|\; \hat{\mathbf U}_{\text{vec}} \;-\; c_{ds}\,|\mathbf D|^{1/2}\,\mathbf D,
\qquad \hat{\mathbf U}_{\text{vec}} := \mathbf U_{\text{vec}}/U,
$$

i.e. growth injects a vector contribution specifically **along the wind's
own bearing** (not merely scaling whatever direction $\mathbf D$ currently
points in), while dissipation removes energy roughly along $\mathbf D$'s
own current direction. This structurally guarantees $\mathbf D(t)$ rotates
toward alignment with $\hat{\mathbf U}_{\text{vec}}$ over the same
relaxation timescale that governs the scalar energy problem — the
magnitude-growth and direction-alignment dynamics share the same $\kappa,
c_{ds}$, i.e. they are not two separate stories but one. A full phase-plane
stability analysis of this vector system (in particular, whether its
linearization about the aligned equilibrium is also a node, as §2.4's
scalar reduction is) is not carried out here and is flagged as an open item
in §6 — this subsection is presented at the same "tied to a real
mechanism, briefly" level the rest of §3 is, not with §2's full rigor.

*Reconstruction caveat.* Because $u_{\text{wsea}}, v_{\text{wsea}}$ are
independently Tucker/DMD-compressed under Convention 2, recovering `wsd`
or `wsh` from a reconstruction via `hypot`/`atan2` inherits the same
low-rank/`hypot`-noncommutativity caveat `variable_taxonomy.md` §3.3
already flags — a pre-existing, general subtlety of this pipeline, not
something new introduced by the coupling proposed here.

### 3.3 Steepness/period–height coupling via dispersion and the Miche breaking limit

**Mechanism.** Deep-water linear dispersion, $\omega^2 = gk$
(`physical_consistency_constraints.md` §2.3), gives wavelength in terms of
period:

$$
T = \frac{2\pi}{\omega} \;\Rightarrow\; k = \frac{\omega^2}{g} = \frac{4\pi^2}{gT^2}
\;\Rightarrow\; L = \frac{2\pi}{k} = \frac{g\,T^2}{2\pi}.
$$

The Miche (1944) breaking criterion bounds wave steepness by depth-limited
wavelength, $H/L \lesssim 0.14\tanh(kh)$; in the deep-water limit ($kh\to
\infty$, $\tanh\to 1$) this reduces to a pure steepness bound $H/L\lesssim
0.14$, and substituting the dispersion result above gives a bound relating
significant height to peak period directly:

$$
H_s \;\lesssim\; 0.14\,L \;=\; 0.14\,\frac{g\,T_p^2}{2\pi}.
$$

Applied to this archive's wind-sea triplet ($H_s\to \mathrm{wsh}$, $T_p\to
\mathrm{wsp}$), this is a **real, physically enforced upper bound**
relating two members of the wind-sea Class C triplet that the current
per-variable pipeline treats as entirely independent tensors. It is not a
new, unrelated fact: the same crest-instability/breaking mechanism that
enforces this instantaneous steepness bound is the physical process behind
§2.1's whitecapping dissipation term $S_{ds}$ — one is the temporal,
energy-integrated consequence of the mechanism (§2.3's $E^{3/2}$
saturation), the other is its instantaneous, geometric signature
(a height/period bound). The archive's `wsh`/`wsp` pair, or equally
`pwh`/`pwp`, should therefore be understood as two coupled readouts of one
breaking-limited spectrum, not two independent scalar fields whose
correlation is incidental.

| Example | Variables | Class(es) per `variable_taxonomy.md` | Mechanism |
|---|---|---|---|
| 3.1 | `was` → `wsh`, `wsp` | B (wind) → C (wind-sea height, period) | $S_{in}$ multiplicative growth; $S_{nl}$ height/period co-evolution |
| 3.2 | `wad_come_from` ↔ `wsd` | B (wind direction) ↔ C (wind-sea direction) | $S_{in}$'s directional spreading, peaked at wind direction |
| 3.3 | `wsh` ↔ `wsp` | C ↔ C (same triplet) | Dispersion relation + Miche breaking limit; same mechanism as $S_{ds}$ |

---

## 4. Correct treatment / specification: joint-compressed-state DMD for correlated groups

§2–§3 establish that several of this archive's variables are dynamically
coupled through a shared governing equation. Per §0's stated subtlety, this
section specifies how to give that coupling an actual mechanism in the
pipeline, **without** touching the per-variable tensor design or the
per-variable Tucker fit that Convention 2 requires.

### 4.1 Identify a correlated group

A **group** $\mathcal V = \{v_1,\dots,v_g\}$ is a small, deliberately chosen
set of variables believed, on the physical grounds of §2–§3, to share a
common driving source term. Two natural candidates, directly motivated by
the worked examples above:

| Group | Members | Shared mechanism |
|---|---|---|
| Wind-sea generation group | `u_wind`, `v_wind`, `u_wsea`, `v_wsea`, `wsp` | $S_{in}$ (§2.1, §3.1, §3.2) — one source term drives wind-sea height, direction, *and* (via $S_{nl}$) period growth |
| Extended swell-transfer group (larger, weaker coupling) | wind-sea group $\cup$ `u_pwave`, `v_pwave`, `pwp` | $S_nl$ energy transfer from wind-sea to lower-frequency peak/swell partitions over fetch/duration |

The first group is the direct, best-motivated candidate: it spans the
Class-B wind vector and the entire Class-C wind-sea triplet, unified by the
single fact that $S_{in}$ is the shared right-hand-side term driving all
five of those tensors' evolution. The second is offered as an example of
how the same logic extends outward, at the cost discussed in §4.3/§6.

### 4.2 Build the joint state from already-compressed coefficients

For each $v_i \in \mathcal V$, take that variable's own, already-fitted,
entirely independent Tucker core (`tucker_decomposition.md` §4.1 — nothing
about this step changes),

$$
C^{(v_i)} \in \mathbb{R}^{r_{\text{lat}}^{(v_i)} \times r_{\text{lon}}^{(v_i)} \times T},
$$

and its per-timestep vectorization (`dynamic_mode_decomposition.md` §4.2's
already-permitted, narrow reshape — nothing about this step changes
either),

$$
x_t^{(v_i)} = \mathrm{vec}\big(C^{(v_i)}_{:,:,t}\big) \in \mathbb{R}^{n_{v_i}}, \qquad n_{v_i} = r_{\text{lat}}^{(v_i)}\, r_{\text{lon}}^{(v_i)}.
$$

Concatenate across the group's members into one joint state per timestep:

$$
x_t^{\mathcal V} = \begin{pmatrix} x_t^{(v_1)} \\ \vdots \\ x_t^{(v_g)} \end{pmatrix} \in \mathbb{R}^{n_{\mathcal V}}, \qquad n_{\mathcal V} = \sum_{i=1}^g n_{v_i}.
$$

Form the shifted snapshot matrices exactly as in `dynamic_mode_decomposition.md`
§2.2, now over the joint state,

$$
X^{\mathcal V} = \big[x_0^{\mathcal V},\dots,x_{T-1}^{\mathcal V}\big], \qquad
X'^{\mathcal V} = \big[x_1^{\mathcal V},\dots,x_T^{\mathcal V}\big],
$$

and fit **one shared** linear operator $A^{\mathcal V} = X'^{\mathcal V}
(X^{\mathcal V})^+$ (via the same truncated-SVD recipe, at some rank
$r_{\text{dmd}}^{\mathcal V} \le \min(n_{\mathcal V}, T-1)$), eigendecomposed
exactly as in that document's §2.2, to obtain joint eigenvalues $\lambda_k$
and joint modes $\Phi_k^{\mathcal V} \in \mathbb{C}^{n_{\mathcal V}}$.

```python
# Illustrative only. Each C_v below is that variable's OWN, already
# independently-fit Tucker core — this step never touches a raw
# (lat, lon, time) tensor, and never changes how any single variable's
# own Tucker fit was computed.
def joint_dmd_state(cores: dict[str, np.ndarray]) -> np.ndarray:
    # cores[v]: (r_lat_v, r_lon_v, T) — one already-compressed core per
    # group member. Vectorize each variable's per-timestep coefficient
    # grid (the narrow, already-permitted reshape), then stack across
    # variables -- never across (lat, lon).
    per_variable = [c.reshape(-1, c.shape[-1]) for c in cores.values()]  # (n_v, T) each
    return np.concatenate(per_variable, axis=0)  # (sum(n_v), T) == x_0^V, ..., x_T^V
```

A group's members must share a common native sampling interval $\Delta t$
and timestamp set — automatically true here, since Tucker leaves the time
mode at (or near) full resolution per variable (`tucker_decomposition.md`
§4.3) and all wave/wind fields in one archive era share a common native
cadence — and, as with single-variable DMD, snapshot pairs must not be
formed across an archive-era boundary.

### 4.3 Precisely distinguishing this from the forbidden operation

Restated once more, because it is the point most likely to be misread: this
construction never (a) shares a Tucker core or factor matrix across
variables, (b) operates on any variable's raw `(lat, lon, time)` tensor
after this point, or (c) reintroduces a "variable" axis into anything
upstream of DMD. Every variable in the group keeps its own, fully
independent Tucker fit, exactly as `tucker_decomposition.md` §4.1 mandates.
The *only* thing that becomes joint is the small numeric coefficient vector
DMD's snapshot-matrix regression consumes — dimension $n_{\mathcal V}$,
the sum of a handful of small per-variable ranks (tens to low hundreds,
typically), not the sum of `(lat, lon)` grid sizes (hundreds of thousands).
This is exactly the scale and place both `tucker_decomposition.md` §4.4 and
`dynamic_mode_decomposition.md` §6 flagged as the legitimate location for a
future cross-variable mechanism to live, if one were ever specified — this
section is that specification.

---

## 5. Results: what a correctly joint-fit operator should show

**Block structure of $A^{\mathcal V}$.** Partition the fitted operator into
blocks aligned with each group member's own coefficient dimension,

$$
A^{\mathcal V} = \begin{pmatrix} A_{11} & A_{12} & \cdots \\ A_{21} & A_{22} & \cdots \\ \vdots & & \ddots \end{pmatrix}, \qquad A_{ij} \in \mathbb{R}^{n_{v_i}\times n_{v_j}}.
$$

An **independent** per-variable DMD fit is exactly the special case where
every off-diagonal block $A_{ij}$, $i\ne j$, is not merely small but
*structurally absent* — never estimated, because each variable's own
`dmd_fit` never sees any other variable's coefficients at all
(`dynamic_mode_decomposition.md` §4.1). A **correctly joint-fit**
$A^{\mathcal V}$, applied to a genuinely coupled group such as §4.1's
wind-sea generation group, should return **non-negligible** off-diagonal
blocks between physically coupled pairs — e.g. the block mapping the
current wind coefficient state to the *next-step* wind-sea coefficient
state — and this is the literal, checkable, fitted evidence of
cross-variable predictive structure: a joint operator whose off-diagonal
blocks are all numerically negligible relative to the diagonal blocks is
telling you the chosen group either has weaker coupling than assumed, or
was fit on too short/noisy a window to resolve it; large, structured
off-diagonal blocks are direct confirmation that the physical mechanism of
§2–§3 is visible in the fitted dynamics.

**Lead–lag structure.** Because $A^{\mathcal V}$ propagates the *entire*
joint state one step, a rising wind-coefficient mode that predictably
precedes a rising wind-sea-height mode by some number of $\Delta t$
steps — the direct discrete-time analogue of the relaxation delay derived
explicitly in §2.4 ($1/|\lambda_2| = 2/(\kappa U_\infty^2)$) — shows up as
a genuine, fittable lead–lag relationship in $A^{\mathcal V}$'s
off-diagonal structure (equivalently, in a shared complex eigenvalue whose
associated mode $\Phi_k^{\mathcal V}$ has a systematic phase offset between
its wind-block entries and its wind-sea-block entries). **An independent
per-variable DMD fit cannot represent this at all, structurally, not just
poorly** — each variable's own $A^{(v)}$ only ever multiplies that
variable's own past coefficients, so no matter how strong the true
cross-variable lead–lag relationship is physically, an independent fit's
per-mode residual (exactly what the mode-level Ornstein–Uhlenbeck stage in
`ornstein_uhlenbeck_processes.md` §4.2 is fit to) will misattribute the
missing cross-variable predictability as unmodeled noise for the driven
variable. A concrete diagnostic that makes this checkable: regress the
independent fit's wind-sea-mode OU residuals against lagged wind-mode DMD
amplitudes — non-trivial explanatory power there is direct evidence of
exactly the cross-variable structure the independent design has no
mechanism to absorb, and that a correctly joint-fit $A^{\mathcal V}$
should instead have already captured in its off-diagonal blocks.

**Rank/data tradeoff, as a diagnostic to watch.** Because $n_{\mathcal V}$
is the *sum* of the group's members' ranks, $r_{\text{dmd}}^{\mathcal V}$
must be chosen relative to this larger ambient dimension under the same
constraint as single-variable DMD, $r_{\text{dmd}}^{\mathcal V} \le
\min(n_{\mathcal V}, T-1)$ (`dynamic_mode_decomposition.md` §4.3) — a
larger group shrinks the data-to-dimension ratio for a fixed training
window $T$, and a joint fit whose singular-value spectrum shows no clear
elbow, or whose off-diagonal blocks are large but unstable across
refits/bootstrap windows, is a sign the group has been grown past what the
available $T$ can support (§6).

---

## 6. Open questions and limitations

- **The physical-consistency static check's dynamical blind spot (flagged
  for that document's owner, not resolved here).** §2.5 derives, from this
  document's own reduced ODE, that a young/duration-limited sea sitting
  below the PM curve is a physically valid transient, not necessarily an
  inconsistent draw, and proposes two concrete refinements — a wave-age- or
  duration-aware tolerance, or a trajectory check against (2.1)'s
  closed-form growth curve rather than only its fixed point. Both require
  recent-wind-history state that `physical_consistency_constraints.md`'s
  current per-query-point-independent design does not carry; implementing
  either is an architectural extension of that document's scope. This
  document proposes the refinement and its physical justification; it does
  not implement it, and `physical_consistency_constraints.md`'s next editor
  is the one positioned to decide how (or whether) to act on it.

- **Group selection is an open modeling choice, not a rule derived here.**
  §4.1 offers two candidate groups and states, in §5, the diagnostic
  signature of a well-chosen one, but gives no general algorithm for
  choosing group membership. The tradeoff is real and unresolved: growing a
  group to capture more genuine physical coupling (e.g. adding the
  peak-wave triplet to capture $S_{nl}$'s energy transfer beyond the
  wind-sea partition) increases $n_{\mathcal V}$ and hence the DMD rank and
  data requirements, risking an underdetermined or overfit joint operator
  for a fixed training window; keeping groups small and conservative avoids
  that risk but reproduces the fully-independent design's blind spot for
  whatever coupling falls outside the chosen group. This mirrors, and
  narrows without closing, the same open cross-variable-coupling question
  `tucker_decomposition.md` §4.4 and `dynamic_mode_decomposition.md` §6
  both already flag.

- **The reduced ODE system (§2.3–§2.4) is a pedagogical simplification, not
  a substitute for the real spectral balance equation.** It discards the
  full frequency–direction spectral shape; it holds the wind-input growth
  coefficient $\kappa$ fixed rather than letting it depend on wave age
  $U/c_p(E)$ as the real $\beta(U/c,\cdot)$ does (itself a further
  self-referencing refinement this reduction omits, since $c_p$ depends on
  the wave field's own state); it omits $S_{nl}$'s cubic self-interaction
  entirely; and it treats $U$'s own dynamics as a trivial linear relaxation
  rather than genuine synoptic meteorology (with the two-way Charnock-type
  feedback noted only as an aside in §2.4, not developed). It should be
  read as demonstrating the qualitative mechanism — why wind and wave-state
  variables are coupled rather than merely correlated, and why that
  coupling's fixed point recovers a known physical constant exactly — not
  as a claim that WAVEWATCH III literally integrates a two-equation system.

- **§3.2's directional-alignment ODE is illustrative, not fully
  stability-analyzed.** Unlike §2.3–§2.4's scalar energy system, the vector
  generalization proposed in §3.2 is not carried through a full
  linearization/eigenvalue analysis of its aligned equilibrium. Whether
  that equilibrium is a node (as the scalar case is) or admits a more
  interesting rotational approach (a Kuramoto-oscillator-style phase
  equation is the natural further reduction, if directional alignment
  dynamics are ever specified with the same rigor as §2) is left open.
