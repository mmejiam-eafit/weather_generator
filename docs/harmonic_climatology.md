# Harmonic Climatology Removal

A graduate-level treatment of the per-cell harmonic-regression climatology
stage of the `weather_generator` stochastic pipeline: what it should
compute, why the regression must be built the way it is built, and how its
output tensors must be represented so that no spatial or cross-variable
structure is silently destroyed downstream.

## 0. Two errors this document corrects

This document replaces an earlier draft set (since deleted) that made two
errors. Both are corrected here, explicitly, because they are easy to
reintroduce by analogy with "how the code currently happens to work."

**Error 1 — this document is prescriptive, not descriptive.** Its job is to
define what the harmonic-climatology stage *should* compute, so that the
actual code in `src/train/climatology.py` can be corrected against it — not
to summarize the current implementation and certify it as correct because
it exists. Nothing below should be read as "here is what
`ClimatologyRemover` computes, and that is why it is right." Everywhere the
current code agrees with the treatment below, that is coincidence worth
confirming, not evidence used to derive the treatment. Where the current
code is relevant, it is flagged inline, briefly, and clearly marked as an
aside — never as the source of the theory.

**Error 2 — no flattening, ever, and no merging across variables.** The
dataset for a single physical field is a **3D tensor** $(\text{lat},
\text{lon}, \text{time})$, already land-masked. It is never a 4D tensor of
shape $(\text{lat}, \text{lon}, \text{time}, \text{num\_variables})$ — every
variable (`sst`, `u_wind`, `v_wind`, `u_wsea`, `wsp`, …) is fit completely
independently, on its own tensor, with its own coefficients. And within one
variable's tensor, the spatial axes are never collapsed: the tensor is
never reshaped to a 2D $(\text{lat}\cdot\text{lon}, \text{time})$ or
$(\text{lat}, \text{lon}\cdot\text{time})$ matrix as an *externally visible*
representation. Flattening destroys the adjacency between neighboring
cells that later pipeline stages (spatial covariance structure, Tucker/DMD
decompositions) depend on, and once two axes are merged into one, that
adjacency cannot be reconstructed from the merged axis alone.

This does **not** mean every cell must be fit with an explicit Python
`for`-loop. Harmonic climatology removal is, by construction, a
*per-cell* regression — the same design matrix $X$ (built once, from the
shared time axis) is fit against a different response vector at every
$(\text{lat}, \text{lon})$ location. Solving all of those regressions in one
batched linear-algebra call — via `np.linalg.lstsq` applied to a
temporarily-reshaped design, or via an `einsum` contraction that keeps
$(\text{lat}, \text{lon})$ as live broadcast axes — is mathematically
identical to looping over every cell independently, coefficient by
coefficient. It is a vectorization of the loop, not a violation of it, and
it loses no local information whatsoever, because each cell's fitted
coefficients still depend only on that cell's own time series. Section 4
states the exact line between this legitimate broadcasting and the
forbidden flattening.

## 1. Summary: separating a deterministic background from a stochastic anomaly

Every field this pipeline ingests — sea-surface temperature, wind
components, wave-partition components, currents — can be modeled, at each
grid cell $s = (\text{lat}, \text{lon})$, as the sum of a **deterministic,
slowly-and-periodically-varying background state** and a **stochastic
anomaly**:

$$
X(s, t) = \mu(s, t) + X'(s, t).
$$

**For height/period fields, $X$ is the `log`-transformed tensor.** Per
`variable_taxonomy.md` §3.4, wave-height fields (`sigwh`, `wsh`, `pswh`)
and period fields (`wsp`, `pwp`, `pswp`) enter this stage already
`log`-transformed ($\log E$, $T$), not in raw physical units — this
document's own per-cell regression is itself an L2/least-squares fit, so
it benefits from the same dynamic-range/leverage argument that motivates
the transform ahead of Tucker and DMD. This has a substantive consequence
for §1's decomposition, worth being explicit about: a *multiplicative*
seasonal cycle in raw units (e.g. "winter storms are systematically $k$
times more energetic than summer's") becomes an *additive* shift in $\mu$
under `log` — the harmonic regression below should be read as fitting that
additive, log-space seasonal shape, not a seasonal cycle in physical wave
height or period directly. Every other field this document discusses
(`sst`, wind/current components) is unaffected — `variable_taxonomy.md`
§3.4 explicitly excludes `sst`, and Class B wind/current magnitudes are
not retained as their own tensor at all (§3.2), only as `u,v` pairs.

This is the same decomposition used throughout applied math for systems
driven by a mix of periodic and secular (slowly varying, non-periodic)
forcing. Consider a linear, forced ODE/PDE such as a damped oscillator or a
diffusion equation with a forcing term that has (a) a periodic component at
one or more known frequencies (annual and semi-annual insolation cycles,
and — for some fields — the diurnal cycle) and (b) a slow secular drift
(a warming trend). The general solution decomposes as

$$
u(t) = u_{\text{particular}}(t) + u_{\text{homogeneous}}(t),
$$

where $u_{\text{particular}}$ tracks the forcing (it inherits the forcing's
periodicities and its secular drift, with some frequency-dependent gain and
phase lag imposed by the system's transfer function) and
$u_{\text{homogeneous}}$ is whatever remains once the forced response is
subtracted — governed by the system's own internal dynamics rather than by
the external forcing.

$\mu(s,t)$ in this pipeline plays the role of $u_{\text{particular}}$: it is
the fitted, deterministic response of each grid cell to solar forcing
(insolation's annual, semi-annual, and — where physically appropriate —
diurnal cycles) plus a slowly varying secular trend. $X'(s,t) = X(s,t) -
\mu(s,t)$, the anomaly, plays the role of $u_{\text{homogeneous}}$: it is
what is left once the known, deterministic, externally-forced part of the
signal has been removed, and it is this residual — not the raw field
$X(s,t)$ — that the rest of the pipeline (standardization, spatial/temporal
decomposition, copula-based dependency modeling) treats as the stochastic
process to be learned. Climatology removal is therefore not a cosmetic
detrending step; it is what makes the downstream stochastic-modeling
problem well-posed in the first place, by ensuring the process being
modeled stochastically has (approximately) zero, stationary mean rather
than a mean that swings with the seasons or drifts with the climate.

## 2. Theoretical foundation

### 2.1 The harmonic design

At each cell $s$, the climatology is modeled as a linear combination of a
fixed set of basis functions of time:

$$
\mu(s, t) = c_0(s) + c_{\text{lin}}(s)\,\Delta\tau(t)
+ \sum_{k=1}^{2} \Big[ a_k(s)\cos(k\,\omega_y t) + b_k(s)\sin(k\,\omega_y t) \Big]
+ \mathbb{1}_{\text{diurnal}}\Big[ a_d(s)\cos(\omega_d t) + b_d(s)\sin(\omega_d t) \Big],
$$

with:

- $\omega_y = 2\pi / 365.25$ rad/day (annual angular frequency), and $k=2$
  giving the semi-annual harmonic $2\omega_y$ — needed because the seasonal
  cycle of most ocean/atmosphere fields is not a pure sinusoid (e.g. a
  double-peaked SST cycle in some regions, or the two-hemisphere-summer
  structure smeared into a single time series);
- $\omega_d = 2\pi$ rad/day (one full cycle per day), included only for
  fields with a genuine solar-driven diurnal signal (§3);
- $\Delta\tau(t)$, a year-normalized time coordinate for the trend term,
  deliberately built around a **different time origin** than the harmonic
  terms (§2.2 — this is the central derivation of this document);
- $c_0(s), c_{\text{lin}}(s), a_k(s), b_k(s), a_d(s), b_d(s)$: the per-cell
  coefficients to be estimated. Every one of them is a scalar field over
  $(\text{lat}, \text{lon})$ — i.e. the full coefficient object for one
  variable is a tensor of shape $(\text{lat}, \text{lon}, n_{\text{terms}})$
  (§4).

### 2.2 Harmonic regression is OLS *because* of the cos/sin parameterization

A single periodic component could equally well be written in
amplitude-phase form,

$$
A\cos(\omega t - \varphi),
$$

which looks more physically transparent ($A$ is the seasonal amplitude,
$\varphi$ the phase lag relative to solar forcing). But fitting $(A,
\varphi)$ directly by least squares is a **nonlinear** regression problem:
$\mu(t)$ is not linear in $\varphi$ (it involves $\cos(\omega t - \varphi)$,
not $\varphi$ or $\cos\varphi$ alone), so minimizing $\sum_t \big(y_t -
A\cos(\omega t-\varphi)\big)^2$ over $(A,\varphi)$ has no closed form, is
non-convex in $\varphi$ (which is periodic, so naive gradient descent can
converge to the wrong branch), and generally needs an iterative solver
(Gauss–Newton, Levenberg–Marquardt) with a reasonable initial guess.

The standard trick — used throughout this design — is the angle-addition
identity:

$$
A\cos(\omega t - \varphi) = \underbrace{(A\cos\varphi)}_{a}\cos(\omega t) + \underbrace{(A\sin\varphi)}_{b}\sin(\omega t) = a\cos(\omega t) + b\sin(\omega t).
$$

For a *fixed, known* frequency $\omega$ (annual, semi-annual, and diurnal
frequencies are known a priori from the physics — they are not being
estimated from the data), $\cos(\omega t)$ and $\sin(\omega t)$ are just two
more fixed basis functions of $t$, exactly like the constant $1$ or the
linear term $\Delta\tau(t)$. The model is therefore **linear in $(a, b)$**
even though it is manifestly nonlinear in $(A, \varphi)$. Fitting $(a, b)$
by ordinary least squares gives a closed-form, globally optimal solution (no
local optima, no initialization sensitivity), and $(A, \varphi)$ can always
be recovered afterward, losslessly, as a deterministic post-hoc
transformation:

$$
A = \sqrt{a^2 + b^2}, \qquad \varphi = \operatorname{atan2}(b, a).
$$

This is the general reason the whole design in §2.1 — intercept, trend, and
every harmonic pair — is written as a **linear combination of fixed basis
functions of time with unknown coefficients**, rather than as unknown
amplitudes and phases: it turns a nonlinear, non-convex fitting problem into
an ordinary least-squares problem, at zero cost in expressiveness.

### 2.3 OLS mechanics: normal equations

Collect the $T$ time samples' design-matrix rows into $X \in
\mathbb{R}^{T\times n_{\text{terms}}}$ (one column per term in §2.1) and one
cell's response into $y\in\mathbb{R}^{T}$. OLS minimizes the residual sum of
squares,

$$
\hat\beta = \arg\min_\beta \|y - X\beta\|_2^2,
$$

whose stationarity condition ($\nabla_\beta \|y-X\beta\|^2 = -2X^\top(y -
X\beta) = 0$) gives the **normal equations**

$$
X^\top X\,\hat\beta = X^\top y \quad\Longrightarrow\quad \hat\beta = (X^\top X)^{-1} X^\top y,
$$

provided $X$ has full column rank (so that the Gram matrix $X^\top X$ is
invertible). Every per-cell fit in §4 is one instance of this equation,
with $y$ replaced by that cell's own $(y_1,\dots,y_T)$ time series and $X$
shared across all cells (since the sampling times are the same everywhere).

### 2.4 Why the trend term needs its own, centered time origin

This is the single most consequential design detail in this document, and
it is worth being able to re-derive on demand — it is a special case of a
completely general fact about OLS: **an intercept and a slope term become
statistically entangled unless the predictor used for the slope is centered
relative to the fitting window.**

Strip the model down to just the two terms that interact: the intercept and
the trend,

$$
y_t = c_0 + c_{\text{lin}}\,\Delta\tau(t) + \varepsilon_t, \qquad t=1,\dots,T,
$$

so the design matrix is $X = \big[\,\mathbf{1} \ \ \Delta\tau\,\big] \in
\mathbb{R}^{T\times 2}$, with $\mathbf{1}$ the all-ones column and
$\Delta\tau = (\Delta\tau(t_1),\dots,\Delta\tau(t_T))^\top$. The Gram matrix
is

$$
X^\top X =
\begin{pmatrix}
\mathbf{1}^\top\mathbf{1} & \mathbf{1}^\top \Delta\tau \\
\Delta\tau^\top \mathbf{1} & \Delta\tau^\top\Delta\tau
\end{pmatrix}
=
\begin{pmatrix}
T & \sum_t \Delta\tau(t) \\
\sum_t \Delta\tau(t) & \sum_t \Delta\tau(t)^2
\end{pmatrix}.
$$

**The off-diagonal Gram-matrix term is exactly $\sum_t \Delta\tau(t)$.**
Inverting this $2\times2$ matrix,

$$
(X^\top X)^{-1} = \frac{1}{\det} \begin{pmatrix} \sum_t\Delta\tau(t)^2 & -\sum_t\Delta\tau(t) \\ -\sum_t\Delta\tau(t) & T \end{pmatrix}, \qquad \det = T\sum_t\Delta\tau(t)^2 - \Big(\sum_t\Delta\tau(t)\Big)^2,
$$

and applying $\hat\beta = (X^\top X)^{-1}X^\top y$ gives

$$
\hat c_0 = \frac{\sum_t\Delta\tau(t)^2 \sum_t y_t \;-\; \sum_t\Delta\tau(t)\sum_t \Delta\tau(t) y_t}{\det}, \qquad
\hat c_{\text{lin}} = \frac{T\sum_t \Delta\tau(t) y_t \;-\; \sum_t\Delta\tau(t)\sum_t y_t}{\det}.
$$

**Uncentered case.** If $\Delta\tau(t)$ is measured from the *start* of the
fitting window (e.g. $\Delta\tau(t) = (t - t_{\text{start}})/1\,\text{yr}$,
so $\Delta\tau(t)\ge 0$ throughout), then $\sum_t \Delta\tau(t) \ne 0$ in
general — it is roughly $T\cdot(\text{window length})/2$ in years. Both
estimators above then mix the $\sum y_t$ row and the $\sum \Delta\tau(t)y_t$
row together: $\hat c_0$ is no longer simply the sample mean $\bar y$, and
$\hat c_{\text{lin}}$ is no longer simply the raw covariance-over-variance
slope — each carries a contribution that depends on the *other* term's
statistic, scaled by $\sum_t\Delta\tau(t)$. Equivalently, the sampling
covariance of the estimator, $\operatorname{Cov}(\hat\beta) = \sigma^2
(X^\top X)^{-1}$, has a nonzero off-diagonal entry
$-\sigma^2\sum_t\Delta\tau(t)/\det$: noise that perturbs the fitted trend
also perturbs the fitted intercept, and vice versa. This is the textbook
multicollinearity between slope and intercept in simple linear regression —
harmonic climatology just inherits it if the trend predictor is left
uncentered.

**Centered case.** Now define $\Delta\tau(t) = (t - t_{\text{ref}})/1\,\text{yr}$
with $t_{\text{ref}}$ chosen as the fitting window's **midpoint**. Over a
(roughly) evenly-sampled window centered on $t_{\text{ref}}$, the positive
and negative deviations balance, so

$$
\sum_t \Delta\tau(t) \approx 0.
$$

The Gram matrix becomes (approximately) diagonal,

$$
X^\top X \approx \begin{pmatrix} T & 0 \\ 0 & \sum_t\Delta\tau(t)^2 \end{pmatrix},
$$

its inverse is diagonal too, and the normal equations **decouple
completely**:

$$
\hat c_0 \approx \bar y = \frac{1}{T}\sum_t y_t, \qquad \hat c_{\text{lin}} \approx \frac{\sum_t \Delta\tau(t)\,y_t}{\sum_t \Delta\tau(t)^2}.
$$

The intercept is now just the sample mean of $y$ over the window, entirely
uninfluenced by whatever the trend coefficient happens to be, and the trend
is the simple (centered) OLS slope, uninfluenced by the intercept. The
off-diagonal covariance term vanishes with the off-diagonal Gram term, so
the two estimates are also statistically decorrelated, not just
algebraically decoupled at the point estimate.

This is exactly the same maneuver as centering a predictor $x$ (fitting $y
= \beta_0 + \beta_1 (x - \bar x)$ instead of $y = \beta_0' + \beta_1 x$) to
decorrelate an estimated intercept from an estimated slope in ordinary
simple linear regression — harmonic climatology's trend term is a special
case of that general principle, not a bespoke trick.

**Why the harmonic terms don't need this treatment.** The harmonic columns
$\cos(k\omega_y t), \sin(k\omega_y t)$ interact with the intercept through
the same Gram-matrix mechanism ($\sum_t \cos(k\omega_y t)$ plays the role of
$\sum_t \Delta\tau(t)$ above), but a sinusoid summed over a window that
spans an integer number of its own periods averages to (approximately)
zero *regardless of where the time origin is placed* — shifting the origin
by $t_0$ only rotates the pair $(a_k, b_k)$ into each other via the
angle-addition identity, it does not reintroduce a mean offset. The trend
term has no such protection: $\Delta\tau(t)$ is monotonic, not periodic, so
its window-average is generically nonzero unless the origin is
*deliberately* placed at the window's centroid. That asymmetry — periodic
terms self-average regardless of origin, a linear term does not — is
precisely why this design uses **two different time origins**: an arbitrary
(but fixed) origin for the harmonic terms, and the window midpoint,
specifically, for the trend term.

## 3. Guidelines: variable-class handling

### 3.1 Directional (Cartesian) components must already be $(u, v)$ before this stage

OLS assumes its response variable is a real number for which linear
combinations and residuals are physically meaningful: $y_t - \hat\mu(t)$
must mean something, and so must $\hat\mu(t) = \sum_j \beta_j x_{tj}$. A
compass-convention direction stored as a raw angle in degrees (a
"coming-from" or "going-to" bearing) is not such a quantity — it wraps at
$0°/360°$, and the wrap is not a physical discontinuity, only an artifact of
the encoding. Two headings of $359°$ and $1°$ are two degrees apart
physically, but a naive residual computes $358°$; a harmonic regression
fit directly to the raw angle would be dragged toward whatever the encoding
happens to place at the branch cut, and its intercept/trend/seasonal terms
would all be contaminated by wraparound artifacts that have nothing to do
with the physical direction field. (This is the same circularity argument
used elsewhere in this repository's variable-taxonomy documentation for why
angle-valued fields cannot be treated as ordinary reals — restated here in
this document's own terms because it is exactly the assumption OLS needs
and a wrapped angle violates.)

The fix used throughout this pipeline is to decompose each `(speed,
direction)` pair into Cartesian components before any harmonic fitting:

$$
u = \text{speed}\cdot\sin(\theta), \qquad v = \text{speed}\cdot\cos(\theta),
$$

with $\theta$ resolved to a single, consistent "going-to" or "coming-from"
convention beforehand (a separate concern from climatology removal). Each
of $u$ and $v$ is then an ordinary signed real-valued scalar field with no
wraparound: differences and linear combinations of $u$ (or of $v$) are
physically meaningful (they are literally components of a vector in a fixed
Cartesian frame), so each becomes a legitimate independent target for the
harmonic regression in §2, on exactly the same footing as `sst`. The
resulting climatology is a climatological *vector field*
$(\mu_u(s,t),\mu_v(s,t))$; note that a decomposed pair's $u$ and $v$
components are still fit **independently of each other** (§4) — the
`(speed, direction)` decomposition happens once, upstream of this stage, not
inside the per-variable regression itself.

### 3.2 Diurnal term: physical reasoning per variable class

The diurnal harmonic pair ($\cos(\omega_d t), \sin(\omega_d t)$, one cycle
per day) should be included only for fields with a genuine,
solar-insolation-driven daily cycle — not decided by a hardcoded list of
variable names, but by asking whether the underlying physical process has a
24-hour solar forcing signature strong enough to be resolvable at the
model's spatiotemporal resolution.

| Variable class | Physical driver | Diurnal term? | Reasoning |
|---|---|---|---|
| `sst` | Direct solar heating of the ocean skin/mixed layer | Yes | Daytime insolation warms the near-surface layer and nighttime radiative cooling reverses it; this is the textbook diurnal SST cycle, well documented and directly solar-forced. |
| `u_wind`, `v_wind` (near-surface wind, from `was`/`wad_come_from`) | Boundary-layer heating cycle (convective mixing by day, stable stratification and decoupling from geostrophic flow at night; sea/land-breeze circulations) | Yes | Near-surface wind speed and direction both have a well-established diurnal modulation driven by the daily heating/cooling of the boundary layer, distinct from but ultimately traceable to solar forcing. |
| `u_wsea`, `v_wsea` (from `wsh`/`wsd`), `wsp`, `u_peak`, `v_peak` (from `pwh`/`pwd`), `u_pswell`, `v_pswell` (from `pswh`/`pswd`) — wave-partition fields | Wind-generated gravity waves: fetch, duration, and storm-system evolution | No | Wave generation integrates wind forcing over fetch and duration; wave energy grows and decays on synoptic timescales (many hours to days), not in lock-step with the sub-daily wind modulation above. Even where the underlying wind has a diurnal signature, the wave field's response is heavily low-pass filtered by wave-growth physics and by propagation away from the generation area, so at typical global-model resolution there is no physically resolvable diurnal signal left to fit — fitting one anyway would just capture noise or wind-field leakage, not a real wave process. |
| `u_current`, `v_current` (from `ocs`/`ocd`) | Wind-driven and thermohaline circulation, plus (in shelf/coastal regions) tidal forcing | No, by default | Ocean currents at global-model resolution are not solar-diurnally forced the way SST or near-surface wind are. Tides do produce genuine sub-daily and near-24-hour periodicities in some regions, but that is a distinct astronomical (lunar/solar tidal) forcing mechanism, not the solar-*heating* diurnal cycle this term is meant to capture, and it is not resolved by typical global GFS/WAVEWATCH III-class products in the first place — including a diurnal term here without a specific tidal justification would misattribute variance. |

The general test to apply to any *new* variable added to this pipeline is
not "is it in the default diurnal list" but: **does the field have an
energetically direct, same-day response to solar insolation, at this
model's resolution?** If yes, include the diurnal pair; if the field's
physical generation mechanism operates on longer (synoptic, seasonal) time
scales even though its ultimate forcing is solar, leave it out.

## 4. Correct treatment for this dataset's tensors

Each physical variable arrives (after land-masking, upstream of this stage)
as its own 3D tensor $Y \in \mathbb{R}^{T\times \text{lat}\times \text{lon}}$
— continuous time on one axis, the intact spatial grid on the other two.
The correct treatment fits one independent OLS regression (§2) **per grid
cell**, for **one variable at a time**:

$$
\hat\beta(s) = \big(X^\top X\big)^{-1} X^\top y(s), \qquad y(s) = Y[:, s] \in \mathbb{R}^T, \quad s \in \{1,\dots,\text{lat}\}\times\{1,\dots,\text{lon}\},
$$

where $X$ (the design matrix of §2.1) is identical for every cell — only
the response vector $y(s)$ changes — and, if a cell is land-masked (NaN at
every timestep), it is simply excluded (or assigned NaN coefficients, which
downstream masking already discards).

**Batched solve, not a flattening.** Because $X$ is shared across all
cells, all of those per-cell regressions can be solved in a single
vectorized linear-algebra call without looping in Python, and — critically
— without ever treating $(\text{lat}, \text{lon})$ as a single collapsed
axis in the tensor's *external* representation. Two equivalent, acceptable
ways to do this:

1. **Batched `lstsq`.** Temporarily reshape $Y$'s spatial axes into one
   axis purely as the argument to a single `np.linalg.lstsq(X, Y_reshaped)`
   call (so the call sees a $(T, \text{lat}\cdot\text{lon})$ matrix), then
   immediately reshape the returned coefficients back to
   $(n_{\text{terms}}, \text{lat}, \text{lon})$ before storing or returning
   them. The reshape exists only inside the call boundary; nothing upstream
   or downstream of it ever sees, stores, or indexes a merged
   $(\text{lat}\cdot\text{lon})$ axis.
2. **Normal equations via `einsum`.** Compute the (shared) Gram matrix $G =
   X^\top X$ once, form $X^\top Y$ by contracting only over the time axis
   (`np.einsum("tc,tij->cij", X, Y)`, keeping `i` (lat) and `j` (lon) as
   live, distinguishable broadcast axes throughout), and solve $G\,\hat B =
   X^\top Y$ for $\hat B$ with shape $(n_{\text{terms}}, \text{lat},
   \text{lon})$ via a batched `np.linalg.solve`/broadcasted matrix solve.
   No reshape of the spatial axes ever happens here at all.

Either route is **numerically and mathematically identical** to looping
over every $(\text{lat},\text{lon})$ cell and calling `lstsq` on its own
$T$-length series one at a time: each cell's fitted coefficients are a
function of that cell's own time series alone, so no cross-cell (local or
otherwise) information is introduced or lost by batching the solve. This is
what §0 means by "broadcasting is not the forbidden flattening."

**What is forbidden** is letting the *result* of that fit be represented,
downstream, as a merged $(\text{cell}, \text{coeff})$ 2D array — i.e.
treating the reshape used in route 1 above as the coefficient tensor's
permanent identity rather than an implementation detail that gets undone
before the function returns. The stored (and consumed-by-later-stages)
representation of one variable's fitted climatology must always be:

- the coefficient tensor $\hat B \in \mathbb{R}^{n_{\text{terms}} \times
  \text{lat} \times \text{lon}}$ — one $(\text{lat},\text{lon})$ coefficient
  map per design-matrix column, with $(\text{lat},\text{lon})$ always
  distinguishable axes; and
- the evaluated climatology itself, $\mu(s,t)$, as a 3D tensor
  $(\text{lat}, \text{lon}, \text{time})$ — reconstructible at any set of
  query times via $\mu = X_{\text{query}} \cdot \hat B$ (again contracting
  only over the coefficient axis, e.g. `einsum("tc,c...->t...", X_query,
  B)`), never as a flattened $(\text{cell}, \text{time})$ matrix.

**Never merged across variables.** `sst`'s tensor, `u_wind`'s tensor,
`u_wsea`'s tensor, and every other variable's tensor are fit completely
independently — different design matrices where the diurnal term differs
(§3.2), different response tensors, different coefficient tensors, stored
separately (e.g. one coefficient tensor per variable name in a mapping, not
concatenated into one array indexed by a combined
`(variable, lat, lon, coeff)` axis). There is no cross-variable regression
step here at all; "harmonic climatology" is, by construction, a
single-variable operation applied once per variable.

*(Aside on the current code, per §0's framing rule: `src/train/climatology.py`
already reshapes each variable's spatial axes back to
`(n_terms, lat, lon)` after its internal `lstsq` call and already loops
over `X.data_vars` per-variable rather than sharing one array across
variables — i.e. it already lands on route 1 above for this specific
concern, which is worth confirming empirically rather than assuming.)*

## 5. Results: interpreting and evaluating the fitted climatology

### 5.1 Reading a fitted trend coefficient physically

Because $\Delta\tau(t)$ is measured in years, $c_{\text{lin}}(s)$ carries
units of *(variable's units) per year* directly — e.g. for `sst`,
$c_{\text{lin}}(s)$ is a local SST warming (or cooling) rate in °C/year at
cell $s$, readable straight off the fit with no further unit conversion.
Mapped over $(\text{lat},\text{lon})$, $c_{\text{lin}}$ is exactly the kind
of spatial warming-rate map used in observational climate-trend studies.

A significance-style check some pipelines apply — comparing
$|c_{\text{lin}}(s)|$ against the anomaly's own standard deviation at that
cell (i.e., how large the trend is relative to the field's natural
variability over the fitting window) — is a reasonable practical filter for
"is this trend big enough to matter compared to the noise it's embedded
in," but it is a separate, downstream diagnostic, not part of the
regression itself: the regression is agnostic to whether $c_{\text{lin}}$
turns out to be practically negligible.

**Directional fields need an extra step of care.** For a decomposed
directional pair, $c_{\text{lin}}(u)$ and $c_{\text{lin}}(v)$ are each fit
independently (§4) and each has straightforward units (e.g. m/s per year).
But a linear trend in a Cartesian component does *not* correspond to a
simple trend in the resultant speed or bearing — speed is
$\sqrt{u^2+v^2}$ and bearing is $\operatorname{atan2}(u,v)$, both nonlinear
in $(u,v)$, so "the wind speed is trending at $X$ m/s/year" cannot be read
directly off $c_{\text{lin}}(u)$ or $c_{\text{lin}}(v)$ alone. To make a
physically meaningful statement about a trend in speed or direction,
reconstruct $(\text{speed}, \theta)$ from the fitted $(u,v)$ climatology
evaluated at two different times (§5.2) and compare those, rather than
interpreting either component's trend coefficient in isolation.

### 5.2 Evaluating the climatology outside the fitting window

$\mu(s,t)$ is, by construction, a smooth, closed-form function of
*continuous* time — a finite sum of an intercept, a linear term, and
sinusoids, all defined for any real $t$, not just the discrete timestamps
used to fit it. Evaluating it at query times outside the training window is
therefore just a matter of plugging those times into the same design-matrix
construction (§2.1) and contracting against the already-fitted
coefficients (§4) — nothing about the functional form requires $t$ to lie
inside $[t_{\text{start}}, t_{\text{end}}]$.

Two caveats attach to this, though, and they are not symmetric:

- **The harmonic terms extrapolate safely** because they are bounded and
  periodic — evaluating $\cos(\omega_y t)$ a decade past the fitting window
  is exactly as well-defined, and exactly as bounded, as evaluating it
  inside the window.
- **The linear trend term does not extrapolate safely.** $\Delta\tau(t)$
  grows without bound as $t$ moves away from $t_{\text{ref}}$, so a
  constant-rate trend fitted over one window, if evaluated far outside it,
  will eventually produce physically implausible values (e.g. an SST
  climatology that keeps rising linearly forever). The trend term should be
  understood as a **local linear approximation valid near the fitting
  window**, not a claim about the field's long-run behavior; evaluating it
  well outside the window it was fit on requires the same caution as any
  linear extrapolation of a real physical trend.

## 6. Open questions and limitations

- **The whole design is a quasi-stationarity assumption.** Modeling
  $\mu(s,t)$ as a *fixed*-shape harmonic pattern (constant amplitude and
  phase for each harmonic) plus a *single* constant linear rate,
  unchanging across the entire fitting window, presumes the underlying
  climatological signal itself is stationary in shape over that window.
  Real seasonal cycles are not perfectly stationary: their amplitude and
  phase can themselves drift over long records (e.g. warming can widen or
  narrow a region's seasonal SST range, shift the timing of a seasonal
  peak, or be modulated by decadal ocean-atmosphere oscillations), which
  this design cannot represent — any such drift gets folded into either the
  harmonic coefficients (as a compromise average shape) or the residual
  anomaly (as spurious low-frequency "noise" that is not actually
  stochastic).

- **A single global linear trend is itself an approximation.** Real
  physical trends are rarely exactly linear over long horizons — they can
  accelerate, decelerate, or reverse. This design captures only the
  best-fit constant rate over the specific fitting window used; it says
  nothing about whether that rate is representative of a longer-run
  process or is itself just the locally linear piece of something more
  complex (e.g. a quadratic acceleration, or a trend correlated with a
  multi-year oscillation that the window happens to catch mid-swing).

- **The fitting window's length and placement change what "the trend"
  means, not just how precisely it's estimated.** Because $c_{\text{lin}}$
  is defined relative to whichever window it was fit on, two fits over
  different windows of the same underlying record are not simply
  noisier/less-noisy estimates of one true number — they can be answers to
  subtly different questions. A short window can have interannual
  variability (e.g. an ENSO event) alias into an apparent trend that a
  longer window would average away; a longer window smooths over such
  events but also averages over any genuine acceleration or deceleration
  within it, and can be sensitive to exactly where its start/end points
  fall relative to such events. Any reported trend coefficient should be
  read together with the window it was estimated over, not treated as a
  window-independent physical constant.

- **The diurnal-inclusion reasoning in §3.2 is resolution-dependent.** The
  claim that wave-partition fields have no resolvable diurnal signal is
  stated relative to typical global-model resolution and forcing data
  (GFS-Wave/WAVEWATCH III-class products); it should be revisited if this
  pipeline is ever pointed at higher-resolution regional wave models, where
  locally wind-driven short-fetch seas forced by diurnal sea/land-breeze
  cycles could, in principle, carry a weak but real diurnal signature that
  this default would currently suppress.
