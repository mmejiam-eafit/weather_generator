# Gaussian Copula and Marginal Gaussianization

## 0. Two errors this document corrects

This document replaces an earlier draft (part of a seven-file documentation
set the user deleted) that made two specific errors. Both are corrected
here, explicitly, before anything else.

**Error 1 — this document is prescriptive, not descriptive.** In the
user's own words:

> "I DO NOT need them to reflect the current code. It's the opposite. I
> need these documents to define what the process IS in order to CORRECT
> the actual code files to what they should be."

Accordingly, this document defines the *correct* mathematical treatment of
marginal Gaussianization and the implicit Gaussian copula for this
pipeline. It is not a walkthrough of `src/train/copula.py` as an
authoritative reference, and there is no "relation to the current
pipeline" section presenting that file's existing behavior as correct.
`src/train/copula.py` is used below only as source material for a
mechanism worth adopting — a well-designed empirical-CDF Gaussianization
algorithm — described here as this document's own prescriptive algorithm.
Two narrow, clearly marked asides note where the current code departs from
the correct treatment; neither is this document's frame.

**Error 2 — no flattening, ever, and no cross-variable stacking.** In the
user's own words:

> "Don't treat the entirety of the variables as a single 4D tensor of size
> (lat, lon, time, num_variables); instead, treat each of them as a single
> 3D tensor (lat, lon, time). Assume that the land mask is already
> applied... and NEVER flatten them from this 3 dimensional
> representation. They should NOT be converted to a 2D matrix of (lat *
> lon, time) nor of (lat, lon * time) or any other combination. Flattening
> has the disadvantage of losing local information, so NEVER flatten."

For *this* document, per-cell Gaussianization is naturally compatible with
this rule: fitting one empirical-CDF lookup table (LUT) per `(lat, lon)`
location, per variable, is inherently a per-cell operation — there is no
version of it that needs `(lat, lon)` collapsed into a single cell-index
axis. But it is easy to get this subtly wrong in implementation, so the
exact line is stated explicitly in §4: computing the fit/transform via a
single vectorized numpy call that broadcasts over the full `(lat, lon)`
grid — reducing only over the time axis, leaving `(lat, lon)` as intact
leading axes throughout — is **not** the forbidden flattening; it is a
vectorization convenience, mathematically identical to looping over every
cell one at a time. What **is** forbidden is ever collapsing `(lat, lon)`
into one flat cell-index axis (e.g. to build a single `(n_cells, n_bins)`
LUT array), and stacking multiple variables into a shared array before or
during this step. Every variable is Gaussianized entirely independently,
on its own native `(lat, lon, time)` tensor.

---

## 1. Summary

Every marginal (per-variable, per-grid-cell) distribution in this dataset
is non-Gaussian: wave heights and speeds are positive and right-skewed,
periods are positive and bounded to a physically plausible band, and
directional Cartesian components inherit the shape of a speed–angle
product. Downstream stages of this pipeline (Tucker decomposition, DMD,
mode-level Ornstein–Uhlenbeck (OU) recovery) are linear, Gaussian-driven
machinery: they are well-founded when acting on jointly Gaussian data and
are not distribution-free.

The fix is the classical probability-integral-transform trick, applied
per grid cell and per variable, entirely independently: replace each raw
physical value with its rank under a nonparametric estimate of its own
marginal CDF (an empirical-CDF lookup table, fit independently at every
`(lat, lon)` cell, for every variable, ignoring all other cells and all
other variables), then push that rank through the standard normal
quantile function $\Phi^{-1}$. This is *marginal* Gaussianization only —
it does not fit, and is not intended to fit, any joint dependence
structure. Sklar's theorem is what licenses treating "marginal shape" and
"dependence structure" as separable problems in the first place; the
empirical-CDF LUT is a nonparametric estimator of each marginal CDF; and
the reason the overall pipeline nonetheless behaves as an *implicit
Gaussian copula* is a closure property of the multivariate Gaussian family
under linear maps, not anything asserted at the Gaussianization step
itself. That implicit choice has a known, derivable cost: zero joint upper
tail dependence, meaning simulated extreme co-occurrences across cells
will be systematically too rare.

Given Error 2, every variable's Gaussianization is a fully independent
1-variable problem on that variable's own `(lat, lon, time)` tensor. The
open design question this document leaves unresolved (§6) is therefore not
"how do we avoid mixing variables together" (settled: never) but *where
along that one variable's own processing chain* the Gaussianization step
should sit — on the raw physical anomalies, on that variable's own
Tucker-core coefficients, or on that variable's own DMD mode
amplitudes/residuals.

---

## 2. Theoretical foundation

### 2.1 The probability integral transform

**Lemma (PIT).** Let $X$ be a real random variable with continuous CDF
$F$. Then $U := F(X)$ is uniformly distributed on $(0,1)$.

*Proof.* For $u \in (0,1)$, continuity and strict monotonicity of $F$ on
the support of $X$ give a well-defined inverse there, so
$$
P(U \le u) = P(F(X) \le u) = P\big(X \le F^{-1}(u)\big) = F\big(F^{-1}(u)\big) = u,
$$
which is exactly the CDF of $\mathrm{Unif}(0,1)$. $\blacksquare$

This is the entire mechanism by which "marginal shape" can be split off
from "dependence": applying $F$ (whatever its shape — skewed, bounded,
heavy-tailed) to $X$ always destroys the shape and leaves a uniform
variable, and applying $\Phi^{-1}$ (the standard normal quantile function)
to that uniform variable always produces a standard normal variable,
regardless of what $F$ was. Composing the two,
$$
Z := \Phi^{-1}\big(F(X)\big) \sim \mathcal N(0,1),
$$
is a strictly increasing (since both $F$ and $\Phi^{-1}$ are strictly
increasing on their relevant domains) transform of $X$ that reshapes its
marginal to standard normal while preserving its rank order exactly.

### 2.2 Sklar's theorem

**Theorem (Sklar, 1959).** Let $H$ be an $n$-dimensional CDF with margins
$F_1,\ldots,F_n$. There exists a copula $C : [0,1]^n \to [0,1]$ such that
$$
H(x_1,\ldots,x_n) = C\big(F_1(x_1),\ldots,F_n(x_n)\big) \qquad \forall\, x \in \mathbb R^n. \tag{2.1}
$$
If every $F_i$ is continuous, $C$ is unique. Conversely, for any copula
$C$ and any margins $F_1,\ldots,F_n$, the right-hand side of (2.1) is a
valid $n$-dimensional CDF with exactly those margins.

*Proof sketch (continuous-margins case).* Let $U_i := F_i(X_i)$; by the
PIT each $U_i \sim \mathrm{Unif}(0,1)$. Define
$$
C(u_1,\ldots,u_n) := H\big(F_1^{-1}(u_1),\ldots,F_n^{-1}(u_n)\big),
$$
using the ordinary inverse (well-defined since each $F_i$ is continuous
and strictly increasing on its support). By construction $C$ is exactly
the joint CDF of $(U_1,\ldots,U_n)$ — a function with uniform margins
inheriting the measure-nonnegativity of $H$ on rectangles — i.e. a genuine
copula. Substituting $u_i = F_i(x_i)$ recovers (2.1). Uniqueness follows
because continuity makes the range of each $F_i$ all of $[0,1]$, so the
defining equation pins $C$ down everywhere on the unit cube, not merely on
a sparse image set. $\blacksquare$

The content of Sklar's theorem for this pipeline: $C$ is, by definition,
the joint law of the *Gaussianized* (or, more generally, uniformized)
variables. Fitting marginals and fitting the copula are two genuinely
separate statistical problems, and — crucially for §2.3 — the copula $C$
of a random vector is **invariant under any strictly increasing
transform applied separately to each coordinate**: if $T_1,\ldots,T_n$ are
strictly increasing, $(T_1(X_1),\ldots,T_n(X_n))$ has the same copula as
$(X_1,\ldots,X_n)$, because a copula is a statement about ranks, and
monotone transforms preserve ranks. In particular, since
$Z_i = \Phi^{-1}(F_i(X_i))$ is such a transform for every $i$,
$$
\operatorname{Cop}(X_1,\ldots,X_n) = \operatorname{Cop}(Z_1,\ldots,Z_n). \tag{2.2}
$$
Gaussianizing every marginal, on its own, changes nothing about the
dependence structure — it only ever reshapes each margin to standard
normal. This identity is what makes §2.3 rigorous rather than heuristic.

### 2.3 A correct empirical-CDF Gaussianization algorithm

The true marginal CDF $F_i$ at a given `(lat, lon)` cell, for a given
variable, is unknown. The standard nonparametric estimator is the
**empirical CDF**: given a sample $x_{(1)} \le x_{(2)} \le \cdots \le
x_{(T)}$ (the order statistics of that cell's time series, land/masked
and missing values excluded),
$$
\hat F_T(x) = \frac{1}{T}\sum_{t=1}^{T} \mathbb 1\{x_t \le x\}.
$$
The Glivenko–Cantelli theorem gives $\sup_x |\hat F_T(x) - F(x)| \to 0$
almost surely as $T \to \infty$, and the Dvoretzky–Kiefer–Wolfowitz
inequality gives an explicit exponential concentration rate — this is the
formal justification for treating $\hat F_T$ as a usable stand-in for the
unknown $F_i$.

The algorithm below is a **quantile-binned** version of the empirical
CDF's inverse (the empirical quantile function), which is what makes it
tractable to invert exactly and cheap to vectorize. It happens to closely
match a well-designed existing implementation (`src/train/copula.py`,
whose docstring this section's mechanism is built to justify), but it is
stated here as its own, independently-derivable algorithm.

**Fit.** Choose $n_{\text{bins}}$ probability levels on a regular grid,
e.g. $p_k = (k + \tfrac12)/n_{\text{bins}}$ for $k = 0,\ldots,n_{\text{bins}}-1$
(a symmetric "plotting position," avoiding placing a bin exactly at the
boundary probabilities $0$ or $1$). For a given cell and variable, compute
the empirical quantile at each $p_k$ from that cell's own time series —
this is exactly $n_{\text{bins}}$ order-statistic-based estimates of
$F_i^{-1}(p_k)$. Store the ascending sequence of $(p_k, \hat x_k)$ pairs as
that cell's LUT.

**Transform (forward, Gaussianize).** Given a new raw value $x$ at that
cell: locate the two adjacent fitted bin values bracketing $x$ (binary
search over the ascending LUT), linearly interpolate between their
probabilities to get an estimated $\hat u = \hat F(x) \in (0,1)$, clip
away from the exact boundaries $\{0,1\}$ (to keep $\Phi^{-1}$ finite), and
return
$$
z = \Phi^{-1}(\hat u).
$$
Values outside the fitted range are handled by clamping the interpolation
to the nearest edge bin, so this direction never diverges even for values
beyond the training record's extremes.

**Inverse (recover a physical value).** Given $z$: compute $\hat u =
\Phi(z)$, then linearly interpolate *within the LUT's value axis* between
the two bins bracketing $\hat u$'s probability, returning a raw value.
Because interpolation only ever occurs between the smallest and largest
fitted quantiles, this direction can **never produce a value outside the
training range** — a direct, useful consequence of using a genuine
empirical-CDF LUT rather than a smooth parametric marginal fit (e.g. a
fitted log-normal or gamma), which would not offer this guarantee.

This LUT/interpolation/inversion mechanism is doing nothing more than an
efficient, vectorized realization of "estimate $F_i$ nonparametrically per
cell, then apply $\Phi^{-1}\!\circ \hat F_i$" — precisely the object
constructed in §2.1–§2.2.

### 2.4 Why this is a *Gaussian* copula, despite no correlation matrix ever being fit

Nothing in §2.3 fits a joint density, a copula-density formula, or a
correlation matrix. It transforms each marginal, independently, cell by
cell, variable by variable. The claim that the overall pipeline is
nonetheless a **Gaussian copula** model rests on the following chain, not
on anything asserted during Gaussianization:

1. **Gaussianization preserves the copula exactly.** By (2.2), the copula
   of the physical variables equals the copula of their Gaussianized
   images $Z_i = \Phi^{-1}(F_i(X_i))$. Marginal Gaussianization is
   copula-neutral by construction.

2. **The Gaussian family is closed under linear maps.** If $Z \sim \mathcal
   N(\mu,\Sigma)$ in $\mathbb R^n$, its characteristic function is
   $\varphi_Z(t) = \mathbb E[e^{i t^\top Z}] = \exp\!\big(i t^\top \mu -
   \tfrac12 t^\top \Sigma t\big)$. For an affine map $Y = AZ + b$,
   $$
   \varphi_Y(t) = e^{i t^\top b}\,\varphi_Z(A^\top t)
   = \exp\!\Big(i t^\top (A\mu + b) - \tfrac12 t^\top (A \Sigma A^\top) t\Big),
   $$
   which is again a Gaussian characteristic function. Hence $Y \sim
   \mathcal N(A\mu+b,\, A\Sigma A^\top)$: **any** linear/affine
   transformation of a jointly Gaussian vector is again jointly Gaussian.
   This is exactly why Tucker's multilinear projections, DMD's linear
   propagator, and OU's linear SDE driven by Gaussian white noise all
   preserve joint-Gaussianity of whatever Gaussian object they act on.

3. **Composing 1 and 2.** If the downstream chain (Tucker → DMD → mode-OU,
   or any subset/reordering of linear, Gaussian-innovation-driven stages)
   models the Gaussianized coefficients — at whatever stage they are
   introduced — as jointly Gaussian with some induced correlation
   structure $\Sigma$, then by point 2 every linear function of them
   downstream (including the reconstructed $Z$ at the original
   Gaussianization point) is also jointly Gaussian, with a correlation
   matrix determined entirely by the composed linear operators. Call this
   implied correlation matrix $\Sigma$. By point 1, the copula of the
   *physical* variables $X$ equals the copula of $Z$, and the copula of a
   jointly Gaussian vector with correlation matrix $\Sigma$ is, by
   definition,
   $$
   C_\Sigma^{\text{Gauss}}(u_1,\ldots,u_n) = \Phi_\Sigma\big(\Phi^{-1}(u_1),\ldots,\Phi^{-1}(u_n)\big),
   $$
   where $\Phi_\Sigma$ is the joint CDF of $\mathcal N(0,\Sigma)$. Hence
   $$
   \operatorname{Cop}(X_1,\ldots,X_n) = C_\Sigma^{\text{Gauss}}.
   $$

No step in this chain ever writes down $\Sigma$ explicitly or fits it via
a sample-correlation formula — $\Sigma$ is whatever the composition of
linear, Gaussian-driven stages happens to imply. But the *mathematical
content* of "marginal-CDF-transform each variable independently, then let
downstream linear-Gaussian machinery define all cross-cell/cross-variable
structure" is, precisely, "assume the joint law is Sklar's decomposition
with a Gaussian copula and nonparametric margins." That equivalence is
what "implicit Gaussian copula" means here, and it is exact, not
approximate — conditional on the downstream machinery actually behaving as
linear-Gaussian end to end (which is itself an assumption those stages'
own documentation is responsible for, not this one).

### 2.5 Zero tail dependence

**Definition.** For a copula $C$ with uniform margins $U,V$, the *upper
tail dependence coefficient* is
$$
\lambda_U := \lim_{u \to 1^-} P(V > u \mid U > u).
$$
$\lambda_U \in (0,1]$ means extreme events in the two variables tend to
co-occur; $\lambda_U = 0$ means they become asymptotically independent in
the extremes, however strong the bulk correlation is.

**Result.** For the bivariate Gaussian copula with correlation $\rho$,
$\lambda_U = 0$ for every $\rho < 1$ (Sibuya, 1959).

*Derivation sketch.* Let $(X,Y)$ be standard bivariate normal with
correlation $\rho$. Write $Y = \rho X + \sqrt{1-\rho^2}\,W$ with $W \sim
\mathcal N(0,1)$ independent of $X$. Fix a large threshold $x$ and
condition on $X > x$; for large $x$, the conditional law of $X$ given
$X>x$ concentrates near $x$ itself (the Gaussian density decays like
$e^{-x^2/2}$, so the conditional distribution above the threshold is
sharply peaked at its left edge — an exponential-tilting/Laplace-method
argument). Then, asymptotically,
$$
P(Y > x \mid X > x) \approx P\!\Big(W > x\sqrt{\tfrac{1-\rho}{1+\rho}}\Big) = \bar\Phi\!\Big(x\sqrt{\tfrac{1-\rho}{1+\rho}}\Big),
$$
where $\bar\Phi = 1-\Phi$. Taking $x \to \infty$ (equivalent to $u \to
1^-$ under $u = \Phi(x)$): if $\rho < 1$, the coefficient
$\sqrt{(1-\rho)/(1+\rho)} > 0$, so the argument of $\bar\Phi$ diverges and
$$
\lambda_U = \lim_{x\to\infty} \bar\Phi\!\Big(x\sqrt{\tfrac{1-\rho}{1+\rho}}\Big) = 0.
$$
Only at $\rho = 1$ does the coefficient vanish, giving $\bar\Phi(0) =
\tfrac12$ and (via the exact known closed form $\lambda_U = 2\bar\Phi(x_0
\sqrt{(1-\rho)/(1+\rho)})$ in the $x\to\infty$ limit) $\lambda_U = 1$ —
perfect dependence, the degenerate case.

**Contrast: the $t$-copula.** For a bivariate $t$-copula with $\nu$
degrees of freedom and correlation $\rho$ (Embrechts, McNeil & Straumann,
2001),
$$
\lambda_U = 2\,\bar t_{\nu+1}\!\Big(\sqrt{(\nu+1)\,\tfrac{1-\rho}{1+\rho}}\Big) > 0 \quad \text{for any finite } \nu \text{ and } \rho > -1.
$$
The qualitative reason this does *not* vanish where the Gaussian case did:
the conditioning variable is no longer light-tailed. A Student-$t$ tail
decays polynomially ($P(X>x) \sim c\,x^{-\nu}$, regularly varying), not
like $e^{-x^2/2}$; the same conditioning argument used above yields a
**fixed, finite** argument for the companion variable's tail probability
as $x \to \infty$ (rather than one that itself diverges), so the limiting
ratio stays strictly positive. This is the formal reason a $t$-copula has
tunable positive tail dependence (via $\nu$) while a Gaussian copula has
none, for any $\rho < 1$.

This is the direct, derivable version of the correct, load-bearing piece
of theory already present in `src/train/copula.py`'s own docstring: *"a
Gaussian copula has zero joint tail dependence, so simulated storm
co-occurrences across cells will be too rare and too mild."* That
statement is genuine theory, worth citing as such — not a description of
this repository's pipeline choices, but a property of the Gaussian-copula
family itself, which any implicit-Gaussian-copula design (§2.4) inherits
regardless of implementation details.

---

## 3. Guidelines: variable-class handling

Every field in this dataset falls into one of three classes with respect
to how per-cell empirical-CDF Gaussianization behaves. In every case the
treatment is the same procedure (§2.3) applied independently per
`(lat, lon)` cell and per variable; what differs is what the LUT is
implicitly buying you, and what it is not.

### Class A — unsigned magnitude scalars

`sigwh` (significant wave height), `wsh`/`pwh`/`pswh` (wind-sea, peak,
primary-swell heights), `was` (wind speed), `ocs` (current speed), `sst`
(sea-surface temperature, signed but a scalar in the same sense).

These are bounded below by zero (except `sst`) and typically right-skewed
— storm-driven heavy right tails over an otherwise moderate bulk. The
per-cell empirical-CDF LUT handles this "for free": no distributional
family needs to be chosen, boundedness at zero is enforced automatically
by the LUT's inversion never producing a value below its smallest fitted
quantile, and skewness is absorbed entirely by the (in general
non-uniform) spacing of the fitted quantile bins — no transform parameter
(e.g. a Box–Cox $\lambda$) needs to be estimated or tuned.

### Class B — periods

`wsp`, `pwp`, `pswp` (wind-sea, peak, primary-swell periods).

Also positive and typically right-skewed, and the LUT mechanism handles
this identically to Class A. The caveat specific to this class: each
period is fit and Gaussianized **entirely independently** of its
partition's paired height and direction (e.g. `wsh`, `wsp`, `wsd` are
physically coupled through fetch/wave-age relationships within the
wind-sea partition, but nothing in per-variable Gaussianization sees that
coupling — each of the three is its own independent 1-D problem, per
Error 2). Under the design corrected in §0, there is consequently no
step anywhere in this document's scope that models within-partition
height/period/direction coupling; whatever correlation exists among them
must be recovered, if at all, by whatever correlation structure the
downstream per-variable linear machinery ends up implying for each of
these now-separate Gaussian channels (§2.4) — which is not guaranteed to
reproduce the true physical coupling. This is a genuine, currently
unaddressed gap, not a solved problem deferred elsewhere in this document.

### Class C — directional Cartesian components

`(u,v)` pairs derived from each `(magnitude, direction)` pair before
Gaussianization: `(u_wind, v_wind)` from `(was, wad)`, `(u_wsea, v_wsea)`
from `(wsh, wsd)`, `(u_peak, v_peak)` from `(pwh, pwd)`, `(u_swell,
v_swell)` from `(pswh, pswd)`, `(u_current, v_current)` from `(ocs, ocd)`.

Each component, once formed, is treated as an ordinary signed scalar field
on its own `(lat, lon, time)` tensor and Gaussianized independently exactly
as in Class A/B — nothing about the LUT mechanism itself cares that $u$
and $v$ were jointly derived from a magnitude and an angle.

**Known open subtlety.** Before Gaussianization, $u$ and $v$ satisfy the
exact algebraic identity $speed = \operatorname{hypot}(u,v)$ at every
cell and time. Because $u$ and $v$ are Gaussianized via two *separate*
per-cell empirical CDFs (each a monotone but different nonlinear
regrading of its own marginal), this identity does not survive into
$(Z_u, Z_v)$-space: $\operatorname{hypot}(Z_u, Z_v)$ is not a meaningful
transform of physical speed, and there is no guarantee that, after
downstream sampling and inverse-transforming $Z_u, Z_v$ back to physical
$(u,v)$, the reconstructed $\operatorname{hypot}(u,v)$ matches whatever
speed marginal was actually fit for that partition. Whether this
discrepancy is small in practice (bounded by how strongly the downstream
linear stage happens to correlate $Z_u$ and $Z_v$) or physically
significant is an open empirical question, not resolved by anything in
this document.

---

## 4. Correct treatment for this dataset's tensors

For each variable $v$ independently, with land already masked to NaN, the
native representation is a single 3D tensor
$$
X_v \in \mathbb R^{n_{\text{lat}} \times n_{\text{lon}} \times n_{\text{time}}}.
$$
No other variable, and no flattened reshaping of this tensor, ever enters
the fit or transform for $v$.

**Composes independently with the `log` pre-transform.** For height/period
variables, `variable_taxonomy.md` §3.4 applies a `log` transform ($\log E$,
$T$) *before* this document's machinery runs, regardless of which
placement (i/ii/iii, §6.1) is chosen for Gaussianization itself — "where
along the chain Gaussianization sits" and "whether a physical `log`
pre-transform happens at the very start" are independent design axes. In
particular, this does not weaken the case for Gaussianization: rank-based
Gaussianization (empirical CDF → inverse-normal) is invariant to any
strictly monotonic pre-transform of its own input, so `log`'s only effect
here is on what $X_v$ numerically *is* by the time it reaches whichever
stage is chosen — it changes nothing about this section's fit/transform
mechanics themselves.

**Fit.** Compute, for a chosen set of bin probabilities $p_0,\ldots,p_{n_{\text{bins}}-1}$,
$$
\text{LUT}_v[k, i, j] = \widehat{F_{v,i,j}}^{-1}(p_k),
$$
the empirical quantile at probability $p_k$ of the time series
$X_v[i,j,:]$, ignoring NaNs. This is computed via a single vectorized call
that reduces only over the time axis and leaves `(lat, lon)` as intact
leading axes of the result, e.g. (schematically) `nanquantile(X_v,
[p_0,...,p_{n_bins-1}], axis=time)`, producing a LUT tensor of shape
`(n_bins, n_lat, n_lon)`.

**The broadcast-vs-flatten distinction, stated explicitly.** This single
vectorized call computes *exactly* the same $n_{\text{lat}} \times
n_{\text{lon}}$ independent 1-D quantile fits that an explicit double loop
over `i` and `j` would compute one cell at a time — no information from
cell $(i,j)$ ever enters the computation at cell $(i',j') \ne (i,j)$.
Vectorizing this way is a performance optimization, not a change to what
is being computed, and `(lat, lon)` remain distinct, addressable axes of
the LUT tensor throughout. This is **not** the forbidden operation. The
forbidden operation is reshaping $X_v$ to $(n_{\text{lat}} \cdot
n_{\text{lon}},\, n_{\text{time}})$ (or any other flattening of `(lat,
lon)` into a single cell-index axis) before or during this step, e.g. to
produce a LUT literally shaped `(n_bins, n_cells)`. Nothing about the
per-cell empirical-CDF mathematics requires or benefits from that
flattening — it is strictly a representational regression, discarding the
grid's 2D adjacency structure for no computational gain, and is exactly
the pattern Error 2 rules out categorically.

**Transform / inverse.** Applied per variable, per cell, exactly as in
§2.3, using each cell's own `LUT_v[:, i, j]` — again vectorized over the
`(lat, lon)` leading axes, never flattened.

*Note on current code (marked aside, not this document's framing).*
`src/train/copula.py`'s `fit`/`transform` reshape each variable's
`(time, lat, lon)` array to `(time, n_cells)` via `.reshape(shape[0],
-1)` before fitting — precisely the flattening pattern ruled out above,
even though the loop over variables is already correctly independent per
Error 2. This is a concrete instance of the gap this document exists to
correct, not a structural choice to defer to.

---

## 5. Results

**What the fitted per-cell LUT represents.** For a fixed variable $v$ and
cell $(i,j)$, `LUT_v[:, i, j]` is a nonparametric point estimate of that
cell's own marginal quantile function $F_{v,i,j}^{-1}$, evaluated at
$n_{\text{bins}}$ fixed probability levels, built entirely from that
cell's own finite (non-NaN) time record. It carries no information about,
and is not influenced by, any other cell or any other variable.

**The forward/inverse round trip.** Forward: a raw value is located
between two adjacent fitted quantiles (interpolated to an estimated
probability), then mapped through $\Phi^{-1}$ to a standard normal value.
Inverse: $\Phi$ recovers the probability, and the same LUT is
interpolated in the opposite direction to recover a raw value, bounded to
the training range by construction. Applying forward then inverse (or
vice versa) to a value already consistent with the fitted LUT recovers it
up to the interpolation error between adjacent bins — exact at the fitted
quantile points themselves, with error bounded by local bin spacing
elsewhere, and controllable by the choice of $n_{\text{bins}}$.

**A well-calibrated per-cell QQ-plot.** For a held-out (or in-sample, as a
fitting-quality check) time series at cell $(i,j)$, plot the empirical
quantiles of $Z_{v,i,j} = \Phi^{-1}(\hat F_{v,i,j}(X_{v,i,j}))$ against the
theoretical quantiles of $\mathcal N(0,1)$. A well-calibrated fit lies
close to the identity line $y=x$ across the bulk of the distribution, with
the largest deviations expected at the extreme tails — where order
statistics are sparsest and the bin-interpolation and boundary-clamping
behavior have the most leverage — and where any finite-$n_{\text{bins}}$
discretization is most visible. Systematic bow-shaped departure from $y=x$
across the whole range (rather than only in the extremes) indicates too
few bins, an undersized fitting record, or a genuine violation of this
cell's assumed continuity (e.g. an unmasked land/ice boundary cell
mixing two physically distinct regimes).

---

## 6. Open questions

### 6.1 Where should Gaussianization sit, per variable?

Because Error 2 rules out any cross-variable joint tensor, each variable's
processing chain — Gaussianize, Tucker-compress, DMD, mode-OU — is now a
fully independent 1-variable pipeline. The open question this document
leaves unresolved is *where along that one chain* the marginal
Gaussianization step (§2.3) should be applied. Three placements are live:

**(i) On the raw, native per-cell physical anomalies (current placement).**
Gaussianize $X_v \in \mathbb R^{n_{\text{lat}}\times n_{\text{lon}}\times
n_{\text{time}}}$ directly, before any compression, exactly as specified
in §4. "Per-cell" here means an actual physical grid location.

- *For:* maximal interpretability — every LUT corresponds to a real
  $(lat,lon)$ point's physical marginal; boundedness/skewness handling
  (§3) is applied at the point where it is most directly meaningful;
  everything downstream (Tucker, DMD) then operates on already
  well-behaved, Gaussian-margined data from the start, which is exactly
  the assumption those linear stages are best justified under.
- *Against:* this is the highest-resolution, most expensive placement —
  one LUT per physical cell, $n_{\text{lat}} \times n_{\text{lon}}$ of
  them per variable (on the order of $10^5$–$10^6$), each needing enough
  time samples to estimate $n_{\text{bins}}$ quantiles reliably; and the
  transform is applied before any noise/signal separation, so raw
  measurement or grid-cell noise is Gaussianized together with genuine
  physical structure — nothing at this stage can distinguish the two.

**(ii) On that variable's own Tucker-core coefficients.**
Fit a per-variable, 3-mode Tucker decomposition on $X_v$ directly
(ranks $(r_{\text{lat}}, r_{\text{lon}}, r_T)$, no variable-mode at all —
unlike a jointly-Tucked design, there is no cross-variable mixing concern
here by construction), yielding a compressed core tensor $G_v \in
\mathbb R^{r_{\text{lat}} \times r_{\text{lon}} \times r_T}$. Gaussianize
each entry of $G_v$ via its own per-entry empirical CDF instead of
Gaussianizing $X_v$ itself.

- *For:* far fewer LUTs to fit ($r_{\text{lat}} r_{\text{lon}} r_T \ll
  n_{\text{lat}} n_{\text{lon}} n_{\text{time}}$); operates in a basis
  DMD/OU already expect to be compact and well-conditioned; each LUT is
  built from more effective samples relative to the complexity it
  describes.
- *Against:* **"per-cell" stops meaning a physical grid location.** A
  Tucker-core coefficient is a linear combination, via the fitted factor
  matrices, across many original `(lat, lon)` grid points for that one
  variable — so there is still a cross-*spatial-mode* mixing concern (even
  though there is no cross-*variable* mixing concern, since each
  variable's Tucker fit is entirely independent per Error 2). Marginally
  Gaussianizing a coefficient that is itself a weighted sum over many
  physical cells has a much less direct physical reading than
  Gaussianizing a physical anomaly, and the calibration diagnostic (§5)
  becomes harder to interpret physically. There is also a genuine
  redundancy question: a linear combination over many grid cells is,
  by a rough central-limit argument, already pulled some distance toward
  Gaussian before this step ever runs — so explicit Gaussianization here
  may be doing comparatively little for coefficients dominated by broad,
  well-sampled spatial patterns, while still doing real work for
  coefficients where a small number of extreme (storm) events dominate
  the linear combination and leave visible skew/kurtosis.

**(iii) On that variable's own DMD mode amplitudes/residuals.**
Push Gaussianization further downstream still, to the AR(1)/OU-residual
sequence $\eta_k(t)$ in DMD mode space (or to the mode amplitudes
directly), rather than to the raw field or the Tucker core.

- *For:* this is the object the mode-OU formulas actually need to be
  (approximately) Gaussian — its $\varphi_{\text{mode}}$,
  $\theta_{\text{mode}}$, $\sigma_{\text{mode}}$ recovery is most directly
  justified when its innovations are genuinely Gaussian white noise, so
  Gaussianizing as close as possible to the point where Gaussianity is
  actually required is the most targeted fix, and it feeds the
  OU-innovation requirement most directly of the three options.
- *Against:* by this point, two linear compressions (Tucker, DMD) have
  already run on **non**-Gaussianized data, so any prior stage's
  non-Gaussianity has already propagated through SVD-based fits that are
  optimal in an $L^2$/Gaussian-noise sense but not distribution-free —
  fixing the marginal only at the very end does not undo that. "Per-cell"
  here has moved even further from a physical grid location — the
  relevant index is a mode number $k$ (further complicated by
  complex-conjugate mode pairs: Gaussianizing a complex amplitude raises
  its own sub-question of whether to treat real/imaginary parts as two
  independent real LUTs or modulus/phase as one magnitude LUT plus a
  circular/directional one, reintroducing the Class C direction-handling
  issue from §3 in a new guise). And critically, this placement gives up
  the direct guarantee of §5: the raw physical field's marginal shape is
  now only preserved to the extent that Tucker's and DMD's linear
  reconstruction maps, applied to a Gaussianized residual, happen to
  reproduce the correct non-Gaussian physical marginal on the way back
  out — which is not automatic and would need its own empirical
  validation (a physical-space QQ-plot after the *full* round trip, not
  just a mode-space one).

| | (i) raw physical anomalies | (ii) Tucker-core coefficients | (iii) DMD mode residuals |
|---|---|---|---|
| "Per-cell" means | physical `(lat,lon)` location | linear combination over many `(lat,lon)` cells | mode index (possibly complex) |
| # LUTs per variable | $n_{\text{lat}} n_{\text{lon}}$ (largest) | $r_{\text{lat}} r_{\text{lon}} r_T$ (small) | $r_{\text{dmd}}$ (smallest) |
| Physical interpretability of each LUT | direct | indirect (spatial-mode mixing) | indirect (mode/phase) |
| Redundancy risk (CLT-type argument) | none | possible for broad, well-sampled modes | possible, compounded by prior stages |
| Directness for OU's Gaussian-innovation need | indirect | more direct | most direct |
| Guarantees on recovered *physical* marginal shape | exact (by construction) | only after Tucker inverse | only after full Tucker+DMD inverse |

This document does not resolve which placement is correct — that is a
live design decision requiring empirical validation (e.g. comparing
physical-space QQ-plots and reconstructed extreme statistics under each
placement) that has not been done.

**Update (2026-09-13): (i) vs. (ii) decided — see `open_questions.md` §4.1.**
Task 07's era-2 rank sweep ran (i) and (ii) at the same Tucker rank and found
them statistically indistinguishable (50.81% vs. 53.04% error at 5% rank) —
the linear-rank-inflation concern this section raised about per-cell
Gaussianization ahead of a linear compression stage (§2.4's point that
copula-invariance under monotone transforms says nothing about invariance of
the linear/Pearson structure Tucker actually uses) turned out not to be the
dominant effect on this data. Placement (ii) was still adopted as canonical,
on the architectural grounds in `open_questions.md` §4.1's update rather than
this empirical near-tie: it keeps Tucker in physically-grounded log-energy
anomaly space and moves Gaussianization to sit immediately before DMD, the
one stage that actually requires Gaussian innovations.

### 6.2 Relation to the tail-dependence fix path

The zero-tail-dependence result of §2.5 is a property of the *Gaussian*
copula specifically, independent of where in the chain (i, ii, or iii
above) the Gaussianization happens to sit — every placement above still
composes, per §2.4, into an implicit Gaussian copula for the physical
variables, as long as the downstream machinery stays linear and
Gaussian-driven. A separately documented, also-unresolved decision (the
physical-consistency-constraints discussion) is *what family* to use
instead of a pure Gaussian copula — e.g. replacing the Gaussianization
step with a $t$-copula construction (tunable $\lambda_U > 0$ via a single
degrees-of-freedom parameter, §2.5) or splicing empirical-CDF bulk
behavior to Generalized Pareto Distribution tails per cell (Coles, 2001)
— as the root-cause fix for insufficient joint tail dependence and
physical-consistency violations, rather than patching either downstream.

These are two independent open axes of the same design space: *where*
(this document, §6.1) and *what kind of copula* (the physical-consistency
document). A resolution on one axis does not resolve the other — a
$t$-copula/GPD-tail construction could in principle be placed at the raw
physical field, at the Tucker core, or at the DMD residual, exactly
mirroring the three-way tradeoff above.
