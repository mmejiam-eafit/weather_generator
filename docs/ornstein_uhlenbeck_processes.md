# Ornstein-Uhlenbeck Processes

Scope: the Ornstein-Uhlenbeck (OU) SDE, its exact discrete-time (AR(1))
solution, its complex-valued extension, and the two places this repository's
generator needs it — the per-mode residual around a variable's own DMD
trajectory, and the per-cell residual around a variable's own coherent
(Tucker + DMD) field reconstruction. This document defines what the OU
treatment **should be**; it is a specification to build/correct code
against, not a description of any particular function's current body.

## 0. Two errors this document corrects

This document replaces an earlier draft that made two errors. Both are
corrected here explicitly so the mistake isn't quietly repeated.

**Error 1 — this document is prescriptive, not descriptive.** A prior draft
included a "relation to the current pipeline" section that presented
`src/train/dmd.py`'s `fit_mode_ou` and `src/train/local_residual.py`'s
`fit_local_residual` as the authoritative definition of what OU fitting
*is* in this codebase. That got the direction of authority backwards. In
the author's own words: *"I DO NOT need them to reflect the current code.
It's the opposite. I need these documents to define what the process IS in
order to CORRECT the actual code files to what they should be."* Both
functions currently assume an upstream tensor design (a joint,
cross-variable, spatially-raveled state) that is itself wrong — see Error 2
— so treating them as ground truth would have propagated that error into
the theory. Below, the OU process is derived and applied on its own terms.
Where current code differs from the corrected treatment, it is called out
in a clearly marked aside, never as the section's framing.

**Error 2 — the tensor-representation error, specifically for the
local-residual fit.** In the author's own words: *"Don't treat the entirety
of the variables as a single 4D tensor of size (lat, lon, time,
num_variables); instead, treat each of them as a single 3D tensor (lat,
lon, time). Assume that the land mask is already applied ... and NEVER
flatten them from this 3 dimensional representation. They should NOT be
converted to a 2D matrix of (lat * lon, time) nor of (lat, lon * time) or
any other combination. Flattening has the disadvantage of losing local
information, so NEVER flatten."* Concretely for the OU fit this means: the
local-residual layer fits one independent OU process per spatial cell per
variable, on the residual $R = Z - Z_{\text{coherent}}$, where **both** $Z$
and $Z_{\text{coherent}}$ are that one variable's own native
`(lat, lon, time)` tensor — never raveled into a `(cell, time)` matrix, and
never stacked with other variables into a 4-tensor. Section 4 below gives
the concrete fitting procedure and explains precisely why a fully
vectorized fit over the whole `(lat, lon)` grid is *not* the forbidden
flattening.

> **Where current code differs (informational, not the framing of this
> document).** `src/train/local_residual.py::fit_local_residual` currently
> takes `Z`/`Z_coherent` typed `(T, N_ocean, V)` — time-first, but with
> `(lat, lon)` already raveled into a single `N_ocean` axis and all
> variables `V` stacked together — and `src/train/dmd.py::fit_mode_ou`
> currently fits from a DMD model built jointly across variables. Both
> assume the upstream joint/raveled tensor design that Error 2 rejects.
> Bringing these two functions in line with Sections 3-4 below (native
> per-variable `(lat, lon, time)` tensors, per-variable DMD/mode index) is
> exactly the correction this document exists to drive.

## 1. Summary

Every additive stochastic layer in this generator is, at its core, the same
object: a **mean-reverting Gaussian (or complex-Gaussian) diffusion** fit to
a *residual* — whatever a deterministic model failed to capture. The
Ornstein-Uhlenbeck process is the unique (up to parameterization)
continuous-time process with this property: linear drift back toward a
mean, constant diffusion, Gaussian stationary law, exponentially decaying
autocorrelation. It is fit **twice**, at two different scales, and the two
fits should be understood as instances of one piece of theory rather than
two different techniques:

| | Mode-level OU | Local-residual OU |
|---|---|---|
| Fit to | residual of one variable's own DMD mode amplitude around its deterministic trajectory $\hat a_k(t) = b_{0,k}e^{\omega_k t}$ | residual of one variable's own field $Z(\mathrm{lat},\mathrm{lon},t)$ around its coherent (Tucker+DMD) reconstruction |
| Domain | complex, one scalar time series per DMD mode $k$ | real, one scalar time series per `(lat, lon)` cell |
| Indexed by | mode index $k = 1,\dots,r_{\mathrm{dmd}}^{(v)}$, per variable $v$ | spatial cell $(\mathrm{lat}, \mathrm{lon})$, per variable $v$ |
| "Small" or "large" | small: $r_{\mathrm{dmd}}^{(v)}$ is tens of numbers | large: one process per ocean grid cell |
| Section | 4.2 | 4.1 |

Both fits reduce, exactly, to the same discrete-time inference problem:
observe a (possibly complex) time series believed to be OU-distributed at a
fixed sampling interval $\Delta t$, and recover $(\theta, \sigma)$ (and,
diagnostically, the half-life and stationary variance they imply). Section
2 derives that inference problem from first principles; Sections 3-5 apply
it correctly to this dataset's tensors; Section 6 flags what is still open.

## 2. Theoretical foundation

### 2.1 The SDE and its explicit solution

The (real, scalar) Ornstein-Uhlenbeck process is defined by

$$
dX_t = -\theta(X_t - \mu)\,dt + \sigma\,dW_t, \qquad \theta > 0,\ \sigma > 0,
$$

with $W_t$ a standard Wiener process. $\theta$ is the mean-reversion rate,
$\mu$ the long-run mean, $\sigma$ the diffusion coefficient (note: with this
parameterization $\sigma$ has units such that $\sigma\,dW_t$ has the same
units as $dX_t$ — i.e. $\sigma$ is *not* the stationary standard deviation;
see 2.3).

**Solving it.** Let $Y_t = X_t - \mu$, so $dY_t = -\theta Y_t\,dt +
\sigma\,dW_t$. Multiply by the integrating factor $e^{\theta t}$ and apply
Itô's product rule (the drift and diffusion of $Y_t$ make this exact, no
extra Itô correction term arises because $e^{\theta t}$ is deterministic):

$$
d\left(e^{\theta t} Y_t\right) = e^{\theta t}\left(dY_t + \theta Y_t\,dt\right) = \sigma e^{\theta t}\,dW_t.
$$

Integrating from $0$ to $t$:

$$
e^{\theta t} Y_t - Y_0 = \sigma \int_0^t e^{\theta s}\,dW_s
\quad\Longrightarrow\quad
Y_t = Y_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)}\,dW_s,
$$

so the explicit solution is

$$
X_t = \mu + (X_0 - \mu)e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)}\,dW_s. \tag{2.1}
$$

The deterministic part decays exponentially toward $\mu$ with rate
$\theta$; the stochastic part is a Wiener integral against an exponentially
*decaying* kernel (most recent noise counts most), which is why $X_t$ is
Gaussian for every $t$ (a linear functional of Gaussian increments) even
though it is not itself Markov-simple like Brownian motion — it has finite
memory set by $1/\theta$.

### 2.2 Stationary distribution

Since the stochastic integral in (2.1) is a mean-zero Gaussian (Wiener
integral of a deterministic integrand), its variance follows from the Itô
isometry, $\mathrm{Var}\left(\int_0^t f(s)\,dW_s\right) = \int_0^t f(s)^2\,ds$:

$$
\mathrm{Var}(X_t) = \sigma^2\int_0^t e^{-2\theta(t-s)}\,ds = \sigma^2\cdot\frac{1 - e^{-2\theta t}}{2\theta}.
$$

As $t \to \infty$ (or if $X_0$ is already drawn from stationarity), this
converges to

$$
X_\infty \sim \mathcal N\!\left(\mu,\ \frac{\sigma^2}{2\theta}\right). \tag{2.2}
$$

This is the process's **stationary/marginal** distribution — the
unconditional law of $X_t$ once transients have died out. It will be
important below (2.5) not to confuse this variance with the *innovation*
variance of the discretized process.

### 2.3 Autocovariance

For $s < t$ at stationarity, write $t = s + \Delta t$ and split (2.1)'s
stochastic term at $s$:

$$
X_t - \mu = (X_s - \mu)e^{-\theta\Delta t} + \sigma\int_s^t e^{-\theta(t-u)}\,dW_u.
$$

The integral term depends only on Wiener increments after time $s$, hence
is independent of $X_s$ (which depends only on increments up to $s$). So

$$
\mathrm{Cov}(X_s, X_t) = e^{-\theta\Delta t}\,\mathrm{Var}(X_s) = \frac{\sigma^2}{2\theta}e^{-\theta\Delta t}. \tag{2.3}
$$

This is the defining diagnostic signature of an OU process: variance
$\sigma^2/2\theta$ at lag $0$, decaying geometrically in continuous time
with rate $\theta$ — i.e. **exponential** autocorrelation
$\rho(\Delta t) = e^{-\theta\Delta t}$, never any other shape (no
oscillation, no power-law tails) for the real scalar process.

### 2.4 The exact discrete-time transition — OU is exactly AR(1)

This is the fact that makes fitting practical: an OU process observed only
at a fixed grid $t, t+\Delta t, t+2\Delta t, \dots$ is not *approximately*
an AR(1) process — it is **exactly** one, for any $\Delta t$, with no
discretization error.

Take (2.1) at $t + \Delta t$ conditional on $X_t$ and split the stochastic
integral at $t$ exactly as in 2.3:

$$
X_{t+\Delta t} = \mu + \phi\,(X_t - \mu) + \varepsilon_t, \qquad \phi := e^{-\theta\Delta t}, \tag{2.4}
$$

$$
\varepsilon_t := \sigma\int_t^{t+\Delta t} e^{-\theta(t+\Delta t - u)}\,dW_u.
$$

*(Notational note: $\phi$ throughout this document denotes the scalar
discrete-time AR(1)/OU coefficient — standard time-series notation. This is
an unrelated, purely coincidental symbol overload with the sibling
`dynamic_mode_decomposition.md`'s $\phi_k$, which denotes a DMD spatial
mode vector. Keep the two apart by context: nothing in that document has a
$\Delta t$ exponent, and nothing here indexes a spatial mode.)*

$\varepsilon_t$ is mean-zero Gaussian (again a Wiener integral of a
deterministic kernel), and by the Itô isometry, substituting $v = t+\Delta
t - u$:

$$
\mathrm{Var}(\varepsilon_t) = \sigma^2\int_t^{t+\Delta t} e^{-2\theta(t+\Delta t-u)}\,du
= \sigma^2\int_0^{\Delta t} e^{-2\theta v}\,dv
= \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta\Delta t}\right)
= \frac{\sigma^2}{2\theta}\left(1 - \phi^2\right). \tag{2.5}
$$

Because $\varepsilon_t$ depends only on Wiener increments strictly after
$t$, it is independent of $X_t$ (and, by the same argument applied at every
step, independent across non-overlapping intervals — $\{\varepsilon_t\}$ is
i.i.d. across the sampling grid). Equations (2.4)-(2.5) say exactly:

> **An OU process sampled at interval $\Delta t$ is a stationary Gaussian
> AR(1) process** with autoregressive coefficient $\phi = e^{-\theta\Delta
> t} \in (0,1)$ and i.i.d. Gaussian innovations of variance
> $\frac{\sigma^2}{2\theta}(1-\phi^2)$.

**This is exact, not Euler-Maruyama.** The naive Euler-Maruyama
discretization of the SDE, $X_{t+\Delta t} \approx X_t - \theta(X_t-\mu)\Delta
t + \sigma\sqrt{\Delta t}\,\xi_t$, is only first-order accurate — it agrees
with (2.4)-(2.5) solely in the limit $\Delta t \to 0$ (where $\phi \to
1-\theta\Delta t$ and the innovation variance $\to \sigma^2\Delta t$,
recovering the Euler step). At the data's actual sampling interval — not
infinitesimal — Euler-Maruyama's $\phi$ and innovation variance are both
biased relative to (2.4)-(2.5), with the bias growing as $\theta\Delta t$
grows. Because (2.4)-(2.5) hold *exactly* for any $\Delta t$, they — not
Euler-Maruyama — are the correct transition to both simulate forward from
and to invert when fitting from data.

**Why this matters for fitting.** Because (2.4) is exactly linear
regression of $X_{t+\Delta t}-\mu$ on $X_t - \mu$, $(\phi, \mathrm{Var}(\varepsilon))$
can be estimated directly from an observed time series by an AR(1) fit, and
then inverted for the physical parameters:

$$
\hat\theta = -\frac{\ln\hat\phi}{\Delta t}, \qquad
\hat\sigma^2 = \hat{\mathrm{Var}}(\varepsilon)\cdot\frac{2\hat\theta}{1-\hat\phi^2}. \tag{2.6}
$$

Every fit in this document (Section 4) is an application of exactly (2.6),
possibly with $\phi$, $\varepsilon$, $\theta$, $\sigma$ complex-valued
(Section 2.6) and possibly computed for many independent series at once via
broadcasting (Section 4).

### 2.5 The critical pitfall: innovation variance $\neq$ marginal variance

Equation (2.6) requires the **innovation** variance
$\mathrm{Var}(\varepsilon_t)$ — the variance of the *one-step-ahead
prediction error* $\varepsilon_t = X_{t+\Delta t} - \mu - \phi(X_t - \mu)$ —
not the raw marginal (stationary, unconditional) variance of the observed
series, $\mathrm{Var}(X)$ from (2.2). These two are related but not
interchangeable, and the relationship is exactly (2.5) rearranged:

$$
\mathrm{Var}(X) = \frac{\sigma^2}{2\theta}, \qquad
\mathrm{Var}(\varepsilon) = \frac{\sigma^2}{2\theta}(1-\phi^2)
\quad\Longrightarrow\quad
\boxed{\ \mathrm{Var}(X) = \frac{\mathrm{Var}(\varepsilon)}{1-\phi^2}\ }. \tag{2.7}
$$

A student can re-derive (2.7) directly from the AR(1) recursion alone,
without any SDE machinery: at stationarity $X_{t+\Delta t}-\mu = \phi(X_t -
\mu) + \varepsilon_t$ with $\varepsilon_t$ independent of $X_t$, so taking
variances of both sides and using $\mathrm{Var}(X_{t+\Delta t}) =
\mathrm{Var}(X_t) = \mathrm{Var}(X)$ at stationarity,

$$
\mathrm{Var}(X) = \phi^2\,\mathrm{Var}(X) + \mathrm{Var}(\varepsilon)
\quad\Longrightarrow\quad
\mathrm{Var}(X)(1-\phi^2) = \mathrm{Var}(\varepsilon),
$$

which is (2.7). This is a completely general fact about any stationary
AR(1) process, independent of where it came from.

**The mistake to avoid:** if $(\theta$ already estimated via $\hat\phi)$ and
the *marginal* variance $\mathrm{Var}(X)$ are substituted for
$\mathrm{Var}(\varepsilon)$ inside the innovation-variance formula in (2.6),

$$
\hat\sigma^2_{\text{wrong}} = \mathrm{Var}(X)\cdot\frac{2\hat\theta}{1-\hat\phi^2}
\overset{(2.7)}{=} \mathrm{Var}(\varepsilon)\cdot\frac{1}{1-\hat\phi^2}\cdot\frac{2\hat\theta}{1-\hat\phi^2}
= \hat\sigma^2_{\text{correct}} \cdot \frac{1}{1-\hat\phi^2},
$$

i.e. the marginal-variance shortcut **double-counts** the $(1-\phi^2)$
factor — it is already baked into $\mathrm{Var}(X)$ by (2.7), and the
formula in (2.6) applies it a second time. The resulting bias factor on
$\sigma$ (not $\sigma^2$) is

$$
\frac{\hat\sigma_{\text{wrong}}}{\hat\sigma_{\text{correct}}} = \frac{1}{\sqrt{1-\phi^2}}.
$$

This factor is negligible when $\phi \approx 0$ (fast decorrelation — one
step is nearly independent of the last, so innovation variance and marginal
variance nearly coincide) but blows up as $\phi \to 1$ (slow decorrelation
— consecutive observations are nearly redundant, and using the raw spread
of a highly autocorrelated series wildly overstates the *new* randomness
injected per step). Concretely, at $\phi = 0.95$:

$$
\frac{1}{\sqrt{1-0.95^2}} = \frac{1}{\sqrt{0.0975}} \approx 3.20,
$$

a 3x+ overestimate of $\sigma$ (and hence over 10x too much simulated
per-step variance, since it enters some downstream quantities squared) from
this single substitution error. **$\sigma$ must always be estimated from
the innovation variance, never the raw marginal/stationary variance
directly** — see Section 5's checklist.

### 2.6 Complex-valued extension (DMD mode residuals)

DMD mode amplitudes are complex: for mode $k$ with (generally complex)
continuous-time eigenvalue $\omega_k$ (the sibling `dynamic_mode_
decomposition.md`'s notation — the continuous-time rate obtained from the
discrete-time DMD eigenvalue $\lambda_k$ via $\omega_k = \ln(\lambda_k)/\Delta
t$), the deterministic trajectory is $\hat a_k(t) = b_{0,k}\,e^{\omega_k
t}$, and the residual $\eta_k(t) =
a_k(t) - \hat a_k(t)$ (measured amplitude minus deterministic prediction)
is a complex-valued time series believed to itself be a mean-reverting
diffusion around $0$. The natural model is a **complex OU process**:

$$
d\eta_t = -\Theta\,\eta_t\,dt + \sigma\,dW_t^{\mathbb C}, \qquad \Theta = \theta + i\omega \in \mathbb C,\ \theta > 0,
$$

where $dW_t^{\mathbb C} = dW_t^{(1)} + i\,dW_t^{(2)}$ is a complex Wiener
increment built from two *independent* standard real Wiener processes,
normalized here so that $\mathbb E\left[dW_t^{\mathbb C}\,\overline{dW_t^{\mathbb
C}}\right] = dt$ and $\mathbb E\left[(dW_t^{\mathbb C})^2\right] = 0$ (circularly
symmetric: the noise has no preferred phase direction in the complex
plane, appropriate since a mode's residual amplitude has no privileged
orientation). $\Theta$ is complex: its real part $\theta$ is a decay rate
exactly as before, but its imaginary part $\omega$ allows the residual to
also rotate in the complex plane (a phase drift relative to the
deterministic trajectory's own rotation, which is already carried by
$\nu_k = \mathrm{Im}(\omega_k)$ in the sibling document's notation).

The entire derivation of 2.1-2.4 goes through unchanged with $\theta \to
\Theta$ and the real Itô isometry replaced by its complex analogue
$\mathrm{Var}(\int f\,dW^{\mathbb C}) = \int |f|^2\,du$ (valid under the
circular-symmetry normalization above). The exact discrete transition is

$$
\eta_{t+\Delta t} = \phi\,\eta_t + \varepsilon_t, \qquad \phi = e^{-\Theta\Delta t} \in \mathbb C, \tag{2.8}
$$

$$
\mathrm{Var}(\varepsilon_t) := \mathbb E\left[|\varepsilon_t|^2\right] = \frac{\sigma^2}{2\theta}\left(1-|\phi|^2\right), \qquad \mathbb E\left[\varepsilon_t^2\right] = 0, \tag{2.9}
$$

i.e. $\varepsilon_t$ is a circularly-symmetric complex Gaussian — exactly
(2.4)-(2.5) with $\theta \to \theta = \mathrm{Re}(\Theta)$ and $\phi^2 \to
|\phi|^2$. All of Section 2.5's pitfall and its $1/\sqrt{1-|\phi|^2}$ bias
factor apply verbatim with $\phi^2$ read as $|\phi|^2$.

**Decay versus rotation.** Writing the principal complex logarithm of $\phi$
as $\log\phi = \log|\phi| + i\arg\phi$ (with $\arg\phi \in (-\pi,\pi]$),
and since $\phi = e^{-\Theta\Delta t} = e^{-\theta\Delta t}e^{-i\omega\Delta
t}$:

$$
\theta = \mathrm{Re}(\Theta) = -\frac{\mathrm{Re}(\log\phi)}{\Delta t} = -\frac{\log|\phi|}{\Delta t}, \qquad
\omega = \mathrm{Im}(\Theta) = \frac{\arg\phi}{\Delta t}\ \ (\text{mod } 2\pi/\Delta t). \tag{2.10}
$$

$\mathrm{Re}(\log\phi)$ and $\log|\phi|$ are the *same number* ($\mathrm{Re}(\log
z) = \log|z|$ for any nonzero complex $z$) — two equivalent routes to the
same decay rate, and both are deliberately **distinct from** $\arg\phi$,
which carries only the residual's phase-rotation frequency and says nothing
about how fast its magnitude decorrelates. A mode-OU fit that only cares
about "how quickly does this residual forget itself" wants $\theta$ (or
equivalently $|\phi|$) alone; $\omega$ is a separate, physically distinct
diagnostic (a leftover rotation the DMD eigenvalue's own $\nu_k =
\mathrm{Im}(\omega_k)$ didn't fully capture), not a component of decay.
Because $\arg\phi$ is only
recoverable modulo $2\pi$, $\omega$ is only identifiable up to aliasing at
$|\omega| < \pi/\Delta t$ — the usual Nyquist limit of the sampling
interval, worth remembering before over-interpreting a fitted rotation.

*(Practical aside, not a correctness claim: some implementations, when
clipping $\phi$ for numerical safety before taking a log, clip and log
$\mathrm{Re}(\phi)$ directly rather than $|\phi|$, specifically so that a
negative real part — indicating step-to-step sign alternation — is pushed
to the fast-decorrelation end rather than assigned whatever slower decay
$|\phi|$ alone would imply for a mode that is alternating rather than
smoothly rotating. That is a defensible numerical-safety choice about how
to handle a degenerate case, layered on top of the decomposition in (2.10);
it does not change what $\theta$ and $\omega$ mean when $\phi$ is
well-behaved.)*

## 3. Guidelines: variable-class handling

The OU math in Section 2 is **variable-class agnostic**: $\theta$, $\sigma$,
$\phi$ and every derivation above operate purely on whatever Gaussianized
coordinate $Z$ the residual is measured in — a real- or complex-valued
number believed (by construction of the upstream transform) to behave like
a stationary Gaussian process. The fitting and simulation code has no
branch on "this is a directional field" versus "this is a magnitude field";
it is one formula applied uniformly.

What *does* depend on variable class is how the fitted parameters and
simulated trajectories should be **interpreted**, and how they must be
**inverse-transformed** back to physical units before that interpretation
is meaningful:

| Variable class | Example physical quantity | Interpretation caveat |
|---|---|---|
| Circular/directional | a direction, decoded to its correct meteorological convention | Gaussian-space excursions correspond to angular displacement; a large simulated jump combined with the marginal transform's wraparound behavior can correspond to a small or large true angular change depending on where on the circle the mean sits. Mean-reversion in $Z$-space is reversion toward the transform's central angle, not necessarily toward $0°$. |
| Non-negative magnitude | a wave height, wind speed, or similar strictly-positive quantity | The physical-space stationary distribution implied by a symmetric Gaussian in $Z$-space is generally **skewed** after inverse transform (e.g. log-normal-like), so a half-life computed in $Z$-space ($\ln 2/\theta$) is time-symmetric, but the corresponding physical-space excursions are not — an upward and downward swing of equal $Z$-space size are not equal-magnitude in physical units. |
| Signed scalar/anomaly | an already-anomaly or already-symmetric physical quantity | Closest to the "native" Gaussian case; least distortion between $Z$-space and physical-space interpretation of $\theta,\sigma$. |

The load-bearing dependency here is **wherever Gaussianization happens
upstream** — which transform, fit at which granularity (per cell? per
variable globally? jointly across variables via a copula?), and whether it
is itself refit alongside the OU layer or held fixed. That question belongs
to the sibling document on Gaussianization/copula transforms
(`notebooks/03_understand_gaussian_transformation_and_copulas.ipynb`,
`notebooks/05_gaussian_transformation_and_copula_on_dmd.ipynb`) and is
treated there as an open design question, not resolved here — see also
Section 6. Everything in this document assumes that transform is already
fixed and its output is the $Z$ this document's residuals are computed
from.

## 4. Correct treatment for this dataset's tensors

Notation: for one physical variable $v$, let $Z_v$ denote its Gaussianized
anomaly field, a native tensor of shape `(T, n_lat, n_lon)` (time-first, per
this codebase's convention), with a boolean land/ice mask of shape
`(n_lat, n_lon)` already applied — land/ice cells hold a fixed sentinel
(e.g. zero) rather than data, and remain at their `(lat, lon)` grid
position; they are never dropped or reindexed (`notebooks/01_understand_
tucker_decomposition.ipynb` documents the mask's construction). Each
variable is handled entirely on its own: no step below ever stacks multiple
variables into a shared axis.

### 4.1 Local-residual OU: broadcast over `(lat, lon)`, never flatten

$Z_{v,\text{coherent}}$ — the deterministic Tucker+DMD reconstruction for
variable $v$ (no mode-OU stochastic term added; see the sibling
`dynamic_mode_decomposition.md`, or reason directly from 4.2 below: it is
the DMD's $\hat a_k(t)$ propagated back through variable $v$'s own Tucker
factors) — has the same native shape `(T, n_lat, n_lon)`. The residual

$$
R_v = Z_v - Z_{v,\text{coherent}} \qquad \text{shape } (T, n_lat, n_lon)
$$

is fit with one independent real OU process **per grid cell**, following
(2.6) applied at every $(\mathrm{lat},\mathrm{lon})$ simultaneously. The
correct implementation is a set of numpy reductions that never touch the
`(n_lat, n_lon)` axes except elementwise, and reduce **only** over the time
axis:

```python
# R: (T, n_lat, n_lon) — one variable's own residual tensor. Never reshaped.
R_prev, R_next = R[:-1], R[1:]                      # (T-1, n_lat, n_lon)

num = np.sum(R_prev * R_next, axis=0)               # (n_lat, n_lon)
den = np.sum(R_prev * R_prev, axis=0) + eps          # (n_lat, n_lon)
phi = np.clip(num / den, -1 + eps, 1 - eps)          # (n_lat, n_lon)

# clip separately before the log: log needs a positive argument, and
# clipping near phi -> 1 avoids theta -> 0 blowing up sigma via 1/(1-phi**2)
theta = -np.log(np.clip(phi, phi_min, phi_max)) / dt  # (n_lat, n_lon)

innovation = R_next - phi[None, :, :] * R_prev        # (T-1, n_lat, n_lon)
innovation_variance = np.mean(innovation ** 2, axis=0)  # (n_lat, n_lon) -- (2.6)'s Var(epsilon), NOT R.var()
sigma = np.sqrt(innovation_variance * 2 * theta / (1 - phi ** 2 + eps))  # (n_lat, n_lon)

var_R = R.var(axis=0)          # (n_lat, n_lon) -- marginal variance, used ONLY for the diagnostic below
var_Z = Z.var(axis=0) + eps    # (n_lat, n_lon)
variance_ratio = var_R / var_Z # (n_lat, n_lon) -- see Section 5
```

repeated independently for each variable $v$ (a small Python-level loop
over variables, each call operating on that one variable's own 3D tensor —
never a 4D `(lat, lon, time, variable)` array, per Error 1).

**Why this is a broadcast, not the forbidden flattening.** Every array
above keeps `(n_lat, n_lon)` as two intact axes end to end. The only
reduction performed is `axis=0` (over time); nothing ever collapses
`(n_lat, n_lon)` into a single combined index, and no cell's identity is
ever merged with another's. This is mathematically and numerically
**exactly equivalent** to an explicit nested loop —

```python
for i in range(n_lat):
    for j in range(n_lon):
        theta[i, j], sigma[i, j], phi[i, j] = fit_ou_1d(R[:, i, j], dt)
```

— producing identical results (up to floating-point summation order,
immaterial). The vectorized form is purely a speed convenience: numpy
dispatches the `axis=0` reductions as compiled loops instead of a Python
double loop, but the *computation* performed for cell $(i,j)$ never sees or
touches any other cell's data. Land cells (per the pre-applied mask) are
simply left with degenerate ($R \approx 0$) statistics at their own
`(lat, lon)` position — the `+eps` guards prevent a `0/0`, and downstream
usage should further gate on the mask directly (Section 5's active-cell
diagnostic naturally comes out near-zero for a masked cell, but the mask
itself, not the diagnostic, is the authoritative reason to skip it).

**This is the exact place vectorizing for speed can accidentally
reintroduce flattening.** The danger is not this reduction — it is
tempting, once cells are held as `(n_lat, n_lon)`, to "simplify" by
reshaping to `(T, n_lat * n_lon)` before running the same reductions, since
the arithmetic gives bit-identical numbers either way (a reshape alone
loses no information). What flattening costs is everything *around* the
arithmetic: (a) every downstream consumer of `theta`/`sigma`/`phi` now
needs an auxiliary index map back to `(lat, lon)`, which is extra
bookkeeping that must be threaded through simulation, diagnostics, and
plotting, and must be rebuilt any time the land mask or grid changes (this
archive already has three different grid/cadence eras, so `N_ocean` is not
even a fixed number across the dataset); (b) a flat index no longer
encodes spatial adjacency — cell $i$ and cell $i+1$ in the flattened axis
are not spatial neighbors — silently foreclosing any later spatially-aware
use of these fields (smoothing a $\theta$-map, neighbor-based diagnostics,
plotting without an unscatter step) without extra unflattening logic; and
(c) nothing in a `(cell, time)` matrix's shape signals "these are
independent parallel channels" the way two preserved `(lat, lon)` axes do —
it reads exactly like the shape of a problem where mixing information
*across* the flattened axis (a covariance, a shared regularizer, a matrix
factorization) would be legitimate, inviting a future edit to accidentally
introduce cross-cell coupling that was never intended. None of that is
about numerical correctness of the specific reductions used above (which
are safe either way) — it is about what the surrounding code and its
future maintainers are, or are not, invited to do next. Keep the two
spatial axes explicit for exactly that reason.

### 4.2 Mode-level OU: per-variable, per-mode, already compressed

By the time a variable's own DMD residual $\eta_k(t)$ exists, it is no
longer indexed by physical `(lat, lon)` at all — the flattening concern in
4.1 does not apply here, for a different reason than "it's been handled
carefully": it has already been through variable $v$'s own Tucker
decomposition, which maps physical space into a small number of compressed
spatial ranks (call them $r_{\mathrm{lat}}, r_{\mathrm{lon}}$, per the
Tucker/DMD documents), and DMD then operates on that compressed core's mode
index $k = 1, \dots, r_{\mathrm{dmd}}^{(v)}$. $\eta_k(t)$ is one scalar
complex time series per mode, and $r_{\mathrm{dmd}}^{(v)}$ is small (tens of
values, not tens of thousands of grid cells) — a genuinely non-spatial (or
abstractly, compressed-index) object, not a reshape of the physical grid.
Fitting all $r_{\mathrm{dmd}}^{(v)}$ modes' complex OU parameters at once
via (2.8)-(2.10), broadcast over the mode axis the same way 4.1 broadcasts
over `(lat, lon)`, is unproblematic for exactly the reasons given in 4.1 —
no information is lost, because there was no spatial identity left to lose
by this stage.

The one design point worth stating explicitly: **this fit is per-variable**.
Each variable $v$ has its own Tucker decomposition, its own DMD, its own
mode index $k$, and therefore its own independent set of
$(\theta_k^{(v)}, \sigma_k^{(v)}, \phi_k^{(v)})$ — there is no joint,
cross-variable mode index shared between variables, consistent with the
per-variable design in the Tucker and DMD documents. A mode-OU fit that
draws on more than one variable's amplitude time series in the same fit
call is out of scope of the corrected design.

## 5. Results: reading the fitted parameters

| Quantity | Formula | Reads as |
|---|---|---|
| $\phi$ | $e^{-\theta\Delta t}$ (real) or $e^{-\Theta\Delta t}$ (complex) | one-step memory: $\phi \to 1$ is a slowly-forgetting (long-memory, "coherent-looking") residual; $\phi \to 0$ is a noise-like, next-to-independent one |
| $\theta$ | $-\ln\phi/\Delta t$ (real) or $-\log\|\phi\|/\Delta t$ (complex, (2.10)) | mean-reversion rate, units of inverse time |
| half-life | $\ln 2/\theta$ | time for the deterministic part of a displacement to decay by half — solve $e^{-\theta t_{1/2}} = 1/2$; the standard, most physically legible way to report $\theta$ |
| $\omega$ (complex only) | $\arg\phi/\Delta t$, (2.10) | residual phase rotation left over after the DMD eigenvalue's own frequency; identifiable only up to $\|\omega\| < \pi/\Delta t$ (Nyquist) |
| $\sigma$ | from *innovation* variance via (2.6)/(2.9), never marginal variance (2.5) | diffusion intensity — sets how much *new* randomness enters per unit time; consistency check: as $\Delta t \to 0$, the fitted per-step innovation std should scale like $\sigma\sqrt{\Delta t}$, matching the SDE's own Brownian scaling |
| stationary variance | $\sigma^2/2\theta$ (2.2), or equivalently innovation variance $/(1-\phi^2)$ (2.7) | the process's own unconditional spread — what a long simulated run should reproduce empirically |
| variance ratio | $\mathrm{Var}(R)/\mathrm{Var}(Z)$, per cell (or per mode) | see below |

**Variance-ratio diagnostic, as a design recommendation.** Simulating one
OU path per grid cell per variable per step is not free, and a cell whose
residual $R$ contributes only a small fraction of that cell's total
variance $Z$ contributes correspondingly little to sample fidelity by being
simulated at all. A sound default is to compute $\mathrm{Var}(R)/\mathrm{Var}(Z)$
per cell and skip simulating (fix at $0$) any cell below a small floor —
this is an **efficiency floor**, not a modeling choice: the OU fit at such
a cell is not wrong, it is simply small enough that omitting it introduces
error below other, already-accepted sources of approximation in the
pipeline (Gaussianization approximation, DMD mode-count truncation,
Monte-Carlo sampling noise itself). Treat the specific threshold as a
tunable efficiency/fidelity tradeoff, not a physically derived constant —
see Section 6.

**Checklist item for every refit.** Whenever this fit is written,
rewritten, or ported, re-verify by construction that the quantity feeding
$\sigma$'s formula is the *innovation* variance — the mean squared one-step
prediction error $\varepsilon_t = R_{t+\Delta t} - \phi R_t$, computed
*after* estimating $\phi$ — and not `R.var(axis=...)` directly (Section
2.5). A cheap empirical check: simulate forward from the fitted
$(\theta,\sigma)$ at equilibrium ($X_0 \sim \mathcal N(\mu, \sigma^2/2\theta)$
or its complex analogue) and confirm the simulated trajectory's own
long-run marginal variance reproduces the *original* residual's empirical
variance — if $\sigma$ was estimated from the marginal variance instead of
the innovation variance, this check will fail by roughly the
$1/\sqrt{1-\phi^2}$ factor derived in 2.5, worse for cells/modes with
larger $\phi$.

## 6. Open questions / limitations

- **Gaussianization-placement dependency.** Every derivation and fitting
  procedure above assumes $Z$ is already in a coordinate system where the
  residual is approximately stationary Gaussian. Exactly where in the
  pipeline that transform is applied, at what granularity, and whether it
  is held fixed while OU parameters are fit or refit jointly with them, is
  an open question the sibling Gaussianization/copula document owns, not
  this one. If that upstream transform changes, every OU parameter derived
  here must be refit, and the variance-ratio floor (Section 5) should be
  re-validated rather than assumed to transfer.

- **Conjugate-pair mode coupling.** DMD complex modes generally arise in
  conjugate pairs $(k, k')$ with $\lambda_{k'} = \overline{\lambda_k}$ and,
  by construction of a real physical field, $\eta_{k'}(t) \equiv
  \overline{\eta_k(t)}$ *exactly* — not merely correlated, but a
  deterministic function of the other mode. Fitting $k$ and $k'$ as two
  independent complex OU processes (as 4.2's per-mode broadcast does) gives
  marginally consistent but not *jointly* consistent parameters: nothing in
  an independent fit enforces that simulating $k$ and $k'$ separately
  reproduces the exact conjugate relationship needed to keep a
  reconstructed physical field real-valued. The two should ultimately be
  treated as one shared stochastic driver (fit once, simulate one member of
  the pair, obtain the other by conjugation) rather than as two
  independently-fit and independently-simulated processes — flagged here as
  unresolved, not yet given a specific fitting procedure.

- **Variance floor as an unvalidated heuristic.** The variance-ratio floor
  in Section 5 is a reasonable default grounded in an efficiency argument,
  but the actual downstream error its omission introduces has not been
  validated end to end (that validation depends on simulation and
  hurricane-season-style checks that are not yet in place). Treat the
  specific numeric threshold as provisional until such a validation exists,
  not as a physically derived cutoff.
