# Dynamic Mode Decomposition (DMD) for Per-Variable Tucker-Core Sequences

## 0. Two errors this document corrects

This document is one of a set that replaces a previously deleted draft. That
draft made two errors, and this document is written specifically to avoid
repeating them.

**This document is prescriptive, not descriptive.** It does not describe
what `src/train/dmd.py` currently does and does not treat that file's
current behavior as ground truth. The opposite: this document defines what
the DMD step *should be*, so that the code can be brought in line with it.
Concretely, `src/train/dmd.py`'s `dmd_fit` today takes a `core: (T, r_ocean,
r_var)` array — a single joint Tucker core spanning **every** modeled
variable at once, produced by an upstream Tucker step that ravels the
ocean-cell (lat, lon) grid into one `ocean_cell` mode and stacks all
variables into a second mode. That upstream shape is itself the wrong
design (see Error 2 and `docs/tucker_decomposition.md`), and DMD built on
top of it inherits the error: a joint cross-variable state that this
document does not endorse. Everywhere below, "the Tucker core" means one
physical variable's own core, never a joint one. Where the current code
differs from what follows, that is a defect in the code to fix, not a
variant this document is reconciling itself to.

**Tensors are per-variable and 3D; nothing gets flattened to reach that
point.** Each physical variable — significant wave height, a wind
component, SST, whatever — lives in its own tensor of shape `(lat, lon,
time)`, land-masked (ocean cells only; land/ice handled via
Tucker's `mask=`-based iterative imputation, as in
`notebooks/01_understand_tucker_decomposition.ipynb`, not by raveling them
out of existence). Variables are never stacked into a fourth mode, and
`lat`/`lon` are never raveled together into a single spatial axis. This
matters for DMD specifically, and it creates a genuine tension that this
document does not paper over: **classical DMD's algorithm is, by
definition, built on a snapshot *matrix*.** At each time step, DMD needs
the entire instantaneous state as one column vector, so it can assemble a
matrix of snapshots side by side and run an SVD-based regression across
time. Even granting a per-variable, never-raveled Tucker core of shape
`(r_lat, r_lon, T)`, applying exact DMD in its classical form still
requires collapsing each time slice's `(r_lat, r_lon)` coefficient grid
into a single length-`r_lat · r_lon` vector before the snapshot matrix can
be built.

Section 4.2 below addresses this honestly rather than resolving it
one way or the other. In brief: this is a **narrower and different** kind
of flattening than the one the tensor-representation correction forbids —
it reshapes an already-compressed, small `(r_lat, r_lon)` *coefficient*
grid whose local spatial adjacency has already been captured separately by
Tucker's own per-mode factor matrices, not the original high-resolution
physical `(lat, lon)` grid whose adjacency structure the correction is
protecting. That distinction is real and worth being deliberate about, not
something to wave away by analogy to Error 2 alone, and this document does
not tell the eventual code which side to land on.

---

## 1. Summary

Dynamic Mode Decomposition (DMD) is a data-driven, equation-free method for
extracting a set of spatially coherent, temporally oscillating/growing/
decaying "modes" from a sequence of snapshots of a dynamical system. It is
the numerical workhorse for approximating the spectrum (eigenvalues and
eigenfunctions) of the **Koopman operator** — an infinite-dimensional but
*linear* operator that governs the exact evolution of observables of a
possibly nonlinear system. That linearity is the entire reason DMD is
useful here: the underlying ocean/atmosphere dynamics are nonlinear, but
DMD fits a best-linear-approximation operator directly from data, in the
coordinate system the upstream pipeline hands it — in this repository,
that coordinate system is a **single physical variable's own Tucker-core
sequence**, never a joint multi-variable state.

Given that per-variable core, DMD produces:

- a small set of discrete-time eigenvalues $\lambda_k \in \mathbb{C}$,
  convertible to continuous-time rates $\omega_k = \ln(\lambda_k)/\Delta t$
  whose real part is a growth/decay rate and whose imaginary part is an
  angular frequency;
- corresponding spatial modes $\phi_k$ (coefficient patterns in the
  Tucker-core's compressed coordinates);
- initial amplitudes $b_0$ that weight each mode's contribution to the
  observed initial condition.

Downstream, this repository fits a per-mode stochastic (Ornstein-Uhlenbeck)
correction on top of the DMD backbone's residual — that step is not this
document's subject, but its correctness depends on DMD being fit correctly
first, which is what this document defines.

---

## 2. Theoretical foundation

### 2.1 Nonlinear dynamics and the Koopman operator

Consider a discrete-time, autonomous, generally **nonlinear** dynamical
system on a state space $\mathcal M \subseteq \mathbb R^n$:

$$
x_{t+1} = F(x_t), \qquad x_t \in \mathcal M, \quad F: \mathcal M \to \mathcal M .
$$

$F$ is not assumed linear, and in general no finite-dimensional linear
operator reproduces its action on $x$ exactly. The Koopman operator
sidesteps this by changing *what* we track: instead of the state $x_t$
itself, track scalar **observables** — functions $g: \mathcal M \to
\mathbb C$ drawn from some function space $\mathcal G$ (e.g. $L^2(\mathcal
M)$), rich enough to include, for instance, each coordinate function $g_i(x)
= x_i$.

Define the **Koopman operator** $\mathcal K : \mathcal G \to \mathcal G$ by
composition with the flow map:

$$
(\mathcal K g)(x) \;=\; (g \circ F)(x) \;=\; g\big(F(x)\big).
$$

$\mathcal K$ is **linear in $g$** regardless of whether $F$ is linear:

$$
\mathcal K(\alpha g_1 + \beta g_2) = (\alpha g_1 + \beta g_2)\circ F = \alpha\, (g_1\circ F) + \beta\, (g_2 \circ F) = \alpha\, \mathcal K g_1 + \beta\, \mathcal K g_2 .
$$

This is the entire trick: composition is linear in the function being
composed even when the map it's composed with is not. The price is that
$\mathcal K$ acts on an infinite-dimensional function space even when
$\mathcal M$ itself is finite-dimensional.

Along a trajectory, applying $\mathcal K$ repeatedly advances the
observable in lock-step with the state:

$$
g(x_{t+1}) = g(F(x_t)) = (\mathcal K g)(x_t) \quad\Longrightarrow\quad g(x_t) = (\mathcal K^{t} g)(x_0).
$$

Suppose $\mathcal K$ has eigenfunctions $\varphi_k \in \mathcal G$ with
eigenvalues $\lambda_k \in \mathbb C$: $\mathcal K \varphi_k = \lambda_k
\varphi_k$. If the vector-valued observable of interest — in particular
the identity observable $g(x) = x$ — can be written in this eigenbasis,

$$
x = \sum_k \varphi_k(x)\, v_k ,
$$

for some fixed (Koopman-mode) vectors $v_k$, then evolving forward in time
is *exactly linear in the eigenfunction coordinates*, no matter how
nonlinear $F$ is:

$$
x_t = \sum_k \big(\mathcal K^t \varphi_k\big)(x_0)\, v_k = \sum_k \lambda_k^{\,t}\, \varphi_k(x_0)\, v_k .
$$

This is the theoretical justification for using a linear spectral method
on nonlinear geophysical dynamics: we are not linearizing $F$ around a
fixed point (that would only be locally valid); we are working in a
(possibly infinite-dimensional) space of observables in which the
evolution operator is exactly linear, globally, by construction.

**Where DMD fits in.** In practice we cannot compute $\mathcal K$'s full
spectrum on an infinite-dimensional space. Exact DMD is the special case of
this program where the chosen observables are simply the state coordinates
themselves — here, a single variable's Tucker-core coordinates — and we
ask for the best-fit **finite-dimensional linear operator** $A$ such that
$x_{t+1} \approx A x_t$ over the observed data. When the state's own
coordinates already span (or nearly span) a Koopman-invariant subspace,
DMD's eigenvalues/modes are consistent estimators of true Koopman
eigenvalues/modes; when they don't, DMD instead returns the best rank-
truncated *linear approximation* of the system's evolution in the chosen
coordinates — still useful, but its fidelity is bounded by how well those
coordinates happen to linearize the dynamics. This caveat is why upstream
Gaussianization quality matters so much (§3).

### 2.2 Exact DMD, derived step by step

Let $x_t \in \mathbb C^{n}$ denote the vectorized per-variable Tucker-core
snapshot at time $t$ (see §4.1 for exactly how $x_t$ is formed from the
core; for this section, take it as given). Given $T+1$ snapshots $x_0,
x_1,\dots,x_T$, form the two shifted snapshot matrices

$$
X = \big[\,x_0,\; x_1,\; \dots,\; x_{T-1}\,\big] \in \mathbb C^{n\times T}, \qquad
X' = \big[\,x_1,\; x_2,\; \dots,\; x_T\,\big] \in \mathbb C^{n\times T},
$$

so that column $j$ of $X'$ is the one-step-ahead image of column $j$ of
$X$. We seek the operator $A \in \mathbb C^{n\times n}$ minimizing

$$
A = \arg\min_{\tilde A} \| X' - \tilde A X \|_F \;=\; X' X^{+},
$$

where $X^+$ is the Moore–Penrose pseudoinverse of $X$. $A$ is exactly the
discrete-time linear operator that best explains the observed one-step
transitions, in a least-squares sense — the finite-dimensional analogue of
$\mathcal K$ restricted to the identity observable.

$n$ (here $n = r_{\mathrm{lat}}\cdot r_{\mathrm{lon}}$, §4.1) may still be
too large to form and eigendecompose $A$ directly, and $X$ is typically
rank-deficient ($T \ll n$ or vice versa). Both problems are handled by
working through the (truncated) SVD of $X$:

$$
X = U\,\Sigma\,V^{*}, \qquad U \in \mathbb C^{n\times r}, \;\; \Sigma \in \mathbb R^{r\times r}_{>0}\text{ diagonal}, \;\; V\in \mathbb C^{T\times r},
$$

truncated to a rank $r = r_{\mathrm{dmd}}$ chosen well below $\min(n, T)$
(§4.3). Then $X^+ = V\Sigma^{-1}U^*$, so

$$
A = X' V \Sigma^{-1} U^{*}.
$$

$A$ is $n\times n$ and still expensive/ill-conditioned to touch directly,
but its restriction to the $r_{\mathrm{dmd}}$-dimensional subspace spanned
by $U$ (the dominant POD modes of the data) is cheap:

$$
\tilde A \;=\; U^{*} A U \;=\; U^{*} X' V \Sigma^{-1} \quad \in \mathbb C^{r_{\mathrm{dmd}}\times r_{\mathrm{dmd}}}.
$$

Eigendecompose the small matrix: $\tilde A W = W \Lambda$, with $\Lambda =
\mathrm{diag}(\lambda_1,\dots,\lambda_{r_{\mathrm{dmd}}})$ and $W$'s columns
the corresponding eigenvectors.

**Exact DMD modes.** Define

$$
\Phi = X'\, V \Sigma^{-1} W \quad \in \mathbb C^{n \times r_{\mathrm{dmd}}}.
$$

These are genuine eigenvectors of the *full* operator $A$, not merely of
its projection $\tilde A$ — which is what makes them "exact" (Tu et al.
2014) rather than merely projected ($UW$ would be the projected-mode
alternative). The proof is a short, satisfying computation:

$$
A\Phi = \big(X'V\Sigma^{-1}U^{*}\big)\big(X'V\Sigma^{-1}W\big)
= X'V\Sigma^{-1}\underbrace{\big(U^{*}X'V\Sigma^{-1}\big)}_{=\ \tilde A}W
= X'V\Sigma^{-1}\,\tilde A W
= X'V\Sigma^{-1}\,W\Lambda
= \Phi\Lambda .
$$

So $A\Phi=\Phi\Lambda$: the columns of $\Phi$ are eigenvectors of $A$ with
eigenvalues $\Lambda$, exactly.

**Initial amplitudes.** Express the first snapshot in the mode basis,
$x_0 \approx \Phi b_0$, i.e. $b_0 = \Phi^{+} x_0$ (pseudoinverse, since
$\Phi$ is $n \times r_{\mathrm{dmd}}$ and generally neither square nor
orthogonal). The DMD reconstruction/forecast at any time $t$ is then

$$
x_t \;\approx\; \Phi\, \Lambda^{t}\, b_0 \;=\; \sum_{k=1}^{r_{\mathrm{dmd}}} \phi_k\, \lambda_k^{\,t}\, b_{0,k},
$$

the direct finite-dimensional analogue of the Koopman expansion in §2.1,
with $\lambda_k \leftrightarrow$ approximate Koopman eigenvalues, $\phi_k
\leftrightarrow$ approximate Koopman modes, and $b_{0,k}$ playing the role
of $\varphi_k(x_0)$.

### 2.3 Discrete-time eigenvalues vs. continuous-time rates

The $\lambda_k$ above are **discrete-time** eigenvalues, tied to whatever
sampling interval $\Delta t$ separates consecutive snapshots. Assuming the
underlying (locally linearized, in the observable space chosen) generator
is a continuous-time linear system $\dot x = \mathcal A x$ sampled every
$\Delta t$,

$$
x_{t+\Delta t} = e^{\mathcal A \Delta t} x_t \quad\Longrightarrow\quad \lambda_k = e^{\omega_k \Delta t} \quad\Longrightarrow\quad \omega_k = \frac{\ln \lambda_k}{\Delta t}.
$$

Writing $\omega_k = \sigma_k + i\,\nu_k$:

- $\sigma_k = \mathrm{Re}(\omega_k)$ is a **growth rate** ($\sigma_k>0$) or
  **decay rate** ($\sigma_k<0$), in inverse time units;
- $\nu_k = \mathrm{Im}(\omega_k)$ is an **angular frequency** (radians per
  unit time); the corresponding period is $2\pi/|\nu_k|$.

Two caveats worth keeping in view: (i) $\ln(\cdot)$ has a branch cut, so
$\omega_k$ is only recoverable modulo $2\pi i/\Delta t$ — frequencies above
the Nyquist limit $\pi/\Delta t$ alias onto lower ones, exactly as in any
sampled-data spectral estimate; (ii) $|\lambda_k| = 1$ corresponds to
$\sigma_k = 0$ — a mode that neither grows nor decays, i.e. sustained
oscillation. $|\lambda_k|>1$ (equivalently $\sigma_k > 0$) is a *growing*
mode, physically implausible as a description of a statistically
stationary ocean/atmosphere state over a long horizon and worth treating as
a diagnostic red flag (§5) rather than accepted at face value.

### 2.4 Complex-conjugate eigenvalue pairs and traveling waves

The per-variable Tucker-core coefficients are real-valued: $x_t \in
\mathbb R^n$ (a subset of $\mathbb C^n$). $X$ and $X'$ are therefore real
matrices, hence so are their SVD factors $U, \Sigma, V$, hence so is
$\tilde A = U^{*}X'V\Sigma^{-1}$ — up to floating-point rounding, $\tilde A$
is a **real** matrix, even though we may carry it in complex arithmetic.

A standard linear-algebra fact then applies: a real matrix's
characteristic polynomial has real coefficients, so any non-real root comes
paired with its complex conjugate. Concretely, if $\tilde A \phi = \lambda
\phi$ with $\tilde A$ real and $\lambda \notin \mathbb R$, conjugating both
sides gives $\tilde A \bar\phi = \bar\lambda\,\bar\phi$ (using
$\overline{\tilde A} = \tilde A$) — so $\bar\lambda$ is also an eigenvalue,
with eigenvector $\bar\phi$. Non-real eigenvalues of a real operator
**necessarily** arise in conjugate pairs $(\lambda, \bar\lambda)$ with
conjugate mode pairs $(\phi,\bar\phi)$; this is not an empirical tendency,
it is forced by $\tilde A$ (and hence $A$) being real.

Physically, this is exactly what makes the reconstructed signal real: a
conjugate pair's joint contribution to $x_t$ is

$$
\phi_k \lambda_k^{t} b_{0,k} + \bar\phi_k \bar\lambda_k^{t}\bar b_{0,k} = 2\,\mathrm{Re}\!\big(\phi_k \lambda_k^{t} b_{0,k}\big),
$$

a real oscillatory term, envelope $e^{\sigma_k t}$ modulating an oscillation
at angular frequency $\nu_k$, spatially patterned by $\mathrm{Re}(\phi_k)$/
$\mathrm{Im}(\phi_k)$ — the data-driven analogue of a **traveling wave** or
standing normal mode: a spatial pattern that oscillates in place (or
propagates, if the real/imaginary spatial patterns are phase-shifted copies
of one another) at a fixed frequency while growing or decaying at a fixed
rate. In the wave/atmosphere context this is precisely the behavior one
hopes DMD recovers: diurnal or synoptic-band oscillations, near-inertial
motions, swell-decay envelopes.

**Design choice to make explicit at simulation time.** The pairing above is
guaranteed *at fit time*, in the exact eigendecomposition of $\tilde A$.
It is a separate question whether that pairing is **preserved at
simulation/sampling time**, once a stochastic correction (e.g. a per-mode
Ornstein–Uhlenbeck residual layer, fit on top of the DMD backbone) is
sampled independently per mode. If two conjugate-paired modes' stochastic
perturbations $\eta_k(t)$ and $\eta_{k'}(t)$ are drawn independently at each
step rather than forced to satisfy $\eta_{k'}(t) = \overline{\eta_k(t)}$,
the reconstructed field can drift complex/lose its realness, which is
unphysical. Whether to **enforce** the conjugate constraint by construction
at every simulated step, or to let it hold only in expectation/fit
statistics and not per-realization, is a genuine design decision for
whatever code performs sampling (e.g. this repository's `weather.simulate`)
to make explicitly and document — this document does not resolve it (see
§6), only flags that it must be resolved somewhere, deliberately.

### 2.5 Relation to PDE normal-mode analysis

For a linear PDE (e.g. the linearized shallow-water or wave-action
equations underlying much of ocean-wave dynamics), separation of variables
yields normal modes: spatial eigenfunctions oscillating/decaying at fixed
temporal rates, found analytically from the linear operator's spectrum.
DMD is the empirical, data-driven analogue of that same idea applied to a
system that is (a) only partially or not at all analytically tractable,
and (b) genuinely nonlinear at the level of the raw physical fields. §2.1's
Koopman framing is what licenses calling DMD's output "modes" in the same
spirit as PDE normal modes: DMD does not linearize the PDE around a base
state in the classical sense, it fits the exactly-linear evolution operator
of a chosen set of observables (here, one variable's Tucker-core
coordinates), which is faithful to the true nonlinear dynamics exactly to
the extent that those observables happen to (near-)span a Koopman-invariant
subspace — a data-driven analogue, not a rigorous PDE spectral method
applied to the literal field equations.

---

## 3. Guidelines: variable-class handling

DMD is **class-agnostic**. By the time DMD sees any input, upstream steps
(Gaussianization/standardization, then a per-variable Tucker fit) have
already mapped whatever the raw physical variable was — a scalar (SST,
significant wave height), a period, or a Cartesian component obtained from
a magnitude/direction decomposition (subject to this archive's direction
conventions: `wsd` and `wad_come_from` correctly decode as
"coming from"; `wad`/`ocd` decode "going to" and must not be treated as
"coming from"; `pwd`/`pswd` are assumed "coming from" pending expert
confirmation) — into one real-valued `(r_lat, r_lon, T)` Tucker core. DMD
operates purely on that abstract numeric coordinate sequence; it has no
notion of what a coordinate physically "means," and needs none.

This genericity is simultaneously a feature and a risk:

| Aspect | Consequence |
|---|---|
| One DMD implementation serves every variable | No variable-specific code path is needed in `dmd_fit` itself. |
| DMD cannot detect upstream mis-preparation | A variable whose Gaussianization was poor, or whose direction convention was decoded backwards, will not raise an error in DMD — it will silently degrade the fit (spurious growth rates, mode mixing, poor reconstruction), diagnosable only after the fact (§5), not by DMD refusing bad input. |
| Validity is entirely inherited, not established | DMD's implicit assumption — that the chosen observables (Tucker-core coordinates) evolve near-linearly — is far more defensible when those coordinates are already close to jointly Gaussian/stationary. Heavy-tailed, bounded, or circular raw variables that were not adequately Gaussianized upstream (or wave/wind angles that were decoded with the wrong convention before being turned into Cartesian components) show up in DMD as unmodeled nonlinearity: degraded reconstruction error, unstable ($|\lambda_k|>1$) modes that shouldn't be there, or physically implausible frequencies. |
| Whether Tucker left time intact matters to what "a snapshot" even is | This document assumes Tucker compresses only the `lat`/`lon` modes and leaves the `time` mode at full resolution $T$ (as in `notebooks/01_understand_tucker_decomposition.ipynb`), so the per-variable core is a length-$T$ sequence of `(r_lat, r_lon)` snapshots — exactly what DMD needs to form snapshot pairs across the *full* observed time resolution. If a future Tucker revision also truncates the time mode, "a snapshot" would instead live in a compressed time basis, which changes what a DMD snapshot pair means and is out of scope for this document — flag explicitly if that ever changes. |

Periods, scalars, and Cartesian vector components all enter DMD on
identical footing: real floating-point numbers, per grid cell, per time
step, within one variable's own core. There is no special-casing to add,
and none should be added — any variable-specific handling belongs upstream
(Gaussianization, direction decoding), never inside `dmd_fit`.

---

## 4. Correct treatment for this dataset

### 4.1 Per-variable DMD on that variable's own Tucker-core sequence

For a single physical variable $v$, the upstream (per `docs/tucker_decomposition.md`)
Tucker fit produces factor matrices $U_{\mathrm{lat}}^{(v)} \in \mathbb
R^{N_{\mathrm{lat}}\times r_{\mathrm{lat}}}$, $U_{\mathrm{lon}}^{(v)} \in
\mathbb R^{N_{\mathrm{lon}}\times r_{\mathrm{lon}}}$, and a core tensor

$$
C^{(v)} \in \mathbb R^{\,r_{\mathrm{lat}} \times r_{\mathrm{lon}} \times T}
$$

(time left at full resolution $T$, per §3). DMD, as defined in this
document, is fit **once per variable, entirely independently** on
$C^{(v)}$ alone. There is no joint state across variables, no `r_var` mode,
no stacking. If a future variant needs cross-variable structure, that is a
different, explicitly-scoped extension (§6) — not something DMD does
implicitly by default.

Each time slice $C^{(v)}_{:,:,t} \in \mathbb R^{r_{\mathrm{lat}} \times
r_{\mathrm{lon}}}$ is the "state" at time $t$ in this variable's own
Tucker-compressed coordinates.

For height/period variables specifically, $C^{(v)}$ traces back to the
`log`-transformed tensor `tucker_decomposition.md` §4.1 now specifies
($\log E$, $T$ — `variable_taxonomy.md` §3.4), not raw physical units.
This is relevant to DMD in particular, beyond just Tucker: `log`
converts the archive's genuinely multiplicative, power-law fetch/duration
growth (§2.1's `correlated_variables.md`-derived motivation) into additive
growth, which is the regime a linear operator fit (this document's whole
subject) is best justified in — a DMD eigenvalue fit directly on raw,
multiplicatively-growing $H^2$ would be fitting a single linear rate to a
process whose absolute growth increment scales with its own current
amplitude, exactly the kind of misspecification this document's rank- and
mode-selection guidance (§4.3, §6) assumes has already been avoided
upstream.

### 4.2 The state-vectorization tension, addressed honestly

Exact DMD (§2.2) is defined on a snapshot **matrix**: it needs each $x_t$
as one column vector so that $X$ and $X'$ can be formed and passed through
an SVD. That means, even having correctly avoided ever raveling the
original `(lat, lon)` physical grid (Error 2), applying classical DMD to
$C^{(v)}$ still requires forming

$$
x_t = \mathrm{vec}\big(C^{(v)}_{:,:,t}\big) \in \mathbb R^{r_{\mathrm{lat}}\cdot r_{\mathrm{lon}}}
$$

for each $t$, i.e. reshaping the `(r_lat, r_lon)` coefficient grid at each
time step into a single length-$(r_{\mathrm{lat}}\cdot r_{\mathrm{lon}})$
vector before DMD's SVD step can act on it.

**Why this is a narrower, different kind of flattening than the one Error
2 forbids.** The tensor-representation correction exists because raveling
the *original, high-resolution physical* `(lat, lon)` grid into one axis
destroys the local adjacency structure between neighboring grid cells —
information a naive matrix decomposition has no way to recover, since nothing
in a flattened vector encodes which entries used to be spatial neighbors.
The reshape here is different in kind: $C^{(v)}_{:,:,t}$ is *already*
Tucker-compressed. Tucker's own per-mode factor matrices
$U_{\mathrm{lat}}^{(v)}$ and $U_{\mathrm{lon}}^{(v)}$ have already captured
the spatial (local-adjacency) structure along each physical axis
*separately*, before this point — that is what a Tucker fit is for. The
small `(r_lat, r_lon)` grid being reshaped here is a coefficient
grid in an abstract, already-compressed basis, not raw grid cells; there
is no remaining raw-grid adjacency left in it to lose by reshaping. Row $i$
of the vectorized $x_t$ does not "used to be a neighbor" of row $i+1$ in
any physically meaningful sense that Tucker hasn't already accounted for
through its factor matrices.

**This distinction is real, but it should be a deliberate one, not an
excuse to stop thinking about it.** Reshaping a small compressed
coefficient grid is a fundamentally milder operation than raveling a raw
physical grid, but it is not *nothing*: it does discard whatever
structure, if any, exists jointly across the `r_lat` and `r_lon` compressed
axes together (as opposed to what each factor matrix captured along its
own axis independently), and it commits to treating the vectorized
$r_{\mathrm{lat}}\cdot r_{\mathrm{lon}}$-dimensional coefficient space as
DMD's ambient state space. A reader implementing this should note the
distinction explicitly in code/comments, not silently assume the
Error-2 prohibition simply doesn't apply here "by analogy" without stating
why.

**An alternative worth citing, not mandating.** If a reader wants to avoid
even this narrower reshape, the DMD literature includes genuinely
tensor-native extensions — higher-order/tensor DMD formulations that act
mode-wise directly on a tensor sequence $C^{(v)}_{:,:,0}, C^{(v)}_{:,:,1},
\dots$ without ever vectorizing the per-timestep state into a single
column, analogous to how Tucker itself avoids vectorizing `(lat, lon)`.
These are the theoretically purest option along this axis, at the cost of
more specialized (and less standardized/battle-tested) numerics than exact
DMD's SVD-based recipe.

**This document does not resolve which approach the eventual code should
take.** Vectorize-then-classical-DMD and tensor-native dynamic-mode
extensions are both legitimate; the choice is flagged here as an open
decision (also tracked in §6), to be made explicitly and documented
wherever it is made — not defaulted into silently by whichever library
happens to be imported first.

### 4.3 Recommended defaults, expressed symbolically

- **DMD truncation rank** $r_{\mathrm{dmd}}$: chosen well below
  $r_{\mathrm{lat}}\cdot r_{\mathrm{lon}}$ (the ambient dimension of the
  vectorized per-timestep state, §4.2) and necessarily $r_{\mathrm{dmd}}
  \le \min(r_{\mathrm{lat}}\cdot r_{\mathrm{lon}},\, T-1)$ (there cannot be
  more DMD modes than snapshot pairs). In practice, select $r_{\mathrm
  dmd}$ from the singular-value spectrum of $X$ — e.g. the smallest rank
  retaining some target fraction of cumulative singular-value energy, or an
  elbow in the spectrum — rather than fixing it to an arbitrary constant
  independent of the data.
- **Sampling interval** $\Delta t$: whatever the archive's native cadence
  is for the era/window being fit (this repository's archive cadence
  varies by era — see `core.archive_eras`). $\Delta t$ sets the Nyquist
  limit $\pi/\Delta t$ on recoverable angular frequency (§2.3); a coarser
  $\Delta t$ irrecoverably aliases faster physical oscillations.
- **Snapshot pairing**: consecutive-in-time pairs $(x_t, x_{t+1})$ as in
  §2.2. If gaps or era boundaries exist in the training window, pairs must
  not be formed across a gap or era boundary (consistent with this
  repository's existing rule that a training window must stay within a
  single archive era).

These are stated symbolically and structurally, deliberately, rather than
as fixed numbers — the right numeric values are a function of the data and
belong in configuration/tuning, not in this document.

---

## 5. Results: reading a DMD fit diagnostically

For one variable's DMD fit, the following diagnostics indicate whether the
fit is behaving as a plausible description of that variable's dynamics, in
the given (Gaussianized, Tucker-compressed) observable coordinates:

- **Eigenvalue spectrum in the complex plane.** Plot each $\lambda_k$
  against the unit circle. Points strictly inside ($|\lambda_k|<1$,
  equivalently $\sigma_k<0$) are decaying — expected for the bulk of modes
  in a statistically stationary system. Points on the circle
  ($\sigma_k\approx 0$) are sustained oscillations — plausible for strong
  periodic forcing (diurnal cycle, near-inertial motion). Points outside
  the circle ($\sigma_k>0$, genuine growth) are a red flag over any fit
  window long enough to be considered representative of stationary
  behavior; treat as a signal to reexamine rank truncation, the
  Gaussianization upstream, or whether the fit window itself contains a
  genuine transient (e.g. storm growth) that a longer/different window
  would average out.
- **Growth/decay rates** $\sigma_k = \mathrm{Re}(\omega_k)$: compare
  $1/|\sigma_k|$ (an implied decorrelation timescale) against physically
  plausible timescales for the variable in question — sea-state
  adjustment on the order of hours to a day, SST anomaly persistence on
  the order of days to weeks, synoptic-scale weather-system evolution on
  the order of several days. A fitted timescale wildly outside physically
  reasonable bounds for the variable being fit suggests overfitting (rank
  too high relative to data) or an upstream data-quality issue rather than
  a genuine physical mode.
- **Oscillation frequencies** $\nu_k = \mathrm{Im}(\omega_k)$, period
  $2\pi/|\nu_k|$: compare against known physical periods — diurnal (~24 h,
  solar heating cycle in SST/wind), near-inertial (period $2\pi/f$ with
  Coriolis parameter $f = 2\Omega\sin(\text{latitude})$, hence
  latitude-dependent), synoptic-band (several days), and, if present in
  the fit window, semidiurnal tidal-band signatures in current fields. A
  recovered frequency that lands close to one of these known bands is
  independent physical corroboration that the mode is real rather than a
  rank-truncation artifact.
- **Modes** $\phi_k$: each is, after un-vectorizing (§4.2), an `(r_lat,
  r_lon)`-shaped complex coefficient pattern in Tucker-compressed
  coordinates — not yet a physical spatial pattern. To visualize the
  physical-space pattern a mode corresponds to, project back through the
  Tucker factor matrices, e.g. $U_{\mathrm{lat}}^{(v)}\,\mathrm{Re}(\phi_k)\,
  (U_{\mathrm{lon}}^{(v)})^{\!\top}$ for the real part (and similarly for
  the imaginary part), yielding an `(N_lat, N_lon)` spatial pattern that
  can be checked against known physical patterns (e.g. a traveling swell
  front, a basin-scale SST anomaly).
- **Initial amplitudes** $b_0$: $|b_{0,k}|$ ranks each mode's contribution
  to the observed initial condition. Combined with $|\lambda_k|$, this
  indicates which modes dominate the reconstruction over the fit horizon
  versus which are low-amplitude and plausibly noise-fit artifacts of
  truncating at $r_{\mathrm{dmd}}$ too generously.
- **Reconstruction residual.** Compare $\hat x_t = \Phi\Lambda^t b_0$
  against the actual (vectorized) core sequence. This residual is exactly
  the signal any downstream per-mode stochastic layer (e.g. a mode-level
  Ornstein–Uhlenbeck fit) is meant to model. A well-specified DMD fit
  should leave a residual that looks like a stationary, mean-reverting
  process per mode — a residual with a visible trend or slow drift signals
  that $r_{\mathrm{dmd}}$ is too low (a slow mode is missing from $\Lambda$
  and is leaking into the "noise" the next stage will try, and fail, to
  characterize as stationary).

---

## 6. Open questions and limitations

DMD is, plainly, the **least-settled concept in this pipeline** relative to
Tucker or the upstream Gaussianization steps — the points below are
genuinely open, not rhetorical.

- **No cross-variable temporal coupling.** Because each variable's DMD fit
  is now entirely independent (§4.1, correcting Error 1's joint
  `(r_ocean, r_var)` state), there is no mechanism in this document for
  capturing cross-variable dynamical coupling — e.g. wind driving wave
  growth, a physically central mechanism in this domain. This ties to the
  same open question raised in `docs/tucker_decomposition.md` regarding
  whether/how cross-variable structure should be captured at all, and is
  left unresolved here as well. Any future joint treatment would need to
  be designed deliberately (e.g. a coupled/multi-variable DMD variant, or
  a separate cross-variable coupling layer applied to per-variable DMD
  outputs) rather than reintroduced by accident via a joint state, which
  is precisely the mistake Error 1 corrects.
- **Conjugate-pair enforcement at simulation time** (§2.4): whether the
  code that samples/simulates forward in time enforces $\lambda_{k'} =
  \bar\lambda_k$, $\phi_{k'} = \bar\phi_k$, and $\eta_{k'}(t) =
  \overline{\eta_k(t)}$ for the stochastic residual at every step, or
  instead allows independent per-mode sampling and accepts that realness
  holds only statistically rather than per-realization, is not decided by
  this document. It must be decided, explicitly, wherever the sampling
  code lives.
- **The state-vectorization tension** (§4.2): whether DMD, as eventually
  implemented, vectorizes each `(r_lat, r_lon)` Tucker-core time slice into
  a single column and proceeds with classical exact DMD (narrower
  flattening than Error 2 forbids, but still a real choice worth being
  deliberate about), or instead uses a tensor-native dynamic-mode
  formulation that never vectorizes the per-timestep state, is left open.
  Both are legitimate; this document takes no position on which the
  eventual code should implement, beyond insisting the choice be made
  consciously and documented at the point it's made.
- **Rank selection** for $r_{\mathrm{dmd}}$ is described only structurally
  (§4.3); no closed-form or automatic data-driven rank-selection rule is
  mandated here, mirroring the same open-endedness in Tucker rank
  selection.
- **Stability of growing modes over long simulation horizons.** If a fit
  genuinely produces $|\lambda_k|>1$ for some mode (as opposed to a
  transient artifact, §5), and DMD's deterministic backbone is used to
  drive long-horizon stochastic simulation, that mode will diverge
  unboundedly. Whether/how to detect and handle such a mode (reject the
  fit, clip the eigenvalue, flag for manual review) is not addressed here
  and should be treated as a real open design question wherever the
  simulation-horizon behavior of DMD's output is specified.
