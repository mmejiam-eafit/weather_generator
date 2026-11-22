# Tucker Decomposition

## 0. Two errors this document corrects

This document is a rewrite following the deletion of an earlier documentation
set that made two errors. Both are stated explicitly here because this
document is the one most affected by the second.

**Error 1 — prescriptive, not descriptive.** These documents do not exist to
describe what the code currently does. They exist to define what the process
*should be*, so that the code can be corrected to match. In particular,
`src/train/tucker.py` as it stands today implements the wrong design — it
fits a single joint Tucker decomposition over a tensor `Z` of shape
`(T, N_ocean, V)`, where `N_ocean` is a raveled "ocean cell" mode (lat and lon
already flattened together and land cells already dropped) and `V` is a
stacked "variable" mode (all modeled variables concatenated along a fourth
axis before flattening). That file, and the design note it cites
(`docs/model_design.md §5.5`, itself part of the deleted/superseded set), are
exactly what needs to change under the correction below — they are not
consulted here as authority, only mentioned as the artifact this document's
prescription will be used to correct.

**Error 2 — the core correction for this document.** Do not treat the
collection of modeled variables as a single 4-dimensional tensor of shape
`(lat, lon, time, num_variables)`. Each physical variable gets its **own**
independent 3-dimensional tensor of shape `(lat, lon, time)`, decomposed on
its own. Within that 3D tensor, lat and lon are never merged into a single
"ocean cell" mode, and lat/lon/time are never flattened into any 2D matrix —
not `(lat·lon, time)`, not `(lat, lon·time)`, nor any other combination.
Flattening spatial axes together destroys the local (neighboring-cell)
structure that a separate lat mode and a separate lon mode preserve; that
loss is the reason flattening is forbidden here, not a stylistic preference.
Land and ice cells are handled by a mask passed into a masked/incomplete
Tucker fit (§2.4), which iteratively imputes those entries from the current
low-rank reconstruction — they are never dropped from the tensor, and the
tensor's three axes are never reduced in number to accommodate them. This
pattern is grounded concretely in `notebooks/01_understand_tucker_decomposition.ipynb`
(not to copy its exploratory rank choices, but to fix the correct mechanism
and shapes) and is elaborated in §3 and §4.

## 1. What this is, why it's here

The pipeline needs a compact representation of large spatiotemporal fields —
significant wave height, sea surface temperature, wind and current
components — so that a later dynamical stage (Dynamic Mode Decomposition,
covered in a sibling document) can regress a linear evolution operator on a
tractable number of coefficients, and so that a stochastic residual model can
be fit to what that linear operator does not explain. A raw training window
of, say, a `(1121, 2880)` global grid at 12-hourly resolution over a season
is far too large, and far too spatially redundant (neighboring grid cells
are highly correlated), to model directly. Tucker decomposition supplies the
right notion of dimensionality reduction for this setting: unlike a naive
"flatten space into one axis and run PCA" approach, it operates on the
tensor's native multi-way structure, extracting a compressed basis
**per axis** (a spatial basis in latitude, a spatial basis in longitude, and
optionally a compressed basis in time) without ever collapsing those axes
into each other. This document defines the correct mathematical treatment
of that reduction: what a Tucker decomposition is, why the specific
per-variable / three-mode / mask-aware form is the right one for this
dataset, and what it does and does not solve.

## 2. Theoretical foundation

### 2.1 Tensors as multilinear maps

A real $N$-way tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times
\cdots \times I_N}$ is an $N$-dimensional array indexed by
$(i_1, \dots, i_N)$, $1 \le i_n \le I_n$. Algebraically, it is more useful to
think of $\mathcal{X}$ not merely as a array of numbers but as a multilinear
map: fixing bases for each of the $N$ vector spaces $\mathbb{R}^{I_1}, \dots,
\mathbb{R}^{I_N}$, $\mathcal{X}$ represents an element of the tensor product
space
$$
\mathcal{X} \in \mathbb{R}^{I_1} \otimes \mathbb{R}^{I_2} \otimes \cdots \otimes \mathbb{R}^{I_N},
$$
equivalently a multilinear functional taking one vector from each of $N$ dual
spaces to a scalar. For $N=2$ this is exactly a matrix, understood as a
bilinear form or as a linear map between two vector spaces. Everything below
is the $N \ge 3$ generalization of matrix linear algebra: mode-$n$
unfolding generalizes "the matrix, read out"; the mode-$n$ product
generalizes "matrix times vector/matrix"; multilinear rank generalizes
matrix rank; and Tucker decomposition generalizes the SVD. None of these
generalizations are unique — several inequivalent ways to define "rank" or
"decomposition" exist for tensors where only one exists for matrices, which
is itself a reason order-$\ge 3$ tensor algebra is more subtle than matrix
algebra (§2.3 returns to this).

For this pipeline, $N = 3$ throughout: each physical variable's data is a
single tensor $\mathcal{X} \in \mathbb{R}^{I_{\text{lat}} \times
I_{\text{lon}} \times I_{\text{time}}}$, one axis per physical/temporal
dimension, per variable (§3–§4).

### 2.2 Mode-$n$ unfolding (matricization)

The **mode-$n$ unfolding** of $\mathcal{X} \in \mathbb{R}^{I_1 \times \cdots
\times I_N}$, written $\mathbf{X}_{(n)} \in \mathbb{R}^{I_n \times
\prod_{k \ne n} I_k}$, is the matrix obtained by fixing the $n$-th index and
laying out all other index combinations as columns:
$$
\mathbf{X}_{(n)}\big[i_n,\, j\big] = \mathcal{X}[i_1, \dots, i_N], \qquad
j = 1 + \sum_{\substack{k=1 \\ k \ne n}}^{N} (i_k - 1) \prod_{\substack{m<k \\ m \ne n}} I_m.
$$
Concretely, for a 3-mode tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2
\times I_3}$: $\mathbf{X}_{(1)}$ has $I_1$ rows and $I_2 I_3$ columns (every
row is one lat-slice's worth of all (lon, time) pairs, laid flat);
$\mathbf{X}_{(2)}$ has $I_2$ rows and $I_1 I_3$ columns; $\mathbf{X}_{(3)}$
has $I_3$ rows and $I_1 I_2$ columns.

**This is not the forbidden flattening of Error 2.** Mode-$n$ unfolding is a
transient linear-algebra step performed *inside* the HOSVD/HOOI algorithm
(§2.3–§2.4), one mode at a time, to reduce "SVD of a tensor along axis $n$"
to the well-understood problem "SVD of a matrix." It is recomputed,
discarded, and never becomes the tensor's representation between pipeline
stages — the object that gets stored, passed to the next stage, or reasoned
about physically is always the full 3D `(lat, lon, time)` tensor, never a
flattened matrix. The forbidden flattening in Error 2 is a *representational*
choice (deciding to store/pass data as `(lat·lon, time)`); mode-$n$
unfolding is an *algorithmic* internal step (LAPACK needs a matrix to run
`svd` on, so the decomposition routine manufactures one, uses it, and moves
on). Conflating the two is precisely the mistake this document exists to
prevent.

### 2.3 The mode-$n$ product and multilinear rank

The **mode-$n$ product** of a tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times
\cdots \times I_N}$ with a matrix $\mathbf{U} \in \mathbb{R}^{J \times I_n}$,
written $\mathcal{X} \times_n \mathbf{U} \in \mathbb{R}^{I_1 \times \cdots
\times I_{n-1} \times J \times I_{n+1} \times \cdots \times I_N}$, is defined
elementwise as
$$
(\mathcal{X} \times_n \mathbf{U})[i_1, \dots, i_{n-1}, j, i_{n+1}, \dots, i_N]
= \sum_{i_n=1}^{I_n} \mathcal{X}[i_1, \dots, i_N] \, \mathbf{U}[j, i_n],
$$
equivalently $\mathbf{U} \mathbf{X}_{(n)}$ unfolded back into tensor form.
It is the natural generalization of matrix-vector/matrix-matrix
multiplication to "multiply along one axis of a tensor, leave the others
untouched." Mode-$n$ products along different modes commute:
$\mathcal{X} \times_m \mathbf{A} \times_n \mathbf{B} = \mathcal{X} \times_n
\mathbf{B} \times_m \mathbf{A}$ for $m \ne n$, which is what makes the
compact "core times one factor per mode" form of Tucker (§2.4) well-defined
regardless of the order the products are applied in.

The **multilinear rank** (also "Tucker rank" or "$n$-rank") of $\mathcal{X}$
is the tuple $(R_1, \dots, R_N)$ with $R_n = \operatorname{rank}
(\mathbf{X}_{(n)})$, the ordinary matrix rank of the mode-$n$ unfolding. This
generalizes matrix rank ($N=2$, where $R_1 = R_2$ always) but for $N \ge 3$
the $R_n$ need not agree with each other, and — critically — a tensor of
multilinear rank $(R_1, \dots, R_N)$ is *not* generally expressible as a sum
of $\max_n R_n$ rank-1 terms the way a matrix is; that different notion
(CP/CANDECOMP-PARAFAC rank) is a separate, harder quantity that this document
does not use. Tucker decomposition targets multilinear rank, not CP rank.

### 2.4 The Tucker decomposition theorem

Given target ranks $(R_1, \dots, R_N)$ with $R_n \le I_n$, the **Tucker
decomposition** of $\mathcal{X} \in \mathbb{R}^{I_1 \times \cdots \times
I_N}$ seeks a core tensor $\mathcal{G} \in \mathbb{R}^{R_1 \times \cdots
\times R_N}$ and, for each mode $n$, a factor matrix $\mathbf{U}^{(n)} \in
\mathbb{R}^{I_n \times R_n}$ with orthonormal columns, such that
$$
\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \cdots \times_N \mathbf{U}^{(N)}.
$$
For $N=3$ (this pipeline's case, one Tucker fit per variable):
$$
\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(\text{lat})} \times_2 \mathbf{U}^{(\text{lon})} \times_3 \mathbf{U}^{(\text{time})},
\qquad
\mathcal{G} \in \mathbb{R}^{R_{\text{lat}} \times R_{\text{lon}} \times R_{\text{time}}}.
$$
Each factor matrix $\mathbf{U}^{(n)}$ is a basis for a rank-$R_n$ subspace of
mode $n$; the core $\mathcal{G}$ carries all the cross-mode interaction
("how much of lat-basis-vector $p$, lon-basis-vector $q$, and
time-basis-vector $r$ appear together"). This is the direct generalization
of the matrix low-rank factorization $\mathbf{X} \approx \mathbf{U} \Sigma
\mathbf{V}^\top$ — the core $\mathcal{G}$ plays the role $\Sigma$ plays for
matrices, except it need not be diagonal once $N \ge 3$, because there is no
generalization of the spectral theorem that simultaneously diagonalizes more
than two orthogonal factor matrices at once.

### 2.5 HOSVD: SVD-based initialization per mode

The **Higher-Order SVD** (HOSVD; Tucker 1966, in the form popularized by De
Lathauwer, De Lathauwer & Vandewalle, 2000) computes each factor matrix
independently: for each mode $n$, take the ordinary matrix SVD of the
mode-$n$ unfolding, $\mathbf{X}_{(n)} = \mathbf{U}^{(n)} \mathbf{\Sigma}^{(n)}
\mathbf{V}^{(n)\top}$, and keep the leading $R_n$ left singular vectors as
$\mathbf{U}^{(n)}$. The core is then obtained by projecting:
$$
\mathcal{G} = \mathcal{X} \times_1 \mathbf{U}^{(1)\top} \times_2 \mathbf{U}^{(2)\top} \times \cdots \times_N \mathbf{U}^{(N)\top}.
$$

**Why HOSVD is generally suboptimal for $N \ge 3$.** For a matrix ($N=2$),
the Eckart–Young theorem guarantees that truncating the SVD to rank $R$
gives the *best* rank-$R$ approximation in Frobenius norm — there is no
better rank-$R$ matrix. HOSVD does not carry this guarantee once $N \ge 3$.
The reason is that HOSVD chooses each $\mathbf{U}^{(n)}$ by looking only at
mode $n$'s unfolding *in isolation*, ignoring how that choice interacts with
the factor matrices chosen for the other modes. Optimizing each mode
independently (a per-mode SVD, done once, non-iteratively) is not the same
optimization problem as jointly minimizing
$\lVert \mathcal{X} - \mathcal{G} \times_1 \mathbf{U}^{(1)} \times \cdots
\times_N \mathbf{U}^{(N)} \rVert_F^2$
over all factor matrices simultaneously — the true objective couples them.
HOSVD gives a good, cheap starting point, and in practice is often close to
optimal, but it is not guaranteed optimal, and can be measurably improved on
in cases where the modes are strongly coupled (which spatial × spatial ×
temporal ocean fields typically are).

### 2.6 HOOI: alternating-least-squares refinement

**Higher-Order Orthogonal Iteration** (HOOI) refines the HOSVD initialization
by alternating least squares: fix all factor matrices except $\mathbf{U}^{(n)}$,
and re-solve for $\mathbf{U}^{(n)}$ as the leading $R_n$ left singular
vectors of the mode-$n$ unfolding of the tensor *with the other modes'
factors already projected out*,
$$
\mathbf{U}^{(n)} \leftarrow \text{leading-}R_n\text{ left singular vectors of }
\Big(\mathcal{X} \times_{k \ne n} \mathbf{U}^{(k)\top}\Big)_{(n)},
$$
then cycle to the next mode, and repeat until the reconstruction error
converges. Each individual step is an exact solution of a matrix
Eckart–Young problem conditional on the other factors, so the overall
reconstruction error is non-increasing at every step — HOOI is a coordinate-
descent scheme on a non-convex objective and is guaranteed to converge to a
*stationary point*, not necessarily the global optimum, but it strictly
improves on (or matches) whatever HOSVD handed it as a starting point. This
is why HOSVD is used as *initialization* and HOOI as *refinement*: HOSVD is
cheap and mode-independent, HOOI is the part of the algorithm that actually
accounts for cross-mode coupling.

### 2.7 Relation to PCA / POD

Restricting to $N=2$ collapses all of the above to familiar territory:
mode-1 and mode-2 unfolding of a matrix $\mathbf{X}$ are $\mathbf{X}$ and
$\mathbf{X}^\top$ themselves, the mode-$n$ product reduces to ordinary
matrix multiplication, HOSVD along both modes reduces to computing the same
SVD twice (once for the row space, once for the column space, both
recoverable from a single $\mathbf{X} = \mathbf{U}\Sigma\mathbf{V}^\top$),
and HOOI has nothing left to iterate on, since the $N=2$ HOSVD is already
Eckart–Young-optimal. Tucker with $N=2$ *is* PCA/Proper Orthogonal
Decomposition on a data matrix, applied here (in the DMD literature, and in
classical POD of spatiotemporal fields) by putting one physical axis on the
rows and one on the columns — typically space on one axis, time or ensemble
member on the other. Tucker's contribution, and the reason it is used here
instead of plain PCA, is exactly what changes once $N \ge 3$: it keeps
*multiple* physical axes (lat, lon, time) each as their own mode with their
own basis, instead of forcing a choice between two axes and flattening
whatever else exists into one of them. That flattening-avoidance is not
incidental to Tucker — it is Tucker's entire reason for existing relative to
PCA, which is also why flattening the tensor before or after calling Tucker
(Error 2) defeats the purpose of choosing Tucker over PCA in the first
place.

### 2.8 Masked / incomplete Tucker (iterative imputation)

Ocean grids contain land and ice cells with no valid observation at any
timestep. These must not be dropped from the tensor (doing so would ravel
lat/lon together to describe "the remaining cells," which is exactly the
forbidden flattening of Error 2) and must not be filled with an arbitrary
sentinel value fed into the decomposition as if it were data (a sentinel
like `-9999`, or leaving `NaN` in place, would corrupt every singular value
and every reconstruction-error computation that touches it). The correct
treatment is a masked/incomplete Tucker fit.

Let $M \in \{0,1\}^{I_1 \times \cdots \times I_N}$ be a binary mask, $1$ for
observed entries and $0$ for entries to be imputed (land/ice cells,
broadcast across every timestep — see §3). Masked Tucker minimizes
reconstruction error **only over the observed entries**:
$$
\min_{\mathcal{G}, \{\mathbf{U}^{(n)}\}} \; \Big\lVert M \odot \Big(\mathcal{X} - \mathcal{G} \times_1 \mathbf{U}^{(1)} \times \cdots \times_N \mathbf{U}^{(N)}\Big) \Big\rVert_F^2,
$$
where $\odot$ is the elementwise (Hadamard) product. This is solved by an
alternating scheme closely related to HOOI: at each iteration, (1) fill the
masked (unobserved) entries with the *current* low-rank reconstruction's
values at those entries — i.e. impute $\hat{\mathcal{X}} = M \odot
\mathcal{X} + (1-M) \odot \big(\mathcal{G} \times_1 \mathbf{U}^{(1)} \times
\cdots \times_N \mathbf{U}^{(N)}\big)$ — then (2) run one ordinary HOOI
sweep (§2.6) on this fully-filled $\hat{\mathcal{X}}$, then (3) recompute the
imputed values from the new factors, and repeat. Intuitively: the observed
entries anchor the fit at every step (they are never overwritten), while the
unobserved entries are always replaced by "whatever the current best
low-rank model predicts belongs there" before the next SVD-based update — so
the imputed values are never treated as independent data, only as
placeholders consistent with the current low-rank structure. As the factors
converge, the imputed entries converge to the low-rank completion implied by
the observed entries and the fitted rank; this is why the mask is best
understood as *iterative imputation from a low-rank model*, not as a
one-time fill-and-forget preprocessing step.

Convergence here inherits HOOI's usual guarantee — non-increasing
reconstruction error on the observed entries, converging to a stationary
point — but not a global-optimality guarantee, and with an added
practical wrinkle: masked HOOI's objective is non-convex even for a fixed
mask, and unlike the fully-observed case, a poor initialization or a mask
with very large contiguous unobserved regions (e.g. a large land mass with
no interior coastline) can converge to a poor local completion of the
interior. See §6 for this as a named limitation.

## 3. Variable-class handling

The variable-taxonomy document (a sibling document to this one) recommends
reducing every modeled physical quantity — whatever its original tensor or
vector character — down to a set of plain numeric scalar fields, each with
shape `(lat, lon, time)`, before any Tucker fitting happens. This document
takes that reduction as given and addresses only what happens once each
variable is already such a field:

| Variable class | Example | What reaches Tucker |
|---|---|---|
| (A) Scalars | `sigwh` (significant wave height), `sst` (sea surface temperature) | One `(lat, lon, time)` tensor, directly. |
| (B) Vector components | `u_wind`, `v_wind` (Cartesian wind components) | Each Cartesian component is its own scalar field once decomposed out of the vector; each gets its own `(lat, lon, time)` tensor. |
| (C) Triplet / directional components | e.g. Cartesian components derived from a direction-and-magnitude field (`wsd`-style variables per the direction-convention finding) | Same as (B): each Cartesian component, once resolved to a plain number per `(lat, lon, time)` cell, is its own tensor. |

The point that matters for this document: by the time anything reaches
Tucker, class (A), (B), and (C) are indistinguishable. There is no
"vector-mode" or "component-mode" special-cased inside Tucker — a two-
component wind field does not become one `(lat, lon, time, 2)` tensor, nor
does it become one `(lat, lon, time)` tensor for "the wind variable" with
components stacked some other way. `u_wind` gets a Tucker fit. `v_wind` gets
a *separate*, independent Tucker fit. Nothing downstream of the
variable-taxonomy reduction ever reintroduces a variable axis, a component
axis, or any other grouping axis into a single tensor passed to Tucker. Each
of the resulting scalar fields — regardless of which physical class it came
from — is decomposed exactly as described in §4.

## 4. Correct treatment for this dataset's tensors

### 4.1 One tensor, one Tucker fit, per variable

For each modeled scalar field $v$ (in the sense of §3 — a scalar, a vector
component, or a triplet component, all equivalent by that point), form
$$
\mathcal{X}^{(v)} \in \mathbb{R}^{I_{\text{lat}} \times I_{\text{lon}} \times I_{\text{time}}}
$$
by stacking that single variable's snapshots along a time axis — mirroring
`notebooks/01_understand_tucker_decomposition.ipynb`'s `np.stack(ssts,
axis=-1)`, which builds exactly this shape (`(1121, 2880, 12)` for `sst`
over the notebook's exploratory window) for one variable at a time. Fit an
independent masked Tucker decomposition (§2.8) to $\mathcal{X}^{(v)}$:
$$
\mathcal{X}^{(v)} \approx \mathcal{G}^{(v)} \times_1 \mathbf{U}^{(v,\text{lat})} \times_2 \mathbf{U}^{(v,\text{lon})} \times_3 \mathbf{U}^{(v,\text{time})}.
$$
No two variables ever share a core, share factor matrices, or appear
together in the same tensor at any point in this process. If there are $V$
modeled variables, there are $V$ independent Tucker fits, each with its own
$(R_{\text{lat}}, R_{\text{lon}}, R_{\text{time}})$ chosen independently
(they need not match across variables — `sst` and `sigwh` have no reason to
compress to the same ranks).

**What $\mathcal{X}^{(v)}$ actually contains, for height/period variables.**
For wave-height fields (`sigwh`, `wsh`, `pswh`) and period fields (`wsp`,
`pwp`, `pswp`), $\mathcal{X}^{(v)}$ is the `log`-transformed tensor
(`variable_taxonomy.md` §3.4 — $\log E := \log(H^2)$, $T:=\log T_{m01}$),
not the raw physical-unit field. Nothing about the Tucker mechanics above
changes because of this — it is a marginal pre-transform applied before
this section's fit, motivated by the same L2-fit dynamic-range/leverage
concern this document's masked-HOOI machinery is already sensitive to
(§2.8) — but reconstructions from these particular fits must be inverted
($H=\exp(\log E/2)$, $T_{m01}=\exp(T)$) before being read as physical
quantities.

### 4.2 Mask construction and use

For variable $v$, build the ocean mask exactly as the notebook does:
$$
\text{ocean\_mask}^{(v)}[\text{lat}, \text{lon}] = \neg \bigvee_{t} \operatorname{isnan}\!\big(\mathcal{X}^{(v)}[\text{lat}, \text{lon}, t]\big),
$$
a `(lat, lon)` boolean array — a cell counts as ocean only if it is
non-`NaN` at *every* training timestep. Broadcast it across the time axis to
build the full 3D mask passed to the masked Tucker fit (§2.8):
$$
M^{(v)} = \text{broadcast}\big(\text{ocean\_mask}^{(v)},\ \mathcal{X}^{(v)}.\text{shape}\big) \in \{0,1\}^{I_{\text{lat}} \times I_{\text{lon}} \times I_{\text{time}}}.
$$
Land/ice entries in $\mathcal{X}^{(v)}$ are filled with a neutral
placeholder (`0.0`, never `NaN`, never a sentinel like `-9999` — see §2.8
for why) purely so the algorithm's initial norm computation is not poisoned;
the mask, not the placeholder value, is what tells the masked-HOOI loop
these entries are to be imputed rather than fit directly. This mask is
computed **per variable** (a cell that is land for `sst` need not coincide
exactly with a cell masked for a different reason in another variable's own
fit) and never reused to justify dropping cells from the tensor's shape —
the tensor stays `(lat, lon, time)` whether a given cell is ocean or land.

### 4.3 Why the time mode should be left at (or near) full rank

Compressing the lat and lon modes is well-motivated on its own: neighboring
grid cells are strongly spatially correlated, so a modest $R_{\text{lat}},
R_{\text{lon}} \ll I_{\text{lat}}, I_{\text{lon}}$ captures most of the
spatial variance. The time mode deserves separate reasoning, because it
feeds directly into the next stage of this pipeline (DMD, covered in a
sibling document), and that stage's requirements — not a stylistic
preference — should drive the rank choice here.

DMD works by regressing a linear operator $\mathbf{A}$ that approximately
advances the system one step in time, $\mathbf{x}_{t+1} \approx \mathbf{A}
\mathbf{x}_t$, fit from a sequence of *paired, consecutive* state vectors
$(\mathbf{x}_1, \mathbf{x}_2), (\mathbf{x}_2, \mathbf{x}_3), \dots$ — one
state vector per original timestep, in original time order. If the Tucker
time mode is compressed to $R_{\text{time}} < I_{\text{time}}$, the
time-mode factor matrix $\mathbf{U}^{(v,\text{time})} \in
\mathbb{R}^{I_{\text{time}} \times R_{\text{time}}}$ mixes multiple original
snapshots into each of the $R_{\text{time}}$ compressed time-coefficients —
by construction, each column of $\mathbf{U}^{(v,\text{time})}$ is a linear
combination (a temporal basis vector) spanning some subset of the original
timesteps, not a single one. That is precisely a loss of the
one-state-per-original-timestep sequence DMD needs to regress a step-to-step
evolution operator: DMD would then be regressing an operator between
*compressed time-coefficients* rather than between actual consecutive
physical states, which changes what the operator means and discards the
finer step-to-step structure DMD is built to capture.

The reasoning generalizes to a design principle a student should be able to
re-derive, not memorize: **compress the modes whose within-mode
correlation you intend to exploit for compression, and leave uncompressed
the mode whose original index sequence a downstream stage needs intact.**
Here that means lat and lon are good candidates for real compression (their
correlation *is* the thing being exploited, and nothing downstream needs
individual grid-cell identity preserved one-to-one), while time is not,
because DMD's entire mechanism depends on indexable, consecutive, one-per-
original-timestep states. Setting $R_{\text{time}} = I_{\text{time}}$ (full
rank in time) is the safe default that avoids this problem entirely; a
carefully chosen $R_{\text{time}} < I_{\text{time}}$ that still preserves a
faithful one-to-one (or at least monotonic, evenly-spaced) mapping to
distinct time-points might be defensible in principle, but should be
treated as a deviation to be justified case-by-case, not the default. The
notebook's own choice of `time_rank = sst_full.shape[-1]` (i.e. full rank in
time) reflects exactly this reasoning, applied to its one exploratory
variable.

To be clear about what is and is not being prescribed here: the notebook's
specific spatial ranks — `latitude_rank = 10`, `longitude_rank = 20` — are
stated in the notebook itself to be arbitrary, chosen only to explore the
mechanism on one variable (`sst`) over a short one-week window. They are not
a recommended default for any variable in production; §5 covers how to
choose ranks diagnostically instead of copying these numbers.

### 4.4 Open question: cross-variable coupling (flagged, not resolved)

Under §4.1, each variable's Tucker fit is entirely independent — there is no
shared factor matrix, no shared core, and in particular **no "variable"
mode anywhere in this design** that could carry cross-variable structure.
This is a deliberate consequence of correcting Error 2, but it has a real
cost: physically meaningful correlations between variables — for instance,
wind speed and wave height tend to co-vary, since wind forcing generates
waves — have no mechanism to be captured *within* Tucker as specified here.
Each variable's core and factor matrices describe only that variable's own
spatial and temporal structure.

**This document does not resolve this.** If cross-variable coupling matters
for the system being modeled (and physically, it plausibly does), it must be
captured by some *other* mechanism operating on the outputs of these
per-variable Tucker fits — for example, a joint model fit on the stacked
per-variable core coefficients after each variable has already been
independently compressed by its own Tucker fit, or some other later stage
entirely. Precisely which mechanism, if any, is the right one is left as an
explicitly open design question for a later document or a later revision of
this one; it should not be quietly reintroduced by putting variables back
into a shared tensor mode, since that is the exact design Error 2 rules out.

## 5. Results: reading reconstruction error and rank choices

Each per-variable Tucker fit reports a **relative Frobenius reconstruction
error** over the observed (non-masked) entries:
$$
\text{reconstruction\_error}^{(v)} = \frac{\big\lVert M^{(v)} \odot \big(\hat{\mathcal{X}}^{(v)} - \mathcal{X}^{(v)}_{\text{filled}}\big) \big\rVert_F}{\big\lVert M^{(v)} \odot \mathcal{X}^{(v)}_{\text{filled}} \big\rVert_F},
$$
where $\hat{\mathcal{X}}^{(v)} = \mathcal{G}^{(v)} \times_1
\mathbf{U}^{(v,\text{lat})} \times_2 \mathbf{U}^{(v,\text{lon})} \times_3
\mathbf{U}^{(v,\text{time})}$ is the reconstruction and $\lVert \cdot
\rVert_F$ is the Frobenius norm ($\lVert \mathcal{T} \rVert_F = \sqrt{\sum
\mathcal{T}[i,j,k]^2}$, the natural tensor generalization of the matrix
Frobenius norm — equivalently the $\ell_2$ norm of the tensor flattened to a
single vector, which is basis-independent and unaffected by which unfolding,
if any, is used to compute it). A value near 0 means the chosen ranks
$(R_{\text{lat}}, R_{\text{lon}}, R_{\text{time}})$ capture nearly all of
$v$'s variance over the observed cells; a value near 1 means the fit is
capturing almost none of it. This is diagnosed **per variable** — there is
no single reconstruction error for "the pipeline," only one number per
variable's own fit, since each variable has entirely separate ranks and an
entirely separate notion of how compressible it is.

In practice, rank selection is a per-variable, per-mode tradeoff: increasing
$R_{\text{lat}}$ or $R_{\text{lon}}$ will monotonically decrease
reconstruction error (more spatial basis vectors can only help fit the
observed data at least as well) at the cost of a larger, less compressed
representation; the usual diagnostic is to fit at several candidate ranks
and look for where the reconstruction-error curve's marginal improvement
flattens out (a scree-plot-style elbow, directly analogous to choosing a
truncation rank in PCA/POD), separately for each variable and, in principle,
separately in each spatial mode.

Physically, the fitted objects for one variable's fit mean:

- $\mathbf{U}^{(v,\text{lat})} \in \mathbb{R}^{I_{\text{lat}} \times
  R_{\text{lat}}}$ and $\mathbf{U}^{(v,\text{lon})} \in
  \mathbb{R}^{I_{\text{lon}} \times R_{\text{lon}}}$ together form a
  $(R_{\text{lat}}, R_{\text{lon}})$ **spatial basis for that one variable**
  — an empirical orthogonal basis of large-scale spatial patterns that
  variable tends to exhibit, analogous to EOFs/POD modes but with lat and
  lon kept as separate factors rather than one flattened spatial mode.
- $\mathbf{U}^{(v,\text{time})}$ (if compressed at all — see §4.3) plays the
  same role along the time axis: a basis of temporal patterns for that
  variable.
- $\mathcal{G}^{(v)} \in \mathbb{R}^{R_{\text{lat}} \times R_{\text{lon}}
  \times R_{\text{time}}}$ is that variable's own core, recording how
  strongly each combination of one lat-pattern, one lon-pattern, and one
  time-pattern contributes to reconstructing $\mathcal{X}^{(v)}$. It is
  specific to variable $v$; it is never combined with another variable's
  core (§4.4).

The notebook's own reported shapes (`matrix_core.shape == (10, 20, 12)` for
its one exploratory `sst` fit) exemplify this format only — they are not
target values for any variable in the trained pipeline.

### 5.1 Spatially-resolved error diagnostics: where does reconstruction error concentrate?

The aggregate reconstruction-error number above is a single scalar; two
ranks (or two variables) can report similar aggregate error while their
errors are distributed completely differently across the domain — one
rank polishing already-easy regions further, another genuinely resolving
harder structure. Task 07's rank-selection work (2026-09-26,
`tasks/07-tucker-dmd-rank-selection.md` §5) developed four complementary
diagnostics for this, implemented in
`notebooks/07_tucker_reconstruction_visual_check.py` and applied to a
handful of cached true-field time-slices against a saved `(core,
factor_lat, factor_lon)` fit:

1. **Coastal vs. open-ocean RMSE.** Ocean cells within a small band
   (default 3 grid cells) of land vs. everything else. Reconstruction
   error is consistently higher near the coastline — a real but moderate
   effect (a ~1.3x ratio measured on `wsh` at 20% rank), since land-ocean
   is the domain's sharpest, most permanent discontinuity, and a Tucker
   fit's factor matrices are smooth basis functions spanning the *entire*
   lat or lon axis — a finite number of smooth global functions cannot
   cleanly represent an arbitrary, irregular masked boundary (the same
   flavor of issue as Gibbs-phenomenon ringing near a discontinuity in a
   truncated Fourier series).
2. **Gradient-conditioned error, restricted to open ocean only.**
   Correlate `|error|` against the true field's local spatial-gradient
   magnitude (`np.gradient`), for ocean cells far enough from land that
   the gradient stencil never touches the land sentinel value. This
   isolates the *same* mechanism as (1) but operating on open-ocean
   phenomena themselves (storm cells, eddies, fronts) rather than the
   coastline: measured at 0.75 (strong) on `wsh` at 20% rank, with
   open-ocean cells binned into gradient-magnitude quartiles showing the
   bottom three quartiles (flattest 75%) sitting close together while the
   sharpest quartile alone carried more than 3x the error of the other
   three combined.
3. **Upper-half / lower-half RMSE ratio**, a single-number summary of (2):
   `RMSE(sharper 50% of open-ocean cells by local gradient) /
   RMSE(flatter 50%)`. Near 1 means error is roughly uniform regardless of
   local structure; increasing above 1 means error is concentrating
   specifically where the true field changes rapidly in space (2.52x on
   `wsh` at 20% rank).
4. **Shape of the open-ocean `|error|` distribution itself** — not the
   per-quartile group-RMSEs in (2), which are only 4 numbers, too few to
   support a moment estimate. Instead: the percentile ratio `(P100 - P50)
   / IQR`, skewness, and excess kurtosis of the full per-cell `|error|`
   sample. Large values of all three indicate a small minority of cells
   carry disproportionate error rather than error being spread evenly
   (24.5 / 3.0 / 10.8 respectively on `wsh` at 20% rank — strongly
   right-skewed and heavy-tailed, consistent with (1)-(3)).

**Why track all four rather than pick one**: they answer related but
distinct questions (how much worse is the coastline specifically; does the
same effect hold in the open ocean; how concentrated is that effect in one
number; what does the full error distribution's shape say independent of
any spatial conditioning). When comparing ranks, agreement across all four
is reassuring but not the interesting case — the useful case is
*disagreement*: e.g. a higher rank lowering the half-ratio while
skewness/kurtosis stay flat would mean the extra rank is reducing error in
the bulk without actually resolving the sharp-transition cells, a
materially different (and less useful) kind of improvement than a drop in
the single aggregate `err_pct` number alone would suggest.

## 6. Open questions and limitations

- **Cross-variable coupling (repeated from §4.4).** The per-variable design
  corrected in this document has no built-in mechanism for cross-variable
  correlation. If wind–wave (or other cross-variable) coupling matters
  physically, something outside of Tucker itself must capture it — a later
  joint stage over the compressed per-variable coefficients, or some other
  mechanism. This document deliberately does not propose which.
- **Masked-HOOI convergence is not guaranteed to be global.** As noted in
  §2.8, masked HOOI converges monotonically to a stationary point of a
  non-convex objective, not provably to the global optimum, and large
  contiguous unobserved regions (large landmasses with little interior
  coastline) are a realistic case where a poor local completion is possible.
  This is a genuine limitation of the masking approach, not merely a
  theoretical caveat — it should be checked empirically (e.g. sensitivity to
  initialization) where large land masses dominate a variable's grid.
- **Rank selection is not automated.** §5's elbow-of-the-reconstruction-
  error-curve approach is a diagnostic, not an algorithm; nothing in this
  document specifies an automatic stopping rule for $(R_{\text{lat}},
  R_{\text{lon}}, R_{\text{time}})$ per variable. Rank choice remains a
  per-variable judgment call to be made (and re-checked) by whoever trains
  the model, informed by the reconstruction-error diagnostic in §5.
