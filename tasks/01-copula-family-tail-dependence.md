# Task 01 — Copula family / tail dependence

**Depends on:** — · **Blocks:** 02 (shares this task's experiment)
**Open question:** `docs/open_questions.md` §4.2 — OPEN (distinct axis from §4.1)
**Primary source docs:** `docs/gaussian_copula_and_marginals.md` §6.2, `docs/physical_consistency_constraints.md` §6

## 1. What question this will try to answer

Independently of *where* Gaussianization sits (§4.1/Task 00), what *kind*
of copula should be used at all? The current design's implicit Gaussian
copula has provably zero upper-tail dependence ($\lambda_U=0$ for
$\rho<1$), which both source docs identify as the root cause of the
physical-consistency violations the rejection-sampling layer
(`physical_consistency_constraints.md`) exists to patch after the fact. A
$t$-copula (tunable $\lambda_U>0$) or GPD-spliced tails are named as
root-cause alternatives, but neither document resolves which to use, nor
whether to do so instead of, or in addition to, rejection sampling.

## 2. Steps to answer it

1. Load a physically-coupled pair with a documented mechanism —
   `was`/`wsh` (wind speed / wind-sea height, coupled via the
   Pierson-Moskowitz relation, `physical_consistency_constraints.md` §2.2)
   — over a multi-month window, at one or more representative ocean cells.
2. Gaussianize both variables via the real, current per-cell empirical-CDF
   copula (`train.copula.GaussianCopula`) — placement (i), today's default.
3. Measure **empirical upper-tail dependence** directly:
   $\hat\lambda_U(q) = \Pr(U_{wsh}>q \mid U_{was}>q)$ at $q$ near 1 (e.g.
   0.95), where $U$ are the Gaussianized values passed through $\Phi$.
4. Compare against the Gaussian copula's own implied joint upper-tail
   probability (from the fitted correlation $\rho$ and a bivariate normal)
   — the gap between implied and empirical is the direct, real-data
   evidence this question currently lacks.
5. **Decision rule:** if empirical $\hat\lambda_U$ is materially nonzero
   and the gap versus the Gaussian-implied value is large, that's a
   concrete case for a $t$-copula or GPD-tail construction (root-cause
   fix). If the gap is small, the current Gaussian copula plus
   rejection-sampling (Task 02's framing) is adequately supported by the
   data as-is. Repeat across several $q$ and several cells/seasons before
   concluding — one cell is illustrative only.

## 3. Notebook(s) needed

- `notebooks/01_copula_family_tail_dependence.ipynb` — loads `was`/`wsh`
  over a 3-month era-3 window, computes empirical $\hat\lambda_U$ at one
  representative cell, and compares it against the Gaussian-implied joint
  tail probability. Also used by Task 02 (§3 there).

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — measured $\hat\lambda_U$ across cells/seasons, whether it changes
`open_questions.md` §4.2's status tag, and any follow-on task this spawns.)_
