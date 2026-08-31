# Task 02 — Rejection sampling vs. copula fix

**Depends on:** 01 (same underlying experiment; architecture-side framing) · **Blocks:** —
**Open question:** `docs/open_questions.md` §8.3 — OPEN ("= §4.2")
**Primary source docs:** `docs/physical_consistency_constraints.md` §6

## 1. What question this will try to answer

`open_questions.md` §8.3 is explicit: this is the same open question as
§4.2 (Task 01), viewed from the constraints side rather than the copula
side. The team has chosen rejection sampling for now (fast, safe, exact
given whatever model exists per the sampling-theoretic proof in
`physical_consistency_constraints.md` §2.4); fixing the copula's tail
dependence remains the more root-cause-oriented alternative. This task
doesn't re-derive the theory — it asks the practical question the theory
can't answer alone: given real archive data, how much would each approach
actually cost/catch?

## 2. Steps to answer it

This task **reuses Task 01's experiment and data directly** — do not
refit a second time. Using the same `was`/`wsh` empirical vs.
Gaussian-implied joint upper-tail comparison from
`notebooks/01_copula_family_tail_dependence.ipynb`:

1. Estimate the **rejection-sampling cost**: given the Gaussian copula's
   implied joint distribution and the empirical constraint pass-rate
   (`physical_consistency_constraints.md` §2.1–2.3's checks), what fraction
   of draws would need to be rejected/redrawn to reach the true joint
   behavior — i.e. the expected `1/Z` draws-per-accepted-sample cost from
   §2.4's proof, estimated from real data rather than assumed.
2. Estimate the **copula-fix benefit**: how much of the gap between
   empirical and Gaussian-implied tail probability (Task 01, step 4) would
   a $t$-copula with a plausibly-fit degrees-of-freedom parameter close,
   roughly?
3. **Decision rule:** if the real rejection rate is low (cheap to keep
   rejection sampling) and/or a $t$-copula's benefit is small on real data,
   staying with rejection sampling for now is empirically supported. If the
   real rejection rate is high (expensive, and per §2.4's proof, a signal
   the copula is a poor fit for the true joint physics there), that's a
   concrete case to prioritize the copula-fix path instead.

## 3. Notebook(s) needed

- *Shares* `notebooks/01_copula_family_tail_dependence.ipynb` (§3 there
  reuses the fitted copula and empirical measurements for this task's
  rejection-rate/benefit estimate).

## 4. Conclusion

_(Blank. Fill in only after Task 01's notebook has been run against real
archive data — estimated rejection rate, estimated copula-fix benefit,
whether it changes `open_questions.md` §8.3's status tag, and the
build/defer recommendation this task should produce.)_
