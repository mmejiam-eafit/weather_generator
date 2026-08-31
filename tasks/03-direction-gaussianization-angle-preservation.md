# Task 03 — Direction preservation under independent (u,v) Gaussianization

**Depends on:** 00 (soft — result is specific to placement (i), the current default; re-check if Task 00 changes placement) · **Blocks:** —
**Open question:** `docs/open_questions.md` §2.4 — PARTIALLY MOOT (magnitude case resolved by §2.3; angle case remains open)
**Primary source docs:** `docs/gaussian_copula_and_marginals.md` §3, `docs/variable_taxonomy.md` §3.4

## 1. What question this will try to answer

`u` and `v` (a Class B/C directional pair) are each Gaussianized via their
own *separate* per-cell empirical CDF. §2.3's resolution (a dedicated
`log E` tensor as the authoritative magnitude source) made the **magnitude**
half of this concern largely moot — `hypot(u,v)` is no longer trusted for
magnitude at all. What's still genuinely open: does independent per-cell
Gaussianization of $u$ and $v$ distort the **direction**
$\operatorname{atan2}(v,u)$, which is still read from `u,v` directly (no
analogous dedicated-tensor fix exists for direction) and has no
alternative source to fall back on.

## 2. Steps to answer it

1. Load `wsh`/`wsd`, decompose into `u_wsea`/`v_wsea` via the real
   `train.directional.ComingFromToUV` transformer (§2's confirmed
   `wsd`="coming_from" convention).
2. Compute the reference angle `atan2(v_wsea, u_wsea)` directly from the
   true, ungaussianized components.
3. Fit `train.copula.GaussianCopula` on `u_wsea`/`v_wsea` independently
   per cell, then round-trip (`transform` → `inverse_transform`) and
   recompute the angle from the reconstructed components. This isolates
   the copula's own (Gaussianize + invert) distortion of angle.
4. **Extend to the real question**, not just the round-trip: draw
   perturbed samples in Gaussianized $Z$-space (not just invert the exact
   fit), invert back, and compare the resulting angle distribution against
   what a directly-decomposed direction (no `u,v`-mediated Gaussianization
   at all) would give. This is the step that actually tests whether
   *sampling* through this representation distorts direction, not just
   whether the fit is self-consistent.
5. **Decision rule:** a materially nonzero, systematic (not just noisy)
   angular bias introduced by step 3/4 is evidence this needs the same
   kind of dedicated-tensor fix §2.3 gave magnitude; a small, unbiased
   scatter suggests the current design is adequate for direction as-is.

## 3. Notebook(s) needed

- `notebooks/03_direction_gaussianization_angle_preservation.ipynb` —
  loads `wsh`/`wsd`, builds `u_wsea`/`v_wsea`, and runs the round-trip
  angular-error diagnostic (step 3). Step 4's perturbation-sampling
  extension is marked as the concrete next cell to add.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — measured angular error/bias, whether it changes `open_questions.md`
§2.4's status tag, and any follow-on task this spawns.)_
