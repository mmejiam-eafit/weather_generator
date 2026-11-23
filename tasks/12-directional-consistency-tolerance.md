# Task 12 — Directional-consistency tolerance calibration

**Depends on:** — · **Blocks:** —
**Open question:** `docs/open_questions.md` §8.1 — PARTIALLY ANSWERED
**Primary source docs:** `docs/physical_consistency_constraints.md` §5, `docs/correlated_variables.md` §3.2, §3.3

## 1. What question this will try to answer

Should the physical-consistency constraints check direction, not just
magnitude? `correlated_variables.md` §3.2/§3.3 already supplies the missing
physical grounding: the wind/wind-sea directional-alignment system has a
derived, stable equilibrium at zero angular offset, with an explicit
relaxation rate shared with the magnitude-growth dynamics. This licenses a
concrete candidate constraint: flag as directionally inconsistent any query
point where the angular offset between wind-sea direction and wind
direction exceeds a tolerance $\delta_{dir}$, **unless** the local sea
state is young/duration-limited — mirroring the existing energy-budget
constraint's young-sea exception. The missing mechanism is now specified;
the missing calibration is not — this task supplies it.

**Update (task 07, 2026-09-29):** two recent commits
(`752d6d6` canonical `(time, lat, lon)` dim order, `c97641c` relaxing the
`wad`/`wad_come_from` consistency check for quantization noise) touch
`_check_wad_wad_come_from_consistency` in `src/train/ingest.py` — but that
is an *internal* redundancy check confirming two encodings of the *same*
wind-direction field agree (tolerating ~0.1% of cells for compass-code
quantization rounding), not this task's physical `wsd`-vs-`wad_come_from`
alignment question. No wave-age proxy, percentile calibration, or
$\delta_{dir}$ logic exists anywhere in `src/`. This task remains fully
open, unrelated to that fix.

## 2. Steps to answer it

1. Load `wad_come_from`, `wsd`, `was`, `wsh` over a multi-month window.
2. Compute the real empirical angular offset $|\theta| =
   |\angle(\text{wsd}) - \angle(\text{wad\_come\_from})|$ (minimum circular
   distance) across all ocean cells and times.
3. Following `physical_consistency_constraints.md` §5's own calibration
   principle: compute the percentile distribution of $|\theta|$ (e.g. 90th,
   95th, 99th) — a tolerance set from real-data percentiles, not guessed.
4. Stratify by a wave-age proxy (illustratively, `wsh/was`, since the
   reduced-ODE fixed point $wsh^* \sim (\kappa/c_{ds})\cdot was^2$ ties
   larger ratios to more-developed seas) into young vs. developed terciles,
   and compare the offset distribution between them — this tests whether
   §8.1's "unless young" exception is empirically justified, i.e. whether
   young-proxy cells really do show larger tolerated offsets.
5. **Decision rule:** set $\delta_{dir}$ so a target percentile (95th/99th)
   of genuine archive observations falls inside the tolerance, the same way
   the existing three constraints' tolerances were calibrated, and report
   whether the young/developed split supports a wave-age-gated tolerance
   or a single fixed one.

## 3. Notebook(s) needed

- `notebooks/12_directional_consistency_tolerance.ipynb` — loads
  `wad_come_from`/`wsd`/`was`/`wsh` over a 3-month era-3 window, computes
  the empirical offset distribution, and stratifies by the wave-age proxy.

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — the calibrated $\delta_{dir}$ value(s), whether the young/developed
stratification supports the proposed exception, whether it changes
`open_questions.md` §8.1's status tag, and any follow-on task this spawns.)_
