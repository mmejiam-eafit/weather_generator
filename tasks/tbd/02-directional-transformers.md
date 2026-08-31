# Task 02 — `ComingFromToUV` / `GoingToToUV` transformers in `train/directional.py`

**Depends on:** 01. **Blocks:** 08.

**Revision note (2026-07-17):** an earlier version of this task split the
two classes into separate `coming_from_uv.py`/`going_to_uv.py` files. Judged
overkill for two closely-related ~30-line transformer classes — both now
live side by side in a single `directional.py`, matching v1's original
single-file layout for this concern. If you're picking up stale context
(from `docs/model_design.md`, `AGENTS.md`, or elsewhere) that still shows
two files, it's been superseded — this file is current.

## Current state

`src/train/directional.py` has one class, `DirectionalDecomposer`, handling
all five archive `(speed, direction)` pairs uniformly as `coming_from` — it
unconditionally applies `+180°` before computing `(u, v)`
(`theta_going_to_rad = np.deg2rad((result[direction_var] + 180.0) % 360.0)`,
line 74). This is wrong for `wad`/`ocd`, which are `going_to` (§2.4). It also
hardcodes `("was", "wad", "u_wind", "v_wind")` as the wind pair (line 17),
which is wrong per the source-of-truth finding (§3.7: `wad_come_from` is
correct, `wad` is unused).

## Changes — `src/train/directional.py` (rewrite in place, same filename)

Replace the single `DirectionalDecomposer` class with two classes,
`ComingFromToUV` and `GoingToToUV`, defined side by side in this one module.

1. **`ComingFromToUV(BaseEstimator, TransformerMixin)`.** Exact behavior per
   §5.3/§2.4:
   - `__init__(self, pairs: list[tuple[str, str, str, str]]) -> None` —
     `pairs` is **required**, no default (§3.7: "no silent defaulting"; an
     agent adding a new directional variable must explicitly choose a
     transformer). Each tuple is
     `(speed_var, direction_var, u_out_name, v_out_name)`.
   - `fit(X, y=None)`: `self.pairs_ = tuple(self.pairs)`. Unlike v1's
     `DirectionalDecomposer.fit`, do **not** filter against a hardcoded
     module-level pair list — the caller states pairs explicitly at
     construction (the canonical list lives only in `weather/model.py`'s
     pipeline construction, task 08).
   - `transform(X)`: for each pair —
     `theta_to = (X[direction_var] + 180.0) % 360.0`;
     `u = speed * sin(deg2rad(theta_to))`; `v = speed * cos(deg2rad(theta_to))`;
     drop `speed_var`/`direction_var`, add `u_out_name`/`v_out_name`.
     Non-owned variables (both non-directional ones and pairs owned by the
     other transformer) pass through unchanged (§3.7, §5.3).
   - `inverse_transform(X)`: `speed = hypot(u, v)`;
     `theta_to = atan2(u, v) mod 360` (**sin, cos argument order — not
     `(y, x)`**, §2.4); `theta_from = (theta_to + 180) mod 360`; drop
     `u_out_name`/`v_out_name`, restore `speed_var`/`direction_var`. Must
     reconstruct `θ_from`, not `θ_to` (§2.4: "`ComingFromToUV.
     inverse_transform` emits: `theta_coming_from`").
2. **`GoingToToUV(BaseEstimator, TransformerMixin)`**, same shape but:
   - `transform`: `u = speed * sin(deg2rad(theta_to))`,
     `v = speed * cos(deg2rad(theta_to))` **directly** — no `+180`.
   - `inverse_transform`: `theta_to = atan2(u, v) mod 360` directly — no
     `+180` shift, emits `theta_to` as-is (§2.4: "`GoingToToUV.
     inverse_transform` emits: `theta_going_to`").
3. Neither class hardcodes a canonical pair list (unlike v1's
   `_DIRECTIONAL_PAIRS` module constant). The canonical tier-1+ assignment
   lives only in `weather/model.py`'s pipeline construction (task 08), per
   §5.3:
   - `ComingFromToUV`: `(wsh, wsd, u_wsea, v_wsea)`,
     `(was, wad_come_from, u_wind, v_wind)` — tier 1;
     `(pwh, pwd, u_peak, v_peak)` — tier 2 (pending expert convention
     confirmation, see `docs/model_design.md` §2.3 and the saved project
     memory on this);
     `(pswh, pswd, u_pswell, v_pswell)` — tier 3 (same pending status).
   - `GoingToToUV`: `(ocs, ocd, u_current, v_current)` — tier 3 only; **empty
     list for tier 1** (§5.3: "Even when `GoingToToUV` has an empty pair
     list (tier 1), the transformer is present in the Pipeline as a
     scaffold").
4. Module docstring should describe both classes and cite §2.4/§3.7/§5.3,
   same as any other module holding more than one closely-related class.

## Tests — `tests/train/test_directional.py` (rewrite in place, mirrors the single source file)

Per this repo's testing convention ("tests live under `tests/` mirroring
`src/`"), one test file for one source file. Replace the old
`DirectionalDecomposer`-only tests with coverage for both classes — organize
as clearly-named functions/groups (e.g. two `test_coming_from_to_uv_*` /
`test_going_to_to_uv_*` prefixes, or two test classes) within this one file:

- `test_cardinal_signs` for **both** classes — four `θ ∈ {0°, 90°, 180°,
  270°}` cases each, from §2.4's respective coming-from/going-to tables.
  **Load-bearing** for both.
- `test_uv_roundtrip` for both — `(speed, θ) → (u,v) → (speed', θ')` within
  `1e-10`.
- `test_inverse_transform_returns_native_convention` for both —
  `ComingFromToUV` reconstructs `θ_from`, `GoingToToUV` reconstructs `θ_to`.
- `test_pass_through_non_paired_vars` for both.

Keep **`tests/train/test_pipeline_composition.py`** as its own separate
file (unaffected by this consolidation — it tests cross-transformer
composition, not either transformer in isolation):
`test_transformers_commute_on_disjoint_pairs` — `ComingFromToUV(...) →
GoingToToUV(...)` and the reverse order produce identical output when their
pair lists are disjoint (they always should be). Import both classes from
`train.directional`.

## Acceptance criteria

- Both transformers independently pass their own four-case cardinal-direction
  table (§2.4) — this is the load-bearing sign-convention check; getting it
  wrong silently mirrors every downstream `(u, v)` field.
- `grep -rn "DirectionalDecomposer" src/ tests/` returns nothing after this
  task (the class is gone; the module path `train/directional.py` is
  unchanged from v1, so it should **not** appear in that grep — only the
  old class name should).
