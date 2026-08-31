# Task 01 — `core/io.py`: direction origin codes

**Depends on:** 00. **Blocks:** 02, 03.

## Current state

`src/core/io.py`'s `_DIRECTION_ORIGIN_CODES` (line 24) has only `wsd`, `pwd`,
`pswd`, `wad` — `wad_come_from` and `ocd` are deliberately absent, and the
comment above it explains why (v1 believed `wad_come_from` was a redundant
duplicate of `wad`, and that `ocd`'s convention was unverified). Both beliefs
are now known wrong (`docs/model_design.md` §2.3, §2.4, §5.1's
"Implementation prerequisite" paragraph).

`decode_compass_code` itself (line 94-109) is **already correct and
convention-agnostic** — no logic change needed, only a docstring fix: it
currently claims the output is always "coming from" convention, which
contradicts §2.3 ("Returns raw decoded angle in the archive's native
convention for that field — either 'coming from' or 'going to'").

## Changes — `src/core/io.py`

1. Add to `_DIRECTION_ORIGIN_CODES`: `"wad_come_from": 1` and `"ocd": 1`.
2. Rewrite the comment above `_DIRECTION_ORIGIN_CODES` (line 20-23) — remove
   the "`ocd` is deliberately absent... `wad_come_from` is deliberately
   absent" language. Replace with: `wad` is still origin-code-decodable but
   unused downstream (`wad_come_from` is the source of truth for wind
   direction — see task 02); `ocd`'s `going_to` convention is provisional
   pending §2.7 ocean-grid re-verification, but the *decode* (origin
   code `1`, same as `wad`) is settled regardless of that open question.
3. Fix `decode_compass_code`'s docstring `Returns` section (line 106-107):
   replace "Meteorological 'coming from' convention" with something like
   "Native on-disk convention for that field (coming_from or going_to — see
   §2.3); convention handling is the caller's responsibility, not this
   function's."
4. Fix `load_snapshot`'s docstring `Raises` section (line 137-139): it's no
   longer true that `ocd`/`wad_come_from` are unverified/unknown — both now
   decode successfully via `_DIRECTION_ORIGIN_CODES`. Update or remove that
   caveat so the docstring doesn't contradict the new behavior.
5. **Add `load_raw`** — a lower-level function distinct from `load_snapshot`,
   needed by task 03's consistency check (which must read raw `wad`/
   `wad_come_from` codes before/regardless of the origin-code allowlist):
   ```python
   def load_raw(data_dir: Path, t: datetime, variable: str) -> np.ndarray:
       """Load one variable's raw on-disk array with no decoding (see §5.1).

       Parameters
       ----------
       data_dir : Path
       t : datetime
           UTC-aware.
       variable : str

       Returns
       -------
       np.ndarray
           Untouched on-disk dtype — int8 for direction fields, float16 for
           continuous fields. Callers decode/upcast themselves.
       """
       return np.load(snapshot_path(data_dir, t, variable))
   ```

## Tests — `tests/core/test_io.py`

- `test_wsd_compass_codes`, `test_wad_compass_codes` (§12.5, already exist —
  confirm still passing; unaffected by this change).
- Add: decode `wad_come_from` and `ocd` codes via
  `decode_compass_code(..., origin_code=1)` and confirm they match `wad`'s
  decode formula exactly (same origin code — §2.3's "no origin-code
  correction" note). This guards against someone re-introducing a different
  origin code for `ocd` by mistake later.
- Add: `load_snapshot` round-trip cases for `wad_come_from` and `ocd`
  (currently these raise `ValueError`; after this task they must return
  decoded `float64` degree arrays like every other direction field).
- Add: `load_raw` returns the untouched on-disk array — `int8` for a
  direction field, `float16` for a continuous field — with no upcasting or
  decoding.

## Acceptance criteria

- `core.io.load_snapshot(data_dir, t, "wad_come_from")` and
  `core.io.load_snapshot(data_dir, t, "ocd")` both succeed and return
  `float64` arrays in `[0, 360)`.
- `core.io.load_raw` never upcasts or decodes.
