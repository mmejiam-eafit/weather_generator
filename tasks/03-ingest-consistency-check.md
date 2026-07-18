# Task 03 — Ingest: `wad`/`wad_come_from` consistency check

**Depends on:** 01. **Blocks:** 08.

## Current state

`src/train/ingest.py`'s `load_training_window` has no consistency check —
it loads exactly the requested `variables` list and nothing else. Before
task 01, `wad_come_from` wasn't even loadable (excluded from
`_DIRECTION_ORIGIN_CODES`); after task 01 it is, which is a prerequisite for
this task's check.

## Changes — `src/train/ingest.py`

Add the check specified verbatim in §5.1:

```python
def _check_wad_wad_come_from_consistency(data_dir: Path, t: datetime) -> None:
    """Guard against upstream archive convention drift (§3.7, §5.1)."""
    wad = decode_compass_code(load_raw(data_dir, t, "wad"), origin_code=1)
    wad_from = decode_compass_code(load_raw(data_dir, t, "wad_come_from"), origin_code=1)
    diff = np.abs(((wad - wad_from - 180.0 + 180.0) % 360.0) - 180.0)   # min circular distance
    max_dev = float(np.nanmax(diff))
    if max_dev > 5.0:   # tolerance for 22.5° discretization + rounding
        raise ValueError(
            f"wad and wad_come_from do not differ by 180° at {t.isoformat()}. "
            f"Max deviation: {max_dev:.2f}°. Archive convention may have changed."
        )
```

- Call once on the **earliest** snapshot with both `wad` and `wad_come_from`
  present, before the main per-variable load loop begins.
- Call again on the **last** snapshot of the window (§5.1: "if the archive
  convention changed *within* the training window, the second check catches
  it").
- Log `max_dev` on success for both checks, via the same
  `sys.stderr`/`warnings` pattern this file already uses for gap handling.
- Runs **unconditionally**, regardless of whether `"wad"`/`"wad_come_from"`
  appear in the caller's `variables` argument — this is an archive-integrity
  guard, not a modeled-variable concern (§5.1; §9 gotcha 5a: "do not remove
  it as 'unnecessary'... if it ever fails, that's the signal — not a false
  alarm — that the archive convention changed").
- Uses `core.io.load_raw` (task 01) + `core.io.decode_compass_code`
  directly — **not** `load_snapshot` — since it needs both `wad` and
  `wad_come_from` regardless of what's in the training `variables` list or
  the `_DIRECTION_ORIGIN_CODES` allowlist state.

## Tests

- `tests/core/test_io.py::test_wad_and_wad_come_from_consistency_check`
  (§12.5) — construct a synthetic snapshot pair that differs by exactly 180°
  (passes) and one that doesn't (raises `ValueError`).
- `tests/train/test_ingest.py`: add a case confirming `load_training_window`
  raises when the *first*-snapshot check fails, and a separate case
  confirming it also checks the *last* snapshot — e.g. corrupt only the last
  snapshot's `wad_come_from` and confirm the whole window load still raises,
  proving the check isn't only run once at the start.

## Acceptance criteria

- A synthetic archive with a mid-window or end-of-window `wad`/
  `wad_come_from` mismatch fails `load_training_window` with a clear
  `ValueError`, not a silent bad fit.
- A correctly-related synthetic archive loads without warnings beyond the
  informational success log line.
