# Task 14 — Verify `copulas` dev dependency stays isolated from `train/copula.py`

**Depends on:** none. **Blocks:** none (informs task 12's import-boundary tests).

## Background / concern

`pyproject.toml` now lists `copulas (>=0.14.1,<0.15.0)` under `[dependency-groups] dev`
(line 34) — added directly by the user, not by any migration task. `copulas`
(the sdv-dev PyPI package) ships ready-made multivariate copula classes
(e.g. `GaussianMultivariate`) that are superficially similar in name/purpose
to this repo's own `train.copula.GaussianCopula`. The concern raised: having
this package installed could tempt a future editor (human or agent) to
"helpfully" swap the hand-rolled implementation for a call into that
library — silently changing behavior (a joint multivariate fit vs. this
module's deliberate per-cell *independent* fit) and dropping the documented
§3.5 zero-tail-dependence caveat. Because `copulas` is a **dev-only**
dependency, any such import inside `src/` would also be a hard
dependency-group violation: `src/` must stay installable/runnable without
the `dev` group.

## Current state (confirmed at task-creation time)

- `grep -rn "copulas" src/ tests/ notebooks/` returns **nothing** — the
  package is not referenced anywhere outside `pyproject.toml`.
- `src/train/copula.py` contains only the original hand-rolled
  implementation: module-level helpers `_searchsorted_per_column`,
  `_value_to_uniform`, `_uniform_to_value`, and the `GaussianCopula(
  BaseEstimator, TransformerMixin)` class (`__init__(n_bins=500)`, `fit`,
  `transform`, `inverse_transform`, fitted attributes `bin_probs_` and
  `sorted_anomalies_binned_`). No `import copulas` / `from copulas import
  ...` anywhere in the file.
- This is the correct starting state. This task exists to make that
  explicit and re-checkable, not because anything is currently wrong.

## Verify

1. `src/train/copula.py` still contains only the original implementation
   described above — no `import copulas` / `from copulas import ...`
   anywhere in the file.
2. `grep -rn "^import copulas\|^from copulas" src/` (repo root) returns
   nothing — `copulas` is not imported anywhere under `src/`.
3. `poetry run pytest tests/train/test_copula.py -v` still passes
   unchanged — confirms behavior, not just import absence.
4. `train.copula.GaussianCopula`'s public API (`__init__(n_bins)`, `fit`,
   `transform`, `inverse_transform`) and its fitted attribute names
   (`bin_probs_`, `sorted_anomalies_binned_`) are unchanged — a library
   swap would very likely rename or restructure these.
5. `copulas` is still declared under `[dependency-groups] dev` in
   `pyproject.toml`, **not** `[project] dependencies` — if it's ever
   promoted to a runtime dependency, that's a signal someone intends
   `src/` to actually depend on it, and this task's assumption needs
   re-litigating with the user, not silently accepted.

## Acceptance criteria

- All five checks above pass with no discrepancy.
- If any discrepancy is found (an import of `copulas` inside `src/`, a
  changed public API, or `copulas` promoted to a runtime dependency), stop
  and flag it for explicit user review before proceeding — do not
  unilaterally "fix" it by picking a side (keep-bespoke vs. adopt-library);
  that's a design decision, not a bug fix.

## Suggested follow-up (not required by this task)

Task 12 ("Integration tests + import boundaries") already plans a
`tests/test_import_boundaries.py`. Once that lands, add a permanent
assertion there so this guard doesn't depend on a one-time manual check
ever again — e.g. a static grep-based test, or asserting `"copulas" not in
{m.split(".")[0] for m in sys.modules}` after importing `train.copula`.
