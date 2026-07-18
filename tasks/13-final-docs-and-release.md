# Task 13 — Final docs and release

**Depends on:** all previous tasks. **Blocks:** nothing (terminal task).

## Goal

Close out the remaining `docs/model_design.md` §12 "done" checklist items
not already covered by earlier tasks.

## Changes

1. **`README.md`** — install instructions, a minimal example (the §1 code
   sketch: `WeatherModel.load`, `sample`, `sample_batch`, `simulate_field`),
   link to `docs/model_design.md` (§12.8).
2. **`CHANGELOG.md`** — Keep a Changelog format; add a `v0.1.0` entry
   describing the initial release (tier-1 variables; hierarchical
   Tucker+DMD+OU+local-residual architecture). Note explicitly that this
   design is v2 of the handoff and v1 is preserved in git history for
   reference (§12.9).
3. Walk every item in `docs/model_design.md` §12 and confirm it's checked:
   - (1) `poetry install` + `poetry run pytest` green.
   - (2) `test_pipeline.ipynb` produces a fitted model + validation PDF
     end-to-end (task 11).
   - (3) Public API (`weather/__init__.py`: `WeatherModel`, `sample`,
     `sample_batch`, `simulate_field`, `full_report`) has full docstrings
     and type hints.
   - (4) CI green on Python 3.14: `black --check`, `isort --check-only`,
     `mypy --strict src/`, `pytest --cov` ≥85%.
   - (5)-(7) the full unit/grid-convention/integration test list from tasks
     01-12 — spot-check the exhaustive list in §12.5-§12.7 against what
     actually exists in `tests/`, not just against what each task file
     claimed to add.
   - (8)-(9) documentation, this task.
4. Tag `v0.1.0` once everything above is green. **This is a release
   action — confirm with the user before tagging or pushing**, per this
   repo's general working agreement (destructive/shared-state actions need
   explicit sign-off even when a design doc describes them as the obvious
   next step).

## Acceptance criteria

- Every item in `docs/model_design.md` §12 is verifiably true, not just
  plausible.
- `poetry run pytest --cov=src` reports ≥85% coverage.
- `poetry run black --check src tests && poetry run isort --check-only src
  tests && poetry run mypy --strict src` all pass.
