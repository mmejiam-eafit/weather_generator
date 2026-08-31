# Task 00 — Bootstrap: docs, AGENTS.md, tooling

**Depends on:** none. **Blocks:** every other task (all of them cite
`docs/model_design.md` section numbers that only exist once this lands).

## Goal

Make `model_design_v2.md` the repo's single source of truth before any code
task starts, and confirm tooling is ready.

## Changes

1. **Replace `docs/model_design.md` (v1, 1447 lines) with the content of
   `model_design_v2.md` (repo root, 1794 lines).** Move, don't merge — v2 is
   fully self-contained (per its own "Audience" note: "every convention,
   formula, and restriction the agent needs is stated here, not by
   reference") and supersedes v1 entirely.
   ```
   git mv model_design_v2.md docs/model_design.md
   ```
2. Confirm no root-level `model_design_v2.md` remains after the move.
3. **Update `AGENTS.md`** (633 lines) — it's the operating manual for coding
   agents in this repo and currently describes v1 module names throughout.
   Fix at least:
   - Module table (~line 62-69): add a row for `train/local_residual.py`;
     remove the `train/ou.py` row (folded into `dmd.py` — task 06).
     `train/directional.py` stays a single file/row — task 02 rewrites its
     one class into two (`ComingFromToUV`, `GoingToToUV`), no file split.
   - Directory tree (~line 324-343): add
     `notebooks/verify_ocean_grid_orientation.ipynb`.
   - Code examples (~line 116-135, 266-304): `exact_dmd` → `dmd_fit`;
     `DirectionalDecomposer` → `ComingFromToUV`/`GoingToToUV` (still
     imported from `train.directional`); `hosvd` → `tucker_fit`.
   - Any `(see docs/model_design.md §X)` citation affected by the
     section-number shifts in `tasks/README.md`.
4. **Tooling — confirm, don't redo.** `pyproject.toml` already pins
   `tensorly (>=0.9.0,<0.10.0)` and `pydmd (>=2025.8.1,<2026.0.0)`; both were
   confirmed installed and importable under Python 3.14 in a prior session
   (`tensorly==0.9.0`, `pydmd==2025.8.1`, harmless `SyntaxWarning`s only — no
   action needed unless meaningful time has passed). `.github/workflows/ci.yml`
   already runs `black --check`, `isort --check-only`, `mypy`, `pytest --cov`
   per §2.8/§12.4 — no action needed there either. This step is a
   verification, not new work:
   ```
   poetry install
   poetry run python -c "import tensorly, pydmd; print(tensorly.__version__)"
   poetry run python -c "from importlib.metadata import version; print(version('pydmd'))"
   ```
   (`pydmd` has no `__version__` attribute — use `importlib.metadata` as
   shown, not `pydmd.__version__`, which raises `AttributeError`.)

## Tests

None — this is a docs/tooling task, not a code change.

## Acceptance criteria

- `docs/model_design.md` is the v2 document (grep for a v2-only string, e.g.
  `"local_residual"`, to confirm); no `model_design_v2.md` remains anywhere
  in the repo.
- `grep -rn "exact_dmd\|hosvd(\|DirectionalDecomposer\|coming_from_uv\.py\|going_to_uv\.py" AGENTS.md` returns nothing.
- Both tooling verification commands in step 4 succeed.
