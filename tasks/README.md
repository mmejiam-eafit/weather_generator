# Implementation tasks: v1 → v2 migration

Derived from `model_design_v2.md` (repo root, to be moved to `docs/model_design.md`
by task 00 — that's the source of truth for every formula/signature cited
below; read the referenced section before implementing, this index does not
repeat rationale). Ordered by dependency; do not skip ahead within a chain.

Legend: **(new)** file doesn't exist yet. **(rewrite)** replaces existing
content wholesale. **(edit)** modifies part of an existing file. **(delete)**
removed entirely.

**Status column** (emoji conventions per [gitmoji.dev](https://gitmoji.dev)
where a direct match exists; two entries have no gitmoji equivalent and are
noted as such):

| Emoji | Meaning |
|---|---|
| ⬜ | Todo — not started (no gitmoji equivalent; neutral placeholder) |
| 🚧 | In progress (`:construction:`) |
| ✅ | Done (`:white_check_mark:`) |
| 🔒 | Blocked (no gitmoji equivalent; repurposed `:lock:` for "can't proceed") |
| ♻️ | Refactored (`:recycle:`) — done, but the approach changed materially from this file's spec; check the task file's notes before trusting it verbatim |
| ⚰️ | Dropped (`:coffin:`) — deliberately not doing this; reason belongs in the task file, not just this table |

| # | Status | Task | Depends on | Key files |
|---|---|---|---|---|
| 00 | ✅ | Bootstrap: docs, AGENTS.md, tooling | — | `docs/model_design.md`, `AGENTS.md` |
| 01 | ✅ | `core/io.py` direction codes | 00 | `src/core/io.py` (edit) |
| 02 | ✅ | `ComingFromToUV`/`GoingToToUV` transformers | 01 | `src/train/directional.py` (rewrite) |
| 03 | ✅ | Ingest: `wad`/`wad_come_from` consistency check | 01 | `src/train/ingest.py` (edit) |
| 04 | ✅ | Climatology linear trend term | — | `src/train/climatology.py` (edit) |
| 05 | ✅ | Tucker → TensorLy wrapper | — | `src/train/tucker.py` (rewrite) |
| 06 | ✅ | DMD → PyDMD wrapper + mode-OU | — | `src/train/dmd.py` (rewrite), `src/train/ou.py` (delete) |
| 07 | ✅ | Local residual OU layer | 05, 06 | `src/train/local_residual.py` (new) |
| 08 | ✅ | `WeatherModel` rewrite | 02, 03, 04, 05, 06, 07 | `src/weather/model.py` (rewrite) |
| 09 | ⬜ | Sampling rewrite (coherent + local) | 08 | `src/weather/simulate.py` (rewrite) |
| 10 | ⬜ | Validation: hurricane-season checks + step-name fixes | 08, 09 | `src/weather/validate.py` (edit) |
| 11 | ⬜ | Notebooks | 08, 09, 10 | `notebooks/verify_ocean_grid_orientation.ipynb` (new), `notebooks/test_pipeline.ipynb` (rewrite) |
| 12 | ⬜ | Integration tests + import boundaries | 10 | `tests/weather/test_integration.py` (new), `tests/test_import_boundaries.py` (new) |
| 13 | ⬜ | Final docs + release | all | `README.md`, `CHANGELOG.md` |
| 14 | ✅ | Verify `copulas` dev dependency isolation | — | `src/train/copula.py` (verify only), `pyproject.toml` (verify only) |
| 15 | ✅ | Evaluate `copulas` as a replacement for `train/copula.py` | — | `notebooks/evaluate_copulas_replacement.ipynb` (new; verdict only, no `src/` change) |
| 16 | ✅ | Archive grid/cadence drift: era registry + ingest/model fixes | 01, 03, 08 | `src/core/archive_eras.py` (new), `src/train/ingest.py` (edit), `src/core/io.py` (edit), `src/weather/model.py` (edit) |

Update the status cell as work proceeds — this table is the at-a-glance
index; each task file is the detail. Keep them in sync: if a task's approach
changes materially from its file's original spec (♻️), note *why* in that
file, not just here.

Each task file states: current state, exact changes, new/updated tests (file
+ test name, cross-referenced to `model_design_v2.md` §12's exhaustive list),
and acceptance criteria. Tests belonging to a module are specified inside
that module's task file, not gathered into a separate pass — the exceptions
are import-boundary enforcement and multi-module integration tests (task 12),
which don't belong to any single module.

## Section-number trap

Once task 00 lands, `docs/model_design.md` *is* v2 — cite its numbering, not
whatever a stale v1 comment in the current source says. Three places the
numbering shifted between v1 and v2:

- v1 §3.9 ("three-package split") is v2 **§3.14** — v2 inserts five new
  subsections (§3.9-§3.13: climatology trend, hierarchical decomposition,
  library adoption, sklearn hybrid, escalation paths) before it.
- v1 §5.7 was "OU fit" (the mode-level fit). In v2, that content moved into
  **§5.6.1** (a subsection of DMD, since the mode-OU fit now lives in
  `train/dmd.py` — task 06). v2's **§5.7** is an entirely different, new
  section: the local-residual layer (task 07).
- v1 §6 was one flat numbered list. v2 §6 is split into **§6.A/§6.B/§6.C/§6.D**
  with items 1-19 running continuously across the split.

## Known doc bug to work around, not propagate

§8.6's "Production runs" one-liner example lists
`variables=["sigwh", "wsh", "wsp", "was", "wad", "wsd"]` — that `"wad"` is a
leftover from before the direction-convention finding was integrated into
this document. Every other section (§3.7, §3.8, §5.3, §10) is explicit that
`wad` is unused and `wad_come_from` is the source of truth for wind. Tasks 08
and 11 call this out at the point it matters; do not copy that one example
line literally.
