# Repository Context

Fast orientation for anyone — human or agent — new to this repo. This is a
snapshot, not a source of truth: [`AGENTS.md`](AGENTS.md) (conventions),
[`docs/model_design.md`](docs/model_design.md) (design/algorithm), and
[`tasks/README.md`](tasks/README.md) (task-by-task status) are authoritative
and win if this file ever drifts from them. Update this file when a decision
below changes or a new one gets made.

## Where we are

Migrating from a v1 prototype to a v2 rewrite
(`docs/model_design.md`, titled "Design Handoff (v2)"). Per
[`tasks/README.md`](tasks/README.md):

- **Done (✅):** core I/O + direction-code decoding, directional
  transformers, ingest, climatology trend, Tucker (TensorLy), DMD (PyDMD)
  + mode-OU, local-residual OU layer, `WeatherModel`, the `copulas`-library
  build-vs-buy evaluation (verdict: no — see
  `notebooks/evaluate_copulas_replacement.ipynb`), and the archive-era
  registry (grid/cadence drift fix, see below).
- **Not started (⬜):** task 09 (sampling rewrite — `sample_batch`/
  `simulate_field` are coherent-layer-only today; local-residual noise and
  DMD conjugate-pair forcing aren't wired in yet), task 10 (hurricane-season
  validation checks), task 11 (notebooks — the training entry point
  `notebooks/test_pipeline.ipynb` is currently stale/broken: it imports
  functions renamed or deleted since task 06), task 12 (integration tests +
  import-boundary enforcement — `tests/test_import_boundaries.py` doesn't
  exist yet), task 13 (final docs + release — `CHANGELOG.md` has no
  `[0.1.0]` entry yet, only `[Unreleased]`).

A `v3_implementation_handoff.md` sits at repo root, untracked, proposing a
post-hoc physical-consistency filtering module. **Don't trust it at face
value** — see "Known open issues" below.

## Decisions and findings worth knowing before you touch code

These came out of empirical investigation, not just reading the design doc
— easy to get wrong by trusting the doc or intuition alone.

### Direction conventions are mixed across the archive

Empirically verified (`notebooks/verify_direction_convention.ipynb`):

| Field | Convention | Confidence |
|---|---|---|
| `wsd`, `wad_come_from` | `coming_from` (standard meteorological) | confirmed |
| `wad`, `ocd` | `going_to` — **not** `coming_from`, despite that being the naive default | confirmed |
| `pwd`, `pswd` | assumed `coming_from` | **not proven** — local trade-wind climatology isn't valid ground truth for swell arriving from distant storms; pending expert check |

`wad_come_from` (not `wad`) is the source of truth for wind direction.
Never decode `wad`/`ocd` as `coming_from` — that reads backwards.

### The archive has three grid/cadence "eras"

`core.archive_eras.ARCHIVE_ERAS` is the authoritative registry. Era 1
(2022-01 to 2023-10, coarse 181×360 global grid, 6-hourly), era 2 (2023-10
to 2025-10, wave grid 621×1440 @ `lat0=-75`), era 3 (2025-10-27 onward,
current/open-ended, wave grid 721×1440 @ `lat0=-90` — now a literal copy of
the wind grid's convention). `core.config`'s wave-grid constants track
**era 3** (updated 2026-07-25 — they used to be era-2-specific values
presented as universal, which crashed on any real era-3 load). Loading an
era-1/era-2 window still mislabels that window's wave-grid latitudes as a
documented, accepted tradeoff, not a bug to fix casually.

**Deliberate decision:** don't unify the three eras onto one grid (via
downsampling) or backfill era 3 to hourly cadence — both were evaluated and
deferred, not rejected. Revisit once era 3 reaches ~18 months of history,
around **2027-04-27**. Before then, era 3's grid/cadence is the system's
canonical resolution, and training windows must stay within a single era
(a cross-era window raises `ValueError` before any file I/O).

## Known open issues

- **`load_training_window` fails on real era-3 dates today.**
  `_check_wad_wad_come_from_consistency` (`src/train/ingest.py`) raises
  `wad and wad_come_from do not differ by 180°... Max deviation: 180.00°` —
  the two fields decode identically instead of as opposites, across every
  era-3 date tried so far (2026-01-01, 2026-03-15, 2025-11-15, 2026-06-01).
  Unrelated to the grid-shape fix above; not yet investigated. Needs its
  own empirical study, similar in spirit to
  `notebooks/verify_direction_convention.ipynb`.
- **`v3_implementation_handoff.md` has stale premises.** It claims
  `docs/model_design.md` §3.15/§5.9 already spec out physical-consistency
  filtering — they don't exist (§3 ends at §3.14, §5 ends at §5.8). It also
  claims "v2 is implemented and running" — five tasks (09-13, see above)
  are still not started. It requires `tests/test_import_boundaries.py` to
  pass while also forbidding new files beyond `physics.py`/`test_physics.py`
  — that test file doesn't exist (it's task 12's). Separately, `AGENTS.md`'s
  "Physical consistency (documented limitation)" section says post-hoc
  projection/filtering layers need explicit authorization and belong in the
  copula, not a downstream patch — a tension the handoff doesn't address.
  Re-verify against current state before implementing anything from it;
  don't follow it as written.

## Conventions

- Run everything through Poetry — `poetry run pytest`, `poetry run python`,
  etc. Never bare `python`/`pip`.
- `AGENTS.md` is this repo's binding conventions doc (style, layering,
  numerical safety rules, testing protocol). Read it before writing code,
  not just this file.

## Where to go next

- [`AGENTS.md`](AGENTS.md) — mandatory code style, architecture layering,
  numerical/scientific safety rules.
- [`docs/model_design.md`](docs/model_design.md) — full design, algorithm,
  and data conventions (the source of truth for "how").
- [`tasks/README.md`](tasks/README.md) — task-by-task status index; each
  `tasks/NN-*.md` file has the detail behind one row.
- `notebooks/` — empirical investigations backing the findings above;
  re-run rather than take on faith if something looks off.
