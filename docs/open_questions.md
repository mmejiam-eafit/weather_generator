# Open Questions Across the Pipeline Documentation

**Provenance.** This document consolidates every open question raised across
the eight sibling study guides in `docs/` (`variable_taxonomy.md`,
`tucker_decomposition.md`, `dynamic_mode_decomposition.md`,
`ornstein_uhlenbeck_processes.md`, `gaussian_copula_and_marginals.md`,
`harmonic_climatology.md`, `physical_consistency_constraints.md`,
`correlated_variables.md`). Each sibling's own "Open questions" section was
read in full to build this list — nothing here is invented independently of
that set, though several items are pushed further than their source
document pushed them, using reasoning available from *reading the set as a
whole* that no single document had access to on its own.

**Update (`docs/noaa_etl/`).** A later pass cross-referenced this list
against `docs/noaa_etl/` (`pipeline.md`, `noaa_ingestion.md`,
`noaa_transform.md`, `noaa_update_data.md`) — documentation of the actual
upstream ETL that produces this archive from raw NOAA GRIB2/HYCOM sources.
Where that ETL's own documentation settles a question directly (source-level
ground truth about the archive's construction, not inference from behavior),
status was updated accordingly, and one new item it surfaced was added; see
§1.1, §1.2, §1.4.

**Scope note.** This repository (`noaa-etl`) contains only the upstream ETL
(`src/task/{noaa_ingestion,noaa_transform,noaa_update_data}.py` and their
`src/util/` helpers) — none of the eight sibling study guides cited above, nor
the files they describe (`core/config.py`'s `OCEAN_GRID_SHAPE`, `core/io.py`'s
`decode_compass_code`, `notebooks/verify_direction_convention.ipynb`), exist
anywhere in this codebase (confirmed by a repo-wide search: zero hits, no
`.ipynb` files at all). They belong to a separate, downstream "modeling"
repository not checked out here. Practically: **only §1** consists of
questions this repo's source can actually adjudicate. Sections 2–8 cite
modules and documents this codebase doesn't contain, so nothing here can
advance, confirm, or correct them further — a scope boundary worth stating
up front rather than leaving implicit.

**How to read the status tags:**

- **OPEN** — genuinely unresolved by any document in this set; no answer is
  given here beyond what was already on the table, possibly with a
  practical mitigation noted.
- **ANSWERED** — this consolidation resolves the question, either by
  pointing at a sibling document that already answers it (sometimes without
  that document realizing it was answering a question raised elsewhere), or
  by deriving a new answer directly in this document.
- **PARTIALLY ANSWERED** — a real answer now exists, but it does not fully
  close the question; the residual piece is stated explicitly.

Nothing in this document authorizes a code change by itself — these are
still specification-level documents, not implementations.

---

## 1. Data and convention-level questions

### 1.1 Is the ocean grid's row-0 latitude/orientation south-first, like the wave and wind grids? — **ANSWERED** (yes, confirmed by direct geo-referenced lookup)

Raised in: `variable_taxonomy.md` §6.

`core/config.py` deliberately leaves the ocean grid's row-0 latitude unset.
Wave and wind grids were confirmed row-0-south via a coastline/land-mask
correlation study; the ocean grid has not had the equivalent study run
against real data. `notebooks/verify_direction_convention.ipynb` uses a
cheaper proxy (zonal-mean `sst` maximum for the equator row, open-ocean
fraction at 40–60° for hemisphere sense) sufficient only to support a few
direction-diagnostic latitude bands, not a full verification.

**Update from `docs/noaa_etl/noaa_transform.md`:** the upstream ETL that
produces this archive's HYCOM-sourced fields (`sst`, `ocs`, `ocd`) confirms
this can't be settled from that codebase either — the HYCOM→uniform-grid
remap (`hycom_index`, backed by `hycom_satocean_idx.npy`) deliberately does
**not** go through `flip_latitude` the way every GFS-sourced field does, and
the ETL's own documentation states plainly that "this repository only
contains the index lookup table, not the geographic (lat/lon-in-degrees)
definition of the output grid it targets... confirm the authoritative grid
definition [with the consuming service] before assuming exact bounds." That
does confirm `OCEAN_GRID_SHAPE`'s value — `hycom_satocean_idx.npy` is
exactly `(1121, 2880, 2)` — but not its orientation.

This changes the *recommended path* to resolution, even though it doesn't
close the question: rather than (or in addition to) an independent
coastline-correlation study, the more direct fix is obtaining the target
grid's geographic definition from whoever generated
`hycom_satocean_idx.npy` — that party necessarily knows the real-world
lat/lon each output cell corresponds to, which no amount of statistical
inference from this archive alone can fully recover. **Remains open**, and
no `sst`/`ocs`/`ocd` field should be treated as geo-referenced until one of
these two paths is completed.

**Additional corroborating evidence (from loading `hycom_satocean_idx.npy`
directly, not just reading about it):** shape `(1121, 2880, 2)`, dtype
`int16`, confirmed. One correction to the bounds cited above: `lon_idx`
actually ranges `0–4498`, not `0–4499`. New evidence not previously noted:
output row 0's `lat_idx` is nearly constant (~425) across every sampled
longitude — consistent with sitting near a true, non-displaced pole, where
all meridians converge to one point — while the opposite edge (row −1) has
widely scattered `lat_idx` across longitude (~2836–3257 in a coarse sample),
consistent with the warped, displaced-pole region of a tripolar mesh. This is
internally consistent with the "one true pole at the South Pole, two
displaced poles in the North" narrative above, but it remains internal
grid-structure evidence only — it still doesn't anchor either edge to a real
lat/lon, so it corroborates without closing the question.

**Provenance, checked directly against `noaa-etl`'s history (not just its
current-state docs).** `git log --follow -p` on
`src/util/geo/hycom_satocean_idx.npy` across the entire repo history returns
exactly one commit — `14d5c19` ("feature process rtofs", author
`s-gonzalez <sebastian@bearing.ai>`, 2025-11-19) — which adds the file as an
opaque binary, no accompanying generation script, README, or comment then or
since. A repo-wide grep for `satocean`/`sat_ocean` turns up only the
filename itself, nowhere else. This confirms, by direct inspection rather
than trusting `noaa_transform.md`'s claim secondhand, that the grid's
geographic definition genuinely is not recoverable from this codebase, its
history, or any comment in it. **Actionable lead this adds beyond what the
docs alone gave:** the commit author is a real, identifiable person at the
same org — asking them directly (or locating whatever ticket/PR that commit
closed) is a far more direct path to closing this than another statistical
correlation study.

**Dimension cross-check against actual usage, not just the file's own
shape.** `hycom_index()` (`src/util/geo/grid_utils.py`) indexes a native
HYCOM grid via `grid_arr[lat_idx, lon_idx]`, where that native grid's shape
(`lat_size`, `lon_size`) is parsed **dynamically from each `.b` metadata
file's header at runtime** (`read_hycom_fields` in
`src/util/hycom/hycom_transform.py`: `lon_size = int(b_file_lines[7]...)`,
`lat_size = int(b_file_lines[8]...)`) — never hardcoded, and never asserted
against `HYCOM_IDX`'s bounds anywhere in the code. So `hycom_satocean_idx.npy`
implicitly requires every incoming native HYCOM grid to satisfy
`lat_size ≥ 3298` and `lon_size ≥ 4499` (one past each index's observed max),
with **no explicit compatibility check anywhere** — if RTOFS's native grid
resolution ever changed, this would raise `IndexError` rather than silently
misalign, which is a real if minor mitigating factor, but the dependency
itself is unvalidated in code. No test fixtures or sample `.b` files exist in
`noaa-etl` to check this against directly, so it can't be fully verified
without a live RTOFS file. As a plausibility check only (external knowledge,
not verified against this repo): `3298 × 4499` is close to RTOFS/HYCOM's
publicly documented global 1/12° operational grid dimensions (commonly cited
as `3298 × 4500`) — consistent with this genuinely targeting the real
operational grid rather than a stale or mismatched artifact, reassuring for
the pipeline's general health, but it still said nothing about orientation
**until the direct check below.**

**Resolved by direct geo-referenced lookup (2026-08-11).** RTOFS publishes
NetCDF diagnostic output (distinct from the `.a`/`.b` binaries `noaa-etl`
parses) to a public, unauthenticated AWS bucket
(`s3://noaa-nws-rtofs-pds/rtofs.{date}/rtofs_glo_2ds_f{fff}_diag.nc`). One
such file (`rtofs.20240127/rtofs_glo_2ds_f000_diag.nc`) was downloaded and
opened directly. It carries `Conventions: CF-1.0` and, critically, explicit
CF-convention 2-D coordinate arrays — `Latitude(Y, X)` and `Longitude(Y, X)`,
both shape `(3298, 4500)`, exactly matching `hycom_satocean_idx.npy`'s index
bounds (`lat_idx` max `3297`, `lon_idx` max `4498`). This means the file
gives the *exact* real-world latitude/longitude of every native HYCOM grid
cell the index table can point to — no statistical inference needed.

Doing the actual remap — `real_lat = Latitude[lat_idx, lon_idx]`,
`real_lon = Longitude[lat_idx, lon_idx] % 360` over the full
`(1121, 2880)` index table — gives a direct, exact geo-reference for every
cell of the archive's `sst`/`ocs`/`ocd` output grid:

- **Latitude / row axis:** row 0 (excluding one `(0,0)` sentinel/fill cell
  per row — 113 such cells total out of 3,228,480, evidently a land/no-data
  placeholder, not a real mapping) is ≈ constant at **−65.00°** across every
  longitude; row 1 ≈ −64.90°; row 100 ≈ −52.51°; row 300 ≈ −27.52°; row 560
  ≈ +4.95°; row 800 ≈ +34.96° — **strictly increasing with row index, and
  effectively zonal (constant across longitude) for every row south of the
  tripolar patch**, exactly as expected for the Mercator-projected part of
  the grid. Row spacing is non-uniform in degrees (≈0.10°/row near −65°,
  ≈0.125°/row near −50…−27°) — the correct signature of a Mercator grid
  (uniform in projected `y`, not in latitude degrees), not an artifact.
  North of ≈47°N (row ≈1000+), rows stop being purely zonal (row 1000 spans
  52.8°–64.0°, row 1120 spans 50.3°–76.6°) — exactly the displaced-pole
  tripolar-patch warping the ETL docs already said exists north of 47°N, and
  consistent with (though not itself a repeat of) the earlier `lat_idx`
  clustering evidence for that region. **Row index increases with latitude
  (south → north)** — the same "row-0-south" convention already confirmed
  for the wave and wind grids, achieved here not via `flip_latitude` (which
  `noaa_transform.md` confirms is *not* applied to HYCOM fields) but because
  `hycom_satocean_idx.npy` itself was evidently built with that row ordering
  baked in.
- **Longitude / column axis:** sampled at row 560 (a purely zonal,
  tripolar-patch-free row), longitude runs from `0.00°` at column 0 up to
  `359.84°` at column 2879, monotonically increasing with exactly one
  wrap-around discontinuity — **column index increases eastward starting
  from 0°E**, the same convention as the GFS-sourced grids
  (`lon = j * 0.25`).

This closes the question: the ocean grid's orientation is **row-0-south,
column-0-at-0°E-increasing-eastward**, matching the wave/wind grids exactly.
`core/config.py`'s `grid_coordinates` for the ocean grid can now be filled
in with confidence — for the Mercator-projected majority of the grid
(rows south of ≈47°N) coordinates are a simple, if non-uniform, per-row
latitude lookup (not a fixed-step formula like the GFS grids); north of
47°N, per-cell `(lat, lon)` should be taken directly from a stored copy of
this remap (or the remap re-derived from a fresh `Latitude`/`Longitude`
pair) rather than approximated, since the tripolar patch is not zonal there.
The commit-author lead and the coastline-correlation study are no longer
necessary — this direct lookup supersedes both as the authoritative answer.

**Implementation gap (2026-08-16): this answer was never wired into code.**
`src/core/config.py`'s `grid_coordinates("ocean")` still raises
`ValueError` unconditionally, so `sst`/`ocs`/`ocd` cannot be loaded through
`load_training_window` at all — confirmed directly, not assumed. The
*question* this section resolves (what the real coordinates are) is
answered; *using* that answer in loadable code is tracked separately as
`tasks/16-ocean-grid-coordinates.md`.

### 1.2 Is `pwd`/`pswd`'s "coming_from" convention correct? — **ANSWERED**

Raised in: `variable_taxonomy.md` §6, `correlated_variables.md` §3.

Previously open because only `wad`/`wad_come_from` (wind) and `wsd`
(wind-sea) had been checked against known persistent wind/current patterns
(trade winds, Southern Ocean westerlies, major equatorial currents) — a
check that works because those fields track *local*, climatologically-persistent
forcing, but is invalid for `pwd`/`pswd` since they can be swell arriving
from a distant, decoupled storm system (confirmed inconclusive in
`notebooks/verify_direction_convention.ipynb`'s Tests 5–6, which got mixed
PASS/FAIL results against the same trade-wind boxes and correctly diagnosed
why that result wasn't diagnostic either way).

**Resolved by `docs/noaa_etl/noaa_transform.md`'s "Direction field encoding"
section**, which doesn't have this confound because it isn't an empirical
behavioral test — it reads NOAA's own field definitions directly. It cites
[NCEP GRIB2 Code Table 4.2-0-2](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-2.shtml)
(`WDIR`: "Wind Direction (from which blowing)") and NDBC's measurement
glossary (`MWD`/`SwD`/`WWD`, the buoy analogs of `dirpw`/`swdir`/`wvdir`,
defined as "the direction from which the waves... are coming"), and its
transform table confirms `pwd` (from `dirpw`) and `pswd` (from `swdir`)
receive `calculate_16degree` only — no conversion — exactly like `wsd`. They
inherit the source field's documented "coming from" semantics unmodified, by
construction, independent of whether the wave field happens to be locally
wind-generated or swell from afar. This is stronger evidence than a
climatology check could ever provide for these two fields specifically.

As a side effect, this also closes the narrower gap of whether `pwd`/`pswd`
truly share `wsd`'s origin-code family (previously assumed by analogy, never
directly verified): checking `src/core/io.py`'s actual `decode_compass_code`
formula, `(code - origin_code) * 22.5 % 360`, against the ETL's documented
sector table (sectors 1–16, sector 16 = North via wraparound) confirms
`origin_code=0` decodes that scheme correctly. **No further empirical study
or domain-expert check is required for this question.**

**Direct source-level reconfirmation:** `src/util/gfs/gfs_transform.py`'s
`VARIABLE_TRANSFORMATION` list (not just `noaa_transform.md`'s paraphrase of
it) shows `dirpw`→`pwd` and `swdir`→`pswd` each get exactly `[flip_latitude,
calculate_16degree]` — a byte-identical transform chain to `wvdir`→`wsd`.
Executing `calculate_16degree` (`src/util/geo/coords_utils.py`) directly
against sample angles also confirms its sector semantics match
`noaa_transform.md`'s table exactly (sector 16 = N at 0°/360°, sector 4 = E
at 90°, sector 8 = S at 180°, sector 12 = W at 270°). Separately: `core/io.py`
and any `decode_compass_code`/`origin_code` do **not** exist anywhere in this
repository — that citation above refers to the downstream consuming
service's decoder, not to this ETL's own code, and should be read as such.

### 1.3 Is `wad` usable as a wind-direction field? — **ANSWERED** (already resolved; restated here for completeness)

Raised in: `variable_taxonomy.md` §6.

No — `wad` is confirmed to be a decoy: it decodes as a plausible
`coming_from` field but is actually `going_to`, exactly 180° from
`wad_come_from` at every checked cell. `wad_come_from` is the only correct
wind-direction source and should be the only one ever wired into any
pipeline stage. This was already a closed, confirmed finding when it was
written into `variable_taxonomy.md` — it is listed here only so this
document is a complete index, not because it is actually in question.

**Direct source-level reconfirmation of the mechanism:** `wad_come_from` and
`wad` both read GRIB `shortName: "wdir"` in `gfs_transform.py`'s
`VARIABLE_TRANSFORMATION`, but `wad_come_from`'s chain is `[flip_latitude,
calculate_16degree]` while `wad`'s is `[flip_latitude, vectorize_16degree]`.
`vectorize_16degree` (`src/util/geo/coords_utils.py`) internally calls
`transform_directions_to_coming_from_vectorized` (a ±180° flip) before
bucketing; `wad_come_from`'s path never applies that flip. Since raw `wdir`
is already "coming from" per NCEP (§1.2), the extra flip applied only to
`wad` necessarily produces the "going to" (180°-opposite) value — a direct
code-level explanation for *why* the decoy relationship holds, not just a
restated confirmation that it does.

### 1.4 Is `pwh` an independent wave-height measurement, distinct from `sigwh`? — **ANSWERED** (no — and this needs correcting elsewhere)

Raised in: not previously identified as an open question anywhere in this
set; surfaced by reading `docs/noaa_etl/noaa_transform.md`.

No. The GFS variable-map table states `pwh` is derived from the same source
GRIB field as `sigwh` (`swh`) through the identical transform chain, and
labels it explicitly: "Significant/primary wave height (**duplicate of
`sigwh`**, kept for naming compatibility)." `pwh` and `sigwh` are the same
array. This is not true of the rest of the "peak wave" triplet — `pwd`
(from `dirpw`) and `pwp` (from `perpw`) are genuinely independent fields —
so the issue is specific to the height component only.

**This directly undermines an assumption in two sibling documents, and both
need a corrective pass, not just a note here:**
- `variable_taxonomy.md`'s Class C treats `pwh`/`pwd`/`pwp` as one coherent
  independent triplet the same way `wsh`/`wsd`/`wsp` is. That's incorrect
  for `pwh` specifically — it carries no information beyond what `sigwh`
  already provides.
- `correlated_variables.md` §3.3 extends its Miche-breaking-limit
  steepness-coupling example to "`wsh`/`wsp`, or equally `pwh`/`pwp`" — that
  extension is misleading. `wsh`/`wsp` are two independent readouts of one
  partition's spectrum, as claimed. `pwh`/`pwp` are not: `pwh` is just
  `sigwh` restated, so the "coupling" there is really `sigwh`/`pwp`, a
  different (and weaker-grounded, since `sigwh` is a composite across
  partitions rather than one partition's own spectrum) physical claim than
  the one actually being made.

**Status: fully closed, including the corrective edits.** The fact was
already settled when this entry was first written; the corrective passes
to `variable_taxonomy.md`, `correlated_variables.md`, and (identified
afterward — it has the same latent assumption and was missed in the
original sweep) `physical_consistency_constraints.md` have now all been
made.

**Direct source-level reconfirmation** (previously only confirmed via
`noaa_transform.md`'s table, one level removed from the pipeline itself):
`gfs_transform.py`'s `VARIABLE_TRANSFORMATION` contains two separate dict
entries both with `short_name: "swh"` and an identical transformation list
`[flip_latitude, round_2]`, differing only in `variable_store_name`
(`"sigwh"` vs `"pwh"`) and matching `store_type` (`np.float16`). This is the
same deterministic operation sequence applied to the same input, so `pwh`
and `sigwh` aren't merely equivalent — they're the same array.

**Independent confirmation directly from NOAA's own documentation**
(checked separately from `noaa-etl`'s code, at the user's request, since
code-level agreement alone doesn't rule out the ETL author having
independently made the same naming choice for unrelated reasons). NCEP's
official GRIB2 parameter table,
[Table 4.2-10-0 ("Oceanographic — Waves")](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-10-0.shtml),
was fetched and its raw HTML grepped directly (not summarized secondhand).
It defines `DIRPW` ("Primary Wave Direction") and `PERPW` ("Primary Wave
Mean Period") as real, independent parameters, but across the table's
entire ~80-parameter wave category — from spectral moments through
freak-wave statistics — **there is no "primary wave height" parameter at
all**. Every other height in this archive traces to a real, distinct NOAA
field (`HTSGW`→`sigwh`, `WVHGT`→wind-sea height, numbered swell-partition
heights→`pswh`); there was structurally never a `PWHGT`-equivalent for
`pwh` to be. This is stronger than the code-level finding alone: it shows
`pwh`'s duplication of `sigwh` isn't an ETL implementation quirk that
happened to reuse `sigwh`, but the only coherent choice available given
that NOAA never defined an independent field for it to measure.

**Corrective edits made (2026-08-11):**
- `variable_taxonomy.md` — Class C summary table, §2.2, §3.3 (main
  correction, with both lines of evidence), the §5 results table, and the
  §6 open-questions bullet on `sigwh`'s RMS relationship all updated to
  exclude `pwh` from any independent-partition treatment.
- `correlated_variables.md` §3.3 — the "`wsh`/`wsp`, or equally `pwh`/`pwp`"
  claim corrected: the physically meaningful analog is `sigwh`/`pwp`, and
  it is explicitly weaker-grounded than the `wsh`/`wsp` case, not equally
  valid.
- `physical_consistency_constraints.md` §2.1 (and its downstream §2.4/§5
  tuples and formulas) — the wave-energy-budget constraint corrected from
  the (now known to be structurally broken — it double-counts `sigwh`
  against itself) three-term $\text{sigwh}^2\approx\text{wsh}^2+\text{pwh}^2+\text{pswh}^2$
  to the two-term $\text{sigwh}^2\approx\text{wsh}^2+\text{pswh}^2$, with
  the corresponding loss of a third independent check made explicit rather
  than silently absorbed into the tolerance band.

### 1.5 Are this archive's `*p` fields (`wsp`, `pwp`, `pswp`) peak periods ($T_p=1/f_p$) or mean periods ($T_{m01}=m_0/m_1$)? — **ANSWERED** (mean period — corrects an error in this same document series)

Raised in: a direct challenge to `variable_taxonomy.md` §2.2's original
claim, sourced from a separate design thread that described NOAA's period
fields as "ratios of different momenta... not a period in the sinusoidal
sense" — in tension with this series' own prior claim that they were peak
periods.

**Mean period, $T_{m01}=m_0/m_1$ — the other thread was right, and
`variable_taxonomy.md` §2.2 was wrong.** Checked directly against source
and NOAA documentation, not assumed from either side's framing.
`noaa-etl`'s `gfs_transform.py` derives `wsp`/`pwp`/`pswp` from GRIB
`shortName`s `mpww`/`perpw`/`mpts`. Cross-checked against NCEP's own
[GRIB2–GRIB1 parameter conversion table](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/GRIB2_parmeter_conversion_table.html),
these are explicitly named **"Mean period of wind waves"** (10-0-6),
**"Primary wave mean period"** (10-0-11), and **"Mean period of total
swell"** — distinct, separately-numbered GRIB2 parameters from NOAA's own
"Peak Wave Period"/"Peak Period of Wind Waves"/"Peak Period of Total
Swell" family, which this archive's ETL never reads at all. Confirmed
against [WAVEWATCH III's user manual](https://polar.ncep.noaa.gov/waves/wavewatch/manual.v5.16.pdf)
and standard oceanographic convention, the quantity reported is
$T_{m01}=m_0/m_1$ — an energy-weighted mean frequency across the whole
partition spectrum, inverted — not $T_p=1/f_p$, the reciprocal of the
single largest spectral bin. The two coincide only for an infinitely
narrow spectral peak.

**Corrective edits made (2026-08-11):**
- `variable_taxonomy.md` §2.2 — corrected to describe $T_{m01}$, with the
  full evidence trail; all downstream uses of the `$T_p$` symbol in that
  document renamed to `$T_{m01}$`.
- `physical_consistency_constraints.md` §2.3 — retitled "Mean-period
  plausibility" (was "Peak-period plausibility"); added a correction
  explaining that the section's plausibility-*bound* use is tolerant of
  either statistic (both occupy the same broad physical regime), so the
  floor/ceiling values survive essentially unchanged, but the framing was
  corrected throughout.
- `correlated_variables.md` §3.3 — this is where the distinction is
  **load-bearing, not just a caveat**: the Miche-breaking-limit derivation
  substitutes a period directly into the dispersion relation to get a
  specific numeric wavelength, and $T_{m01}$ is not literally the period
  of the single wave component that relation is about. Corrected to flag
  that using `wsp` ($T_{m01}$) unmodified systematically underestimates
  the bound (conservative/over-rejecting, not unsafe), and that a
  shape-dependent correction factor $T_p\approx\alpha\,T_{m01}$ (commonly
  cited in the rough range $\alpha\approx1.05\text{–}1.4$ for JONSWAP-type
  wind-sea spectra, not independently verified against this archive) is
  needed before the `0.14` Miche constant is applied — an empirical
  calibration task, not resolved here.

**Follow-on (2026-08-11):** confirming $T_{m01}$ as the archive's actual
period statistic directly motivated formalizing a `log`-transform
convention for it — $T := \log T_{m01}$, alongside a matching $\log E :=
\log(H^2)$ for wave heights — now specified in `variable_taxonomy.md`
§3.4. See §2.3 below for the tensor-representation consequence
(resolves what had been an open `hypot`-reconstruction question) and
§3.4/§3.5 for the further consequence for cross-channel coupling
diagnostics.

---

## 2. Representation-level questions (taxonomy, Tucker, Gaussianization)

### 2.1 Does `sigwh² ≈ wsh² + pswh²` require disjoint partitions, and is that assumption safe? — **ANSWERED**

Raised in: `variable_taxonomy.md` §6 ("`sigwh`'s RMS relationship to
partition heights assumes disjoint partitions").

Yes, the identity is exact only if the wind-sea/peak/swell spectral
partitioning is non-overlapping, and to the extent it isn't, the identity
is approximate. This is not actually a loose end — it is precisely the
situation `physical_consistency_constraints.md` §2.2 already characterizes
correctly: that document derives the energy-budget relation as an
approximate, tolerance-banded check (`|sigwh² − wsh² − pswh²| < atol`)
*because* real partitions have some spectral overlap and because
higher-order swell contributions are neglected — it never treats the
relation as an exact identity to begin with. **The two documents' concerns
are the same fact, viewed from two sides**: the taxonomy document flags the
assumption from the decomposition side, and the physical-consistency
document already builds the correct tolerance-based treatment for it from
the constraint side. No further resolution is needed; treat
`physical_consistency_constraints.md` §2.2 as this question's answer.

### 2.2 Does independent per-variable Gaussianization discard the physical coupling within a height/period/direction triplet? — **PARTIALLY ANSWERED**

Raised in: `variable_taxonomy.md` §6 ("flagged here because it originates in
the taxonomy, not the copula machinery... a concern for the Gaussian-copula
document to resolve").

`gaussian_copula_and_marginals.md` does not, in fact, resolve this — its own
open-questions section (§6) is about *where* Gaussianization sits in the
per-variable chain, not about *restoring cross-member coupling within one
triplet*. Those are different questions. The actual answer to *this*
question comes from `correlated_variables.md`, whose whole §3–§5 is a
specification for exactly this: grouping a triplet's members (e.g. `wsh`,
`wsp`, and the `wsd`-derived Cartesian pair) into one joint DMD state built
from their already-independently-Tucker-compressed coefficients, so that
the coupling discarded by independent per-variable, per-cell treatment
upstream (Gaussianization *and* Tucker *and* DMD, all per-variable) is
reintroduced at the one stage designed to model temporal/cross-member
dynamics jointly.

This is a real answer, not a deflection — but it is not a full closure:
`correlated_variables.md` §6 itself notes that **which** variables belong in
a joint group is an open modeling choice (see §3.2 below), so the
mechanism for recovering the coupling is now specified, while its exact
scope (how many groups, how large) is not.

### 2.3 Does low-rank Tucker reconstruction preserve `speed = hypot(u, v)` for a directional field? — **ANSWERED** (resolved via a dedicated `log E` tensor)

Raised in: `variable_taxonomy.md` §6 ("Low-rank approximation does not
commute with `hypot`"), explicitly deferred to `tucker_decomposition.md`.

`tucker_decomposition.md` did not originally address this (confirmed by
reading its full "Open questions and limitations" section at the time).
**Resolved (2026-08-11):** the "concrete mitigation" this entry previously
named as unadopted — decompose height as its own separate scalar tensor,
in addition to `u,v` — has now been adopted, with a specific, justified
form. `variable_taxonomy.md` §3.4 specifies $\log E := \log(H^2)$ (not raw
$H$, for the dynamic-range/power-law-growth reasons that section derives)
as a fourth, independently Tucker-fit tensor per Class C triplet (and
alongside `sigwh`), and designates $H=\exp(\log E/2)$ — not
`hypot(u_dir,v_dir)` — as the authoritative magnitude-reconstruction path.
`physical_consistency_constraints.md` §4 now requires this explicitly for
its own inputs. The redundancy this entry flagged as a cost is accepted
deliberately: `u_dir,v_dir` are kept magnitude-scaled (not switched to
unit vectors) for the "shrinks to origin when energy is negligible"
property `variable_taxonomy.md` §3.3 already valued, so magnitude is now
represented twice by design, with $\log E$'s reconstruction the one that
counts.

### 2.4 Does per-cell independent Gaussianization of $u$ and $v$ preserve `speed = hypot(u, v)`? — **PARTIALLY MOOT** (magnitude case resolved by 2.3; angle case remains open)

Raised in: `gaussian_copula_and_marginals.md` §3 ("Known open subtlety").

This was flagged as a distinct failure mode from §2.3, occurring earlier
in the chain: $u$ and $v$ are each Gaussianized via their own *separate*
per-cell empirical CDF, so $speed=\operatorname{hypot}(u,v)$ does not
survive into Gaussianized $(Z_u,Z_v)$-space either, compounding with §2.3's
Tucker-level distortion. **§2.3's resolution changes what's actually at
stake here.** Since $H$ is now read from $\log E$'s own reconstruction and
`hypot(u_dir,v_dir)` is no longer the authoritative magnitude source at
all, whether Gaussianization distorts `hypot(u,v)` specifically is
**largely moot for magnitude** — nothing downstream is supposed to trust
that value for magnitude anymore. What is **not** resolved, and remains a
genuinely open question: whether independent per-cell Gaussianization of
$u$ and $v$ distorts the **direction** $\operatorname{atan2}(v,u)$, which
is still read from `u_dir,v_dir` (never from $\log E$) and has no analogous
dedicated-tensor fix. This is a narrower, restated version of the original
concern — angle preservation only, not magnitude — and it inherits §2.3's
old status: open, unresolved by any sibling document.

---

## 3. Cross-variable coupling

### 3.1 What mechanism captures cross-variable dynamical coupling, given that Tucker and DMD are now strictly per-variable? — **ANSWERED**

Raised in: `tucker_decomposition.md` §4.4/§6, `dynamic_mode_decomposition.md`
§6 (both explicitly declined to propose a mechanism and flagged this as
open).

`correlated_variables.md` answers this directly: identify physically
justified correlated-variable groups (e.g. `{was, wsh, wsp}`,
`{wad_come_from, wsd}`), and for each group, concatenate the members'
*already independently Tucker-compressed* per-variable coefficient
sequences into one joint state before running DMD's snapshot-pair
regression on that group — never touching or re-raveling the original
per-variable spatial `(lat, lon, time)` tensors, so this does not
reintroduce the four-dimensional-stacking mistake the tensor-representation
rule forbids. The physical grounding is the shared wave-action balance
equation (`correlated_variables.md` §2): wind and wind-sea variables are
coupled through a real, shared governing PDE, not merely correlated by
coincidence, which is exactly the kind of structure a joint (rather than
independent) dynamical fit is suited to capture — a jointly-fit linear
operator can represent genuine lead-lag structure (e.g. a rising `was` mode
predicting future `wsh` growth) that two independent per-variable DMD fits
cannot represent at all, since each only sees its own autocorrelation.

### 3.2 Which variables belong in a joint group, and how many groups are needed? — **OPEN**

Raised in: `correlated_variables.md` §6, explicitly described there as
"mirrors, and narrows without closing" the question in §3.1.

No general algorithm for group membership is given anywhere. The tradeoff
is real: growing a group (e.g. adding the peak-wave triplet to capture
$S_{nl}$'s energy transfer beyond the wind-sea partition) increases the
joint state's dimension and hence the DMD rank and data requirements,
risking an underdetermined or overfit joint operator for a fixed training
window; keeping groups small and conservative avoids that risk but
reproduces the independent-fit blind spot for whatever coupling falls
outside the chosen group. **Remains open** — a modeling/hyperparameter
choice to be made empirically (e.g. by checking, per §5 of
`correlated_variables.md`, whether a candidate group's jointly-fit operator
actually shows non-negligible off-diagonal coupling terms; if it doesn't,
the group was drawn too widely for no benefit).

**Update (2026-11-21): tooling for that check now exists.**
`correlated_variables.md` §4.4 recasts the joint-group regression as a
Linear Inverse Model and specifies the measurements:

- a tau test of the linear-Markov assumption;
- correlated within-group noise;
- information lost by independent fits, measured in nats: Granger/transfer
  entropy into each member, and the mutual information of the joint
  innovations.

These are implemented in `src/train/linear_inverse.py`. A candidate group
whose coupling information is indistinguishable from a bootstrap baseline
gains nothing from a joint fit. The question itself stays open until
per-variable cores exist for a candidate group (only `wsh` has one so far)./wind-sea-direction alignment equilibrium a stable node, as the scalar energy case is? — **ANSWERED HERE**

Raised in: `correlated_variables.md` §3.2 and §6 ("§3.2's directional-alignment
ODE is illustrative, not fully stability-analyzed... left open").

This can be answered directly from the vector ODE `correlated_variables.md`
§3.2 already writes down:

$$
\dot{\mathbf D} = \kappa\, U\,|\mathbf D|\, \hat{\mathbf U}_{\text{vec}} - c_{ds}\,|\mathbf D|^{1/2}\,\mathbf D,
\qquad \hat{\mathbf U}_{\text{vec}} = \mathbf U_{\text{vec}}/U .
$$

Write $\mathbf D = r(\cos\theta,\sin\theta)$ in polar coordinates relative
to the (quasi-static) wind bearing, so $\theta$ is the angular offset
between wind-sea direction and wind direction and $r = |\mathbf D| = H_s$.
Projecting $\dot{\mathbf D}$ onto the radial and tangential unit vectors
gives, after simplification,

$$
\dot r = \kappa U r\cos\theta - c_{ds}r^{3/2}, \qquad
\dot\theta = -\kappa U \sin\theta .
$$

The angular equation decouples completely from $r$ — it is exactly the
classical overdamped-pendulum / Adler phase equation. Its fixed points are
$\theta=0$ (aligned) and $\theta=\pi$ (anti-aligned); linearizing,
$\dot\theta\approx-\kappa U\,\theta$ near $\theta=0$ (stable, decay rate
$\kappa U$) and $\dot\epsilon\approx+\kappa U\,\epsilon$ near $\theta=\pi+\epsilon$
(unstable). So $\theta=0$ attracts **every** initial condition except the
single unstable equilibrium at exactly $\theta=\pi$ — wind-sea direction
aligns with wind direction from any starting offset, matching the physical
picture $S_{in}$'s directional peaking already motivates.

At $\theta=0$, the radial equation reduces to exactly the scalar energy
ODE of §2.3–§2.4, with fixed point $r^\star = (\kappa U/c_{ds})^2$ and
linearized rate $-\tfrac12 \kappa U$ there (same computation as the scalar
case). Because $\partial\dot\theta/\partial r \equiv 0$ and
$\partial\dot r/\partial\theta|_{\theta=0}=-\kappa U r\sin\theta|_{\theta=0}=0$,
the full 2D Jacobian at $(\theta,r)=(0,r^\star)$ is **diagonal**, with real
eigenvalues $-\kappa U$ (angular) and $-\tfrac12\kappa U$ (radial). Both
negative and real: **the aligned equilibrium is a stable node**, exactly as
the scalar case is — and angular relaxation is exactly twice as fast as
radial relaxation ($|\lambda_\theta| = 2\,|\lambda_r|$), a clean, falsifiable
quantitative prediction of this reduction worth checking against real
fetch-limited growth data if this proposal is ever implemented.

This inherits the same caveat the scalar analysis already carries (§2.4 of
`correlated_variables.md`): it assumes $U$ (and hence the wind bearing) is
quasi-static relative to the alignment relaxation timescale, consistent
with how that document already treats $U$'s own dynamics elsewhere.

### 3.4 Does within-group channel coupling (one partition's `u, v, E, T`) need a joint multi-channel Tucker, or is it recoverable downstream of §4's independent per-variable fits? — **OPEN**

Raised in: a competing design document (`physical_handoff.md`) proposed
stacking one wave partition's derived channels `u, v, E, T` into a single
4-mode `(lat, lon, t, C)` tensor and fitting one joint Tucker across it,
so the channel factor `A_C` could discover cross-channel structure (e.g. a
steepness-like direction) directly — a real alternative to
`tucker_decomposition.md` §4.1's strict per-variable rule and
`correlated_variables.md` §4's downstream-joint-model proposal, not
something dismissable by citing that rule alone.

There is a genuine, structural asymmetry between the two designs: each
independent per-variable Tucker fit truncates to its own top modes based
on that one channel's own marginal variance, *before* any cross-channel
structure is considered — so a coupling that is weak in each channel's
own marginal variance but strong jointly (plausible for a steepness
signature riding on otherwise differently-driven `E`/`T` variability)
would be discarded before any downstream joint model ever sees it, a
known blind spot of "reduce independently, then correlate" pipelines
relative to genuinely joint methods (CCA/PLS/joint-tensor-factorization).
The joint design pays for this with a real, untested cost of its own: it
forces `u, v, E, T` to share spatial/temporal factor matrices, assuming
their dominant EOFs coincide. Separately, the physical relationship
motivating this (the Miche steepness bound) is a pointwise inequality
that no linear decomposition — either architecture — can *guarantee*
holds at every point; that enforcement job belongs to
`physical_consistency_constraints.md`'s rejection layer regardless of how
this question resolves, so what's actually at stake is representation
fidelity (and possibly downstream rejection-rate reduction), not a
substitute for that layer.

**Remains open** — full derivation, the asymmetry argument, and a
concrete, bounded experiment design (one partition — `wsh`/`wsd`/`wsp` —
joint 4-channel Tucker vs. independent per-channel Tucker + CCA/joint-DMD
on the coefficients) are in `correlated_variables.md` §6. **Update:** the
experiment now checks channels `{u,v,E,F}`, not `{u,v,E,T}`, and the
`E − 2T ≈ const` diagnostic originally proposed is no longer used as-is —
see §3.5 below for why. This mirrors §4.1 below in requiring empirical
validation before either design is committed to, not a decision from
reasoning alone.

### 3.5 Is the `E − 2T ≈ const` diagnostic proposed in §3.4 actually clean evidence of Miche/dispersion coupling, or a shared-moment artifact? — **ANSWERED** (artifact risk confirmed; fix proposed, itself unverified)

Raised in: a direct algebraic check of §3.4's own proposed channels,
prompted by the observation that `T = log(Tm01)` and `E = 2log(Hs)` both
reduce to expressions in the same spectral moments.

**Confirmed: `E` and `T` are not two independently-constructed channels —
they are a linear reparametrization of the same two spectral moments,
sharing one term with matching sign.** Using `variable_taxonomy.md`'s own
definitions and moment convention (§2.2, ordinary frequency $f$):

$$E = 2\log H_s = \log(16\,m_0) = \log16 + \log m_0, \qquad T = \log T_{m01} = \log(m_0/m_1) = \log m_0 - \log m_1.$$

Both carry $\log m_0$ with coefficient $+1$, so $T = (E-\log16) - \log
m_1$ and $\operatorname{Cov}(E,T) = \operatorname{Var}(\log m_0) -
\operatorname{Cov}(\log m_0,\log m_1)$ — structurally present regardless
of any real physical steepness coupling, purely because both channels
share an overlapping moment. The specific diagnostic §3.4 proposed
checking, $E - 2T = -\log m_0 + \log16 + 2\log m_1$, does **not** cancel
this shared term (the $\log m_0$ coefficient flips sign but survives). So
a joint Tucker's `A_C` — or any downstream correlation check — finding a
strong `E`/`T`-coupled direction is not, by itself, evidence it found the
Miche/dispersion relationship specifically; it could equally be recovering
this algebraic artifact. This is answered as a mathematical fact; what
remains open is only the fix below.

**Proposed fix, itself open:** replace $\{E,T\}$ with $\{E,F\}$, $F := E
- T = \log16+\log m_1$ — a basis where $E$ depends only on $m_0$ and $F$
only on $m_1$, with no shared additive term, so any observed `E`/`F`
correlation is attributable to genuine co-variation in the real data
rather than guaranteed by the basis choice. **Not itself validated**
against real archive spectra, and not established as the better channel
choice beyond this one diagnostic. §3.4's experiment has been updated to
run on `{u,v,E,F}` with a shuffled/synthetic null-baseline control (to
measure how much apparent coupling survives even with no real physics
present), rather than `{u,v,E,T}` with no baseline — see
`correlated_variables.md` §6 for the full experiment.

---

## 4. Gaussianization placement

### 4.1 Where, along each variable's independent processing chain, should Gaussianization happen? — **DECIDED** (placement ii, Tucker on log-energy anomalies then Gaussianize the core, before DMD — supersedes the 2026-08-19 placement-i decision; see 2026-09-13 update)

Raised in: `gaussian_copula_and_marginals.md` §6.1 (the deepest treatment),
echoed as a dependency in `tucker_decomposition.md`, `dynamic_mode_decomposition.md`,
and `ornstein_uhlenbeck_processes.md` — all four assume this is settled
upstream and explicitly decline to settle it themselves.

Three placements remain live and are laid out with a full pro/con table in
`gaussian_copula_and_marginals.md` §6.1: (i) on raw per-cell physical
anomalies (current default; maximal interpretability, most LUTs, most
expensive); (ii) on that variable's own Tucker-core coefficients (fewer
LUTs, but "per-cell" stops meaning a physical grid location); (iii) on that
variable's DMD mode amplitudes/residuals (most directly serves the
OU-innovation Gaussianity requirement, but by then two prior linear
compressions have already run on non-Gaussianized data, and complex-mode
Gaussianization reintroduces the Class-C direction-handling problem in a
new guise). **This is the single most consequential open question in the
whole documentation set** — it is a live design decision requiring empirical
validation (comparing physical-space QQ-plots and reconstructed extreme
statistics under each placement) that has not been performed by any
document here, and no amount of further documentation-level reasoning
closes it; it needs to be settled empirically, once, against real training
data.

**Update (2026-08-16): a fair placement comparison turns out to be blocked
on §5.3, not independent of it.** An empirical run in
`notebooks/00_gaussianization_placement.ipynb` (30-day `sigwh` training
window, held-out 10-day test period, fixed `R_LAT=10, R_LON=20, r_dmd=20`)
initially found placement (iii) catastrophically unstable out-of-sample
(held-out 99th-percentile reconstruction error +255.89, vs. -0.02 for
placement (i) and +0.15 for placement (ii)) — a result that read as strong
evidence for Gaussianizing before any decomposition. A follow-up
sensitivity sweep in the same notebook traced that instability to
`dmd.Phi`'s condition number (~1.2e6 at those ranks) rather than to
placement (iii) per se: dropping `r_dmd` from 20 to 5 on the *same* data
cut the condition number ~5.5x and the held-out error ~100x; matching the
Tucker core down to a 32-dimensional state (`r_lat=4, r_lon=8, r_dmd=5`)
made the held-out reconstruction essentially exact (p99 error -0.05).
Widening the spatial crop alone, holding rank fixed, helped much less
(~2.5x on conditioning) — full table and derivation in §5.3's own update
below. **Consequence: a placement-(iii) failure measured at an
arbitrarily over-parameterized rank is not evidence against that
placement in general, only against that rank choice.** §4.1 cannot be
fairly re-evaluated until §5.3 has a real answer (or at minimum, until
whoever next runs this comparison reports `dmd.Phi`'s condition number
alongside the reconstruction error for every rank tested, rather than
trusting one unjustified rank).

**Decision (2026-08-19): adopting placement (i) — Gaussianize after
physical transformations and climatology removal, before Tucker — as the
standing default, without waiting for §5.3 to fully close.** This is a
deliberate design choice, not a proven empirical result: rigorously closing
the "before-Tucker (i) vs. after-Tucker/before-DMD (ii)" comparison would
need task 07's rank-selection work to land first, and no evidence gathered
so far makes that comparison decisively one way. The choice rests on two
grounds: (a) it is the natural consequence of how the physical fields are
already processed — fit the marginal-normalizing transform on the same
climatology-removed, physically-transformed residual scale that all
downstream linear compression (Tucker, then DMD) subsequently operates on,
rather than on an opaque, rank-and-basis-dependent coefficient space; (b)
it matches this project's own held-out empirical result immediately above:
placements (i) and (ii) were nearly indistinguishable (<1e-2 error) once
the DMD-rank confound was controlled for, so picking (i) costs nothing on
the evidence actually in hand.

An external check was also run before finalizing this: four independent
multi-agent literature/experiment reviews (run outside this repo, pasted
in for consideration) were asked the same before/after-Tucker/after-DMD
question. Of eight distinct answers returned (four runs, one set
duplicated by a context compaction), six favored placement (i) as a
default, one favored placement (ii), and one refused to pick a default,
reporting a synthetic experiment where pre-reduction Gaussianization
*worsened* physical-space reconstruction even while improving latent
Gaussianity/dependence diagnostics. None of the bundle's own "code
experiments" ran the real Tucker/DMD stack — most failed outright
(`numpy`/`scipy` unavailable in the sandbox) and the rest were toy 1-D/2-D
synthetic analogues (rank-1 "Tucker-like" projections, scalar AR(1) in
place of DMD) — so the bundle is literature-grounded theoretical framing,
not independent empirical confirmation, and did not by itself decide this.
The one point where it is consistent and useful: placement (iii)
(Gaussianize after DMD directly) is essentially never recommended as a
default, on the Koopman/DMD-theory grounds that a nonlinear transform
applied after a linear operator has already been fit cannot retroactively
linearize that fit — this reinforces, rather than newly establishes, this
project's own finding above that placement (iii) is the one that fails
badly at plausible ranks.

**Left as pending work, deliberately not blocking:** whether placement
(ii) is actually equivalent to, or better than, (i) once §5.3's
rank-selection work is real. Pick this up either as a follow-on empirical
task (re-running task 00's held-out comparison at a defensible,
task-07-derived rank) or via a more targeted literature search than the
one summarized above — whichever comes first. It does not need to resolve
before implementation proceeds with (i) as the default.

**Update (2026-09-13): superseded — placement (ii) adopted as canonical.**
Task 07's era-2 cloud rank sweep (see `tasks/07-tucker-dmd-rank-selection.md`
§5) ran the exact task-07-derived comparison this section called for:
Tucker at rank (31,72) — 5% of era 2's (621,1440) wave grid — on the
`ClimatologyRemover` + `PerCellStandardizer` output, once with `GaussianCopula`
applied before Tucker (placement i) and once with Tucker applied directly to
the un-Gaussianized standardized anomalies (placement ii). Result:
`err_pct` 50.81% (i) vs. 53.04% (ii) — **statistically indistinguishable**,
not the large gap a linear-rank-inflation hypothesis (per-cell Gaussianization
preserves rank/copula dependence exactly, by Sklar's theorem in §2.2, but not
the linear/Pearson covariance structure Tucker actually optimizes against —
see `gaussian_copula_and_marginals.md` §2.4) predicted going in. That
hypothesis is not supported by this data; the high error at low rank on this
window is evidently about rank sufficiency, not Gaussianization placement.

Despite the near-tie, **placement (ii) is adopted as canonical** on
architectural grounds, not empirical superiority:

1. Tucker operates on the physically-grounded, climatology-removed
   log-energy anomaly field rather than copula(Z)-space — keeping the
   compression step in the same units/interpretation as the rest of the
   physical pipeline.
2. Gaussianization moves to immediately before DMD (on the much smaller
   Tucker-core coefficients), i.e. applied exactly at the one stage
   (`mode-OU`'s innovation-Gaussianity requirement) that actually needs it,
   rather than broadly upstream of a stage (Tucker) that doesn't benefit
   from it.
3. This placement was also the one that surfaced (and let the team fix) a
   real latent gap in `PerCellStandardizer` — a handful of near-zero-variance
   cells (intermittently ice-covered) had no floor on the division, producing
   `~1e15`-magnitude values; previously silently absorbed by `GaussianCopula`'s
   clipping under placement (i), never visible until placement (ii) fed
   raw standardized anomalies straight to Tucker. Fixed in `src/train/
   standardize.py` (`_floored_divisor`, floors the divisor only, never
   `sigma_cell_` itself — see that file's module docstring for why).

**The specific Tucker rank is not yet decided** — see
`tasks/07-tucker-dmd-rank-selection.md` §5 for the era-2 sweep's rank-vs-error
numbers and the practical (cost, not memory) ceiling found around 20-30% rank
on the current VM. 20% is the current working baseline; finalizing a
production rank is deferred pending either more compute budget or a cheaper
Tucker fitting path.

### 4.2 What copula family should be used — Gaussian, or one with tail dependence? — **OPEN** (distinct axis from 4.1)

Raised in: `gaussian_copula_and_marginals.md` §6.2, `physical_consistency_constraints.md`
§6.

Independently of *where* Gaussianization sits (§4.1), there is a second,
orthogonal open question: *what kind* of copula to use at all. The current
design's implicit Gaussian copula has provably zero upper-tail dependence
($\lambda_U=0$ for $\rho<1$), which both documents identify as the root
cause of the physical-consistency violations the rejection-sampling layer
(`physical_consistency_constraints.md`) exists to patch after the fact. A
$t$-copula (tunable $\lambda_U>0$ via one degrees-of-freedom parameter) or
splicing Generalized Pareto Distribution tails onto each cell's empirical
CDF are both named as root-cause alternatives. **Neither document resolves
which to use, nor whether to do so instead of, or in addition to,
rejection sampling.** A $t$-copula/GPD-tail construction could in principle
be placed at any of the three §4.1 locations, so resolving §4.1 does not
resolve this, and vice versa — they are genuinely independent axes of the
same design space, and both remain open.

---

## 5. DMD-specific questions

### 5.1 Should DMD vectorize each Tucker-core time slice into a state vector (classical exact DMD), or use a tensor-native formulation? — **ANSWERED** (recommendation)

Raised in: `dynamic_mode_decomposition.md` §4.2/§6, which explicitly declines
to take a position, calling both "legitimate."

Reading the reasoning already laid out in that document, a default is
defensible: **use the narrow reshape** — vectorize each `(r_lat, r_lon)`
Tucker-core time slice into a single column and proceed with classical
exact DMD. The document's own analysis already establishes this is *not*
the forbidden representational flattening (it reshapes an already-compressed
coefficient grid whose spatial adjacency was captured separately by
Tucker's own factor matrices, not the raw physical grid), so there is no
principled reason tied to this documentation set's core tensor-representation
concern to avoid it. Tensor-native (e.g. higher-order) DMD variants remain
a legitimate escalation path if empirical evidence later shows the
vectorized fit loses meaningful structure — consistent with this repo's
general pattern of an escalation ladder (plain method first, fancier
variant only once a concrete deficiency is observed) rather than a reason
to default to the more complex option up front. **Recommendation, not a
closed empirical fact** — flagged as such.

### 5.2 Should conjugate-paired DMD/OU modes be sampled independently, or forced to satisfy the exact conjugate relationship at every simulated step? — **ANSWERED**, concrete procedure validated (2026-09-24); production port still task 9's job

Raised in: `dynamic_mode_decomposition.md` §2.4/§6, `ornstein_uhlenbeck_processes.md`
§6.

The *principle* is not actually in doubt once stated precisely: a real
physical field requires $\eta_{k'}(t) = \overline{\eta_k(t)}$ exactly, not
merely in distribution — independently sampling two conjugate-paired modes'
stochastic residuals will, with probability 1, produce a complex-valued
reconstructed field, which is unphysical. So the two modes must be treated
as **one shared stochastic driver**: fit once, simulate one member of the
pair, obtain the other by conjugation — not as two independently-fit,
independently-simulated processes. Both sibling documents already lean
toward this conclusion without quite stating it as a decision. **What
remains genuinely open is the specific fitting/simulation procedure** for
that shared-driver treatment (exactly how the single complex OU fit is
parameterized and sampled such that conjugation is exact rather than
approximate) — that implementation detail is left to whatever code
performs sampling.

**Update (2026-09-24), task 07's DMD+OU generative fidelity check
(`tasks/07-tucker-dmd-rank-selection.md` §8):** the missing procedure is
now concretely worked out and empirically validated, in an experimental
script — not yet ported to the production sampler
(`weather/simulate.py`), which remains task 9's job. For a conjugate pair
$(k,k')$, $|\eta_k(t)| = |\eta_{k'}(t)|$ exactly, always — so only the
*magnitude* needs a stochastic model at all; phase is purely
deterministic, advancing at the mode's own frequency $\arg(\lambda_k)$,
continued from the true measured phase at the end of training. One
shared magnitude draw per pair, phase negated for the partner, makes the
two exact conjugates by construction, with no explicit forcing step.
Magnitude itself is fit in **log** space (de-meaned, mean restored at
generation) since it is non-negative with a nonzero mean, unlike the
zero-mean-reverting residual `fit_mode_ou`'s existing math assumes —
consistent with this project's log-space convention for
physically-positive quantities. Validated empirically
(`notebooks/07_dmd_ou_generative_check.py`): the naive independent-
sampling approach this section originally warned against produced
`imag_real_ratio_mean` ≈0.93-1.04 at every DMD rank tested (comparably
imaginary as real — not safely-discardable noise); the fix brought it to
≈3×10⁻¹⁷-6×10⁻¹⁷ (machine precision) at every rank 5-200 on the real
35,712-dimensional core.

### 5.3 How should Tucker ranks $(r_{\text{lat}}, r_{\text{lon}}, r_{\text{time}})$ and DMD rank $r_{\text{dmd}}$ be chosen? — **DECIDED**: `r_dmd=100` (2026-09-25); Tucker rank 40% `(248, 576)` (2026-09-29, pragmatic/budget-bound, no elbow found)

Raised in: `tucker_decomposition.md` §6, `dynamic_mode_decomposition.md` §6
(the same underlying issue, applied to two different decompositions).

Both documents describe only a diagnostic (elbow of the reconstruction-error
curve for Tucker; eigenvalue-spectrum/growth-rate plausibility for DMD), not
an automatic stopping rule. **Remains open** as a per-variable (and, per
§3.2 above, per-group) judgment call to be made and re-checked by whoever
trains the model — no document proposes a data-driven automatic rank
selector, and none is proposed here either, since doing so responsibly
requires empirical cross-validation this documentation-only exercise cannot
perform.

**Update (2026-08-16): empirical evidence that rank choice matters more
than expected, plus a cheap candidate diagnostic.** A sensitivity sweep in
`notebooks/00_gaussianization_placement.ipynb` (`sigwh`, 30-day training
window, southeastern-tropical-Pacific crop) fit Tucker+DMD at several
`(crop, r_lat, r_lon, r_dmd)` combinations and measured both `dmd.Phi`'s
condition number and held-out reconstruction error at the 99th percentile:

| crop cells | $r_{\text{lat}}$ | $r_{\text{lon}}$ | $r_{\text{dmd}}$ | $\operatorname{cond}(\Phi)$ | held-out p99 error |
|---|---|---|---|---|---|
| 6384  | 10 | 20 | 20 | 1.17e6 | +255.89 |
| 6384  | 10 | 20 | 5  | 2.10e5 | +2.36   |
| 6384  | 4  | 8  | 5  | 1.50e5 | -0.05   |
| 10080 | 10 | 20 | 20 | 4.72e5 | +26.59  |

`dmd.Phi`'s condition number tracks held-out reconstruction quality closely
and, unlike the error itself, can be computed cheaply
(`np.linalg.cond(dmd.Phi)`) with no held-out data at all — a candidate
practical diagnostic: treat a rank choice as suspect if it leaves
`dmd.Phi` far worse-conditioned than a smaller-rank fit on the same data
achieves, rather than trusting whatever rank was chosen a priori. Crop size
(spatial extent) also improves conditioning, but far less than rank choice
does at fixed rank (~2.5x vs. ~5.5x here) — some, but not most, of the
effect is the "narrow slice of state space" the DMD/Koopman framing would
predict. This is not a full automatic rank-selection rule — no document
proposes one, and this finding doesn't either — but it is a concrete,
falsifiable signal no prior document identified, and it directly explains
why §4.1's placement comparison was initially confounded (see that
section's update).

**Update (2026-08-20, task 07): full rank sweep on real `wsh`/`wsp`
(task 17's corrected mask) finds no elbow at all within a feasible range —
the binding constraint is memory, not a plateau.** Swept
`(5,10),(10,20),(20,40),(40,80),(80,160)` on the same 280×230/`T=1448`
crop for `sigwh`, `wsh` (log-energy, task 17's height-floor sentinel), and
`wsp` (log-period, task 17's `T_SENTINEL_S` sentinel). All three curves
decrease monotonically and materially at every step, steepest at the top
of the range — `sigwh` 9.64%→0.43%, `wsh` 63.16%→23.81%, `wsp`
14.37%→5.84% (full table: `tasks/07-tucker-dmd-rank-selection.md` §4).
Peak memory for a single masked `partial_tucker` fit on this crop,
measured via an isolated `RLIMIT_AS`-capped subprocess, is **~12.25–
12.27GB and essentially rank-independent** from `(40,80)` through
`(80,160)` — dominated by fixed per-ALS-iteration tensor
unfolding/reconstruction cost, not by the target rank's core size (~74MB
even at `(80,160)`). `(80,160)` is the largest candidate tested not
because error stopped improving there, but because it is close to the
largest rank this machine (31GB total / ~18GB available) can fit at all.
Recommendation: `(r_lat=80, r_lon=160)` as a hardware-bounded default for
`wsh`/`wsp`, re-evaluate on any machine with materially more available
memory rather than trusting it as a long-term ceiling.

**Update (2026-09-18, task 07): DMD half run against a real era-2 core
under the now-canonical placement (ii); no elbow found, but a settled,
important side question — climatology removal is not the cause and
should not be reversed.** Full writeup and raw results:
`tasks/07-tucker-dmd-rank-selection.md` §6.

Swept `r_dmd \in \{5,10,20,40,50,80,100,150,200\}`
(`notebooks/07_dmd_rank_sweep.py`) against the one Tucker core currently
trusted (era 2, 9-month window, `(r_lat=124, r_lon=288)`, 20% rank —
§4.1's placement decision), fitting both the copula and DMD on an 80/20
train/holdout split of that core (never on the whole window before
splitting, to avoid a second leakage channel beyond the Tucker factors'
own mild one). **No growing modes anywhere in the tested range**,
`cond(\Phi)` climbing smoothly from 2.2 to 227 with no sharp knee (unlike
the 10x jump between `r_dmd=10` and `20` in the 2026-08-16 finding above).
A one-step-ahead prediction error stayed at 98.5-99.7% across the *entire*
range, essentially flat despite `cond(\Phi)` spanning two orders of
magnitude — **the 2026-08-16 finding that `cond(\Phi)` tracks held-out
quality does not clearly hold here**, at least not within this range; that
finding was established with DMD fit on a raw, non-Gaussianized core
(testing placement iii), and this is the first time it's been checked
against a Gaussianized-input (placement ii) DMD fit.

Root-caused the flat ~99% error to rank capacity, not prediction quality:
a POD/SVD of the same Gaussianized core shows only 31.32% of its variance
captured by the leading 200 (of 1754 possible) modes — 50% needs ~411,
90% needs ~1287. (Getting a trustworthy number here took two failed
metric attempts, both left in `07_dmd_rank_sweep.py`'s history as a
caution for whoever revisits this: the first evaluated the deterministic
trajectory $b_0\lambda^t$ from the training window's absolute $t=0$, which
underflows to numerical zero by $t\approx n_{\text{train}}$ regardless of
`r_dmd` since every $|\lambda|<1$; the fix re-anchors the amplitude at the
*measured* state at the end of training and propagates it forward by
$\lambda^{\text{lead}}$ for small lead times — the standard DMD state-
propagation evaluation, not a replay of the deterministic trajectory from
the fit's own origin.)

**Follow-up test of a real hypothesis this raised**: does removing
climatology (§3.9) starve DMD of the one genuinely low-rank, oscillatory
signal it would otherwise be good at, and should climatology be left in
the anomaly DMD actually sees instead? Tested directly
(`notebooks/07_dmd_climatology_only_check.py`): projected
`ClimatologyRemover`'s fitted signal *alone* (its 6-term-per-cell
harmonic+trend prediction — `wsh` is not diurnal — zero anomaly, zero
noise) through the same Tucker factors and checked its own POD spectrum
and DMD fit. Result: needs only **3 modes for 90% of its variance, 5-6 for
~100%** — in sharp contrast to the real anomaly core above. At `r_dmd=6`,
DMD reconstructs it to 0.07% error and recovers the *exact* physical
periods baked into it (365.3 days annual, 182.6 days semi-annual, plus a
static/trend mode); under-parameterizing at `r_dmd=4` (below the true
~6-dimensional rank) destabilizes the fit (2 growing modes, nonphysical
233-day/3467-day "periods") — a clean confirmation that under-choosing
`r_dmd` for even a purely deterministic low-rank signal breaks DMD, not
just a theoretical concern.

**Conclusion: the mechanism is real, but it argues against reversing
climatology removal, not for it.** Climatology genuinely is the
low-rank, DMD-friendly part of the signal, and removing it does leave DMD
facing only the high-rank part. But that low-rank structure is already
captured *exactly and cheaply* by `ClimatologyRemover`'s 6 coefficients
per cell — asking DMD to re-derive it from noisy data would spend modes
re-learning something already known exactly, for no benefit, since
climatology and the anomaly are separable, near-orthogonal problems: the
anomaly's own rank requirement doesn't shrink just because a few "free"
easy modes are added alongside it, and a low-rank climatology component
sitting in the same fit could actively mislead rank selection (masking
the hard part's error at low `r_dmd` for the wrong reason). **Decision:
climatology removal stays exactly where it is (before Tucker, before
DMD).** The anomaly's high effective rank is treated as an intrinsic
property of weather-scale variability once the predictable seasonal part
is gone — consistent with the architecture's own split between DMD
(whatever *is* low-rank/oscillatory) and mode-OU (everything else, as
stochastic noise) — not a pipeline defect.

**`r_dmd` itself remains undecided.** Unlike Tucker, this sweep was not
memory- or cost-bound (every fit costs the same ~30s regardless of
`r_dmd`, confirmed empirically, and runs entirely locally, no VM needed)
— extending the candidate list past 200 to look for an actual elbow is
cheap and a natural next step, just not yet done. The old crop-based
`r_dmd\approx20` vs. `r_dmd=10` contradiction (`tasks/07-tucker-dmd-rank-
selection.md` §4) is superseded by this sweep at a completely different
scale (35,712-dimensional core vs. 800) and should be treated as moot,
not resolved in either direction.

**Update (2026-09-19): physical-forecasting context for interpreting the
"no elbow found" result above.** Prompted by a natural follow-on question
— how much of the post-climatology anomaly is genuinely learnable
structure a higher-rank DMD could still capture, versus irreducibly
random/OU-appropriate — this researched what real physics-based
ocean/atmosphere forecasting models actually do. Full source list in
`tasks/07-tucker-dmd-rank-selection.md` §7.

*Operational wave models do not solve Navier-Stokes.* WAVEWATCH III,
SWAN, and WAM solve the **phase-averaged spectral wave action balance
equation** — a statistical energy-transport description across
wavenumber-direction space, with parameterized source terms for wind
input, nonlinear resonant interactions, whitecapping/dissipation, and
bottom friction — not a phase-resolving fluid simulation. True
Navier-Stokes/Boussinesq/RANS ("phase-resolving") models exist, but a
recent review (arXiv:2511.21856) puts them at 10-200x the computational
cost of phase-averaged models and restricted to local/regional domains
(10⁻²-10² km², minutes-to-days), versus phase-averaged models' basin/
global scale (10²-10⁷ km², days-to-decades). Basin-scale, multi-day
forecasting has never been an NS problem operationally, for cost reasons
that don't go away with more compute — this project's own Tucker/DMD
approach (a coarse, statistical/spectral-style compression, not a
phase-resolving simulation) is in the same family as the operational
approach, not a simplification of a "more correct" NS-based method.

*Forecast horizons give a concrete number for "how much is actually
learnable."* Operational global wave models run hourly forecasts to 120h
(5 days), 3-hourly to 180h (7.5 days). Atmospheric predictability has a
hard, chaos-imposed ceiling around two weeks (Lorenz's classical result),
with current models reliably skillful to ~9-10 days and large-scale
phase/position retaining some skill to ~15 days. Past that horizon, poor
skill is not a model-quality problem — it is provably unpredictable from
initial conditions, no matter the model or compute budget. This reframes
the DMD sweep's ~1287-mode requirement for 90% variance (above): plausibly
a mix of (a) genuinely deterministic structure a smarter or higher-rank
fit could still capture, up to some horizon, and (b) an irreducibly
chaotic component beyond that horizon where more modes fit noise rather
than generalizing — not evidence that the sweep or the pipeline is wrong.

*Swell vs. wind-sea — a flagged, not-yet-pursued idea.* Swell (waves that
have propagated away from their generation region) is far more
predictable at long range than local wind-sea, because swell reduces to
near-linear dispersive propagation while wind-sea stays tightly coupled to
the chaotic local wind field (arXiv:2510.06484, Navy ESPC v2 long-range
wave spectra skill). Separating swell from wind-sea before Tucker/DMD —
analogous in spirit to climatology removal — could plausibly concentrate
more of the genuinely learnable structure into DMD's domain, leaving a
cleaner, more honestly-stochastic residual for OU. **Not actioned this
round** — recorded as a candidate follow-on, not a decision.

*The DMD+OU architecture is independently validated by operational
practice, not just internally consistent.* NWP already uses the same
deterministic-dynamics-plus-stochastic-residual pattern operationally
(SPPT, SKEB — stochastic perturbations representing unresolved sub-grid
physics), and Ornstein-Uhlenbeck processes specifically are used to model
the temporal autocorrelation of dominant EOF/mode amplitudes in stochastic
climate models (arXiv:2603.26693) — the same technique as this project's
own mode-OU stage, not a bespoke choice.

**Update (2026-09-24): two experiments testing whether "no elbow" means
real missing structure or intrinsic noise — decisive on the first count,
suggestive of a second consideration on the practical-generation side.**
Full method and results: `tasks/07-tucker-dmd-rank-selection.md` §8.

*Phase-randomized surrogate test — decisive.* Built 20 independent
Prichard-Theiler multivariate FFT surrogates of the Gaussianized core
(stationary Gaussian processes sharing the real data's exact second-order/
covariance statistics, nothing beyond) and ran the identical DMD rank
sweep on each. At *every* `r_dmd` from 5 to 200, for both the 1-step and
48-step held-out lead, real error sits inside the 20-surrogate band (e.g.
`r_dmd=200`: real 98.49% vs. surrogate mean 98.65% [98.08, 99.33]). A
linear predictor is Bayes-optimal for a Gaussian/linear process, so the
surrogate band traces the covariance-implied error floor for any rank —
real data never beats it. **This is the decisive version of the point
this section's 2026-09-19 update raised speculatively**: the ~69% of
variance DMD can't reach by `r_dmd=200` is statistically indistinguishable
from correlated noise with the anomaly's own second-order statistics, not
recoverable structure a higher or different rank could capture. Extending
the sweep past 200 (previously flagged as cheap and open) is no longer
expected to find anything — the ceiling isn't a search artifact.

*Full DMD+OU generative fidelity check — consistent, plus one new lead.*
Using §5.2's now-validated conjugate-pair-coherent OU sampling procedure,
generated 10 independent synthetic holdout realizations per rank and
compared their statistics against the real held-out data. Spectral match
(`log_psd_distance`) improves steadily from `r_dmd=5` (46.6) to `r_dmd=150`
(1.82), then is flat-to-slightly-worse at `r_dmd=200` (2.17) — consistent
with the surrogate test's ceiling. The marginal-distribution match
(`ks_d_statistic`) is **not** monotonic: it improves to a minimum around
`r_dmd=50-80` (≈0.06) and then gets *worse* from `r_dmd=100` to `200`
(0.066 → 0.180) — plausibly `cond(Φ)`'s well-established rise across this
range (2.2→227, §6) degrading the numerical stability of the per-mode
magnitude/phase estimates used for generation, though this specific
mechanism is a hypothesis here, not separately verified. If it holds up
under further scrutiny, it's a *second*, independent reason (distinct
from "no missing structure") to prefer a more modest `r_dmd`: not just
"higher doesn't help," but "past roughly `r_dmd=100`, generative fidelity
may actively degrade." Combined, both experiments point away from
chasing `r_dmd` toward 200 and toward a more modest working value
(tentatively somewhere in the 50-100 range per the KS minimum), though a
final production `r_dmd` is still not picked — see
[[project_climatology_removal_confirmed_correct]].

**Update (2026-09-25): real-data (no surrogate) 8-fold walk-forward CV
confirms the above is not a single-split artifact.** Full method/results:
`tasks/07-tucker-dmd-rank-selection.md` §9. Re-ran the held-out-error
comparison using only real data — 8 genuinely different real test
windows (`TimeSeriesSplit`, expanding training window) instead of the one
80/20 split every prior check used, with a rolling anchor through each
~190-step real test block (142 anchors/fold) rather than a single
boundary anchor. Result: one-step predictive accuracy barely moves across
the *entire* `r_dmd=5-200` range — under 0.3 percentage points from worst
to best, roughly 4x smaller than the original single-split test's ~1.2-
point spread — and this near-flatness **replicates almost identically
across all 8 independent real windows** (fold-to-fold std at any given
rank: 0.07-0.13 points, small relative to the already-small overall
spread). This is the real-data confirmation the surrogate test's
synthetic-noise-band argument implied but couldn't directly demonstrate:
rank truly doesn't matter much for predictive accuracy, checked directly
against real held-out weather, not just against a statistical null model.
Recommendation unchanged, now on firmer footing: **`r_dmd` in 50-100**.

**Decided (2026-09-25): `r_dmd = 100`.** Full justification:
`tasks/07-tucker-dmd-rank-selection.md` §10 (plots:
`notebooks/dmd_rank_fidelity_tradeoff.png`,
`notebooks/dmd_rank_cv_predictive_error.png`). The generative check's
two fidelity metrics disagree on where "best" is — spectral
(`log_psd_distance`) at `r_dmd=150`, marginal (`ks_d_statistic`) at
`r_dmd=80` — and `r_dmd=100` is the point between them where that
disagreement stops being worth chasing in either direction: moving to 80
would buy an 11.9% better `ks_d` at the cost of a 121% worse
`log_psd_distance`; moving to 150 would buy a 25.7% better
`log_psd_distance` at the cost of a 49.8% worse `ks_d`. The real-data CV
result is flat across all three candidates regardless (99.60-99.64%,
inside the fold-to-fold noise band), so it imposes no penalty on this
choice. This closes the DMD half of §5.3 — Tucker's own rank remains
open.

**Decided (2026-09-29): Tucker rank = 40% `(248, 576)`.** Full record:
`tasks/07-tucker-dmd-rank-selection.md` §12. The final VM run hit its 11h
cap at iteration 170/300 with 23.30% reconstruction error, one checkpoint
short of its own early-stop threshold — better than 30% (27.47%, 8h cap)
and 20%'s fully converged 32.90%. Unlike `r_dmd`, this is a pragmatic
choice made when the Azure credit ran out, not a swept optimum: no elbow
was found, and a higher rank was never tested to convergence. Production
copula and DMD (`r_dmd=100`) were fit on the full 40% core.

### 5.4 Is masked-HOOI's imputation of land cells guaranteed to converge well? — **OPEN** (mitigation available)

Raised in: `tucker_decomposition.md` §6.

No — masked HOOI converges monotonically to a stationary point of a
non-convex objective, not provably to the global optimum, and large
contiguous unobserved regions (large landmasses with little interior
coastline to anchor the imputation) are a realistic case for a poor local
completion. This is a genuine numerical limitation, not just a theoretical
caveat. **Remains open** as a property of the method, but the standard
numerical-optimization mitigation applies and is worth stating explicitly
since no document does: run masked HOOI from multiple random initializations
and compare reconstruction error/imputed-region stability across them,
flagging variables or regions where the imputation is sensitive to
initialization for closer inspection, rather than trusting a single run
blindly.

### 5.5 What should happen if a fitted DMD mode is genuinely growing ($|\lambda_k| > 1$) rather than just noisy? — **OPEN** (mitigation available)

Raised in: `dynamic_mode_decomposition.md` §6.

If DMD's deterministic backbone drives long-horizon stochastic simulation,
a genuinely growing mode diverges unboundedly. No document specifies
detect-and-handle logic. **Remains open**, but a mitigation consistent with
this documentation set's own established pattern (`physical_consistency_constraints.md`'s
rejection-sampling philosophy) is a reasonable default worth naming: clip
$|\lambda_k|$ to $1$ at simulation time for modes exceeding it by a small,
plausibly-noise-attributable margin, but flag (rather than silently clip)
any mode whose growth exceeds a larger threshold for manual review — the
same "detect, don't silently absorb" philosophy the physical-consistency
document already applies to individual samples, applied here to a fitted
model artifact instead.

### 5.6 Does NaN in a partition-energy field mean land, or a genuine "no partition present" physical zero? — **DECIDED** (policy validated empirically, task 17)

Raised in: task 07's `wsh` rank sweep (0-11% of cells passed even a
90%-valid tolerance over 6 months, forcing a `sigwh`-substitution
workaround).

No document (`model_design.md:308,313,315`; `variable_taxonomy.md` §4)
distinguished these two NaN causes before this -- both were described
purely as land/ice. For the partition triplets (wind-sea, peak, primary
swell), NaN also occurs at genuine ocean cells whenever that partition is
simply absent -- a real physical state, not missing data. There is also no
authoritative, variable-independent land mask anywhere in the codebase;
every mask was inferred from a variable's own NaN pattern, which conflates
the two causes for exactly the partition variables where it matters.

**Decided (`tasks/17-partition-nan-sentinel-policy.md` §1, notebook +
`src/train/masking.py`-scoped only, not yet wired into production):**

1. **Sentinel semantics.** At a genuine ocean cell where a partition
   variable is NaN, fill with a sentinel and mark **VALID** -- "no
   partition" is a real physical observation Tucker/DMD fit directly on,
   not missing data for masked HOOI to impute. Land cells keep today's
   treatment (zero-filled, INVALID/imputed).
2. **Blast radius.** `src/train/masking.py` (new module) + notebooks
   only. `src/weather/model.py` and `src/train/ingest.py` are untouched.
   Production wiring is a deliberately deferred follow-on task.
3. **Log-space companions differ by variable, both decided now, not
   deferred:** height's sentinel `H=0` is the clean physical limit as
   $m_0\to0$ ($H_{m0}=4\sqrt{m_0}$), so only a numerical floor is needed
   before `log E:=log(H^2)` (`H_FLOOR_M=0.01`, 1cm). Period's
   $T_{m01}=m_0/m_1$ has no analogous clean limit -- a genuine $0/0$ form
   -- but is not arbitrary either: if a fading partition keeps a roughly
   fixed spectral shape while only its amplitude shrinks, the ratio
   tends to a well-defined, shape-determined limit (an L'Hopital-style
   cancellation), estimated empirically from real near-vanishing
   partitions rather than invented or taken as an archive-wide typical
   value.

**Empirical categorization and validation (real archive data, 2026-08-20,
280x230 South Pacific crop, 6-month window -- full derivation and numbers
in `tasks/17-partition-nan-sentinel-policy.md` §4):** `wsd`, `pwd`, `pwp`,
`pswd` carry zero ocean-cell NaN; `wsh`/`wsp` (36.5%) and `pswh`/`pswp`
(1.5%, milder) do. `wsd` is never NaN even where `wsh` is, and is not a
stale filler (its variance is *higher*, not collapsed, during `wsh`-NaN
periods) -- plausibly `wad`-derived, an open data-provenance question not
resolved here. The derived period sentinel is `T_SENTINEL_S ~= 3.84`s
(extrapolated zero-height limit, cross-validated against a conditional-
median estimator, 0.51s agreement at the chosen bin width). Rerunning
`wsh`'s and `wsp`'s Tucker rank sweeps under the corrected mask fixes the
original pathology outright: coverage goes from 7-11% to **100%**, and
reconstruction error now **decreases** with rank (was increasing).

**Still open:** whether/how this gets wired into production, and the
`wsd`-vs-`wad` provenance question above.

---

## 6. Ornstein–Uhlenbeck-specific questions

### 6.1 Is the local-residual variance-ratio floor a physically sound cutoff, or just a reasonable-looking default? — **OPEN**

Raised in: `ornstein_uhlenbeck_processes.md` §6.

The floor is grounded in an efficiency argument (don't bother simulating a
cell/variable whose local residual explains negligible variance), which is
sound as far as it goes, but the actual downstream error its omission
introduces has not been validated end to end — that validation depends on
simulation and hurricane-season-style checks that are not yet in place
anywhere in this repo. **Remains open**; treat any specific numeric
threshold as provisional until such a validation exists.

### 6.2 Where does the local residual live under placement (ii)? — **DEFERRED** (2026-11-22: not part of the first era-3 model)

Raised in: `model_design.md` v3 §3.10 and §5.7, while bringing the handoff
in line with placement (ii) (§4.1).

The local residual is fit after the mode-level OU, as $R = Z - Z_{\text{coherent}}$,
with one independent OU per grid cell (`ornstein_uhlenbeck_processes.md`
§4.1). Its $Z$ was v2's per-cell Gaussianized field. Placement (ii) moved
the copula onto the Tucker core, so that field no longer exists, and the
local residual has never been fit on real data (task 07 validated only the
mode-level OU; task 10 is open). Three candidates:

1. **Standardized-anomaly space.** $Z$ is the standardized log-energy
   anomaly that Tucker consumes; $Z_{\text{coherent}}$ is DMD's
   deterministic core, copula-inverted, then Tucker-inverted. The residual
   is not Gaussian, which the OU fit assumes.
2. **A per-cell copula on the residual**, then the OU. Gaussian innovations
   again, at the cost of a second set of per-cell lookup tables (several GB)
   and an extra inverse step at sampling.
3. **Leave it out.**

**Decision (2026-11-22): option 3 for the first era-3 model**
(`include_local_residual=False`). Revisit once validation shows whether the
missing small-scale variance matters. At 40% Tucker rank, about 23% of the
standardized anomaly's norm lies outside the coherent layer (task 07 §12).
Choosing between options 1 and 2 stays open until then.

---

## 7. Harmonic climatology's quasi-stationarity assumptions

### 7.1 Is a fixed harmonic shape plus one linear trend, held constant over the whole fitting window, an adequate model of the climatological background? — **OPEN**

Raised in: `harmonic_climatology.md` §6 (three related bullets, consolidated
here as one question with three facets, since they are the same underlying
assumption viewed from different angles).

Three linked facts, none resolved by any document: (a) real seasonal cycles'
amplitude and phase can themselves drift over long records (e.g. from
decadal ocean-atmosphere oscillations), which a fixed-shape harmonic design
cannot represent — any such drift is folded into either the harmonic
coefficients as a compromise average or the residual as spurious
low-frequency "noise"; (b) a single linear trend is itself only the
best-fit constant rate over the specific window used, and says nothing
about whether that rate generalizes or is itself the locally-linear piece
of something curved; (c) because the trend coefficient is defined relative
to whichever window it is fit on, two fits over different windows of the
same record are not simply noisier/cleaner estimates of one true number —
they can be honest answers to subtly different questions (a short window
can have interannual variability, e.g. an ENSO event, alias into an
apparent trend that a longer window would average away).

**Remains open.** Point (c) is not purely a climatology-document concern —
it connects directly to a broader, still-open decision elsewhere in this
project (which single archive era, or subset of the archive, to actually
train the production model on — a decision this documentation set does not
touch and does not resolve). Any reported trend coefficient should be read
together with the specific window it was estimated over, not treated as a
window-independent physical constant, until that broader training-window
decision is made.

### 7.2 Is the "waves have no diurnal signal" reasoning resolution-dependent? — **OPEN**, conditionally

Raised in: `harmonic_climatology.md` §6.

Yes, explicitly and narrowly: the claim is stated relative to typical
global-model resolution and forcing data (GFS-Wave/WAVEWATCH III-class
products). It should be revisited only if this pipeline is ever pointed at
a higher-resolution regional wave model, where locally wind-driven
short-fetch seas forced by diurnal sea/land-breeze cycles could carry a
weak but real diurnal signature. **Not an active concern at the current
archive's resolution** — listed here for completeness, not because it
needs attention now.

---

## 8. Physical-consistency constraints

### 8.1 Should the physical-consistency constraints check direction, not just magnitude? — **PARTIALLY ANSWERED**

Raised in: `physical_consistency_constraints.md` §6 ("direction-blind scope
gap... a candidate future extension once a defensible veer/backing
tolerance is derived and validated").

`correlated_variables.md` §3.2 (and §3.3 of this document) now supplies
exactly the physical grounding that was missing: the wind/wind-sea
directional-alignment system has a derived, stable equilibrium at zero
angular offset, with an explicit relaxation rate $\kappa U$ shared with the
magnitude-growth dynamics. This licenses a concrete **candidate constraint,
proposed here for the first time**: flag as directionally inconsistent any
query point where the angular offset $|\theta| = |\angle(\mathbf D) -
\angle(\mathbf U_{\text{vec}})|$ between wind-sea direction and wind
direction exceeds a tolerance $\delta_{\text{dir}}$ — **unless** the local
sea state is young/duration-limited (small $r/r^\star$, i.e. wave age not yet
saturated), exactly mirroring the existing energy-budget constraint's
young-sea exception, since §3.3's derivation shows magnitude and direction
relax on the same shared timescale.

This is a real, physically-grounded proposal, not just a restatement of the
gap — but it is not a closed answer: $\delta_{\text{dir}}$ still needs
empirical calibration against real archive data exactly the way the
existing three constraints' tolerances were calibrated
(`physical_consistency_constraints.md` §5), and the "unless young" exception
requires the same recent-wind-history state that §8.4 below notes is
missing from the current per-query-point-independent design. **Status:
the missing mechanism is now specified; the missing calibration and the
missing state-tracking are not.**

### 8.2 Can a per-query-point-independent check ever catch cross-cell or cross-time physical inconsistency? — **OPEN**, and structurally so

Raised in: `physical_consistency_constraints.md` §6.

No, and this is not a gap to be patched incrementally — it is a structural
property of the rejection-sampling design as specified. A storm's wave
field advecting across adjacent cells at an implausible speed, or a field
at $t$ and $t+\Delta t$ that is mutually inconsistent, would satisfy every
per-point constraint individually and be accepted in full. Fixing this
requires a fundamentally different mechanism (a joint constraint evaluated
over a spatiotemporal neighborhood), which no document in this set
specifies. **Remains open and explicitly out of scope** for the
per-point design as currently specified — not a defect to silently accept,
but also not something to be resolved by refining tolerances.

### 8.3 Rejection sampling, or fix the copula? — **OPEN** (same axis as §4.2)

Raised in: `physical_consistency_constraints.md` §6.

Restated here only to make the cross-reference explicit: this is the same
open question as §4.2 above, viewed from the constraints side rather than
the copula side. The team has chosen rejection sampling for now (fast,
safe, exact given whatever model exists); fixing the copula's tail
dependence remains the more root-cause-oriented alternative. **Remains
open** — see §4.2 for the full framing; resolving it requires the same
empirical work either document alone can specify but not perform.

### 8.4 Does a static Pierson-Moskowitz check misfire on physically valid transient ("young") sea states? — **ANSWERED** (proposal exists; not yet incorporated)

Raised in: `correlated_variables.md` §2.5 and §6, addressed *to*
`physical_consistency_constraints.md`.

Yes — `correlated_variables.md` §2.5 derives this directly from its own
reduced ODE: a duration-limited sea genuinely sitting below the PM
equilibrium curve is a physically valid transient, not necessarily an
inconsistent draw, and a static per-point tolerance-band check has no way
to distinguish "transient-but-valid" from "genuinely inconsistent," since it
only examines one instant. Two concrete refinements are proposed there: a
wave-age- or duration-aware tolerance, or a trajectory check against the
closed-form growth curve rather than only its fixed point. **The answer
exists and is specific** — but both refinements require recent-wind-history
state that `physical_consistency_constraints.md`'s current
per-query-point-independent design does not carry, so implementing either
is an architectural extension of that document's scope, not a parameter
tweak. This consolidation treats the *question* as answered (yes, it
misfires, and here is a specific fix) while flagging that the *code-level
integration* of that fix is still to be done by whoever next edits
`physical_consistency_constraints.md`.

---

## 9. Explicitly out of scope (not open questions — scope boundaries stated by design)

These are not unresolved questions so much as places where a sibling
document deliberately drew a boundary. Listed here only to avoid them being
mistaken for open items that need chasing:

- **`correlated_variables.md`'s reduced ODE system is a pedagogical
  simplification**, not a claim that WAVEWATCH III literally integrates a
  two-equation system. It discards the full frequency-direction spectral
  shape, holds the wind-input coefficient fixed rather than wave-age
  dependent, omits $S_{nl}$'s self-interaction, and treats wind's own
  dynamics as trivial relaxation. This is intentional scope-narrowing for
  pedagogical clarity, not an oversight.
- **`physical_consistency_constraints.md`'s three constraints are
  deliberately narrow** (wave-energy budget, fetch-limited
  Pierson-Moskowitz, peak-period bounds) — "physical consistency" here
  means exactly these three checks, not a general physical-validity
  guarantee. Additional constraints are explicitly deferred to a future
  revision, not silently assumed absent.

---

## Summary table

| # | Question | Status |
|---|---|---|
| 1.1 | Ocean-grid orientation | ANSWERED (row-0-south, col-0-at-0°E-eastward, via direct RTOFS NetCDF lookup) |
| 1.2 | `pwd`/`pswd` convention | ANSWERED |
| 1.3 | `wad` usability | ANSWERED (no) |
| 1.4 | `pwh` independence from `sigwh` | ANSWERED (no — confirmed via NOAA GRIB2 tables too; sibling docs corrected) |
| 1.5 | `*p` fields: peak period or mean period? | ANSWERED (mean period, $T_{m01}$; sibling docs corrected) |
| 2.1 | `sigwh` disjoint-partition assumption | ANSWERED |
| 2.2 | Triplet coupling lost by independent Gaussianization | PARTIALLY ANSWERED |
| 2.3 | `hypot` vs. low-rank Tucker | ANSWERED (resolved via dedicated `log E` tensor) |
| 2.4 | `hypot` vs. independent Gaussianization | PARTIALLY MOOT (magnitude resolved by 2.3; angle open) |
| 3.1 | Cross-variable coupling mechanism | ANSWERED |
| 3.2 | Joint-group selection | OPEN |
| 3.3 | Directional-alignment equilibrium stability | ANSWERED HERE |
| 3.4 | Joint multi-channel Tucker vs. downstream joint model for within-group coupling | OPEN (experiment updated to `{E,F}`, see 3.5) |
| 3.5 | Is `E − 2T ≈ const` clean evidence, or a shared-moment artifact? | ANSWERED (artifact risk confirmed; `{E,F}` fix proposed, unverified) |
| 4.1 | Gaussianization placement | DECIDED (ii, Tucker on log-energy anomalies then Gaussianize the core, before DMD — supersedes the earlier placement-i default; 2026-09-13) |
| 4.2 | Copula family (Gaussian vs. t/GPD) | OPEN |
| 5.1 | DMD state-vectorization | ANSWERED (recommendation) |
| 5.2 | Conjugate-pair sampling | ANSWERED, procedure validated 2026-09-24 (production port deferred to task 9) |
| 5.3 | Tucker/DMD rank selection | DECIDED — DMD: **r_dmd=100** (2026-09-25, knee point between spectral- and marginal-fidelity optima); Tucker: **40% (248, 576)** (2026-09-29, pragmatic/budget-bound, not a swept optimum) |
| 5.4 | Masked-HOOI convergence | OPEN (mitigation proposed) |
| 5.5 | Growing DMD modes | OPEN (mitigation proposed) |
| 5.6 | Partition-absence NaN vs. land/ice NaN | DECIDED (policy validated empirically, task 17) |
| 6.1 | OU variance-ratio floor | OPEN |
| 6.2 | Local residual under placement (ii) | DEFERRED (not in the first era-3 model, 2026-11-22; space still open) |
| 7.1 | Climatology quasi-stationarity | OPEN |
| 7.2 | Diurnal reasoning resolution-dependence | OPEN (conditional, inactive) |
| 8.1 | Direction-blind physical constraints | PARTIALLY ANSWERED |
| 8.2 | Cross-cell/cross-time blindness | OPEN (structural) |
| 8.3 | Rejection vs. copula fix | OPEN (= 4.2) |
| 8.4 | Static PM check vs. young-sea transients | ANSWERED (not yet integrated) |
