# Changelog

Version history for codt_tools and the CODT interface changes it tracks.

**This is not just a record.** Several entries carry facts you need in order
to read *archived* output correctly — what `psigma` meant before CODT 3.1.0,
how the LEM smallest-eddy scale is derived, an eddy-header field that changed
meaning without changing type, and why aerosol seeding must be detected by
probing rather than by a conventions string. Check the run's `code_version`
against these when analyzing old data.

Current behavior lives in [CLAUDE.md](CLAUDE.md).

## 1.0.0 — the case/run/simulation refactor

*Requires CODT 3.1.0 or newer.*

`config.py` / `runner.py` / `slurm.py` / `experiment.py` / the six-table
registry became four layers: `case/`, `run/`, `simulation/`, `registry/`.
A clean break with no compatibility aliases; code pinned to the old names
uses the `v0.8.0` tag.

- **`case/`** — `CODTConfig`/`InjectionData`/`ParcelInput` →
  `Case`/`Aerosol`/`Parcel`. File locations moved out of the case and into
  `write_inputs`. `experiment.py` deleted.
- **Designs** — a point is a dict, a design is a list of dicts; `cross()` is
  the only combinatorial helper. `_SWEEP_ABBREV` auto-naming deleted.
- **`run/`** — `CODTRunner` → `Run` plus `write_local` / `write_slurm_array`.
  **All SLURM submission removed**: codt_tools generates a script, you run
  `sbatch`. `codt_tools/slurm.py` deleted (recover with
  `git show v0.8.0:codt_tools/slurm.py`).
- **`simulation/`** — `CODTSimulation` → `Simulation`, `Simulation.from_run`,
  and the analysis→registry import cut.
- **`registry/`** — six tables and a status lifecycle → one table of seven
  columns. No status, no experiments, no migrations. Pre-1.0 databases are
  neither migrated nor read.

Requires a **CODT 3.1.0+** binary.

## Merged

- **Minimal registry** (Stage 5 of the 1.0 refactor): `registry/api.py` + `db.py` +
  `versions.py` → one `registry/store.py` with a single seven-column table.
  No status, no experiments, no parameters, no migrations. Old databases are
  neither migrated nor read. This also removed the last copy of
  `SUPPORTED_CONVENTIONS`. See "Simulation Registry (rewritten in 0.9.0)".
- **Simulation layer** (Stage 4 of the 1.0 refactor): `simulation.py` + `plotting.py` +
  `trajectory_io.py` → `codt_tools/simulation/`, `CODTSimulation` →
  `Simulation`, `Simulation.from_run`, and the analysis→registry import cut.
  No top-level `conventions.py` — each reader owns its format string instead.
  Breaking, no aliases. See "Simulation Layer (0.9.0)" above.
- **Run layer** (Stage 3 of the 1.0 refactor): `runner.py` + `slurm.py` → `codt_tools/run/`, `CODTRunner` →
  `Run` + `write_local`/`write_slurm_array`. **All SLURM submission removed**
  — the package generates a script and you launch it. No registry coupling.
  `docs/running-on-slurm.md` rewritten as prose guidance. Breaking, no aliases.
  See "Run Layer (0.9.0)" above.
- **Case layer** (Stage 1 of the 1.0 refactor): `config.py` → `codt_tools/case/`, `CODTConfig`/`InjectionData`/
  `ParcelInput` → `Case`/`Aerosol`/`Parcel`, `experiment.py` deleted, and file
  locations moved out of the case and into `write_inputs`. Breaking, no aliases.
  See "Case Layer (0.9.0)" above.
- **Arch-aware array SLURM submission** (codt_tools v0.8.0): new
  `codt_tools/slurm.py`, job-array submission, registry schema v4
  (`build_arch`). No CODT-side change. See "Architecture-aware SLURM
  submission" above and `docs/running-on-slurm.md`.
- **CODT 3.1.0 psigma/n_blob redefinition** (codt_tools v0.7.0): `psigma` is now
  the total domain fraction replaced per entrainment event; validation is
  `int(psigma*N) >= n_blob`. See "CODT 3.1.0 psigma/n_blob redefinition" below.
- **CODT 3.0.0 LEM turbulence scales** (codt_tools v0.6.0): `kolmogorov_length_scale` removed from `&TURBULENCE_LEM`; smallest eddy derived from grid + diffusivities. See "CODT 3.0.0 LEM turbulence scales" below.
- **CODT v3 interface** (codt_tools v0.5.0, for CODT 2.0.0): waypoint-leg parcel input, aerosol seeding, registry input-schema gating. See "CODT v3 interface" below.
- **v0.6.0**: Aerosol detrainment/entrainment during blob events, standalone `entrainment.f90` module, `&ENTRAINMENT` namelist, entrainment budget variables in output NC.
- **v0.5.x**: Flat output directory (no auto-subdirectory), relative paths, removed input file copies, `{name}_DONE` marker.
- **v0.4.0**: Version embedding (`code_version`, `git_commit` in output NC), radiation module, DGM Rosenbrock solver, parcel entrainment framework, pressure_limit.
- **LEM** (PR #11): `simulation_mode='parcel'` with periodic BCs, periodic triplet map, -5/3 eddy selection.
- **Collision-coalescence** (PR #13): Event-driven 1D CC with Hall/Long/unity kernels.

## CODT 3.1.0 psigma/n_blob redefinition

> **Status: implemented in codt_tools v0.7.0.** CODT side is `c529ae9`, tag
> `v3.1.0`. No file format changed; no conventions string moved.

**What changed in CODT.** Through 3.0.1, `psigma` was the size of **one** blob,
so an entrainment event replaced `n_blob * psigma` of the domain and the event
interval carried an `n_blob` factor to compensate. From 3.1.0, `psigma` is the
**total** fraction replaced per event and `n_blob` only subdivides that fixed
volume into evenly sized chunks (`int(psigma*N)` split `n_blob` ways, integer
remainder spread one cell at a time so the total does not drift with `n_blob`).

Consequences for anything reading or writing these parameters:

- **Event timing depends on `psigma` alone.** `dt = (psigma/(1−psigma))/(ent_rate·|vel|)`.
- **`n_blob` is now a pure mixing axis.** Sweeping it holds the entrainment rate
  and per-event volume fixed and varies only the spatial distribution
  (inhomogeneous → homogeneous). Previously such a sweep changed the entrained
  volume instead.
- **Validation changed:** `psigma * n_blob < 1` → `int(psigma*N) >= n_blob`.
  `psigma` alone must still be in (0, 1); `n_blob` is still capped at 10.
- **This departs from EMPM**, which keeps the per-blob reading. Deliberate.

**What codt_tools does about it (v0.7.0).** The new rule needs `N`, which the
parcel writer never received, so `write_parcel` and `ParcelInput.write` take an
optional `n_grid` (the `&PARAMETERS N` the run will use). Pass it and
`int(psigma*N) >= n_blob` is checked per leg; omit it and that check is left to
CODT at run time — existing callers keep working. `Case.write_inputs` supplies
it automatically. `Case.validate()` mirrors CODT's startup abort for the
`&ENTRAINMENT` scalars *and* the per-leg schedule when `do_entrainment` is on.
Note the parcel branch runs `_validate_lem_scales()` first, so a too-coarse `N`
raises there before the entrainment check is reached.

**Output impact.** Same variable and attribute names (`psigma`, `n_blob`,
`PARCEL.psigma`, `PARCEL.n_blob`) carrying the **new** meaning — so old and new
runs are indistinguishable by name. The `long_name` text was reworded
(`psigma`: "Blob Fraction of Domain" → "Total Domain Fraction Replaced per
Entrainment Event"), which is the only in-file hint; otherwise distinguish by
`code_version`. Attribute text only — `CODT_output_v1` is unchanged.

**Comparability of existing runs.** `n_blob = 1` runs are bit-identical across
the change. `n_blob > 1` runs are **not comparable**: an old
`n_blob = 5, psigma = 0.1` event replaced 50% of the domain, a new one replaces
10%. Affected ensembles (including `EXP001_sf01_seeding`) need re-running before
their realization spread can be read as stochastic. Old behaviour is reproducible
exactly by setting `psigma` to the old `n_blob * psigma` with `n_blob = 1`.

## CODT 3.0.0 LEM turbulence scales

> **Status: implemented in codt_tools v0.6.0**, verified against a CODT v3.0.0
> binary (`e1c03be`) in both modes. CODT 3.0.0 removed a namelist parameter and
> changed the meaning of an eddy-header field.

**What changed in CODT**

`kolmogorov_length_scale` was **removed** from `&TURBULENCE_LEM`. A namelist that
still declares it is a **fatal read error** (the Fortran namelist read cannot
match the object name). The smallest turbulence scale is now derived:

| derived quantity | formula |
|---|---|
| `actual_kolmogorov_scale` | `(nu**3 / dissipation_rate) ** 0.25` — **diagnostic only, never governs** |
| `grid_eddy_scale` | `6 * dz`, where `dz = H / N` |
| `diffusivity_length_scale` | `(max(kT, Dv) / (0.1 * dissipation_rate ** (1/3))) ** 0.75` |
| `smallest_eddy_gridpoints` | `max(6, 3 * ceil(max(grid_eddy_scale, diffusivity_length_scale) / (3 * dz)))` |
| `smallest_eddy_scale` | `smallest_eddy_gridpoints * dz` |
| `diffusivity_enhancement` | `(smallest_eddy_scale / diffusivity_length_scale) ** (4/3)` — always ≥ 1 |

Constants in CODT `src/globals.f90`: `nu = 1.488e-5`, `kT = 1.96e-5`,
`Dv = 2.2705e-5` m²/s.

`smallest_eddy_scale` drives the eddy-sampler lower bound, the sampler's
gridpoint floor, and the Reynolds number
`(integral_length_scale / smallest_eddy_scale) ** (4/3)`.

The **LEM** diffusivities are molecular scaled by the same
`diffusivity_enhancement`: `kT * f` and `Dv * f`. Both take the same factor, so
Pr and Sc are preserved, and `f >= 1` means LEM diffusion is never slower than
molecular. **Droplet growth is unaffected** — the DGM computes its own T- and
p-dependent conductivity and vapor diffusivity internally (CODT
`src/DGM.f90:204-205`). Chamber mode is unaffected too.

`f` is a **step function** of the grid, not ≈1: in the diffusivity-limited regime
the 3-cell quantum can be up to half of `diffusivity_length_scale`, so `f` reaches
`(3/2) ** (4/3) = 1.717` near the crossover — at `eps=0.01, H=1`, `N=1025` gives
1.001 but `N=1045` gives 1.675. Read it from `LEM.diffusivity_enhancement`; do
not assume a value.

CODT now also **aborts at startup** when `smallest_eddy_gridpoints > N` or
`smallest_eddy_scale >= integral_length_scale`, and warns when
`integral_length_scale / smallest_eddy_scale < 3`. Generated configs should keep
`integral_length_scale` comfortably above `max(6*H/N, diffusivity_length_scale)`.

**What codt_tools does about it**

- `kolmogorov_length_scale` is gone from the `turbulence_lem` defaults, so
  `write_namelist` no longer emits it.
- `config.lem_turbulence_scales(n, h, dissipation_rate)` reproduces CODT's
  derivation (a mirror of `derive_turbulence_scales` in `src/LEM.f90`) and
  returns all six quantities. **It must track that subroutine.** It deliberately
  reproduces CODT's single-precision `1./3.` and `4./3.` literals; agreement with
  the output attributes is then ~4e-9 relative, the floor set by gfortran's `**`
  vs libm `pow`.
- `Case.validate()` mirrors the new startup aborts: parcel raises on
  `smallest_eddy_gridpoints > N` and `smallest_eddy_scale >= integral_length_scale`
  and warns when the scale separation is < 3 (**this warning fires on the
  defaults**, matching CODT on its bundled `params.nml`); chamber raises when
  `lmin < 6` or `lmin % 3 != 0` (new `src/ODT.f90` guard).
- `load_eddies()` renames parcel header field 1 to `smallest_eddy_scale`
  **unconditionally**. Field count/order/dtype are unchanged (3 × f8), so a
  pre-3.0.0 file still reads without error, but the value under that key is the
  old namelist input — check `code_version`.
- `Simulation.lem_scales()` returns whichever `LEM.*` attributes the file
  carries (empty dict for chamber or pre-3.0.0 runs); `info()` prints them.

**New optional global attributes** on parcel output — additive, so
`CODT_output_v1` is unchanged. Detect by presence, do not require:
`LEM.actual_kolmogorov_scale`, `LEM.grid_eddy_scale`,
`LEM.diffusivity_length_scale`, `LEM.smallest_eddy_scale`,
`LEM.smallest_eddy_gridpoints` (int), `LEM.diffusivity_enhancement`. The old
`TURBULENCE_LEM.kolmogorov_length_scale` attribute is absent from new files.
Consistency check: `smallest_eddy_scale == smallest_eddy_gridpoints * H/N`.

**Version impact.** No conventions string changes — `CODT_output_v1` and all
input strings are untouched, so `SUPPORTED_CONVENTIONS` /
`SUPPORTED_INPUT_CONVENTIONS` were **not** edited. CODT took a MAJOR bump
(3.0.0) for the breaking namelist; codt_tools went 0.5.0 → 0.6.0. There is no
back-compatible writer path: the incompatibility lives in CODT's namelist
parser, so **codt_tools 0.6.0 requires a CODT 3.0.0 binary** and 0.5.x requires
a 2.x one.

**Also in CODT 3.0.0, no I/O impact:** `a760c51` unified droplet eddy transport
for LEM and ODT and fixed the map direction (droplets were advected through the
inverse permutation for eddy lengths ≥ 9). Results change in **both** modes, so
pre-3.0.0 output is not comparable — but no file format moved.

**Migration for existing configs:** delete the `kolmogorov_length_scale` line.
To influence the model's smallest eddy, change `N`/`H` (which sets `6*dz`);
`actual_kolmogorov_scale` depends only on `dissipation_rate`.

## CODT v3 interface

Supported as of codt_tools **v0.5.0**, targeting CODT **2.0.0** (the
`feature/time-varying-entrainment` @ `45631ca` line). Running these inputs needs
a CODT 2.0.0 binary.

- **Parcel input (breaking).** Only `CODT_parcel_input_v3` is accepted;
  `read_parcel` rejects v1/v2 outright rather than converting, so archival
  parcel files need an older codt_tools checkout. v3 is **waypoint legs**:
  `segment_coord` holds per-leg *targets* (height or pressure per `&PARCEL
  vertical_axis`), `velocity` is signed, the leg counter (not position) keys the
  lookup, and completing the last leg **ends the run before `tmax`**. `ent_rate`
  is **1/km** (was 1/m — a 1000× physical change if a template is reused
  verbatim). New `&PARCEL`: `vertical_axis`, `pressure_mode`, `initial_height`.
- **Aerosol seeding (additive).** The conventions string is still
  `CODT_aerosol_input_v1` — CODT never bumped it, and `droplets.f90:1243` hard-
  rejects anything else, so v1 is *current*, not legacy. Seeded and unseeded
  files are both v1. Adds an optional `bin_type(bin)` and an optional
  all-or-nothing **seed group** (`seed_bin`/`seed_edge`/`seed_event` dims)
  carrying a second aerosol population with its own bins, CDF, and event
  schedule. New `&MICROPHYSICS`: `do_seeding`, `seed_hydration`,
  `seed_growth_time`. Build one via `case.aerosol.set_seed_group(...)`.
- **Version strings can't answer the questions you'll ask.** Output conventions
  stayed `CODT_output_v1`, so new output vars (`parcel_height_env`, per-leg
  `ent_rate`) must be detected by *variable presence*. Aerosol stayed v1, so
  seeding must be detected by *seed-group presence* (probe the `seed_bin`
  dimension, as CODT's own `read_seed_group` does). The registry records both
  facts per run (registry schema v3; all of this was removed in 0.9.0).
- **`do_seeding` is the sole controller of seeding — the gate is one-way.**
  `do_seeding=.true.` with no seed group aborts the run (fatal). But
  `do_seeding=.false.` with a seed group present is **fine**: CODT never reads
  the group, just warns, so one aerosol file can serve both a seeded and an
  unseeded run (the bundled `input/aerosol_input.nc` ships with a group for
  exactly this). `Case.validate()` mirrors this (raises on the fatal case,
  warns on a dormant group), and `Registry.register_run` re-checks against the
  staged file (catching a swapped shared input) and records `has_seed_group`.
- **Two gotchas that will bite a writer:**
  - `category` is dimensioned `(bin)`, **not** `(aerosol_type)` — the old CODT
    docs had this wrong; the code always read it as `(bin)`.
  - CODT's docs list dimensions in **Fortran** order (`bin, time`) while
    `ncdump`/Python see the reverse (`time, bin`). Taking those docs literally
    from Python yields a transposed CDF that reads fine and samples wrong. Trust
    `ncdump`.
- **`aerosol_type > 1` used to be silently ignored** (the reader hardcoded
  `start=[1]`, reading only row 1). All rows are now read — if anything here ever
  wrote multi-row composition tables, those rows were being dropped and now aren't.
- **Testing gotcha:** a parcel e2e run needs `tmax` long enough to actually fly
  the legs, or it silently hits the time limit mid-leg and seed events never fire
  (`Seeded: 0`) — which looks like a passing run. Distance/speed sets the floor.

## Post-v0.3.0 fixes

- `codt_tools.__version__` added (`importlib.metadata`-based).
- `budget_closure()` scale bug: was `abs(inject) or ...`, so parcel mode's near-zero-but-nonzero inject term inflated `relative_residual` (~8 for a perfectly closed budget). Now normalizes by the largest budget term.
- `budget_closure()` and `budget_totals()` report mass terms (`*_mass`, `budget_condensation`) in g/m³ (domain-mean concentration, comparable to LWC) instead of the kg stored in the NC; `budget_totals()` also converts `*_WV` terms kg/kg → g/kg, fixes the `units` attrs accordingly, and leaves K / count terms as written. `relative_residual` unchanged (dimensionless).

## codt_tools v0.3.0 — CODT v1.0.0 compatibility pass

- **config.py**: restructured `Namelist._DEFAULTS` to match CODT's namelist groups (`do_entrainment`→`&PARAMETERS`, `pressure_limit`→`&PARCEL`, new `&ENTRAINMENT` group); aligned all defaults to CODT code defaults; `radiation_method` default `'1d'`. Verified a generated namelist is accepted by the v1.0.0 binary in both modes (chamber run completes; parcel+entrainment passes namelist read).
- **simulation.py**: `load_collisions` now reads the trailing `coalesced` (i1) byte; entrainment budget vars (`budget_detrain_*`, `budget_entrain_*`, `budget_n_detrained/entrained`) added to `_FIELD_REGISTRY`. New `domain_volume()`, `budget_totals(cumulative=True)`, and `budget_closure()` (liquid-water mass closure vs LWC).
- `domain_volume = volume_scaling * domain_width**2 * H`, with hardcoded `domain_width = 0.001 m` (globals.f90). LWC is g/m³ (not kg/m³).
