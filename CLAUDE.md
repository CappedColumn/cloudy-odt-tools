# CODT Tools

Python framework for configuring, running, and analyzing Cloudy One-Dimensional Turbulence (CODT) simulations.

- CODT (Fortran): https://github.com/CappedColumn/cloudy-odt
- codt_tools (Python): https://github.com/CappedColumn/cloudy-odt-tools
- Deps: numpy, xarray, netCDF4, matplotlib, f90nml
- Cluster: SLURM

## CODT Interface

For full I/O specifications (namelist groups, input schemas, output file formats, NetCDF variables/attributes), see the `codt-io` skill. For versioning and format compatibility, see the `codt-versioning` skill.

### Executable Contract

```
./CODT <NAMELIST_PATH>       # path MUST contain '/' (use ./params.nml)
./CODT --help                # print usage and namelist group summary
./CODT --version             # print "CODT <version> (<git_commit>)"
```

- `--help` / `-h` and `--version` / `-v` exit cleanly without loading simulation modules
- `aerosol_file` resolves relative to namelist parent dir
- `output_directory` can be absolute or relative (resolved from cwd); parent directory must exist
- Output files go directly into `output_directory/`, prefixed with `simulation_name` (no auto-subdirectory)
- Stdout redirected to `{output_dir}/{sim_name}.log` (single log file, all init + runtime messages)
- All error messages go to stderr (`error_unit`) — visible on terminal even after stdout redirect
- `{sim_name}_DONE` file = success marker; exit 0 = success, 1 = error (via `call exit()`)
- `overwrite=.false.` (default) rejects if output `.nc` exists
- Input files (namelist, aerosol NC, parcel NC) are NOT copied to output directory
- Parameter validation runs before output initialization: N, tmax, H, Tref, pres, volume_scaling, write_timer, write_buffer, simulation_mode are range-checked; all errors reported at once
- Cross-namelist consistency warnings: do_radiation without do_microphysics, write_eddies without do_turbulence, do_entrainment in chamber mode

Build: `./build.sh` (injects version+git hash into `src/version.f90`, then `fpm build`) -> `./build/*/app/CODT`

### Source Architecture

- `app/main.f90` — lightweight CLI (--help, --version, arg parsing). Only uses `version` module; simulation modules loaded in `run_codt` subroutine.
- `src/version.f90` — `code_version` and `git_commit` strings (injected by `build.sh`)
- `src/CODT.f90` — `run_simulation()` subroutine with the time loop and all physics modules

### Output Files (`{output_directory}/`)

| File | Description |
|------|-------------|
| `{name}.nc` | Main output (profiles + time series, schema `CODT_output_v1`) |
| `{name}.log` | Redirected stdout (all initialization and runtime messages) |
| `{name}_particles.nc` | Trajectories (CF ragged array, schema `CODT_particle_output_v1`, if enabled) |
| `{name}_collisions.bin` | Collision events (unformatted stream, if enabled) |
| `{name}_eddies.bin` | Eddy events (unformatted stream, if enabled) |
| `{name}_DONE` | Completion marker with timestamp |

Global attrs on NC files: `conventions`, `code_version` (git-describe string from build time, e.g. `v1.0.0` or `v1.0.0-6-g90dabe0-dirty` — **not** bare semver), `git_commit` (short hash, `-dirty` if uncommitted). Namelist params as `PARAMETERS.N`, `MICROPHYSICS.write_trajectories`, etc. Bools as int (0/1). Mode-guarded. `CODTRunner._query_version` and `sim.info()` treat `code_version` as an opaque string (no semver parsing).

### Chamber vs. Parcel Mode

| Aspect | Chamber | Parcel |
|--------|---------|--------|
| **Turbulence** | ODT (`&TURBULENCE_ODT`) | LEM (`&TURBULENCE_LEM`) |
| **Boundary conditions** | Dirichlet | Periodic |
| **Scalar representation** | Dual: nondimensional + dimensional | Dimensional only |
| **Time step** | Adaptive | Fixed |
| **Forcing** | Temperature gradient (`Tdiff`) | Adiabatic ascent (`parcel_file`) |
| **Particle init** | Injected over time (aerosol NC schedule) | Pre-loaded at init (`aerosol_concentration`) + Köhler equilibration |
| **Particle fallout** | Gravitational removal | Periodic wrapping |
| **`Tref`** | Bottom boundary T (°C→K) | Uniform initial T (°C→K) |
| **`pres`** | Constant reference pressure | Initial pressure, evolves hydrostatically |
| **`initial_RH`** | Ignored | Sets initial WV field (0–1) |
| **`aerosol_concentration`** | Ignored | Number concentration (cm⁻³) |
| **`parcel_file`** | Not used | Required for ascent rate |
| **`Tdiff`** | Top-bottom ΔT | Not used |
| **Entrainment** | Not yet implemented (planned) | Blob method with aerosol detrainment/entrainment (`do_entrainment` in `&PARAMETERS`, params in `&ENTRAINMENT`) |
| **`pressure_limit`** | Not used | Stop simulation at target pressure (Pa) |
| **Seeding** | Events keyed to **time** [s] | Events keyed to the **vertical coordinate** (m or Pa per `vertical_axis`). Same driver, same `seed_coord` variable — only the axis differs |

## Simulation Registry

`codt_tools/registry/` tracks experiments and runs in SQLite (docs in `docs/registry-*.md`). DB location convention: `--db` or `$CODT_REGISTRY_DB`; the DB lives on home/group space, never scratch. All access goes through `Registry` / the `codt-registry` CLI (pragmas, retry, txn coupling live in Python — never raw sqlite3 writes). Gate rule: `SUPPORTED_CONVENTIONS` (registry/versions.py) is the single source of truth for readable output conventions; `CODTSimulation.__init__` and `record_completion` warn on mismatch (`strict=True` raises). `SUPPORTED_INPUT_CONVENTIONS` is the input-side equivalent (parcel v3 only, aerosol v1) — `register_run` records each NetCDF input's conventions plus the aerosol seed-group flag, and cross-checks `do_seeding`. Bump both via the codt-versioning skill when formats change. Experiments: `ExperimentSpec` YAML → `create_experiment_runs` → `{data_root}/{experiment_id}/{experiment.yaml, shared_inputs/, runs/{run_id}/}` with content-hash dedup of shared inputs as relative symlinks; run_id = `{stamp}_{model}_{descriptor}`.

## Runner Layout

`CODTRunner.setup_run` creates `{base}/{run_name}/` with `inputs/` (params.nml, aerosol_input.nc) and an empty `output/` (model writes here). The namelist `output_directory` is set to the absolute `output/` path; `executable` and `base_output_dir` are resolved to absolute in `__init__`, so local runs and generated sbatch scripts are cwd-independent. `aerosol_file` stays relative (`aerosol_input.nc`) and CODT resolves it against the namelist's parent (`inputs/`).

`CODTRunner` queries `codt --version` at init and stores the result in `self.codt_version` (None if binary is missing or doesn't support `--version`).

`CODTConfig.validate()` mirrors the Fortran-side checks: range validation (N, tmax, H, pres, volume_scaling, Tref, simulation_mode) raises `ValueError`; cross-namelist inconsistencies (do_radiation without do_microphysics, write_eddies without do_turbulence, do_entrainment in chamber mode) issue `warnings.warn`.

### Namelist group placement (must match what CODT reads)

`Namelist._DEFAULTS` groups map 1:1 to the Fortran namelist of the same name. A param in the wrong group makes CODT reject the file (`Invalid parameter in &GROUP`). Defaults are aligned to CODT's **code** defaults (`docs/input_parameters.md`). Mode/switch gating in `_groups_for_write`: chamber omits `turbulence_lem`/`parcel`/`entrainment`; parcel omits `turbulence_odt`/`specialeffects`; `radiation` only when `do_radiation`; `entrainment` only when `do_entrainment`.

- `do_entrainment` → `&PARAMETERS` (not `&PARCEL`)
- `pressure_limit` → `&PARCEL` (not `&PARAMETERS`)
- `ent_rate, n_blob, psigma, random_entrainment` → standalone `&ENTRAINMENT` (not `&PARCEL`)
- `radiation_method` must be `'1d'` or `'3d'` (not `'two_stream'`)
- `do_seeding, seed_hydration, seed_growth_time` → `&MICROPHYSICS` (not a `&SEEDING` group — there isn't one)

`{name}_DONE` discovery: `collect()` (runner.py) and `CODTSimulation._discover_files` both correctly use `{name}_DONE`.

## Binary File Readers

### Eddy Binary (`{name}_eddies.bin`)

Unformatted Fortran stream. Mode-aware header:
1. `mode_flag` (i1): 0 = chamber, 1 = parcel
2. `N` (i4), `H` (f8) — shared fields
3. Mode-specific fields (f8 array):
   - Chamber: C2, ZC2, Tdiff, Tref (4 values)
   - Parcel: integral_length_scale, smallest_eddy_scale, dissipation_rate (3 values)
     - Field 1 was `kolmogorov_length_scale` before CODT 3.0.0. Same
       count/order/dtype, different quantity, so old files read without error
       under the new label. `load_eddies` uses the new label unconditionally —
       check `code_version` when reading archived runs.

Per-eddy record: M(i4), L(i4), time(f8). Raw grid indices for replay via `implement_eddy(L, M)`.

```python
# Read header
mode_flag = np.fromfile(f, dtype='<i1', count=1)[0]
N, H = np.fromfile(f, dtype=[('N','<i4'),('H','<f8')], count=1)[0]
if mode_flag == 0:  # chamber
    hdr = np.fromfile(f, dtype='<f8', count=4)  # C2, ZC2, Tdiff, Tref
else:  # parcel
    hdr = np.fromfile(f, dtype='<f8', count=3)  # L_int, smallest_eddy_scale, epsilon
dt_eddy = np.dtype([('M','<i4'),('L','<i4'),('time','<f8')])
```

### Collision Binary (`{name}_collisions.bin`)

Unformatted Fortran stream. Header: N(i4), H(f8), domain_width(f8), volume_scaling(f8). Per-event: id_keep(i4), id_kill(i4), r_keep(f8), r_kill(f8), r_after(f8), position(f8), time(f8), coalesced(i1). `load_collisions` reads the trailing `coalesced` (i1) flag (1 = merged, 0 = bounce; 49-byte packed record).

> **`time` changed meaning in CODT 3.0.1 (`c67809c`) — layout unchanged.** Through
> 3.0.0 this field held the event's time *within* the current collision-coalescence
> window (0 → `delta_time`), not absolute time: values were tiny (order 1e-8–1e-2 s),
> reset every window, so the stream was **non-monotonic**. From 3.0.1 it is absolute
> simulation time, on the same axis as the output NC `time` coordinate. The field was
> always `f8`, so the 49-byte record is **identical** and readers parse both eras
> without error — only the values differ. Detect by the run's `code_version` /
> `git_commit`, or by testing whether any `time` exceeds one `delta_time`.
>
> For pre-3.0.1 files, absolute time is recoverable by partitioning events into write
> intervals using the cumulative `N_collisions` from the output NC (the binary is
> written in event order): the k-th block of `N_collisions[k]` events belongs to
> interval k. Binary totals match `N_collisions`/`N_coalescences` exactly in both
> eras, so that is a safe cross-check.

```python
dt_header = np.dtype([('N','<i4'),('H','<f8'),('domain_width','<f8'),('volume_scaling','<f8')])
dt_record = np.dtype([('id_keep','<i4'),('id_kill','<i4'),('r_keep','<f8'),('r_kill','<f8'),('r_after','<f8'),('position','<f8'),('time','<f8'),('coalesced','<i1')])
```

## Merged

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

## Remaining Work

- Snakemake example Snakefile
- `test_runs/workshop_verify/setup_cases.py` still calls the pre-v3 parcel API
  (`set_parcel(time=...)`); unrelated to the LEM change, but it will not run.

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
CODT at run time — existing callers keep working. `CODTConfig.write` supplies it
automatically. `CODTConfig.validate()` mirrors CODT's startup abort for the
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
- `CODTConfig.validate()` mirrors the new startup aborts: parcel raises on
  `smallest_eddy_gridpoints > N` and `smallest_eddy_scale >= integral_length_scale`
  and warns when the scale separation is < 3 (**this warning fires on the
  defaults**, matching CODT on its bundled `params.nml`); chamber raises when
  `lmin < 6` or `lmin % 3 != 0` (new `src/ODT.f90` guard).
- `load_eddies()` renames parcel header field 1 to `smallest_eddy_scale`
  **unconditionally**. Field count/order/dtype are unchanged (3 × f8), so a
  pre-3.0.0 file still reads without error, but the value under that key is the
  old namelist input — check `code_version`.
- `CODTSimulation.lem_scales()` returns whichever `LEM.*` attributes the file
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
  `seed_growth_time`. Build one via `cfg.injection.set_seed_group(...)`.
- **Version strings can't answer the questions you'll ask.** Output conventions
  stayed `CODT_output_v1`, so new output vars (`parcel_height_env`, per-leg
  `ent_rate`) must be detected by *variable presence*. Aerosol stayed v1, so
  seeding must be detected by *seed-group presence* (probe the `seed_bin`
  dimension, as CODT's own `read_seed_group` does). The registry records both
  facts per run — see `docs/registry-schema.md`.
- **`do_seeding` is the sole controller of seeding — the gate is one-way.**
  `do_seeding=.true.` with no seed group aborts the run (fatal). But
  `do_seeding=.false.` with a seed group present is **fine**: CODT never reads
  the group, just warns, so one aerosol file can serve both a seeded and an
  unseeded run (the bundled `input/aerosol_input.nc` ships with a group for
  exactly this). `CODTConfig.validate()` mirrors this (raises on the fatal case,
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

## Workshop materials (`examples/`)

Four step-by-step tutorials (chamber simple/advanced, parcel simple/advanced) for interactive-Python + manual `CODT ./params.nml` workflows, plus `templates/{chamber,parcel}_template.py` end-to-end scripts and a pandoc site build (`site/build_site.sh` → `site/html/`, gitignored; publish by copying to `~/public_html/codt_workshop/`). Workshop conda env: `cloudy-odt`. Physics-verified settings: chamber `tmax=600` (~45 s wall), parcel `tmax=240` (~4.5 min wall); parcel cases need small CCN (`edge_radii=[30,60,120]` nm — the 291–1000 nm chamber aerosol never passes its Köhler critical radius, so `Nact` stays 0) and `initial_rh=0.98`; entrainment needs `ent_rate≈0.002` m⁻¹ with a humid sounding (`RH=[0.95,0.92,0.88]`) — `ent_rate=2.0` (the code default) or a dry sounding evaporates the entire cloud. Entrainment event spacing: `dt = (psigma/(1−psigma))/(ent_rate·|vel|)` — **independent of `n_blob`** from CODT 3.1.0 (it carried an `n_blob` factor through 3.0.1; see "CODT 3.1.0 psigma/n_blob redefinition" below).

## Post-v0.3.0 fixes

- `codt_tools.__version__` added (`importlib.metadata`-based).
- `budget_closure()` scale bug: was `abs(inject) or ...`, so parcel mode's near-zero-but-nonzero inject term inflated `relative_residual` (~8 for a perfectly closed budget). Now normalizes by the largest budget term.
- `budget_closure()` and `budget_totals()` report mass terms (`*_mass`, `budget_condensation`) in g/m³ (domain-mean concentration, comparable to LWC) instead of the kg stored in the NC; `budget_totals()` also converts `*_WV` terms kg/kg → g/kg, fixes the `units` attrs accordingly, and leaves K / count terms as written. `relative_residual` unchanged (dimensionless).

## codt_tools v0.3.0 — CODT v1.0.0 compatibility pass

- **config.py**: restructured `Namelist._DEFAULTS` to match CODT's namelist groups (`do_entrainment`→`&PARAMETERS`, `pressure_limit`→`&PARCEL`, new `&ENTRAINMENT` group); aligned all defaults to CODT code defaults; `radiation_method` default `'1d'`. Verified a generated namelist is accepted by the v1.0.0 binary in both modes (chamber run completes; parcel+entrainment passes namelist read).
- **simulation.py**: `load_collisions` now reads the trailing `coalesced` (i1) byte; entrainment budget vars (`budget_detrain_*`, `budget_entrain_*`, `budget_n_detrained/entrained`) added to `_FIELD_REGISTRY`. New `domain_volume()`, `budget_totals(cumulative=True)`, and `budget_closure()` (liquid-water mass closure vs LWC).
- `domain_volume = volume_scaling * domain_width**2 * H`, with hardcoded `domain_width = 0.001 m` (globals.f90). LWC is g/m³ (not kg/m³).
