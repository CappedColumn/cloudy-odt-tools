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
   - Parcel: integral_length_scale, kolmogorov_length_scale, dissipation_rate (3 values)

Per-eddy record: M(i4), L(i4), time(f8). Raw grid indices for replay via `implement_eddy(L, M)`.

```python
# Read header
mode_flag = np.fromfile(f, dtype='<i1', count=1)[0]
N, H = np.fromfile(f, dtype=[('N','<i4'),('H','<f8')], count=1)[0]
if mode_flag == 0:  # chamber
    hdr = np.fromfile(f, dtype='<f8', count=4)  # C2, ZC2, Tdiff, Tref
else:  # parcel
    hdr = np.fromfile(f, dtype='<f8', count=3)  # L_int, eta, epsilon
dt_eddy = np.dtype([('M','<i4'),('L','<i4'),('time','<f8')])
```

### Collision Binary (`{name}_collisions.bin`)

Unformatted Fortran stream. Header: N(i4), H(f8), domain_width(f8), volume_scaling(f8). Per-event: id_keep(i4), id_kill(i4), r_keep(f8), r_kill(f8), r_after(f8), position(f8), time(f8), coalesced(i1). `load_collisions` reads the trailing `coalesced` (i1) flag (1 = merged, 0 = bounce; 49-byte packed record).

```python
dt_header = np.dtype([('N','<i4'),('H','<f8'),('domain_width','<f8'),('volume_scaling','<f8')])
dt_record = np.dtype([('id_keep','<i4'),('id_kill','<i4'),('r_keep','<f8'),('r_kill','<f8'),('r_after','<f8'),('position','<f8'),('time','<f8'),('coalesced','<i1')])
```

## Merged

- **CODT v3 interface** (codt_tools v0.5.0, for CODT 2.0.0): waypoint-leg parcel input, aerosol seeding, registry input-schema gating. See "CODT v3 interface" below.
- **v0.6.0**: Aerosol detrainment/entrainment during blob events, standalone `entrainment.f90` module, `&ENTRAINMENT` namelist, entrainment budget variables in output NC.
- **v0.5.x**: Flat output directory (no auto-subdirectory), relative paths, removed input file copies, `{name}_DONE` marker.
- **v0.4.0**: Version embedding (`code_version`, `git_commit` in output NC), radiation module, DGM Rosenbrock solver, parcel entrainment framework, pressure_limit.
- **LEM** (PR #11): `simulation_mode='parcel'` with periodic BCs, periodic triplet map, -5/3 eddy selection.
- **Collision-coalescence** (PR #13): Event-driven 1D CC with Hall/Long/unity kernels.

## Remaining Work

- Snakemake example Snakefile

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

Four step-by-step tutorials (chamber simple/advanced, parcel simple/advanced) for interactive-Python + manual `CODT ./params.nml` workflows, plus `templates/{chamber,parcel}_template.py` end-to-end scripts and a pandoc site build (`site/build_site.sh` → `site/html/`, gitignored; publish by copying to `~/public_html/codt_workshop/`). Workshop conda env: `cloudy-odt`. Physics-verified settings: chamber `tmax=600` (~45 s wall), parcel `tmax=240` (~4.5 min wall); parcel cases need small CCN (`edge_radii=[30,60,120]` nm — the 291–1000 nm chamber aerosol never passes its Köhler critical radius, so `Nact` stays 0) and `initial_rh=0.98`; entrainment needs `ent_rate≈0.002` m⁻¹ with a humid sounding (`RH=[0.95,0.92,0.88]`) — `ent_rate=2.0` (the code default) or a dry sounding evaporates the entire cloud. Entrainment event spacing: `dt = (n_blob/ent_rate)·(psigma/(1−psigma))/|vel|`.

## Post-v0.3.0 fixes

- `codt_tools.__version__` added (`importlib.metadata`-based).
- `budget_closure()` scale bug: was `abs(inject) or ...`, so parcel mode's near-zero-but-nonzero inject term inflated `relative_residual` (~8 for a perfectly closed budget). Now normalizes by the largest budget term.
- `budget_closure()` and `budget_totals()` report mass terms (`*_mass`, `budget_condensation`) in g/m³ (domain-mean concentration, comparable to LWC) instead of the kg stored in the NC; `budget_totals()` also converts `*_WV` terms kg/kg → g/kg, fixes the `units` attrs accordingly, and leaves K / count terms as written. `relative_residual` unchanged (dimensionless).

## codt_tools v0.3.0 — CODT v1.0.0 compatibility pass

- **config.py**: restructured `Namelist._DEFAULTS` to match CODT's namelist groups (`do_entrainment`→`&PARAMETERS`, `pressure_limit`→`&PARCEL`, new `&ENTRAINMENT` group); aligned all defaults to CODT code defaults; `radiation_method` default `'1d'`. Verified a generated namelist is accepted by the v1.0.0 binary in both modes (chamber run completes; parcel+entrainment passes namelist read).
- **simulation.py**: `load_collisions` now reads the trailing `coalesced` (i1) byte; entrainment budget vars (`budget_detrain_*`, `budget_entrain_*`, `budget_n_detrained/entrained`) added to `_FIELD_REGISTRY`. New `domain_volume()`, `budget_totals(cumulative=True)`, and `budget_closure()` (liquid-water mass closure vs LWC).
- `domain_volume = volume_scaling * domain_width**2 * H`, with hardcoded `domain_width = 0.001 m` (globals.f90). LWC is g/m³ (not kg/m³).
