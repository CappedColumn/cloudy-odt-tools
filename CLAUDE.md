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

## Runner Layout

`CODTRunner.setup_run` creates `{base}/{sim_name}/run/` with params.nml, aerosol_input.nc. Output goes to `{base}/`.

`CODTRunner` queries `codt --version` at init and stores the result in `self.codt_version` (None if binary is missing or doesn't support `--version`).

`CODTConfig.validate()` mirrors the Fortran-side checks: range validation (N, tmax, H, pres, volume_scaling, Tref, simulation_mode) raises `ValueError`; cross-namelist inconsistencies (do_radiation without do_microphysics, write_eddies without do_turbulence, do_entrainment in chamber mode) issue `warnings.warn`.

### Namelist group placement (must match what CODT reads)

`Namelist._DEFAULTS` groups map 1:1 to the Fortran namelist of the same name. A param in the wrong group makes CODT reject the file (`Invalid parameter in &GROUP`). Defaults are aligned to CODT's **code** defaults (`docs/input_parameters.md`). Mode/switch gating in `_groups_for_write`: chamber omits `turbulence_lem`/`parcel`/`entrainment`; parcel omits `turbulence_odt`/`specialeffects`; `radiation` only when `do_radiation`; `entrainment` only when `do_entrainment`.

- `do_entrainment` → `&PARAMETERS` (not `&PARCEL`)
- `pressure_limit` → `&PARCEL` (not `&PARAMETERS`)
- `ent_rate, n_blob, psigma, random_entrainment` → standalone `&ENTRAINMENT` (not `&PARCEL`)
- `radiation_method` must be `'1d'` or `'3d'` (not `'two_stream'`)

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

- **v0.6.0**: Aerosol detrainment/entrainment during blob events, standalone `entrainment.f90` module, `&ENTRAINMENT` namelist, entrainment budget variables in output NC.
- **v0.5.x**: Flat output directory (no auto-subdirectory), relative paths, removed input file copies, `{name}_DONE` marker.
- **v0.4.0**: Version embedding (`code_version`, `git_commit` in output NC), radiation module, DGM Rosenbrock solver, parcel entrainment framework, pressure_limit.
- **LEM** (PR #11): `simulation_mode='parcel'` with periodic BCs, periodic triplet map, -5/3 eddy selection.
- **Collision-coalescence** (PR #13): Event-driven 1D CC with Hall/Long/unity kernels.

## Remaining Work

- Snakemake example Snakefile

## codt_tools v0.3.0 — CODT v1.0.0 compatibility pass

- **config.py**: restructured `Namelist._DEFAULTS` to match CODT's namelist groups (`do_entrainment`→`&PARAMETERS`, `pressure_limit`→`&PARCEL`, new `&ENTRAINMENT` group); aligned all defaults to CODT code defaults; `radiation_method` default `'1d'`. Verified a generated namelist is accepted by the v1.0.0 binary in both modes (chamber run completes; parcel+entrainment passes namelist read).
- **simulation.py**: `load_collisions` now reads the trailing `coalesced` (i1) byte; entrainment budget vars (`budget_detrain_*`, `budget_entrain_*`, `budget_n_detrained/entrained`) added to `_FIELD_REGISTRY`. New `domain_volume()`, `budget_totals(cumulative=True)`, and `budget_closure()` (liquid-water mass closure vs LWC).
- `domain_volume = volume_scaling * domain_width**2 * H`, with hardcoded `domain_width = 0.001 m` (globals.f90). LWC is g/m³ (not kg/m³).
