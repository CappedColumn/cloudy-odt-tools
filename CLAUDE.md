# CODT Tools

Python framework for configuring, running, and analyzing Cloudy One-Dimensional Turbulence (CODT) simulations.

- CODT (Fortran): https://github.com/CappedColumn/cloudy-odt
- codt_tools (Python): https://github.com/CappedColumn/cloudy-odt-tools
- Deps: numpy, xarray, netCDF4, matplotlib, f90nml
- Cluster: SLURM

## CODT Interface

For I/O specifications — namelist groups, input schemas, output file formats, NetCDF variables and attributes — see [docs/file-formats.md](docs/file-formats.md).

### Executable Contract

```
./CODT <NAMELIST_PATH>       # path MUST contain '/' (use ./params.nml)
./CODT --help                # print usage and namelist group summary
./CODT --version             # print "CODT <version> (<git_commit>)"
```

- `--help` / `-h` and `--version` / `-v` exit cleanly without loading simulation modules
- `aerosol_file` resolves relative to namelist parent dir
- `output_directory` can be absolute or relative (resolved from cwd). Its parent directory must exist
- Output files go directly into `output_directory/`, prefixed with `simulation_name` (no auto-subdirectory)
- Stdout redirected to `{output_dir}/{sim_name}.log` (single log file, all init + runtime messages)
- All error messages go to stderr (`error_unit`) — visible on terminal even after stdout redirect
- `{sim_name}_DONE` file = success marker. Exit 0 = success, exit 1 = error (via `call exit()`)
- `overwrite=.false.` (default) rejects if output `.nc` exists
- Input files (namelist, aerosol NC, parcel NC) are NOT copied to output directory
- Parameter validation runs before output initialization. CODT range-checks N, tmax, H, Tref, pres, volume_scaling, write_timer, write_buffer and simulation_mode. It reports all errors at once
- Cross-namelist consistency warnings: do_radiation without do_microphysics, write_eddies without do_turbulence, do_entrainment in chamber mode

Build: `./build.sh` (injects version+git hash into `src/version.f90`, then `fpm build`) -> `./build/*/app/CODT`

### Source Architecture

- `app/main.f90` — lightweight CLI (--help, --version, arg parsing). Uses only the `version` module. The `run_codt` subroutine loads the simulation modules.
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

Global attrs on NC files: `conventions`, `code_version` (git-describe string from build time, e.g. `v1.0.0` or `v1.0.0-6-g90dabe0-dirty` — **not** bare semver), `git_commit` (short hash, `-dirty` if uncommitted). Namelist params as `PARAMETERS.N`, `MICROPHYSICS.write_trajectories`, etc. Bools as int (0/1). Mode-guarded. `check_executable` and `sim.info()` treat `code_version` as an opaque string (no semver parsing).

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

`codt_tools/registry/` is **a list of the simulations that were run** — one
SQLite file, one `runs` table, seven columns (`run_id`, `workdir`,
`created_at`, `executable`, `code_version`, `tags`, `notes`). Docs:
`docs/registry-quickstart.md`. Locate the database with `--db` or
`$CODT_REGISTRY_DB`, on home or group space, never on scratch.

```python
with Registry("~/codt_runs.db") as reg:      # created if absent
    reg.add_many(runs, tags="EXP005")
    reg.list(tag="EXP005")
```

`Registry` (`registry/store.py`): `add(run)`, `add_many(runs)`,
`add_directory(workdir)`, `list(tag=, since=)`, `get(run_id)`,
`remove(run_id)`, `len()`. `add` takes a **`Run`** and reads name, workdir and
executable off it. It is `INSERT OR REPLACE`, so re-staging an ensemble is not
an error. `codt_tools.run.codt_version()` captures `code_version`
automatically, and caches it **once per executable**, not per run. `list()`
returns plain `list[dict]`, newest first.

CLI: `codt-registry list [--tag] [--since] | show <id> | add <dir> | remove <id>`.
No `init` — opening creates.

**Deliberately absent, and not to be re-added without asking:**

- **No status of any kind.** No lifecycle, no `status_events`, no exit codes.
  "Did it finish?" is `run.is_complete` (the `_DONE` marker), the only thing
  ever actually true. Generated launch scripts contain no registry calls, so
  nothing *could* report from inside a job.
- **No experiments table.** Grouping is the free-text `tags` column, matched
  as a substring. Hypothesis/conclusion belong in a README in the experiment
  directory.
- **No namelist parameters, input checksums or executable archiving.** A run's
  configuration lives in its own staged `inputs/`.
- **No migrations.** Opening a database stamps `PRAGMA user_version=1`.
  Nothing ever reads it back.

**Nothing in the package imports the registry** — pinned by
`tests/registry/test_store.py::TestOptional`, which checks in a fresh
interpreter that importing `codt_tools`, `codt_tools.run` and
`codt_tools.simulation` leaves `sys.modules` free of `codt_tools.registry`.

## Case Layer

`codt_tools/case/` holds `Case`, `Namelist`, `Aerosol` (attribute
`case.aerosol`), `Parcel`, and `case/validate.py` (`validate_case`,
`lem_turbulence_scales`, `initial_launch_level`). `case/aerosol.py` and
`case/parcel.py` carry the aerosol and parcel file readers and writers, and
those functions are public.

Set namelist parameters with `case.set(**namelist_params)`. Set the components
with `case.aerosol.set(...)` and `case.parcel.set(...)`. Assigning any other
attribute **raises** instead of silently creating a dead one.

### Sweeps: designs and points

`codt_tools/case/mutate.py`. A **point** is a dict of `path -> value`
(`params.*`, `aerosol.*`, `parcel.*`, or a bare component name to swap the whole
object). A **design** is a list of points. `case.apply(point)` applies one in
place, as the multi-component sibling of `case.set()`. `case.sweep(design)`
returns one independent deep copy per point. `cross(*axes)` is the only
combinatorial helper. An axis is either a list of points, or a dict of
`path -> values` that expands to independent axes. A list of points needs no
special syntax for values that vary *together*, such as LHS samples and paired
parameters. `cross` **raises when two crossed axes set the same path**. Use `+`
to union two designs. A callable value transforms the current value instead of
replacing it. Everything else — filtering, control groups, derived values — is
plain Python on a list of dicts. There is no sweep DSL, deliberately.
`docs/designs.md` holds the recipes for conditional axes, ragged branches,
replicates, filtering and post-hoc derivation.

Run names default to `{base_name}_{index}`, zero-padded.
`name=(index, point) -> str` overrides. **The case layer does not own the
index → parameters mapping.** Each run's staged `inputs/` holds its own
configuration. The registry does not hold the mapping either. Write a design
manifest yourself if you want to read the design as a table.

### Path ownership — where CODT reads and writes

`aerosol_file`, `parcel_file` and `mie_data_file` resolve against the
namelist's parent directory. `output_directory` resolves against the process's
working directory. `docs/file-formats.md` has the table and the consequences.

The three input paths are **staging-owned**. A `Case` carries none of them,
`case.set()` raises on all three, and
`Case.write_inputs(directory, output_directory=None)` is the only place that
assigns them. Source references: `src/droplets.f90:996`, `src/parcel.f90:119`,
`src/radiation.f90:670`, `src/initialize.f90:246`, `app/main.f90:43`, and
`resolve_path` at `globals.f90:417`.

One invariant makes bare filenames safe. codt_tools always invokes the model
with an **absolute** namelist path (`runner.run_local`, both sbatch bodies).
`tests/run/test_run.py::TestExecuteLocal::test_invokes_with_an_absolute_namelist_path`
tests it.

## Run Layer

`codt_tools/run/` holds the run layer. **Nothing in the package submits a
SLURM job.** No Python code calls `sbatch`, `squeue` or `sacct`. Batch
execution is *artifact generation*. codt_tools writes a script. You launch it.

`Run(case, executable, workdir)` — no account, partition, qos or registry.

| member | does |
|---|---|
| `stage()` | calls `case.write_inputs(workdir/"inputs", output_directory=workdir/"output")`. Returns the staged `Namelist`. **Does not modify the case** — one case stages any number of runs. |
| `execute_local(**kw)` | runs the binary on the **absolute** `inputs/params.nml`, blocking. Returns `CompletedProcess`. Warns with a plain-language cause when the binary dies on a signal (SIGILL = built for another CPU). |
| `open_simulation()` | delegates to `Simulation.from_run(self)`. Raises unless `is_complete` |
| `is_complete` / `is_staged` | the `{case.name}_DONE` marker / the namelist exists |
| `name`, `inputs_dir`, `output_dir`, `namelist_path`, `done_marker` | derived from `workdir`. `name` is the *directory* name. `done_marker` uses `case.name` (they need not match) |
| `Run.for_cases(cases, exe, base_dir)` | one run per case at `base_dir/{case.name}`. **Raises on duplicate names** |

`__init__` resolves `executable` and `workdir` absolute. That is what makes
bare staged filenames resolve, because CODT takes `argv[1]`'s parent *as
typed*.
`tests/run/test_run.py::TestExecuteLocal::test_invokes_with_an_absolute_namelist_path`
pins it.

`check_executable(exe) -> str | None` gives a one-line reason a binary will not
run (missing, not executable, missing runtime lib, SIGILL, or no `--version`
support), or None. It judges the binary **on the host it runs on**, so
it cannot catch an arch mismatch that only appears on a compute node. The
generated scripts run `"$EXE" --version` in situ for that.

### Launch scripts (`run/launcher.py`)

`write_local(runs, path, jobs=N)` and `write_slurm_array(runs, path,
runs_per_task=1, module=None, account=..., partition=..., qos=...,
constraint=..., time=..., mem=..., array_throttle=..., requeue=True,
**directives)`. Both take `list[Run]`, return the script path, and **execute
nothing**. Each script uses a per-run subshell, a `_DONE` skip guard
(`compgen -G "$RUN_DIR/output/*_DONE"`), `wait`, and a timing line per run
(`run, wall seconds, exit code` → `timing.txt`).

- The `_DONE` guard makes both scripts **idempotent**. A second submission
  resumes. That is what makes `--requeue` safe, so the two ship together by
  default. The guard works at *run* granularity only. CODT still has no
  checkpoint/restart, so an interrupted individual run restarts.
- `write_slurm_array` also writes `{stem}_runs.txt`, one whitespace-separated
  line of run dirs per array task, read via `$SLURM_ARRAY_TASK_ID`.
- Omitted `account`/`partition`/`time` render as `<ACCOUNT>`/`<PARTITION>`/
  `<TIME>` placeholders — inert in shell, refused by SLURM until filled.
- Both **raise** if the batch mixes executables (a script names one `$EXE`)
  or is empty.
- Two tests enforce the no-submission rule: `launcher.py` may not import
  `subprocess`/`os`/`Popen`, and generating both scripts with `subprocess.run`
  monkeypatched to raise must succeed.

codt_tools does no cluster or architecture detection. It passes `account`,
`partition`, `qos`, `constraint` and `module` through as opaque strings.
`docs/running-on-slurm.md` covers task sizing and preemption.

`Run` has no registry coupling. Generated
scripts contain no `codt-registry` text. You record a run with an explicit
call.

`Case.validate()` (`case/validate.py`) mirrors the Fortran-side checks. Range validation (N, tmax, H, pres, volume_scaling, Tref, simulation_mode) raises `ValueError`. Cross-namelist inconsistencies (do_radiation without do_microphysics, write_eddies without do_turbulence, do_entrainment in chamber mode) issue `warnings.warn`.

## Simulation Layer

`codt_tools/simulation/` holds the analysis layer:

| module | contents |
|---|---|
| `simulation.py` | `Simulation` — one class |
| `plotting.py` | the `plot_*` helpers, plus the public `get_label` and `ensure_ax` |
| `trajectory.py` | the particle trajectory reader |

`Simulation.from_run(run)` opens a completed run's output. It raises if the run
has no `_DONE` marker. `Run.open_simulation()` delegates to it, so the two
cannot drift. `simulation.py` imports `Run` under `TYPE_CHECKING` only, and
`open_simulation` imports `Simulation` inside the method. Staging and launching
a run should not pay for xarray and matplotlib.

**Analysis does not import the registry.** Reading your own output must not
pull in the bookkeeping layer.
`tests/simulation/test_layer.py::TestNoRegistryDependency` pins that. It
asserts in a *fresh interpreter* that neither `import codt_tools` nor
`import codt_tools.simulation` leaves anything under `codt_tools.registry` in
`sys.modules`.

### Conventions — each reader owns its format string

There is deliberately **no registry of "supported" conventions** to keep in
sync, and no top-level `conventions.py`. Each reader names the one format
string it accepts.

| Reader | Constant | On mismatch |
|---|---|---|
| `simulation/simulation.py` | `OUTPUT_CONVENTIONS = "CODT_output_v1"` | **warns**, opens anyway |
| `case/aerosol.py` | `_CONVENTIONS = "CODT_aerosol_input_v1"` | **raises** |
| `case/parcel.py` | `CONVENTIONS = "CODT_parcel_input_v3"` | **raises** |

The asymmetry is the point. Output already exists, and a scientist must be able
to look at it, so a mismatch only warns and names expected versus found. A
wrong-format *input* silently produces a wrong simulation, so that raises.

`"CODT_output_v1"` exists in exactly one place, so the rule above is literally
true. There is no "supported conventions" set anywhere.

### Namelist group placement

Each `Namelist` group maps 1:1 to the Fortran namelist of the same name. A
param in the wrong group makes CODT reject the file
(`Invalid parameter in &GROUP`). `Namelist.groups_for_write()` gates by mode
and switch. `docs/file-formats.md` lists the placements that catch people out.

`{name}_DONE` discovery: `Run.done_marker` and `Simulation._discover_files`
both correctly use `{name}_DONE`. The generated launch scripts glob `*_DONE`
instead, because bash does not know `simulation_name`.

## Binary File Readers

`docs/file-formats.md` holds the eddy and collision binary layouts, with the
numpy dtypes for both.

## Remaining Work

- Snakemake example Snakefile.
- A design manifest (`run_id` -> parameters) has no first-class home. The
  habit documented in `docs/designs.md` ("Recording the design") is the
  current answer.

## Settings verified against real runs

**`examples/` is not part of this repository** and never has been. The
workshop tutorials are maintained separately and do not track this repo. These
settings are kept here because real runs verified them, and they are the
quickest way to get a demonstration case that behaves.

- Chamber `tmax=600` (~45 s wall). Parcel `tmax=240` (~4.5 min wall).
- Parcel cases need small CCN (`edge_radii=[30,60,120]` nm) and
  `initial_rh=0.98`. The 291–1000 nm chamber aerosol never passes its Köhler
  critical radius, so `Nact` stays 0.
- Entrainment needs `ent_rate≈0.002` m⁻¹ with a humid sounding
  (`RH=[0.95,0.92,0.88]`). `ent_rate=2.0` (the code default) evaporates the
  entire cloud, and so does a dry sounding.
- Entrainment event spacing is `dt = (psigma/(1−psigma))/(ent_rate·|vel|)`. It
  is **independent of `n_blob`**.

## History

[CHANGELOG.md](CHANGELOG.md) holds the version history and the CODT interface
changes that codt_tools tracks. Read it before you trust output from an
archived run, because several entries carry facts you need to parse an older
file correctly.
