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

## Simulation Registry (rewritten in 0.9.0)

`codt_tools/registry/` is **a list of the simulations that were run** — one
SQLite file, one `runs` table, seven columns (`run_id`, `workdir`,
`created_at`, `executable`, `code_version`, `tags`, `notes`). Docs:
`docs/registry-quickstart.md`. DB location convention unchanged: `--db` or
`$CODT_REGISTRY_DB`, on home/group space, never scratch.

```python
with Registry("~/codt_runs.db") as reg:      # created if absent
    reg.add_many(runs, tags="EXP005")
    reg.list(tag="EXP005")
```

`Registry` (`registry/store.py`): `add(run)`, `add_many(runs)`,
`add_directory(workdir)`, `list(tag=, since=)`, `get(run_id)`,
`remove(run_id)`, `len()`. `add` takes a **`Run`** and reads name/workdir/
executable off it; it is `INSERT OR REPLACE`, so re-staging an ensemble is not
an error. `code_version` is captured automatically via
`codt_tools.run.codt_version()` and cached **once per executable**, not per
run. `list()` returns plain `list[dict]`, newest first.

CLI: `codt-registry list [--tag] [--since] | show <id> | add <dir> | remove <id>`.
No `init` — opening creates.

**Deliberately absent, and not to be re-added without asking:**

- **No status of any kind.** No lifecycle, no `status_events`, no exit codes.
  "Did it finish?" is `run.is_complete` (the `_DONE` marker), the only thing
  ever actually true. Generated launch scripts contain no registry calls
  (Stage 3), so nothing *could* report from inside a job.
- **No experiments table.** Grouping is the free-text `tags` column, matched
  as a substring. Hypothesis/conclusion belong in a README in the experiment
  directory.
- **No namelist parameters, input checksums or executable archiving.** A run's
  configuration lives in its own staged `inputs/`.
- **No migrations.** `PRAGMA user_version=1` is stamped at creation and never
  read.

**Nothing in the package imports the registry** — pinned by
`tests/registry/test_store.py::TestOptional`, which checks in a fresh
interpreter that importing `codt_tools`, `codt_tools.run` and
`codt_tools.simulation` leaves `sys.modules` free of `codt_tools.registry`.

Deleted in 0.9.0: `registry/api.py` (720 lines), `registry/db.py` (6 tables,
4 migrations), `registry/versions.py`, `docs/registry-schema.md`,
`docs/using-a-shared-registry.md`. **Deleting `versions.py` resolved the Stage
4 duplication** — `"CODT_output_v1"` now exists in exactly one place,
`simulation/simulation.py`, so "each reader owns its format string" is
literally true. Pre-0.9.0 databases are **not migrated and not read**; they
stay queryable from the pinned `codt08` environment.

## Case Layer (0.9.0)

`codt_tools/case/` replaces `config.py`: `Case` (was `CODTConfig`), `Namelist`,
`Aerosol` (was `InjectionData`, attribute `case.aerosol`), `Parcel` (was
`ParcelInput`), plus `case/validate.py` (`validate_case`,
`lem_turbulence_scales`, `initial_launch_level`). The old `aerosol_io.py` /
`parcel_io.py` are folded into `case/aerosol.py` / `case/parcel.py`, their
functions still public. No compatibility aliases — 0.9.0 is a clean break; code
pinned to the old names uses the `v0.8.0` tag (`codt08` env).

Removed with the rename: `set_injection` / `set_bins` / `set_parcel` (use
`case.aerosol.set(...)` / `case.parcel.set(...)`) and the `__getattr__` /
`__setattr__` dot-proxy. `case.set(**namelist_params)` stays; assigning any
other attribute now **raises** instead of silently creating a dead one.

### Sweeps: designs and points (0.9.0)

`codt_tools/case/mutate.py`. A **point** is a dict of `path -> value`
(`params.*`, `aerosol.*`, `parcel.*`, or a bare component name to swap the whole
object); a **design** is a list of points. `case.apply(point)` applies one in
place — the multi-component sibling of `case.set()`; `case.sweep(design)`
returns one independent deep copy per point. `cross(*axes)` is the only
combinatorial helper: an axis is a list of points (so values that vary
*together* — LHS samples, paired parameters — need no special syntax) or a dict
of `path -> values` (expanded to independent axes). It **raises when two crossed
axes set the same path**; `+` is how two designs are unioned. A callable value
transforms the current value instead of replacing it. Everything else —
filtering, control groups, derived values — is plain Python on a list of dicts.
No sweep DSL, deliberately. Recipes for conditional axes, ragged branches,
replicates, filtering and post-hoc derivation: `docs/designs.md`.

Run names default to `{base_name}_{index}` zero-padded (`_SWEEP_ABBREV` is
gone — it stringified arrays into directory names and collided silently).
`name=(index, point) -> str` overrides. **The case layer does not own the
index → parameters mapping**; that is each run's staged `inputs/` plus the
registry. A design manifest artifact belongs to the Run layer (Stage 3).

### Path ownership — where CODT reads and writes

Four namelist keys locate files, and they resolve two different ways:

| Key | Read at (CODT) | Resolved against |
|---|---|---|
| `aerosol_file` | `src/droplets.f90:996` | `namelist_dir` |
| `parcel_file` | `src/parcel.f90:119` | `namelist_dir` |
| `mie_data_file` | `src/radiation.f90:670` | `namelist_dir` |
| `output_directory` | `src/initialize.f90:246` | **the process's cwd** |

`namelist_dir` is the parent of argv[1] *as typed* (`app/main.f90:43`);
`resolve_path` (`globals.f90:417`) takes absolute values as-is. So a relative
`output_directory` lands wherever the job started, and a stale absolute one
points at another run's directory (only caught when `overwrite=.false.` finds
the `.nc`).

codt_tools therefore treats the first three as **staging-owned**:

- a `Case` carries none of them — defaults are `output_directory=""`,
  `aerosol_file="aerosol_input.nc"`, `parcel_file=""`, and `from_input_dir` /
  `from_simulation` normalize back to those after using the stored paths to
  *find* the files;
- `case.set()` raises on all three, naming `write_inputs` instead;
- **`Case.write_inputs(directory, output_directory=None)` is the only place
  they are assigned**, and it assigns all of them from its own arguments every
  call, returning the staged `Namelist` (defaults to `directory/output`,
  resolved absolute, and creates it so CODT's parent-exists check passes);
- `Namelist.write()` refuses a namelist whose `output_directory` is empty;
- `mie_data_file` is user-owned as a source path and **copied into the input
  directory** at write time, referenced by basename — previously a relative Mie
  path resolved against `inputs/`, where nothing was ever staged.

The invariant that makes bare filenames safe: the model is always invoked with
an **absolute** namelist path (`runner.run_local`, both sbatch bodies), tested
in `tests/run/test_run.py::TestExecuteLocal::test_invokes_with_an_absolute_namelist_path`.

## Run Layer (0.9.0)

`codt_tools/run/` replaces `runner.py`. **Nothing in the package submits a
SLURM job** — no `sbatch`, `squeue` or `sacct` is called from Python. Batch
execution is *artifact generation*: codt_tools writes a script, you launch it.

`Run(case, executable, workdir)` — no account, partition, qos or registry.

| member | does |
|---|---|
| `stage()` | `case.write_inputs(workdir/"inputs", output_directory=workdir/"output")`; returns the staged `Namelist`. **Does not modify the case** — one case stages any number of runs. |
| `execute_local(**kw)` | runs the binary on the **absolute** `inputs/params.nml`, blocking; returns `CompletedProcess`. Warns with a plain-language cause when the binary dies on a signal (SIGILL = built for another CPU). |
| `open_simulation()` | delegates to `Simulation.from_run(self)`; raises unless `is_complete` |
| `is_complete` / `is_staged` | the `{case.name}_DONE` marker / the namelist exists |
| `name`, `inputs_dir`, `output_dir`, `namelist_path`, `done_marker` | derived from `workdir`; `name` is the *directory* name, `done_marker` uses `case.name` (they need not match) |
| `Run.for_cases(cases, exe, base_dir)` | one run per case at `base_dir/{case.name}`; **raises on duplicate names** |

`executable` and `workdir` are resolved absolute in `__init__` — that is what
preserves the Stage 1 invariant that bare staged filenames resolve, since CODT
takes `argv[1]`'s parent *as typed*. Pinned by
`tests/run/test_run.py::TestExecuteLocal::test_invokes_with_an_absolute_namelist_path`.

`check_executable(exe) -> str | None` gives a one-line reason a binary will not
run (missing, not executable, missing runtime lib, SIGILL, pre-0.4.0 CODT with
no `--version`), or None. It judges the binary **on the host it runs on**, so
it cannot catch an arch mismatch that only appears on a compute node — the
generated scripts run `"$EXE" --version` in situ for that.

### Launch scripts (`run/launcher.py`)

`write_local(runs, path, jobs=N)` and `write_slurm_array(runs, path,
runs_per_task=1, module=None, account=..., partition=..., qos=...,
constraint=..., time=..., mem=..., array_throttle=..., requeue=True,
**directives)`. Both take `list[Run]`, return the script path, and **execute
nothing**. Modeled on the hand-written `Rome_validate.sh`: per-run subshell, a
`_DONE` skip guard (`compgen -G "$RUN_DIR/output/*_DONE"`), `wait`, and a
timing line per run (`run, wall seconds, exit code` → `timing.txt`).

- The `_DONE` guard makes both scripts **idempotent** — re-running resumes.
  That is what makes `--requeue` safe, so the two ship together by default.
  Preemption is only handled at *run* granularity; CODT still has no
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

Retired with `codt_tools/slurm.py` in 0.9.0: `submit()`, `status()`, the arch
preflight, `detect_build_arch`, `ARCH_CONSTRAINT`, the `sinfo`/`mychpc`
introspection and the even-batch packing math. That knowledge is now prose in
`docs/running-on-slurm.md` (arch table, krueger entitlement, task sizing,
`readelf` recipe). Recover the code with `git show v0.8.0:codt_tools/slurm.py`.

Registry schema **v4**'s `build_arch` column stays, but **nothing fills it in
automatically** — pass `build_arch=` to `register_run` yourself. `Run` has no
registry coupling at all and generated scripts contain no `codt-registry`
text; recording is an explicit call (Stage 5 replaces the registry).

`Case.validate()` (`case/validate.py`) mirrors the Fortran-side checks: range validation (N, tmax, H, pres, volume_scaling, Tref, simulation_mode) raises `ValueError`; cross-namelist inconsistencies (do_radiation without do_microphysics, write_eddies without do_turbulence, do_entrainment in chamber mode) issue `warnings.warn`.

## Simulation Layer (0.9.0)

`codt_tools/simulation/` replaces the loose `simulation.py` / `plotting.py` /
`trajectory_io.py`:

| module | contents |
|---|---|
| `simulation.py` | `Simulation` (was `CODTSimulation`) — moved intact, one class |
| `plotting.py` | the `plot_*` helpers; `_get_label`/`_ensure_ax` are now public `get_label`/`ensure_ax` |
| `trajectory.py` | was `trajectory_io.py` |

`Simulation.from_run(run)` opens a completed run's output and raises if it has
no `_DONE` marker; `Run.open_simulation()` delegates to it, so the two cannot
drift. `Run` is imported under `TYPE_CHECKING` only, and `open_simulation`
imports `Simulation` inside the method — staging and launching a run should not
pay for xarray and matplotlib.

**Analysis no longer imports the registry.** `simulation.py` used to do
`from codt_tools.registry.versions import check_conventions`, which meant
reading your own output pulled in the bookkeeping layer. Pinned by
`tests/simulation/test_layer.py::TestNoRegistryDependency`, which asserts in a
*fresh interpreter* that neither `import codt_tools` nor
`import codt_tools.simulation` leaves anything under `codt_tools.registry` in
`sys.modules`.

### Conventions — each reader owns its format string

There is deliberately **no registry of "supported" conventions** to keep in
sync, and no top-level `conventions.py` (the refactor plan proposed one; it
was not needed once the input-side gating turned out to be registry-only).

| Reader | Constant | On mismatch |
|---|---|---|
| `simulation/simulation.py` | `OUTPUT_CONVENTIONS = "CODT_output_v1"` | **warns**, opens anyway |
| `case/aerosol.py` | `_CONVENTIONS = "CODT_aerosol_input_v1"` | **raises** |
| `case/parcel.py` | `CONVENTIONS = "CODT_parcel_input_v3"` | **raises** (names the retired v1/v2) |

The asymmetry is the point: output already exists and a scientist must be able
to look at it, so a mismatch is a warning naming expected vs found. A
wrong-format *input* silently produces a wrong simulation, so that raises.

**Resolved in Stage 5:** `registry/versions.py` was deleted with `api.py`, so
`"CODT_output_v1"` now exists in exactly one place and the rule above is
literally true — there is no "supported conventions" set anywhere.

### Namelist group placement (must match what CODT reads)

`Namelist._DEFAULTS` groups map 1:1 to the Fortran namelist of the same name. A param in the wrong group makes CODT reject the file (`Invalid parameter in &GROUP`). Defaults are aligned to CODT's **code** defaults (CODT's own `docs/input_parameters.md`, in the CODT repo). Mode/switch gating in `Namelist.groups_for_write()` (public since 0.9.0): chamber omits `turbulence_lem`/`parcel`/`entrainment`; parcel omits `turbulence_odt`/`specialeffects`; `radiation` only when `do_radiation`; `entrainment` only when `do_entrainment`.

- `do_entrainment` → `&PARAMETERS` (not `&PARCEL`)
- `pressure_limit` → `&PARCEL` (not `&PARAMETERS`)
- `ent_rate, n_blob, psigma, random_entrainment` → standalone `&ENTRAINMENT` (not `&PARCEL`)
- `radiation_method` must be `'1d'` or `'3d'` (not `'two_stream'`)
- `do_seeding, seed_hydration, seed_growth_time` → `&MICROPHYSICS` (not a `&SEEDING` group — there isn't one)

`{name}_DONE` discovery: `Run.done_marker` and `Simulation._discover_files` both correctly use `{name}_DONE`; the generated launch scripts glob `*_DONE` instead, since bash does not know `simulation_name`.

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

## Remaining Work

- Snakemake example Snakefile.
- A design manifest (`run_id` -> parameters) has no first-class home; the
  habit documented in `docs/designs.md` ("Recording the design") is the
  current answer. Deferred past 1.0 deliberately.

## Workshop materials (not in this repo)

**`examples/` is not part of this repository** and never has been. The
workshop tutorials live in the deployed copy at
`~krueger-group11/cbois/codt-workshop/` (a separate, deliberately stripped,
non-git copy of codt_tools, used by the `cloudy-odt` env — which is pinned to
`~/dev/CODT_tools-v0.8.0` and unaffected by the 1.0 refactor). Kept here
because the physics settings below were verified against real runs.

Four step-by-step tutorials (chamber simple/advanced, parcel simple/advanced) for interactive-Python + manual `CODT ./params.nml` workflows, plus `templates/{chamber,parcel}_template.py` end-to-end scripts and a pandoc site build (`site/build_site.sh` → `site/html/`, gitignored; publish by copying to `~/public_html/codt_workshop/`). Workshop conda env: `cloudy-odt`. Physics-verified settings: chamber `tmax=600` (~45 s wall), parcel `tmax=240` (~4.5 min wall); parcel cases need small CCN (`edge_radii=[30,60,120]` nm — the 291–1000 nm chamber aerosol never passes its Köhler critical radius, so `Nact` stays 0) and `initial_rh=0.98`; entrainment needs `ent_rate≈0.002` m⁻¹ with a humid sounding (`RH=[0.95,0.92,0.88]`) — `ent_rate=2.0` (the code default) or a dry sounding evaporates the entire cloud. Entrainment event spacing: `dt = (psigma/(1−psigma))/(ent_rate·|vel|)` — **independent of `n_blob`** from CODT 3.1.0 (it carried an `n_blob` factor through 3.0.1; see "CODT 3.1.0 psigma/n_blob redefinition" below).

## History

Version history, and the CODT interface changes codt_tools tracks, are in
[CHANGELOG.md](CHANGELOG.md). Several entries there are **needed to read
archived output correctly** — the `psigma` redefinition in CODT 3.1.0, the
LEM scale derivation in 3.0.0, the eddy-header field that changed meaning,
and why seeding must be detected by probing. Check a run's `code_version`
against them before trusting an old file.
