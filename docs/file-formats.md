# File formats and paths

What CODT reads, what it writes, and where it looks for each. Read this when
you are writing an input by hand, reading a binary output file, or debugging a
namelist CODT rejected.

## Output files

CODT writes everything into `output_directory/`, prefixed with
`simulation_name`. There is no automatic subdirectory.

| File | Description |
|------|-------------|
| `{name}.nc` | Main output: profiles and time series, schema `CODT_output_v1` |
| `{name}.log` | Redirected stdout, with all initialization and runtime messages |
| `{name}_particles.nc` | Trajectories, CF ragged array, schema `CODT_particle_output_v1`, if enabled |
| `{name}_collisions.bin` | Collision events, unformatted stream, if enabled |
| `{name}_eddies.bin` | Eddy events, unformatted stream, if enabled |
| `{name}_DONE` | Completion marker with a timestamp |

`{name}_DONE` is the only reliable answer to "did this run finish". CODT writes
it on a clean exit. `run.is_complete` reads it.

## Namelist group placement

Each `Namelist` group maps 1:1 to the Fortran namelist of the same name. **A
parameter in the wrong group makes CODT reject the whole file** with
`Invalid parameter in &GROUP`. The defaults match CODT's own code defaults.

These five are the ones that catch people out:

- `do_entrainment` → `&PARAMETERS` (not `&PARCEL`)
- `pressure_limit` → `&PARCEL` (not `&PARAMETERS`)
- `ent_rate, n_blob, psigma, random_entrainment` → standalone `&ENTRAINMENT`
  (not `&PARCEL`)
- `radiation_method` must be `'1d'` or `'3d'`
- `do_seeding, seed_hydration, seed_growth_time` → `&MICROPHYSICS`. There is no
  `&SEEDING` group.

`Namelist.groups_for_write()` gates by mode and switch, so a written namelist
holds only the groups that run. Chamber omits `turbulence_lem`, `parcel` and
`entrainment`. Parcel omits `turbulence_odt` and `specialeffects`. It writes
`radiation` only when `do_radiation`, and `entrainment` only when
`do_entrainment`.

## Where CODT looks for files

Four namelist keys locate files, and they resolve two different ways:

| Key | Resolved against |
|---|---|
| `aerosol_file` | the namelist's parent directory |
| `parcel_file` | the namelist's parent directory |
| `mie_data_file` | the namelist's parent directory |
| `output_directory` | **the process's working directory** |

The namelist's parent directory is the parent of `argv[1]` *as typed*. Absolute
values are taken as-is. So a relative `output_directory` lands wherever the job
started, and a stale absolute one points at another run's directory. CODT
catches that second case only when `overwrite=.false.` finds an existing `.nc`.

**codt_tools owns the first three paths, so you never set them.**

- A `Case` carries none of them. `case.set()` raises on all three and names
  `write_inputs` instead.
- `Case.write_inputs(directory, output_directory=None)` is the only place that
  assigns them. It assigns all of them from its own arguments on every call and
  returns the staged `Namelist`. `output_directory` defaults to
  `directory/output`, resolves absolute, and is created so CODT's
  parent-exists check passes.
- `Namelist.write()` refuses a namelist whose `output_directory` is empty.
- `mie_data_file` is yours as a source path. `write_inputs` copies it into the
  input directory and references it by basename.

Bare filenames are safe because codt_tools always invokes the model with an
**absolute** namelist path.

## Conventions strings

Each reader accepts exactly one format string. There is no central list of
supported conventions.

| Reader | Constant | On mismatch |
|---|---|---|
| output | `CODT_output_v1` | **warns**, opens anyway |
| aerosol input | `CODT_aerosol_input_v1` | **raises** |
| parcel input | `CODT_parcel_input_v3` | **raises** |

The asymmetry is deliberate. Output already exists and you must be able to look
at it, so a mismatch only warns and names expected versus found. A wrong-format
*input* silently produces a wrong simulation, so that raises.

## NetCDF global attributes

`conventions`, `code_version` (a git-describe string from build time, for
example `v1.0.0` or `v1.0.0-6-g90dabe0-dirty`, **not** bare semver), and
`git_commit` (short hash, with `-dirty` if the tree was uncommitted).

Namelist parameters appear as `PARAMETERS.N`, `MICROPHYSICS.write_trajectories`
and so on. Booleans are stored as integers (0/1). The set is mode-guarded.
Treat `code_version` as an opaque string — nothing parses it as semver.

## Eddy binary (`{name}_eddies.bin`)

Unformatted Fortran stream with a mode-aware header:

1. `mode_flag` (i1): 0 = chamber, 1 = parcel
2. `N` (i4), `H` (f8) — shared fields
3. Mode-specific fields (f8 array):
   - Chamber: C2, ZC2, Tdiff, Tref (4 values)
   - Parcel: integral_length_scale, smallest_eddy_scale, dissipation_rate
     (3 values)

Per-eddy record: M(i4), L(i4), time(f8). These are raw grid indices, so you can
replay an eddy with `implement_eddy(L, M)`.

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

## Collision binary (`{name}_collisions.bin`)

Unformatted Fortran stream. Header: N(i4), H(f8), domain_width(f8),
volume_scaling(f8). Per-event: id_keep(i4), id_kill(i4), r_keep(f8),
r_kill(f8), r_after(f8), position(f8), time(f8), coalesced(i1). The packed
record is 49 bytes. `load_collisions` reads the trailing `coalesced` flag,
where 1 = merged and 0 = bounce.

`time` is absolute simulation time, on the same axis as the output NC `time`
coordinate. CODT writes the binary in event order. Binary totals match
`N_collisions` and `N_coalescences` exactly, so that is a safe cross-check.

```python
dt_header = np.dtype([('N','<i4'),('H','<f8'),('domain_width','<f8'),('volume_scaling','<f8')])
dt_record = np.dtype([('id_keep','<i4'),('id_kill','<i4'),('r_keep','<f8'),('r_kill','<f8'),('r_after','<f8'),('position','<f8'),('time','<f8'),('coalesced','<i1')])
```

## See also

- [starting-a-project.md](starting-a-project.md) — the three-script project layout
- [designs.md](designs.md) — expressing an ensemble
