# CODT Tools

Python framework for configuring, running, and analyzing Cloudy One-Dimensional Turbulence (CODT) simulations.

- CODT (Fortran): https://github.com/CappedColumn/cloudy-odt
- codt_tools (Python): https://github.com/CappedColumn/cloudy-odt-tools
- Deps: numpy, xarray, netCDF4, matplotlib, f90nml
- Cluster: SLURM

## Executable Contract

```
./CODT <NAMELIST_PATH>       # path MUST contain '/' (use ./params.nml)
```

- `aerosol_file`, `bin_data_file` resolve relative to namelist parent dir
- `output_directory` must be absolute
- Stdout -> `{output_dir}/{sim_name}/{sim_name}.log`, stderr -> output path only
- `DONE` file = success marker; exit 0 = success, 1 = error
- `overwrite=.false.` (default) rejects if output `.nc` exists

Build: `source fpm_env && fpm build` -> `./build/*/app/CODT`

## Namelist Groups

`&PARAMETERS`: N, Lmin, Lprob, tmax, Tdiff, Tref, pres, H, volume_scaling, max_accept_prob, same_random, simulation_name, output_directory, overwrite, write_timer, write_buffer, write_eddies, do_turbulence, do_microphysics, do_special_effects, simulation_mode ('chamber'|'parcel'), integral_length_scale, kolmogorov_length_scale, dissipation_rate

`&MICROPHYSICS`: init_drop_each_gridpoint, expected_Ndrops_per_gridpoint, initial_wet_radius, aerosol_file, bin_data_file, write_trajectories, trajectory_start, trajectory_end, trajectory_timer, do_collisions, do_coalescence, coalescence_kernel ('hall'|'long'|'unity'), wmax_collision, write_collisions

`&SPECIALEFFECTS`: do_sidewalls, area_sw, area_bot, C_sw, sw_nudging_time, T_sw, RH_sw, P_sw, do_random_fallout, random_fallout_rate

## Output Files (`{output_dir}/{sim_name}/`)

| File | Description |
|------|-------------|
| `{name}.nc` | Main output (profiles + time series) |
| `{name}.log` | Redirected stdout |
| `{name}.nml` | Namelist copy |
| `aerosol_input.nc` | Aerosol input copy |
| `{name}_particles.nc` | Trajectories (CF ragged array, if enabled) |
| `{name}_collisions.bin` | Collision events (unformatted stream, if enabled) |
| `{name}_eddies.bin` | Eddy events (unformatted stream, if enabled) |
| `DONE` | Completion marker |

### Main netCDF (`{name}.nc`)

Dims: `time` (unlimited), `z`, `radius`/`radius_edges` (if microphysics).
Always: `z`, `time`, `T`, `QV`, `Tv`, `S`, `W`.
Microphysics: `radius`, `radius_edges`, `DSD`, `DSD_1`, `DSD_2`, `Np`, `Nact`, `Nun`, `Ravg`, `LWC`.
Budgets (always defined, `time` dim, double, accumulated per write interval then reset):
`budget_inject_solute_mass`, `budget_inject_liquid_mass` (kg);
`budget_fallout_liquid_mass`, `budget_fallout_solute_mass` (kg);
`budget_condensation` (kg, net cond/evap);
`budget_dgm_delta_T` (K, sum of per-droplet ΔT from DGM);
`budget_diffusion_delta_T` (K), `budget_diffusion_delta_WV` (kg/kg) — domain-sum change from diffusion (≈0 for LEM periodic);
`budget_sidewall_delta_T` (K), `budget_sidewall_delta_WV` (kg/kg) — 0 unless sidewalls enabled;
`budget_n_injected`, `budget_n_fellout`, `budget_n_coalesced` (counts, stored as double);
`N_collisions`, `N_coalescences` (per-interval counts).
Global attrs: namelist params as `PARAMETERS.N`, `MICROPHYSICS.write_trajectories`, etc. Bools as int (0/1). Machine paths excluded.

### Particle NetCDF (`{name}_particles.nc`)

CF contiguous ragged array. Dims: `record` (unlimited), `time_step` (unlimited).
Per-record: particle_id, aerosol_id, gridcell, position, temperature, water_vapor, supersaturation, radius, solute_radius, activated, aerosol_category, n_collisions, n_coalescences, radius_before_coalescence.
Per-time-step: time, row_sizes (`cf_role="ragged_row_sizes"`).

### Eddy Binary (`{name}_eddies.bin`)

Unformatted Fortran stream. Header: N(i4), H(f8), C2(f8), ZC2(f8), Tdiff(f8), Tref(f8). Per-eddy: M(i4), L(i4), time(f8). Raw grid indices for replay via `implement_eddy(L, M)`.

```python
dt_header = np.dtype([('N','<i4'),('H','<f8'),('C2','<f8'),('ZC2','<f8'),('Tdiff','<f8'),('Tref','<f8')])
dt_eddy = np.dtype([('M','<i4'),('L','<i4'),('time','<f8')])
```

### Collision Binary (`{name}_collisions.bin`)

Unformatted Fortran stream. Header: N(i4), H(f8), domain_width(f8), volume_scaling(f8). Per-event: id_keep(i4), id_kill(i4), r_keep(f8), r_kill(f8), r_after(f8), position(f8), time(f8).

```python
dt_header = np.dtype([('N','<i4'),('H','<f8'),('domain_width','<f8'),('volume_scaling','<f8')])
dt_record = np.dtype([('id_keep','<i4'),('id_kill','<i4'),('r_keep','<f8'),('r_kill','<f8'),('r_after','<f8'),('position','<f8'),('time','<f8')])
```

### Aerosol Input (`aerosol_input.nc`, schema `CODT_aerosol_input_v1`)

Dims: aerosol_type, edge (N bins+1), bin (N bins), time (injection steps).
Vars: n_ions, molar_mass, solute_density, edge_radii (nm), category, cumulative_frequency (time,bin in Python -> bin,time in Fortran), injection_time, injection_rate.
Attrs: `conventions="CODT_aerosol_input_v1"`, `aerosol_name`.

### Bin Data (`bin_data.txt`)

Header "N Bin-Edges", int count, then one float per line (microns).

## Runner Layout

`CODTRunner.setup_run` creates `{base}/{sim_name}/run/` with params.nml, aerosol_input.nc, bin_data.txt. Output goes to `{base}/{sim_name}/`.

## In Progress

- **LEM** (`implement-LEM`): merged (PR #11). `simulation_mode='parcel'` with periodic BCs, periodic triplet map, -5/3 eddy selection.
- **Collision-coalescence** (`coll-coal`): merged (PR #13). Event-driven 1D CC with Hall/Long/unity kernels. New namelist params in `&MICROPHYSICS`.

## Remaining Work

- Eddy binary reader for `_eddies.bin`
- Snakemake example Snakefile
- `sim.budget_totals()` cumsum helper
- Budget closure check (inject - fallout + condensation ≈ ΔLWC)
