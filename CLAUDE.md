---
# CODT Simulation Framework — Design Document

## Developer Context
- We use Python and modern Fortran
- Prioritize correct physics and readability over optimization
- CODT source code: https://github.com/CappedColumn/cloudy-odt
- codt_tools source code: https://github.com/CappedColumn/cloudy-odt-tools
- All Fortran I/O and execution changes are merged to main on GitHub
- Cluster uses SLURM, jobs run on [your partition/account info]
- Python environment has numpy, xarray, netCDF4, matplotlib

## 1. Overview

A Python framework for configuring, running, and post-processing simulations from the
Cloudy One-Dimensional Turbulence (CODT) model. The framework has three layers:

1. **Configuration** (`CODTConfig`) — build and modify input files
2. **Execution** (`CODTRunner`) — submit jobs via SLURM, pack multiple serial runs per node
3. **Analysis** (`CODTSimulation`) — load output, compute diagnostics, produce plots

Repository: https://github.com/CappedColumn/cloudy-odt-tools

---

## 2. Model I/O Specification

### 2.1 Input Files

The model reads input from the **working directory**. Three files are required:

#### `params.nml` — Fortran namelist (3 groups)

```
&PARAMETERS
  N, Lmin, Lprob, tmax, Tdiff, Tref, pres, H,
  volume_scaling, max_accept_prob, same_random,
  simulation_name, output_directory, overwrite,
  write_timer, write_buffer, write_eddies,
  do_turbulence, do_microphysics, do_special_effects
/

&MICROPHYSICS
  init_drop_each_gridpoint, expected_Ndrops_per_gridpoint,
  initial_wet_radius,
  aerosol_file, bin_data_file,
  write_trajectories, trajectory_start, trajectory_end, trajectory_timer
/

&SPECIALEFFECTS
  do_sidewalls, area_sw, area_bot, C_sw, sw_nudging_time,
  T_sw, RH_sw, P_sw,
  do_random_fallout, random_fallout_rate
/
```

**Key behavior**: The namelist path is passed as a command-line argument.
Relative paths inside the namelist (`aerosol_file`, `bin_data_file`) are
resolved relative to the namelist's parent directory. `output_directory` must
be an absolute path. Output is written to `{output_directory}/{simulation_name}/`.

#### `input/aerosol_input.nc` — Aerosol specification (NetCDF4, `CODT_aerosol_input_v1`)

**Dimensions:**

| Dimension | Size | Description |
|-----------|------|-------------|
| `aerosol_type` | N types (1 for now) | Aerosol chemistry types |
| `edge` | N bins + 1 | Bin edge count |
| `bin` | N bins | Size bin count |
| `time` | N injection times | Injection schedule steps |

**Variables:**

| Variable | Dimensions | Type | Units | Description |
|----------|-----------|------|-------|-------------|
| `n_ions` | (aerosol_type) | int | — | Number of dissociating ions |
| `molar_mass` | (aerosol_type) | double | kg mol⁻¹ | Solute molar mass |
| `solute_density` | (aerosol_type) | double | kg m⁻³ | Solute density |
| `edge_radii` | (edge) | double | nm | Dry aerosol bin edges |
| `category` | (bin) | int | — | Aerosol type index per bin |
| `cumulative_frequency` | (time, bin)* | double | — | CDF thresholds [0,1] per bin |
| `injection_time` | (time) | double | s | Injection schedule times |
| `injection_rate` | (time) | double | m⁻³ s⁻¹ | Aerosol injection rate |

*Note: `cumulative_frequency` dimensions are `(time, bin)` in Python/C-order,
which becomes `(bin, time)` in Fortran column-major order.

**Global attributes:**

| Attribute | Value | Description |
|-----------|-------|-------------|
| `conventions` | `"CODT_aerosol_input_v1"` | Schema version (validated on read) |
| `aerosol_name` | e.g., `"NaCl"` | Human-readable aerosol label |

**Python creation example:**
```python
import netCDF4 as nc
import numpy as np

ds = nc.Dataset("aerosol_input.nc", "w", format="NETCDF4")
ds.conventions = "CODT_aerosol_input_v1"
ds.aerosol_name = "NaCl"

ds.createDimension("aerosol_type", 1)
ds.createDimension("edge", 3)      # 2 bins + 1
ds.createDimension("bin", 2)
ds.createDimension("time", 1)

v = ds.createVariable("n_ions", "i4", ("aerosol_type",))
v[:] = [2]

v = ds.createVariable("molar_mass", "f8", ("aerosol_type",))
v.units = "kg mol-1"
v[:] = [58.4428e-3]

v = ds.createVariable("solute_density", "f8", ("aerosol_type",))
v.units = "kg m-3"
v[:] = [2163.0]

v = ds.createVariable("edge_radii", "f8", ("edge",))
v.units = "nm"
v[:] = [60.0, 70.0, 4930.0]

v = ds.createVariable("category", "i4", ("bin",))
v[:] = [1, 2]

# NOTE: (time, bin) in Python -> (bin, time) in Fortran
v = ds.createVariable("cumulative_frequency", "f8", ("time", "bin"))
v[:] = [[1.0, 1.0]]

v = ds.createVariable("injection_time", "f8", ("time",))
v.units = "s"
v[:] = [0.0]

v = ds.createVariable("injection_rate", "f8", ("time",))
v.units = "m-3 s-1"
v[:] = [5.5e5]

ds.close()
```

#### `input/bin_data.txt` — Droplet size distribution bin edges

```
N Bin-Edges
<int>                           # e.g., 201
Droplet Bin-Edges (microns)
<float>                         # one per line, 201 values
<float>                         # spanning ~0.049 to 65.26 microns
...
```

### 2.2 Output Files

All output is written to `{output_directory}/{simulation_name}/`. The simulation
produces the following files (where `{name}` = `simulation_name`):

| File | Format | Description |
|------|--------|-------------|
| `{name}.nc` | netCDF4 (HDF5) | Main output: profiles + time series |
| `{name}.log` | ASCII | Stdout redirected here by the model |
| `{name}.nml` | ASCII | Verbatim copy of the original namelist |
| `aerosol_input.nc` | netCDF4 | Copy of aerosol input (original basename) |
| `{name}_particles.nc` | netCDF4 (HDF5) | Particle trajectory data (if `write_trajectories=.true.`) |
| `{name}_eddies.bin` | Unformatted stream | Accepted eddy events (if `write_eddies=.true.`) |
| `DONE` | ASCII | Completion marker (timestamp), only on success |

### 2.3 Main netCDF Structure (`{name}.nc`)

**Dimensions:**
- `time` — UNLIMITED (e.g., 7201 for 7200s at 1s write interval)
- `z` — number of grid cells (e.g., 2000)
- `radius` — number of droplet bins (e.g., 200 = N_bin_edges - 1) *(only when `do_microphysics=.true.`)*
- `radius_edges` — number of bin edges (e.g., 201) *(only when `do_microphysics=.true.`)*

**Variables (always present):**

| Variable | Dims | `long_name` | Units |
|----------|------|-------------|-------|
| `z` | (z) | Height | meters |
| `time` | (time) | Time | seconds |
| `T` | (time, z) | Temperature | celsius |
| `QV` | (time, z) | Water Vapor Mixing Ratio | g/kg |
| `Tv` | (time, z) | Virtual Temperature | celsius |
| `S` | (time, z) | Supersaturation | % |
| `W` | (time, z) | W-Velocity | m/s |

**Variables (only when `do_microphysics=.true.`):**

| Variable | Dims | `long_name` | Units |
|----------|------|-------------|-------|
| `radius` | (radius) | Droplet Bin Centers | microns |
| `radius_edges` | (radius_edges) | Droplet Bin Edges | microns |
| `DSD` | (time, radius) | Droplet Size Distribution | # |
| `DSD_1` | (time, radius) | Droplet Size Distribution - 1 | # |
| `DSD_2` | (time, radius) | Droplet Size Distribution - 2 | # |
| `Np` | (time) | Number of Particles | # |
| `Nact` | (time) | Number of Activated Particles | # |
| `Nun` | (time) | Number of Unactivated Particles | # |
| `Ravg` | (time) | Average Particle Radius (wet) | um |
| `LWC` | (time) | Liquid Water Content | g/m3 |

**Global Attributes:** Namelist parameters are stored as dot-prefixed global
attributes grouped by namelist (e.g., `PARAMETERS.N`, `MICROPHYSICS.write_trajectories`,
`SPECIALEFFECTS.do_sidewalls`). Machine-specific paths (`output_directory`,
`aerosol_file`, `bin_data_file`) are excluded. Booleans are stored as integers
(1=true, 0=false). MICROPHYSICS and SPECIALEFFECTS attributes are only present
when those modules are enabled. `CODTSimulation` should prefer these over the
`.nml` file when available.

Python parsing example:
```python
params = {k: v for k, v in ds.attrs.items() if k.startswith("PARAMETERS.")}
micro = {k: v for k, v in ds.attrs.items() if k.startswith("MICROPHYSICS.")}
```

### 2.4 Particle NetCDF Structure (`{name}_particles.nc`)

Written when `write_trajectories = .true.` and simulation time overlaps
`[trajectory_start, trajectory_end]`. Uses a CF contiguous ragged array
layout: all particle records are flattened into a single `record` dimension,
and a `row_sizes` variable indicates how many records belong to each time step.

**Dimensions:**
- `record` — UNLIMITED, total particle records across all time steps
- `time_step` — UNLIMITED, number of trajectory write events

**Per-record variables** (on `record` dimension, all compressed with `deflate_level=1, shuffle=true`):

| Variable | Type | Units | Description |
|----------|------|-------|-------------|
| `particle_id` | INT | — | Unique particle identifier |
| `aerosol_id` | INT | — | Aerosol type ID |
| `gridcell` | INT | — | Grid cell index |
| `position` | FLOAT | m | Vertical position in domain |
| `temperature` | FLOAT | celsius | Local temperature |
| `water_vapor` | FLOAT | g/kg | Water vapor mixing ratio |
| `supersaturation` | FLOAT | % | Local supersaturation |
| `radius` | FLOAT | um | Wet droplet radius |
| `solute_radius` | FLOAT | m | Dry solute radius |
| `activated` | INT | 0/1 | Activation flag (`flag_values=0,1`) |
| `aerosol_category` | INT | — | Aerosol category label |

**Per-time-step variables** (on `time_step` dimension):

| Variable | Type | Units | Attributes |
|----------|------|-------|------------|
| `time` | DOUBLE | seconds | Simulation time |
| `row_sizes` | INT | — | `cf_role="ragged_row_sizes"`, `sample_dimension="record"` |

**Recovering trajectories:**
- All particles at time step `t`: use `cumsum(row_sizes)` to slice into `record`
- Trajectory of particle `P`: filter all records where `particle_id == P`

**Python reading example:**
```python
import xarray as xr
import numpy as np

ds = xr.open_dataset("sim_particles.nc")
row_sizes = ds["row_sizes"].values
time = ds["time"].values

# Get all particles at time step t
offsets = np.concatenate([[0], np.cumsum(row_sizes)])
start, end = offsets[t], offsets[t + 1]
particles_at_t = ds.isel(record=slice(start, end))

# Get trajectory of particle P
mask = ds["particle_id"] == P
trajectory = ds.sel(record=mask)
```

### 2.5 Eddy Binary Format (`{name}_eddies.bin`)

Written when `write_eddies = .true.`. Contains accepted eddy events as an
unformatted Fortran stream file. Each eddy is written directly (no buffering)
as it is accepted during the simulation.

**Header** (written once at file creation):

| Field | Type | Description |
|-------|------|-------------|
| `N` | `integer(i4)` | Grid size (number of cells) |
| `H` | `real(dp)` | Domain height (meters) |
| `C2` | `real(dp)` | ODT turbulent strength parameter |
| `ZC2` | `real(dp)` | ODT viscous cut-off parameter |
| `Tdiff` | `real(dp)` | Top-bottom temperature difference (K) |
| `Tref` | `real(dp)` | Bottom boundary temperature (K, after Kelvin conversion) |

**Per-eddy record** (repeated, one per accepted eddy):

| Field | Type | Description |
|-------|------|-------------|
| `M` | `integer(i4)` | Eddy start cell index (passed to `implement_eddy`) |
| `L` | `integer(i4)` | Eddy length in cells (passed to `implement_eddy`) |
| `time` | `real(dp)` | Time of eddy event (seconds) |

**Reading in Fortran:**
```fortran
open(unit, file='..._eddies.bin', form='unformatted', access='stream', status='old')
read(unit) N, H, C2, ZC2, Tdiff, Tref   ! header
do
    read(unit, iostat=ios) M, L, etime    ! per-eddy
    if (ios /= 0) exit
end do
```

**Reading in Python:**
```python
import numpy as np

dt_header = np.dtype([('N', '<i4'), ('H', '<f8'), ('C2', '<f8'),
                      ('ZC2', '<f8'), ('Tdiff', '<f8'), ('Tref', '<f8')])
dt_eddy = np.dtype([('M', '<i4'), ('L', '<i4'), ('time', '<f8')])

with open(path, 'rb') as f:
    header = np.frombuffer(f.read(dt_header.itemsize), dtype=dt_header)[0]
    eddies = np.frombuffer(f.read(), dtype=dt_eddy)
```

**Design intent:** Stores raw grid indices so eddies can be replayed in another
simulation via `implement_eddy(L, M)` without renormalization. The importing
simulation must have the same `N` (or explicitly rescale indices).

### 2.6 Executable Interface Contract

This section defines the interface between `codt_tools` (Python) and the CODT
executable (Fortran). Both projects must honor this contract.

**Status: Implemented** and merged to main in the Fortran source.

### Invocation

```
./CODT <NAMELIST_PATH>
```

- A namelist path is **required**. If no argument is provided, the executable
  prints an error and exits with code 1.
- The path must contain at least one `/` — use `./params.nml` for the current
  directory. A bare filename (`params.nml`) is rejected.

### Path Resolution Rules

| Path in namelist | Resolved relative to |
|------------------|---------------------|
| `aerosol_file` | Namelist parent directory |
| `bin_data_file` | Namelist parent directory |
| `output_directory` | **Absolute path required** (must start with `/`) |

Absolute paths (starting with `/`) are used as-is for any field.

### Model Behavior

- **Stdout** is redirected by the model to `{output_dir}/{sim_name}/{sim_name}.log`
- **Stderr** prints only the output directory path (for SLURM visibility)
- **DONE marker**: On successful completion, a `DONE` file with a timestamp is
  written to the output directory. Absence = failed or still running.
- **Exit codes**: 0 = success, 1 = error (enables SLURM status detection)
- **Overwrite protection**: If `overwrite = .false.` (default) and the output
  `.nc` file already exists, the simulation stops with exit code 1.
- **Output copies**: The original namelist is copied as `{sim_name}.nml` and
  the aerosol input file is copied with its original basename.

### Runner File Layout

When `CODTRunner` sets up a run, it:
1. Creates `{run_dir}/` containing `params.nml`, `aerosol_input.nc`, `bin_data.txt`
2. Sets `aerosol_file = "aerosol_input.nc"` and `bin_data_file = "bin_data.txt"`
   (relative to namelist location)
3. Sets `output_directory` to an absolute path
4. Invokes: `./CODT {run_dir}/params.nml`

Output lands in `{output_directory}/{simulation_name}/`:
```
{output_directory}/{simulation_name}/
    {simulation_name}.nc
    {simulation_name}.log
    {simulation_name}.nml
    aerosol_input.nc
    DONE
    ...
```

### Build

```bash
source fpm_env && fpm build
# Executable: ./build/*/app/CODT
# Or install to a fixed location:
fpm install --prefix ~/simulations/CODT
# Executable: ~/simulations/CODT/bin/CODT
```

---

## 3. Class Design

### 3.1 `CODTConfig`

**Status: Implemented.** Bundles namelist, aerosol injection, and bin data into a
single configuration object. Uses composition — wraps `Namelist`, `InjectionData`,
and `BinData` (which remain available for standalone read/inspection).

```python
class CODTConfig:
    """Complete set of CODT input parameters."""

    def __init__(self, namelist_path=None):
        """Load from existing params.nml (+ sibling data files), or defaults."""

    # --- Components (public, for direct access when needed) ---
    params: Namelist           # Fortran namelist parameters
    injection: InjectionData   # Aerosol injection spec (NetCDF)
    bins: BinData              # Droplet bin edges

    # --- Convenience setters ---
    def set(self, **kwargs):
        """Set namelist params by name (delegates to Namelist.set).
        Example: config.set(tref=22.0, tmax=3600)
        """

    def set_injection(self, **kwargs):
        """Set injection attributes (delegates to InjectionData.set).
        Example: config.set_injection(aerosol_name="KCl", injection_rate=1e5)
        """

    def set_bins(self, edges):
        """Set droplet bin edges in microns (delegates to BinData.set)."""

    @property
    def name(self) -> str:
        """Simulation name (from namelist)."""

    # --- I/O ---
    def write(self, directory):
        """Write params.nml, aerosol_input.nc, bin_data.txt to directory.
        Sets aerosol_file/bin_data_file to relative names automatically.
        """

    def validate(self):
        """Check internal consistency:
        - injection CDF dimensions match N_bins x N_injection_times
        - trajectory_start < trajectory_end < tmax (if trajectories enabled)
        """

    # --- Sweep generation ---
    @staticmethod
    def sweep(base_config, **param_ranges) -> list['CODTConfig']:
        """Generate Cartesian product of parameter variations.

        Example:
            configs = CODTConfig.sweep(base,
                tref=[20.0, 21.0, 22.0],
                volume_scaling=[13, 50]
            )
            # Returns 6 configs with auto-generated names:
            # "base_Tref20.0_VS13", "base_Tref20.0_VS50", ...
        """

    def copy(self) -> 'CODTConfig':
        """Deep copy for manual parameter variation."""
```

**Component classes** (`Namelist`, `InjectionData`, `BinData`) are exported at
package level for read/inspection use. Only `CODTConfig` should be used to write
simulation input sets — it ensures the three files stay consistent and co-located.

Namelist parsing uses `f90nml` (required dependency).

### 3.2 `CODTRunner`

**Status: Implemented.** Manages directory setup, local execution, and SLURM
job submission. Methods are stateless — `setup_run` returns paths, `submit`
takes paths — making them composable with Snakemake or other orchestrators.

```python
class CODTRunner:
    """Set up and run CODT simulations locally or on a SLURM cluster."""

    def __init__(self, executable, base_output_dir,
                 account, partition, cores_per_node=40):
        """No executable validation at construction time (deferred to
        run_local/submit). Enables test fixtures and ahead-of-time config.
        """

    def setup_run(self, config: CODTConfig) -> Path:
        """Create a run directory with all input files.
        Sets output_directory to base_output_dir (absolute).
        Returns {base_output_dir}/{simulation_name} (sim root).

        Directory structure:
            {base_output_dir}/{simulation_name}/
                run/
                    params.nml
                    aerosol_input.nc
                    bin_data.txt
        """

    def setup_runs(self, configs: list[CODTConfig]) -> list[Path]:
        """Batch setup — calls setup_run for each config."""

    def run_local(self, config: CODTConfig) -> CompletedProcess:
        """Set up and run a single simulation locally (blocking).
        No stdout capture — model handles its own logging.
        """

    def submit(self, run_dirs: list[Path],
               walltime='24:00:00', dry_run=False) -> list[str]:
        """Submit to SLURM. Batches into groups of cores_per_node.
        dry_run=True writes scripts and returns paths without submitting.
        dry_run=False calls sbatch and returns job IDs.
        """

    def status(self, job_ids: list[str]) -> dict[str, str]:
        """Query SLURM for job status via squeue."""

    def collect(self, sim_names: list[str]) -> list[CODTSimulation]:
        """Load completed simulations. Skips sims without DONE marker."""

    def _generate_sbatch(self, run_dirs, walltime, batch_id=0) -> str:
        """Generate SLURM batch script with taskset core pinning.

        Template:
            #!/bin/bash
            #SBATCH --account={account}
            #SBATCH --partition={partition}
            #SBATCH --nodes=1
            #SBATCH --ntasks={len(run_dirs)}
            #SBATCH --time={walltime}
            #SBATCH --job-name=CODT_{batch_id}

            taskset -c 0 {executable} {run_dir_0}/run/params.nml &
            taskset -c 1 {executable} {run_dir_1}/run/params.nml &
            ...
            wait
        """
```

**Snakemake integration**: The runner API is Snakemake-compatible without changes.
Primary mode: Snakemake manages SLURM (`--executor slurm`), runner provides
`setup_run()`. For batch packing, use Snakemake's `group` directive. The runner's
own `submit()` is for interactive/notebook use.

### 3.3 `CODTSimulation`

**Status: Implemented.**

The primary analysis object. Points at a completed simulation's output directory.
Variable access uses netCDF variable names directly via `__getattr__`.

```python
class CODTSimulation:
    """Load and analyze output from a completed CODT simulation."""

    def __init__(self, path):
        """
        path: directory containing output files, OR path to the .nc file.
        Auto-discovers all associated files based on simulation_name.
        """

    # --- Metadata ---
    name: str                    # simulation name
    path: pathlib.Path           # output directory
    params: dict                 # namelist parameters (from netCDF global attrs)
    completed: bool              # True if DONE marker exists

    # Convenience accessors for commonly-used parameters
    @property
    def N(self) -> int: ...       # grid cells
    @property
    def tmax(self) -> float: ...
    @property
    def Tref(self) -> float: ...
    @property
    def H(self) -> float: ...
    @property
    def volume_scaling(self) -> float: ...
    @property
    def dz(self) -> float: ...    # H / N

    # --- Coordinates (loaded once on init) ---
    time: np.ndarray              # shape (n_times,), seconds
    z: np.ndarray                 # shape (n_z,), meters
    bin_edges: np.ndarray         # shape (n_bins+1,), microns (from radius_edges)
    radius: np.ndarray            # shape (n_bins,), microns (bin centers)

    # --- Field access (dynamic via __getattr__, uses netCDF names) ---
    # Access any netCDF variable as an attribute:
    sim.T           # Temperature (time, z), °C
    sim.QV          # Water Vapor Mixing Ratio (time, z), g/kg
    sim.Tv          # Virtual Temperature (time, z), °C
    sim.S           # Supersaturation (time, z), %
    sim.W           # W-Velocity (time, z), m/s
    sim.Np          # Number of Particles (time)
    sim.Nact        # Number of Activated Particles (time)
    sim.Nun         # Number of Unactivated Particles (time)
    sim.Ravg        # Average Particle Radius (time), µm
    sim.LWC         # Liquid Water Content (time), g/m³
    sim.DSD         # Droplet Size Distribution (time, radius)
    sim.DSD_1       # DSD category 1 (time, radius)
    sim.DSD_2       # DSD category 2 (time, radius)

    # --- Derived quantities ---
    def profile(self, variable, t) -> xr.DataArray:
        """Extract profile at time nearest to t."""

    def time_average(self, variable, t_start, t_end) -> xr.DataArray:
        """Time-averaged profile over [t_start, t_end]."""

    def domain_average(self, variable) -> xr.DataArray:
        """Spatial mean time series of a (time, z) variable."""

    def dsd_average(self, t_start, t_end, ...) -> xr.DataArray:
        """Time-averaged DSD with normalization options."""

    def set_core_region(self, z_min=None, z_max=None, auto=False):
        """Set spatial bounds for core-region analysis."""

    def spectral_width(self, t=None, t_start=None, t_end=None):
        """Number-weighted std dev of DSD. Returns scalar or time series."""

    def activation_fraction(self) -> xr.DataArray:
        """Nact / Np time series."""

    @staticmethod
    def compare(simulations, variable, plot_type='timeseries', **kwargs):
        """Multi-simulation overlay plots (timeseries, profile, spectrum).
        Auto-assigns evenly-spaced colors from a colormap."""

    # --- Trajectory loading ---
    def load_trajectories(self) -> xr.Dataset:
        """Load particle trajectory data from {name}_particles.nc."""

    def trajectory_of(self, pid) -> xr.Dataset:
        """Extract full trajectory of a single particle by PID."""

    def particles_at_time(self, t) -> xr.Dataset:
        """Extract all particles at time step index t."""

    def particle_ids(self) -> np.ndarray:
        """Return sorted array of unique particle IDs."""

    has_trajectories: bool  # property: True if _particles.nc exists

    # --- Standard plots ---
    def plot_timeheight(self, variable, ax=None, **kwargs):
        """Time-height (Hovmöller) contour plot."""

    def plot_profile(self, variable, times, ax=None, **kwargs):
        """Vertical profiles at specified times."""

    def plot_timeseries(self, variable, ax=None, **kwargs):
        """Time series of scalar or domain-mean variables."""

    def plot_spectrum(self, times, z=None, ax=None, **kwargs):
        """Droplet size distribution at specified times."""

    def plot_dsd_evolution(self, times, ax=None, **kwargs):
        """DSD evolution over multiple time steps."""

    def plot_trajectory(self, pid, variable='position', ax=None, **kwargs):
        """Single-particle trajectory plot (variable vs time)."""

    # --- Utilities ---
    def info(self):
        """Pretty-print simulation metadata."""

    def fields(self) -> list[str]:
        """List available netCDF variables."""

    def close(self):
        """Close netCDF file handles."""
```

### 3.4 `trajectory_io` module

**Status: Implemented.**

Particle trajectory data is stored in `{name}_particles.nc` using a CF
contiguous ragged array (see section 2.4). Reading is straightforward
with xarray — no custom binary parsing needed.

```python
# codt_tools/trajectory_io.py

def load_particles(nc_path) -> xr.Dataset:
    """Open the _particles.nc file. Returns the full Dataset."""

def particles_at_timestep(ds, t) -> xr.Dataset:
    """Extract all particles at time step index t."""

def trajectory_of(ds, pid) -> xr.Dataset:
    """Extract the trajectory of particle with given ID."""

def unique_particle_ids(ds) -> np.ndarray:
    """Return sorted array of unique particle IDs."""

def record_times(ds) -> np.ndarray:
    """Expand per-time-step time to per-record using row_sizes."""
```

---

## 4. Package Structure

```
codt_tools/
├── __init__.py              # Exports: CODTConfig, CODTRunner, CODTSimulation,
│                            #          Namelist, InjectionData, BinData
├── config.py                # Namelist, InjectionData, BinData, CODTConfig
├── runner.py                # CODTRunner
├── simulation.py            # CODTSimulation
├── trajectory_io.py         # NetCDF particle trajectory reader
├── aerosol_io.py            # aerosol_input.nc NetCDF reader/writer
└── plotting.py              # Plot functions (called by Simulation methods)
```

Namelist parsing is handled by `f90nml` in `config.py` (no separate `namelist_io.py`).

---

## 5. Fortran Modifications

### 5.1 Completed (namelist-execution branch)

- Command-line namelist path argument (required, no fallback)
- Relative path resolution from namelist parent directory
- Namelist variables: `aerosol_file`, `bin_data_file`
- `overwrite` namelist parameter
- `output_directory` must be absolute
- Stdout redirect to log file, stderr prints output path only
- `DONE` completion marker with timestamp
- Non-zero exit codes on error
- Namelist verbatim copy (replaces Fortran dump)
- Aerosol input file copied to output directory

### 5.2 Completed (netcdf-improvements branch)

- Namelist parameters as netCDF global attributes (dot-prefixed by namelist group)
- Bin edges stored as `radius_edges` variable (201 values) alongside bin centers
- Fix "celcius" → "celsius" typo in `Tv` units attribute
- Fix `long name` → `long_name` (CF convention) across all variables
- Add missing `long_name` attributes to `z`, `time`, `radius`, `radius_edges`
- Guard particle statistics (`Np`, `Nact`, `Nun`, `Ravg`, `LWC`) and DSD variables
  behind `do_microphysics` — absent from netCDF when microphysics is disabled

### 5.3 Completed (netcdf-writeout-cleanup branch)

- `nf90_sync` after each profile buffer flush in `writeout.f90`
- `nf90_sync` after each particle buffer flush in `write_particle.f90`
- Cached NetCDF variable IDs in particle writer
- Added `aerosol_category` to particle NetCDF output
- Eddy output converted from formatted text (`_eddies.txt`) to unformatted
  binary stream (`_eddies.bin`) with header containing N, H, C2, ZC2, Tdiff, Tref
- Eddy records store raw grid indices (M, L) and dimensional time for direct replay
- Removed manual eddy buffering (OS handles stream I/O)

### 5.4 Completed (aerosol-netcdf-input branch)

- Replaced text-based `injection_data.txt` with NetCDF `aerosol_input.nc`
  (`CODT_aerosol_input_v1` schema)
- Aerosol chemistry stored as per-type variables via `aerosol_type` dimension
- Size distribution uses shared bins with CDF thresholds and category mapping
- Time-varying injection rates and frequencies via `time` dimension
- Namelist variable renamed: `inj_data_file` → `aerosol_file`
- Old text format removed entirely (clean break)

### 5.5 Completed (netcdf-particle-trajectories branch)

- Particle trajectory output converted from binary to NetCDF4 (CF ragged array)
- New `_particles.nc` file with `record` and `time_step` unlimited dimensions
- Per-record variables: particle_id, aerosol_id, gridcell, position, temperature,
  water_vapor, supersaturation, radius, solute_radius, activated
- Per-time-step variables: time, row_sizes (with `cf_role="ragged_row_sizes"`)
- All variables use zlib compression (`deflate_level=1, shuffle=true`)
- Removed legacy binary output: `_particle.bin`, `_PID.bin`, `_particle_meta.txt`
- Removed `w_particle` type, `write_particle_id`, `close_droplets` from Fortran source
- Trajectory calls guarded behind `do_microphysics .and. write_trajectories`

### 5.6 Completed (io-robustness-fixes branch)

- Cleaned up output path variables: `sim_output_dir` and `file_prefix` are now
  explicit globals in `globals.f90`, replacing the overloaded `filename` local
  variable that changed meaning mid-function
  - `sim_output_dir` = `{output_directory}/{simulation_name}/`
  - `file_prefix` = `{sim_output_dir}{simulation_name}` (base path for `.nc`, `.nml`, `.log`)
- `copy_file()` now stops with exit code 1 on error (was silently returning)
- `initial_wet_radius` warning message clarified
- `initialize_microphysics()` no longer takes a filename parameter
- Merged aerosol-netcdf-input changes into this branch

---

## 6. Implementation Status

### Phase 1: Core I/O — Complete
- `aerosol_io.py` — read/write `CODT_aerosol_input_v1` NetCDF schema
- `trajectory_io.py` — CF ragged array reader via xarray
- Namelist parsing via `f90nml` in `config.py`

### Phase 2: CODTSimulation — Complete
- File discovery, metadata from netCDF global attrs
- Dynamic variable access via `__getattr__`
- Derived quantities: `profile`, `time_average`, `domain_average`, `dsd_average`, `spectral_width`, `activation_fraction`
- Plot methods: `plot_timeheight`, `plot_profile`, `plot_timeseries`, `plot_spectrum`, `plot_dsd_evolution`, `plot_trajectory`
- Multi-simulation comparison: `compare()` with auto color scaling (`comparison_colors`)
- Trajectory support: `load_trajectories`, `trajectory_of`, `particles_at_time`, `particle_ids`

### Phase 3: CODTConfig — Complete
- Composes `Namelist`, `InjectionData`, `BinData`
- `write(directory)`, `validate()`, `copy()`, `sweep()`

### Phase 4: CODTRunner — Complete
- `setup_run` / `setup_runs` with `{base}/{sim}/run/` layout
- `run_local` for blocking execution
- `_generate_sbatch` / `submit` with `taskset` core pinning and `dry_run` support
- `status` and `collect` for job monitoring

### Phase 5: Fortran Modifications — Complete
All Fortran-side I/O and execution changes are merged to main. See section 5.

### Remaining Work
- Eddy binary reader (`_eddies.bin`)
- Snakemake example Snakefile (design done, no code yet)

---

## 7. Dependencies

**Python (runtime):**
- `numpy` — array operations, binary I/O
- `xarray` + `netCDF4` — netCDF reading
- `matplotlib` — plotting
- `f90nml` — Fortran namelist parsing (required)

**Python (development):**
- `pytest` — testing

**System:**
- SLURM (`sbatch`, `squeue`, `scancel`)
- Compiled CODT executable

---

## 8. Open Items / Future Work

- **Eddy data reader**: `write_eddies` produces `_eddies.bin` (unformatted stream,
  see section 2.5). A Python reader using `numpy.frombuffer` is straightforward —
  add to `codt_tools` when needed.

- **Snakemake integration**: Design complete (Snakemake manages SLURM, runner
  provides `setup_run`). Example Snakefile not yet written.

- **Restart capability**: Not currently in the framework. Could be added if the model
  supports it.

- **Aerosol category DSDs**: `DSD_1` and `DSD_2` exist. The framework should support
  an arbitrary number of DSD categories (matching the injection category labels).