# CODT Tools

Python framework for configuring, running, and analyzing simulations from the
Cloudy One-Dimensional Turbulence (CODT) model.

## Features

- **`CODTConfig`** — Build simulation input files (namelist, aerosol injection, bin data) with parameter sweeps
- **`CODTRunner`** — Run simulations locally or submit SLURM batch jobs with core pinning
- **`CODTSimulation`** — Load output, compute diagnostics, and produce publication-quality plots
- **Multi-simulation comparison** — Overlay time series, profiles, and spectra across parameter sweeps
- **Simulation registry** — SQLite-backed tracking of experiments and runs (parameters, code versions, status history, data location) with the `codt-registry` CLI

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from codt_tools import CODTConfig, CODTRunner, CODTSimulation

# Configure
config = CODTConfig()
config.set(tmax=3600, tref=21.0, do_microphysics=True)

# Run
runner = CODTRunner(
    executable="/path/to/CODT",
    base_output_dir="/path/to/output",
    account="my-account",
    partition="my-partition",
)
runner.run_local(config)

# Analyze
sim = CODTSimulation("/path/to/output/default_sim")
sim.plot_timeseries("LWC")
sim.plot_timeheight("T")

# Compare multiple runs
configs = CODTConfig.sweep(config, tref=[20.0, 21.0, 22.0])
sims = [CODTSimulation(f"/path/to/output/{c.name}") for c in configs]
CODTSimulation.compare(sims, "LWC", plot_type="timeseries")
```

## Simulation Registry

Track every run in a single SQLite database: full namelist parameters,
executable checksum and `--version`, status history, and where the data
lives. Experiments are defined declaratively in YAML (base config +
parameter sweep) and expanded into registered runs:

```python
from codt_tools import ExperimentSpec, create_experiment_runs
from codt_tools.registry import Registry

spec = ExperimentSpec.from_yaml("experiment.yaml")
with Registry("~/codt_registry.db") as reg:
    runner, run_dirs = create_experiment_runs(spec, reg, "/path/to/CODT")
    runner.submit(run_dirs, walltime="12:00:00")
```

```bash
codt-registry list --experiment EXP001 --status failed
codt-registry export --experiment EXP001 --csv runs.csv
```

See [docs/registry-quickstart.md](docs/registry-quickstart.md) for the
full define → create → run → query → conclude workflow,
[docs/registry-schema.md](docs/registry-schema.md) for the schema, and
[docs/using-a-shared-registry.md](docs/using-a-shared-registry.md) for
shared/group databases.

## Dependencies

- numpy, xarray, netCDF4, matplotlib, f90nml, pyyaml

## Testing

```bash
pip install -e ".[dev]"
pytest
```
