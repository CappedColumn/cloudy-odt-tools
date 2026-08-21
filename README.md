# CODT Tools

Python framework for configuring, running, and analyzing simulations from the
Cloudy One-Dimensional Turbulence (CODT) model.

## Features

- **`Case`** — Build simulation input files (namelist, aerosol injection, parcel trajectory) with parameter sweeps
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
from codt_tools import Case, CODTRunner, CODTSimulation

# Configure
case = Case()
case.set(tmax=3600, tref=21.0, do_microphysics=True)

# Run
runner = CODTRunner(
    executable="/path/to/CODT",
    base_output_dir="/path/to/output",
    account="my-account",
    partition="my-partition",
)
runner.run_local(case)

# Analyze
sim = CODTSimulation("/path/to/output/default_sim")
sim.plot_timeseries("LWC")
sim.plot_timeheight("T")

# Compare multiple runs
cases = Case.sweep(case, tref=[20.0, 21.0, 22.0])
sims = [CODTSimulation(f"/path/to/output/{c.name}") for c in cases]
CODTSimulation.compare(sims, "LWC", plot_type="timeseries")
```

## Simulation Registry

Track every run in a single SQLite database: full namelist parameters,
executable checksum and `--version`, status history, and where the data
lives. Attach a registry to a runner and every run it stages is recorded:

```python
from codt_tools.registry import Registry

with Registry("~/codt_registry.db") as reg:
    reg.create_experiment("EXP001", "Title", data_root="/path/to/data")
    runner = CODTRunner("/path/to/CODT", "/path/to/data/EXP001/runs",
                        account="my-account", partition="my-partition",
                        registry=reg, experiment_id="EXP001")
    run_dirs = runner.setup_runs(cases)
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
