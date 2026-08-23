# CODT Tools

Python framework for configuring, running, and analyzing simulations from the
Cloudy One-Dimensional Turbulence (CODT) model.

## Features

- **`Case`** — Build simulation input files (namelist, aerosol injection, parcel trajectory), and expand a design into an ensemble of them
- **`Run`** — Stage a case into a run directory and execute it locally; generate launch scripts for a batch (local or SLURM) that you submit yourself
- **`CODTSimulation`** — Load output, compute diagnostics, and produce publication-quality plots
- **Multi-simulation comparison** — Overlay time series, profiles, and spectra across parameter sweeps
- **Simulation registry** — SQLite-backed tracking of experiments and runs (parameters, code versions, status history, data location) with the `codt-registry` CLI

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from codt_tools import Case, CODTSimulation, Run

# Configure
case = Case()
case.set(tmax=3600, tref=21.0, do_microphysics=True)

# Run
run = Run(case, executable="/path/to/CODT", workdir="/path/to/output/control")
run.stage()          # writes inputs/, creates output/
run.execute_local()  # blocking

# Analyze
sim = run.open_simulation()
sim.plot_timeseries("LWC")
sim.plot_timeheight("T")

# Compare multiple runs
cases = case.sweep({"params.tref": [20.0, 21.0, 22.0]})   # see docs/designs.md
sims = [CODTSimulation(f"/path/to/output/{c.name}") for c in cases]
CODTSimulation.compare(sims, "LWC", plot_type="timeseries")
```

## Simulation Registry

Track every run in a single SQLite database: full namelist parameters,
executable checksum and `--version`, status history, and where the data
lives. Registration is an explicit call, so what gets recorded is up to you:

```python
from codt_tools.registry import Registry
from codt_tools import Run, write_slurm_array

with Registry("~/codt_registry.db") as reg:
    reg.create_experiment("EXP001", "Title", data_root="/path/to/data")

    runs = Run.for_cases(cases, "/path/to/CODT", "/path/to/data/EXP001/runs")
    for run in runs:
        staged = run.stage()
        reg.register_run(run.name, run.case, run.workdir,
                         namelist=staged, experiment_id="EXP001")

    write_slurm_array(runs, "/path/to/data/EXP001/array.sh",
                      runs_per_task=64, account="my-account",
                      partition="my-partition", time="12:00:00")
    # then, yourself:  sbatch /path/to/data/EXP001/array.sh
```

```bash
codt-registry list --experiment EXP001 --status failed
codt-registry export --experiment EXP001 --csv runs.csv
```

See [docs/designs.md](docs/designs.md) for describing an ensemble —
including conditional, ragged and sampled designs —
[docs/registry-quickstart.md](docs/registry-quickstart.md) for the
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
