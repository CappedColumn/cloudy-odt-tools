# CODT Tools

Python framework for configuring, running, and analyzing simulations from the
Cloudy One-Dimensional Turbulence (CODT) model.

## Features

- **`Case`** — Build simulation input files (namelist, aerosol injection, parcel trajectory), and expand a design into an ensemble of them
- **`Run`** — Stage a case into a run directory and execute it locally; generate launch scripts for a batch (local or SLURM) that you submit yourself
- **`Simulation`** — Load output, compute diagnostics, and produce publication-quality plots. Depends on nothing but the output files
- **Multi-simulation comparison** — Overlay time series, profiles, and spectra across parameter sweeps
- **Simulation registry** — SQLite-backed tracking of experiments and runs (parameters, code versions, status history, data location) with the `codt-registry` CLI

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from codt_tools import Case, Run, Simulation

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
sims = [Simulation(f"/path/to/output/{c.name}") for c in cases]
Simulation.compare(sims, "LWC", plot_type="timeseries")
```

## Recording what you ran

An optional list of the simulations that were run: one SQLite file, one
table, seven columns. Nothing else in codt_tools needs it.

```python
from codt_tools.registry import Registry

with Registry("~/codt_runs.db") as reg:      # created if absent
    reg.add_many(runs, tags="EXP005_seeding")

    for row in reg.list(tag="EXP005"):
        print(row["run_id"], row["code_version"])
```

```bash
codt-registry list --tag EXP005
codt-registry show EXP005_000
```

There is deliberately no status tracking and no experiments table: whether a
run finished is `run.is_complete` (the `_DONE` marker on disk), and grouping
is the free-text `tags` column. See
[docs/registry-quickstart.md](docs/registry-quickstart.md).

## Dependencies

- numpy, xarray, netCDF4, matplotlib, f90nml, pyyaml

## Testing

```bash
pip install -e ".[dev]"
pytest
```
