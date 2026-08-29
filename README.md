# CODT Tools

Python framework for configuring, running, and analyzing simulations from the
Cloudy One-Dimensional Turbulence (CODT) model.

## The shape of it

**A case describes a simulation. A run stages one into a directory and
executes it. A simulation reads the output. The registry is optional.**

| | |
|---|---|
| **`Case`** | The complete input description — namelist, aerosol, parcel trajectory. Expands into an ensemble via a *design*. Knows nothing about where a `Run` will stage it. |
| **`Run`** | One case + one binary + one directory. Stages inputs, runs the model locally, opens the output. Knows nothing about SLURM. |
| **`Simulation`** | Read-only analysis: fields, profiles, averages, budgets, spectra, DSDs, trajectories, plots, and multi-run comparison. Depends on nothing but the output files. |
| **`Registry`** | An optional list of the runs you have done. One table, seven columns. |

Batch execution is **artifact generation**. codt_tools writes a launch script,
local or SLURM. You run it. Nothing here calls `sbatch`.

New to it? Start with
[docs/starting-a-project.md](docs/starting-a-project.md) — a project is three
short scripts, and that page has all three.

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
run = Run(case, executable="/path/to/CODT", workdir="/path/to/runs/control")
run.stage()          # writes inputs/, creates output/
run.execute_local()  # blocking

# Analyze
sim = run.open_simulation()
sim.plot_timeseries("LWC")
sim.plot_timeheight("T")

# Compare multiple runs
cases = case.sweep({"params.tref": [20.0, 21.0, 22.0]})   # see docs/designs.md
# stage() writes the .nc into workdir/output/. Point Simulation at that dir.
sims = [Simulation(f"/path/to/runs/{c.name}/output") for c in cases]
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

The registry deliberately has no status tracking and no experiments table.
`run.is_complete` tells you whether a run finished, by looking for the `_DONE`
marker on disk. The free-text `tags` column does the grouping. See
[docs/registry-quickstart.md](docs/registry-quickstart.md).

## Documentation

| | |
|---|---|
| [starting-a-project.md](docs/starting-a-project.md) | the three-script project layout — start here |
| [designs.md](docs/designs.md) | expressing an ensemble: points, designs, `cross`, and the irregular cases |
| [running-on-slurm.md](docs/running-on-slurm.md) | generating a batch script, sizing a task, preemption |
| [file-formats.md](docs/file-formats.md) | what CODT reads and writes, and where it looks |
| [registry-quickstart.md](docs/registry-quickstart.md) | recording what you ran |
| [CHANGELOG.md](CHANGELOG.md) | version history — **and facts needed to read archived output** |

Requires a **CODT 3.1.0+** binary.

## Dependencies

- numpy, xarray, netCDF4, matplotlib, f90nml

## Testing

```bash
pip install -e ".[dev]"
pytest
```
