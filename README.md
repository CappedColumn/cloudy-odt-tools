# CODT Tools

Python framework for configuring, running, and analyzing simulations from the
Cloudy One-Dimensional Turbulence (CODT) model.

## Features

- **`CODTConfig`** — Build simulation input files (namelist, aerosol injection, bin data) with parameter sweeps
- **`CODTRunner`** — Run simulations locally or submit SLURM batch jobs with core pinning
- **`CODTSimulation`** — Load output, compute diagnostics, and produce publication-quality plots
- **Multi-simulation comparison** — Overlay time series, profiles, and spectra across parameter sweeps

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

## Dependencies

- numpy, xarray, netCDF4, matplotlib, f90nml

## Testing

```bash
pip install -e ".[dev]"
pytest
```
