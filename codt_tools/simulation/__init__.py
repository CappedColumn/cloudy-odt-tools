"""The simulation layer: reading and analyzing completed CODT output.

``Simulation`` is the whole interface — it discovers a run's output files
from one path, exposes every field as an attribute, and carries the
profiles, averages, budgets, spectra and plots built on them.

Analysis is read-only and self-contained. Nothing here imports the registry,
so reading your own data never requires the bookkeeping layer.
"""

from codt_tools.simulation.plotting import (
    comparison_colors,
    ensure_ax,
    get_label,
    plot_dsd_evolution,
    plot_profile,
    plot_spectrum,
    plot_timeheight,
    plot_timeseries,
)
from codt_tools.simulation.simulation import OUTPUT_CONVENTIONS, Simulation
from codt_tools.simulation.trajectory import (
    load_particles,
    particles_at_timestep,
    record_times,
    trajectory_of,
    unique_particle_ids,
)

__all__ = [
    "OUTPUT_CONVENTIONS",
    "Simulation",
    "comparison_colors",
    "ensure_ax",
    "get_label",
    "load_particles",
    "particles_at_timestep",
    "plot_dsd_evolution",
    "plot_profile",
    "plot_spectrum",
    "plot_timeheight",
    "plot_timeseries",
    "record_times",
    "trajectory_of",
    "unique_particle_ids",
]
