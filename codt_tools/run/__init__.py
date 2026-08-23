"""The run layer: staging a case into a directory, and launching it.

A :class:`Run` pairs a :class:`~codt_tools.case.Case` with a binary and a
directory. It stages inputs, runs the model locally, and opens the output.

Launching a *batch* is deliberately not something this package does. It
generates a script — :func:`write_local` for this machine,
:func:`write_slurm_array` for a cluster — and you launch it. No ``sbatch``,
``squeue`` or ``sacct`` is called from Python, so nothing here needs to know
your account, your partition, or which cluster you are on.
"""

from codt_tools.run.launcher import write_local, write_slurm_array
from codt_tools.run.run import Run, check_executable

__all__ = [
    "Run",
    "check_executable",
    "write_local",
    "write_slurm_array",
]
