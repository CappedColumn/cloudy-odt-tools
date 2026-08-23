"""CODT Tools — configuration, execution, and analysis of CODT simulations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("codt-tools")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "unknown"

from codt_tools.case import Aerosol, Case, Namelist, Parcel
from codt_tools.run import Run, check_executable, write_local, write_slurm_array
from codt_tools.simulation import Simulation

__all__ = [
    "Aerosol",
    "Case",
    "Namelist",
    "Parcel",
    "Run",
    "Simulation",
    "check_executable",
    "write_local",
    "write_slurm_array",
]
