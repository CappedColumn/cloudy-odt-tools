"""CODT Tools — configuration, execution, and analysis of CODT simulations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("codt-tools")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "unknown"

from codt_tools.case import Aerosol, Case, Namelist, Parcel
from codt_tools.runner import CODTRunner
from codt_tools.simulation import CODTSimulation

__all__ = [
    "Aerosol",
    "CODTRunner",
    "CODTSimulation",
    "Case",
    "Namelist",
    "Parcel",
]
