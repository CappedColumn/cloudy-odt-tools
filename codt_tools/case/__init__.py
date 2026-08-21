"""The case layer: a CODT input description and the files it is written to.

``Case`` bundles the three input components — ``Namelist`` (params.nml),
``Aerosol`` (aerosol_input.nc) and ``Parcel`` (parcel_input.nc). It describes
what to simulate; where a run reads and writes is decided by
:meth:`Case.write_inputs`, never carried on the case itself.
"""

from codt_tools.case.aerosol import (
    SEED_HYDRATION_MODES,
    Aerosol,
    make_seed_group,
    read_aerosol,
    write_aerosol,
)
from codt_tools.case.case import (
    AEROSOL_FILENAME,
    PARCEL_FILENAME,
    Case,
)
from codt_tools.case.mutate import apply_point, cross
from codt_tools.case.namelist import Namelist
from codt_tools.case.parcel import (
    PRESSURE_MODES,
    VERTICAL_AXES,
    Parcel,
    read_parcel,
    validate_legs,
    write_parcel,
)
from codt_tools.case.validate import lem_turbulence_scales, validate_case

__all__ = [
    "AEROSOL_FILENAME",
    "Aerosol",
    "Case",
    "Namelist",
    "apply_point",
    "cross",
    "PARCEL_FILENAME",
    "PRESSURE_MODES",
    "Parcel",
    "SEED_HYDRATION_MODES",
    "VERTICAL_AXES",
    "lem_turbulence_scales",
    "make_seed_group",
    "read_aerosol",
    "read_parcel",
    "validate_case",
    "validate_legs",
    "write_aerosol",
    "write_parcel",
]
