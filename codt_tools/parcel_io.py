"""Read and write CODT parcel input files (``parcel_input.nc``).

Supports both ``CODT_parcel_input_v1`` (velocity segments only) and
``CODT_parcel_input_v2`` (adds environmental sounding for entrainment).

Functions
---------
read_parcel
    Read a parcel_input.nc file into a plain dict.
write_parcel
    Write a plain dict to a parcel_input.nc file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import netCDF4 as nc
import numpy as np


def read_parcel(path: Union[str, Path]) -> dict[str, Any]:
    """Read a parcel_input.nc file.

    Parameters
    ----------
    path : str or Path
        Path to the parcel NetCDF file.

    Returns
    -------
    dict
        Keys:

        - ``time`` : np.ndarray, shape (n_segments,), seconds
        - ``velocity`` : np.ndarray, shape (n_segments,), m/s
        - ``env_pressure`` : np.ndarray or None, shape (n_levels,), Pa
        - ``env_temperature`` : np.ndarray or None, shape (n_levels,), K
        - ``env_RH`` : np.ndarray or None, shape (n_levels,), 0–1

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the ``conventions`` attribute is not a recognized parcel schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Parcel input file not found: {path}")

    with nc.Dataset(path, "r") as ds:
        conventions = getattr(ds, "conventions", None)
        if conventions not in ("CODT_parcel_input_v1", "CODT_parcel_input_v2"):
            raise ValueError(
                f"Expected CODT_parcel_input_v1 or v2, "
                f"got '{conventions}' in {path}"
            )

        data: dict[str, Any] = {
            "time": ds["time"][:].data.copy(),
            "velocity": ds["velocity"][:].data.copy(),
            "env_pressure": None,
            "env_temperature": None,
            "env_RH": None,
        }

        if conventions == "CODT_parcel_input_v2":
            data["env_pressure"] = ds["env_pressure"][:].data.copy()
            data["env_temperature"] = ds["env_temperature"][:].data.copy()
            data["env_RH"] = ds["env_RH"][:].data.copy()

    return data


def write_parcel(path: Union[str, Path], data: dict[str, Any]) -> None:
    """Write a parcel_input.nc file.

    Writes v1 if no environmental profile is present, v2 if it is.

    Parameters
    ----------
    path : str or Path
        Destination file path. Parent directories are created if needed.
    data : dict
        Must contain ``time`` and ``velocity``. May contain
        ``env_pressure``, ``env_temperature``, ``env_RH`` for v2.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    time = np.atleast_1d(np.asarray(data["time"], dtype=np.float64))
    velocity = np.atleast_1d(np.asarray(data["velocity"], dtype=np.float64))

    has_env = data.get("env_pressure") is not None

    if has_env:
        conventions = "CODT_parcel_input_v2"
        env_pressure = np.asarray(data["env_pressure"], dtype=np.float64)
        env_temperature = np.asarray(data["env_temperature"], dtype=np.float64)
        env_RH = np.asarray(data["env_RH"], dtype=np.float64)
    else:
        conventions = "CODT_parcel_input_v1"

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.conventions = conventions

        ds.createDimension("segment", len(time))

        v = ds.createVariable("time", "f8", ("segment",))
        v.units = "s"
        v[:] = time

        v = ds.createVariable("velocity", "f8", ("segment",))
        v.units = "m/s"
        v[:] = velocity

        if has_env:
            ds.createDimension("level", len(env_pressure))

            v = ds.createVariable("env_pressure", "f8", ("level",))
            v.units = "Pa"
            v[:] = env_pressure

            v = ds.createVariable("env_temperature", "f8", ("level",))
            v.units = "K"
            v[:] = env_temperature

            v = ds.createVariable("env_RH", "f8", ("level",))
            v.units = "1"
            v[:] = env_RH
