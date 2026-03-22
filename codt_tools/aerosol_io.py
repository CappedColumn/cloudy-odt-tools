"""Read and write CODT aerosol input files (``aerosol_input.nc``).

The NetCDF4 file follows the ``CODT_aerosol_input_v1`` schema. See CLAUDE.md
section 2.1 for the full specification.

Functions
---------
read_aerosol
    Read an aerosol_input.nc file into a plain dict.
write_aerosol
    Write a plain dict to an aerosol_input.nc file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import netCDF4 as nc
import numpy as np

# Schema version string validated on read
_CONVENTIONS = "CODT_aerosol_input_v1"


def read_aerosol(path: Union[str, Path]) -> dict[str, Any]:
    """Read an aerosol_input.nc file.

    Parameters
    ----------
    path : str or Path
        Path to the aerosol NetCDF file.

    Returns
    -------
    dict
        Keys:

        - ``aerosol_name`` : str
        - ``n_ions`` : np.ndarray, shape (n_types,)
        - ``molar_mass`` : np.ndarray, shape (n_types,)
        - ``solute_density`` : np.ndarray, shape (n_types,)
        - ``edge_radii`` : np.ndarray, shape (n_edges,), nanometres
        - ``category`` : np.ndarray, shape (n_bins,)
        - ``cumulative_frequency`` : np.ndarray, shape (n_times, n_bins)
        - ``injection_time`` : np.ndarray, shape (n_times,), seconds
        - ``injection_rate`` : np.ndarray, shape (n_times,), m⁻³ s⁻¹

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the ``conventions`` global attribute does not match
        ``CODT_aerosol_input_v1``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Aerosol input file not found: {path}")

    with nc.Dataset(path, "r") as ds:
        conventions = getattr(ds, "conventions", None)
        if conventions != _CONVENTIONS:
            raise ValueError(
                f"Expected conventions='{_CONVENTIONS}', "
                f"got '{conventions}' in {path}"
            )

        data: dict[str, Any] = {
            "aerosol_name": getattr(ds, "aerosol_name", ""),
            "n_ions": ds["n_ions"][:].data.copy(),
            "molar_mass": ds["molar_mass"][:].data.copy(),
            "solute_density": ds["solute_density"][:].data.copy(),
            "edge_radii": ds["edge_radii"][:].data.copy(),
            "category": ds["category"][:].data.copy(),
            "cumulative_frequency": ds["cumulative_frequency"][:].data.copy(),
            "injection_time": ds["injection_time"][:].data.copy(),
            "injection_rate": ds["injection_rate"][:].data.copy(),
        }

    return data


def write_aerosol(path: Union[str, Path], data: dict[str, Any]) -> None:
    """Write an aerosol_input.nc file.

    Parameters
    ----------
    path : str or Path
        Destination file path. Parent directories are created if needed.
    data : dict
        Must contain the same keys returned by :func:`read_aerosol`.
        See that function's docstring for the expected shapes and types.

    Raises
    ------
    KeyError
        If a required key is missing from *data*.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_ions = np.atleast_1d(np.asarray(data["n_ions"], dtype=np.int32))
    molar_mass = np.atleast_1d(np.asarray(data["molar_mass"], dtype=np.float64))
    solute_density = np.atleast_1d(np.asarray(data["solute_density"], dtype=np.float64))
    edge_radii = np.asarray(data["edge_radii"], dtype=np.float64)
    category = np.asarray(data["category"], dtype=np.int32)
    cumulative_frequency = np.atleast_2d(
        np.asarray(data["cumulative_frequency"], dtype=np.float64)
    )
    injection_time = np.atleast_1d(
        np.asarray(data["injection_time"], dtype=np.float64)
    )
    injection_rate = np.atleast_1d(
        np.asarray(data["injection_rate"], dtype=np.float64)
    )

    n_types = len(n_ions)
    n_edges = len(edge_radii)
    n_bins = len(category)
    n_times = len(injection_time)

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        # Global attributes
        ds.conventions = _CONVENTIONS
        ds.aerosol_name = str(data["aerosol_name"])

        # Dimensions
        ds.createDimension("aerosol_type", n_types)
        ds.createDimension("edge", n_edges)
        ds.createDimension("bin", n_bins)
        ds.createDimension("time", n_times)

        # Per-type variables
        v = ds.createVariable("n_ions", "i4", ("aerosol_type",))
        v[:] = n_ions

        v = ds.createVariable("molar_mass", "f8", ("aerosol_type",))
        v.units = "kg mol-1"
        v[:] = molar_mass

        v = ds.createVariable("solute_density", "f8", ("aerosol_type",))
        v.units = "kg m-3"
        v[:] = solute_density

        # Bin structure
        v = ds.createVariable("edge_radii", "f8", ("edge",))
        v.units = "nm"
        v[:] = edge_radii

        v = ds.createVariable("category", "i4", ("bin",))
        v[:] = category

        # Time-varying injection
        v = ds.createVariable("cumulative_frequency", "f8", ("time", "bin"))
        v[:] = cumulative_frequency

        v = ds.createVariable("injection_time", "f8", ("time",))
        v.units = "s"
        v[:] = injection_time

        v = ds.createVariable("injection_rate", "f8", ("time",))
        v.units = "m-3 s-1"
        v[:] = injection_rate
