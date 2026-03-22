"""Particle trajectory I/O for CODT simulations.

Reads ``{name}_particles.nc`` files produced by CODT, which use a CF
contiguous ragged array layout (see CLAUDE.md section 2.4).

All functions operate on :class:`xarray.Dataset` objects opened from
these files.  The key idea: all particle records are flattened into a
single ``record`` dimension, and ``row_sizes`` tells you how many
records belong to each time step.

Functions
---------
load_particles
    Open a ``_particles.nc`` file and return the Dataset.
particles_at_timestep
    Extract all particle records at a given time step index.
trajectory_of
    Extract the full trajectory of a single particle by ID.
unique_particle_ids
    Return sorted array of all particle IDs in the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr


def load_particles(path: Union[str, Path]) -> xr.Dataset:
    """Open a ``_particles.nc`` file.

    Parameters
    ----------
    path : str or Path
        Path to the particle trajectory NetCDF file.

    Returns
    -------
    xarray.Dataset
        The full dataset with ``record`` and ``time_step`` dimensions.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Particle file not found: {path}")
    return xr.open_dataset(path, decode_timedelta=False)


def _record_offsets(ds: xr.Dataset) -> np.ndarray:
    """Compute cumulative record offsets from ``row_sizes``.

    Returns an array of length ``n_time_steps + 1`` where
    ``offsets[t]`` is the first record index for time step *t*
    and ``offsets[-1]`` is the total number of records.
    """
    row_sizes = ds["row_sizes"].values
    return np.concatenate([[0], np.cumsum(row_sizes)])


def particles_at_timestep(ds: xr.Dataset, t: int) -> xr.Dataset:
    """Extract all particle records at time step index *t*.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by :func:`load_particles`.
    t : int
        Time step index (0-based).

    Returns
    -------
    xarray.Dataset
        Subset of *ds* along the ``record`` dimension containing only
        the particles at time step *t*.

    Raises
    ------
    IndexError
        If *t* is out of range.
    """
    offsets = _record_offsets(ds)
    n_steps = len(offsets) - 1
    if t < 0 or t >= n_steps:
        raise IndexError(
            f"Time step {t} out of range [0, {n_steps - 1}]"
        )
    start, end = int(offsets[t]), int(offsets[t + 1])
    return ds.isel(record=slice(start, end))


def trajectory_of(ds: xr.Dataset, pid: int) -> xr.Dataset:
    """Extract the full trajectory of a single particle.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by :func:`load_particles`.
    pid : int
        Particle ID to extract.

    Returns
    -------
    xarray.Dataset
        Subset of *ds* along the ``record`` dimension containing only
        records where ``particle_id == pid``.

    Raises
    ------
    ValueError
        If no records with the given *pid* are found.
    """
    mask = ds["particle_id"].values == pid
    if not mask.any():
        raise ValueError(f"No records found for particle_id={pid}")
    return ds.sel(record=mask)


def unique_particle_ids(ds: xr.Dataset) -> np.ndarray:
    """Return sorted array of unique particle IDs.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by :func:`load_particles`.

    Returns
    -------
    numpy.ndarray
        Sorted 1-D array of unique particle IDs.
    """
    return np.unique(ds["particle_id"].values)


def record_times(ds: xr.Dataset) -> np.ndarray:
    """Return the simulation time for each record.

    Expands the per-time-step ``time`` variable to the ``record``
    dimension using ``row_sizes``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by :func:`load_particles`.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``n_records``, giving the simulation time
        (in seconds) for each record.
    """
    return np.repeat(ds["time"].values, ds["row_sizes"].values)
