"""Read and write CODT aerosol input files (``aerosol_input.nc``).

The NetCDF4 file follows the ``CODT_aerosol_input_v1`` schema. See CLAUDE.md
section 2.1 for the full specification.

A file may carry an optional **seed group**: a second aerosol population with
its own bins, CDF, and release schedule, used when ``&MICROPHYSICS do_seeding``
is enabled. The two populations deliberately do not share a distribution.

Three per-bin labels are easy to conflate:

- ``bin`` — the atom: one sampleable dry radius. The CDF is over bins only.
- ``category(bin)`` — an **output label only**, selecting which ``DSD_n`` the
  particle is counted in.
- ``bin_type(bin)`` — a row of the composition table, setting solute physics.

They are independent: neither ``category`` nor ``bin_type`` subdivides the CDF.
A type is seed material *iff* ``seed_bin_type`` references it — there is no
``is_seed`` flag, so a chemically identical seed needs a **duplicate**
composition row.

Functions
---------
read_aerosol
    Read an aerosol_input.nc file into a plain dict.
write_aerosol
    Write a plain dict to an aerosol_input.nc file.
make_seed_group
    Build and validate a seed group dict for :func:`write_aerosol`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Union

import netCDF4 as nc
import numpy as np

# Schema version string validated on read
_CONVENTIONS = "CODT_aerosol_input_v1"

#: Wet radius modes accepted by ``&MICROPHYSICS seed_hydration``.
SEED_HYDRATION_MODES: tuple[str, ...] = ("equilibrium", "double_growth", "dry")

#: Seed-group variables. All-or-nothing: emit every one, or none.
SEED_KEYS: tuple[str, ...] = (
    "seed_edge_radii",
    "seed_category",
    "seed_bin_type",
    "seed_frequency",
    "seed_coord",
    "seed_concentration",
)


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
        - ``bin_type`` : np.ndarray, shape (n_bins,) — composition row per bin.
          Defaults to all ones when the variable is absent.
        - ``cumulative_frequency`` : np.ndarray, shape (n_times, n_bins)
        - ``injection_time`` : np.ndarray, shape (n_times,), seconds
        - ``injection_rate`` : np.ndarray, shape (n_times,), m⁻³ s⁻¹
        - ``dsd_bin_edges`` : np.ndarray, shape (n_dsd_edges,), microns

        Seed-group keys, each ``None`` when the file carries no seed group:

        - ``seed_edge_radii`` : np.ndarray, shape (n_seed_edges,), nanometres
        - ``seed_category`` : np.ndarray, shape (n_seed_bins,)
        - ``seed_bin_type`` : np.ndarray, shape (n_seed_bins,)
        - ``seed_frequency`` : np.ndarray, shape (n_seed_events, n_seed_bins)
        - ``seed_coord`` : np.ndarray, shape (n_seed_events,), s / m / Pa
        - ``seed_concentration`` : np.ndarray, shape (n_seed_events,), cm⁻³

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the ``conventions`` global attribute does not match
        ``CODT_aerosol_input_v1``, or the seed group is partially present.
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

        category = ds["category"][:].data.copy()

        data: dict[str, Any] = {
            "aerosol_name": getattr(ds, "aerosol_name", ""),
            "n_ions": ds["n_ions"][:].data.copy(),
            "molar_mass": ds["molar_mass"][:].data.copy(),
            "solute_density": ds["solute_density"][:].data.copy(),
            "edge_radii": ds["edge_radii"][:].data.copy(),
            "category": category,
            "cumulative_frequency": ds["cumulative_frequency"][:].data.copy(),
            "injection_time": ds["injection_time"][:].data.copy(),
            "injection_rate": ds["injection_rate"][:].data.copy(),
            "dsd_bin_edges": ds["dsd_bin_edges"][:].data.copy(),
        }

        # bin_type is optional; absent means every bin is composition row 1.
        if "bin_type" in ds.variables:
            data["bin_type"] = ds["bin_type"][:].data.copy()
        else:
            data["bin_type"] = np.ones(len(category), dtype=np.int32)

        present = [key for key in SEED_KEYS if key in ds.variables]
        if present and len(present) != len(SEED_KEYS):
            missing = sorted(set(SEED_KEYS) - set(present))
            raise ValueError(
                f"{path} has a partial seed group: found {sorted(present)} "
                f"but is missing {missing}. The seed group is all-or-nothing."
            )
        for key in SEED_KEYS:
            data[key] = ds[key][:].data.copy() if present else None

    return data


def make_seed_group(
    seed_edge_radii: np.ndarray | list,
    seed_category: np.ndarray | list,
    seed_bin_type: np.ndarray | list,
    seed_frequency: np.ndarray | list,
    seed_coord: np.ndarray | list,
    seed_concentration: np.ndarray | list,
    n_types: int,
    bin_type: np.ndarray | list | None = None,
    category: np.ndarray | list | None = None,
) -> dict[str, Any]:
    """Build and validate a seed group for :func:`write_aerosol`.

    Mirrors the rules CODT enforces at read time so a malformed seed group is
    rejected here rather than aborting a submitted run.

    Parameters
    ----------
    seed_edge_radii : array-like
        Seed bin edge radii in nm. Length must be ``len(seed_category) + 1``.
    seed_category : array-like
        Output category per seed bin. Categories are not checked for
        collisions with the background; see the warning note below.
    seed_bin_type : array-like
        Composition row per seed bin, in ``1..n_types``. A type is seed
        material *iff* it is referenced here, so a chemically identical seed
        needs a duplicate composition row.
    seed_frequency : array-like
        Cumulative size distribution per event, shape
        ``(n_seed_events, n_seed_bins)`` in netCDF declaration order. Each
        event's row must run to 1.0.
    seed_coord : array-like
        Axis value triggering each event: time (s) in chamber mode, height (m)
        or pressure (Pa) in parcel mode per ``&PARCEL vertical_axis``. Events
        fire once, on first arrival. Duplicates are rejected.
    seed_concentration : array-like
        Concentration released by each event, in cm⁻³. Must be >= 0.
    n_types : int
        Number of rows in the composition table (the ``aerosol_type`` length).
    bin_type : array-like, optional
        The background's ``bin_type``. When given, enables the checks that no
        type is referenced by both groups and that every type is referenced.
    category : array-like, optional
        The background's ``category``. When given, warns on category values
        shared with the seed group.

    Returns
    -------
    dict
        Seed-group keys ready to merge into a :func:`write_aerosol` dict.

    Raises
    ------
    ValueError
        If any of CODT's seed-group rules is violated.

    Warns
    -----
    UserWarning
        If a seed category collides with a background category. CODT does not
        check this, and the two silently merge into the same ``DSD_n``.
    """
    seed_edge_radii = np.atleast_1d(np.asarray(seed_edge_radii, dtype=np.float64))
    seed_category = np.atleast_1d(np.asarray(seed_category, dtype=np.int32))
    seed_bin_type = np.atleast_1d(np.asarray(seed_bin_type, dtype=np.int32))
    seed_frequency = np.atleast_2d(np.asarray(seed_frequency, dtype=np.float64))
    seed_coord = np.atleast_1d(np.asarray(seed_coord, dtype=np.float64))
    seed_concentration = np.atleast_1d(
        np.asarray(seed_concentration, dtype=np.float64)
    )

    n_seed_bins = len(seed_category)
    n_seed_events = len(seed_coord)

    if len(seed_bin_type) != n_seed_bins:
        raise ValueError(
            f"seed_bin_type must have one value per seed bin ({n_seed_bins}), "
            f"got {len(seed_bin_type)}"
        )
    if len(seed_edge_radii) != n_seed_bins + 1:
        raise ValueError(
            f"seed_edge_radii must have seed_bin + 1 = {n_seed_bins + 1} "
            f"values, got {len(seed_edge_radii)}"
        )
    if seed_frequency.shape != (n_seed_events, n_seed_bins):
        raise ValueError(
            f"seed_frequency must have shape (n_seed_events, n_seed_bins) = "
            f"{(n_seed_events, n_seed_bins)}, got {seed_frequency.shape}. "
            f"Note this is netCDF declaration order; CODT's docs list the "
            f"reverse (Fortran) order."
        )
    if len(seed_concentration) != n_seed_events:
        raise ValueError(
            f"seed_concentration must have one value per event "
            f"({n_seed_events}), got {len(seed_concentration)}"
        )

    if not np.allclose(seed_frequency[:, -1], 1.0):
        raise ValueError("each seed_frequency event row must run to 1.0")
    if np.any(seed_concentration < 0.0):
        raise ValueError("seed_concentration must be >= 0")
    if len(np.unique(seed_coord)) != n_seed_events:
        raise ValueError(
            "duplicate seed_coord values are rejected; write one larger event "
            "instead"
        )
    if np.any(seed_bin_type < 1) or np.any(seed_bin_type > n_types):
        raise ValueError(f"seed_bin_type values must lie in 1..{n_types}")

    if bin_type is not None:
        bin_type = np.atleast_1d(np.asarray(bin_type, dtype=np.int32))
        shared = sorted(set(bin_type.tolist()) & set(seed_bin_type.tolist()))
        if shared:
            raise ValueError(
                f"composition row(s) {shared} are referenced by both bin_type "
                f"and seed_bin_type; a type is seed material iff seed_bin_type "
                f"references it, so seeding a chemical twin of the background "
                f"needs a duplicate composition row"
            )
        referenced = set(bin_type.tolist()) | set(seed_bin_type.tolist())
        unreferenced = sorted(set(range(1, n_types + 1)) - referenced)
        if unreferenced:
            raise ValueError(
                f"composition row(s) {unreferenced} are referenced by no bin "
                f"in either group; every type must be referenced at least once"
            )

    if category is not None:
        category = np.atleast_1d(np.asarray(category, dtype=np.int32))
        collisions = sorted(set(category.tolist()) & set(seed_category.tolist()))
        if collisions:
            warnings.warn(
                f"seed categories {collisions} are also used by the background; "
                f"CODT does not check this and both will silently merge into "
                f"the same DSD_n. Number seed categories after the "
                f"background's unless merging is intended.",
                stacklevel=2,
            )

    return {
        "seed_edge_radii": seed_edge_radii,
        "seed_category": seed_category,
        "seed_bin_type": seed_bin_type,
        "seed_frequency": seed_frequency,
        "seed_coord": seed_coord,
        "seed_concentration": seed_concentration,
    }


def write_aerosol(path: Union[str, Path], data: dict[str, Any]) -> None:
    """Write an aerosol_input.nc file.

    Parameters
    ----------
    path : str or Path
        Destination file path. Parent directories are created if needed.
    data : dict
        Must contain the same keys returned by :func:`read_aerosol`.
        See that function's docstring for the expected shapes and types.
        ``bin_type`` is optional and defaults to all ones. The seed group is
        optional and all-or-nothing; build it with :func:`make_seed_group`.

    Raises
    ------
    KeyError
        If a required key is missing from *data*.
    ValueError
        If ``bin_type`` is the wrong length or out of range, or the seed group
        is partially present.

    Notes
    -----
    ``&MICROPHYSICS do_seeding`` is the sole controller. ``do_seeding = .true.``
    with no seed group is **fatal** in CODT. ``do_seeding = .false.`` with a seed
    group present is fine — CODT ignores the dormant group (and warns), so a file
    written with a seed group can still run unseeded.
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
    dsd_bin_edges = np.asarray(data["dsd_bin_edges"], dtype=np.float64)

    n_types = len(n_ions)
    n_edges = len(edge_radii)
    n_bins = len(category)
    n_times = len(injection_time)
    n_dsd_edges = len(dsd_bin_edges)

    # bin_type is optional on input; write it explicitly either way.
    if data.get("bin_type") is None:
        bin_type = np.ones(n_bins, dtype=np.int32)
    else:
        bin_type = np.atleast_1d(np.asarray(data["bin_type"], dtype=np.int32))
        if len(bin_type) != n_bins:
            raise ValueError(
                f"bin_type must have one value per bin ({n_bins}), "
                f"got {len(bin_type)}"
            )
        if np.any(bin_type < 1) or np.any(bin_type > n_types):
            raise ValueError(f"bin_type values must lie in 1..{n_types}")

    seed_present = [key for key in SEED_KEYS if data.get(key) is not None]
    if seed_present and len(seed_present) != len(SEED_KEYS):
        missing = sorted(set(SEED_KEYS) - set(seed_present))
        raise ValueError(
            f"partial seed group: got {sorted(seed_present)} but missing "
            f"{missing}. The seed group is all-or-nothing; build it with "
            f"make_seed_group()."
        )
    seed: dict[str, np.ndarray] | None = None
    if seed_present:
        seed = {
            "seed_edge_radii": np.atleast_1d(
                np.asarray(data["seed_edge_radii"], dtype=np.float64)),
            "seed_category": np.atleast_1d(
                np.asarray(data["seed_category"], dtype=np.int32)),
            "seed_bin_type": np.atleast_1d(
                np.asarray(data["seed_bin_type"], dtype=np.int32)),
            "seed_frequency": np.atleast_2d(
                np.asarray(data["seed_frequency"], dtype=np.float64)),
            "seed_coord": np.atleast_1d(
                np.asarray(data["seed_coord"], dtype=np.float64)),
            "seed_concentration": np.atleast_1d(
                np.asarray(data["seed_concentration"], dtype=np.float64)),
        }

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        # Global attributes
        ds.conventions = _CONVENTIONS
        ds.aerosol_name = str(data["aerosol_name"])

        # Dimensions
        ds.createDimension("aerosol_type", n_types)
        ds.createDimension("edge", n_edges)
        ds.createDimension("bin", n_bins)
        ds.createDimension("time", n_times)
        ds.createDimension("dsd_edge", n_dsd_edges)

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

        v = ds.createVariable("bin_type", "i4", ("bin",))
        v.long_name = "aerosol_type row for each bin"
        v[:] = bin_type

        # Time-varying injection
        v = ds.createVariable("cumulative_frequency", "f8", ("time", "bin"))
        v[:] = cumulative_frequency

        v = ds.createVariable("injection_time", "f8", ("time",))
        v.units = "s"
        v[:] = injection_time

        v = ds.createVariable("injection_rate", "f8", ("time",))
        v.units = "m-3 s-1"
        v[:] = injection_rate

        v = ds.createVariable("dsd_bin_edges", "f8", ("dsd_edge",))
        v.units = "um"
        v[:] = dsd_bin_edges

        # Seed group: a second population with its own bins, CDF, and schedule.
        if seed is not None:
            n_seed_bins = len(seed["seed_category"])
            n_seed_events = len(seed["seed_coord"])

            ds.createDimension("seed_bin", n_seed_bins)
            ds.createDimension("seed_edge", n_seed_bins + 1)
            ds.createDimension("seed_event", n_seed_events)

            v = ds.createVariable("seed_edge_radii", "f8", ("seed_edge",))
            v.units = "nm"
            v[:] = seed["seed_edge_radii"]

            v = ds.createVariable("seed_category", "i4", ("seed_bin",))
            v.long_name = "output category for each seed bin"
            v[:] = seed["seed_category"]

            v = ds.createVariable("seed_bin_type", "i4", ("seed_bin",))
            v.long_name = "aerosol_type row for each seed bin"
            v[:] = seed["seed_bin_type"]

            v = ds.createVariable(
                "seed_frequency", "f8", ("seed_event", "seed_bin")
            )
            v.long_name = "cumulative size distribution per seeding event"
            v[:] = seed["seed_frequency"]

            v = ds.createVariable("seed_coord", "f8", ("seed_event",))
            v.long_name = "axis value triggering each seeding event"
            v[:] = seed["seed_coord"]

            v = ds.createVariable("seed_concentration", "f8", ("seed_event",))
            v.units = "cm-3"
            v[:] = seed["seed_concentration"]
