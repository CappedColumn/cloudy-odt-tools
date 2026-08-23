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

``Aerosol`` is the in-memory form; the three functions below are the file
format itself, kept public because they are the format's only specification.

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


# ======================================================================
# Aerosol
# ======================================================================


class Aerosol:
    """In-memory representation of a CODT aerosol injection specification.

    Reads and writes the ``aerosol_input.nc`` NetCDF4 format
    (``CODT_aerosol_input_v1`` schema).

    Parameters
    ----------
    path : str or Path, optional
        Path to an existing ``aerosol_input.nc`` file. If ``None``, creates
        an instance with default NaCl values.

    Examples
    --------
    Load from file:

    >>> inj = Aerosol("input/aerosol_input.nc")
    >>> inj.aerosol_name
    'NaCl'

    Create from defaults and customise — scalars are wrapped automatically:

    >>> inj = Aerosol()
    >>> inj.set(aerosol_name="KCl", n_ions=2,
    ...         molar_mass=0.07455, solute_density=1984.0)
    >>> inj.set(injection_rate=5.5e5,
    ...         edge_radii=[60.0, 70.0, 4930.0],
    ...         cumulative_frequency=[1.0, 1.0])
    >>> inj.write("run_dir/aerosol_input.nc")
    """

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        if path is None:
            self._init_defaults()
        else:
            self._read(Path(path))

    def _init_defaults(self) -> None:
        """Populate with default NaCl aerosol values."""
        self.aerosol_name: str = "NaCl"
        self.n_ions: np.ndarray = np.array([2], dtype=np.int32)
        self.molar_mass: np.ndarray = np.array([58.4428e-3])
        self.solute_density: np.ndarray = np.array([2.163e3])
        self.edge_radii: np.ndarray = np.array([291.0, 500.0, 1000.0])
        self.category: np.ndarray = np.array([1, 2], dtype=np.int32)
        self.bin_type: np.ndarray = np.array([1, 1], dtype=np.int32)
        self.cumulative_frequency: np.ndarray = np.array([[0.5, 1.0]])
        self.injection_time: np.ndarray = np.array([0.0])
        self.injection_rate: np.ndarray = np.array([6.66e4])
        self.dsd_bin_edges: np.ndarray = np.geomspace(0.049, 60.0, num=201)
        self._init_seed_defaults()

    def _init_seed_defaults(self) -> None:
        """Clear the optional seed group."""
        self.seed_edge_radii: np.ndarray | None = None
        self.seed_category: np.ndarray | None = None
        self.seed_bin_type: np.ndarray | None = None
        self.seed_frequency: np.ndarray | None = None
        self.seed_coord: np.ndarray | None = None
        self.seed_concentration: np.ndarray | None = None

    def _read(self, path: Path) -> None:
        """Read from an aerosol_input.nc file."""
        data = read_aerosol(path)
        self.aerosol_name = data["aerosol_name"]
        self.n_ions = data["n_ions"]
        self.molar_mass = data["molar_mass"]
        self.solute_density = data["solute_density"]
        self.edge_radii = data["edge_radii"]
        self.category = data["category"]
        self.bin_type = data["bin_type"]
        self.cumulative_frequency = data["cumulative_frequency"]
        self.injection_time = data["injection_time"]
        self.injection_rate = data["injection_rate"]
        self.dsd_bin_edges = data["dsd_bin_edges"]
        for key in SEED_KEYS:
            setattr(self, key, data[key])

    # ------------------------------------------------------------------
    # Access / modification
    # ------------------------------------------------------------------

    # Fields that are 1D arrays in the NetCDF schema.  Scalars passed for
    # these are wrapped in a length-1 array automatically.
    _INT_1D_FIELDS: set[str] = {"n_ions", "category", "bin_type"}
    _FLOAT_1D_FIELDS: set[str] = {
        "molar_mass", "solute_density", "edge_radii",
        "injection_time", "injection_rate", "dsd_bin_edges",
    }
    # cumulative_frequency is always 2D (time, bin).
    _FLOAT_2D_FIELDS: set[str] = {"cumulative_frequency"}

    def set(self, **kwargs: Any) -> None:
        """Set one or more attributes by name.

        Scalars are automatically wrapped into arrays where appropriate.
        For example, ``n_ions=2`` becomes ``np.array([2])``, and a 1-D
        ``cumulative_frequency`` is promoted to 2-D (single time step).

        Parameters
        ----------
        **kwargs
            Attribute name-value pairs.

        Raises
        ------
        AttributeError
            If an attribute name does not exist.

        Examples
        --------
        >>> inj.set(aerosol_name="KCl", n_ions=2,
        ...         molar_mass=0.07455, solute_density=1984.0)
        >>> inj.set(injection_rate=1.0e5)
        >>> inj.set(cumulative_frequency=[0.3, 0.7, 1.0])  # single time step
        """
        for key, value in kwargs.items():
            if key not in self._field_names():
                raise AttributeError(
                    f"'{key}' is not a valid Aerosol attribute. "
                    f"Valid attributes: {self._field_names()}"
                )
            value = self._coerce(key, value)
            setattr(self, key, value)

    @staticmethod
    def _coerce(key: str, value: Any) -> Any:
        """Coerce *value* to the expected type/shape for *key*.

        For ``cumulative_frequency``, if the values look like a PDF
        (rows sum to ~1 but the last element is not ~1), they are
        converted to a CDF via ``cumsum``.  A warning is issued if the
        final CDF values do not equal 1.
        """
        if key == "aerosol_name":
            return str(value)

        if key in Aerosol._INT_1D_FIELDS:
            return np.atleast_1d(np.asarray(value, dtype=np.int32))

        if key in Aerosol._FLOAT_1D_FIELDS:
            return np.atleast_1d(np.asarray(value, dtype=np.float64))

        if key in Aerosol._FLOAT_2D_FIELDS:
            arr = np.atleast_2d(np.asarray(value, dtype=np.float64))
            arr = Aerosol._validate_cdf(arr)
            return arr

        return value

    @staticmethod
    def _validate_cdf(arr: np.ndarray) -> np.ndarray:
        """Check cumulative_frequency and auto-convert PDF to CDF.

        Parameters
        ----------
        arr : np.ndarray
            Shape (n_times, n_bins). Each row should be a CDF ending at 1.

        Returns
        -------
        np.ndarray
            Validated (and possibly converted) CDF array.
        """
        for i, row in enumerate(arr):
            last = row[-1]
            row_sum = row.sum()

            if np.isclose(last, 1.0):
                # Already a valid CDF — check monotonicity
                if np.any(np.diff(row) < -1e-12):
                    warnings.warn(
                        f"cumulative_frequency row {i} is not monotonically "
                        f"non-decreasing: {row}. Expected a CDF.",
                        UserWarning,
                        stacklevel=4,
                    )
                continue

            # Last value is not 1 — check if it looks like a PDF
            if np.isclose(row_sum, 1.0) and np.all(row >= 0):
                arr[i] = np.cumsum(row)
                warnings.warn(
                    f"cumulative_frequency row {i} looks like a PDF "
                    f"(sums to {row_sum:.6g}). Converted to CDF via cumsum.",
                    UserWarning,
                    stacklevel=4,
                )
            else:
                warnings.warn(
                    f"cumulative_frequency row {i} does not end at 1.0 "
                    f"(last value = {last:.6g}, sum = {row_sum:.6g}). "
                    f"Expected a CDF with final value 1.0.",
                    UserWarning,
                    stacklevel=4,
                )

        return arr

    @property
    def n_types(self) -> int:
        """Number of aerosol types."""
        return len(self.n_ions)

    @property
    def n_bins(self) -> int:
        """Number of aerosol size bins."""
        return len(self.category)

    @property
    def n_edges(self) -> int:
        """Number of bin edges."""
        return len(self.edge_radii)

    @property
    def n_times(self) -> int:
        """Number of injection time steps."""
        return len(self.injection_time)

    def _field_names(self) -> list[str]:
        """Return list of public attribute names."""
        return [
            "aerosol_name", "n_ions", "molar_mass", "solute_density",
            "edge_radii", "category", "bin_type", "cumulative_frequency",
            "injection_time", "injection_rate", "dsd_bin_edges",
            *SEED_KEYS,
        ]

    @property
    def has_seed_group(self) -> bool:
        """Whether a seed group is present.

        ``&MICROPHYSICS do_seeding`` is the sole controller: seeding on with no
        group is fatal, but a group present with seeding off is ignored (CODT
        warns), so one file can serve both a seeded and an unseeded run.
        """
        return self.seed_coord is not None

    def set_seed_group(self, **kwargs: Any) -> None:
        """Build, validate, and attach a seed group.

        Thin wrapper over :func:`aerosol_io.make_seed_group` that supplies the
        background's ``n_types``, ``bin_type``, and ``category`` so the
        cross-population rules are checked. See that function for the
        arguments and the rules enforced.
        """
        kwargs.setdefault("n_types", self.n_types)
        kwargs.setdefault("bin_type", self.bin_type)
        kwargs.setdefault("category", self.category)
        for key, value in make_seed_group(**kwargs).items():
            setattr(self, key, value)

    def clear_seed_group(self) -> None:
        """Remove the seed group."""
        self._init_seed_defaults()

    # ------------------------------------------------------------------
    # Conversion to/from dict (for aerosol_io)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to the dict format used by :mod:`aerosol_io`.

        Returns
        -------
        dict
            Keys match those returned by :func:`aerosol_io.read_aerosol`.
        """
        return {name: getattr(self, name) for name in self._field_names()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Aerosol":
        """Create an instance from a dict (as returned by ``read_aerosol``).

        Parameters
        ----------
        data : dict
            Must contain all keys from :func:`aerosol_io.read_aerosol`.
        """
        obj = cls.__new__(cls)
        obj.aerosol_name = data["aerosol_name"]
        obj.n_ions = np.atleast_1d(np.asarray(data["n_ions"], dtype=np.int32))
        obj.molar_mass = np.atleast_1d(np.asarray(data["molar_mass"], dtype=np.float64))
        obj.solute_density = np.atleast_1d(np.asarray(data["solute_density"], dtype=np.float64))
        obj.edge_radii = np.asarray(data["edge_radii"], dtype=np.float64)
        obj.category = np.asarray(data["category"], dtype=np.int32)
        obj.cumulative_frequency = np.atleast_2d(
            np.asarray(data["cumulative_frequency"], dtype=np.float64)
        )
        obj.injection_time = np.atleast_1d(
            np.asarray(data["injection_time"], dtype=np.float64)
        )
        obj.injection_rate = np.atleast_1d(
            np.asarray(data["injection_rate"], dtype=np.float64)
        )
        obj.dsd_bin_edges = np.asarray(data["dsd_bin_edges"], dtype=np.float64)

        # bin_type is optional; absent means every bin is composition row 1.
        if data.get("bin_type") is None:
            obj.bin_type = np.ones(len(obj.category), dtype=np.int32)
        else:
            obj.bin_type = np.asarray(data["bin_type"], dtype=np.int32)

        obj._init_seed_defaults()
        for key in SEED_KEYS:
            if data.get(key) is not None:
                dtype = (
                    np.int32
                    if key in ("seed_category", "seed_bin_type")
                    else np.float64
                )
                setattr(obj, key, np.asarray(data[key], dtype=dtype))
        return obj

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print(self) -> None:
        """Pretty-print the injection data."""
        print(f"Aerosol Name:          {self.aerosol_name}")
        print(f"N Types:               {self.n_types}")
        print(f"  N-Ions:              {self.n_ions}")
        print(f"  Molar Mass (kg/mol): {self.molar_mass}")
        print(f"  Density (kg/m3):     {self.solute_density}")
        print(f"N Bin Edges:           {self.n_edges}")
        print(f"  Edges (nm):          {self.edge_radii}")
        print(f"  Categories:          {self.category}")
        print(f"N Injection Times:     {self.n_times}")
        print(f"  Times (s):           {self.injection_time}")
        print(f"  Rates (m-3 s-1):     {self.injection_rate}")
        print(f"DSD Bin Edges:         {len(self.dsd_bin_edges)} edges")
        print(f"  Range (um):          {self.dsd_bin_edges[0]:.4f} - {self.dsd_bin_edges[-1]:.4f}")
        print(f"Cumulative Freq:       shape {self.cumulative_frequency.shape}")
        for i, row in enumerate(self.cumulative_frequency):
            print(f"  Time {i}: {row}")

    def __repr__(self) -> str:
        return (
            f"Aerosol(aerosol='{self.aerosol_name}', "
            f"n_types={self.n_types}, n_bins={self.n_bins}, "
            f"n_times={self.n_times})"
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, path: Union[str, Path]) -> None:
        """Write the injection data to an aerosol_input.nc file.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created if needed.
        """
        write_aerosol(path, self.to_dict())

