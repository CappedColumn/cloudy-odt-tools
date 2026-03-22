"""CODT input configuration: namelist, injection data, and bin data.

All classes can be imported via::

    from codt_tools.config import Namelist, InjectionData, BinData
"""

from __future__ import annotations

import copy
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Union

import f90nml
import numpy as np

from codt_tools.aerosol_io import read_aerosol, write_aerosol


# ======================================================================
# Namelist
# ======================================================================


class Namelist:
    """In-memory representation of a CODT Fortran namelist (params.nml).

    Parameters
    ----------
    path : str or Path, optional
        Path to an existing params.nml file. If ``None``, creates a namelist
        populated with default values.

    Examples
    --------
    >>> nml = Namelist("input/params.nml")
    >>> nml.get("tref")
    21.5
    >>> nml.set(tref=22.0, tmax=7200.0, simulation_name="new_run")
    >>> nml.write("output_dir/params.nml")

    Create a blank namelist with defaults:

    >>> nml = Namelist()
    >>> nml.set(tref=22.0, simulation_name="my_sim")
    """

    # Default values for all namelist parameters.
    # Types must match Fortran declarations exactly:
    #   INTEGER  -> int
    #   REAL(8)  -> float
    #   LOGICAL  -> bool
    #   CHARACTER -> str
    _DEFAULTS: dict[str, dict[str, Any]] = {
        "parameters": {
            "n":                  2000,
            "lmin":               6,
            "lprob":              18,
            "tmax":               3600.0,
            "tdiff":              3.0,
            "tref":               21.5,
            "pres":               1.0e5,
            "h":                  1.0,
            "volume_scaling":     100,
            "max_accept_prob":    0.1,
            "same_random":        False,
            "simulation_name":    "default_sim",
            "output_directory":   "./output",
            "write_timer":        1.0,
            "write_buffer":       200,
            "write_eddies":       False,
            "do_turbulence":      True,
            "do_microphysics":    True,
            "do_special_effects": False,
            "overwrite":          False,
        },
        "microphysics": {
            "init_drop_each_gridpoint":       False,
            "expected_ndrops_per_gridpoint":   3,
            "initial_wet_radius":              1.5,
            "aerosol_file":                    "aerosol_input.nc",
            "bin_data_file":                   "bin_data.txt",
            "write_trajectories":              False,
            "trajectory_start":                0.0,
            "trajectory_end":                  0.0,
            "trajectory_timer":                1.0,
        },
        "specialeffects": {
            "do_sidewalls":         False,
            "area_sw":              4,
            "area_bot":             2,
            "c_sw":                 0.42,
            "sw_nudging_time":      0.85,
            "t_sw":                 14.85,
            "rh_sw":                0.96,
            "p_sw":                 7,
            "do_random_fallout":    False,
            "random_fallout_rate":  0.33333,
        },
    }

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        if path is None:
            # Deep copy defaults so each instance is independent
            self._data: dict[str, dict[str, Any]] = {
                group: dict(params)
                for group, params in self._DEFAULTS.items()
            }
        else:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Namelist file not found: {path}")

            nml = f90nml.read(path)
            self._data = {group: dict(nml[group]) for group in nml}

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def groups(self) -> list[str]:
        """List of namelist group names (lowercase)."""
        return list(self._data.keys())

    def get(self, key: str) -> Any:
        """Get a parameter value by name, searching across all groups.

        Parameters
        ----------
        key : str
            Parameter name (case-insensitive).

        Returns
        -------
        Any
            The parameter value.

        Raises
        ------
        KeyError
            If *key* is not found in any group.
        """
        group = self._find_group(key)
        return self._data[group][key.lower()]

    def set(self, **kwargs: Any) -> None:
        """Set one or more parameter values, auto-routing to the correct group.

        Numeric types (int, float) are freely interchangeable. Boolean
        parameters accept ``True``/``False`` or ``1``/``0``.

        Parameters
        ----------
        **kwargs
            Parameter name-value pairs. Names are case-insensitive.

        Raises
        ------
        KeyError
            If a parameter name is not found in any group.
        TypeError
            If the new value type is incompatible (e.g. str for a numeric).

        Examples
        --------
        >>> nml.set(tref=22.0, tmax=7200.0, simulation_name="test_run")
        >>> nml.set(do_turbulence=0)       # 0 -> False
        >>> nml.set(volume_scaling=13.5)   # float -> OK for int fields
        """
        for key, value in kwargs.items():
            group = self._find_group(key)
            key_lower = key.lower()
            old_value = self._data[group][key_lower]
            old_type = type(old_value)
            new_type = type(value)

            if old_type is bool:
                # Accept bool, int (0/1), or raise
                if new_type is bool:
                    pass
                elif new_type is int and value in (0, 1):
                    value = bool(value)
                else:
                    raise TypeError(
                        f"Type mismatch for '{key}': expected bool or "
                        f"int (0/1), got {new_type.__name__} ({value!r})"
                    )
            elif old_type in (int, float):
                # Note: bool is a subclass of int in Python, so check
                # for bool first above to avoid treating True/False as 1/0
                if new_type is bool:
                    raise TypeError(
                        f"Type mismatch for '{key}': expected numeric "
                        f"(int/float), got bool"
                    )
                elif new_type in (int, float):
                    # Freely convert between int and float
                    value = old_type(value)
                else:
                    raise TypeError(
                        f"Type mismatch for '{key}': expected numeric "
                        f"(int/float), got {new_type.__name__}"
                    )
            elif old_type is str:
                if new_type is not str:
                    raise TypeError(
                        f"Type mismatch for '{key}': expected str, "
                        f"got {new_type.__name__}"
                    )

            self._data[group][key_lower] = value

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print(self, group: str | None = None) -> None:
        """Pretty-print namelist contents.

        Parameters
        ----------
        group : str, optional
            If given, print only that group (case-insensitive).
            If ``None``, print all groups.

        Raises
        ------
        KeyError
            If *group* is not found in the namelist.
        """
        if group is not None:
            group = group.lower()
            if group not in self._data:
                raise KeyError(
                    f"Group '{group}' not found. "
                    f"Available groups: {self.groups}"
                )
            groups = {group: self._data[group]}
        else:
            groups = self._data

        for grp_name, params in groups.items():
            print(f"&{grp_name.upper()}")
            for key, val in params.items():
                print(f"  {key} = {val!r}")
            print()

    def __repr__(self) -> str:
        params_count = sum(len(p) for p in self._data.values())
        return (
            f"Namelist({self.groups}, {params_count} parameters)"
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, path: Union[str, Path]) -> None:
        """Write the namelist to a file in Fortran namelist format.

        Warns if the write directory doesn't match the paths referenced
        by ``aerosol_file`` or ``bin_data_file`` in the namelist.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created if needed.
        """
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check that aerosol_file and bin_data_file are consistent
        # with the namelist write location
        nml_dir = path.parent
        for key in ("aerosol_file", "bin_data_file"):
            try:
                data_path = Path(self.get(key))
            except KeyError:
                continue
            expected = (nml_dir / data_path).resolve()
            if not expected.is_file():
                warnings.warn(
                    f"Namelist '{key}' is set to '{data_path}', but "
                    f"'{expected}' does not exist. The CODT model will "
                    f"resolve this relative to the namelist's parent "
                    f"directory ({nml_dir}). Make sure to write the "
                    f"corresponding data file there.",
                    UserWarning,
                    stacklevel=2,
                )

        nml = f90nml.Namelist(self._data)
        nml.write(path, force=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_group(self, key: str) -> str:
        """Find which namelist group contains *key* (case-insensitive)."""
        key_lower = key.lower()
        for group, params in self._data.items():
            if key_lower in params:
                return group
        raise KeyError(
            f"Parameter '{key}' not found in any namelist group. "
            f"Use .print() to see available parameters."
        )


# ======================================================================
# InjectionData
# ======================================================================


class InjectionData:
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

    >>> inj = InjectionData("input/aerosol_input.nc")
    >>> inj.aerosol_name
    'NaCl'

    Create from defaults and customise — scalars are wrapped automatically:

    >>> inj = InjectionData()
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
        self.cumulative_frequency: np.ndarray = np.array([[0.5, 1.0]])
        self.injection_time: np.ndarray = np.array([0.0])
        self.injection_rate: np.ndarray = np.array([6.66e4])

    def _read(self, path: Path) -> None:
        """Read from an aerosol_input.nc file."""
        data = read_aerosol(path)
        self.aerosol_name = data["aerosol_name"]
        self.n_ions = data["n_ions"]
        self.molar_mass = data["molar_mass"]
        self.solute_density = data["solute_density"]
        self.edge_radii = data["edge_radii"]
        self.category = data["category"]
        self.cumulative_frequency = data["cumulative_frequency"]
        self.injection_time = data["injection_time"]
        self.injection_rate = data["injection_rate"]

    # ------------------------------------------------------------------
    # Access / modification
    # ------------------------------------------------------------------

    # Fields that are 1D arrays in the NetCDF schema.  Scalars passed for
    # these are wrapped in a length-1 array automatically.
    _INT_1D_FIELDS: set[str] = {"n_ions", "category"}
    _FLOAT_1D_FIELDS: set[str] = {
        "molar_mass", "solute_density", "edge_radii",
        "injection_time", "injection_rate",
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
                    f"'{key}' is not a valid InjectionData attribute. "
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

        if key in InjectionData._INT_1D_FIELDS:
            return np.atleast_1d(np.asarray(value, dtype=np.int32))

        if key in InjectionData._FLOAT_1D_FIELDS:
            return np.atleast_1d(np.asarray(value, dtype=np.float64))

        if key in InjectionData._FLOAT_2D_FIELDS:
            arr = np.atleast_2d(np.asarray(value, dtype=np.float64))
            arr = InjectionData._validate_cdf(arr)
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
            "edge_radii", "category", "cumulative_frequency",
            "injection_time", "injection_rate",
        ]

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
    def from_dict(cls, data: dict[str, Any]) -> "InjectionData":
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
        print(f"Cumulative Freq:       shape {self.cumulative_frequency.shape}")
        for i, row in enumerate(self.cumulative_frequency):
            print(f"  Time {i}: {row}")

    def __repr__(self) -> str:
        return (
            f"InjectionData(aerosol='{self.aerosol_name}', "
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


# ======================================================================
# BinData
# ======================================================================


class BinData:
    """In-memory representation of CODT droplet bin edge data.

    Parameters
    ----------
    path : str or Path, optional
        Path to an existing bin_data.txt file. If ``None``, creates an
        instance with a default log-spaced bin edge array.

    Examples
    --------
    >>> bins = BinData("input/bin_data.txt")
    >>> bins.edges
    array([0.049, 0.051, ...])
    >>> bins.n_edges
    201
    """

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        if path is None:
            self._init_defaults()
        else:
            self._read(Path(path))

    def _init_defaults(self) -> None:
        """Create default log-spaced bin edges from ~0.049 to ~60 microns."""
        self.edges: np.ndarray = np.geomspace(0.049, 60.0, num=201)

    def _read(self, path: Path) -> None:
        """Parse a bin_data.txt file."""
        if not path.is_file():
            raise FileNotFoundError(f"Bin data file not found: {path}")

        with open(path, "r") as f:
            lines = f.readlines()

        # Line 1: "N Bin-Edges"
        # Line 2: integer count
        n_edges = int(lines[1].strip())

        # Line 3: "Droplet Bin-Edges (microns)"
        # Lines 4+: one value per line
        self.edges = np.array(
            [float(lines[i].strip()) for i in range(3, 3 + n_edges)],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def n_edges(self) -> int:
        """Number of bin edges."""
        return len(self.edges)

    @property
    def n_bins(self) -> int:
        """Number of bins (edges - 1)."""
        return len(self.edges) - 1

    @property
    def centers(self) -> np.ndarray:
        """Bin center values (midpoints of adjacent edges) in microns."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def set(self, edges: Union[np.ndarray, list]) -> None:
        """Replace the bin edges.

        Parameters
        ----------
        edges : array-like
            New bin edge values in microns.
        """
        self.edges = np.asarray(edges, dtype=np.float64)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print(self) -> None:
        """Pretty-print the bin data summary."""
        print(f"N Bin Edges:  {self.n_edges}")
        print(f"N Bins:       {self.n_bins}")
        print(f"Range (um):   {self.edges[0]:.4f} - {self.edges[-1]:.4f}")

    def __repr__(self) -> str:
        return (
            f"BinData(n_edges={self.n_edges}, "
            f"range=[{self.edges[0]:.4f}, {self.edges[-1]:.4f}] um)"
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, path: Union[str, Path]) -> None:
        """Write the bin data to a file in CODT format.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write("N Bin-Edges\n")
            f.write(f"{self.n_edges}\n")
            f.write("Droplet Bin-Edges (microns)\n")
            for edge in self.edges:
                f.write(f"{edge:.18e}\n")


# ======================================================================
# CODTConfig
# ======================================================================


class CODTConfig:
    """Complete set of CODT input parameters.

    Bundles a :class:`Namelist`, :class:`InjectionData`, and :class:`BinData`
    into a single object that can be written to a directory, copied, and
    swept over for parameter studies.

    Parameters
    ----------
    namelist_path : str or Path, optional
        Path to an existing ``params.nml`` file.  If provided, sibling
        ``aerosol_input.nc`` and ``bin_data.txt`` files are also loaded
        (based on the ``aerosol_file`` and ``bin_data_file`` namelist
        parameters).  If ``None``, all components are initialised with
        defaults.

    Examples
    --------
    Create from defaults and customise:

    >>> cfg = CODTConfig()
    >>> cfg.set(simulation_name="my_sim", tref=22.0, tmax=7200.0)
    >>> cfg.set_injection(injection_rate=1.0e5)
    >>> cfg.write("/scratch/my_sim/run")

    Load from existing files:

    >>> cfg = CODTConfig("run_dir/params.nml")
    >>> cfg.params.get("tref")
    22.0
    """

    def __init__(self, namelist_path: Union[str, Path, None] = None) -> None:
        self.params: Namelist = Namelist(namelist_path)

        if namelist_path is not None:
            nml_dir = Path(namelist_path).resolve().parent
            aerosol_path = nml_dir / self.params.get("aerosol_file")
            bin_path = nml_dir / self.params.get("bin_data_file")
            self.injection: InjectionData = InjectionData(
                aerosol_path if aerosol_path.is_file() else None
            )
            self.bins: BinData = BinData(
                bin_path if bin_path.is_file() else None
            )
        else:
            self.injection = InjectionData()
            self.bins = BinData()

    # ------------------------------------------------------------------
    # Convenience setters
    # ------------------------------------------------------------------

    def set(self, **kwargs: Any) -> None:
        """Set namelist parameters by name.

        Delegates to :meth:`Namelist.set` — parameters are automatically
        routed to the correct namelist group.

        Parameters
        ----------
        **kwargs
            Parameter name-value pairs (case-insensitive).

        Examples
        --------
        >>> cfg.set(tref=22.0, tmax=7200.0, simulation_name="test")
        """
        self.params.set(**kwargs)

    def set_injection(self, **kwargs: Any) -> None:
        """Set aerosol injection attributes.

        Delegates to :meth:`InjectionData.set`.

        Parameters
        ----------
        **kwargs
            Attribute name-value pairs.

        Examples
        --------
        >>> cfg.set_injection(aerosol_name="KCl", injection_rate=1.0e5)
        """
        self.injection.set(**kwargs)

    def set_bins(self, edges: Union[np.ndarray, list]) -> None:
        """Set droplet size distribution bin edges.

        Delegates to :meth:`BinData.set`.

        Parameters
        ----------
        edges : array-like
            Bin edge values in microns.
        """
        self.bins.set(edges)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Simulation name (from namelist ``simulation_name``)."""
        return self.params.get("simulation_name")

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, directory: Union[str, Path]) -> None:
        """Write all input files to a directory.

        Writes ``aerosol_input.nc`` and ``bin_data.txt`` first, then
        ``params.nml`` (so the namelist path-existence check passes).
        The namelist ``aerosol_file`` and ``bin_data_file`` parameters
        are set to the bare filenames before writing.

        Parameters
        ----------
        directory : str or Path
            Target directory.  Created if it does not exist.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Set relative paths for data files
        self.params.set(
            aerosol_file="aerosol_input.nc",
            bin_data_file="bin_data.txt",
        )

        # Write data files before namelist (avoids Namelist.write warning)
        self.injection.write(directory / "aerosol_input.nc")
        self.bins.write(directory / "bin_data.txt")
        self.params.write(directory / "params.nml")

    def validate(self) -> None:
        """Check internal consistency.

        Raises
        ------
        ValueError
            If any consistency check fails.
        """
        # CDF dimensions must match bins
        n_bins = self.injection.n_bins
        cdf_cols = self.injection.cumulative_frequency.shape[1]
        if cdf_cols != n_bins:
            raise ValueError(
                f"cumulative_frequency has {cdf_cols} columns but "
                f"there are {n_bins} aerosol bins."
            )

        # Trajectory window (if enabled)
        if self.params.get("write_trajectories"):
            t_start = self.params.get("trajectory_start")
            t_end = self.params.get("trajectory_end")
            tmax = self.params.get("tmax")
            if t_start >= t_end:
                raise ValueError(
                    f"trajectory_start ({t_start}) must be less than "
                    f"trajectory_end ({t_end})."
                )
            if t_end > tmax:
                raise ValueError(
                    f"trajectory_end ({t_end}) exceeds tmax ({tmax})."
                )

    # ------------------------------------------------------------------
    # Copy / sweep
    # ------------------------------------------------------------------

    def copy(self) -> "CODTConfig":
        """Return a deep copy of this configuration."""
        return copy.deepcopy(self)

    # Short-name abbreviations for sweep-generated simulation names.
    _SWEEP_ABBREV: dict[str, str] = {
        "tref": "Tref",
        "tdiff": "Tdiff",
        "tmax": "tmax",
        "volume_scaling": "VS",
        "n": "N",
        "h": "H",
        "pres": "P",
        "lmin": "Lmin",
        "lprob": "Lprob",
        "max_accept_prob": "MAP",
        "expected_ndrops_per_gridpoint": "Ndrops",
        "initial_wet_radius": "Rw",
        "write_timer": "dt",
    }

    @staticmethod
    def sweep(
        base: "CODTConfig", **param_ranges: list
    ) -> list["CODTConfig"]:
        """Generate a Cartesian product of parameter variations.

        Each returned config is an independent deep copy with a unique
        ``simulation_name`` auto-generated from the base name and the
        varied parameter values.

        Parameters
        ----------
        base : CODTConfig
            Base configuration to vary.
        **param_ranges
            Keyword arguments mapping parameter names to lists of values.

        Returns
        -------
        list[CODTConfig]
            One config per combination.

        Examples
        --------
        >>> configs = CODTConfig.sweep(
        ...     base,
        ...     tref=[20.0, 21.0, 22.0],
        ...     volume_scaling=[13, 50],
        ... )
        >>> len(configs)
        6
        >>> configs[0].name
        'default_sim_Tref20.0_VS13'
        """
        if not param_ranges:
            return [base.copy()]

        param_names = list(param_ranges.keys())
        param_values = [param_ranges[k] for k in param_names]
        base_name = base.name

        configs: list[CODTConfig] = []
        for combo in product(*param_values):
            cfg = base.copy()
            # Build name suffix
            parts: list[str] = []
            for name, value in zip(param_names, combo):
                cfg.set(**{name: value})
                abbrev = CODTConfig._SWEEP_ABBREV.get(
                    name.lower(), name
                )
                parts.append(f"{abbrev}{value}")
            cfg.set(simulation_name=f"{base_name}_{'_'.join(parts)}")
            configs.append(cfg)

        return configs

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CODTConfig(name='{self.name}', "
            f"injection={self.injection!r}, "
            f"bins={self.bins!r})"
        )
