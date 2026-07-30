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

from codt_tools.aerosol_io import (
    SEED_HYDRATION_MODES,
    SEED_KEYS,
    make_seed_group,
    read_aerosol,
    write_aerosol,
)
from codt_tools.parcel_io import (
    PRESSURE_MODES,
    VERTICAL_AXES,
    _MAX_N_BLOB,
    _validate_entrainment,
    read_parcel,
    validate_legs,
    write_parcel,
)


# ======================================================================
# Derived LEM turbulence scales
# ======================================================================

# Molecular constants from CODT ``src/globals.f90`` (m**2/s), and the triplet
# map's structural floor ``min_eddy_gridpoints``.
_NU = 1.488e-5
_KT = 1.96e-5
_DV = 2.2705e-5
_MIN_EDDY_GRIDPOINTS = 6

# CODT writes these exponents as the literals `1./3.` and `4./3.`, which Fortran
# evaluates in *default* (single) precision before promoting to real(dp). Using
# the exact double 1/3 instead shifts diffusivity_length_scale by ~4e-8 relative.
# Reproducing the literals brings agreement with the LEM.* output attributes down
# to ~4e-9 relative, which is the floor set by gfortran's `**` differing from
# libm's `pow`; it cannot be closed from Python. That residual only matters for a
# configuration sitting within ~1e-8 of a 3-cell boundary in the `ceiling` below,
# where CODT's own value is the authority.
_ONE_THIRD = float(np.float32(1.0) / np.float32(3.0))
_FOUR_THIRDS = float(np.float32(4.0) / np.float32(3.0))


def lem_turbulence_scales(
    n: int, h: float, dissipation_rate: float
) -> dict[str, float | int]:
    """Derive the LEM turbulence scales the way CODT does.

    Mirrors ``derive_turbulence_scales`` in CODT ``src/LEM.f90``. As of CODT
    3.0.0 the smallest eddy is derived from the grid and the molecular
    diffusivities rather than supplied via the namelist (the old
    ``kolmogorov_length_scale`` input was removed), so this is the only way to
    know it before a run. **This must track ``src/LEM.f90``** — the same values
    are written to the output NC as the ``LEM.*`` global attributes, which is
    what the end-to-end check compares against.

    Parameters
    ----------
    n : int
        Number of grid cells (``N``).
    h : float
        Domain height in m.
    dissipation_rate : float
        TKE dissipation rate in m**2/s**3.

    Returns
    -------
    dict
        ``actual_kolmogorov_scale`` (diagnostic only, never governs),
        ``grid_eddy_scale``, ``diffusivity_length_scale``,
        ``smallest_eddy_scale`` (m), ``smallest_eddy_gridpoints`` (cells) and
        ``diffusivity_enhancement`` (unitless, always >= 1).

    Notes
    -----
    ``diffusivity_enhancement`` is a *step* function of the grid, not ~1: the
    3-cell quantum can overshoot ``diffusivity_length_scale`` by up to 50%, so
    it reaches ``(3/2)**(4/3) = 1.717`` near the crossover. Read it, do not
    assume it.
    """
    dz = h / n
    grid_eddy_scale = _MIN_EDDY_GRIDPOINTS * dz
    diffusivity_length_scale = (
        max(_KT, _DV) / (0.1 * dissipation_rate**_ONE_THIRD)
    ) ** 0.75

    # ceiling, not round-to-nearest: rounding down would put the smallest eddy
    # below diffusivity_length_scale and drive the enhancement factor under 1.
    gridpoints = 3 * int(
        np.ceil(max(grid_eddy_scale, diffusivity_length_scale) / (3.0 * dz))
    )
    gridpoints = max(_MIN_EDDY_GRIDPOINTS, gridpoints)
    smallest_eddy_scale = gridpoints * dz

    return {
        "actual_kolmogorov_scale": (_NU**3 / dissipation_rate) ** 0.25,
        "grid_eddy_scale": grid_eddy_scale,
        "diffusivity_length_scale": diffusivity_length_scale,
        "smallest_eddy_scale": smallest_eddy_scale,
        "smallest_eddy_gridpoints": gridpoints,
        "diffusivity_enhancement": (
            smallest_eddy_scale / diffusivity_length_scale
        ) ** _FOUR_THIRDS,
    }


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
    # Defaults mirror CODT's *code* defaults (docs/input_parameters.md). Each
    # internal group maps directly to the Fortran namelist of the same name,
    # so a parameter MUST live in the group CODT reads it from — putting it
    # elsewhere makes CODT reject the namelist ("Invalid parameter in &GROUP").
    # Parameters CODT marks "required" have no code default; the placeholders
    # here (names, paths, write cadence) just keep the config usable.
    _DEFAULTS: dict[str, dict[str, Any]] = {
        "parameters": {
            "n":                  2000,
            "tmax":               100.0,
            "tref":               20.0,
            "pres":               1.0e5,
            "h":                  1.0,
            "volume_scaling":     10.0,
            "same_random":        False,
            "simulation_name":    "default_sim",
            "output_directory":   "./output",
            "overwrite":          False,
            "write_timer":        1.0,
            "write_buffer":       200,
            "write_eddies":       False,
            "do_turbulence":      True,
            "do_microphysics":    True,
            "do_special_effects": False,
            "do_radiation":       False,
            "do_entrainment":     False,
            "simulation_mode":    "chamber",
        },
        "parcel": {
            "parcel_file":          "",
            "initial_rh":           1.0,
            "pressure_limit":       0.0,
            "vertical_axis":        "height",
            "pressure_mode":        "hydrostatic",
            "initial_height":       0.0,
        },
        "entrainment": {
            "ent_rate":             2.0,   # 1/km
            "n_blob":               1,
            "psigma":               0.1,
            "random_entrainment":   True,
        },
        "turbulence_odt": {
            "tdiff":              10.0,
            "lmin":               6,
            "lprob":              18,
            "max_accept_prob":    0.1,
            "c2":                 1.5e3,
            "zc2":                1.0e5,
        },
        "turbulence_lem": {
            "integral_length_scale":    0.01,
            "dissipation_rate":         0.01,
        },
        "microphysics": {
            "aerosol_file":                    "aerosol_input.nc",
            "aerosol_concentration":           0.0,
            "init_drop_each_gridpoint":        False,
            "expected_ndrops_per_gridpoint":   1.0,
            "initial_wet_radius":              1.5,
            "write_trajectories":              False,
            "trajectory_start":                0.0,
            "trajectory_end":                  0.0,
            "trajectory_timer":                1.0,
            "do_collisions":                   False,
            "do_coalescence":                  False,
            "coalescence_kernel":              "hall",
            "wmax_collision":                  10.0,
            "write_collisions":                False,
            "do_seeding":                      False,
            "seed_hydration":                  "equilibrium",
            "seed_growth_time":                5.0,
        },
        "specialeffects": {
            "do_sidewalls":         False,
            "area_sw":              4.0,
            "area_bot":             2.0,
            "c_sw":                 0.42,
            "sw_nudging_time":      0.85,
            "t_sw":                 14.85,
            "rh_sw":                0.96,
            "p_sw":                 7.0,
            "do_random_fallout":    False,
            "random_fallout_rate":  1.0,
        },
        "radiation": {
            "radiation_method":     "1d",
            "mie_data_file":        "",
            "eps_top":              1.0,
            "eps_bot":              1.0,
            "sky_temp":             263.15,
            "sky_cooling_flag":     False,
            "rad_call_interval":    0.0,
            "nphotons":             700000,
            "nbins":                30,
            "lx_rad":               2.0,
            "ly_rad":               2.0,
            "t_side":               293.15,
            "max_droplets_per_cell": 20,
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

        Warns if the write directory doesn't match the path referenced
        by ``aerosol_file`` in the namelist.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created if needed.
        """
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check that aerosol_file is consistent with the write location
        nml_dir = path.parent
        for key in ("aerosol_file",):
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

        nml = f90nml.Namelist(self._groups_for_write())
        nml.write(path, force=True)

    # Groups that are only relevant to one simulation mode.
    _CHAMBER_ONLY_GROUPS: set[str] = {"turbulence_odt", "specialeffects"}
    _PARCEL_ONLY_GROUPS: set[str] = {"turbulence_lem", "parcel", "entrainment"}

    def _groups_for_write(self) -> dict[str, dict[str, Any]]:
        """Return namelist groups filtered by simulation mode and switches.

        Chamber mode excludes ``turbulence_lem``, ``parcel``, and
        ``entrainment``. Parcel mode excludes ``turbulence_odt`` and
        ``specialeffects``. The ``radiation`` group is only included when
        ``do_radiation`` is enabled, and ``entrainment`` only when
        ``do_entrainment`` is enabled (CODT reads each group only when its
        switch is on).
        """
        try:
            mode = self.get("simulation_mode")
        except KeyError:
            mode = "chamber"

        if mode == "parcel":
            skip = set(self._CHAMBER_ONLY_GROUPS)
        else:
            skip = set(self._PARCEL_ONLY_GROUPS)

        try:
            do_radiation = self.get("do_radiation")
        except KeyError:
            do_radiation = False
        if not do_radiation:
            skip.add("radiation")

        try:
            do_entrainment = self.get("do_entrainment")
        except KeyError:
            do_entrainment = False
        if not do_entrainment:
            skip.add("entrainment")

        return {g: p for g, p in self._data.items() if g not in skip}

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
# ParcelInput
# ======================================================================


class ParcelInput:
    """In-memory representation of a CODT parcel input file.

    Reads and writes the ``parcel_input.nc`` format
    (``CODT_parcel_input_v3``). The trajectory is a sequence of **waypoint
    legs**: leg *i* says "proceed to ``segment_coord[i]`` at ``velocity[i]``".
    Targets need no monotonic order, velocities are signed, and completing the
    last leg ends the simulation — possibly before ``tmax``.

    Parameters
    ----------
    path : str or Path, optional
        Path to an existing ``parcel_input.nc`` file. If ``None``, creates
        an instance with a single leg rising to 1000 m at 1 m/s.

    Examples
    --------
    Create from defaults, then fly up, down, and up again:

    >>> pi = ParcelInput()
    >>> pi.set(segment_coord=[1000.0, 400.0, 1500.0], velocity=[1.0, -0.5, 1.0])

    Load from file:

    >>> pi = ParcelInput("input/parcel_input.nc")
    """

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        if path is None:
            self._init_defaults()
        else:
            self._read(Path(path))

    def _init_defaults(self) -> None:
        self.segment_coord: np.ndarray = np.array([1000.0])
        self.velocity: np.ndarray = np.array([1.0])
        self.ent_rate: np.ndarray | None = None
        self.n_blob: np.ndarray | None = None
        self.psigma: np.ndarray | None = None
        self.env_height: np.ndarray | None = None
        self.env_pressure: np.ndarray | None = None
        self.env_temperature: np.ndarray | None = None
        self.env_RH: np.ndarray | None = None

    def _read(self, path: Path) -> None:
        data = read_parcel(path)
        self.segment_coord = data["segment_coord"]
        self.velocity = data["velocity"]
        self.ent_rate = data["ent_rate"]
        self.n_blob = data["n_blob"]
        self.psigma = data["psigma"]
        self.env_height = data["env_height"]
        self.env_pressure = data["env_pressure"]
        self.env_temperature = data["env_temperature"]
        self.env_RH = data["env_RH"]

    # ------------------------------------------------------------------
    # Access / modification
    # ------------------------------------------------------------------

    _VALID_ATTRS: set[str] = {
        "segment_coord", "velocity",
        "ent_rate", "n_blob", "psigma",
        "env_height", "env_pressure", "env_temperature", "env_RH",
    }

    def set(self, **kwargs: Any) -> None:
        """Set one or more attributes by name.

        Array-like values are converted to numpy arrays. Setting any member of
        the sounding or the entrainment schedule to ``None`` clears just that
        value; use :meth:`clear_env_profile` or :meth:`clear_entrainment` to
        clear a whole group.

        Parameters
        ----------
        **kwargs
            Attribute name-value pairs.

        Raises
        ------
        AttributeError
            If an attribute name is not valid.
        """
        for key, value in kwargs.items():
            if key not in self._VALID_ATTRS:
                raise AttributeError(
                    f"'{key}' is not a valid ParcelInput attribute. "
                    f"Valid: {sorted(self._VALID_ATTRS)}"
                )
            if value is None:
                setattr(self, key, None)
            elif key == "n_blob":
                setattr(self, key,
                        np.atleast_1d(np.asarray(value, dtype=np.int32)))
            else:
                setattr(self, key,
                        np.atleast_1d(np.asarray(value, dtype=np.float64)))

    @property
    def n_legs(self) -> int:
        """Number of waypoint legs."""
        return len(self.segment_coord)

    @property
    def has_env_profile(self) -> bool:
        """Whether an environmental sounding is present."""
        return self.env_pressure is not None

    @property
    def has_entrainment_schedule(self) -> bool:
        """Whether a per-leg entrainment schedule is present."""
        return self.ent_rate is not None

    def set_env_profile(
        self,
        height: np.ndarray | list,
        pressure: np.ndarray | list,
        temperature: np.ndarray | list,
        RH: np.ndarray | list,
    ) -> None:
        """Set the environmental sounding.

        Required when ``do_entrainment`` is enabled or
        ``&PARCEL pressure_mode='environment'``.

        Parameters
        ----------
        height : array-like
            Environmental height in m (strictly increasing).
        pressure : array-like
            Environmental pressure in Pa (strictly decreasing).
        temperature : array-like
            Environmental temperature in K.
        RH : array-like
            Environmental relative humidity (0–1).
        """
        self.env_height = np.asarray(height, dtype=np.float64)
        self.env_pressure = np.asarray(pressure, dtype=np.float64)
        self.env_temperature = np.asarray(temperature, dtype=np.float64)
        self.env_RH = np.asarray(RH, dtype=np.float64)

    def clear_env_profile(self) -> None:
        """Remove the environmental sounding."""
        self.env_height = None
        self.env_pressure = None
        self.env_temperature = None
        self.env_RH = None

    def set_entrainment_schedule(
        self,
        ent_rate: np.ndarray | list,
        n_blob: np.ndarray | list,
        psigma: np.ndarray | list,
    ) -> None:
        """Set a per-leg entrainment schedule.

        When present these override the constant ``&ENTRAINMENT`` values
        leg-by-leg; ``random_entrainment`` still comes from the namelist.

        Parameters
        ----------
        ent_rate : array-like
            Fractional entrainment rate per leg, in **1/km** (not 1/m).
        n_blob : array-like
            Number of evenly sized chunks the replaced volume is split into,
            per leg (1–10). Purely a mixing axis — it does not change how much
            is entrained, nor the event spacing.
        psigma : array-like
            Total domain fraction replaced per entrainment event, per leg, in
            (0, 1). Each chunk needs at least one gridcell, so
            ``int(psigma * N) >= n_blob``.

        Notes
        -----
        This is the CODT >= 3.1.0 meaning of ``psigma``. Through CODT 3.0.1 it
        was the size of *one* blob, so an event replaced ``n_blob * psigma`` of
        the domain; schedules carried over from that era entrain ``n_blob``
        times less now.
        """
        self.ent_rate = np.asarray(ent_rate, dtype=np.float64)
        self.n_blob = np.asarray(n_blob, dtype=np.int32)
        self.psigma = np.asarray(psigma, dtype=np.float64)

    def clear_entrainment_schedule(self) -> None:
        """Remove the per-leg entrainment schedule."""
        self.ent_rate = None
        self.n_blob = None
        self.psigma = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to the dict format used by :mod:`parcel_io`."""
        return {key: getattr(self, key) for key in self._VALID_ATTRS}

    def write(
        self,
        path: Union[str, Path],
        initial_level: float | None = None,
        vertical_axis: str = "height",
        n_grid: int | None = None,
    ) -> None:
        """Write the parcel input to a NetCDF file.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created if needed.
        initial_level : float, optional
            Launch level to validate leg directions against. If None,
            validation is left to CODT at run time.
        vertical_axis : {"height", "pressure"}
            Axis that ``segment_coord`` is expressed on.
        n_grid : int, optional
            Grid size ``N``, used to check the entrainment schedule's
            ``int(psigma * N) >= n_blob`` rule. If None that check is left to
            CODT at run time.
        """
        write_parcel(
            path, self.to_dict(), initial_level, vertical_axis, n_grid
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print(self) -> None:
        """Pretty-print the parcel input data."""
        print(f"N Legs:        {self.n_legs}")
        print(f"  Targets:     {self.segment_coord}")
        print(f"  Vel (m/s):   {self.velocity}")
        if self.has_entrainment_schedule:
            print(f"Entrainment:   per-leg schedule")
            print(f"  Rate (1/km): {self.ent_rate}")
        else:
            print("Entrainment:   none (namelist constants)")
        if self.env_pressure is not None:
            print(f"Env Profile:   {len(self.env_pressure)} levels")
            print(f"  P range:     {self.env_pressure[0]:.0f} - "
                  f"{self.env_pressure[-1]:.0f} Pa")
        else:
            print("Env Profile:   none")

    def __repr__(self) -> str:
        return f"ParcelInput(n_legs={self.n_legs}, schema=v3)"


# ======================================================================
# CODTConfig
# ======================================================================


class CODTConfig:
    """Complete set of CODT input parameters.

    Bundles a :class:`Namelist` and :class:`InjectionData` into a single
    object that can be written to a directory, copied, and swept over for
    parameter studies.

    Parameters
    ----------
    namelist_path : str or Path, optional
        Path to an existing ``params.nml`` file.  If provided, a sibling
        ``aerosol_input.nc`` file is also loaded (based on the
        ``aerosol_file`` namelist parameter).  If ``None``, all components
        are initialised with defaults.

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
            self.injection: InjectionData = InjectionData(
                aerosol_path if aerosol_path.is_file() else None
            )
            try:
                parcel_file = self.params.get("parcel_file")
            except KeyError:
                parcel_file = ""
            if parcel_file:
                parcel_path = nml_dir / parcel_file
                self.parcel: ParcelInput = ParcelInput(
                    parcel_path if parcel_path.is_file() else None
                )
            else:
                self.parcel = ParcelInput()
        else:
            self.injection = InjectionData()
            self.parcel = ParcelInput()

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_simulation(
        cls,
        path: Union[str, Path],
        run_dir: Union[str, Path, None] = None,
    ) -> "CODTConfig":
        """Build a config from a completed simulation's output.

        Recovers the configuration from:

        - **Namelist parameters** from netCDF global attributes (always
          available in CODT v0.4+).
        - **DSD bin edges** from the ``radius_edges`` variable in the
          output netCDF.
        - **Aerosol injection / parcel input** from the run directory
          (required — CODT v0.5.x no longer copies input files to the
          output directory).

        Parameters
        ----------
        path : str or Path
            Path to the output ``.nc`` file, or a directory containing
            exactly one.
        run_dir : str or Path
            Path to the run directory containing ``params.nml``,
            ``aerosol_input.nc``, and optionally ``parcel_input.nc``.
            Required because CODT v0.5.x does not copy input files to
            the output directory.

        Returns
        -------
        CODTConfig
            A new config populated from the simulation output.

        Raises
        ------
        FileNotFoundError
            If *run_dir* is not provided or does not contain the
            expected input files (``aerosol_input.nc``).

        Examples
        --------
        >>> cfg = CODTConfig.from_simulation(
        ...     "output/old_run.nc",
        ...     run_dir="output/old_run/run",
        ... )
        """
        from codt_tools.simulation import CODTSimulation

        sim = CODTSimulation(path)

        # -- Build the config object without calling __init__ --
        obj = cls.__new__(cls)

        # Namelist: prefer simulation's parsed params (from netCDF attrs)
        if sim.params is not None:
            obj.params = copy.deepcopy(sim.params)
        else:
            obj.params = Namelist()

        # Input files: require run_dir
        if run_dir is None:
            sim.close()
            raise FileNotFoundError(
                "run_dir is required: CODT v0.5.x does not copy input "
                "files to the output directory. Pass the path to the "
                "run directory (e.g. '{base}/{name}/run/')."
            )

        run_dir = Path(run_dir)
        aerosol_path = run_dir / "aerosol_input.nc"
        if not aerosol_path.is_file():
            sim.close()
            raise FileNotFoundError(
                f"aerosol_input.nc not found in run_dir: {run_dir}. "
                f"Expected: {aerosol_path}"
            )
        obj.injection = InjectionData(aerosol_path)

        parcel_path = run_dir / "parcel_input.nc"
        obj.parcel = ParcelInput(
            parcel_path if parcel_path.is_file() else None
        )

        # DSD bin edges: prefer radius_edges from the netCDF output
        try:
            edges = sim.bin_edges
            obj.injection.set(dsd_bin_edges=edges)
        except (KeyError, AttributeError):
            pass

        sim.close()
        return obj

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

        Parameters
        ----------
        edges : array-like
            Bin edge values in microns.
        """
        self.injection.set(dsd_bin_edges=np.asarray(edges, dtype=np.float64))

    def set_parcel(self, **kwargs: Any) -> None:
        """Set parcel input attributes.

        Delegates to :meth:`ParcelInput.set`.

        Parameters
        ----------
        **kwargs
            Attribute name-value pairs.

        Examples
        --------
        Waypoint legs: ascend to 500 m at 1 m/s, then to 900 m at 0.5 m/s.
        ``time`` is not a v3 field — the lookup is keyed to the leg counter.

        >>> cfg.set_parcel(segment_coord=[500.0, 900.0], velocity=[1.0, 0.5])
        """
        self.parcel.set(**kwargs)

    # ------------------------------------------------------------------
    # Dot-access for namelist parameters
    # ------------------------------------------------------------------

    # Attributes that belong to the CODTConfig instance itself (not the
    # namelist).  These must bypass the namelist delegation in __setattr__.
    _OWN_ATTRS: set[str] = {"params", "injection", "parcel"}

    def __getattr__(self, name: str) -> Any:
        """Attribute-style read access to namelist parameters.

        Only called when normal attribute lookup fails, so instance
        attributes (``params``, ``injection``) and properties
        (``name``) are unaffected.

        Examples
        --------
        >>> cfg = CODTConfig()
        >>> cfg.tref
        21.5
        """
        # Guard against recursion during deepcopy/pickle — params may not
        # exist yet when Python is reconstructing the object.
        if name == "params" or "params" not in self.__dict__:
            raise AttributeError(name)
        try:
            return self.params.get(name)
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Attribute-style write access to namelist parameters.

        If *name* is a known namelist parameter, the value is routed
        through :meth:`Namelist.set` (with type checking).  Otherwise
        normal attribute assignment is used.

        Examples
        --------
        >>> cfg = CODTConfig()
        >>> cfg.tref = 22.0
        >>> cfg.tref
        22.0
        """
        # During __init__ or for own instance attributes, use normal path.
        if name in self._OWN_ATTRS or not hasattr(self, "params"):
            object.__setattr__(self, name, value)
            return

        # Check if name is a namelist parameter.
        try:
            self.params._find_group(name)
        except KeyError:
            object.__setattr__(self, name, value)
        else:
            self.params.set(**{name: value})

    def __dir__(self) -> list[str]:
        """Include namelist parameter names for tab-completion."""
        base = set(super().__dir__())
        if hasattr(self, "params"):
            for group_params in self.params._data.values():
                base.update(group_params.keys())
        return sorted(base)

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

        Writes data files first, then ``params.nml`` (so the namelist
        path-existence check passes). For parcel mode, also writes
        ``parcel_input.nc`` and sets ``parcel_file`` accordingly.

        Parameters
        ----------
        directory : str or Path
            Target directory.  Created if it does not exist.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Set relative path for data file
        self.params.set(aerosol_file="aerosol_input.nc")

        # Write data files before namelist (avoids Namelist.write warning)
        self.injection.write(directory / "aerosol_input.nc")

        if self.params.get("simulation_mode") == "parcel":
            self.params.set(parcel_file="parcel_input.nc")
            self.parcel.write(
                directory / "parcel_input.nc",
                n_grid=self.params.get("n"),
            )

        self.params.write(directory / "params.nml")

    def validate(self) -> None:
        """Check internal consistency.

        Validates mode-specific constraints:

        - **Chamber mode** requires ``tdiff > 0``.
        - **Parcel mode** requires a non-empty ``parcel_file`` and
          ``initial_rh`` in [0, 1]. If ``do_entrainment`` is enabled,
          the parcel input must have an environmental sounding (v2).
        - Trajectory windows must be valid when enabled.
        - CDF dimensions must match aerosol bins.

        Raises
        ------
        ValueError
            If any consistency check fails.
        """
        mode = self.params.get("simulation_mode")

        # -- Range checks --
        if self.params.get("n") <= 0:
            raise ValueError(f"N must be positive, got {self.params.get('n')}.")
        if self.params.get("tmax") <= 0:
            raise ValueError(f"tmax must be positive, got {self.params.get('tmax')}.")
        if self.params.get("h") <= 0:
            raise ValueError(f"H must be positive, got {self.params.get('h')}.")
        if self.params.get("pres") <= 0:
            raise ValueError(f"pres must be positive, got {self.params.get('pres')}.")
        if self.params.get("volume_scaling") <= 0:
            raise ValueError(
                f"volume_scaling must be positive, got {self.params.get('volume_scaling')}."
            )
        if self.params.get("tref") <= -273.15:
            raise ValueError(f"Tref must be > -273.15 C, got {self.params.get('tref')}.")
        if mode not in ("chamber", "parcel"):
            raise ValueError(
                f"simulation_mode must be 'chamber' or 'parcel', got '{mode}'."
            )

        # -- Cross-namelist consistency warnings --
        if self.params.get("do_radiation") and not self.params.get("do_microphysics"):
            warnings.warn(
                "do_radiation has no effect without do_microphysics.",
                UserWarning,
                stacklevel=2,
            )
        if self.params.get("write_eddies") and not self.params.get("do_turbulence"):
            warnings.warn(
                "write_eddies has no effect without do_turbulence.",
                UserWarning,
                stacklevel=2,
            )
        if self.params.get("do_entrainment") and mode == "chamber":
            warnings.warn(
                "do_entrainment is not yet implemented in chamber mode.",
                UserWarning,
                stacklevel=2,
            )

        # -- Mode-specific checks --
        if mode == "chamber":
            tdiff = self.params.get("tdiff")
            if tdiff <= 0:
                raise ValueError(
                    f"Chamber mode requires tdiff > 0, got {tdiff}."
                )

            # Mirrors the ODT startup guard (CODT src/ODT.f90). Below six cells
            # each of the triplet map's three segments is a single cell and the
            # eddy carries no sub-eddy structure.
            if self.params.get("do_turbulence"):
                lmin = self.params.get("lmin")
                if lmin < _MIN_EDDY_GRIDPOINTS:
                    raise ValueError(
                        f"Lmin = {lmin} is below the smallest representable "
                        f"eddy ({_MIN_EDDY_GRIDPOINTS} gridpoints)."
                    )
                if lmin % 3 != 0:
                    raise ValueError(
                        f"Lmin = {lmin} must be a multiple of 3 "
                        f"(triplet map segments)."
                    )

        elif mode == "parcel":
            self._validate_lem_scales()

            parcel_file = self.params.get("parcel_file")
            if not parcel_file:
                raise ValueError(
                    "Parcel mode requires parcel_file to be set."
                )

            initial_rh = self.params.get("initial_rh")
            if not 0.0 <= initial_rh <= 1.0:
                raise ValueError(
                    f"initial_rh must be in [0, 1], got {initial_rh}."
                )

            vertical_axis = self.params.get("vertical_axis")
            if vertical_axis not in VERTICAL_AXES:
                raise ValueError(
                    f"vertical_axis must be one of {list(VERTICAL_AXES)}, "
                    f"got {vertical_axis!r}."
                )

            pressure_mode = self.params.get("pressure_mode")
            if pressure_mode not in PRESSURE_MODES:
                raise ValueError(
                    f"pressure_mode must be one of {list(PRESSURE_MODES)}, "
                    f"got {pressure_mode!r}."
                )

            needs_sounding = (
                self.params.get("do_entrainment")
                or pressure_mode == "environment"
            )
            if needs_sounding and not self.parcel.has_env_profile:
                raise ValueError(
                    "do_entrainment=.true. or pressure_mode='environment' "
                    "requires an environmental sounding in the parcel input. "
                    "Use cfg.parcel.set_env_profile(...)."
                )

            # Mirror CODT's entrainment startup abort (>= 3.1.0): psigma is the
            # total domain fraction replaced per event, split into n_blob evenly
            # sized chunks, so each chunk needs at least one gridcell. The
            # per-leg schedule overrides the namelist scalars where present.
            if self.params.get("do_entrainment"):
                n = self.params.get("n")
                psigma = self.params.get("psigma")
                n_blob = self.params.get("n_blob")
                if not 0.0 < psigma < 1.0:
                    raise ValueError(
                        f"psigma must be in (0, 1), got {psigma}."
                    )
                if not 1 <= n_blob <= _MAX_N_BLOB:
                    raise ValueError(
                        f"n_blob must be in 1..{_MAX_N_BLOB}, got {n_blob}."
                    )
                if int(psigma * n) < n_blob:
                    raise ValueError(
                        f"int(psigma * N) = {int(psigma * n)} is fewer than "
                        f"n_blob = {n_blob} chunks (psigma={psigma}, N={n}); "
                        f"every chunk needs at least one gridcell."
                    )
                leg_ent_rate = self.parcel.ent_rate
                leg_n_blob = self.parcel.n_blob
                leg_psigma = self.parcel.psigma
                if (leg_ent_rate is not None
                        and leg_n_blob is not None
                        and leg_psigma is not None):
                    _validate_entrainment(
                        leg_ent_rate,
                        leg_n_blob,
                        leg_psigma,
                        self.parcel.n_legs,
                        n,
                    )

            if self.parcel.n_legs < 1:
                raise ValueError("Parcel input must have at least one leg.")

            # Mirror CODT's leg-direction validation. On the pressure axis the
            # launch level is the initial pressure; in environment mode CODT
            # resets that from the sounding at initial_height before validating,
            # so do the same here.
            initial_height = self.params.get("initial_height")
            if vertical_axis == "pressure":
                env_height = self.parcel.env_height
                env_pressure = self.parcel.env_pressure
                if (pressure_mode == "environment"
                        and env_height is not None
                        and env_pressure is not None):
                    initial_level = float(
                        np.interp(initial_height, env_height, env_pressure)
                    )
                else:
                    initial_level = self.params.get("pres")
            else:
                initial_level = initial_height

            validate_legs(
                self.parcel.segment_coord,
                self.parcel.velocity,
                initial_level,
                vertical_axis,
            )

        # -- Seeding: do_seeding is the sole controller (mirrors CODT) --
        # Enabling it requires a seed group to act on; without one there is
        # nothing to seed, so that is fatal. With seeding off, a seed group in
        # the file is simply ignored (never read) — CODT warns but runs, which
        # is what lets one file serve both a seeded and an unseeded run. So a
        # dormant group is a warning here, not an error.
        do_seeding = self.params.get("do_seeding")
        if do_seeding and not self.injection.has_seed_group:
            raise ValueError(
                "do_seeding=.true. but the aerosol input has no seed group. "
                "CODT aborts on this. Add one with "
                "cfg.injection.set_seed_group(...) or disable do_seeding."
            )
        if self.injection.has_seed_group and not do_seeding:
            warnings.warn(
                "The aerosol input carries a seed group but do_seeding=.false.; "
                "the group will be ignored. Set do_seeding=True to use it, or "
                "cfg.injection.clear_seed_group() to drop it."
            )

        if do_seeding:
            seed_hydration = self.params.get("seed_hydration")
            if seed_hydration not in SEED_HYDRATION_MODES:
                raise ValueError(
                    f"seed_hydration must be one of "
                    f"{list(SEED_HYDRATION_MODES)}, got {seed_hydration!r}."
                )
            if self.params.get("seed_growth_time") <= 0.0:
                raise ValueError("seed_growth_time must be > 0.")

        # -- Injection data consistency --
        n_bins = self.injection.n_bins
        cdf_cols = self.injection.cumulative_frequency.shape[1]
        if cdf_cols != n_bins:
            raise ValueError(
                f"cumulative_frequency has {cdf_cols} columns but "
                f"there are {n_bins} aerosol bins."
            )

        # -- Trajectory window --
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

    def _validate_lem_scales(self) -> None:
        """Check the derived LEM scales, mirroring CODT's startup aborts.

        CODT 3.0.0 derives the smallest eddy rather than taking it from the
        namelist, and refuses to start on grids that cannot represent it
        (``validate_turbulence_scales`` in ``src/LEM.f90``). Reproducing that
        here turns a wasted allocation into an immediate ``ValueError``.
        """
        if not self.params.get("do_turbulence"):
            return

        n = self.params.get("n")
        h = self.params.get("h")
        eps = self.params.get("dissipation_rate")
        l_int = self.params.get("integral_length_scale")
        scales = lem_turbulence_scales(n, h, eps)
        gridpoints = scales["smallest_eddy_gridpoints"]
        l_small = scales["smallest_eddy_scale"]

        if gridpoints > n:
            raise ValueError(
                f"The domain cannot contain the smallest eddy: "
                f"smallest eddy = {gridpoints} cells, but N = {n}. Required "
                f"scale: {l_small:.4e} m (set by the larger of 6*dz and the "
                f"diffusivity length scale). Increase H, or increase N so "
                f"6*dz falls below it, or raise dissipation_rate."
            )

        # The -5/3 eddy sampler forms (L**(-5/3) - l_small**(-5/3)); if
        # l_small >= L that bracket is <= 0 and raising it to -3/5 yields a
        # silent NaN.
        if l_small >= l_int:
            raise ValueError(
                f"No inertial range -- the smallest eddy is not smaller than "
                f"the integral length scale: smallest_eddy_scale = "
                f"{l_small:.4e} m >= integral_length_scale = {l_int:.4e} m. "
                f"Increase integral_length_scale, or increase N to refine "
                f"the grid."
            )

        scale_separation = l_int / l_small
        if scale_separation < 3.0:
            warnings.warn(
                f"The inertial range is nearly absent: "
                f"integral_length_scale / smallest_eddy_scale = "
                f"{scale_separation:.2f} (Re = "
                f"{scale_separation ** _FOUR_THIRDS:.3f}). Eddy statistics are "
                f"drawn from a very narrow band; maps_per_event will be small "
                f"and the turbulence poorly resolved. Consider a larger "
                f"integral_length_scale or a finer grid.",
                UserWarning,
                stacklevel=3,
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
        "c2": "C2",
        "zc2": "ZC2",
        "expected_ndrops_per_gridpoint": "Ndrops",
        "initial_wet_radius": "Rw",
        "aerosol_concentration": "Naer",
        "write_timer": "dt",
        "integral_length_scale": "Lint",
        "dissipation_rate": "eps",
        "initial_rh": "RH",
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
            f"injection={self.injection!r})"
        )
