"""The CODT Fortran namelist (``params.nml``).

``Namelist`` is a faithful container: it holds exactly what a namelist holds,
including the four path-bearing keys, and does not decide what they should be.
Deciding where a run reads and writes is :meth:`codt_tools.case.Case.write_inputs`'s
job — see the path-ownership rules there.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Union

import f90nml


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
    # Defaults mirror CODT's *code* defaults (see docs/input_parameters.md
    # in the CODT repository, not this one). Each
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
            "output_directory":   "",     # staged at write time
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
        # Deep copy defaults so each instance is independent
        self._data: dict[str, dict[str, Any]] = {
            group: dict(params)
            for group, params in self._DEFAULTS.items()
        }
        if path is None:
            return

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Namelist file not found: {path}")

        # A file's values are laid *over* the defaults rather than replacing
        # them. A namelist on disk carries only the groups its mode needs
        # (CODT never reads the others), so replacing would leave the object
        # missing parameters that are perfectly well defined — and switching
        # such a namelist to the other mode would then fail on lookup.
        # Groups and keys the defaults do not know are kept verbatim.
        nml = f90nml.read(path)
        for group in nml:
            self._data.setdefault(group, {}).update(dict(nml[group]))

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

        Raises
        ------
        ValueError
            If ``output_directory`` is empty. CODT resolves a relative
            ``output_directory`` against the *process's* working directory
            (``src/initialize.f90:246``), so an empty value silently sends the
            run's output wherever the job happened to start. Deciding that
            path is :meth:`codt_tools.case.Case.write_inputs`'s job.
        """
        path = Path(path).resolve()

        try:
            output_directory = self.get("output_directory")
        except KeyError:
            output_directory = None
        if output_directory is not None and not str(output_directory).strip():
            raise ValueError(
                "output_directory is empty: this namelist has not been "
                "staged, and CODT would resolve the empty path against "
                "whatever directory the job starts in. Write it through "
                "Case.write_inputs(dir, output_directory=...), or set "
                "output_directory explicitly first."
            )

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

        nml = f90nml.Namelist(self.groups_for_write())
        nml.write(path, force=True)

    # Groups that are only relevant to one simulation mode.
    _CHAMBER_ONLY_GROUPS: set[str] = {"turbulence_odt", "specialeffects"}
    _PARCEL_ONLY_GROUPS: set[str] = {"turbulence_lem", "parcel", "entrainment"}

    def groups_for_write(self) -> dict[str, dict[str, Any]]:
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

