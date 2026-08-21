"""Read and write CODT parcel input files (``parcel_input.nc``).

Supports ``CODT_parcel_input_v3`` only. The v3 schema describes a trajectory as
an ordered sequence of **waypoint legs**: each leg is an instruction "proceed to
``segment_coord(i)`` at ``velocity(i)``". The active leg advances when its target
is reached, so the lookup key is the leg counter, not the parcel's position, and
targets need no monotonic order. Completing the last leg ends the simulation,
possibly before ``tmax``.

The v1 (time segments) and v2 (time segments + sounding) schemas are rejected by
CODT and are not readable here; see ``docs/codt_v3_migration.md`` to convert.

``Parcel`` is the in-memory form; the three functions below are the file format
itself, kept public because they are the format's only specification.

Functions
---------
read_parcel
    Read a parcel_input.nc file into a plain dict.
write_parcel
    Write a plain dict to a parcel_input.nc file.
validate_legs
    Check leg targets and signed velocities the way CODT does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import netCDF4 as nc
import numpy as np

CONVENTIONS: str = "CODT_parcel_input_v3"

#: Vertical axes accepted by ``&PARCEL vertical_axis``.
VERTICAL_AXES: tuple[str, ...] = ("height", "pressure")

#: Pressure modes accepted by ``&PARCEL pressure_mode``.
PRESSURE_MODES: tuple[str, ...] = ("hydrostatic", "environment")

#: Per-leg entrainment schedule variables. Present together or not at all.
_ENTRAINMENT_KEYS: tuple[str, ...] = ("ent_rate", "n_blob", "psigma")

#: Environmental sounding variables. Present together or not at all.
_SOUNDING_KEYS: tuple[str, ...] = (
    "env_height",
    "env_pressure",
    "env_temperature",
    "env_RH",
)

_MAX_N_BLOB: int = 10


def validate_legs(
    segment_coord: np.ndarray | list,
    velocity: np.ndarray | list,
    initial_level: float,
    vertical_axis: str = "height",
) -> None:
    """Validate leg targets and signed velocities against a launch level.

    Mirrors CODT's ``validate_legs`` (``src/parcel.f90``) so that a malformed
    trajectory is rejected when the file is built rather than after a submit.

    Parameters
    ----------
    segment_coord : array-like
        Leg target levels, in metres or pascals per *vertical_axis*.
    velocity : array-like
        Signed leg velocities in m/s. Must point from the previous level
        toward the leg's target.
    initial_level : float
        The launch level the first leg is measured from: ``initial_height``
        (m) on the height axis, or the initial pressure (Pa) on the pressure
        axis. In ``pressure_mode='environment'`` CODT resets the initial
        pressure from the sounding *before* validating, so pass the effective
        launch level rather than the raw namelist ``pres``.
    vertical_axis : {"height", "pressure"}
        Axis that *segment_coord* is expressed on. On the pressure axis, "up"
        means a *decreasing* target, so the direction test is negated.

    Raises
    ------
    ValueError
        If *vertical_axis* is unknown, the arrays disagree in length, a leg has
        zero velocity, a target equals the previous level, or a velocity points
        away from its target.
    """
    if vertical_axis not in VERTICAL_AXES:
        raise ValueError(
            f"vertical_axis must be one of {list(VERTICAL_AXES)}, "
            f"got {vertical_axis!r}"
        )

    segment_coord = np.atleast_1d(np.asarray(segment_coord, dtype=np.float64))
    velocity = np.atleast_1d(np.asarray(velocity, dtype=np.float64))

    if len(segment_coord) != len(velocity):
        raise ValueError(
            f"segment_coord and velocity must have the same length, got "
            f"{len(segment_coord)} and {len(velocity)}"
        )
    if len(segment_coord) < 1:
        raise ValueError("at least one leg is required")

    prev = float(initial_level)
    for i, (target, vel) in enumerate(zip(segment_coord, velocity), start=1):
        if vel == 0.0:
            raise ValueError(
                f"leg {i} has zero velocity (the leg could never complete)"
            )
        if target == prev:
            raise ValueError(f"leg {i} target equals the previous level: {prev}")

        toward = target - prev
        if vertical_axis == "pressure":
            toward = -toward
        if toward * vel < 0.0:
            raise ValueError(
                f"leg {i} velocity {vel} points away from its target {target} "
                f"(previous level {prev})"
            )
        prev = float(target)


def _validate_entrainment(
    ent_rate: np.ndarray,
    n_blob: np.ndarray,
    psigma: np.ndarray,
    n_legs: int,
    n_grid: int | None = None,
) -> None:
    """Validate a per-leg entrainment schedule.

    Parameters
    ----------
    ent_rate : np.ndarray
        Fractional entrainment rate per leg, in 1/km (**not** 1/m).
    n_blob : np.ndarray
        Number of chunks the replaced volume is split into, per leg.
    psigma : np.ndarray
        Total domain fraction replaced per entrainment event, per leg
        (CODT >= 3.1.0 meaning; through 3.0.1 this was the size of *one*
        blob).
    n_legs : int
        Expected length of each array.
    n_grid : int, optional
        Grid size ``N`` the run will use. Required to check CODT's
        ``int(psigma * N) >= n_blob`` rule (every chunk needs at least one
        gridcell); when None that check is left to CODT at run time.

    Raises
    ------
    ValueError
        If any array is the wrong length or a value is out of range.
    """
    for name, arr in (
        ("ent_rate", ent_rate),
        ("n_blob", n_blob),
        ("psigma", psigma),
    ):
        if len(arr) != n_legs:
            raise ValueError(
                f"{name} must have one value per leg ({n_legs}), got {len(arr)}"
            )

    if np.any(ent_rate <= 0.0):
        raise ValueError("ent_rate must be > 0 (units are 1/km, not 1/m)")
    if np.any(n_blob < 1) or np.any(n_blob > _MAX_N_BLOB):
        raise ValueError(f"n_blob must be in 1..{_MAX_N_BLOB}")
    if np.any(psigma <= 0.0) or np.any(psigma >= 1.0):
        raise ValueError("psigma must be in (0, 1)")
    if n_grid is not None:
        cells = (psigma * n_grid).astype(np.int64)
        bad = np.nonzero(cells < n_blob)[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"leg {i}: int(psigma * N) = {int(cells[i])} is fewer than "
                f"n_blob = {int(n_blob[i])} chunks (psigma={float(psigma[i])}, "
                f"N={n_grid}); every chunk needs at least one gridcell"
            )


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

        - ``segment_coord`` : np.ndarray, shape (n_legs,), m or Pa
        - ``velocity`` : np.ndarray, shape (n_legs,), m/s, signed
        - ``ent_rate`` : np.ndarray or None, shape (n_legs,), 1/km
        - ``n_blob`` : np.ndarray or None, shape (n_legs,)
        - ``psigma`` : np.ndarray or None, shape (n_legs,)
        - ``env_height`` : np.ndarray or None, shape (n_levels,), m
        - ``env_pressure`` : np.ndarray or None, shape (n_levels,), Pa
        - ``env_temperature`` : np.ndarray or None, shape (n_levels,), K
        - ``env_RH`` : np.ndarray or None, shape (n_levels,), 0–1

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the ``conventions`` attribute is not ``CODT_parcel_input_v3``, or if
        an optional group is only partially present.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Parcel input file not found: {path}")

    with nc.Dataset(path, "r") as ds:
        conventions = getattr(ds, "conventions", None)
        if conventions != CONVENTIONS:
            extra = ""
            if conventions in ("CODT_parcel_input_v1", "CODT_parcel_input_v2"):
                extra = (
                    " (v1/v2 parcel inputs are no longer supported; regenerate "
                    "the file in the v3 waypoint format — see "
                    "docs/codt_v3_migration.md)"
                )
            raise ValueError(
                f"Expected conventions='{CONVENTIONS}', "
                f"got '{conventions}' in {path}{extra}"
            )

        data: dict[str, Any] = {
            "segment_coord": ds["segment_coord"][:].data.copy(),
            "velocity": ds["velocity"][:].data.copy(),
        }

        for group in (_ENTRAINMENT_KEYS, _SOUNDING_KEYS):
            present = [key for key in group if key in ds.variables]
            if present and len(present) != len(group):
                missing = sorted(set(group) - set(present))
                raise ValueError(
                    f"{path} has a partial variable group: found "
                    f"{sorted(present)} but is missing {missing}. These "
                    f"variables must be present together or not at all."
                )
            for key in group:
                data[key] = (
                    ds[key][:].data.copy() if present else None
                )

    return data


def write_parcel(
    path: Union[str, Path],
    data: dict[str, Any],
    initial_level: float | None = None,
    vertical_axis: str = "height",
    n_grid: int | None = None,
) -> None:
    """Write a parcel_input.nc file in the v3 waypoint-leg schema.

    Parameters
    ----------
    path : str or Path
        Destination file path. Parent directories are created if needed.
    data : dict
        Must contain ``segment_coord`` and ``velocity``. May contain the
        entrainment schedule (``ent_rate``, ``n_blob``, ``psigma``) and the
        sounding (``env_height``, ``env_pressure``, ``env_temperature``,
        ``env_RH``); each group is all-or-nothing.
    initial_level : float, optional
        Launch level to validate leg directions against, matching the
        ``&PARCEL initial_height`` (height axis) or initial ``pres`` (pressure
        axis) the run will use. If None, leg-direction validation is skipped
        and CODT will perform it at run time instead.
    vertical_axis : {"height", "pressure"}
        Axis that ``segment_coord`` is expressed on, matching
        ``&PARCEL vertical_axis``.
    n_grid : int, optional
        Grid size ``N`` the run will use, needed to check the entrainment
        schedule's ``int(psigma * N) >= n_blob`` rule. If None that check is
        left to CODT at run time.

    Raises
    ------
    ValueError
        If the legs, the entrainment schedule, or the sounding are invalid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    segment_coord = np.atleast_1d(
        np.asarray(data["segment_coord"], dtype=np.float64)
    )
    velocity = np.atleast_1d(np.asarray(data["velocity"], dtype=np.float64))
    n_legs = len(segment_coord)

    if len(velocity) != n_legs:
        raise ValueError(
            f"segment_coord and velocity must have the same length, got "
            f"{n_legs} and {len(velocity)}"
        )

    if initial_level is not None:
        validate_legs(segment_coord, velocity, initial_level, vertical_axis)
    elif vertical_axis not in VERTICAL_AXES:
        raise ValueError(
            f"vertical_axis must be one of {list(VERTICAL_AXES)}, "
            f"got {vertical_axis!r}"
        )

    ent_rate: np.ndarray | None = None
    n_blob: np.ndarray | None = None
    psigma: np.ndarray | None = None
    if data.get("ent_rate") is not None:
        ent_rate = np.atleast_1d(np.asarray(data["ent_rate"], dtype=np.float64))
        n_blob = np.atleast_1d(np.asarray(data["n_blob"], dtype=np.int32))
        psigma = np.atleast_1d(np.asarray(data["psigma"], dtype=np.float64))
        _validate_entrainment(ent_rate, n_blob, psigma, n_legs, n_grid)

    env: dict[str, np.ndarray] | None = None
    if data.get("env_pressure") is not None:
        env = {
            key: np.atleast_1d(np.asarray(data[key], dtype=np.float64))
            for key in _SOUNDING_KEYS
        }
        n_levels = len(env["env_pressure"])
        for key, arr in env.items():
            if len(arr) != n_levels:
                raise ValueError(
                    f"sounding variables must all have the same length; "
                    f"{key} has {len(arr)}, expected {n_levels}"
                )
        if np.any(np.diff(env["env_height"]) <= 0.0):
            raise ValueError("env_height must be strictly increasing")
        if np.any(np.diff(env["env_pressure"]) >= 0.0):
            raise ValueError("env_pressure must be strictly decreasing")

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.conventions = CONVENTIONS

        ds.createDimension("segment", n_legs)

        v = ds.createVariable("segment_coord", "f8", ("segment",))
        v.units = "Pa" if vertical_axis == "pressure" else "m"
        v.long_name = "leg target level"
        v[:] = segment_coord

        v = ds.createVariable("velocity", "f8", ("segment",))
        v.units = "m/s"
        v.long_name = "signed leg velocity"
        v[:] = velocity

        if ent_rate is not None:
            v = ds.createVariable("ent_rate", "f8", ("segment",))
            v.units = "1/km"
            v[:] = ent_rate

            v = ds.createVariable("n_blob", "i4", ("segment",))
            v[:] = n_blob

            v = ds.createVariable("psigma", "f8", ("segment",))
            v[:] = psigma

        if env is not None:
            ds.createDimension("level", len(env["env_pressure"]))

            v = ds.createVariable("env_height", "f8", ("level",))
            v.units = "m"
            v[:] = env["env_height"]

            v = ds.createVariable("env_pressure", "f8", ("level",))
            v.units = "Pa"
            v[:] = env["env_pressure"]

            v = ds.createVariable("env_temperature", "f8", ("level",))
            v.units = "K"
            v[:] = env["env_temperature"]

            v = ds.createVariable("env_RH", "f8", ("level",))
            v.units = "1"
            v[:] = env["env_RH"]


# ======================================================================
# Parcel
# ======================================================================


class Parcel:
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

    >>> pi = Parcel()
    >>> pi.set(segment_coord=[1000.0, 400.0, 1500.0], velocity=[1.0, -0.5, 1.0])

    Load from file:

    >>> pi = Parcel("input/parcel_input.nc")
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
                    f"'{key}' is not a valid Parcel attribute. "
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
        return f"Parcel(n_legs={self.n_legs}, schema=v3)"

