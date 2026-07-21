"""Read and write CODT parcel input files (``parcel_input.nc``).

Supports ``CODT_parcel_input_v3`` only. The v3 schema describes a trajectory as
an ordered sequence of **waypoint legs**: each leg is an instruction "proceed to
``segment_coord(i)`` at ``velocity(i)``". The active leg advances when its target
is reached, so the lookup key is the leg counter, not the parcel's position, and
targets need no monotonic order. Completing the last leg ends the simulation,
possibly before ``tmax``.

The v1 (time segments) and v2 (time segments + sounding) schemas are rejected by
CODT and are not readable here; see ``docs/codt_v3_migration.md`` to convert.

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
) -> None:
    """Validate a per-leg entrainment schedule.

    Parameters
    ----------
    ent_rate : np.ndarray
        Fractional entrainment rate per leg, in 1/km (**not** 1/m).
    n_blob : np.ndarray
        Blobs per entrainment event per leg.
    psigma : np.ndarray
        Blob fraction per leg.
    n_legs : int
        Expected length of each array.

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
    if np.any(psigma * n_blob >= 1.0):
        raise ValueError("psigma * n_blob must be < 1 for every leg")


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
        _validate_entrainment(ent_rate, n_blob, psigma, n_legs)

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
