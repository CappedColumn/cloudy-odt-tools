"""Validation: everything CODT would refuse at startup, checked here first.

The checks mirror CODT's own startup aborts so a bad configuration fails in
Python instead of after an allocation lands on a compute node. Each one names
the Fortran guard it mirrors; when CODT's guards move, these must follow.

:func:`lem_turbulence_scales` reproduces ``derive_turbulence_scales`` in CODT's
``src/LEM.f90`` and is public — it is the only way to know the smallest eddy a
grid can carry before running.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from codt_tools.case.aerosol import SEED_HYDRATION_MODES
from codt_tools.case.parcel import (
    PRESSURE_MODES,
    VERTICAL_AXES,
    _MAX_N_BLOB,
    _validate_entrainment,
    validate_legs,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from codt_tools.case.case import Case
    from codt_tools.case.namelist import Namelist
    from codt_tools.case.parcel import Parcel


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



def initial_launch_level(params: "Namelist", parcel: "Parcel") -> float:
    """The level a parcel run launches from, on its own vertical axis.

    Mirrors CODT's leg-direction validation. On the pressure axis the launch
    level is the initial pressure; in ``pressure_mode='environment'`` CODT
    resets that from the sounding at ``initial_height`` before validating, so
    do the same here.

    Shared by :func:`validate_case` and ``Case.write_inputs`` so the two cannot
    disagree about which direction a leg points.
    """
    initial_height = params.get("initial_height")
    if params.get("vertical_axis") != "pressure":
        return initial_height

    env_height = parcel.env_height
    env_pressure = parcel.env_pressure
    if (params.get("pressure_mode") == "environment"
            and env_height is not None
            and env_pressure is not None):
        return float(np.interp(initial_height, env_height, env_pressure))
    return params.get("pres")


def validate_case(case: Case) -> None:
    """Check internal consistency.

    Validates mode-specific constraints:

    - **Chamber mode** requires ``tdiff > 0``.
    - **Parcel mode** requires at least one trajectory leg and ``initial_rh``
      in [0, 1]. If ``do_entrainment`` is enabled, the parcel input must have
      an environmental sounding.
    - Radiation requires a ``mie_data_file`` that exists.
    - Trajectory windows must be valid when enabled.
    - CDF dimensions must match aerosol bins.

    Path keys are not validated here: ``output_directory``, ``aerosol_file``
    and ``parcel_file`` are assigned by ``Case.write_inputs`` and are empty or
    nominal until then.

    Raises
    ------
    ValueError
        If any consistency check fails.
    """
    mode = case.params.get("simulation_mode")

    # -- Range checks --
    if case.params.get("n") <= 0:
        raise ValueError(f"N must be positive, got {case.params.get('n')}.")
    if case.params.get("tmax") <= 0:
        raise ValueError(f"tmax must be positive, got {case.params.get('tmax')}.")
    if case.params.get("h") <= 0:
        raise ValueError(f"H must be positive, got {case.params.get('h')}.")
    if case.params.get("pres") <= 0:
        raise ValueError(f"pres must be positive, got {case.params.get('pres')}.")
    if case.params.get("volume_scaling") <= 0:
        raise ValueError(
            f"volume_scaling must be positive, got {case.params.get('volume_scaling')}."
        )
    if case.params.get("tref") <= -273.15:
        raise ValueError(f"Tref must be > -273.15 C, got {case.params.get('tref')}.")
    if mode not in ("chamber", "parcel"):
        raise ValueError(
            f"simulation_mode must be 'chamber' or 'parcel', got '{mode}'."
        )

    # -- Cross-namelist consistency warnings --
    if case.params.get("do_radiation") and not case.params.get("do_microphysics"):
        warnings.warn(
            "do_radiation has no effect without do_microphysics.",
            UserWarning,
            stacklevel=3,
        )
    if case.params.get("write_eddies") and not case.params.get("do_turbulence"):
        warnings.warn(
            "write_eddies has no effect without do_turbulence.",
            UserWarning,
            stacklevel=3,
        )
    if case.params.get("do_entrainment") and mode == "chamber":
        warnings.warn(
            "do_entrainment is not yet implemented in chamber mode.",
            UserWarning,
            stacklevel=3,
        )

    # -- Mode-specific checks --
    if mode == "chamber":
        tdiff = case.params.get("tdiff")
        if tdiff <= 0:
            raise ValueError(
                f"Chamber mode requires tdiff > 0, got {tdiff}."
            )

        # Mirrors the ODT startup guard (CODT src/ODT.f90). Below six cells
        # each of the triplet map's three segments is a single cell and the
        # eddy carries no sub-eddy structure.
        if case.params.get("do_turbulence"):
            lmin = case.params.get("lmin")
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
        _validate_lem_scales(case)

        initial_rh = case.params.get("initial_rh")
        if not 0.0 <= initial_rh <= 1.0:
            raise ValueError(
                f"initial_rh must be in [0, 1], got {initial_rh}."
            )

        vertical_axis = case.params.get("vertical_axis")
        if vertical_axis not in VERTICAL_AXES:
            raise ValueError(
                f"vertical_axis must be one of {list(VERTICAL_AXES)}, "
                f"got {vertical_axis!r}."
            )

        pressure_mode = case.params.get("pressure_mode")
        if pressure_mode not in PRESSURE_MODES:
            raise ValueError(
                f"pressure_mode must be one of {list(PRESSURE_MODES)}, "
                f"got {pressure_mode!r}."
            )

        needs_sounding = (
            case.params.get("do_entrainment")
            or pressure_mode == "environment"
        )
        if needs_sounding and not case.parcel.has_env_profile:
            raise ValueError(
                "do_entrainment=.true. or pressure_mode='environment' "
                "requires an environmental sounding in the parcel input. "
                "Use case.parcel.set_env_profile(...)."
            )

        # Mirror CODT's entrainment startup abort (>= 3.1.0): psigma is the
        # total domain fraction replaced per event, split into n_blob evenly
        # sized chunks, so each chunk needs at least one gridcell. The
        # per-leg schedule overrides the namelist scalars where present.
        if case.params.get("do_entrainment"):
            n = case.params.get("n")
            psigma = case.params.get("psigma")
            n_blob = case.params.get("n_blob")
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
            leg_ent_rate = case.parcel.ent_rate
            leg_n_blob = case.parcel.n_blob
            leg_psigma = case.parcel.psigma
            if (leg_ent_rate is not None
                    and leg_n_blob is not None
                    and leg_psigma is not None):
                _validate_entrainment(
                    leg_ent_rate,
                    leg_n_blob,
                    leg_psigma,
                    case.parcel.n_legs,
                    n,
                )

        if case.parcel.n_legs < 1:
            raise ValueError("Parcel input must have at least one leg.")

        # Mirror CODT's leg-direction validation, from the same launch level
        # Case.write_inputs hands to write_parcel.
        initial_level = initial_launch_level(case.params, case.parcel)

        validate_legs(
            case.parcel.segment_coord,
            case.parcel.velocity,
            initial_level,
            vertical_axis,
        )

    # -- Radiation --
    # CODT aborts when do_radiation is on and mie_data_file is empty
    # (src/radiation.f90:665) and again when the file cannot be opened
    # (radiation.f90:847), resolving a relative name against the namelist's
    # directory. Case.write_inputs stages the file next to params.nml, so the
    # value here is the *source* path: check that it exists now.
    if case.params.get("do_radiation"):
        mie_data_file = str(case.params.get("mie_data_file")).strip()
        if not mie_data_file:
            raise ValueError(
                "do_radiation=.true. requires mie_data_file. Set it to the "
                "Mie table on disk; Case.write_inputs copies it next to "
                "params.nml so the run directory is self-contained."
            )
        if not Path(mie_data_file).expanduser().is_file():
            raise ValueError(
                f"mie_data_file not found: {mie_data_file}. Give a path that "
                f"exists now — it is copied into the run's input directory "
                f"at write time."
            )

    # -- Seeding: do_seeding is the sole controller (mirrors CODT) --
    # Enabling it requires a seed group to act on; without one there is
    # nothing to seed, so that is fatal. With seeding off, a seed group in
    # the file is simply ignored (never read) — CODT warns but runs, which
    # is what lets one file serve both a seeded and an unseeded run. So a
    # dormant group is a warning here, not an error.
    do_seeding = case.params.get("do_seeding")
    if do_seeding and not case.aerosol.has_seed_group:
        raise ValueError(
            "do_seeding=.true. but the aerosol input has no seed group. "
            "CODT aborts on this. Add one with "
            "case.aerosol.set_seed_group(...) or disable do_seeding."
        )
    if case.aerosol.has_seed_group and not do_seeding:
        warnings.warn(
            "The aerosol input carries a seed group but do_seeding=.false.; "
            "the group will be ignored. Set do_seeding=True to use it, or "
            "case.aerosol.clear_seed_group() to drop it.",
            UserWarning,
            stacklevel=3,
        )

    if do_seeding:
        seed_hydration = case.params.get("seed_hydration")
        if seed_hydration not in SEED_HYDRATION_MODES:
            raise ValueError(
                f"seed_hydration must be one of "
                f"{list(SEED_HYDRATION_MODES)}, got {seed_hydration!r}."
            )
        if case.params.get("seed_growth_time") <= 0.0:
            raise ValueError("seed_growth_time must be > 0.")

    # -- Injection data consistency --
    n_bins = case.aerosol.n_bins
    cdf_cols = case.aerosol.cumulative_frequency.shape[1]
    if cdf_cols != n_bins:
        raise ValueError(
            f"cumulative_frequency has {cdf_cols} columns but "
            f"there are {n_bins} aerosol bins."
        )

    # -- Trajectory window --
    if case.params.get("write_trajectories"):
        t_start = case.params.get("trajectory_start")
        t_end = case.params.get("trajectory_end")
        tmax = case.params.get("tmax")
        if t_start >= t_end:
            raise ValueError(
                f"trajectory_start ({t_start}) must be less than "
                f"trajectory_end ({t_end})."
            )
        if t_end > tmax:
            raise ValueError(
                f"trajectory_end ({t_end}) exceeds tmax ({tmax})."
            )

def _validate_lem_scales(case: Case) -> None:
    """Check the derived LEM scales, mirroring CODT's startup aborts.

    CODT 3.0.0 derives the smallest eddy rather than taking it from the
    namelist, and refuses to start on grids that cannot represent it
    (``validate_turbulence_scales`` in ``src/LEM.f90``). Reproducing that
    here turns a wasted allocation into an immediate ``ValueError``.
    """
    if not case.params.get("do_turbulence"):
        return

    n = case.params.get("n")
    h = case.params.get("h")
    eps = case.params.get("dissipation_rate")
    l_int = case.params.get("integral_length_scale")
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
            stacklevel=4,
        )
