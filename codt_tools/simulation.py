"""Load and analyze output from a completed CODT simulation.

Primary usage::

    from codt_tools import CODTSimulation

    sim = CODTSimulation("path/to/SWS100_C")
    sim.T           # Temperature DataArray (time, z)
    sim.LWC         # Liquid water content DataArray (time,)
    sim.profile("S", t=3600)
"""

from __future__ import annotations

import pathlib
from typing import Union

import numpy as np
import xarray as xr

from codt_tools.config import Namelist
from codt_tools.plotting import (
    _ensure_ax,
    _get_label,
    comparison_colors,
    plot_dsd_evolution as _plot_dsd_evolution,
    plot_profile as _plot_profile,
    plot_spectrum as _plot_spectrum,
    plot_timeheight as _plot_timeheight,
    plot_timeseries as _plot_timeseries,
)
from codt_tools.trajectory_io import (
    load_particles as _load_particles,
    particles_at_timestep as _particles_at_timestep,
    record_times as _record_times,
    trajectory_of as _trajectory_of,
    unique_particle_ids as _unique_particle_ids,
)


# netCDF variable name -> (long_name, dims_hint)
# dims_hint: "tz" = (time, z), "t" = (time,), "tr" = (time, radius)
_FIELD_REGISTRY: dict[str, str] = {
    "T": "tz",
    "QV": "tz",
    "Tv": "tz",
    "S": "tz",
    "Np": "t",
    "Nact": "t",
    "Nun": "t",
    "Ravg": "t",
    "LWC": "t",
    "DSD": "tr",
    # Budget diagnostics — accumulated per write interval then reset.
    "budget_inject_solute_mass": "t",
    "budget_inject_liquid_mass": "t",
    "budget_fallout_liquid_mass": "t",
    "budget_fallout_solute_mass": "t",
    "budget_condensation": "t",
    "budget_dgm_delta_T": "t",
    "budget_diffusion_delta_T": "t",
    "budget_diffusion_delta_WV": "t",
    "budget_sidewall_delta_T": "t",
    "budget_sidewall_delta_WV": "t",
    "budget_n_injected": "t",
    "budget_n_fellout": "t",
    "budget_n_coalesced": "t",
    "N_collisions": "t",
    "N_coalescences": "t",
}
# DSD_1, DSD_2, ... are discovered dynamically from the netCDF file.


class CODTSimulation:
    """Load and analyze output from a completed CODT simulation.

    Parameters
    ----------
    path : str or Path
        Path to the simulation output directory, or directly to the
        main ``.nc`` file. Associated files (``.nml``, ``_particles.nc``,
        etc.) are auto-discovered from the simulation name.

    Examples
    --------
    >>> sim = CODTSimulation("output/SWS100_C")
    >>> sim.T
    <xarray.DataArray 'T' (time: 7201, z: 2000)>
    >>> sim.profile("T", t=3600)
    <xarray.DataArray 'T' (z: 2000)>
    """

    def __init__(self, path: Union[str, pathlib.Path]) -> None:
        self.params: Namelist | None = None
        self._core_region: tuple[float, float] | None = None
        self._particles_ds: xr.Dataset | None = None

        path = pathlib.Path(path).resolve()
        self._discover_files(path)
        self._ds: xr.Dataset = xr.open_dataset(
            self._nc_path, decode_timedelta=False
        )
        self._load_params()

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _discover_files(self, path: pathlib.Path) -> None:
        """Locate associated files from a path.

        Accepts either:
        - A directory containing ``{name}.nc``
        - A direct path to the ``.nc`` file
        - A path with no extension (treated as ``{path}.nc``)
        """
        if path.suffix == ".nc":
            nc_path = path
            directory = path.parent
            name = path.stem
        elif path.is_dir():
            directory = path
            # Look for a single .nc file (exclude *_particles.nc, aerosol_input.nc)
            nc_files = [
                f for f in directory.glob("*.nc")
                if not f.name.endswith("_particles.nc")
                and f.name != "aerosol_input.nc"
            ]
            if len(nc_files) == 0:
                raise FileNotFoundError(
                    f"No netCDF file found in {directory}"
                )
            if len(nc_files) > 1:
                raise ValueError(
                    f"Multiple netCDF files found in {directory}: "
                    f"{[f.name for f in nc_files]}. "
                    f"Pass the .nc path directly."
                )
            nc_path = nc_files[0]
            name = nc_path.stem
        else:
            directory = path.parent
            nc_path = directory / f"{path.name}.nc"
            if not nc_path.is_file():
                raise FileNotFoundError(
                    f"Cannot find netCDF file. Tried: {nc_path} "
                    f"and {path} as a directory."
                )
            name = path.name

        if not nc_path.is_file():
            raise FileNotFoundError(f"netCDF file not found: {nc_path}")

        self.name = name
        self.path = directory
        self._nc_path = nc_path

        # Optional associated files
        nml_path = directory / f"{name}.nml"
        self._nml_path: pathlib.Path | None = (
            nml_path if nml_path.is_file() else None
        )

        particles_path = directory / f"{name}_particles.nc"
        self._particles_path: pathlib.Path | None = (
            particles_path if particles_path.is_file() else None
        )

        collisions_path = directory / f"{name}_collisions.bin"
        self._collisions_path: pathlib.Path | None = (
            collisions_path if collisions_path.is_file() else None
        )

        eddies_path = directory / f"{name}_eddies.bin"
        self._eddies_path: pathlib.Path | None = (
            eddies_path if eddies_path.is_file() else None
        )

        self._done_path: pathlib.Path | None = (
            directory / "DONE" if (directory / "DONE").is_file() else None
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _load_params(self) -> None:
        """Load namelist parameters from netCDF global attributes.

        Parameters are stored as dot-prefixed global attributes
        (e.g. ``PARAMETERS.N``, ``MICROPHYSICS.write_trajectories``).
        Booleans are stored as integers (1 = true, 0 = false).
        Falls back to the ``.nml`` file if no attributes are present.
        """
        # Try netCDF global attributes first
        nc_attrs = dict(self._ds.attrs)
        param_attrs = {
            k: v for k, v in nc_attrs.items()
            if "." in k and k.split(".")[0] in (
                "PARAMETERS", "MICROPHYSICS", "SPECIALEFFECTS"
            )
        }

        if param_attrs:
            self.params = self._params_from_nc_attrs(param_attrs)
        elif self._nml_path is not None:
            self.params = Namelist(self._nml_path)
        else:
            self.params = None

    @staticmethod
    def _params_from_nc_attrs(attrs: dict) -> Namelist:
        """Build a Namelist from dot-prefixed netCDF global attributes.

        Converts ``PARAMETERS.N`` → group ``parameters``, key ``n``.
        Integer-encoded booleans are converted back using the known
        boolean fields from the Namelist defaults.
        """
        # Collect known boolean fields from defaults
        bool_keys: set[str] = set()
        for group_params in Namelist._DEFAULTS.values():
            for k, v in group_params.items():
                if isinstance(v, bool):
                    bool_keys.add(k)

        nml = Namelist()  # start from defaults
        for attr_name, value in attrs.items():
            group_upper, key = attr_name.split(".", 1)
            key_lower = key.lower()

            # Convert numpy scalars to Python types
            if hasattr(value, "item"):
                value = value.item()

            # Convert int-encoded booleans back
            if key_lower in bool_keys and isinstance(value, int):
                value = bool(value)

            try:
                nml.set(**{key_lower: value})
            except (KeyError, TypeError):
                # Skip attributes not in the defaults (e.g. removed paths)
                pass

        return nml

    def _param(self, key: str) -> float | int | str | bool | None:
        """Get a single namelist parameter, or None if unavailable."""
        if self.params is None:
            return None
        try:
            return self.params.get(key)
        except KeyError:
            return None

    @property
    def N(self) -> int | None:
        """Number of grid cells."""
        return self._param("n")

    @property
    def tmax(self) -> float | None:
        """Maximum simulation time (seconds)."""
        return self._param("tmax")

    @property
    def Tref(self) -> float | None:
        """Reference temperature (K)."""
        return self._param("tref")

    @property
    def H(self) -> float | None:
        """Domain height (m)."""
        return self._param("h")

    @property
    def dz(self) -> float | None:
        """Grid spacing (m), computed as H / N."""
        h = self.H
        n = self.N
        if h is not None and n is not None:
            return h / n
        return None

    @property
    def volume_scaling(self) -> float | None:
        """Volume scaling factor."""
        return self._param("volume_scaling")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_time_idx(self, t: float) -> int:
        """Return the index of the time step nearest to *t* (seconds).

        Uses absolute difference — safe even when the time coordinate
        contains duplicate values (which would break ``sel(method="nearest")``).
        """
        return int(np.argmin(np.abs(self.time - t)))

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    @property
    def completed(self) -> bool:
        """Whether the simulation completed successfully (DONE marker exists)."""
        return self._done_path is not None

    @property
    def time(self) -> np.ndarray:
        """Time coordinate in seconds."""
        return self._ds["time"].values

    @property
    def z(self) -> np.ndarray:
        """Vertical coordinate in meters."""
        return self._ds["z"].values

    @property
    def radius(self) -> np.ndarray:
        """Droplet bin-center radii in microns."""
        return self._ds["radius"].values

    @property
    def bin_edges(self) -> np.ndarray:
        """Droplet bin edges in microns from the ``radius_edges`` netCDF variable."""
        return self._ds["radius_edges"].values

    # ------------------------------------------------------------------
    # Field access — netCDF variable names as properties
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> xr.DataArray:
        """Provide attribute-style access to netCDF variables.

        Variable names match the netCDF file exactly: ``T``, ``QV``,
        ``S``, ``W``, ``Np``, ``Nact``, ``LWC``, ``DSD``, etc.
        """
        if name in _FIELD_REGISTRY or name in self._ds.data_vars:
            return self._ds[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    @property
    def fields(self) -> list[str]:
        """List of available netCDF variable names."""
        registered = [v for v in _FIELD_REGISTRY if v in self._ds]
        dynamic = [v for v in self._ds.data_vars
                   if v.startswith("DSD_") and v not in registered]
        return registered + sorted(dynamic)

    def _dims_of(self, variable: str) -> str:
        """Return the dimension hint for a variable name.

        Returns
        -------
        str
            ``"tz"`` for (time, z), ``"t"`` for (time,),
            ``"tr"`` for (time, radius).

        Raises
        ------
        KeyError
            If *variable* is not a recognized field.
        """
        if variable in _FIELD_REGISTRY:
            return _FIELD_REGISTRY[variable]
        if variable.startswith("DSD_") and variable in self._ds.data_vars:
            return "tr"
        raise KeyError(
            f"Unknown variable '{variable}'. "
            f"Available: {self.fields}"
        )

    # ------------------------------------------------------------------
    # Core region
    # ------------------------------------------------------------------

    @property
    def core_region(self) -> tuple[float, float] | None:
        """The currently set core region ``(z_min, z_max)`` in meters.

        Returns ``None`` if no core region has been set.
        """
        return self._core_region

    def set_core_region(
        self,
        z_min: float | None = None,
        z_max: float | None = None,
        method: str | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
        sigma: float = 3.0,
    ) -> tuple[float, float]:
        """Set the core region, excluding boundary layers.

        The core region is used as the default spatial subset for
        ``domain_average``, ``time_average``, and other statistics methods.

        Call with explicit bounds::

            sim.set_core_region(z_min=0.05, z_max=0.95)

        Or auto-detect from the temperature and water vapor profiles::

            sim.set_core_region(method="gradient", t_start=3600)

        Parameters
        ----------
        z_min, z_max : float, optional
            Explicit core region bounds in meters. If both are given,
            *method* is ignored.
        method : {"gradient"}, optional
            Auto-detection method. Currently only ``"gradient"`` is
            supported, which identifies boundary layers from the
            second derivative of the time-averaged T and QV profiles.
        t_start : float, optional
            Start time (seconds) for the time-averaging window used
            by gradient detection. Boundary layers develop over time,
            so this should typically skip the initial transient.
            Defaults to ``tmax / 2``.
        t_end : float, optional
            End time (seconds) for the averaging window. Defaults to
            the last available time.
        sigma : float, optional
            Threshold multiplier for gradient detection. The boundary
            layer is where the curvature of T or QV exceeds
            ``mean + sigma * std`` of the interior curvature.
            Higher values produce a smaller (more conservative) BL
            estimate. Default is 3.0.

        Returns
        -------
        tuple of float
            The ``(z_min, z_max)`` that was set.
        """
        if z_min is not None and z_max is not None:
            self._core_region = (float(z_min), float(z_max))
            return self._core_region

        if method == "gradient":
            self._core_region = self._detect_boundary_layers(
                t_start=t_start, t_end=t_end, sigma=sigma,
            )
            return self._core_region

        raise ValueError(
            "Provide both z_min and z_max, or set method='gradient'."
        )

    def _detect_boundary_layers(
        self,
        t_start: float | None,
        t_end: float | None,
        sigma: float,
    ) -> tuple[float, float]:
        """Detect boundary layers from T and QV curvature profiles.

        The algorithm:

        1. Time-average T(z) and QV(z) over ``[t_start, t_end]``.
        2. Compute the second spatial derivative (curvature) of each.
        3. Estimate an interior noise level from the middle 50% of the
           domain as ``mean(|d²f/dz²|) + sigma * std(|d²f/dz²|)``.
        4. From each boundary, scan inward until the curvature of
           *both* T and QV drops below the threshold. This captures
           the full BL including the diffusion-dominated gradient
           and the transition region where curvature decays into
           the uniform core.
        """
        z = self.z
        n = len(z)

        if t_start is None:
            t_start = float(self.time[-1]) / 2.0
        if t_end is None:
            t_end = float(self.time[-1])

        T_avg = (
            self._ds["T"]
            .sel(time=slice(t_start, t_end))
            .mean(dim="time")
            .values
        )
        QV_avg = (
            self._ds["QV"]
            .sel(time=slice(t_start, t_end))
            .mean(dim="time")
            .values
        )

        # Second derivative (curvature)
        d2T = np.abs(np.gradient(np.gradient(T_avg, z), z))
        d2QV = np.abs(np.gradient(np.gradient(QV_avg, z), z))

        # Interior noise level from middle 50%
        q1, q3 = n // 4, 3 * n // 4
        thresh_T = np.mean(d2T[q1:q3]) + sigma * np.std(d2T[q1:q3])
        thresh_QV = np.mean(d2QV[q1:q3]) + sigma * np.std(d2QV[q1:q3])

        # Scan from bottom: find first point where BOTH are below threshold
        z_bot = z[0]
        for i in range(n // 2):
            if d2T[i] < thresh_T and d2QV[i] < thresh_QV:
                z_bot = z[i]
                break

        # Scan from top: find first point (going inward) where BOTH below
        z_top = z[-1]
        for i in range(n - 1, n // 2, -1):
            if d2T[i] < thresh_T and d2QV[i] < thresh_QV:
                z_top = z[i]
                break

        return (float(z_bot), float(z_top))

    def _z_slice(
        self,
        z_min: float | None = None,
        z_max: float | None = None,
        full_domain: bool = False,
    ) -> slice:
        """Resolve z bounds for derived-quantity methods.

        Resolution order:

        1. ``full_domain=True`` — bypass core region entirely.
        2. Explicit ``z_min``/``z_max`` — override core region for
           this call only.
        3. ``core_region`` — if previously set via ``set_core_region``.
        4. Full domain — fallback when nothing else is specified.

        Raises
        ------
        ValueError
            If *full_domain* is ``True`` and *z_min* or *z_max* are
            also given.
        """
        if full_domain and (z_min is not None or z_max is not None):
            raise ValueError(
                "Cannot specify both full_domain=True and z_min/z_max."
            )
        if full_domain:
            return slice(float(self.z[0]), float(self.z[-1]))
        if z_min is None and z_max is None and self._core_region is not None:
            z_min, z_max = self._core_region
        if z_min is None:
            z_min = float(self.z[0])
        if z_max is None:
            z_max = float(self.z[-1])
        return slice(z_min, z_max)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def profile(
        self,
        variable: str,
        t: float,
        z_min: float | None = None,
        z_max: float | None = None,
        full_domain: bool = False,
    ) -> xr.DataArray:
        """Extract a vertical profile at the time nearest to *t*.

        Parameters
        ----------
        variable : str
            A (time, z) variable name (e.g. ``"T"``, ``"S"``).
        t : float
            Target time in seconds.
        z_min, z_max : float, optional
            Spatial bounds in meters. If not given, uses the core
            region (if set), otherwise the full domain.
        full_domain : bool, optional
            Side flag that bypasses the core region and uses the
            full vertical domain. Cannot be combined with
            *z_min*/*z_max*.

        Returns
        -------
        xr.DataArray
            Profile with dimension (z,).
        """
        da = self._ds[variable]
        idx = self._nearest_time_idx(t)
        return da.isel(time=idx).sel(
            z=self._z_slice(z_min, z_max, full_domain)
        )

    def time_average(
        self,
        variable: str,
        t_start: float,
        t_end: float,
        z_min: float | None = None,
        z_max: float | None = None,
        full_domain: bool = False,
    ) -> xr.DataArray:
        """Time-averaged profile over [t_start, t_end].

        Parameters
        ----------
        variable : str
            A (time, z) variable name.
        t_start, t_end : float
            Time bounds in seconds.
        z_min, z_max : float, optional
            Spatial bounds in meters. If not given, uses the core
            region (if set), otherwise the full domain.
        full_domain : bool, optional
            Side flag that bypasses the core region and uses the
            full vertical domain. Cannot be combined with
            *z_min*/*z_max*.

        Returns
        -------
        xr.DataArray
            Mean profile with dimension (z,).
        """
        da = self._ds[variable]
        subset = da.sel(time=slice(t_start, t_end))
        if "z" in da.dims:
            subset = subset.sel(z=self._z_slice(z_min, z_max, full_domain))
        return subset.mean(dim="time")

    def domain_average(
        self,
        variable: str,
        z_min: float | None = None,
        z_max: float | None = None,
        full_domain: bool = False,
    ) -> xr.DataArray:
        """Spatial average time series of a (time, z) variable.

        Parameters
        ----------
        variable : str
            A (time, z) variable name.
        z_min, z_max : float, optional
            Spatial bounds in meters. If not given, uses the core
            region (if set), otherwise the full domain.
        full_domain : bool, optional
            Side flag that bypasses the core region and uses the
            full vertical domain. Cannot be combined with
            *z_min*/*z_max*.

        Returns
        -------
        xr.DataArray
            Time series with dimension (time,).
        """
        da = self._ds[variable]
        subset = da.sel(z=self._z_slice(z_min, z_max, full_domain))
        return subset.mean(dim="z")

    def dsd_average(
        self,
        t_start: float,
        t_end: float,
        normalize: str | None = "dlogr",
    ) -> dict[str, xr.DataArray]:
        """Time-averaged droplet size distributions.

        Averages the total DSD and each sub-category DSD (``DSD_1``,
        ``DSD_2``, etc.) found in the dataset.

        Parameters
        ----------
        t_start, t_end : float
            Time bounds in seconds.
        normalize : {"dlogr", "dr", None}, optional
            Normalization mode applied to each DSD independently:

            - ``"dlogr"`` (default): ``dC(r) / C / dlog10(r)``
              in µm⁻¹. Requires bin edges.
            - ``"dr"``: ``dC(r) / C / dr`` in µm⁻¹.
              Requires bin edges.
            - ``None``: raw time-averaged counts.

        Returns
        -------
        dict[str, xr.DataArray]
            Keyed by variable name: ``{"DSD": ..., "DSD_1": ...,
            "DSD_2": ..., ...}``. Each value has dimension (radius,).
        """
        if normalize is not None and normalize not in ("dlogr", "dr"):
            raise ValueError(
                f"Unknown normalize={normalize!r}. "
                f"Use 'dlogr', 'dr', or None."
            )

        # Find all DSD variables in the dataset
        dsd_vars = [v for v in self._ds.data_vars if v == "DSD" or v.startswith("DSD_")]

        # Precompute bin widths
        if normalize == "dlogr":
            bin_width = np.diff(np.log10(self.bin_edges))
        elif normalize == "dr":
            bin_width = np.diff(self.bin_edges)

        result: dict[str, xr.DataArray] = {}
        for var in dsd_vars:
            avg = self._ds[var].sel(time=slice(t_start, t_end)).mean(dim="time")

            if normalize is not None:
                total = float(avg.sum(dim="radius"))
                if total == 0:
                    avg = avg * 0.0
                else:
                    avg = avg / total / bin_width

            result[var] = avg

        return result

    def activation_fraction(self) -> xr.DataArray:
        """Activation fraction (Nact / Np) as a time series.

        Returns
        -------
        xr.DataArray
            Dimensioned (time,).  NaN where Np is zero.
        """
        return self._ds["Nact"] / self._ds["Np"]

    def spectral_width(
        self,
        t: float | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> float | xr.DataArray:
        """Number-weighted standard deviation of the droplet size distribution.

        Parameters
        ----------
        t : float, optional
            Single time in seconds.  Returns a scalar ``float``.
        t_start, t_end : float, optional
            Time range.  Returns an ``xr.DataArray`` with dimension
            (time,) over that range.  If neither *t* nor *t_start*/*t_end*
            is given, returns the full time series.

        Returns
        -------
        float or xr.DataArray
            Standard deviation of the DSD in microns.
        """
        dsd = self._ds["DSD"]
        r = dsd.radius  # xr.DataArray with dim "radius" for proper broadcasting

        if t is not None:
            dsd = dsd.isel(time=self._nearest_time_idx(t))
            total = dsd.sum(dim="radius")
            if float(total) == 0:
                return 0.0
            mean_r = float((dsd * r).sum(dim="radius") / total)
            var = float((dsd * (r - mean_r) ** 2).sum(dim="radius") / total)
            return float(np.sqrt(var))

        if t_start is not None or t_end is not None:
            dsd = dsd.sel(time=slice(t_start, t_end))

        total = dsd.sum(dim="radius")
        # Replace zeros with NaN to avoid division issues
        total_safe = total.where(total > 0)
        mean_r = (dsd * r).sum(dim="radius") / total_safe
        variance = (dsd * (r - mean_r) ** 2).sum(dim="radius") / total_safe
        return np.sqrt(variance)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_timeheight(
        self,
        variable: str,
        t_start: float | None = None,
        t_end: float | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        full_domain: bool = False,
        ax: object = None,
        **kwargs,
    ) -> object:
        """Time-height (Hovmoller) contour plot for a (time, z) variable.

        Parameters
        ----------
        variable : str
            A (time, z) variable name.
        t_start, t_end : float, optional
            Time bounds in seconds. Defaults to full time range.
        z_min, z_max : float, optional
            Spatial bounds in meters.
        full_domain : bool, optional
            Side flag that bypasses the core region and uses the
            full vertical domain. Cannot be combined with
            *z_min*/*z_max*.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates a new figure if ``None``.
        **kwargs
            Passed to ``pcolormesh``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        da = self._ds[variable]
        if t_start is not None or t_end is not None:
            da = da.sel(time=slice(t_start, t_end))
        if "z" in da.dims:
            da = da.sel(z=self._z_slice(z_min, z_max, full_domain))
        return _plot_timeheight(da, ax=ax, **kwargs)

    def plot_profile(
        self,
        variable: str,
        times: list[float],
        z_min: float | None = None,
        z_max: float | None = None,
        full_domain: bool = False,
        ax: object = None,
        **kwargs,
    ) -> object:
        """Vertical profiles at specified times, overlaid on one axis.

        Parameters
        ----------
        variable : str
            A (time, z) variable name.
        times : list of float
            Times in seconds at which to extract profiles.
        z_min, z_max : float, optional
            Spatial bounds in meters.
        full_domain : bool, optional
            Side flag that bypasses the core region and uses the
            full vertical domain. Cannot be combined with
            *z_min*/*z_max*.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates a new figure if ``None``.
        **kwargs
            Passed to each ``plot`` call.

        Returns
        -------
        matplotlib.axes.Axes
        """
        profiles = []
        for t in times:
            da = self.profile(variable, t, z_min=z_min, z_max=z_max,
                              full_domain=full_domain)
            profiles.append((da, f"t = {float(da.time):.0f} s"))
        return _plot_profile(profiles, ax=ax, **kwargs)

    def plot_timeseries(
        self,
        variables: str | list[str],
        z_min: float | None = None,
        z_max: float | None = None,
        full_domain: bool = False,
        ax: object = None,
        **kwargs,
    ) -> object:
        """Time series plot.

        For (time, z) variables, automatically applies ``domain_average``.
        For (time,) variables, plots directly.

        Parameters
        ----------
        variables : str or list of str
            One or more variable names.
        z_min, z_max : float, optional
            Spatial bounds for domain averaging of (time, z) variables.
        full_domain : bool, optional
            Side flag that bypasses the core region and uses the
            full vertical domain. Cannot be combined with
            *z_min*/*z_max*.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates a new figure if ``None``.
        **kwargs
            Passed to each ``plot`` call.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if isinstance(variables, str):
            variables = [variables]

        series = []
        for var in variables:
            dims = self._dims_of(var)
            if dims == "tz":
                da = self.domain_average(var, z_min=z_min, z_max=z_max,
                                         full_domain=full_domain)
            else:
                da = self._ds[var]
            series.append((da, var))
        return _plot_timeseries(series, ax=ax, **kwargs)

    def plot_spectrum(
        self,
        t_start: float,
        t_end: float,
        normalize: str | None = "dlogr",
        ax: object = None,
        **kwargs,
    ) -> object:
        """Droplet size distribution plot.

        Uses ``dsd_average`` to compute the time-averaged, normalized
        DSD and plots the total and each sub-category.

        Parameters
        ----------
        t_start, t_end : float
            Time bounds in seconds for the DSD average.
        normalize : {"dlogr", "dr", None}, optional
            Normalization mode. Default is ``"dlogr"``.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates a new figure if ``None``.
        **kwargs
            Passed to each ``plot`` call.

        Returns
        -------
        matplotlib.axes.Axes
        """
        spectra = self.dsd_average(t_start, t_end, normalize=normalize)
        return _plot_spectrum(
            spectra, self.radius, normalize=normalize, ax=ax, **kwargs
        )

    def plot_dsd_evolution(
        self,
        variables: str | list[str] | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
        cmaps: list[str] | None = None,
        alpha: float = 0.7,
        ax: object = None,
        **kwargs,
    ) -> object:
        """Radius-time evolution of the DSD.

        Parameters
        ----------
        variables : str, list of str, or None
            Which DSD variables to plot. Examples: ``"DSD"``,
            ``["DSD", "DSD_2"]``, ``["DSD_1", "DSD_2"]``.
            If ``None``, plots all DSD variables in the dataset.
        t_start, t_end : float, optional
            Time bounds in seconds. Defaults to full time range.
        cmaps : list of str, optional
            Matplotlib colormap names, one per variable.
        alpha : float, optional
            Transparency for each layer. Default is 0.7.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates a new figure if ``None``.
        **kwargs
            Passed to each ``pcolormesh`` call.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if variables is None:
            variables = ["DSD"]
        elif isinstance(variables, str):
            variables = [variables]

        dsd_dict = {
            v: self._ds[v].sel(time=slice(t_start, t_end))
            for v in variables
        }
        return _plot_dsd_evolution(
            dsd_dict, cmaps=cmaps, alpha=alpha, ax=ax, **kwargs
        )

    # ------------------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------------------

    @property
    def has_trajectories(self) -> bool:
        """Whether particle trajectory data is available."""
        return self._particles_path is not None

    def load_trajectories(self) -> xr.Dataset:
        """Load particle trajectory data from ``{name}_particles.nc``.

        The dataset is cached after the first call.

        Returns
        -------
        xarray.Dataset
            CF ragged array dataset with ``record`` and ``time_step``
            dimensions. Use :func:`trajectory_io.particles_at_timestep`
            or :func:`trajectory_io.trajectory_of` to slice it.

        Raises
        ------
        FileNotFoundError
            If no ``_particles.nc`` file was found.
        """
        if self._particles_ds is not None:
            return self._particles_ds
        if self._particles_path is None:
            raise FileNotFoundError(
                f"No particle trajectory file found for '{self.name}'. "
                f"Expected: {self.path / f'{self.name}_particles.nc'}"
            )
        self._particles_ds = _load_particles(self._particles_path)
        return self._particles_ds

    def trajectory_of(self, pid: int) -> xr.Dataset:
        """Extract full trajectory of a single particle by ID.

        Parameters
        ----------
        pid : int
            Particle ID.

        Returns
        -------
        xarray.Dataset
            Records for this particle across all time steps.
        """
        return _trajectory_of(self.load_trajectories(), pid)

    def particles_at_time(self, t: int) -> xr.Dataset:
        """Extract all particles at a given time step index.

        Parameters
        ----------
        t : int
            Time step index (0-based).

        Returns
        -------
        xarray.Dataset
            All particle records at time step *t*.
        """
        return _particles_at_timestep(self.load_trajectories(), t)

    def particle_ids(self) -> np.ndarray:
        """Return sorted array of unique particle IDs."""
        return _unique_particle_ids(self.load_trajectories())

    def plot_trajectory(
        self,
        pid: int,
        variable: str = "position",
        ax: object = None,
        **kwargs,
    ) -> object:
        """Plot a single particle's trajectory.

        Parameters
        ----------
        pid : int
            Particle ID.
        variable : str, optional
            Trajectory variable to plot on the y-axis (default
            ``"position"``).  Available variables: ``position``,
            ``radius``, ``temperature``, ``supersaturation``,
            ``water_vapor``, ``solute_radius``.
        ax : matplotlib axes, optional
            Axes to plot on.
        **kwargs
            Passed to ``ax.plot()``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        ds = self.load_trajectories()
        times = _record_times(ds)
        mask = ds["particle_id"].values == pid
        if not mask.any():
            raise ValueError(f"No records for particle_id={pid}")

        t = times[mask]
        y = ds[variable].values[mask]

        ax = _ensure_ax(ax)
        kwargs.setdefault("label", f"Particle {pid}")
        ax.plot(t, y, **kwargs)
        ax.set_xlabel("Time (s)")

        units = ds[variable].attrs.get("units", "")
        if units:
            ax.set_ylabel(f"{variable} ({units})")
        else:
            ax.set_ylabel(variable)

        ax.legend()
        return ax

    # ------------------------------------------------------------------
    # Collision data
    # ------------------------------------------------------------------

    @property
    def has_collisions(self) -> bool:
        """Whether collision binary data is available."""
        return self._collisions_path is not None

    def load_collisions(self) -> dict[str, np.ndarray]:
        """Load collision events from ``{name}_collisions.bin``.

        Returns
        -------
        dict
            ``"header"`` — structured array with N, H, domain_width,
            volume_scaling.
            ``"events"`` — structured array with id_keep, id_kill,
            r_keep, r_kill, r_after, position, time.

        Raises
        ------
        FileNotFoundError
            If no ``_collisions.bin`` file was found.
        """
        if self._collisions_path is None:
            raise FileNotFoundError(
                f"No collision file found for '{self.name}'. "
                f"Expected: {self.path / f'{self.name}_collisions.bin'}"
            )

        dt_header = np.dtype([
            ('N', '<i4'), ('H', '<f8'),
            ('domain_width', '<f8'), ('volume_scaling', '<f8'),
        ])
        dt_event = np.dtype([
            ('id_keep', '<i4'), ('id_kill', '<i4'),
            ('r_keep', '<f8'), ('r_kill', '<f8'), ('r_after', '<f8'),
            ('position', '<f8'), ('time', '<f8'),
        ])

        raw = np.fromfile(self._collisions_path, dtype=np.uint8)
        header = np.frombuffer(raw[:dt_header.itemsize], dtype=dt_header)
        events = np.frombuffer(raw[dt_header.itemsize:], dtype=dt_event)
        return {"header": header, "events": events}

    # ------------------------------------------------------------------
    # Multi-simulation comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare(
        simulations: list["CODTSimulation"],
        variable: str,
        plot_type: str = "timeseries",
        t: float | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        full_domain: bool = False,
        normalize: str | None = "dlogr",
        labels: list[str] | None = None,
        styles: list[dict] | None = None,
        ax: object = None,
        **kwargs,
    ) -> object:
        """Overlay the same diagnostic from multiple simulations.

        Parameters
        ----------
        simulations : list[CODTSimulation]
            Simulations to compare.
        variable : str
            NetCDF variable name (e.g. ``"T"``, ``"LWC"``, ``"S"``).
        plot_type : {"timeseries", "profile", "spectrum"}, optional
            Type of comparison plot (default ``"timeseries"``).
        t : float, optional
            For ``"profile"``: time in seconds at which to extract
            profiles.
        t_start, t_end : float, optional
            For ``"timeseries"``: time range to plot.
            For ``"spectrum"``: averaging window (required).
        z_min, z_max : float, optional
            Spatial bounds in meters.
        full_domain : bool, optional
            If ``True``, ignore core region settings.
        normalize : str or None, optional
            For ``"spectrum"``: normalization mode (default ``"dlogr"``).
        labels : list[str], optional
            Per-simulation labels (defaults to ``sim.name``).
        styles : list[dict], optional
            Per-simulation style kwargs (e.g. color, linestyle).
        ax : matplotlib axes, optional
            Axes to plot on.
        **kwargs
            Common styling passed to every plot call.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> CODTSimulation.compare([sim1, sim2], "LWC")
        >>> CODTSimulation.compare([sim1, sim2], "T",
        ...     plot_type="profile", t=30.0)
        >>> CODTSimulation.compare([sim1, sim2], "DSD",
        ...     plot_type="spectrum", t_start=30, t_end=60)
        """
        if not simulations:
            raise ValueError("Must provide at least one simulation.")

        labels = labels or [sim.name for sim in simulations]

        # Auto-assign evenly-spaced colors when no explicit styles given
        if styles is None:
            colors = comparison_colors(len(simulations))
            styles = [{"color": c} for c in colors]

        if plot_type == "timeseries":
            return CODTSimulation._compare_timeseries(
                simulations, variable, t_start, t_end,
                z_min, z_max, full_domain, labels, styles, ax, **kwargs,
            )
        elif plot_type == "profile":
            return CODTSimulation._compare_profile(
                simulations, variable, t,
                z_min, z_max, full_domain, labels, styles, ax, **kwargs,
            )
        elif plot_type == "spectrum":
            return CODTSimulation._compare_spectrum(
                simulations, t_start, t_end, normalize,
                labels, styles, ax, **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown plot_type={plot_type!r}. "
                f"Use 'timeseries', 'profile', or 'spectrum'."
            )

    @staticmethod
    def _compare_timeseries(
        simulations, variable, t_start, t_end,
        z_min, z_max, full_domain, labels, styles, ax, **kwargs,
    ):
        """Build time series comparison plot."""
        series: list[tuple[xr.DataArray, str]] = []
        for i, sim in enumerate(simulations):
            dims = sim._dims_of(variable)
            if dims == "tz":
                da = sim.domain_average(
                    variable, z_min=z_min, z_max=z_max,
                    full_domain=full_domain,
                )
            else:
                da = sim._ds[variable]
            if t_start is not None or t_end is not None:
                da = da.sel(time=slice(t_start, t_end))
            series.append((da, labels[i]))

        if styles is not None:
            ax = _ensure_ax(ax)
            for i, (da, label) in enumerate(series):
                merged = {**kwargs, **styles[i]}
                ax.plot(da.time.values, da.values, label=label, **merged)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(_get_label(series[0][0]))
            ax.legend()
            return ax

        return _plot_timeseries(series, ax=ax, **kwargs)

    @staticmethod
    def _compare_profile(
        simulations, variable, t,
        z_min, z_max, full_domain, labels, styles, ax, **kwargs,
    ):
        """Build profile comparison plot."""
        if t is None:
            raise ValueError("Must specify t for profile comparison.")

        profiles: list[tuple[xr.DataArray, str]] = []
        for i, sim in enumerate(simulations):
            da = sim.profile(
                variable, t, z_min=z_min, z_max=z_max,
                full_domain=full_domain,
            )
            profiles.append((da, labels[i]))

        if styles is not None:
            ax = _ensure_ax(ax)
            for i, (da, label) in enumerate(profiles):
                merged = {**kwargs, **styles[i]}
                ax.plot(da.values, da.z.values, label=label, **merged)
            ax.set_xlabel(_get_label(profiles[0][0]))
            ax.set_ylabel("Height (m)")
            ax.legend()
            return ax

        return _plot_profile(profiles, ax=ax, **kwargs)

    @staticmethod
    def _compare_spectrum(
        simulations, t_start, t_end, normalize,
        labels, styles, ax, **kwargs,
    ):
        """Build spectrum comparison plot."""
        if t_start is None or t_end is None:
            raise ValueError(
                "Must specify t_start and t_end for spectrum comparison."
            )

        ax = _ensure_ax(ax)
        for i, sim in enumerate(simulations):
            spectra = sim.dsd_average(t_start, t_end, normalize=normalize)
            da = spectra["DSD"]
            style = ({**kwargs, **styles[i]}) if styles else kwargs
            ax.plot(sim.radius, da.values, label=labels[i], **style)

        ax.set_xscale("log")
        ax.set_xlabel(r"Droplet Radius ($\mu$m)")
        if normalize == "dlogr":
            ax.set_ylabel(
                r"$\mathrm{d}C(r)\;/\;C\;\mathrm{d}\log(r)$ ($\mu$m$^{-1}$)"
            )
        elif normalize == "dr":
            ax.set_ylabel(
                r"$\mathrm{d}C(r)\;/\;C\;\mathrm{d}r$ ($\mu$m$^{-1}$)"
            )
        else:
            ax.set_ylabel("Counts")
        ax.legend()
        return ax

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def info(self) -> None:
        """Print a summary of the simulation."""
        print(f"Simulation:  {self.name}")
        print(f"Directory:   {self.path}")
        print(f"NC file:     {self._nc_path.name}")
        print(f"NML file:    {self._nml_path.name if self._nml_path else 'not found'}")
        print(f"Completed:   {self.completed}")
        print()

        print(f"Time:    {len(self.time)} steps, "
              f"[{self.time[0]:.1f}, {self.time[-1]:.1f}] s")
        print(f"Z:       {len(self.z)} cells, "
              f"[{self.z[0]:.4f}, {self.z[-1]:.4f}] m")
        print(f"Radius:  {len(self.radius)} bins, "
              f"[{self.radius[0]:.3f}, {self.radius[-1]:.3f}] um")
        print()

        if self.params is not None:
            print("Parameters:")
            for key in ("n", "tmax", "tref", "tdiff", "pres", "h",
                        "volume_scaling", "do_turbulence",
                        "do_microphysics", "do_special_effects"):
                val = self._param(key)
                if val is not None:
                    print(f"  {key:25s} = {val}")

            if self._param("do_collisions"):
                print()
                print("Collisions:")
                for key in ("coalescence_kernel", "do_coalescence"):
                    val = self._param(key)
                    if val is not None:
                        print(f"  {key:25s} = {val}")
                if "N_collisions" in self._ds:
                    total_coll = int(self._ds["N_collisions"].values.sum())
                    total_coal = int(self._ds["N_coalescences"].values.sum())
                    print(f"  {'total collisions':25s} = {total_coll}")
                    print(f"  {'total coalescences':25s} = {total_coal}")

    def __repr__(self) -> str:
        parts = [f"'{self.name}'"]
        if self.tmax is not None:
            parts.append(f"tmax={self.tmax:.0f}")
        if self.N is not None:
            parts.append(f"N={self.N}")
        if self.Tref is not None:
            parts.append(f"Tref={self.Tref:.1f}")
        return f"CODTSimulation({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all underlying netCDF datasets."""
        self._ds.close()
        if self._particles_ds is not None:
            self._particles_ds.close()

    def __del__(self) -> None:
        try:
            self._ds.close()
        except Exception:
            pass
        try:
            if self._particles_ds is not None:
                self._particles_ds.close()
        except Exception:
            pass
