"""Plotting functions for CODT simulation data.

These are called by the ``CODTSimulation.plot_*`` methods. They can
also be used standalone with raw xarray DataArrays.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def comparison_colors(n: int, cmap: str = "viridis") -> list[str]:
    """Return *n* evenly-spaced hex colors from a colormap.

    Samples the colormap at equally spaced intervals, avoiding the
    extreme endpoints (which are often too light or too dark) by
    using the range [0.1, 0.9].

    Parameters
    ----------
    n : int
        Number of colors needed.
    cmap : str, optional
        Matplotlib colormap name (default ``"viridis"``).

    Returns
    -------
    list[str]
        Hex color strings.
    """
    cm = plt.colormaps[cmap]
    if n == 1:
        positions = [0.5]
    else:
        positions = np.linspace(0.1, 0.9, n)
    return [mcolors.to_hex(cm(p)) for p in positions]


def _get_label(da: xr.DataArray) -> str:
    """Build an axis label from a DataArray's attributes."""
    long_name = da.attrs.get("long_name", da.attrs.get("long name", da.name or ""))
    units = da.attrs.get("units", "")
    if units:
        return f"{long_name} ({units})"
    return long_name


def _ensure_ax(ax: matplotlib.axes.Axes | None) -> matplotlib.axes.Axes:
    """Return the given axes, or create a new figure and axes."""
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_timeheight(
    da: xr.DataArray,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Time-height (Hovmoller) contour plot for a (time, z) variable.

    Parameters
    ----------
    da : xr.DataArray
        Data with dimensions (time, z).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if ``None``.
    **kwargs
        Passed to ``ax.pcolormesh``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("shading", "auto")
    mesh = ax.pcolormesh(
        da.time.values, da.z.values, da.values.T, **kwargs
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title(_get_label(da))
    ax.figure.colorbar(mesh, ax=ax)
    return ax


def plot_profile(
    profiles: list[tuple[xr.DataArray, str]],
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Vertical profiles overlaid on one axis.

    Parameters
    ----------
    profiles : list of (DataArray, label) tuples
        Each DataArray has dimension (z,). The label is used in the
        legend.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if ``None``.
    **kwargs
        Passed to each ``ax.plot`` call.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    for da, label in profiles:
        ax.plot(da.values, da.z.values, label=label, **kwargs)
    if profiles:
        ax.set_xlabel(_get_label(profiles[0][0]))
    ax.set_ylabel("Height (m)")
    ax.legend()
    return ax


def plot_timeseries(
    series: list[tuple[xr.DataArray, str]],
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Time series overlaid on one axis.

    Parameters
    ----------
    series : list of (DataArray, label) tuples
        Each DataArray has dimension (time,).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if ``None``.
    **kwargs
        Passed to each ``ax.plot`` call.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)
    for da, label in series:
        ax.plot(da.time.values, da.values, label=label, **kwargs)
    ax.set_xlabel("Time (s)")
    if series:
        ax.set_ylabel(_get_label(series[0][0]))
    if len(series) > 1:
        ax.legend()
    return ax


def plot_spectrum(
    spectra: dict[str, xr.DataArray],
    radius: np.ndarray,
    normalize: str | None = "dlogr",
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Droplet size distribution plot.

    Parameters
    ----------
    spectra : dict[str, xr.DataArray]
        Output from ``CODTSimulation.dsd_average``. Keys are variable
        names (``"DSD"``, ``"DSD_1"``, etc.).
    radius : np.ndarray
        Bin-center radii in microns (x-axis values).
    normalize : {"dlogr", "dr", None}, optional
        Used only for labeling the y-axis. Should match the
        normalization used when computing *spectra*.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if ``None``.
    **kwargs
        Passed to each ``ax.plot`` call.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _ensure_ax(ax)

    for name, da in spectra.items():
        ax.plot(radius, da.values, label=name, **kwargs)

    ax.set_xscale("log")
    ax.set_xlabel(r"Droplet Radius ($\mu$m)")

    if normalize == "dlogr":
        ax.set_ylabel(r"$\mathrm{d}C(r)\;/\;C\;\mathrm{d}\log(r)$ ($\mu$m$^{-1}$)")
    elif normalize == "dr":
        ax.set_ylabel(r"$\mathrm{d}C(r)\;/\;C\;\mathrm{d}r$ ($\mu$m$^{-1}$)")
    else:
        ax.set_ylabel("Counts")

    ax.legend()
    return ax


# Default colormaps for DSD categories, cycled if more are needed.
_DSD_CMAPS = ["Blues", "Reds", "Greens", "Purples", "Oranges"]


def plot_dsd_evolution(
    dsd_arrays: dict[str, xr.DataArray],
    cmaps: list[str] | None = None,
    alpha: float = 0.7,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Radius-time evolution of the droplet size distribution.

    Overlays each DSD category as a pcolormesh layer with its own
    colormap. Zero-count regions are fully transparent.

    Parameters
    ----------
    dsd_arrays : dict[str, xr.DataArray]
        Keyed by variable name (``"DSD_1"``, ``"DSD_2"``, etc.).
        Each DataArray has dimensions (time, radius). The total
        ``"DSD"`` key is excluded automatically if sub-categories
        are present.
    cmaps : list of str, optional
        Matplotlib colormap names, one per category. Defaults to
        ``["Blues", "Reds", "Greens", "Purples", "Oranges"]``.
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
    ax = _ensure_ax(ax)
    if cmaps is None:
        cmaps = _DSD_CMAPS

    for i, key in enumerate(dsd_arrays):
        da = dsd_arrays[key]
        data = da.values.astype(float)
        time = da.time.values

        # Use bin centers for the y-axis; bin_edges are used to set
        # the visual extent via y-axis limits.
        y = da.radius.values

        # Build a colormap where zero maps to fully transparent
        base_cmap = plt.colormaps[cmaps[i % len(cmaps)]]
        colors = base_cmap(np.linspace(0, 1, 256))
        colors[0, 3] = 0.0  # transparent for the lowest value
        cmap = mcolors.ListedColormap(colors)

        # Mask zeros so they don't render
        masked = np.where(data > 0, data, np.nan)

        # pcolormesh(X, Y, C) with shading='nearest':
        # X=time (n_time,), Y=bin_centers (n_bins,), C.T=(n_bins, n_time)
        mesh = ax.pcolormesh(
            time, y, masked.T, cmap=cmap, alpha=alpha,
            shading="nearest", **kwargs
        )

        ax.figure.colorbar(mesh, ax=ax, label=key, pad=0.01 + i * 0.05)

    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Droplet Radius ($\mu$m)")
    ax.set_title("DSD Evolution")
    return ax
