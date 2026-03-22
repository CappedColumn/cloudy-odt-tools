"""Shared fixtures for codt_tools tests.

Creates minimal synthetic netCDF files that mimic real CODT output,
so tests don't require actual simulation data.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_main_nc(path: Path, name: str = "test_sim", n_time: int = 11,
                    n_z: int = 20, n_bins: int = 10,
                    microphysics: bool = True) -> Path:
    """Write a minimal main output netCDF file.

    Returns the path to the created file.
    """
    nc_path = path / f"{name}.nc"
    times = np.arange(n_time, dtype=np.float64) * 10.0  # 0, 10, ..., 100
    z = np.linspace(0.005, 0.995, n_z)
    rng = np.random.default_rng(42)

    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        # Global attributes (namelist params)
        ds.setncattr("PARAMETERS.N", n_z)
        ds.setncattr("PARAMETERS.tmax", float(times[-1]))
        ds.setncattr("PARAMETERS.Tref", 293.15)
        ds.setncattr("PARAMETERS.H", 1.0)
        ds.setncattr("PARAMETERS.volume_scaling", 13)
        ds.setncattr("PARAMETERS.simulation_name", name)
        ds.setncattr("PARAMETERS.do_turbulence", 1)
        ds.setncattr("PARAMETERS.do_microphysics", 1 if microphysics else 0)

        # Dimensions
        ds.createDimension("time", None)  # unlimited
        ds.createDimension("z", n_z)

        v = ds.createVariable("time", "f8", ("time",))
        v.units = "seconds"
        v.long_name = "Time"
        v[:] = times

        v = ds.createVariable("z", "f8", ("z",))
        v.units = "meters"
        v.long_name = "Height"
        v[:] = z

        # Always-present fields
        for vname, long in [("T", "Temperature"), ("QV", "Water Vapor Mixing Ratio"),
                            ("S", "Supersaturation"), ("W", "W-Velocity")]:
            v = ds.createVariable(vname, "f4", ("time", "z"))
            v.long_name = long
            v.units = "celsius" if vname == "T" else ("g/kg" if vname == "QV" else ("%" if vname == "S" else "m/s"))
            v[:] = rng.normal(size=(n_time, n_z)).astype(np.float32)

        if microphysics:
            n_edges = n_bins + 1
            ds.createDimension("radius", n_bins)
            ds.createDimension("radius_edges", n_edges)

            bin_edges = np.logspace(-1, 2, n_edges)  # ~0.1 to 100 microns
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            v = ds.createVariable("radius", "f8", ("radius",))
            v.long_name = "Droplet Bin Centers"
            v.units = "microns"
            v[:] = bin_centers

            v = ds.createVariable("radius_edges", "f8", ("radius_edges",))
            v.long_name = "Droplet Bin Edges"
            v.units = "microns"
            v[:] = bin_edges

            # DSD — put some counts in the middle bins
            dsd = rng.poisson(5, size=(n_time, n_bins)).astype(np.float32)
            # Zero out first timestep to test NaN handling
            dsd[0, :] = 0
            v = ds.createVariable("DSD", "f4", ("time", "radius"))
            v.long_name = "Droplet Size Distribution"
            v.units = "#"
            v[:] = dsd

            for cat in ["DSD_1", "DSD_2"]:
                v = ds.createVariable(cat, "f4", ("time", "radius"))
                v.long_name = f"Droplet Size Distribution - {cat[-1]}"
                v.units = "#"
                v[:] = rng.poisson(2, size=(n_time, n_bins)).astype(np.float32)

            # Scalar time series
            np_vals = rng.integers(100, 500, size=n_time).astype(np.float32)
            np_vals[0] = 0
            nact_vals = (np_vals * rng.uniform(0.1, 0.5, size=n_time)).astype(np.float32)
            nact_vals[0] = 0

            for vname, long, units, vals in [
                ("Np", "Number of Particles", "#", np_vals),
                ("Nact", "Number of Activated Particles", "#", nact_vals),
                ("Nun", "Number of Unactivated Particles", "#", np_vals - nact_vals),
                ("Ravg", "Average Particle Radius", "um",
                 rng.uniform(1, 10, size=n_time).astype(np.float32)),
                ("LWC", "Liquid Water Content", "g/m3",
                 rng.uniform(0, 0.01, size=n_time).astype(np.float32)),
            ]:
                v = ds.createVariable(vname, "f4", ("time",))
                v.long_name = long
                v.units = units
                v[:] = vals

    return nc_path


def _create_particles_nc(path: Path, name: str = "test_sim",
                         n_timesteps: int = 5,
                         n_particles: int = 4) -> Path:
    """Write a minimal particle trajectory netCDF file.

    Creates a CF ragged array with *n_particles* particles across
    *n_timesteps* time steps, where each particle appears at every step.
    """
    nc_path = path / f"{name}_particles.nc"
    rng = np.random.default_rng(99)

    total_records = n_timesteps * n_particles
    row_sizes = np.full(n_timesteps, n_particles, dtype=np.int32)
    times = np.arange(n_timesteps, dtype=np.float64) * 10.0

    # Per-record arrays
    pids = np.tile(np.arange(1, n_particles + 1), n_timesteps)
    positions = rng.uniform(0, 1, size=total_records)
    radii = rng.uniform(1, 10, size=total_records)
    activated = rng.integers(0, 2, size=total_records)

    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("record", None)
        ds.createDimension("time_step", None)

        v = ds.createVariable("time", "f8", ("time_step",))
        v.units = "seconds"
        v[:] = times

        v = ds.createVariable("row_sizes", "i4", ("time_step",))
        v.cf_role = "ragged_row_sizes"
        v.sample_dimension = "record"
        v[:] = row_sizes

        for vname, dtype, data in [
            ("particle_id", "i4", pids),
            ("aerosol_id", "i4", np.ones(total_records, dtype=np.int32)),
            ("gridcell", "i4", rng.integers(1, 20, size=total_records)),
            ("position", "f4", positions),
            ("temperature", "f4", rng.normal(20, 1, size=total_records)),
            ("water_vapor", "f4", rng.uniform(5, 10, size=total_records)),
            ("supersaturation", "f4", rng.normal(0, 0.5, size=total_records)),
            ("radius", "f4", radii),
            ("solute_radius", "f4", rng.uniform(0.01, 0.1, size=total_records)),
            ("activated", "i4", activated),
            ("aerosol_category", "i4", np.ones(total_records, dtype=np.int32)),
        ]:
            v = ds.createVariable(vname, dtype, ("record",))
            v[:] = data

    return nc_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_dir(tmp_path):
    """Create a simulation output directory with main NC + DONE marker.

    Returns the directory path (not the .nc path).
    """
    name = "test_sim"
    _create_main_nc(tmp_path, name=name)
    (tmp_path / "DONE").write_text("2026-03-22 12:00:00\n")
    return tmp_path


@pytest.fixture
def sim_dir_with_particles(tmp_path):
    """Simulation directory with main NC + particle trajectories + DONE."""
    name = "test_sim"
    _create_main_nc(tmp_path, name=name)
    _create_particles_nc(tmp_path, name=name)
    (tmp_path / "DONE").write_text("2026-03-22 12:00:00\n")
    return tmp_path


@pytest.fixture
def sim_dir_no_micro(tmp_path):
    """Simulation directory without microphysics variables."""
    name = "test_sim"
    _create_main_nc(tmp_path, name=name, microphysics=False)
    (tmp_path / "DONE").write_text("2026-03-22 12:00:00\n")
    return tmp_path


@pytest.fixture
def particles_nc(tmp_path):
    """Standalone particle trajectory file for trajectory_io tests."""
    return _create_particles_nc(tmp_path)


@pytest.fixture
def aerosol_data():
    """Minimal aerosol data dict for aerosol_io tests."""
    return {
        "aerosol_name": "NaCl",
        "n_ions": np.array([2], dtype=np.int32),
        "molar_mass": np.array([58.4428e-3]),
        "solute_density": np.array([2163.0]),
        "edge_radii": np.array([60.0, 70.0, 4930.0]),
        "category": np.array([1, 2], dtype=np.int32),
        "cumulative_frequency": np.array([[1.0, 1.0]]),
        "injection_time": np.array([0.0]),
        "injection_rate": np.array([5.5e5]),
    }
