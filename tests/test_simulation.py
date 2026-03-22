"""Tests for codt_tools.simulation (CODTSimulation)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

from codt_tools.simulation import CODTSimulation


# ── Initialization & file discovery ──────────────────────────────────

class TestInit:

    def test_from_directory(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert sim.name == "test_sim"
        assert sim.path == sim_dir

    def test_from_nc_path(self, sim_dir):
        sim = CODTSimulation(sim_dir / "test_sim.nc")
        assert sim.name == "test_sim"

    def test_from_stem_path(self, sim_dir):
        sim = CODTSimulation(sim_dir / "test_sim")
        assert sim.name == "test_sim"

    def test_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CODTSimulation(tmp_path / "nonexistent")

    def test_no_nc_in_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CODTSimulation(tmp_path)


# ── Metadata & properties ────────────────────────────────────────────

class TestMetadata:

    def test_completed(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert sim.completed is True

    def test_not_completed(self, sim_dir):
        (sim_dir / "DONE").unlink()
        sim = CODTSimulation(sim_dir)
        assert sim.completed is False

    def test_params_from_nc_attrs(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert sim.params is not None
        assert sim.N == 20
        assert sim.Tref == 293.15
        assert sim.H == 1.0

    def test_tmax(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert sim.tmax == 100.0

    def test_dz(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert sim.dz == pytest.approx(1.0 / 20)

    def test_volume_scaling(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert sim.volume_scaling == 13


# ── Coordinates ──────────────────────────────────────────────────────

class TestCoordinates:

    def test_time(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert len(sim.time) == 11
        np.testing.assert_allclose(sim.time[0], 0.0)
        np.testing.assert_allclose(sim.time[-1], 100.0)

    def test_z(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert len(sim.z) == 20

    def test_radius(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert len(sim.radius) == 10

    def test_bin_edges(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert len(sim.bin_edges) == 11


# ── Dynamic field access ─────────────────────────────────────────────

class TestFieldAccess:

    def test_getattr_2d(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        T = sim.T
        assert isinstance(T, xr.DataArray)
        assert T.dims == ("time", "z")

    def test_getattr_1d(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        lwc = sim.LWC
        assert isinstance(lwc, xr.DataArray)
        assert lwc.dims == ("time",)

    def test_getattr_missing(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        with pytest.raises(AttributeError):
            _ = sim.NONEXISTENT_VARIABLE

    def test_fields_list(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        f = sim.fields
        assert "T" in f
        assert "LWC" in f
        assert "DSD" in f


# ── Derived quantities ───────────────────────────────────────────────

class TestProfile:

    def test_shape(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        p = sim.profile("T", t=50.0)
        assert p.dims == ("z",)
        assert len(p.z) == 20

    def test_nearest_time(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        # t=53 should snap to t=50
        p = sim.profile("T", t=53.0)
        assert p.dims == ("z",)


class TestTimeAverage:

    def test_shape(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        avg = sim.time_average("T", t_start=20.0, t_end=60.0)
        assert avg.dims == ("z",)

    def test_subset(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        avg_full = sim.time_average("T", t_start=0.0, t_end=100.0)
        avg_half = sim.time_average("T", t_start=0.0, t_end=50.0)
        # Different windows should (very likely) give different means
        assert not np.allclose(avg_full.values, avg_half.values)


class TestDomainAverage:

    def test_shape(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        da = sim.domain_average("T")
        assert da.dims == ("time",)
        assert len(da.time) == 11

    def test_is_spatial_mean(self, sim_dir):
        """domain_average should equal manual mean over z."""
        sim = CODTSimulation(sim_dir)
        da = sim.domain_average("T")
        expected = sim.T.mean(dim="z")
        np.testing.assert_allclose(da.values, expected.values)


class TestDsdAverage:

    def test_returns_dict(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        result = sim.dsd_average(t_start=20.0, t_end=60.0)
        assert isinstance(result, dict)
        assert "DSD" in result

    def test_dsd_shape(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        result = sim.dsd_average(t_start=20.0, t_end=60.0)
        assert result["DSD"].dims == ("radius",)
        assert len(result["DSD"].radius) == 10


class TestActivationFraction:

    def test_shape(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        af = sim.activation_fraction()
        assert af.dims == ("time",)
        assert len(af.time) == 11

    def test_nan_when_no_particles(self, sim_dir):
        """First timestep has Np=0, should produce NaN."""
        sim = CODTSimulation(sim_dir)
        af = sim.activation_fraction()
        assert np.isnan(af.values[0])

    def test_range(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        af = sim.activation_fraction()
        valid = af.values[~np.isnan(af.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 1)


class TestSpectralWidth:

    def test_time_series(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        sw = sim.spectral_width()
        assert hasattr(sw, "time")
        assert len(sw.time) == 11

    def test_single_time(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        sw = sim.spectral_width(t=50.0)
        assert isinstance(sw, float)
        assert sw >= 0

    def test_time_range(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        sw = sim.spectral_width(t_start=20.0, t_end=60.0)
        assert len(sw.time) == 5  # t=20,30,40,50,60

    def test_nan_when_empty_dsd(self, sim_dir):
        """First timestep has DSD=0, should produce NaN."""
        sim = CODTSimulation(sim_dir)
        sw = sim.spectral_width()
        assert np.isnan(sw.values[0])

    def test_zero_dsd_returns_zero(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        sw = sim.spectral_width(t=0.0)
        assert sw == 0.0


# ── Core region ──────────────────────────────────────────────────────

class TestCoreRegion:

    def test_default_none(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert sim.core_region is None

    def test_set_core_region(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        sim.set_core_region(z_min=0.2, z_max=0.8)
        assert sim.core_region == (0.2, 0.8)

    def test_profile_respects_core(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        sim.set_core_region(z_min=0.2, z_max=0.8)
        p = sim.profile("T", t=50.0)
        assert float(p.z.min()) >= 0.2
        assert float(p.z.max()) <= 0.8

    def test_full_domain_overrides_core(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        sim.set_core_region(z_min=0.2, z_max=0.8)
        p = sim.profile("T", t=50.0, full_domain=True)
        assert len(p.z) == 20


# ── Plotting ─────────────────────────────────────────────────────────

class TestPlotMethods:
    """Smoke tests — just verify they run without errors."""

    def test_plot_timeheight(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        ax = sim.plot_timeheight("T")
        assert ax is not None

    def test_plot_profile(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        ax = sim.plot_profile("T", times=[30.0, 60.0])
        assert ax is not None

    def test_plot_timeseries(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        ax = sim.plot_timeseries("LWC")
        assert ax is not None

    def test_plot_spectrum(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        ax = sim.plot_spectrum(t_start=20.0, t_end=60.0)
        assert ax is not None

    def test_plot_dsd_evolution(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        ax = sim.plot_dsd_evolution()
        assert ax is not None


# ── Trajectories ─────────────────────────────────────────────────────

class TestTrajectories:

    def test_has_trajectories_true(self, sim_dir_with_particles):
        sim = CODTSimulation(sim_dir_with_particles)
        assert sim.has_trajectories is True

    def test_has_trajectories_false(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert sim.has_trajectories is False

    def test_load_trajectories(self, sim_dir_with_particles):
        sim = CODTSimulation(sim_dir_with_particles)
        ds = sim.load_trajectories()
        assert isinstance(ds, xr.Dataset)
        assert "particle_id" in ds

    def test_particle_ids(self, sim_dir_with_particles):
        sim = CODTSimulation(sim_dir_with_particles)
        pids = sim.particle_ids()
        np.testing.assert_array_equal(pids, [1, 2, 3, 4])

    def test_trajectory_of(self, sim_dir_with_particles):
        sim = CODTSimulation(sim_dir_with_particles)
        traj = sim.trajectory_of(1)
        assert np.all(traj["particle_id"].values == 1)

    def test_particles_at_time(self, sim_dir_with_particles):
        sim = CODTSimulation(sim_dir_with_particles)
        subset = sim.particles_at_time(0)
        assert len(subset.record) == 4

    def test_plot_trajectory(self, sim_dir_with_particles):
        sim = CODTSimulation(sim_dir_with_particles)
        ax = sim.plot_trajectory(pid=1, variable="radius")
        assert ax is not None


# ── Compare (static method) ──────────────────────────────────────────

class TestCompare:

    def test_timeseries(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        ax = CODTSimulation.compare([sim, sim], "LWC",
                                    plot_type="timeseries",
                                    labels=["A", "B"])
        assert ax is not None

    def test_profile(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        ax = CODTSimulation.compare([sim, sim], "T",
                                    plot_type="profile", t=50.0,
                                    labels=["A", "B"])
        assert ax is not None

    def test_spectrum(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        ax = CODTSimulation.compare([sim, sim], "DSD",
                                    plot_type="spectrum",
                                    t_start=20.0, t_end=60.0,
                                    labels=["A", "B"])
        assert ax is not None

    def test_domain_averaged_2d(self, sim_dir):
        """Timeseries compare on a (time,z) variable auto-averages."""
        sim = CODTSimulation(sim_dir)
        ax = CODTSimulation.compare([sim], "T", plot_type="timeseries")
        assert ax is not None

    def test_custom_styles(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        styles = [{"color": "red", "linestyle": "--"},
                  {"color": "blue", "linestyle": "-"}]
        ax = CODTSimulation.compare([sim, sim], "LWC",
                                    styles=styles, labels=["A", "B"])
        assert ax is not None

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CODTSimulation.compare([], "LWC")

    def test_bad_plot_type(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        with pytest.raises(ValueError, match="Unknown plot_type"):
            CODTSimulation.compare([sim], "LWC", plot_type="bad")

    def test_profile_requires_t(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        with pytest.raises(ValueError, match="Must specify t"):
            CODTSimulation.compare([sim], "T", plot_type="profile")

    def test_spectrum_requires_time_range(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        with pytest.raises(ValueError, match="Must specify t_start"):
            CODTSimulation.compare([sim], "DSD", plot_type="spectrum")


# ── Utilities ────────────────────────────────────────────────────────

class TestUtilities:

    def test_info(self, sim_dir, capsys):
        sim = CODTSimulation(sim_dir)
        sim.info()
        captured = capsys.readouterr()
        assert "test_sim" in captured.out
        assert "Completed" in captured.out

    def test_close_runs(self, sim_dir):
        """close() should not raise."""
        sim = CODTSimulation(sim_dir)
        sim.close()


# ── No microphysics ─────────────────────────────────────────────────

class TestNoMicrophysics:

    def test_no_dsd(self, sim_dir_no_micro):
        sim = CODTSimulation(sim_dir_no_micro)
        assert "DSD" not in sim.fields

    def test_profile_still_works(self, sim_dir_no_micro):
        sim = CODTSimulation(sim_dir_no_micro)
        p = sim.profile("T", t=50.0)
        assert p.dims == ("z",)
