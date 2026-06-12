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
        (sim_dir / "test_sim_DONE").unlink()
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

    def test_core_region_rejected_for_parcel(self, tmp_path):
        from conftest import _create_main_nc

        _create_main_nc(tmp_path, name="parcel_sim", mode="parcel")
        (tmp_path / "parcel_sim_DONE").write_text("2026-05-27\n")
        sim = CODTSimulation(tmp_path)
        with pytest.raises(ValueError, match="chamber mode"):
            sim.set_core_region(z_min=0.1, z_max=0.9)


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


# ── Eddy binary reader ─────────────────────────────────────────────

def _write_eddy_bin(path, mode="chamber", n_events=5):
    """Write a synthetic eddy binary for testing."""
    buf = bytearray()
    mode_flag = 0 if mode == "chamber" else 1
    buf += np.array([mode_flag], dtype='<i1').tobytes()
    buf += np.array([(100, 1.0)], dtype=[('N', '<i4'), ('H', '<f8')])[0].tobytes()
    if mode == "chamber":
        buf += np.array([1500.0, 1e5, 10.0, 293.15], dtype='<f8').tobytes()
    else:
        buf += np.array([0.1, 0.001, 0.01], dtype='<f8').tobytes()
    for i in range(n_events):
        buf += np.array([i + 5], dtype='<i4').tobytes()
        buf += np.array([i + 2], dtype='<i4').tobytes()
        buf += np.array([float(i) * 10.0], dtype='<f8').tobytes()
    path.write_bytes(bytes(buf))


class TestEddyReader:

    def test_has_eddies_false(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        assert not sim.has_eddies()

    def test_load_eddies_raises_without_file(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        with pytest.raises(FileNotFoundError):
            sim.load_eddies()

    def test_chamber_eddies(self, sim_dir):
        _write_eddy_bin(sim_dir / "test_sim_eddies.bin", mode="chamber", n_events=3)
        sim = CODTSimulation(sim_dir)
        assert sim.has_eddies()
        result = sim.load_eddies()
        hdr = result["header"]
        assert hdr["mode"] == "chamber"
        assert hdr["N"] == 100
        assert hdr["H"] == 1.0
        assert hdr["C2"] == 1500.0
        assert hdr["Tdiff"] == 10.0
        assert len(result["events"]) == 3
        assert result["events"][0]["M"] == 5
        assert result["events"][0]["L"] == 2
        assert result["events"][0]["time"] == 0.0

    def test_parcel_eddies(self, tmp_path):
        from conftest import _create_main_nc
        _create_main_nc(tmp_path, name="parcel_test", mode="parcel")
        (tmp_path / "parcel_test_DONE").write_text("2026-05-27\n")
        _write_eddy_bin(tmp_path / "parcel_test_eddies.bin", mode="parcel", n_events=2)
        sim = CODTSimulation(tmp_path)
        result = sim.load_eddies()
        hdr = result["header"]
        assert hdr["mode"] == "parcel"
        assert hdr["integral_length_scale"] == 0.1
        assert hdr["kolmogorov_length_scale"] == 0.001
        assert hdr["dissipation_rate"] == 0.01
        assert len(result["events"]) == 2


# ── Collision binary reader ────────────────────────────────────────

def _write_collision_bin(path, events):
    """Write a synthetic collision binary. *events* is a list of
    (id_keep, id_kill, r_keep, r_kill, r_after, position, time, coalesced).
    """
    buf = bytearray()
    buf += np.array([(100, 1.0, 0.001, 13.0)],
                    dtype=[('N', '<i4'), ('H', '<f8'),
                           ('domain_width', '<f8'),
                           ('volume_scaling', '<f8')])[0].tobytes()
    for ev in events:
        idk, idl, rk, rl, ra, pos, t, coal = ev
        buf += np.array([idk], dtype='<i4').tobytes()
        buf += np.array([idl], dtype='<i4').tobytes()
        buf += np.array([rk, rl, ra, pos, t], dtype='<f8').tobytes()
        buf += np.array([coal], dtype='<i1').tobytes()
    path.write_bytes(bytes(buf))


class TestCollisionReader:

    def test_load_collisions_raises_without_file(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        with pytest.raises(FileNotFoundError):
            sim.load_collisions()

    def test_coalesced_flag_roundtrips(self, sim_dir):
        # One coalescence (flag=1, r_after set), one bounce (flag=0, r_after=0).
        events = [
            (7, 8, 5.0e-6, 4.0e-6, 6.0e-6, 0.5, 1.0, 1),
            (3, 9, 2.0e-6, 1.0e-6, 0.0, 0.2, 2.0, 0),
        ]
        _write_collision_bin(sim_dir / "test_sim_collisions.bin", events)
        sim = CODTSimulation(sim_dir)
        result = sim.load_collisions()

        hdr = result["header"]
        assert hdr["N"][0] == 100
        assert hdr["volume_scaling"][0] == 13.0

        ev = result["events"]
        # The trailing i1 byte must be read, so the record stride stays
        # aligned and both events decode (a missing byte would mis-slice).
        assert len(ev) == 2
        assert "coalesced" in ev.dtype.names
        assert ev["coalesced"][0] == 1
        assert ev["coalesced"][1] == 0
        assert ev["id_keep"][0] == 7
        assert ev["id_kill"][1] == 9
        assert ev["r_after"][0] == 6.0e-6
        assert ev["time"][1] == 2.0


# ── Budgets ────────────────────────────────────────────────────────

def _create_budget_nc(path, name="test_sim"):
    """Write a main NC with a closing liquid-water budget for tests."""
    import netCDF4 as nc
    nc_path = path / f"{name}.nc"
    n_time = 4
    times = np.arange(n_time, dtype=np.float64) * 10.0
    vol = 13.0 * 0.001 ** 2 * 1.0  # volume_scaling * domain_width**2 * H

    # Per-interval liquid mass changes (kg): inject + condensation - fallout.
    inj = np.array([0.0, 2.0, 1.0, 0.0])
    cond = np.array([0.0, 0.5, 0.5, 0.5])
    fall = np.array([0.0, 0.0, 0.5, 1.0])
    net_increment = inj + cond - fall          # liquid mass added each step
    m_liquid = np.cumsum(net_increment)        # total liquid mass (kg)
    lwc = m_liquid * 1000.0 / vol              # kg -> g/m**3

    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.setncattr("PARAMETERS.volume_scaling", 13)
        ds.setncattr("PARAMETERS.H", 1.0)
        ds.setncattr("PARAMETERS.do_microphysics", 1)
        ds.setncattr("PARAMETERS.simulation_mode", "chamber")
        ds.createDimension("time", None)
        ds.createDimension("z", 2)
        v = ds.createVariable("time", "f8", ("time",)); v[:] = times
        v = ds.createVariable("z", "f8", ("z",)); v[:] = [0.0, 1.0]
        for vname, data in [
            ("budget_inject_liquid_mass", inj),
            ("budget_condensation", cond),
            ("budget_fallout_liquid_mass", fall),
            ("budget_diffusion_delta_WV", np.full(n_time, 1e-4)),  # kg/kg
            ("LWC", lwc),
        ]:
            v = ds.createVariable(vname, "f8", ("time",))
            v[:] = data
    (path / f"{name}_DONE").write_text("2026-06-08\n")
    return path


class TestBudgets:

    def test_domain_volume(self, sim_dir):
        sim = CODTSimulation(sim_dir)
        # volume_scaling=13, H=1.0, domain_width=0.001
        assert sim.domain_volume() == pytest.approx(13.0 * 0.001 ** 2 * 1.0)

    def test_budget_totals_cumulative(self, tmp_path):
        _create_budget_nc(tmp_path)
        sim = CODTSimulation(tmp_path)
        totals = sim.budget_totals(cumulative=True)
        # Mass terms are converted kg -> g/m**3 of domain volume.
        kg_to_g_m3 = 1000.0 / (13.0 * 0.001 ** 2 * 1.0)
        # cumulative inject = 0+2+1+0 = 3 kg
        assert float(totals["budget_inject_liquid_mass"][-1]) == pytest.approx(
            3.0 * kg_to_g_m3)
        assert totals["budget_inject_liquid_mass"].attrs["units"] == "g/m3"
        # WV terms are converted kg/kg -> g/kg
        assert float(totals["budget_diffusion_delta_WV"][-1]) == pytest.approx(
            4 * 1e-4 * 1000.0)
        assert totals["budget_diffusion_delta_WV"].attrs["units"] == "g/kg"
        # raw increments preserved when cumulative=False
        raw = sim.budget_totals(cumulative=False)
        assert float(raw["budget_inject_liquid_mass"][1]) == pytest.approx(
            2.0 * kg_to_g_m3)

    def test_budget_totals_requires_budgets(self, sim_dir_no_micro):
        sim = CODTSimulation(sim_dir_no_micro)
        with pytest.raises(ValueError, match="budget"):
            sim.budget_totals()

    def test_budget_closure_residual_near_zero(self, tmp_path):
        _create_budget_nc(tmp_path)
        sim = CODTSimulation(tmp_path)
        result = sim.budget_closure()
        # Terms are reported in g/m**3 of domain volume.
        vol = 13.0 * 0.001 ** 2 * 1.0
        # inject(3.0) - fallout(1.5) + condensation(1.5) = 3.0 kg
        expected = 3.0 * 1000.0 / vol
        assert result["sources_net"] == pytest.approx(expected)
        assert result["delta_lwc_mass"] == pytest.approx(expected)
        assert result["residual"] == pytest.approx(0.0, abs=1e-6 * expected)
        assert abs(result["relative_residual"]) < 1e-6
