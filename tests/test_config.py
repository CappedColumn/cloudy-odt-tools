"""Tests for CODTConfig."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codt_tools.config import (
    CODTConfig,
    InjectionData,
    Namelist,
    ParcelInput,
    lem_turbulence_scales,
)


# ======================================================================
# Derived LEM turbulence scales (mirror of CODT src/LEM.f90)
# ======================================================================


class TestLEMTurbulenceScales:
    """Tests for lem_turbulence_scales()."""

    def test_grid_limited_regime(self) -> None:
        # eps = 0.01, H = 1, N = 1025: 6*dz just exceeds the diffusivity
        # length scale, so the floor is exactly 6 cells and f ~ 1.
        s = lem_turbulence_scales(1025, 1.0, 0.01)
        assert s["smallest_eddy_gridpoints"] == 6
        assert s["diffusivity_enhancement"] == pytest.approx(1.001, abs=1e-3)

    def test_diffusivity_limited_regime_overshoots(self) -> None:
        # N = 1045 puts 6*dz just below the handoff, so the 3-cell quantum
        # rounds up to 9 cells and f jumps -- f is a step function of the
        # grid, not ~1.
        s = lem_turbulence_scales(1045, 1.0, 0.01)
        assert s["smallest_eddy_gridpoints"] == 9
        assert s["diffusivity_enhancement"] == pytest.approx(1.675, abs=1e-3)

    def test_enhancement_never_below_one(self) -> None:
        for n in range(200, 3000, 37):
            s = lem_turbulence_scales(n, 1.0, 0.01)
            assert s["diffusivity_enhancement"] >= 1.0

    def test_gridpoints_consistent_with_scale(self) -> None:
        n, h = 2000, 1.0
        s = lem_turbulence_scales(n, h, 0.01)
        assert s["smallest_eddy_scale"] == pytest.approx(
            s["smallest_eddy_gridpoints"] * h / n
        )
        assert s["smallest_eddy_gridpoints"] % 3 == 0

    def test_matches_codt_reference_values(self) -> None:
        # Pinned against the LEM.* global attributes of a real CODT v3.0.0
        # (e1c03be) parcel run at N=2000, H=1, eps=0.01. Guards the
        # single-precision `1./3.` / `4./3.` literals: dropping them shifts
        # diffusivity_length_scale by ~4e-8 relative, which this catches.
        s = lem_turbulence_scales(2000, 1.0, 0.01)
        assert s["diffusivity_length_scale"] == pytest.approx(
            0.005849128279333603, rel=1e-7
        )
        assert s["actual_kolmogorov_scale"] == pytest.approx(
            0.00075762133825086, rel=1e-7
        )
        assert s["diffusivity_enhancement"] == pytest.approx(
            1.0345388541686187, rel=1e-7
        )
        assert s["smallest_eddy_gridpoints"] == 12
        assert s["smallest_eddy_scale"] == pytest.approx(0.006)

    def test_kolmogorov_scale_is_diagnostic_only(self) -> None:
        # (nu**3/eps)**0.25 is always ~8x below the diffusivity handoff, so
        # it can never win the max that sets the smallest eddy.
        s = lem_turbulence_scales(2000, 1.0, 0.01)
        assert s["actual_kolmogorov_scale"] < s["diffusivity_length_scale"]
        assert s["actual_kolmogorov_scale"] < s["smallest_eddy_scale"]


# ======================================================================
# Init
# ======================================================================


class TestCODTConfigInit:
    """Tests for CODTConfig construction."""

    def test_defaults(self) -> None:
        cfg = CODTConfig()
        assert cfg.name == "default_sim"
        assert isinstance(cfg.params, Namelist)
        assert isinstance(cfg.injection, InjectionData)

    def test_name_property(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_name="my_sim")
        assert cfg.name == "my_sim"

    def test_from_namelist_path(self, tmp_path: Path) -> None:
        """Load from an existing directory with all three files."""
        # Write a config to disk, then reload
        cfg = CODTConfig()
        cfg.set(simulation_name="reload_test", tref=25.0)
        cfg.write(tmp_path)

        reloaded = CODTConfig(tmp_path / "params.nml")
        assert reloaded.name == "reload_test"
        assert reloaded.params.get("tref") == 25.0
        assert reloaded.injection.aerosol_name == "NaCl"
        assert len(reloaded.injection.dsd_bin_edges) == 201

    def test_from_namelist_missing_data_files(self, tmp_path: Path) -> None:
        """Loading from namelist alone falls back to defaults for data."""
        nml = Namelist()
        nml.set(simulation_name="nml_only")
        nml.write(tmp_path / "params.nml")

        cfg = CODTConfig(tmp_path / "params.nml")
        assert cfg.name == "nml_only"
        # Should fall back to defaults (not crash)
        assert isinstance(cfg.injection, InjectionData)


# ======================================================================
# Setters
# ======================================================================


class TestCODTConfigSetters:
    """Tests for set, set_injection, set_bins."""

    def test_set_namelist_params(self) -> None:
        cfg = CODTConfig()
        cfg.set(tref=22.0, tmax=7200.0, simulation_name="setter_test")
        assert cfg.params.get("tref") == 22.0
        assert cfg.params.get("tmax") == 7200.0
        assert cfg.name == "setter_test"

    def test_set_injection(self) -> None:
        cfg = CODTConfig()
        cfg.set_injection(aerosol_name="KCl", injection_rate=1.0e5)
        assert cfg.injection.aerosol_name == "KCl"
        np.testing.assert_allclose(cfg.injection.injection_rate, [1.0e5])

    def test_set_bins(self) -> None:
        cfg = CODTConfig()
        new_edges = np.linspace(0.1, 50.0, 101)
        cfg.set_bins(new_edges)
        assert len(cfg.injection.dsd_bin_edges) == 101
        np.testing.assert_allclose(cfg.injection.dsd_bin_edges, new_edges)

    def test_set_bad_param_raises(self) -> None:
        cfg = CODTConfig()
        with pytest.raises(KeyError):
            cfg.set(nonexistent_param=42)

    def test_set_injection_bad_attr_raises(self) -> None:
        cfg = CODTConfig()
        with pytest.raises(AttributeError):
            cfg.set_injection(nonexistent_attr=42)


# ======================================================================
# Dot-access
# ======================================================================


class TestCODTConfigDotAccess:
    """Tests for attribute-style get/set of namelist parameters."""

    def test_get(self) -> None:
        cfg = CODTConfig()
        assert cfg.tref == 20.0
        assert cfg.simulation_name == "default_sim"
        assert cfg.do_microphysics is True

    def test_set(self) -> None:
        cfg = CODTConfig()
        cfg.tref = 22.0
        cfg.simulation_name = "dot_test"
        assert cfg.tref == 22.0
        assert cfg.params.get("tref") == 22.0
        assert cfg.simulation_name == "dot_test"

    def test_set_type_checking(self) -> None:
        cfg = CODTConfig()
        with pytest.raises(TypeError):
            cfg.tref = "not a number"

    def test_invalid_attr_raises(self) -> None:
        cfg = CODTConfig()
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_param

    def test_own_attrs_unaffected(self) -> None:
        cfg = CODTConfig()
        assert isinstance(cfg.params, Namelist)
        assert isinstance(cfg.injection, InjectionData)

    def test_dir_includes_params(self) -> None:
        cfg = CODTConfig()
        d = dir(cfg)
        assert "tref" in d
        assert "simulation_name" in d
        assert "do_microphysics" in d
        assert "params" in d


# ======================================================================
# Write
# ======================================================================


class TestCODTConfigWrite:
    """Tests for write()."""

    def test_creates_all_files(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_name="write_test")
        cfg.write(tmp_path / "run")

        assert (tmp_path / "run" / "params.nml").is_file()
        assert (tmp_path / "run" / "aerosol_input.nc").is_file()

    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "dir"
        cfg = CODTConfig()
        cfg.write(target)
        assert target.is_dir()
        assert (target / "params.nml").is_file()

    def test_sets_relative_data_paths(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.write(tmp_path)

        reloaded = Namelist(tmp_path / "params.nml")
        assert reloaded.get("aerosol_file") == "aerosol_input.nc"

    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_name="roundtrip", tref=23.5, volume_scaling=50)
        cfg.set_injection(injection_rate=1.0e5)
        cfg.write(tmp_path)

        reloaded = CODTConfig(tmp_path / "params.nml")
        assert reloaded.name == "roundtrip"
        assert reloaded.params.get("tref") == 23.5
        assert reloaded.params.get("volume_scaling") == 50
        np.testing.assert_allclose(
            reloaded.injection.injection_rate, [1.0e5]
        )

    def test_chamber_excludes_parcel_groups(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="chamber")
        cfg.write(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "turbulence_odt" in nml
        assert "turbulence_lem" not in nml
        assert "parcel" not in nml

    def test_parcel_excludes_chamber_groups(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel")
        cfg.write(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "turbulence_lem" in nml
        assert "parcel" in nml
        assert "turbulence_odt" not in nml
        assert "specialeffects" not in nml

    def test_parcel_omits_removed_kolmogorov_scale(self, tmp_path: Path) -> None:
        # CODT 3.0.0 removed kolmogorov_length_scale from &TURBULENCE_LEM;
        # a namelist still declaring it is a fatal read error.
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel")
        cfg.write(tmp_path)

        assert "kolmogorov_length_scale" not in (
            tmp_path / "params.nml"
        ).read_text()

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "kolmogorov_length_scale" not in nml["turbulence_lem"]

    def test_radiation_excluded_when_disabled(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(do_radiation=False)
        cfg.write(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "radiation" not in nml

    def test_namelist_group_placement_matches_codt(self, tmp_path: Path) -> None:
        """Params must land in the group CODT reads them from, else CODT
        rejects the namelist with 'Invalid parameter in &GROUP'.

        CODT v1.0: do_entrainment is read in &PARAMETERS, pressure_limit in
        &PARCEL, and the entrainment params in a standalone &ENTRAINMENT.
        """
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="parcel.nc",
                do_entrainment=True)
        cfg.write(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        # do_entrainment in &PARAMETERS, not &PARCEL
        assert "do_entrainment" in nml["parameters"]
        assert "do_entrainment" not in nml["parcel"]
        # pressure_limit in &PARCEL, not &PARAMETERS
        assert "pressure_limit" in nml["parcel"]
        assert "pressure_limit" not in nml["parameters"]
        # entrainment params in standalone &ENTRAINMENT, not &PARCEL
        assert "entrainment" in nml
        assert "ent_rate" in nml["entrainment"]
        assert "ent_rate" not in nml["parcel"]

    def test_entrainment_excluded_when_disabled(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="parcel.nc",
                do_entrainment=False)
        cfg.write(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "entrainment" not in nml

    def test_radiation_method_default_valid(self) -> None:
        """CODT only accepts '1d' or '3d' for radiation_method."""
        cfg = CODTConfig()
        assert cfg.params.get("radiation_method") in ("1d", "3d")

    def test_radiation_included_when_enabled(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(do_radiation=True)
        cfg.write(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "radiation" in nml


# ======================================================================
# Copy
# ======================================================================


class TestCODTConfigCopy:
    """Tests for copy()."""

    def test_deep_copy_independence(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_name="original", tref=20.0)

        clone = cfg.copy()
        clone.set(simulation_name="clone", tref=25.0)

        assert cfg.name == "original"
        assert cfg.params.get("tref") == 20.0
        assert clone.name == "clone"
        assert clone.params.get("tref") == 25.0

    def test_deep_copy_injection_independence(self) -> None:
        cfg = CODTConfig()
        clone = cfg.copy()
        clone.set_injection(aerosol_name="KCl")

        assert cfg.injection.aerosol_name == "NaCl"
        assert clone.injection.aerosol_name == "KCl"


# ======================================================================
# Validate
# ======================================================================


class TestCODTConfigValidate:
    """Tests for validate()."""

    def test_valid_chamber_passes(self) -> None:
        cfg = CODTConfig()
        cfg.validate()  # should not raise

    def test_valid_parcel_passes(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="parcel_input.nc")
        cfg.validate()

    def test_chamber_tdiff_zero_fails(self) -> None:
        cfg = CODTConfig()
        cfg.set(tdiff=0.0)
        with pytest.raises(ValueError, match="tdiff"):
            cfg.validate()

    def test_parcel_no_file_fails(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel")
        with pytest.raises(ValueError, match="parcel_file"):
            cfg.validate()

    def test_parcel_bad_rh_fails(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc", initial_rh=1.5)
        with pytest.raises(ValueError, match="initial_rh"):
            cfg.validate()

    def test_parcel_entrainment_needs_env(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                do_entrainment=True)
        with pytest.raises(ValueError, match="environmental sounding"):
            cfg.validate()

    def test_parcel_entrainment_with_env_passes(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                do_entrainment=True)
        cfg.parcel.set_env_profile(
            height=[0.0, 2000.0],
            pressure=[100000.0, 80000.0],
            temperature=[300.0, 285.0],
            RH=[0.8, 0.5],
        )
        cfg.validate()

    def _entraining_parcel(self) -> CODTConfig:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                do_entrainment=True)
        cfg.parcel.set_env_profile(
            height=[0.0, 2000.0],
            pressure=[100000.0, 80000.0],
            temperature=[300.0, 285.0],
            RH=[0.8, 0.5],
        )
        return cfg

    def test_parcel_entrainment_chunk_needs_a_gridcell(self) -> None:
        # int(0.001 * 1024) = 1 cell cannot be split into 2 chunks.
        cfg = self._entraining_parcel()
        cfg.set(n=1024, psigma=0.001, n_blob=2)
        with pytest.raises(ValueError, match=r"int\(psigma \* N\)"):
            cfg.validate()

    def test_parcel_entrainment_psigma_times_n_blob_over_one_passes(
        self,
    ) -> None:
        # Rejected before CODT 3.1.0: psigma is now the total fraction
        # replaced, so n_blob no longer multiplies it.
        cfg = self._entraining_parcel()
        cfg.set(psigma=0.5, n_blob=3)
        cfg.validate()

    def test_parcel_entrainment_bad_psigma_fails(self) -> None:
        cfg = self._entraining_parcel()
        cfg.set(psigma=1.0)
        with pytest.raises(ValueError, match="psigma must be in"):
            cfg.validate()

    def test_parcel_entrainment_leg_schedule_checked(self) -> None:
        cfg = self._entraining_parcel()
        cfg.set(n=1024)
        cfg.parcel.set_entrainment_schedule(
            ent_rate=[1.0], n_blob=[5], psigma=[0.001]
        )
        with pytest.raises(ValueError, match=r"int\(psigma \* N\)"):
            cfg.validate()

    def test_parcel_environment_mode_needs_env(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                pressure_mode="environment")
        with pytest.raises(ValueError, match="environmental sounding"):
            cfg.validate()

    def test_parcel_bad_vertical_axis_fails(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                vertical_axis="altitude")
        with pytest.raises(ValueError, match="vertical_axis"):
            cfg.validate()

    def test_parcel_bad_pressure_mode_fails(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                pressure_mode="isobaric")
        with pytest.raises(ValueError, match="pressure_mode"):
            cfg.validate()

    def test_parcel_leg_pointing_away_fails(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc")
        cfg.set_parcel(segment_coord=[1000.0], velocity=[-1.0])
        with pytest.raises(ValueError, match="points away from its target"):
            cfg.validate()

    def test_parcel_legs_validate_against_initial_height(self) -> None:
        # Launching at 1200 m makes a 1000 m target a descent, so the
        # ascending velocity that would pass from the ground now fails.
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                initial_height=1200.0)
        cfg.set_parcel(segment_coord=[1000.0], velocity=[1.0])
        with pytest.raises(ValueError, match="points away from its target"):
            cfg.validate()

    # -- Derived LEM scales (CODT 3.0.0) --

    def test_parcel_domain_too_small_for_smallest_eddy_fails(self) -> None:
        # N = 4 cannot hold the 6-cell structural floor.
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc", n=4, h=1.0)
        with pytest.raises(ValueError, match="cannot contain the smallest eddy"):
            cfg.validate()

    def test_parcel_no_inertial_range_fails(self) -> None:
        # A coarse grid pushes 6*dz above integral_length_scale.
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                n=100, h=1.0, integral_length_scale=0.01)
        with pytest.raises(ValueError, match="No inertial range"):
            cfg.validate()

    def test_parcel_narrow_inertial_range_warns(self) -> None:
        # The defaults themselves sit here (l_small = 6 mm vs L = 10 mm),
        # matching the warning CODT emits on its bundled params.nml.
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc")
        with pytest.warns(UserWarning, match="inertial range is nearly absent"):
            cfg.validate()

    def test_lem_scales_not_checked_without_turbulence(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", parcel_file="p.nc",
                n=4, h=1.0, do_turbulence=False)
        cfg.validate()  # should not raise

    def test_chamber_lmin_below_floor_fails(self) -> None:
        cfg = CODTConfig()
        cfg.set(lmin=3)
        with pytest.raises(ValueError, match="smallest representable eddy"):
            cfg.validate()

    def test_chamber_lmin_not_multiple_of_three_fails(self) -> None:
        cfg = CODTConfig()
        cfg.set(lmin=8)
        with pytest.raises(ValueError, match="multiple of 3"):
            cfg.validate()

    def test_chamber_lmin_valid_passes(self) -> None:
        cfg = CODTConfig()
        cfg.set(lmin=9)
        cfg.validate()

    def test_bad_trajectory_window(self) -> None:
        cfg = CODTConfig()
        cfg.set(
            write_trajectories=True,
            trajectory_start=100.0,
            trajectory_end=50.0,
            tmax=3600.0,
        )
        with pytest.raises(ValueError, match="trajectory_start"):
            cfg.validate()

    def test_trajectory_exceeds_tmax(self) -> None:
        cfg = CODTConfig()
        cfg.set(
            write_trajectories=True,
            trajectory_start=100.0,
            trajectory_end=5000.0,
            tmax=3600.0,
        )
        with pytest.raises(ValueError, match="trajectory_end"):
            cfg.validate()


# ======================================================================
# Sweep
# ======================================================================


class TestCODTConfigSweep:
    """Tests for sweep()."""

    def test_single_param(self) -> None:
        base = CODTConfig()
        base.set(simulation_name="sweep_base")

        configs = CODTConfig.sweep(base, tref=[20.0, 21.0, 22.0])

        assert len(configs) == 3
        for cfg in configs:
            assert "Tref" in cfg.name

        trefs = [cfg.params.get("tref") for cfg in configs]
        assert trefs == [20.0, 21.0, 22.0]

    def test_cartesian_product(self) -> None:
        base = CODTConfig()
        base.set(simulation_name="sweep_base")

        configs = CODTConfig.sweep(
            base,
            tref=[20.0, 21.0, 22.0],
            volume_scaling=[13, 50],
        )

        assert len(configs) == 6

    def test_unique_names(self) -> None:
        base = CODTConfig()
        base.set(simulation_name="sweep")

        configs = CODTConfig.sweep(
            base, tref=[20.0, 21.0], volume_scaling=[13, 50]
        )

        names = [cfg.name for cfg in configs]
        assert len(set(names)) == len(names)  # all unique

    def test_independence_from_base(self) -> None:
        base = CODTConfig()
        base.set(simulation_name="base", tref=20.0)

        configs = CODTConfig.sweep(base, tref=[25.0, 30.0])

        # Modifying sweep results should not affect base
        configs[0].set(tmax=9999.0)
        assert base.params.get("tmax") != 9999.0

    def test_empty_sweep_returns_copy(self) -> None:
        base = CODTConfig()
        base.set(simulation_name="base")

        configs = CODTConfig.sweep(base)

        assert len(configs) == 1
        assert configs[0].name == "base"
        # Should be a copy, not the same object
        configs[0].set(tref=99.0)
        assert base.params.get("tref") != 99.0


# ======================================================================
# from_simulation
# ======================================================================


class TestCODTConfigFromSimulation:
    """Tests for CODTConfig.from_simulation()."""

    @pytest.fixture
    def run_dir(self, tmp_path: Path) -> Path:
        """Create a run directory with default input files."""
        d = tmp_path / "run_inputs"
        d.mkdir()
        InjectionData().write(d / "aerosol_input.nc")
        return d

    def test_recovers_params(self, sim_dir: Path, run_dir: Path) -> None:
        cfg = CODTConfig.from_simulation(sim_dir, run_dir=run_dir)
        assert cfg.tmax == 100.0
        assert cfg.simulation_name == "test_sim"
        assert cfg.volume_scaling == 13

    def test_recovers_bin_edges(self, sim_dir: Path, run_dir: Path) -> None:
        cfg = CODTConfig.from_simulation(sim_dir, run_dir=run_dir)
        assert len(cfg.injection.dsd_bin_edges) == 11
        assert cfg.injection.dsd_bin_edges[0] > 0

    def test_no_run_dir_raises(self, sim_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="run_dir is required"):
            CODTConfig.from_simulation(sim_dir)

    def test_missing_aerosol_raises(self, sim_dir: Path, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="aerosol_input.nc"):
            CODTConfig.from_simulation(sim_dir, run_dir=empty_dir)

    def test_recovers_aerosol(
        self, sim_dir: Path, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        inj = InjectionData()
        inj.set(aerosol_name="KCl", injection_rate=1e5)
        inj.write(run_dir / "aerosol_input.nc")

        cfg = CODTConfig.from_simulation(sim_dir, run_dir=run_dir)
        assert cfg.injection.aerosol_name == "KCl"
        np.testing.assert_allclose(cfg.injection.injection_rate, [1e5])

    def test_recovers_parcel(
        self, sim_dir: Path, run_dir: Path
    ) -> None:
        pi = ParcelInput()
        pi.set(segment_coord=[1000.0, 1500.0], velocity=[1.0, 0.5])
        pi.write(run_dir / "parcel_input.nc", initial_level=0.0)

        cfg = CODTConfig.from_simulation(sim_dir, run_dir=run_dir)
        assert cfg.parcel.n_legs == 2
        np.testing.assert_allclose(cfg.parcel.velocity, [1.0, 0.5])

    def test_parcel_defaults_without_file(
        self, sim_dir: Path, run_dir: Path
    ) -> None:
        cfg = CODTConfig.from_simulation(sim_dir, run_dir=run_dir)
        assert isinstance(cfg.parcel, ParcelInput)
        assert cfg.parcel.n_legs == 1

    def test_can_modify_and_write(
        self, sim_dir: Path, run_dir: Path, tmp_path: Path
    ) -> None:
        cfg = CODTConfig.from_simulation(sim_dir, run_dir=run_dir)
        cfg.tref = 25.0
        cfg.simulation_name = "new_run"

        out = tmp_path / "new_run" / "run"
        cfg.write(out)

        cfg2 = CODTConfig(out / "params.nml")
        assert cfg2.params.get("tref") == 25.0
        assert cfg2.name == "new_run"
        assert (out / "aerosol_input.nc").is_file()
