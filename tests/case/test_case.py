"""Tests for Case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codt_tools.case import Aerosol, Case, Namelist, Parcel


# ======================================================================
# Init
# ======================================================================


class TestCaseInit:
    """Tests for Case construction."""

    def test_defaults(self) -> None:
        cfg = Case()
        assert cfg.name == "default_sim"
        assert isinstance(cfg.params, Namelist)
        assert isinstance(cfg.aerosol, Aerosol)

    def test_name_property(self) -> None:
        cfg = Case()
        cfg.set(simulation_name="my_sim")
        assert cfg.name == "my_sim"

    def test_from_namelist_path(self, tmp_path: Path) -> None:
        """Load from an existing directory with all three files."""
        # Write a config to disk, then reload
        cfg = Case()
        cfg.set(simulation_name="reload_test", tref=25.0)
        cfg.write_inputs(tmp_path)

        reloaded = Case.from_input_dir(tmp_path)
        assert reloaded.name == "reload_test"
        assert reloaded.params.get("tref") == 25.0
        assert reloaded.aerosol.aerosol_name == "NaCl"
        assert len(reloaded.aerosol.dsd_bin_edges) == 201

    def test_from_namelist_missing_data_files(self, tmp_path: Path) -> None:
        """Loading from namelist alone falls back to defaults for data."""
        nml = Namelist()
        nml.set(simulation_name="nml_only",
                output_directory=str(tmp_path / "output"))
        nml.write(tmp_path / "params.nml")

        cfg = Case.from_input_dir(tmp_path)
        assert cfg.name == "nml_only"
        # Should fall back to defaults (not crash)
        assert isinstance(cfg.aerosol, Aerosol)


# ======================================================================
# Setters
# ======================================================================


class TestCaseSetters:
    """Tests for Case.set and the component setters."""

    def test_set_namelist_params(self) -> None:
        cfg = Case()
        cfg.set(tref=22.0, tmax=7200.0, simulation_name="setter_test")
        assert cfg.params.get("tref") == 22.0
        assert cfg.params.get("tmax") == 7200.0
        assert cfg.name == "setter_test"

    def test_set_injection(self) -> None:
        cfg = Case()
        cfg.aerosol.set(aerosol_name="KCl", injection_rate=1.0e5)
        assert cfg.aerosol.aerosol_name == "KCl"
        np.testing.assert_allclose(cfg.aerosol.injection_rate, [1.0e5])

    def test_set_bins(self) -> None:
        cfg = Case()
        new_edges = np.linspace(0.1, 50.0, 101)
        cfg.aerosol.set(dsd_bin_edges=new_edges)
        assert len(cfg.aerosol.dsd_bin_edges) == 101
        np.testing.assert_allclose(cfg.aerosol.dsd_bin_edges, new_edges)

    def test_set_bad_param_raises(self) -> None:
        cfg = Case()
        with pytest.raises(KeyError):
            cfg.set(nonexistent_param=42)

    def test_set_injection_bad_attr_raises(self) -> None:
        cfg = Case()
        with pytest.raises(AttributeError):
            cfg.aerosol.set(nonexistent_attr=42)


# ======================================================================
# Dot-access
# ======================================================================


class TestCaseAttributeGuard:
    """A Case has three attributes; anything else is an error, not a no-op."""

    def test_own_attrs_unaffected(self) -> None:
        cfg = Case()
        assert isinstance(cfg.params, Namelist)
        assert isinstance(cfg.aerosol, Aerosol)
        assert isinstance(cfg.parcel, Parcel)

    def test_reading_a_namelist_param_as_an_attribute_raises(self) -> None:
        cfg = Case()
        with pytest.raises(AttributeError):
            _ = cfg.tref

    def test_setting_a_namelist_param_as_an_attribute_raises(self) -> None:
        """``cfg.tref = 22.0`` used to work; ``cfg.tefr = 22.0`` used to
        silently create a dead attribute. Both now raise, and the message
        names the call that works."""
        cfg = Case()
        with pytest.raises(AttributeError, match=r"case\.set\(tref="):
            cfg.tref = 22.0

    def test_setting_an_unknown_attribute_raises(self) -> None:
        cfg = Case()
        with pytest.raises(AttributeError, match="no attribute 'tefr'"):
            cfg.tefr = 22.0

    def test_components_can_be_replaced(self) -> None:
        cfg = Case()
        cfg.aerosol = Aerosol()
        cfg.parcel = Parcel()
        cfg.params = Namelist()


# ======================================================================
# Write
# ======================================================================


class TestCaseWrite:
    """Tests for write()."""

    def test_creates_all_files(self, tmp_path: Path) -> None:
        cfg = Case()
        cfg.set(simulation_name="write_test")
        cfg.write_inputs(tmp_path / "run")

        assert (tmp_path / "run" / "params.nml").is_file()
        assert (tmp_path / "run" / "aerosol_input.nc").is_file()

    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "dir"
        cfg = Case()
        cfg.write_inputs(target)
        assert target.is_dir()
        assert (target / "params.nml").is_file()

    def test_sets_relative_data_paths(self, tmp_path: Path) -> None:
        cfg = Case()
        cfg.write_inputs(tmp_path)

        reloaded = Namelist(tmp_path / "params.nml")
        assert reloaded.get("aerosol_file") == "aerosol_input.nc"

    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = Case()
        cfg.set(simulation_name="roundtrip", tref=23.5, volume_scaling=50)
        cfg.aerosol.set(injection_rate=1.0e5)
        cfg.write_inputs(tmp_path)

        reloaded = Case.from_input_dir(tmp_path)
        assert reloaded.name == "roundtrip"
        assert reloaded.params.get("tref") == 23.5
        assert reloaded.params.get("volume_scaling") == 50
        np.testing.assert_allclose(
            reloaded.aerosol.injection_rate, [1.0e5]
        )

    def test_chamber_excludes_parcel_groups(self, tmp_path: Path) -> None:
        cfg = Case()
        cfg.set(simulation_mode="chamber")
        cfg.write_inputs(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "turbulence_odt" in nml
        assert "turbulence_lem" not in nml
        assert "parcel" not in nml

    def test_parcel_excludes_chamber_groups(self, tmp_path: Path) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel")
        cfg.write_inputs(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "turbulence_lem" in nml
        assert "parcel" in nml
        assert "turbulence_odt" not in nml
        assert "specialeffects" not in nml

    def test_parcel_omits_removed_kolmogorov_scale(self, tmp_path: Path) -> None:
        # CODT 3.0.0 removed kolmogorov_length_scale from &TURBULENCE_LEM;
        # a namelist still declaring it is a fatal read error.
        cfg = Case()
        cfg.set(simulation_mode="parcel")
        cfg.write_inputs(tmp_path)

        assert "kolmogorov_length_scale" not in (
            tmp_path / "params.nml"
        ).read_text()

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "kolmogorov_length_scale" not in nml["turbulence_lem"]

    def test_radiation_excluded_when_disabled(self, tmp_path: Path) -> None:
        cfg = Case()
        cfg.set(do_radiation=False)
        cfg.write_inputs(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "radiation" not in nml

    def test_namelist_group_placement_matches_codt(self, tmp_path: Path) -> None:
        """Params must land in the group CODT reads them from, else CODT
        rejects the namelist with 'Invalid parameter in &GROUP'.

        CODT v1.0: do_entrainment is read in &PARAMETERS, pressure_limit in
        &PARCEL, and the entrainment params in a standalone &ENTRAINMENT.
        """
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                do_entrainment=True)
        cfg.write_inputs(tmp_path)

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
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                do_entrainment=False)
        cfg.write_inputs(tmp_path)

        import f90nml
        nml = f90nml.read(tmp_path / "params.nml")
        assert "entrainment" not in nml

    def test_radiation_method_default_valid(self) -> None:
        """CODT only accepts '1d' or '3d' for radiation_method."""
        cfg = Case()
        assert cfg.params.get("radiation_method") in ("1d", "3d")

    def test_radiation_included_when_enabled(self, tmp_path: Path) -> None:
        mie = tmp_path / "src" / "mie_absorption.txt"
        mie.parent.mkdir()
        mie.write_text("1.0 2.0\n")

        cfg = Case()
        cfg.set(do_radiation=True, mie_data_file=str(mie))
        cfg.write_inputs(tmp_path / "run")

        import f90nml
        nml = f90nml.read(tmp_path / "run" / "params.nml")
        assert "radiation" in nml

    def test_radiation_stages_the_mie_table(self, tmp_path: Path) -> None:
        """CODT resolves mie_data_file against the namelist's directory, so
        the table has to be copied there — a bare source path never resolves."""
        mie = tmp_path / "elsewhere" / "mie_absorption.txt"
        mie.parent.mkdir()
        mie.write_text("1.0 2.0\n")

        cfg = Case()
        cfg.set(do_radiation=True, mie_data_file=str(mie))
        staged = cfg.write_inputs(tmp_path / "run")

        assert (tmp_path / "run" / "mie_absorption.txt").is_file()
        assert staged.get("mie_data_file") == "mie_absorption.txt"
        assert cfg.params.get("mie_data_file") == str(mie)

    def test_radiation_without_a_mie_table_raises(self, tmp_path: Path) -> None:
        cfg = Case()
        cfg.set(do_radiation=True, mie_data_file=str(tmp_path / "absent.txt"))
        with pytest.raises(FileNotFoundError, match="mie_data_file"):
            cfg.write_inputs(tmp_path / "run")


# ======================================================================
# Copy
# ======================================================================


class TestCaseCopy:
    """Tests for copy()."""

    def test_deep_copy_independence(self) -> None:
        cfg = Case()
        cfg.set(simulation_name="original", tref=20.0)

        clone = cfg.copy()
        clone.set(simulation_name="clone", tref=25.0)

        assert cfg.name == "original"
        assert cfg.params.get("tref") == 20.0
        assert clone.name == "clone"
        assert clone.params.get("tref") == 25.0

    def test_deep_copy_injection_independence(self) -> None:
        cfg = Case()
        clone = cfg.copy()
        clone.aerosol.set(aerosol_name="KCl")

        assert cfg.aerosol.aerosol_name == "NaCl"
        assert clone.aerosol.aerosol_name == "KCl"


# ======================================================================
# Sweep
# ======================================================================


class TestCaseSweep:
    """Tests for sweep(). Design-building itself lives in test_mutate.py."""

    def test_dict_shorthand(self) -> None:
        base = Case()
        base.set(simulation_name="sweep_base")

        cases = base.sweep({"params.tref": [20.0, 21.0, 22.0]})

        assert [c.params.get("tref") for c in cases] == [20.0, 21.0, 22.0]
        assert [c.name for c in cases] == [
            "sweep_base_000", "sweep_base_001", "sweep_base_002"
        ]

    def test_dict_shorthand_is_cartesian(self) -> None:
        base = Case()
        cases = base.sweep({
            "params.tref": [20.0, 21.0, 22.0],
            "params.volume_scaling": [13, 50],
        })
        assert len(cases) == 6

    def test_design_list(self) -> None:
        base = Case()
        base.set(simulation_name="sweep")

        cases = base.sweep([
            {"params.tref": 20.0, "params.n_blob": 1},
            {"params.tref": 22.0, "params.n_blob": 5},
        ])

        assert len(cases) == 2
        assert cases[1].params.get("tref") == 22.0
        assert cases[1].params.get("n_blob") == 5

    def test_names_are_unique_and_path_safe(self) -> None:
        """Sweep names become run directory names."""
        base = Case()
        base.set(simulation_name="sweep")

        cases = base.sweep({"aerosol.injection_rate": [[5e4], [6e4]]})

        names = [c.name for c in cases]
        assert names == ["sweep_000", "sweep_001"]
        assert len(set(names)) == len(names)
        for name in names:
            assert not (set(name) & set(" /[](),"))

    def test_name_callable_overrides(self) -> None:
        base = Case()
        cases = base.sweep(
            {"params.tref": [20.0, 21.0]},
            name=lambda i, point: f"EXP002_Tref{point['params.tref']:.0f}",
        )
        assert [c.name for c in cases] == ["EXP002_Tref20", "EXP002_Tref21"]

    def test_name_callable_must_not_collide(self) -> None:
        base = Case()
        with pytest.raises(ValueError, match="unique"):
            base.sweep({"params.tref": [20.0, 21.0]}, name=lambda i, p: "same")

    def test_independence_from_base_and_siblings(self) -> None:
        base = Case()
        base.set(simulation_name="base", tref=20.0)

        cases = base.sweep({"params.tref": [25.0, 30.0]})
        cases[0].set(tmax=9999.0)

        assert base.params.get("tmax") != 9999.0
        assert cases[1].params.get("tmax") != 9999.0
        assert base.params.get("tref") == 20.0

    def test_component_objects_are_not_shared(self) -> None:
        base = Case()
        cases = base.sweep({"params.tref": [20.0, 21.0]})

        cases[0].aerosol.set(aerosol_name="KCl")

        assert cases[1].aerosol.aerosol_name != "KCl"
        assert base.aerosol.aerosol_name != "KCl"

    def test_empty_design_gives_nothing(self) -> None:
        base = Case()
        assert base.sweep([]) == []

    def test_no_axes_gives_one_copy(self) -> None:
        """The empty product is one point that changes nothing."""
        base = Case()
        base.set(simulation_name="base")

        cases = base.sweep({})

        assert len(cases) == 1
        assert cases[0].name == "base_000"
        cases[0].set(tref=99.0)
        assert base.params.get("tref") != 99.0


# ======================================================================
# from_simulation
# ======================================================================


class TestCaseFromSimulation:
    """Tests for Case.from_simulation()."""

    @pytest.fixture
    def run_dir(self, tmp_path: Path) -> Path:
        """Create a run directory with default input files."""
        d = tmp_path / "run_inputs"
        d.mkdir()
        Aerosol().write(d / "aerosol_input.nc")
        return d

    def test_recovers_params(self, sim_dir: Path, run_dir: Path) -> None:
        cfg = Case.from_simulation(sim_dir, run_dir=run_dir)
        assert cfg.params.get("tmax") == 100.0
        assert cfg.name == "test_sim"
        assert cfg.params.get("volume_scaling") == 13

    def test_recovers_bin_edges(self, sim_dir: Path, run_dir: Path) -> None:
        cfg = Case.from_simulation(sim_dir, run_dir=run_dir)
        assert len(cfg.aerosol.dsd_bin_edges) == 11
        assert cfg.aerosol.dsd_bin_edges[0] > 0

    def test_no_run_dir_raises(self, sim_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="run_dir is required"):
            Case.from_simulation(sim_dir)

    def test_missing_aerosol_raises(self, sim_dir: Path, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="aerosol_input.nc"):
            Case.from_simulation(sim_dir, run_dir=empty_dir)

    def test_recovers_aerosol(
        self, sim_dir: Path, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        inj = Aerosol()
        inj.set(aerosol_name="KCl", injection_rate=1e5)
        inj.write(run_dir / "aerosol_input.nc")

        cfg = Case.from_simulation(sim_dir, run_dir=run_dir)
        assert cfg.aerosol.aerosol_name == "KCl"
        np.testing.assert_allclose(cfg.aerosol.injection_rate, [1e5])

    def test_recovers_parcel(
        self, sim_dir: Path, run_dir: Path
    ) -> None:
        pi = Parcel()
        pi.set(segment_coord=[1000.0, 1500.0], velocity=[1.0, 0.5])
        pi.write(run_dir / "parcel_input.nc", initial_level=0.0)

        cfg = Case.from_simulation(sim_dir, run_dir=run_dir)
        assert cfg.parcel.n_legs == 2
        np.testing.assert_allclose(cfg.parcel.velocity, [1.0, 0.5])

    def test_parcel_defaults_without_file(
        self, sim_dir: Path, run_dir: Path
    ) -> None:
        cfg = Case.from_simulation(sim_dir, run_dir=run_dir)
        assert isinstance(cfg.parcel, Parcel)
        assert cfg.parcel.n_legs == 1

    def test_can_modify_and_write(
        self, sim_dir: Path, run_dir: Path, tmp_path: Path
    ) -> None:
        cfg = Case.from_simulation(sim_dir, run_dir=run_dir)
        cfg.set(tref=25.0, simulation_name="new_run")

        out = tmp_path / "new_run" / "run"
        cfg.write_inputs(out)

        cfg2 = Case.from_input_dir(out)
        assert cfg2.params.get("tref") == 25.0
        assert cfg2.name == "new_run"
        assert (out / "aerosol_input.nc").is_file()
