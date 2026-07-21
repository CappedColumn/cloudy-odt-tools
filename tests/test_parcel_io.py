"""Tests for parcel_io and ParcelInput (CODT_parcel_input_v3)."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from codt_tools.config import CODTConfig, ParcelInput
from codt_tools.parcel_io import read_parcel, validate_legs, write_parcel


def _sounding() -> dict[str, list[float]]:
    """A humid sounding spanning 0–2000 m, keyed for the parcel_io dict."""
    return {
        "env_height": [0.0, 500.0, 1000.0, 2000.0],
        "env_pressure": [1.0e5, 9.5e4, 9.0e4, 8.0e4],
        "env_temperature": [300.0, 297.0, 294.0, 288.0],
        "env_RH": [0.95, 0.92, 0.88, 0.80],
    }


def _sounding_kwargs() -> dict[str, list[float]]:
    """The same sounding, keyed for ParcelInput.set_env_profile."""
    return {
        key.removeprefix("env_"): value for key, value in _sounding().items()
    }


# ======================================================================
# Leg validation
# ======================================================================


class TestValidateLegs:
    """Tests for validate_legs, mirroring CODT's src/parcel.f90."""

    def test_simple_ascent_passes(self) -> None:
        validate_legs([1000.0], [1.0], 0.0, "height")

    def test_up_down_up_passes(self) -> None:
        validate_legs(
            [1000.0, 400.0, 1500.0], [1.0, -0.5, 1.0], 0.0, "height"
        )

    def test_zero_velocity_fails(self) -> None:
        with pytest.raises(ValueError, match="zero velocity"):
            validate_legs([1000.0], [0.0], 0.0, "height")

    def test_target_equals_previous_fails(self) -> None:
        with pytest.raises(ValueError, match="equals the previous level"):
            validate_legs([0.0], [1.0], 0.0, "height")

    def test_consecutive_duplicate_targets_fail(self) -> None:
        with pytest.raises(ValueError, match="leg 2 target equals"):
            validate_legs([1000.0, 1000.0], [1.0, 1.0], 0.0, "height")

    def test_velocity_points_away_fails(self) -> None:
        with pytest.raises(ValueError, match="points away from its target"):
            validate_legs([1000.0], [-1.0], 0.0, "height")

    def test_descent_needs_negative_velocity(self) -> None:
        # Second leg descends, so a positive velocity points away.
        with pytest.raises(ValueError, match="leg 2 velocity"):
            validate_legs([1000.0, 400.0], [1.0, 0.5], 0.0, "height")

    def test_pressure_axis_ascent_is_decreasing_target(self) -> None:
        # On the pressure axis "up" means a lower target pressure, so rising
        # to 90 kPa from 100 kPa takes a positive velocity.
        validate_legs([9.0e4], [1.0], 1.0e5, "pressure")

    def test_pressure_axis_sign_inverted_fails(self) -> None:
        with pytest.raises(ValueError, match="points away from its target"):
            validate_legs([9.0e4], [-1.0], 1.0e5, "pressure")

    def test_pressure_axis_descent(self) -> None:
        validate_legs([1.05e5], [-1.0], 1.0e5, "pressure")

    def test_bad_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="vertical_axis"):
            validate_legs([1000.0], [1.0], 0.0, "altitude")

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            validate_legs([1000.0, 2000.0], [1.0], 0.0, "height")


# ======================================================================
# Low-level read/write
# ======================================================================


class TestParcelIO:
    """Tests for read_parcel / write_parcel."""

    def test_roundtrip_minimal(self, tmp_path: Path) -> None:
        data = {"segment_coord": [1000.0], "velocity": [1.0]}
        path = tmp_path / "parcel_input.nc"
        write_parcel(path, data, initial_level=0.0)
        loaded = read_parcel(path)

        np.testing.assert_allclose(loaded["segment_coord"], [1000.0])
        np.testing.assert_allclose(loaded["velocity"], [1.0])
        assert loaded["env_pressure"] is None
        assert loaded["ent_rate"] is None

    def test_roundtrip_up_down_up(self, tmp_path: Path) -> None:
        data = {
            "segment_coord": [1000.0, 400.0, 1500.0],
            "velocity": [1.0, -0.5, 1.0],
        }
        path = tmp_path / "parcel_input.nc"
        write_parcel(path, data, initial_level=0.0)
        loaded = read_parcel(path)

        np.testing.assert_allclose(
            loaded["segment_coord"], [1000.0, 400.0, 1500.0]
        )
        np.testing.assert_allclose(loaded["velocity"], [1.0, -0.5, 1.0])

    def test_roundtrip_with_sounding(self, tmp_path: Path) -> None:
        data = {"segment_coord": [1000.0], "velocity": [1.0], **_sounding()}
        path = tmp_path / "parcel_input.nc"
        write_parcel(path, data, initial_level=0.0)
        loaded = read_parcel(path)

        for key, expected in _sounding().items():
            np.testing.assert_allclose(loaded[key], expected)

    def test_roundtrip_with_entrainment_schedule(self, tmp_path: Path) -> None:
        data = {
            "segment_coord": [1000.0, 1500.0],
            "velocity": [1.0, 0.5],
            "ent_rate": [0.5, 2.0],
            "n_blob": [1, 2],
            "psigma": [0.1, 0.2],
            **_sounding(),
        }
        path = tmp_path / "parcel_input.nc"
        write_parcel(path, data, initial_level=0.0)
        loaded = read_parcel(path)

        np.testing.assert_allclose(loaded["ent_rate"], [0.5, 2.0])
        np.testing.assert_array_equal(loaded["n_blob"], [1, 2])
        np.testing.assert_allclose(loaded["psigma"], [0.1, 0.2])

    def test_pressure_axis_roundtrip(self, tmp_path: Path) -> None:
        data = {"segment_coord": [9.0e4, 9.5e4], "velocity": [1.0, -0.5]}
        path = tmp_path / "parcel_input.nc"
        write_parcel(
            path, data, initial_level=1.0e5, vertical_axis="pressure"
        )
        loaded = read_parcel(path)
        np.testing.assert_allclose(loaded["segment_coord"], [9.0e4, 9.5e4])

    def test_conventions_is_v3(self, tmp_path: Path) -> None:
        path = tmp_path / "parcel_input.nc"
        write_parcel(
            path, {"segment_coord": [1000.0], "velocity": [1.0]},
            initial_level=0.0,
        )
        with nc.Dataset(path, "r") as ds:
            assert ds.conventions == "CODT_parcel_input_v3"
            assert "time" not in ds.variables

    def test_writes_no_optional_groups_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "parcel_input.nc"
        write_parcel(
            path, {"segment_coord": [1000.0], "velocity": [1.0]},
            initial_level=0.0,
        )
        with nc.Dataset(path, "r") as ds:
            assert "ent_rate" not in ds.variables
            assert "env_height" not in ds.variables

    def test_write_validates_legs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="points away"):
            write_parcel(
                tmp_path / "p.nc",
                {"segment_coord": [1000.0], "velocity": [-1.0]},
                initial_level=0.0,
            )

    def test_write_skips_validation_without_initial_level(
        self, tmp_path: Path
    ) -> None:
        # Direction is only checkable against a launch level; without one the
        # file is written and CODT validates at run time.
        write_parcel(
            tmp_path / "p.nc",
            {"segment_coord": [1000.0], "velocity": [-1.0]},
        )
        assert (tmp_path / "p.nc").is_file()

    def test_ent_rate_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ent_rate must be > 0"):
            write_parcel(
                tmp_path / "p.nc",
                {
                    "segment_coord": [1000.0], "velocity": [1.0],
                    "ent_rate": [0.0], "n_blob": [1], "psigma": [0.1],
                },
                initial_level=0.0,
            )

    def test_psigma_times_n_blob_must_be_under_one(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"psigma \* n_blob"):
            write_parcel(
                tmp_path / "p.nc",
                {
                    "segment_coord": [1000.0], "velocity": [1.0],
                    "ent_rate": [1.0], "n_blob": [5], "psigma": [0.3],
                },
                initial_level=0.0,
            )

    def test_n_blob_out_of_range(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="n_blob must be in"):
            write_parcel(
                tmp_path / "p.nc",
                {
                    "segment_coord": [1000.0], "velocity": [1.0],
                    "ent_rate": [1.0], "n_blob": [11], "psigma": [0.01],
                },
                initial_level=0.0,
            )

    def test_env_height_must_increase(self, tmp_path: Path) -> None:
        bad = _sounding()
        bad["env_height"] = [0.0, 500.0, 400.0, 2000.0]
        with pytest.raises(ValueError, match="env_height must be strictly"):
            write_parcel(
                tmp_path / "p.nc",
                {"segment_coord": [1000.0], "velocity": [1.0], **bad},
                initial_level=0.0,
            )

    def test_env_pressure_must_decrease(self, tmp_path: Path) -> None:
        bad = _sounding()
        bad["env_pressure"] = [1.0e5, 9.5e4, 9.6e4, 8.0e4]
        with pytest.raises(ValueError, match="env_pressure must be strictly"):
            write_parcel(
                tmp_path / "p.nc",
                {"segment_coord": [1000.0], "velocity": [1.0], **bad},
                initial_level=0.0,
            )

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_parcel("/nonexistent/path.nc")

    def test_rejects_v1_file(self, tmp_path: Path) -> None:
        path = tmp_path / "old.nc"
        with nc.Dataset(path, "w") as ds:
            ds.conventions = "CODT_parcel_input_v1"
            ds.createDimension("segment", 1)
            ds.createVariable("time", "f8", ("segment",))[:] = [0.0]
            ds.createVariable("velocity", "f8", ("segment",))[:] = [1.0]

        with pytest.raises(ValueError, match="no longer supported"):
            read_parcel(path)

    def test_rejects_partial_sounding(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.nc"
        with nc.Dataset(path, "w") as ds:
            ds.conventions = "CODT_parcel_input_v3"
            ds.createDimension("segment", 1)
            ds.createDimension("level", 2)
            ds.createVariable("segment_coord", "f8", ("segment",))[:] = [1000.0]
            ds.createVariable("velocity", "f8", ("segment",))[:] = [1.0]
            ds.createVariable("env_height", "f8", ("level",))[:] = [0.0, 100.0]

        with pytest.raises(ValueError, match="partial variable group"):
            read_parcel(path)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "parcel_input.nc"
        write_parcel(
            path, {"segment_coord": [1000.0], "velocity": [1.0]},
            initial_level=0.0,
        )
        assert path.is_file()


# ======================================================================
# ParcelInput class
# ======================================================================


class TestParcelInput:
    """Tests for the ParcelInput class."""

    def test_defaults(self) -> None:
        pi = ParcelInput()
        assert pi.n_legs == 1
        np.testing.assert_allclose(pi.segment_coord, [1000.0])
        np.testing.assert_allclose(pi.velocity, [1.0])
        assert not pi.has_env_profile
        assert not pi.has_entrainment_schedule

    def test_set(self) -> None:
        pi = ParcelInput()
        pi.set(segment_coord=[1000.0, 400.0], velocity=[1.0, -0.5])
        assert pi.n_legs == 2
        np.testing.assert_allclose(pi.velocity, [1.0, -0.5])

    def test_set_bad_attr_raises(self) -> None:
        pi = ParcelInput()
        with pytest.raises(AttributeError):
            pi.set(nonexistent=42)

    def test_time_is_no_longer_an_attribute(self) -> None:
        # v3 dropped the time axis; catch templates carried over from v1/v2.
        pi = ParcelInput()
        with pytest.raises(AttributeError):
            pi.set(time=[0.0, 300.0])

    def test_set_env_profile(self) -> None:
        pi = ParcelInput()
        pi.set_env_profile(**_sounding_kwargs())
        assert pi.has_env_profile
        np.testing.assert_allclose(pi.env_height, [0.0, 500.0, 1000.0, 2000.0])
        np.testing.assert_allclose(pi.env_pressure, [1.0e5, 9.5e4, 9.0e4, 8.0e4])

    def test_clear_env_profile(self) -> None:
        pi = ParcelInput()
        pi.set_env_profile(**_sounding_kwargs())
        assert pi.has_env_profile
        pi.clear_env_profile()
        assert not pi.has_env_profile
        assert pi.env_height is None

    def test_set_entrainment_schedule(self) -> None:
        pi = ParcelInput()
        pi.set(segment_coord=[1000.0, 1500.0], velocity=[1.0, 0.5])
        pi.set_entrainment_schedule(
            ent_rate=[0.5, 2.0], n_blob=[1, 2], psigma=[0.1, 0.2]
        )
        assert pi.has_entrainment_schedule
        np.testing.assert_allclose(pi.ent_rate, [0.5, 2.0])

    def test_clear_entrainment_schedule(self) -> None:
        pi = ParcelInput()
        pi.set_entrainment_schedule(
            ent_rate=[0.5], n_blob=[1], psigma=[0.1]
        )
        pi.clear_entrainment_schedule()
        assert not pi.has_entrainment_schedule
        assert pi.n_blob is None

    def test_roundtrip_file(self, tmp_path: Path) -> None:
        pi = ParcelInput()
        pi.set(segment_coord=[1000.0, 400.0, 1500.0], velocity=[1.0, -0.5, 1.0])
        pi.write(tmp_path / "parcel_input.nc", initial_level=0.0)

        loaded = ParcelInput(tmp_path / "parcel_input.nc")
        assert loaded.n_legs == 3
        np.testing.assert_allclose(loaded.velocity, [1.0, -0.5, 1.0])

    def test_roundtrip_file_with_sounding(self, tmp_path: Path) -> None:
        pi = ParcelInput()
        pi.set(segment_coord=[1000.0], velocity=[1.0])
        pi.set_env_profile(**_sounding_kwargs())
        pi.write(tmp_path / "parcel_input.nc", initial_level=0.0)

        loaded = ParcelInput(tmp_path / "parcel_input.nc")
        assert loaded.has_env_profile
        np.testing.assert_allclose(loaded.env_pressure, [1.0e5, 9.5e4, 9.0e4, 8.0e4])

    def test_repr(self) -> None:
        pi = ParcelInput()
        assert "v3" in repr(pi)
        assert "n_legs=1" in repr(pi)


# ======================================================================
# CODTConfig parcel integration
# ======================================================================


class TestCODTConfigParcel:
    """Tests for parcel support in CODTConfig."""

    def test_default_has_parcel(self) -> None:
        cfg = CODTConfig()
        assert isinstance(cfg.parcel, ParcelInput)

    def test_set_parcel(self) -> None:
        cfg = CODTConfig()
        cfg.set_parcel(segment_coord=[1000.0, 400.0], velocity=[1.0, -0.5])
        assert cfg.parcel.n_legs == 2

    def test_write_parcel_mode(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", simulation_name="parcel_test")
        cfg.set_parcel(segment_coord=[1000.0], velocity=[1.0])
        cfg.write(tmp_path / "run")

        assert (tmp_path / "run" / "params.nml").is_file()
        assert (tmp_path / "run" / "aerosol_input.nc").is_file()
        assert (tmp_path / "run" / "parcel_input.nc").is_file()

        from codt_tools.config import Namelist
        nml = Namelist(tmp_path / "run" / "params.nml")
        assert nml.get("parcel_file") == "parcel_input.nc"
        assert nml.get("vertical_axis") == "height"
        assert nml.get("pressure_mode") == "hydrostatic"

    def test_write_chamber_mode_no_parcel_file(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="chamber", simulation_name="chamber_test")
        cfg.write(tmp_path / "run")

        assert not (tmp_path / "run" / "parcel_input.nc").is_file()

    def test_roundtrip_parcel_mode(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", simulation_name="rt_test")
        cfg.set_parcel(
            segment_coord=[1000.0, 400.0, 1500.0], velocity=[1.0, -0.5, 1.0]
        )
        cfg.write(tmp_path / "run")

        reloaded = CODTConfig(tmp_path / "run" / "params.nml")
        assert reloaded.parcel.n_legs == 3
        np.testing.assert_allclose(reloaded.parcel.velocity, [1.0, -0.5, 1.0])

    def test_copy_independence(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel")
        cfg.set_parcel(segment_coord=[1000.0, 1500.0], velocity=[1.0, 0.5])

        clone = cfg.copy()
        clone.set_parcel(velocity=[2.0, 1.0])

        np.testing.assert_allclose(cfg.parcel.velocity, [1.0, 0.5])
        np.testing.assert_allclose(clone.parcel.velocity, [2.0, 1.0])
