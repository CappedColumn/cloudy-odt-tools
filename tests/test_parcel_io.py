"""Tests for parcel_io and ParcelInput."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codt_tools.config import CODTConfig, ParcelInput
from codt_tools.parcel_io import read_parcel, write_parcel


# ======================================================================
# Low-level read/write
# ======================================================================


class TestParcelIO:
    """Tests for read_parcel / write_parcel."""

    def test_roundtrip_v1(self, tmp_path: Path) -> None:
        data = {
            "time": np.array([0.0, 300.0, 600.0]),
            "velocity": np.array([1.0, 0.5, 0.0]),
            "env_pressure": None,
            "env_temperature": None,
            "env_RH": None,
        }
        path = tmp_path / "parcel_input.nc"
        write_parcel(path, data)
        loaded = read_parcel(path)

        np.testing.assert_allclose(loaded["time"], data["time"])
        np.testing.assert_allclose(loaded["velocity"], data["velocity"])
        assert loaded["env_pressure"] is None

    def test_roundtrip_v2(self, tmp_path: Path) -> None:
        data = {
            "time": np.array([0.0, 600.0]),
            "velocity": np.array([1.0, 0.5]),
            "env_pressure": np.array([100000.0, 90000.0, 80000.0]),
            "env_temperature": np.array([300.0, 295.0, 290.0]),
            "env_RH": np.array([0.8, 0.6, 0.4]),
        }
        path = tmp_path / "parcel_input.nc"
        write_parcel(path, data)
        loaded = read_parcel(path)

        np.testing.assert_allclose(loaded["time"], data["time"])
        np.testing.assert_allclose(loaded["velocity"], data["velocity"])
        np.testing.assert_allclose(loaded["env_pressure"], data["env_pressure"])
        np.testing.assert_allclose(loaded["env_temperature"], data["env_temperature"])
        np.testing.assert_allclose(loaded["env_RH"], data["env_RH"])

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_parcel("/nonexistent/path.nc")

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        data = {
            "time": np.array([0.0]),
            "velocity": np.array([1.0]),
            "env_pressure": None,
            "env_temperature": None,
            "env_RH": None,
        }
        path = tmp_path / "deep" / "nested" / "parcel_input.nc"
        write_parcel(path, data)
        assert path.is_file()


# ======================================================================
# ParcelInput class
# ======================================================================


class TestParcelInput:
    """Tests for the ParcelInput class."""

    def test_defaults(self) -> None:
        pi = ParcelInput()
        assert pi.n_segments == 1
        np.testing.assert_allclose(pi.time, [0.0])
        np.testing.assert_allclose(pi.velocity, [1.0])
        assert not pi.has_env_profile

    def test_set(self) -> None:
        pi = ParcelInput()
        pi.set(time=[0.0, 300.0], velocity=[1.0, 0.5])
        assert pi.n_segments == 2
        np.testing.assert_allclose(pi.velocity, [1.0, 0.5])

    def test_set_bad_attr_raises(self) -> None:
        pi = ParcelInput()
        with pytest.raises(AttributeError):
            pi.set(nonexistent=42)

    def test_set_env_profile(self) -> None:
        pi = ParcelInput()
        pi.set_env_profile(
            pressure=[100000.0, 90000.0],
            temperature=[300.0, 295.0],
            RH=[0.8, 0.6],
        )
        assert pi.has_env_profile
        np.testing.assert_allclose(pi.env_pressure, [100000.0, 90000.0])

    def test_clear_env_profile(self) -> None:
        pi = ParcelInput()
        pi.set_env_profile(
            pressure=[100000.0], temperature=[300.0], RH=[0.8]
        )
        assert pi.has_env_profile
        pi.clear_env_profile()
        assert not pi.has_env_profile

    def test_roundtrip_file(self, tmp_path: Path) -> None:
        pi = ParcelInput()
        pi.set(time=[0.0, 300.0, 600.0], velocity=[1.0, 0.5, 0.0])
        pi.write(tmp_path / "parcel_input.nc")

        loaded = ParcelInput(tmp_path / "parcel_input.nc")
        assert loaded.n_segments == 3
        np.testing.assert_allclose(loaded.velocity, [1.0, 0.5, 0.0])

    def test_roundtrip_v2_file(self, tmp_path: Path) -> None:
        pi = ParcelInput()
        pi.set(time=[0.0, 600.0], velocity=[1.0, 0.5])
        pi.set_env_profile(
            pressure=[100000.0, 80000.0],
            temperature=[300.0, 285.0],
            RH=[0.8, 0.5],
        )
        pi.write(tmp_path / "parcel_input.nc")

        loaded = ParcelInput(tmp_path / "parcel_input.nc")
        assert loaded.has_env_profile
        np.testing.assert_allclose(loaded.env_pressure, [100000.0, 80000.0])

    def test_repr(self) -> None:
        pi = ParcelInput()
        assert "v1" in repr(pi)
        pi.set_env_profile(
            pressure=[100000.0], temperature=[300.0], RH=[0.8]
        )
        assert "v2" in repr(pi)


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
        cfg.set_parcel(time=[0.0, 300.0], velocity=[1.0, 0.5])
        assert cfg.parcel.n_segments == 2

    def test_write_parcel_mode(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", simulation_name="parcel_test")
        cfg.set_parcel(time=[0.0, 600.0], velocity=[1.0, 0.5])
        cfg.write(tmp_path / "run")

        assert (tmp_path / "run" / "params.nml").is_file()
        assert (tmp_path / "run" / "aerosol_input.nc").is_file()
        assert (tmp_path / "run" / "parcel_input.nc").is_file()

        # Verify parcel_file is set in namelist
        from codt_tools.config import Namelist
        nml = Namelist(tmp_path / "run" / "params.nml")
        assert nml.get("parcel_file") == "parcel_input.nc"

    def test_write_chamber_mode_no_parcel_file(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="chamber", simulation_name="chamber_test")
        cfg.write(tmp_path / "run")

        assert not (tmp_path / "run" / "parcel_input.nc").is_file()

    def test_roundtrip_parcel_mode(self, tmp_path: Path) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel", simulation_name="rt_test")
        cfg.set_parcel(time=[0.0, 300.0, 600.0], velocity=[1.0, 0.5, 0.0])
        cfg.write(tmp_path / "run")

        reloaded = CODTConfig(tmp_path / "run" / "params.nml")
        assert reloaded.parcel.n_segments == 3
        np.testing.assert_allclose(reloaded.parcel.velocity, [1.0, 0.5, 0.0])

    def test_copy_independence(self) -> None:
        cfg = CODTConfig()
        cfg.set(simulation_mode="parcel")
        cfg.set_parcel(time=[0.0, 300.0], velocity=[1.0, 0.5])

        clone = cfg.copy()
        clone.set_parcel(velocity=[2.0, 1.0])

        np.testing.assert_allclose(cfg.parcel.velocity, [1.0, 0.5])
        np.testing.assert_allclose(clone.parcel.velocity, [2.0, 1.0])
