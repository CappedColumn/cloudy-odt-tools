"""Tests for codt_tools.aerosol_io (read/write aerosol_input.nc)."""

from __future__ import annotations

import numpy as np
import pytest

from codt_tools.aerosol_io import read_aerosol, write_aerosol


class TestWriteRead:
    """Roundtrip write → read tests."""

    def test_roundtrip_values(self, tmp_path, aerosol_data):
        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, aerosol_data)
        result = read_aerosol(path)

        assert result["aerosol_name"] == "NaCl"
        np.testing.assert_array_equal(result["n_ions"], [2])
        np.testing.assert_allclose(result["molar_mass"], [58.4428e-3])
        np.testing.assert_allclose(result["solute_density"], [2163.0])
        np.testing.assert_allclose(result["edge_radii"], [60.0, 70.0, 4930.0])
        np.testing.assert_array_equal(result["category"], [1, 2])
        np.testing.assert_allclose(result["cumulative_frequency"], [[1.0, 1.0]])
        np.testing.assert_allclose(result["injection_time"], [0.0])
        np.testing.assert_allclose(result["injection_rate"], [5.5e5])

    def test_roundtrip_shapes(self, tmp_path, aerosol_data):
        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, aerosol_data)
        result = read_aerosol(path)

        assert result["n_ions"].shape == (1,)
        assert result["edge_radii"].shape == (3,)
        assert result["category"].shape == (2,)
        assert result["cumulative_frequency"].shape == (1, 2)
        assert result["injection_time"].shape == (1,)

    def test_creates_parent_directories(self, tmp_path, aerosol_data):
        path = tmp_path / "deep" / "nested" / "aerosol_input.nc"
        write_aerosol(path, aerosol_data)
        assert path.is_file()

    def test_multiple_injection_times(self, tmp_path, aerosol_data):
        aerosol_data["injection_time"] = np.array([0.0, 30.0, 60.0])
        aerosol_data["injection_rate"] = np.array([5.5e5, 3.0e5, 1.0e5])
        aerosol_data["cumulative_frequency"] = np.array([
            [1.0, 1.0],
            [0.8, 1.0],
            [0.5, 1.0],
        ])
        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, aerosol_data)
        result = read_aerosol(path)

        assert result["injection_time"].shape == (3,)
        assert result["cumulative_frequency"].shape == (3, 2)
        np.testing.assert_allclose(result["injection_rate"], [5.5e5, 3.0e5, 1.0e5])


class TestReadErrors:
    """Error handling in read_aerosol."""

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_aerosol(tmp_path / "nonexistent.nc")

    def test_bad_conventions(self, tmp_path):
        import netCDF4 as nc

        path = tmp_path / "bad.nc"
        with nc.Dataset(path, "w") as ds:
            ds.conventions = "WRONG_SCHEMA"
            ds.createDimension("x", 1)

        with pytest.raises(ValueError, match="CODT_aerosol_input_v1"):
            read_aerosol(path)


class TestWriteErrors:
    """Error handling in write_aerosol."""

    def test_missing_key(self, tmp_path):
        with pytest.raises(KeyError):
            write_aerosol(tmp_path / "bad.nc", {"aerosol_name": "test"})
