"""Tests for codt_tools.aerosol_io (read/write aerosol_input.nc)."""

from __future__ import annotations

import numpy as np
import pytest

import netCDF4 as nc

from codt_tools.case.aerosol import (
    SEED_KEYS,
    make_seed_group,
    read_aerosol,
    write_aerosol,
)


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


# ======================================================================
# bin_type
# ======================================================================


class TestBinType:
    """Tests for the optional per-bin composition row."""

    def test_defaults_to_ones_when_absent(self, tmp_path, aerosol_data):
        # A file written without bin_type reads back as all type 1, which is
        # the behaviour CODT had before bin_type existed.
        assert "bin_type" not in aerosol_data
        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, aerosol_data)
        result = read_aerosol(path)
        np.testing.assert_array_equal(result["bin_type"], [1, 1])

    def test_written_explicitly(self, tmp_path, aerosol_data):
        # Even an all-ones bin_type is written out, so the variable is always
        # present in files this package produces.
        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, aerosol_data)
        with nc.Dataset(path, "r") as ds:
            assert "bin_type" in ds.variables
            assert ds["bin_type"].dimensions == ("bin",)

    def test_multi_row_composition_table(self, tmp_path, aerosol_data):
        # aerosol_type > 1 used to be silently ignored by CODT (it read only
        # row 1); all rows are read now, so round-trip them faithfully.
        aerosol_data["n_ions"] = np.array([2, 2], dtype=np.int32)
        aerosol_data["molar_mass"] = np.array([58.4428e-3, 74.55e-3])
        aerosol_data["solute_density"] = np.array([2163.0, 1984.0])
        aerosol_data["bin_type"] = np.array([1, 2], dtype=np.int32)

        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, aerosol_data)
        result = read_aerosol(path)
        np.testing.assert_array_equal(result["bin_type"], [1, 2])
        np.testing.assert_allclose(result["solute_density"], [2163.0, 1984.0])

    def test_wrong_length_rejected(self, tmp_path, aerosol_data):
        aerosol_data["bin_type"] = np.array([1, 1, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="one value per bin"):
            write_aerosol(tmp_path / "bad.nc", aerosol_data)

    def test_out_of_range_rejected(self, tmp_path, aerosol_data):
        aerosol_data["bin_type"] = np.array([1, 5], dtype=np.int32)
        with pytest.raises(ValueError, match="must lie in 1..1"):
            write_aerosol(tmp_path / "bad.nc", aerosol_data)


# ======================================================================
# Seed group
# ======================================================================


def _two_type_background(aerosol_data):
    """Background with a spare composition row for the seed to reference."""
    aerosol_data["n_ions"] = np.array([2, 2], dtype=np.int32)
    aerosol_data["molar_mass"] = np.array([58.4428e-3, 58.4428e-3])
    aerosol_data["solute_density"] = np.array([2163.0, 2163.0])
    aerosol_data["bin_type"] = np.array([1, 1], dtype=np.int32)
    return aerosol_data


def _seed_kwargs(**overrides):
    """A two-event, two-bin seed group referencing composition row 2."""
    kwargs = {
        "seed_edge_radii": [500.0, 1000.0, 2000.0],
        "seed_category": [7, 8],
        "seed_bin_type": [2, 2],
        "seed_frequency": [[0.5, 1.0], [0.25, 1.0]],
        "seed_coord": [500.0, 900.0],
        "seed_concentration": [10.0, 5.0],
        "n_types": 2,
        "bin_type": [1, 1],
        "category": [1, 2],
    }
    kwargs.update(overrides)
    return kwargs


class TestSeedGroup:
    """Tests for the optional seed group."""

    def test_no_seed_group_roundtrips_unchanged(self, tmp_path, aerosol_data):
        # Seeding is additive: a file with no seed group must be untouched.
        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, aerosol_data)
        result = read_aerosol(path)

        assert all(result[key] is None for key in SEED_KEYS)
        with nc.Dataset(path, "r") as ds:
            assert "seed_bin" not in ds.dimensions
            assert ds.conventions == "CODT_aerosol_input_v1"

    def test_roundtrip_with_seed_group(self, tmp_path, aerosol_data):
        data = _two_type_background(aerosol_data)
        data.update(make_seed_group(**_seed_kwargs()))

        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, data)
        result = read_aerosol(path)

        np.testing.assert_allclose(result["seed_edge_radii"], [500.0, 1000.0, 2000.0])
        np.testing.assert_array_equal(result["seed_category"], [7, 8])
        np.testing.assert_array_equal(result["seed_bin_type"], [2, 2])
        np.testing.assert_allclose(result["seed_coord"], [500.0, 900.0])
        np.testing.assert_allclose(result["seed_concentration"], [10.0, 5.0])
        np.testing.assert_allclose(
            result["seed_frequency"], [[0.5, 1.0], [0.25, 1.0]]
        )

    def test_seed_frequency_dimension_order(self, tmp_path, aerosol_data):
        # CODT's docs list Fortran order (bin, event); ncdump and netCDF4 see
        # the reverse. Getting this backwards reads fine and samples wrong.
        data = _two_type_background(aerosol_data)
        data.update(make_seed_group(**_seed_kwargs()))

        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, data)
        with nc.Dataset(path, "r") as ds:
            assert ds["seed_frequency"].dimensions == ("seed_event", "seed_bin")
            assert ds.dimensions["seed_edge"].size == 3
            assert ds.dimensions["seed_bin"].size == 2
            assert ds.dimensions["seed_event"].size == 2

    def test_transposed_frequency_rejected(self):
        with pytest.raises(ValueError, match="netCDF declaration order"):
            make_seed_group(**_seed_kwargs(
                seed_frequency=[[0.5, 0.25], [1.0, 1.0]],
                seed_coord=[500.0, 900.0, 1200.0],
            ))

    def test_edge_must_be_bin_plus_one(self):
        with pytest.raises(ValueError, match="seed_bin \\+ 1"):
            make_seed_group(**_seed_kwargs(seed_edge_radii=[500.0, 1000.0]))

    def test_cdf_must_reach_one(self):
        with pytest.raises(ValueError, match="must run to 1.0"):
            make_seed_group(**_seed_kwargs(
                seed_frequency=[[0.5, 0.9], [0.25, 1.0]]
            ))

    def test_negative_concentration_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            make_seed_group(**_seed_kwargs(seed_concentration=[10.0, -1.0]))

    def test_duplicate_coord_rejected(self):
        with pytest.raises(ValueError, match="duplicate seed_coord"):
            make_seed_group(**_seed_kwargs(seed_coord=[500.0, 500.0]))

    def test_bin_type_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="must lie in 1..2"):
            make_seed_group(**_seed_kwargs(seed_bin_type=[2, 3]))

    def test_type_shared_with_background_rejected(self):
        # A type is seed material iff seed_bin_type references it, so sharing
        # a row with the background is ambiguous and CODT rejects it.
        with pytest.raises(ValueError, match="referenced by both"):
            make_seed_group(**_seed_kwargs(seed_bin_type=[1, 1]))

    def test_unreferenced_type_rejected(self):
        with pytest.raises(ValueError, match="referenced by no bin"):
            make_seed_group(**_seed_kwargs(
                seed_bin_type=[2, 2], n_types=3
            ))

    def test_category_collision_warns(self):
        # CODT does not check this; the two silently merge into one DSD_n.
        with pytest.warns(UserWarning, match="also used by the background"):
            make_seed_group(**_seed_kwargs(seed_category=[1, 8]))

    def test_distinct_categories_do_not_warn(self, recwarn):
        make_seed_group(**_seed_kwargs())
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_partial_seed_group_rejected_on_write(self, tmp_path, aerosol_data):
        data = _two_type_background(aerosol_data)
        data.update(make_seed_group(**_seed_kwargs()))
        del data["seed_coord"]

        with pytest.raises(ValueError, match="partial seed group"):
            write_aerosol(tmp_path / "bad.nc", data)

    def test_partial_seed_group_rejected_on_read(self, tmp_path, aerosol_data):
        data = _two_type_background(aerosol_data)
        data.update(make_seed_group(**_seed_kwargs()))
        path = tmp_path / "aerosol_input.nc"
        write_aerosol(path, data)

        with nc.Dataset(path, "a") as ds:
            ds.renameVariable("seed_coord", "seed_coord_x")

        with pytest.raises(ValueError, match="partial seed group"):
            read_aerosol(path)
