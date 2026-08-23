"""Tests for path ownership: who decides where CODT reads and writes.

Four namelist keys point CODT at data, and they resolve two different ways:

- ``aerosol_file`` / ``parcel_file`` / ``mie_data_file`` against the directory
  of the namelist path as typed (``app/main.f90:43``, ``globals.f90:417``);
- ``output_directory`` against the **process's** working directory
  (``initialize.f90:246``).

A ``Case`` therefore holds none of them: staging assigns all four, every time.
These tests pin that rule, because getting it wrong points a run at another
run's data and CODT only notices when ``overwrite=.false.`` happens to catch it.
"""

from __future__ import annotations

from pathlib import Path

import f90nml
import pytest

from codt_tools.case import Case, Namelist


class TestCaseCarriesNoPaths:
    """The staged keys are neutral on a case and cannot be set on one."""

    def test_defaults_are_neutral(self) -> None:
        case = Case()
        assert case.params.get("output_directory") == ""
        assert case.params.get("aerosol_file") == "aerosol_input.nc"
        assert case.params.get("parcel_file") == ""

    @pytest.mark.parametrize(
        "key", ["output_directory", "aerosol_file", "parcel_file"]
    )
    def test_set_refuses_staged_keys(self, key: str) -> None:
        case = Case()
        with pytest.raises(ValueError, match="write_inputs"):
            case.set(**{key: "/somewhere/else"})

    def test_mie_data_file_is_settable(self, tmp_path: Path) -> None:
        """It names a *source* file the user owns, not a staged location."""
        case = Case()
        case.set(mie_data_file=str(tmp_path / "mie.txt"))
        assert case.params.get("mie_data_file") == str(tmp_path / "mie.txt")

    def test_from_input_dir_drops_the_paths_it_read(
        self, tmp_path: Path
    ) -> None:
        """A case loaded from a finished run must not inherit its output dir.

        Otherwise writing that case again aims the new run at the old run's
        directory, which CODT only refuses when the .nc already exists.
        """
        source = tmp_path / "old_run" / "inputs"
        source.mkdir(parents=True)
        Case().aerosol.write(source / "oddly_named.nc")
        nml = Namelist()
        nml.set(
            simulation_name="old",
            output_directory=str(tmp_path / "old_run" / "output"),
            aerosol_file="oddly_named.nc",
        )
        nml.write(source / "params.nml")

        case = Case.from_input_dir(source)

        assert case.name == "old"
        assert case.params.get("output_directory") == ""
        assert case.params.get("aerosol_file") == "aerosol_input.nc"
        # the data was still found and loaded through the path it named
        assert case.aerosol.n_bins > 0


class TestWriteInputs:
    """write_inputs assigns every path, from its own arguments."""

    def test_stages_absolute_output_directory(self, tmp_path: Path) -> None:
        case = Case()
        staged = case.write_inputs(
            tmp_path / "run" / "inputs",
            output_directory=tmp_path / "run" / "output",
        )

        value = staged.get("output_directory")
        assert Path(value).is_absolute()
        assert value == str(tmp_path / "run" / "output")
        on_disk = f90nml.read(tmp_path / "run" / "inputs" / "params.nml")
        assert on_disk["parameters"]["output_directory"] == value

    def test_creates_the_output_directory(self, tmp_path: Path) -> None:
        """CODT aborts when output_directory's parent does not exist."""
        case = Case()
        case.write_inputs(
            tmp_path / "run" / "inputs",
            output_directory=tmp_path / "run" / "deep" / "output",
        )
        assert (tmp_path / "run" / "deep" / "output").is_dir()

    def test_defaults_output_beside_the_inputs(self, tmp_path: Path) -> None:
        case = Case()
        staged = case.write_inputs(tmp_path / "run")
        assert staged.get("output_directory") == str(tmp_path / "run" / "output")

    def test_data_files_are_referenced_by_bare_name(
        self, tmp_path: Path
    ) -> None:
        """CODT joins these onto the namelist's own directory, so a bare name
        is what makes a run directory relocatable."""
        case = Case()
        case.set(simulation_mode="parcel")
        staged = case.write_inputs(tmp_path / "run")

        assert staged.get("aerosol_file") == "aerosol_input.nc"
        assert staged.get("parcel_file") == "parcel_input.nc"
        assert (tmp_path / "run" / "aerosol_input.nc").is_file()
        assert (tmp_path / "run" / "parcel_input.nc").is_file()

    def test_chamber_mode_clears_parcel_file(self, tmp_path: Path) -> None:
        case = Case()
        case.set(simulation_mode="chamber")
        staged = case.write_inputs(tmp_path / "run")

        assert staged.get("parcel_file") == ""
        assert not (tmp_path / "run" / "parcel_input.nc").exists()

    def test_leaves_the_case_untouched(self, tmp_path: Path) -> None:
        case = Case()
        case.set(simulation_mode="parcel")
        before = case.params.groups_for_write()

        case.write_inputs(tmp_path / "run", output_directory=tmp_path / "out")

        assert case.params.groups_for_write() == before
        assert case.params.get("output_directory") == ""

    def test_second_write_is_not_polluted_by_the_first(
        self, tmp_path: Path
    ) -> None:
        """The bug this rule exists to prevent: two runs, one output dir."""
        case = Case()
        first = case.write_inputs(
            tmp_path / "a" / "inputs", output_directory=tmp_path / "a" / "output"
        )
        second = case.write_inputs(
            tmp_path / "b" / "inputs", output_directory=tmp_path / "b" / "output"
        )

        assert first.get("output_directory") == str(tmp_path / "a" / "output")
        assert second.get("output_directory") == str(tmp_path / "b" / "output")

    def test_validates_leg_direction_at_write_time(
        self, tmp_path: Path
    ) -> None:
        """write_parcel got no launch level before, so a leg pointing the
        wrong way was only caught by validate() — or by CODT."""
        case = Case()
        case.set(simulation_mode="parcel", initial_height=0.0)
        case.parcel.set(segment_coord=[1000.0], velocity=[-1.0])

        with pytest.raises(ValueError):
            case.write_inputs(tmp_path / "run")


class TestNamelistWriteGuard:
    """An unstaged namelist must not reach disk."""

    def test_empty_output_directory_raises(self, tmp_path: Path) -> None:
        nml = Namelist()
        with pytest.raises(ValueError, match="output_directory is empty"):
            nml.write(tmp_path / "params.nml")

    def test_explicit_output_directory_writes(self, tmp_path: Path) -> None:
        nml = Namelist()
        nml.set(output_directory=str(tmp_path / "out"))
        nml.write(tmp_path / "params.nml")
        assert (tmp_path / "params.nml").is_file()
