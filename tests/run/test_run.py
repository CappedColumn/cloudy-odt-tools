"""Tests for ``Run``: staging a case into a directory and executing it.

The invariant these guard most carefully is the one inherited from Stage 1:
CODT resolves ``aerosol_file`` and ``parcel_file`` against the parent of
``argv[1]`` *as typed* (``app/main.f90:43``), so a run is only correct when
the model is launched with an **absolute** namelist path. That was pinned by
``tests/test_runner.py::TestAbsoluteNamelistPath`` and is pinned here.

The stub binaries below are real executable shell scripts rather than
monkeypatched ``subprocess.run`` calls, so what is tested is the actual
invocation.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from codt_tools.case import Case
from codt_tools.run import Run, check_executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub(path: Path, body: str) -> Path:
    """Write an executable shell script standing in for the CODT binary."""
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return path


@pytest.fixture
def case() -> Case:
    """A minimal chamber case."""
    cfg = Case()
    cfg.set(simulation_name="test_sim")
    return cfg


@pytest.fixture
def run(case: Case, tmp_path: Path) -> Run:
    """An unstaged run in a temp directory."""
    return Run(case, tmp_path / "bin" / "CODT", tmp_path / "runs" / "control")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

class TestPaths:
    """Every path is derived from workdir, and every path is absolute."""

    def test_derived_paths(self, run: Run) -> None:
        assert run.inputs_dir == run.workdir / "inputs"
        assert run.output_dir == run.workdir / "output"
        assert run.namelist_path == run.workdir / "inputs" / "params.nml"
        assert run.name == "control"

    def test_relative_paths_are_resolved_absolute(
        self, case: Case, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        run = Run(case, "bin/CODT", "runs/control")

        assert run.executable.is_absolute()
        assert run.workdir.is_absolute()
        assert run.namelist_path.is_absolute()

    def test_done_marker_uses_the_simulation_name_not_the_directory(
        self, case: Case, tmp_path: Path
    ) -> None:
        # A run directory need not be named after the simulation; the model
        # builds the marker from simulation_name either way.
        run = Run(case, tmp_path / "CODT", tmp_path / "run_017")

        assert run.done_marker == run.output_dir / "test_sim_DONE"


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

class TestStage:
    """Staging writes the inputs and leaves the case alone."""

    def test_creates_the_directory_layout(self, run: Run) -> None:
        run.stage()

        assert run.inputs_dir.is_dir()
        assert run.output_dir.is_dir()
        assert run.namelist_path.is_file()
        assert (run.inputs_dir / "aerosol_input.nc").is_file()

    def test_is_staged_tracks_the_namelist(self, run: Run) -> None:
        assert not run.is_staged
        run.stage()
        assert run.is_staged

    def test_staged_namelist_points_at_this_runs_output(self, run: Run) -> None:
        staged = run.stage()

        assert staged.get("output_directory") == str(run.output_dir)
        assert staged.get("aerosol_file") == "aerosol_input.nc"

    def test_does_not_mutate_the_case(self, run: Run) -> None:
        run.stage()

        assert run.case.params.get("output_directory") == ""
        assert run.case.params.get("aerosol_file") == "aerosol_input.nc"

    def test_one_case_stages_two_independent_runs(
        self, case: Case, tmp_path: Path
    ) -> None:
        first = Run(case, tmp_path / "CODT", tmp_path / "a")
        second = Run(case, tmp_path / "CODT", tmp_path / "b")

        staged_a = first.stage()
        staged_b = second.stage()

        assert staged_a.get("output_directory") != staged_b.get(
            "output_directory"
        )
        assert staged_a.get("output_directory") == str(first.output_dir)
        assert staged_b.get("output_directory") == str(second.output_dir)


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

class TestIsComplete:
    """Completion is the model's own ``_DONE`` marker, nothing else."""

    def test_false_before_and_true_after(self, run: Run) -> None:
        run.stage()
        assert not run.is_complete

        run.done_marker.write_text("2026-08-23 12:00:00\n")
        assert run.is_complete

    def test_output_files_alone_do_not_count(self, run: Run) -> None:
        run.stage()
        (run.output_dir / "test_sim.nc").write_text("partial")

        assert not run.is_complete


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

class TestExecuteLocal:
    """What actually gets handed to the binary."""

    def test_invokes_with_an_absolute_namelist_path(
        self, case: Case, tmp_path: Path, monkeypatch
    ) -> None:
        """The Stage 1 invariant: bare staged filenames only resolve when
        CODT is launched with an absolute namelist path."""
        argv_log = tmp_path / "argv.txt"
        exe = _stub(tmp_path / "CODT", f'echo "$1" > {argv_log}')
        run = Run(case, exe, tmp_path / "runs" / "control")
        run.stage()

        # Run from elsewhere: a relative path would resolve differently here.
        monkeypatch.chdir(tmp_path)
        run.execute_local()

        passed = Path(argv_log.read_text().strip())
        assert passed.is_absolute()
        assert passed == run.namelist_path

    def test_returns_the_completed_process(
        self, case: Case, tmp_path: Path
    ) -> None:
        exe = _stub(tmp_path / "CODT", "exit 0")
        run = Run(case, exe, tmp_path / "run")
        run.stage()

        assert run.execute_local().returncode == 0

    def test_propagates_a_failure_code(
        self, case: Case, tmp_path: Path
    ) -> None:
        exe = _stub(tmp_path / "CODT", "exit 1")
        run = Run(case, exe, tmp_path / "run")
        run.stage()

        assert run.execute_local().returncode == 1

    def test_missing_executable_raises(self, run: Run) -> None:
        run.stage()
        with pytest.raises(FileNotFoundError, match="Executable not found"):
            run.execute_local()

    def test_non_executable_binary_raises(
        self, case: Case, tmp_path: Path
    ) -> None:
        exe = tmp_path / "CODT"
        exe.write_text("#!/bin/bash\nexit 0\n")
        exe.chmod(0o644)
        run = Run(case, exe, tmp_path / "run")
        run.stage()

        with pytest.raises(PermissionError, match="chmod"):
            run.execute_local()

    def test_unstaged_run_raises(self, case: Case, tmp_path: Path) -> None:
        exe = _stub(tmp_path / "CODT", "exit 0")
        run = Run(case, exe, tmp_path / "run")

        with pytest.raises(FileNotFoundError, match="stage"):
            run.execute_local()

    def test_fatal_signal_is_explained(
        self, case: Case, tmp_path: Path
    ) -> None:
        """A wrong-CPU binary dies on SIGILL; raw output says only
        'Illegal instruction', so the warning has to name the cause."""
        exe = _stub(tmp_path / "CODT", "kill -ILL $$")
        run = Run(case, exe, tmp_path / "run")
        run.stage()

        with pytest.warns(RuntimeWarning, match="different CPU"):
            proc = run.execute_local()

        assert proc.returncode < 0


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

class TestForCases:
    """One line turns a design into a list of runs."""

    def test_names_directories_from_case_names(self, tmp_path: Path) -> None:
        base = Case()
        base.set(simulation_name="sweep")
        cases = base.sweep({"params.tref": [20.0, 21.0, 22.0]})

        runs = Run.for_cases(cases, tmp_path / "CODT", tmp_path / "ensemble")

        assert [r.name for r in runs] == ["sweep_000", "sweep_001", "sweep_002"]
        assert all(r.workdir.parent == (tmp_path / "ensemble") for r in runs)

    def test_shares_one_executable(self, tmp_path: Path) -> None:
        base = Case()
        base.set(simulation_name="sweep")
        cases = base.sweep({"params.tref": [20.0, 21.0]})

        runs = Run.for_cases(cases, tmp_path / "CODT", tmp_path / "ensemble")

        assert {r.executable for r in runs} == {(tmp_path / "CODT").resolve()}

    def test_duplicate_names_raise(self, tmp_path: Path) -> None:
        """Two cases with one name would stage into the same directory,
        the second silently overwriting the first."""
        first, second = Case(), Case()
        first.set(simulation_name="same")
        second.set(simulation_name="same")

        with pytest.raises(ValueError, match="same directory"):
            Run.for_cases([first, second], tmp_path / "CODT", tmp_path / "e")

    def test_each_run_stages_independently(self, tmp_path: Path) -> None:
        base = Case()
        base.set(simulation_name="sweep")
        cases = base.sweep({"params.tref": [20.0, 21.0]})
        runs = Run.for_cases(cases, tmp_path / "CODT", tmp_path / "ensemble")

        for one in runs:
            one.stage()

        assert runs[0].namelist_path.is_file()
        assert runs[1].namelist_path.is_file()
        assert runs[0].output_dir != runs[1].output_dir


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class TestOpenSimulation:
    """Opening output is gated on the run having actually finished."""

    def test_opens_a_completed_run(
        self, case: Case, completed_run_dir: Path
    ) -> None:
        run = Run(case, completed_run_dir / "CODT", completed_run_dir)

        sim = run.open_simulation()

        assert sim.T.shape[0] > 0

    def test_incomplete_run_raises(self, run: Run) -> None:
        run.stage()
        with pytest.raises(FileNotFoundError, match="not complete"):
            run.open_simulation()


# ---------------------------------------------------------------------------
# Executable check
# ---------------------------------------------------------------------------

class TestCheckExecutable:
    """A one-line answer to 'why will this binary not run?'."""

    def test_working_binary_reports_nothing(self, tmp_path: Path) -> None:
        exe = _stub(tmp_path / "CODT", 'echo "CODT 3.1.0 (abc1234)"')

        assert check_executable(exe) is None

    def test_missing_file(self, tmp_path: Path) -> None:
        assert check_executable(tmp_path / "nope") == "no such file"

    def test_directory(self, tmp_path: Path) -> None:
        assert "directory" in check_executable(tmp_path)

    def test_not_executable(self, tmp_path: Path) -> None:
        exe = tmp_path / "CODT"
        exe.write_text("#!/bin/bash\nexit 0\n")
        exe.chmod(0o644)

        assert "chmod" in check_executable(exe)

    def test_nonzero_exit_reports_the_first_line(
        self, tmp_path: Path
    ) -> None:
        """The real case this catches: a binary whose runtime libraries
        are missing, which fails in the loader before main()."""
        exe = _stub(
            tmp_path / "CODT",
            'echo "libgfortran.so.5: version GFORTRAN_10 not found" >&2\n'
            "exit 1",
        )

        problem = check_executable(exe)

        assert "GFORTRAN_10" in problem
        assert "exited 1" in problem

    def test_sigill_names_the_cpu_mismatch(self, tmp_path: Path) -> None:
        exe = _stub(tmp_path / "CODT", "kill -ILL $$")

        problem = check_executable(exe)

        assert "SIGILL" in problem
        assert "different CPU" in problem

    @pytest.mark.skipif(
        os.geteuid() == 0, reason="root ignores the execute bit"
    )
    def test_never_raises(self, tmp_path: Path) -> None:
        """Callers check several binaries and print a table; an exception
        from one would abort the lot."""
        for candidate in [tmp_path / "nope", tmp_path, tmp_path / "CODT"]:
            assert check_executable(candidate) is None or isinstance(
                check_executable(candidate), str
            )
