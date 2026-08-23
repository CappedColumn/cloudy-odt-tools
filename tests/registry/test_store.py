"""Tests for the minimal registry: a list of the simulations that were run.

What is *not* tested here, because it no longer exists: run status, status
events, experiments, namelist parameter storage, input checksums, executable
archiving, schema migrations. The registry records that a run happened; it
does not model the workflow.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from codt_tools.case import Case
from codt_tools.registry import COLUMNS, Registry
from codt_tools.registry.__main__ import main
from codt_tools.run import Run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def exe(tmp_path: Path) -> Path:
    """A stub binary that answers --version, as CODT does."""
    path = tmp_path / "CODT"
    path.write_text('#!/bin/bash\necho "CODT v3.1.0 (abc1234)"\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def runs(exe: Path, tmp_path: Path) -> list[Run]:
    """Three runs of one ensemble, sharing one binary."""
    base = Case()
    base.set(simulation_name="sweep")
    cases = base.sweep({"params.tref": [20.0, 21.0, 22.0]})
    return Run.for_cases(cases, exe, tmp_path / "ens")


@pytest.fixture
def reg(tmp_path: Path):
    """An open registry on a fresh file."""
    with Registry(tmp_path / "runs.db") as registry:
        yield registry


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------

class TestOpening:
    """One file, created on demand."""

    def test_creates_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "new.db"
        assert not path.exists()

        with Registry(path) as registry:
            assert len(registry) == 0

        assert path.is_file()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "runs.db"

        with Registry(path):
            pass

        assert path.is_file()

    def test_create_false_refuses_a_missing_file(self, tmp_path: Path) -> None:
        """A typo in a shared DB path should not silently start an empty
        list."""
        with pytest.raises(FileNotFoundError, match="create=True"):
            Registry(tmp_path / "nope.db", create=False)

    def test_reopening_keeps_the_rows(
        self, tmp_path: Path, runs: list[Run]
    ) -> None:
        path = tmp_path / "runs.db"
        with Registry(path) as registry:
            registry.add_many(runs)

        with Registry(path, create=False) as registry:
            assert len(registry) == 3

    def test_schema_is_one_table(self, reg: Registry) -> None:
        """The whole design: one table, seven columns."""
        tables = [
            row[0] for row in reg._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        assert tables == ["runs"]

        columns = [row[1] for row in reg._conn.execute("PRAGMA table_info(runs)")]
        assert tuple(columns) == COLUMNS


# ---------------------------------------------------------------------------
# Adding
# ---------------------------------------------------------------------------

class TestAdd:
    """Recording that a run exists."""

    def test_add_returns_the_run_name(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        assert reg.add(runs[0]) == "sweep_000"

    def test_records_the_paths_off_the_run(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        reg.add(runs[0])
        row = reg.get("sweep_000")

        assert row["workdir"] == str(runs[0].workdir)
        assert row["executable"] == str(runs[0].executable)
        assert Path(row["workdir"]).is_absolute()

    def test_captures_the_code_version(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        """Which CODT produced a result cannot be recovered later, so it is
        the one thing captured automatically."""
        reg.add(runs[0])

        assert reg.get("sweep_000")["code_version"] == "CODT v3.1.0 (abc1234)"

    def test_probes_the_version_once_per_executable(
        self, reg: Registry, runs: list[Run], exe: Path
    ) -> None:
        """250 runs of one binary must not mean 250 subprocess calls."""
        reg.add_many(runs)

        assert list(reg._version_cache) == [str(exe)]

    def test_run_id_can_be_overridden(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        reg.add(runs[0], run_id="EXP005_control")

        assert reg.get("EXP005_control") is not None
        assert reg.get("sweep_000") is None

    def test_readding_updates_rather_than_raising(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        """Re-staging an ensemble should not explode."""
        reg.add(runs[0], tags="first")
        reg.add(runs[0], tags="second")

        assert len(reg) == 1
        assert reg.get("sweep_000")["tags"] == "second"

    def test_add_many_returns_every_id(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        assert reg.add_many(runs, tags="EXP005") == [
            "sweep_000", "sweep_001", "sweep_002"
        ]
        assert len(reg) == 3

    def test_notes_and_tags_are_free_text(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        reg.add(runs[0], tags="EXP005_seeding", notes="control, rerun of 003")
        row = reg.get("sweep_000")

        assert row["tags"] == "EXP005_seeding"
        assert row["notes"] == "control, rerun of 003"

    def test_add_directory_without_a_run_object(
        self, reg: Registry, tmp_path: Path
    ) -> None:
        """For listing runs someone else produced."""
        elsewhere = tmp_path / "someone_elses_run"
        elsewhere.mkdir()

        assert reg.add_directory(elsewhere, tags="borrowed") == "someone_elses_run"

        row = reg.get("someone_elses_run")
        assert row["workdir"] == str(elsewhere)
        assert row["code_version"] is None      # no executable given

    def test_add_directory_captures_version_when_given_an_exe(
        self, reg: Registry, tmp_path: Path, exe: Path
    ) -> None:
        elsewhere = tmp_path / "run_x"
        elsewhere.mkdir()

        reg.add_directory(elsewhere, executable=exe)

        assert reg.get("run_x")["code_version"] == "CODT v3.1.0 (abc1234)"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class TestList:
    """The list, and the one way of narrowing it."""

    def test_empty_registry(self, reg: Registry) -> None:
        assert reg.list() == []
        assert len(reg) == 0

    def test_returns_plain_dicts(self, reg: Registry, runs: list[Run]) -> None:
        """No pandas, no row objects — just dicts."""
        reg.add_many(runs)
        rows = reg.list()

        assert all(isinstance(r, dict) for r in rows)
        assert set(rows[0]) == set(COLUMNS)

    def test_tag_is_a_substring_match(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        """tag='EXP005' finds tags='EXP005_seeding'."""
        reg.add(runs[0], tags="EXP005_seeding")
        reg.add(runs[1], tags="EXP005_control")
        reg.add(runs[2], tags="EXP006_other")

        assert len(reg.list(tag="EXP005")) == 2
        assert len(reg.list(tag="EXP006")) == 1
        assert reg.list(tag="nothing") == []

    def test_since_filters_by_date(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        reg.add_many(runs)

        assert len(reg.list(since="2000-01-01")) == 3
        assert reg.list(since="2999-01-01") == []

    def test_get_returns_none_for_an_unknown_run(self, reg: Registry) -> None:
        assert reg.get("never_ran") is None


class TestRemove:
    """Dropping a row, and only a row."""

    def test_removes_the_row(self, reg: Registry, runs: list[Run]) -> None:
        reg.add_many(runs)

        assert reg.remove("sweep_001") is True
        assert len(reg) == 2
        assert reg.get("sweep_001") is None

    def test_returns_false_when_absent(self, reg: Registry) -> None:
        assert reg.remove("never_ran") is False

    def test_leaves_the_files_alone(
        self, reg: Registry, runs: list[Run]
    ) -> None:
        runs[0].stage()
        reg.add(runs[0])

        reg.remove("sweep_000")

        assert runs[0].namelist_path.is_file()


# ---------------------------------------------------------------------------
# Decoupling
# ---------------------------------------------------------------------------

class TestOptional:
    """The registry is a side utility, not part of the workflow."""

    def test_nothing_in_the_package_imports_it(self) -> None:
        """Staging, launching and analyzing must all work without it —
        pinned so a future convenience import cannot quietly re-couple."""
        import subprocess
        import sys

        proc = subprocess.run(
            [
                sys.executable, "-c",
                "import codt_tools, codt_tools.run, codt_tools.simulation, sys;"
                "print([m for m in sys.modules"
                " if m.startswith('codt_tools.registry')])",
            ],
            capture_output=True, text=True, check=True,
        )

        assert proc.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    """Four commands."""

    def test_list_empty(self, tmp_path: Path, capsys) -> None:
        assert main(["--db", str(tmp_path / "runs.db"), "list"]) == 0
        assert "No runs recorded" in capsys.readouterr().out

    def test_add_then_list_then_show(
        self, tmp_path: Path, exe: Path, capsys
    ) -> None:
        db = str(tmp_path / "runs.db")
        run_dir = tmp_path / "a_run"
        run_dir.mkdir()

        assert main(["--db", db, "add", str(run_dir), "--executable", str(exe),
                     "--tag", "EXP005"]) == 0
        assert main(["--db", db, "list"]) == 0
        out = capsys.readouterr().out
        assert "a_run" in out
        assert "1 run(s)" in out

        assert main(["--db", db, "show", "a_run"]) == 0
        assert "CODT v3.1.0" in capsys.readouterr().out

    def test_list_filters_by_tag(
        self, tmp_path: Path, capsys
    ) -> None:
        db = str(tmp_path / "runs.db")
        for name, tag in [("one", "EXP005"), ("two", "EXP006")]:
            directory = tmp_path / name
            directory.mkdir()
            main(["--db", db, "add", str(directory), "--tag", tag])
        capsys.readouterr()

        main(["--db", db, "list", "--tag", "EXP005"])
        out = capsys.readouterr().out
        assert "one" in out and "two" not in out

    def test_show_unknown_run_exits_nonzero(self, tmp_path: Path) -> None:
        assert main(["--db", str(tmp_path / "runs.db"), "show", "nope"]) == 1

    def test_remove(self, tmp_path: Path) -> None:
        db = str(tmp_path / "runs.db")
        directory = tmp_path / "gone"
        directory.mkdir()
        main(["--db", db, "add", str(directory)])

        assert main(["--db", db, "remove", "gone"]) == 0
        assert main(["--db", db, "remove", "gone"]) == 1

    def test_missing_db_argument_is_an_error(
        self, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("CODT_REGISTRY_DB", raising=False)

        assert main(["list"]) == 2
        assert "CODT_REGISTRY_DB" in capsys.readouterr().err

    def test_db_comes_from_the_environment(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("CODT_REGISTRY_DB", str(tmp_path / "env.db"))

        assert main(["list"]) == 0
        assert (tmp_path / "env.db").is_file()
