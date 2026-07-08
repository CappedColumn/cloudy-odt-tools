"""Tests for codt_tools.registry: schema, API, gating, and concurrency."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
from pathlib import Path

import netCDF4 as nc
import pytest

from codt_tools.config import CODTConfig
from codt_tools.registry import (
    SCHEMA_VERSION,
    IncompatibleConventionsError,
    Registry,
    check_conventions,
    connect,
    sha256_file,
)
from codt_tools.registry.versions import archive_executable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    """A fresh registry with a tmp database."""
    reg = Registry(tmp_path / "registry.db")
    yield reg
    reg.close()


@pytest.fixture
def run_dir(tmp_path):
    """A run directory with inputs/ written by a real CODTConfig."""
    config = CODTConfig()
    config.set(simulation_name="test_run")
    rdir = tmp_path / "exp" / "runs" / "test_run"
    inputs = rdir / "inputs"
    inputs.mkdir(parents=True)
    config.injection.write(inputs / "aerosol_input.nc")
    config.params.set(aerosol_file="aerosol_input.nc")
    config.params.write(inputs / "params.nml")
    return rdir, config


def _register(reg: Registry, run_dir_config, run_id: str = "20260708_000000_codt_test",
              **kwargs) -> str:
    rdir, config = run_dir_config
    return reg.register_run(run_id, config, rdir, **kwargs)


# ---------------------------------------------------------------------------
# Schema / connection
# ---------------------------------------------------------------------------

class TestSchema:
    def test_fresh_db_has_schema_version(self, tmp_path):
        conn = connect(tmp_path / "r.db")
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        conn.close()

    def test_pragmas(self, tmp_path):
        conn = connect(tmp_path / "r.db")
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        conn.close()

    def test_tables_exist(self, tmp_path):
        conn = connect(tmp_path / "r.db")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "experiments",
            "runs",
            "input_files",
            "namelist_parameters",
            "status_events",
            "output_files",
        } <= tables
        conn.close()

    def test_reopen_idempotent(self, tmp_path):
        connect(tmp_path / "r.db").close()
        conn = connect(tmp_path / "r.db")
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        conn.close()

    def test_missing_db_without_create_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            connect(tmp_path / "missing.db", create=False)

    def test_newer_schema_rejected(self, tmp_path):
        path = tmp_path / "r.db"
        connect(path).close()
        raw = sqlite3.connect(path)
        raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        raw.close()
        with pytest.raises(RuntimeError, match="newer than supported"):
            connect(path)

    def test_status_check_constraint(self, tmp_path):
        conn = connect(tmp_path / "r.db")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO runs (run_id, run_dir, status, created_at) "
                "VALUES ('x', '.', 'bogus', '2026-01-01')"
            )
        conn.close()


# ---------------------------------------------------------------------------
# Versions / gating / hashing
# ---------------------------------------------------------------------------

class TestVersions:
    def test_supported_conventions_pass(self):
        assert check_conventions("CODT_output_v1") is True

    def test_unsupported_warns(self):
        with pytest.warns(UserWarning, match="not in supported set"):
            assert check_conventions("CODT_output_v99") is False

    def test_strict_raises(self):
        with pytest.raises(IncompatibleConventionsError):
            check_conventions(None, strict=True)

    def test_sha256_matches_hashlib(self, tmp_path):
        import hashlib

        f = tmp_path / "data.bin"
        f.write_bytes(b"codt" * 1000)
        assert sha256_file(f) == hashlib.sha256(b"codt" * 1000).hexdigest()

    def test_archive_executable_dedups(self, tmp_path):
        exe = tmp_path / "CODT"
        exe.write_bytes(b"\x7fELF fake binary")
        archive = tmp_path / "archive"
        first = archive_executable(exe, archive)
        second = archive_executable(exe, archive)
        assert first == second
        assert first.read_bytes() == exe.read_bytes()
        assert len(list(archive.rglob("CODT"))) == 1


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

class TestExperiments:
    def test_create_and_get(self, registry):
        registry.create_experiment("exp1", "Title", hypothesis="Does X affect Y?")
        exp = registry.get_experiment("exp1")
        assert exp["status"] == "planned"
        assert exp["hypothesis"] == "Does X affect Y?"

    def test_conclude(self, registry):
        registry.create_experiment("exp1", "Title")
        registry.conclude_experiment("exp1", "X does not affect Y.", ["a.png"])
        exp = registry.get_experiment("exp1")
        assert exp["status"] == "concluded"
        assert exp["concluded_at"] is not None
        assert "a.png" in exp["analysis_artifacts"]

    def test_list_filter(self, registry):
        registry.create_experiment("exp1", "One")
        registry.create_experiment("exp2", "Two")
        registry.conclude_experiment("exp2", "done")
        assert len(registry.list_experiments()) == 2
        assert [e["experiment_id"] for e in registry.list_experiments("concluded")] == [
            "exp2"
        ]

    def test_update_rejects_bad_column(self, registry):
        registry.create_experiment("exp1", "One")
        with pytest.raises(ValueError):
            registry.update_experiment("exp1", created_at="2020-01-01")

    def test_unknown_experiment_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_experiment("nope")


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class TestRuns:
    def test_register_run_basics(self, registry, run_dir):
        run_id = _register(registry, run_dir)
        run = registry.get_run(run_id)
        assert run["status"] == "registered"
        assert run["codt_tools_version"] is not None

    def test_input_files_checksummed(self, registry, run_dir):
        run_id = _register(registry, run_dir)
        rows = registry._conn.execute(
            "SELECT * FROM input_files WHERE run_id = ?", (run_id,)
        ).fetchall()
        by_type = {r["file_type"]: r for r in rows}
        assert set(by_type) == {"namelist", "aerosol_input"}
        rdir, _ = run_dir
        expected = sha256_file(rdir / "inputs" / "params.nml")
        assert by_type["namelist"]["checksum"] == expected

    def test_symlinked_input_recorded(self, registry, tmp_path, run_dir):
        rdir, config = run_dir
        shared = rdir.parent.parent / "shared_inputs"
        shared.mkdir()
        real = rdir / "inputs" / "aerosol_input.nc"
        target = shared / "aerosol_input.nc"
        real.rename(target)
        real.symlink_to(Path("..") / ".." / ".." / "shared_inputs" / "aerosol_input.nc")
        run_id = _register(registry, (rdir, config))
        row = registry._conn.execute(
            "SELECT * FROM input_files WHERE run_id = ? AND file_type = "
            "'aerosol_input'",
            (run_id,),
        ).fetchone()
        assert row["is_symlink"] == 1
        assert "shared_inputs" in row["link_target"]
        assert row["checksum"] == sha256_file(target)

    def test_namelist_parameters_stored(self, registry, run_dir):
        run_id = _register(registry, run_dir)
        params = registry.run_parameters(run_id)
        assert params["parameters"]["simulation_name"] == "test_run"
        row = registry._conn.execute(
            "SELECT param_type FROM namelist_parameters WHERE run_id = ? "
            "AND param_name = 'n'",
            (run_id,),
        ).fetchone()
        assert row["param_type"] == "integer"

    def test_run_linked_to_experiment_relative_dir(self, registry, run_dir):
        rdir, config = run_dir
        data_root = rdir.parent.parent  # tmp/exp
        registry.create_experiment("exp1", "One", data_root=data_root)
        run_id = _register(registry, (rdir, config), experiment_id="exp1")
        run = registry.get_run(run_id)
        assert run["run_dir"] == "runs/test_run"

    def test_executable_archived(self, registry, run_dir, tmp_path):
        exe = tmp_path / "CODT"
        exe.write_bytes(b"\x7fELF fake binary")
        run_id = _register(registry, run_dir, executable_path=exe)
        run = registry.get_run(run_id)
        assert run["executable_checksum"] == sha256_file(exe)
        archived = Path(run["executable_archive_path"])
        assert archived.exists()
        assert archived.read_bytes() == exe.read_bytes()

    def test_duplicate_run_id_rejected(self, registry, run_dir):
        _register(registry, run_dir)
        with pytest.raises(sqlite3.IntegrityError):
            _register(registry, run_dir)


# ---------------------------------------------------------------------------
# Status lifecycle
# ---------------------------------------------------------------------------

class TestStatus:
    def test_transitions_and_events(self, registry, run_dir):
        run_id = _register(registry, run_dir)
        registry.update_status(run_id, "queued", slurm_job_id="123")
        registry.update_status(run_id, "running")
        registry.update_status(run_id, "completed", exit_code=0)
        run = registry.get_run(run_id)
        assert run["status"] == "completed"
        assert run["exit_code"] == 0
        assert run["slurm_job_id"] == "123"
        assert run["started_at"] is not None
        assert run["completed_at"] is not None
        events = registry.run_events(run_id)
        assert [e["status"] for e in events] == [
            "registered",
            "queued",
            "running",
            "completed",
        ]

    def test_invalid_status_rejected(self, registry, run_dir):
        run_id = _register(registry, run_dir)
        with pytest.raises(ValueError):
            registry.update_status(run_id, "bogus")

    def test_unknown_run_raises(self, registry):
        with pytest.raises(KeyError):
            registry.update_status("nope", "running")

    def test_data_status(self, registry, run_dir):
        run_id = _register(registry, run_dir)
        registry.set_data_status(run_id, "on_group")
        assert registry.get_run(run_id)["data_status"] == "on_group"


# ---------------------------------------------------------------------------
# record_completion / conventions gate
# ---------------------------------------------------------------------------

class TestRecordCompletion:
    def _make_output(self, path: Path, conventions: str) -> Path:
        out = path / "out.nc"
        with nc.Dataset(out, "w") as ds:
            ds.setncattr("conventions", conventions)
            ds.setncattr("code_version", "0.6.0")
            ds.setncattr("git_commit", "894cafc")
        return out

    def test_records_versions(self, registry, run_dir, tmp_path):
        run_id = _register(registry, run_dir)
        out = self._make_output(tmp_path, "CODT_output_v1")
        registry.record_completion(run_id, out)
        run = registry.get_run(run_id)
        assert run["conventions"] == "CODT_output_v1"
        assert run["code_version"] == "0.6.0"
        assert run["git_commit"] == "894cafc"

    def test_unsupported_conventions_warns_but_records(
        self, registry, run_dir, tmp_path
    ):
        run_id = _register(registry, run_dir)
        out = self._make_output(tmp_path, "CODT_output_v99")
        with pytest.warns(UserWarning):
            registry.record_completion(run_id, out)
        assert registry.get_run(run_id)["conventions"] == "CODT_output_v99"


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestQueries:
    def test_query_by_param_value(self, registry, run_dir):
        run_id = _register(registry, run_dir)
        name = registry.run_parameters(run_id)["parameters"]["simulation_name"]
        hits = registry.query_runs(param="simulation_name", value=name)
        assert [r["run_id"] for r in hits] == [run_id]
        assert registry.query_runs(param="simulation_name", value="nope") == []

    def test_query_by_status_and_experiment(self, registry, run_dir):
        registry.create_experiment("exp1", "One")
        run_id = _register(registry, run_dir, experiment_id="exp1")
        registry.update_status(run_id, "running")
        assert registry.query_runs(experiment_id="exp1", status="running")
        assert not registry.query_runs(experiment_id="exp1", status="failed")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def _hammer(db_path: str, run_id: str, n_updates: int) -> None:
    reg = Registry(db_path)
    for _ in range(n_updates):
        reg.update_status(run_id, "running")
    reg.close()


class TestConcurrency:
    def test_no_lost_events(self, registry, run_dir):
        """8 processes x 25 updates: every event must land."""
        run_id = _register(registry, run_dir)
        n_procs, n_updates = 8, 25
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_hammer, args=(str(registry.db_path), run_id, n_updates))
            for _ in range(n_procs)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0
        events = registry.run_events(run_id)
        assert len(events) == 1 + n_procs * n_updates  # registered + updates
