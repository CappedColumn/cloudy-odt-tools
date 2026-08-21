"""Tests for codt_tools.registry: schema, API, gating, and concurrency."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import warnings
from pathlib import Path

import netCDF4 as nc
import pytest

from codt_tools.case import Case
from codt_tools.registry import (
    SCHEMA_VERSION,
    IncompatibleConventionsError,
    Registry,
    check_conventions,
    check_input_conventions,
    check_seeding_consistency,
    connect,
    inspect_input_file,
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
    """A run directory with inputs/ written by a real Case."""
    config = Case()
    config.set(simulation_name="test_run")
    rdir = tmp_path / "exp" / "runs" / "test_run"
    staged = config.write_inputs(rdir / "inputs", output_directory=rdir / "output")
    return rdir, config, staged


def _register(reg: Registry, run_dir_config, run_id: str = "20260708_000000_codt_test",
              **kwargs) -> str:
    rdir, config, staged = run_dir_config
    kwargs.setdefault("namelist", staged)
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

    @staticmethod
    def _make_old_db(path, version):
        """Build a database at an older schema version from the current DDL.

        Columns added after *version* are stripped line-wise, so each
        migration step is exercised against a schema that genuinely lacks
        what it adds.
        """
        from codt_tools.registry.db import SCHEMA_SQL

        added_after = {
            1: ("permanent_data_root", "schema_conventions",
                "has_seed_group", "build_arch"),
            2: ("schema_conventions", "has_seed_group", "build_arch"),
            3: ("build_arch",),
        }[version]
        old_sql = "\n".join(
            line
            for line in SCHEMA_SQL.splitlines()
            if not any(col in line for col in added_after)
        )
        raw = sqlite3.connect(path)
        raw.executescript(old_sql)
        raw.execute(f"PRAGMA user_version = {version}")
        raw.commit()
        raw.close()

    def _columns(self, conn, table):
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    def test_v1_database_migrates_to_current(self, tmp_path):
        """A v1 DB upgrades in place through every intermediate step."""
        path = tmp_path / "r.db"
        self._make_old_db(path, 1)

        conn = connect(path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert "permanent_data_root" in self._columns(conn, "experiments")
        assert {"schema_conventions", "has_seed_group"} <= self._columns(
            conn, "input_files"
        )
        assert "build_arch" in self._columns(conn, "runs")
        conn.close()

    def test_v3_database_gains_build_arch(self, tmp_path):
        """A v3 DB gains build_arch; existing runs read NULL."""
        path = tmp_path / "r.db"
        self._make_old_db(path, 3)
        raw = sqlite3.connect(path)
        raw.execute(
            "INSERT INTO runs (run_id, run_dir, created_at) "
            "VALUES ('r1', 'd', 't')"
        )
        raw.commit()
        raw.close()

        conn = connect(path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        row = conn.execute("SELECT build_arch FROM runs").fetchone()
        # NULL: the architecture of a past run's binary is unrecoverable.
        assert row["build_arch"] is None
        conn.close()

    def test_v2_database_migrates_to_v3(self, tmp_path):
        """A v2 DB gains the input-schema columns; existing rows read NULL."""
        path = tmp_path / "r.db"
        self._make_old_db(path, 2)
        raw = sqlite3.connect(path)
        raw.execute(
            "INSERT INTO runs (run_id, run_dir, created_at) VALUES ('r1', 'd', 't')"
        )
        raw.execute(
            "INSERT INTO input_files (run_id, file_type, file_path) "
            "VALUES ('r1', 'aerosol_input', 'aerosol_input.nc')"
        )
        raw.commit()
        raw.close()

        conn = connect(path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        row = conn.execute(
            "SELECT schema_conventions, has_seed_group FROM input_files"
        ).fetchone()
        # NULL means "not recorded", distinct from a recorded absence (0).
        assert row["schema_conventions"] is None
        assert row["has_seed_group"] is None
        conn.close()

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

    def test_parcel_v3_accepted(self):
        assert check_input_conventions("parcel_input", "CODT_parcel_input_v3") is True

    @pytest.mark.parametrize("retired", ["CODT_parcel_input_v1", "CODT_parcel_input_v2"])
    def test_retired_parcel_versions_explain_why(self, retired):
        with pytest.warns(UserWarning, match="ent_rate as 1/km"):
            assert check_input_conventions("parcel_input", retired) is False

    def test_aerosol_v1_is_current_not_legacy(self):
        """v1 is the v3-era aerosol string; seeded files still declare it."""
        assert check_input_conventions("aerosol_input", "CODT_aerosol_input_v1") is True

    def test_unknown_file_type_passes_unchecked(self):
        assert check_input_conventions("namelist", None) is True

    def test_input_strict_raises(self):
        with pytest.raises(IncompatibleConventionsError):
            check_input_conventions("parcel_input", "CODT_parcel_input_v1", strict=True)

    @pytest.mark.parametrize(
        "do_seeding,has_group",
        # do_seeding is the sole controller, so the gate is one-way. Only
        # seeding-on with no group is fatal; a dormant group (off + present) is
        # explicitly fine — one file serving a seeded and an unseeded run.
        [(True, True), (False, False), (False, True)],
    )
    def test_seeding_one_way_gate_passes(self, do_seeding, has_group):
        assert check_seeding_consistency(do_seeding, has_group) is True

    def test_dormant_seed_group_does_not_warn(self, recwarn):
        """Seeding off with a group present is valid and silent at this layer."""
        assert check_seeding_consistency(False, True) is True
        assert len(recwarn) == 0

    def test_seeding_flag_without_group_warns(self):
        with pytest.warns(UserWarning, match="no seed group"):
            assert check_seeding_consistency(True, False) is False

    def test_seeding_strict_raises(self):
        with pytest.raises(IncompatibleConventionsError):
            check_seeding_consistency(True, False, strict=True)

    def test_inspect_detects_seed_group(self, tmp_path):
        """Probes the seed_bin dimension, as CODT's own reader does."""
        config = Case()
        plain = tmp_path / "plain.nc"
        config.aerosol.write(plain)
        assert inspect_input_file(plain) == ("CODT_aerosol_input_v1", False)

        config.aerosol.set_seed_group(
            seed_edge_radii=[500.0, 1000.0, 2000.0],
            seed_category=[7, 8],
            seed_bin_type=[2, 2],
            seed_frequency=[[0.5, 1.0], [0.25, 1.0]],
            seed_coord=[500.0, 900.0],
            seed_concentration=[10.0, 5.0],
            n_types=2,
        )
        seeded = tmp_path / "seeded.nc"
        config.aerosol.write(seeded)
        # Same conventions string either way — presence is the only signal.
        assert inspect_input_file(seeded) == ("CODT_aerosol_input_v1", True)

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
        rdir, _, _ = run_dir
        expected = sha256_file(rdir / "inputs" / "params.nml")
        assert by_type["namelist"]["checksum"] == expected

    def test_input_conventions_recorded(self, registry, run_dir):
        """NetCDF inputs record their schema; the namelist has none."""
        run_id = _register(registry, run_dir)
        rows = {
            r["file_type"]: r
            for r in registry._conn.execute(
                "SELECT * FROM input_files WHERE run_id = ?", (run_id,)
            )
        }
        aerosol = rows["aerosol_input"]
        assert aerosol["schema_conventions"] == "CODT_aerosol_input_v1"
        # Recorded absence (0), not "unknown" (NULL).
        assert aerosol["has_seed_group"] == 0
        assert rows["namelist"]["schema_conventions"] is None
        assert rows["namelist"]["has_seed_group"] is None

    def test_seeded_run_is_queryable(self, registry, run_dir):
        """A seeded run is identifiable despite sharing v1 with unseeded runs."""
        rdir, config, staged = run_dir
        config.aerosol.set_seed_group(
            seed_edge_radii=[500.0, 1000.0, 2000.0],
            seed_category=[7, 8],
            seed_bin_type=[2, 2],
            seed_frequency=[[0.5, 1.0], [0.25, 1.0]],
            seed_coord=[500.0, 900.0],
            seed_concentration=[10.0, 5.0],
            n_types=2,
        )
        config.aerosol.write(rdir / "inputs" / "aerosol_input.nc")
        config.params.set(do_seeding=True)
        run_id = _register(registry, (rdir, config, staged))

        seeded = registry._conn.execute(
            "SELECT run_id FROM input_files WHERE has_seed_group = 1"
        ).fetchall()
        assert [r["run_id"] for r in seeded] == [run_id]

    def test_do_seeding_without_seed_group_warns_at_registration(
        self, registry, run_dir
    ):
        """CODT aborts on this; surface it before a submit, not after."""
        rdir, config, staged = run_dir
        staged.set(do_seeding=True)
        with pytest.warns(UserWarning, match="no seed group"):
            _register(registry, (rdir, config, staged))

    def test_dormant_seed_group_registers_cleanly(self, registry, run_dir):
        """Seed group present but do_seeding off is valid — CODT ignores it.

        The group is still recorded (a run that could seed but didn't), and
        registration does not warn about the one-way gate.
        """
        rdir, config, staged = run_dir
        config.aerosol.set_seed_group(
            seed_edge_radii=[500.0, 1000.0, 2000.0],
            seed_category=[7, 8],
            seed_bin_type=[2, 2],
            seed_frequency=[[0.5, 1.0], [0.25, 1.0]],
            seed_coord=[500.0, 900.0],
            seed_concentration=[10.0, 5.0],
            n_types=2,
        )
        config.aerosol.write(rdir / "inputs" / "aerosol_input.nc")
        # do_seeding stays False (the run_dir default).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_id = _register(registry, (rdir, config, staged))
        row = registry._conn.execute(
            "SELECT has_seed_group FROM input_files WHERE run_id = ? AND "
            "file_type = 'aerosol_input'",
            (run_id,),
        ).fetchone()
        assert row["has_seed_group"] == 1

    def test_symlinked_input_recorded(self, registry, tmp_path, run_dir):
        rdir, config, staged = run_dir
        shared = rdir.parent.parent / "shared_inputs"
        shared.mkdir()
        real = rdir / "inputs" / "aerosol_input.nc"
        target = shared / "aerosol_input.nc"
        real.rename(target)
        real.symlink_to(Path("..") / ".." / ".." / "shared_inputs" / "aerosol_input.nc")
        run_id = _register(registry, (rdir, config, staged))
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
        rdir, config, staged = run_dir
        data_root = rdir.parent.parent  # tmp/exp
        registry.create_experiment("exp1", "One", data_root=data_root)
        run_id = _register(registry, (rdir, config, staged), experiment_id="exp1")
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

    def test_accepts_output_directory(self, registry, run_dir, tmp_path):
        """The sbatch wrapper passes the output dir; nc found via _DONE."""
        run_id = _register(registry, run_dir)
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with nc.Dataset(out_dir / "my_sim.nc", "w") as ds:
            ds.setncattr("conventions", "CODT_output_v1")
        (out_dir / "my_sim_DONE").touch()

        registry.record_completion(run_id, out_dir)
        assert registry.get_run(run_id)["conventions"] == "CODT_output_v1"

    def test_directory_without_marker_rejected(
        self, registry, run_dir, tmp_path
    ):
        run_id = _register(registry, run_dir)
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="_DONE"):
            registry.record_completion(run_id, out_dir)

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

class TestCLI:
    def _run(self, db, *argv) -> int:
        from codt_tools.registry.__main__ import main

        return main(["--db", str(db), *argv])

    def test_init_and_experiment_roundtrip(self, tmp_path, capsys):
        db = tmp_path / "r.db"
        assert self._run(db, "init") == 0
        assert self._run(db, "experiment", "create", "exp1", "Title") == 0
        assert self._run(db, "experiment", "list") == 0
        assert "exp1" in capsys.readouterr().out

    def test_update_status_and_show(self, registry, run_dir, capsys):
        run_id = _register(registry, run_dir)
        db = registry.db_path
        assert self._run(db, "update-status", run_id, "running", "--job-id", "42") == 0
        assert self._run(db, "show", run_id) == 0
        out = capsys.readouterr().out
        assert "running" in out and "42" in out

    def test_export_csv(self, registry, run_dir, tmp_path, capsys):
        run_id = _register(registry, run_dir)
        csv_path = tmp_path / "runs.csv"
        assert self._run(registry.db_path, "export", "--csv", str(csv_path)) == 0
        content = csv_path.read_text()
        assert run_id in content
        assert "parameters.simulation_name" in content

    def test_unknown_run_exits_nonzero(self, registry, capsys):
        assert self._run(registry.db_path, "show", "nope") == 1

    def test_missing_db_arg(self, monkeypatch, capsys):
        monkeypatch.delenv("CODT_REGISTRY_DB", raising=False)
        from codt_tools.registry.__main__ import main

        assert main(["list"]) == 2


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
