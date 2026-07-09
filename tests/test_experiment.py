"""Tests for codt_tools.experiment (ExperimentSpec, run creation)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from codt_tools.experiment import (
    ExperimentSpec,
    create_experiment_runs,
    dedup_shared_inputs,
)
from codt_tools.registry import Registry, sha256_file


@pytest.fixture
def spec(tmp_path: Path) -> ExperimentSpec:
    """A small chamber sweep spec rooted in a tmp dir."""
    return ExperimentSpec(
        experiment_id="EXP001",
        title="Tref sensitivity",
        data_root=tmp_path / "data",
        hypothesis="Warmer base temperature increases LWC.",
        permanent_data_root=tmp_path / "group",
        base_parameters={"tmax": 60.0},
        parameter_sweep={"tref": [20.0, 22.0], "volume_scaling": [13, 50]},
        execution_context="unit-test",
        slurm_options={
            "account": "owner-guest",
            "partition": "notchpeak-guest",
            "cores_per_node": 4,
            "walltime": "01:00:00",
        },
    )


@pytest.fixture
def registry(tmp_path: Path) -> Generator[Registry, None, None]:
    reg = Registry(tmp_path / "registry.db")
    yield reg
    reg.close()


@pytest.fixture
def codt_exe(tmp_path: Path) -> Path:
    """A stub CODT executable that reports a valid version."""
    exe = tmp_path / "codt"
    exe.write_text('#!/bin/bash\necho "CODT v1.0.0 (abc1234)"\n')
    exe.chmod(0o755)
    return exe


class TestSpecYaml:

    def test_roundtrip(self, spec: ExperimentSpec, tmp_path: Path) -> None:
        path = tmp_path / "experiment.yaml"
        spec.to_yaml(path)
        loaded = ExperimentSpec.from_yaml(path)

        assert loaded.experiment_id == "EXP001"
        assert loaded.base_parameters == {"tmax": 60.0}
        assert loaded.parameter_sweep == {
            "tref": [20.0, 22.0], "volume_scaling": [13, 50]
        }
        assert loaded.slurm_options["walltime"] == "01:00:00"
        assert Path(loaded.permanent_data_root).name == "group"

    def test_permanent_data_root_optional(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.yaml"
        path.write_text("experiment_id: X\ntitle: t\ndata_root: /tmp\n")
        assert ExperimentSpec.from_yaml(path).permanent_data_root is None

    def test_unknown_field_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "experiment_id: X\ntitle: t\ndata_root: /tmp\ntypo_field: 1\n"
        )
        with pytest.raises(ValueError, match="typo_field"):
            ExperimentSpec.from_yaml(path)


class TestExpand:

    def test_cartesian_product(self, spec: ExperimentSpec) -> None:
        configs = spec.expand()
        assert len(configs) == 4
        names = {cfg.name for cfg in configs}
        assert "EXP001_Tref20.0_VS13" in names

    def test_base_parameters_applied(self, spec: ExperimentSpec) -> None:
        for cfg in spec.expand():
            assert cfg.params.get("tmax") == 60.0

    def test_no_sweep_single_run(self, spec: ExperimentSpec) -> None:
        spec.parameter_sweep = {}
        configs = spec.expand()
        assert len(configs) == 1
        assert spec.descriptor(configs[0]) == "base"

    def test_descriptor(self, spec: ExperimentSpec) -> None:
        configs = spec.expand()
        assert spec.descriptor(configs[0]) == "Tref20.0_VS13"


class TestDedupSharedInputs:

    def _make_runs(self, tmp_path: Path, contents: list[bytes]) -> list[Path]:
        run_dirs = []
        for i, content in enumerate(contents):
            inputs = tmp_path / "runs" / f"run_{i}" / "inputs"
            inputs.mkdir(parents=True)
            (inputs / "params.nml").write_text("&PARAMETERS /\n")
            (inputs / "aerosol_input.nc").write_bytes(content)
            run_dirs.append(inputs.parent)
        return run_dirs

    def test_identical_files_symlinked(self, tmp_path: Path) -> None:
        run_dirs = self._make_runs(tmp_path, [b"same", b"same", b"same"])
        shared = tmp_path / "shared_inputs"

        deduped = dedup_shared_inputs(run_dirs, shared)

        assert len(deduped) == 1
        target = shared / "aerosol_input.nc"
        assert target.is_file() and not target.is_symlink()
        for rd in run_dirs:
            link = rd / "inputs" / "aerosol_input.nc"
            assert link.is_symlink()
            assert not Path(link.readlink()).is_absolute()
            assert link.read_bytes() == b"same"

    def test_unique_files_untouched(self, tmp_path: Path) -> None:
        run_dirs = self._make_runs(tmp_path, [b"one", b"two"])
        deduped = dedup_shared_inputs(run_dirs, tmp_path / "shared_inputs")

        assert deduped == {}
        assert not (tmp_path / "shared_inputs").exists()
        for rd in run_dirs:
            assert not (rd / "inputs" / "aerosol_input.nc").is_symlink()

    def test_namelist_never_deduped(self, tmp_path: Path) -> None:
        run_dirs = self._make_runs(tmp_path, [b"a", b"b"])
        dedup_shared_inputs(run_dirs, tmp_path / "shared_inputs")
        for rd in run_dirs:
            assert not (rd / "inputs" / "params.nml").is_symlink()

    def test_idempotent(self, tmp_path: Path) -> None:
        run_dirs = self._make_runs(tmp_path, [b"same", b"same"])
        shared = tmp_path / "shared_inputs"
        dedup_shared_inputs(run_dirs, shared)
        deduped = dedup_shared_inputs(run_dirs, shared)

        assert deduped == {}
        for rd in run_dirs:
            assert (rd / "inputs" / "aerosol_input.nc").read_bytes() == b"same"


class TestCreateExperimentRuns:

    def test_layout_and_registration(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path
    ) -> None:
        runner, run_dirs = create_experiment_runs(
            spec, registry, codt_exe
        )

        exp_dir = spec.experiment_dir
        assert (exp_dir / "experiment.yaml").is_file()
        assert len(run_dirs) == 4

        exp = registry.get_experiment("EXP001")
        assert exp["status"] == "planned"

        runs = registry.query_runs(experiment_id="EXP001")
        assert len(runs) == 4
        descriptors = {r["descriptor"] for r in runs}
        assert "Tref20.0_VS13" in descriptors
        for r in runs:
            assert r["status"] == "registered"
            assert r["execution_context"] == "unit-test"
            # run_id = {YYYYMMDD}_{HHMMSS}_{model}_{descriptor}
            assert r["run_id"].split("_", 2)[2] == f"codt_{r['descriptor']}"

    def test_shared_inputs_deduped(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path
    ) -> None:
        _, run_dirs = create_experiment_runs(
            spec, registry, codt_exe
        )

        shared = spec.experiment_dir / "shared_inputs"
        assert (shared / "aerosol_input.nc").is_file()
        for rd in run_dirs:
            link = rd / "inputs" / "aerosol_input.nc"
            assert link.is_symlink()

        # Registered input_files rows reflect the symlink layout and
        # checksum the resolved content.
        runs = registry.query_runs(experiment_id="EXP001")
        with registry._conn as conn:
            rows = conn.execute(
                "SELECT * FROM input_files WHERE run_id = ? "
                "AND file_type = 'aerosol_input'",
                (runs[0]["run_id"],),
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["is_symlink"] == 1
        assert rows[0]["link_target"].startswith("../../../shared_inputs/")
        assert rows[0]["checksum"] == sha256_file(shared / "aerosol_input.nc")

    def test_namelists_differ_per_run(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path
    ) -> None:
        _, run_dirs = create_experiment_runs(
            spec, registry, codt_exe
        )
        contents = {
            (rd / "inputs" / "params.nml").read_text() for rd in run_dirs
        }
        assert len(contents) == 4  # every sweep point has a distinct namelist

    def test_relocate_experiment(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path,
        tmp_path: Path
    ) -> None:
        """Scratch -> group workflow: move tree, verify, update registry."""
        import shutil

        create_experiment_runs(spec, registry, codt_exe)

        # Move the whole experiment tree (as rsync/mv would).
        new_root = Path(spec.permanent_data_root)
        new_root.mkdir()
        shutil.move(str(spec.experiment_dir), str(new_root))

        registry.relocate_experiment("EXP001", new_root)

        exp = registry.get_experiment("EXP001")
        assert exp["data_root"] == str(new_root.resolve())
        for run in registry.query_runs(experiment_id="EXP001"):
            assert run["data_status"] == "on_group"
            assert (new_root / run["run_dir"] / "inputs" / "params.nml").is_file()

    def test_relocate_missing_data_rejected(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path,
        tmp_path: Path
    ) -> None:
        create_experiment_runs(spec, registry, codt_exe)
        empty_root = tmp_path / "empty"
        empty_root.mkdir()

        with pytest.raises(FileNotFoundError):
            registry.relocate_experiment("EXP001", empty_root)
        # Registry unchanged on failure.
        exp = registry.get_experiment("EXP001")
        assert exp["data_root"] != str(empty_root.resolve())

    def test_relocate_checksum_mismatch_rejected(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path,
        tmp_path: Path
    ) -> None:
        import shutil

        _, run_dirs = create_experiment_runs(
            spec, registry, codt_exe
        )
        new_root = Path(spec.permanent_data_root)
        new_root.mkdir()
        shutil.move(str(spec.experiment_dir), str(new_root))

        # Corrupt one relocated shared input.
        shared = new_root / "EXP001" / "shared_inputs" / "aerosol_input.nc"
        shared.write_bytes(b"corrupted")

        with pytest.raises(ValueError, match="Checksum mismatch"):
            registry.relocate_experiment("EXP001", new_root)

    def test_relocate_via_cli(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path
    ) -> None:
        import shutil

        from codt_tools.registry.__main__ import main

        create_experiment_runs(spec, registry, codt_exe)
        new_root = Path(spec.permanent_data_root)
        new_root.mkdir()
        shutil.move(str(spec.experiment_dir), str(new_root))

        rc = main([
            "--db", str(registry.db_path),
            "relocate", "EXP001", str(new_root),
        ])
        assert rc == 0
        exp = registry.get_experiment("EXP001")
        assert exp["data_root"] == str(new_root.resolve())

    def test_missing_executable_rejected(
        self, spec: ExperimentSpec, registry: Registry
    ) -> None:
        with pytest.raises(FileNotFoundError, match="executable"):
            create_experiment_runs(spec, registry, "/nonexistent/codt")
        # Nothing was registered.
        with pytest.raises(KeyError):
            registry.get_experiment("EXP001")

    def test_bad_build_rejected(
        self, spec: ExperimentSpec, registry: Registry, tmp_path: Path
    ) -> None:
        """A binary without injected version info must be refused."""
        exe = tmp_path / "codt_bad"
        exe.write_text('#!/bin/bash\necho "CODT VERSION_PLACEHOLDER"\n')
        exe.chmod(0o755)

        with pytest.raises(ValueError, match="Rebuild CODT"):
            create_experiment_runs(spec, registry, exe)
        with pytest.raises(KeyError):
            registry.get_experiment("EXP001")

    def test_no_version_support_rejected(
        self, spec: ExperimentSpec, registry: Registry, tmp_path: Path
    ) -> None:
        exe = tmp_path / "codt_old"
        exe.write_text("#!/bin/bash\nexit 1\n")   # --version unsupported
        exe.chmod(0o755)

        with pytest.raises(ValueError, match="Rebuild CODT"):
            create_experiment_runs(spec, registry, exe)

    def test_permanent_data_root_recorded(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path
    ) -> None:
        create_experiment_runs(spec, registry, codt_exe)
        exp = registry.get_experiment("EXP001")
        assert exp["permanent_data_root"].endswith("group")

    def test_relocate_defaults_to_permanent_root(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path
    ) -> None:
        import shutil

        create_experiment_runs(spec, registry, codt_exe)
        new_root = Path(spec.permanent_data_root)
        new_root.mkdir()
        shutil.move(str(spec.experiment_dir), str(new_root))

        registry.relocate_experiment("EXP001")   # no new_root argument

        exp = registry.get_experiment("EXP001")
        assert exp["data_root"] == str(new_root.resolve())

    def test_relocate_no_root_anywhere_rejected(
        self, registry: Registry
    ) -> None:
        registry.create_experiment("bare", "no roots")
        with pytest.raises(ValueError, match="permanent_data_root"):
            registry.relocate_experiment("bare")

    def test_submit_uses_spec_walltime_and_marks_running(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path,
        monkeypatch,
    ) -> None:
        import subprocess as sp

        runner, run_dirs = create_experiment_runs(spec, registry, codt_exe)

        def fake_sbatch(cmd, **kwargs):
            class P:
                stdout = "Submitted batch job 999"
                returncode = 0
            return P()

        monkeypatch.setattr(sp, "run", fake_sbatch)
        runner.submit(run_dirs)   # no walltime argument

        script = (runner.base_output_dir / "CODT_batch_0.sh").read_text()
        assert "#SBATCH --time=01:00:00" in script   # from spec.slurm_options
        assert registry.get_experiment("EXP001")["status"] == "running"
        for run in registry.query_runs(experiment_id="EXP001"):
            assert run["status"] == "queued"

    def test_sbatch_records_completion_and_failure_detail(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path
    ) -> None:
        runner, run_dirs = create_experiment_runs(spec, registry, codt_exe)
        script = runner._generate_sbatch(run_dirs, "01:00:00")

        # Output metadata captured in-job after success.
        assert "complete" in script
        assert str(run_dirs[0] / "output") in script
        # Failure detail points at the logs.
        assert "--detail" in script and ".log" in script
        # SLURM job stdout lands somewhere known.
        assert "#SBATCH --output=" in script

    def test_end_to_end_dry_run(
        self, spec: ExperimentSpec, registry: Registry, codt_exe: Path
    ) -> None:
        """Plan verification: spec -> registered runs -> dry-run sbatch."""
        runner, run_dirs = create_experiment_runs(
            spec, registry, codt_exe
        )
        scripts = runner.submit(run_dirs, walltime="01:00:00", dry_run=True)

        # 4 runs / 4 cores_per_node = 1 batch script.
        assert len(scripts) == 1
        content = Path(scripts[0]).read_text()
        assert "#SBATCH --partition=notchpeak-guest" in content
        assert "codt-registry --db" in content
        for rd in run_dirs:
            assert str(rd / "inputs" / "params.nml") in content
        assert content.count("taskset") == 4
