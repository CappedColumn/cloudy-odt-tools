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
        self, spec: ExperimentSpec, registry: Registry
    ) -> None:
        runner, run_dirs = create_experiment_runs(
            spec, registry, "/usr/local/bin/codt"
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
        self, spec: ExperimentSpec, registry: Registry
    ) -> None:
        _, run_dirs = create_experiment_runs(
            spec, registry, "/usr/local/bin/codt"
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
        self, spec: ExperimentSpec, registry: Registry
    ) -> None:
        _, run_dirs = create_experiment_runs(
            spec, registry, "/usr/local/bin/codt"
        )
        contents = {
            (rd / "inputs" / "params.nml").read_text() for rd in run_dirs
        }
        assert len(contents) == 4  # every sweep point has a distinct namelist

    def test_end_to_end_dry_run(
        self, spec: ExperimentSpec, registry: Registry
    ) -> None:
        """Plan verification: spec -> registered runs -> dry-run sbatch."""
        runner, run_dirs = create_experiment_runs(
            spec, registry, "/usr/local/bin/codt"
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
