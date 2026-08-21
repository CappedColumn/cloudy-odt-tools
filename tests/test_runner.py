"""Tests for CODTRunner."""

from __future__ import annotations

from pathlib import Path

import pytest

from codt_tools.config import CODTConfig
from codt_tools.runner import CODTRunner


@pytest.fixture(autouse=True)
def no_scheduler(monkeypatch):
    """Keep unit tests off the live scheduler.

    ``CODTRunner`` queries sinfo/mychpc at construction and preflights the
    target at submit. Both are stubbed out so tests are hermetic; the
    resolution and preflight logic is covered explicitly in
    ``TestArchResolution`` / ``TestPreflight`` and in ``test_slurm.py``.
    """
    from codt_tools import slurm as slurm_mod

    monkeypatch.setattr(slurm_mod, "detect_build_arch", lambda exe: None)
    monkeypatch.setattr(slurm_mod, "partition_cluster", lambda p: None)
    monkeypatch.setattr(
        slurm_mod, "cores_for_constraint",
        lambda p, c=None, f=None, nodes=None: None,
    )
    monkeypatch.setattr(
        slurm_mod, "validate_target",
        lambda *a, **k: [],
    )


@pytest.fixture
def runner(tmp_path: Path) -> CODTRunner:
    """Create a runner with temp paths."""
    return CODTRunner(
        executable="/usr/local/bin/codt",
        base_output_dir=tmp_path / "output",
        account="owner-guest",
        partition="notchpeak-guest",
        cores_per_node=4,
    )


@pytest.fixture
def config() -> CODTConfig:
    """Create a default config."""
    cfg = CODTConfig()
    cfg.set(simulation_name="test_sim")
    return cfg


class TestSetupRun:
    """Tests for setup_run and setup_runs."""

    def test_creates_directory_structure(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        sim_dir = runner.setup_run(config)

        assert sim_dir.is_dir()
        assert (sim_dir / "inputs" / "params.nml").is_file()
        assert (sim_dir / "inputs" / "aerosol_input.nc").is_file()
        assert (sim_dir / "output").is_dir()

    def test_sim_dir_path(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        sim_dir = runner.setup_run(config)

        expected = runner.base_output_dir / "test_sim"
        assert sim_dir == expected

    def test_auto_sets_output_directory(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        runner.setup_run(config)

        assert config.params.get("output_directory") == str(
            runner.base_output_dir / "test_sim" / "output"
        )

    def test_auto_sets_data_paths(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        runner.setup_run(config)

        assert config.params.get("aerosol_file") == "aerosol_input.nc"

    def test_setup_runs_batch(self, runner: CODTRunner) -> None:
        configs = []
        for i in range(3):
            cfg = CODTConfig()
            cfg.set(simulation_name=f"batch_{i}")
            configs.append(cfg)

        sim_dirs = runner.setup_runs(configs)

        assert len(sim_dirs) == 3
        for i, sim_dir in enumerate(sim_dirs):
            assert sim_dir.name == f"batch_{i}"
            assert (sim_dir / "inputs" / "params.nml").is_file()

    def test_namelist_content_readable(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        """Verify the written namelist can be read back."""
        sim_dir = runner.setup_run(config)

        reloaded = CODTConfig(sim_dir / "inputs" / "params.nml")
        assert reloaded.name == "test_sim"


class TestGenerateArraySbatch:
    """Tests for _generate_array_sbatch."""

    def test_basic_script(self, runner: CODTRunner) -> None:
        batches = [[Path("/scratch/sim_0"), Path("/scratch/sim_1")]]
        script = runner._generate_array_sbatch(
            batches, "12:00:00", "job1", Path("/scratch/job1.manifest")
        )

        assert "#!/bin/bash" in script
        assert "#SBATCH --account=owner-guest" in script
        assert "#SBATCH --partition=notchpeak-guest" in script
        assert "#SBATCH --nodes=1" in script
        assert "#SBATCH --ntasks=2" in script
        assert "#SBATCH --time=12:00:00" in script
        assert "#SBATCH --job-name=job1" in script
        assert "#SBATCH --array=0-0" in script
        assert '/scratch/job1.manifest' in script
        assert 'read -ra RUNS' in script
        assert script.strip().endswith("wait")

    def test_array_spec_and_throttle(self, runner: CODTRunner) -> None:
        batches = [[Path(f"/s/{i}")] for i in range(5)]
        script = runner._generate_array_sbatch(
            batches, "01:00:00", "j", Path("/m"), array_throttle=2
        )
        assert "#SBATCH --array=0-4%2" in script

        no_throttle = runner._generate_array_sbatch(
            batches, "01:00:00", "j", Path("/m")
        )
        assert "#SBATCH --array=0-4\n" in no_throttle

    def test_ntasks_is_largest_batch(self, runner: CODTRunner) -> None:
        batches = [[Path(f"/s/{i}") for i in range(4)], [Path("/s/x")]]
        script = runner._generate_array_sbatch(
            batches, "01:00:00", "j", Path("/m")
        )
        assert "#SBATCH --ntasks=4" in script

    def test_core_pinning_is_dynamic(self, runner: CODTRunner) -> None:
        """Cores are pinned by array index, not baked per run directory."""
        batches = [[Path("/scratch/sim_0")]]
        script = runner._generate_array_sbatch(
            batches, "01:00:00", "j", Path("/m")
        )
        assert 'taskset -c "$i"' in script
        assert 'for i in "${!RUNS[@]}"' in script

    def test_executable_in_script(self, runner: CODTRunner) -> None:
        script = runner._generate_array_sbatch(
            [[Path("/scratch/sim_0")]], "01:00:00", "j", Path("/m")
        )
        assert str(runner.executable) in script

    def test_qos_and_constraint_conditional(self, runner: CODTRunner) -> None:
        script = runner._generate_array_sbatch(
            [[Path("/s/0")]], "01:00:00", "j", Path("/m")
        )
        assert "--qos=" not in script
        assert "--constraint=" not in script
        assert "--mem=" not in script

        runner.qos = "notchpeak-freecycle"
        runner.constraint = "rom"
        runner.mem_per_task = "2G"
        script = runner._generate_array_sbatch(
            [[Path("/s/0"), Path("/s/1")]], "01:00:00", "j", Path("/m")
        )
        assert "#SBATCH --qos=notchpeak-freecycle" in script
        assert "#SBATCH --constraint=rom" in script
        # 2G per task x 2 tasks
        assert "#SBATCH --mem=4G" in script

    @pytest.mark.parametrize(
        "per_task,ntasks,expected",
        [("2G", 4, "8G"), ("512M", 2, "1024M"), ("4", 3, "12M"),
         ("weird", 4, "weird")],
    )
    def test_total_mem(
        self, runner: CODTRunner, per_task: str, ntasks: int, expected: str
    ) -> None:
        runner.mem_per_task = per_task
        assert runner._total_mem(ntasks) == expected


class TestArchResolution:
    """Auto-resolution of constraint / cores_per_node from the binary."""

    def _runner(self, tmp_path, monkeypatch, arch, cores=64, **kwargs):
        from codt_tools import slurm as slurm_mod

        monkeypatch.setattr(slurm_mod, "detect_build_arch", lambda exe: arch)
        monkeypatch.setattr(
            slurm_mod, "partition_cluster", lambda p: "notchpeak"
        )
        monkeypatch.setattr(
            slurm_mod, "cores_for_constraint",
            lambda p, c=None, f=None, nodes=None: cores,
        )
        return CODTRunner(
            executable="/usr/local/bin/codt",
            base_output_dir=tmp_path / "out",
            account="krueger",
            partition="notchpeak-freecycle",
            **kwargs,
        )

    def test_zen2_resolves_rom_constraint(self, tmp_path, monkeypatch) -> None:
        r = self._runner(tmp_path, monkeypatch, "zen2")
        assert r.build_arch == "zen2"
        assert r.constraint == "rom"
        assert r.cores_per_node == 64
        assert r.cluster == "notchpeak"

    def test_skylake_renders_or_constraint(self, tmp_path, monkeypatch) -> None:
        r = self._runner(tmp_path, monkeypatch, "skylake", cores=32)
        assert r.constraint == "skl|csl"
        assert r.cores_per_node == 32

    def test_baseline_arch_gets_no_constraint(
        self, tmp_path, monkeypatch
    ) -> None:
        r = self._runner(tmp_path, monkeypatch, "nehalem")
        assert r.constraint is None

    def test_unknown_arch_gets_no_constraint(
        self, tmp_path, monkeypatch
    ) -> None:
        r = self._runner(tmp_path, monkeypatch, None)
        assert r.constraint is None

    def test_explicit_args_win(self, tmp_path, monkeypatch) -> None:
        r = self._runner(
            tmp_path, monkeypatch, "zen2",
            constraint="gen", cores_per_node=96, cluster="granite",
        )
        assert r.constraint == "gen"
        assert r.cores_per_node == 96
        assert r.cluster == "granite"

    def test_falls_back_when_introspection_fails(
        self, tmp_path, monkeypatch
    ) -> None:
        """Off-cluster, packing keeps the historical default of 40."""
        r = self._runner(tmp_path, monkeypatch, "zen2", cores=None)
        assert r.cores_per_node == CODTRunner.DEFAULT_CORES_PER_NODE


class TestPreflight:
    """submit() refuses targets that cannot run the binary."""

    def test_raises_on_mismatch(
        self, runner: CODTRunner, config: CODTConfig, monkeypatch
    ) -> None:
        from codt_tools import slurm as slurm_mod

        monkeypatch.setattr(
            slurm_mod, "validate_target",
            lambda *a, **k: ["no rom nodes in this partition"],
        )
        sim_dir = runner.setup_run(config)
        with pytest.raises(ValueError, match="no rom nodes"):
            runner.submit([sim_dir], dry_run=True)

    def test_force_downgrades_to_warning(
        self, runner: CODTRunner, config: CODTConfig, monkeypatch
    ) -> None:
        from codt_tools import slurm as slurm_mod

        monkeypatch.setattr(
            slurm_mod, "validate_target",
            lambda *a, **k: ["no rom nodes in this partition"],
        )
        sim_dir = runner.setup_run(config)
        with pytest.warns(UserWarning, match="no rom nodes"):
            results = runner.submit([sim_dir], dry_run=True, force=True)
        assert Path(results[0]).is_file()


class TestSubmit:
    """Tests for submit (dry_run mode)."""

    def test_dry_run_writes_scripts(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        sim_dir = runner.setup_run(config)

        results = runner.submit([sim_dir], walltime="01:00:00", dry_run=True)

        assert len(results) == 1
        script_path = Path(results[0])
        assert script_path.is_file()
        assert script_path.parent.name == "slurm"
        content = script_path.read_text()
        assert "#!/bin/bash" in content
        # Run dirs live in the manifest, not the script.
        manifest = script_path.with_suffix(".manifest")
        assert manifest.is_file()
        assert str(sim_dir) in manifest.read_text()

    def test_empty_run_dirs(self, runner: CODTRunner) -> None:
        assert runner.submit([], dry_run=True) == []

    def test_batching(self, runner: CODTRunner) -> None:
        """With cores_per_node=4, 7 sims produce one array of 2 tasks."""
        configs = []
        for i in range(7):
            cfg = CODTConfig()
            cfg.set(simulation_name=f"batch_test_{i}")
            configs.append(cfg)

        sim_dirs = runner.setup_runs(configs)
        results = runner.submit(sim_dirs, dry_run=True)

        # One script for the whole ensemble, not one per batch.
        assert len(results) == 1
        script_path = Path(results[0])
        assert "#SBATCH --array=0-1" in script_path.read_text()

        lines = script_path.with_suffix(".manifest").read_text().splitlines()
        assert [len(line.split()) for line in lines] == [4, 3]
        # Every run appears exactly once.
        listed = [x for line in lines for x in line.split()]
        assert sorted(listed) == sorted(str(d) for d in sim_dirs)

    def test_batches_are_evenly_sized(self, runner: CODTRunner) -> None:
        """Runs spread evenly so no array task over-reserves cores.

        cores_per_node is a per-task capacity, not a fill target: 10 runs
        at 4/node is 3 tasks of 4/3/3, never 4/4/2.
        """
        assert [len(b) for b in runner._batch([Path(f"/s/{i}")
                                               for i in range(10)])] == [4, 3, 3]

    @pytest.mark.parametrize(
        "n,capacity,expected",
        [
            (200, 64, [50, 50, 50, 50]),   # the motivating case
            (7, 4, [4, 3]),
            (8, 4, [4, 4]),                # exact fit is unchanged
            (1, 4, [1]),
            (5, 64, [5]),                  # fewer runs than one node holds
        ],
    )
    def test_batch_sizes(
        self, runner: CODTRunner, n: int, capacity: int, expected: list[int]
    ) -> None:
        runner.cores_per_node = capacity
        batches = runner._batch([Path(f"/s/{i}") for i in range(n)])
        assert [len(b) for b in batches] == expected
        # No task exceeds what the node can host, and nothing is lost.
        assert max(len(b) for b in batches) <= capacity
        assert sum(len(b) for b in batches) == n

    def test_ntasks_matches_largest_batch(self, runner: CODTRunner) -> None:
        runner.cores_per_node = 64
        batches = runner._batch([Path(f"/s/{i}") for i in range(200)])
        script = runner._generate_array_sbatch(
            batches, "01:00:00", "j", Path("/m")
        )
        assert "#SBATCH --ntasks=50" in script
        assert "#SBATCH --array=0-3" in script

    def test_script_name_is_unique_per_submit(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        """Repeat submits must not overwrite each other's scripts."""
        sim_dir = runner.setup_run(config)
        first = runner.submit([sim_dir], dry_run=True)[0]
        runner.experiment_id = "exp2"
        second = runner.submit([sim_dir], dry_run=True)[0]
        assert first != second
        assert "exp2" in Path(second).name


class TestCollect:
    """Tests for collect() output directory resolution."""

    def test_explicit_output_dir(self, runner: CODTRunner, tmp_path: Path) -> None:
        from conftest import _create_main_nc

        out = tmp_path / "custom_output"
        out.mkdir()
        name = "explicit_sim"
        _create_main_nc(out, name=name)
        (out / f"{name}_DONE").write_text("2026-05-27\n")

        results = runner.collect([name], output_dir=out)
        assert len(results) == 1
        assert results[0].name == name

    def test_reads_namelist_for_output_dir(
        self, runner: CODTRunner, tmp_path: Path
    ) -> None:
        from conftest import _create_main_nc

        base = runner.base_output_dir
        name = "nml_sim"

        # Set up run directory with namelist pointing to custom output
        out = tmp_path / "real_output"
        out.mkdir(parents=True)
        cfg = CODTConfig()
        cfg.set(simulation_name=name, output_directory=str(out))
        inputs_dir = base / name / "inputs"
        inputs_dir.mkdir(parents=True)
        cfg.params.write(inputs_dir / "params.nml")

        # Put output in the custom directory
        _create_main_nc(out, name=name)
        (out / f"{name}_DONE").write_text("2026-05-27\n")

        results = runner.collect([name])
        assert len(results) == 1
        assert results[0].name == name

    def test_falls_back_to_base(self, runner: CODTRunner) -> None:
        from conftest import _create_main_nc

        base = runner.base_output_dir
        base.mkdir(parents=True, exist_ok=True)
        name = "fallback_sim"
        _create_main_nc(base, name=name)
        (base / f"{name}_DONE").write_text("2026-05-27\n")

        results = runner.collect([name])
        assert len(results) == 1
        assert results[0].name == name

    def test_skips_incomplete(self, runner: CODTRunner) -> None:
        from conftest import _create_main_nc

        base = runner.base_output_dir
        base.mkdir(parents=True, exist_ok=True)
        _create_main_nc(base, name="no_done")

        with pytest.warns(UserWarning, match="no DONE marker"):
            results = runner.collect(["no_done"])
        assert len(results) == 0


class TestRegistryHooks:
    """Runner-registry integration (and the registry=None no-op path)."""

    @pytest.fixture
    def reg_runner(self, tmp_path: Path):
        from codt_tools.registry import Registry

        registry = Registry(tmp_path / "registry.db")
        registry.create_experiment("exp1", "Test experiment")
        runner = CODTRunner(
            executable="/usr/local/bin/codt",
            base_output_dir=tmp_path / "output",
            account="owner-guest",
            partition="notchpeak-guest",
            cores_per_node=4,
            registry=registry,
            experiment_id="exp1",
        )
        yield runner, registry
        registry.close()

    def test_setup_run_registers(self, reg_runner, config: CODTConfig) -> None:
        runner, registry = reg_runner
        runner.setup_run(config, run_id="20260708_000000_codt_test")
        run = registry.get_run("20260708_000000_codt_test")
        assert run["status"] == "registered"
        assert run["experiment_id"] == "exp1"

    def test_setup_run_defaults_to_sim_name(
        self, reg_runner, config: CODTConfig
    ) -> None:
        runner, registry = reg_runner
        sim_dir = runner.setup_run(config)
        assert sim_dir.name == "test_sim"
        assert registry.get_run("test_sim")["status"] == "registered"

    def test_sbatch_contains_registry_lines(
        self, reg_runner, config: CODTConfig
    ) -> None:
        runner, registry = reg_runner
        sim_dir = runner.setup_run(config)
        script = runner._generate_array_sbatch(
            [[sim_dir]], "01:00:00", "j", Path("/m")
        )
        assert f"codt-registry --db {registry.db_path}" in script
        # Run names are derived at runtime from the manifest, so the script
        # is run-agnostic; assert the generic reporting shape instead.
        assert 'RUN_NAME="$(basename "$RUN_DIR")"' in script
        assert '$REGISTRY update-status "$RUN_NAME" running' in script
        assert '$REGISTRY update-status "$RUN_NAME" completed' in script
        assert '$REGISTRY update-status "$RUN_NAME" failed' in script
        assert '$REGISTRY complete "$RUN_NAME"' in script
        # The array task id is what gets recorded against each run.
        assert 'JOBID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in script
        assert "|| true" in script

    def test_sbatch_without_registry_has_no_hooks(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        sim_dir = runner.setup_run(config)
        script = runner._generate_array_sbatch(
            [[sim_dir]], "01:00:00", "j", Path("/m")
        )
        assert "codt-registry" not in script

    def test_collect_records_completion(self, reg_runner) -> None:
        from conftest import _create_main_nc

        runner, registry = reg_runner
        cfg = CODTConfig()
        cfg.set(simulation_name="col_sim")
        sim_dir = runner.setup_run(cfg)
        out = sim_dir / "output"
        _create_main_nc(out, name="col_sim")
        (out / "col_sim_DONE").write_text("2026-07-08\n")

        results = runner.collect(["col_sim"])
        assert len(results) == 1
        run = registry.get_run("col_sim")
        assert run["status"] == "collected"
        assert run["conventions"] == "CODT_output_v1"

    def test_collect_run_id_differs_from_sim_name(self, reg_runner) -> None:
        from conftest import _create_main_nc

        runner, registry = reg_runner
        cfg = CODTConfig()
        cfg.set(simulation_name="innername")
        sim_dir = runner.setup_run(cfg, run_id="20260708_000000_codt_x")
        out = sim_dir / "output"
        _create_main_nc(out, name="innername")
        (out / "innername_DONE").write_text("2026-07-08\n")

        results = runner.collect(["20260708_000000_codt_x"])
        assert len(results) == 1
        assert results[0].name == "innername"
        assert registry.get_run("20260708_000000_codt_x")["status"] == "collected"


class TestRepr:
    """Test string representation."""

    def test_repr(self, runner: CODTRunner) -> None:
        r = repr(runner)
        assert "CODTRunner" in r
        assert "owner-guest" in r
        assert "notchpeak-guest" in r
