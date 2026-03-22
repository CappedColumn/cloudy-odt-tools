"""Tests for CODTRunner."""

from __future__ import annotations

from pathlib import Path

import pytest

from codt_tools.config import CODTConfig
from codt_tools.runner import CODTRunner


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
        assert (sim_dir / "run" / "params.nml").is_file()
        assert (sim_dir / "run" / "aerosol_input.nc").is_file()
        assert (sim_dir / "run" / "bin_data.txt").is_file()

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
            runner.base_output_dir
        )

    def test_auto_sets_data_paths(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        runner.setup_run(config)

        assert config.params.get("aerosol_file") == "aerosol_input.nc"
        assert config.params.get("bin_data_file") == "bin_data.txt"

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
            assert (sim_dir / "run" / "params.nml").is_file()

    def test_namelist_content_readable(
        self, runner: CODTRunner, config: CODTConfig
    ) -> None:
        """Verify the written namelist can be read back."""
        sim_dir = runner.setup_run(config)

        reloaded = CODTConfig(sim_dir / "run" / "params.nml")
        assert reloaded.name == "test_sim"


class TestGenerateSbatch:
    """Tests for _generate_sbatch."""

    def test_basic_script(self, runner: CODTRunner) -> None:
        run_dirs = [Path("/scratch/sim_0"), Path("/scratch/sim_1")]
        script = runner._generate_sbatch(run_dirs, "12:00:00", batch_id=0)

        assert "#!/bin/bash" in script
        assert "#SBATCH --account=owner-guest" in script
        assert "#SBATCH --partition=notchpeak-guest" in script
        assert "#SBATCH --nodes=1" in script
        assert "#SBATCH --ntasks=2" in script
        assert "#SBATCH --time=12:00:00" in script
        assert "#SBATCH --job-name=CODT_0" in script
        assert "taskset -c 0" in script
        assert "taskset -c 1" in script
        assert script.strip().endswith("wait")

    def test_core_pinning(self, runner: CODTRunner) -> None:
        run_dirs = [Path(f"/scratch/sim_{i}") for i in range(4)]
        script = runner._generate_sbatch(run_dirs, "24:00:00")

        for i in range(4):
            assert f"taskset -c {i}" in script

    def test_executable_in_script(self, runner: CODTRunner) -> None:
        run_dirs = [Path("/scratch/sim_0")]
        script = runner._generate_sbatch(run_dirs, "01:00:00")

        assert str(runner.executable) in script

    def test_namelist_path_in_script(self, runner: CODTRunner) -> None:
        run_dirs = [Path("/scratch/sim_0")]
        script = runner._generate_sbatch(run_dirs, "01:00:00")

        assert "/scratch/sim_0/run/params.nml" in script


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
        content = script_path.read_text()
        assert "#!/bin/bash" in content
        assert str(sim_dir) in content

    def test_batching(self, runner: CODTRunner) -> None:
        """With cores_per_node=4, 7 sims should produce 2 batches."""
        configs = []
        for i in range(7):
            cfg = CODTConfig()
            cfg.set(simulation_name=f"batch_test_{i}")
            configs.append(cfg)

        sim_dirs = runner.setup_runs(configs)
        results = runner.submit(sim_dirs, dry_run=True)

        assert len(results) == 2  # ceil(7/4) = 2

        # First batch should have 4 sims, second should have 3
        script_0 = Path(results[0]).read_text()
        script_1 = Path(results[1]).read_text()
        assert script_0.count("taskset") == 4
        assert script_1.count("taskset") == 3


class TestRepr:
    """Test string representation."""

    def test_repr(self, runner: CODTRunner) -> None:
        r = repr(runner)
        assert "CODTRunner" in r
        assert "owner-guest" in r
        assert "notchpeak-guest" in r
