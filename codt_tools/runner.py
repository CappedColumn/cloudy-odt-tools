"""CODT simulation runner: directory setup, local execution, and SLURM submission.

Example usage::

    from codt_tools import CODTConfig, CODTRunner

    runner = CODTRunner(
        executable="~/simulations/CODT/bin/CODT_exec",
        base_output_dir="/scratch/CODT_output",
        account="owner-guest",
        partition="notchpeak-guest",
    )

    cfg = CODTConfig()
    cfg.set(simulation_name="test_run", tref=22.0, tmax=3600.0)

    # Local (blocking) execution
    result = runner.run_local(cfg)

    # Or SLURM batch submission
    sim_dir = runner.setup_run(cfg)
    job_ids = runner.submit([sim_dir], walltime="04:00:00")
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Union

from codt_tools.config import CODTConfig


class CODTRunner:
    """Set up and run CODT simulations locally or on a SLURM cluster.

    Parameters
    ----------
    executable : str or Path
        Path to the compiled CODT binary.
    base_output_dir : str or Path
        Root directory for all simulation runs and output.
    account : str
        SLURM account string (e.g. ``"owner-guest"``).
    partition : str
        SLURM partition name (e.g. ``"notchpeak-guest"``).
    cores_per_node : int, optional
        Maximum number of simulations to pack per SLURM node (default 40).
    """

    def __init__(
        self,
        executable: Union[str, Path],
        base_output_dir: Union[str, Path],
        account: str,
        partition: str,
        cores_per_node: int = 40,
    ) -> None:
        self.executable: Path = Path(executable).expanduser().resolve()
        self.base_output_dir: Path = Path(base_output_dir).expanduser().resolve()
        self.account: str = account
        self.partition: str = partition
        self.cores_per_node: int = cores_per_node

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------

    def setup_run(self, config: CODTConfig) -> Path:
        """Create a run directory with all input files.

        Directory structure::

            {base_output_dir}/{simulation_name}/
                run/
                    params.nml
                    aerosol_input.nc
                    bin_data.txt

        The namelist ``output_directory`` is set to ``base_output_dir``
        (absolute) so model output lands in
        ``{base_output_dir}/{simulation_name}/``.

        Parameters
        ----------
        config : CODTConfig
            Simulation configuration.

        Returns
        -------
        Path
            The simulation directory (``{base_output_dir}/{simulation_name}``).
        """
        sim_name = config.name
        sim_dir = self.base_output_dir / sim_name
        run_dir = sim_dir / "run"

        config.params.set(output_directory=str(self.base_output_dir))
        config.write(run_dir)

        return sim_dir

    def setup_runs(self, configs: list[CODTConfig]) -> list[Path]:
        """Create run directories for multiple simulations.

        Parameters
        ----------
        configs : list[CODTConfig]
            List of simulation configurations.

        Returns
        -------
        list[Path]
            Simulation directories, one per config.
        """
        return [self.setup_run(cfg) for cfg in configs]

    # ------------------------------------------------------------------
    # Local execution
    # ------------------------------------------------------------------

    def run_local(self, config: CODTConfig) -> subprocess.CompletedProcess:
        """Set up and run a single simulation locally (blocking).

        The CODT model handles its own stdout redirection to a log file,
        so no output capture is performed here.

        Parameters
        ----------
        config : CODTConfig
            Simulation configuration.

        Returns
        -------
        subprocess.CompletedProcess
            The completed process.  Check ``.returncode`` for success (0)
            or failure (1).

        Raises
        ------
        FileNotFoundError
            If the executable does not exist.
        """
        if not self.executable.is_file():
            raise FileNotFoundError(
                f"Executable not found: {self.executable}"
            )

        sim_dir = self.setup_run(config)
        nml_path = sim_dir / "run" / "params.nml"

        return subprocess.run(
            [str(self.executable), str(nml_path)],
            check=False,
        )

    # ------------------------------------------------------------------
    # SLURM submission
    # ------------------------------------------------------------------

    def _generate_sbatch(
        self,
        run_dirs: list[Path],
        walltime: str,
        batch_id: int = 0,
    ) -> str:
        """Generate a SLURM batch script for a set of simulations.

        Each simulation is pinned to a specific core via ``taskset``.

        Parameters
        ----------
        run_dirs : list[Path]
            Simulation directories (each containing ``run/params.nml``).
        walltime : str
            SLURM wall-clock time (e.g. ``"24:00:00"``).
        batch_id : int, optional
            Batch index for the job name (default 0).

        Returns
        -------
        str
            The batch script content.
        """
        lines = [
            "#!/bin/bash",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --partition={self.partition}",
            "#SBATCH --nodes=1",
            f"#SBATCH --ntasks={len(run_dirs)}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --job-name=CODT_{batch_id}",
            "",
        ]

        for i, sim_dir in enumerate(run_dirs):
            nml_path = sim_dir / "run" / "params.nml"
            lines.append(
                f"taskset -c {i} {self.executable} {nml_path} &"
            )

        lines.append("wait")
        return "\n".join(lines) + "\n"

    def submit(
        self,
        run_dirs: list[Path],
        walltime: str = "24:00:00",
        dry_run: bool = False,
    ) -> list[str]:
        """Submit simulations to SLURM (or generate scripts only).

        Simulations are batched into groups of ``cores_per_node``, each
        group becoming one SLURM job.

        Parameters
        ----------
        run_dirs : list[Path]
            Simulation directories (from :meth:`setup_run`).
        walltime : str, optional
            SLURM wall-clock time (default ``"24:00:00"``).
        dry_run : bool, optional
            If ``True``, write batch scripts but do not submit.
            Returns script file paths instead of job IDs.

        Returns
        -------
        list[str]
            SLURM job IDs (if ``dry_run=False``) or script file paths
            (if ``dry_run=True``).
        """
        # Split into batches
        batches: list[list[Path]] = []
        for i in range(0, len(run_dirs), self.cores_per_node):
            batches.append(run_dirs[i : i + self.cores_per_node])

        results: list[str] = []
        for batch_id, batch in enumerate(batches):
            script = self._generate_sbatch(batch, walltime, batch_id)
            script_path = (
                self.base_output_dir / f"CODT_batch_{batch_id}.sh"
            )
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script)

            if dry_run:
                results.append(str(script_path))
            else:
                if not self.executable.is_file():
                    raise FileNotFoundError(
                        f"Executable not found: {self.executable}"
                    )
                proc = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # Parse "Submitted batch job 12345"
                job_id = proc.stdout.strip().split()[-1]
                results.append(job_id)

        return results

    # ------------------------------------------------------------------
    # Status and collection
    # ------------------------------------------------------------------

    def status(self, job_ids: list[str]) -> dict[str, str]:
        """Query SLURM for job status.

        Parameters
        ----------
        job_ids : list[str]
            SLURM job IDs to query.

        Returns
        -------
        dict[str, str]
            Mapping of job ID to state string (e.g. ``"RUNNING"``,
            ``"PENDING"``, ``"COMPLETED"``).
        """
        id_str = ",".join(job_ids)
        proc = subprocess.run(
            ["squeue", f"--jobs={id_str}", "--noheader", "--format=%i %T"],
            capture_output=True,
            text=True,
            check=False,
        )

        active: dict[str, str] = {}
        for line in proc.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                active[parts[0]] = parts[1]

        # Jobs not in squeue are assumed completed
        return {
            jid: active.get(jid, "COMPLETED") for jid in job_ids
        }

    def collect(self, sim_names: list[str]) -> list:
        """Load completed simulations as CODTSimulation objects.

        Parameters
        ----------
        sim_names : list[str]
            Simulation names to collect.

        Returns
        -------
        list[CODTSimulation]
            One object per completed simulation.  Simulations without
            a ``DONE`` marker are skipped with a warning.
        """
        from codt_tools.simulation import CODTSimulation

        results = []
        for name in sim_names:
            sim_dir = self.base_output_dir / name
            done_marker = sim_dir / "DONE"
            if not done_marker.is_file():
                warnings.warn(
                    f"Simulation '{name}' has no DONE marker at "
                    f"{done_marker} — skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            results.append(CODTSimulation(sim_dir))

        return results

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CODTRunner(executable='{self.executable}', "
            f"base_output_dir='{self.base_output_dir}', "
            f"account='{self.account}', partition='{self.partition}')"
        )
