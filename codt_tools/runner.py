"""CODT simulation runner: directory setup, local execution, and SLURM submission.

Run directory layout (one directory per run under ``base_output_dir``)::

    {base_output_dir}/{run_name}/
        inputs/
            params.nml
            aerosol_input.nc
            [parcel_input.nc]
        output/
            {sim_name}.nc, {sim_name}.log, {sim_name}_DONE, ...

When a :class:`codt_tools.registry.Registry` is attached, every run is
registered at setup time and its status is updated through the lifecycle
(locally via the API, on SLURM via ``codt-registry`` CLI calls embedded in
the batch script). Without a registry, behavior is unchanged.

Example usage::

    from codt_tools import CODTConfig, CODTRunner
    from codt_tools.registry import Registry

    runner = CODTRunner(
        executable="~/simulations/CODT/bin/CODT_exec",
        base_output_dir="/scratch/CODT_output",
        account="owner-guest",
        partition="notchpeak-guest",
        registry=Registry("~/codt_registry.db"),   # optional
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

import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Union

from codt_tools.config import CODTConfig

if TYPE_CHECKING:
    from codt_tools.registry import Registry


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
    registry : Registry, optional
        Simulation registry. When given, runs are registered at setup and
        status transitions are tracked. All registry hooks are no-ops when
        this is None.
    experiment_id : str, optional
        Experiment to attach registered runs to (requires ``registry``).
    """

    def __init__(
        self,
        executable: Union[str, Path],
        base_output_dir: Union[str, Path],
        account: str,
        partition: str,
        cores_per_node: int = 40,
        registry: "Registry | None" = None,
        experiment_id: str | None = None,
    ) -> None:
        self.executable: Path = Path(executable).expanduser().resolve()
        self.base_output_dir: Path = Path(base_output_dir).expanduser().resolve()
        self.account: str = account
        self.partition: str = partition
        self.cores_per_node: int = cores_per_node
        self.registry = registry
        self.experiment_id = experiment_id
        self.default_walltime: str | None = None
        self.codt_version: str | None = self._query_version()

    # ------------------------------------------------------------------
    # Version
    # ------------------------------------------------------------------

    def _query_version(self) -> str | None:
        """Run ``codt --version`` and return the version string, or None."""
        if not self.executable.is_file():
            return None
        try:
            proc = subprocess.run(
                [str(self.executable), "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------

    def setup_run(
        self,
        config: CODTConfig,
        run_id: str | None = None,
        register: bool = True,
    ) -> Path:
        """Create a run directory with input files and an output directory.

        Directory structure::

            {base_output_dir}/{run_name}/
                inputs/   (params.nml, aerosol_input.nc, ...)
                output/   (created empty; model writes here)

        ``run_name`` is *run_id* if given, else the simulation name. The
        namelist ``output_directory`` is set to the absolute ``output/``
        path. With a registry attached, the run is registered under
        ``run_name`` with status ``registered``.

        Parameters
        ----------
        config : CODTConfig
            Simulation configuration.
        run_id : str, optional
            Registry run identifier and directory name
            (e.g. ``20260708_143022_codt_v2.1_control``).
        register : bool, optional
            If ``False``, skip registry registration even when a registry
            is attached. Used when input files will be post-processed
            (e.g. shared-input dedup) before registration.

        Returns
        -------
        Path
            The run directory (``{base_output_dir}/{run_name}``).
        """
        run_name = run_id if run_id is not None else config.name
        sim_dir = self.base_output_dir / run_name
        inputs_dir = sim_dir / "inputs"
        output_dir = sim_dir / "output"

        output_dir.mkdir(parents=True, exist_ok=True)
        config.params.set(output_directory=str(output_dir))
        config.write(inputs_dir)

        if register and self.registry is not None:
            self.registry.register_run(
                run_name,
                config,
                sim_dir,
                experiment_id=self.experiment_id,
                executable_path=self.executable,
                code_version=self.codt_version,
            )
        return sim_dir

    def setup_runs(
        self,
        configs: list[CODTConfig],
        run_ids: list[str] | None = None,
    ) -> list[Path]:
        """Create run directories for multiple simulations.

        Parameters
        ----------
        configs : list[CODTConfig]
            List of simulation configurations.
        run_ids : list[str], optional
            Registry run identifiers, one per config.

        Returns
        -------
        list[Path]
            Run directories, one per config.
        """
        if run_ids is None:
            return [self.setup_run(cfg) for cfg in configs]
        if len(run_ids) != len(configs):
            raise ValueError("run_ids and configs must have the same length")
        return [
            self.setup_run(cfg, run_id=rid) for cfg, rid in zip(configs, run_ids)
        ]

    # ------------------------------------------------------------------
    # Local execution
    # ------------------------------------------------------------------

    def run_local(
        self,
        config: CODTConfig,
        run_id: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Set up and run a single simulation locally (blocking).

        The CODT model handles its own stdout redirection to a log file,
        so no output capture is performed here. With a registry attached,
        status transitions (running -> completed/failed) and output
        version metadata are recorded.

        Parameters
        ----------
        config : CODTConfig
            Simulation configuration.
        run_id : str, optional
            Registry run identifier (defaults to the simulation name).

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

        sim_dir = self.setup_run(config, run_id=run_id)
        run_name = sim_dir.name
        nml_path = sim_dir / "inputs" / "params.nml"

        if self.registry is not None:
            self.registry.update_status(run_name, "running")

        proc = subprocess.run(
            [str(self.executable), str(nml_path)],
            check=False,
        )

        if self.registry is not None:
            status = "completed" if proc.returncode == 0 else "failed"
            self.registry.update_status(
                run_name, status, exit_code=proc.returncode
            )
            output_nc = sim_dir / "output" / f"{config.name}.nc"
            if proc.returncode == 0 and output_nc.is_file():
                self.registry.record_completion(run_name, output_nc)
        return proc

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

        Each simulation is pinned to a specific core via ``taskset``. With
        a registry attached, each simulation is wrapped in a subshell that
        reports running/completed/failed status through the
        ``codt-registry`` CLI; registry failures never kill the run
        (``|| true``).

        Parameters
        ----------
        run_dirs : list[Path]
            Run directories (each containing ``inputs/params.nml``).
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
            f"#SBATCH --output={self.base_output_dir}/CODT_batch_{batch_id}_%j.out",
            "",
        ]

        if self.registry is None:
            for i, sim_dir in enumerate(run_dirs):
                nml_path = sim_dir / "inputs" / "params.nml"
                lines.append(
                    f"taskset -c {i} {self.executable} {nml_path} &"
                )
        else:
            db = self.registry.db_path
            # Absolute CLI path so status reporting works regardless of
            # the job's PATH/environment; fall back to bare name.
            cli = shutil.which("codt-registry") or "codt-registry"
            lines.append(f'REGISTRY="{cli} --db {db}"')
            lines.append("")
            for i, sim_dir in enumerate(run_dirs):
                run_name = sim_dir.name
                nml_path = sim_dir / "inputs" / "params.nml"
                output_dir = sim_dir / "output"
                lines.extend([
                    "(",
                    f'  $REGISTRY update-status {run_name} running '
                    f'--job-id "$SLURM_JOB_ID" || true',
                    f"  taskset -c {i} {self.executable} {nml_path}",
                    "  rc=$?",
                    f"  if [ $rc -eq 0 ]; then",
                    f'    $REGISTRY update-status {run_name} completed '
                    f'--exit-code $rc --job-id "$SLURM_JOB_ID" || true',
                    f"    $REGISTRY complete {run_name} {output_dir} || true",
                    f"  else",
                    f'    $REGISTRY update-status {run_name} failed '
                    f'--exit-code $rc --job-id "$SLURM_JOB_ID" '
                    f'--detail "exit $rc; logs: {output_dir}/*.log and '
                    f'{self.base_output_dir}/CODT_batch_{batch_id}_$SLURM_JOB_ID.out" '
                    f"|| true",
                    f"  fi",
                    ") &",
                ])

        lines.append("wait")
        return "\n".join(lines) + "\n"

    def submit(
        self,
        run_dirs: list[Path],
        walltime: str | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Submit simulations to SLURM (or generate scripts only).

        Simulations are batched into groups of ``cores_per_node``, each
        group becoming one SLURM job. With a registry attached, runs are
        marked ``queued`` (with their job ID) on successful submission.

        Parameters
        ----------
        run_dirs : list[Path]
            Run directories (from :meth:`setup_run`).
        walltime : str, optional
            SLURM wall-clock time. Defaults to ``self.default_walltime``
            (set from the experiment spec's ``slurm_options`` by
            ``create_experiment_runs``), else ``"24:00:00"``.
        dry_run : bool, optional
            If ``True``, write batch scripts but do not submit.
            Returns script file paths instead of job IDs.

        Returns
        -------
        list[str]
            SLURM job IDs (if ``dry_run=False``) or script file paths
            (if ``dry_run=True``).
        """
        if walltime is None:
            walltime = self.default_walltime or "24:00:00"

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

                if self.registry is not None:
                    for sim_dir in batch:
                        self.registry.update_status(
                            sim_dir.name, "queued", slurm_job_id=job_id
                        )

        if (
            not dry_run
            and results
            and self.registry is not None
            and self.experiment_id is not None
        ):
            self.registry.update_experiment(
                self.experiment_id, status="running"
            )

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

    def _resolve_output_dir(self, name: str) -> Path:
        """Find the output directory for a run.

        Checks, in order: ``{base}/{name}/output/`` (standard layout),
        the ``output_directory`` value in ``{base}/{name}/inputs/params.nml``,
        and finally ``base_output_dir``.
        """
        output_dir = self.base_output_dir / name / "output"
        if output_dir.is_dir():
            return output_dir
        nml_path = self.base_output_dir / name / "inputs" / "params.nml"
        if nml_path.is_file():
            from codt_tools.config import Namelist
            nml = Namelist(nml_path)
            out = Path(nml.get("output_directory"))
            if not out.is_absolute():
                out = (nml_path.parent / out).resolve()
            return out
        return self.base_output_dir

    def collect(
        self,
        run_names: list[str],
        output_dir: Union[str, Path, None] = None,
    ) -> list:
        """Load completed simulations as CODTSimulation objects.

        For each run, the output directory is resolved in order:

        1. *output_dir* argument (if given).
        2. ``{base}/{run_name}/output/`` (standard layout).
        3. ``output_directory`` from ``{base}/{run_name}/inputs/params.nml``.
        4. ``base_output_dir`` as a last resort.

        With a registry attached, collected runs get their output version
        metadata recorded and status set to ``collected``.

        Parameters
        ----------
        run_names : list[str]
            Run directory names to collect. For runs where the run name
            differs from the namelist ``simulation_name``, output files
            are found by globbing the ``_DONE`` marker.
        output_dir : str or Path, optional
            Explicit output directory. Overrides resolution.

        Returns
        -------
        list[CODTSimulation]
            One object per completed simulation.  Simulations without
            a ``{name}_DONE`` marker are skipped with a warning.
        """
        from codt_tools.simulation import CODTSimulation

        explicit_dir = Path(output_dir) if output_dir is not None else None

        results = []
        for name in run_names:
            out = explicit_dir if explicit_dir is not None else self._resolve_output_dir(name)
            done_markers = sorted(out.glob("*_DONE"))
            done_marker = out / f"{name}_DONE"
            if not done_marker.is_file() and len(done_markers) == 1:
                # Run directory name differs from simulation_name
                done_marker = done_markers[0]
            if not done_marker.is_file():
                warnings.warn(
                    f"Simulation '{name}' has no DONE marker at "
                    f"{done_marker} — skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            sim_name = done_marker.name.removesuffix("_DONE")
            nc_path = out / f"{sim_name}.nc"
            results.append(CODTSimulation(nc_path))

            if self.registry is not None:
                self.registry.record_completion(name, nc_path)
                self.registry.update_status(name, "collected")

        return results

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        version = f", version='{self.codt_version}'" if self.codt_version else ""
        return (
            f"CODTRunner(executable='{self.executable}', "
            f"base_output_dir='{self.base_output_dir}', "
            f"account='{self.account}', partition='{self.partition}'"
            f"{version})"
        )
