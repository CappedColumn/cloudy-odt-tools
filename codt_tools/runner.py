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

import math
import re
import shutil
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from codt_tools import slurm as slurm_mod
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
    qos : str, optional
        SLURM QOS. Required where it differs from the partition name (on
        granite, partition ``granite`` uses qos ``granite-freecycle``).
    cores_per_node : int, optional
        Simulations to pack per SLURM node. ``None`` (the default) resolves
        it from the real core count of the nodes the job will land on; pass
        an int to override.
    constraint : str, optional
        SLURM node feature constraint. ``None`` (the default) resolves it
        from the executable's build architecture; pass a string to override.
    mem_per_task : str, optional
        Memory per packed simulation (e.g. ``"2G"``). When given, an
        explicit ``#SBATCH --mem`` of ``mem_per_task * ntasks`` is emitted.
        Node sharing is the CHPC default, so leaving this unset silently
        takes 2G/core.
    cluster : str, optional
        Cluster the partition lives on. ``None`` resolves it from ``sinfo``;
        used to pass ``-M`` to sbatch/squeue/sacct.
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
        cores_per_node: int | None = None,
        registry: "Registry | None" = None,
        experiment_id: str | None = None,
        qos: str | None = None,
        constraint: str | None = None,
        mem_per_task: str | None = None,
        cluster: str | None = None,
    ) -> None:
        self.executable: Path = Path(executable).expanduser().resolve()
        self.base_output_dir: Path = Path(base_output_dir).expanduser().resolve()
        self.account: str = account
        self.partition: str = partition
        self.qos: str | None = qos
        self.cores_per_node: int | None = cores_per_node
        self.constraint: str | None = constraint
        self.mem_per_task: str | None = mem_per_task
        self.cluster: str | None = cluster
        self.registry = registry
        self.experiment_id = experiment_id
        self.default_walltime: str | None = None
        self.default_array_throttle: int | None = None
        self.codt_version: str | None = self._query_version()
        self.build_arch: str | None = slurm_mod.detect_build_arch(
            self.executable
        )
        self._features: tuple[str, ...] = ()
        self._resolve_slurm_target()

    # ------------------------------------------------------------------
    # SLURM target resolution
    # ------------------------------------------------------------------

    DEFAULT_CORES_PER_NODE: int = 40

    def _resolve_slurm_target(self) -> None:
        """Fill in constraint, cluster and packing density where unset.

        Explicitly passed values always win. Anything that cannot be
        resolved (off-cluster, no ``sinfo``, unknown architecture) falls
        back to the historical behavior: no constraint, 40 tasks per node.
        """
        if self.constraint is None:
            features = (
                slurm_mod.ARCH_CONSTRAINT.get(self.build_arch)
                if self.build_arch is not None else None
            )
            if features:
                self._features = tuple(features)
                self.constraint = slurm_mod.render_constraint(features)
        else:
            # An explicit constraint may be an OR expression ("skl|csl").
            self._features = tuple(
                f.strip() for f in self.constraint.split("|") if f.strip()
            )

        if self.cluster is None:
            self.cluster = slurm_mod.partition_cluster(self.partition)

        if self.cores_per_node is None:
            resolved = slurm_mod.cores_for_constraint(
                self.partition, self.cluster, self._features or None
            )
            self.cores_per_node = (
                resolved if resolved else self.DEFAULT_CORES_PER_NODE
            )

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
                build_arch=self.build_arch,
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

    def _batch(self, run_dirs: list[Path]) -> list[list[Path]]:
        """Split run directories into evenly sized per-node batches.

        The batch *count* is set by ``cores_per_node`` (the most runs a node
        can host), but the runs are then spread evenly across those batches
        rather than filling each to capacity and leaving a small remainder.

        This matters because one array header covers every task: ``--ntasks``
        is sized to the largest batch, so a filled-then-remainder split makes
        the final task reserve a full node's worth of cores to run a handful
        of simulations. Evening the batches out keeps the reservation honest
        without changing the number of tasks.

        200 runs at 64 cores/node give ``[50, 50, 50, 50]`` rather than
        ``[64, 64, 64, 8]`` — still four tasks, but ``--ntasks`` drops from
        64 to 50 and nothing is over-reserved.

        Wall time is unaffected (a task's simulations all run in parallel),
        and the smaller request schedules better under node sharing: a task
        needing all 64 cores of a node can only start on an empty one, while
        50 can share with another user's small job.
        """
        capacity = self.cores_per_node or self.DEFAULT_CORES_PER_NODE
        n_batches = math.ceil(len(run_dirs) / capacity)
        base, remainder = divmod(len(run_dirs), n_batches)

        batches: list[list[Path]] = []
        start = 0
        for i in range(n_batches):
            # The first `remainder` batches take one extra run.
            size = base + 1 if i < remainder else base
            batches.append(run_dirs[start : start + size])
            start += size
        return batches

    def _task_body(self) -> list[str]:
        """Shell lines running one array task's simulations in parallel.

        Each simulation is pinned to its own core with ``taskset``. With a
        registry attached, each is wrapped in a subshell reporting
        running/completed/failed through the ``codt-registry`` CLI; every
        registry call is ``|| true`` so status reporting can never kill a
        simulation.
        """
        lines: list[str] = []
        if self.registry is None:
            lines.extend([
                'for i in "${!RUNS[@]}"; do',
                f'  taskset -c "$i" {self.executable} '
                '"${RUNS[$i]}/inputs/params.nml" &',
                "done",
            ])
            return lines

        db = self.registry.db_path
        # Absolute CLI path so status reporting works regardless of the
        # job's PATH/environment; fall back to bare name.
        cli = shutil.which("codt-registry") or "codt-registry"
        lines.extend([
            f'REGISTRY="{cli} --db {db}"',
            'JOBID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
            "",
            'for i in "${!RUNS[@]}"; do',
            '  RUN_DIR="${RUNS[$i]}"',
            '  RUN_NAME="$(basename "$RUN_DIR")"',
            "  (",
            '    $REGISTRY update-status "$RUN_NAME" running '
            '--job-id "$JOBID" || true',
            f'    taskset -c "$i" {self.executable} '
            '"$RUN_DIR/inputs/params.nml"',
            "    rc=$?",
            "    if [ $rc -eq 0 ]; then",
            '      $REGISTRY update-status "$RUN_NAME" completed '
            '--exit-code $rc --job-id "$JOBID" || true',
            '      $REGISTRY complete "$RUN_NAME" "$RUN_DIR/output" || true',
            "    else",
            '      $REGISTRY update-status "$RUN_NAME" failed '
            '--exit-code $rc --job-id "$JOBID" '
            '--detail "exit $rc; logs: $RUN_DIR/output/*.log and '
            f'{self.base_output_dir}/slurm/$JOBID.out" || true',
            "    fi",
            "  ) &",
            "done",
        ])
        return lines

    def _generate_array_sbatch(
        self,
        batches: list[list[Path]],
        walltime: str,
        job_name: str,
        manifest_path: Path,
        array_throttle: int | None = None,
    ) -> str:
        """Generate a single SLURM job-array script for a whole ensemble.

        One array task per batch; each task reads its own line of the
        manifest, so the script stays small no matter how many runs there
        are.

        Parameters
        ----------
        batches : list[list[Path]]
            Run directories grouped per array task.
        walltime : str
            SLURM wall-clock time (e.g. ``"24:00:00"``).
        job_name : str
            SLURM job name; also used for the output file pattern.
        manifest_path : Path
            Path the batch->run mapping is written to.
        array_throttle : int, optional
            Cap on concurrently running array tasks (renders ``%N``).

        Returns
        -------
        str
            The batch script content.
        """
        ntasks = max(len(b) for b in batches)
        array_spec = f"0-{len(batches) - 1}"
        if array_throttle:
            array_spec += f"%{array_throttle}"

        lines = [
            "#!/bin/bash",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --partition={self.partition}",
        ]
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        lines.extend([
            "#SBATCH --nodes=1",
            f"#SBATCH --ntasks={ntasks}",
        ])
        if self.mem_per_task:
            lines.append(f"#SBATCH --mem={self._total_mem(ntasks)}")
        lines.extend([
            f"#SBATCH --time={walltime}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --array={array_spec}",
            f"#SBATCH --output={self.base_output_dir}/slurm/"
            f"{job_name}_%A_%a.out",
            "",
            f'mapfile -t BATCHES < "{manifest_path}"',
            'read -ra RUNS <<< "${BATCHES[$SLURM_ARRAY_TASK_ID]}"',
            "",
        ])
        lines.extend(self._task_body())
        lines.append("wait")
        return "\n".join(lines) + "\n"

    def _total_mem(self, ntasks: int) -> str:
        """Scale ``mem_per_task`` by task count into a SLURM --mem value."""
        text = str(self.mem_per_task).strip()
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?B?)", text, re.I)
        if not match:
            # Unrecognized format: pass through untouched rather than
            # silently mis-scaling the request.
            return text
        value, unit = match.groups()
        total = float(value) * ntasks
        total_int = int(total) if total == int(total) else total
        return f"{total_int}{unit.upper() or 'M'}"

    def submit(
        self,
        run_dirs: list[Path],
        walltime: str | None = None,
        dry_run: bool = False,
        array_throttle: int | None = None,
        force: bool = False,
    ) -> list[str]:
        """Submit an ensemble to SLURM as a single job array.

        Runs are packed ``cores_per_node`` to an array task. Before writing
        anything, the target is preflight-checked: submitting an
        architecture-tuned binary to a partition without matching nodes
        raises rather than producing jobs that hang unschedulable or crash
        with SIGILL.

        Parameters
        ----------
        run_dirs : list[Path]
            Run directories (from :meth:`setup_run`).
        walltime : str, optional
            SLURM wall-clock time. Defaults to ``self.default_walltime``
            (set from the experiment spec's ``slurm_options`` by
            ``create_experiment_runs``), else ``"24:00:00"``.
        dry_run : bool, optional
            If ``True``, write the script and manifest but do not submit.
            Returns the script path instead of job IDs.
        array_throttle : int, optional
            Maximum array tasks running at once. Defaults to
            ``self.default_array_throttle`` (set from the experiment spec's
            ``slurm_options`` by ``create_experiment_runs``). Worth setting on
            preemptable partitions, where a large burst is both antisocial
            and more exposed to preemption.
        force : bool, optional
            Downgrade preflight failures to warnings. An escape hatch, not
            a default.

        Returns
        -------
        list[str]
            A single-element list: the array job ID (``dry_run=False``) or
            the script path (``dry_run=True``).

        Raises
        ------
        ValueError
            If the preflight check fails and ``force`` is False.
        """
        if not run_dirs:
            return []
        if walltime is None:
            walltime = self.default_walltime or "24:00:00"
        if array_throttle is None:
            array_throttle = self.default_array_throttle

        problems = slurm_mod.validate_target(
            self.partition,
            self.account,
            self.qos,
            features=self._features or None,
            cluster=self.cluster,
            build_arch=self.build_arch,
        )
        if problems:
            message = "SLURM target preflight failed:\n" + "\n".join(
                f"  - {p}" for p in problems
            )
            if force:
                warnings.warn(message + "\nProceeding anyway (force=True).")
            else:
                raise ValueError(
                    message + "\nPass force=True to submit anyway."
                )

        batches = self._batch(run_dirs)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        job_name = f"{self.experiment_id or 'CODT'}_{stamp}"

        slurm_dir = self.base_output_dir / "slurm"
        slurm_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = slurm_dir / f"{job_name}.manifest"
        manifest_path.write_text(
            "\n".join(" ".join(str(d) for d in batch) for batch in batches)
            + "\n"
        )

        script_path = slurm_dir / f"{job_name}.sh"
        script_path.write_text(
            self._generate_array_sbatch(
                batches, walltime, job_name, manifest_path, array_throttle
            )
        )

        if dry_run:
            return [str(script_path)]

        if not self.executable.is_file():
            raise FileNotFoundError(f"Executable not found: {self.executable}")

        cmd = ["sbatch"]
        if self.cluster:
            cmd += ["-M", self.cluster]
        cmd.append(str(script_path))
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
        # Parse "Submitted batch job 12345"
        array_job_id = proc.stdout.strip().split()[-1]

        if self.registry is not None:
            for task_idx, batch in enumerate(batches):
                for sim_dir in batch:
                    self.registry.update_status(
                        sim_dir.name,
                        "queued",
                        slurm_job_id=f"{array_job_id}_{task_idx}",
                    )
            if self.experiment_id is not None:
                self.registry.update_experiment(
                    self.experiment_id, status="running"
                )

        return [array_job_id]

    # ------------------------------------------------------------------
    # Status and collection
    # ------------------------------------------------------------------

    def status(self, job_ids: list[str]) -> dict[str, str]:
        """Query SLURM for job status.

        Jobs missing from ``squeue`` have left the queue, but that alone
        does not mean they succeeded: a preempted, timed-out or OOM-killed
        run looks identical there. Those IDs are therefore resolved through
        ``sacct``, which retains terminal states. Only IDs that neither
        command knows about fall back to ``"COMPLETED"``.

        Parameters
        ----------
        job_ids : list[str]
            SLURM job IDs to query. Array tasks use ``{array}_{task}``.

        Returns
        -------
        dict[str, str]
            Mapping of job ID to state string (e.g. ``"RUNNING"``,
            ``"PENDING"``, ``"COMPLETED"``, ``"PREEMPTED"``).
        """
        if not job_ids:
            return {}

        id_str = ",".join(job_ids)
        cluster_args = ["-M", self.cluster] if self.cluster else []

        proc = subprocess.run(
            ["squeue", *cluster_args, f"--jobs={id_str}",
             "--noheader", "--format=%i %T"],
            capture_output=True,
            text=True,
            check=False,
        )

        active: dict[str, str] = {}
        for line in proc.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                active[parts[0]] = parts[1]

        missing = [jid for jid in job_ids if jid not in active]
        if missing:
            active.update(self._sacct_status(missing, cluster_args))

        return {jid: active.get(jid, "COMPLETED") for jid in job_ids}

    def _sacct_status(
        self, job_ids: list[str], cluster_args: list[str]
    ) -> dict[str, str]:
        """Look up terminal job states in the accounting database."""
        proc = subprocess.run(
            ["sacct", *cluster_args, "-j", ",".join(job_ids),
             "-X", "-n", "-P", "--format=JobID,State"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return {}

        states: dict[str, str] = {}
        for line in proc.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 2 and parts[0]:
                # "CANCELLED by 12345" -> "CANCELLED"
                states[parts[0]] = parts[1].split()[0] if parts[1] else ""
        return states

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
