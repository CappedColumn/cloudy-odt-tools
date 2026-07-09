"""Experiment specifications: YAML definition, sweep expansion, run creation.

An experiment spec is a YAML file describing a hypothesis-driven set of
CODT runs: a base configuration plus a parameter sweep. The spec is
Snakemake-ready (plain data, no code) and is consumed here to expand
configs, lay out the experiment directory tree, deduplicate shared input
files, and register everything in the simulation registry.

Experiment directory layout (see registry docs)::

    {data_root}/{experiment_id}/
        experiment.yaml
        shared_inputs/   (inputs identical across runs, by content hash)
        runs/{run_id}/
            inputs/      (params.nml + symlinks -> ../../../shared_inputs/*)
            output/

Primary usage::

    from codt_tools.experiment import ExperimentSpec, create_experiment_runs
    from codt_tools.registry import Registry

    spec = ExperimentSpec.from_yaml("experiment.yaml")
    with Registry("registry.db") as reg:
        runner, run_dirs = create_experiment_runs(spec, reg, "/path/to/CODT")
        runner.submit(run_dirs, walltime=spec.slurm_options["walltime"])
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

from codt_tools.config import CODTConfig
from codt_tools.registry import Registry, sha256_file
from codt_tools.runner import CODTRunner


@dataclass
class ExperimentSpec:
    """A declarative description of one experiment.

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier (used as registry key, directory
        name under ``data_root``, and base simulation name).
    title : str
        One-line human-readable title.
    data_root : str or Path
        Root directory under which the experiment tree is created.
    hypothesis : str, optional
        The question this experiment answers.
    model : str, optional
        Model name recorded with each run (default ``"codt"``).
    base_parameters : dict, optional
        Namelist parameter overrides applied to every run (flat
        ``param -> value`` mapping; groups are resolved by ``CODTConfig``).
    parameter_sweep : dict, optional
        ``param -> list of values`` mapping, expanded as a Cartesian
        product via :meth:`CODTConfig.sweep`. Empty means a single run.
    execution_context : str, optional
        Free-text note on where/how runs execute (e.g. ``"notchpeak"``).
    permanent_data_root : str or Path, optional
        Where the experiment tree should live long-term (e.g. group
        space). Runs execute under ``data_root`` (typically scratch);
        after validation the tree is moved here and the registry
        updated via ``Registry.relocate_experiment``.
    slurm_options : dict, optional
        SLURM settings for the runner: ``account``, ``partition``,
        ``cores_per_node``, ``walltime``.
    """

    experiment_id: str
    title: str
    data_root: Union[str, Path]
    hypothesis: str | None = None
    model: str = "codt"
    base_parameters: dict[str, Any] = field(default_factory=dict)
    parameter_sweep: dict[str, list] = field(default_factory=dict)
    execution_context: str | None = None
    permanent_data_root: Union[str, Path, None] = None
    slurm_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentSpec":
        """Load a spec from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the experiment YAML file.

        Returns
        -------
        ExperimentSpec

        Raises
        ------
        KeyError
            If a required field (experiment_id, title, data_root) is
            missing.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown experiment spec fields: {sorted(unknown)}"
            )
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write the spec to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "data_root": str(self.data_root),
            "hypothesis": self.hypothesis,
            "model": self.model,
            "base_parameters": self.base_parameters,
            "parameter_sweep": self.parameter_sweep,
            "execution_context": self.execution_context,
            "permanent_data_root": (
                str(self.permanent_data_root)
                if self.permanent_data_root is not None else None
            ),
            "slurm_options": self.slurm_options,
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @property
    def experiment_dir(self) -> Path:
        """The experiment root: ``{data_root}/{experiment_id}``."""
        return Path(self.data_root).expanduser().resolve() / self.experiment_id

    def expand(self) -> list[CODTConfig]:
        """Expand base parameters + sweep into one config per run.

        The base simulation name is ``experiment_id``; sweep configs get
        the standard sweep suffix (e.g. ``{experiment_id}_Tref20.0_VS13``).

        Returns
        -------
        list[CODTConfig]
            One config per sweep point (a single config if no sweep).
        """
        base = CODTConfig()
        base.set(simulation_name=self.experiment_id)
        if self.base_parameters:
            base.set(**self.base_parameters)
        return CODTConfig.sweep(base, **self.parameter_sweep)

    def descriptor(self, config: CODTConfig) -> str:
        """The sweep-point descriptor of an expanded config.

        The part of the simulation name after the experiment_id prefix
        (e.g. ``Tref20.0_VS13``), or ``"base"`` for an unswept run.
        """
        prefix = f"{self.experiment_id}_"
        if config.name.startswith(prefix):
            return config.name[len(prefix):]
        return "base"


def dedup_shared_inputs(
    run_dirs: list[Path], shared_dir: Union[str, Path]
) -> dict[str, list[Path]]:
    """Deduplicate identical input files across runs via symlinks.

    Input files (everything in each run's ``inputs/`` except
    ``params.nml``) whose content hash appears in more than one run are
    moved once into *shared_dir* and replaced by **relative** symlinks,
    so the experiment tree can be relocated as a whole and CODT reads
    through the links. Dedup is by content hash only, never by filename;
    if two distinct contents share a filename, later ones get a hash
    suffix in *shared_dir*.

    Parameters
    ----------
    run_dirs : list[Path]
        Run directories (each containing ``inputs/``).
    shared_dir : str or Path
        Directory for the shared copies (created on first dedup).

    Returns
    -------
    dict[str, list[Path]]
        Mapping of content hash -> symlinked input paths, for files
        that were deduplicated.
    """
    shared_dir = Path(shared_dir)

    by_hash: dict[str, list[Path]] = {}
    for run_dir in run_dirs:
        inputs_dir = Path(run_dir) / "inputs"
        for path in sorted(inputs_dir.iterdir()):
            if path.name == "params.nml" or path.is_symlink():
                continue
            by_hash.setdefault(sha256_file(path), []).append(path)

    deduped: dict[str, list[Path]] = {}
    for digest, files in by_hash.items():
        if len(files) < 2:
            continue
        shared_dir.mkdir(parents=True, exist_ok=True)
        target = shared_dir / files[0].name
        if target.exists() and sha256_file(target) != digest:
            target = shared_dir / (
                f"{target.stem}_{digest[:8]}{target.suffix}"
            )
        if not target.exists():
            files[0].rename(target)
        for path in files:
            if path.exists() and not path.is_symlink():
                path.unlink()
            if not path.is_symlink():
                path.symlink_to(os.path.relpath(target, path.parent))
        deduped[digest] = files
    return deduped


def create_experiment_runs(
    spec: ExperimentSpec,
    registry: Registry,
    executable: Union[str, Path],
) -> tuple[CODTRunner, list[Path]]:
    """Create and register all runs of an experiment.

    Creates the experiment in the registry (status ``planned``), writes
    ``experiment.yaml`` into the experiment directory, expands the sweep,
    sets up each run directory under ``runs/``, deduplicates shared
    inputs into ``shared_inputs/``, and registers each run (so the
    recorded ``input_files`` rows reflect the post-dedup symlink layout).

    Run IDs are ``{YYYYMMDD_HHMMSS}_{model}_{descriptor}`` with a single
    timestamp shared by the whole experiment.

    Parameters
    ----------
    spec : ExperimentSpec
        The experiment specification.
    registry : Registry
        Open registry to record the experiment and runs in.
    executable : str or Path
        Path to the CODT executable.

    Returns
    -------
    tuple[CODTRunner, list[Path]]
        The configured runner (use it for ``submit``/``run_local``) and
        the run directories, in sweep order.

    Raises
    ------
    FileNotFoundError
        If the executable does not exist.
    ValueError
        If the executable does not report a proper version (improper
        build) — rebuild CODT with ``./build.sh`` and point
        *executable* at the new binary.
    """
    exp_dir = spec.experiment_dir
    slurm = spec.slurm_options
    runner = CODTRunner(
        executable=executable,
        base_output_dir=exp_dir / "runs",
        account=slurm.get("account", ""),
        partition=slurm.get("partition", ""),
        cores_per_node=slurm.get("cores_per_node", 40),
        registry=registry,
        experiment_id=spec.experiment_id,
    )
    runner.default_walltime = slurm.get("walltime")

    # Build gate: refuse to create runs from a binary whose provenance
    # can't be recorded. Fails BEFORE anything is registered or written.
    if not runner.executable.is_file():
        raise FileNotFoundError(
            f"CODT executable not found: {runner.executable}. "
            "Fix the 'executable' argument."
        )
    if runner.codt_version is None or "PLACEHOLDER" in runner.codt_version:
        raise ValueError(
            f"Executable {runner.executable} does not report a valid "
            f"version (got {runner.codt_version!r}) — it was not built "
            "properly and its runs would be untraceable. Rebuild CODT "
            "with ./build.sh (which injects version/commit) and pass "
            "the new binary."
        )

    registry.create_experiment(
        spec.experiment_id,
        spec.title,
        hypothesis=spec.hypothesis,
        data_root=str(Path(spec.data_root).expanduser().resolve()),
        permanent_data_root=(
            str(Path(spec.permanent_data_root).expanduser().resolve())
            if spec.permanent_data_root is not None else None
        ),
    )
    spec.to_yaml(exp_dir / "experiment.yaml")

    configs = spec.expand()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ids = [
        f"{stamp}_{spec.model}_{spec.descriptor(cfg)}" for cfg in configs
    ]

    run_dirs = [
        runner.setup_run(cfg, run_id=rid, register=False)
        for cfg, rid in zip(configs, run_ids)
    ]
    dedup_shared_inputs(run_dirs, exp_dir / "shared_inputs")

    for cfg, rid, run_dir in zip(configs, run_ids, run_dirs):
        registry.register_run(
            rid,
            cfg,
            run_dir,
            experiment_id=spec.experiment_id,
            descriptor=spec.descriptor(cfg),
            execution_context=spec.execution_context,
            executable_path=runner.executable,
            code_version=runner.codt_version,
        )
    return runner, run_dirs
