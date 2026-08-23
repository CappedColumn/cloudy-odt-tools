"""Generate launch scripts for a batch of runs. Nothing here executes them.

codt_tools writes a shell script; you read it and launch it. No ``sbatch``,
``squeue`` or ``sacct`` is ever called from Python, which means the generated
script is the whole interface — reviewable before it runs, editable after, and
runnable on a cluster this package knows nothing about.

Both writers follow the script this was modeled on, which had been in use by
hand: one subshell per run, a ``_DONE`` skip guard that makes re-running the
script resume rather than redo, a timing line per run, and a final ``wait``.

Examples
--------
>>> write_local(runs, base / "run_all.sh", jobs=4)
>>> write_slurm_array(runs, base / "array.sh", runs_per_task=8,
...                   account="krueger", partition="notchpeak-freecycle")
"""

from __future__ import annotations

import math
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, Union

from codt_tools.run.run import Run

#: Default number of runs executing at once under :func:`write_local`.
DEFAULT_JOBS: int = 4

#: Placeholders written for SLURM directives left unspecified. They are
#: syntactically inert (``#SBATCH`` lines are shell comments), so the script
#: still parses; SLURM rejects it until they are filled in, which is the
#: intent — a wrong account is worse than an obvious blank.
PLACEHOLDER_ACCOUNT: str = "<ACCOUNT>"
PLACEHOLDER_PARTITION: str = "<PARTITION>"
PLACEHOLDER_TIME: str = "<TIME>"


def write_local(
    runs: Sequence[Run],
    path: Union[str, Path],
    jobs: int = DEFAULT_JOBS,
    timing_file: Union[str, Path, None] = None,
) -> Path:
    """Write a script that runs a batch locally, *jobs* at a time.

    The script is idempotent: any run whose ``_DONE`` marker already exists
    is skipped, so re-running it after an interruption resumes.

    Parameters
    ----------
    runs : sequence of Run
        The runs to launch. Must already be staged when the script runs;
        staging is not the script's job.
    path : str or Path
        Where to write the script.
    jobs : int, optional
        How many runs execute concurrently. ``jobs=1`` gives the serial
        case. Each CODT run is single-threaded, so this is a core count.
    timing_file : str or Path, optional
        Where per-run timings are appended. Defaults to ``timing.txt``
        beside the script.

    Returns
    -------
    Path
        The script path, absolute.

    Raises
    ------
    ValueError
        If *runs* is empty, if *jobs* is below 1, or if the runs do not all
        share one executable (the script names a single binary).

    Notes
    -----
    Launch it detached, so it survives logout::

        nohup bash run_all.sh > run_all.log 2>&1 &
    """
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")

    path = Path(path).expanduser().resolve()
    executable = _one_executable(runs)
    timing = (
        Path(timing_file).expanduser().resolve()
        if timing_file is not None
        else path.parent / "timing.txt"
    )

    lines = [
        "#!/bin/bash",
        *_preamble(len(runs)),
        "#",
        "# Launch detached, so it survives logout:",
        f"#     nohup bash {path.name} > {path.stem}.log 2>&1 &",
        "",
        f"EXE={shlex.quote(str(executable))}",
        f"TIMING={shlex.quote(str(timing))}",
        f"JOBS={jobs}",
        "",
        *_runs_array(runs),
        "",
        'echo "host $(hostname) -- started $(date)"',
        '"$EXE" --version',
        "",
        'for RUN_DIR in "${RUNS[@]}"; do',
        *_run_one_subshell(indent="  "),
        "  ) &",
        "",
        "  # Hold at JOBS concurrent runs; wait -n returns as each finishes.",
        '  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done',
        "done",
        "wait",
        "",
        'echo "all runs finished $(date)"',
    ]
    return _write(path, lines)


def write_slurm_array(
    runs: Sequence[Run],
    path: Union[str, Path],
    runs_per_task: int = 1,
    module: str | None = None,
    job_name: str | None = None,
    account: str | None = None,
    partition: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
    time: str | None = None,
    mem: str | None = None,
    array_throttle: int | None = None,
    requeue: bool = True,
    manifest: Union[str, Path, None] = None,
    **directives: Any,
) -> Path:
    """Write a SLURM job-array script for a batch, plus its run manifest.

    One array task per group of *runs_per_task* runs, which execute in
    parallel within the task. The run directories live in a sibling manifest
    file — one whitespace-separated line per task — which each task reads by
    ``$SLURM_ARRAY_TASK_ID``, so the script stays the same size whether the
    ensemble has ten runs or a thousand.

    Nothing is submitted. Read the script, fill in any placeholders, then::

        sbatch array.sh

    Parameters
    ----------
    runs : sequence of Run
        The runs to launch, already staged.
    path : str or Path
        Where to write the script.
    runs_per_task : int, optional
        Runs sharing one array task, executing concurrently. This is also
        ``--ntasks``, so it should not exceed the cores a task will get.
    module : str, optional
        An LMOD module to load first (e.g. ``"gcc/11.2.0"``). Needed when
        the binary has no RPATH to its runtime libraries.
    job_name : str, optional
        SLURM job name. Defaults to the script's stem.
    account, partition, qos, constraint, time, mem : str, optional
        SLURM directives. ``account``, ``partition`` and ``time`` are
        written as ``<ACCOUNT>``/``<PARTITION>``/``<TIME>`` placeholders
        when omitted; the rest are simply left out.
    array_throttle : int, optional
        Cap on array tasks running at once (renders ``--array=0-N%throttle``).
        Worth setting on preemptable partitions, where a large burst is both
        antisocial and more exposed to preemption.
    requeue : bool, optional
        Emit ``#SBATCH --requeue``. On by default, and safe, because the
        ``_DONE`` guard means a requeued job skips what already finished.
        That pairing is the entire preemption story — CODT has no
        checkpoint/restart, so an interrupted *individual* run restarts from
        the beginning.
    manifest : str or Path, optional
        Where the task -> run-directory mapping is written. Defaults to
        ``{script_stem}_runs.txt`` beside the script.
    **directives
        Further ``#SBATCH`` options; underscores become dashes, so
        ``mail_type="END"`` writes ``#SBATCH --mail-type=END``.

    Returns
    -------
    Path
        The script path, absolute. The manifest sits beside it.

    Raises
    ------
    ValueError
        If *runs* is empty, if *runs_per_task* is below 1, or if the runs do
        not all share one executable.
    """
    if runs_per_task < 1:
        raise ValueError(
            f"runs_per_task must be at least 1, got {runs_per_task}"
        )

    path = Path(path).expanduser().resolve()
    executable = _one_executable(runs)
    name = job_name if job_name is not None else path.stem
    manifest_path = (
        Path(manifest).expanduser().resolve()
        if manifest is not None
        else path.parent / f"{path.stem}_runs.txt"
    )
    timing = path.parent / "timing.txt"

    tasks = _chunk(runs, runs_per_task)
    ntasks = max(len(task) for task in tasks)
    array_spec = f"0-{len(tasks) - 1}"
    if array_throttle:
        array_spec += f"%{array_throttle}"

    sbatch = [
        f"#SBATCH --account={account or PLACEHOLDER_ACCOUNT}",
        f"#SBATCH --partition={partition or PLACEHOLDER_PARTITION}",
    ]
    if qos:
        sbatch.append(f"#SBATCH --qos={qos}")
    if constraint:
        sbatch.append(f"#SBATCH --constraint={constraint}")
    sbatch.extend([
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={ntasks}",
    ])
    if mem:
        sbatch.append(f"#SBATCH --mem={mem}")
    sbatch.extend([
        f"#SBATCH --time={time or PLACEHOLDER_TIME}",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --output={path.parent}/{name}_%A_%a.out",
    ])
    for key, value in directives.items():
        sbatch.append(f"#SBATCH --{key.replace('_', '-')}={value}")
    if requeue:
        sbatch.extend([
            "# Requeue-safe: the loop below skips any run that already wrote",
            "# its _DONE marker, so a preempted job resumes rather than redoes.",
            "#SBATCH --requeue",
        ])

    lines = [
        "#!/bin/bash",
        *sbatch,
        "",
        *_preamble(len(runs)),
        f"# {len(tasks)} array task(s), up to {ntasks} run(s) each.",
        "#",
        "# Fill in any <PLACEHOLDER> above, then: sbatch " + path.name,
        "",
    ]
    if module:
        lines.extend([f"module load {module}", ""])
    lines.extend([
        f"EXE={shlex.quote(str(executable))}",
        f"TIMING={shlex.quote(str(timing))}",
        "",
        f"mapfile -t TASKS < {shlex.quote(str(manifest_path))}",
        'read -ra RUNS <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"',
        "",
        'echo "task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID '
        'on $(hostname)"',
        '"$EXE" --version',
        "",
        'for RUN_DIR in "${RUNS[@]}"; do',
        *_run_one_subshell(indent="  "),
        "  ) &",
        "done",
        "wait",
        "",
        'echo "task $SLURM_ARRAY_TASK_ID finished $(date)"',
    ])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            " ".join(str(run.workdir) for run in task) for task in tasks
        )
        + "\n"
    )
    return _write(path, lines)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _run_one_subshell(indent: str = "") -> list[str]:
    """The per-run subshell: skip if done, else run it and record timing.

    The guard globs for ``*_DONE`` rather than naming the marker, because
    a run directory's name need not match the ``simulation_name`` the model
    builds the marker from.
    """
    body = [
        "(",
        "  # Idempotent: a run that already finished is not redone.",
        '  if compgen -G "$RUN_DIR/output/*_DONE" > /dev/null; then',
        '    echo "skip $(basename "$RUN_DIR") -- already complete"',
        "    exit 0",
        "  fi",
        "  start=$(date +%s)",
        '  "$EXE" "$RUN_DIR/inputs/params.nml"',
        "  rc=$?",
        "  # columns: run, wall seconds, exit code",
        "  printf '%s %s %s\\n' \"$(basename \"$RUN_DIR\")\" "
        '"$(( $(date +%s) - start ))" "$rc" >> "$TIMING"',
    ]
    return [indent + line for line in body]


def _runs_array(runs: Sequence[Run]) -> list[str]:
    """The ``RUNS=(...)`` array of absolute run directories."""
    return [
        "RUNS=(",
        *[f"  {shlex.quote(str(run.workdir))}" for run in runs],
        ")",
    ]


def _preamble(n_runs: int) -> list[str]:
    """Header comments identifying the generator and the batch size."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        f"# Generated by codt_tools on {stamp} for {n_runs} run(s).",
        "# Regenerate rather than edit by hand where you can.",
    ]


def _one_executable(runs: Sequence[Run]) -> Path:
    """The single binary shared by every run, or an error explaining why not.

    A generated script names one ``$EXE``, so a batch mixing binaries would
    silently run some of it with the wrong one.
    """
    if not runs:
        raise ValueError("No runs to launch.")

    executables = {run.executable for run in runs}
    if len(executables) > 1:
        listed = "\n".join(f"  {exe}" for exe in sorted(executables))
        raise ValueError(
            f"These runs use {len(executables)} different executables, but a "
            f"generated script names one:\n{listed}\n"
            f"Write one script per executable."
        )
    return executables.pop()


def _chunk(runs: Sequence[Run], size: int) -> list[Sequence[Run]]:
    """Split *runs* into consecutive groups of at most *size*."""
    n_tasks = math.ceil(len(runs) / size)
    return [runs[i * size : (i + 1) * size] for i in range(n_tasks)]


def _write(path: Path, lines: list[str]) -> Path:
    """Write a script and make it executable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)
    return path
