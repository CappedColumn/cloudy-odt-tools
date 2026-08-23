"""A single CODT run: a case staged into a directory, and its execution.

A :class:`Run` is the pairing of a :class:`~codt_tools.case.Case` (*what* to
simulate) with a binary and a working directory (*where* it happens). It owns
one directory and nothing outside it::

    {workdir}/
        inputs/
            params.nml
            aerosol_input.nc
            [parcel_input.nc]
            [mie table]
        output/
            {sim_name}.nc, {sim_name}.log, {sim_name}_DONE, ...

Nothing here knows about SLURM, accounts or partitions. Running a batch of
these is a separate concern: :mod:`codt_tools.run.launcher` writes a script
that you launch yourself.

Examples
--------
>>> run = Run(case, "~/bin/CODT", "/scratch/me/exp/control")
>>> run.stage()
>>> run.execute_local()
>>> sim = run.open_simulation()
"""

from __future__ import annotations

import os
import signal
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Union

from codt_tools.case import Case, Namelist

if TYPE_CHECKING:
    from codt_tools.simulation import Simulation

#: Name of the namelist inside a run's ``inputs/`` directory.
NAMELIST_FILENAME: str = "params.nml"

#: Subdirectory names within a run directory.
INPUTS_DIRNAME: str = "inputs"
OUTPUT_DIRNAME: str = "output"

#: Seconds to wait for ``--version``, which exits before CODT loads any
#: simulation module.
PROBE_TIMEOUT_S: float = 10.0


class Run:
    """One CODT simulation: a case, a binary, and a directory to work in.

    Parameters
    ----------
    case : Case
        What to simulate. Never modified — :meth:`stage` derives the staged
        namelist rather than assigning paths onto the case, so one case can
        back any number of runs.
    executable : str or Path
        The CODT binary. Resolved to an absolute path.
    workdir : str or Path
        The run directory. Created by :meth:`stage`; resolved to an absolute
        path so that a run is independent of the working directory it was
        defined in.

    Notes
    -----
    Both paths are resolved absolute at construction. That is what preserves
    the staging invariant: CODT resolves ``aerosol_file`` and ``parcel_file``
    against the parent of ``argv[1]`` *as typed* (``app/main.f90:43``), so the
    bare filenames written into ``inputs/`` only resolve correctly when the
    model is launched with an absolute namelist path.
    """

    def __init__(
        self,
        case: Case,
        executable: Union[str, Path],
        workdir: Union[str, Path],
    ) -> None:
        self.case: Case = case
        self.executable: Path = Path(executable).expanduser().resolve()
        self.workdir: Path = Path(workdir).expanduser().resolve()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The run's name: its directory name."""
        return self.workdir.name

    @property
    def inputs_dir(self) -> Path:
        """Directory the input files are staged into."""
        return self.workdir / INPUTS_DIRNAME

    @property
    def output_dir(self) -> Path:
        """Directory CODT writes results into."""
        return self.workdir / OUTPUT_DIRNAME

    @property
    def namelist_path(self) -> Path:
        """Absolute path of the staged namelist — CODT's ``argv[1]``."""
        return self.inputs_dir / NAMELIST_FILENAME

    @property
    def done_marker(self) -> Path:
        """The success marker CODT writes on a clean exit.

        Named from the case's ``simulation_name``, which is what the model
        uses — not from the run directory name, which need not match.
        """
        return self.output_dir / f"{self.case.name}_DONE"

    @property
    def is_staged(self) -> bool:
        """True once the namelist exists on disk."""
        return self.namelist_path.is_file()

    @property
    def is_complete(self) -> bool:
        """True once CODT has written this run's ``_DONE`` marker."""
        return self.done_marker.is_file()

    # ------------------------------------------------------------------
    # Staging
    # ------------------------------------------------------------------

    def stage(self) -> Namelist:
        """Write this run's input files into ``inputs/``.

        Creates ``inputs/`` and ``output/``, writes the aerosol file, the
        parcel file (parcel mode), the Mie table (when ``do_radiation``) and
        ``params.nml``. The staged namelist points ``output_directory`` at
        this run's absolute ``output/``.

        The case is not modified.

        Returns
        -------
        Namelist
            The namelist as written, staged paths included. This — not
            ``case.params`` — is what a run actually reads, so it is what
            provenance records should capture.
        """
        return self.case.write_inputs(
            self.inputs_dir, output_directory=self.output_dir
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_local(self, **kwargs: Any) -> subprocess.CompletedProcess:
        """Run the binary here and now, blocking until it exits.

        CODT does its own stdout redirection (into
        ``output/{sim_name}.log``), so nothing is captured by default;
        errors still reach the terminal on stderr.

        Parameters
        ----------
        **kwargs
            Passed through to :func:`subprocess.run`.

        Returns
        -------
        subprocess.CompletedProcess
            Check ``.returncode``: 0 is success, 1 is a CODT error.

        Raises
        ------
        FileNotFoundError
            If the executable is missing, or the run has not been staged.
        """
        if not self.executable.is_file():
            raise FileNotFoundError(f"Executable not found: {self.executable}")
        if not os.access(self.executable, os.X_OK):
            raise PermissionError(
                f"Not executable (chmod +x it): {self.executable}"
            )
        if not self.is_staged:
            raise FileNotFoundError(
                f"No namelist at {self.namelist_path} — call stage() first."
            )

        kwargs.setdefault("check", False)
        # An absolute namelist path: see the class docstring.
        proc = subprocess.run(
            [str(self.executable), str(self.namelist_path)], **kwargs
        )

        # A negative return code is a fatal signal, not a CODT error code.
        # SIGILL in particular is unreadable as raw output ("Illegal
        # instruction"), so name it here where it happens.
        if proc.returncode < 0:
            warnings.warn(
                f"Run '{self.name}': the binary "
                f"{_signal_problem(-proc.returncode)}",
                RuntimeWarning,
                stacklevel=2,
            )
        return proc

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def open_simulation(self) -> "Simulation":
        """Open this run's output for analysis.

        Returns
        -------
        Simulation
            Reader for ``output/``.

        Raises
        ------
        FileNotFoundError
            If the run has not completed (no ``_DONE`` marker), which is
            reported in preference to whatever partial output exists.
        """
        # Imported here, not at module scope: the analysis layer pulls in
        # xarray and matplotlib, which staging and launching do not need.
        from codt_tools.simulation import Simulation

        return Simulation.from_run(self)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def for_cases(
        cls,
        cases: Iterable[Case],
        executable: Union[str, Path],
        base_dir: Union[str, Path],
    ) -> list["Run"]:
        """Build one run per case, each in ``base_dir/{case.name}``.

        The one line every ensemble writes. Case names come from
        :meth:`Case.sweep`, which numbers them ``{base}_000``, so the run
        directories inherit that ordering.

        Parameters
        ----------
        cases : iterable of Case
            The design, already expanded into cases.
        executable : str or Path
            The CODT binary, shared by every run.
        base_dir : str or Path
            Parent directory for the run directories.

        Returns
        -------
        list of Run
            One run per case, in order.

        Raises
        ------
        ValueError
            If two cases share a name, which would silently stage them into
            the same directory.
        """
        base_dir = Path(base_dir).expanduser().resolve()
        runs: list["Run"] = [
            cls(case, executable, base_dir / case.name) for case in cases
        ]

        seen: dict[str, int] = {}
        for index, run in enumerate(runs):
            if run.name in seen:
                raise ValueError(
                    f"Cases {seen[run.name]} and {index} are both named "
                    f"'{run.name}'; they would stage into the same directory. "
                    f"Give Case.sweep a name= function, or set distinct "
                    f"simulation_name values."
                )
            seen[run.name] = index
        return runs

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = "complete" if self.is_complete else (
            "staged" if self.is_staged else "unstaged"
        )
        return f"Run(name='{self.name}', workdir='{self.workdir}', {state})"


# ---------------------------------------------------------------------------
# Is this binary usable?
# ---------------------------------------------------------------------------


def check_executable(executable: Union[str, Path]) -> str | None:
    """Report why a CODT binary is unusable, or None if it works.

    Probes with ``--version``, which CODT's CLI answers before loading any
    simulation module (``app/main.f90``), so this starts the binary without
    touching input files.

    Parameters
    ----------
    executable : str or Path
        Path to the CODT binary.

    Returns
    -------
    str or None
        A one-line description of the first problem found, suitable for
        printing as-is; None when the binary ran and reported its version.
        Never raises.

    Notes
    -----
    This judges the binary *on the host it runs on*. An architecture-tuned
    build can pass here on a login node and still die with SIGILL on every
    compute node, so it does not replace the ``--version`` line that the
    generated launch scripts run in place — it only catches the mistakes
    that travel: a wrong path, a missing execute bit, a broken build.

    Examples
    --------
    >>> problem = check_executable("~/bin/CODT")
    >>> if problem:
    ...     print(f"CODT binary unusable: {problem}")
    """
    path = Path(executable).expanduser().resolve()

    if not path.exists():
        return "no such file"
    if path.is_dir():
        return "is a directory, not a binary"
    if not os.access(path, os.X_OK):
        return "not executable (chmod +x it)"

    try:
        proc = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            f"--version did not return within {PROBE_TIMEOUT_S:g}s; this may "
            f"not be a CODT binary"
        )
    except OSError as exc:
        # ENOEXEC and friends: a text file, a wrong-format binary.
        return f"could not be executed: {exc.strerror or exc}"

    if proc.returncode < 0:
        return _signal_problem(-proc.returncode)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        first = detail[0] if detail else "no output"
        return (
            f"--version exited {proc.returncode}: {first} "
            f"(CODT supports --version from 0.4.0)"
        )
    return None


def _signal_problem(signum: int) -> str:
    """Explain a fatal signal in the terms that matter for a CODT binary."""
    if signum == signal.SIGILL:
        return (
            "died on an illegal instruction (SIGILL) — it is built for a "
            "different CPU than this host. Run it on a matching node, or "
            "rebuild for this one."
        )
    if signum == signal.SIGSEGV:
        return (
            "crashed with a segmentation fault (SIGSEGV) — the build is "
            "likely broken or mismatched to its shared libraries."
        )
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = f"signal {signum}"
    return f"was killed by {name}"
