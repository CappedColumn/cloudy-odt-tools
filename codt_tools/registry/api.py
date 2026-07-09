"""
Registry API: experiment and run tracking for CODT simulations.

All database access goes through the :class:`Registry` class (or the
``codt-registry`` CLI, which wraps it). Status changes are recorded twice in
one transaction: an append-only row in ``status_events`` (audit trail) and
the derived current state on ``runs``.
"""

from __future__ import annotations

import json
import socket
import sqlite3
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codt_tools.registry.db import RUN_STATUSES, connect
from codt_tools.registry.versions import (
    archive_executable,
    check_conventions,
    sha256_file,
)

if TYPE_CHECKING:
    from codt_tools.config import CODTConfig

# File types recognized in a run's inputs/ directory, mapped from file name.
_INPUT_FILE_TYPES: dict[str, str] = {
    "params.nml": "namelist",
    "aerosol_input.nc": "aerosol_input",
    "parcel_input.nc": "parcel_input",
}

# update_status retry schedule on 'database is locked' [s].
_RETRY_DELAYS_S: tuple[float, ...] = (0.5, 2.0, 8.0)


def _utcnow() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _param_type(value: Any) -> str:
    """Map a Python namelist value to its Fortran-side type label."""
    if isinstance(value, bool):
        return "logical"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "real"
    if isinstance(value, (list, tuple)):
        return "array"
    return "character"


def _codt_tools_version() -> str:
    """Installed codt-tools version, or 'unknown' in a bare source tree."""
    try:
        return version("codt-tools")
    except PackageNotFoundError:
        return "unknown"


class Registry:
    """
    SQLite-backed registry of CODT experiments and runs.

    Parameters
    ----------
    db_path : str or Path
        Path to the registry database. Created (with schema) if missing
        unless ``create`` is False.
    exe_archive_dir : str or Path or None
        Root of the content-addressed executable archive. Defaults to an
        ``executables/`` directory next to the database file. Set to the
        string ``"off"`` to disable executable archiving.
    create : bool
        Passed through to :func:`codt_tools.registry.db.connect`.
    """

    def __init__(
        self,
        db_path: str | Path,
        exe_archive_dir: str | Path | None = None,
        create: bool = True,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self._conn = connect(self.db_path, create=create)
        if exe_archive_dir == "off":
            self.exe_archive_dir: Path | None = None
        elif exe_archive_dir is None:
            self.exe_archive_dir = self.db_path.parent / "executables"
        else:
            self.exe_archive_dir = Path(exe_archive_dir).expanduser()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        experiment_id: str,
        title: str,
        hypothesis: str | None = None,
        data_root: str | Path | None = None,
        permanent_data_root: str | Path | None = None,
    ) -> None:
        """Create a new experiment in status 'planned'."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO experiments "
                "(experiment_id, title, hypothesis, data_root, "
                "permanent_data_root, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    title,
                    hypothesis,
                    str(data_root) if data_root is not None else None,
                    str(permanent_data_root)
                    if permanent_data_root is not None else None,
                    _utcnow(),
                ),
            )

    def update_experiment(self, experiment_id: str, **fields: Any) -> None:
        """
        Update columns of an experiment record.

        Parameters
        ----------
        experiment_id : str
            Experiment to update.
        **fields
            Column/value pairs; allowed columns are title, hypothesis,
            status, conclusion, analysis_artifacts, data_root.
        """
        allowed = {
            "title",
            "hypothesis",
            "status",
            "conclusion",
            "analysis_artifacts",
            "data_root",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update experiment column(s): {sorted(bad)}")
        if not fields:
            return
        assignments = ", ".join(f"{col} = ?" for col in fields)
        with self._conn:
            cur = self._conn.execute(
                f"UPDATE experiments SET {assignments} WHERE experiment_id = ?",
                (*fields.values(), experiment_id),
            )
        if cur.rowcount == 0:
            raise KeyError(f"Unknown experiment: {experiment_id}")

    def conclude_experiment(
        self,
        experiment_id: str,
        conclusion: str,
        artifacts: list[str] | None = None,
    ) -> None:
        """Record a conclusion and mark the experiment 'concluded'."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE experiments SET status = 'concluded', conclusion = ?, "
                "analysis_artifacts = ?, concluded_at = ? WHERE experiment_id = ?",
                (
                    conclusion,
                    json.dumps(artifacts) if artifacts is not None else None,
                    _utcnow(),
                    experiment_id,
                ),
            )
        if cur.rowcount == 0:
            raise KeyError(f"Unknown experiment: {experiment_id}")

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Return one experiment record as a dict."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        return dict(row)

    def list_experiments(self, status: str | None = None) -> list[dict[str, Any]]:
        """List experiments, optionally filtered by status."""
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM experiments ORDER BY created_at"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM experiments WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def register_run(
        self,
        run_id: str,
        config: CODTConfig,
        run_dir: str | Path,
        *,
        experiment_id: str | None = None,
        descriptor: str | None = None,
        execution_context: str | None = None,
        executable_path: str | Path | None = None,
        code_version: str | None = None,
        git_commit: str | None = None,
        git_branch: str | None = None,
        build_info: str | None = None,
        data_status: str | None = "on_scratch",
        notes: str | None = None,
    ) -> str:
        """
        Register a run whose inputs have already been written to disk.

        Checksums every recognized file in ``{run_dir}/inputs/`` (following
        symlinks; symlink targets recorded), stores all namelist parameters
        from the config's mode-gated groups, and archives the executable
        (content-addressed, deduplicated) unless archiving is disabled.

        Parameters
        ----------
        run_id : str
            Unique run identifier (``YYYYMMDD_HHMMSS_model_vX_descriptor``).
        config : CODTConfig
            The configuration the inputs were written from.
        run_dir : str or Path
            The run directory containing ``inputs/`` (absolute here; stored
            relative to the experiment's data_root when one is set).

        Returns
        -------
        str
            The registered ``run_id``.
        """
        run_dir = Path(run_dir).expanduser().resolve()
        inputs_dir = run_dir / "inputs"
        now = _utcnow()

        stored_run_dir = str(run_dir)
        if experiment_id is not None:
            data_root = self.get_experiment(experiment_id)["data_root"]
            if data_root:
                stored_run_dir = str(run_dir.relative_to(Path(data_root).resolve()))

        exe_path = exe_checksum = exe_archive = None
        if executable_path is not None:
            exe = Path(executable_path).expanduser().resolve()
            exe_path = str(exe)
            if exe.is_file():
                exe_checksum = sha256_file(exe)
                if self.exe_archive_dir is not None:
                    exe_archive = str(archive_executable(exe, self.exe_archive_dir))

        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, experiment_id, descriptor, "
                "execution_context, run_dir, status, created_at, code_version, "
                "git_commit, git_branch, codt_tools_version, executable_path, "
                "executable_checksum, executable_archive_path, build_info, "
                "data_status, notes) "
                "VALUES (?, ?, ?, ?, ?, 'registered', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    experiment_id,
                    descriptor,
                    execution_context,
                    stored_run_dir,
                    now,
                    code_version,
                    git_commit,
                    git_branch,
                    _codt_tools_version(),
                    exe_path,
                    exe_checksum,
                    exe_archive,
                    build_info,
                    data_status,
                    notes,
                ),
            )
            self._conn.execute(
                "INSERT INTO status_events (run_id, status, timestamp, hostname) "
                "VALUES (?, 'registered', ?, ?)",
                (run_id, now, socket.gethostname()),
            )
            for name, file_type in _INPUT_FILE_TYPES.items():
                path = inputs_dir / name
                if not path.exists():
                    continue
                self._conn.execute(
                    "INSERT INTO input_files (run_id, file_type, file_path, "
                    "checksum, size_bytes, is_symlink, link_target) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        file_type,
                        name,
                        sha256_file(path),
                        path.stat().st_size,
                        int(path.is_symlink()),
                        str(path.readlink()) if path.is_symlink() else None,
                    ),
                )
            for group, params in config.params._groups_for_write().items():
                for pname, pvalue in params.items():
                    self._conn.execute(
                        "INSERT INTO namelist_parameters (run_id, group_name, "
                        "param_name, param_value, param_type) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (run_id, group, pname, str(pvalue), _param_type(pvalue)),
                    )
        return run_id

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        exit_code: int | None = None,
        slurm_job_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """
        Record a status transition (event row + derived columns, one txn).

        Retries with backoff on 'database is locked' so packed SLURM tasks
        finishing simultaneously never lose an update.
        """
        if status not in RUN_STATUSES:
            raise ValueError(f"Invalid status {status!r}; expected {RUN_STATUSES}")
        now = _utcnow()

        derived = ["status = ?"]
        values: list[Any] = [status]
        if status == "running":
            derived.append("started_at = ?")
            values.append(now)
        if status in ("completed", "failed"):
            derived.append("completed_at = ?")
            values.append(now)
        if exit_code is not None:
            derived.append("exit_code = ?")
            values.append(exit_code)
        if slurm_job_id is not None:
            derived.append("slurm_job_id = ?")
            values.append(slurm_job_id)

        cur = None
        for delay_s in (*_RETRY_DELAYS_S, None):
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO status_events (run_id, status, timestamp, "
                        "hostname, slurm_job_id, exit_code, detail) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            run_id,
                            status,
                            now,
                            socket.gethostname(),
                            slurm_job_id,
                            exit_code,
                            detail,
                        ),
                    )
                    cur = self._conn.execute(
                        f"UPDATE runs SET {', '.join(derived)} WHERE run_id = ?",
                        (*values, run_id),
                    )
                break
            except sqlite3.IntegrityError as err:
                if "FOREIGN KEY" in str(err):
                    raise KeyError(f"Unknown run: {run_id}") from err
                raise
            except sqlite3.OperationalError as err:
                if "locked" not in str(err) or delay_s is None:
                    raise
                time.sleep(delay_s)
        if cur is not None and cur.rowcount == 0:
            raise KeyError(f"Unknown run: {run_id}")

    def record_completion(self, run_id: str, output_path: str | Path) -> None:
        """
        Store version metadata read from a run's output netCDF file.

        Reads the ``conventions``, ``code_version``, and ``git_commit``
        global attributes, checks the conventions gate (warns if
        unsupported), and updates the run record.

        *output_path* may be the main output ``.nc`` file or the run's
        output directory — in the latter case the file is located via
        the ``{name}_DONE`` completion marker.
        """
        import netCDF4 as nc

        output_path = Path(output_path)
        if output_path.is_dir():
            markers = sorted(output_path.glob("*_DONE"))
            if len(markers) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one *_DONE marker in {output_path}, "
                    f"found {len(markers)}; pass the .nc file explicitly."
                )
            name = markers[0].name.removesuffix("_DONE")
            output_path = output_path / f"{name}.nc"

        with nc.Dataset(output_path) as ds:
            attrs = {
                key: (str(ds.getncattr(key)) if key in ds.ncattrs() else None)
                for key in ("conventions", "code_version", "git_commit")
            }
        check_conventions(attrs["conventions"])
        with self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET conventions = ?, "
                "code_version = COALESCE(?, code_version), "
                "git_commit = COALESCE(?, git_commit) WHERE run_id = ?",
                (
                    attrs["conventions"],
                    attrs["code_version"],
                    attrs["git_commit"],
                    run_id,
                ),
            )
        if cur.rowcount == 0:
            raise KeyError(f"Unknown run: {run_id}")

    def set_data_status(self, run_id: str, data_status: str) -> None:
        """Update where a run's output data currently lives (or that it's gone)."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET data_status = ? WHERE run_id = ?",
                (data_status, run_id),
            )
        if cur.rowcount == 0:
            raise KeyError(f"Unknown run: {run_id}")

    def relocate_experiment(
        self,
        experiment_id: str,
        new_root: str | Path | None = None,
        *,
        data_status: str = "on_group",
        verify: bool = True,
    ) -> None:
        """Record that an experiment tree has moved to a new data root.

        Call this **after** the tree has been copied/moved (e.g. from
        scratch to group space). The registry update only commits if
        the data verifiably exists at the new root: every run directory
        must be present, and (with *verify*) recorded input-file
        checksums must match the relocated content. On success,
        ``experiments.data_root`` and all runs' ``data_status`` are
        updated in one transaction. ``run_dir`` values are relative and
        unchanged.

        Parameters
        ----------
        experiment_id : str
            The experiment to relocate.
        new_root : str or Path, optional
            The new data root (parent of the experiment directory).
            Defaults to the experiment's recorded
            ``permanent_data_root``.
        data_status : str, optional
            New data location status for all runs (default
            ``"on_group"``).
        verify : bool, optional
            If ``True`` (default), re-checksum each run's recorded
            input files at the new location and require a match.

        Raises
        ------
        FileNotFoundError
            If a run directory or recorded input file is missing at
            the new root.
        ValueError
            If a relocated file's checksum does not match the registry,
            or *new_root* is omitted and the experiment has no
            ``permanent_data_root``.
        """
        if new_root is None:
            new_root = self.get_experiment(experiment_id).get(
                "permanent_data_root"
            )
            if not new_root:
                raise ValueError(
                    f"Experiment {experiment_id!r} has no recorded "
                    "permanent_data_root; pass new_root explicitly."
                )
        new_root = Path(new_root).expanduser().resolve()
        runs = self.query_runs(experiment_id=experiment_id)

        for run in runs:
            if run["data_status"] == "deleted":
                continue
            run_dir = new_root / run["run_dir"]
            if not run_dir.is_dir():
                raise FileNotFoundError(
                    f"Run directory not found at new root: {run_dir}"
                )
            if not verify:
                continue
            rows = self._conn.execute(
                "SELECT file_path, checksum FROM input_files "
                "WHERE run_id = ?",
                (run["run_id"],),
            ).fetchall()
            for row in rows:
                path = run_dir / "inputs" / row["file_path"]
                if not path.exists():
                    raise FileNotFoundError(
                        f"Recorded input file missing after move: {path}"
                    )
                if row["checksum"] and sha256_file(path) != row["checksum"]:
                    raise ValueError(
                        f"Checksum mismatch after move: {path} — "
                        "data may be corrupted; registry not updated."
                    )

        with self._conn:
            cur = self._conn.execute(
                "UPDATE experiments SET data_root = ? WHERE experiment_id = ?",
                (str(new_root), experiment_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Unknown experiment: {experiment_id}")
            self._conn.execute(
                "UPDATE runs SET data_status = ? WHERE experiment_id = ? "
                "AND (data_status IS NULL OR data_status != 'deleted')",
                (data_status, experiment_id),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return one run record as a dict."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown run: {run_id}")
        return dict(row)

    def query_runs(
        self,
        *,
        experiment_id: str | None = None,
        status: str | None = None,
        param: str | None = None,
        group: str | None = None,
        value: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search runs by experiment, status, date range, and/or parameter.

        Parameters
        ----------
        param, group, value : str or None
            Filter to runs whose namelist contains ``param`` (optionally
            within ``group``), optionally with the given value (compared as
            text, e.g. ``value="256"``).
        since, until : str or None
            ISO-8601 bounds on ``created_at``.
        """
        clauses: list[str] = []
        args: list[Any] = []
        if experiment_id is not None:
            clauses.append("r.experiment_id = ?")
            args.append(experiment_id)
        if status is not None:
            clauses.append("r.status = ?")
            args.append(status)
        if since is not None:
            clauses.append("r.created_at >= ?")
            args.append(since)
        if until is not None:
            clauses.append("r.created_at <= ?")
            args.append(until)
        if param is not None:
            sub = "SELECT 1 FROM namelist_parameters np WHERE np.run_id = r.run_id "
            sub += "AND np.param_name = ?"
            args.append(param.lower())
            if group is not None:
                sub += " AND np.group_name = ?"
                args.append(group.lower())
            if value is not None:
                sub += " AND np.param_value = ?"
                args.append(value)
            clauses.append(f"EXISTS ({sub})")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT r.* FROM runs r {where} ORDER BY r.created_at", args
        ).fetchall()
        return [dict(r) for r in rows]

    def run_parameters(self, run_id: str) -> dict[str, dict[str, str]]:
        """Return a run's namelist parameters as {group: {param: value}}."""
        rows = self._conn.execute(
            "SELECT group_name, param_name, param_value FROM namelist_parameters "
            "WHERE run_id = ? ORDER BY group_name, param_name",
            (run_id,),
        ).fetchall()
        params: dict[str, dict[str, str]] = {}
        for row in rows:
            params.setdefault(row["group_name"], {})[row["param_name"]] = row[
                "param_value"
            ]
        return params

    def run_events(self, run_id: str) -> list[dict[str, Any]]:
        """Return a run's full status history, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM status_events WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
