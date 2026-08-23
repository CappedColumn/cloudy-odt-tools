"""A list of the simulations that were run.

One SQLite file, one table, seven columns. That is the whole design.

What this deliberately does **not** do: track a run's status while it
executes, model experiments as entities, store namelist parameters, checksum
inputs, or archive executables. Those made sense when the registry drove the
workflow; nothing drives the workflow from here any more. A run's inputs are
in its own ``inputs/`` directory, and whether it finished is
:attr:`~codt_tools.run.Run.is_complete`, which reads the ``_DONE`` marker —
the only thing that is ever actually true.

Examples
--------
>>> from codt_tools.registry import Registry
>>> with Registry("~/codt_runs.db") as reg:       # created if absent
...     reg.add_many(runs, tags="EXP005_seeding")
...     for row in reg.list(tag="EXP005"):
...         print(row["run_id"], row["code_version"])
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Union

from codt_tools.run import codt_version

if TYPE_CHECKING:
    from codt_tools.run import Run

#: The one table. Seven columns, no indexes, no migration machinery.
SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,   -- the run's name
    workdir      TEXT NOT NULL,      -- absolute path to the run directory
    created_at   TEXT NOT NULL,      -- ISO-8601 UTC, when it was added
    executable   TEXT,               -- the binary's path
    code_version TEXT,               -- CODT --version, captured at add time
    tags         TEXT,               -- free text, for grouping
    notes        TEXT                -- free text
);
"""

#: Stamped once at creation and never read. A free hook for a future
#: maintainer that costs no versioning code today.
USER_VERSION: int = 1

COLUMNS: tuple[str, ...] = (
    "run_id", "workdir", "created_at", "executable",
    "code_version", "tags", "notes",
)


class Registry:
    """A list of runs, backed by one SQLite file.

    Parameters
    ----------
    db_path : str or Path
        The database file. Created, with its parent directories, if it does
        not exist and *create* is True.
    create : bool, optional
        If False, opening a path that does not exist raises rather than
        creating an empty database — useful when a typo in a shared DB path
        would otherwise silently start a new, empty list.

    Examples
    --------
    >>> with Registry("~/codt_runs.db") as reg:
    ...     reg.add(run, tags="EXP005")
    'EXP005_000'

    Notes
    -----
    Keep the database on home or group space, never on scratch — the list
    should outlive the data it points at.
    """

    def __init__(
        self, db_path: Union[str, Path], create: bool = True
    ) -> None:
        self.db_path: Path = Path(db_path).expanduser()
        if not self.db_path.exists():
            if not create:
                raise FileNotFoundError(
                    f"No registry at {self.db_path}. Pass create=True to "
                    f"start a new one."
                )
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        # WAL so a reader is never blocked by a writer; foreign keys are
        # irrelevant with one table.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.execute(f"PRAGMA user_version={USER_VERSION}")
        self._conn.commit()

        # One --version probe per executable, not per run.
        self._version_cache: dict[str, str | None] = {}

    # ------------------------------------------------------------------
    # Adding
    # ------------------------------------------------------------------

    def add(
        self,
        run: "Run",
        *,
        tags: str | None = None,
        notes: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Record that a run exists.

        Parameters
        ----------
        run : Run
            The run. Its name, working directory and executable are read
            off it; nothing is executed except one cached ``--version``.
        tags : str, optional
            Free text for grouping, e.g. an experiment name. Matched as a
            substring by :meth:`list`.
        notes : str, optional
            Free text.
        run_id : str, optional
            Identifier, defaulting to ``run.name`` (the directory name).

        Returns
        -------
        str
            The ``run_id`` recorded.

        Notes
        -----
        Re-adding the same ``run_id`` **updates** its row rather than
        raising: re-staging an ensemble should not be an error.
        """
        identifier = run_id if run_id is not None else run.name
        self._conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, workdir, created_at, executable, code_version, "
            " tags, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                identifier,
                str(run.workdir),
                _utcnow(),
                str(run.executable),
                self._version_of(run.executable),
                tags,
                notes,
            ),
        )
        self._conn.commit()
        return identifier

    def add_many(
        self,
        runs: Iterable["Run"],
        *,
        tags: str | None = None,
        notes: str | None = None,
    ) -> list[str]:
        """Record a whole ensemble. One line after staging one.

        Parameters
        ----------
        runs : iterable of Run
            The runs to record.
        tags, notes : str, optional
            Applied to every run.

        Returns
        -------
        list of str
            The recorded ``run_id`` values, in order.
        """
        return [self.add(run, tags=tags, notes=notes) for run in runs]

    def add_directory(
        self,
        workdir: Union[str, Path],
        *,
        executable: Union[str, Path, None] = None,
        tags: str | None = None,
        notes: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Record a run directory directly, without a :class:`Run` object.

        For listing runs someone else produced, or ones staged before this
        registry existed. ``code_version`` is captured only if *executable*
        is given.

        Parameters
        ----------
        workdir : str or Path
            The run directory. Resolved absolute.
        executable : str or Path, optional
            The binary that produced it.
        tags, notes : str, optional
            Free text.
        run_id : str, optional
            Identifier, defaulting to the directory's name.

        Returns
        -------
        str
            The ``run_id`` recorded.
        """
        path = Path(workdir).expanduser().resolve()
        self._conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, workdir, created_at, executable, code_version, "
            " tags, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id if run_id is not None else path.name,
                str(path),
                _utcnow(),
                str(Path(executable).expanduser().resolve())
                if executable is not None else None,
                self._version_of(executable) if executable is not None
                else None,
                tags,
                notes,
            ),
        )
        self._conn.commit()
        return run_id if run_id is not None else path.name

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def list(
        self,
        tag: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """The runs on record, newest first.

        Parameters
        ----------
        tag : str, optional
            Keep only runs whose ``tags`` contain this substring, so
            ``tag="EXP005"`` matches ``tags="EXP005_seeding"``.
        since : str, optional
            Keep only runs added on or after this ISO-8601 date or
            timestamp, e.g. ``"2026-08-01"``.

        Returns
        -------
        list of dict
            One dict per run, with the seven columns as keys.
        """
        sql = "SELECT * FROM runs"
        clauses: list[str] = []
        values: list[Any] = []
        if tag is not None:
            clauses.append("tags LIKE ?")
            values.append(f"%{tag}%")
        if since is not None:
            clauses.append("created_at >= ?")
            values.append(since)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, run_id"

        return [dict(row) for row in self._conn.execute(sql, values)]

    def get(self, run_id: str) -> dict[str, Any] | None:
        """One run by identifier, or None if it is not on record."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def __len__(self) -> int:
        """How many runs are on record."""
        return self._conn.execute("SELECT count(*) FROM runs").fetchone()[0]

    # ------------------------------------------------------------------
    # Removing
    # ------------------------------------------------------------------

    def remove(self, run_id: str) -> bool:
        """Drop a run from the list.

        Removes the row only. Nothing on disk is touched.

        Returns
        -------
        bool
            True if a row was removed, False if there was none.
        """
        cursor = self._conn.execute(
            "DELETE FROM runs WHERE run_id = ?", (run_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _version_of(self, executable: Union[str, Path]) -> str | None:
        """``CODT --version`` for this binary, probed at most once."""
        key = str(executable)
        if key not in self._version_cache:
            self._version_cache[key] = codt_version(executable)
        return self._version_cache[key]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "Registry":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Registry('{self.db_path}', {len(self)} runs)"


def _utcnow() -> str:
    """Current UTC time, ISO-8601, to the second."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
