"""
SQLite connection handling and schema for the CODT simulation registry.

The schema is versioned via ``PRAGMA user_version``. ``connect()`` opens a
connection with the pragmas required for safe concurrent access (WAL mode,
busy timeout, foreign keys) and applies any pending migrations.

Notes
-----
WAL mode requires working POSIX file locks. CHPC home-directory NFS supports
these; do not place the database on filesystems with broken locking.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION: int = 3

# Milliseconds to wait on a locked database before failing. Bursts of tiny
# status writes arrive when up to 40 packed SLURM tasks finish on one node.
BUSY_TIMEOUT_MS: int = 30_000

RUN_STATUSES: tuple[str, ...] = (
    "registered",
    "queued",
    "running",
    "completed",
    "failed",
    "collected",
)

EXPERIMENT_STATUSES: tuple[str, ...] = (
    "planned",
    "running",
    "analyzed",
    "concluded",
)

DATA_STATUSES: tuple[str, ...] = (
    "on_scratch",
    "on_group",
    "archived",
    "deleted",
)


def _quoted(values: tuple[str, ...]) -> str:
    """Render a tuple of strings as a quoted SQL list for CHECK constraints."""
    return ", ".join(f"'{v}'" for v in values)


SCHEMA_SQL: str = f"""
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    hypothesis TEXT,
    status TEXT NOT NULL DEFAULT 'planned'
        CHECK (status IN ({_quoted(EXPERIMENT_STATUSES)})),
    conclusion TEXT,
    analysis_artifacts TEXT,          -- JSON list of paths
    data_root TEXT,
    permanent_data_root TEXT,         -- intended post-QC home (v2)
    created_at TEXT NOT NULL,         -- ISO-8601 UTC
    concluded_at TEXT
);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    model_name TEXT NOT NULL DEFAULT 'codt',
    descriptor TEXT,
    execution_context TEXT,           -- 'local' | 'slurm:<cluster>'
    run_dir TEXT NOT NULL,            -- relative to experiments.data_root
    status TEXT NOT NULL DEFAULT 'registered'
        CHECK (status IN ({_quoted(RUN_STATUSES)})),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    exit_code INTEGER,
    slurm_job_id TEXT,
    conventions TEXT,                 -- from output attrs at completion
    code_version TEXT,
    git_commit TEXT,
    git_branch TEXT,
    codt_tools_version TEXT,
    executable_path TEXT,
    executable_checksum TEXT,
    executable_archive_path TEXT,
    build_info TEXT,
    data_status TEXT
        CHECK (data_status IS NULL OR data_status IN ({_quoted(DATA_STATUSES)})),
    notes TEXT
);
CREATE INDEX idx_runs_experiment ON runs(experiment_id);
CREATE INDEX idx_runs_status ON runs(status);

CREATE TABLE input_files (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    file_type TEXT NOT NULL,          -- 'namelist' | 'aerosol_input' | 'parcel_input'
    file_path TEXT NOT NULL,          -- relative to the run's inputs/ directory
    checksum TEXT,                    -- SHA256 of resolved content
    size_bytes INTEGER,
    is_symlink INTEGER NOT NULL DEFAULT 0,
    link_target TEXT,                 -- relative, e.g. ../../shared_inputs/x.nc
    schema_conventions TEXT,          -- NetCDF 'conventions' attr (v3)
    has_seed_group INTEGER,           -- aerosol only; NULL for other types (v3)
    UNIQUE (run_id, file_path)
);

CREATE TABLE namelist_parameters (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    group_name TEXT NOT NULL,
    param_name TEXT NOT NULL,
    param_value TEXT NOT NULL,
    param_type TEXT
        CHECK (param_type IN ('integer', 'real', 'logical', 'character', 'array')),
    UNIQUE (run_id, group_name, param_name)
);
CREATE INDEX idx_nlp_param ON namelist_parameters(param_name, param_value);

CREATE TABLE status_events (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    hostname TEXT,
    slurm_job_id TEXT,
    exit_code INTEGER,
    detail TEXT
);
CREATE INDEX idx_events_run ON status_events(run_id);

-- Phase 1.5: populated by a future inventory-outputs step.
CREATE TABLE output_files (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    size_bytes INTEGER,
    checksum TEXT
);
"""

# Sequential migrations: MIGRATIONS[n] upgrades a version-n database to n+1.
# Version 0 (fresh database) is initialized directly from SCHEMA_SQL.
MIGRATIONS: dict[int, str] = {
    1: "ALTER TABLE experiments ADD COLUMN permanent_data_root TEXT;",
    # v3: record input-file schema identity. Existing rows stay NULL — the
    # conventions of an already-registered file are not recoverable here, and
    # NULL correctly reads as "not recorded" rather than "absent".
    2: (
        "ALTER TABLE input_files ADD COLUMN schema_conventions TEXT;"
        "ALTER TABLE input_files ADD COLUMN has_seed_group INTEGER;"
    ),
}


def connect(db_path: str | Path, create: bool = True) -> sqlite3.Connection:
    """
    Open a registry database, applying pragmas and pending migrations.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    create : bool
        If True, initialize the schema when the database is new. If False,
        raise ``FileNotFoundError`` for a missing database file.

    Returns
    -------
    sqlite3.Connection
        Connection with WAL journal mode, 30 s busy timeout, foreign keys
        enabled, and ``sqlite3.Row`` row factory.

    Raises
    ------
    FileNotFoundError
        If the database does not exist and ``create`` is False.
    RuntimeError
        If the database schema version is newer than this code supports.
    """
    db_path = Path(db_path).expanduser()
    if not create and not db_path.exists():
        raise FileNotFoundError(f"Registry database not found: {db_path}")

    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")

    _apply_migrations(conn)
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Initialize a fresh schema or upgrade an older one to SCHEMA_VERSION."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Registry schema version {version} is newer than supported "
            f"({SCHEMA_VERSION}); upgrade codt_tools."
        )

    with conn:  # single transaction for the whole upgrade
        if version == 0:
            conn.executescript(SCHEMA_SQL)
        else:
            for step in range(version, SCHEMA_VERSION):
                conn.executescript(MIGRATIONS[step])
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
