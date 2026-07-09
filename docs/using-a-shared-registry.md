# Using a Shared Registry

How to work against a registry you didn't create — a group database, or
your own from another machine or an older codt_tools.

## Pointing at the database

All tools resolve the database from `--db` or the `CODT_REGISTRY_DB`
environment variable:

```bash
export CODT_REGISTRY_DB=/uufs/chpc.utah.edu/common/home/group/codt_registry.db
codt-registry list --status running
```

In Python: `Registry(os.environ["CODT_REGISTRY_DB"])`. If the schema of
the database is **newer** than your codt_tools supports, opening raises
with a message to upgrade; older schemas are migrated in place
automatically (so opening someone else's DB with a newer codt_tools
*writes* to it — coordinate before doing that).

## Read-only querying

For pure analysis on a shared DB, avoid accidental writes and lock
contention by querying through the CLI (`list`, `show`, `export`) or
SQLite's read-only URI mode:

```python
import sqlite3
conn = sqlite3.connect("file:/path/to/registry.db?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
```

Never modify a shared DB with raw `sqlite3` writes — the pragmas,
retry/backoff, and status/event transactional coupling live in the
Python layer. Write through `Registry` or `codt-registry` only.

## The conventions gate

Each output NetCDF carries a `conventions` global attribute (e.g.
`CODT_output_v1`). Your codt_tools declares which conventions it can
read in `codt_tools.registry.SUPPORTED_CONVENTIONS`. The gate is
checked:

- when `CODTSimulation` opens an output file, and
- when `record_completion` stores a run's output metadata.

By default a mismatch **warns** and proceeds (`check_conventions(...,
strict=True)` raises instead). If you hit the warning on someone
else's runs, don't fight it — analyze with a codt_tools version whose
`SUPPORTED_CONVENTIONS` matches the run's `conventions` value (stored
in the `runs` table, so you can check before opening any files).

## WAL and filesystems (CHPC)

The registry uses SQLite WAL mode, which requires POSIX advisory
locks:

- **Fine**: CHPC home directories and group spaces (NFS with working
  locks), local disks.
- **Do not** put the registry on filesystems with unreliable locking,
  and never on scratch (purged).
- Concurrent writers (packed SLURM tasks reporting status) are the
  design case: `busy_timeout=30000` plus retry/backoff in
  `update_status` absorb bursts of ≤ 40 tasks/node. If you see
  `database is locked` errors persistently, check that every writer
  goes through the Python layer/CLI.

## Data location and `data_status` conventions

The DB is authoritative for where run data lives:
`experiments.data_root` (absolute) + `runs.run_dir` (relative). If an
experiment tree is moved, update `data_root` once
(`codt-registry`/`update_experiment`) — run rows don't change.

Keep `data_status` truthful when you touch other people's data areas:

| value | meaning |
|---|---|
| `on_scratch` | live under `data_root` on scratch (purgeable!) |
| `on_group` | moved to group/project space |
| `archived` | in cold storage (e.g. Pando); `data_root` may be stale |
| `deleted` | data gone; params + conclusion remain the record |

## Shared-input symlinks

Run inputs may be **relative** symlinks into the experiment's
`shared_inputs/` directory (content-hash dedup). Relative links keep
the tree relocatable as a unit. Two caveats:

- Archive experiment trees with `tar -h` (follow links) **or** always
  move/archive the whole `{experiment_id}` directory so links stay
  tree-internal.
- `input_files.checksum` is of the resolved content; `is_symlink` /
  `link_target` record the physical layout for audits.
