# Recording what you ran

An optional list of the simulations that were run. One SQLite file, one
table, seven columns.

Nothing else in codt_tools needs it. You can build a case, stage a run, launch
a batch and analyze output without it. You record a run in a separate,
explicit step.

## The whole thing

```python
from pathlib import Path

from codt_tools import Case, Run
from codt_tools.registry import Registry

EXE = Path.home() / "dev/CODT/build/gfortran_2A3B4C5D6E7F8091/app/CODT"
BASE_DIR = Path.home() / "scratch" / "CODT" / "EXP005"

base = Case()
base.set(simulation_name="EXP005", tmax=3600.0)
cases = base.sweep({"params.tref": [20.0, 21.0, 22.0]})

runs = Run.for_cases(cases, EXE, BASE_DIR)
for run in runs:
    run.stage()

with Registry("~/codt_runs.db") as reg:      # created if it doesn't exist
    reg.add_many(runs, tags="EXP005_seeding")
```

Later:

```python
with Registry("~/codt_runs.db") as reg:
    for row in reg.list(tag="EXP005"):
        print(row["run_id"], row["workdir"], row["code_version"])
```

Or from a shell:

```bash
export CODT_REGISTRY_DB=~/codt_runs.db
codt-registry list --tag EXP005
codt-registry show EXP005_000
```

## What a row holds

| Column | What it is |
|---|---|
| `run_id` | the run's name. Defaults to its directory name |
| `workdir` | absolute path to the run directory |
| `created_at` | ISO-8601 UTC, when it was added to the list |
| `executable` | path to the binary |
| `code_version` | `CODT --version`, captured automatically when you add |
| `tags` | free text, for grouping |
| `notes` | free text |

`code_version` is the one thing the registry captures for you. A rebuild
destroys the evidence of which CODT produced a result, and no later step can
recover it. The registry probes the binary once, not once per run.

## What it deliberately does not do

**No status.** There is no queued/running/completed, no event history, no
exit codes. Ask `run.is_complete` whether a run finished. It reads the `_DONE`
marker, which CODT itself writes. That marker is the only thing that is ever
actually true:

```python
done = [r.name for r in runs if r.is_complete]
```

**No experiments.** Grouping is the `tags` string, matched as a substring, so
`list(tag="EXP005")` finds `tags="EXP005_seeding"`. A hypothesis and a
conclusion belong in a README in the experiment directory, where you will
actually read them again — not in a database schema.

**No parameters, checksums or archived binaries.** Each run keeps its own
configuration in its `inputs/` directory, next to the output it produced.
That is where you need it. The registry says a run happened, and where it is.

## The API

```python
Registry(db_path, create=True)
    .add(run, *, tags=None, notes=None, run_id=None) -> str
    .add_many(runs, *, tags=None, notes=None) -> list[str]
    .add_directory(workdir, *, executable=None, tags=None, notes=None) -> str
    .list(tag=None, since=None) -> list[dict]      # newest first
    .get(run_id) -> dict | None
    .remove(run_id) -> bool
    len(reg)
```

`add` is `INSERT OR REPLACE` on `run_id`: re-adding a run updates its row, so
re-staging an ensemble is not an error. `add_directory` records a directory
that has no `Run` object — useful for runs someone else produced.

`create=False` refuses to open a path that does not exist, so a typo in a
shared database path fails instead of silently starting a new, empty list.

## Where to keep it

Keep it on home or group space, never on scratch. The list should outlive the
data it points at. The `--db` flag and `$CODT_REGISTRY_DB` both work. The
environment variable is the convenient default.

It is a plain SQLite file with one table, so anything can read it:

```bash
sqlite3 ~/codt_runs.db "select run_id, code_version from runs;"
```

```python
import sqlite3, pandas as pd
pd.read_sql("select * from runs", sqlite3.connect("codt_runs.db"))
```

codt_tools reads only that one-table shape. It leaves any other database
untouched rather than converting it, so start a new list for new work.
