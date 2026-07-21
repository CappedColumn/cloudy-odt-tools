# Registry Schema

SQLite database, six tables. Schema is versioned via
`PRAGMA user_version` (current version: see
`codt_tools.registry.SCHEMA_VERSION`); `connect()` applies pending
migrations from the `MIGRATIONS` dict automatically on open. Every
connection sets `journal_mode=WAL`, `busy_timeout=30000`,
`foreign_keys=ON`, `synchronous=NORMAL` — always open the database
through `codt_tools.registry.connect()` / the `Registry` class / the
`codt-registry` CLI, never raw `sqlite3` (the pragmas and retry logic
live in Python).

All timestamps are ISO-8601 UTC text.

## Tables

### experiments

One row per hypothesis-driven experiment.

| Column | Meaning |
|---|---|
| `experiment_id` | PK, e.g. `EXP001_tref_sensitivity` |
| `title` | one-line title |
| `hypothesis` | the question being answered |
| `status` | `planned` / `running` / `analyzed` / `concluded` (CHECK) |
| `conclusion` | free text, set by `conclude_experiment` |
| `analysis_artifacts` | JSON list of artifact paths |
| `data_root` | root under which the experiment tree currently lives |
| `permanent_data_root` | intended post-QC home; default destination for `relocate` (schema v2) |
| `created_at`, `concluded_at` | timestamps |

### runs

One row per simulation. Indexes on `experiment_id` and `status`.

| Column | Meaning |
|---|---|
| `run_id` | PK, `{YYYYMMDD_HHMMSS}_{model}_{descriptor}` |
| `experiment_id` | FK → experiments (nullable for standalone runs) |
| `model_name` | default `codt` |
| `descriptor` | the run's sweep point only, e.g. `Tref18.0_VS13` |
| `execution_context` | e.g. `local`, `slurm:notchpeak` |
| `run_dir` | **relative to `experiments.data_root`** (absolute for standalone runs) |
| `status` | `registered`/`queued`/`running`/`completed`/`failed`/`collected` (CHECK) |
| `created_at`, `started_at`, `completed_at` | lifecycle timestamps |
| `exit_code`, `slurm_job_id` | from execution |
| `conventions`, `code_version`, `git_commit` | read from output NC attrs at completion |
| `git_branch`, `build_info` | optional build provenance |
| `codt_tools_version` | recorded at registration |
| `executable_path`, `executable_checksum` | SHA256 of the binary |
| `executable_archive_path` | content-addressed archive copy |
| `data_status` | `on_scratch`/`on_group`/`archived`/`deleted` (CHECK) |
| `notes` | free text |

`data_root` + `run_dir` make the DB authoritative for data location:
relocating an experiment updates one field, and `data_status` answers
"which concluded experiments still have data on scratch" before a
purge.

### input_files

One row per input file per run. `UNIQUE(run_id, file_path)`; CASCADE
on run deletion.

| Column | Meaning |
|---|---|
| `file_type` | `namelist` / `aerosol_input` / `parcel_input` |
| `file_path` | relative to the run's `inputs/` |
| `checksum` | SHA256 **of resolved content** (through symlinks) |
| `size_bytes` | resolved size |
| `is_symlink`, `link_target` | physical layout after shared-input dedup (`link_target` relative, e.g. `../../../shared_inputs/aerosol_input.nc`) |
| `schema_conventions` | the file's NetCDF `conventions` attribute; NULL for the namelist and for rows registered before schema v3 |
| `has_seed_group` | aerosol only: 1/0 for seed-group presence; NULL for other file types and pre-v3 rows |

#### Input schema gating

`SUPPORTED_INPUT_CONVENTIONS` (`registry/versions.py`) is the source of truth
for readable input formats, alongside `SUPPORTED_CONVENTIONS` for outputs. The
two input types behave differently, and the asymmetry is deliberate:

- **`parcel_input`** — the version string is load-bearing. Only
  `CODT_parcel_input_v3` is accepted. v1/v2 are *retired*, not merely old: v3
  re-keys the segment lookup to a leg counter and redefines `ent_rate` as 1/km
  (a 1000× change), so an old file is silently wrong rather than unreadable.
  `check_input_conventions` refuses them with that explanation.
- **`aerosol_input`** — the version string is inert. `CODT_aerosol_input_v1` is
  the only string there has ever been, and it is still current: CODT v3 added
  seeding as *optional* dims/vars rather than bumping the schema
  (`droplets.f90:1243` hard-rejects anything else). Seeded and unseeded files
  are both v1, so the version cannot distinguish them — `has_seed_group` is
  what carries that information.

Seed-group detection probes for the `seed_bin` dimension, mirroring CODT's
`read_seed_group`, which returns early when that dimension is absent. The group
is all-or-nothing: once `seed_bin` exists, the remaining seed dims and variables
are mandatory.

`register_run` also cross-checks `&MICROPHYSICS do_seeding` against the staged
aerosol file (`check_seeding_consistency`). `do_seeding` is the sole controller,
so the gate is **one-way**: `do_seeding=.true.` with no seed group is fatal
(nothing to seed), but `do_seeding=.false.` with a group present is **fine** —
CODT ignores the dormant group, which is what lets one file serve both a seeded
and an unseeded run. Only the fatal direction warns. `CODTConfig.validate()`
makes the same check against the in-memory config (and warns, not raises, on a
dormant group); the registry check is against the file actually on disk, which
additionally catches a shared or symlinked aerosol input swapped after the
config was built.

The fatal-direction check warns by default and raises
`IncompatibleConventionsError` under `strict=True`.

### namelist_parameters

Every parameter of every run, for cross-run SQL queries.
`UNIQUE(run_id, group_name, param_name)`; index on
`(param_name, param_value)`; CASCADE.

Values are stored as text with `param_type`
(`integer`/`real`/`logical`/`character`/`array`) for interpretation.

### status_events

Append-only status history: `run_id`, `status`, `timestamp`,
`hostname`, `slurm_job_id`, `exit_code`, `detail`. The `runs.status`
column is the current state; this table is the audit trail.

### output_files

Created now, populated later by a planned inventory step (run_id,
file_name, size_bytes, checksum).

## Example queries

```sql
-- All failed runs of an experiment, with exit codes
SELECT run_id, exit_code, completed_at FROM runs
WHERE experiment_id = 'EXP001_tref_sensitivity' AND status = 'failed';

-- Every run ever done with tref = 18.0
SELECT r.run_id, r.status, r.code_version
FROM runs r JOIN namelist_parameters p USING (run_id)
WHERE p.param_name = 'tref' AND p.param_value = '18.0';

-- Concluded experiments whose data is still on scratch (purge check)
SELECT DISTINCT e.experiment_id FROM experiments e
JOIN runs r USING (experiment_id)
WHERE e.status = 'concluded' AND r.data_status = 'on_scratch';

-- Status timeline of one run
SELECT status, timestamp, hostname, detail FROM status_events
WHERE run_id = ? ORDER BY id;
```

## Migration policy

- Schema changes bump `SCHEMA_VERSION` and add an entry to
  `MIGRATIONS` in `codt_tools/registry/db.py` (`old_version -> SQL`).
- `connect()` applies migrations in order inside a transaction and
  stamps the new `user_version`.
- Never edit the schema of an existing database by hand; never
  downgrade. Back up the `.db` file before major codt_tools upgrades
  (it is a single file; `cp` while no writers are active).
