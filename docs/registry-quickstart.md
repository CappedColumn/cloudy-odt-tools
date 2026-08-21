# Registry Quickstart

The simulation registry tracks CODT experiments and runs in a single
SQLite file: what was run, with which parameters and code version,
where the data lives, and what was concluded. This walkthrough covers
the full lifecycle: **define → create → run → query → validate/move →
conclude**.

## 0. Setup

Pick a database location and (optionally) export it so the CLI and
SLURM jobs find it without `--db`:

```bash
export CODT_REGISTRY_DB=/uufs/chpc.utah.edu/common/home/$USER/codt_registry.db
codt-registry init
```

The registry lives on your **home directory or group space, not
scratch** — it must outlive the run data. WAL mode requires a
filesystem with POSIX locks (home NFS on CHPC is fine; see
[using-a-shared-registry.md](using-a-shared-registry.md) for caveats).

## 1. Define an experiment (`experiment.yaml`)

```yaml
# Unique ID: registry key, directory name, and base simulation name.
experiment_id: EXP001_tref_sensitivity
title: LWC sensitivity to base temperature
hypothesis: >
  Warmer chamber base temperature increases steady-state LWC.

# Recorded with every run.
model: codt
execution_context: notchpeak

# Where the experiment tree is created (runs execute here; scratch is
# the normal choice):
#   {data_root}/{experiment_id}/{experiment.yaml, shared_inputs/, runs/}
data_root: /scratch/general/vast/u1342804/CODT/experiments

# Optional: where the tree should live long-term. Runs go to scratch,
# and after validation the tree is moved here (see step 5a).
permanent_data_root: /uufs/chpc.utah.edu/common/home/group-space/CODT/experiments

# Namelist overrides applied to every run (flat param -> value;
# groups are resolved automatically).
base_parameters:
  tmax: 3600.0
  do_microphysics: true

# Cartesian-product sweep, one run per combination (2 x 2 = 4 runs).
# Empty/omitted means a single run.
parameter_sweep:
  tref: [18.0, 22.0]
  volume_scaling: [13, 50]

slurm_options:
  account: krueger
  partition: notchpeak-freecycle
  qos: notchpeak-freecycle          # partition != qos on some clusters
  walltime: "12:00:00"
  array_throttle: 8                 # cap concurrent array tasks
  mem_per_task: 2G
  # cores_per_node: omit -> resolved from the node type (rom -> 64)
  # constraint:     omit -> resolved from the executable's build arch
```

Node targeting is automatic: the runner detects the executable's build
architecture, constrains the job to nodes that can run it, and packs to
their real core count. See `docs/running-on-slurm.md`.

## 2. Create the runs

```python
from codt_tools import ExperimentSpec, create_experiment_runs
from codt_tools.registry import Registry

spec = ExperimentSpec.from_yaml("experiment.yaml")
with Registry("~/codt_registry.db") as reg:
    runner, run_dirs = create_experiment_runs(spec, reg, "~/dev/CODT/codt")
```

Before anything is written, the executable is gated: it must exist and
report a valid version (`codt --version`). A binary that was not built
properly (missing/placeholder version info) is refused with
instructions to rebuild via `./build.sh` — its runs would be
untraceable.

This registers the experiment (status `planned`), expands the sweep,
and creates:

```
{data_root}/{experiment_id}/
├── experiment.yaml
├── shared_inputs/          # inputs identical across runs (content-hash dedup)
└── runs/{run_id}/
    ├── inputs/             # params.nml + relative symlinks -> shared_inputs/
    └── output/             # CODT writes here
```

Run IDs are `{YYYYMMDD_HHMMSS}_{model}_{descriptor}`, e.g.
`20260709_101500_codt_Tref18.0_VS13`. Each run is registered with all
namelist parameters, input-file checksums, the executable's SHA256
(archived content-addressed), `codt --version` output, and the binary's
target CPU architecture (`build_arch`).

`submit()` refuses a target whose nodes cannot run the executable — an
architecture-tuned binary aimed at a partition without matching hardware
raises rather than producing jobs that hang unschedulable or die with
SIGILL. Pass `force=True` to override.

## 3. Run

```python
# SLURM: the whole ensemble goes as ONE job array, packed cores_per_node
# runs per array task. Walltime defaults to spec.slurm_options["walltime"].
job_ids = runner.submit(run_dirs)      # -> ['4812345'] (one array job id)

# Or locally, one at a time (blocking):
proc = runner.run_local(spec.expand()[0])
```

On submission the experiment flips to `running`. Inside the job each
run reports `running → completed/failed` (with its array task ID,
`{array_job_id}_{task_index}`), and
on success its output metadata (`conventions`, `code_version`,
`git_commit`) is captured immediately — no separate `collect` step is
required for bookkeeping. Every transition is appended to
`status_events` with timestamp and hostname; failures record a
`detail` pointing at the run's `.log` and the batch job's
`CODT_batch_{i}_{jobid}.out` (both under the experiment tree /
`base_output_dir`).

## 4. Query

```bash
codt-registry list --experiment EXP001_tref_sensitivity
codt-registry list --status failed --since 2026-07-01
codt-registry list --param tref --value 18.0
codt-registry show 20260709_101500_codt_Tref18.0_VS13
codt-registry export --experiment EXP001_tref_sensitivity --csv runs.csv
```

Or in Python: `reg.query_runs(...)`, `reg.run_parameters(run_id)`,
`reg.run_events(run_id)`. Analysis starts from the registry, not from
directory listings:

```python
from codt_tools import CODTSimulation

run = reg.get_run("20260709_101500_codt_Tref18.0_VS13")
exp = reg.get_experiment(run["experiment_id"])
sim = CODTSimulation(Path(exp["data_root"]) / run["run_dir"] / "output")
```

Loading output emits a warning if the file's `conventions` attribute
is not supported by your codt_tools version (the *conventions gate* —
see [using-a-shared-registry.md](using-a-shared-registry.md)).

## 5. Move validated data off scratch

The normal lifecycle is **run on scratch → validate/QC → move to
group space**. Once the runs pass quality control (all `_DONE`
markers present, budgets close, output loads cleanly):

```bash
# 1. Move the WHOLE experiment directory (keeps relative symlinks valid)
rsync -a /scratch/.../experiments/EXP001_tref_sensitivity \
    /uufs/.../group-space/CODT/experiments/

# 2. Verify + update the registry (destination defaults to the
#    permanent_data_root recorded at creation)
codt-registry relocate EXP001_tref_sensitivity
```

`relocate` refuses to update the registry unless every run directory
exists at the new root and the recorded input-file checksums match the
relocated content — a botched copy can't silently become the recorded
truth. On success it sets `experiments.data_root` and flips every
run's `data_status` to `on_group` (override with `--data-status`;
skip checksums with `--no-verify`). Only then delete the scratch copy.

The intended destination is recorded up front as
`permanent_data_root` in `experiment.yaml`, so this step needs no
decisions.

## 6. Conclude

```bash
codt-registry experiment conclude EXP001_tref_sensitivity \
    "LWC increases ~8% per K of Tref over 18-22 C." \
    --artifact analysis/lwc_vs_tref.png
```

Concluding records the conclusion text, artifact paths, and timestamp,
and sets the experiment to `concluded`. Once run data is moved or
purged, update `codt-registry set-data-status <run_id> archived`
(or `deleted`) — the parameters and conclusion remain as the permanent
"never rerun this" record.

## When things fail

The workflow is designed so each failure surfaces at a well-defined
point with a specific fix:

| Failure | Where it surfaces | What to change |
|---|---|---|
| Executable missing | `create_experiment_runs` raises `FileNotFoundError` before anything is written | The `executable` argument |
| Improper build (no version info) | `create_experiment_runs` raises `ValueError` before anything is written | Rebuild with `./build.sh`, pass the new binary |
| Duplicate `experiment_id` | `create_experiment_runs` raises `IntegrityError` | Pick a new `experiment_id` in the YAML |
| Invalid namelist values | `cfg.validate()` / CODT rejects at startup (stderr + run `failed`) | `base_parameters` / `parameter_sweep` in the YAML |
| sbatch rejected (bad account/partition) | `runner.submit` raises `CalledProcessError` | `slurm_options` in the YAML |
| Run crashes / exits nonzero | Run marked `failed` with exit code; `detail` names the run's `.log` and the batch `*.out` file | Diagnose from those logs (see `codt-registry show <run_id>`) |
| Job killed (walltime, OOM, node death) | Run stuck in `running`; SLURM job gone | Cross-check `sacct`, backfill status from the `_DONE` marker; raise `walltime` in the YAML and resubmit |
| Botched copy at relocation | `codt-registry relocate` refuses (missing files / checksum mismatch); registry untouched | Re-run the `rsync`, then relocate again |
| Output unreadable by tools | `CODTSimulation` warns on the `conventions` gate | Use a codt_tools version matching the run's recorded `conventions` |
