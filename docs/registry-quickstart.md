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
  account: owner-guest
  partition: notchpeak-guest
  cores_per_node: 40
  walltime: "12:00:00"
```

## 2. Create the runs

```python
from codt_tools import ExperimentSpec, create_experiment_runs
from codt_tools.registry import Registry

spec = ExperimentSpec.from_yaml("experiment.yaml")
with Registry("~/codt_registry.db") as reg:
    runner, run_dirs = create_experiment_runs(spec, reg, "~/dev/CODT/codt")
```

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
(archived content-addressed), and `codt --version` output.

## 3. Run

```python
# SLURM (batched cores_per_node runs per job; each run reports status
# back to the registry from inside the job):
job_ids = runner.submit(run_dirs, walltime="12:00:00")

# Or locally, one at a time (blocking):
proc = runner.run_local(spec.expand()[0])
```

Statuses flow `registered → queued → running → completed/failed`;
every transition is appended to `status_events` with timestamp and
hostname.

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

# 2. Verify + update the registry in one step
codt-registry relocate EXP001_tref_sensitivity \
    /uufs/.../group-space/CODT/experiments
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
