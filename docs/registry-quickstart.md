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

Node targeting is **not** automatic. Pick the constraint deliberately — see
`docs/running-on-slurm.md` for the architecture table and the krueger group's
entitlements.

## 2. Create the runs

```python
from codt_tools import Case, Run, check_executable
from codt_tools.registry import Registry

EXE = "~/dev/CODT/codt"
assert check_executable(EXE) is None, check_executable(EXE)

base = Case()
base.set(simulation_name="EXP001", tmax=3600.0)
cases = base.sweep({"params.tref": [18.0, 21.0, 24.0]})

with Registry("~/codt_registry.db") as reg:
    reg.create_experiment("EXP001_tref_sensitivity", "Tref sensitivity",
                          data_root="/path/to/data")

    runs = Run.for_cases(
        cases, EXE, "/path/to/data/EXP001_tref_sensitivity/runs"
    )
    for run in runs:
        staged = run.stage()
        reg.register_run(run.name, run.case, run.workdir, namelist=staged,
                         experiment_id="EXP001_tref_sensitivity",
                         executable_path=EXE)
```

> The YAML spec layer (`ExperimentSpec` / `create_experiment_runs`) was removed
> in codt_tools 0.9.0; the YAML above is still a fine way to *record* a design,
> but nothing reads it. Shared-input dedup went with it, and in 0.9.0 the
> `slurm_options` block is documentation only — nothing submits for you.

This creates:

```
{data_root}/{experiment_id}/runs/{run_id}/
    ├── inputs/             # params.nml, aerosol_input.nc, [parcel_input.nc]
    └── output/             # CODT writes here
```

Each run is registered with all
namelist parameters, input-file checksums, the executable's SHA256
(archived content-addressed) and `codt --version` output. The `build_arch`
column exists but nothing fills it in automatically — pass `build_arch=`
yourself if you want it (`docs/running-on-slurm.md` has the `readelf`
recipe).

## 3. Run

```python
from codt_tools import write_slurm_array

# SLURM: generate the array script, then submit it yourself.
script = write_slurm_array(runs, base_dir / "array.sh", runs_per_task=64,
                           account=..., partition=..., time="12:00:00")
#   sbatch <script>

# Or locally, one at a time (blocking):
runs[0].execute_local()
```

Status is **not** reported from inside the job — generated scripts contain no
registry calls. Record transitions yourself around a local run, or reconcile
after a batch from the `_DONE` markers:

```python
for run in runs:
    if run.is_complete:
        reg.update_status(run.name, "completed", exit_code=0)
        reg.record_completion(run.name, run.output_dir / f"{run.case.name}.nc")
```

`record_completion` captures the output's `conventions`, `code_version` and
`git_commit`. Every transition is appended to `status_events` with timestamp
and hostname.

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
from codt_tools import Simulation

run = reg.get_run("20260709_101500_codt_Tref18.0_VS13")
exp = reg.get_experiment(run["experiment_id"])
sim = Simulation(Path(exp["data_root"]) / run["run_dir"] / "output")
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
| Executable missing / not executable / wrong CPU | `check_executable(exe)` returns a one-line reason; `Run.execute_local` raises | The `executable` argument; see `docs/running-on-slurm.md` |
| Improper build (no version info) | `check_executable` reports the `--version` failure | Rebuild with `./build.sh`, pass the new binary |
| Duplicate `experiment_id` | `Registry.create_experiment` raises `IntegrityError` | Pick a new `experiment_id` in the YAML |
| Invalid namelist values | `case.validate()` / CODT rejects at startup (stderr + run `failed`) | `base_parameters` / `parameter_sweep` in the YAML |
| sbatch rejected (bad account/partition) | `sbatch` says so when *you* submit the generated script | The directives passed to `write_slurm_array` |
| Run crashes / exits nonzero | Run marked `failed` with exit code; `detail` names the run's `.log` and the batch `*.out` file | Diagnose from those logs (see `codt-registry show <run_id>`) |
| Job killed (walltime, OOM, node death) | Run stuck in `running`; SLURM job gone | Cross-check `sacct`, backfill status from the `_DONE` marker; raise `walltime` in the YAML and resubmit |
| Botched copy at relocation | `codt-registry relocate` refuses (missing files / checksum mismatch); registry untouched | Re-run the `rsync`, then relocate again |
| Output written in an unknown format | `Simulation` warns (naming expected vs found) and opens it anyway | Nothing, usually — check anything surprising against the file |
