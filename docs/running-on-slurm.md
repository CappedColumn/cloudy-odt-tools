# Running CODT on SLURM

How `CODTRunner` targets nodes, packs them, and submits an ensemble as a
single job array. Companion to `docs/registry-quickstart.md`, which covers
the experiment/registry side.

## Why architecture matters

CODT ships architecture-tuned builds. An arch-tuned binary is **not
portable**: a `-march=znver2` build raises SIGILL (illegal instruction) on
Intel Skylake/Cascade Lake nodes, because those CPUs do not implement the
instructions gcc emitted for AMD Rome.

Partitions at CHPC are heterogeneous. `notchpeak-freecycle` mixes `rom`,
`skl` and `csl` nodes, so an unconstrained zen2 binary submitted there
crashes on roughly half the nodes at random, depending on where SLURM
happens to place each task.

`CODTRunner` therefore detects what the binary was built for and constrains
the job to matching nodes.

## How the architecture is detected

It is read off the binary — no CODT source change, and it works
retroactively on binaries that already exist.

CODT's `fpm.toml` link-time flags embed an `-Wl,-rpath=` pointing at the
spack netCDF build it linked against, and CHPC's spack tree is named by
target microarchitecture:

```bash
$ readelf -d build/gfortran_*/app/CODT | grep -oE 'linux-rocky8-[a-z0-9_]+' | sort -u
linux-rocky8-zen2
```

A baseline (`release` profile) build shows `linux-rocky8-nehalem` instead.

`codt_tools.slurm.detect_build_arch()` does exactly this and returns the
token (`"zen2"`).

**Caveat.** This reads *which netCDF build was linked*, not CODT's own
`-march`. It is a reliable proxy only because CODT's fpm profiles pair an
`optimized-<tag>` feature with its matching `netcdf-<tag>` atomically — a
documented invariant in CODT's `fpm.toml.template` ("Optional:
architecture-tuned build"). A hand-mixed profile would make the signal lie.
Note `fpm.toml` itself is gitignored (per-checkout, site-specific); the
template is the stable reference.

If the RPATH names more than one architecture, or names none at all,
detection returns `None` — *unknown*, which is deliberately not the same as
*portable*. No constraint is applied, and `validate_target` warns if the
target partition mixes CPU types.

## The ARCH_CONSTRAINT table

`codt_tools/slurm.py` maps each spack architecture token to the SLURM node
*features* that can run that code:

```python
ARCH_CONSTRAINT = {
    "nehalem": None, "x86_64": None,        # portable baselines
    "westmere": None, "sandybridge": None,
    "skylake": ("skl", "csl"),              # Cascade Lake runs Skylake code
    "skylake_avx512": ("skl", "csl"),
    "zen2": ("rom",),                       # notchpeak Rome
    "zen4": ("gen",),                       # granite Genoa
}
```

**This table is the one place to update when CODT adds a build profile.** It
is the codt_tools-side mirror of the `(compiler, arch)` table in CODT's
`fpm_env`. A tuple renders as an OR constraint (`--constraint="skl|csl"`).

ISA-superset relations are deliberately *not* encoded — znver2 code does run
on znver4 hardware, but exact match is the safe default. Broadening is an
explicit user override via `constraint=`.

## Entitlement for the krueger group

`myallocation` / `mychpc batch` are authoritative; copy the
partition/qos/account triples from them verbatim. As of 2026-08:

| Target | Route | Notes |
|---|---|---|
| notchpeak Rome (`rom`, 64-core) | **freecycle only** | No general notchpeak allocation. `--partition=notchpeak-freecycle --qos=notchpeak-freecycle --account=krueger`. Preemptable. |
| `notchpeak-shared-short` | general, no allocation needed | **No Rome nodes** (`csl`, `npl` only) — a zen2 binary here is unschedulable by construction. ≤ 8 h. |
| granite Genoa (`gen`, 96-core) | preemptable, `--account=krueger --qos=granite-freecycle` | Needs a **zen4** CODT build. |
| kingspeak / lonepeak | general | Older Intel; baseline build. |

Note partition ≠ qos on granite (partition `granite`, qos
`granite-freecycle`), which is why `qos` is a separate constructor argument.

**Preemption is not handled.** CODT has no checkpoint/restart, so a
preempted simulation loses all its work. There is no `--requeue` and no
skip-if-`_DONE` logic; that needs its own design. On freecycle, size
`walltime` accordingly and expect to re-run losses.

## Automatic resolution

`CODTRunner.__init__` resolves three things, unless you pass them
explicitly (an explicit value always wins):

| Attribute | Resolved from |
|---|---|
| `constraint` | `ARCH_CONSTRAINT[build_arch]` |
| `cores_per_node` | the **minimum** core count over nodes in the partition matching the constraint |
| `cluster` | `sinfo -M all`, for `sbatch -M` |

The minimum core count is deliberate: `skl` spans 32- and 36-core nodes, so
packing 36 would oversubscribe the 32-core ones.

When introspection is unavailable (off-cluster, no `sinfo`), packing falls
back to 40 tasks per node and no constraint — the historical behavior.

```python
runner = CODTRunner(
    executable="~/dev/CODT/build/gfortran_*/app/CODT",
    base_output_dir="/scratch/general/vast/$USER/EXP002",
    account="krueger",
    partition="notchpeak-freecycle",
    qos="notchpeak-freecycle",
)
runner.build_arch      # 'zen2'
runner.constraint      # 'rom'
runner.cores_per_node  # 64
runner.cluster         # 'notchpeak'
```

## Preflight

`submit()` validates the target *before* writing anything:

1. **Entitlement** — the (partition, account, qos) triple appears in
   `mychpc batch`.
2. **Schedulability** — the partition actually has nodes carrying the
   required feature. This is what catches a zen2 binary aimed at
   `notchpeak-shared-short`.
3. **Silent-SIGILL risk** — unknown build architecture on a partition that
   mixes CPU types.

Any problem raises `ValueError` naming the offending triple and suggesting a
fix. `submit(..., force=True)` downgrades these to warnings — an escape
hatch, not a default.

Introspection that is unavailable yields no problems, rather than false
failures, so off-cluster use and tests keep working.

## Job arrays

An ensemble submits as **one** job array rather than N loose `sbatch` calls.
Runs are packed `cores_per_node` to an array task, each pinned to its own
core with `taskset`.

The batch→run mapping lives in a **manifest** next to the script
(`{base_output_dir}/slurm/{job_name}.manifest`, one line per array index),
so the script stays small no matter how many runs there are:

```bash
mapfile -t BATCHES < ".../job.manifest"
read -ra RUNS <<< "${BATCHES[$SLURM_ARRAY_TASK_ID]}"
for i in "${!RUNS[@]}"; do
  ( ... taskset -c "$i" CODT "${RUNS[$i]}/inputs/params.nml" ... ) &
done
wait
```

Scripts, manifests and array `.out` files go under
`{base_output_dir}/slurm/`, named `{experiment_id}_{timestamp}`, so repeat
or concurrent submits never overwrite each other.

`array_throttle` caps concurrently running tasks (`--array=0-N%K`). Worth
setting on freecycle, where a large burst is both antisocial and more
exposed to preemption.

One array header covers every task, so `--ntasks` is sized to the
**largest** batch. Runs are therefore spread *evenly* across tasks rather
than filling each node to capacity and leaving a remainder: 200 runs at 64
cores/node batch as `[50, 50, 50, 50]`, not `[64, 64, 64, 8]`. Same number
of tasks, but `--ntasks` is 50 instead of 64 and no task reserves cores it
will not use. `cores_per_node` therefore sets the *capacity* of a task, not
its exact size.

Wall time is unaffected — every simulation in a task runs in parallel, so a
50-run task and a 64-run task both take one simulation's duration.

The even split also **schedules better under node sharing**. A task asking
for all 64 cores of a Rome node can only start on a completely empty node;
at 50 it can share with someone else's small job. Since most jobs on these
partitions use only a few cores, the smaller request finds a slot sooner —
which matters most on freecycle, where you are competing for whatever is
left over.

```python
run_dirs = runner.setup_runs(configs)
job_ids = runner.submit(run_dirs, walltime="12:00:00", array_throttle=8)
# ['4812345'] — a single array job id
```

Each run is recorded in the registry with `slurm_job_id = "{array}_{task}"`,
so a run maps to the actual array task that ran it.

## From an experiment spec

```yaml
slurm_options:
  account: krueger
  partition: notchpeak-freecycle
  qos: notchpeak-freecycle
  walltime: "12:00:00"
  array_throttle: 8
  mem_per_task: 2G
  # cores_per_node:  omit -> resolved from the node type (rom -> 64)
  # constraint:      omit -> resolved from the executable's build arch
  # cluster:         omit -> resolved from sinfo
```

`mem_per_task` is scaled by task count into a single `#SBATCH --mem`. Node
sharing is the CHPC default, so leaving it unset silently takes 2G/core;
setting it explicitly is recommended.

## Provenance

The detected architecture is recorded per run in the registry as
`build_arch` (schema v4), alongside `code_version` and `git_commit`. It
shows up in `codt-registry show <run_id>` and `codt-registry export --csv`.
Runs registered before v4 read `build_arch` NULL — the architecture of a
past run's binary is not recoverable after the fact.

## Storage

Point `base_output_dir` at VAST (`/scratch/general/vast/$USER/...`), never
`$HOME` or group space: CODT writes NetCDF and binary output per write
interval, and NFS under a packed 64-way node is slow. Copy results to group
space before the 60-day inactivity purge.
