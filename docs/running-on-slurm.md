# Running CODT on SLURM

**codt_tools does not submit jobs.** It writes a batch script; you read it and
run `sbatch` yourself. Nothing in the package calls `sbatch`, `squeue` or
`sacct`, which means it needs to know nothing about your account, your
partition, or which cluster you are on — and the script is the whole
interface, reviewable before it runs and editable after.

This document is the knowledge that automation used to encode: how to pick a
constraint, what the krueger group is entitled to, and how to size a task.
Companion to `docs/designs.md` (building the ensemble) and
`docs/registry-quickstart.md` (recording it).

## The shape of it

```python
from codt_tools import Case, Run, write_slurm_array

cases = base.sweep({"params.tref": [20.0, 21.0, 22.0]})
runs = Run.for_cases(cases, EXE, "/scratch/general/vast/$USER/EXP002")
for run in runs:
    run.stage()

write_slurm_array(
    runs, "/scratch/general/vast/$USER/EXP002/array.sh",
    runs_per_task=64,
    account="krueger", partition="notchpeak-freecycle",
    qos="notchpeak-freecycle", constraint="rom",
    time="12:00:00", module="gcc/11.2.0",
)
```

Then read the script and submit it:

```bash
sbatch /scratch/general/vast/$USER/EXP002/array.sh
```

Two artifacts are written: the script, and a sibling `array_runs.txt`
manifest holding one whitespace-separated line of run directories per array
task. Each task reads its own line via `$SLURM_ARRAY_TASK_ID`, so the script
stays the same size whether the ensemble has ten runs or a thousand.

Any directive you leave out is written as a `<PLACEHOLDER>`. The script still
parses (`#SBATCH` lines are comments) but SLURM refuses it until you fill it
in, which is the intent — an obvious blank beats a wrong default.

For a local batch instead, `write_local(runs, path, jobs=N)` writes the same
per-run loop throttled to `N` concurrent runs; launch it with
`nohup bash run_all.sh > run_all.log 2>&1 &`.

## Preemption: `--requeue` plus the `_DONE` guard

Both generated scripts wrap each run in a subshell that **skips any run whose
`_DONE` marker already exists**:

```bash
if compgen -G "$RUN_DIR/output/*_DONE" > /dev/null; then
  echo "skip $(basename "$RUN_DIR") -- already complete"
  exit 0
fi
```

That is what makes `#SBATCH --requeue` safe, and the two ship together by
default. A preempted array task is requeued and resumes at the first
unfinished run instead of redoing the batch.

The limit is worth stating plainly: **CODT has no checkpoint/restart**, so an
individual simulation that is interrupted mid-run starts over from the
beginning. The guard works at run granularity, not within a run. On
freecycle, keep individual runs short enough that losing one is cheap.

Re-running a script by hand is idempotent for the same reason — it resumes
rather than redoes, so it is the normal way to mop up a partially completed
ensemble.

## Why architecture matters

CODT ships architecture-tuned builds. An arch-tuned binary is **not
portable**: a `-march=znver2` build raises SIGILL (illegal instruction) on
Intel Skylake/Cascade Lake nodes, because those CPUs do not implement the
instructions gcc emitted for AMD Rome.

Partitions at CHPC are heterogeneous. `notchpeak-freecycle` mixes `rom`,
`skl` and `csl` nodes, so an **unconstrained** zen2 binary submitted there
crashes on roughly half the nodes at random, depending on where SLURM happens
to place each task. Set `constraint=` deliberately.

Both generated scripts run `"$EXE" --version` before the loop, so a
mismatched binary announces itself at the top of the job's `.out` file rather
than as a wall of identical failures.

`codt_tools.check_executable(path)` performs the same probe from Python and
returns a one-line reason or None. Note it judges the binary **on the host it
runs on** — an arch-tuned build can pass on a login node and still SIGILL on
every compute node, so it catches the portable mistakes (wrong path, missing
execute bit, missing runtime library) and not the arch mismatch.

## Finding out what a binary was built for

Read it off the binary. CODT's `fpm.toml` link-time flags embed an
`-Wl,-rpath=` pointing at the spack netCDF build it linked against, and
CHPC's spack tree is named by target microarchitecture:

```bash
$ readelf -d build/gfortran_*/app/CODT | grep -oE 'linux-rocky8-[a-z0-9_]+' | sort -u
linux-rocky8-zen2
```

A baseline (`release` profile) build shows `linux-rocky8-nehalem` instead.

**Caveat.** This reads *which netCDF build was linked*, not CODT's own
`-march`. It is a reliable proxy only because CODT's fpm profiles pair an
`optimized-<tag>` feature with its matching `netcdf-<tag>` atomically — a
documented invariant in CODT's `fpm.toml.template` ("Optional:
architecture-tuned build"). A hand-mixed profile would make the signal lie.
Note `fpm.toml` itself is gitignored (per-checkout, site-specific); the
template is the stable reference.

If the RPATH names more than one architecture, or names none at all, treat
the build as *unknown* — which is deliberately not the same as *portable*.

## Choosing a constraint

Map the architecture token to the SLURM node *features* that can run that
code:

| Build arch | `--constraint` | Notes |
|---|---|---|
| `nehalem`, `x86_64`, `westmere`, `sandybridge` | *(none)* | portable baselines |
| `skylake`, `skylake_avx512` | `skl\|csl` | Cascade Lake runs Skylake code |
| `zen2` | `rom` | notchpeak Rome |
| `zen4` | `gen` | granite Genoa |

**Update this table when CODT adds a build profile.** It mirrors the
`(compiler, arch)` table in CODT's `fpm_env`. A tuple renders as an OR
constraint (`--constraint="skl|csl"`).

ISA-superset relations are deliberately *not* encoded — znver2 code does run
on znver4 hardware, but exact match is the safe default; broaden it yourself
when you mean to.

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
`granite-freecycle`), which is why `qos=` is its own argument.

Some builds need a module for their runtime libraries — the Rome build has no
libgfortran RPATH and cannot start without `module load gcc/11.2.0`. Pass
`module="gcc/11.2.0"` and the script loads it before the loop.

## Sizing a task

`runs_per_task` is how many runs share one array task, executing
concurrently; it is also what `--ntasks` is set to, so **it should not exceed
the cores a task will actually get**. Rome nodes are 64-core, granite Genoa
96, `skl` spans 32 and 36 — size to the *smallest* node the constraint can
land on, or a 36-run task oversubscribes a 32-core node.

Wall time does not depend on it: every simulation in a task runs in parallel,
so a 50-run task and a 64-run task both take one simulation's duration.

It is worth asking for **fewer** cores than a node has. Node sharing is the
CHPC default, so a task asking for all 64 cores of a Rome node can only start
on a completely empty one, while 50 can share with someone else's small job.
Most jobs on these partitions use a few cores, so the smaller request finds a
slot sooner — which matters most on freecycle, where you compete for
leftovers. Spreading 200 runs as `4 x 50` rather than `64, 64, 64, 8` costs
nothing and schedules better.

`array_throttle=N` caps concurrently running tasks (`--array=0-M%N`). Worth
setting on freecycle, where a large burst is both antisocial and more exposed
to preemption.

Memory: `mem="8G"` sets one `#SBATCH --mem` for the task. Node sharing means
leaving it unset silently takes 2G/core; set it deliberately.

## Recording what ran

Registration is an explicit call — no `Run` touches the registry, and no
generated script contains `codt-registry` text. Record runs from Python
before or after submitting, at whatever granularity you want; see
`docs/registry-quickstart.md`.

The registry keeps a `build_arch` column (schema v4), but nothing fills it in
automatically now that the detection table is retired. Pass it yourself with
the token from the `readelf` recipe above if you want it recorded.

## Storage

Point the base directory at VAST (`/scratch/general/vast/$USER/...`), never
`$HOME` or group space: CODT writes NetCDF and binary output per write
interval, and NFS under a packed 64-way node is slow. Copy results to group
space before the 60-day inactivity purge.

## What was retired in 0.9.0

`CODTRunner.submit()` / `.status()`, the `sinfo`/`mychpc` introspection, the
`ARCH_CONSTRAINT` table and the submit-time preflight were removed with
`codt_tools/slurm.py`. Everything they knew is in this document. The code
itself is one command away if it is ever wanted back:

```bash
git show v0.8.0:codt_tools/slurm.py
```
