# Running CODT on SLURM

**codt_tools does not submit jobs.** It writes a batch script. You read it and
run `sbatch` yourself. Nothing in the package calls `sbatch`, `squeue` or
`sacct`. The package therefore needs to know nothing about your account, your
partition, or your cluster. The script is the whole interface. You can review
it before it runs, and edit it afterwards.

This document covers generating the script, sizing a task, and surviving
preemption. It is a companion to `docs/designs.md`, which builds the ensemble,
and `docs/registry-quickstart.md`, which records it.

Every value below that names an account, a partition, a queue or a node
feature comes from **your** cluster. Take them from your site's documentation
or from whoever administers the machine. codt_tools passes them through
without interpreting them.

## The shape of it

```python
from codt_tools import Case, Run, write_slurm_array

cases = base.sweep({"params.tref": [20.0, 21.0, 22.0]})
runs = Run.for_cases(cases, EXE, BASE_DIR)
for run in runs:
    run.stage()

write_slurm_array(
    runs, BASE_DIR / "array.sh",
    runs_per_task=16,
    account="<your account>", partition="<your partition>",
    time="12:00:00",
)
```

Then read the script and submit it:

```bash
sbatch <base_dir>/array.sh
```

`write_slurm_array` also takes `qos`, `constraint`, `mem`, `module`,
`array_throttle` and `requeue`, plus arbitrary `**directives` for any other
`#SBATCH` line. Pass whichever your cluster needs and omit the rest.

The call writes two artifacts. The first is the script. The second is a
sibling `array_runs.txt` manifest, which holds one whitespace-separated line
of run directories per array task. Each task reads its own line via
`$SLURM_ARRAY_TASK_ID`. The script therefore stays the same size whether the
ensemble has ten runs or a thousand.

Any directive you leave out becomes a `<PLACEHOLDER>`. The script still parses,
because `#SBATCH` lines are comments. SLURM refuses it until you fill the
placeholder in. That is the intent: an obvious blank beats a wrong default.

For a local batch instead, use `write_local(runs, path, jobs=N)`. It writes the
same per-run loop, throttled to `N` concurrent runs. Launch it with
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

The limit is worth stating plainly. **CODT has no checkpoint/restart.** An
interruption mid-run therefore restarts that simulation from the beginning.
The guard works at run granularity, not within a run. On a preemptable queue,
keep individual runs short enough that losing one is cheap.

Submitting a script again is safe for the same reason. It resumes instead of
repeating work, so it is the normal way to complete a partial ensemble.

## Checking the binary runs

Both generated scripts run `"$EXE" --version` before the loop, so a binary that
cannot start announces itself at the top of the job's `.out` file rather than
as a wall of identical failures.

`codt_tools.check_executable(path)` performs the same probe from Python and
returns a one-line reason or None. It judges the binary **on the host it runs
on**, which catches the portable mistakes: a wrong path, a missing execute bit,
a missing runtime library.

If your cluster has more than one kind of compute node and your CODT was built
with optimizations for one of them, the binary may start on a login node and
still fail on a compute node. Use `constraint=` to restrict the job to nodes
the binary can run on. Your cluster's documentation names the available node
features. If some builds need a module for their runtime libraries, pass
`module="<module name>"` and the script loads it before the loop.

## Naming your account and partition

`account`, `partition` and `qos` are cluster-specific strings, and codt_tools
passes them through untouched. Take them from your site's documentation or
from your cluster's own reporting tools, and copy them verbatim.

Note that a partition and a qos are not always the same string, which is why
`qos=` is a separate argument. Pass it only when your cluster needs one.

## Sizing a task

`runs_per_task` is how many runs share one array task and run concurrently.
The script also passes it to `--ntasks`, so **it should not exceed the cores a
task will actually get**. If a `constraint` lets the job land on nodes of
different sizes, size the task to the *smallest* one. Otherwise a task can
oversubscribe the node it happens to get.

Wall time does not depend on it. Every simulation in a task runs in parallel,
so a 50-run task and a 64-run task both take one simulation's duration.

It is worth asking for **fewer** cores than a node has, on any cluster that
shares nodes between jobs. A task asking for every core of a node can only
start on a completely empty one, while a slightly smaller task can share with
someone else's small job and so finds a slot sooner. Spreading 200 runs as
`4 x 50` rather than `64, 64, 64, 8` costs nothing and schedules better.

`array_throttle=N` caps concurrently running tasks (`--array=0-M%N`). Worth
setting on a shared or preemptable queue, where a large burst is both
antisocial and more exposed to preemption.

Memory: `mem="8G"` sets one `#SBATCH --mem` for the task. Clusters that share
nodes usually apply a per-core default when you leave it unset, so set it
deliberately.

## Recording what ran

You record a run with an explicit call. No `Run` touches the registry, and no
generated script contains `codt-registry` text. Record runs from Python before
or after you submit, at whatever granularity you want. See
`docs/registry-quickstart.md`.

The `runs` table has seven columns, and `notes` is free text. Put anything else
you want recorded about a batch there.

## Storage

Point the base directory at the fastest scratch filesystem your cluster
provides, not at your home directory. CODT writes NetCDF and binary output
every write interval, and a network filesystem is slow under a full node of
concurrent runs.

Scratch filesystems are usually purged on a schedule. Copy results somewhere
durable once you trust them, and note in the experiment README where they
went.
