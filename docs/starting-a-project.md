# Starting a project

A CODT project is three short scripts and a directory. This is the shape
every real project has converged on; copy the three files, edit them, and you
have a working ensemble.

```
EXP006_seeding/
    README.md        what you are asking, and later, what you found
    design.py        what to simulate
    stage.py         write the run directories and a launch script
    analyze.py       read the output back
    design.csv       written by design.py: index -> parameters
```

The scripts live with your project — in `~/simulations/CODT/projects/`, a git
repo, wherever — **not** in the run directory. Run data goes to scratch; the
scripts and the README are small and should live somewhere permanent.

The split matters because the middle step is the slow one. Designing is
instant, analyzing is interactive, but staging hundreds of directories and
waiting on a queue is neither — so it gets its own file you can re-run.

---

## 1. `design.py` — what to simulate

Build one base `Case`, expand it into a design, and **validate before
anything touches the disk**. `case.validate()` mirrors CODT's own startup
checks, so a bad parameter fails here in seconds rather than after a queue
wait.

```python
"""EXP006: does seeding change the activated fraction at fixed updraft?"""

from pathlib import Path
import csv

from codt_tools import Case
from codt_tools.case import cross

EXP = "EXP006"
BASE_DIR = Path(f"/scratch/general/vast/{Path.home().name}/CODT/{EXP}")


def build():
    """The base case and the design expanded from it."""
    base = Case()
    base.set(
        simulation_name=EXP,
        simulation_mode="chamber",
        tmax=3600.0,
        do_microphysics=True,
    )

    # A design is a list of points; a point is a dict of path -> value.
    design = cross(
        {"params.tref": [18.0, 21.0, 24.0]},
        {"params.volume_scaling": [13, 50]},
    )                                             # 6 runs

    cases = base.sweep(design)                    # EXP006_000 ... EXP006_005
    for case in cases:
        case.validate()                           # fails fast, and locally
    return design, cases


if __name__ == "__main__":
    design, cases = build()
    print(f"{len(cases)} runs: {cases[0].name} ... {cases[-1].name}")

    # index -> parameters, kept next to the project. Nothing reads this back;
    # it is how a future reader (you) knows what run 004 was.
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(Path(__file__).parent / "design.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id"] + sorted(design[0])
        )
        writer.writeheader()
        for case, point in zip(cases, design):
            writer.writerow({"run_id": case.name, **point})
```

`cross` is the only combinatorial helper. Anything it cannot express —
paired parameters, LHS samples, conditional axes, control runs, filtering —
is plain Python on a list of dicts. See [designs.md](designs.md) for the
recipes.

---

## 2. `stage.py` — write the directories and a launch script

**Check the binary first.** Staging 200 directories with an executable that
cannot run wastes the whole round trip, and the failure — an illegal
instruction on a node with the wrong CPU — is unreadable when it finally
arrives.

```python
"""Stage EXP006 and write its launch script. Does not submit anything."""

from pathlib import Path

from codt_tools import Run, check_executable, write_slurm_array
from codt_tools.registry import Registry

from design import BASE_DIR, EXP, build

EXE = Path.home() / "dev/CODT/build/gfortran_*/app/CODT"   # a real path
REGISTRY = Path.home() / "codt_runs.db"


if __name__ == "__main__":
    problem = check_executable(EXE)
    if problem:
        raise SystemExit(f"CODT binary unusable: {problem}")

    _, cases = build()
    runs = Run.for_cases(cases, EXE, BASE_DIR)
    for run in runs:
        run.stage()                     # writes inputs/, creates output/

    script = write_slurm_array(
        runs, BASE_DIR / "array.sh",
        runs_per_task=6,                # <= cores a task will get
        account="krueger",
        partition="notchpeak-freecycle",
        qos="notchpeak-freecycle",
        constraint="rom",               # match the binary's build!
        time="12:00:00",
        module="gcc/11.2.0",
    )

    # Optional: record that these runs exist.
    with Registry(REGISTRY) as reg:
        reg.add_many(runs, tags=EXP)

    print(f"Staged {len(runs)} runs.\nSubmit with:  sbatch {script}")
```

Then, yourself:

```bash
sbatch /scratch/.../EXP006/array.sh
```

**codt_tools never submits anything.** It writes the script; you read it and
run `sbatch`. Read it the first time — the `#SBATCH` header is right at the
top, and any directive you left out appears as `<PLACEHOLDER>`.

Pick `constraint` deliberately to match how the binary was built;
[running-on-slurm.md](running-on-slurm.md) has the architecture table and the
group's entitlements.

**On this machine instead of a cluster**, swap one call:

```python
write_local(runs, BASE_DIR / "run_all.sh", jobs=6)
#   nohup bash run_all.sh > run_all.log 2>&1 &
```

Both scripts skip any run that already finished, so **re-running the script is
how you mop up an incomplete ensemble** — no filtering, just submit it again.

---

## 3. `analyze.py` — read the output back

```python
"""Read EXP006 back."""

import matplotlib.pyplot as plt

from codt_tools import Run, Simulation

from design import BASE_DIR, build
from stage import EXE


if __name__ == "__main__":
    _, cases = build()
    runs = Run.for_cases(cases, EXE, BASE_DIR)

    missing = [r.name for r in runs if not r.is_complete]
    if missing:
        print(f"{len(missing)} runs incomplete: {missing}")

    sims = [r.open_simulation() for r in runs if r.is_complete]

    sims[0].info()

    out = BASE_DIR / "analysis"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    Simulation.compare(sims, "LWC", plot_type="timeseries", ax=ax)
    fig.tight_layout()
    fig.savefig(out / "lwc.pdf")
```

Completion is `run.is_complete` — the `_DONE` marker CODT writes on a clean
exit. That is the only reliable answer; the registry does not track status.

For what `Simulation` exposes — profiles, averages, budgets, spectra, DSDs,
trajectories, collision and eddy binaries — see the `analyze-codt` skill.

---

## 4. The README beside the data

Write it when you start, finish it when you conclude:

```markdown
# EXP006 — seeding and activated fraction

**Question.** Does hygroscopic seeding change Nact at fixed updraft?
**Design.** 3 tref x 2 volume_scaling, chamber, tmax=3600. See design.csv.
**Binary.** CODT v3.1.0 (ee8f004).

## Result
...written after analysis...

## Data
/scratch/.../EXP006  ->  moved to group space YYYY-MM-DD
```

The registry has nowhere to put a hypothesis or a conclusion, by design. This
file is the record, and it should sit next to the runs it describes.

**Scratch is purged after 60 days of inactivity, without warning.** Copy the
tree to group space once you trust the results, and note in the README where
it went.

---

## See also

- [designs.md](designs.md) — expressing an ensemble that is not a plain
  product: LHS samples, paired parameters, ragged branches, control groups.
- [running-on-slurm.md](running-on-slurm.md) — picking a constraint, sizing a
  task, and what to do about preemption.
- [registry-quickstart.md](registry-quickstart.md) — the optional list of runs.
