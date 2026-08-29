# Designs: describing an ensemble

A single simulation is a `Case`. A set of simulations is a **design**. This
document shows you how to record a design, including the irregular ones. Most
real designs are irregular.

## The vocabulary

There are exactly three ideas.

**A path** addresses one thing in a case:

| path | means |
|---|---|
| `"params.tref"` | a namelist parameter (the group is resolved for you) |
| `"aerosol.injection_rate"` | a field of the aerosol input |
| `"parcel.velocity"` | a field of the parcel input |
| `"aerosol"` | the whole component — swap in a prebuilt `Aerosol` |

**A point** is a dict of `path -> value`: one case's worth of changes.

```python
{"params.tref": 22.0, "params.do_seeding": True, "aerosol": seeded}
```

**A design** is a list of points — one entry per run. That's it. A design is an
ordinary Python list of ordinary Python dicts, and everything below follows from
that.

```python
from codt_tools import Case
from codt_tools.case import cross

base = Case()
base.set(simulation_name="EXP001", tmax=3600.0)   # what every run shares

cases = base.sweep(design)                         # one Case per point
```

Two helpers do the only work that isn't ordinary Python:

- `case.apply(point)` — apply one point to one case, in place.
- `cross(*axes)` — Cartesian product of axes. An **axis** is either a list of
  points, or a dict of `path -> list of values` (which expands into one
  independent axis per key).

`case.sweep(design)` deep-copies the base once per point, applies the point, and
names the result. The base is never modified, and the cases never share
components.

## The regular case

```python
cases = base.sweep({
    "params.tref":           [20.0, 21.0, 22.0],
    "params.volume_scaling": [13, 50],
})                                    # 6 cases: EXP001_000 ... EXP001_005
```

A dict of `path -> values` handed straight to `sweep` is shorthand for the plain
Cartesian product. If that is all you need, stop reading here.

## Irregular designs

Real ensembles are rarely a clean product. Every pattern below is just list and
dict manipulation, so they compose freely.

### Values that vary together

An axis is a *list of points*, so you write sampled or paired values directly.
There is no zip helper and no special syntax.

```python
# a Latin hypercube: 10 points in 4 dimensions, NOT 10**4 combinations
lhs = [{"params.ent_rate": r, "params.tref": t,
        "params.pres": p, "params.aerosol_concentration": n}
       for r, t, p, n in samples]

# a paired ladder: rep 1..5 carries n_blob 1,1,2,2,5
reps = [{"params.n_blob": nb} for _, nb in [(1,1),(2,1),(3,2),(4,2),(5,5)]]

design = cross(lhs, reps)             # 10 x 5 = 50
```

### Derived values

Compute a value that follows from another where you build the axis:

```python
axis = [{"params.ent_rate": r, "params.do_entrainment": r > 0.0}
        for r in rates]
```

If the value depends on paths that only meet *after* a cross, derive it in a
pass over the finished design:

```python
design = cross(rate_axis, mode_axis)
design = [{**p, "params.do_entrainment": p["params.ent_rate"] > 0.0}
          for p in design]
```

### Conditional axes

A parameter that is meaningless unless a switch is on doesn't belong in the
product — it belongs in a branch. Build each branch, then concatenate:

```python
off = [{"params.do_entrainment": False}]                  # 1 run
on  = cross({"params.do_entrainment": [True]},
            {"params.ent_rate": [0.5, 1.0]},
            {"params.n_blob":   [1, 5]})                  # 4 runs

design = off + on                                          # 5 runs
```

Crossing `n_blob` over the `off` branch would have given four identical
entrainment-free runs wearing different labels.

### Ragged branches

Different sub-ranges per branch, same idea:

```python
design = (
    cross({"params.simulation_mode": ["chamber"]},
          {"params.tdiff": [5.0, 10.0]})
  + cross({"params.simulation_mode": ["parcel"]},
          {"params.initial_rh": [0.95, 0.98, 1.0]})
)                                                          # 2 + 3 = 5
```

### Control runs

A reference run beside a sweep is one more point:

```python
design = swept + [{"params.tref": 21.0, "params.do_entrainment": False}]
```

### Replicates

CODT has no integer RNG seed (`same_random` is on/off only), so repeats sample
RNG variability. A repeated point is a repeated run:

```python
design = [p for p in design for _ in range(5)]     # 5 realizations of each
```

The default index naming keeps them distinct (`exp_000` … `exp_004`), and
because a design is a list, the replicate index is just position.

### Dropping invalid combinations

Filter with a comprehension. This one mirrors CODT's entrainment rule
(`int(psigma * N) >= n_blob`), so the bad combinations never reach a queue:

```python
grid = cross({"params.psigma": [0.001, 0.1]},
             {"params.n_blob": [1, 5]},
             {"params.n":      [2000]})
design = [p for p in grid
          if int(p["params.psigma"] * p["params.n"]) >= p["params.n_blob"]]
```

`case.validate()` catches the same thing later. Filter the design as well, to
keep the run count honest.

### Whole components per branch

An axis can carry an object, which is how a seeding axis works — each mode is a
separately built `Aerosol`:

```python
modes = [{"aerosol": aerosols[m], "params.do_seeding": m != "NO"}
         for m in ("NO", "CB2.5", "CB5", "MC2.5", "MC5")]
```

Values can also be **callables**, which transform rather than replace. The
callable receives the current value and must return the new one:

```python
{"params.tmax": lambda t: 2 * t}
{"aerosol": lambda a: with_seed_group(a, radius_nm=2500)}
```

### Patching one point

A design is a list, so fix a single entry in place:

```python
design[7] = {**design[7], "params.tmax": 7200.0}
```

### De-duplicating after a union

Concatenating designs can repeat a point. For designs whose values are all
scalars:

```python
design = list({tuple(sorted(p.items())): p for p in design}.values())
```

(Points carrying objects or arrays are not hashable this way — dedupe those on
whichever scalar keys identify them.)

## What `cross` refuses

Crossed axes must be independent. If two of them set the same path, `cross`
raises rather than silently letting one win:

```python
cross({"params.tref": [20.0]}, {"params.tref": [22.0]})
# ValueError: Crossed axes both set params.tref. ... to put two designs side
# by side, add them: design_a + design_b.
```

That is nearly always a design bug. When you genuinely want one branch to
override another's default, they are two designs — concatenate them.

## Naming

Run names become run *directory* names, so the default is an index:
`{base_name}_000`, `_001`, … (widened past a thousand runs). It is always
path-safe and can never collide, whatever the design varies.

**The case layer does not store the mapping from index to parameters.** Each run
holds its own configuration in its staged inputs: `inputs/params.nml` plus the
NetCDF input files. Open run 004 and you can read what it was. The registry
does not hold the mapping either. Its seven columns record that a run happened,
and where it is.

Write the mapping yourself when you build the design if you want to read the
design as a table. See "Recording the design" below.

For a project convention, pass `name=`, which receives `(index, point)`:

```python
cases = base.sweep(design, name=lambda i, p: f"EXP002_{i:03d}")
```

Anything not in the point — a replicate label, a mode name — can be carried in
a parallel list and indexed by `i`:

```python
labels = ["r1", "r2", "r3"]
cases = base.sweep(axis, name=lambda i, p: f"EXP002_{labels[i]}")
```

Duplicate names raise: two runs cannot share a directory.

## What this deliberately doesn't do

- **No sweep DSL.** No axis objects, conditional-axis syntax, or constraint
  expressions. Everything above is list and dict manipulation. You can print,
  slice and debug it with the tools you already have. A DSL could not express
  the branch and filter cases without growing into a small language.
- **No adaptive designs.** A design is fixed before anything runs. To choose
  the next points from finished results, write a second sweep after the
  analysis.
- **No manifest file written for you.** The staged inputs are the record of
  what each run was. Write your own index if you want to read the design as a
  table. See "Recording the design" below.

## Recording the design

The design is data, so keep it if you want it:

```python
import pandas as pd
pd.DataFrame(design).to_csv(exp_dir / "design.csv", index=False)   # scalars
```

That is documentation for a future reader, not something codt_tools reads back.

## A full example

The shape of a real campaign — 10 LHS points × 5 seeding modes × 5 paired
realizations, filtered, with a control group:

```python
from codt_tools import Case, Run, write_slurm_array
from codt_tools.case import cross

base = Case.from_input_dir("~/dev/CODT/input")
base.set(simulation_mode="parcel", simulation_name="EXP002", tmax=2200.0)

lhs   = [{"params.ent_rate": r, "params.tref": t, "params.pres": p,
          "params.aerosol_concentration": n,
          "params.do_entrainment": r > 0.0}
         for r, t, p, n in lhs_samples]                     # 10
modes = [{"aerosol": aerosols[m], "params.do_seeding": m != "NO"}
         for m in MODES]                                     # 5
reps  = [{"params.n_blob": nb} for _, nb in REALIZATIONS]    # 5

design = cross(lhs, modes, reps)                             # 250

n, psigma = base.params.get("n"), base.params.get("psigma")  # CODT's rule
design = [p for p in design
          if int(psigma * n) >= p["params.n_blob"]]
print(len(design), "runs")                                   # check before staging

cases = base.sweep(design, name=lambda i, p: f"EXP002_{i:03d}")
for case in cases:
    case.validate()                                          # fails in seconds,
                                                             # not after a queue wait

runs = Run.for_cases(cases, executable, base_dir)
for run in runs:
    run.stage()

write_slurm_array(runs, base_dir / "array.sh", runs_per_task=64,
                  account=..., partition=..., time="12:00:00")
# then, yourself:  sbatch array.sh
```

## See also

- `docs/file-formats.md` — what CODT reads and writes, and where.
- `docs/running-on-slurm.md` — turning a staged ensemble into a batch script.
- `docs/registry-quickstart.md` — recording an ensemble in the registry.
- `codt_tools/case/mutate.py` — the implementation. It is short.
