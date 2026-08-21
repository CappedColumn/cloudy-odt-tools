# codt_tools Refactor — Overarching Plan

## Context

`docs/refactor.txt` proposes restructuring codt_tools into four layers — **case**
(pure input description), **run** (one case + one executable + one workdir),
**simulation** (read-only analysis), **registry** (optional bookkeeping) — because
`config.py`, `runner.py`, `experiment.py`, and `registry/api.py` are mutually
entangled today.

Exploration confirms the diagnosis and sharpens it:

| Finding | Evidence |
|---|---|
| `CODTConfig` is already ~90% of the proposed `Case` | it owns `params`/`injection`/`parcel` and has `copy()`, `write()`, `validate()`, `from_simulation()` — the refactor is mostly a split + rename + de-mutation, not a rewrite |
| Two real hidden mutations | `config.py:1447,1453` (`write()` overwrites `aerosol_file`/`parcel_file`); `runner.py:236` (`setup_run()` writes `output_directory` into the *caller's* config) |
| `sweep()` is namelist-only | `config.py:1835` routes every value through `Namelist.set` — aerosol and parcel fields are unreachable |
| The YAML experiment layer is dead | `ExperimentSpec` / `create_experiment_runs`: **zero callers** across all 59 consumer files. The real pattern is a hand-rolled `config.py` + `submit.py` + `analyze.py` per project |
| Registry is required, not optional | `experiment.py:238` takes `Registry` as a **required positional**; `runner.py:388-397` embeds `codt-registry` CLI text into generated sbatch scripts |
| Analysis leaks into registry | `simulation.py:22` imports `check_conventions` from `registry.versions` — the one dependency that makes analysis need the registry package |
| SLURM introspection is fine; its *placement* isn't | `slurm.py` is 492 lines of pure, injectable, testable functions with zero coupling. `runner.py` is what drags it into the core path |

**Intended outcome:** a package where building a case is obvious, running one
locally is trivial, generating many runs is explicit, SLURM support is artifact
generation rather than cluster policy, analysis is stable, and provenance is
optional.

### Decisions taken (2026-08-20)

1. **Land v0.8.0 first.** The arch-aware SLURM work is complete and green
   (422 tests collect; 170 pass in the four affected files). Commit and tag it
   before any refactor commit.
2. **Clean break, no compatibility shims.** `CODTConfig`/`CODTRunner`/
   `CODTSimulation` are renamed, not aliased. The 59 files in
   `~/simulations/CODT/projects/` are **explicitly out of scope** — they pin to
   the `v0.8.0` tag.
3. **Full minimal registry** per the doc (~9 columns). The existing 554-run DB is
   archived read-only.
4. **Delete `experiment.py`**, salvaging sweep expansion and shared-input dedup.
5. **Update each Claude skill in the stage that breaks it.**

### Consequence to handle in Stage 0

Every project under `~/simulations/CODT/projects/` imports from an *editable*
install of this tree, so the first breaking commit breaks all 14 projects at once.
Before Stage 1, stand up a second environment pinned to the tag:

```bash
git -C ~/dev/CODT_tools tag v0.8.0            # Stage 0
git clone ~/dev/CODT_tools ~/dev/CODT_tools-v0.8.0 -b v0.8.0
conda create -n codt08 --clone codt
conda activate codt08 && pip install -e ~/dev/CODT_tools-v0.8.0
```

Old project scripts then run under `codt08`; `codt` follows the refactor.

---

## Target layout

```
codt_tools/
  __init__.py            Case, Run, Simulation  (three names, nothing else)
  conventions.py         moved out of registry/versions.py — the format gate
  chpc.py                renamed from slurm.py; CHPC-only, optional, off the core path
  case/
    __init__.py          Case
    case.py              Case: from_input_dir, clone, apply, sweep, validate, write_inputs
    namelist.py          from config.py:128-511
    aerosol.py           InjectionData -> Aerosol   (+ aerosol_io.py folded in as _io)
    parcel.py            ParcelInput   -> Parcel    (+ parcel_io.py folded in as _io)
    mutate.py            path-based apply/sweep, callable mutators
    validate.py          lem_turbulence_scales + the validate() body
  run/
    __init__.py          Run
    run.py               Run(case, executable, workdir): stage, execute_local, open_simulation
    launcher.py          shell / parallel / SLURM-array artifact generation
    validate_exec.py     validate_executable() -> report object
  simulation/
    __init__.py          Simulation
    simulation.py        from simulation.py, largely intact
    plotting.py
    trajectory.py
  registry/
    __init__.py          Registry (minimal)
    store.py             ~9-column SQLite + CLI
    legacy.py            read-only reader for the archived v4 DB
```

Notes on deviations from `docs/refactor.txt`:

- `slurm.py` → **`chpc.py`** (flat module, not `run/slurm.py`). The rename is the
  point: the name states "site policy," and nothing in `run/` imports it.
  `run/launcher.py` emits generic placeholder templates only.
- `aerosol_io.py` / `parcel_io.py` fold **into** `case/aerosol.py` / `case/parcel.py`
  rather than a shared `case/io.py`. They are leaf modules whose only consumer is
  the matching model class.
- `conventions.py` sits at **top level**, not under `registry/`. It is the
  `SUPPORTED_CONVENTIONS` source of truth that `case` and `simulation` both need;
  leaving it under `registry/` is exactly the dependency the refactor removes.
  (The `codt-versioning` skill must learn the new path.)

## Target API

```python
from codt_tools import Case, Run, Simulation

case = Case.from_input_dir("~/dev/CODT/input")
case.params.set(tref=21.0, tmax=3600.0)
case.aerosol.set(injection_rate=[6.6e4])
case.validate()

run = Run(case=case, executable="/path/to/CODT", workdir="/scratch/.../test1")
run.stage()                 # writes files; mutates nothing on `case`
run.execute_local()

sim = Simulation.from_run(run)
sim.plot_timeseries("LWC")
```

```python
cases = case.sweep({
    "params.tref":            [20.0, 21.0, 22.0],
    "aerosol.injection_rate": [[5e4], [6e4], [7e4]],
    "parcel.velocity":        [[1.0, 0.5], [0.8, 0.4]],
})
runs = [Run(c, EXE, BASE / c.name) for c in cases]

from codt_tools.run.launcher import generate_parallel_launcher, generate_slurm_array
generate_parallel_launcher(runs, jobs=20, path=BASE / "run_all.sh")
generate_slurm_array(runs, path=BASE / "array.sh")   # + runs.txt
```

---

## Stages

Each stage is one branch off `main`, merged when green. Stages 1–6 each get their
own plan-mode session before implementation. PR size guidance (300–800 lines) is
measured in **logic changed**, not lines moved — the file splits inflate raw diffs.

### Stage 0 — Baseline (no sub-plan; do first)

Commit the uncommitted v0.8.0 changeset and tag it; stand up the `codt08` pin env
described above. Also commit or remove the three untracked docs
(`refactor.txt`, `running-on-slurm.md`, `arch_aware_slurm_handoff.md`).

**Done when:** `git status` clean, `v0.8.0` tag exists, `codt08` env runs an old
project script unchanged.

### Stage 1 — `case/` — branch `refactor/case-layer`

Split `config.py` (1853 lines) into `case/`. Rename `CODTConfig`→`Case`,
`InjectionData`→`Aerosol`, `ParcelInput`→`Parcel`, attribute `injection`→`aerosol`.
Fold `aerosol_io.py`/`parcel_io.py` into the model modules. Add
`Case.from_input_dir()`. **Delete `experiment.py` and `tests/test_experiment.py`**
here — it is dead and it is a `CODTConfig` consumer, so removing it now avoids
carrying an alias.

The defining change: **`Case` never mutates itself.** `write_inputs(dir)` derives a
staging namelist (relative `aerosol_file`/`parcel_file`, absolute
`output_directory`) and writes that, leaving `case.params` untouched. Fix the
related bug that `write()` calls `write_parcel` without `initial_level`/
`vertical_axis`, skipping leg-direction validation at write time.

Open question for the sub-plan: keep `CODTConfig.__getattr__`/`__setattr__`
dot-proxying onto namelist params (`config.py:1359,1383`), or drop it now that
`case.apply({"params.tref": ...})` supersedes it?

Touches: `codt_tools/config.py`, `aerosol_io.py`, `parcel_io.py`, `experiment.py`;
call sites in `runner.py:236,240` and `registry/api.py:363,365` (the latter reaches
into private `params._groups_for_write()` — give it a public accessor).
Skill: **configure-run-codt**.

### Stage 2 — Sweeps — branch `refactor/sweep-api`

`case/mutate.py`: path-based `Case.apply({"params.tref": 21.0})` resolving
`params.*` / `aerosol.* `/ `parcel.*`, and `Case.sweep(dict)` taking the Cartesian
product over the whole case, plus callable mutators. Salvage the expansion logic
from the deleted `experiment.py:152`.

Replace `_SWEEP_ABBREV` auto-naming (`config.py:1768`) — it stringifies arrays into
directory names and can collide. Sub-plan decides: explicit `name=` callable with a
sane default, or an index-based default.

Skill: **configure-run-codt** (sweep section), **design-experiment** (rewrite around
`Case.sweep` instead of `ExperimentSpec` YAML).

### Stage 3 — `run/` — branch `refactor/run-layer`

`Run(case, executable, workdir)` — no `account`, `partition`, `qos`, `registry`,
`experiment_id`. `stage()` materializes `workdir/{inputs,output}` without touching
`case`. `execute_local()` runs the binary. `open_simulation()` returns a `Simulation`.

`run/launcher.py` emits three artifact kinds: serial shell script, parallel
launcher with a job cap (EXP003 ran 20-way `nohup` on kingspeak12 — this is a
first-class need, not a SLURM afterthought), and a generic SLURM array template
with `<ACCOUNT>`/`<PARTITION>`/`<TIME>` placeholders plus `runs.txt`.

**Model the array template on `/scratch/general/vast/u1342804/CODT/SAM_sims/slurm/Act_SAM.sh`**
— the script actually written and used by hand. It carries `--requeue` and a
`_DONE`-marker skip loop, which is how preemption is survived without CODT-side
checkpointing. Emit that guard by default.

`run/validate_exec.py`: `validate_executable(executable, smoke_case=None)` →
report (exists, `--version`, optional `detect_build_arch`, optional dry local run).

Rename `slurm.py`→`chpc.py`; strip `runner.py` down to nothing and delete it,
along with the `codt-registry` CLI text embedded in generated sbatch
(`runner.py:388-397`).

Skill: **configure-run-codt** (execution section), **partition-tree** /
**job-triage** cross-check.

### Stage 4 — `simulation/` — branch `refactor/simulation-namespace`

Mostly organizational and low-risk. Move `simulation.py`/`plotting.py`/
`trajectory_io.py` under `simulation/`; rename `CODTSimulation`→`Simulation`; add
`Simulation.from_run(run)`. Promote the private `_get_label`/`_ensure_ax` that
`simulation.py:23-32` imports out of `plotting.py` to public names. Extract
`registry/versions.py`'s gating into top-level `conventions.py`, cutting the
analysis→registry edge.

Skill: **analyze-codt**, **codt-versioning** (the `SUPPORTED_CONVENTIONS` path moves).

### Stage 5 — Minimal registry — branch `refactor/minimal-registry`

New `registry/store.py`: `run_id, case_name, workdir, executable_path,
executable_version, input_hash, created_at, notes, tags`. SQLite, not JSONL —
WAL handles concurrent writes from array tasks on NFS, which appended JSONL does
not. Recording is an explicit call (`registry.add(run)`), never a side effect of
`stage()` or `execute_local()`.

Delete `registry/api.py` (711 lines), the 6-table schema, and the migration chain.
Keep `registry/legacy.py`: a read-only reader for the archived v4 DB so the 554
existing runs and 26,688 parameter rows stay queryable. New CLI: `add-run`,
`list-runs`, `show-run`.

Also clears out confirmed dead code: `Registry.update_run` (`api.py:499`, zero
callers), the `output_files` table (never written), `runs.model_name` (never written).

Skill: **registry-admin**, **experiment-status**, **conclude-experiment** — all
three describe workflows that no longer exist; rewrite or retire.

### Stage 6 — Docs, scaffold, release — branch `refactor/docs-and-release`

Rewrite `README.md` around Case → Run → Simulation. Rewrite
`docs/running-on-slurm.md` around launcher generation with CHPC as *example*.
Rewrite `docs/registry-quickstart.md` as optional bookkeeping; delete
`docs/registry-schema.md`'s v1–v4 migration history in favor of the new schema.
Rewrite the CODT Tools sections of `CLAUDE.md`.

Add the missing docs that existing code already cites:
`docs/codt_v3_migration.md` (referenced by `versions.py:48`) and
`docs/input_parameters.md` (referenced by `CLAUDE.md`).

Consider shipping a project scaffold that emits the `config.py`/`submit.py`/
`analyze.py` trio — that hand-rolled trio is the pattern all 14 real projects
converged on, and it is better formalized than reinvented.

Version bump: **0.8.0 → 1.0.0** (recommended — this is a full, deliberate API
break and the API is meant to stick). Follow the `codt-versioning` skill.

---

## Verification

Per stage, in the `codt` env:

```bash
conda activate codt && pytest          # 422 tests today; expect the suite to shrink
```

Test files track the stages: `test_config.py` (734) → `tests/case/`;
`test_runner.py` (565) → `tests/run/`; `test_experiment.py` (443) deleted in
Stage 1; `test_registry.py` (673) mostly replaced in Stage 5;
`test_simulation.py`/`test_aerosol_io.py`/`test_parcel_io.py`/`test_trajectory_io.py`/
`test_plotting.py` survive with import updates.

End-to-end, after Stage 3 and again after Stage 6, against a real binary
(`~/dev/CODT/build/*/app/CODT`):

```python
case = Case.from_input_dir("~/dev/CODT/input")
case.params.set(simulation_mode="chamber", tmax=600.0)   # ~45 s wall
run = Run(case, EXE, workdir=SCRATCH / "smoke_chamber"); run.stage(); run.execute_local()
sim = Simulation.from_run(run); sim.info()
assert case.params.get("output_directory") == ""         # staging mutated nothing
```

Repeat for parcel (`tmax=240`, ~4.5 min, `edge_radii=[30,60,120]` nm,
`initial_rh=0.98` — per the physics-verified workshop settings in `CLAUDE.md`).

Launcher artifacts: `bash -n` each generated script; `sbatch --test-only` the array
template with placeholders filled.

Registry: fresh DB, `add-run` → `list-runs` → `show-run`; confirm
`registry/legacy.py` still reads the archived v4 DB and reports 554 runs.

---

## Known defects to fix along the way

| Defect | Location | Stage |
|---|---|---|
| `write()` overwrites `aerosol_file` / `parcel_file` on self | `config.py:1447,1453` | 1 |
| `setup_run()` mutates the caller's config | `runner.py:236` | 1/3 |
| `write()` skips leg-direction validation (no `initial_level`/`vertical_axis`) | `config.py:1454` | 1 |
| Registry reaches into private `params._groups_for_write()` | `registry/api.py:365` | 1 |
| `simulation.py` imports private `_get_label`/`_ensure_ax` | `simulation.py:23-32` | 4 |
| `codt-registry` CLI text embedded in generated sbatch | `runner.py:388-397` | 3 |
| `Registry.update_run` — zero callers | `registry/api.py:499` | 5 |
| `output_files` table never written; `runs.model_name` never written | `registry/db.py` | 5 |
| `notebooks/getting_started.ipynb` imports removed `BinData` | notebook cell 1 | 6 |
| `test_runs/workshop_verify/setup_cases.py` uses pre-v3 `set_parcel(time=...)` | that file | 6 |
| `docs/codt_v3_migration.md`, `docs/input_parameters.md` cited but absent | `versions.py:48`, `CLAUDE.md` | 6 |

## Out of scope

- Migrating the 59 files in `~/simulations/CODT/projects/` — they pin to `v0.8.0`.
- Any CODT (Fortran) change. This refactor is Python-only; no file format,
  namelist group, or conventions string moves.
- Preemption checkpoint/restart — CODT has none. The array template's `_DONE`
  skip guard is the whole of the requeue story.
