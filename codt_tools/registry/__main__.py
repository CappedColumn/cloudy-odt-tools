"""
Command-line interface for the CODT simulation registry.

Usage
-----
    codt-registry [--db PATH] <command> [args]
    python -m codt_tools.registry <command> [args]

The database path is taken from ``--db`` or the ``CODT_REGISTRY_DB``
environment variable. This CLI is the supported way for shell scripts
(e.g. SLURM job wrappers) to talk to the registry: it enforces the
pragmas, retry logic, and event/derived-status transaction that raw
``sqlite3`` calls would bypass.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

from codt_tools.registry.api import Registry
from codt_tools.registry.db import DATA_STATUSES, RUN_STATUSES

_RUN_LIST_COLUMNS: tuple[str, ...] = (
    "run_id",
    "experiment_id",
    "status",
    "created_at",
    "completed_at",
    "exit_code",
    "code_version",
    "data_status",
)


def _print_record(record: dict[str, Any]) -> None:
    """Print a single record as aligned key/value lines."""
    width = max(len(k) for k in record)
    for key, val in record.items():
        print(f"  {key:<{width}}  {val if val is not None else '-'}")


def _print_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    """Print selected columns of a list of records as a fixed-width table."""
    if not rows:
        print("(no matches)")
        return
    widths = {
        c: max(len(c), *(len(str(r.get(c) or "-")) for r in rows)) for c in columns
    }
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c) or "-").ljust(widths[c]) for c in columns))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codt-registry",
        description="Query and update the CODT simulation registry.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("CODT_REGISTRY_DB"),
        help="Registry database path (default: $CODT_REGISTRY_DB).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create the database schema.")

    p_status = sub.add_parser("update-status", help="Record a run status change.")
    p_status.add_argument("run_id")
    p_status.add_argument("status", choices=RUN_STATUSES)
    p_status.add_argument("--exit-code", type=int, default=None)
    p_status.add_argument("--job-id", default=None, help="SLURM job ID.")
    p_status.add_argument("--detail", default=None)

    p_complete = sub.add_parser(
        "complete", help="Record versions from a run's output netCDF file."
    )
    p_complete.add_argument("run_id")
    p_complete.add_argument("output_path")

    p_show = sub.add_parser("show", help="Show one run in full.")
    p_show.add_argument("run_id")

    p_list = sub.add_parser("list", help="List/search runs.")
    p_list.add_argument("--experiment", default=None)
    p_list.add_argument("--status", default=None, choices=RUN_STATUSES)
    p_list.add_argument("--param", default=None)
    p_list.add_argument("--group", default=None)
    p_list.add_argument("--value", default=None)
    p_list.add_argument("--since", default=None, help="ISO date lower bound.")
    p_list.add_argument("--until", default=None, help="ISO date upper bound.")

    p_export = sub.add_parser("export", help="Export runs (with params) to CSV.")
    p_export.add_argument("--experiment", default=None)
    p_export.add_argument("--status", default=None, choices=RUN_STATUSES)
    p_export.add_argument("--csv", required=True, dest="csv_path")

    p_data = sub.add_parser("set-data-status", help="Update a run's data location.")
    p_data.add_argument("run_id")
    p_data.add_argument(
        "data_status", choices=("on_scratch", "on_group", "archived", "deleted")
    )

    p_reloc = sub.add_parser(
        "relocate",
        help="Record an experiment tree's move to a new data root "
        "(run after copying; verifies data before updating).",
    )
    p_reloc.add_argument("experiment_id")
    p_reloc.add_argument(
        "new_root",
        nargs="?",
        default=None,
        help="New data root (default: the experiment's recorded "
        "permanent_data_root).",
    )
    p_reloc.add_argument(
        "--data-status",
        default="on_group",
        choices=DATA_STATUSES,
        help="New data_status for all runs (default: on_group).",
    )
    p_reloc.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification of relocated input files.",
    )

    p_exp = sub.add_parser("experiment", help="Experiment operations.")
    exp_sub = p_exp.add_subparsers(dest="exp_command", required=True)
    p_ec = exp_sub.add_parser("create")
    p_ec.add_argument("experiment_id")
    p_ec.add_argument("title")
    p_ec.add_argument("--hypothesis", default=None)
    p_ec.add_argument("--data-root", default=None)
    p_ec.add_argument("--permanent-data-root", default=None)
    p_es = exp_sub.add_parser("show")
    p_es.add_argument("experiment_id")
    p_el = exp_sub.add_parser("list")
    p_el.add_argument(
        "--status",
        default=None,
        choices=("planned", "running", "analyzed", "concluded"),
    )
    p_econ = exp_sub.add_parser("conclude")
    p_econ.add_argument("experiment_id")
    p_econ.add_argument("conclusion")
    p_econ.add_argument(
        "--artifact",
        action="append",
        default=None,
        help="Analysis artifact path (repeatable).",
    )

    return parser


def _cmd_export(reg: Registry, args: argparse.Namespace) -> None:
    """Export runs to CSV with one column per namelist parameter."""
    runs = reg.query_runs(experiment_id=args.experiment, status=args.status)
    rows = []
    for run in runs:
        flat = dict(run)
        for group, params in reg.run_parameters(run["run_id"]).items():
            for name, value in params.items():
                flat[f"{group}.{name}"] = value
        rows.append(flat)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(k for k in row if k not in fieldnames)
    with open(args.csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.csv_path}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    if args.db is None:
        print(
            "error: no database given (use --db or set CODT_REGISTRY_DB)",
            file=sys.stderr,
        )
        return 2

    try:
        with Registry(args.db, create=(args.command == "init")) as reg:
            if args.command == "init":
                print(f"Initialized registry: {args.db}")
            elif args.command == "update-status":
                reg.update_status(
                    args.run_id,
                    args.status,
                    exit_code=args.exit_code,
                    slurm_job_id=args.job_id,
                    detail=args.detail,
                )
                print(f"{args.run_id}: {args.status}")
            elif args.command == "complete":
                reg.record_completion(args.run_id, args.output_path)
                print(f"{args.run_id}: versions recorded from {args.output_path}")
            elif args.command == "show":
                run = reg.get_run(args.run_id)
                _print_record(run)
                print("\n  parameters:")
                print(
                    json.dumps(reg.run_parameters(args.run_id), indent=4)
                )
                print("\n  events:")
                for event in reg.run_events(args.run_id):
                    print(
                        f"    {event['timestamp']}  {event['status']:<10} "
                        f"host={event['hostname'] or '-'} "
                        f"exit={event['exit_code'] if event['exit_code'] is not None else '-'}"
                    )
            elif args.command == "list":
                _print_table(
                    reg.query_runs(
                        experiment_id=args.experiment,
                        status=args.status,
                        param=args.param,
                        group=args.group,
                        value=args.value,
                        since=args.since,
                        until=args.until,
                    ),
                    _RUN_LIST_COLUMNS,
                )
            elif args.command == "export":
                _cmd_export(reg, args)
            elif args.command == "set-data-status":
                reg.set_data_status(args.run_id, args.data_status)
                print(f"{args.run_id}: data_status={args.data_status}")
            elif args.command == "relocate":
                reg.relocate_experiment(
                    args.experiment_id,
                    args.new_root,
                    data_status=args.data_status,
                    verify=not args.no_verify,
                )
                new_root = reg.get_experiment(args.experiment_id)["data_root"]
                print(
                    f"{args.experiment_id}: data_root={new_root}, "
                    f"runs data_status={args.data_status}"
                )
            elif args.command == "experiment":
                if args.exp_command == "create":
                    reg.create_experiment(
                        args.experiment_id,
                        args.title,
                        hypothesis=args.hypothesis,
                        data_root=args.data_root,
                        permanent_data_root=args.permanent_data_root,
                    )
                    print(f"Created experiment: {args.experiment_id}")
                elif args.exp_command == "show":
                    _print_record(reg.get_experiment(args.experiment_id))
                elif args.exp_command == "list":
                    _print_table(
                        reg.list_experiments(args.status),
                        ("experiment_id", "status", "title", "created_at"),
                    )
                elif args.exp_command == "conclude":
                    reg.conclude_experiment(
                        args.experiment_id, args.conclusion, args.artifact
                    )
                    print(f"Concluded experiment: {args.experiment_id}")
    except (KeyError, FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
