"""``codt-registry`` — read and edit the list of runs from a shell.

Four commands::

    codt-registry list [--tag EXP005]
    codt-registry show <run_id>
    codt-registry add <run_dir> [--executable ...] [--tag ...] [--notes ...]
    codt-registry remove <run_id>

The database comes from ``--db`` or ``$CODT_REGISTRY_DB``. There is no
``init``: opening a path creates it.
"""

from __future__ import annotations

import argparse
import os
import sys

from codt_tools.registry.store import Registry

ENV_VAR: str = "CODT_REGISTRY_DB"


def build_parser() -> argparse.ArgumentParser:
    """The ``codt-registry`` command-line interface."""
    parser = argparse.ArgumentParser(
        prog="codt-registry",
        description="A list of the CODT simulations that were run.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get(ENV_VAR),
        help=f"Registry database (default: ${ENV_VAR}).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List runs, newest first.")
    p_list.add_argument("--tag", help="Only runs whose tags contain this.")
    p_list.add_argument("--since", help="Only runs added on or after this date.")

    p_show = sub.add_parser("show", help="Show one run.")
    p_show.add_argument("run_id")

    p_add = sub.add_parser("add", help="Record a run directory.")
    p_add.add_argument("run_dir")
    p_add.add_argument("--executable", help="Binary, to capture its version.")
    p_add.add_argument("--tag", help="Free text, for grouping.")
    p_add.add_argument("--notes", help="Free text.")
    p_add.add_argument("--run-id", help="Defaults to the directory name.")

    p_remove = sub.add_parser("remove", help="Drop a run from the list.")
    p_remove.add_argument("run_id")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a shell exit code."""
    args = build_parser().parse_args(argv)

    if not args.db:
        print(
            f"No database given. Pass --db or set ${ENV_VAR}.",
            file=sys.stderr,
        )
        return 2

    with Registry(args.db) as reg:
        if args.command == "list":
            rows = reg.list(tag=args.tag, since=args.since)
            if not rows:
                print("No runs recorded.")
                return 0
            _print_table(rows)
            print(f"\n{len(rows)} run(s).")
            return 0

        if args.command == "show":
            row = reg.get(args.run_id)
            if row is None:
                print(f"No run '{args.run_id}'.", file=sys.stderr)
                return 1
            width = max(len(k) for k in row)
            for key, value in row.items():
                print(f"{key:<{width}}  {value if value is not None else '-'}")
            return 0

        if args.command == "add":
            run_id = reg.add_directory(
                args.run_dir,
                executable=args.executable,
                tags=args.tag,
                notes=args.notes,
                run_id=args.run_id,
            )
            print(f"Recorded {run_id}.")
            return 0

        if args.command == "remove":
            if reg.remove(args.run_id):
                print(f"Removed {args.run_id}.")
                return 0
            print(f"No run '{args.run_id}'.", file=sys.stderr)
            return 1

    return 2


def _print_table(rows: list[dict]) -> None:
    """Print the columns worth seeing at a glance, aligned."""
    shown = ("run_id", "created_at", "code_version", "tags")
    widths = {
        key: max(len(key), max(len(_cell(r, key)) for r in rows))
        for key in shown
    }
    header = "  ".join(f"{key:<{widths[key]}}" for key in shown)
    print(header)
    print("  ".join("-" * widths[key] for key in shown))
    for row in rows:
        print("  ".join(f"{_cell(row, key):<{widths[key]}}" for key in shown))


def _cell(row: dict, key: str) -> str:
    value = row.get(key)
    return "-" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
