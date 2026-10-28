from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import yaml

from .attribution_scope import load_scope, scope_summary
from .config import EXPERIMENT_NAMES, load_experiment, repo_root


def _experiment_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)


def _run_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    _experiment_argument(parser)
    parser.add_argument("--scope")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--check-data", action="store_true")

    show = subparsers.add_parser("show-config")
    _experiment_argument(show)

    for command in ("prepare-dataset", "compute-attribution", "train-lds", "evaluate"):
        _run_parser(subparsers, command)
    return parser


def _default_scope(config: dict) -> str | None:
    return config.get("attribution", {}).get("default_scope")


def _plan(command: str, experiment: str, scope_name: str | None) -> dict:
    root = repo_root()
    config = load_experiment(experiment, root)
    selected_scope = scope_name or _default_scope(config)
    scope = None
    if selected_scope is not None:
        scope = scope_summary(load_scope(experiment, selected_scope, root))
    output_root = Path(
        os.environ.get("PINT_OUTPUT_ROOT", str(root / "outputs"))
    ).expanduser().resolve()
    if selected_scope:
        output_root = output_root / experiment / selected_scope
    else:
        output_root = output_root / experiment
    return {
        "command": command,
        "experiment": experiment,
        "scope": scope,
        "dataset_manifest": config["dataset"]["manifest"],
        "queries": config["queries"]["count"],
        "checkpoint_count": config["training"]["checkpoint_count"],
        "output_root": str(output_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    if args.command == "validate":
        for name in EXPERIMENT_NAMES:
            config = load_experiment(name, root)
            default_scope = _default_scope(config)
            if default_scope:
                scope = load_scope(name, default_scope, root)
                if args.check_data and scope.selection == "explicit_indices":
                    if scope.indices_path is None or not scope.indices_path.is_file():
                        raise FileNotFoundError(scope.indices_path)
            print(f"[ok] {name}")
        return 0
    if args.command == "show-config":
        print(yaml.safe_dump(load_experiment(args.experiment, root), sort_keys=False))
        return 0

    plan = _plan(args.command, args.experiment, args.scope)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    print(
        "Backend migration is not complete; rerun with --dry-run to inspect the plan.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
