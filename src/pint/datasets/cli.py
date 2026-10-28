from __future__ import annotations

import argparse
from typing import Sequence

from pint.config import EXPERIMENT_NAMES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a PINT experiment dataset", add_help=False
    )
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)
    args, backend_args = parser.parse_known_args(argv)
    if args.experiment == "x3":
        from .x3 import main as prepare
    elif args.experiment == "shape3d":
        from .shape3d import main as prepare
    elif args.experiment == "celeba":
        from .celeba import main as prepare
    else:
        from .artbench2 import main as prepare
    return prepare(backend_args)


if __name__ == "__main__":
    raise SystemExit(main())
