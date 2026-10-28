from __future__ import annotations

import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a PINT experiment dataset", add_help=False
    )
    parser.add_argument("--experiment", required=True, choices=("shape3d", "celeba"))
    args, backend_args = parser.parse_known_args(argv)
    if args.experiment == "shape3d":
        from .shape3d import main as prepare
    else:
        from .celeba import main as prepare
    return prepare(backend_args)


if __name__ == "__main__":
    raise SystemExit(main())
