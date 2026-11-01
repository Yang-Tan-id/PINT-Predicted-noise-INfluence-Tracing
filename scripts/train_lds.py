#!/usr/bin/env python3
from __future__ import annotations

from pint.evaluation.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["--stage", "train", *__import__("sys").argv[1:]]))
