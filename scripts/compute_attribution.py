#!/usr/bin/env python3
from __future__ import annotations

import sys

from pint.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["compute-attribution", *sys.argv[1:]]))
