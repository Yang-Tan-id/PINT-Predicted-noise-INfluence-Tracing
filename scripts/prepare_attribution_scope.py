#!/usr/bin/env python3
from __future__ import annotations

import sys

from pint.datasets.scope_manifest import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
