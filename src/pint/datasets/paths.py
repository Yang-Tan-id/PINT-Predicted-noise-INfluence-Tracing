from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    value = os.environ.get("PINT_DATA_ROOT")
    if not value:
        raise RuntimeError(
            "PINT_DATA_ROOT is not set. Choose an external dataset directory, for example: "
            "export PINT_DATA_ROOT=/scratch/$USER/pint-data"
        )
    return Path(value).expanduser().resolve()
