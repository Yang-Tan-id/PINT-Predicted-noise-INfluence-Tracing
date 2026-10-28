from __future__ import annotations

import os
from pathlib import Path

from pint.config import repo_root


def data_root() -> Path:
    value = os.environ.get("PINT_DATA_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return repo_root() / "data"
