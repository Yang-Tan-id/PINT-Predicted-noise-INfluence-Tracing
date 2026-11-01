from pathlib import Path
import sys

from pint.evaluation.cli import main


if __name__ == "__main__":
    experiment = Path(__file__).resolve().parent.name
    raise SystemExit(main(["--stage", "evaluate", "--experiment", experiment, *sys.argv[1:]]))
