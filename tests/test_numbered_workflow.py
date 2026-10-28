from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pint.config import EXPERIMENT_NAMES, repo_root
from pint.datasets.x3 import generate_csv
from pint.training.cli import training_plan


class NumberedWorkflowTests(unittest.TestCase):
    def test_every_experiment_has_identical_numbered_entrypoints(self) -> None:
        root = repo_root() / "experiments"
        prepare_contents = {
            (root / name / "00_prepare_dataset.py").read_text(encoding="utf-8")
            for name in EXPERIMENT_NAMES
        }
        train_contents = {
            (root / name / "01_train_model.py").read_text(encoding="utf-8")
            for name in EXPERIMENT_NAMES
        }
        self.assertEqual(len(prepare_contents), 1)
        self.assertEqual(len(train_contents), 1)

    def test_x3_generator_writes_expected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x3.csv"
            generate_csv(path, num_samples=3, seed=67)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["id"], "0")
        self.assertEqual(set(rows[0]), {"id", *[f"tile_{i}" for i in range(1, 10)], "label_1", "label_2", "label_3"})

    def test_training_plan_uses_fixed_checkpoint_count(self) -> None:
        with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as output:
            with patch.dict(
                os.environ,
                {"PINT_DATA_ROOT": data, "PINT_OUTPUT_ROOT": output},
                clear=False,
            ):
                shape = training_plan("shape3d")
                art = training_plan("artbench2")
        self.assertEqual(shape["checkpoint_count"], 50)
        self.assertEqual(art["checkpoint_count"], 25)
        self.assertTrue(shape["progress_bar"])

    def test_paths_default_to_repository_directories(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            shape = training_plan("shape3d")
        self.assertEqual(
            shape["dataset"], str(repo_root() / "data" / "shape3d" / "20000" / "dataset.npz")
        )
        self.assertEqual(
            shape["output"], str(repo_root() / "outputs" / "shape3d" / "model_training")
        )


if __name__ == "__main__":
    unittest.main()
