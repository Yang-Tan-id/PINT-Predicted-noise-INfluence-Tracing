from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pint.training.cli import main


class TrainingDispatchTests(unittest.TestCase):
    def fake_module(self, name: str) -> tuple[types.ModuleType, Mock]:
        module = types.ModuleType(name)
        entrypoint = Mock(return_value=0)
        module.main = entrypoint
        return module, entrypoint

    def test_x3_dispatches_to_pytorch_backend(self) -> None:
        module, entrypoint = self.fake_module("pint.training.x3_training")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "x3"
            with patch.dict(os.environ, {}, clear=False):
                with patch.dict(sys.modules, {module.__name__: module}):
                    result = main(
                        [
                            "--experiment",
                            "x3",
                            "--gpu-id",
                            "2",
                            "--output-dir",
                            str(output),
                            "--family",
                            "prompted",
                        ]
                    )
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(result, 0)
        entrypoint.assert_called_once_with(
            ["--output-dir", str(output), "--family", "prompted"]
        )

    def test_shape3d_dispatches_to_jax_backend(self) -> None:
        module, entrypoint = self.fake_module("pint.training.jax_training")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shape3d"
            with patch.dict(sys.modules, {module.__name__: module}):
                result = main(
                    [
                        "--experiment",
                        "shape3d",
                        "--output-dir",
                        str(output),
                        "--batch-size",
                        "16",
                    ]
                )
        self.assertEqual(result, 0)
        entrypoint.assert_called_once_with(
            [
                "--experiment",
                "shape3d",
                "--output-dir",
                str(output),
                "--batch-size",
                "16",
            ]
        )

    def test_celeba_dispatches_to_jax_backend(self) -> None:
        module, entrypoint = self.fake_module("pint.training.jax_training")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "celeba"
            with patch.dict(sys.modules, {module.__name__: module}):
                result = main(
                    ["--experiment", "celeba", "--output-dir", str(output)]
                )
        self.assertEqual(result, 0)
        entrypoint.assert_called_once_with(
            ["--experiment", "celeba", "--output-dir", str(output)]
        )

    def test_artbench2_dispatches_to_lora_backend(self) -> None:
        name = "pint.training.backends.artbench_lora"
        module, entrypoint = self.fake_module(name)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artbench2"
            with patch.dict(sys.modules, {name: module}):
                result = main(
                    [
                        "--experiment",
                        "artbench2",
                        "--output-dir",
                        str(output),
                        "--epochs",
                        "1",
                    ]
                )
        self.assertEqual(result, 0)
        entrypoint.assert_called_once_with(
            ["--output-dir", str(output), "--epochs", "1"]
        )


if __name__ == "__main__":
    unittest.main()
