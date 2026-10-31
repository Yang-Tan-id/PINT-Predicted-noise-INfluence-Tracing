from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pint.sampling.cli import main


class SamplingDispatchTests(unittest.TestCase):
    def fake_module(self, name: str) -> tuple[types.ModuleType, Mock]:
        module = types.ModuleType(name)
        entrypoint = Mock(return_value=0)
        module.main = entrypoint
        return module, entrypoint

    def test_x3_custom_query_count_is_forwarded(self) -> None:
        module, entrypoint = self.fake_module("pint.sampling.x3")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "queries"
            model = Path(directory) / "models"
            with patch.dict(sys.modules, {module.__name__: module}):
                result = main(
                    [
                        "--experiment",
                        "x3",
                        "--num-queries",
                        "13",
                        "--seed",
                        "90",
                        "--model-dir",
                        str(model),
                        "--output-dir",
                        str(output),
                    ]
                )
        self.assertEqual(result, 0)
        entrypoint.assert_called_once_with(
            [
                "--model-dir",
                str(model),
                "--output-dir",
                str(output),
                "--num-queries",
                "13",
                "--seed",
                "90",
            ]
        )

    def test_celeba_uses_jax_sampler(self) -> None:
        module, entrypoint = self.fake_module("pint.sampling.jax_pixel")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "queries"
            model = Path(directory) / "models"
            with patch.dict(sys.modules, {module.__name__: module}):
                result = main(
                    [
                        "--experiment",
                        "celeba",
                        "--model-dir",
                        str(model),
                        "--output-dir",
                        str(output),
                    ]
                )
        self.assertEqual(result, 0)
        entrypoint.assert_called_once_with(
            [
                "--experiment",
                "celeba",
                "--model-dir",
                str(model),
                "--output-dir",
                str(output),
                "--num-queries",
                "100",
                "--seed",
                "0",
            ]
        )

    def test_artbench_uses_60_queries_by_default(self) -> None:
        module, entrypoint = self.fake_module("pint.sampling.artbench2")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "queries"
            model = Path(directory) / "models"
            with patch.dict(sys.modules, {module.__name__: module}):
                result = main(
                    [
                        "--experiment",
                        "artbench2",
                        "--model-dir",
                        str(model),
                        "--output-dir",
                        str(output),
                    ]
                )
        self.assertEqual(result, 0)
        entrypoint.assert_called_once_with(
            [
                "--model-dir",
                str(model),
                "--output-dir",
                str(output),
                "--num-queries",
                "60",
                "--seed",
                "0",
            ]
        )


if __name__ == "__main__":
    unittest.main()
