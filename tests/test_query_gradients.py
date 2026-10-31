from __future__ import annotations

import os
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from pint.query_gradients.cli import main, query_gradient_plan
from pint.query_gradients.common import (
    checkpoint_pair_range,
    normalization_for,
    timestep_count_for,
    timestep_grid,
    write_run_manifest,
)


class QueryGradientTests(unittest.TestCase):
    def test_checkpoint_pair_shards_are_balanced_contiguous_and_complete(self) -> None:
        shards = [checkpoint_pair_range(10, index, 3) for index in range(3)]
        self.assertEqual(
            [(shard.start, shard.stop) for shard in shards],
            [(0, 4), (4, 7), (7, 10)],
        )
        self.assertEqual(
            [value for shard in shards for value in shard], list(range(10))
        )

    def test_checkpoint_pair_shards_reject_invalid_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            checkpoint_pair_range(10, 3, 3)
        with self.assertRaises(ValueError):
            checkpoint_pair_range(10, 0, 0)

    def test_experiment_normalization_contract(self) -> None:
        for experiment in ("x3", "shape3d", "celeba", "artbench2"):
            self.assertEqual(
                normalization_for(experiment), "predicted_noise_delta_l2"
            )

    def test_timestep_grids_match_sampling(self) -> None:
        for experiment in ("x3", "shape3d"):
            grid = timestep_grid(experiment)
            self.assertEqual(grid.shape, (100,))
            self.assertEqual((int(grid[0]), int(grid[-1])), (999, 0))
            self.assertEqual(timestep_count_for(experiment), 100)
        self.assertTrue(
            np.array_equal(
                timestep_grid("x3"),
                np.linspace(999, 0, 100, dtype=np.int32),
            )
        )
        for experiment in ("celeba", "artbench2"):
            grid = timestep_grid(experiment)
            self.assertEqual(grid.shape, (50,))
            self.assertEqual((int(grid[0]), int(grid[-1])), (999, 19))
            self.assertEqual(timestep_count_for(experiment), 50)

    def test_plan_defaults_to_4096_and_checkpoint_major(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            with patch.dict(os.environ, {"PINT_OUTPUT_ROOT": output}, clear=False):
                plan = query_gradient_plan("shape3d")
        self.assertEqual(plan["projection_dim"], 4096)
        self.assertEqual(plan["timesteps"], 100)
        self.assertTrue(plan["checkpoint_major"])
        self.assertEqual(query_gradient_plan("celeba")["timesteps"], 50)

    def test_cli_rejects_timestep_override(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--experiment", "x3", "--timesteps", "50", "--dry-run"])

    def test_manifest_rejects_mixed_projection_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_manifest = root / "queries.json"
            query_manifest.write_text('{"queries": [{"query_id": 0}]}')
            arguments = {
                "output_dir": root / "output",
                "experiment": "x3",
                "query_manifest": query_manifest,
                "checkpoints": [root / "one.pt", root / "two.pt"],
                "timesteps": timestep_grid("x3"),
                "batch_size": 2,
            }
            write_run_manifest(projection_dim=4096, **arguments)
            with self.assertRaises(RuntimeError):
                write_run_manifest(projection_dim=2048, **arguments)

    def test_cli_forwards_batch_and_projection_to_backend(self) -> None:
        module = types.ModuleType("pint.query_gradients.x3")
        entrypoint = Mock(return_value=0)
        module.main = entrypoint
        with tempfile.TemporaryDirectory() as output:
            with patch.dict(
                os.environ, {"PINT_OUTPUT_ROOT": output}, clear=False
            ), patch.dict(sys.modules, {module.__name__: module}):
                result = main(
                    [
                        "--experiment", "x3",
                        "--batch-size", "17",
                        "--projection-dim", "2048",
                    ]
                )
        self.assertEqual(result, 0)
        forwarded = entrypoint.call_args.args[0]
        self.assertEqual(forwarded[forwarded.index("--batch-size") + 1], "17")
        self.assertNotIn("--timesteps", forwarded)
        self.assertEqual(forwarded[forwarded.index("--projection-dim") + 1], "2048")

    @patch("pint.query_gradients.cli.subprocess.Popen")
    def test_cli_splits_checkpoint_ranges_across_gpu_workers(self, popen: Mock) -> None:
        workers = [Mock(), Mock()]
        for worker in workers:
            worker.wait.return_value = 0
        popen.side_effect = workers
        with tempfile.TemporaryDirectory() as output:
            with patch.dict(os.environ, {"PINT_OUTPUT_ROOT": output}, clear=False):
                result = main(["--experiment", "x3", "--gpu-ids", "2,5"])
        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 2)
        for index, (call, gpu_id) in enumerate(zip(popen.call_args_list, (2, 5))):
            command = call.args[0]
            self.assertEqual(
                command[command.index("--shard-index") + 1], str(index)
            )
            self.assertEqual(command[command.index("--shard-count") + 1], "2")
            self.assertEqual(
                call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(gpu_id)
            )

    def test_x3_backend_processes_complete_query_batch(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the unit-test environment")
        from pint.query_gradients.x3 import main as x3_main
        from pint.training.backends.x3_model import CondEpsModel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "models" / "base" / "prompted"
            query_dir = root / "queries"
            model_dir.mkdir(parents=True)
            query_dir.mkdir()
            vocab = {"red": 0, "blue": 1}
            model = CondEpsModel(3, 2, base_ch=8, time_dim=8)
            for epoch in (1, 2, 3):
                if epoch > 1:
                    with torch.no_grad():
                        next(model.parameters()).add_(0.01)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "seed": 7,
                        "cond_dim": 2,
                        "T": 1000,
                        "vocab": vocab,
                        "config": {"base_channels": 8, "time_embedding_dim": 8},
                    },
                    model_dir / f"epoch_{epoch:04d}.pt",
                )
            records = []
            for query_id, label in enumerate(vocab):
                path = query_dir / f"q{query_id:02d}"
                path.mkdir()
                np.save(path / "final_state.npy", np.zeros((1, 3, 3, 3), np.float32))
                np.save(path / "trajectory_t.npy", timestep_grid("x3"))
                records.append(
                    {
                        "query_id": query_id,
                        "labels": [label],
                        "dir": str(path),
                        "model_seed": 7,
                    }
                )
            manifest = query_dir / "manifest.json"
            manifest.write_text(json.dumps(records), encoding="utf-8")
            output = root / "gradients"
            result = x3_main(
                [
                    "--experiment", "x3",
                    "--model-dir", str(root / "models"),
                    "--query-manifest", str(manifest),
                    "--output-dir", str(output),
                    "--batch-size", "2",
                    "--projection-dim", "16",
                    "--device", "cpu",
                ]
            )
            with np.load(output / "checkpoint_0000.npz") as payload:
                self.assertEqual(payload["query_features"].shape, (2, 100, 16))
                self.assertEqual(
                    str(payload["normalization"]), "predicted_noise_delta_l2"
                )
            sharded_output = root / "sharded_gradients"
            sharded_result = x3_main(
                [
                    "--experiment", "x3",
                    "--model-dir", str(root / "models"),
                    "--query-manifest", str(manifest),
                    "--output-dir", str(sharded_output),
                    "--batch-size", "2",
                    "--projection-dim", "16",
                    "--device", "cpu",
                    "--shard-index", "1",
                    "--shard-count", "2",
                ]
            )
            self.assertFalse((sharded_output / "checkpoint_0000.npz").exists())
            with np.load(sharded_output / "checkpoint_0001.npz") as payload:
                self.assertEqual(int(payload["checkpoint_index"]), 1)
        self.assertEqual(result, 0)
        self.assertEqual(sharded_result, 0)


if __name__ == "__main__":
    unittest.main()
