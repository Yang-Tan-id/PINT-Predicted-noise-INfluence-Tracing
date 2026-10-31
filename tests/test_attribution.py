from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from pint.attribution.cli import attribution_plan, main
from pint.attribution.common import datapoint_shard, merge_shards, save_shard


class AttributionTests(unittest.TestCase):
    @patch("pint.baselines.attribution_cli.main")
    def test_numbered_attribution_entrypoint_dispatches_baseline_method(
        self, baseline_main: Mock
    ) -> None:
        baseline_main.return_value = 0
        arguments = ["--experiment", "celeba", "--method", "retrac", "--dry-run"]
        self.assertEqual(main(arguments), 0)
        baseline_main.assert_called_once_with(arguments)

    def test_datapoint_shards_are_balanced_contiguous_and_complete(self) -> None:
        shards = [datapoint_shard(10, index, 3) for index in range(3)]
        self.assertEqual(
            [(shard.start, shard.stop) for shard in shards],
            [(0, 4), (4, 7), (7, 10)],
        )
        self.assertEqual([value for shard in shards for value in shard], list(range(10)))

    def test_plan_inherits_fixed_timestep_and_mc_contract(self) -> None:
        self.assertEqual(attribution_plan("shape3d")["timesteps"], 100)
        self.assertEqual(attribution_plan("celeba")["timesteps"], 50)
        self.assertEqual(attribution_plan("artbench2")["mc_samples"], 1)
        self.assertEqual(attribution_plan("shape3d")["scope"], "paper_5k")
        self.assertEqual(attribution_plan("x3")["scope"], "all")
        self.assertEqual(attribution_plan("x3")["score_normalization"], "raw")
        self.assertEqual(
            attribution_plan("artbench2")["score_normalization"],
            "query_train_l2",
        )
        self.assertEqual(
            attribution_plan("x3")["parameter_sources"],
            ["raw_gradient", "adamw_full"],
        )
        self.assertEqual(
            attribution_plan("x3", adamw_enabled=False)["parameter_sources"],
            ["raw_gradient"],
        )

    def test_merge_preserves_datapoint_order_and_scores(self) -> None:
        metadata = {
            "format_version": 2,
            "experiment": "x3",
            "method": "pint",
            "normalization": "predicted_noise_delta_l2",
            "score_normalization": "raw",
            "projection": "countsketch",
            "projection_dim": 4,
            "timesteps": [9, 0],
            "checkpoint_pairs": 2,
            "query_manifest_sha256": "digest",
            "scope": "all",
            "scope_size": 3,
            "model_seed": 7,
            "mc_samples": 1,
            "mc_pairing": "shared_with_query_by_checkpoint_and_timestep",
            "reduction": "checkpoint_sum_of_timestep_mean_dot_products",
            "learning_rate_weighted": True,
            "parameter_sources": ["raw_gradient", "adamw_full"],
            "adamw_enabled": True,
            "adamw_semantics": "next full AdamW direction from saved moments; learning rate applied once",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_shard(
                root, 0,
                raw_scores=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                adamw_scores=np.asarray([[10.0, 20.0], [30.0, 40.0]]),
                train_indices=np.asarray([4, 7]),
                query_ids=np.asarray([10, 11]),
                metadata={**metadata, "shard_index": 0},
            )
            save_shard(
                root, 1,
                raw_scores=np.asarray([[5.0], [6.0]]),
                adamw_scores=np.asarray([[50.0], [60.0]]),
                train_indices=np.asarray([9]),
                query_ids=np.asarray([10, 11]),
                metadata={**metadata, "shard_index": 1},
            )
            output = merge_shards(root, 2)
            with np.load(output, allow_pickle=False) as payload:
                self.assertTrue(np.array_equal(payload["train_indices"], [4, 7, 9]))
                self.assertTrue(
                    np.array_equal(payload["raw_scores"], [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])
                )
                self.assertTrue(
                    np.array_equal(payload["adamw_scores"], [[10.0, 20.0, 50.0], [30.0, 40.0, 60.0]])
                )
                manifest = json.loads(str(payload["manifest_json"].item()))
                self.assertEqual(manifest["shard_count"], 2)

    def test_x3_backend_scores_a_small_datapoint_batch(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the unit-test environment")
        from pint.attribution.x3 import main as x3_attribution
        from pint.datasets.x3 import generate_csv
        from pint.training.backends.x3_dataset import ColorGridDataset
        from pint.training.backends.x3_model import CondEpsModel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "x3.csv"
            generate_csv(dataset, num_samples=3, seed=67)
            loaded_dataset = ColorGridDataset(str(dataset), grid_size=3)
            model_root = root / "models"
            checkpoint_root = model_root / "base" / "prompted"
            checkpoint_root.mkdir(parents=True)
            model = CondEpsModel(3, len(loaded_dataset.vocab), base_ch=8, time_dim=8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            payload = {
                "model_state": model.state_dict(),
                "seed": 67,
                "cond_dim": len(loaded_dataset.vocab),
                "T": 1000,
                "vocab": loaded_dataset.vocab,
                "config": {
                    "base_channels": 8,
                    "time_embedding_dim": 8,
                    "weight_decay": 1e-4,
                    "grad_clip_norm": 1.0,
                },
                "learning_rate_at_checkpoint": 1e-4,
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(payload, checkpoint_root / "epoch_0001.pt")
            with torch.no_grad():
                next(model.parameters()).add_(0.01)
            payload = {**payload, "model_state": model.state_dict()}
            torch.save(payload, checkpoint_root / "epoch_0002.pt")
            (checkpoint_root / "training_config.json").write_text(
                json.dumps({"dataset": str(dataset)}), encoding="utf-8"
            )
            query_manifest = root / "queries.json"
            query_manifest.write_text('{"queries": [{"query_id": 0}]}', encoding="utf-8")
            gradients = root / "query_gradients"
            gradients.mkdir()
            manifest = {
                "experiment": "x3",
                "normalization": "predicted_noise_delta_l2",
                "projection": "countsketch",
                "projection_dim": 16,
                "checkpoint_pairs": 1,
                "timesteps": [999, 0],
            }
            (gradients / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.savez(
                gradients / "checkpoint_0000.npz",
                checkpoint_index=np.asarray(0),
                timesteps=np.asarray([999, 0], dtype=np.int32),
                projection_dim=np.asarray(16),
                query_ids=np.asarray([0], dtype=np.int32),
                query_features=np.ones((1, 2, 16), dtype=np.float32),
            )
            output = root / "attribution"
            result = x3_attribution(
                [
                    "--experiment", "x3",
                    "--model-dir", str(model_root),
                    "--query-manifest", str(query_manifest),
                    "--query-gradient-dir", str(gradients),
                    "--output-dir", str(output),
                    "--batch-size", "2",
                    "--device", "cpu",
                ]
            )
            with np.load(output / "shards" / "shard_0000.npz", allow_pickle=False) as scored:
                self.assertEqual(scored["raw_scores"].shape, (1, 3))
                self.assertEqual(scored["adamw_scores"].shape, (1, 3))
                self.assertTrue(np.isfinite(scored["raw_scores"]).all())
                self.assertTrue(np.isfinite(scored["adamw_scores"]).all())
        self.assertEqual(result, 0)

    def test_artbench_linear_hooks_match_scalar_per_sample_gradients(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the unit-test environment")
        from pint.attribution.artbench2 import LinearGradSampleCapture

        module = torch.nn.Sequential(
            torch.nn.Linear(3, 4, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(4, 2, bias=False),
        )
        parameters = list(module.parameters())
        inputs = torch.tensor(
            [[1.0, 2.0, -1.0], [-3.0, 4.0, 0.5], [2.0, -1.0, 3.0]]
        )
        targets = torch.tensor([[0.5, -0.5], [1.0, 2.0], [-1.0, 0.25]])
        capture = LinearGradSampleCapture(module, parameters)
        capture.start(len(inputs))
        losses = torch.nn.functional.mse_loss(
            module(inputs), targets, reduction="none"
        ).mean(dim=1)
        actual = capture.gradients(losses)
        expected_by_sample = []
        for index in range(len(inputs)):
            loss = torch.nn.functional.mse_loss(
                module(inputs[index : index + 1]),
                targets[index : index + 1],
            )
            expected_by_sample.append(torch.autograd.grad(loss, parameters))
        expected = [
            torch.stack([sample[slot] for sample in expected_by_sample])
            for slot in range(len(parameters))
        ]
        for captured, scalar in zip(actual, expected):
            self.assertTrue(torch.allclose(captured, scalar, atol=1e-6, rtol=1e-5))
        capture.close()

    def test_batched_adamw_direction_matches_first_step_formula(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the unit-test environment")
        from pint.attribution.torch_adamw import adamw_full_direction_batch

        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.AdamW(
            [parameter], lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1
        )
        gradients = [torch.tensor([[2.0, -4.0], [-3.0, 1.0]])]
        directions, learning_rate = adamw_full_direction_batch(
            [parameter], gradients, optimizer, max_grad_norm=None
        )
        expected = -(
            gradients[0] / (gradients[0].abs() + 1e-8)
            + 0.1 * parameter.detach().unsqueeze(0)
        )
        self.assertTrue(torch.allclose(directions[0], expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(learning_rate, 3e-4)


if __name__ == "__main__":
    unittest.main()
