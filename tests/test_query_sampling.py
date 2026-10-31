from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from pint.sampling.cli import main, sampling_plan
from pint.sampling.artbench2 import _model_seed as artbench_model_seed
from pint.sampling.jax_pixel import (
    _final_checkpoint,
    default_ddim_steps,
    inference_timesteps,
)
from pint.sampling.queries import (
    artbench2_queries,
    celeba_queries,
    default_query_count,
    default_query_seed,
    shape3d_queries,
    x3_queries,
)
from pint.sampling.x3 import _model_seed as x3_model_seed


class QuerySamplingTests(unittest.TestCase):
    def test_default_counts(self) -> None:
        self.assertEqual(default_query_count("x3"), 100)
        self.assertEqual(default_query_count("shape3d"), 100)
        self.assertEqual(default_query_count("celeba"), 100)
        self.assertEqual(default_query_count("artbench2"), 60)

    def test_plan_accepts_custom_count(self) -> None:
        self.assertEqual(sampling_plan("shape3d", 17)["query_count"], 17)

    def test_query_seed_defaults_match_experiment_contracts(self) -> None:
        self.assertEqual(default_query_seed("shape3d"), 100)
        self.assertEqual(sampling_plan("shape3d")["seed"], 100)
        self.assertEqual(default_query_seed("x3"), 0)

    def test_custom_seed_offsets_initial_seeds(self) -> None:
        self.assertEqual(shape3d_queries(3, seed=90)[0]["initial_seed"], 90)
        self.assertEqual(shape3d_queries(3, seed=90)[-1]["initial_seed"], 92)
        artbench = artbench2_queries(4, seed=50)
        self.assertEqual(
            [record["initial_seed"] for record in artbench], [50, 50, 51, 51]
        )

    def test_shape3d_queries_match_paper_lds_contract(self) -> None:
        records = shape3d_queries(100)
        self.assertEqual(len(records), 100)
        for record in records:
            labels = record["labels"]
            self.assertEqual(len(labels), 4)
            self.assertTrue(labels[0].startswith("shape_"))
            self.assertTrue(labels[1].startswith("object_hue_"))
            self.assertTrue(labels[2].startswith("wall_hue_"))
            self.assertTrue(labels[3].startswith("floor_hue_"))
        self.assertEqual(records[0]["initial_seed"], 100)
        self.assertEqual(records[-1]["initial_seed"], 199)

    def test_refined_ddim_timestep_grid(self) -> None:
        self.assertEqual(inference_timesteps(1000, 50), list(range(999, -1, -20)))
        self.assertEqual(inference_timesteps(1000, 1000), list(range(999, -1, -1)))

    def test_dataset_specific_default_ddim_steps(self) -> None:
        self.assertEqual(default_ddim_steps("celeba"), 50)
        self.assertEqual(default_ddim_steps("shape3d"), 1000)

    def test_jax_checkpoint_directory_rejects_multiple_model_seeds(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seed_1_epoch_0200.ckpt").touch()
            (root / "seed_2_epoch_0200.ckpt").touch()
            with self.assertRaisesRegex(RuntimeError, "exactly one model seed"):
                _final_checkpoint(root)

    def test_jax_checkpoint_selects_latest_epoch_for_one_seed(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            early = root / "seed_42_epoch_0004.ckpt"
            final = root / "seed_42_epoch_0200.ckpt"
            early.touch()
            final.touch()
            self.assertEqual(_final_checkpoint(root), final)

    def test_torch_samplers_read_model_seed_metadata(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x3_root = root / "x3"
            x3_meta = x3_root / "base" / "prompted" / "meta.json"
            x3_meta.parent.mkdir(parents=True)
            x3_meta.write_text(json.dumps({"seed": 67}), encoding="utf-8")
            self.assertEqual(x3_model_seed(x3_root, "prompted", {"seed": 67}), 67)

            artbench_root = root / "artbench2"
            artbench_root.mkdir()
            (artbench_root / "training_config.json").write_text(
                json.dumps({"seed": 42}), encoding="utf-8"
            )
            self.assertEqual(artbench_model_seed(artbench_root), 42)

    def test_x3_sampler_rejects_conflicting_model_seed_metadata(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "base" / "prompted" / "meta.json"
            meta.parent.mkdir(parents=True)
            meta.write_text(json.dumps({"seed": 67}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Conflicting x3 model seeds"):
                x3_model_seed(root, "prompted", {"seed": 42})

    def test_x3_queries_are_all_prompted(self) -> None:
        vocabulary = (
            "shape_color_red",
            "shape_color_blue",
            "background_color_red",
            "background_color_blue",
            "shape_circle",
            "shape_square",
        )
        records = x3_queries(100, vocabulary)
        self.assertEqual(len(records), 100)
        self.assertEqual({record["family"] for record in records}, {"prompted"})
        self.assertTrue(all(len(record["labels"]) == 3 for record in records))

    def test_celeba_queries_use_prompted_attr40_contract(self) -> None:
        names = ["Blurry", *[f"Attribute_{index}" for index in range(39)]]
        attributes = {}
        partitions = {}
        for index in range(30):
            filename = f"{index:06d}.jpg"
            row = np.full(40, -1, dtype=np.int8)
            row[1:5] = 1
            attributes[filename] = row
            partitions[filename] = 1
        records = celeba_queries(100, names, attributes, partitions, seed=10)
        self.assertEqual(len(records), 100)
        self.assertEqual({record["family"] for record in records}, {"prompted"})
        self.assertEqual([record["initial_seed"] for record in records[:4]], [10, 11, 12, 13])
        self.assertEqual(len(records[0]["positive_attributes"]), 4)

    def test_artbench_queries_alternate_classes(self) -> None:
        records = artbench2_queries(60)
        self.assertEqual(len(records), 60)
        self.assertEqual(records[0]["label"], "post_impressionism")
        self.assertEqual(records[1]["label"], "ukiyo_e")
        self.assertEqual(records[0]["initial_seed"], 0)
        self.assertEqual(records[1]["initial_seed"], 0)
        self.assertEqual(records[-1]["initial_seed"], 29)

    def test_dry_run_selects_cuda_device_without_loading_backend(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                main(
                    [
                        "--experiment",
                        "artbench2",
                        "--num-queries",
                        "7",
                        "--gpu-id",
                        "2",
                        "--dry-run",
                    ]
                ),
                0,
            )
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")


if __name__ == "__main__":
    unittest.main()
