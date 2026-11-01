from __future__ import annotations

import tempfile
import unittest
import os
import pickle
import csv
import io
import json
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from pint.evaluation.common import (
    LDS_MODEL_COUNT,
    LDS_SUBSET_SIZES,
    EvaluationJob,
    _topk_columns,
    artbench_ranking_sign,
    merge_true_f,
    prepare_lds_jobs,
    prepare_artbench_jobs,
    true_f_complete,
)
from pint.evaluation.cli import main as evaluation_main
from pint.evaluation.metrics_artbench2 import (
    _expected_timesteps,
    _is_all_black,
    _trajectory_metrics,
)
from pint.evaluation.metrics_jax import (
    simple_loss_probe_seed,
    simple_loss_timesteps,
)
from pint.evaluation.reporting import finite_spearman, write_reports
from pint.training.model_versions import (
    ARTBENCH_CLIP_MODEL_REVISION,
    ARTBENCH_DINO_REVISION,
)


class EvaluationTests(unittest.TestCase):
    def test_artbench_cli_defaults_to_single_adamw_source(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch(
            "pint.evaluation.cli.output_root", return_value=Path(directory)
        ), patch("pint.evaluation.cli.prepare_artbench_jobs", return_value=[]) as prepare:
            with redirect_stdout(output):
                self.assertEqual(
                    evaluation_main([
                        "--experiment", "artbench2", "--stage", "train", "--dry-run"
                    ]),
                    0,
                )
        self.assertEqual(prepare.call_args.kwargs["parameter_source"], "adamw")
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["parameter_source"], "adamw")
        self.assertEqual(plan["model_parameters"], "lora")

    def test_jax_simple_loss_matches_latest_paper_probe_contracts(self) -> None:
        shape_timesteps = simple_loss_timesteps("shape3d", 1000)
        celeba_timesteps = simple_loss_timesteps("celeba", 1000)
        self.assertTrue(np.array_equal(
            shape_timesteps, np.linspace(999, 0, 100, dtype=np.int32)
        ))
        self.assertTrue(np.array_equal(
            celeba_timesteps, np.linspace(999, 0, 50, dtype=np.int32)
        ))
        self.assertEqual(len(np.unique(shape_timesteps)), 100)
        self.assertEqual(len(np.unique(celeba_timesteps)), 50)
        self.assertIsNone(simple_loss_probe_seed("shape3d", 7))
        self.assertEqual(simple_loss_probe_seed("celeba", 0), 2026)
        self.assertEqual(simple_loss_probe_seed("celeba", 7), 702_047)

    def test_paper_lds_contract(self) -> None:
        self.assertEqual(LDS_MODEL_COUNT, 192)
        self.assertEqual(LDS_SUBSET_SIZES["x3"], 12_500)
        self.assertEqual(LDS_SUBSET_SIZES["shape3d"], 2_500)
        self.assertEqual(LDS_SUBSET_SIZES["celeba"], 2_500)

    def test_shape3d_subsets_are_deterministic_and_use_three_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = prepare_lds_jobs("shape3d", root)
            first_zero = np.load(first[0].train_indices)
            first_64 = np.load(first[64].train_indices)
            second = prepare_lds_jobs("shape3d", root)
            second_zero = np.load(second[0].train_indices)
        self.assertEqual(len(first), 192)
        self.assertEqual(len(first_zero), 2_500)
        self.assertTrue(np.array_equal(first_zero, second_zero))
        self.assertFalse(np.array_equal(first_zero, first_64))
        self.assertEqual(first[0].subset_id, 0)
        self.assertEqual(first[-1].subset_id, 191)

    def test_x3_first_mask_matches_original_direct_rng_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            jobs = prepare_lds_jobs("x3", Path(directory))
            actual = np.load(jobs[0].train_indices)
        expected = np.sort(
            np.random.default_rng(0).choice(50_000, size=12_500, replace=False)
        )
        self.assertTrue(np.array_equal(actual, expected))

    def test_artbench_builds_one_removal_model_per_query_and_score_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            data = root / "data"
            score_dir = output / "artbench2" / "attribution" / "timesteps_50"
            score_dir.mkdir(parents=True)
            np.savez_compressed(
                score_dir / "scores.npz",
                query_ids=np.asarray([3, 7, 11]),
                train_indices=np.arange(10),
                raw_scores=np.arange(30, dtype=np.float64).reshape(3, 10),
                adamw_scores=np.arange(30, dtype=np.float64).reshape(3, 10)[:, ::-1],
            )
            index_dir = data / "artbench2" / "indices"
            index_dir.mkdir(parents=True)
            with (index_dir / "idx-train.pkl").open("wb") as handle:
                pickle.dump(list(range(100, 110)), handle)
            with patch.dict(
                os.environ,
                {"PINT_OUTPUT_ROOT": str(output), "PINT_DATA_ROOT": str(data)},
                clear=False,
            ):
                jobs = prepare_artbench_jobs(
                    output / "artbench2" / "evaluation",
                    method="pint", projection_dim=4096,
                    parameter_source="adamw", topk=2,
                )
        self.assertEqual(len(jobs), 3)
        self.assertEqual({job.query_id for job in jobs}, {3, 7, 11})
        self.assertEqual({job.parameter_source for job in jobs}, {"adamw"})

    def test_artbench_ranking_matches_original_raw_and_adamw_orientations(self) -> None:
        self.assertEqual(artbench_ranking_sign("pint", "raw"), 1)
        self.assertEqual(artbench_ranking_sign("pint", "adamw"), 1)
        for method in ("gas", "tracincp", "retrac"):
            self.assertEqual(artbench_ranking_sign(method, "raw"), 1)
            self.assertEqual(artbench_ranking_sign(method, "adamw"), 1)
        values = np.asarray([3.0, np.nan, -4.0, 2.0])
        self.assertTrue(np.array_equal(_topk_columns(values, 2, +1), [0, 3]))
        self.assertTrue(np.array_equal(_topk_columns(values, 2, -1), [2, 3]))

    def test_artbench_trajectory_metrics_are_latent_state_mse(self) -> None:
        reference = np.zeros((3, 4, 2, 2), dtype=np.float32)
        candidate = np.ones_like(reference)
        endpoint, trajectory = _trajectory_metrics(reference, candidate)
        self.assertEqual(endpoint, 1.0)
        self.assertEqual(trajectory, 1.0)

    def test_artbench_evaluation_rejects_schedule_drift_and_black_images(self) -> None:
        from PIL import Image

        self.assertEqual(
            _expected_timesteps(1000, 50), list(range(999, -1, -20))
        )
        with self.assertRaises(ValueError):
            _expected_timesteps(1000, 60)
        self.assertTrue(_is_all_black(Image.new("RGB", (2, 2), (0, 0, 0))))
        self.assertFalse(_is_all_black(Image.new("RGB", (2, 2), (0, 0, 1))))

    def test_artbench_perceptual_models_use_immutable_revisions(self) -> None:
        for revision in (ARTBENCH_CLIP_MODEL_REVISION, ARTBENCH_DINO_REVISION):
            self.assertEqual(len(revision), 40)
            self.assertTrue(all(character in "0123456789abcdef" for character in revision))

    def test_artbench_merge_contains_only_requested_removal_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for query_id in (2, 5):
                model = root / f"q{query_id:02d}"
                model.mkdir()
                indices = model / "train_indices.pkl"
                with indices.open("wb") as handle:
                    pickle.dump([0, 1], handle)
                values = np.asarray([query_id / 10], dtype=np.float64)
                np.savez_compressed(
                    model / "true_f.npz", query_ids=np.asarray([query_id]),
                    endpoint_mse=values, trajectory_mse=values,
                    pixel_l2=values, clip_distance=values,
                    dino_distance=values, lpips=values,
                )
                jobs.append(EvaluationJob(
                    job_id=f"pint_raw_q{query_id:02d}", experiment="artbench2",
                    output_dir=model, train_indices=indices, query_id=query_id,
                    method="pint", parameter_source="raw",
                ))
            merged = merge_true_f(jobs, root / "metrics.npz")
            with np.load(merged) as payload:
                self.assertNotIn("simple_loss", payload.files)
                self.assertEqual(payload["endpoint_mse"].shape, (2,))
                self.assertEqual(str(payload["trajectory_reduction"]), "latent_state_mse")

    def test_lds_true_f_merge_preserves_model_and_query_axes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for subset_id in range(2):
                model = root / f"subset_{subset_id:04d}"
                model.mkdir()
                indices = model / "train_indices.npy"
                np.save(indices, np.asarray([subset_id, subset_id + 2]))
                np.savez_compressed(
                    model / "true_f.npz",
                    query_ids=np.asarray([4, 9]),
                    simple_loss=np.asarray([subset_id + 0.1, subset_id + 0.2]),
                    trajectory_deviation=np.asarray([subset_id + 1.1, subset_id + 1.2]),
                )
                jobs.append(EvaluationJob(
                    job_id=model.name, experiment="shape3d", output_dir=model,
                    train_indices=indices, subset_id=subset_id, subset_seed=10 + subset_id,
                ))
            merged = merge_true_f(jobs, root / "true_f.npz")
            with np.load(merged) as payload:
                self.assertEqual(payload["simple_loss"].shape, (2, 2))
                self.assertEqual(payload["trajectory_deviation"].shape, (2, 2))
                self.assertTrue(np.array_equal(payload["query_ids"], [4, 9]))
                self.assertTrue(np.array_equal(payload["subset_ids"], [0, 1]))
                self.assertEqual(str(payload["trajectory_reduction"]), "mean_snapshot_mse")

    def test_x3_merge_records_l2_squared_without_dimension_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "subset_0000"
            model.mkdir()
            indices = model / "train_indices.npy"
            np.save(indices, np.asarray([0, 1]))
            np.savez_compressed(
                model / "true_f.npz", query_ids=np.asarray([0]),
                simple_loss=np.asarray([0.1]),
                trajectory_deviation=np.asarray([0.2]),
            )
            job = EvaluationJob(
                job_id="subset_0000", experiment="x3", output_dir=model,
                train_indices=indices, subset_id=0, subset_seed=0,
            )
            merged = merge_true_f([job], root / "true_f.npz")
            with np.load(merged) as payload:
                self.assertEqual(
                    str(payload["trajectory_reduction"]), "mean_snapshot_l2_squared"
                )

    def test_true_f_completion_requires_every_requested_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "true_f.npz"
            np.savez_compressed(path, trajectory_deviation=np.asarray([1.0]))
            self.assertFalse(true_f_complete(path, "x3"))
            np.savez_compressed(
                path, simple_loss=np.asarray([1.0]),
                trajectory_deviation=np.asarray([2.0]),
            )
            self.assertTrue(true_f_complete(path, "x3"))

    def test_finite_spearman_uses_average_tie_ranks(self) -> None:
        self.assertAlmostEqual(
            finite_spearman(np.asarray([1, 1, 3]), np.asarray([1, 2, 3])),
            0.8660254037844387,
        )

    def test_lds_report_writes_query_csv_and_method_mean_std(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output_root = base / "outputs"
            evaluation = output_root / "x3" / "evaluation"
            true_f = evaluation / "lds" / "true_f.npz"
            true_f.parent.mkdir(parents=True)
            np.savez_compressed(
                true_f,
                query_ids=np.asarray([4, 9]),
                train_indices=np.asarray([[0, 1], [1, 2], [2, 3]]),
                simple_loss=np.asarray([[0, 2], [1, 1], [2, 0]], dtype=float),
                trajectory_deviation=np.asarray([[0, 2], [1, 1], [2, 0]], dtype=float),
            )
            scores = output_root / "x3" / "attribution" / "timesteps_100" / "scores.npz"
            scores.parent.mkdir(parents=True)
            values = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=float)
            np.savez_compressed(
                scores, query_ids=np.asarray([4, 9]), train_indices=np.arange(4),
                raw_scores=values, adamw_scores=values,
            )
            with patch.dict(os.environ, {"PINT_OUTPUT_ROOT": str(output_root)}, clear=False):
                report, query_csv = write_reports("x3", evaluation)
            with query_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual({row["prediction_sign"] for row in rows}, {"-1.0"})
            self.assertIn("pint | raw | simple_loss | 2 |", report.read_text())

    def test_topk_report_includes_all_six_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation"
            merged = root / "topk_removal" / "pint" / "true_f.npz"
            merged.parent.mkdir(parents=True)
            values = np.asarray([1.0, 3.0])
            np.savez_compressed(
                merged,
                query_ids=np.asarray([0, 1]), methods=np.asarray(["pint", "pint"]),
                parameter_sources=np.asarray(["raw", "raw"]),
                endpoint_mse=values, trajectory_mse=values, pixel_l2=values,
                clip_distance=values, dino_distance=values, lpips=values,
            )
            report, query_csv = write_reports("artbench2", root)
            with query_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            text = report.read_text()
            for metric in (
                "endpoint_mse", "trajectory_mse", "pixel_l2",
                "clip_distance", "dino_distance", "lpips",
            ):
                self.assertIn(f"pint | raw | {metric} | 2 | 2.00000000 | 1.00000000", text)


if __name__ == "__main__":
    unittest.main()
