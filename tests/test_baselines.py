from __future__ import annotations

import unittest

import numpy as np

from pint.baselines.attribution_cli import attribution_plan
from pint.baselines.artbench_projection import artbench_projection_seed
from pint.baselines.common import baseline_spec, orient_adamw_scores, run_tag
from pint.baselines.query_cli import query_plan


class BaselineTests(unittest.TestCase):
    def test_baseline_adamw_scores_are_negated_at_write_orientation(self):
        values = np.asarray([[1.0, -2.0]])
        for method in ("gas", "tracincp", "retrac"):
            self.assertTrue(np.array_equal(
                orient_adamw_scores(method, values), np.asarray([[-1.0, 2.0]])
            ))

    def test_dataset_timestamp_and_mc_contract(self):
        for method in ("gas", "tracincp", "retrac"):
            self.assertEqual(
                (baseline_spec(method, "x3").timestep_count, baseline_spec(method, "x3").mc_samples),
                (100, 10),
            )
            self.assertEqual(
                (baseline_spec(method, "shape3d").timestep_count, baseline_spec(method, "shape3d").mc_samples),
                (100, 1),
            )
            for experiment in ("celeba", "artbench2"):
                self.assertEqual(
                    (baseline_spec(method, experiment).timestep_count, baseline_spec(method, experiment).mc_samples),
                    (50, 1),
                )

    def test_method_normalization_contract(self):
        self.assertEqual(baseline_spec("gas", "x3").score_normalization, "projected_both_l2")
        self.assertEqual(baseline_spec("tracincp", "x3").score_normalization, "raw")
        self.assertEqual(
            baseline_spec("retrac", "x3").score_normalization,
            "full_l2_before_projection",
        )

    def test_query_filename_tag_carries_method_sampling_and_projection(self):
        spec = baseline_spec("retrac", "celeba")
        self.assertEqual(run_tag(spec, 4096), "retrac_t50_mc1_projected4096")
        self.assertEqual(run_tag(spec, 2048), "retrac_t50_mc1_projected2048")

    def test_query_and_attribution_plans_share_the_tag(self):
        query = query_plan("shape3d", "gas", projection_dim=2048)
        attribution = attribution_plan("shape3d", "gas", projection_dim=2048)
        self.assertEqual(query["tag"], attribution["tag"])
        self.assertIn("/methods/gas/queries", query["output"])
        self.assertEqual(attribution["parameter_sources"], ["raw_gradient", "adamw_full"])

    def test_artbench_projection_default_and_checkpoint_seed_contract(self):
        query = query_plan("artbench2", "gas")
        attribution = attribution_plan("artbench2", "gas")
        self.assertEqual(query["projection_dim"], 32768)
        self.assertIn("projected32768", query["tag"])
        self.assertEqual(query["tag"], attribution["tag"])
        self.assertEqual(artbench_projection_seed(2), artbench_projection_seed(2))
        self.assertNotEqual(artbench_projection_seed(2), artbench_projection_seed(3))


if __name__ == "__main__":
    unittest.main()
