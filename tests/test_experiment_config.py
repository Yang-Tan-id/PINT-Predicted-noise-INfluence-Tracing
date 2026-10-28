from __future__ import annotations

import unittest

from pint.config import EXPERIMENT_NAMES, load_experiment


class ExperimentConfigTests(unittest.TestCase):
    def test_all_experiment_configs_validate(self) -> None:
        configs = {name: load_experiment(name) for name in EXPERIMENT_NAMES}
        self.assertEqual(configs["x3"]["queries"]["count"], 100)
        self.assertEqual(configs["shape3d"]["queries"]["count"], 100)
        self.assertEqual(configs["celeba"]["queries"]["count"], 100)
        self.assertEqual(configs["artbench2"]["queries"]["count"], 60)
        self.assertEqual(configs["shape3d"]["lds"]["num_subsets"], 192)
        self.assertEqual(configs["celeba"]["lds"]["num_subsets"], 192)


if __name__ == "__main__":
    unittest.main()
