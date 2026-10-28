from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np

from pint.config import repo_root
from pint.datasets.scope_manifest import build_scope_indices, build_scope_mask


class DatasetManifestTests(unittest.TestCase):
    def test_shape3d_paper_scope(self) -> None:
        path = repo_root() / "dataset_manifests/shape3d/attribution_5k_indices.npy"
        indices = np.load(path)
        np.testing.assert_array_equal(indices, build_scope_indices(20_000, 5_000, 42))
        self.assertEqual(len(np.unique(indices)), 5_000)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "ac1bfae0332205885d6d9496d0b3bbc6ef5bd9f5fc6f7106e9e87cfe60eb2c16",
        )

    def test_celeba_paper_scope_and_mask(self) -> None:
        root = repo_root() / "dataset_manifests/celeba"
        indices_path = root / "attribution_5k_indices.npy"
        mask_path = root / "attribution_5k_mask.npy"
        indices = np.load(indices_path)
        mask = np.load(mask_path)
        np.testing.assert_array_equal(indices, build_scope_indices(10_000, 5_000, 42))
        np.testing.assert_array_equal(mask, build_scope_mask(10_000, indices))
        self.assertEqual(
            hashlib.sha256(indices_path.read_bytes()).hexdigest(),
            "8e6b914190f6cb2ca3888a21b63244d3b03585967bffd74eff489dd974a01434",
        )
        self.assertEqual(
            hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "4ce1873c32da75b73a73ccd5ca2f81ea0565aee549d9f459ecc33e846a80e408",
        )


if __name__ == "__main__":
    unittest.main()
