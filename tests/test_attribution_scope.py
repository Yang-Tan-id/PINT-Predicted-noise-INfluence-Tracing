from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pint.attribution_scope import AttributionScope, resolve_indices


class AttributionScopeTests(unittest.TestCase):
    def test_all_scope_returns_full_dataset(self) -> None:
        scope = AttributionScope(name="all", selection="all")
        np.testing.assert_array_equal(resolve_indices(scope, 4), np.arange(4))

    def test_explicit_scope_preserves_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.npy"
            np.save(path, np.asarray([4, 1, 3], dtype=np.int64))
            scope = AttributionScope(
                name="paper",
                selection="explicit_indices",
                expected_size=3,
                indices_path=path,
            )
            np.testing.assert_array_equal(resolve_indices(scope, 5), [4, 1, 3])

    def test_explicit_scope_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.npy"
            np.save(path, np.asarray([1, 1], dtype=np.int64))
            scope = AttributionScope(
                name="bad", selection="explicit_indices", indices_path=path
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                resolve_indices(scope, 5)


if __name__ == "__main__":
    unittest.main()
