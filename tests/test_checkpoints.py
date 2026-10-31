from __future__ import annotations

import unittest

from pint.training.checkpoints import evenly_spaced_epochs


class CheckpointPlacementTests(unittest.TestCase):
    def test_one_checkpoint_is_the_final_epoch(self) -> None:
        self.assertEqual(evenly_spaced_epochs(200, 1), (200,))

    def test_non_divisible_count_is_even_and_includes_end(self) -> None:
        epochs = evenly_spaced_epochs(200, 7)
        self.assertEqual(epochs, (29, 58, 86, 115, 143, 172, 200))
        self.assertEqual(len(epochs), 7)
        self.assertEqual(epochs[-1], 200)

    def test_every_epoch_can_be_saved(self) -> None:
        self.assertEqual(evenly_spaced_epochs(200, 200), tuple(range(1, 201)))

    def test_count_cannot_exceed_epochs(self) -> None:
        with self.assertRaises(ValueError):
            evenly_spaced_epochs(100, 101)


if __name__ == "__main__":
    unittest.main()
