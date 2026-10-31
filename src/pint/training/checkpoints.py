"""Checkpoint placement shared by all training backends."""

import math


def evenly_spaced_epochs(num_epochs: int, num_checkpoints: int) -> tuple[int, ...]:
    """Return evenly distributed 1-based epochs, always including the end."""

    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    if not 1 <= num_checkpoints <= num_epochs:
        raise ValueError("num_checkpoints must be between 1 and num_epochs")
    epochs = tuple(
        math.ceil(index * num_epochs / num_checkpoints)
        for index in range(1, num_checkpoints + 1)
    )
    if len(set(epochs)) != num_checkpoints or epochs[-1] != num_epochs:
        raise AssertionError("checkpoint epoch construction failed")
    return epochs
