from __future__ import annotations

import math
from typing import Iterable

from pint.query_gradients.common import stable_torch_seed


class CountSketchProjector:
    """A reusable CountSketch map for one checkpoint's parameter ordering."""

    def __init__(self, parameters: Iterable, dimension: int, *seed_parts: object):
        import torch

        self.dimension = dimension
        self.specs = []
        for index, parameter in enumerate(parameters):
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(
                stable_torch_seed(
                    *seed_parts, "tensor", index, parameter.numel(), dimension
                )
            )
            buckets = torch.randint(
                dimension,
                (parameter.numel(),),
                generator=generator,
                device=parameter.device,
            )
            signs = torch.randint(
                2,
                (parameter.numel(),),
                generator=generator,
                device=parameter.device,
                dtype=torch.int8,
            ).float().mul_(2).sub_(1)
            self.specs.append((buckets, signs))
        self.scale = 1.0 / math.sqrt(float(dimension))

    def __call__(self, gradients):
        import torch

        if len(gradients) != len(self.specs):
            raise ValueError("gradient and CountSketch parameter counts differ")
        output = torch.zeros(
            self.dimension, device=gradients[0].device, dtype=torch.float32
        )
        for gradient, (buckets, signs) in zip(gradients, self.specs):
            output.index_add_(0, buckets, signs * gradient.reshape(-1).float())
        return output * self.scale

    def project_batch(self, gradients):
        """Project a gradient tree with a shared leading example axis."""
        import torch

        if len(gradients) != len(self.specs):
            raise ValueError("gradient and CountSketch parameter counts differ")
        batch_size = gradients[0].shape[0]
        if any(gradient.shape[0] != batch_size for gradient in gradients):
            raise ValueError("batched gradients must share their leading axis")
        output = torch.zeros(
            (batch_size, self.dimension),
            device=gradients[0].device,
            dtype=torch.float32,
        )
        for gradient, (buckets, signs) in zip(gradients, self.specs):
            flat = gradient.reshape(batch_size, -1).float()
            output.scatter_add_(
                1,
                buckets.unsqueeze(0).expand(batch_size, -1),
                flat * signs.unsqueeze(0),
            )
        return output * self.scale
