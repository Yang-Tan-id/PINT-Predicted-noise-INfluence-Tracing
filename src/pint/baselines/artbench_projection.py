from __future__ import annotations

import hashlib
from dataclasses import dataclass


def artbench_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


@dataclass(frozen=True)
class ArtBenchProjector:
    """The global-coordinate CountSketch used by the ArtBench-2 originals."""

    dimension: int
    seed: int
    chunk_size: int = 1_000_000

    def project(self, tensors, *, full_l2_normalize: bool = False):
        import torch

        norm = torch.sqrt(sum(t.detach().float().square().sum() for t in tensors))
        scale = norm.clamp_min(1e-12).reciprocal() if full_l2_normalize else 1.0
        output = torch.zeros(self.dimension, dtype=torch.float32, device=tensors[0].device)
        bucket_a = 6364136223846793005
        bucket_b = (1442695040888963407 + self.seed * 2_654_435_761) & 0x7FFFFFFFFFFFFFFF
        sign_a = 2862933555777941757
        sign_b = (3037000493 + self.seed * 1_000_003) & 0x7FFFFFFFFFFFFFFF
        offset = 0
        for tensor in tensors:
            flat = tensor.detach().float().reshape(-1)
            for start in range(0, flat.numel(), self.chunk_size):
                end = min(start + self.chunk_size, flat.numel())
                coordinates = torch.arange(offset + start, offset + end, dtype=torch.int64, device=flat.device)
                buckets = torch.remainder(coordinates * bucket_a + bucket_b, self.dimension)
                signs = torch.bitwise_and(coordinates * sign_a + sign_b, 1).float().mul_(2).sub_(1)
                output.scatter_add_(0, buckets, flat[start:end] * signs * scale)
            offset += flat.numel()
        return output

    def project_batch(self, tensors, *, full_l2_normalize: bool = False):
        import torch

        batch = tensors[0].shape[0]
        norms = torch.sqrt(
            sum(t.detach().float().reshape(batch, -1).square().sum(dim=1) for t in tensors)
        )
        scales = norms.clamp_min(1e-12).reciprocal() if full_l2_normalize else torch.ones_like(norms)
        output = torch.zeros((batch, self.dimension), dtype=torch.float32, device=tensors[0].device)
        bucket_a = 6364136223846793005
        bucket_b = (1442695040888963407 + self.seed * 2_654_435_761) & 0x7FFFFFFFFFFFFFFF
        sign_a = 2862933555777941757
        sign_b = (3037000493 + self.seed * 1_000_003) & 0x7FFFFFFFFFFFFFFF
        offset = 0
        for tensor in tensors:
            flat = tensor.detach().float().reshape(batch, -1)
            for start in range(0, flat.shape[1], self.chunk_size):
                end = min(start + self.chunk_size, flat.shape[1])
                coordinates = torch.arange(offset + start, offset + end, dtype=torch.int64, device=flat.device)
                buckets = torch.remainder(coordinates * bucket_a + bucket_b, self.dimension)
                signs = torch.bitwise_and(coordinates * sign_a + sign_b, 1).float().mul_(2).sub_(1)
                values = flat[:, start:end] * signs.unsqueeze(0) * scales.unsqueeze(1)
                output.scatter_add_(1, buckets.unsqueeze(0).expand(batch, -1), values)
            offset += flat.shape[1]
        return output
