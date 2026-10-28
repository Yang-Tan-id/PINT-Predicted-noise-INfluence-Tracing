"""Dataset preparation entrypoints."""

from .scope_manifest import build_scope_indices, build_scope_mask

__all__ = ["build_scope_indices", "build_scope_mask"]
