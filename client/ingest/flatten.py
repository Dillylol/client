"""Utilities for flattening nested telemetry structures."""
from __future__ import annotations

from typing import Any, Dict


def flatten_tree(data: Any, *, sep: str = ".") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    def _walk(node: Any, prefix: str) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                next_prefix = key if not prefix else f"{prefix}{sep}{key}"
                _walk(value, next_prefix)
            return
        if isinstance(node, list):
            for idx, value in enumerate(node):
                next_prefix = f"{prefix}{sep}{idx}" if prefix else str(idx)
                _walk(value, next_prefix)
            return
        if isinstance(node, (str, int, float, bool)):
            flat[prefix] = node
        else:
            try:
                flat[prefix] = repr(node)
            except Exception:  # noqa: BLE001
                pass

    _walk(data, "")
    return flat

