"""Apply dotted-path patches to telemetry trees."""
from __future__ import annotations

from typing import Any, Dict, MutableMapping, MutableSequence


def apply_patch(tree: MutableMapping[str, Any], patch: Dict[str, Any]) -> None:
    for path, value in patch.items():
        parts = path.split(".") if path else []
        node: Any = tree
        for idx, part in enumerate(parts):
            is_last = idx == len(parts) - 1
            if isinstance(node, MutableSequence):
                index = _ensure_index(node, part)
                if is_last:
                    node[index] = value
                else:
                    child = node[index]
                    if not isinstance(child, (MutableMapping, MutableSequence)):
                        child = {}
                        node[index] = child
                    node = child
            else:
                if not isinstance(node, MutableMapping):
                    return
                if is_last:
                    node[part] = value
                else:
                    if part not in node or not isinstance(node[part], (MutableMapping, MutableSequence)):
                        node[part] = {}
                    node = node[part]


def _ensure_index(seq: MutableSequence[Any], token: str) -> int:
    try:
        index = int(token)
    except ValueError:
        raise KeyError(f"Non-integer index '{token}' in list path") from None
    while len(seq) <= index:
        seq.append({})
    return index

