"""Simple ring buffer for time-series telemetry."""
from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Tuple


class RingBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self._capacity = capacity
        self._data: Deque[Tuple[int, object]] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, ts_ms: int, value: object) -> None:
        self._data.append((ts_ms, value))

    def pop_left(self) -> None:
        if self._data:
            self._data.popleft()

    def values(self) -> List[Tuple[int, object]]:
        return list(self._data)

    def since(self, min_ts: int) -> List[Tuple[int, object]]:
        return [(ts, val) for ts, val in self._data if ts >= min_ts]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __iter__(self) -> Iterable[Tuple[int, object]]:  # pragma: no cover - trivial
        return iter(self._data)

