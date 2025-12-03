"""Maintain live telemetry state, flattened map, and time-series buffers."""
from __future__ import annotations

import copy
import csv
import io
import time
from typing import Any, Dict, Iterable, List

from .flatten import flatten_tree
from .patcher import apply_patch
from .ringbuf import RingBuffer


class StateStore:
    def __init__(
        self,
        *,
        series_capacity: int = 10_000,
        throttle_ms: int = 100,
        global_cap: int = 2_000_000,
    ) -> None:
        self._live_tree: Dict[str, Any] = {}
        self._flat: Dict[str, Any] = {}
        self._series: Dict[str, RingBuffer] = {}
        self._last_series_ts: Dict[str, int] = {}
        self._last_values: Dict[str, Any] = {}
        self._series_capacity = series_capacity
        self._throttle_ms = throttle_ms
        self._global_cap = global_cap
        self._total_samples = 0

    # ------------------------------------------------------------------
    def handle_snapshot(self, data: Dict[str, Any], ts_ms: int) -> List[str]:
        self._live_tree = copy.deepcopy(data)
        return self._sync_flat(ts_ms)

    def handle_diff(self, patch: Dict[str, Any], ts_ms: int) -> List[str]:
        apply_patch(self._live_tree, patch)
        return self._sync_flat(ts_ms)

    def update_branch(self, path: str, value: Any, ts_ms: int) -> List[str]:
        apply_patch(self._live_tree, {path: value})
        return self._sync_flat(ts_ms)

    def get_live_tree(self) -> Dict[str, Any]:
        return copy.deepcopy(self._live_tree)

    def get_flat(self) -> Dict[str, Any]:
        return dict(self._flat)

    def get_series(self, key: str) -> List[tuple[int, Any]]:
        buf = self._series.get(key)
        if not buf:
            return []
        return buf.values()

    def export_csv(self, keys: Iterable[str], window_ms: int) -> bytes:
        now_ms = int(time.time() * 1000)
        min_ts = now_ms - window_ms if window_ms > 0 else 0
        buckets: Dict[int, Dict[str, Any]] = {}
        for key in keys:
            series = self._series.get(key)
            if not series:
                continue
            for ts, value in series:
                if ts < min_ts:
                    continue
                bucket = buckets.setdefault(ts, {})
                bucket[key] = value
        rows = sorted(buckets.items())
        output = io.StringIO()
        writer = csv.writer(output)
        header = ["ts_ms", *keys]
        writer.writerow(header)
        for ts, values in rows:
            row = [ts]
            for key in keys:
                row.append(self._format_csv_value(values.get(key)))
            writer.writerow(row)
        return output.getvalue().encode("utf-8")

    # ------------------------------------------------------------------
    def _sync_flat(self, ts_ms: int) -> List[str]:
        new_flat = flatten_tree(self._live_tree)
        changed: List[str] = []
        for key, value in new_flat.items():
            if self._flat.get(key) != value:
                changed.append(key)
                self._record_series(key, value, ts_ms)
        for key in list(self._flat.keys()):
            if key not in new_flat:
                changed.append(key)
                self._record_series(key, None, ts_ms)
        self._flat = new_flat
        return changed

    def _record_series(self, key: str, value: Any, ts_ms: int) -> None:
        last_ts = self._last_series_ts.get(key, 0)
        last_val = self._last_values.get(key)
        if last_val == value and ts_ms - last_ts < self._throttle_ms:
            return
        buf = self._series.setdefault(key, RingBuffer(self._series_capacity))
        prev_len = len(buf)
        buf.append(ts_ms, value)
        new_len = len(buf)
        added = max(0, new_len - prev_len)
        if added:
            self._total_samples += added
            self._enforce_global_cap()
        self._last_series_ts[key] = ts_ms
        self._last_values[key] = value

    def _enforce_global_cap(self) -> None:
        if self._total_samples <= self._global_cap:
            return
        while self._total_samples > self._global_cap:
            for buf in self._series.values():
                if not buf:
                    continue
                buf.pop_left()
                self._total_samples -= 1
                if self._total_samples <= self._global_cap:
                    break
            else:
                break

    def _format_csv_value(self, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

