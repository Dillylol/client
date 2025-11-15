"""Connectivity helpers for the FTC Jules bridge."""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Union

import requests

import codec

TelemetryList = Optional[List[Dict[str, Any]]]
CommandLike = Union[str, Dict[str, Any]]


class DataStream:
    """Maintain a live connection to the Jules HTTP bridge."""

    def __init__(
        self,
        base_url: str = "http://192.168.43.1:58080",
        *,
        stream_path: str = "/jules/stream",
        dump_path: str = "/jules/dump",
        command_path: str = "/jules/command",
        token: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.stream_path = stream_path
        self.dump_path = dump_path
        self.command_path = command_path
        self.token = token.strip() if token else None

        self.session = requests.Session()

        self.last_latency_ms: Optional[float] = None
        self.last_error: Optional[str] = None
        self.last_fetch_ts: Optional[float] = None
        self.last_message_ts: Optional[float] = None
        self.connected_since: Optional[float] = None
        self.messages_received: int = 0

        self._state: str = "DISCONNECTED"
        self._reconnect_attempts: int = 0

        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=512)
        self._seen_signatures: Deque[str] = deque(maxlen=512)
        self._signature_set: set[str] = set()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._update_urls()

    # ------------------------------------------------------------------
    # Connection management
    def connect(self, base_url: Optional[str] = None, *, token: Optional[str] = None) -> None:
        if base_url:
            self.base_url = base_url.rstrip("/")
        if token is not None:
            token = token.strip()
            self.token = token or None
        self._update_urls()
        self.last_error = None
        self._start_stream_thread()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None
        self._state = "DISCONNECTED"
        self._reconnect_attempts = 0

    def _update_urls(self) -> None:
        self.stream_url = f"{self.base_url}{self.stream_path}"
        self.dump_url = f"{self.base_url}{self.dump_path}"
        self.command_url = f"{self.base_url}{self.command_path}"

    def _headers(self) -> Dict[str, str]:
        if not self.token:
            return {}
        return {
            "X-Jules-Token": self.token,
            "Authorization": f"Bearer {self.token}",
        }

    def _start_stream_thread(self) -> None:
        self.disconnect()
        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=512)
        self._state = "CONNECTING"
        self.messages_received = 0
        self.connected_since = None
        self.last_message_ts = None
        self._seen_signatures.clear()
        self._signature_set.clear()
        self._thread = threading.Thread(target=self._stream_loop, name="JulesStream", daemon=True)
        self._thread.start()

    def _stream_loop(self) -> None:
        backoff = 1.0
        self._state = "CONNECTING"
        self._reconnect_attempts = 0

        while not self._stop_event.is_set():
            self._state = "CONNECTING"
            try:
                with self.session.get(
                    self.stream_url,
                    headers=self._headers(),
                    stream=True,
                    timeout=10,
                ) as response:
                    if response.status_code != 200:
                        self.last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        self._state = "ERROR"
                        self._sleep_with_stop(backoff)
                        backoff = min(backoff * 2, 5.0)
                        self._reconnect_attempts += 1
                        continue

                    self._state = "CONNECTED"
                    self.connected_since = time.time()
                    self.last_error = None
                    backoff = 1.0
                    self._reconnect_attempts = 0

                    event_lines: List[str] = []
                    for raw in response.iter_lines(decode_unicode=True):
                        if self._stop_event.is_set():
                            break
                        if raw is None:
                            continue
                        if not raw:
                            self._handle_sse_event(event_lines)
                            event_lines = []
                            continue
                        if raw.startswith(":"):
                            continue
                        if raw.startswith("data:"):
                            event_lines.append(raw[5:].lstrip())
                        else:
                            event_lines.append(raw)

                    if event_lines:
                        self._handle_sse_event(event_lines)
                        event_lines = []

                    self._state = "RECONNECTING"
                    self._reconnect_attempts += 1

            except requests.exceptions.RequestException as exc:
                self.last_error = str(exc)
                self._state = "ERROR"
                self._reconnect_attempts += 1

            if self._stop_event.is_set():
                break

            self._sleep_with_stop(backoff)
            backoff = min(backoff * 2, 5.0)

        self._state = "DISCONNECTED"

    def _sleep_with_stop(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if self._stop_event.is_set():
                break
            time.sleep(0.1)

    def _handle_sse_event(self, event_lines: List[str]) -> None:
        if not event_lines:
            return
        try:
            text = "\n".join(event_lines)
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    message = self._normalize_message(item)
                    self._enqueue_message(message)
                return
            message = self._normalize_message(data)
            self._enqueue_message(message)
        except json.JSONDecodeError:
            # Wrap malformed payloads so the UI can show them.
            message = {
                "type": "event",
                "payload": {"raw": "\n".join(event_lines)},
                "ts": int(time.time() * 1000),
            }
            self._enqueue_message(message)

    def _enqueue_message(self, message: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                pass

        self.messages_received += 1
        self.last_message_ts = time.time()
        self.last_error = None

    def _normalize_message(self, message: Any) -> Dict[str, Any]:
        if isinstance(message, dict):
            normalized = dict(message)
            if "ts" not in normalized:
                normalized["ts"] = int(time.time() * 1000)
            if "type" not in normalized:
                normalized["type"] = codec.TELEMETRY_TYPE
            return normalized
        return {
            "type": "event",
            "payload": {"value": message},
            "ts": int(time.time() * 1000),
        }

    # ------------------------------------------------------------------
    # Telemetry and dump fallback
    def get_data(self) -> TelemetryList:
        drained: List[Dict[str, Any]] = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if drained:
            self.last_fetch_ts = time.time()
            return drained

        if self._state not in {"CONNECTED", "CONNECTING"}:
            fallback = self._fetch_dump()
            if fallback:
                return fallback

        return None

    def _fetch_dump(self) -> TelemetryList:
        try:
            start = time.perf_counter()
            resp = self.session.get(self.dump_url, headers=self._headers(), timeout=2)
            resp.raise_for_status()
            data = resp.json()
            envelopes: List[Dict[str, Any]] = []
            for item in data if isinstance(data, list) else []:
                payload = dict(item)
                envelopes.append(
                    {
                        "v": payload.get("v", codec.ENVELOPE_VERSION),
                        "type": payload.get("type", codec.TELEMETRY_TYPE),
                        "ts": payload.get("ts") or payload.get("t") or int(time.time() * 1000),
                        "payload": payload,
                    }
                )

            self.last_latency_ms = (time.perf_counter() - start) * 1000
            self.last_error = None
            self.last_fetch_ts = time.time()
            if envelopes:
                return envelopes
        except requests.exceptions.RequestException as exc:
            self.last_error = str(exc)
        except json.JSONDecodeError as exc:
            self.last_error = str(exc)

        self.last_latency_ms = None
        return None

    def is_new_payload(self, payload: Dict[str, Any]) -> bool:
        signature = self._signature_for(payload)
        return self._remember_signature(signature)

    def _signature_for(self, payload: Dict[str, Any]) -> str:
        if "id" in payload:
            return f"id:{payload['id']}"
        inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
        return json.dumps({"type": payload.get("type"), "payload": inner}, sort_keys=True, default=str)

    def _remember_signature(self, signature: str) -> bool:
        if signature in self._signature_set:
            return False
        if len(self._seen_signatures) == self._seen_signatures.maxlen:
            oldest = self._seen_signatures.popleft()
            self._signature_set.discard(oldest)
        self._seen_signatures.append(signature)
        self._signature_set.add(signature)
        return True

    # ------------------------------------------------------------------
    # Commands
    def send_command(self, command: CommandLike) -> Dict[str, Any]:
        if isinstance(command, dict):
            envelope = codec.ensure_command_envelope(command)
            command_str = json.dumps(envelope)
        else:
            envelope = None
            command_str = command

        try:
            start = time.perf_counter()
            response = self.session.post(
                self.command_url,
                headers=self._headers(),
                json={"command": command_str},
                timeout=3,
            )
            round_trip = (time.perf_counter() - start) * 1000
            self.last_latency_ms = round_trip

            if response.ok:
                try:
                    data = response.json()
                except ValueError:
                    data = {"ok": True, "text": response.text}
                if envelope is not None:
                    data.setdefault("request", envelope)
                self.last_error = None
                return data
            return {"ok": False, "status": response.status_code, "text": response.text}
        except requests.exceptions.RequestException as exc:
            self.last_error = str(exc)
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Introspection
    def connection_snapshot(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "state": self._state,
            "last_latency_ms": self.last_latency_ms,
            "last_error": self.last_error,
            "last_fetch_ts": self.last_fetch_ts,
            "last_message_ts": self.last_message_ts,
            "connected_since": self.connected_since,
            "messages_received": self.messages_received,
            "token": self.token,
            "reconnect_attempts": self._reconnect_attempts,
        }


__all__ = ["DataStream"]
