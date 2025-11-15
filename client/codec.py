"""Utilities for constructing and parsing JULES bridge envelopes."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterable, Optional

COMMAND_TYPE = "command"
TELEMETRY_TYPE = "telemetry"
ANNOUNCE_TYPE = "announce"
ACK_TYPE = "ack"
ERROR_TYPE = "error"
PING_TYPE = "ping"
PONG_TYPE = "pong"

ENVELOPE_VERSION = 1


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def new_message_id(prefix: str = "c") -> str:
    """Generate a reasonably unique message identifier."""
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}-{_timestamp_ms()}-{suffix}"


def build_envelope(message_type: str, payload: Dict[str, Any], *, message_id: Optional[str] = None,
                   version: int = ENVELOPE_VERSION, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "v": version,
        "type": message_type,
        "ts": timestamp_ms if timestamp_ms is not None else _timestamp_ms(),
        "payload": payload,
    }
    if message_id:
        envelope["id"] = message_id
    return envelope


def build_command(name: str, args: Optional[Dict[str, Any]] = None, *,
                  message_id: Optional[str] = None, timestamp_ms: Optional[int] = None,
                  extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"name": name}
    if args:
        payload["args"] = args
    if extras:
        payload.update(extras)
    return build_envelope(COMMAND_TYPE, payload, message_id=message_id or new_message_id(),
                          timestamp_ms=timestamp_ms)


def ensure_command_envelope(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either a raw payload (e.g., {"name": "drive"}) or a full envelope.
    Returns a command envelope, generating ids/timestamps as needed.
    """
    if not data:
        raise ValueError("Command payload is empty")

    message_type = data.get("type")
    if message_type == COMMAND_TYPE:
        envelope = dict(data)
        if "v" not in envelope:
            envelope["v"] = ENVELOPE_VERSION
        if "ts" not in envelope:
            envelope["ts"] = _timestamp_ms()
        if "id" not in envelope:
            envelope["id"] = new_message_id()
        return envelope

    # Treat the dict as a raw payload
    name = data.get("name")
    if not name:
        raise ValueError("Command payload must include a 'name' field")
    args = data.get("args")
    extras = {k: v for k, v in data.items() if k not in {"name", "args"}}
    return build_command(name, args=args, extras=extras)


def pretty_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def summarize_payload(payload: Dict[str, Any], *, keys: Iterable[str] | None = None, limit: int = 3) -> str:
    """Create a short human-readable summary of a payload."""
    keys_to_use = list(keys) if keys is not None else ["name", "args", "ip", "battery"]
    pairs = []
    for key in keys_to_use:
        if key in payload and payload[key] not in (None, ""):
            value = payload[key]
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)
            pairs.append(f"{key}={value}")
        if len(pairs) >= limit:
            break
    if not pairs:
        return json.dumps(payload, ensure_ascii=False)
    return ", ".join(pairs)
