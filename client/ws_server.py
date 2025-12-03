"""WebSocket server orchestrating robot ↔ client shot planning."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from .config import ClientConfig
from .evaluator import ShotEvaluator
from .rpm_model import RpmModel


@dataclass
class SessionState:
    """Track lifecycle information for the connected robot."""

    session_id: Optional[str] = None
    seq_out: int = 0
    last_obs_ts_ms: Optional[int] = None
    clock_skew_ms: Optional[float] = None
    last_seq_by_type: Dict[str, int] = None
    token_valid: bool = True

    def __post_init__(self) -> None:
        if self.last_seq_by_type is None:
            self.last_seq_by_type = {}

    def next_seq(self) -> int:
        self.seq_out += 1
        return self.seq_out

    def register_seq(self, msg_type: str, seq: Optional[int]) -> bool:
        try:
            seq_int = int(seq)
        except (TypeError, ValueError):
            return False
        last = self.last_seq_by_type.get(msg_type)
        if last is not None and seq_int <= last:
            return False
        self.last_seq_by_type[msg_type] = seq_int
        return True

    def mark_obs(self, ts_ms: Optional[int], now_ms: int) -> None:
        try:
            robot_ts = int(ts_ms) if ts_ms is not None else None
        except (TypeError, ValueError):
            robot_ts = None
        if robot_ts is not None:
            self.last_obs_ts_ms = now_ms
            if self.clock_skew_ms is None:
                self.clock_skew_ms = float(robot_ts - now_ms)
            else:
                self.clock_skew_ms = 0.9 * self.clock_skew_ms + 0.1 * (robot_ts - now_ms)
        else:
            self.last_obs_ts_ms = now_ms

    def offline(self, now_ms: int) -> bool:
        if self.last_obs_ts_ms is None:
            return False
        return (now_ms - self.last_obs_ts_ms) > 1500


class ShotServer:
    """Server managing shot planning, logging, and UI fan-out."""

    def __init__(self, config_path: Path) -> None:
        self.config = ClientConfig.load(config_path)
        self.model = RpmModel.load(self.config.paths.model_path)
        self.evaluator = ShotEvaluator(self.config, self.model)
        self.session = SessionState()
        self.event_buffer: Deque[Dict[str, Any]] = deque(maxlen=50)

        self.robot_ws: Optional[WebSocketServerProtocol] = None
        self.gui_clients: Set[WebSocketServerProtocol] = set()
        self._pinger_task: Optional[asyncio.Task[None]] = None
        self._health_task: Optional[asyncio.Task[None]] = None
        self._logger = logging.getLogger(__name__)

    # -----------------------------------------------------------------
    async def robot_handler(self, ws: WebSocketServerProtocol) -> None:
        self.robot_ws = ws
        self.session = SessionState()
        if self._pinger_task:
            self._pinger_task.cancel()
        if self._health_task:
            self._health_task.cancel()
        self._pinger_task = asyncio.create_task(self._pinger())
        self._health_task = asyncio.create_task(self._health_monitor())
        try:
            async for message in ws:
                for line in message.splitlines():
                    await self._handle_robot_message(ws, line)
        finally:
            if self._pinger_task:
                self._pinger_task.cancel()
            if self._health_task:
                self._health_task.cancel()
            self.robot_ws = None

    def _record_event(self, event: Dict[str, Any]) -> None:
        self.event_buffer.append(event)


    async def _handle_robot_message(self, ws: WebSocketServerProtocol, line: str) -> None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        msg_type = obj.get("type")
        seq = obj.get("seq")
        now_ms = _now_ms()

        if msg_type == "hello":
            await self._handle_hello(ws, obj, now_ms)
            return
        if self.session.session_id is None:
            return
        if obj.get("session_id") != self.session.session_id:
            return
        if msg_type != "request_shot_plan" and not self.session.register_seq(msg_type, seq):
            return

        if msg_type == "request_shot_plan":
            await self._handle_request(ws, obj, now_ms)
        elif msg_type == "obs":
            self.session.mark_obs(obj.get("ts_ms"), now_ms)
            self._record_event({"type": "obs", "payload": obj})
            await self._broadcast_ui({"type": "obs", "payload": obj})
        elif msg_type == "token_shot_fired":
            self.session.register_seq(msg_type, seq)
            self.evaluator.record_shot_fired(obj)
            self._record_event({"type": "token_shot_fired", "payload": obj})
        elif msg_type == "shot_result":
            update = self.evaluator.apply_shot_result(obj, now_ms)
            self._record_event({"type": "shot_result", "payload": obj})
            await self._broadcast_ui({"type": "shot_result", "payload": obj})
            if update:
                await self._send_model_update(ws, update, now_ms)
        elif msg_type == "pong":
            await self._handle_pong(obj, now_ms)
        elif msg_type == "ack":
            # Preserve prior behaviour by fanning out a UI notification
            self._record_event({"type": "ack", "payload": obj})
            await self._broadcast_ui({"type": "ack", "payload": obj})
        else:
            self._record_event({"type": msg_type or "unknown", "payload": obj})

    async def _handle_hello(self, ws: WebSocketServerProtocol, obj: Dict[str, Any], now_ms: int) -> None:
        session_id = obj.get("session_id")
        seq = obj.get("seq")
        token = obj.get("token", "")
        if not isinstance(session_id, str):
            return
        if self.config.require_token and token != self.config.token:
            await ws.send(json.dumps({"type": "error", "code": "auth", "message": "invalid token"}))
            await ws.close(code=4403, reason="invalid token")
            return
        self.session = SessionState(session_id=session_id)
        self.session.register_seq("hello", seq)
        self.evaluator.start_session(session_id)
        self._record_event({"type": "hello", "payload": obj})
        ack = {
            "type": "hello_ack",
            "session_id": session_id,
            "seq": self.session.next_seq(),
            "policy": self.config.policy.name,
            "policy_flags": {
                "send_abs_rpm": self.config.policy.send_abs_rpm,
                "bias_step_rpm": self.config.policy.bias_step_rpm,
                "bias_cap_rpm": self.config.policy.bias_cap_rpm,
                "start_with_anchor": self.config.policy.start_with_anchor,
                "anchor_samples_target": self.config.policy.anchor_samples_target,
            },
            "model_version": self.model.model_version,
            "ts_ms": now_ms,
        }
        await ws.send(json.dumps(ack))
        await self._broadcast_ui({"type": "hello", "payload": ack})

    async def _handle_request(self, ws: WebSocketServerProtocol, obj: Dict[str, Any], now_ms: int) -> None:
        if self.session.offline(now_ms):
            self._logger.warning("Robot marked offline; issuing safe noop command")
            command = self.evaluator.safe_noop(now_ms, reason="offline")
        else:
            command = self.evaluator.plan_shot(obj, now_ms)
        command_msg = dict(command)
        command_msg.update(
            {
                "session_id": self.session.session_id,
                "seq": self.session.next_seq(),
                "ts_ms": now_ms,
                "valid_ms": command.get("valid_ms", self.config.cmd_valid_ms),
            }
        )
        await ws.send(json.dumps(command_msg))
        self._record_event({"type": "cmd", "payload": command_msg})
        await self._broadcast_ui({"type": "cmd", "payload": command_msg})
        # track seq for request after sending to avoid blocking duplicates
        self.session.register_seq("request_shot_plan", obj.get("seq"))

    async def _send_model_update(self, ws: WebSocketServerProtocol, update: Dict[str, Any], now_ms: int) -> None:
        payload = dict(update)
        payload.update(
            {
                "type": "rpm_model_update",
                "session_id": self.session.session_id,
                "seq": self.session.next_seq(),
                "update_id": str(uuid.uuid4()),
                "valid_ms": max(self.config.cmd_valid_ms, 1000),
                "ts_ms": now_ms,
            }
        )
        await ws.send(json.dumps(payload))
        self._record_event({"type": "rpm_model_update", "payload": payload})
        await self._broadcast_ui({"type": "rpm_model_update", "payload": payload})

    async def _handle_pong(self, obj: Dict[str, Any], now_ms: int) -> None:
        robot_ts = obj.get("ts_ms")
        if robot_ts is not None:
            self.session.mark_obs(robot_ts, now_ms)
        self._record_event({"type": "pong", "payload": obj})
        await self._broadcast_ui({"type": "pong", "payload": obj})

    async def _pinger(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            if self.robot_ws is None or not self.robot_ws.open or self.session.session_id is None:
                continue
            ping_msg = {
                "type": "ping",
                "session_id": self.session.session_id,
                "seq": self.session.next_seq(),
                "ts_ms": _now_ms(),
                "id": str(uuid.uuid4()),
            }
            try:
                await self.robot_ws.send(json.dumps(ping_msg))
                self._record_event({"type": "ping", "payload": ping_msg})
            except Exception:
                break

    async def _health_monitor(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            now_ms = _now_ms()
            offline = self.session.offline(now_ms)
            await self._broadcast_ui(
                {
                    "type": "health",
                    "session_id": self.session.session_id,
                    "offline": offline,
                    "clock_skew_ms": self.session.clock_skew_ms,
                    "ts_ms": now_ms,
                }
            )

    async def gui_handler(self, ws: WebSocketServerProtocol) -> None:
        self.gui_clients.add(ws)
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "state_snapshot",
                        "session_id": self.session.session_id,
                        "event_buffer": list(self.event_buffer),
                        "clock_skew_ms": self.session.clock_skew_ms,
                    }
                )
            )
            async for _ in ws:
                pass
        finally:
            self.gui_clients.discard(ws)

    async def _broadcast_ui(self, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload)
        if not self.gui_clients:
            return
        await asyncio.gather(
            *(client.send(message) for client in list(self.gui_clients) if client.open),
            return_exceptions=True,
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


async def main(config_path: Path | str = "config.yml") -> None:
    server = ShotServer(Path(config_path))
    robot_server = await websockets.serve(server.robot_handler, "0.0.0.0", 8765, ping_interval=None)
    ui_server = await websockets.serve(server.gui_handler, "0.0.0.0", 8766)

    print("Listening: ws://0.0.0.0:8765/stream  (robot)")
    print("Listening: ws://0.0.0.0:8766/ui      (gui)")

    await asyncio.gather(robot_server.wait_closed(), ui_server.wait_closed())


if __name__ == "__main__":
    asyncio.run(main())
