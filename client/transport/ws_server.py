"""Pedro shot-planning WebSocket server bound to the client port."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

import websockets
from websockets import ConnectionClosed
from websockets.server import WebSocketServerProtocol, serve

try:  # Support both package and script execution layouts
    from client.client.config import ClientConfig
    from client.client.evaluator import ShotEvaluator
    from client.client.rpm_model import RpmModel
except ModuleNotFoundError:  # pragma: no cover - fallback when run from repo root
    from client.config import ClientConfig
    from client.evaluator import ShotEvaluator
    from client.rpm_model import RpmModel

logger = logging.getLogger(__name__)

_JULES_PATH = "/jules"


@dataclass
class SessionState:
    """Track connection sequencing and health."""

    session_id: Optional[str] = None
    seq_out: int = 0
    last_obs_ts_ms: Optional[int] = None
    clock_skew_ms: Optional[float] = None
    last_seq_by_type: Dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.session_id = None
        self.seq_out = 0
        self.last_obs_ts_ms = None
        self.clock_skew_ms = None
        self.last_seq_by_type.clear()

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
            if self.clock_skew_ms is None:
                self.clock_skew_ms = float(robot_ts - now_ms)
            else:
                self.clock_skew_ms = 0.9 * self.clock_skew_ms + 0.1 * (robot_ts - now_ms)
        self.last_obs_ts_ms = now_ms

    def offline(self, now_ms: int) -> bool:
        if self.last_obs_ts_ms is None:
            return False
        return (now_ms - self.last_obs_ts_ms) > 1500


class ShotServer:
    """Coordinate RPM-only planning and telemetry over WebSockets."""

    def __init__(
        self,
        config: ClientConfig,
        model: RpmModel,
        *,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.evaluator = ShotEvaluator(config, model)
        self.session = SessionState()
        self._robot_ws: Optional[WebSocketServerProtocol] = None
        self._on_event = on_event
        self._on_status = on_status

    async def handle(self, ws: WebSocketServerProtocol, path: str) -> None:
        if path != _JULES_PATH:
            logger.warning("Rejecting connection on unexpected path %s", path)
            await ws.close(code=4404, reason="invalid path")
            return
        if self._robot_ws is not None and self._robot_ws.open:
            logger.warning("Rejecting connection; robot slot already in use")
            await ws.close(code=4409, reason="already connected")
            return

        logger.info("Robot connected from %s", ws.remote_address)
        self._emit_status(
            {
                "state": "robot_connected",
                "remote": str(ws.remote_address),
            }
        )
        self._robot_ws = ws
        self.session.reset()
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.warning("Dropping non-UTF8 frame")
                        continue
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    await self._handle_message(ws, line)
        except ConnectionClosed:
            logger.info("Robot connection closed")
        finally:
            self._robot_ws = None
            self.session.reset()
            self._emit_status({"state": "robot_disconnected"})

    async def _handle_message(self, ws: WebSocketServerProtocol, line: str) -> None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON frame: %s", line)
            return

        msg_type = obj.get("type")
        seq = obj.get("seq")
        now_ms = _now_ms()

        if msg_type == "hello":
            self._emit_event(
                {
                    "direction": "inbound",
                    "type": msg_type,
                    "payload": obj,
                }
            )
            await self._handle_hello(ws, obj, seq, now_ms)
            return

        if self.session.session_id is None:
            logger.debug("Ignoring %s before hello", msg_type)
            return
        if obj.get("session_id") != self.session.session_id:
            logger.debug("Ignoring %s for stale session", msg_type)
            return
        if msg_type != "request_shot_plan" and not self.session.register_seq(str(msg_type), seq):
            logger.debug("Dropping duplicate %s seq=%s", msg_type, seq)
            return

        if msg_type == "request_shot_plan":
            self._emit_event(
                {
                    "direction": "inbound",
                    "type": msg_type,
                    "payload": obj,
                }
            )
            await self._handle_request(ws, obj, now_ms)
        elif msg_type == "obs":
            self.session.mark_obs(obj.get("ts_ms"), now_ms)
            self._emit_event(
                {
                    "direction": "inbound",
                    "type": msg_type,
                    "payload": obj,
                }
            )
        elif msg_type == "token_shot_fired":
            self.session.register_seq(msg_type, seq)
            self.evaluator.record_shot_fired(obj)
            self._emit_event(
                {
                    "direction": "inbound",
                    "type": msg_type,
                    "payload": obj,
                }
            )
        elif msg_type == "shot_result":
            self.session.mark_obs(obj.get("ts_ms"), now_ms)
            update = self.evaluator.apply_shot_result(obj, now_ms)
            self._emit_event(
                {
                    "direction": "inbound",
                    "type": msg_type,
                    "payload": obj,
                }
            )
            if update:
                await self._send_model_update(ws, update, now_ms)
        elif msg_type == "rpm_model_update":
            logger.info("Robot acknowledged rpm_model_update: %s", obj.get("update_id"))
            self._emit_event(
                {
                    "direction": "inbound",
                    "type": msg_type,
                    "payload": obj,
                }
            )
        elif msg_type == "pong":
            self.session.mark_obs(obj.get("ts_ms"), now_ms)
            self._emit_event(
                {
                    "direction": "inbound",
                    "type": msg_type,
                    "payload": obj,
                }
            )
        else:
            logger.debug("Unhandled message type: %s", msg_type)
            self._emit_event(
                {
                    "direction": "inbound",
                    "type": msg_type or "unknown",
                    "payload": obj,
                }
            )

    async def _handle_hello(
        self,
        ws: WebSocketServerProtocol,
        obj: Dict[str, Any],
        seq: Optional[int],
        now_ms: int,
    ) -> None:
        session_id = obj.get("session_id")
        token = obj.get("token", "")
        if not isinstance(session_id, str):
            logger.warning("hello missing session_id")
            await ws.close(code=4400, reason="missing session_id")
            return
        if self.config.require_token and token != self.config.token:
            logger.warning("hello rejected due to token mismatch")
            await ws.close(code=4403, reason="invalid token")
            return

        self.session.reset()
        self.session.session_id = session_id
        self.session.register_seq("hello", seq)
        self.evaluator.start_session(session_id)
        ack = {
            "type": "hello_ack",
            "session_id": session_id,
            "seq": self.session.next_seq(),
            "policy": self.config.policy.name,
            "model_version": self.model.model_version,
            "ts_ms": now_ms,
        }
        await ws.send(json.dumps(ack))
        logger.info("hello_ack sent for session %s", session_id)
        self._emit_status(
            {
                "state": "session_started",
                "session_id": session_id,
            }
        )
        self._emit_event(
            {
                "direction": "outbound",
                "type": "hello_ack",
                "payload": ack,
            }
        )

    async def _handle_request(
        self,
        ws: WebSocketServerProtocol,
        obj: Dict[str, Any],
        now_ms: int,
    ) -> None:
        offline = self.session.offline(now_ms)
        command = self.evaluator.plan_shot(obj, now_ms, offline=offline)
        cmd = dict(command)
        cmd.setdefault("valid_ms", self.config.cmd_valid_ms)
        cmd.update(
            {
                "type": "cmd",
                "session_id": self.session.session_id,
                "seq": self.session.next_seq(),
                "ts_ms": now_ms,
            }
        )
        self._enforce_rpm_surface(cmd)
        await ws.send(json.dumps(cmd))
        logger.debug("Sent cmd %s", cmd.get("cmd_id"))
        self.session.register_seq("request_shot_plan", obj.get("seq"))
        self._emit_event(
            {
                "direction": "outbound",
                "type": "cmd",
                "payload": cmd,
            }
        )

    async def _send_model_update(
        self,
        ws: WebSocketServerProtocol,
        update: Dict[str, Any],
        now_ms: int,
    ) -> None:
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
        logger.info(
            "Pushed rpm_model_update version=%s", payload.get("model_version")
        )
        self._emit_event(
            {
                "direction": "outbound",
                "type": "rpm_model_update",
                "payload": payload,
            }
        )

    def _emit_event(self, event: Dict[str, Any]) -> None:
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001
                logger.exception("RL event callback raised")

    def _emit_status(self, status: Dict[str, Any]) -> None:
        if self._on_status:
            try:
                self._on_status(status)
            except Exception:  # noqa: BLE001
                logger.exception("RL status callback raised")

    def _enforce_rpm_surface(self, cmd: Dict[str, Any]) -> None:
        has_bias = "rpm_bias" in cmd and cmd["rpm_bias"] is not None
        has_abs = "rpm_target_abs" in cmd and cmd["rpm_target_abs"] is not None
        if has_bias and has_abs:
            if self.config.policy.send_abs_rpm:
                cmd.pop("rpm_bias", None)
            else:
                cmd.pop("rpm_target_abs", None)
        if not has_bias and not has_abs:
            cmd["rpm_bias"] = 0

    async def run(self, *, stop_event: Optional[asyncio.Event] = None) -> None:
        logger.info(
            "Listening for robot on ws://%s:%d%s",
            self.config.server_host,
            self.config.server_port,
            _JULES_PATH,
        )
        self._emit_status(
            {
                "state": "listening",
                "endpoint": f"ws://{self.config.server_host}:{self.config.server_port}{_JULES_PATH}",
            }
        )
        server = await serve(
            self.handle,
            self.config.server_host,
            self.config.server_port,
            ping_interval=None,
        )
        try:
            if stop_event is None:
                await asyncio.Future()
            else:
                await stop_event.wait()
        finally:
            server.close()
            await server.wait_closed()
            self._emit_status(
                {
                    "state": "stopped",
                    "endpoint": f"ws://{self.config.server_host}:{self.config.server_port}{_JULES_PATH}",
                }
            )


FrameCallback = Callable[[Dict[str, Any]], None]
StatusCallback = Callable[[Dict[str, Any]], None]


class WSServer:
    """Compatibility WebSocket server for the GUI link manager."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        path: str = _JULES_PATH,
        token: Optional[str],
        loop: asyncio.AbstractEventLoop,
        on_frame: FrameCallback,
        on_status: StatusCallback,
    ) -> None:
        self._host = host
        self._port = port
        self._path = path
        self._token = token
        self._loop = loop
        self._on_frame = on_frame
        self._on_status = on_status

        self._stop_event = asyncio.Event()
        self._send_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._active_ws: Optional[WebSocketServerProtocol] = None

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(lambda: self._stop_event.set())

    async def run(self) -> None:
        async with websockets.serve(self._handler, self._host, self._port, ping_interval=None):
            self._emit_status(
                {
                    "transport": "ws_server",
                    "state": "listening",
                    "endpoint": f"ws://{self._host}:{self._port}{self._path}",
                }
            )
            await self._stop_event.wait()
        self._emit_status(
            {
                "transport": "ws_server",
                "state": "stopped",
                "endpoint": f"ws://{self._host}:{self._port}{self._path}",
            }
        )

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        parsed = urlparse(ws.path)
        if parsed.path != self._path:
            await ws.close(code=4000, reason="invalid path")
            return
        if self._token and not self._check_token(parsed, ws):
            await ws.close(code=4001, reason="unauthorized")
            return
        if self._active_ws is not None:
            await ws.close(code=4002, reason="already connected")
            return

        self._active_ws = ws
        sender = asyncio.create_task(self._sender(ws))
        receiver = asyncio.create_task(self._receiver(ws))
        self._emit_status(
            {
                "transport": "ws_server",
                "state": "connected",
                "endpoint": f"ws://{self._host}:{self._port}{self._path}",
            }
        )
        try:
            done, pending = await asyncio.wait(
                {sender, receiver},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            self._active_ws = None
            self._emit_status(
                {
                    "transport": "ws_server",
                    "state": "ready",
                    "endpoint": f"ws://{self._host}:{self._port}{self._path}",
                }
            )

    async def _sender(self, ws: WebSocketServerProtocol) -> None:
        while not ws.closed:
            msg = await self._send_queue.get()
            await ws.send(msg)

    async def _receiver(self, ws: WebSocketServerProtocol) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    self._emit_status(
                        {
                            "transport": "ws_server",
                            "state": "bad_frame",
                            "raw": line,
                        }
                    )
                    continue
                self._on_frame(frame)

    def _check_token(self, parsed, ws: WebSocketServerProtocol) -> bool:
        query = parse_qs(parsed.query)
        if "token" in query and query["token"]:
            return query["token"][0] == self._token
        auth = ws.request_headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1] == self._token
        return False

    def queue_json(self, payload: Dict[str, Any]) -> None:
        text = json.dumps(payload)
        self.queue_text(text)

    def queue_text(self, text: str) -> None:
        async def _enqueue() -> None:
            await self._send_queue.put(text)

        asyncio.run_coroutine_threadsafe(_enqueue(), self._loop)

    def _emit_status(self, status: Dict[str, Any]) -> None:
        self._on_status(status)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def main(config_path: Path | str = "config.yml") -> None:
    config = ClientConfig.load(Path(config_path))
    model = RpmModel.load(config.paths.model_path)
    server = ShotServer(config, model)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
