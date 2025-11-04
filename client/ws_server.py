"""WebSocket server orchestrating robot ↔ client shot planning."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from .evaluator import ShotEvaluator


class RobotSession:
    """Track heartbeat and status information for the connected robot."""

    def __init__(self) -> None:
        self.last_hb: Optional[int] = None
        self.battery_v: Optional[float] = None
        self.active_opmode: str = "NoOpMode"
        self.seq: int = 0
        self.ping_inflight: Dict[str, int] = {}

    def apply_heartbeat(self, obj: Dict[str, object]) -> None:
        self.last_hb = int(obj.get("ts_ms", time.time() * 1000))
        self.seq = int(obj.get("seq", self.seq))
        battery = obj.get("battery_v")
        if battery is not None:
            self.battery_v = float(battery)
        active = obj.get("active_opmode")
        if isinstance(active, str):
            self.active_opmode = active

    def heartbeat_age_ms(self) -> Optional[int]:
        if self.last_hb is None:
            return None
        return int(time.time() * 1000) - int(self.last_hb)


class ShotServer:
    """Server managing shot planning and UI fan-out."""

    def __init__(self, evaluator: ShotEvaluator) -> None:
        self.session = RobotSession()
        self.evaluator = evaluator
        self.clients: Set[WebSocketServerProtocol] = set()
        self.robot_ws: Optional[WebSocketServerProtocol] = None

    # --- Robot connection handling ---------------------------------
    async def robot_handler(self, ws: WebSocketServerProtocol) -> None:
        self.robot_ws = ws
        async for msg in ws:
            for line in msg.splitlines():
                await self._handle_robot_message(ws, line)

    async def _handle_robot_message(self, ws: WebSocketServerProtocol, line: str) -> None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return

        msg_type = obj.get("type")
        if msg_type == "heartbeat":
            self.session.apply_heartbeat(obj)
            await self._broadcast_ui(
                {
                    "type": "ui_update",
                    "battery_v": self.session.battery_v,
                    "heartbeat_age_ms": self.session.heartbeat_age_ms(),
                    "active_opmode": self.session.active_opmode,
                    "seq": self.session.seq,
                }
            )
        elif msg_type == "pong":
            await self._handle_robot_pong(obj)
        elif msg_type == "request_shot_plan":
            plan = self.evaluator.handle_request_shot_plan(obj)
            await ws.send(json.dumps(plan))
        elif msg_type == "obs_shot":
            update = self.evaluator.handle_obs_shot(obj)
            if update is not None:
                await ws.send(json.dumps(update))
            await self._broadcast_ui({"type": "obs_shot", "payload": obj})
        else:
            # keep existing acknowledgement behavior untouched
            if msg_type == "ack":
                await self._broadcast_ui({"type": "ack", "payload": obj})

    async def _handle_robot_pong(self, obj: Dict[str, object]) -> None:
        pid = obj.get("id")
        if not isinstance(pid, str):
            return
        t0 = self.session.ping_inflight.pop(pid, None)
        if t0 is not None:
            rtt = int(time.time() * 1000 - t0)
            await self._broadcast_ui({"type": "ping", "rtt_ms": rtt})

    async def pinger(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            if self.robot_ws is None or not self.robot_ws.open:
                continue
            pid = str(uuid.uuid4())
            ts_ms = int(time.time() * 1000)
            self.session.ping_inflight[pid] = ts_ms
            await self.robot_ws.send(json.dumps({"type": "ping", "id": pid, "t0": ts_ms}))

    # --- GUI fan-out ------------------------------------------------
    async def gui_handler(self, ws: WebSocketServerProtocol) -> None:
        self.clients.add(ws)
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "ui_update",
                        "battery_v": self.session.battery_v,
                        "heartbeat_age_ms": self.session.heartbeat_age_ms(),
                        "active_opmode": self.session.active_opmode,
                        "seq": self.session.seq,
                    }
                )
            )
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)

    async def _broadcast_ui(self, payload: Dict[str, object]) -> None:
        if not self.clients:
            return
        message = json.dumps(payload)
        await asyncio.gather(*(client.send(message) for client in list(self.clients) if client.open), return_exceptions=True)


async def main(model_path: str | Path = "rpm_model.json") -> None:
    evaluator = ShotEvaluator(Path(model_path))
    server = ShotServer(evaluator)

    robot_server = await websockets.serve(server.robot_handler, "0.0.0.0", 8765, ping_interval=None)
    ui_server = await websockets.serve(server.gui_handler, "0.0.0.0", 8766)

    print("Listening: ws://0.0.0.0:8765/stream  (robot)")
    print("Listening: ws://0.0.0.0:8766/ui      (gui)")

    await asyncio.gather(server.pinger())


if __name__ == "__main__":
    asyncio.run(main())
