import asyncio, json, time, uuid
import websockets

class RobotSession:
    def __init__(self):
        self.last_hb = None
        self.battery_v = None
        self.active_opmode = "NoOpMode"
        self.seq = 0
        self.ping_inflight = {}  # id -> t0

    def apply_heartbeat(self, obj):
        self.last_hb = obj.get("ts_ms", int(time.time()*1000))
        self.seq = obj.get("seq", self.seq)
        self.battery_v = obj.get("battery_v", self.battery_v)
        self.active_opmode = obj.get("active_opmode", self.active_opmode)

    def heartbeat_age_ms(self):
        if self.last_hb is None: return None
        return int(time.time()*1000) - int(self.last_hb)

SESSION = RobotSession()
CLIENTS = set()

async def robot_handler(ws):
    async for msg in ws:
        for line in msg.splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "heartbeat":
                SESSION.apply_heartbeat(obj)
                # fan-out to GUI listeners
                payload = {
                    "type":"ui_update",
                    "battery_v": SESSION.battery_v,
                    "heartbeat_age_ms": SESSION.heartbeat_age_ms(),
                    "active_opmode": SESSION.active_opmode,
                    "seq": SESSION.seq
                }
                if CLIENTS:
                    await asyncio.gather(*[c.send(json.dumps(payload)) for c in CLIENTS if c.open])
            elif t == "pong":
                pid = obj.get("id")
                t0 = SESSION.ping_inflight.pop(pid, None)
                if t0 is not None and CLIENTS:
                    rtt = int(time.time()*1000 - t0)
                    await asyncio.gather(*[c.send(json.dumps({"type":"ping","rtt_ms":rtt})) for c in CLIENTS if c.open])

async def gui_handler(ws):
    CLIENTS.add(ws)
    try:
        # push initial snapshot so the UI lights up
        await ws.send(json.dumps({
            "type":"ui_update",
            "battery_v": SESSION.battery_v,
            "heartbeat_age_ms": SESSION.heartbeat_age_ms(),
            "active_opmode": SESSION.active_opmode,
            "seq": SESSION.seq
        }))
        async for _ in ws:
            pass
    finally:
        CLIENTS.discard(ws)

async def pinger(robot_ws):
    while True:
        await asyncio.sleep(2.0)
        if robot_ws.open:
            pid = str(uuid.uuid4())
            SESSION.ping_inflight[pid] = int(time.time()*1000)
            await robot_ws.send(json.dumps({"type":"ping","id":pid,"t0":SESSION.ping_inflight[pid]}))

async def main():
    # Two endpoints:
    # 1) /stream accepts the ROBOT connection (one)
    # 2) /ui accepts any GUI panels (your Tkinter/Qt/… code can attach via local WS)
    robot_server = await websockets.serve(robot_handler, "0.0.0.0", 8765, create_protocol=None, ping_interval=None, process_request=None, subprotocols=None)
    ui_server    = await websockets.serve(gui_handler   , "0.0.0.0", 8766)

    print("Listening: ws://0.0.0.0:8765/stream  (robot)")
    print("Listening: ws://0.0.0.0:8766/ui      (gui)")

    # optional: keep a local robot client reference for pinging if desired
    await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
