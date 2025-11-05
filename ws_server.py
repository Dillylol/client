"""Entry point for the Pedro shot-planning WebSocket server."""
from __future__ import annotations

import asyncio
from pathlib import Path

from transport.ws_server import main


if __name__ == "__main__":
    asyncio.run(main(Path("config.yml")))
