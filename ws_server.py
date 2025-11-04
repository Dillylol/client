"""Entry point that proxies to :mod:`client.ws_server`."""
from __future__ import annotations

import asyncio
from pathlib import Path

from client.ws_server import main


if __name__ == "__main__":
    asyncio.run(main(Path("rpm_model.json")))
