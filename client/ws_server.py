"""Entry point for the Pedro shot-planning WebSocket server."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure the project root (the ancestor that contains the `transport` package)
# is on sys.path so this script can be executed directly from any cwd.
# We walk up ancestors and add the first one that contains a `transport` dir.
_this_file = Path(__file__).resolve()
for _ancestor in ([_this_file.parent] + list(_this_file.parents)):
    if (_ancestor / "transport").is_dir():
        sys.path.insert(0, str(_ancestor))
        break

from transport.ws_server import main


if __name__ == "__main__":
    asyncio.run(main(Path("config.yml")))
