import asyncio
import sys
import logging
from pathlib import Path

# NEW: Import the updater
import updater

# Defines the version of THIS code. Incremement this before building a new Release.
APP_VERSION = "1.0.0"

# Setup basic logging to see updater output
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------
# FIX: Set the correct event loop policy *before* any other imports
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -----------------------------------------------------------------

from gui import DevControllerApp

if __name__ == "__main__":
    # 1. Check for updates immediately
    # If an update is found, this function will restart the app and script execution stops here.
    updater.check_and_update(APP_VERSION)

    # 2. Determine paths based on execution mode (exe vs source)
    if getattr(sys, 'frozen', False):
        APP_ROOT = Path(sys.executable).parent
    else:
        APP_ROOT = Path(__file__).parent.resolve()
    
    CONFIG_PATH = APP_ROOT / "config.yml"

    # 3. Launch App
    app = DevControllerApp(config_path=CONFIG_PATH)
    # Optional: Display version in title
    app.title(f"JULES Dev Controller v{APP_VERSION}")
    app.mainloop()