import asyncio
import sys
from pathlib import Path  # Import the Path object

# -----------------------------------------------------------------
# FIX: Set the correct event loop policy *before* any other imports
# that might import asyncio.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -----------------------------------------------------------------

# NOW import your app, which will subsequently import asyncio
from gui import DevControllerApp

if __name__ == "__main__":
    # 1. Get the directory where main.py is located
    #    (e.g., ".../dillylol/client/")
    APP_ROOT = Path(__file__).parent.resolve()
    
    # 2. Create a full, absolute path to the config.yml file in that directory
    #    (e.g., ".../dillylol/client/config.yml")
    CONFIG_PATH = APP_ROOT / "config.yml"

    # 3. Pass this guaranteed-correct path to your app
    app = DevControllerApp(config_path=CONFIG_PATH)
    app.mainloop()