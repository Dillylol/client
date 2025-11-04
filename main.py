import asyncio
import sys

# -----------------------------------------------------------------
# FIX: Set the correct event loop policy *before* any other imports
# that might import asyncio.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -----------------------------------------------------------------

# NOW import your app, which will subsequently import asyncio
from gui import DevControllerApp

if __name__ == "__main__":
    app = DevControllerApp()
    app.mainloop()