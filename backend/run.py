import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    dev  = os.environ.get("RENDER") is None   # RENDER est injecté par Render en prod
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=dev)
