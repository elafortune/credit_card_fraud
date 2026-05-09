import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=True)
