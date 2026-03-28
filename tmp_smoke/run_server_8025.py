import sys
from pathlib import Path
ROOT = Path(r"C:\Users\rahul\Nyaymalaw-5.0")
sys.path.insert(0, str(ROOT))
import uvicorn
import api_server
uvicorn.run(api_server.app, host="127.0.0.1", port=8025)
