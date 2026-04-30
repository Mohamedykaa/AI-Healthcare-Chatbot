import os
import sys

import uvicorn
from dotenv import load_dotenv

if __name__ == "__main__":
    # Ensure current directory is in python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    # Load .env so config overrides (e.g. CHROMA_PERSIST_DIR) are available
    load_dotenv()

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=True)
