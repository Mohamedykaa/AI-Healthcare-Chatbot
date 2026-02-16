import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure current directory is in python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
