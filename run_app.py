import subprocess
import sys
import os

def main():
    # Ensure current directory is in python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    cmd = [sys.executable, "-m", "chainlit", "run", "src/ui/app.py"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
