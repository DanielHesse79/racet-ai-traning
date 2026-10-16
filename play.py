"""Starts the race: cars that learn to drive, with all-MiniLM-L6-v2 or Laya on the GPU.

    python play.py               opens the race in the browser
    python play.py --no-browser  only starts the server, at http://127.0.0.1:8765
"""
from engine import run

if __name__ == "__main__":
    run()
