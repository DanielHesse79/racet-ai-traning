"""Startar racet: bilar som lär sig köra, med all-MiniLM-L6-v2 eller Laya på grafikkortet.

    python play.py               öppnar racet i webbläsaren
    python play.py --no-browser  startar bara servern, på http://127.0.0.1:8765
"""
from engine import run

if __name__ == "__main__":
    run()
