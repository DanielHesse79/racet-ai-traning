"""Startar spelen. Varje drag bestäms av en beslutsmodell på grafikkortet: Nimble via Ollama eller Laya.

    python play.py                             öppnar spelen i webbläsaren
    python play.py --model laya                med Laya vald från start
    python play.py --game learn --headless 50  spelar 50 omgångar i terminalen
"""
from car import CarGame
from car_learn import LearningCar
from engine import run
from lake import Lake
from snake import Snake

GAMES = {"car": CarGame, "learn": LearningCar, "lake": Lake, "snake": Snake}
# modell -> var den körs. nimble = hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q8_0 i Ollama.
MODELS = {"nimble": "ollama", "laya": "laya"}

if __name__ == "__main__":
    run(GAMES, MODELS)
