"""Gemensam motor: frågar en beslutsmodell om nästa drag och visar spelet.

Modellen är antingen Nimble i Ollama eller Laya, som körs direkt på grafikkortet med PyTorch.

Ett spel är en underklass till Game (se snake.py och car.py).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
HEADLESS_MAX_STEPS = 1000

class ModelService:
    """En PyTorch-modell på grafikkortet, laddad i bakgrunden första gången den behövs:
    PyTorch används bara här, och det tar en stund att lägga en modell på grafikkortet."""
    loading = threading.Lock()  # en modell i taget: transformers tål inte att importeras från två trådar

    def __init__(self, name, create):
        self.name = name
        self.create = create  # bygger modellen; körs i bakgrundstråden
        self.driver = None
        self.error = None
        self.thread = None
        self.loaded = threading.Event()
        self.lock = threading.Lock()  # ett anrop åt gången på grafikkortet

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._load, daemon=True)
            self.thread.start()

    def status(self):
        self.start()
        status = {"ready": self.driver is not None, "error": self.error}
        if self.driver:
            status.update(self.driver.vocabulary(), version=self.driver.version, updates=self.driver.updates)
        return status

    def _load(self):
        try:
            with ModelService.loading:
                self.driver = self.create()
        except ImportError as error:
            self.error = f"{self.name} kräver PyTorch med CUDA och transformers i den Python som kör play.py ({error})."
        except Exception as error:  # saknad modell i cachen, CUDA-fel med mera: visas på sidan i stället
            self.error = f"Kunde inte ladda {self.name}: {error}"
        finally:
            self.loaded.set()

    def unload(self):
        """Släpp modellen ur grafikminnet. Det den lärt sig ligger kvar i sin fil i memory/."""
        if self.thread is None:
            return
        self.loaded.wait()  # en laddning som pågår får bli klar först
        with self.lock:
            driver, self.driver = self.driver, None
            self.thread, self.error = None, None
            self.loaded.clear()
        if driver is not None:
            import gc
            import torch
            del driver
            gc.collect()
            torch.cuda.empty_cache()

    def choose(self, context, question, choices):
        self.start()
        self.loaded.wait()
        if not self.driver:
            raise ValueError(self.error)
        with self.lock:
            return self.driver.choose(context, question, choices)


MEMORY = Path(__file__).parent / "memory"


def create_laya():
    import race_laya
    return race_laya.LayaDriver(MEMORY / "race_laya.pt")


def create_minibert():
    import race_bert
    return race_bert.TinyBertDriver(MEMORY / "race_minibert.pt")


laya = ModelService("Laya", create_laya)
# Förarna i racet: Laya, och all-MiniLM-L6-v2, en väldigt liten BERT som tränas från noll.
# Bara en av dem ligger på grafikkortet åt gången: väljs den ena släpps den andra.
RACERS = {"laya": laya, "minibert": ModelService("all-MiniLM-L6-v2", create_minibert)}


def decide(model, backend, context, question, choices):
    """Låter modellen välja ett av `choices`. Returnerar (val, {val: sannolikhet}, ms, tokenanvändning).

    backend är "laya" för Laya direkt på grafikkortet, annars "ollama" (en beslutsmodell i Ollama).
    """
    if backend == "laya":
        started = time.perf_counter()
        probs, usage = laya.choose(context, question, choices)
        return max(probs, key=probs.get), probs, round((time.perf_counter() - started) * 1000), usage
    # Ollamas beslutsgränssnitt (samma form som Jev-API:et) bygger Nimbles prompt och läser av
    # sannolikheterna för alternativen.
    body = {"model": model, "state": context, "questions": {"move": {
        "type": "choice", "instructions": question, "criteria": {choice: "" for choice in choices}}}}
    request = urllib.request.Request(
        OLLAMA_URL + "/v1/systemone", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    ms = round((time.perf_counter() - started) * 1000)
    answer = result["answers"]["move"]
    probs = {choice: answer["probabilities"][choice] for choice in choices}
    # Ollama redovisar inte grafikkortets tid här, så hela svarstiden räknas.
    tokens = result.get("usage", {}).get("input_tokens", 0)
    return answer["choice"], probs, ms, {"tokens": tokens, "computed": tokens, "gpu_ms": ms}


class Game:
    """Basklass. Ett spel anger konstanterna och metoderna start, context, apply och cells."""
    TITLE = ""
    UNIT = ""       # vad poängen räknar, t.ex. "äpplen"
    COLS = ROWS = 0
    MOVES = {}      # drag -> etikett i gränssnittet
    QUESTION = ""   # frågan modellen får tillsammans med context()

    def __init__(self, model, backend):
        self.model = model
        self.backend = backend
        self.best = 0
        self.rounds = []  # poängen i varje avslutad omgång
        self.reset()

    def reset(self):
        self.score = 0
        self.steps = 0
        self.alive = True
        self.won = False
        self.reason = ""
        self.last = None
        self.start()

    def die(self, reason, won=False):
        self.alive = False
        self.won = won
        self.reason = reason
        self.rounds.append(self.score)

    def memory_info(self):
        """Självlärande spel returnerar en kort text om vad de har lärt sig."""
        return None

    def step(self):
        if not self.alive:
            return
        move, probs, ms, usage = decide(self.model, self.backend, self.context(), self.QUESTION, list(self.MOVES))
        self.last = {"move": move, "probs": probs, "ms": ms, "usage": usage}
        self.steps += 1
        self.apply(move)
        self.best = max(self.best, self.score)

    def state(self):
        return {
            "title": self.TITLE, "unit": self.UNIT, "cols": self.COLS, "rows": self.ROWS,
            "moves": self.MOVES, "cells": self.cells(), "score": self.score, "best": self.best,
            "steps": self.steps, "alive": self.alive, "won": self.won, "reason": self.reason,
            "last": self.last, "model": self.model, "rounds": self.rounds[-60:],
            "memory": self.memory_info(),
        }


class LearningGame(Game):
    """Spel som minns något mellan omgångar och körningar.

    Minnet ligger i self.knowledge och sparas som memory/<NAME>.<modell>.json efter varje omgång.
    """
    NAME = ""  # filnamnets början

    def __init__(self, model, backend):
        super().__init__(model, backend)
        model_name = re.sub(r"[^\w.-]+", "_", model)
        self.file = Path(__file__).parent / "memory" / f"{self.NAME}.{model_name}.json"
        self.knowledge = self.blank()
        if self.file.exists():
            self.knowledge = json.loads(self.file.read_text(encoding="utf-8"))
            self.rounds = self.knowledge.pop("rounds")
            self.best = max(self.rounds, default=0)

    def blank(self):
        """Ett tomt minne."""
        return {}

    def save(self):
        self.file.parent.mkdir(exist_ok=True)
        self.file.write_text(json.dumps({**self.knowledge, "rounds": self.rounds}), encoding="utf-8")

    def forget(self):
        self.knowledge, self.rounds, self.best = self.blank(), [], 0
        self.save()
        self.reset()

    def step(self):
        was_alive = self.alive
        super().step()
        if was_alive and not self.alive:
            self.save()


def ollama_error(error):
    if isinstance(error, urllib.error.HTTPError):
        detail = error.read().decode(errors="replace")
        return f"Ollama svarade {error.code}: {detail}"
    if isinstance(error, urllib.error.URLError):
        return f"Når inte Ollama på {OLLAMA_URL} – är Ollama igång? ({error.reason})"
    return str(error)


def gpu_status():
    """Grafikkortets belastning enligt nvidia-smi, och vilka modeller Ollama har i grafikminnet."""
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=5, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"Kunde inte läsa nvidia-smi: {error}"}
    name, *values = (value.strip() for value in output.splitlines()[0].split(","))

    def number(text):
        try:
            return round(float(text))
        except ValueError:
            return None  # "[N/A]" på kort som inte rapporterar värdet

    util, used, total, temperature, power, limit = map(number, values)
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/ps", timeout=2) as response:
            models = [{"name": m["name"], "vram": m.get("size_vram", 0)} for m in json.load(response).get("models", [])]
    except (OSError, ValueError):
        models = None
    return {"name": name, "util": util, "memoryUsed": used, "memoryTotal": total, "temperature": temperature,
            "power": power, "powerLimit": limit, "models": models}


def serve(games, models, default_game, default_model, port, open_browser):
    """Serverar alla spel i `games` (id -> klass) med alla modeller i `models` (modell -> "ollama" eller "laya")."""
    lock = threading.Lock()
    page = Path(__file__).with_name("index.html")
    race = Path(__file__).with_name("race.html")  # racet simuleras i webbläsaren
    files = {"/": (page, "text/html"), "/race": (race, "text/html"),
             "/gpu.js": (Path(__file__).with_name("gpu.js"), "text/javascript"),
             "/theme.css": (Path(__file__).with_name("theme.css"), "text/css")}
    gpu = {"at": 0.0, "status": None, "lock": threading.Lock()}  # nvidia-smi högst en gång per 0,8 s
    menu = [[game_id, game_class.TITLE] for game_id, game_class in games.items()]
    running = {}  # (spel-id, modell) -> spel, så att varje modell har sin egen poäng och historik

    def pick(game_id, model):
        if game_id not in games:
            game_id = default_game
        if model not in models:
            model = default_model
        if (game_id, model) not in running:
            running[game_id, model] = games[game_id](model, models[model])
        return game_id, running[game_id, model]

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_state(self, game_id, game, status=200, error=None):
            state = {**game.state(), "id": game_id, "games": menu, "models": list(models), "error": error}
            self.send(status, json.dumps(state).encode())

        def route(self):
            """/api/<spel>/<åtgärd>?model=<modell> -> (spel-id, spel, åtgärd), eller None."""
            url = urllib.parse.urlsplit(self.path)
            parts = url.path.strip("/").split("/")
            if len(parts) != 3 or parts[0] != "api":
                return None
            model = urllib.parse.parse_qs(url.query).get("model", [""])[0]
            return (*pick(parts[1], model), parts[2])

        def send_json(self, value, status=200):
            self.send(status, json.dumps(value).encode())

        def body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def gpu(self):
            with gpu["lock"]:
                if time.monotonic() - gpu["at"] > 0.8:
                    gpu["status"], gpu["at"] = gpu_status(), time.monotonic()
                status = dict(gpu["status"])
            loaded = [racer for racer in RACERS.values() if racer.driver]
            if loaded and "error" not in status:  # samma process, så grafikminnet räknas en gång
                status["models"] = (status["models"] or []) + [
                    {"name": " + ".join(r.name for r in loaded), "vram": loaded[0].driver.vram()}]
            return status

        def racer(self):
            """Föraren i racet enligt ?model=: laya eller minibert."""
            model = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("model", ["laya"])[0]
            return RACERS.get(model)

        def race(self, action):
            """Racets modellanrop: decide, learn och reset."""
            racer = self.racer()
            if racer is None:
                return self.send_json({"error": "Okänd modell."}, 404)
            driver = racer.driver
            if not driver:
                return self.send_json({"error": racer.error or f"{racer.name} laddas fortfarande."}, 503)
            request = self.body()
            with racer.lock:
                if action == "decide":
                    self.send_json(driver.decide(request["keys"]))
                elif action == "learn":
                    self.send_json(driver.learn(request["samples"], request["best"]))
                elif action == "reset":
                    self.send_json(driver.reset())
                else:
                    self.send(404, b"{}")

        def do_GET(self):
            route = self.route()
            if self.path in files:
                file, content_type = files[self.path]
                self.send(200, file.read_bytes(), content_type)
            elif self.path == "/api/gpu":
                self.send_json(self.gpu())
            elif self.path.startswith("/api/race/status"):
                racer = self.racer()
                if not racer:
                    return self.send_json({"ready": False, "error": "Okänd modell."})
                for other in RACERS.values():  # bara en förare åt gången på grafikkortet
                    if other is not racer:
                        other.unload()
                self.send_json(racer.status())
            elif route and route[2] == "state":
                self.send_state(*route[:2])
            else:
                self.send(404, b"{}")

        def do_POST(self):
            if self.path.startswith("/api/race/"):
                try:
                    return self.race(urllib.parse.urlsplit(self.path).path.rsplit("/", 1)[1])
                except (OSError, ValueError) as error:
                    return self.send_json({"error": ollama_error(error)}, 502)
            route = self.route()
            if not route:
                return self.send(404, b"{}")
            game_id, game, action = route
            with lock:
                if action == "reset":
                    game.reset()
                    self.send_state(game_id, game)
                elif action == "forget" and hasattr(game, "forget"):
                    game.forget()
                    self.send_state(game_id, game)
                elif action == "step":
                    try:
                        game.step()
                        self.send_state(game_id, game)
                    except (OSError, ValueError) as error:
                        self.send_state(game_id, game, 502, ollama_error(error))
                else:
                    self.send(404, b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/#{default_game}/{urllib.parse.quote(default_model, safe='')}"
    print(f"Spelen körs på {url}  (Ctrl+C avslutar)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def play_headless(game, rounds):
    for number in range(1, rounds + 1):
        game.reset()
        total_ms = tokens = gpu_ms = 0
        while game.alive and game.steps < HEADLESS_MAX_STEPS:
            game.step()
            total_ms += game.last["ms"]
            tokens += game.last["usage"]["computed"]
            gpu_ms += game.last["usage"]["gpu_ms"]
        ending = game.reason if not game.alive else f"Avbröt efter {HEADLESS_MAX_STEPS} drag"
        print(f"Omgång {number}: {game.score} {game.UNIT} på {game.steps} drag, "
              f"{total_ms / game.steps:.0f} ms/drag, {tokens / max(gpu_ms, 1) * 1000:,.0f} tokens/s. {ending}.",
              flush=True)
    print(f"Bästa: {game.best} {game.UNIT}.")


def run(games, models):
    """Startar spelen i `games` (id -> klass) med `models` (modell -> "ollama" eller "laya"); de första är förvalda."""
    parser = argparse.ArgumentParser(description="Spel som spelas av en beslutsmodell på grafikkortet.")
    parser.add_argument("--game", choices=list(games), default=next(iter(games)),
                        help="spel som visas först, eller som spelas med --headless")
    parser.add_argument("--model", default=next(iter(models)),
                        help=f"modell som är vald från start ({', '.join(models)}, eller en Ollama-modell "
                             "som förstår Nimbles promptformat)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--headless", type=int, metavar="N", help="spela N omgångar i terminalen")
    parser.add_argument("--no-browser", action="store_true", help="öppna inte webbläsaren automatiskt")
    sys.stdout.reconfigure(encoding="utf-8")  # å, ä, ö även när utdata skickas vidare i Windows
    args = parser.parse_args()

    if args.model not in models:
        models = {**models, args.model: "ollama"}

    if args.headless:
        try:
            play_headless(games[args.game](args.model, models[args.model]), args.headless)
        except (OSError, ValueError) as error:
            raise SystemExit(ollama_error(error))
    else:
        serve(games, models, args.game, args.model, args.port, not args.no_browser)
