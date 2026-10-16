"""Servern bakom racet: modellerna på grafikkortet, GPU-status och sidorna.

Racet simuleras i webbläsaren (race.html). Servern svarar på modellens beslut och tränar den
mellan generationerna. Modellen är all-MiniLM-L6-v2 eller Laya, som båda körs med PyTorch.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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


MEMORY = Path(__file__).parent / "memory"


def create_laya():
    import race_laya
    return race_laya.LayaDriver(MEMORY / "race_laya.pt")


def create_minibert():
    import race_bert
    return race_bert.TinyBertDriver(MEMORY / "race_minibert.pt")


# Förarna: all-MiniLM-L6-v2, en väldigt liten BERT som tränas från noll, och Laya.
# Bara en av dem ligger på grafikkortet åt gången: väljs den ena släpps den andra.
RACERS = {"minibert": ModelService("all-MiniLM-L6-v2", create_minibert), "laya": ModelService("Laya", create_laya)}


def gpu_status():
    """Grafikkortets belastning enligt nvidia-smi."""
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
    return {"name": name, "util": util, "memoryUsed": used, "memoryTotal": total, "temperature": temperature,
            "power": power, "powerLimit": limit}


def serve(port, open_browser):
    here = Path(__file__).parent
    files = {"/": (here / "race.html", "text/html"), "/race": (here / "race.html", "text/html"),
             "/gpu.js": (here / "gpu.js", "text/javascript"), "/theme.css": (here / "theme.css", "text/css")}
    gpu = {"at": 0.0, "status": None, "lock": threading.Lock()}  # nvidia-smi högst en gång per 0,8 s

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status=200):
            self.send(status, json.dumps(value).encode())

        def body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def gpu(self):
            with gpu["lock"]:
                if time.monotonic() - gpu["at"] > 0.8:
                    gpu["status"], gpu["at"] = gpu_status(), time.monotonic()
                return dict(gpu["status"])

        def racer(self):
            """Föraren enligt ?model=: minibert eller laya."""
            model = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("model", ["minibert"])[0]
            return RACERS.get(model)

        def race(self, action):
            """Modellanropen: decide, learn och reset."""
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
            else:
                self.send(404, b"{}")

        def do_POST(self):
            if not self.path.startswith("/api/race/"):
                return self.send(404, b"{}")
            try:
                self.race(urllib.parse.urlsplit(self.path).path.rsplit("/", 1)[1])
            except (OSError, ValueError) as error:
                self.send_json({"error": str(error)}, 502)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Racet körs på {url}  (Ctrl+C avslutar)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def run():
    parser = argparse.ArgumentParser(description="Racet: bilar som lär sig köra, med en modell på grafikkortet.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="öppna inte webbläsaren automatiskt")
    sys.stdout.reconfigure(encoding="utf-8")  # å, ä, ö även när utdata skickas vidare i Windows
    args = parser.parse_args()
    serve(args.port, not args.no_browser)
