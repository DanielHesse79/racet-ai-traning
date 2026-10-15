"""Isvaken: gå över en frusen sjö till skatten utan att gå genom isen.

Vakarna syns inte, men de ligger på samma ställe varje omgång. Modellen lär sig en karta:
var isen håller, var vakarna är och hur långt det verkar vara till skatten från varje ruta.
Startas med play.py.
"""
from engine import LearningGame

# S = start, T = skatt, # = vak. Ändra gärna kartan, men det måste finnas en väg fram.
MAP = [
    "S...#...",
    ".#.#..#.",
    "...#.#..",
    "##...#.#",
    "...#....",
    ".###.##.",
    ".....#..",
    ".#.#...T",
]
SIZE = len(MAP)
SQUARES = {(x, y): mark for y, row in enumerate(MAP) for x, mark in enumerate(row)}
START = next(cell for cell, mark in SQUARES.items() if mark == "S")
TREASURE = next(cell for cell, mark in SQUARES.items() if mark == "T")
HOLES = {cell for cell, mark in SQUARES.items() if mark == "#"}
MAX_STEPS = 60
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def distance(cell):
    return abs(cell[0] - TREASURE[0]) + abs(cell[1] - TREASURE[1])


def key(cell):
    return f"{cell[0]},{cell[1]}"


class Lake(LearningGame):
    TITLE = "Isvaken – självlärande"
    NAME = "lake"
    UNIT = "steg närmare skatten"
    COLS = ROWS = SIZE
    MOVES = {"up": "↑ Upp", "down": "↓ Ner", "left": "← Vänster", "right": "→ Höger"}
    QUESTION = (
        "The best next move toward the treasure. Never step into a known hole and never walk into the "
        "shore. Pick the move with the fewest estimated steps to the treasure."
    )

    def blank(self):
        # Rutor som [kolumn, rad]. estimate: "kolumn,rad" -> inlärt antal steg till skatten.
        return {"holes": [], "safe": [list(START)], "estimate": {}}

    def start(self):
        self.position = START
        self.fell = False

    def estimate(self, cell):
        """Antal steg till skatten: det inlärda värdet, annars avståndet rakt dit."""
        return self.knowledge["estimate"].get(key(cell), distance(cell))

    def learn(self, cell):
        """En ruta är minst ett steg längre bort än sin bästa granne (LRTA*).

        Så blir återvändsgränder dyrare för varje besök, tills en annan väg ser bättre ut.
        """
        around = [(cell[0] + dx, cell[1] + dy) for dx, dy in DIRECTIONS.values()]
        reachable = [self.estimate(c) for c in around if c in SQUARES and list(c) not in self.knowledge["holes"]]
        if 1 + min(reachable) > self.estimate(cell):
            self.knowledge["estimate"][key(cell)] = 1 + min(reachable)

    def context(self):
        x, y = self.position
        lines = [
            f"Ice lake game on a {SIZE}x{SIZE} grid. Columns 0-{SIZE - 1} run left to right, rows 0-{SIZE - 1} "
            "run top to bottom. Some squares are holes in the ice that cannot be seen. Falling into a hole "
            "ends the round. The holes are in the same squares every round.",
            f"You are at column {x}, row {y}. The treasure is at column {TREASURE[0]}, row {TREASURE[1]}.",
            "What you know about each possible move:",
        ]
        targets = {move: (x + dx, y + dy) for move, (dx, dy) in DIRECTIONS.items()}
        open_ice = [c for c in targets.values() if c in SQUARES and list(c) not in self.knowledge["holes"]]
        shortest = min(self.estimate(c) for c in open_ice)
        for move, cell in targets.items():
            if cell not in SQUARES:
                lines.append(f"- {move}: blocked by the shore.")
            elif cell == TREASURE:
                lines.append(f"- {move}: reaches the treasure.")
            elif list(cell) in self.knowledge["holes"]:
                lines.append(f"- {move}: a known hole. You fell in here in an earlier round.")
            else:
                ice = "safe ice, walked before" if list(cell) in self.knowledge["safe"] else "unknown ice, never tried"
                # Modellerna är svaga på att jämföra tal, så det minsta pekas ut i klartext.
                best = " This is the fewest." if self.estimate(cell) == shortest else ""
                lines.append(f"- {move}: {ice}. Estimated {self.estimate(cell)} steps from the treasure.{best}")
        return "\n".join(lines)

    def apply(self, move):
        origin = self.position
        dx, dy = DIRECTIONS[move]
        cell = (origin[0] + dx, origin[1] + dy)
        if cell in HOLES:
            if list(cell) not in self.knowledge["holes"]:
                self.knowledge["holes"].append(list(cell))
            self.position = cell
            self.fell = True
            self.learn(origin)
            return self.die("Gick genom isen")
        self.learn(origin)
        if cell in SQUARES:  # annars står den kvar vid strandkanten
            self.position = cell
            if list(cell) not in self.knowledge["safe"]:
                self.knowledge["safe"].append(list(cell))
            self.score = max(self.score, distance(START) - distance(cell))
            if cell == TREASURE:
                return self.die(f"Hittade skatten på {self.steps} drag", won=True)
        if self.steps >= MAX_STEPS:
            self.die(f"Orkade inte längre ({MAX_STEPS} drag)")

    def memory_info(self):
        known = self.knowledge
        return (f"{len(known['holes'])} vakar hittade, {len(known['safe'])} säkra rutor, "
                f"{len(known['estimate'])} omvärderade avstånd")

    def cells(self):
        # Siffran i en säker ruta är modellens nuvarande uppskattning av stegen till skatten.
        cells = [[x, y, "safe", str(self.estimate((x, y)))] for x, y in self.knowledge["safe"]]
        cells += [[x, y, "hole"] for x, y in self.knowledge["holes"]]
        cells.append([*TREASURE, "goal"])
        cells.append([*self.position, "fell" if self.fell else "player"])
        return cells
