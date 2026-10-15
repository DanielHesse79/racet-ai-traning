"""Bilspel: väj för mötande bilar på en trefilig väg. Startas med play.py."""
import random

from engine import Game

LANES = ["left", "middle", "right"]
ROAD_ROWS = 8
CAR_ROW = ROAD_ROWS - 1  # din bil står alltid längst ner
STEER = {"left": -1, "stay": 0, "right": 1}
TRAFFIC_DENSITY = 0.8  # chans att en ny bil dyker upp när det finns plats


class CarGame(Game):
    TITLE = "Bilspelet"
    UNIT = "omkörda bilar"
    COLS, ROWS = len(LANES), ROAD_ROWS
    MOVES = {"left": "← Vänster", "stay": "↑ Rakt fram", "right": "→ Höger"}
    QUESTION = "The move that keeps your car from crashing. If more than one move is safe, prefer to stay."

    def start(self):
        self.lane = 1
        self.traffic = set()  # (fil, rad)
        self.crashed = False

    def ahead(self, lane):
        """Antal rader till närmaste bil i filen (0 = precis bredvid), eller None om filen är tom."""
        distances = [CAR_ROW - row for other, row in self.traffic if other == lane]
        return min(distances, default=None)

    def context(self):
        lines = [
            "Driving game on a road with 3 lanes: left, middle and right. "
            f"Your car is in the {LANES[self.lane]} lane. Other cars come 1 row closer every turn.",
            "Result of each possible move:",
        ]
        for move, delta in STEER.items():
            lane = self.lane + delta
            if not 0 <= lane < len(LANES):
                lines.append(f"- {move}: leaves the road. Crash.")
                continue
            distance = self.ahead(lane)
            if distance == 0:
                lines.append(f"- {move}: hits the car right beside you. Crash.")
            elif distance == 1:
                lines.append(f"- {move}: the car 1 row ahead in that lane hits you. Crash.")
            elif distance is None:
                lines.append(f"- {move}: safe. That lane is empty.")
            else:
                lines.append(f"- {move}: safe. The next car in that lane is {distance} rows ahead.")
        return "\n".join(lines)

    def apply(self, move):
        lane = self.lane + STEER[move]
        if not 0 <= lane < len(LANES):
            return self.crash("Körde av vägen")
        if (lane, CAR_ROW) in self.traffic:
            return self.crash("Körde in i bilen bredvid")
        self.lane = lane
        self.traffic = {(other, row + 1) for other, row in self.traffic}
        if (lane, CAR_ROW) in self.traffic:
            return self.crash("Krockade med bilen framför")
        passed = {cell for cell in self.traffic if cell[1] > CAR_ROW}
        self.traffic -= passed
        self.score += len(passed)
        # Högst en ny bil varannan rad, så att det alltid finns en väg förbi.
        if all(row != 1 for _, row in self.traffic) and random.random() < TRAFFIC_DENSITY:
            self.traffic.add((random.randrange(len(LANES)), 0))

    def crash(self, reason):
        self.crashed = True
        self.die(reason)

    def cells(self):
        cells = [[lane, row, "traffic"] for lane, row in sorted(self.traffic) if row <= CAR_ROW]
        cells.append([self.lane, CAR_ROW, "crash" if self.crashed else "car"])
        return cells

