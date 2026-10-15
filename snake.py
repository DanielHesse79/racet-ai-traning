"""Snake. Startas med play.py."""
import random
from collections import deque

from engine import Game

SIZE = 10
STARVE_AFTER = 100  # drag utan äpple innan ormen svälter
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class Snake(Game):
    TITLE = "Snake"
    UNIT = "äpplen"
    COLS = ROWS = SIZE
    MOVES = {"up": "↑ Upp", "down": "↓ Ner", "left": "← Vänster", "right": "→ Höger"}
    QUESTION = (
        "The best next move for the snake. Never pick a move that kills the snake. Among safe "
        "moves prefer the one closest to the apple, but avoid moves that leave few reachable open cells."
    )

    def start(self):
        mid = SIZE // 2
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]  # huvudet först
        self.hungry = 0
        self.place_apple()

    def place_apple(self):
        free = [(x, y) for x in range(SIZE) for y in range(SIZE) if (x, y) not in self.snake]
        self.apple = random.choice(free) if free else None

    def look(self, move):
        """Vad händer om ormen går åt `move`? Returnerar (utfall, ny huvudruta)."""
        dx, dy = DIRECTIONS[move]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        if not (0 <= head[0] < SIZE and 0 <= head[1] < SIZE):
            return "wall", head
        if head == self.apple:
            return "apple", head
        if head in self.snake[:-1]:  # svansspetsen hinner flytta sig
            return "body", head
        return "free", head

    def room(self, head):
        """Antal rutor ormen kan nå från `head` – litet tal betyder återvändsgränd."""
        blocked = set(self.snake[:-1])
        seen, queue = {head}, deque([head])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS.values():
                cell = (x + dx, y + dy)
                if 0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE and cell not in blocked and cell not in seen:
                    seen.add(cell)
                    queue.append(cell)
        return len(seen)

    def context(self):
        hx, hy = self.snake[0]
        ax, ay = self.apple
        lines = [
            f"Snake game on a {SIZE}x{SIZE} grid. Columns 0-{SIZE - 1} run left to right, "
            f"rows 0-{SIZE - 1} run top to bottom. The snake dies if it hits a wall or its own body.",
            f"Snake head: column {hx}, row {hy}. Apple: column {ax}, row {ay}. Snake length: {len(self.snake)}.",
            "Result of each possible move:",
        ]
        for move in DIRECTIONS:
            outcome, (x, y) = self.look(move)
            if outcome == "wall":
                lines.append(f"- {move}: hits the wall. The snake dies.")
            elif outcome == "body":
                lines.append(f"- {move}: hits the snake's own body. The snake dies.")
            else:
                distance = abs(x - ax) + abs(y - ay)
                what = "eats the apple" if outcome == "apple" else f"safe, then {distance} steps from the apple"
                lines.append(f"- {move}: {what}. Open cells reachable afterwards: {self.room((x, y))}.")
        return "\n".join(lines)

    def apply(self, move):
        outcome, head = self.look(move)
        if outcome in ("wall", "body"):
            self.die("Körde in i väggen" if outcome == "wall" else "Bet sig själv")
            return
        self.snake.insert(0, head)
        if outcome == "apple":
            self.score += 1
            self.hungry = 0
            self.place_apple()
            if self.apple is None:
                self.die("Brädet är fullt – ormen vann!")
        else:
            self.snake.pop()
            self.hungry += 1
            if self.hungry >= STARVE_AFTER:
                self.die(f"Svalt ihjäl ({STARVE_AFTER} drag utan äpple)")

    def cells(self):
        cells = [[x, y, "snake"] for x, y in self.snake[:0:-1]]
        cells.append([*self.snake[0], "head"])
        if self.apple:
            cells.append([*self.apple, "apple"])
        return cells

