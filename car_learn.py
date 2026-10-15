"""Bilspelet med självinlärning.

Modellen får inga regler och inga färdigtolkade drag – bara läget och vad som hände
tidigare gånger den var i exakt samma läge. Startas med play.py.
"""
from car import LANES, STEER, CarGame
from engine import LearningGame

SIGHT = 2   # så många rader framåt ser föraren
GOAL = 300  # drag utan krock för att komma i mål


class LearningCar(LearningGame, CarGame):
    TITLE = "Bilspelet – självlärande"
    NAME = "car_learn"
    QUESTION = (
        "The move that keeps your car from crashing. Go by what happened in earlier rounds in this "
        "exact situation: never pick a move that crashed, and prefer a move that survived."
    )

    def blank(self):
        return {"experience": {}}  # läge -> drag -> [överlevde, krockade]

    def seen(self, lane):
        distance = self.ahead(lane)
        if distance is None or distance > SIGHT:
            return "no car in sight"
        if distance == 0:
            return "a car is right beside you"
        return f"a car is {distance} row{'s' if distance > 1 else ''} ahead"

    def situation(self):
        lanes = [f"- {name} lane{' (your lane)' if index == self.lane else ''}: {self.seen(index)}."
                 for index, name in enumerate(LANES)]
        return "\n".join(lanes)

    def context(self):
        lines = [
            "Driving game on a road with 3 lanes: left, middle and right. "
            f"Your car is in the {LANES[self.lane]} lane. You can see {SIGHT} rows ahead.",
            self.situation(),
            "What happened in earlier rounds when you made each move in exactly this situation:",
        ]
        tried = self.knowledge["experience"].get(self.situation(), {})
        for move in STEER:
            survived, crashed = tried.get(move, (0, 0))
            if crashed:
                lines.append(f"- {move}: crashed.")
            elif survived:
                lines.append(f"- {move}: survived ({survived} time{'s' if survived > 1 else ''}).")
            else:
                lines.append(f"- {move}: not tried yet.")
        return "\n".join(lines)

    def apply(self, move):
        outcome = self.knowledge["experience"].setdefault(self.situation(), {}).setdefault(move, [0, 0])
        super().apply(move)
        outcome[self.crashed] += 1
        if self.alive and self.steps >= GOAL:
            self.die("Kom i mål utan krock", won=True)

    def memory_info(self):
        experience = self.knowledge["experience"]
        lessons = sum(1 for moves in experience.values() for _, crashed in moves.values() if crashed)
        return f"{len(experience)} lägen sedda, {lessons} krockar ihågkomna"
