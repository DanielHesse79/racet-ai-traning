"""Racets lägestext: det en bil berättar för modellen, och manövrerna modellen väljer mellan.

Delas av förarna och kräver inte PyTorch. Nyckeln är sju tecken: fart, hur vägen svänger
längre fram, och avståndet till vägkanten i fem riktningar. Med sensorerna avstängda är allt
utom farten "-": bilen är blind.
"""
QUESTION = "How should the race car steer to stay on the road and keep moving forward?"
OPTIONS = {  # samma ordning som ACTIONS i race.html
    "sharp left": "turn hard to the left and slow down",
    "left": "steer gently to the left",
    "straight": "keep straight and speed up",
    "right": "steer gently to the right",
    "sharp right": "turn hard to the right and slow down",
    "brake": "keep straight and slow down",
}
SPEEDS = ["slow", "medium", "fast", "very fast"]
CURVES = ["turns sharply left", "bends left", "goes straight", "bends right", "turns sharply right"]
DISTANCES = ["very close", "close", "some room", "far", "open"]
SENSORS = ["far left", "left", "straight ahead", "right", "far right"]


def describe(key):
    """Läget som text, t.ex. key "2314021" -> "Speed: fast. The road ahead bends right. Distance to ..."."""
    if key[1] == "-":
        return f"Speed: {SPEEDS[int(key[0])]}. The sensors are off: the car cannot see the road or where it bends."
    speed, curve, *rays = (int(c) for c in key)
    edges = ", ".join(f"{name} {DISTANCES[d]}" for name, d in zip(SENSORS, rays))
    return f"Speed: {SPEEDS[speed]}. The road ahead {CURVES[curve]}. Distance to the road edge: {edges}."


def vocabulary():
    return {"question": QUESTION, "options": list(OPTIONS), "speeds": SPEEDS, "curves": CURVES,
            "distances": DISTANCES, "sensors": SENSORS}
