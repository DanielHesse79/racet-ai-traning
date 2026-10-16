# The Race – AI training you can watch

Cars that learn to drive. A small language model, **all-MiniLM-L6-v2** (or **Laya**), reads
what each car sees as text, picks a maneuver, and is trained on the GPU between generations.
A teaching example of how a model is trained in practice: best practice, crashes as
punishment, temperature, and three tracks that show whether the model has learned to drive
or only memorized the track. The design follows the Laya Training Visualization Tool
(4Front 2 Market): dark frame, light track, gauges for the GPU.

```bash
python play.py
```

Opens the race at http://localhost:8765.

**Getting started.** The race needs an NVIDIA GPU and a Python with PyTorch (CUDA) and
transformers. The models are read from the Hugging Face cache, so download them once first:

```bash
pip install torch transformers huggingface_hub --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```

```bash
huggingface-cli download convaiinnovations/laya
```

## How the race works

Two drivers, chosen at the top, both trained on the GPU between generations. Only the chosen
one is on the GPU: switching drivers releases the other, and what it has learned is loaded
from its file again when you switch back.

| Driver | Model | What is trained | License |
|---|---|---|---|
| **all-MiniLM-L6-v2** (default) | `sentence-transformers/all-MiniLM-L6-v2`, a very small BERT, 22M parameters, knows nothing about racing at the start | the whole model, plus a new decision layer | Apache-2.0 |
| **Laya** | `convaiinnovations/laya`, ModernBERT-large with a decision head, 421M parameters, pre-trained for decisions | only the decision head; the large encoder is frozen | Apache-2.0 |

A car describes what it sees in text – *"Speed: fast. The road ahead bends right. Distance to
the road edge: far left close, …"* – and the model picks one of six maneuvers: sharp left,
left, straight and accelerate, right, sharp right, straight and brake.

**Part 1 · around the track.**
1. In the first generation every car chooses for itself and crashes. The one that got
   furthest becomes **best practice** – shown with two decimals, e.g. *11.59% of the track*.
2. After that, all cars drive exactly like best practice until 0.5 s before its crash (the
   purple line on the track). There the model takes over and the cars make slightly
   different choices: they explore a little to a lot, i.e. drive with temperature 0.2–1.0.
   One decision in ten is also completely random (epsilon-greedy).
3. The car that gets furthest becomes the new best practice. If no record is set, the point
   where the cars start driving themselves moves 0.5 s further back per generation, at most 3 s.

**Part 2 · cut the time.** Once the first car makes it around, every car gets its own point
where it starts driving itself, spread over the whole record lap (purple dots). The cars try
new lines in many places at once, and some crash on the way. With all-MiniLM-L6-v2 the lap
time went from 34.5 s to 15.5 s in one generation and to 14.5 s in generation 24.

**The test car** (teal) drives with temperature 0 – always what the model believes in most –
and never follows best practice. It drives on the model alone from the start, so its curve
(*model alone*) shows what the model itself has learned. The other curves get help from best
practice.

**Track: learned or memorized?** The controls have three tracks: *Training*, *Mirrored*
(left turns become right turns) and *New track*. The model comes along between tracks, but
best practice and crashes belong to each track. Best practice is memorized steering and only
works on its own track; what the model has learned about speed, curves and road edges works
everywhere. Measured with all-MiniLM-L6-v2 after 7 generations on the training track: on the
first try the test car got 47% around the mirrored track and 38% around the new one, and in
both cases one of the exploring cars made it around the whole lap in the very first generation.

**Sensors off** makes the cars blind: they only know their own speed, not where the road edge
is or how the road bends. Applies from the next decision.

**Cars** are chosen in the controls: 10, 100, 1,000 or 10,000. Switching restarts the
current generation. Measured time per simulation step in the browser, with every car reading
its sensors (the worst case):

| Cars | Per step | Top speed |
|---|---|---|
| 100 | 0.15 ms | ~110× real time |
| 1,000 | 0.74 ms | ~22× |
| 10,000 | 6.1 ms | ~2.7× |

The limit is the browser's CPU, not the models: cars in the same situation share answers, so
a generation has a few hundred to a couple of thousand different situations regardless of the
number of cars. Training uses at most 3,000 situations per generation, and Laya keeps at most
6,000 situations on the GPU (about 1.1 GB). With all-MiniLM-L6-v2 and 10,000 cars a
generation took 7–70 s; the time goes to the simulation, while the model and the training
take a few seconds.

**Training.** Between generations: choices that led to a crash within eight decisions are
punished, choices that moved the car forward are reinforced, and the model imitates best
practice and the generation's ten best cars. Every crash shows as a red cross on the track.

Measured, 30–40 generations per driver:

| | Around the whole track | Lap time later |
|---|---|---|
| all-MiniLM-L6-v2 | generation 3 (52.5 s) | 16.6 s in generation 25 |
| Laya | generation 13 (49.7 s) | 46.3 s in generation 40 |

What the drivers have learned is saved in `memory/race_laya.pt` and `memory/race_minibert.pt`,
and the race records per track in the browser. **Restart** resets the driver and its records
on every track. **Space** plays and pauses.

## The GPU

The gauges show the GPU load and the model's tokens/s. The load comes from `nvidia-smi` via
the server, once per second. Tokens/s covers the time the model is actually computing.

The race loads the GPU in short bursts: the model answers new situations and trains between
generations, while the cars are simulated in the browser. all-MiniLM-L6-v2 trains 6 passes
over the situations per generation (about 0.5–1 s); 16 passes did not learn better but took
three times as long.

Laya on the CPU instead of the GPU: measured 198 ms per decision versus 25 ms, and 13 s versus
0.16 s for 100 decisions at once. It works, but training between generations takes minutes
instead of seconds.

## Files

- `play.py` – starts the race.
- `engine.py` – the server: the models on the GPU, GPU status and the pages.
- `laya_model.py` – Laya's model, loaded from the Hugging Face cache (from convaiinnovations/laya, Apache-2.0).
- `race_text.py` – the situation text and the maneuvers.
- `race_train.py` – training between generations, shared by the drivers.
- `race_laya.py`, `race_bert.py` – the drivers Laya and all-MiniLM-L6-v2.
- `race.html` – the race: the simulation, the tracks and the dashboard; `theme.css` – the
  design; `gpu.js` – GPU status and gauges.
- `memory/` – what the drivers have learned (created at runtime, not in the repo).
