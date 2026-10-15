"""Det förarna i racet har gemensamt: sannolikheter per läge, och träningen mellan generationerna.

En förare (Laya, den lilla BERT:en) ärver TrainedRacer och anger:
  self.trainable  – namn -> parameter som tränas
  _logits(keys)   – modellens värden för varje manöver, plus antal tokens
  _prepare(keys)  – valfritt: förarbete för nya lägen, returnerar antal tokens
"""
import random
import time

import torch

import race_text

BATCH = 256            # lägen per gradientsteg
IMITATION = 1.0        # hur mycket de bästa bilarnas val väger
ENTROPY = 0.03         # håller kvar lite osäkerhet, så att bilarna fortsätter prova
MAX_STATES = 3000      # högst så många lägen per träning, även med 10 000 bilar


class TrainedRacer:
    trains = True
    EPOCHS = 16  # varv över alla lägen per träning; en förare kan välja färre

    def _setup(self, memory_file):
        """Anropas av underklassen när self.trainable finns."""
        self.initial = {name: p.detach().clone() for name, p in self.trainable.items()}
        self.file = memory_file
        self.updates = 0
        self.optimizer = torch.optim.AdamW(self._param_groups())
        if self.file.exists():
            saved = torch.load(self.file, map_location="cuda")
            self._set_weights(saved["weights"])
            self.updates = saved["updates"]
        self.version = 0  # ökar när modellen ändras, så att webbläsaren vet att svaren är gamla

    def _param_groups(self):
        return [{"params": list(self.trainable.values()), "lr": self.LEARNING_RATE}]

    def _prepare(self, keys):
        return 0

    @staticmethod
    def vocabulary():
        return race_text.vocabulary()

    @staticmethod
    def vram():
        return torch.cuda.memory_reserved()

    def _set_weights(self, weights):
        with torch.no_grad():
            for name, p in self.trainable.items():
                p.copy_(weights[name])

    @staticmethod
    def _pad(rows, value):
        length = max(len(r) for r in rows)
        out = torch.full((len(rows), length, *rows[0].shape[1:]), value, dtype=rows[0].dtype, device="cuda")
        mask = torch.zeros((len(rows), length), dtype=torch.long, device="cuda")
        for i, r in enumerate(rows):
            out[i, :len(r)] = r
            mask[i, :len(r)] = 1
        return out, mask

    def decide(self, keys):
        """Sannolikheterna för varje manöver, per läge."""
        started = time.perf_counter()
        tokens = self._prepare(keys)
        probs = []
        with torch.no_grad():
            for start in range(0, len(keys), BATCH):
                logits, used = self._logits(keys[start:start + BATCH])
                probs += torch.softmax(logits.float(), -1).tolist()
                tokens += used
        torch.cuda.synchronize()
        return {"probs": probs, "version": self.version,
                "usage": {"tokens": tokens, "gpu_ms": round((time.perf_counter() - started) * 1000, 1)}}

    def learn(self, samples, best):
        """samples: [läge, manöver, summa belöning, antal]. best: [läge, manöver, antal] från de bästa bilarna."""
        started = time.perf_counter()
        # Jämför varje manöver med snittet för samma läge: bättre än snittet förstärks, sämre försvagas.
        per_key = {}
        for key, action, total, count in samples:
            per_key.setdefault(key, [0.0, 0])
            per_key[key][0] += total
            per_key[key][1] += count
        rows = []  # (läge, manöver, fördel, vikt)
        for key, action, total, count in samples:
            baseline = per_key[key][0] / per_key[key][1]
            rows.append((key, action, total / count - baseline, count / (count + 2)))
        spread = (sum(a * a for _, _, a, _ in rows) / max(1, len(rows))) ** 0.5 or 1.0
        imitate = {(key, action): count for key, action, count in best}
        keys = list(dict.fromkeys([k for k, _ in imitate] + [r[0] for r in rows]))
        if len(keys) > MAX_STATES:  # behåll lägena som ska härmas, och ett slumpat urval av resten
            copied = len(dict.fromkeys(k for k, _ in imitate))
            keys = keys[:min(copied, MAX_STATES)] + random.sample(keys[copied:], max(0, MAX_STATES - copied))
            kept = set(keys)
            rows = [r for r in rows if r[0] in kept]
            imitate = {ka: c for ka, c in imitate.items() if ka[0] in kept}
        tokens = self._prepare(keys)
        losses = []
        for _ in range(self.EPOCHS):
            order = torch.randperm(len(keys)).tolist()
            for start in range(0, len(keys), BATCH):
                batch = [keys[i] for i in order[start:start + BATCH]]
                where = {key: i for i, key in enumerate(batch)}
                logits, used = self._logits(batch)
                tokens += 3 * used  # framåt och bakåt
                logp = torch.log_softmax(logits.float(), -1)
                chosen = [(where[k], a, adv / spread, w) for k, a, adv, w in rows if k in where]
                loss = logp.new_zeros(())
                if chosen:
                    i, a, adv, w = (torch.tensor(x, device="cuda") for x in zip(*chosen))
                    loss = loss - (w * adv * logp[i, a]).sum() / w.sum()
                copied = [(where[k], a, c) for (k, a), c in imitate.items() if k in where]
                if copied:
                    i, a, c = (torch.tensor(x, device="cuda") for x in zip(*copied))
                    loss = loss - IMITATION * (c * logp[i, a]).sum() / c.sum()
                loss = loss + ENTROPY * (logp.exp() * logp).sum(-1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable.values(), 1.0)
                self.optimizer.step()
                losses.append(float(loss))
        self.updates += 1
        self.version += 1
        self.save()
        # Svara direkt på lägena modellen just tränats på, så att nästa generation börjar med dem i
        # minnet i stället för att stanna och fråga om varje läge igen.
        answers = self.decide(keys)
        return {"version": self.version, "updates": self.updates, "states": len(keys), "steps": len(losses),
                "loss": round(sum(losses) / max(1, len(losses)), 4), "keys": keys, "probs": answers["probs"],
                "usage": {"tokens": tokens + answers["usage"]["tokens"],
                          "gpu_ms": round((time.perf_counter() - started) * 1000, 1)}}

    def save(self):
        self.file.parent.mkdir(exist_ok=True)
        torch.save({"weights": {n: p.detach() for n, p in self.trainable.items()}, "updates": self.updates}, self.file)

    def reset(self):
        self._set_weights(self.initial)
        self.optimizer = torch.optim.AdamW(self._param_groups())
        self.updates = 0
        self.version += 1
        self.file.unlink(missing_ok=True)
        return {"version": self.version, "updates": 0}
