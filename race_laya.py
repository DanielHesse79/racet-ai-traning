"""Laya kör racet: bilarna beskriver vad de ser i text, Laya väljer manöver,
och Layas beslutshuvud tränas mellan generationerna på hur det gick.

Kodaren (ModernBERT-large) är fryst, så dess utdata för ett läge kan sparas och återanvändas.
Bara beslutshuvudet (två transformer-lager och värderingen av alternativen) tränas.
"""
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch

import laya_model
from race_text import OPTIONS, QUESTION, describe
from race_train import TrainedRacer

MAX_CACHED = 6000  # lägen vars kodarutdata sparas på grafikkortet, runt 1,1 GB


class LayaDriver(TrainedRacer):
    LEARNING_RATE = 3e-4

    def __init__(self, memory_file):
        self.model, self.tokenizer, config = laya_model.load()
        self.model.requires_grad_(False)
        self.trainable = {name: p for name, p in self.model.named_parameters()
                          if name.startswith(("head.", "scorer.", "type_emb."))}
        for p in self.trainable.values():
            p.requires_grad_(True)
        self.question, self.markers = laya_model.question_ids(
            self.tokenizer, QUESTION, [f"{name}: {text}" for name, text in OPTIONS.items()])
        self.temperatures = config["temperature_by_options"]
        self.temperature = self.temperatures["choice:6-10"]  # racets sex manövrer
        self.max_len = config["max_len"]
        self.hidden = OrderedDict()  # läge -> kodarens utdata på grafikkortet, senast använda sist
        self._setup(memory_file)

    @contextmanager
    def _original_head(self):
        """Layas ursprungliga huvud, för spelen – inte det som racet har tränat."""
        trained = {name: p.detach().clone() for name, p in self.trainable.items()}
        self._set_weights(self.initial)
        try:
            yield
        finally:
            self._set_weights(trained)

    def choose(self, state, question, options):
        """En fråga med egna alternativ. Returnerar ({alternativ: sannolikhet}, tokenanvändning)."""
        started = time.perf_counter()
        ids, markers = laya_model.question_ids(self.tokenizer, question, options)
        room = self.max_len - len(ids) - 1
        ids = ids + self.tokenizer(state, add_special_tokens=False)["input_ids"][:room] + [self.tokenizer.sep_token_id]
        ids, mask = self._pad([torch.tensor(ids)], self.tokenizer.pad_token_id)
        marker_pos = torch.tensor([markers], device="cuda")
        bucket = "choice:2" if len(options) <= 2 else "choice:3-5" if len(options) <= 5 else "choice:6-10"
        with self._original_head(), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model.decide(self.model.encode(ids, mask), mask, marker_pos,
                                       torch.ones_like(marker_pos, dtype=torch.bool))
        probs = torch.softmax(logits[0].float() / self.temperatures[bucket], -1).tolist()
        torch.cuda.synchronize()
        tokens = int(mask.sum())
        gpu_ms = round((time.perf_counter() - started) * 1000, 1)
        return dict(zip(options, probs)), {"tokens": tokens, "computed": tokens, "gpu_ms": gpu_ms}

    def _prepare(self, keys):
        """Kör den frysta kodaren för lägen den inte har sett. Returnerar antal tokens."""
        missing = [key for key in dict.fromkeys(keys) if key not in self.hidden]
        tokens = 0
        for start in range(0, len(missing), 128):
            chunk = missing[start:start + 128]
            rows = [self.question + self.tokenizer(describe(key), add_special_tokens=False)["input_ids"]
                    + [self.tokenizer.sep_token_id] for key in chunk]
            ids, mask = self._pad([torch.tensor(r) for r in rows], self.tokenizer.pad_token_id)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.model.encode(ids, mask)
            for i, key in enumerate(chunk):
                self.hidden[key] = hidden[i, :len(rows[i])].to(torch.bfloat16).clone()
            tokens += int(mask.sum())
        for key in keys:
            self.hidden.move_to_end(key)
        while len(self.hidden) > max(MAX_CACHED, len(keys)):  # släpp de lägen som använts längst bak i tiden
            self.hidden.popitem(last=False)
        return tokens

    def _logits(self, keys):
        hidden, mask = self._pad([self.hidden[key] for key in keys], 0)
        markers = torch.tensor([self.markers] * len(keys), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model.decide(hidden, mask, markers, torch.ones_like(markers, dtype=torch.bool))
        return logits / self.temperature, int(mask.sum())
