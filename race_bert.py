"""En väldigt liten BERT kör racet: all-MiniLM-L6-v2, 6 lager och 22 miljoner parametrar.

Den kan ingenting om bilkörning från början. Ett nytt beslutslager väljer manöver ur
medelvärdet av BERT:ens utdata för lägestexten, och hela modellen tränas mellan generationerna.
"""
import torch
import torch.nn as nn

from race_text import OPTIONS, describe
from race_train import TrainedRacer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class TinyBertDriver(TrainedRacer):
    LEARNING_RATE = 1e-4   # BERT:en
    EPOCHS = 6             # hela modellen tränas, så 6 varv lär lika bra som 16 men går tre gånger fortare
    HEAD_RATE = 1e-3       # det nya beslutslagret lär sig snabbare

    def __init__(self, memory_file):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).cuda().eval()  # eval: ingen dropout
        self.head = nn.Linear(self.encoder.config.hidden_size, len(OPTIONS)).cuda()
        with torch.no_grad():  # små vikter: från början är alla manövrer lika sannolika
            self.head.weight.normal_(0, 0.01)
            self.head.bias.zero_()
        self.trainable = {**{"encoder." + n: p for n, p in self.encoder.named_parameters()},
                          **{"head." + n: p for n, p in self.head.named_parameters()}}
        self._setup(memory_file)

    def _param_groups(self):
        return [{"params": list(self.encoder.parameters()), "lr": self.LEARNING_RATE},
                {"params": list(self.head.parameters()), "lr": self.HEAD_RATE}]

    def _logits(self, keys):
        batch = self.tokenizer([describe(key) for key in keys], padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1)
        return self.head(pooled), int(batch["attention_mask"].sum())
