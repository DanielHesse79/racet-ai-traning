"""Laya – beslutsmodellen som kör racet – laddad direkt från Hugging Face-cachen.

Modellen och hur en fråga byggs ihop kommer från convaiinnovations/laya (rl_common.py,
Apache-2.0), nedkortat till det racet behöver. Kräver PyTorch med CUDA och transformers.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

LAYA_REPO = "convaiinnovations/laya"


def find_checkpoint():
    """Mappen med Layas engelska checkpoint i Hugging Face-cachen."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(LAYA_REPO, allow_patterns=["encoder/*", "tokenizer/*", "model.safetensors",
                                                            "rl_agent_config.json"], local_files_only=True))


class DecisionModel(nn.Module):
    """ModernBERT-large som kodare plus Layas beslutshuvud.

    Varje alternativ får en [MASK]-markör, och huvudet ger varje markör ett värde.
    Softmax över värdena är modellens sannolikheter för alternativen.
    """

    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))  # används inte här
        self.register_buffer("temperature", torch.ones(3))

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def decide(self, hidden, attention_mask, marker_pos, marker_mask):
        """Huvudet ovanpå kodarens utdata. Det är den här delen racet tränar."""
        h = hidden + self.type_emb(torch.zeros_like(attention_mask[:, 0]))[:, None, :]  # 0 = choice
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        index = marker_pos[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, index)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def load(device="cuda"):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    folder = find_checkpoint()
    config = json.loads((folder / "rl_agent_config.json").read_text())
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(folder / "encoder"), attn_implementation="sdpa")
    model = DecisionModel(encoder, config["head_layers"], len(config["act_costs"]) + 1)
    model.load_state_dict(load_file(folder / "model.safetensors"), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")
    return model.to(device).eval(), tokenizer, config


def question_ids(tokenizer, instructions, options):
    """[CLS] choice question: <fråga> [SEP] [MASK] alt 1 [MASK] alt 2 ... [SEP] – samma för alla bilar."""
    ids = [tokenizer.cls_token_id] + tokenizer(f"choice question: {instructions}", add_special_tokens=False)["input_ids"]
    ids.append(tokenizer.sep_token_id)
    markers = []
    for option in options:
        markers.append(len(ids))
        ids += [tokenizer.mask_token_id] + tokenizer(" " + option, add_special_tokens=False)["input_ids"]
    ids.append(tokenizer.sep_token_id)
    return ids, markers
