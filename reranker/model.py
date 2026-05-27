"""
Sequence-Aware Re-Ranker

Kombiniert SASRec-Sequenz-Embedding mit Item-Features und User-Kontext.
Input:  sasrec_emb (64-dim) + kategorische/numerische Features
Output: Klick-Wahrscheinlichkeit (0-1)
"""

import torch
import torch.nn as nn


CATEGORICAL_FEATURES = {
    "item_id":             8432,
    "brand":                456,
    "category1":            333,
    "category2":            332,
    "category3":            292,
    "category4":            291,
    "geo_country":          100,
    "device_type":            5,
    "device_os":             11,
    "device_model":         824,
    "device_manufacturer":   64,
    "device_browser":        81,
    "device_type_detail":     6,
    "user_id":           360256,
}

NUMERIC_FEATURES = ["price", "geo_latitude", "geo_longitude"]


class SequenceAwareReranker(nn.Module):
    def __init__(self, sasrec_dim=64, embedding_dim=16,
                 mlp_hidden_units=None, dropout=0.1, batch_norm=True):
        super().__init__()
        if mlp_hidden_units is None:
            mlp_hidden_units = [256, 128, 64]

        # Embeddings für kategorische Features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            for name, vocab_size in CATEGORICAL_FEATURES.items()
        })

        # Input-Dimension: SASRec-Emb + alle Cat-Embeddings + numerische Features
        cat_dim = len(CATEGORICAL_FEATURES) * embedding_dim
        num_dim = len(NUMERIC_FEATURES)
        input_dim = sasrec_dim + cat_dim + num_dim

        # MLP
        layers = []
        prev_dim = input_dim
        for hidden in mlp_hidden_units:
            layers.append(nn.Linear(prev_dim, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch):
        parts = [batch["sasrec_emb"]]

        for name in CATEGORICAL_FEATURES:
            if name in batch:
                emb = self.embeddings[name](batch[name])
                parts.append(emb)

        for name in NUMERIC_FEATURES:
            if name in batch:
                parts.append(batch[name].unsqueeze(-1))

        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)
