#!/usr/bin/env python3
"""
Baut den Trainingsdatensatz für den Re-Ranker.

Für jede Interaktion in webshop.inter:
  - Sequenz = alle vorherigen Items des Nutzers
  - SASRec-Embedding der Sequenz extrahieren (64-dim)
  - Positives: das tatsächlich geklickte Item
  - Negatives: zufällig gesampelte Items

Output: reranker/data/reranker_dataset.npz

Usage:
    python reranker/build_dataset.py
    python reranker/build_dataset.py --max_users 500 --neg_samples 4
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore")

REPO_DIR    = Path(__file__).parent.parent
DATASET_DIR = REPO_DIR / "dataset"
INTER_FILE  = REPO_DIR / "retrieval" / "data" / "webshop" / "webshop.inter"
VOCAB_FILE  = DATASET_DIR / "feature_vocab.json"
SASREC_CKPT = REPO_DIR / "trained" / "SASRec-May-25-2026_22-31-38.pth"
SASREC_VOCAB = REPO_DIR / "trained" / "sasrec_item_vocab.json"
OUT_DIR     = Path(__file__).parent / "data"
sys.path.insert(0, str(REPO_DIR / "training" / "models" / "AFN"))


def load_sasrec():
    from recbole.model.sequential_recommender import SASRec

    with open(SASREC_VOCAB) as f:
        sv = json.load(f)

    class _MinimalDataset:
        def num(self, field):
            return sv["n_items"]

    ckpt = torch.load(SASREC_CKPT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    config["device"] = torch.device("cpu")

    model = SASRec(config, _MinimalDataset()).to("cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, sv, config


def get_sequence_embedding(model, rb_seq, max_len=50):
    """Gibt das SASRec-Sequenz-Embedding (hidden state) für eine Sequenz zurück."""
    rb_seq = rb_seq[-max_len:]
    seq_tensor = torch.tensor(rb_seq, dtype=torch.long).unsqueeze(0)
    len_tensor = torch.tensor([len(rb_seq)], dtype=torch.long)
    with torch.no_grad():
        emb = model.forward(seq_tensor, len_tensor)  # [1, hidden_size]
    return emb.squeeze(0).numpy()


def build(max_users, neg_samples, batch_size):
    print("[1/4] Lade Daten ...")
    with open(VOCAB_FILE) as f:
        vocab = json.load(f)

    item_vocab_afn = vocab["item_id"]   # token → afn_id
    user_vocab_afn = vocab["user_id"]   # token → afn_id

    with open(REPO_DIR / "trained" / "item_features.json") as f:
        raw = json.load(f)
    item_features = {int(k): v for k, v in raw.items()}

    with open(REPO_DIR / "trained" / "user_context.json") as f:
        raw = json.load(f)
    user_ctx = {int(k): v for k, v in raw.items()}

    df = pd.read_csv(INTER_FILE, sep="\t", dtype=str)
    df["timestamp"] = df["timestamp:float"].astype(float)
    df = df.rename(columns={"user_id:token": "user_id", "item_id:token": "item_id"})

    counts = df.groupby("user_id")["item_id"].transform("count")
    df = df[counts >= 3].copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    all_items_afn = np.array(list(item_vocab_afn.values()), dtype=np.int64)
    all_items_tokens = np.array(list(item_vocab_afn.keys()))

    print("[2/4] Lade SASRec ...")
    sasrec, sv, config = load_sasrec()
    token2id = sv["token2id"]
    id2token = sv["id2token"]
    max_len  = sv["MAX_ITEM_LIST_LENGTH"]

    print("[3/4] Extrahiere Embeddings und baue Datensatz ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx_keys = ["geo_country", "geo_latitude", "geo_longitude",
                "device_type", "device_os", "device_model",
                "device_manufacturer", "device_browser", "device_type_detail"]
    feat_keys = ["brand", "category1", "category2", "category3", "category4", "price"]

    sasrec_dim = config["hidden_size"]

    # Akkumulatoren
    rows_emb      = []
    rows_item_id  = []
    rows_label    = []
    rows_user_id  = []
    rows_ctx      = {k: [] for k in ctx_keys}
    rows_feat     = {k: [] for k in feat_keys}

    rng = np.random.default_rng(2024)
    n_users_done = 0

    for user_token, grp in df.groupby("user_id", sort=False):
        items_tokens = grp["item_id"].tolist()
        if len(items_tokens) < 3:
            continue

        afn_uid = user_vocab_afn.get(user_token, 0)
        ctx = user_ctx.get(afn_uid, {k: 0 for k in ctx_keys})

        # Sliding window: für jede Position ab Index 1
        rb_seq = []
        for i, item_token in enumerate(items_tokens):
            rb_id = token2id.get(item_token, 0)

            if i > 0 and rb_seq:  # ab der zweiten Interaktion
                afn_iid_pos = item_vocab_afn.get(item_token, 0)
                if afn_iid_pos > 0:
                    emb = get_sequence_embedding(sasrec, rb_seq, max_len)

                    # Positives
                    rows_emb.append(emb)
                    rows_item_id.append(afn_iid_pos)
                    rows_label.append(1)
                    rows_user_id.append(afn_uid)
                    for k in ctx_keys:
                        rows_ctx[k].append(ctx.get(k, 0))
                    feat = item_features.get(afn_iid_pos, {})
                    for k in feat_keys:
                        rows_feat[k].append(feat.get(k, 0))

                    # Negatives: zufällig aus allen Items außer der aktuellen Sequenz
                    seen_afn = set(
                        item_vocab_afn.get(t, 0)
                        for t in items_tokens[:i+1]
                        if item_vocab_afn.get(t, 0) > 0
                    )
                    neg_pool = all_items_afn[~np.isin(all_items_afn, list(seen_afn))]
                    neg_sample = rng.choice(neg_pool, size=min(neg_samples, len(neg_pool)),
                                            replace=False)
                    for neg_id in neg_sample:
                        rows_emb.append(emb)
                        rows_item_id.append(int(neg_id))
                        rows_label.append(0)
                        rows_user_id.append(afn_uid)
                        for k in ctx_keys:
                            rows_ctx[k].append(ctx.get(k, 0))
                        feat = item_features.get(int(neg_id), {})
                        for k in feat_keys:
                            rows_feat[k].append(feat.get(k, 0))

            if rb_id > 0:
                rb_seq.append(rb_id)

        n_users_done += 1
        if n_users_done % 500 == 0:
            print(f"  {n_users_done:,} / {max_users or '?'} Nutzer  |  "
                  f"{len(rows_label):,} Samples bisher")

        if max_users and n_users_done >= max_users:
            break

    print(f"\n[4/4] Speichere Datensatz ({len(rows_label):,} Samples) ...")
    data = {
        "sasrec_emb":   np.array(rows_emb,     dtype=np.float32),
        "item_id":      np.array(rows_item_id,  dtype=np.int64),
        "user_id":      np.array(rows_user_id,  dtype=np.int64),
        "label":        np.array(rows_label,    dtype=np.float32),
    }
    for k in ctx_keys:
        data[k] = np.array(rows_ctx[k], dtype=np.float32
                           if k in {"geo_latitude", "geo_longitude"} else np.int64)
    for k in feat_keys:
        data[k] = np.array(rows_feat[k], dtype=np.float32
                           if k == "price" else np.int64)

    out_path = OUT_DIR / "reranker_dataset.npz"
    np.savez_compressed(out_path, **data)
    print(f"Gespeichert: {out_path}")
    print(f"  Positives: {int(np.sum(data['label'])):,}")
    print(f"  Negatives: {int(np.sum(1 - data['label'])):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users",   type=int, default=None,
                        help="Nur erste N Nutzer verarbeiten (default: alle)")
    parser.add_argument("--neg_samples", type=int, default=4,
                        help="Negative Samples pro Interaktion (default: 4)")
    parser.add_argument("--batch_size",  type=int, default=256,
                        help="Batch-Größe für SASRec (default: 256)")
    args = parser.parse_args()
    build(args.max_users, args.neg_samples, args.batch_size)
