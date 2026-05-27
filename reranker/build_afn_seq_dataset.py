#!/usr/bin/env python3
"""
Erweitert die bestehenden AFN-Trainingsdaten um SASRec-Sequenz-Embeddings.

Fuer jeden User wird das SASRec-Embedding seiner vollen Interaktionshistorie
berechnet und als 64 neue numerische Features (seq_emb_0..63) hinzugefuegt.

Output:
  dataset/train_seq.npz
  dataset/valid_seq.npz
  dataset/test_seq.npz
  dataset/webshop_data/feature_map_seq.json

Usage:
    python reranker/build_afn_seq_dataset.py
    python reranker/build_afn_seq_dataset.py --max_users 500   # Smoke-Test
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

REPO_DIR     = Path(__file__).parent.parent
DATASET_DIR  = REPO_DIR / "dataset"
INTER_FILE   = REPO_DIR / "retrieval" / "data" / "webshop" / "webshop.inter"
SASREC_CKPT  = REPO_DIR / "trained" / "SASRec-May-25-2026_22-31-38.pth"
SASREC_VOCAB = REPO_DIR / "trained" / "sasrec_item_vocab.json"
VOCAB_FILE   = DATASET_DIR / "feature_vocab.json"

SEQ_DIM = 64   # SASRec hidden_size


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
    return model, sv


def compute_user_embeddings(sasrec, sv, max_users=None):
    """Berechnet SASRec-Embedding fuer jeden User (volle Sequenz)."""
    print("  Lade Interaktionen ...")
    df = pd.read_csv(INTER_FILE, sep="\t", dtype=str)
    df["timestamp"] = df["timestamp:float"].astype(float)
    df = df.rename(columns={"user_id:token": "user_id", "item_id:token": "item_id"})
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    with open(VOCAB_FILE) as f:
        vocab = json.load(f)
    user_vocab = vocab["user_id"]   # raw_token -> afn_id
    token2id   = sv["token2id"]     # item_token -> recbole_id
    max_len    = sv["MAX_ITEM_LIST_LENGTH"]

    user_embeddings = {}   # afn_user_id -> np.array(64,)
    n_done = 0

    for user_token, grp in df.groupby("user_id", sort=False):
        afn_uid = user_vocab.get(user_token, 0)
        if afn_uid == 0:
            continue

        rb_seq = [token2id[tok] for tok in grp["item_id"].tolist()
                  if tok in token2id and token2id[tok] > 0]
        if not rb_seq:
            continue

        rb_seq = rb_seq[-max_len:]
        seq_t  = torch.tensor(rb_seq, dtype=torch.long).unsqueeze(0)
        len_t  = torch.tensor([len(rb_seq)], dtype=torch.long)

        with torch.no_grad():
            emb = sasrec.forward(seq_t, len_t).squeeze(0).numpy()

        user_embeddings[afn_uid] = emb
        n_done += 1

        if n_done % 10000 == 0:
            print(f"    {n_done:,} User verarbeitet ...")

        if max_users and n_done >= max_users:
            break

    print(f"  {n_done:,} User-Embeddings berechnet.")
    return user_embeddings


def extend_split(split_name, user_embeddings):
    """Fuegt SASRec-Embeddings zu einem Split hinzu und speichert ihn."""
    src = DATASET_DIR / f"{split_name}.csv.npz"
    dst = DATASET_DIR / f"{split_name}_seq.npz"

    data     = np.load(src)
    user_ids = data["user_id"]   # AFN-encoded
    n        = len(user_ids)

    # Embedding-Matrix aufbauen: default = Nullvektor fuer unbekannte User
    emb_matrix = np.zeros((n, SEQ_DIM), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        if int(uid) in user_embeddings:
            emb_matrix[i] = user_embeddings[int(uid)]

    new_data = dict(data)
    for d in range(SEQ_DIM):
        new_data[f"seq_emb_{d}"] = emb_matrix[:, d]

    np.savez_compressed(dst, **new_data)
    print(f"  {split_name}_seq.npz gespeichert  ({n:,} Zeilen, {SEQ_DIM} neue Features)")


def update_feature_map():
    """Kopiert feature_map.json und fuegt die 64 neuen numerischen Features hinzu."""
    src     = DATASET_DIR / "webshop_data" / "feature_map.json"
    dst_dir = DATASET_DIR / "webshop_data_seq"
    dst     = dst_dir / "feature_map.json"
    dst_dir.mkdir(exist_ok=True)

    with open(src) as f:
        fm = json.load(f)

    fm["dataset_id"] = "webshop_data_seq"

    new_features = []
    for d in range(SEQ_DIM):
        new_features.append({f"seq_emb_{d}": {
            "active": True,
            "dtype":  "float",
            "type":   "numeric",
            "source": "user",
        }})

    fm["features"].extend(new_features)
    fm["total_features"] = fm.get("total_features", 0) + SEQ_DIM

    with open(dst, "w") as f:
        json.dump(fm, f, indent=2)
    print(f"  webshop_data_seq/feature_map.json gespeichert  (+{SEQ_DIM} Features)")


def build(max_users):
    print("[1/4] Lade SASRec ...")
    sasrec, sv = load_sasrec()

    print("[2/4] Berechne User-Embeddings ...")
    user_embeddings = compute_user_embeddings(sasrec, sv, max_users)

    print("[3/4] Erweitere Splits ...")
    for split in ["train", "valid", "test"]:
        extend_split(split, user_embeddings)

    print("[4/4] Aktualisiere Feature-Map ...")
    update_feature_map()

    print("\nFertig. Naechster Schritt:")
    print("  python training/train/train.py --model AFN --dataset dataset_seq")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=None,
                        help="Nur N User verarbeiten (Smoke-Test)")
    args = parser.parse_args()
    build(args.max_users)
