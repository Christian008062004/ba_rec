#!/usr/bin/env python3
"""
Re-Ranker Training Script

Usage:
    python reranker/train.py
    python reranker/train.py --gpu 0
    python reranker/train.py --max_samples 10000   # Smoke-Test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, random_split

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from model import SequenceAwareReranker, CATEGORICAL_FEATURES, NUMERIC_FEATURES

CONFIG_FILE  = Path(__file__).parent / "config.yaml"
CKPT_DIR     = Path(__file__).parent / "checkpoints"


def load_dataset(path, max_samples=None):
    print(f"  Lade Datensatz: {path}")
    data = np.load(path)

    n = len(data["label"])
    if max_samples:
        idx = np.random.default_rng(2024).choice(n, size=min(max_samples, n), replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    tensors = {}
    tensors["sasrec_emb"] = torch.FloatTensor(data["sasrec_emb"][idx])
    tensors["label"]      = torch.FloatTensor(data["label"][idx])

    for name in CATEGORICAL_FEATURES:
        if name in data:
            arr = np.clip(data[name][idx], 0, CATEGORICAL_FEATURES[name] - 1)
            tensors[name] = torch.LongTensor(arr)

    for name in NUMERIC_FEATURES:
        if name in data:
            tensors[name] = torch.FloatTensor(data[name][idx])

    print(f"  {len(idx):,} Samples geladen  "
          f"(pos: {int(tensors['label'].sum()):,} / neg: {int((1-tensors['label']).sum()):,})")
    return tensors


def make_loader(tensors, indices, batch_size, shuffle):
    subset = {k: v[indices] for k, v in tensors.items()}
    keys   = list(subset.keys())
    ds     = TensorDataset(*[subset[k] for k in keys])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, keys


def train(args):
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0
                          and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_path = REPO_DIR / cfg["dataset_path"]
    if not dataset_path.exists():
        print(f"Datensatz nicht gefunden: {dataset_path}")
        print("Bitte zuerst:  python reranker/build_dataset.py")
        sys.exit(1)

    tensors = load_dataset(dataset_path, args.max_samples)

    # Train/Val Split (90/10)
    n = len(tensors["label"])
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(n)
    split = int(0.9 * n)
    train_idx = torch.LongTensor(idx[:split])
    val_idx   = torch.LongTensor(idx[split:])

    train_loader, keys = make_loader(tensors, train_idx, cfg["batch_size"], shuffle=True)
    val_loader,   _    = make_loader(tensors, val_idx,   cfg["batch_size"], shuffle=False)

    label_key_idx = keys.index("label")

    model = SequenceAwareReranker(
        sasrec_dim=cfg["sasrec_dim"],
        embedding_dim=cfg["embedding_dim"],
        mlp_hidden_units=cfg["mlp_hidden_units"],
        dropout=cfg["dropout"],
        batch_norm=cfg["batch_norm"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Modell-Parameter: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.BCELoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_auc    = 0.0
    patience_ct = 0

    print(f"\nStarte Training  (epochs={cfg['epochs']}, batch={cfg['batch_size']}) ...\n")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for batch_tensors in train_loader:
            batch = {k: batch_tensors[i].to(device) for i, k in enumerate(keys)}
            labels = batch.pop("label")

            preds = model(batch)
            loss  = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_tensors in val_loader:
                batch  = {k: batch_tensors[i].to(device) for i, k in enumerate(keys)}
                labels = batch.pop("label")
                preds  = model(batch)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        print(f"  Epoch {epoch:>2}  |  Loss: {avg_loss:.4f}  |  Val-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            patience_ct = 0
            ckpt_path = CKPT_DIR / "reranker.pt"
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_auc":    auc,
                "config":     cfg,
            }, ckpt_path)
            print(f"           Bestes Modell gespeichert (AUC={auc:.4f})")
        else:
            patience_ct += 1
            if patience_ct >= cfg["patience"]:
                print(f"\nEarly Stopping nach Epoch {epoch}.")
                break

    print(f"\nBestes Val-AUC: {best_auc:.4f}  |  Checkpoint: {CKPT_DIR}/reranker.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",         type=int, default=-1,
                        help="GPU-Index (-1 = CPU)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max. Samples fuer Smoke-Test (default: alle)")
    args = parser.parse_args()
    train(args)
