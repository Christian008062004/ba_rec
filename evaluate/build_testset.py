#!/usr/bin/env python3
"""
Baut den Benchmark-Testdatensatz für die zweistufige Pipeline.

Logik (identisch zum RecBole leave-one-out-Split von SASRec):
  - Nutzer mit >= 3 Interaktionen
  - Sortiert nach Timestamp
  - Letzte Interaktion = Ground Truth (was der Nutzer wirklich geklickt hat)
  - Alle vorherigen = Eingabe-Sequenz für SASRec

Output: evaluate/testset.csv
  Spalten: user_id | sequence (Pipe-getrennt, ältestes Item zuerst) | ground_truth

Usage:
    python evaluate/build_testset.py
    python evaluate/build_testset.py --min_seq_len 2 --output evaluate/testset.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_DIR      = Path(__file__).parent.parent
INTER_FILE    = REPO_DIR / "retrieval" / "data" / "webshop" / "webshop.inter"
VOCAB_FILE    = REPO_DIR / "dataset" / "feature_vocab.json"
DEFAULT_OUT   = Path(__file__).parent / "testset.csv"

MIN_INTERACTIONS = 3   # same as SASRec config: user_inter_num_interval "[3,inf)"


def build(min_seq_len: int, output: Path):
    print("Lade Interaktionen ...")
    df = pd.read_csv(INTER_FILE, sep="\t", dtype=str)
    df["timestamp"] = df["timestamp:float"].astype(float)
    df = df.rename(columns={"user_id:token": "user_id", "item_id:token": "item_id"})
    df = df[["user_id", "item_id", "timestamp"]]

    print(f"  Gesamt: {len(df):,} Interaktionen, {df['user_id'].nunique():,} Nutzer")

    # ── Nutzer mit >= MIN_INTERACTIONS behalten ──────────────────────────────
    counts = df.groupby("user_id")["item_id"].transform("count")
    df = df[counts >= MIN_INTERACTIONS].copy()
    print(f"  Nach Nutzer-Filter (>={MIN_INTERACTIONS}): {len(df):,} Interaktionen,"
          f" {df['user_id'].nunique():,} Nutzer")

    # ── Pro Nutzer nach Timestamp sortieren ──────────────────────────────────
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # ── Leave-one-out: letztes Item = Ground Truth, Rest = Sequenz ───────────
    print("Baue Testsamples ...")
    records = []
    for user_id, grp in df.groupby("user_id", sort=False):
        items = grp["item_id"].tolist()

        ground_truth = items[-1]
        sequence     = items[:-1]          # alles vor dem letzten Item

        # Mindestens min_seq_len Items in der Sequenz
        if len(sequence) < min_seq_len:
            continue

        records.append({
            "user_id":      user_id,
            "sequence":     "|".join(sequence),   # Pipe-getrennt, ältestes zuerst
            "ground_truth": ground_truth,
            "seq_len":      len(sequence),
        })

    testset = pd.DataFrame(records)
    print(f"  Testsamples: {len(testset):,}")
    print(f"  Sequenzlänge — min: {testset['seq_len'].min()},"
          f" median: {testset['seq_len'].median():.0f},"
          f" max: {testset['seq_len'].max()}")

    # ── Validierung gegen AFN-Vokabular ──────────────────────────────────────
    with open(VOCAB_FILE) as f:
        vocab = json.load(f)
    item_vocab = set(vocab["item_id"].keys())
    user_vocab = set(vocab["user_id"].keys())

    n_total = len(testset)
    testset = testset[testset["user_id"].isin(user_vocab)]
    testset = testset[testset["ground_truth"].isin(item_vocab)]
    print(f"  Nach Vokabular-Filter: {len(testset):,} / {n_total:,} Samples behalten")

    # ── Speichern ─────────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    testset[["user_id", "sequence", "ground_truth"]].to_csv(output, index=False)
    print(f"\nTestdatensatz gespeichert: {output}  ({len(testset):,} Zeilen)")
    print("\nBeispiel:")
    row = testset.iloc[0]
    seq_preview = "|".join(row["sequence"].split("|")[-3:])   # letzte 3 Items
    print(f"  user_id:      {row['user_id']}")
    print(f"  sequence:     ...{seq_preview}  ({row['sequence'].count('|') + 1} Items)")
    print(f"  ground_truth: {row['ground_truth']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_seq_len", type=int, default=2,
                        help="Mindest-Sequenzlänge (ohne Ground Truth, default: 2)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help=f"Ausgabepfad (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    build(args.min_seq_len, args.output)


if __name__ == "__main__":
    main()
