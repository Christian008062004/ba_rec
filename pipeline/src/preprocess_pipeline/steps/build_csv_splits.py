#!/usr/bin/env python3
"""Schritt 6: Staging → label-encoded train/valid/test CSVs + item_embedding_matrix.npz

Liest staging/.inter, .item, .user, joint sie, label-encodiert kategorische Features,
normalisiert numerische Features (Min-Max), splittet chronologisch in train/valid/test.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_DIR = ROOT_DIR.parent / "dataset"
STAGING_DIR = DATASET_DIR / "model_input" / "staging"
EMBEDDINGS_NPZ = DATASET_DIR / "model_input" / "embeddings.npz"
DATASET_YAML = DATASET_DIR / "dataset.yaml"

CATEGORICAL_COLS = [
    "user_id", "item_id", "brand",
    "category1", "category2", "category3", "category4",
    "geo_country", "device_type", "device_os", "device_model",
    "device_manufacturer", "device_browser", "device_type_detail",
    "day_of_week", "hour_of_day", "month", "is_holiday",
]
SHARED_CATEGORY_COLS = ["category1", "category2", "category3", "category4"]
NUMERIC_COLS = ["price", "geo_latitude", "geo_longitude"]
LABEL_COL = "rating"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1


def _strip_type_suffix(col: str) -> str:
    return re.sub(r":[a-z_]+$", "", col)


def _read_tsv(path: Path, row_limit: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [_strip_type_suffix(c) for c in df.columns]
    if row_limit > 0:
        df = df.iloc[:row_limit]
    return df


def _label_encode(df: pd.DataFrame, col: str) -> tuple[pd.Series, dict]:
    unique_vals = sorted(df[col].dropna().unique())
    mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
    return df[col].map(mapping).fillna(0).astype(int), mapping


def _shared_label_encode(df: pd.DataFrame, cols: list[str]) -> tuple[dict[str, pd.Series], dict, int]:
    all_vals = sorted({v for col in cols if col in df.columns for v in df[col].dropna().unique()})
    mapping = {v: i + 1 for i, v in enumerate(all_vals)}
    encoded = {col: df[col].map(mapping).fillna(0).astype(int) for col in cols if col in df.columns}
    vocab_size = len(mapping) + 1
    return encoded, mapping, vocab_size


def _minmax_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series.fillna(0).astype(float) * 0
    return ((series - lo) / (hi - lo)).fillna(0).astype(float)


def build_csv_splits(
    staging_dir: Path = STAGING_DIR,
    embeddings_path: Path = EMBEDDINGS_NPZ,
    dataset_yaml: Path = DATASET_YAML,
    output_dir: Path = DATASET_DIR,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    row_limit: int = 0,
) -> None:
    print("Lese Staging-Dateien ...")
    inter = _read_tsv(staging_dir / "dataset.inter", row_limit=row_limit)
    item = _read_tsv(staging_dir / "dataset.item").drop(columns=["embedding", "product_name"], errors="ignore")
    user = _read_tsv(staging_dir / "dataset.user")

    inter = inter.drop(columns=["event_type"], errors="ignore")

    print("Joine Staging-Dateien ...")
    df = inter.merge(item, on="item_id", how="left")
    df = df.merge(user, on="user_id", how="left")

    # Numerische Spalten konvertieren
    for col in NUMERIC_COLS + ["rating", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Label-Spalte: 0 = kein Klick (syntheticView), 1 = Klick (view/basket/order)
    if LABEL_COL in df.columns:
        df["label"] = (df[LABEL_COL].astype(float) > 0).astype(int)
        df = df.drop(columns=[LABEL_COL])
    elif "label" not in df.columns:
        raise ValueError("Keine label/rating-Spalte gefunden.")

    # Versionsspalten: nur Hauptversionsnummer als Integer behalten
    for col in ["device_os_version", "device_browser_version"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r"^(\d+)").fillna(0).astype(int)

    # Shared Encoding für Kategorie-Spalten
    shared_encoded, _, shared_vocab_size = _shared_label_encode(df, SHARED_CATEGORY_COLS)
    for col, encoded in shared_encoded.items():
        df[col] = encoded

    # Label-Encoding kategorischer Features
    item_id_mapping: dict = {}
    all_mappings: dict[str, dict] = {}
    for col in CATEGORICAL_COLS:
        if col in SHARED_CATEGORY_COLS:
            continue
        if col not in df.columns:
            print(f"  [Warnung] Spalte {col!r} fehlt, überspringe.")
            continue
        df[col], mapping = _label_encode(df, col)
        all_mappings[col] = mapping
        if col == "item_id":
            item_id_mapping = mapping

    # Min-Max-Normalisierung numerischer Features
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = _minmax_normalize(df[col])

    # Chronologischer Split
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop(columns=["timestamp"], errors="ignore")
    n = len(df)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_df = df.iloc[:n_train]
    valid_df = df.iloc[n_train:n_train + n_valid]
    test_df  = df.iloc[n_train + n_valid:]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_cols = list(df.columns)
    float_cols_set = set(NUMERIC_COLS + ["label"])
    dtype_map = {col: np.float32 if col in float_cols_set else np.int32 for col in all_cols}

    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        csv_path = output_dir / f"{split_name}.csv"
        npz_path = output_dir / f"{split_name}.csv.npz"
        split_df.to_csv(csv_path, index=False)
        arrays = {
            col: split_df[col].fillna(0).values.astype(dtype_map.get(col, np.float32))
            for col in all_cols
        }
        np.savez(npz_path, **arrays)

    print(f"CSVs + NPZs geschrieben: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

    # Feature-Vocab speichern: Originalwert → encoded Integer
    vocab_path = output_dir / "feature_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(all_mappings, f, ensure_ascii=False, indent=2)
    print(f"feature_vocab.json gespeichert: {len(all_mappings)} Features → {vocab_path}")

    # vocab_sizes in dataset.yaml aktualisieren
    if dataset_yaml.exists():
        with open(dataset_yaml) as f:
            config = yaml.safe_load(f)
        updated = 0
        for spec in config.get("feature_specs", []):
            col = spec.get("name")
            if col not in CATEGORICAL_COLS or col not in df.columns:
                continue
            new_size = int(df[col].max()) + 1
            if spec.get("vocab_size") != new_size:
                spec["vocab_size"] = new_size
                updated += 1
        with open(dataset_yaml, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"dataset.yaml: {updated} vocab_sizes aktualisiert.")

    # item_embedding_matrix.npz aufbauen
    if embeddings_path.exists() and item_id_mapping:
        emb_data = np.load(embeddings_path, allow_pickle=True)
        string_ids: list[str] = list(emb_data["ids"])
        vectors: np.ndarray = emb_data["embeddings"]

        vocab_size = int(df["item_id"].max()) + 1
        emb_dim = vectors.shape[1]
        matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)

        matched = 0
        for str_id, vec in zip(string_ids, vectors):
            int_id = item_id_mapping.get(str_id)
            if int_id is not None:
                matrix[int_id] = vec
                matched += 1

        out_path = output_dir / "item_embedding_matrix.npz"
        np.savez(out_path, key=np.arange(vocab_size), value=matrix)
        print(f"item_embedding_matrix.npz: {matched}/{len(string_ids)} Items gemapped → {out_path}")
    elif not embeddings_path.exists():
        print(f"  [Warnung] {embeddings_path} nicht gefunden — item_embedding_matrix.npz nicht erstellt.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    row_limit = int(os.environ.get("DEV_ROW_LIMIT", "0"))
    build_csv_splits(row_limit=row_limit)
