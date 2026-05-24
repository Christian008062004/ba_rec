#!/usr/bin/env python3
"""Berechnet Embeddings aus embedding_input.json und speichert sie als .npz.

Bereits gecachte IDs werden übersprungen — nur neue werden berechnet.

Eingabe:  model_input/temp/embedding_input.json  { parentId: embedder_input }
Ausgabe:
  model_input/embeddings.npz  – enthält 'ids' (parentProductNumbers) und 'embeddings' (Float32-Matrix)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_DIR = ROOT_DIR.parent / "dataset"

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = DATASET_DIR / "model_input" / "temp" / "embedding_input.json"
OUTPUT_DIR = DATASET_DIR / "model_input"
MODEL_NAME = "Gameselo/STS-multilingual-mpnet-base-v2"


def _load_existing(out_path: Path) -> tuple[list[str], list[list[float]]]:
    """Lädt bereits berechnete Embeddings aus der .npz, falls vorhanden."""
    if not out_path.exists():
        return [], []
    data = np.load(out_path, allow_pickle=True)
    return list(data["ids"]), list(data["embeddings"])


def build_embeddings(
    input_file: Path = INPUT_FILE,
    output_dir: Path = OUTPUT_DIR,
    model_name: str = MODEL_NAME,
) -> Path:
    with input_file.open(encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)

    out_path = output_dir / "embeddings.npz"
    existing_ids, existing_vectors = _load_existing(out_path)
    cached = set(existing_ids)

    missing_ids = [pid for pid in data if pid not in cached]
    missing_texts = [data[pid] for pid in missing_ids]

    print(f"{len(data)} Produkte total, {len(cached)} bereits gecacht, {len(missing_ids)} neu.")

    if missing_ids:
        print(f"Lade Modell: {model_name}")
        model = SentenceTransformer(model_name)

        print(f"Berechne Embeddings für {len(missing_ids)} neue Produkte...")
        new_vectors = model.encode(missing_texts, show_progress_bar=True, convert_to_numpy=True)

        all_ids = existing_ids + missing_ids
        all_vectors = existing_vectors + list(new_vectors)
    else:
        print("Nichts zu tun — alle Embeddings bereits vorhanden.")
        return out_path

    embedding_matrix = np.array(all_vectors, dtype=np.float32)
    id_array = np.array(all_ids, dtype=str)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, ids=id_array, embeddings=embedding_matrix)

    print(f"Embeddings: {embedding_matrix.shape}  → {out_path}")
    return out_path


if __name__ == "__main__":
    build_embeddings()
