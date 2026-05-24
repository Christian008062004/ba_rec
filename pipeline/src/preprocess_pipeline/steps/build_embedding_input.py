#!/usr/bin/env python3
"""Baut den Embedder-Input aus product_cache.json.

Liest alle Varianten, gruppiert nach parentProductNumber und
speichert { parentId: embedder_input } nach model_input/temp/embedding_input.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_DIR = ROOT_DIR.parent / "dataset"
PRODUCT_CACHE = ROOT_DIR / "cache" / "product_cache.json"
OUTPUT_FILE = DATASET_DIR / "model_input" / "temp" / "embedding_input.json"

FIELDS = {
    "name":            "Produkt",
    "brand":           "Marke",
    "categories":      "Kategorie",
    "metaDescription": "Kurzbeschreibung",
    "description":     "Beschreibung",
}


def clean_text(text: str) -> str:
    text = re.sub(r"\|.*raiffeisenmarkt\.de.*", "", text)
    text = re.sub(r"Jetzt günstig online kaufen!?", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_embedder_input(product: dict) -> str:
    parts = []
    for field, label in FIELDS.items():
        value = product.get(field)
        if not value:
            continue
        if isinstance(value, list):
            text = ", ".join(str(v) for v in value)
        else:
            text = clean_text(str(value))
        parts.append(f"{label}: {text}.")
    return " ".join(parts)


def build_embedding_input(
    cache_path: Path = PRODUCT_CACHE,
    output_file: Path = OUTPUT_FILE,
) -> Path:
    with cache_path.open(encoding="utf-8") as f:
        cache = json.load(f)

    # cache ist {variant_id: product_dict | null}
    result: dict[str, str] = {}
    skipped = 0
    start = time.time()

    entries = [(k, v) for k, v in cache.items() if v and "parentProductNumber" in v]
    total = len(entries)
    print(f"{total} Varianten mit Produktdaten gefunden.")

    for i, (_, product) in enumerate(entries, 1):
        parent_id = product["parentProductNumber"]
        if parent_id in result:
            continue  # erster Variant pro Parent reicht
        embedder_input = build_embedder_input(product)
        if not embedder_input:
            skipped += 1
            continue
        result[parent_id] = embedder_input

        if i % 500 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  {i}/{total} ({i/total*100:.1f}%) | {rate:.0f}/s | ~{remaining:.0f}s")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{len(result)} Parent-Produkte gespeichert → {output_file}")
    if skipped:
        print(f"  ({skipped} übersprungen — kein Text vorhanden)")
    return output_file


if __name__ == "__main__":
    build_embedding_input()
