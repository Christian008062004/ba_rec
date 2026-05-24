#!/usr/bin/env python3
"""Schritt 2: Variant-IDs → parentProductNumber in .inter und .item

Liest product_cache.json, baut eine Variant→Parent-Zuordnung und
schreibt .inter und .item mit den gemappten IDs neu raus.
Duplikate (mehrere Varianten desselben Parents) werden zusammengeführt.
Anschließend wird k-core-Filterung erneut angewendet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocess_pipeline.utils.atomic_files import write_inter_file, write_item_file

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_DIR = ROOT_DIR.parent / "dataset"
STAGING_DIR = DATASET_DIR / "model_input" / "staging"
CACHE_DIR = ROOT_DIR / "cache"

DEFAULT_INTER_FILE = STAGING_DIR / "dataset.inter"
DEFAULT_PRODUCT_CACHE = CACHE_DIR / "product_cache.json"


def load_variant_to_parent(cache_path: Path) -> Dict[str, str]:
    with cache_path.open(encoding="utf-8") as f:
        cache: dict = json.load(f)
    return {
        variant_id: data["parentProductNumber"]
        for variant_id, data in cache.items()
        if data and data.get("parentProductNumber")
    }


def read_inter(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_item(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def filter_k_core(
    rows: List[dict], min_user: int, min_item: int
) -> List[dict]:
    filtered = rows
    while True:
        user_counts = Counter(r["user_id:token"] for r in filtered)
        item_counts = Counter(r["item_id:token"] for r in filtered)
        next_filtered = [
            r for r in filtered
            if user_counts[r["user_id:token"]] >= min_user
            and item_counts[r["item_id:token"]] >= min_item
        ]
        if len(next_filtered) == len(filtered):
            return next_filtered
        filtered = next_filtered


def enrich_parent_ids(
    inter_path: Path = DEFAULT_INTER_FILE,
    cache_path: Path = DEFAULT_PRODUCT_CACHE,
    min_user_interactions: int = 2,
    min_item_interactions: int = 2,
) -> None:
    variant_to_parent = load_variant_to_parent(cache_path)
    print(f"Variant→Parent: {len(variant_to_parent)} Mappings geladen")

    item_path = inter_path.with_suffix(".item")
    user_path = inter_path.with_suffix(".user")

    # .inter remappen und deduplizieren
    inter_rows = read_inter(inter_path)
    remapped = 0
    seen = set()
    new_inter: List[dict] = []
    dropped = 0
    for row in inter_rows:
        old_id = row["item_id:token"]
        parent_id = variant_to_parent.get(old_id)
        if parent_id is None:
            dropped += 1
            continue
        row["item_id:token"] = parent_id
        remapped += 1
        key = (row["user_id:token"], row["item_id:token"], row["timestamp:float"], row["event_type:token"])
        if key not in seen:
            seen.add(key)
            new_inter.append(row)

    print(f".inter: {len(inter_rows)} → {len(new_inter)} Zeilen nach Remap+Dedupe ({remapped} IDs ersetzt, {dropped} ohne Parent verworfen)")

    # k-core neu anwenden
    new_inter = filter_k_core(new_inter, min_user_interactions, min_item_interactions)
    print(f".inter: {len(new_inter)} Zeilen nach k-core-Filterung")

    valid_item_ids = {row["item_id:token"] for row in new_inter}
    valid_user_ids = {row["user_id:token"] for row in new_inter}

    # .inter schreiben (als dicts für write_inter_file umformen)
    inter_dicts = [
        {
            "user_id": r["user_id:token"],
            "item_id": r["item_id:token"],
            "rating": float(r["rating:float"]),
            "timestamp": r["timestamp:float"],
            "event_type": r["event_type:token"],
            "day_of_week": r.get("day_of_week:token", ""),
            "hour_of_day": r.get("hour_of_day:token", ""),
            "month": r.get("month:token", ""),
            "is_holiday": r.get("is_holiday:token", ""),
        }
        for r in new_inter
    ]
    write_inter_file(inter_dicts, inter_path)

    # .item remappen, deduplizieren, auf valide IDs filtern
    item_rows = read_item(item_path)
    seen_items: set[str] = set()
    new_item: Dict[str, dict] = {}
    for row in item_rows:
        old_id = row["item_id:token"]
        parent_id = variant_to_parent.get(old_id)
        if parent_id is None or parent_id not in valid_item_ids:
            continue
        if parent_id not in seen_items:
            seen_items.add(parent_id)
            row["item_id:token"] = parent_id
            new_item[parent_id] = {
                "item_id": parent_id,
                "product_name": row.get("product_name:token", parent_id),
                "category1": row.get("category1:token", ""),
                "category2": row.get("category2:token", ""),
                "category3": row.get("category3:token", ""),
                "category4": row.get("category4:token", ""),
            }

    print(f".item: {len(item_rows)} → {len(new_item)} Einträge nach Remap+Dedupe")
    write_item_file(new_item, item_path)

    print(f"Fertig. Valide Produkte: {len(valid_item_ids)}, Nutzer: {len(valid_user_ids)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Schritt 2: Variant-IDs → parentProductNumber")
    parser.add_argument("--inter", type=Path, default=DEFAULT_INTER_FILE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_PRODUCT_CACHE)
    parser.add_argument("--min-user-interactions", type=int, default=2)
    parser.add_argument("--min-item-interactions", type=int, default=2)
    args = parser.parse_args(argv)
    enrich_parent_ids(
        inter_path=args.inter,
        cache_path=args.cache,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )


if __name__ == "__main__":
    main()
