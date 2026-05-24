#!/usr/bin/env python3
"""Schritt 3: .item mit API-Daten anreichern (Preis, Marke, Kategorien)

Liest product_cache.json und ergänzt die bestehende dataset.item
um Preis, Marke und API-Kategorien.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_DIR = ROOT_DIR.parent / "dataset"
STAGING_DIR = DATASET_DIR / "model_input" / "staging"
CACHE_DIR = ROOT_DIR / "cache"

DEFAULT_ITEM_FILE = STAGING_DIR / "dataset.item"
DEFAULT_PRODUCT_CACHE = CACHE_DIR / "product_cache.json"


def load_parent_to_api(cache_path: Path) -> Dict[str, dict]:
    with cache_path.open(encoding="utf-8") as f:
        cache: dict = json.load(f)
    result: Dict[str, dict] = {}
    for data in cache.values():
        if not data:
            continue
        parent = data.get("parentProductNumber")
        if parent and parent not in result:
            result[parent] = data
    return result


def enrich_item_api(
    item_path: Path = DEFAULT_ITEM_FILE,
    cache_path: Path = DEFAULT_PRODUCT_CACHE,
) -> None:
    parent_to_api = load_parent_to_api(cache_path)
    print(f"API-Cache: {len(parent_to_api)} Parent-Produkte geladen")

    with item_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    print(f".item: {len(rows)} Einträge werden angereichert...")

    item_path.parent.mkdir(parents=True, exist_ok=True)
    with item_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "item_id:token",
            "product_name:token",
            "price:float",
            "brand:token",
            "category1:token",
            "category2:token",
            "category3:token",
            "category4:token",
        ])
        enriched = 0
        for row in rows:
            item_id = row["item_id:token"]
            api = parent_to_api.get(item_id, {})
            if api:
                enriched += 1

            name = api.get("name") or row.get("product_name:token", item_id)
            price = api.get("price", "")
            brand = api.get("brand", "")

            api_cats = api.get("categories") or []
            cats = (api_cats + ["", "", "", ""])[:4] if api_cats else [
                row.get("category1:token", ""),
                row.get("category2:token", ""),
                row.get("category3:token", ""),
                row.get("category4:token", ""),
            ]

            writer.writerow([item_id, name, price, brand,
                             cats[0], cats[1], cats[2], cats[3]])

    print(f".item angereichert: {enriched}/{len(rows)} Einträge mit API-Daten → {item_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Schritt 3: .item mit API-Daten anreichern")
    parser.add_argument("--item", type=Path, default=DEFAULT_ITEM_FILE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_PRODUCT_CACHE)
    args = parser.parse_args(argv)
    enrich_item_api(item_path=args.item, cache_path=args.cache)


if __name__ == "__main__":
    main()
