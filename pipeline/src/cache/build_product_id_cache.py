#!/usr/bin/env python3
"""Sammelt alle Produkt-IDs aus eTracker-ZIP-Exporten und speichert sie in product_id_cache.json.

Bereits bekannte IDs werden behalten — nur neue ZIPs werden eingelesen.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Sequence

PIPELINE_DIR = Path(__file__).parent.parent.parent
DEFAULT_RAW_DIR = PIPELINE_DIR / "raw"
DEFAULT_CACHE_PATH = PIPELINE_DIR / "cache" / "product_id_cache.json"


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": [], "product_ids": [], "stats": {}}


def save_cache(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def collect_ids_from_zip(zip_path: Path) -> set[str]:
    ids: set[str] = set()
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            csv_names = [n for n in archive.namelist() if n.endswith("-onsite.csv")]
            if not csv_names:
                return ids
            with archive.open(csv_names[0]) as handle:
                reader = csv.DictReader(
                    (line.decode("utf-8", "ignore") for line in handle),
                    delimiter=";",
                )
                for row in reader:
                    pid = (row.get("product_id") or "").strip()
                    if pid and pid not in {"", "<NULL>", "0"}:
                        ids.add(pid)
    except zipfile.BadZipFile:
        print(f"  Warnung: {zip_path.name} ist keine gültige ZIP-Datei")
    return ids


def build_product_id_cache(
    raw_dir: Path = DEFAULT_RAW_DIR,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> None:
    cache = load_cache(cache_path)
    known_files: set[str] = set(cache.get("files", []))
    known_ids: set[str] = set(cache.get("product_ids", []))

    zip_paths = sorted(raw_dir.glob("*.zip"))
    new_files = [z for z in zip_paths if z.name not in known_files]

    print(f"{len(zip_paths)} ZIPs gefunden, {len(known_files)} bereits verarbeitet, {len(new_files)} neu.")

    for zip_path in new_files:
        ids = collect_ids_from_zip(zip_path)
        new_ids = ids - known_ids
        known_ids.update(ids)
        known_files.add(zip_path.name)
        print(f"  {zip_path.name}: {len(ids)} IDs ({len(new_ids)} neu)")

    cache["files"] = sorted(known_files)
    cache["product_ids"] = sorted(known_ids)
    cache["stats"] = {
        "file_count": len(known_files),
        "unique_product_id_count": len(known_ids),
    }

    save_cache(cache, cache_path)
    print(f"\nFertig: {len(known_ids)} eindeutige Produkt-IDs in {cache_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Produkt-IDs aus eTracker-ZIPs sammeln")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    args = parser.parse_args(argv)
    build_product_id_cache(raw_dir=args.raw_dir, cache_path=args.cache)


if __name__ == "__main__":
    main()
