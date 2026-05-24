#!/usr/bin/env python3
"""Liest Produkt-IDs aus eTracker-ZIP-Exporten und zeigt eine Übersicht.

Dient zur Diagnose, bevor build_product_cache.py aufgerufen wird.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path
from collections import Counter

PIPELINE_DIR = Path(__file__).parent.parent.parent   # pipeline/
DEFAULT_RAW_DIR = PIPELINE_DIR / "raw"


def inspect(raw_dir: Path, show_samples: int = 10) -> None:
    if not raw_dir.exists():
        print(f"FEHLER: raw-Verzeichnis nicht gefunden: {raw_dir}")
        return

    zips = sorted(raw_dir.glob("*.zip"))
    print(f"raw-Verzeichnis : {raw_dir}")
    print(f"ZIP-Dateien     : {len(zips)}")
    if not zips:
        return

    product_ids: set[str] = set()
    zip_stats: list[dict] = []

    for zip_path in zips:
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                csv_names = [n for n in archive.namelist() if n.endswith("-onsite.csv")]
                if not csv_names:
                    zip_stats.append({"zip": zip_path.name, "onsite_csv": None, "ids": 0})
                    continue
                csv_name = csv_names[0]
                local_ids: set[str] = set()
                with archive.open(csv_name) as handle:
                    reader = csv.DictReader(
                        (line.decode("utf-8", "ignore") for line in handle),
                        delimiter=";",
                    )
                    columns = None
                    for row in reader:
                        if columns is None:
                            columns = list(row.keys())
                        pid = (row.get("product_id") or "").strip()
                        if pid and pid not in {"", "<NULL>", "0"}:
                            local_ids.add(pid)
                product_ids.update(local_ids)
                zip_stats.append({
                    "zip": zip_path.name,
                    "onsite_csv": csv_name,
                    "ids": len(local_ids),
                    "columns": columns,
                })
        except zipfile.BadZipFile:
            print(f"  Warnung: {zip_path.name} ist keine gültige ZIP-Datei")

    print(f"\nErgebnis:")
    print(f"  Eindeutige Produkt-IDs gesamt : {len(product_ids)}")
    print()

    # Spaltenübersicht aus erster ZIP
    first = next((s for s in zip_stats if s.get("columns")), None)
    if first:
        cols = first["columns"]
        pid_idx = cols.index("product_id") if "product_id" in cols else None
        print(f"Spalten in onsite-CSV ({len(cols)} gesamt):")
        for i, c in enumerate(cols):
            marker = " <-- product_id" if i == pid_idx else ""
            print(f"  [{i:2d}] {c}{marker}")
        print()

    # Pro-ZIP-Übersicht
    print(f"Pro-ZIP-Übersicht (erste {min(10, len(zip_stats))}):")
    for s in zip_stats[:10]:
        onsite = s.get("onsite_csv") or "KEINE onsite-CSV"
        print(f"  {s['zip']:35s}  {onsite:40s}  {s['ids']:6d} IDs")

    if len(zip_stats) > 10:
        print(f"  ... ({len(zip_stats) - 10} weitere)")

    # Beispiel-IDs
    if product_ids and show_samples > 0:
        samples = sorted(product_ids)[:show_samples]
        print(f"\nBeispiel-IDs ({show_samples} von {len(product_ids)}):")
        for pid in samples:
            print(f"  {pid}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose: Produkt-IDs aus eTracker-ZIPs auslesen"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
        help=f"Verzeichnis mit eTracker-ZIP-Exporten (default: {DEFAULT_RAW_DIR})"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Anzahl Beispiel-IDs die ausgegeben werden"
    )
    args = parser.parse_args()
    inspect(raw_dir=args.raw_dir, show_samples=args.samples)


if __name__ == "__main__":
    main()
