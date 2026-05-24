#!/usr/bin/env python3
"""Schritt 4: .user mit Geokoordinaten anreichern

Liest dataset.user, ermittelt Koordinaten per GeoNames-DE-Index
und Geocoding-API und schreibt die Datei mit geo_latitude/geo_longitude neu.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocess_pipeline.utils.geocoding import (
    DEFAULT_GEOCODE_CACHE,
    DEFAULT_GEONAMES_DE_FILE,
    enrich_geo_coordinates,
)
from preprocess_pipeline.utils.atomic_files import USER_DEVICE_COLUMNS

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_DIR = ROOT_DIR.parent / "dataset"
STAGING_DIR = DATASET_DIR / "model_input" / "staging"

DEFAULT_USER_FILE = STAGING_DIR / "dataset.user"


def enrich_user_geo(
    user_path: Path = DEFAULT_USER_FILE,
    cache_path: Path = DEFAULT_GEOCODE_CACHE,
    geonames_de_file: Path = DEFAULT_GEONAMES_DE_FILE,
    geocode_enabled: bool = True,
    geocode_limit: int = 0,
    geocode_delay: float = 1.0,
    geocode_insecure: bool = True,
) -> None:
    with user_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    print(f".user: {len(rows)} Nutzer geladen")

    user_features: Dict[str, dict] = {
        row["user_id:token"]: {
            "user_id": row["user_id:token"],
            "geo_country": row.get("geo_country:token", ""),
            "geo_region": row.get("geo_region:token", ""),
            "geo_city": row.get("geo_city:token", ""),
            **{col: row.get(f"{col}:token", "") for col in USER_DEVICE_COLUMNS},
        }
        for row in rows
    }

    enrich_geo_coordinates(
        user_features,
        cache_path=cache_path,
        enabled=geocode_enabled,
        limit=geocode_limit,
        delay=geocode_delay,
        insecure=geocode_insecure,
        geonames_de_file=geonames_de_file,
    )

    device_cols = list(USER_DEVICE_COLUMNS.keys())
    user_path.parent.mkdir(parents=True, exist_ok=True)
    with user_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "user_id:token",
            "geo_country:token",
            "geo_latitude:float",
            "geo_longitude:float",
            *[f"{col}:token" for col in device_cols],
        ])
        for user_id, features in sorted(user_features.items()):
            writer.writerow([
                user_id,
                features.get("geo_country", ""),
                features.get("geo_latitude", ""),
                features.get("geo_longitude", ""),
                *[features.get(col, "") for col in device_cols],
            ])

    print(f".user mit Koordinaten geschrieben → {user_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Schritt 4: .user mit Geokoordinaten anreichern")
    parser.add_argument("--user", type=Path, default=DEFAULT_USER_FILE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_GEOCODE_CACHE)
    parser.add_argument("--geonames-de-file", type=Path, default=DEFAULT_GEONAMES_DE_FILE)
    parser.add_argument("--geocode", action="store_true", help="API-Geocoding aktivieren")
    parser.add_argument("--geocode-limit", type=int, default=0)
    parser.add_argument("--geocode-delay", type=float, default=1.0)
    parser.add_argument("--geocode-insecure", action="store_true", default=True)
    args = parser.parse_args(argv)
    enrich_user_geo(
        user_path=args.user,
        cache_path=args.cache,
        geonames_de_file=args.geonames_de_file,
        geocode_enabled=args.geocode,
        geocode_limit=args.geocode_limit,
        geocode_delay=args.geocode_delay,
        geocode_insecure=args.geocode_insecure,
    )


if __name__ == "__main__":
    main()
