#!/usr/bin/env python3
"""Schritt 1: eTracker-ZIPs → Basis-Atomic-Files in model_input/staging/

Liest alle onsite-CSVs aus den ZIP-Dateien, filtert relevante Events,
wendet k-core-Filterung an und schreibt .inter, .item und .user
ausschließlich auf Basis der eTracker-Rohdaten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from preprocess_pipeline.utils.atomic_files import (
    USER_DEVICE_COLUMNS,
    write_inter_file,
    write_item_file,
    write_user_file,
)

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_DIR = ROOT_DIR.parent / "dataset"
DATA_DIR = ROOT_DIR / "raw"
OUTPUT_DIR = DATASET_DIR / "model_input" / "staging"
STAGING_CACHE_DIR = ROOT_DIR / "cache" / "staging_cache"
STAGING_MANIFEST = STAGING_CACHE_DIR / "manifest.json"

DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "dataset.inter"

EVENT_ALIASES = {
    "viewproduct": "viewProduct",
    "inserttobasket": "insertToBasket",
    "inserttowatchlist": "insertToWatchlist",
    "order": "order",
    "syntheticproductoforder": "order",
    # Echtes Negativ: Produkt in Liste angezeigt, aber nicht angeklickt
    "syntheticviewproductinlist": "syntheticViewProductInList",
}

RELEVANT_EVENTS = set(EVENT_ALIASES.values())

# Impressionen ohne Klick (label=0 nach Binarisierung in build_csv_splits)
IMPRESSION_EVENTS = {"syntheticViewProductInList"}

DEFAULT_EVENT_WEIGHTS = {
    "syntheticViewProductInList": 0.0,
    "viewProduct": 1.0,
    "insertToWatchlist": 2.0,
    "insertToBasket": 3.0,
    "order": 5.0,
}

GEO_COLUMNS = {
    "geo_country": "geo_country",
    "geo_region": "geo_region",
    "geo_city": "geo_city",
}

DEVICE_LABELS = {
    "STR_CC_ATTR_VALUE_DEVICE_TYPE_MOBILE_PHONE": "Mobile phone",
    "STR_CC_ATTR_VALUE_DEVICE_TYPE_DESKTOP": "Desktop",
    "STR_CC_ATTR_VALUE_DEVICE_TYPE_TABLET": "Tablet",
    "STR_CC_ATTR_VALUE_DEVICE_TYPE_OTHERS": "Other",
    "STR_CC_ATTR_VALUE_DEVICE_TYPE_DETAIL_SMARTPHONE": "Smartphone",
    "STR_CC_ATTR_VALUE_DEVICE_TYPE_DETAIL_NON_SMARTPHONE": "Feature phone",
    "STR_CC_ATTR_VALUE_DEVICE_TYPE_DETAIL_SMARTTV": "Smart TV",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schritt 1: eTracker-ZIPs → Basis-Atomic-Files")
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--min-user-interactions", type=int, default=2)
    parser.add_argument("--min-item-interactions", type=int, default=2)
    parser.add_argument("--max-zip-files", type=int, default=0)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-max-rows", type=int, default=10000)
    return parser


def _zip_fingerprint(zip_path: Path) -> str:
    stat = zip_path.stat()
    raw = f"{zip_path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_manifest() -> dict:
    if STAGING_MANIFEST.exists():
        return json.loads(STAGING_MANIFEST.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    STAGING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _cache_path(zip_name: str, fingerprint: str) -> Path:
    return STAGING_CACHE_DIR / f"{zip_name}_{fingerprint}.pkl"


def _load_cached(zip_name: str, fingerprint: str) -> tuple[List[dict], Dict[str, dict]] | None:
    path = _cache_path(zip_name, fingerprint)
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return None


def _save_cached(zip_name: str, fingerprint: str, rows: List[dict], features: Dict[str, dict]) -> None:
    STAGING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(zip_name, fingerprint)
    with path.open("wb") as f:
        pickle.dump((rows, features), f, protocol=pickle.HIGHEST_PROTOCOL)


def find_onsite_files(data_dir: Path, max_zip_files: int = 0) -> List[tuple[Path, str]]:
    onsite_files: List[tuple[Path, str]] = []
    zip_paths = sorted(data_dir.glob("*.zip"))
    if max_zip_files > 0:
        zip_paths = zip_paths[:max_zip_files]
    for zip_path in zip_paths:
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                for filename in archive.namelist():
                    if filename.endswith("-onsite.csv"):
                        onsite_files.append((zip_path, filename))
                        break
        except zipfile.BadZipFile:
            print(f"Warnung: {zip_path.name} ist keine gueltige ZIP-Datei (ueberspringe)")
        except Exception as exc:
            print(f"Warnung: Fehler beim Lesen von {zip_path.name}: {exc}")
    return onsite_files


def _is_valid_product_id(value: str | None) -> bool:
    return str(value or "").strip() not in {"", "<NULL>", "0"}


def parse_timestamp(raw_value: str) -> int:
    normalized = raw_value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def clean_value(raw_value: str | None, fallback: str = "") -> str:
    value = str(raw_value or "").strip()
    if value in {"", "<NULL>"}:
        return fallback
    return " ".join(value.split())


def normalize_event_type(raw_value: str | None) -> str:
    return EVENT_ALIASES.get(str(raw_value or "").strip().lower(), "")


def _easter(year: int) -> date:
    """Berechnet Ostersonntag nach dem Gauss-Algorithmus."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _german_holidays(year: int) -> frozenset[date]:
    """Gibt bundesweite gesetzliche Feiertage für ein Jahr zurück."""
    from datetime import timedelta
    easter = _easter(year)
    return frozenset({
        date(year, 1, 1),                    # Neujahr
        easter - timedelta(days=2),          # Karfreitag
        easter + timedelta(days=1),          # Ostermontag
        date(year, 5, 1),                    # Tag der Arbeit
        easter + timedelta(days=39),         # Christi Himmelfahrt
        easter + timedelta(days=50),         # Pfingstmontag
        date(year, 10, 3),                   # Tag der Deutschen Einheit
        date(year, 12, 25),                  # 1. Weihnachtstag
        date(year, 12, 26),                  # 2. Weihnachtstag
    })


_HOLIDAY_CACHE: dict[int, frozenset[date]] = {}


def extract_time_features(ts: int) -> dict:
    """Extrahiert kategorische Zeitfeatures aus einem Unix-Timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    year = dt.year
    if year not in _HOLIDAY_CACHE:
        _HOLIDAY_CACHE[year] = _german_holidays(year)

    return {
        "day_of_week": dt.weekday(),           # 0=Mo … 6=So
        "hour_of_day": dt.hour,                # 0–23
        "month": dt.month,                     # 1–12
        "is_holiday": int(dt.date() in _HOLIDAY_CACHE[year]),  # 0/1
    }


def normalize_readable_value(raw_value: str | None, fallback: str = "") -> str:
    value = clean_value(raw_value, fallback=fallback)
    return DEVICE_LABELS.get(value, value)


def read_mapped_columns(row: dict, mapping: Dict[str, str], fallback: str = "") -> dict:
    return {
        target: normalize_readable_value(row.get(source), fallback=fallback)
        for target, source in mapping.items()
    }


def read_item_features(row: dict, item_id: str) -> dict:
    return {
        "item_id": item_id,
        "product_name": clean_value(row.get("product_name"), fallback=item_id),
        "category1": clean_value(row.get("1. Category")),
        "category2": clean_value(row.get("2. Category")),
        "category3": clean_value(row.get("3. Category")),
        "category4": clean_value(row.get("4. Category")),
    }


def process_onsite_file(zip_path: Path, csv_filename: str) -> tuple[List[dict], Dict[str, dict]]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(csv_filename) as handle:
            reader = csv.DictReader(
                (line.decode("utf-8", "ignore") for line in handle),
                delimiter=";",
            )
            result_rows: List[dict] = []
            item_features: Dict[str, dict] = {}
            seen_keys = set()
            for row in reader:
                event_type = normalize_event_type(row.get("event_type"))
                if event_type not in RELEVANT_EVENTS:
                    continue

                user_id = (row.get("user_id") or "").strip()
                timestamp = (row.get("timestamp") or "").strip()
                product_id = (row.get("product_id") or "").strip()
                if not user_id or not timestamp or not _is_valid_product_id(product_id):
                    continue

                try:
                    ts = parse_timestamp(timestamp)
                except ValueError:
                    continue

                normalized_event = "order" if event_type in {"syntheticProductOfOrder", "order"} else event_type
                item_features.setdefault(product_id, read_item_features(row, product_id))

                normalized = {
                    "user_id": user_id,
                    "item_id": product_id,
                    "timestamp": ts,
                    "event_type": normalized_event,
                    "rating": DEFAULT_EVENT_WEIGHTS.get(normalized_event, 1.0),
                    **extract_time_features(ts),
                    **read_mapped_columns(row, USER_DEVICE_COLUMNS),
                    **read_mapped_columns(row, GEO_COLUMNS),
                }
                dedupe_key = (user_id, product_id, ts, normalized_event)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                result_rows.append(normalized)

    result_rows.sort(key=lambda r: (r["timestamp"], r["user_id"], r["item_id"], r["event_type"]))
    print(f"{csv_filename}: {len(result_rows)} relevante Events gefunden")
    return result_rows, item_features


def filter_k_core(
    rows: Iterable[dict], min_user_interactions: int, min_item_interactions: int
) -> List[dict]:
    all_rows = list(rows)

    # Impressionen vor k-core rausfiltern — reduziert die zu filternde Menge massiv
    impressions = [r for r in all_rows if r["event_type"] in IMPRESSION_EVENTS]
    filtered = [r for r in all_rows if r["event_type"] not in IMPRESSION_EVENTS]

    while True:
        user_counts = Counter(row["user_id"] for row in filtered)
        item_counts = Counter(row["item_id"] for row in filtered)
        next_filtered = [
            row for row in filtered
            if user_counts[row["user_id"]] >= min_user_interactions
            and item_counts[row["item_id"]] >= min_item_interactions
        ]
        if len(next_filtered) == len(filtered):
            break
        filtered = next_filtered

    # Nur Impressionen von User/Item-Paaren behalten die den k-core überlebt haben
    valid_users = {row["user_id"] for row in filtered}
    valid_items = {row["item_id"] for row in filtered}
    kept_impressions = [
        r for r in impressions
        if r["user_id"] in valid_users and r["item_id"] in valid_items
    ]

    return filtered + kept_impressions


def build_user_features(rows: Iterable[dict]) -> Dict[str, dict]:
    rows = list(rows)
    country_buckets: Dict[str, Counter] = defaultdict(Counter)
    device_buckets: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    geo_region_buckets: Dict[str, Counter] = defaultdict(Counter)
    geo_city_buckets: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        user_id = row["user_id"]
        if country := row.get("geo_country", ""):
            country_buckets[user_id][country] += 1
        if region := row.get("geo_region", ""):
            geo_region_buckets[user_id][region] += 1
        if city := row.get("geo_city", ""):
            geo_city_buckets[user_id][city] += 1
        for col in USER_DEVICE_COLUMNS:
            if val := row.get(col, ""):
                device_buckets[user_id][col][val] += 1

    return {
        user_id: {
            "user_id": user_id,
            "geo_country": country_buckets[user_id].most_common(1)[0][0] if country_buckets.get(user_id) else "",
            "geo_region": geo_region_buckets[user_id].most_common(1)[0][0] if geo_region_buckets.get(user_id) else "",
            "geo_city": geo_city_buckets[user_id].most_common(1)[0][0] if geo_city_buckets.get(user_id) else "",
            **{
                col: (device_buckets[user_id][col].most_common(1)[0][0] if device_buckets[user_id][col] else "")
                for col in USER_DEVICE_COLUMNS
            },
        }
        for user_id in {row["user_id"] for row in rows}
    }


def build_staging_base(
    input_dir: Path = DATA_DIR,
    output_path: Path = DEFAULT_OUTPUT_FILE,
    min_user_interactions: int = 2,
    min_item_interactions: int = 2,
    max_zip_files: int = 0,
    test_mode: bool = False,
    test_max_rows: int = 10000,
) -> List[dict]:
    if test_mode and max_zip_files == 0:
        max_zip_files = 1
    onsite_files = find_onsite_files(input_dir, max_zip_files=max_zip_files)
    print(f"Gefunden: {len(onsite_files)} onsite-Dateien")
    if not onsite_files:
        raise FileNotFoundError(f"Keine onsite-Dateien in {input_dir} gefunden")

    manifest = _load_manifest()
    all_rows: List[dict] = []
    item_features: Dict[str, dict] = {}
    new_manifest: dict = {}

    for zip_path, csv_filename in onsite_files:
        fp = _zip_fingerprint(zip_path)
        new_manifest[zip_path.name] = fp
        cached = _load_cached(zip_path.name, fp)
        if cached is not None:
            rows, features = cached
            print(f"  Cache: {zip_path.name} ({len(rows)} Events)")
        else:
            print(f"  Lese: {zip_path.name}...")
            rows, features = process_onsite_file(zip_path, csv_filename)
            _save_cached(zip_path.name, fp, rows, features)
            # Alte Cache-Datei für diese ZIP löschen falls Fingerprint geändert hat
            old_fp = manifest.get(zip_path.name)
            if old_fp and old_fp != fp:
                old = _cache_path(zip_path.name, old_fp)
                if old.exists():
                    old.unlink()
        all_rows.extend(rows)
        item_features.update(features)

    _save_manifest(new_manifest)

    if not all_rows:
        raise ValueError("Keine relevanten Daten gefunden")

    if test_mode:
        all_rows = all_rows[:test_max_rows]
        print(f"[TESTMODUS] Daten auf {len(all_rows)} Zeilen begrenzt")

    rows = filter_k_core(all_rows, min_user_interactions, min_item_interactions)
    rows.sort(key=lambda r: (r["user_id"], r["timestamp"], r["item_id"], r["event_type"]))

    filtered_item_features = {
        item_id: item_features.get(item_id, {"item_id": item_id, "product_name": item_id,
                                              "category1": "", "category2": "", "category3": "", "category4": ""})
        for item_id in {row["item_id"] for row in rows}
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_inter_file(rows, output_path)
    write_item_file(filtered_item_features, output_path.with_suffix(".item"))
    write_user_file(build_user_features(rows), output_path.with_suffix(".user"))

    print(f".inter → {output_path}")
    print(f".item  → {output_path.with_suffix('.item')}")
    print(f".user  → {output_path.with_suffix('.user')}")
    print(f"Interaktionen: {len(rows)}")
    print(f"Nutzer:        {len({row['user_id'] for row in rows})}")
    print(f"Produkte:      {len({row['item_id'] for row in rows})}")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_staging_base(
        input_dir=args.input_dir,
        output_path=args.output,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_zip_files=args.max_zip_files,
        test_mode=args.test_mode,
        test_max_rows=args.test_max_rows,
    )


if __name__ == "__main__":
    main()
