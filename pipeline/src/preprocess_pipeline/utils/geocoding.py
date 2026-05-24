#!/usr/bin/env python3
"""Geocoding-Hilfsfunktionen: Cache, GeoNames-DE-Index und API-Abfragen."""

from __future__ import annotations

import csv
import json
import ssl
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).parent.parent.parent.parent
CACHE_DIR = ROOT_DIR / "cache"

DEFAULT_GEOCODE_CACHE = CACHE_DIR / "geocoding_cache.json"
DEFAULT_GEONAMES_DE_FILE = ROOT_DIR / "raw" / "DE" / "DE.txt"

COUNTRY_CODES = {
    "Austria": "AT",
    "Belgium": "BE",
    "Czechia": "CZ",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "France": "FR",
    "Germany": "DE",
    "Italy": "IT",
    "Luxembourg": "LU",
    "Netherlands": "NL",
    "Poland": "PL",
    "Spain": "ES",
    "Switzerland": "CH",
    "United Kingdom": "GB",
    "United States": "US",
}

COUNTRY_QIDS = {
    "Austria": "Q40",
    "Belgium": "Q31",
    "Czech Republic": "Q213",
    "Czechia": "Q213",
    "Denmark": "Q35",
    "France": "Q142",
    "Germany": "Q183",
    "Luxembourg": "Q32",
    "Netherlands": "Q55",
    "Poland": "Q36",
    "Switzerland": "Q39",
}

GERMAN_REGION_QIDS = {
    "Baden-Wuerttemberg": "Q985",
    "Baden-Württemberg": "Q985",
    "Bayern": "Q980",
    "Berlin": "Q64",
    "Brandenburg": "Q1208",
    "Bremen": "Q1209",
    "Hamburg": "Q1055",
    "Hessen": "Q1199",
    "Mecklenburg-Vorpommern": "Q1196",
    "Niedersachsen": "Q1197",
    "Nordrhein-Westfalen": "Q1198",
    "Rheinland-Pfalz": "Q1200",
    "Saarland": "Q1201",
    "Sachsen": "Q1202",
    "Sachsen-Anhalt": "Q1206",
    "Schleswig-Holstein": "Q1194",
    "Thueringen": "Q1205",
    "Thüringen": "Q1205",
}

DE_NEIGHBOR_COUNTRIES = {
    "Austria",
    "Belgium",
    "Czech Republic",
    "Czechia",
    "Denmark",
    "France",
    "Luxembourg",
    "Netherlands",
    "Poland",
    "Switzerland",
}

FOREIGN_VPN_GEO = {
    "geo_latitude": -90.0,
    "geo_longitude": -180.0,
}


def clean_value(raw_value: str | None, fallback: str = "") -> str:
    value = str(raw_value or "").strip()
    if value in {"", "<NULL>"}:
        return fallback
    return " ".join(value.split())


def load_geocode_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warnung: Geocoding-Cache {cache_path} ist ungueltig; starte mit leerem Cache")
        return {}


def save_geocode_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_geo_key(country: str, region: str, city: str) -> str:
    return "|".join([country, region, city])


def parse_geo_key(key: str) -> tuple[str, str, str]:
    parts = key.split("|")
    return (parts + ["", "", ""])[:3]


def normalize_geocode_query(value: str) -> str:
    value = urllib.parse.unquote(clean_value(value)).replace("\\'", "'").replace("\\", "")
    replacements = {
        "Ae": "Ä",
        "Oe": "Ö",
        "Ue": "Ü",
        "ae": "ä",
        "oe": "ö",
        "ue": "ü",
    }
    normalized = value
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def normalize_lookup_key(value: str) -> str:
    normalized = normalize_geocode_query(clean_value(value)).casefold()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("ß", "ss")
    return " ".join(normalized.split())


def geo_city_variants(city: str) -> List[str]:
    cleaned = urllib.parse.unquote(clean_value(city)).replace("\\'", "'").replace("\\", "")
    if not cleaned:
        return []

    variants = [cleaned]
    normalized_cleaned = normalize_geocode_query(cleaned)
    if normalized_cleaned != cleaned:
        variants.append(normalized_cleaned)
    if "(" in cleaned and ")" in cleaned:
        before_parenthesis = cleaned.split("(", 1)[0].strip()
        inside_parenthesis = cleaned.split("(", 1)[1].split(")", 1)[0].strip()
        variants.extend([inside_parenthesis, before_parenthesis])

        if "-" in inside_parenthesis:
            variants.append(inside_parenthesis.split("-", 1)[0].strip())
        if "-" in before_parenthesis:
            variants.append(before_parenthesis.split("-", 1)[0].strip())
    elif "-" in cleaned:
        before_hyphen, after_hyphen = cleaned.split("-", 1)
        variants.append(before_hyphen.strip())
        variants.append(after_hyphen.strip())
        variants.append(cleaned.replace("-", " "))

    result: List[str] = []
    seen = set()
    for variant in variants:
        if not variant:
            continue
        key = normalize_lookup_key(variant)
        if key in seen:
            continue
        seen.add(key)
        result.append(variant)
    return result


def load_geonames_de_index(path: Path) -> Dict[str, List[dict]]:
    index: Dict[str, List[dict]] = defaultdict(list)
    if not path.exists():
        print(f"Warnung: GeoNames-Datei {path} nicht gefunden")
        return index

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 15:
                continue
            name = clean_value(row[1])
            ascii_name = clean_value(row[2])
            alternate_names = [clean_value(value) for value in row[3].split(",") if value]
            try:
                latitude = float(row[4])
                longitude = float(row[5])
            except ValueError:
                continue

            try:
                population = int(row[14] or 0)
            except ValueError:
                population = 0

            entry = {
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "feature_class": row[6],
                "feature_code": row[7],
                "population": population,
            }
            for candidate_name in [name, ascii_name, *alternate_names]:
                key = normalize_lookup_key(candidate_name)
                if key:
                    index[key].append(entry)

    print(f"GeoNames-DE geladen: {len(index)} Ortsnamen aus {path}")
    return index


def geocode_germany_from_geonames(
    city: str,
    geonames_index: Dict[str, List[dict]],
) -> dict:
    for variant in geo_city_variants(city):
        candidates = geonames_index.get(normalize_lookup_key(variant), [])
        if not candidates:
            continue

        def score(candidate: dict) -> tuple[int, int, int]:
            return (
                int(candidate.get("feature_class") == "P"),
                int(
                    candidate.get("feature_code")
                    in {"PPL", "PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4"}
                ),
                int(candidate.get("population") or 0),
            )

        best = max(candidates, key=score)
        return {
            "geo_latitude": best["latitude"],
            "geo_longitude": best["longitude"],
        }
    return {}


def build_ssl_context(insecure: bool) -> ssl.SSLContext | None:
    if insecure:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    try:
        import importlib

        certifi = importlib.import_module("certifi")
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def geocode_open_meteo(
    country: str,
    region: str,
    city: str,
    ssl_context: ssl.SSLContext | None = None,
    timeout: int = 10,
) -> dict:
    if not city:
        return {}

    query_city = normalize_geocode_query(city)
    params = {
        "name": query_city,
        "count": 10,
        "language": "en",
        "format": "json",
    }
    country_code = COUNTRY_CODES.get(country)
    if country_code:
        params["countryCode"] = country_code

    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bach-recbole-dataset-geocoder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
        payload = json.loads(response.read().decode("utf-8"))

    candidates = payload.get("results") or []
    if not candidates:
        return {}

    region_lower = region.lower()
    country_code_lower = (country_code or "").lower()

    def score(candidate: dict) -> tuple[int, int]:
        candidate_region = str(candidate.get("admin1") or "").lower()
        candidate_country = str(candidate.get("country_code") or "").lower()
        return (
            int(bool(region_lower and candidate_region == region_lower)),
            int(bool(country_code_lower and candidate_country == country_code_lower)),
        )

    best = max(candidates, key=score)
    return {
        "geo_latitude": best.get("latitude"),
        "geo_longitude": best.get("longitude"),
    }


def geocode_nominatim(
    country: str,
    region: str,
    city: str,
    ssl_context: ssl.SSLContext | None = None,
    timeout: int = 10,
) -> dict:
    if not city:
        return {}

    query = ", ".join(
        part for part in [normalize_geocode_query(city), region, country] if part
    )
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bach-recbole-dataset-geocoder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not payload:
        return {}

    best = payload[0]
    return {
        "geo_latitude": best.get("lat"),
        "geo_longitude": best.get("lon"),
    }


def geocode_location(
    country: str,
    region: str,
    city: str,
    ssl_context: ssl.SSLContext | None = None,
    timeout: int = 10,
) -> dict:
    result = geocode_open_meteo(
        country,
        region,
        city,
        ssl_context=ssl_context,
        timeout=timeout,
    )
    if result:
        return result
    return geocode_nominatim(
        country,
        region,
        city,
        ssl_context=ssl_context,
        timeout=timeout,
    )


def enrich_geo_coordinates(
    user_features: Dict[str, dict],
    cache_path: Path,
    enabled: bool,
    limit: int,
    delay: float,
    insecure: bool,
    geonames_de_file: Path,
) -> None:
    for features in user_features.values():
        features.setdefault("geo_latitude", "")
        features.setdefault("geo_longitude", "")

    if not enabled:
        return

    cache = load_geocode_cache(cache_path)
    geonames_de_index = load_geonames_de_index(geonames_de_file)
    location_counts = Counter(
        build_geo_key(
            features.get("geo_country", ""),
            features.get("geo_region", ""),
            features.get("geo_city", ""),
        )
        for features in user_features.values()
        if features.get("geo_city")
    )
    local_de_count = 0
    foreign_vpn_count = 0
    api_candidates: List[str] = []

    for key, _ in location_counts.most_common():
        country, region, city = parse_geo_key(key)

        if country == "Germany":
            local_result = geocode_germany_from_geonames(city, geonames_de_index)
            if local_result:
                local_result["geo_api_region"] = region
                cache[key] = local_result
                local_de_count += 1
            elif key not in cache:
                cache[key] = {}
            continue

        if country not in DE_NEIGHBOR_COUNTRIES:
            cache[key] = FOREIGN_VPN_GEO
            foreign_vpn_count += 1
            continue

        if key not in cache or not cache[key]:
            api_candidates.append(key)

    pending_keys = api_candidates
    if limit > 0:
        pending_keys = pending_keys[:limit]

    print(
        "Geocoding: "
        f"{len(location_counts)} eindeutige Orte, "
        f"{local_de_count} lokale DE-Treffer, "
        f"{foreign_vpn_count} Ausland/VPN-Festwerte, "
        f"{len(pending_keys)} neue Nachbarland-API-Abfragen, "
        f"{len(cache)} Cache-Eintraege"
    )
    for index, key in enumerate(pending_keys, start=1):
        country, region, city = parse_geo_key(key)
        try:
            cache[key] = geocode_location(
                country,
                region,
                city,
                ssl_context=build_ssl_context(insecure),
            )
        except Exception as exc:
            print(f"Warnung: Geocoding fehlgeschlagen fuer {key}: {exc}")
            cache[key] = {}
        if index % 100 == 0:
            save_geocode_cache(cache, cache_path)
            print(f"Geocoding: {index}/{len(pending_keys)} neue Orte verarbeitet")
        if delay > 0:
            time.sleep(delay)

    save_geocode_cache(cache, cache_path)
    for features in user_features.values():
        key = build_geo_key(
            features.get("geo_country", ""),
            features.get("geo_region", ""),
            features.get("geo_city", ""),
        )
        features.update(cache.get(key) or {})
