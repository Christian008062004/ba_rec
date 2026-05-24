#!/usr/bin/env python3
"""Vervollstaendigt den Geocoding-Cache ohne erneutes Einlesen der Roh-ZIPs."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess_pipeline.utils.geocoding import (
    COUNTRY_QIDS,
    DEFAULT_GEOCODE_CACHE,
    DEFAULT_GEONAMES_DE_FILE,
    FOREIGN_VPN_GEO,
    GERMAN_REGION_QIDS,
    build_ssl_context,
    build_geo_key,
    geo_city_variants,
    geocode_germany_from_geonames,
    load_geocode_cache,
    load_geonames_de_index,
    parse_geo_key,
    save_geocode_cache,
)
from preprocess_pipeline.utils.atomic_files import write_user_file

DEFAULT_USER_FILE = Path(__file__).parent.parent.parent.parent / "dataset" / "model_input" / "model_input.user"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

REGION_FALLBACKS = {
    "Germany|Baden-Wuerttemberg": (48.6616, 9.3501),
    "Germany|Bayern": (48.7904, 11.4979),
    "Germany|Berlin": (52.52, 13.405),
    "Germany|Brandenburg": (52.4125, 12.5316),
    "Germany|Bremen": (53.0793, 8.8017),
    "Germany|Hamburg": (53.5511, 9.9937),
    "Germany|Hessen": (50.6521, 9.1624),
    "Germany|Mecklenburg-Vorpommern": (53.6127, 12.4296),
    "Germany|Niedersachsen": (52.6367, 9.8451),
    "Germany|Nordrhein-Westfalen": (51.4332, 7.6616),
    "Germany|Rheinland-Pfalz": (49.9134, 7.4497),
    "Germany|Saarland": (49.3964, 7.023),
    "Germany|Sachsen": (51.1045, 13.2017),
    "Germany|Sachsen-Anhalt": (51.9503, 11.6923),
    "Germany|Schleswig-Holstein": (54.2194, 9.6961),
    "Germany|Thueringen": (50.8612, 11.0527),
}

COUNTRY_FALLBACKS = {
    "Austria": (47.5162, 14.5501),
    "Belgium": (50.5039, 4.4699),
    "Czech Republic": (49.8175, 15.473),
    "Czechia": (49.8175, 15.473),
    "Denmark": (56.2639, 9.5018),
    "France": (46.2276, 2.2137),
    "Luxembourg": (49.8153, 6.1296),
    "Netherlands": (52.1326, 5.2913),
    "Poland": (51.9194, 19.1451),
    "Switzerland": (46.8182, 8.2275),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuellt leere Eintraege in recbole_data/geocoding_cache.json per GeoNames und Wikidata."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_GEOCODE_CACHE,
        help="Pfad zum Geocoding-Cache.",
    )
    parser.add_argument(
        "--geonames-de-file",
        type=Path,
        default=DEFAULT_GEONAMES_DE_FILE,
        help="GeoNames-DE.txt fuer lokale deutsche Ortskoordinaten.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximale Anzahl offener Cache-Eintraege verarbeiten; 0 verarbeitet alle.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Pause zwischen Wikidata-Abfragen in Sekunden.",
    )
    parser.add_argument(
        "--update-user-file",
        action="store_true",
        help="Aktualisiert anschliessend die bestehende .user Datei aus dem Cache.",
    )
    parser.add_argument(
        "--keep-unresolved",
        action="store_true",
        help="Laesst nicht gefundene Orte leer, statt regionale Fallback-Koordinaten zu setzen.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=True,
        help="Deaktiviert SSL-Zertifikatspruefung fuer Wikidata, falls lokale CA-Kette fehlschlaegt.",
    )
    parser.add_argument(
        "--user-file",
        type=Path,
        default=DEFAULT_USER_FILE,
        help="Pfad zur bestehenden .user Datei fuer --update-user-file.",
    )
    return parser.parse_args(argv)


def cache_entry_has_coordinates(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("geo_latitude") not in {"", None}
        and entry.get("geo_longitude") not in {"", None}
    )


def sparql_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_wikidata_query(country: str, region: str, city: str) -> str:
    label_filters = " UNION ".join(
        f'{{ ?place rdfs:label {sparql_string(variant)}@de. }}'
        for variant in geo_city_variants(city)
    )
    if not label_filters:
        label_filters = f'{{ ?place rdfs:label {sparql_string(city)}@de. }}'

    region_qid = GERMAN_REGION_QIDS.get(region)
    country_qid = COUNTRY_QIDS.get(country)
    containment_filter = ""
    if region_qid:
        containment_filter = f"?place wdt:P131* wd:{region_qid}."
    elif country_qid:
        containment_filter = f"?place wdt:P17 wd:{country_qid}."

    return f"""
SELECT ?place ?placeLabel ?coords ?lat ?long WHERE {{
  {label_filters}
  ?place wdt:P625 ?coords.
  {containment_filter}
  BIND(geof:latitude(?coords) AS ?lat)
  BIND(geof:longitude(?coords) AS ?long)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "de,en". }}
}}
LIMIT 5
"""


def query_wikidata(
    country: str,
    region: str,
    city: str,
    insecure: bool,
    timeout: int = 30,
) -> dict:
    query = build_wikidata_query(country, region, city)
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode("utf-8")
    request = urllib.request.Request(
        WIKIDATA_ENDPOINT,
        data=data,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": "bach-recbole-geocoding-cache/1.0",
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout,
        context=build_ssl_context(insecure),
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))

    bindings = payload.get("results", {}).get("bindings", [])
    if not bindings:
        return {}

    row = bindings[0]
    return {
        "geo_latitude": float(row["lat"]["value"]),
        "geo_longitude": float(row["long"]["value"]),
        "geo_api_name": row.get("placeLabel", {}).get("value", city),
        "geo_api_region": region,
        "geo_api_country": country,
    }


def fallback_coordinates(country: str, region: str, city: str) -> dict:
    region_key = f"{country}|{region}"
    if region_key in REGION_FALLBACKS:
        latitude, longitude = REGION_FALLBACKS[region_key]
        return {
            "geo_latitude": latitude,
            "geo_longitude": longitude,
            "geo_api_name": f"Region fallback: {region}",
            "geo_api_region": region,
            "geo_api_country": country,
        }

    if country in COUNTRY_FALLBACKS:
        latitude, longitude = COUNTRY_FALLBACKS[country]
        return {
            "geo_latitude": latitude,
            "geo_longitude": longitude,
            "geo_api_name": f"Country fallback: {country}",
            "geo_api_region": region,
            "geo_api_country": country,
        }

    return FOREIGN_VPN_GEO


def complete_cache(
    cache_path: Path,
    geonames_de_file: Path,
    limit: int,
    delay: float,
    insecure: bool,
    keep_unresolved: bool,
) -> dict:
    cache = load_geocode_cache(cache_path)
    missing_keys = [
        key
        for key, entry in cache.items()
        if not cache_entry_has_coordinates(entry)
    ]
    if limit > 0:
        missing_keys = missing_keys[:limit]

    geonames_index = load_geonames_de_index(geonames_de_file)
    stats = {
        "missing_before": len(missing_keys),
        "geonames_hits": 0,
        "wikidata_hits": 0,
        "foreign_vpn": 0,
        "fallback_hits": 0,
        "still_missing": 0,
    }

    for index, key in enumerate(missing_keys, start=1):
        country, region, city = parse_geo_key(key)
        result = {}

        if country == "Germany":
            result = geocode_germany_from_geonames(city, geonames_index)
            if result:
                result["geo_api_region"] = region
                cache[key] = result
                stats["geonames_hits"] += 1

        if not result and country not in COUNTRY_QIDS:
            cache[key] = FOREIGN_VPN_GEO
            stats["foreign_vpn"] += 1
            result = cache[key]

        if not result:
            try:
                result = query_wikidata(country, region, city, insecure=insecure)
            except Exception as exc:
                print(f"Warnung: Wikidata fehlgeschlagen fuer {key}: {exc}")
                result = {}
            if result:
                cache[key] = result
                stats["wikidata_hits"] += 1
            elif not keep_unresolved:
                cache[key] = fallback_coordinates(country, region, city)
                stats["fallback_hits"] += 1
                result = cache[key]
            elif not cache_entry_has_coordinates(cache.get(key)):
                stats["still_missing"] += 1

        if index % 50 == 0:
            save_geocode_cache(cache, cache_path)
            print(f"{index}/{len(missing_keys)} offene Cache-Eintraege verarbeitet")

        if delay > 0 and not result:
            time.sleep(delay)
        elif delay > 0 and country != "Germany":
            time.sleep(delay)

    save_geocode_cache(cache, cache_path)
    return stats


def update_user_file_from_cache(user_file: Path, cache_path: Path) -> None:
    cache = load_geocode_cache(cache_path)
    user_features = {}
    with user_file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            user_id = row["user_id:token"]
            features = {
                "user_id": user_id,
                "geo_country": row.get("geo_country:token", ""),
                "geo_region": row.get("geo_region:token", ""),
                "geo_city": row.get("geo_city:token", ""),
                "geo_latitude": row.get("geo_latitude:float", ""),
                "geo_longitude": row.get("geo_longitude:float", ""),
                "geo_api_name": row.get("geo_api_name:token", ""),
                "geo_api_region": row.get("geo_api_region:token", ""),
                "geo_api_country": row.get("geo_api_country:token", ""),
            }
            cache_key = build_geo_key(
                features["geo_country"],
                features["geo_region"],
                features["geo_city"],
            )
            if cache_entry_has_coordinates(cache.get(cache_key)):
                features.update(cache[cache_key])
            elif not features["geo_city"]:
                features.update(fallback_coordinates(
                    features["geo_country"],
                    features["geo_region"],
                    features["geo_city"],
                ))
            user_features[user_id] = features

    write_user_file(user_features, user_file)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stats = complete_cache(
        cache_path=args.cache,
        geonames_de_file=args.geonames_de_file,
        limit=args.limit,
        delay=args.delay,
        insecure=args.insecure,
        keep_unresolved=args.keep_unresolved,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=True))
    if args.update_user_file:
        update_user_file_from_cache(args.user_file, args.cache)
        print(f"User-Datei aktualisiert: {args.user_file}")


if __name__ == "__main__":
    main()
