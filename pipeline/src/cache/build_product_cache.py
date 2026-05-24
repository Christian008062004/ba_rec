#!/usr/bin/env python3
"""Baut den Produkt-Cache auf.

Liest alle Produkt-IDs aus product_id_cache.json und fragt noch nicht
gecachte Produkte beim Shopware-Backend ab. Ergebnisse werden in
cache/product_cache.json gespeichert.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

PIPELINE_DIR = Path(__file__).parent.parent.parent
DEFAULT_CACHE_PATH = PIPELINE_DIR / "cache" / "product_cache.json"
DEFAULT_PRODUCT_ID_CACHE_PATH = PIPELINE_DIR / "cache" / "product_id_cache.json"

SHOPWARE_BASE_URL = "https://raiffeisenmarkt-de.ddev.site"
TOKEN_URL = "https://raiffeisenmarkt.ddev.site/api/oauth/token"
PRODUCT_URL = f"{SHOPWARE_BASE_URL}/api/bachelor/products"

TOKEN_LIFETIME_SECONDS = 550  # etwas unter 10 Minuten, um sicher zu sein


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ[key] = value


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_token(client_id: str, client_secret: str) -> str:
    payload = json.dumps({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"Kein access_token in Antwort: {data}")
    return token


def fetch_product(product_id: str, token: str) -> dict | None:
    url = f"{PRODUCT_URL}/{urllib.parse.quote(product_id, safe='')}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shopware-Produkt-Cache aufbauen")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--product-id-cache", type=Path, default=DEFAULT_PRODUCT_ID_CACHE_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Max neue Abfragen (0 = alle)")
    parser.add_argument("--delay", type=float, default=0.05, help="Pause zwischen Abfragen in Sekunden")
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--client-secret", default=None)
    args = parser.parse_args(argv)

    load_env_file(PIPELINE_DIR.parent / ".env")
    load_env_file(PIPELINE_DIR / ".env")

    client_id = args.client_id or os.environ.get("SHOPWARE_CLIENT_ID", "").strip()
    client_secret = args.client_secret or os.environ.get("SHOPWARE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        parser.error("SHOPWARE_CLIENT_ID und SHOPWARE_CLIENT_SECRET müssen gesetzt sein")

    # IDs aus product_id_cache.json laden
    id_cache = load_json(args.product_id_cache)
    all_ids: list[str] = id_cache.get("product_ids", [])
    print(f"{len(all_ids)} Produkt-IDs geladen aus {args.product_id_cache}")

    # Bereits gecachte Produkte laden
    product_cache: dict = load_json(args.cache) if args.cache.exists() else {}
    missing = [pid for pid in all_ids if product_cache.get(pid) is None]
    if args.limit > 0:
        missing = missing[:args.limit]

    print(f"{len(product_cache)} bereits gecacht, {len(missing)} offen")

    if not missing:
        print("Cache vollständig.")
        return

    token = fetch_token(client_id, client_secret)
    token_fetched_at = time.monotonic()
    print("Bearer-Token geholt.")

    fetched = not_found = errors = 0
    total = len(missing)
    start_time = time.monotonic()

    for index, pid in enumerate(missing, start=1):
        if time.monotonic() - token_fetched_at >= TOKEN_LIFETIME_SECONDS:
            token = fetch_token(client_id, client_secret)
            token_fetched_at = time.monotonic()

        try:
            result = fetch_product(pid, token)
        except Exception as exc:
            print(f"\n  Fehler bei {pid}: {exc} — überspringe")
            errors += 1
            continue

        product_cache[pid] = result
        if result is None:
            not_found += 1
        else:
            fetched += 1

        elapsed = time.monotonic() - start_time
        rate = index / elapsed if elapsed > 0 else 0
        eta = (total - index) / rate if rate > 0 else 0
        eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s" if eta > 0 else "--:--"
        pct = index / total * 100
        print(
            f"\r  {index}/{total} ({pct:.1f}%)  "
            f"{rate:.1f}/s  ETA {eta_str}  "
            f"[ok={fetched} 404={not_found} err={errors}]   ",
            end="",
            flush=True,
        )

        if index % 500 == 0:
            save_json(product_cache, args.cache)

        if args.delay > 0:
            time.sleep(args.delay)

    print()  # Zeilenumbruch nach der letzten Fortschrittszeile

    save_json(product_cache, args.cache)
    print(f"\nFertig: {fetched} gefunden, {not_found} nicht gefunden, {errors} Fehler")
    print(f"Cache gesamt: {len(product_cache)} Einträge in {args.cache}")


if __name__ == "__main__":
    main()
