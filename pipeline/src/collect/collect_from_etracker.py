#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import ssl
import sys
import urllib.error
import urllib.request

API_URL = "https://ws.etracker.com/api/v6/rawdata/download/{}"
DOTENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
RAWDATA_FILENAME_RE = re.compile(r"^rawdata-(\d{4}-\d{2}-\d{2})\.zip$")


def load_dotenv(path: str = DOTENV_PATH) -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as dotenv_file:
        for line in dotenv_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value



def parse_args():
    parser = argparse.ArgumentParser(
        description="Hole die eTracker Rohdaten der letzten X Tage als ZIP-Dateien."
    )
    parser.add_argument(
        "-n",
        "--days",
        type=int,
        default=7,
        help="Anzahl der vergangenen Tage, die heruntergeladen werden sollen (Standard: 7)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Enddatum im Format JJJJ-MM-TT. Standard: heute.",
    )
    parser.add_argument(
        "--export-token",
        type=str,
        default=os.environ.get("ETRACKER_EXPORT_TOKEN"),
        help=(
            "eTracker Export-Token. Alternativ kann die Datei .env im Skriptverzeichnis "
            "genutzt werden oder die Umgebungsvariable ETRACKER_EXPORT_TOKEN."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="raw",
        help="Ausgabeverzeichnis für die ZIP-Dateien. Standard: raw.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Vorhandene Dateien überschreiben.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=True,
        help=(
            "Deaktiviert die SSL-Zertifikatsprüfung. Nur verwenden, wenn die lokale CA-" 
            "Kette das eTracker-Zertifikat nicht prüft."
        ),
    )
    return parser.parse_args()


def to_date(value: str) -> datetime.date:
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Ungültiges Datum '{value}': Verwende JJJJ-MM-TT."
        ) from exc


def build_ssl_context(insecure: bool) -> ssl.SSLContext:
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


def find_downloaded_dates(output_dir: str) -> set[datetime.date]:
    downloaded_dates: set[datetime.date] = set()
    if not os.path.isdir(output_dir):
        return downloaded_dates

    for entry in os.listdir(output_dir):
        match = RAWDATA_FILENAME_RE.match(entry)
        if not match:
            continue
        try:
            downloaded_dates.add(to_date(match.group(1)))
        except argparse.ArgumentTypeError:
            continue
    return downloaded_dates


def download_date(
    date: datetime.date,
    token: str,
    output_dir: str,
    overwrite: bool,
    ssl_context: ssl.SSLContext,
) -> None:
    url = API_URL.format(date.isoformat())
    filename = f"rawdata-{date.isoformat()}.zip"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path) and not overwrite:
        print(f"Überspringe {filename}: Datei existiert bereits.")
        return

    request = urllib.request.Request(url, headers={"X-ET-Token": token})
    print(f"Lade {date.isoformat()} herunter...", end=" ")

    try:
        with urllib.request.urlopen(request, context=ssl_context) as response:
            content = response.read()
    except urllib.error.HTTPError as exc:
        print(f"fehlgeschlagen ({exc.code}) - {exc.reason}")
        return
    except urllib.error.URLError as exc:
        print(f"fehlgeschlagen - {exc.reason}")
        return

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as output_file:
        output_file.write(content)

    print(f"fertig ({len(content)} Bytes)")


def main():
    load_dotenv()
    args = parse_args()
    ssl_context = build_ssl_context(args.insecure)

    if args.days < 1:
        print("Fehler: --days muss größer als 0 sein.")
        sys.exit(1)

    if not args.export_token:
        print(
            "Fehler: Kein Export-Token angegeben. Verwende --export-token oder setze "
            "die Umgebungsvariable ETRACKER_EXPORT_TOKEN."
        )
        sys.exit(1)

    if args.end_date:
        end_date = to_date(args.end_date)
    else:
        end_date = datetime.date.today()

    requested_dates = [
        end_date - datetime.timedelta(days=offset)
        for offset in range(args.days - 1, -1, -1)
    ]
    downloaded_dates = find_downloaded_dates(args.output_dir)

    if args.overwrite:
        missing_dates = requested_dates
        print(
            f"Overwrite aktiv: lade {len(missing_dates)} Tage neu nach "
            f"({requested_dates[0]} bis {requested_dates[-1]})."
        )
    else:
        missing_dates = [date for date in requested_dates if date not in downloaded_dates]
        existing_count = len(requested_dates) - len(missing_dates)
        print(
            f"Bereich {requested_dates[0]} bis {requested_dates[-1]}: "
            f"{existing_count} vorhanden, {len(missing_dates)} fehlen."
        )

    if not missing_dates:
        print("Keine fehlenden Tage gefunden. Es muss nichts heruntergeladen werden.")
        return

    for current_date in missing_dates:
        download_date(
            current_date,
            args.export_token,
            args.output_dir,
            args.overwrite,
            ssl_context,
        )


if __name__ == "__main__":
    main()
