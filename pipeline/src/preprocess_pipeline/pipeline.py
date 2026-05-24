#!/usr/bin/env python3
"""Preprocess-Pipeline: eTracker-Rohdaten → dataset/train.csv, valid.csv, test.csv + item_embedding_matrix.npz"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / ".env")
sys.path.insert(0, str(ROOT_DIR / "src"))

from preprocess_pipeline.steps.build_staging_base import (
    DEFAULT_OUTPUT_FILE,
    build_staging_base,
)
from preprocess_pipeline.steps.enrich_parent_ids import enrich_parent_ids
from preprocess_pipeline.steps.enrich_item_api import enrich_item_api
from preprocess_pipeline.steps.enrich_user_geo import enrich_user_geo
from preprocess_pipeline.steps.build_embedding_input import build_embedding_input
from preprocess_pipeline.steps.build_embeddings import build_embeddings
from preprocess_pipeline.steps.build_csv_splits import build_csv_splits
from preprocess_pipeline.steps.fuxictr_prep import (
    fix_vocab_sizes,
    build_feature_map_json,
    prepare_feature_vocab,
    load_params,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess-Pipeline")
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR / "raw")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--min-user-interactions", type=int, default=2)
    parser.add_argument("--min-item-interactions", type=int, default=2)
    parser.add_argument("--max-zip-files", type=int, default=0)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-max-rows", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    row_limit = int(os.environ.get("DEV_ROW_LIMIT", "0"))
    if row_limit:
        print(f"[DEV] DEV_ROW_LIMIT={row_limit} — nur erste {row_limit:,} Interaktionen werden verarbeitet")

    print("=== Schritt 1: eTracker-Daten → Basis-Staging ===")
    build_staging_base(
        input_dir=args.input_dir,
        output_path=args.output,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_zip_files=args.max_zip_files,
        test_mode=args.test_mode,
        test_max_rows=args.test_max_rows,
    )

    print("=== Schritt 2: Variant-IDs → parentProductNumber ===")
    enrich_parent_ids(inter_path=args.output)

    print("=== Schritt 3: .item mit API-Daten anreichern ===")
    enrich_item_api(item_path=args.output.with_suffix(".item"))

    print("=== Schritt 4: .user mit Geokoordinaten anreichern ===")
    enrich_user_geo(user_path=args.output.with_suffix(".user"), geocode_enabled=True)

    print("=== Schritt 5: Embedding-Input aufbauen ===")
    build_embedding_input()

    print("=== Schritt 6: Embeddings → embeddings.npz ===")
    build_embeddings()

    print("=== Schritt 7: Staging → CSVs + item_embedding_matrix.npz ===")
    build_csv_splits(row_limit=row_limit)

    print("=== Schritt 9: FuxiCTR-Vorbereitung ===")
    fix_vocab_sizes()
    params = load_params()
    build_feature_map_json(params)
    prepare_feature_vocab(params)

    print("=== Pipeline abgeschlossen ===")


if __name__ == "__main__":
    main()
