#!/usr/bin/env python3
"""Vorhersagen mit trainiertem FuxiCTR-Modell generieren.

Lädt ein trainiertes Checkpoint aus training/checkpoints/ und
erzeugt Click-Through-Score-Vorhersagen auf dem Test-Split.

Beispiel:
    python predict.py --model AFN
    python predict.py --model DCNv2 --split test
    python predict.py --model FinalMLP --split valid --output predictions.csv
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader

INFERENCE_DIR = Path(__file__).parent
REPO_DIR = INFERENCE_DIR.parent
TRAINING_DIR = REPO_DIR / "training"
DATASET_DIR = REPO_DIR / "dataset"


def load_params(model_name: str) -> dict:
    with open(DATASET_DIR / "dataset.yaml") as f:
        dataset_params = yaml.safe_load(f)
    with open(TRAINING_DIR / "train" / "train.yaml") as f:
        train_params = yaml.safe_load(f)
    with open(TRAINING_DIR / "models" / model_name / f"{model_name}.yaml") as f:
        model_params = yaml.safe_load(f)
    return {**dataset_params, **train_params, **model_params}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Vorhersagen mit trainiertem Modell")
    parser.add_argument("--model", required=True, help="Modellname, z.B. AFN, DCNv2, FinalMLP")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"],
                        help="Datensplit für Vorhersage (default: test)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Ausgabepfad für CSV mit Scores (default: inference/<model>_<split>_predictions.csv)")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU-Index, -1 für CPU")
    args = parser.parse_args(argv)

    params = load_params(args.model)
    params["gpu"] = args.gpu

    # Checkpoint-Pfad: training/checkpoints/webshop_data/<MODEL>.model
    checkpoint_dir = str(TRAINING_DIR / "checkpoints" / params.get("dataset_id", "webshop_data"))
    params["model_root"] = checkpoint_dir + "/"
    params["model_id"] = args.model

    # FeatureMap laden
    json_file = str(DATASET_DIR / params["dataset_id"] / "feature_map.json")
    feature_map = FeatureMap(dataset_id=params["dataset_id"], data_dir=str(DATASET_DIR))
    feature_map.load(json_file, params)

    # Modellklasse dynamisch laden
    sys.path.insert(0, str(TRAINING_DIR / "models" / args.model))
    ModelClass = getattr(importlib.import_module(args.model), args.model)
    model = ModelClass(feature_map, **params)
    model.load_weights(f"{checkpoint_dir}/{args.model}.model")
    print(f"Checkpoint geladen: {checkpoint_dir}/{args.model}.model")

    # Daten laden
    split_data = params[f"{args.split}_data"]
    data_gen = RankDataLoader(
        feature_map,
        stage="test",
        test_data=split_data,
        batch_size=params["batch_size"],
        shuffle=False,
    ).make_iterator()

    # Vorhersagen berechnen
    print(f"Berechne Vorhersagen auf Split: {args.split} ...")
    scores = model.predict(data_gen)

    # Ausgabe speichern
    output_path = args.output or INFERENCE_DIR / f"{args.model}_{args.split}_predictions.csv"
    source_csv = Path(split_data)
    if source_csv.exists():
        df = pd.read_csv(source_csv)
        df["score"] = scores
    else:
        df = pd.DataFrame({"score": scores})

    df.to_csv(output_path, index=False)
    print(f"Vorhersagen gespeichert: {output_path}  ({len(df):,} Zeilen)")

    # Kurze Auswertung falls Label vorhanden
    if "label" in df.columns:
        from sklearn.metrics import roc_auc_score, log_loss
        auc = roc_auc_score(df["label"], df["score"])
        ll = log_loss(df["label"], df["score"])
        print(f"\nAUC:     {auc:.4f}")
        print(f"Logloss: {ll:.4f}")


if __name__ == "__main__":
    main()
