"""FuxiCTR-spezifische Preprocessing-Schritte: vocab_sizes, feature_map, feature_vocab, NPZ-Konvertierung."""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_DEFAULT_YAML = str(Path(__file__).parent.parent.parent.parent.parent / "dataset" / "dataset.yaml")


def load_params(dataset_yaml: str = _DEFAULT_YAML) -> dict:
    with open(dataset_yaml) as f:
        return yaml.safe_load(f)


def fix_vocab_sizes(dataset_yaml: str = _DEFAULT_YAML) -> None:
    """Recalculates vocab_sizes in dataset.yaml from the actual CSV data."""
    with open(dataset_yaml) as f:
        config = yaml.safe_load(f)

    splits = [config["train_data"], config["valid_data"], config["test_data"]]
    df = pd.concat([pd.read_csv(s) for s in splits], ignore_index=True)

    updated = 0
    for spec in config["feature_specs"]:
        if spec.get("type") != "categorical":
            continue
        name = spec["name"]
        correct = int(df[name].max()) + 1
        old = spec.get("vocab_size", 0)
        if correct != old:
            logging.info(f"  vocab_size {name}: {old} -> {correct}")
            spec["vocab_size"] = correct
            updated += 1

    with open(dataset_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logging.info(f"fix_vocab_sizes: {updated} vocab_sizes aktualisiert.")


def build_feature_map_json(params: dict) -> None:
    """Generates feature_map.json required by FuxiCTR's FeatureMap.load()."""
    label_names = []
    features = []
    for spec in params.get("feature_specs", []):
        spec = dict(spec)
        name = spec.pop("name")
        if spec.get("dtype") == "float" and "type" not in spec:
            label_names.append(name)
            spec["type"] = "meta"
        if spec.get("active", True):
            features.append({name: spec})
    data = {
        "dataset_id": params["dataset_id"],
        "labels": label_names,
        "total_features": 0,
        "input_length": 0,
        "features": features,
    }
    out_dir = os.path.join(params["data_root"], params["dataset_id"])
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "feature_map.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Generated feature_map.json at {json_path}")



def prepare_pretrained_emb(params: dict) -> None:
    """Converts .npy embedding files to .npz format expected by FuxiCTR."""
    for spec in params.get("feature_specs", []):
        npz_name = spec.get("pretrained_emb", "")
        if not npz_name.endswith(".npz"):
            continue
        npz_path = os.path.join(params["data_root"], npz_name)
        if not os.path.exists(npz_path):
            npy_path = npz_path.replace(".npz", ".npy")
            emb = np.load(npy_path)
            np.savez(npz_path, key=np.arange(len(emb)), value=emb)
            logging.info(f"Converted {npy_path} -> {npz_path}")


def prepare_npz_data(params: dict) -> None:
    """Converts CSV splits to NPZ format expected by FuxiCTR's RankDataLoader.

    Fills NaN in numeric columns and binarizes labels (score > 0 -> 1).
    Skips splits that are already converted.
    """
    dtype_map = {
        spec["name"]: np.float32 if spec.get("dtype") == "float" else np.int32
        for spec in params.get("feature_specs", [])
    }
    numeric_cols = [s["name"] for s in params.get("feature_specs", []) if s.get("type") == "numeric"]
    label_cols = [
        s["name"] for s in params.get("feature_specs", [])
        if s.get("dtype") == "float" and "type" not in s
    ]
    for split in ["train", "valid", "test"]:
        csv_path = params[f"{split}_data"]
        npz_path = csv_path + ".npz"
        if os.path.exists(npz_path):
            logging.info(f"Already exists, skipping: {npz_path}")
            continue
        df = pd.read_csv(csv_path)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        for col in label_cols:
            df[col] = (df[col] > 0).astype(np.float32)
        arrays = {col: df[col].values.astype(dtype_map.get(col, np.float32)) for col in df.columns}
        np.savez(npz_path, **arrays)
        logging.info(f"Converted {csv_path} -> {npz_path}")


def run(dataset_yaml: str = _DEFAULT_YAML) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fix_vocab_sizes(dataset_yaml)
    params = load_params(dataset_yaml)
    prepare_pretrained_emb(params)
    prepare_npz_data(params)
    build_feature_map_json(params)
    logging.info("FuxiCTR-Prep abgeschlossen.")


if __name__ == "__main__":
    run()
