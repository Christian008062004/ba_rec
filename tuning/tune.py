#!/usr/bin/env python3
"""Hyperparameter-Tuning mit Optuna für alle FuxiCTR-Modelle.

Optuna sucht mit TPE (Tree-structured Parzen Estimator) nach optimalen
Hyperparametern. Schlechte Trials werden per Pruning frühzeitig abgebrochen.
Ergebnisse werden in einer SQLite-DB gespeichert — mehrere Prozesse/Nodes
können parallel an derselben Study arbeiten.

Beispiele:
    # Einzelnes Modell tunen (50 Trials)
    python tune.py --model AFN --trials 50

    # Parallel auf zwei HPC-Nodes (gleiche DB, gleicher Study-Name)
    python tune.py --model DCNv2 --trials 25 --study-name DCNv2_v1
    python tune.py --model DCNv2 --trials 25 --study-name DCNv2_v1  # auf Node 2

    # Alle Modelle nacheinander
    python tune.py --all --trials 30

    # Ergebnisse anzeigen
    python tune.py --model AFN --show-results
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import yaml
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.utils import set_logger

TUNING_DIR  = Path(__file__).parent
REPO_DIR    = TUNING_DIR.parent
TRAINING_DIR = REPO_DIR / "training"
DATASET_DIR  = REPO_DIR / "dataset"
DB_PATH      = TUNING_DIR / "tuning.db"
RESULTS_DIR  = TUNING_DIR / "results"


# ---------------------------------------------------------------------------
# Suchräume je Modell
# ---------------------------------------------------------------------------

def _suggest_AFN(trial: optuna.Trial) -> dict:
    return {
        "embedding_dim":       trial.suggest_categorical("embedding_dim", [8, 16, 32]),
        "logarithmic_neurons": trial.suggest_categorical("logarithmic_neurons", [200, 400, 600]),
        "afn_hidden_units":    trial.suggest_categorical("afn_hidden_units",
                                   [[256, 128, 64], [400, 400], [512, 256, 128]]),
        "dnn_hidden_units":    trial.suggest_categorical("dnn_hidden_units",
                                   [[256, 128], [400, 400], [512, 256]]),
        "net_dropout":         trial.suggest_float("net_dropout", 0.0, 0.4),
        "ensemble_dnn":        trial.suggest_categorical("ensemble_dnn", [True, False]),
        "batch_norm":          True,
        "dnn_activations":     "relu",
    }


def _suggest_DCNv2(trial: optuna.Trial) -> dict:
    return {
        "embedding_dim":              trial.suggest_categorical("embedding_dim", [8, 16, 32]),
        "num_cross_layers":           trial.suggest_int("num_cross_layers", 2, 5),
        "parallel_dnn_hidden_units":  trial.suggest_categorical("parallel_dnn_hidden_units",
                                          [[256, 128], [400, 400, 400], [512, 256, 128]]),
        "net_dropout":                trial.suggest_float("net_dropout", 0.0, 0.4),
        "model_structure":            trial.suggest_categorical("model_structure",
                                          ["parallel", "stacked", "stacked_parallel"]),
        "batch_norm":                 True,
        "dnn_activations":            "relu",
        "stacked_dnn_hidden_units":   [],
    }


def _suggest_FinalMLP(trial: optuna.Trial) -> dict:
    hidden = trial.suggest_categorical("hidden_units",
                 [[256, 128, 64], [400, 400, 400], [512, 256, 128]])
    return {
        "embedding_dim":           trial.suggest_categorical("embedding_dim", [8, 16, 32]),
        "mlp1_hidden_units":       hidden,
        "mlp2_hidden_units":       hidden,
        "mlp1_hidden_activations": "relu",
        "mlp2_hidden_activations": "relu",
        "use_fs":                  trial.suggest_categorical("use_fs", [True, False]),
        "fs_hidden_units":         trial.suggest_categorical("fs_hidden_units",
                                       [[256, 128], [400, 400]]),
        "num_heads":               trial.suggest_categorical("num_heads", [1, 2, 4]),
        "net_dropout":             trial.suggest_float("net_dropout", 0.0, 0.4),
        "batch_norm":              True,
        "fs1_context":             [],
        "fs2_context":             [],
    }


def _suggest_FinalNet(trial: optuna.Trial) -> dict:
    hidden = trial.suggest_categorical("hidden_units",
                 [[256, 128, 64], [400, 400, 400], [512, 256, 128]])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    return {
        "embedding_dim":            trial.suggest_categorical("embedding_dim", [8, 16, 32]),
        "block_type":               trial.suggest_categorical("block_type", ["1B", "2B"]),
        "block1_hidden_units":      hidden,
        "block2_hidden_units":      hidden,
        "block1_hidden_activations": "relu",
        "block2_hidden_activations": "relu",
        "block1_dropout":           dropout,
        "block2_dropout":           dropout,
        "use_feature_gating":       trial.suggest_categorical("use_feature_gating", [True, False]),
        "residual_type":            trial.suggest_categorical("residual_type", ["concat", "sum"]),
        "batch_norm":               True,
    }


def _suggest_PNN(trial: optuna.Trial) -> dict:
    return {
        "embedding_dim":      trial.suggest_categorical("embedding_dim", [8, 16, 32]),
        "dnn_hidden_units":   trial.suggest_categorical("dnn_hidden_units",
                                  [[256, 128, 64], [400, 400, 400], [512, 256, 128]]),
        "use_inner_product":  trial.suggest_categorical("use_inner_product", [True, False]),
        "use_outer_product":  trial.suggest_categorical("use_outer_product", [True, False]),
        "net_dropout":        trial.suggest_float("net_dropout", 0.0, 0.4),
        "batch_norm":         True,
        "dnn_activations":    "relu",
    }


def _suggest_EulerNet(trial: optuna.Trial) -> dict:
    return {
        "embedding_dim":  trial.suggest_categorical("embedding_dim", [8, 16, 32]),
        "order":          trial.suggest_int("order", 2, 4),
        "hidden_units":   trial.suggest_categorical("hidden_units",
                              [[256, 128, 64], [400, 400, 400], [512, 256, 128]]),
        "net_dropout":    trial.suggest_float("net_dropout", 0.0, 0.4),
        "batch_norm":     True,
    }


SEARCH_SPACES: dict[str, callable] = {
    "AFN":      _suggest_AFN,
    "DCNv2":    _suggest_DCNv2,
    "FinalMLP": _suggest_FinalMLP,
    "FinalNet": _suggest_FinalNet,
    "PNN":      _suggest_PNN,
    "EulerNet": _suggest_EulerNet,
}


# ---------------------------------------------------------------------------
# Basis-Params laden
# ---------------------------------------------------------------------------

def _load_base_params(model_name: str) -> dict:
    with open(DATASET_DIR / "dataset.yaml") as f:
        dataset_params = yaml.safe_load(f)
    with open(TRAINING_DIR / "train" / "train.yaml") as f:
        train_params = yaml.safe_load(f)
    return {**dataset_params, **train_params}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _make_objective(model_name: str, gpu: int):
    base_params = _load_base_params(model_name)
    suggest_fn  = SEARCH_SPACES[model_name]

    # FeatureMap einmalig laden — wird für alle Trials wiederverwendet
    json_file   = str(DATASET_DIR / base_params["dataset_id"] / "feature_map.json")
    feature_map = FeatureMap(
        dataset_id=base_params["dataset_id"],
        data_dir=str(DATASET_DIR),
    )
    feature_map.load(json_file, base_params)

    sys.path.insert(0, str(TRAINING_DIR / "models" / model_name))
    ModelClass = getattr(importlib.import_module(model_name), model_name)

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            **base_params,
            **suggest_fn(trial),
            "gpu":          gpu,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "model_id":     f"{model_name}_trial{trial.number}",
            # Weniger Epochen beim Tuning — Pruning fängt schlechte Trials früh ab
            "epochs":       min(base_params.get("epochs", 10), 5),
            "verbose":      0,
        }

        train_gen, valid_gen = RankDataLoader(
            feature_map,
            stage="train",
            train_data=trial_params["train_data"],
            valid_data=trial_params["valid_data"],
            batch_size=trial_params["batch_size"],
            shuffle=trial_params["shuffle"],
        ).make_iterator()

        model = ModelClass(feature_map, **trial_params)

        # Pruning-Callback: meldet Validation-AUC nach jeder Epoche an Optuna
        model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=trial_params["epochs"],
            callbacks=[OptunaFuxiCTRCallback(trial)],
        )

        valid_auc = model.best_metric  # FuxiCTR speichert beste Validation-Metrik
        return valid_auc

    return objective


class OptunaFuxiCTRCallback:
    """Meldet Validation-AUC nach jeder Epoche an Optuna und bricht ab wenn pruned."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        self.epoch = 0

    def on_epoch_end(self, logs: dict | None = None):
        self.epoch += 1
        auc = (logs or {}).get("val_AUC", None)
        if auc is None:
            return
        self.trial.report(auc, self.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ---------------------------------------------------------------------------
# Ergebnisse anzeigen
# ---------------------------------------------------------------------------

def _show_results(model_name: str):
    storage = f"sqlite:///{DB_PATH}"
    study_name = model_name
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        print(f"Keine Ergebnisse für {model_name} gefunden.")
        return

    best = study.best_trial
    print(f"\n=== Beste Parameter für {model_name} ===")
    print(f"  Validation-AUC: {best.value:.4f}")
    print("  Parameter:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print(f"\nAlle Trials: {len(study.trials)}")
    print(f"Abgeschlossen: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Gepruned:      {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter-Tuning mit Optuna")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(SEARCH_SPACES.keys()),
                       help="Modell zum Tunen")
    group.add_argument("--all", action="store_true",
                       help="Alle Modelle nacheinander tunen")

    parser.add_argument("--trials",       type=int, default=50,
                        help="Anzahl Trials pro Modell (default: 50)")
    parser.add_argument("--study-name",   default=None,
                        help="Study-Name für paralleles Tuning (default: <Modellname>)")
    parser.add_argument("--gpu",          type=int, default=-1,
                        help="GPU-Index, -1 für CPU (default: -1)")
    parser.add_argument("--show-results", action="store_true",
                        help="Beste gefundene Parameter anzeigen, nicht tunen")
    args = parser.parse_args()

    models = list(SEARCH_SPACES.keys()) if args.all else [args.model]

    if args.show_results:
        for m in models:
            _show_results(m)
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    storage = f"sqlite:///{DB_PATH}"

    # Optuna-Logging dämpfen
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for model_name in models:
        study_name = args.study_name or model_name
        print(f"\n{'='*60}")
        print(f"Tuning: {model_name}  |  Study: {study_name}  |  Trials: {args.trials}")
        print(f"DB: {DB_PATH}")
        print('='*60)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",        # Validation-AUC maximieren
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            load_if_exists=True,         # Bestehende Study fortsetzen (paralleles Tuning)
        )

        objective = _make_objective(model_name, args.gpu)
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

        # Beste Parameter in YAML speichern
        best_yaml = RESULTS_DIR / f"{model_name}_best.yaml"
        with open(best_yaml, "w") as f:
            yaml.dump(
                {"model": model_name, "auc": study.best_value, "params": study.best_params},
                f, default_flow_style=False, allow_unicode=True,
            )

        print(f"\nBestes Ergebnis: AUC={study.best_value:.4f}")
        print(f"Gespeichert in:  {best_yaml}")
        _show_results(model_name)


if __name__ == "__main__":
    main()
