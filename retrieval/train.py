#!/usr/bin/env python3
"""Retrieval-Modell mit RecBole trainieren.

Beispiele:
    python retrieval/train.py --model LightGCN
    python retrieval/train.py --model BPR
    python retrieval/train.py --model NeuMF --config retrieval/neuMF_config.yaml
"""

import argparse
import sys
from recbole.quick_start import run_recbole

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="RecBole-Modellname, z.B. LightGCN, BPR, NeuMF")
    parser.add_argument("--config", default=None, help="Pfad zur Config-YAML (default: retrieval/<model>_config.yaml)")
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]  # RecBole's eigenen Arg-Parser leeren

    config_file = args.config or f"retrieval/{args.model.lower()}_config.yaml"
    run_recbole(model=args.model, dataset="webshop", config_file_list=[config_file])

if __name__ == "__main__":
    main()
