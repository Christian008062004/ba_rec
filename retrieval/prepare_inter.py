#!/usr/bin/env python3
"""Bereitet dataset.inter für GRU4Rec auf.

Behält alle echten Interaktionen (kein syntheticViewProductInList), mit
Timestamp für die Session-Rekonstruktion durch RecBole.
"""

import pandas as pd
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
SOURCE_FILE = REPO_DIR / "dataset" / "model_input" / "staging" / "dataset.inter"
OUTPUT_FILE = Path(__file__).parent / "data" / "webshop" / "webshop.inter"

df = pd.read_csv(SOURCE_FILE, sep="\t", dtype=str)
print(f"Vorher: {len(df):,} Interaktionen")
print(df["event_type:token"].value_counts())

filtered = df[df["event_type:token"] != "syntheticViewProductInList"][
    ["user_id:token", "item_id:token", "timestamp:float"]
]
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
filtered.to_csv(OUTPUT_FILE, sep="\t", index=False)
print(f"\nNachher: {len(filtered):,} Interaktionen")
