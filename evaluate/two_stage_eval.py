#!/usr/bin/env python3
"""
Benchmark der zweistufigen Pipeline gegen evaluate/testset.csv.

  Stage 1: SASRec  → top-K (=20) Kandidaten
  Stage 2: AFN CTR → Re-Ranking → top-5
  Metriken: Hit@5, Recall@5, NDCG@5

Usage:
    python evaluate/two_stage_eval.py
    python evaluate/two_stage_eval.py --testset evaluate/testset.csv --candidates 20 --topk 5
    python evaluate/two_stage_eval.py --max_users 1000   # schneller Smoke-Test
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))   # damit predict.py importierbar ist

from predict import recommend, _load_all          # noqa: E402  (nach sys.path-Patch)

DEFAULT_TESTSET = Path(__file__).parent / "testset.csv"


# ── Metriken ──────────────────────────────────────────────────────────────────

def hit_at_k(ranked, gt, k):
    return int(gt in ranked[:k])

def recall_at_k(ranked, gt, k):
    return hit_at_k(ranked, gt, k)

def ndcg_at_k(ranked, gt, k):
    for rank, item in enumerate(ranked[:k], start=1):
        if item == gt:
            return 1.0 / math.log2(rank + 1)
    return 0.0


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(testset_path: Path, candidates: int, topk: int, max_users: int | None):
    df = pd.read_csv(testset_path, dtype=str)
    print(f"Testdatensatz: {testset_path}  ({len(df):,} Samples)")

    if max_users:
        df = df.head(max_users)
        print(f"  Limitiert auf {max_users:,} Nutzer (--max_users)")

    print("\nLade Modelle ...")
    _load_all()

    hits_pipe,    recalls_pipe,    ndcgs_pipe    = [], [], []
    hits_sasrec,  recalls_sasrec,  ndcgs_sasrec  = [], [], []
    n_skipped = 0

    print(f"\nStarte Evaluation  (candidates={candidates}, topk={topk}) ...\n")
    print(f"  {'Nutzer':>8}   "
          f"{'Hit@'+str(topk):>8} {'Recall@'+str(topk):>10} {'NDCG@'+str(topk):>8}   "
          f"{'[SASRec]':>10} {'[SASRec]':>10} {'[SASRec]':>8}")
    print("  " + "-" * 80)

    for i, row in enumerate(df.itertuples(index=False), start=1):
        sequence     = row.sequence.split("|")
        ground_truth = row.ground_truth
        user_id      = row.user_id

        try:
            results = recommend(
                sequence=sequence,
                user_id=user_id,
                candidates=candidates,
                topk=topk,
            )
        except ValueError:
            n_skipped += 1
            continue

        pipeline_items = [r["item_id"] for r in results["pipeline"]]
        sasrec_items   = results["sasrec_only"]

        hits_pipe.append(hit_at_k(pipeline_items, ground_truth, topk))
        recalls_pipe.append(recall_at_k(pipeline_items, ground_truth, topk))
        ndcgs_pipe.append(ndcg_at_k(pipeline_items, ground_truth, topk))

        hits_sasrec.append(hit_at_k(sasrec_items, ground_truth, topk))
        recalls_sasrec.append(recall_at_k(sasrec_items, ground_truth, topk))
        ndcgs_sasrec.append(ndcg_at_k(sasrec_items, ground_truth, topk))

        if i % 1000 == 0:
            print(f"  {i:>7,} / {len(df):,}   "
                  f"{np.mean(hits_pipe):>8.4f} {np.mean(recalls_pipe):>10.4f} {np.mean(ndcgs_pipe):>8.4f}   "
                  f"{np.mean(hits_sasrec):>10.4f} {np.mean(recalls_sasrec):>10.4f} {np.mean(ndcgs_sasrec):>8.4f}")

    n_eval = len(hits_pipe)
    w = 57
    print(f"\n{'='*w}")
    print(f"  Evaluiert   : {n_eval:,} Nutzer  |  Uebersprungen: {n_skipped:,}")
    print(f"  Kandidaten  : SASRec top-{candidates} -> AFN top-{topk}")
    print(f"{'='*w}")
    print(f"  {'Metrik':<14}  {'SASRec+AFN':>12}  {'SASRec only':>12}  {'Differenz':>10}")
    print(f"  {'-'*52}")
    for label, pipe_vals, sr_vals in [
        (f"Hit@{topk}",    hits_pipe,    hits_sasrec),
        (f"Recall@{topk}", recalls_pipe, recalls_sasrec),
        (f"NDCG@{topk}",   ndcgs_pipe,   ndcgs_sasrec),
    ]:
        p  = float(np.mean(pipe_vals))
        s  = float(np.mean(sr_vals))
        diff = p - s
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<14}  {p:>12.4f}  {s:>12.4f}  {sign}{diff:>9.4f}")
    print(f"{'='*w}\n")

    return {
        f"pipeline_Hit@{topk}":    float(np.mean(hits_pipe)),
        f"pipeline_Recall@{topk}": float(np.mean(recalls_pipe)),
        f"pipeline_NDCG@{topk}":   float(np.mean(ndcgs_pipe)),
        f"sasrec_Hit@{topk}":      float(np.mean(hits_sasrec)),
        f"sasrec_Recall@{topk}":   float(np.mean(recalls_sasrec)),
        f"sasrec_NDCG@{topk}":     float(np.mean(ndcgs_sasrec)),
        "n_evaluated": n_eval,
        "n_skipped":   n_skipped,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset",    type=Path,  default=DEFAULT_TESTSET,
                        help=f"Pfad zur testset.csv (default: {DEFAULT_TESTSET})")
    parser.add_argument("--candidates", type=int,   default=20,
                        help="SASRec Kandidatenpool-Größe (default: 20)")
    parser.add_argument("--topk",       type=int,   default=5,
                        help="Finale Empfehlungen nach AFN-Re-Ranking (default: 5)")
    parser.add_argument("--max_users",  type=int,   default=None,
                        help="Nur erste N Nutzer evaluieren (default: alle)")
    args = parser.parse_args()

    evaluate(args.testset, args.candidates, args.topk, args.max_users)
