#!/usr/bin/env python3
"""
Grid-Sweep: evaluiert alle (candidates x topk)-Kombinationen.

Lädt SASRec + AFN einmal, iteriert dann über alle Kombinationen und
schreibt Ergebnisse inkrementell nach evaluate/results/.

Output:
    evaluate/results/sweep_<timestamp>.csv   -- maschinenlesbar
    evaluate/results/sweep_<timestamp>.log   -- menschenlesbar (identisch mit stdout)

Usage:
    python evaluate/sweep.py
    python evaluate/sweep.py --candidates 20 50 100 200 --topk 1 5 10 20
    python evaluate/sweep.py --max_users 500   # Smoke-Test
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))

from predict import recommend, _load_all  # noqa: E402


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CANDIDATES = [20, 50, 100, 200]
DEFAULT_TOPK       = [1, 5, 10, 20]
DEFAULT_TESTSET    = EVAL_DIR / "testset.csv"
RESULTS_DIR        = EVAL_DIR / "results"


# ── Metriken ──────────────────────────────────────────────────────────────────

def hit_at_k(ranked, gt, k):
    return int(gt in ranked[:k])

def ndcg_at_k(ranked, gt, k):
    for rank, item in enumerate(ranked[:k], start=1):
        if item == gt:
            return 1.0 / math.log2(rank + 1)
    return 0.0


# ── Tee: stdout + Logdatei gleichzeitig ───────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


# ── Eine Kombination evaluieren ───────────────────────────────────────────────

def run_combination(
    df: pd.DataFrame,
    candidates: int,
    topk: int,
    log_interval: int = 1000,
) -> dict:
    hits_pipe,   ndcgs_pipe   = [], []
    hits_sasrec, ndcgs_sasrec = [], []
    n_skipped = 0
    t0 = time.time()

    print(f"\n[{_ts()}]  candidates={candidates:>4}  topk={topk:>2}  "
          f"({len(df):,} User)", flush=True)
    print(f"  {'User':>8}   {'Hit@'+str(topk):>7} {'NDCG@'+str(topk):>8}   "
          f"{'[SASRec Hit]':>13} {'[SASRec NDCG]':>13}   {'elapsed':>8}")
    print("  " + "-" * 74)

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
        ndcgs_pipe.append(ndcg_at_k(pipeline_items, ground_truth, topk))
        hits_sasrec.append(hit_at_k(sasrec_items, ground_truth, topk))
        ndcgs_sasrec.append(ndcg_at_k(sasrec_items, ground_truth, topk))

        if i % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"  {i:>8,} / {len(df):,}   "
                f"{np.mean(hits_pipe):>7.4f} {np.mean(ndcgs_pipe):>8.4f}   "
                f"{np.mean(hits_sasrec):>13.4f} {np.mean(ndcgs_sasrec):>13.4f}   "
                f"{elapsed:>7.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    n_eval  = len(hits_pipe)

    pipe_hit  = float(np.mean(hits_pipe))   if hits_pipe   else 0.0
    pipe_ndcg = float(np.mean(ndcgs_pipe))  if ndcgs_pipe  else 0.0
    sr_hit    = float(np.mean(hits_sasrec)) if hits_sasrec else 0.0
    sr_ndcg   = float(np.mean(ndcgs_sasrec))if ndcgs_sasrec else 0.0

    print(f"\n  Fertig: {n_eval:,} evaluiert, {n_skipped:,} übersprungen  ({elapsed:.0f}s)")
    print(f"  Pipeline  Hit@{topk}={pipe_hit:.4f}  NDCG@{topk}={pipe_ndcg:.4f}")
    print(f"  SASRec    Hit@{topk}={sr_hit:.4f}  NDCG@{topk}={sr_ndcg:.4f}")
    print(f"  Delta     Hit@{topk}={pipe_hit-sr_hit:+.4f}  NDCG@{topk}={pipe_ndcg-sr_ndcg:+.4f}",
          flush=True)

    return {
        "candidates":    candidates,
        "topk":          topk,
        "n_evaluated":   n_eval,
        "n_skipped":     n_skipped,
        "pipeline_hit":  pipe_hit,
        "pipeline_ndcg": pipe_ndcg,
        "sasrec_hit":    sr_hit,
        "sasrec_ndcg":   sr_ndcg,
        "delta_hit":     pipe_hit - sr_hit,
        "delta_ndcg":    pipe_ndcg - sr_ndcg,
        "duration_s":    round(elapsed, 1),
    }


def _ts():
    return datetime.now().strftime("%H:%M:%S")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, nargs="+", default=DEFAULT_CANDIDATES,
                        help=f"Kandidatenpool-Größen (default: {DEFAULT_CANDIDATES})")
    parser.add_argument("--topk",       type=int, nargs="+", default=DEFAULT_TOPK,
                        help=f"Finale Empfehlungsanzahlen (default: {DEFAULT_TOPK})")
    parser.add_argument("--testset",    type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--max_users",  type=int,  default=None,
                        help="Nur erste N User (Smoke-Test)")
    args = parser.parse_args()

    # Kombinationen: topk muss <= candidates sein
    combos = [(c, k) for c, k in product(sorted(args.candidates), sorted(args.topk)) if k <= c]
    if not combos:
        print("Keine gültigen Kombinationen (topk <= candidates nötig).")
        sys.exit(1)

    # Output-Verzeichnis
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"sweep_{stamp}.csv"
    log_path = RESULTS_DIR / f"sweep_{stamp}.log"

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"[{_ts()}] Sweep gestartet: {len(combos)} Kombinationen", flush=True)
    print(f"  candidates: {sorted(args.candidates)}")
    print(f"  topk:       {sorted(args.topk)}")
    print(f"  testset:    {args.testset}")
    print(f"  CSV:        {csv_path}")
    print(f"  Log:        {log_path}")
    if args.max_users:
        print(f"  max_users:  {args.max_users}")

    # Testdaten laden
    df = pd.read_csv(args.testset, dtype=str)
    print(f"\n[{_ts()}] Testdatensatz geladen: {len(df):,} Samples")
    if args.max_users:
        df = df.head(args.max_users)
        print(f"  Limitiert auf {args.max_users:,} User")

    # Modelle einmal laden
    print(f"\n[{_ts()}] Lade Modelle ...")
    _load_all()

    # CSV initialisieren
    fieldnames = [
        "candidates", "topk", "n_evaluated", "n_skipped",
        "pipeline_hit", "pipeline_ndcg",
        "sasrec_hit",   "sasrec_ndcg",
        "delta_hit",    "delta_ndcg",
        "duration_s",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    # Sweep
    t_total = time.time()
    all_results = []

    for idx, (cands, k) in enumerate(combos, start=1):
        print(f"\n[{_ts()}] -- Kombination {idx}/{len(combos)} --------------------------",
              flush=True)
        result = run_combination(df, cands, k)
        all_results.append(result)

        writer.writerow(result)
        csv_file.flush()

    # Abschluss-Tabelle
    total_elapsed = time.time() - t_total
    w = 74
    print(f"\n\n[{_ts()}] {'='*w}")
    print(f"  Sweep abgeschlossen: {len(all_results)} Kombinationen  "
          f"(gesamt {total_elapsed/60:.1f} min)")
    print(f"  {'candidates':>10} {'topk':>5}  "
          f"{'Pipe Hit':>9} {'Pipe NDCG':>10}  "
          f"{'SR Hit':>8} {'SR NDCG':>9}  "
          f"{'dHit':>7} {'dNDCG':>8}")
    print(f"  {'-'*w}")
    for r in all_results:
        print(
            f"  {r['candidates']:>10} {r['topk']:>5}  "
            f"{r['pipeline_hit']:>9.4f} {r['pipeline_ndcg']:>10.4f}  "
            f"{r['sasrec_hit']:>8.4f} {r['sasrec_ndcg']:>9.4f}  "
            f"{r['delta_hit']:>+7.4f} {r['delta_ndcg']:>+8.4f}"
        )
    print(f"  {'='*w}")
    print(f"\n  Ergebnisse: {csv_path}", flush=True)

    csv_file.close()
    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"Fertig. Ergebnisse in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
