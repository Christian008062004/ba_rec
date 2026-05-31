#!/usr/bin/env python3
"""
Two-stage recommendation inference — funktioniert für bekannte UND unbekannte Nutzer.
  Stage 1: SASRec  → top-K Kandidaten aus der Klick-Sequenz (kein user_id nötig)
  Stage 2: AFN CTR → bewertet Kandidaten mit Session-Kontext → top-N

Unbekannte Nutzer (cold-start):
    Kein user_id erforderlich. AFN nutzt Gerät, Land und aktuelle Zeit —
    alles was auch für neue Nutzer aus der laufenden Session bekannt ist.

Usage:
    # Unbekannter Nutzer — nur Sequenz + optionaler Session-Kontext
    python evaluate/predict.py --sequence P-90243228 P-77380870 P-12345678
    python evaluate/predict.py --sequence P-90243228 P-77380870 \\
        --country Germany --device "Mobile phone" --os Android

    # Bekannter Nutzer — Kontext aus Trainingsdaten
    python evaluate/predict.py --sequence P-90243228 P-77380870 \\
        --user_id 10000015002444790544
"""

import argparse
import importlib
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

REPO_DIR    = Path(__file__).parent.parent
DATASET_DIR = REPO_DIR / "dataset"
TRAINING_DIR = REPO_DIR / "training"
sys.path.insert(0, str(TRAINING_DIR / "models" / "AFN"))

SASREC_CKPT   = REPO_DIR / "trained" / "SASRec-May-25-2026_22-31-38.pth"
AFN_CKPT      = REPO_DIR / "saved" / "AFN.model"
FEATURE_VOCAB = DATASET_DIR / "feature_vocab.json"

_FLOAT_FEATS = {"price", "geo_latitude", "geo_longitude"}


# ── model loading (cached globally) ──────────────────────────────────────────

_sasrec_model  = None
_sasrec_dataset = None
_afn_model     = None
_feature_map   = None
_item_vocab_afn  = None   # token_str  -> afn_encoded_id
_user_vocab_afn  = None   # token_str  -> afn_encoded_id
_recbole_to_afn  = None   # recbole internal id -> afn encoded id
_item_features   = None   # afn_item_id -> {brand, category1-4, price}
_user_ctx_lookup = None   # afn_user_id -> context features
_ctx_vocab       = None   # feature_name -> {label -> encoded_int}


def _load_all():
    global _sasrec_model, _sasrec_dataset, _afn_model, _feature_map
    global _item_vocab_afn, _user_vocab_afn, _recbole_to_afn
    global _item_features, _user_ctx_lookup, _ctx_vocab

    # ── vocabulary ──────────────────────────────────────────────────────────
    print("  [1/5] Lade Vokabular ...")
    with open(FEATURE_VOCAB) as f:
        vocab = json.load(f)
    _item_vocab_afn = vocab["item_id"]
    _user_vocab_afn = vocab["user_id"]
    _ctx_vocab = {k: vocab[k] for k in
                  ["geo_country", "device_type", "device_os",
                   "device_manufacturer", "device_browser", "device_type_detail"]
                  if k in vocab}

    # ── item feature lookup ─────────────────────────────────────────────────
    print("  [2/5] Lade Item-Features ...")
    with open(REPO_DIR / "trained" / "item_features.json") as f:
        raw = json.load(f)
    _item_features = {int(k): v for k, v in raw.items()}

    # ── user context lookup ─────────────────────────────────────────────────
    print("  [3/5] Lade User-Kontext ...")
    with open(REPO_DIR / "trained" / "user_context.json") as f:
        raw = json.load(f)
    _user_ctx_lookup = {int(k): v for k, v in raw.items()}

    # ── SASRec ──────────────────────────────────────────────────────────────
    print("  [4/5] Lade SASRec ...")
    from recbole.model.sequential_recommender import SASRec

    sasrec_vocab_path = REPO_DIR / "trained" / "sasrec_item_vocab.json"
    with open(sasrec_vocab_path) as f:
        sv = json.load(f)

    # SASRec braucht beim Initialisieren nur dataset.num(item_field) → Anzahl Items.
    # Alles andere kommt aus config. Kein create_dataset() mehr nötig.
    class _MinimalDataset:
        def num(self, field):
            return sv["n_items"]

    ckpt = torch.load(SASREC_CKPT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    config["device"] = torch.device("cpu")

    _sasrec_model = SASRec(config, _MinimalDataset()).to("cpu")
    _sasrec_model.load_state_dict(ckpt["state_dict"])
    _sasrec_model.eval()

    # Leichtgewichtiges Vocab-Objekt für Token-Lookups
    _sasrec_dataset = type("SASRecVocab", (), {
        "token2id":               lambda self, field, tok: sv["token2id"].get(tok, 0),
        "id2token":               lambda self, field, iid: sv["id2token"].get(str(iid), "[UNK]"),
        "iid_field":              sv["iid_field"],
        "item_id_list_field":     sv["item_list_field"],
        "item_list_length_field": sv["item_list_length_field"],
        "config":                 {"MAX_ITEM_LIST_LENGTH": sv["MAX_ITEM_LIST_LENGTH"]},
    })()

    n_items = sv["n_items"]
    _recbole_to_afn = np.zeros(n_items, dtype=np.int64)
    for rb_id in range(1, n_items):
        tok = sv["id2token"].get(str(rb_id), "")
        _recbole_to_afn[rb_id] = _item_vocab_afn.get(tok, 0)

    # ── AFN ─────────────────────────────────────────────────────────────────
    print("  [5/5] Lade AFN ...")
    with open(DATASET_DIR / "dataset.yaml") as f:
        dp = yaml.safe_load(f)
    with open(TRAINING_DIR / "train" / "train.yaml") as f:
        tp = yaml.safe_load(f)
    with open(TRAINING_DIR / "models" / "AFN" / "AFN.yaml") as f:
        mp = yaml.safe_load(f)
    params = {**dp, **tp, **mp}
    params["gpu"] = -1
    params["model_id"] = AFN_CKPT.stem

    from fuxictr.features import FeatureMap
    _feature_map = FeatureMap(dataset_id=params["dataset_id"], data_dir=str(DATASET_DIR))
    _feature_map.load(str(DATASET_DIR / params["dataset_id"] / "feature_map.json"), params)

    AFN = getattr(importlib.import_module("AFN"), "AFN")
    _afn_model = AFN(_feature_map, **params)
    _afn_model.load_weights(str(AFN_CKPT))
    _afn_model.eval()
    print("  Modelle bereit.\n")


# ── inference ─────────────────────────────────────────────────────────────────

def recommend(
    sequence: list[str],
    user_id: str | None = None,
    country: str | None = None,
    device: str | None = None,
    os: str | None = None,
    candidates: int = 20,
    topk: int = 5,
) -> list[dict]:
    """
    Two-stage recommendation für bekannte und unbekannte Nutzer.

    Args:
        sequence  : Klick-Sequenz als Produkt-Tokens (ältestes zuerst)
        user_id   : Optional — bekannter Nutzer, Kontext aus Trainingsdaten
        country   : Optional — Ländername, z.B. "Germany"  (für unbekannte Nutzer)
        device    : Optional — Gerätetyp, z.B. "Mobile phone" / "Desktop" / "Tablet"
        os        : Optional — Betriebssystem, z.B. "Android" / "iOS" / "Windows"
        candidates: SASRec Kandidatenpool-Größe (default: 20)
        topk      : Anzahl finaler Empfehlungen (default: 5)

    Returns:
        list of dicts: rank, item_id, ctr_score, sasrec_rank
    """
    if _sasrec_model is None:
        _load_all()

    iid_field = _sasrec_dataset.iid_field

    # ── resolve sequence to RecBole internal ids ────────────────────────────
    rb_seq = []
    skipped = []
    for tok in sequence:
        rb_id = _sasrec_dataset.token2id(iid_field, tok)
        if rb_id != 0:          # 0 = OOV / padding in RecBole
            rb_seq.append(rb_id)
        else:
            skipped.append(tok)

    if skipped:
        print(f"  [warn] {len(skipped)} unknown item(s) skipped: {skipped}")

    if not rb_seq:
        raise ValueError("No known items in sequence — cannot generate recommendations.")

    max_len = _sasrec_dataset.config["MAX_ITEM_LIST_LENGTH"]
    rb_seq  = rb_seq[-max_len:]          # keep most recent max_len items

    # ── Stage 1: SASRec full-sort prediction ────────────────────────────────
    seq_tensor = torch.tensor(rb_seq, dtype=torch.long).unsqueeze(0)       # [1, seq_len]
    len_tensor = torch.tensor([len(rb_seq)], dtype=torch.long)

    from recbole.data.interaction import Interaction
    inter = Interaction({
        _sasrec_dataset.item_id_list_field:   seq_tensor,
        _sasrec_dataset.item_list_length_field: len_tensor,
    })

    with torch.no_grad():
        scores = _sasrec_model.full_sort_predict(inter)   # [1, n_items]

    scores = scores.squeeze(0).numpy()
    scores[0] = -np.inf          # mask padding (index 0)
    scores[rb_seq[-1]] = -np.inf  # mask aktuell betrachtetes Produkt (letztes in Sequenz)

    topk_rb = np.argsort(-scores)[:candidates]   # RecBole ids, best first

    # ── map candidates to AFN ids ────────────────────────────────────────────
    sasrec_ranks  = []    # SASRec rank of each valid candidate
    cand_afn_ids  = []
    cand_tokens   = []
    for sasrec_rank, rb_id in enumerate(topk_rb, start=1):
        tok    = _sasrec_dataset.id2token(iid_field, int(rb_id))
        afn_id = _item_vocab_afn.get(tok, 0)
        if afn_id > 0:
            cand_afn_ids.append(afn_id)
            cand_tokens.append(tok)
            sasrec_ranks.append(sasrec_rank)

    if not cand_afn_ids:
        raise ValueError("None of the SASRec candidates are in the AFN item vocabulary.")

    # ── Session-Kontext aufbauen ─────────────────────────────────────────────
    # Basis: aktuelle Uhrzeit (immer bekannt), alles andere 0 (unbekannt)
    now = datetime.now()
    ctx = {
        "geo_country":         0,
        "geo_latitude":        0.0,
        "geo_longitude":       0.0,
        "device_type":         0,
        "device_os":           0,
        "device_model":        0,
        "device_manufacturer": 0,
        "device_browser":      0,
        "device_type_detail":  0,
        "day_of_week":         now.weekday() + 1,  # Mo=1 … So=7
        "hour_of_day":         now.hour,
        "month":               now.month,
        "is_holiday":          0,
    }
    afn_user_id = 0  # 0 = unbekannter Nutzer (OOV-Embedding)

    if user_id is not None:
        # Bekannter Nutzer: Geo/Device aus Trainingsdaten laden
        afn_uid = _user_vocab_afn.get(str(user_id), 0)
        if afn_uid > 0 and afn_uid in _user_ctx_lookup:
            ctx.update(_user_ctx_lookup[afn_uid])
        else:
            print(f"  [warn] user_id '{user_id}' unbekannt — nutze Standard-Kontext.")
        afn_user_id = afn_uid
    else:
        # Unbekannter Nutzer: explizit übergebene Session-Infos enkodieren
        if country:
            ctx["geo_country"] = _ctx_vocab.get("geo_country", {}).get(country, 0)
            if ctx["geo_country"] == 0:
                print(f"  [warn] Land '{country}' unbekannt — wird ignoriert.")
        if device:
            ctx["device_type"] = _ctx_vocab.get("device_type", {}).get(device, 0)
            if ctx["device_type"] == 0:
                print(f"  [warn] Gerät '{device}' unbekannt. Gültig: "
                      f"{list(_ctx_vocab.get('device_type', {}).keys())}")
        if os:
            ctx["device_os"] = _ctx_vocab.get("device_os", {}).get(os, 0)
            if ctx["device_os"] == 0:
                print(f"  [warn] OS '{os}' unbekannt. Gültig: "
                      f"{list(_ctx_vocab.get('device_os', {}).keys())}")

    # Zeit immer live überschreiben (auch bei bekanntem Nutzer)
    ctx["day_of_week"] = now.weekday() + 1
    ctx["hour_of_day"] = now.hour
    ctx["month"]       = now.month

    # ── Stage 2: AFN CTR scoring ─────────────────────────────────────────────
    n = len(cand_afn_ids)
    batch = {}
    batch["user_id"] = np.full(n, afn_user_id, dtype=np.int64)
    batch["item_id"] = np.array(cand_afn_ids, dtype=np.int64)

    # user context (same for all candidates)
    for k, v in ctx.items():
        dtype = np.float32 if k in _FLOAT_FEATS else np.int64
        batch[k] = np.full(n, v, dtype=dtype)

    # per-item features
    for feat in ["brand", "category1", "category2", "category3", "category4", "price"]:
        dtype = np.float32 if feat in _FLOAT_FEATS else np.int64
        batch[feat] = np.array(
            [_item_features.get(iid, {}).get(feat, 0) for iid in cand_afn_ids],
            dtype=dtype,
        )

    # Kategorische Werte auf gültigen Bereich begrenzen (out-of-range → 0)
    vocab_sizes = {
        name: spec["vocab_size"]
        for name, spec in _feature_map.features.items()
        if spec.get("type") == "categorical"
    }

    tensor_batch = {}
    for feat_name, arr in batch.items():
        if arr.dtype in (np.float32, np.float64):
            tensor_batch[feat_name] = torch.FloatTensor(arr)
        else:
            if feat_name in vocab_sizes:
                arr = np.where(arr < vocab_sizes[feat_name], arr, 0)
            tensor_batch[feat_name] = torch.LongTensor(arr)

    _afn_model.eval()
    with torch.no_grad():
        out = _afn_model(tensor_batch)
    ctr_scores = out["y_pred"].cpu().numpy().reshape(-1)

    # ── SASRec-only top-K (vor AFN Re-Ranking) ───────────────────────────────
    sasrec_only = [cand_tokens[i] for i in range(min(topk, len(cand_tokens)))]

    # ── rank by CTR score and return top-N ───────────────────────────────────
    ranked_idx = np.argsort(-ctr_scores)
    pipeline = []
    for final_rank, idx in enumerate(ranked_idx[:topk], start=1):
        pipeline.append({
            "rank":        final_rank,
            "item_id":     cand_tokens[idx],
            "ctr_score":   float(ctr_scores[idx]),
            "sasrec_rank": sasrec_ranks[idx],
        })

    return {"pipeline": pipeline, "sasrec_only": sasrec_only}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage recommendation: SASRec candidates → AFN CTR re-ranking"
    )
    parser.add_argument(
        "--sequence", nargs="+", required=True,
        help="Ordered product tokens (oldest first), e.g. P-90243228 P-77380870 P-12345678"
    )
    parser.add_argument("--user_id",    default=None,
                        help="Bekannter Nutzer-Token (optional)")
    parser.add_argument("--country",    default=None,
                        help='Land für unbekannte Nutzer, z.B. "Germany"')
    parser.add_argument("--device",     default=None,
                        help='Gerät, z.B. "Mobile phone" / "Desktop" / "Tablet"')
    parser.add_argument("--os",         default=None,
                        help='Betriebssystem, z.B. "Android" / "iOS" / "Windows"')
    parser.add_argument("--candidates", type=int, default=20,
                        help="SASRec Kandidatenpool-Größe (default: 20)")
    parser.add_argument("--topk",       type=int, default=5,
                        help="Finale Empfehlungen (default: 5)")
    args = parser.parse_args()

    print("Lade Modelle ...")
    results = recommend(
        sequence=args.sequence,
        user_id=args.user_id,
        country=args.country,
        device=args.device,
        os=args.os,
        candidates=args.candidates,
        topk=args.topk,
    )

    print(f"\nTop-{args.topk} Empfehlungen für Sequenz: {args.sequence}\n")
    print(f"{'Rang':<6} {'Item-ID':<20} {'CTR-Score':>10}  {'SASRec-Rang':>12}")
    print("-" * 54)
    for r in results["pipeline"]:
        print(f"{r['rank']:<6} {r['item_id']:<20} {r['ctr_score']:>10.4f}  {r['sasrec_rank']:>12}")

    print(f"\nSASRec-only Top-{args.topk}: {results['sasrec_only']}")


if __name__ == "__main__":
    main()
