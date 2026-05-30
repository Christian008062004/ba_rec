#!/usr/bin/env python3
"""Cascaded Training: SASRec-Kandidaten als harte Negatives für AFN.

Ablauf:
  1. SASRec-Modell laden (RecBole .pth)
  2. Für jeden User Top-K Kandidaten generieren
  3. Positives aus train/valid/test.csv entnehmen
  4. Negatives aus SASRec Top-K sampeln (kein Zufalls-Sampling aus Gesamtkatalog)
  5. Neue CSVs + NPZs in dataset/cascaded/ schreiben

Beispiel:
    python retrieval/build_cascaded_splits.py --model retrieval/saved/SASRec-xxx.pth --topk 100
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_DIR = Path(__file__).parent.parent
DATASET_DIR = REPO_DIR / "dataset"


def generate_candidates(model, dataset, config, topk: int) -> dict[int, list[int]]:
    """Gibt {user_internal_id: [item_internal_id, ...]} zurück."""
    model.eval()
    item_field = dataset.iid_field
    uid_field = dataset.uid_field
    item_seq_field = model.ITEM_SEQ        # 'item_id_list'
    item_seq_len_field = model.ITEM_SEQ_LEN  # 'item_id_list_len'
    max_seq_len = config["MAX_ITEM_LIST_LENGTH"] if "MAX_ITEM_LIST_LENGTH" in config else 50

    # User-Sequenzen aus dem Dataset aufbauen
    inter = dataset.inter_feat
    user_seq: dict[int, list[int]] = {}
    for uid, iid in zip(inter[uid_field].tolist(), inter[item_field].tolist()):
        user_seq.setdefault(uid, []).append(iid)

    all_user_ids = list(range(1, dataset.user_num))  # 0 ist Padding
    candidates: dict[int, list[int]] = {}

    batch_size = 512
    log.info(f"Generiere Top-{topk} Kandidaten für {len(all_user_ids):,} User ...")

    for start in range(0, len(all_user_ids), batch_size):
        batch_uids = all_user_ids[start : start + batch_size]

        seqs, seq_lens = [], []
        for uid in batch_uids:
            seq = user_seq.get(uid, [])[-max_seq_len:]
            seq_lens.append(len(seq))
            seqs.append([0] * (max_seq_len - len(seq)) + seq)  # links padden

        uid_tensor = torch.tensor(batch_uids, dtype=torch.long, device=config["device"])
        seq_tensor = torch.tensor(seqs, dtype=torch.long, device=config["device"])
        seq_len_tensor = torch.tensor(seq_lens, dtype=torch.long, device=config["device"])

        interaction = Interaction({
            uid_field: uid_tensor,
            item_seq_field: seq_tensor,
            item_seq_len_field: seq_len_tensor,
        })
        interaction = interaction.to(config["device"])

        with torch.no_grad():
            scores = model.full_sort_predict(interaction)  # [B, num_items]

        scores = scores.cpu()

        # Items maskieren die der User bereits gesehen hat
        history_matrix = dataset.history_item_matrix()  # [num_users, max_hist_len]

        for i, uid in enumerate(batch_uids):
            user_scores = scores[i].clone()
            seen = history_matrix[uid].long()
            seen = seen[seen != 0]
            user_scores[seen] = -float("inf")
            user_scores[0] = -float("inf")  # Padding

            _, topk_iids = torch.topk(user_scores, min(topk, dataset.item_num - 1))
            candidates[uid] = topk_iids.tolist()

        if (start // batch_size) % 20 == 0:
            log.info(f"  {start + len(batch_uids):,} / {len(all_user_ids):,} User verarbeitet")

    return candidates


def build_cascaded_split(
    split_df: pd.DataFrame,
    candidates: dict[int, list[int]],
    uid_to_internal: dict,
    iid_to_internal: dict,
    internal_to_iid: dict,
    iid_mapping: dict,  # encoded_int -> original string (für Rückübersetzung nicht nötig, aber zum Debug)
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Baut einen Split-DataFrame mit harten SASRec-Negatives."""
    rows = []

    positives = split_df[split_df["label"] == 1]
    # negatives aus dem originalen Split für Feature-Lookup
    split_indexed = split_df.set_index(["user_id", "item_id"])

    for _, pos_row in positives.iterrows():
        uid_enc = int(pos_row["user_id"])
        iid_enc = int(pos_row["item_id"])

        uid_internal = uid_to_internal.get(uid_enc)
        if uid_internal is None:
            continue

        cands = candidates.get(uid_internal, [])
        # Das positive Item aus den Kandidaten entfernen
        iid_internal = iid_to_internal.get(iid_enc)
        hard_negs_internal = [c for c in cands if c != iid_internal]

        if not hard_negs_internal:
            continue

        # Positive Row übernehmen
        rows.append(pos_row.to_dict())

        # Negative sampeln
        sampled = rng.choice(
            hard_negs_internal,
            size=min(negatives_per_positive, len(hard_negs_internal)),
            replace=False,
        )

        for neg_internal in sampled:
            neg_iid_enc = internal_to_iid.get(neg_internal)
            if neg_iid_enc is None:
                continue

            # Feature-Zeile für dieses Item suchen (aus split oder train)
            if (uid_enc, neg_iid_enc) in split_indexed.index:
                neg_row = split_indexed.loc[(uid_enc, neg_iid_enc)].iloc[0].to_dict() if hasattr(
                    split_indexed.loc[(uid_enc, neg_iid_enc)], "iloc"
                ) else split_indexed.loc[(uid_enc, neg_iid_enc)].to_dict()
                neg_row["label"] = 0
                rows.append(neg_row)
            else:
                # Kein Kontext-Row vorhanden: Item-Features aus positivem Row übernehmen,
                # nur item_id austauschen (Preis etc. unbekannt → 0)
                neg_row = pos_row.to_dict()
                neg_row["item_id"] = neg_iid_enc
                neg_row["label"] = 0
                # Item-spezifische Features nullen
                for col in ["price", "brand", "category1", "category2", "category3", "category4"]:
                    if col in neg_row:
                        neg_row[col] = 0
                rows.append(neg_row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Pfad zur SASRec .pth-Datei")
    parser.add_argument("--topk", type=int, default=100, help="Anzahl SASRec-Kandidaten pro User")
    parser.add_argument("--neg-ratio", type=int, default=4, help="Harte Negatives pro Positive")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # SASRec laden
    log.info(f"Lade SASRec-Modell: {args.model}")
    config, model, dataset, *_ = load_data_and_model(args.model)

    # Kandidaten generieren
    candidates = generate_candidates(model, dataset, config, args.topk)

    # Mapping: RecBole-interne IDs ↔ encoded item_ids aus AFN-Dataset
    # RecBole speichert Items als Tokens (original string IDs)
    item_field = dataset.iid_field
    uid_field = dataset.uid_field

    # RecBole token2id: original string → internal int
    recbole_item_token2id = dataset.field2token_id[item_field]   # {"P-12345": 3, ...}
    recbole_user_token2id = dataset.field2token_id[uid_field]

    # AFN feature_vocab: original string → encoded int
    import json
    vocab_path = DATASET_DIR / "feature_vocab.json"
    with open(vocab_path) as f:
        feature_vocab = json.load(f)

    item_vocab = feature_vocab.get("item_id", {})  # {"P-12345": 7, ...}
    user_vocab = feature_vocab.get("user_id", {})

    # Brücke: AFN encoded int → RecBole internal int
    # AFN enc → original string → RecBole internal
    afn_enc_to_recbole: dict[int, int] = {}
    recbole_to_afn_enc: dict[int, int] = {}
    for orig_str, afn_enc in item_vocab.items():
        rb_int = recbole_item_token2id.get(orig_str)
        if rb_int is not None:
            afn_enc_to_recbole[afn_enc] = rb_int
            recbole_to_afn_enc[rb_int] = afn_enc

    uid_afn_to_recbole: dict[int, int] = {}
    for orig_str, afn_enc in user_vocab.items():
        rb_int = recbole_user_token2id.get(str(orig_str))
        if rb_int is not None:
            uid_afn_to_recbole[afn_enc] = rb_int

    log.info(f"Item-Mapping: {len(afn_enc_to_recbole):,} Items gemapped")
    log.info(f"User-Mapping: {len(uid_afn_to_recbole):,} User gemapped")

    # AFN-Splits laden
    log.info("Lade AFN-Splits ...")
    train_df = pd.read_csv(DATASET_DIR / "train.csv")
    valid_df = pd.read_csv(DATASET_DIR / "valid.csv")
    test_df  = pd.read_csv(DATASET_DIR / "test.csv")

    out_dir = DATASET_DIR / "cascaded"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cols = list(train_df.columns)
    float_cols = {"price", "geo_latitude", "geo_longitude", "label"}
    dtype_map = {col: np.float32 if col in float_cols else np.int32 for col in all_cols}

    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        log.info(f"Baue cascaded {split_name}-Split ...")
        result = build_cascaded_split(
            split_df=split_df,
            candidates=candidates,
            uid_to_internal=uid_afn_to_recbole,
            iid_to_internal=afn_enc_to_recbole,
            internal_to_iid=recbole_to_afn_enc,
            iid_mapping=item_vocab,
            negatives_per_positive=args.neg_ratio,
            rng=rng,
        )

        result = result[all_cols]  # Spaltenreihenfolge sicherstellen

        csv_path = out_dir / f"{split_name}.csv"
        npz_path = out_dir / f"{split_name}.csv.npz"
        result.to_csv(csv_path, index=False)

        arrays = {
            col: result[col].fillna(0).values.astype(dtype_map.get(col, np.float32))
            for col in all_cols
        }
        np.savez(npz_path, **arrays)

        pos = int((result["label"] == 1).sum())
        neg = int((result["label"] == 0).sum())
        log.info(f"  {split_name}: {len(result):,} Rows (pos={pos:,}, neg={neg:,}) → {csv_path}")

    log.info("Fertig. Trainiere AFN mit:")
    log.info("  python training/train/train.py --model AFN --data-dir dataset/cascaded/")


if __name__ == "__main__":
    main()
