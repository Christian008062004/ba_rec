#!/usr/bin/env python3
"""Top-N Empfehlungen aus einer Item-Sequenz (GRU4Rec, SASRec, ...).

Beispiel:
    python retrieval/recommend.py \
        --model saved/SASRec-<datum>.pth \
        --sequence P-90243228 P-77380870 P-12345678 \
        --topk 10
"""

import argparse
import torch
from recbole.quick_start import load_data_and_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Pfad zur .pth-Datei, z.B. saved/GRU4Rec-....pth")
    parser.add_argument("--sequence", nargs="+", required=True, help="Geordnete Item-IDs (älteste zuerst)")
    parser.add_argument("--topk", type=int, default=10, help="Anzahl Empfehlungen")
    args = parser.parse_args()

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model)
    model.eval()

    # Item-IDs in interne Indizes umwandeln (unbekannte Items überspringen)
    item_field = dataset.iid_field
    valid_items = []
    for pid in args.sequence:
        if pid in dataset.field2token_id[item_field]:
            valid_items.append(dataset.token2id(item_field, pid))
        else:
            print(f"Warnung: '{pid}' nicht im Datensatz – wird übersprungen.")

    if not valid_items:
        print("Keine bekannten Items in der Sequenz.")
        return

    seq_tensor = torch.tensor(valid_items, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    seq_len = torch.tensor([len(valid_items)], dtype=torch.long)

    interaction = {
        dataset.item_id_list_field: seq_tensor,
        dataset.item_list_length_field: seq_len,
    }
    from recbole.data.interaction import Interaction
    inter = Interaction(interaction)
    inter = inter.to(config["device"])

    with torch.no_grad():
        # Scores über alle Items berechnen
        scores = model.full_sort_predict(inter)  # [1, num_items]

    scores = scores.squeeze(0).cpu()
    # Bereits gesehene Items ausblenden
    for iid in valid_items:
        scores[iid] = -float("inf")

    topk_score, topk_iid_list = torch.topk(scores, args.topk)
    topk_iid_list = topk_iid_list.numpy()
    topk_score = topk_score.numpy()

    print(f"\nTop-{args.topk} Empfehlungen für Sequenz: {args.sequence}\n")
    print(f"{'Rang':<6} {'Item-ID':<20} {'Score':.6}")
    print("-" * 36)
    for rank, (iid, score) in enumerate(zip(topk_iid_list, topk_score), 1):
        token = dataset.id2token(item_field, iid)
        print(f"{rank:<6} {token:<20} {score:.4f}")


if __name__ == "__main__":
    main()
