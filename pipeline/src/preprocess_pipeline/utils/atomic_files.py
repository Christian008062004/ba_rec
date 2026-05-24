#!/usr/bin/env python3
"""Schreibt RecBole Atomic Files (.inter, .item, .user) aus vorbereiteten Dicts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable

USER_DEVICE_COLUMNS = {
    "device_type": "device_type",
    "device_os": "device_os",
    "device_os_version": "device_os_version",
    "device_model": "device_model",
    "device_manufacturer": "device_manufacturer",
    "device_browser": "device_browser",
    "device_browser_version": "device_browser_version",
    "device_type_detail": "device_type_detail",
}


def write_inter_file(rows: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "user_id:token",
                "item_id:token",
                "rating:float",
                "timestamp:float",
                "event_type:token",
                "day_of_week:token",
                "hour_of_day:token",
                "month:token",
                "is_holiday:token",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["user_id"],
                    row["item_id"],
                    f'{row["rating"]:.1f}',
                    row["timestamp"],
                    row["event_type"],
                    row.get("day_of_week", ""),
                    row.get("hour_of_day", ""),
                    row.get("month", ""),
                    row.get("is_holiday", ""),
                ]
            )


def write_item_file(
    item_features: Dict[str, dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "item_id:token",
                "product_name:token",
                "category1:token",
                "category2:token",
                "category3:token",
                "category4:token",
            ]
        )
        for item_id, features in sorted(item_features.items()):
            writer.writerow(
                [
                    item_id,
                    features.get("product_name", item_id),
                    features.get("category1", ""),
                    features.get("category2", ""),
                    features.get("category3", ""),
                    features.get("category4", ""),
                ]
            )


def write_user_file(user_features: Dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_cols = list(USER_DEVICE_COLUMNS.keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "user_id:token",
                "geo_country:token",
                "geo_region:token",
                "geo_city:token",
                *[f"{col}:token" for col in device_cols],
            ]
        )
        for user_id, features in sorted(user_features.items()):
            writer.writerow(
                [
                    user_id,
                    features.get("geo_country", ""),
                    features.get("geo_region", ""),
                    features.get("geo_city", ""),
                    *[features.get(col, "") for col in device_cols],
                ]
            )
