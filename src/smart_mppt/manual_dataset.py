"""Prepare the Lagos 30 W field measurements for model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smart_mppt.dataset import PROJECT_ROOT, PROCESSED_DIRECTORY, file_sha256
from smart_mppt.time_features import calendar_features


DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "Manual_Collection.csv"
DEFAULT_DATASET_PATH = PROCESSED_DIRECTORY / "lagos_30w_training.csv"
DEFAULT_METADATA_PATH = PROCESSED_DIRECTORY / "lagos_30w_training.metadata.json"
SOURCE_COLUMNS = [
    "TIME",
    "LIGHT",
    "TEMPERATURE",
    "PANEL_VOLTAGE",
    "PANEL_CURRENT",
]
OUTPUT_COLUMNS = [
    "timestamp",
    "collection_date",
    "session_id",
    "light_lux",
    "log_light_lux",
    "temperature_c",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "max_power_voltage_v",
    "max_power_current_a",
    "max_power_w",
]


def _validate_source(frame: pd.DataFrame) -> None:
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Manual collection is missing columns: {', '.join(missing)}")


def prepare_manual_dataset(
    source_path: Path = DEFAULT_SOURCE_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[Path, Path]:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing Lagos field data: {source_path}")

    source = pd.read_csv(source_path)
    _validate_source(source)
    selected = source[SOURCE_COLUMNS].rename(
        columns={
            "TIME": "timestamp",
            "LIGHT": "light_lux",
            "TEMPERATURE": "temperature_c",
            "PANEL_VOLTAGE": "max_power_voltage_v",
            "PANEL_CURRENT": "max_power_current_a",
        }
    )
    rows_received = len(selected)
    selected = selected.drop_duplicates()
    exact_duplicates_removed = rows_received - len(selected)

    selected["timestamp"] = pd.to_datetime(
        selected["timestamp"], format="%m/%d/%Y %H:%M:%S", errors="raise"
    )
    numeric_columns = [
        "light_lux",
        "temperature_c",
        "max_power_voltage_v",
        "max_power_current_a",
    ]
    selected[numeric_columns] = selected[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(selected[numeric_columns].to_numpy()).all():
        raise ValueError("Manual collection contains non-finite numeric values")
    if (selected[numeric_columns] < 0).any().any():
        raise ValueError("Manual collection contains negative physical measurements")

    rows_before_timestamp_aggregation = len(selected)
    conflicting_timestamp_groups = int(
        (selected.groupby("timestamp", sort=False).size() > 1).sum()
    )
    prepared = (
        selected.groupby("timestamp", as_index=False, sort=True)[numeric_columns]
        .median()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    timestamp_rows_aggregated = rows_before_timestamp_aggregation - len(prepared)

    gaps = prepared["timestamp"].diff()
    new_session = gaps.isna() | (gaps > pd.Timedelta(minutes=5))
    prepared["session_id"] = new_session.cumsum().map(lambda value: f"session-{value:02d}")
    prepared["collection_date"] = prepared["timestamp"].dt.strftime("%Y-%m-%d")
    prepared["log_light_lux"] = np.log1p(prepared["light_lux"])
    prepared = pd.concat([prepared, calendar_features(prepared["timestamp"])], axis=1)
    prepared["max_power_w"] = (
        prepared["max_power_voltage_v"] * prepared["max_power_current_a"]
    )
    prepared = prepared[OUTPUT_COLUMNS]

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(dataset_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")

    metadata = {
        "dataset": "Lagos 30 W manual PV collection",
        "source_path": str(source_path.relative_to(PROJECT_ROOT)),
        "source_sha256": file_sha256(source_path),
        "timezone": "Africa/Lagos",
        "panel_rating_w": 30,
        "light_unit": "lux",
        "temperature_unit": "degrees Celsius",
        "rows_received": rows_received,
        "exact_duplicates_removed": exact_duplicates_removed,
        "timestamp_rows_aggregated": timestamp_rows_aggregated,
        "conflicting_timestamp_groups": conflicting_timestamp_groups,
        "training_rows": int(len(prepared)),
        "collection_dates": sorted(prepared["collection_date"].unique().tolist()),
        "session_count": int(prepared["session_id"].nunique()),
        "columns": OUTPUT_COLUMNS,
        "preprocessing": [
            "Select owner-confirmed input and target columns",
            "Remove exact duplicates across selected columns",
            "Median-aggregate multiple readings with the same timestamp",
            "Sort chronologically and split sessions at gaps over five minutes",
            "Apply log1p to lux and cyclic encodings to local time and day of year",
            "Calculate maximum power as target voltage multiplied by target current",
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return dataset_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path, metadata_path = prepare_manual_dataset(
        args.source, args.dataset, args.metadata
    )
    print(f"Prepared {dataset_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
