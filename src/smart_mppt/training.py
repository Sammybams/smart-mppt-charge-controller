"""Train the Lagos 30 W maximum-power-point regression models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from smart_mppt.dataset import PROJECT_ROOT, file_sha256
from smart_mppt.manual_dataset import (
    DEFAULT_DATASET_PATH,
    prepare_manual_dataset,
)


MODEL_DIRECTORY = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIRECTORY / "smart_mppt.joblib"
DEFAULT_REPORT_PATH = MODEL_DIRECTORY / "training_report.json"
FEATURE_COLUMNS = [
    "light_lux",
    "log_light_lux",
    "temperature_c",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]
TARGET_COLUMNS = ["max_power_voltage_v", "max_power_current_a"]
RANDOM_SEED = 42


def _build_voltage_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=8,
        max_features=0.9,
        n_jobs=1,
        random_state=RANDOM_SEED,
    )


def _build_current_model() -> DummyRegressor:
    # Leave-one-day-out experiments show no generalizable current relationship
    # in this collection. A median is safer than fitting the quantization noise.
    return DummyRegressor(strategy="median")


def _predict(
    voltage_model: RandomForestRegressor,
    current_model: DummyRegressor,
    features: pd.DataFrame,
) -> np.ndarray:
    voltage = voltage_model.predict(features)
    current = current_model.predict(features)
    return np.column_stack([voltage, current])


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    actual_power = actual[:, 0] * actual[:, 1]
    predicted_power = predicted[:, 0] * predicted[:, 1]
    return {
        "voltage_mae_v": round(float(mean_absolute_error(actual[:, 0], predicted[:, 0])), 6),
        "voltage_r2": round(float(r2_score(actual[:, 0], predicted[:, 0])), 6),
        "current_mae_a": round(float(mean_absolute_error(actual[:, 1], predicted[:, 1])), 6),
        "current_r2": round(float(r2_score(actual[:, 1], predicted[:, 1])), 6),
        "power_mae_w": round(float(mean_absolute_error(actual_power, predicted_power)), 6),
    }


def _evaluate_by_day(frame: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    splitter = LeaveOneGroupOut()
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    fold_metrics: list[dict[str, object]] = []

    for training_index, testing_index in splitter.split(
        frame, groups=frame["collection_date"]
    ):
        training = frame.iloc[training_index]
        testing = frame.iloc[testing_index]
        voltage_model = _build_voltage_model()
        current_model = _build_current_model()
        voltage_model.fit(training[FEATURE_COLUMNS], training[TARGET_COLUMNS[0]])
        current_model.fit(training[FEATURE_COLUMNS], training[TARGET_COLUMNS[1]])
        predicted = _predict(voltage_model, current_model, testing[FEATURE_COLUMNS])
        actual = testing[TARGET_COLUMNS].to_numpy()
        actual_parts.append(actual)
        predicted_parts.append(predicted)
        fold_metrics.append(
            {
                "held_out_date": str(testing["collection_date"].iloc[0]),
                "rows": int(len(testing)),
                **_metrics(actual, predicted),
            }
        )

    return _metrics(np.vstack(actual_parts), np.vstack(predicted_parts)), fold_metrics


def train(
    dataset_path: Path | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH
    if not dataset_path.exists():
        dataset_path, _ = prepare_manual_dataset(dataset_path=dataset_path)

    frame = pd.read_csv(dataset_path)
    overall_metrics, fold_metrics = _evaluate_by_day(frame)

    voltage_model = _build_voltage_model()
    current_model = _build_current_model()
    voltage_model.fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMNS[0]])
    current_model.fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMNS[1]])

    timestamps = pd.to_datetime(frame["timestamp"])
    local_hours = (
        timestamps.dt.hour
        + timestamps.dt.minute / 60
        + timestamps.dt.second / 3600
    )
    input_ranges = {
        column: {
            "minimum": float(frame[column].min()),
            "maximum": float(frame[column].max()),
        }
        for column in ("light_lux", "temperature_c")
    }
    input_ranges["time_of_day_hour"] = {
        "minimum": float(local_hours.min()),
        "maximum": float(local_hours.max()),
    }
    input_ranges["day_of_year"] = {
        "minimum": float(timestamps.dt.dayofyear.min()),
        "maximum": float(timestamps.dt.dayofyear.max()),
    }
    artifact = {
        "artifact_version": 2,
        "voltage_model": voltage_model,
        "current_model": current_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "input_ranges": input_ranges,
        "timestamp_range": {
            "minimum": str(frame["timestamp"].min()),
            "maximum": str(frame["timestamp"].max()),
        },
        "timezone": "Africa/Lagos",
        "panel_rating_w": 30,
        "light_unit": "lux",
        "dataset": "Lagos 30 W manual PV collection",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".tmp")
    joblib.dump(artifact, temporary_model, compress=3)
    temporary_model.replace(model_path)

    report: dict[str, object] = {
        "artifact_version": 2,
        "model_type": {
            "voltage": "RandomForestRegressor",
            "current": "DummyRegressor(median)",
        },
        "model_selection": {
            "voltage": "Random forest outperformed Extra Trees, gradient boosting, and a median baseline in leave-one-day-out validation.",
            "current": "No tested feature model beat a day-isolated median baseline; the median avoids claiming a learned relationship that did not generalize.",
        },
        "random_seed": RANDOM_SEED,
        "dataset": "Lagos 30 W manual PV collection",
        "dataset_sha256": file_sha256(dataset_path),
        "model_sha256": file_sha256(model_path),
        "scikit_learn_version": sklearn.__version__,
        "training_rows": int(len(frame)),
        "collection_days": int(frame["collection_date"].nunique()),
        "collection_dates": sorted(frame["collection_date"].unique().tolist()),
        "split_policy": "Leave one complete collection date out per fold",
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "input_ranges": input_ranges,
        "leave_one_day_out_metrics": overall_metrics,
        "per_day_metrics": fold_metrics,
        "known_limitations": [
            "Only four independent collection dates are available.",
            "The 15-row 2026-06-22 fold is too small for a stable per-day R2 score.",
            "PANEL_CURRENT has no validated relationship with lux, temperature, or timestamp in this collection.",
            "Predicted power may exceed the nominal 30 W rating because targets are retained without clipping.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train(args.dataset, args.model, args.report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
