"""Train the hybrid Lagos 30 W maximum-power-point models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from smart_mppt.augmentation import (
    DEFAULT_AUGMENTED_PATH,
    MAX_REASONABLE_POWER_W,
    SURROGATE_ISC_A,
    SURROGATE_VOC_V,
    prepare_augmented_dataset,
)
from smart_mppt.dataset import PROJECT_ROOT, file_sha256
from smart_mppt.manual_dataset import DEFAULT_DATASET_PATH, prepare_manual_dataset


MODEL_DIRECTORY = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIRECTORY / "smart_mppt.joblib"
DEFAULT_REPORT_PATH = MODEL_DIRECTORY / "training_report.json"
VOLTAGE_FEATURE_COLUMNS = [
    "light_lux",
    "log_light_lux",
    "temperature_c",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "solar_elevation_sin",
    "daylight_factor",
    "clearness_proxy",
    "sensor_range_ratio",
]
CURRENT_FEATURE_COLUMNS = [
    "light_lux",
    "log_light_lux",
    "temperature_c",
    "solar_elevation_sin",
    "daylight_factor",
    "clearness_proxy",
    "sensor_range_ratio",
]
TARGET_COLUMNS = ["max_power_voltage_v", "max_power_current_a"]
RANDOM_SEED = 42
CURRENT_AUGMENTATION_WEIGHT = 0.05


def _build_voltage_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=8,
        max_features=0.9,
        n_jobs=1,
        random_state=RANDOM_SEED,
    )


def _build_current_model() -> HistGradientBoostingRegressor:
    monotonic = [0] * len(CURRENT_FEATURE_COLUMNS)
    monotonic[CURRENT_FEATURE_COLUMNS.index("log_light_lux")] = 1
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=16,
        l2_regularization=5.0,
        monotonic_cst=monotonic,
        random_state=RANDOM_SEED,
    )


def _fit_current_model(
    local: pd.DataFrame, augmented: pd.DataFrame
) -> HistGradientBoostingRegressor:
    model = _build_current_model()
    features = pd.concat(
        [local[CURRENT_FEATURE_COLUMNS], augmented[CURRENT_FEATURE_COLUMNS]],
        ignore_index=True,
    )
    target = pd.concat(
        [local[TARGET_COLUMNS[1]], augmented[TARGET_COLUMNS[1]]],
        ignore_index=True,
    )
    weights = np.concatenate(
        [
            np.ones(len(local)),
            np.full(len(augmented), CURRENT_AUGMENTATION_WEIGHT),
        ]
    )
    model.fit(features, target, sample_weight=weights)
    return model


def _predict(
    voltage_model: RandomForestRegressor,
    current_model: HistGradientBoostingRegressor,
    frame: pd.DataFrame,
) -> np.ndarray:
    voltage = voltage_model.predict(frame[VOLTAGE_FEATURE_COLUMNS])
    current = current_model.predict(frame[CURRENT_FEATURE_COLUMNS])
    return np.column_stack([voltage, current])


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_power = actual[:, 0] * actual[:, 1]
    predicted_power = predicted[:, 0] * predicted[:, 1]
    return {
        "voltage_mae_v": round(float(mean_absolute_error(actual[:, 0], predicted[:, 0])), 6),
        "voltage_r2": round(float(r2_score(actual[:, 0], predicted[:, 0])), 6),
        "current_mae_a": round(float(mean_absolute_error(actual[:, 1], predicted[:, 1])), 6),
        "current_r2": round(float(r2_score(actual[:, 1], predicted[:, 1])), 6),
        "power_mae_w": round(float(mean_absolute_error(actual_power, predicted_power)), 6),
    }


def _evaluate_field_days(
    local: pd.DataFrame, augmented: pd.DataFrame
) -> tuple[dict[str, float], list[dict[str, object]]]:
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    fold_metrics: list[dict[str, object]] = []
    for training_index, testing_index in LeaveOneGroupOut().split(
        local, groups=local["collection_date"]
    ):
        training = local.iloc[training_index]
        testing = local.iloc[testing_index]
        voltage_model = _build_voltage_model()
        voltage_model.fit(
            training[VOLTAGE_FEATURE_COLUMNS], training[TARGET_COLUMNS[0]]
        )
        current_model = _fit_current_model(training, augmented)
        predicted = _predict(voltage_model, current_model, testing)
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


def _evaluate_augmented_current(
    local: pd.DataFrame, augmented: pd.DataFrame
) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    testing_mask = rng.random(len(augmented)) < 0.2
    development = augmented.loc[~testing_mask]
    testing = augmented.loc[testing_mask]
    model = _fit_current_model(local, development)
    predicted = np.maximum(0, model.predict(testing[CURRENT_FEATURE_COLUMNS]))
    actual = testing[TARGET_COLUMNS[1]].to_numpy()
    return {
        "rows": int(len(testing)),
        "current_mae_a": round(float(mean_absolute_error(actual, predicted)), 6),
        "current_r2": round(float(r2_score(actual, predicted)), 6),
    }


def _input_ranges(local: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    timestamps = pd.to_datetime(local["timestamp"])
    local_hours = (
        timestamps.dt.hour
        + timestamps.dt.minute / 60
        + timestamps.dt.second / 3600
    )
    collected = {
        "light_lux": {
            "minimum": float(local["light_lux"].min()),
            "maximum": float(local["light_lux"].max()),
        },
        "temperature_c": {
            "minimum": float(local["temperature_c"].min()),
            "maximum": float(local["temperature_c"].max()),
        },
        "time_of_day_hour": {
            "minimum": float(local_hours.min()),
            "maximum": float(local_hours.max()),
        },
        "day_of_year": {
            "minimum": float(timestamps.dt.dayofyear.min()),
            "maximum": float(timestamps.dt.dayofyear.max()),
        },
    }
    supported = {
        "light_lux": {"minimum": 1.0, "maximum": 100_000.0},
        "temperature_c": {"minimum": 24.0, "maximum": 58.0},
        "time_of_day_hour": {"minimum": 5.5, "maximum": 19.5},
        "day_of_year": {"minimum": 1.0, "maximum": 365.0},
    }
    return collected, supported


def train(
    dataset_path: Path | None = None,
    augmented_path: Path | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    augmented_path = augmented_path or DEFAULT_AUGMENTED_PATH
    if not dataset_path.exists():
        dataset_path, _ = prepare_manual_dataset(dataset_path=dataset_path)
    if not augmented_path.exists():
        augmented_path, _ = prepare_augmented_dataset(dataset_path=augmented_path)

    local = pd.read_csv(dataset_path)
    augmented = pd.read_csv(augmented_path)
    field_metrics, fold_metrics = _evaluate_field_days(local, augmented)
    augmented_metrics = _evaluate_augmented_current(local, augmented)

    voltage_model = _build_voltage_model()
    voltage_model.fit(local[VOLTAGE_FEATURE_COLUMNS], local[TARGET_COLUMNS[0]])
    current_model = _fit_current_model(local, augmented)
    collected_ranges, supported_ranges = _input_ranges(local)

    artifact = {
        "artifact_version": 3,
        "voltage_model": voltage_model,
        "current_model": current_model,
        "voltage_feature_columns": VOLTAGE_FEATURE_COLUMNS,
        "current_feature_columns": CURRENT_FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "collected_ranges": collected_ranges,
        "supported_ranges": supported_ranges,
        "timezone": "Africa/Lagos",
        "panel_rating_w": 30.0,
        "light_unit": "lux",
        "dataset": "Hybrid Lagos field + lux-domain 30 W physics model",
        "output_constraints": {
            "maximum_voltage_v": SURROGATE_VOC_V,
            "maximum_current_a": SURROGATE_ISC_A,
            "maximum_power_w": MAX_REASONABLE_POWER_W,
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".tmp")
    joblib.dump(artifact, temporary_model, compress=3)
    temporary_model.replace(model_path)

    report: dict[str, object] = {
        "artifact_version": 3,
        "model_type": {
            "voltage": "RandomForestRegressor trained on Lagos field labels",
            "current": "Monotonic HistGradientBoostingRegressor trained on weighted Lagos and physics-guided labels",
        },
        "random_seed": RANDOM_SEED,
        "local_dataset_sha256": file_sha256(dataset_path),
        "augmented_dataset_sha256": file_sha256(augmented_path),
        "model_sha256": file_sha256(model_path),
        "scikit_learn_version": sklearn.__version__,
        "local_training_rows": int(len(local)),
        "augmented_training_rows": int(len(augmented)),
        "current_augmentation_weight_per_row": CURRENT_AUGMENTATION_WEIGHT,
        "collection_days": int(local["collection_date"].nunique()),
        "collection_dates": sorted(local["collection_date"].unique().tolist()),
        "field_split_policy": "Leave one complete Lagos collection date out per fold",
        "voltage_features": VOLTAGE_FEATURE_COLUMNS,
        "current_features": CURRENT_FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "collected_ranges": collected_ranges,
        "supported_ranges": supported_ranges,
        "field_leave_one_day_out_metrics": field_metrics,
        "field_per_day_metrics": fold_metrics,
        "held_out_augmented_current_metrics": augmented_metrics,
        "output_constraints": artifact["output_constraints"],
        "interpretation": {
            "field_metrics": "Agreement with the supplied Lagos labels on unseen dates.",
            "augmented_metrics": "Agreement with held-out physics-guided current labels; this is not a real-world accuracy claim.",
            "prediction": "Expected MPP starting point for a short controller verification search, not proof of the global peak from one lux sensor.",
        },
        "known_limitations": [
            "Only four independent Lagos collection dates are available.",
            "The exact 30 W panel datasheet is unavailable, so synthetic electrical limits are explicit surrogate assumptions.",
            "The local current labels have no validated feature relationship and conflict with low-light photovoltaic behavior.",
            "One ambient-light sensor cannot identify the spatial shape of partial shade.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--augmented", type=Path, default=DEFAULT_AUGMENTED_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train(args.dataset, args.augmented, args.model, args.report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
