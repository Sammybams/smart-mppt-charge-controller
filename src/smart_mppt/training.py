"""Train the startup maximum-power-point regression model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor

from smart_mppt.dataset import PROJECT_ROOT, PROCESSED_DIRECTORY, prepare_dataset


MODEL_DIRECTORY = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIRECTORY / "smart_mppt.joblib"
DEFAULT_REPORT_PATH = MODEL_DIRECTORY / "training_report.json"
FEATURE_COLUMNS = [
    "sun_intensity_w_m2",
    "panel_voltage_v",
    "panel_current_a",
    "ambient_temperature_c",
    "time_of_day_hour",
]
TARGET_COLUMNS = ["max_power_voltage_v", "max_power_current_a"]
RANDOM_SEED = 42


def _build_estimator() -> MultiOutputRegressor:
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=160,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
    )


def _round_metrics(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(
    dataset_path: Path | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    if dataset_path is None:
        dataset_path = PROCESSED_DIRECTORY / "startup_training.csv"
    if not dataset_path.exists():
        dataset_path, _ = prepare_dataset()

    frame = pd.read_csv(dataset_path)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    training_index, testing_index = next(
        splitter.split(frame, groups=frame["condition_id"])
    )
    training = frame.iloc[training_index]
    testing = frame.iloc[testing_index]

    evaluation_model = _build_estimator()
    evaluation_model.fit(training[FEATURE_COLUMNS], training[TARGET_COLUMNS])
    predictions = evaluation_model.predict(testing[FEATURE_COLUMNS])
    actual = testing[TARGET_COLUMNS]
    predicted_power = predictions[:, 0] * predictions[:, 1]
    actual_power = actual.iloc[:, 0].to_numpy() * actual.iloc[:, 1].to_numpy()

    metrics = {
        "voltage_current_mae": _round_metrics(
            mean_absolute_error(actual, predictions, multioutput="raw_values")
        ),
        "voltage_current_r2": _round_metrics(
            r2_score(actual, predictions, multioutput="raw_values")
        ),
        "power_mae_w": round(
            float(mean_absolute_error(actual_power, predicted_power)), 6
        ),
    }

    final_model = _build_estimator()
    final_model.fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMNS])
    ranges = {
        column: {
            "minimum": float(frame[column].min()),
            "maximum": float(frame[column].max()),
        }
        for column in FEATURE_COLUMNS
    }
    artifact = {
        "artifact_version": 1,
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "training_ranges": ranges,
        "dataset_doi": "10.17632/z93gzbptf7.1",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".tmp")
    joblib.dump(artifact, temporary_model, compress=3)
    temporary_model.replace(model_path)

    report: dict[str, object] = {
        "artifact_version": 1,
        "model_type": "MultiOutputRegressor(HistGradientBoostingRegressor)",
        "random_seed": RANDOM_SEED,
        "dataset_doi": "10.17632/z93gzbptf7.1",
        "dataset_sha256": _file_sha256(dataset_path),
        "model_sha256": _file_sha256(model_path),
        "scikit_learn_version": sklearn.__version__,
        "training_rows": int(len(frame)),
        "source_curves": int(frame["condition_id"].nunique()),
        "holdout_rows": int(len(testing)),
        "holdout_curves": int(testing["condition_id"].nunique()),
        "split_policy": "20% group holdout by source curve",
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "training_ranges": ranges,
        "holdout_metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROCESSED_DIRECTORY / "startup_training.csv",
    )
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

