"""Load the trained model and predict a startup maximum power point."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from smart_mppt.dataset import PROJECT_ROOT


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "smart_mppt.joblib"


@dataclass(frozen=True)
class StartupMeasurement:
    sun_intensity: float
    panel_voltage: float
    panel_current: float
    ambient_temperature: float
    time_of_day_hour: float

    def as_features(self) -> dict[str, float]:
        return {
            "sun_intensity_w_m2": self.sun_intensity,
            "panel_voltage_v": self.panel_voltage,
            "panel_current_a": self.panel_current,
            "ambient_temperature_c": self.ambient_temperature,
            "time_of_day_hour": self.time_of_day_hour,
        }


@dataclass(frozen=True)
class Prediction:
    voltage: float
    current: float
    power: float
    within_training_range: bool
    warnings: tuple[str, ...]


class MPPPredictor:
    """Inference wrapper around the packaged UCP-trained estimator."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Run scripts/train_model.py."
            )
        artifact = joblib.load(model_path)
        if artifact.get("artifact_version") != 1:
            raise ValueError("Unsupported model artifact version")
        self.model = artifact["model"]
        self.feature_columns: list[str] = artifact["feature_columns"]
        self.training_ranges: dict[str, dict[str, float]] = artifact[
            "training_ranges"
        ]
        self.dataset_doi: str = artifact["dataset_doi"]

    def _range_warnings(self, features: dict[str, float]) -> tuple[str, ...]:
        warnings: list[str] = []
        for name, value in features.items():
            limits = self.training_ranges[name]
            minimum = limits["minimum"]
            maximum = limits["maximum"]
            if not minimum <= value <= maximum:
                warnings.append(
                    f"{name}={value:g} is outside the trained range "
                    f"[{minimum:g}, {maximum:g}]"
                )
        return tuple(warnings)

    def predict(self, measurement: StartupMeasurement) -> Prediction:
        features = measurement.as_features()
        frame = pd.DataFrame([features], columns=self.feature_columns)
        prediction = self.model.predict(frame)[0]
        voltage = max(0.0, float(prediction[0]))
        current = max(0.0, float(prediction[1]))
        warnings = self._range_warnings(features)
        return Prediction(
            voltage=round(voltage, 3),
            current=round(current, 3),
            power=round(voltage * current, 3),
            within_training_range=not warnings,
            warnings=warnings,
        )


@lru_cache(maxsize=1)
def get_predictor() -> MPPPredictor:
    return MPPPredictor()

