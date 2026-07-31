"""Load the Lagos model and predict a startup maximum power point."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smart_mppt.time_features import (
    BH1750_STANDARD_MAX_LUX,
    environmental_features_for_timestamp,
    to_lagos_local,
)


def default_model_path() -> Path:
    configured = os.environ.get("SMART_MPPT_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd() / "models" / "smart_mppt.joblib"


@dataclass(frozen=True)
class StartupMeasurement:
    light_lux: float
    temperature_c: float
    timestamp: datetime

    def as_features(self) -> dict[str, float]:
        return {
            "light_lux": self.light_lux,
            "log_light_lux": float(np.log1p(self.light_lux)),
            "temperature_c": self.temperature_c,
            **environmental_features_for_timestamp(self.timestamp, self.light_lux),
        }


@dataclass(frozen=True)
class Prediction:
    voltage: float
    current: float
    power: float
    within_training_range: bool
    warnings: tuple[str, ...]


class MPPPredictor:
    """Inference wrapper around the packaged Lagos 30 W estimators."""

    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = default_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run scripts/train_model.py."
            )
        artifact = joblib.load(model_path)
        if artifact.get("artifact_version") != 3:
            raise ValueError(
                "Unsupported model artifact version; retrain with scripts/train_model.py"
            )
        self.voltage_model = artifact["voltage_model"]
        self.current_model = artifact["current_model"]
        self.voltage_feature_columns: list[str] = artifact[
            "voltage_feature_columns"
        ]
        self.current_feature_columns: list[str] = artifact[
            "current_feature_columns"
        ]
        self.input_ranges: dict[str, dict[str, float]] = artifact[
            "supported_ranges"
        ]
        self.collected_ranges: dict[str, dict[str, float]] = artifact[
            "collected_ranges"
        ]
        self.dataset: str = artifact["dataset"]
        self.panel_rating_w: float = artifact["panel_rating_w"]
        self.output_constraints: dict[str, float] = artifact["output_constraints"]
        self.current_local_anchor_a: float = artifact["current_local_anchor_a"]
        self.current_physics_blend: float = artifact["current_physics_blend"]
        self.current_dark_gate_lux: float = artifact["current_dark_gate_lux"]

    def _range_warnings(self, measurement: StartupMeasurement) -> tuple[str, ...]:
        local = to_lagos_local(measurement.timestamp)
        values = {
            "light_lux": measurement.light_lux,
            "temperature_c": measurement.temperature_c,
            "time_of_day_hour": (
                local.hour + local.minute / 60 + local.second / 3600
            ),
            "day_of_year": float(local.timetuple().tm_yday),
        }
        warnings: list[str] = []
        for name, value in values.items():
            limits = self.input_ranges[name]
            minimum = limits["minimum"]
            maximum = limits["maximum"]
            if not minimum <= value <= maximum:
                warnings.append(
                    f"{name}={value:g} is outside the supported range "
                    f"[{minimum:g}, {maximum:g}]"
                )
        if measurement.light_lux >= 0.95 * BH1750_STANDARD_MAX_LUX:
            warnings.append(
                "light_lux is near or above the BH1750 standard range; "
                "verify that extended-range sensor settings are enabled"
            )
        return tuple(warnings)

    def predict(self, measurement: StartupMeasurement) -> Prediction:
        features = measurement.as_features()
        frame = pd.DataFrame([features])
        voltage = float(
            self.voltage_model.predict(frame[self.voltage_feature_columns])[0]
        )
        physics_current = float(
            self.current_model.predict(frame[self.current_feature_columns])[0]
        )
        daylight_gate = 1 - np.exp(
            -measurement.light_lux / self.current_dark_gate_lux
        )
        current = float(
            daylight_gate
            * (
                (1 - self.current_physics_blend) * self.current_local_anchor_a
                + self.current_physics_blend * max(0, physics_current)
            )
        )
        voltage = float(
            np.clip(
                voltage,
                0,
                self.output_constraints["maximum_voltage_v"],
            )
        )
        current = float(
            np.clip(
                current,
                0,
                self.output_constraints["maximum_current_a"],
            )
        )
        power = voltage * current
        if power > self.output_constraints["maximum_power_w"] and voltage > 0:
            current = self.output_constraints["maximum_power_w"] / voltage
            power = voltage * current
        warnings = self._range_warnings(measurement)
        return Prediction(
            voltage=round(voltage, 3),
            current=round(current, 3),
            power=round(power, 3),
            within_training_range=not warnings,
            warnings=warnings,
        )


@lru_cache(maxsize=1)
def get_predictor() -> MPPPredictor:
    return MPPPredictor()
