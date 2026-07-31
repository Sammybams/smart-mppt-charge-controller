"""One-shot startup API for the Lagos 30 W MPPT model."""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from smart_mppt import __version__
from smart_mppt.predictor import StartupMeasurement, get_predictor


class StartupPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "light_lux": 42000,
                "temperature_c": 38.5,
                "timestamp": "2026-07-22T12:30:00+01:00",
            }
        }
    )

    light_lux: float = Field(
        ge=0,
        le=200_000,
        description="Illuminance reported by the project light sensor in lux",
    )
    temperature_c: float = Field(
        ge=-50,
        le=100,
        description="Ambient temperature in degrees Celsius",
    )
    timestamp: datetime = Field(
        description=(
            "Measurement date and time. Include a UTC offset when possible; "
            "a timestamp without one is interpreted as Africa/Lagos time."
        )
    )


class MaximumPowerPoint(BaseModel):
    voltage_v: float = Field(description="Predicted MPP panel voltage in volts")
    current_a: float = Field(description="Predicted MPP panel current in amperes")
    power_w: float = Field(description="Voltage multiplied by current in watts")


class StartupPredictionResponse(BaseModel):
    max_power_point: MaximumPowerPoint
    within_training_range: bool
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


app = FastAPI(
    title="AI Enabled Smart MPPT Charge Controller",
    version=__version__,
    description=(
        "Predicts the Sunshine Solar AP-PM-30W panel's expected maximum-power "
        "voltage and current once from a Lagos BH1750 lux, temperature, and "
        "timestamp reading."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        version=__version__,
        model=predictor.dataset,
    )


@app.post("/predict", response_model=StartupPredictionResponse)
def predict(request: StartupPredictionRequest) -> StartupPredictionResponse:
    result = get_predictor().predict(
        StartupMeasurement(
            light_lux=request.light_lux,
            temperature_c=request.temperature_c,
            timestamp=request.timestamp,
        )
    )
    return StartupPredictionResponse(
        max_power_point=MaximumPowerPoint(
            voltage_v=result.voltage,
            current_a=result.current,
            power_w=result.power,
        ),
        within_training_range=result.within_training_range,
        warnings=list(result.warnings),
    )
