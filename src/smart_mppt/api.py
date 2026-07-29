"""HTTP API matching the requested one-shot startup prediction interface."""

from __future__ import annotations

from datetime import time

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from smart_mppt import __version__
from smart_mppt.predictor import StartupMeasurement, get_predictor


class StartupPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sun_intensity": 850,
                "panel_voltage": 24.5,
                "panel_current": 7.8,
                "ambient_temperature": 32,
                "time_of_day": "12:30:00",
            }
        }
    )

    sun_intensity: float = Field(
        ge=0,
        le=1500,
        description="Measured solar irradiance in W/m²",
    )
    panel_voltage: float = Field(
        ge=0,
        le=100,
        description="Panel voltage at device startup in volts",
    )
    panel_current: float = Field(
        ge=0,
        le=100,
        description="Panel current at device startup in amperes",
    )
    ambient_temperature: float = Field(
        ge=-50,
        le=100,
        description="Ambient temperature in degrees Celsius",
    )
    time_of_day: time = Field(
        description="Local device time in HH:MM or HH:MM:SS format",
    )


class MaximumPowerPoint(BaseModel):
    voltage: float = Field(description="Predicted MPP voltage in volts")
    current: float = Field(description="Predicted MPP current in amperes")
    power: float = Field(description="Predicted maximum power in watts")


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
        "Predicts the maximum power point once from the measurements supplied "
        "when the charge controller starts."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        version=__version__,
        model=f"UCP {predictor.dataset_doi}",
    )


@app.post("/predict", response_model=StartupPredictionResponse)
def predict(request: StartupPredictionRequest) -> StartupPredictionResponse:
    time_as_hour = (
        request.time_of_day.hour
        + request.time_of_day.minute / 60
        + request.time_of_day.second / 3600
    )
    result = get_predictor().predict(
        StartupMeasurement(
            sun_intensity=request.sun_intensity,
            panel_voltage=request.panel_voltage,
            panel_current=request.panel_current,
            ambient_temperature=request.ambient_temperature,
            time_of_day_hour=time_as_hour,
        )
    )
    return StartupPredictionResponse(
        max_power_point=MaximumPowerPoint(
            voltage=result.voltage,
            current=result.current,
            power=result.power,
        ),
        within_training_range=result.within_training_range,
        warnings=list(result.warnings),
    )

