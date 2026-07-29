import json
from pathlib import Path

from fastapi.testclient import TestClient

from smart_mppt.api import app


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def test_health_loads_packaged_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "model": "UCP 10.17632/z93gzbptf7.1",
    }


def test_documented_startup_request_matches_documented_response() -> None:
    request = json.loads((ROOT / "examples" / "startup_request.json").read_text())
    expected = json.loads(
        (ROOT / "examples" / "startup_response.json").read_text()
    )

    response = client.post("/predict", json=request)

    assert response.status_code == 200
    assert response.json() == expected


def test_short_time_format_is_accepted() -> None:
    response = client.post(
        "/predict",
        json={
            "sun_intensity": 850,
            "panel_voltage": 24.5,
            "panel_current": 7.8,
            "ambient_temperature": 32,
            "time_of_day": "12:30",
        },
    )

    assert response.status_code == 200


def test_invalid_physical_measurement_is_rejected() -> None:
    response = client.post(
        "/predict",
        json={
            "sun_intensity": -1,
            "panel_voltage": 24.5,
            "panel_current": 7.8,
            "ambient_temperature": 32,
            "time_of_day": "12:30:00",
        },
    )

    assert response.status_code == 422


def test_out_of_training_range_is_reported() -> None:
    response = client.post(
        "/predict",
        json={
            "sun_intensity": 300,
            "panel_voltage": 24.5,
            "panel_current": 7.8,
            "ambient_temperature": 32,
            "time_of_day": "07:00:00",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["within_training_range"] is False
    assert len(body["warnings"]) == 2
    assert body["max_power_point"]["power"] > 0

