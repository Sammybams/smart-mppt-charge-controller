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
        "version": "0.4.0",
        "model": "Sunshine Solar AP-PM-30W hybrid Lagos model",
    }


def test_documented_startup_request_matches_documented_response() -> None:
    request = json.loads((ROOT / "examples" / "startup_request.json").read_text())
    expected = json.loads((ROOT / "examples" / "startup_response.json").read_text())

    response = client.post("/predict", json=request)

    assert response.status_code == 200
    assert response.json() == expected


def test_naive_lagos_timestamp_is_accepted() -> None:
    response = client.post(
        "/predict",
        json={
            "light_lux": 42_000,
            "temperature_c": 38.5,
            "timestamp": "2026-07-22T12:30:00",
        },
    )

    assert response.status_code == 200


def test_invalid_physical_measurement_is_rejected() -> None:
    response = client.post(
        "/predict",
        json={
            "light_lux": -1,
            "temperature_c": 38.5,
            "timestamp": "2026-07-22T12:30:00+01:00",
        },
    )

    assert response.status_code == 422


def test_out_of_collection_range_is_reported() -> None:
    response = client.post(
        "/predict",
        json={
            "light_lux": 150_000,
            "temperature_c": 38.5,
            "timestamp": "2026-07-22T07:00:00+01:00",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["within_training_range"] is False
    assert len(body["warnings"]) == 2
    assert body["max_power_point"]["power_w"] > 0


def test_bh1750_standard_range_warning_is_reported() -> None:
    response = client.post(
        "/predict",
        json={
            "light_lux": 65_000,
            "temperature_c": 38.5,
            "timestamp": "2026-07-22T12:30:00+01:00",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["within_training_range"] is False
    assert "BH1750 standard range" in body["warnings"][0]
