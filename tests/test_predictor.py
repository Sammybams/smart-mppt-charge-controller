from datetime import datetime

from smart_mppt.predictor import MPPPredictor, StartupMeasurement


def test_packaged_model_predicts_a_30w_panel_mpp() -> None:
    result = MPPPredictor().predict(
        StartupMeasurement(
            light_lux=42_000,
            temperature_c=38.5,
            timestamp=datetime.fromisoformat("2026-07-22T12:30:00+01:00"),
        )
    )

    assert 0 < result.voltage < 30
    assert 0 < result.current < 2
    assert result.power > 0
    assert result.within_training_range is True
    assert result.warnings == ()


def test_naive_timestamp_is_interpreted_as_lagos_time() -> None:
    predictor = MPPPredictor()
    naive = predictor.predict(
        StartupMeasurement(42_000, 38.5, datetime.fromisoformat("2026-07-22T12:30:00"))
    )
    lagos = predictor.predict(
        StartupMeasurement(
            42_000, 38.5, datetime.fromisoformat("2026-07-22T12:30:00+01:00")
        )
    )

    assert naive == lagos
