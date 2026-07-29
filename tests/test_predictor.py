from smart_mppt.predictor import MPPPredictor, StartupMeasurement


def test_packaged_model_predicts_a_positive_mpp() -> None:
    result = MPPPredictor().predict(
        StartupMeasurement(
            sun_intensity=900,
            panel_voltage=25,
            panel_current=8,
            ambient_temperature=30,
            time_of_day_hour=12,
        )
    )

    assert result.voltage > 0
    assert result.current > 0
    assert result.power > 0
    assert result.within_training_range is True
    assert result.warnings == ()
