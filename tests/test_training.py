import joblib

from smart_mppt.augmentation import DEFAULT_AUGMENTED_PATH
from smart_mppt.manual_dataset import DEFAULT_DATASET_PATH
from smart_mppt.training import train


def test_training_uses_day_isolated_validation(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    report = train(
        dataset_path=DEFAULT_DATASET_PATH,
        augmented_path=DEFAULT_AUGMENTED_PATH,
        model_path=model_path,
        report_path=tmp_path / "report.json",
    )
    artifact = joblib.load(model_path)

    assert artifact["artifact_version"] == 3
    assert artifact["panel_rating_w"] == 30
    assert artifact["light_unit"] == "lux"
    assert report["field_split_policy"] == (
        "Leave one complete Lagos collection date out per fold"
    )
    assert report["collection_days"] == 4
    assert len(report["field_per_day_metrics"]) == 4
    assert report["field_leave_one_day_out_metrics"]["voltage_mae_v"] < 2.5
    assert report["augmented_training_rows"] == 30_000
    assert "Monotonic" in report["model_type"]["current"]
    assert artifact["output_constraints"]["maximum_power_w"] == 33
