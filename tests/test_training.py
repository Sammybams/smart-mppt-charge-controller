import joblib

from smart_mppt.augmentation import generate_augmented_samples
from smart_mppt.manual_dataset import DEFAULT_DATASET_PATH
from smart_mppt.training import train


def test_training_uses_day_isolated_validation(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    augmented_path = tmp_path / "augmented.csv"
    generate_augmented_samples(2_000).to_csv(augmented_path, index=False)
    report = train(
        dataset_path=DEFAULT_DATASET_PATH,
        augmented_path=augmented_path,
        model_path=model_path,
        report_path=tmp_path / "report.json",
    )
    artifact = joblib.load(model_path)

    assert artifact["artifact_version"] == 4
    assert artifact["panel_rating_w"] == 30
    assert artifact["light_unit"] == "lux"
    assert report["field_split_policy"] == (
        "Leave one complete Lagos collection date out per fold"
    )
    assert report["collection_days"] == 4
    assert len(report["field_per_day_metrics"]) == 4
    assert report["field_leave_one_day_out_metrics"]["voltage_mae_v"] < 2.5
    assert report["augmented_training_rows"] == 2_000
    assert "physics model" in report["model_type"]["current"]
    assert artifact["current_physics_blend"] == 0.2
    assert artifact["panel_specifications"]["model"] == "AP-PM-30W"
    calibration = artifact["voltage_label_calibration"]
    assert calibration["raw_rows_above_nameplate_voc"] == 7_785
    assert 0.79 < calibration["scale_factor"] < 0.81
    assert artifact["output_constraints"] == {
        "maximum_voltage_v": 23.16,
        "maximum_current_a": 1.67,
        "maximum_power_w": 33.0,
    }
