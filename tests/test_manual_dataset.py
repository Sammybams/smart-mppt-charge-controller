import json

import numpy as np
import pandas as pd

from smart_mppt.manual_dataset import prepare_manual_dataset


def test_manual_preparation_is_reproducible_and_time_aware(tmp_path) -> None:
    dataset_path, metadata_path = prepare_manual_dataset(
        dataset_path=tmp_path / "training.csv",
        metadata_path=tmp_path / "metadata.json",
    )
    frame = pd.read_csv(dataset_path)
    metadata = json.loads(metadata_path.read_text())

    assert len(frame) == 11_581
    assert metadata["rows_received"] == 12_477
    assert metadata["exact_duplicates_removed"] == 294
    assert metadata["timestamp_rows_aggregated"] == 602
    assert metadata["conflicting_timestamp_groups"] == 601
    assert metadata["collection_dates"] == [
        "2026-06-22",
        "2026-06-23",
        "2026-07-21",
        "2026-07-22",
    ]
    assert metadata["light_unit"] == "lux"
    assert metadata["panel_rating_w"] == 30
    assert frame["timestamp"].is_monotonic_increasing
    assert np.allclose(
        frame["max_power_w"],
        frame["max_power_voltage_v"] * frame["max_power_current_a"],
    )
    assert frame[["hour_sin", "hour_cos"]].abs().le(1).all().all()
    assert frame["solar_elevation_sin"].between(-1, 1).all()
    assert frame["daylight_factor"].between(0, 1).all()
    assert frame["sensor_range_ratio"].between(0, 1).all()
