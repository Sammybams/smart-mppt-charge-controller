import json

import pandas as pd

from smart_mppt.dataset import prepare_dataset


def test_ucp_preparation_is_reproducible(tmp_path) -> None:
    dataset_path, metadata_path = prepare_dataset(output_directory=tmp_path)
    frame = pd.read_csv(dataset_path)
    metadata = json.loads(metadata_path.read_text())

    assert len(frame) == 14_725
    assert frame["condition_id"].nunique() == 496
    assert set(frame["source_condition"]) == {
        "uniform",
        "partial_0.1",
        "partial_0.2",
        "partial_0.3",
        "partial_0.4",
        "partial_0.5",
    }
    assert (frame["max_power_w"] > 0).all()
    assert (frame["max_power_voltage_v"] > 0).all()
    assert (frame["max_power_current_a"] > 0).all()
    assert metadata["doi"] == "10.17632/z93gzbptf7.1"
    assert metadata["training_row_count"] == len(frame)

