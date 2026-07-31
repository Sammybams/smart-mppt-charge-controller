import json

import pandas as pd

from smart_mppt.augmentation import (
    MAX_REASONABLE_POWER_W,
    generate_augmented_samples,
    prepare_augmented_dataset,
)


def test_augmentation_is_reproducible_and_physically_bounded() -> None:
    first = generate_augmented_samples(1_000, random_seed=7)
    second = generate_augmented_samples(1_000, random_seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert (first["light_lux"] > 0).all()
    assert (first["max_power_voltage_v"] >= 0).all()
    assert (first["max_power_current_a"] >= 0).all()
    assert first["max_power_w"].max() <= MAX_REASONABLE_POWER_W + 1e-9

    low = first.nsmallest(100, "light_lux")["max_power_current_a"].median()
    high = first.nlargest(100, "light_lux")["max_power_current_a"].median()
    assert high > low * 4


def test_augmentation_writes_assumptions(tmp_path) -> None:
    dataset_path, metadata_path = prepare_augmented_dataset(
        dataset_path=tmp_path / "augmented.csv",
        metadata_path=tmp_path / "metadata.json",
        sample_count=100,
        random_seed=9,
    )
    metadata = json.loads(metadata_path.read_text())

    assert len(pd.read_csv(dataset_path)) == 100
    assert metadata["rows"] == 100
    assert metadata["light_unit"] == "lux"
    assert metadata["panel_nameplate"]["model"] == "AP-PM-30W"
    assert metadata["panel_nameplate"]["vmp_v"] == 19.3
    assert metadata["panel_nameplate"]["imp_a"] == 1.56
    assert metadata["panel_nameplate"]["voc_v"] == 23.16
    assert metadata["panel_nameplate"]["isc_a"] == 1.67
