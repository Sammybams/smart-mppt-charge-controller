from datetime import datetime

from smart_mppt.time_features import environmental_features_for_timestamp


def test_lagos_solar_elevation_distinguishes_noon_and_night() -> None:
    noon = environmental_features_for_timestamp(
        datetime.fromisoformat("2026-07-22T12:30:00+01:00"), 42_000
    )
    night = environmental_features_for_timestamp(
        datetime.fromisoformat("2026-07-22T00:30:00+01:00"), 0
    )

    assert noon["solar_elevation_sin"] > 0.9
    assert noon["daylight_factor"] == noon["solar_elevation_sin"]
    assert 0 < noon["clearness_proxy"] < 1
    assert night["solar_elevation_sin"] < 0
    assert night["daylight_factor"] == 0


def test_utc_and_lagos_timestamps_produce_the_same_features() -> None:
    lagos = environmental_features_for_timestamp(
        datetime.fromisoformat("2026-07-22T12:30:00+01:00"), 42_000
    )
    utc = environmental_features_for_timestamp(
        datetime.fromisoformat("2026-07-22T11:30:00+00:00"), 42_000
    )

    assert lagos == utc
