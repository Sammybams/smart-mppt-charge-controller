"""Shared Lagos-local light and solar features for training and inference."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


LAGOS_TIMEZONE = ZoneInfo("Africa/Lagos")
SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.2425
LAGOS_LATITUDE_DEGREES = 6.5244
LAGOS_LONGITUDE_DEGREES = 3.3792
LAGOS_UTC_OFFSET_MINUTES = 60.0
BH1750_STANDARD_MAX_LUX = 65_535.0


def to_lagos_local(timestamp: datetime) -> datetime:
    """Interpret naive datetimes as Lagos time and convert aware values."""
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=LAGOS_TIMEZONE)
    return timestamp.astimezone(LAGOS_TIMEZONE)


def calendar_features(timestamps: pd.Series) -> pd.DataFrame:
    """Encode daily and annual cycles without a midnight/year discontinuity."""
    values = pd.to_datetime(timestamps, errors="raise")
    seconds = (
        values.dt.hour * 3600
        + values.dt.minute * 60
        + values.dt.second
        + values.dt.microsecond / 1_000_000
    )
    day_angle = 2 * np.pi * seconds / SECONDS_PER_DAY
    year_angle = 2 * np.pi * (values.dt.dayofyear - 1 + seconds / SECONDS_PER_DAY) / DAYS_PER_YEAR
    return pd.DataFrame(
        {
            "hour_sin": np.sin(day_angle),
            "hour_cos": np.cos(day_angle),
            "day_of_year_sin": np.sin(year_angle),
            "day_of_year_cos": np.cos(year_angle),
        },
        index=timestamps.index,
    )


def solar_features(timestamps: pd.Series, light_lux: pd.Series) -> pd.DataFrame:
    """Approximate Lagos solar elevation and a lux/clear-sky ratio.

    The position equations are the compact NOAA approximation. The ratio is a
    sensor-domain context feature, not a conversion from lux to W/m2.
    """
    values = pd.to_datetime(timestamps, errors="raise")
    decimal_hour = (
        values.dt.hour
        + values.dt.minute / 60
        + values.dt.second / 3600
        + values.dt.microsecond / 3_600_000_000
    )
    gamma = 2 * np.pi / 365 * (
        values.dt.dayofyear - 1 + (decimal_hour - 12) / 24
    )
    equation_of_time_minutes = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    solar_minutes = (
        decimal_hour * 60
        + equation_of_time_minutes
        + 4 * LAGOS_LONGITUDE_DEGREES
        - LAGOS_UTC_OFFSET_MINUTES
    ) % 1440
    hour_angle = np.radians(solar_minutes / 4 - 180)
    latitude = np.radians(LAGOS_LATITUDE_DEGREES)
    elevation_sin = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    ).clip(-1, 1)
    daylight_factor = elevation_sin.clip(lower=0)
    expected_clear_lux = BH1750_STANDARD_MAX_LUX * daylight_factor.clip(lower=0.08)
    clearness_proxy = (light_lux.astype(float) / expected_clear_lux).clip(0, 2)
    return pd.DataFrame(
        {
            "solar_elevation_sin": elevation_sin,
            "daylight_factor": daylight_factor,
            "clearness_proxy": clearness_proxy,
            "sensor_range_ratio": (
                light_lux.astype(float) / BH1750_STANDARD_MAX_LUX
            ).clip(0, 2),
        },
        index=timestamps.index,
    )


def environmental_features(
    timestamps: pd.Series, light_lux: pd.Series
) -> pd.DataFrame:
    return pd.concat(
        [calendar_features(timestamps), solar_features(timestamps, light_lux)],
        axis=1,
    )


def calendar_features_for_timestamp(timestamp: datetime) -> dict[str, float]:
    local = to_lagos_local(timestamp).replace(tzinfo=None)
    frame = calendar_features(pd.Series([local]))
    return {column: float(frame.iloc[0][column]) for column in frame.columns}


def environmental_features_for_timestamp(
    timestamp: datetime, light_lux: float
) -> dict[str, float]:
    local = to_lagos_local(timestamp).replace(tzinfo=None)
    frame = environmental_features(pd.Series([local]), pd.Series([light_lux]))
    return {column: float(frame.iloc[0][column]) for column in frame.columns}
