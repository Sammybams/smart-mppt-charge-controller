"""Shared Lagos-local calendar features for training and inference."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


LAGOS_TIMEZONE = ZoneInfo("Africa/Lagos")
SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.2425


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


def calendar_features_for_timestamp(timestamp: datetime) -> dict[str, float]:
    local = to_lagos_local(timestamp).replace(tzinfo=None)
    frame = calendar_features(pd.Series([local]))
    return {column: float(frame.iloc[0][column]) for column in frame.columns}
