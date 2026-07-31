"""Create reproducible 30 W physics-guided samples in the BH1750 lux domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smart_mppt.dataset import PROCESSED_DIRECTORY, file_sha256
from smart_mppt.panel import PANEL
from smart_mppt.time_features import (
    BH1750_STANDARD_MAX_LUX,
    environmental_features,
)


DEFAULT_AUGMENTED_PATH = PROCESSED_DIRECTORY / "lagos_30w_augmented.csv"
DEFAULT_AUGMENTED_METADATA_PATH = (
    PROCESSED_DIRECTORY / "lagos_30w_augmented.metadata.json"
)
RANDOM_SEED = 42
DEFAULT_SAMPLE_COUNT = 30_000

PANEL_POWER_W = PANEL.maximum_power_w
PANEL_VMP_V = PANEL.maximum_power_voltage_v
PANEL_IMP_A = PANEL.maximum_power_current_a
PANEL_VOC_V = PANEL.open_circuit_voltage_v
PANEL_ISC_A = PANEL.short_circuit_current_a
CURRENT_TEMPERATURE_COEFFICIENT = 0.0006
VOLTAGE_TEMPERATURE_COEFFICIENT = -0.0035
MAX_REASONABLE_POWER_W = 33.0


def _candidate_timestamps(rng: np.random.Generator, count: int) -> pd.Series:
    day_offsets = rng.integers(0, 365, size=count)
    seconds = rng.integers(5 * 3600 + 30 * 60, 19 * 3600 + 31 * 60, size=count)
    return pd.Series(
        pd.Timestamp("2026-01-01")
        + pd.to_timedelta(day_offsets, unit="D")
        + pd.to_timedelta(seconds, unit="s")
    )


def generate_augmented_samples(
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate expected global-MPP targets without changing the API unit."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(random_seed)

    # Draw extra candidates because some sampled clock times are below the
    # horizon and are not useful for charge-controller operation.
    candidate_count = max(sample_count * 2, 1_000)
    timestamps = _candidate_timestamps(rng, candidate_count)
    provisional_lux = pd.Series(np.ones(candidate_count), index=timestamps.index)
    solar = environmental_features(timestamps, provisional_lux)
    daylight = solar["daylight_factor"].to_numpy()
    daylight_indices = np.flatnonzero(daylight > 0.02)
    if len(daylight_indices) < sample_count:
        raise RuntimeError("Not enough daylight timestamps were generated")
    selected = daylight_indices[:sample_count]
    timestamps = timestamps.iloc[selected].reset_index(drop=True)
    daylight = daylight[selected]

    # Transmission covers clear sky, clouds, and strong local/partial shade.
    shaded = rng.random(sample_count) < 0.35
    transmission = np.where(
        shaded,
        10 ** rng.uniform(-2.0, -0.08, sample_count),
        0.25 + 0.9 * rng.beta(2.5, 1.4, sample_count),
    ).clip(0.01, 1.15)
    true_lux = BH1750_STANDARD_MAX_LUX * daylight * transmission

    # The BH1750 datasheet allows substantial unit-to-unit/optical variation.
    # Targets come from the latent light; the noisy sensor value is the input.
    sensor_factor = rng.normal(1.0, 0.12, sample_count).clip(0.8, 1.2)
    measured_lux = (true_lux * sensor_factor).clip(1.0, 100_000.0)
    temperature = (
        27.0 + 18.0 * daylight + 5.0 * transmission + rng.normal(0, 2.5, sample_count)
    ).clip(24.0, 58.0)

    normalized_light = (true_lux / BH1750_STANDARD_MAX_LUX).clip(0, 1.2)
    current = (
        PANEL_IMP_A
        * normalized_light**0.92
        * (1 + CURRENT_TEMPERATURE_COEFFICIENT * (temperature - 25.0))
    )

    # Spatial shade can activate bypass diodes. Lux alone cannot identify the
    # pattern, so these branches teach the expected rather than guaranteed peak.
    shade_strength = (1 - transmission).clip(0, 1)
    branch_draw = rng.random(sample_count)
    bypass_fraction = np.ones(sample_count)
    bypass_fraction[branch_draw < 0.28 * shade_strength] = 2 / 3
    bypass_fraction[branch_draw < 0.08 * shade_strength] = 1 / 3
    current *= (1 - 0.12 * shade_strength).clip(0.75, 1)

    voltage_light_factor = (
        (1 - np.exp(-45 * normalized_light))
        * (0.74 + 0.26 * np.log1p(9 * normalized_light) / np.log(10))
    ).clip(0, 1.04)
    voltage = (
        PANEL_VMP_V
        * voltage_light_factor
        * (1 + VOLTAGE_TEMPERATURE_COEFFICIENT * (temperature - 25.0))
        * bypass_fraction
    ).clip(0, PANEL_VOC_V)
    current = current.clip(0, PANEL_ISC_A)

    power = voltage * current
    excessive = power > MAX_REASONABLE_POWER_W
    current[excessive] *= MAX_REASONABLE_POWER_W / power[excessive]
    power = voltage * current

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "light_lux": measured_lux,
            "log_light_lux": np.log1p(measured_lux),
            "temperature_c": temperature,
        }
    )
    frame = pd.concat(
        [
            frame,
            environmental_features(frame["timestamp"], frame["light_lux"]),
        ],
        axis=1,
    )
    frame["max_power_voltage_v"] = voltage
    frame["max_power_current_a"] = current
    frame["max_power_w"] = power
    frame["source"] = "physics_augmented"
    return frame


def prepare_augmented_dataset(
    dataset_path: Path = DEFAULT_AUGMENTED_PATH,
    metadata_path: Path = DEFAULT_AUGMENTED_METADATA_PATH,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    random_seed: int = RANDOM_SEED,
) -> tuple[Path, Path]:
    frame = generate_augmented_samples(sample_count, random_seed)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dataset_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    metadata = {
        "dataset": "Physics-guided 30 W BH1750 lux augmentation",
        "rows": int(len(frame)),
        "random_seed": random_seed,
        "light_unit": "lux",
        "purpose": (
            "Provide physically varying current and partial-shading examples "
            "while preserving the deployed lux input."
        ),
        "panel_nameplate": {
            "manufacturer": PANEL.manufacturer,
            "model": PANEL.model,
            "rated_power_w": PANEL_POWER_W,
            "vmp_v": PANEL_VMP_V,
            "imp_a": PANEL_IMP_A,
            "voc_v": PANEL_VOC_V,
            "isc_a": PANEL_ISC_A,
            "status": "Read from the supplied AP-PM-30W panel label.",
        },
        "assumptions": [
            "BH1750 input variation is sampled within plus/minus 20 percent.",
            "Lux remains the model input and is not exposed as W/m2.",
            "Cloud and shade transmission spans 1 to 115 percent of the clear reference.",
            "Bypass branches represent expected one-third and two-thirds voltage regions.",
            "Maximum synthetic power is limited to 33 W.",
            "Temperature coefficients remain engineering assumptions because they are not printed on the panel label.",
        ],
        "dataset_sha256": file_sha256(dataset_path),
        "columns": frame.columns.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return dataset_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_AUGMENTED_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_AUGMENTED_METADATA_PATH)
    parser.add_argument("--rows", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path, metadata_path = prepare_augmented_dataset(
        args.dataset, args.metadata, args.rows, args.seed
    )
    print(f"Prepared {dataset_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
