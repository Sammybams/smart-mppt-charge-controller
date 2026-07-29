"""Prepare model-ready startup samples from the UCP PV curve summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "source"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
OUTPUT_COLUMNS = [
    "condition_id",
    "source_condition",
    "sun_intensity_w_m2",
    "panel_voltage_v",
    "panel_current_a",
    "ambient_temperature_c",
    "time_of_day_hour",
    "max_power_voltage_v",
    "max_power_current_a",
    "max_power_w",
]

# UCP contains no time-of-day field. These representative startup times retain
# the requested API feature without inventing a relationship between clock time
# and MPP: each physical curve is repeated at every time with the same label.
REPRESENTATIVE_STARTUP_HOURS = (8.0, 10.0, 12.0, 14.0, 16.0)


@dataclass(frozen=True)
class OperatingPoint:
    voltage: float
    current: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_points(points: Iterable[OperatingPoint]) -> list[OperatingPoint]:
    return [
        point
        for point in points
        if np.isfinite(point.voltage)
        and np.isfinite(point.current)
        and point.voltage >= 0
        and point.current >= 0
    ]


def _sample_rows(
    *,
    condition_id: str,
    source_condition: str,
    irradiance: float,
    temperature: float,
    points: Iterable[OperatingPoint],
    target_voltage: float,
    target_current: float,
    target_power: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for point in _finite_points(points):
        for hour in REPRESENTATIVE_STARTUP_HOURS:
            rows.append(
                {
                    "condition_id": condition_id,
                    "source_condition": source_condition,
                    "sun_intensity_w_m2": irradiance,
                    "panel_voltage_v": point.voltage,
                    "panel_current_a": point.current,
                    "ambient_temperature_c": temperature,
                    "time_of_day_hour": hour,
                    "max_power_voltage_v": target_voltage,
                    "max_power_current_a": target_current,
                    "max_power_w": target_power,
                }
            )
    return rows


def prepare_uniform(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    prepared: list[dict[str, float | str]] = []
    for row_number, row in frame.iterrows():
        target_voltage = float(row["V_MP"])
        target_current = float(row["I_MP"])
        points = [
            OperatingPoint(0.0, float(row["I_SC"])),
            OperatingPoint(float(row["V_1"]), float(row["I_1"])),
            OperatingPoint(target_voltage, target_current),
            OperatingPoint(float(row["V_2"]), float(row["I_2"])),
            OperatingPoint(float(row["V_OC"]), 0.0),
        ]
        prepared.extend(
            _sample_rows(
                condition_id=f"uniform-{row_number:04d}",
                source_condition="uniform",
                irradiance=float(row["Irradiance"]),
                temperature=float(row["Temperature"]),
                points=points,
                target_voltage=target_voltage,
                target_current=target_current,
                target_power=float(row["P_MP"]),
            )
        )
    return prepared


def prepare_partial(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    prepared: list[dict[str, float | str]] = []
    for row_number, row in frame.iterrows():
        first_power = float(row["P_MP,0"])
        second_power = float(row["P_MP,1"])
        peak_index = 0 if first_power >= second_power else 1
        target_voltage = float(row[f"V_MP,{peak_index}"])
        target_current = float(row[f"I_MP,{peak_index}"])
        target_power = float(row[f"P_MP,{peak_index}"])

        points = [
            OperatingPoint(0.0, float(row["I_SC"])),
            *[
                OperatingPoint(float(row[f"V_{index}"]), float(row[f"I_{index}"]))
                for index in range(4)
            ],
            OperatingPoint(float(row["V_MP,0"]), float(row["I_MP,0"])),
            OperatingPoint(float(row["V_MP,1"]), float(row["I_MP,1"])),
            OperatingPoint(float(row["V_OC"]), 0.0),
        ]
        shading = float(row["Partial_Shading"])
        prepared.extend(
            _sample_rows(
                condition_id=f"partial-{row_number:04d}",
                source_condition=f"partial_{shading:.1f}",
                irradiance=float(row["Irradiance"]),
                temperature=float(row["Temperature"]),
                points=points,
                target_voltage=target_voltage,
                target_current=target_current,
                target_power=target_power,
            )
        )
    return prepared


def prepare_dataset(
    source_directory: Path = SOURCE_DIRECTORY,
    output_directory: Path = PROCESSED_DIRECTORY,
) -> tuple[Path, Path]:
    uniform_path = source_directory / "uniform_irradiance_summary.csv"
    partial_path = source_directory / "partial_shading_summary.csv"
    missing = [path for path in (uniform_path, partial_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing source data: {names}. Run scripts/download_dataset.py first."
        )

    uniform = pd.read_csv(uniform_path)
    partial = pd.read_csv(partial_path)
    rows = prepare_uniform(uniform) + prepare_partial(partial)
    dataset = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    dataset = dataset.drop_duplicates().reset_index(drop=True)

    output_directory.mkdir(parents=True, exist_ok=True)
    dataset_path = output_directory / "startup_training.csv"
    metadata_path = output_directory / "startup_training.metadata.json"
    dataset.to_csv(dataset_path, index=False)

    metadata = {
        "dataset": "PV Panel: Irradiance, Temperature, Partial Shading - IV Curves",
        "doi": "10.17632/z93gzbptf7.1",
        "licence": "CC BY 4.0",
        "derivation": (
            "Each UCP curve is converted into startup operating-point samples. "
            "For partial shading, the larger published local-peak power is the "
            "global target. Time values are label-preserving augmentation."
        ),
        "source_files": {
            uniform_path.name: file_sha256(uniform_path),
            partial_path.name: file_sha256(partial_path),
        },
        "source_curve_count": int(
            dataset["condition_id"].nunique()
        ),
        "training_row_count": int(len(dataset)),
        "source_condition_counts": {
            str(key): int(value)
            for key, value in dataset["source_condition"].value_counts().items()
        },
        "columns": OUTPUT_COLUMNS,
        "representative_startup_hours": list(REPRESENTATIVE_STARTUP_HOURS),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return dataset_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIRECTORY)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIRECTORY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path, metadata_path = prepare_dataset(args.source, args.output)
    print(f"Prepared {dataset_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

