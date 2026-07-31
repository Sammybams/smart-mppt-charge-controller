# Lagos 30 W manual collection

`Manual_Collection.csv` is the project-specific field dataset and the primary
source for the production model.

## Provenance

- Location: Lagos, Nigeria
- Panel rating: 30 W
- Collection dates: 2026-06-22, 2026-06-23, 2026-07-21, and 2026-07-22
- Rows as received: 12,477
- SHA-256: `09ce21869b7133c6319af2a574daf190a3049abcf0566d383d92a755c7b9b734`

The original CSV is retained unchanged so that every preprocessing decision is
reproducible.

## Authoritative field contract

The project owner confirmed the following interpretation:

| Source column | Meaning | Unit | Model role |
| --- | --- | --- | --- |
| `LIGHT` | Light measured by the GY-302/BH1750 sensor | lux | Input |
| `TEMPERATURE` | Ambient temperature | degrees Celsius | Input |
| `TIME` | Local measurement time in Lagos | `M/D/YYYY HH:MM:SS` | Input |
| `PANEL_VOLTAGE` | Voltage at the maximum-power operating point | volts | Target |
| `PANEL_CURRENT` | Current at the maximum-power operating point | amperes | Target |

`BATTERY_CURRENT`, `BATTERY_VOLTAGE`, both unnamed columns, and `EXPLANATION`
are not model features or targets. The first-row explanation is retained as
received, but the owner-confirmed contract above governs the training pipeline.

## Important limitations

- Only four collection dates are represented. The timestamps contain useful
  solar-cycle context, but there is not enough independent day-level history to
  justify an LSTM or another sequence-heavy neural network.
- Duplicate rows and duplicate timestamps exist. Preparation removes exact
  duplicates and aggregates conflicting readings from the same second by their
  median.
- `PANEL_CURRENT` is quantized in 0.01 A steps from 0.80 A to 1.49 A. This
  limits the current model's attainable accuracy and should be reviewed against
  the sensor/logger implementation.
- Recorded voltage multiplied by current can exceed the nominal 30 W rating.
  The original labels are preserved. Runtime predictions are limited to 33 W.
- Lux is illuminance, not solar irradiance. It is not directly interchangeable
  with the W/m2 irradiance in the public UCP dataset without a calibrated,
  spectrum- and sensor-specific conversion.

The production current prediction now blends a physics-guided lux model with
the Lagos median current. This gives a light-responsive result while keeping
the uncertain generated data at only 20% of the current estimate.
