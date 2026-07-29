# AI Enabled Smart MPPT Charge Controller

This service accepts the five measurements supplied once when the device starts
and predicts the maximum power point (MPP) to which the charge controller
should move before searching locally to the left and right.

The packaged model is trained from the public UCP photovoltaic I-V curve
dataset and supports both uniform irradiance and partial-shading examples.

## Request and response

Send:

```json
{
  "sun_intensity": 850,
  "panel_voltage": 24.5,
  "panel_current": 7.8,
  "ambient_temperature": 32,
  "time_of_day": "12:30:00"
}
```

Receive:

```json
{
  "max_power_point": {
    "voltage": 28.678,
    "current": 7.077,
    "power": 202.957
  },
  "within_training_range": true,
  "warnings": []
}
```

The field units are:

- `sun_intensity`: W/m²
- `panel_voltage`: volts
- `panel_current`: amperes
- `ambient_temperature`: °C
- `time_of_day`: local `HH:MM` or `HH:MM:SS`
- predicted voltage/current/power: volts, amperes, and watts

`within_training_range` is `false` when a request is outside conditions found
in the training data. A prediction is still returned, and `warnings` identifies
the values that were out of range.

## Run locally

Python 3.11 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
smart-mppt-api
```

The service listens on `http://localhost:8000`. Interactive API documentation
is available at `http://localhost:8000/docs`.

Run the command from the project directory. If the packaged model is stored
elsewhere, set `SMART_MPPT_MODEL_PATH` to its absolute path.

Call it with:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data @examples/startup_request.json
```

Check service and model readiness:

```bash
curl http://localhost:8000/health
```

## Run with Docker

```bash
docker build -t smart-mppt .
docker run --rm -p 8000:8000 smart-mppt
```

## Dataset and training

The project uses:

> kalaimohan T S (2021), “PV Panel: Irradiance, Temperature, Partial
> Shading - IV Curves”, Mendeley Data, V1,
> <https://doi.org/10.17632/z93gzbptf7.1>

The source is licensed under CC BY 4.0. Attribution, archive checksum, and
derivation details are in [`data/SOURCE.md`](data/SOURCE.md).

The two compact source summaries needed for training are committed so the
project works offline. To independently download and verify the original
74 MB archive:

```bash
python scripts/download_dataset.py
```

To reproduce the processed startup samples and model:

```bash
python scripts/prepare_dataset.py
python scripts/train_model.py
```

Training converts each published curve into examples with exactly the five API
inputs. For partial-shading curves, the published local peak with the greatest
power becomes the global MPP target. Because UCP does not contain time of day,
representative startup times are label-preserving augmentation: the time
changes while the physical curve and its target remain the same.

The committed training report is in
[`models/training_report.json`](models/training_report.json). Its group-based
holdout keeps complete source curves together and reports:

- MPP voltage MAE: 0.563 V
- MPP current MAE: 0.022 A
- Maximum power MAE: 4.696 W

## Test

```bash
pytest
```

The tests verify the exact sample contract, input validation, range warnings,
packaged-model inference, and reproducible UCP data preparation.
