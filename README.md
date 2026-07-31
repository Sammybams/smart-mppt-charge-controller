# AI Enabled Smart MPPT Charge Controller

This project predicts a good starting point for a 30 W solar panel's maximum
power point (MPP).

The device sends only three values:

- light from the GY-302/BH1750 sensor, in lux;
- ambient temperature, in degrees Celsius; and
- the current date and time.

The API returns the expected MPP panel voltage and current. The controller can
move near that point and do a short search to the left and right.

## Simple example

Send this to `POST /predict`:

```json
{
  "light_lux": 42000,
  "temperature_c": 38.5,
  "timestamp": "2026-07-22T12:30:00+01:00"
}
```

Example response:

```json
{
  "max_power_point": {
    "voltage_v": 20.626,
    "current_a": 1.121,
    "power_w": 23.118
  },
  "within_training_range": true,
  "warnings": []
}
```

Use a timestamp with a timezone when possible. If no timezone is supplied, the
API treats it as Lagos time.

## How the model works

The model uses two kinds of training data:

1. **Real Lagos readings.** These teach the model how the actual panel voltage
   behaves with lux, temperature, and time.
2. **Generated 30 W examples.** These teach the current model how solar-panel
   current should change in low light, bright light, clouds, and partial shade.

The generated examples still use lux as their input. The device never needs to
send irradiance in W/m2.

Time is used to calculate the sun's approximate height above Lagos. This is
more useful than treating `12:00` as only a number. The same clock time in
different months can have different sunlight conditions.

The final current is a careful blend:

```text
80% anchor from the middle Lagos current reading
20% current from the physics-guided model
```

A low-light gate reduces current toward zero when lux is very low. This means
current is no longer a fixed value, but the uncertain generated data cannot
overpower the real panel data.

## The exact panel

The supplied nameplate identifies a Sunshine Solar AP-PM-30W made in the Lekki
Free Zone, Nigeria. Its standard-test values are:

- maximum power voltage: 19.3 V;
- maximum power current: 1.56 A;
- open-circuit voltage: 23.16 V;
- short-circuit current: 1.67 A; and
- rated power: 30 W.

Runtime output is limited to 23.16 V, 1.67 A, and 33 W. The extra 3 W is a
small safety margin above the nameplate power, not a new panel rating.

The Lagos voltage logger does not match the nameplate scale: 7,785 prepared
rows are above the panel's 23.16 V open-circuit voltage. Training keeps the raw
CSV unchanged, but multiplies voltage labels by `0.7981803143`. This makes the
brightest quarter's median voltage match the real 19.3 V Vmp. The complete
extraction is in [`data/PANEL_DATASHEET.md`](data/PANEL_DATASHEET.md).

The BH1750 normally measures up to about 65,535 lux. The API warns when a value
is close to or above that standard range. Configure the sensor's extended
range if the hardware uses readings above that level.

## What the result means

The result is the **expected best starting point**. It is not proof that the
point is always the global maximum under every shade pattern.

One light sensor cannot see how shade is spread across individual panel cells.
Two different shadow shapes can produce the same lux value but different power
curves. Keep the short controller verification search after moving to the
predicted voltage.

## Measured results

Testing holds out one complete Lagos day at a time. This is harder and more
honest than randomly mixing neighbouring readings between training and test
data.

| Check | Result |
| --- | ---: |
| Voltage error against calibrated labels | 1.363 V MAE |
| Voltage R2 against calibrated labels | 0.777 |
| Voltage error against raw labels | 5.467 V MAE |
| Current error against Lagos current column | 0.238 A MAE |
| Power error against calibrated Lagos columns | 4.754 W MAE |
| Current error on held-out generated examples | 0.046 A MAE |

The local current column has almost no dependable relationship with lux,
temperature, or time. The hybrid current deliberately follows light more than
the old fixed-median model, so its agreement with that column is worse. The
generated current result checks the implemented assumptions; it is not a claim
of real-world accuracy.

## Run the API

Python 3.11 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
smart-mppt-api
```

Open `http://localhost:8000/docs` for interactive API documentation.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data @examples/startup_request.json
```

## Rebuild the data and model

Run these commands in order:

```bash
python scripts/prepare_manual_dataset.py
python scripts/generate_augmented_dataset.py
python scripts/train_model.py
```

The original CSV is never changed. Generated CSV files are ignored by Git.
The finished model and its report are saved in `models/`.

More detail is available in:

- [`data/MANUAL_COLLECTION.md`](data/MANUAL_COLLECTION.md) — the real data;
- [`data/PANEL_DATASHEET.md`](data/PANEL_DATASHEET.md) — the panel nameplate;
- [`data/AUGMENTATION.md`](data/AUGMENTATION.md) — the generated data;
- [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md) — all training and
  prediction steps; and
- [`models/training_report.json`](models/training_report.json) — exact metrics
  and checksums from the latest training run.

## Run tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Why no LSTM or LLM?

There are many rows but only four separate collection days. An LSTM could
memorize those days instead of learning a general rule. The present tree
models are smaller, faster, easier to test, and work from one startup reading.

An LLM is also unnecessary. This is a numeric control problem that needs fast,
repeatable results without an internet connection.
