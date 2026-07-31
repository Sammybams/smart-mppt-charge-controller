# AI Enabled Smart MPPT Charge Controller

This project provides a one-shot startup prediction for the 30 W photovoltaic
panel measured in Lagos, Nigeria. The controller sends light, temperature, and
timestamp once; the API predicts an expected maximum-power-point (MPP) voltage
and current. Firmware can move near that point and then search locally to the
left and right.

The packaged production model is trained on the supplied
`data/Manual_Collection.csv`, not on the earlier public UCP model. UCP remains
in the repository as a reproducible partial-shading research benchmark, but it
describes a simulated 250 W panel and uses irradiance in W/m2 rather than lux.
Those measurements are not mixed as though they were equivalent.

## API contract

Send one `POST /predict` request:

```json
{
  "light_lux": 42000,
  "temperature_c": 38.5,
  "timestamp": "2026-07-22T12:30:00+01:00"
}
```

The service returns:

```json
{
  "max_power_point": {
    "voltage_v": 25.829,
    "current_a": 1.15,
    "power_w": 29.704
  },
  "within_training_range": true,
  "warnings": []
}
```

The timestamp should contain its UTC offset. A timestamp without an offset is
interpreted as Lagos time (`Africa/Lagos`). `power_w` is calculated from the
unrounded voltage and current predictions; it is not a separately fitted
target.

The range flag does not guarantee prediction accuracy. It indicates whether
lux, temperature, local clock time, and day of year fall inside the conditions
represented in the four collection days. A prediction is still returned for
an out-of-range request, with a warning for each extrapolated input.

## Run locally

Python 3.11 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
smart-mppt-api
```

The service listens on `http://localhost:8000`; interactive OpenAPI
documentation is at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data @examples/startup_request.json

curl http://localhost:8000/health
```

If the model is elsewhere, set `SMART_MPPT_MODEL_PATH` to its absolute path.

## Run with Docker

```bash
docker build -t smart-mppt .
docker run --rm -p 8000:8000 smart-mppt
```

## Reproduce preparation and training

The original manual CSV is committed unchanged. Rebuild all generated data and
the model with:

```bash
python scripts/prepare_manual_dataset.py
python scripts/train_model.py
```

Preparation produces ignored, reproducible files:

- `data/processed/lagos_30w_training.csv`
- `data/processed/lagos_30w_training.metadata.json`

Training replaces the committed runtime artifacts:

- `models/smart_mppt.joblib`
- `models/training_report.json`

The detailed source audit, preprocessing rules, time encoding, model selection,
validation, metrics, limitations, and runtime transformation are documented in
[`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md). The local data's
provenance and field contract are in
[`data/MANUAL_COLLECTION.md`](data/MANUAL_COLLECTION.md).

## Why this is time-aware, but not an LSTM

The collection contains 11,581 prepared timestamp readings but only four
independent days. Adjacent readings are highly related, so counting rows alone
would greatly overstate the amount of sequence evidence. An LSTM could easily
memorize those few daily traces and appear strong under a random row split.

The implemented model instead uses:

- cyclic local time-of-day features;
- cyclic day-of-year features;
- lux and `log1p(lux)`;
- ambient temperature; and
- leave-one-complete-day-out validation.

This preserves the useful daily/seasonal context while keeping the API truly
one-shot—no hidden recurrent state or mandatory history window is needed.

## Current measured performance

Across leave-one-day-out predictions, where an entire date is unseen during
each fold:

| Metric | Result |
| --- | ---: |
| MPP voltage MAE | 1.882936 V |
| MPP voltage R2 | 0.649073 |
| MPP current MAE | 0.173991 A |
| MPP current R2 | -0.000220 |
| Derived power MAE | 4.684770 W |

Voltage has a learnable relationship in the supplied data. Current does not:
the current values are nearly uniformly quantized from 0.80 A to 1.49 A and no
tested feature model beat a day-isolated median baseline. The shipped current
regressor therefore returns the training median instead of fitting noise. This
is an explicit limitation, not a hidden success claim.

## Does it solve global MPPT under partial shading?

It implements the requested startup predictor using the field meanings
confirmed by the project owner. It can give the controller a useful initial
location and reduce a search from zero.

It does **not yet prove** that the returned point is the global maximum under
arbitrary partial shading. Each manual row contains one confirmed target
voltage/current pair, not a contemporaneous full I-V or P-V sweep containing
all local peaks. A supervised model can only learn the supplied labels; it
cannot verify that an unmeasured, higher peak did not exist.

For rigorous global-peak training, collect repeated fast I-V sweeps at varied
shading patterns and store, per sweep, lux or calibrated irradiance,
temperature, timestamp, every voltage/current point, and the selected global
peak. Until then, firmware should use the prediction as a starting point and
retain the requested left/right safety search.

## Public datasets and why they are not merged

The original [UCP dataset](https://doi.org/10.17632/z93gzbptf7.1) contains
uniform and partial-shading curves with identified peaks, but it is a simulated
250 W, 60-cell panel dataset with irradiance in W/m2. Normalizing voltage,
current, and power by nameplate ratings does not solve the missing calibrated
lux-to-irradiance mapping or different panel topology.

A closer published study uses a real 30 W ET-M53630WW panel and records
irradiance, temperature, voltage, and current across 55 I-V sweeps. It is useful
for designing the next Lagos collection, but it uses W/m2, a different aged
panel in Morocco, and uniform irradiance; merging it would introduce a domain
shift rather than reliably augment this sensor-panel pair. See the
[Scientific Reports study](https://doi.org/10.1038/s41598-026-39626-w).

An LLM is not suitable for the numeric control path. It cannot provide the
deterministic, low-latency, calibrated regression needed by an embedded MPPT
controller. A small tabular model is the appropriate tool for the data
currently available.

## Test

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The environment variable prevents unrelated globally installed pytest plugins
from affecting the project suite. Tests cover raw-data preparation, timestamp
encoding, day-isolated training, packaged inference, API validation, exact
examples, and range warnings.
