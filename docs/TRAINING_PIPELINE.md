# Model training and runtime pipeline

This document describes exactly how the committed smart MPPT model was
constructed, evaluated, saved, and used by the prediction API. It is intended
to make every manipulation between the public source archive and the returned
maximum power point auditable and reproducible.

## 1. Implemented objective

The requested device interaction is a one-shot startup prediction.

The request contains:

| Requested value | API field | Model feature | Unit |
|---|---|---|---|
| Sun intensity | `sun_intensity` | `sun_intensity_w_m2` | W/m² |
| Panel voltage | `panel_voltage` | `panel_voltage_v` | V |
| Panel current | `panel_current` | `panel_current_a` | A |
| Ambient temperature | `ambient_temperature` | `ambient_temperature_c` | °C |
| Time of day | `time_of_day` | `time_of_day_hour` | Decimal local hour |

The model directly predicts:

| Model target | Meaning | Unit |
|---|---|---|
| `max_power_voltage_v` | Voltage at the predicted global MPP | V |
| `max_power_current_a` | Current at the predicted global MPP | A |

The API calculates maximum power as predicted voltage multiplied by predicted
current. Power is not trained as a third independent model output. This keeps
the returned voltage, current, and power physically consistent.

```mermaid
flowchart LR
    A[UCP ZIP archive] --> B[Two published summary CSV files]
    B --> C[Select startup operating points]
    C --> D[Select global MPP labels]
    D --> E[Time-of-day augmentation]
    E --> F[Filter and deduplicate]
    F --> G[Group holdout by complete PV curve]
    G --> H[Evaluate regression model]
    H --> I[Refit on all prepared rows]
    I --> J[Packaged joblib artifact]
    J --> K[POST /predict]
```

## 2. Software and fixed versions

The implementation uses Python 3.11 and the following principal libraries:

- pandas 2.x for loading and creating tabular datasets
- NumPy 1.24–1.x for finite-value checks and numeric operations
- scikit-learn 1.3.0 for splitting, modelling, and metrics
- joblib 1.x for the compressed trained artifact
- FastAPI and Pydantic for the runtime request/response contract
- Uvicorn for serving the HTTP API

The scikit-learn version is fixed to 1.3.0 because persisted scikit-learn
models are not guaranteed to load safely across arbitrary library versions.
All dependency constraints are declared in `pyproject.toml`.

## 3. Public source dataset

The source is:

> kalaimohan T S (2021), “PV Panel: Irradiance, Temperature, Partial
> Shading - IV Curves”, Mendeley Data, V1,
> <https://doi.org/10.17632/z93gzbptf7.1>

It is published under CC BY 4.0.

The Mendeley archive is `PV_Dataset.zip`. The downloader pins both its public
file identifier and SHA-256 checksum:

```text
22e39cc0b074d9ffd09459851c34898a54652ae9113661a118e2cc6270a08ae8
```

`scripts/download_dataset.py` performs these operations:

1. Downloads the archive to `data/raw/PV_Dataset.zip` unless it already exists.
2. Calculates the archive SHA-256.
3. Stops with an error if the checksum differs from the pinned checksum.
4. Extracts only the two publisher-provided inferred summary files.
5. Renames them to stable project filenames under `data/source/`.

The selected source files are:

| Project file | Original archive member | Data rows |
|---|---|---:|
| `uniform_irradiance_summary.csv` | `PV_Dataset/Uniform_Irradiance_Data/new_inferred_file_4line_FS.csv` | 341 |
| `partial_shading_summary.csv` | `PV_Dataset/Partial_Shading_Data/new_inferred_file_7line_PS.csv` | 155 |

The archive also contains much larger point-by-point curve files. They are not
used in this implementation because the publisher's summaries already contain
the operating samples and identified maximum-power points needed to construct
the requested one-shot input/output examples. The raw 74 MB archive is
reproducibly downloadable but excluded from Git; the two small source summaries
are committed so training can run offline.

Further attribution and source-file checksums are recorded in
`data/SOURCE.md` and in the generated processed-data metadata.

## 4. Source conditions

### 4.1 Uniform irradiance

The uniform summary contains 341 curves:

- Irradiance: 500 to 1,000 W/m² in 50 W/m² increments
- Temperature: 20 to 50 °C in 1 °C increments
- One published MPP per curve: `P_MP`, `V_MP`, and `I_MP`
- Open- and short-circuit values: `V_OC` and `I_SC`
- Two additional sampled operating points: `(V_1, I_1)` and `(V_2, I_2)`

### 4.2 Partial shading

The partial-shading summary contains 155 curves:

- Partial-shading fraction: 0.1, 0.2, 0.3, 0.4, or 0.5
- Irradiance: 1,000 W/m²
- Temperature: 20 to 50 °C in 1 °C increments
- Two published local MPP candidates per curve:
  - `P_MP,0`, `V_MP,0`, and `I_MP,0`
  - `P_MP,1`, `V_MP,1`, and `I_MP,1`
- Open- and short-circuit values: `V_OC` and `I_SC`
- Four additional sampled operating points: `(V_0, I_0)` through
  `(V_3, I_3)`

The partial-shading fraction is retained in `source_condition` for auditing,
but it is deliberately not a model input because the original requested
startup payload does not contain a shading measurement.

## 5. Target-label construction

`src/smart_mppt/dataset.py` constructs one global target for each source curve.

For a uniform curve, the publisher's single identified MPP is used unchanged:

```text
target voltage = V_MP
target current = I_MP
reference power = P_MP
```

For a partial-shading curve, the two published local peaks are compared:

```text
global peak index = argmax(P_MP,0, P_MP,1)
target voltage = V_MP,<global peak index>
target current = I_MP,<global peak index>
reference power = P_MP,<global peak index>
```

If the two published powers are exactly equal, peak 0 is selected
deterministically. No target is inferred from the filename, row order, or
partial-shading fraction.

The reference power is retained in the processed data for auditing and
evaluation preparation. Training fits voltage and current only.

## 6. Startup operating-point construction

Each physical curve describes one environmental condition, but a device may
start at different positions on that curve. The preparation step therefore
creates multiple startup measurements from the operating points published in
each summary.

Five startup points are constructed for every uniform curve:

1. Short circuit: `(0, I_SC)`
2. Publisher sample: `(V_1, I_1)`
3. Published MPP: `(V_MP, I_MP)`
4. Publisher sample: `(V_2, I_2)`
5. Open circuit: `(V_OC, 0)`

Eight startup points are constructed for every partial-shading curve:

1. Short circuit: `(0, I_SC)`
2. Publisher sample: `(V_0, I_0)`
3. Publisher sample: `(V_1, I_1)`
4. Publisher sample: `(V_2, I_2)`
5. Publisher sample: `(V_3, I_3)`
6. First published local MPP: `(V_MP,0, I_MP,0)`
7. Second published local MPP: `(V_MP,1, I_MP,1)`
8. Open circuit: `(V_OC, 0)`

Every startup point from a curve receives that curve's same global MPP label.
This teaches the model to return the curve's maximum from different possible
startup voltage/current positions.

## 7. Mapping temperature and time

The UCP field named `Temperature` is mapped directly to the requested
`ambient_temperature` input. It is not corrected, rescaled, or converted.
Deployment sensors should therefore use degrees Celsius and should be
calibrated against the temperature interpretation used by the source setup.

The source dataset does not contain time of day. To retain the exact requested
input without creating a false relationship, every constructed operating point
is repeated at these representative local hours:

```text
08:00, 10:00, 12:00, 14:00, 16:00
```

All five copies keep the same physical inputs and same MPP target. Only time
changes. This is label-preserving augmentation: it lets the runtime accept time
of day while not claiming that clock time changes the target when irradiance,
voltage, current, and temperature are unchanged.

At inference, `HH:MM:SS` is converted to decimal hour using:

```text
hour + minute / 60 + second / 3600
```

No date, timezone, latitude, longitude, or season is added.

## 8. Filtering and final row counts

Before a startup operating point is accepted:

- Voltage and current must both be finite.
- Voltage and current must both be greater than or equal to zero.

After all rows are assembled:

1. Positive and negative infinity are replaced with missing values.
2. Rows containing any missing value are dropped.
3. Exact duplicate rows are dropped.
4. The remaining row index is reset.

There is no mean/median imputation, outlier clipping, standardization,
normalization, logarithmic transform, categorical encoding, or random noise
injection.

The deterministic row derivation is:

| Source condition | Curves | Startup points per curve | Time copies | Prepared rows |
|---|---:|---:|---:|---:|
| Uniform | 341 | 5 | 5 | 8,525 |
| Partial shading | 155 | 8 | 5 | 6,200 |
| **Total** | **496** |  |  | **14,725** |

The generated files are:

- `data/processed/startup_training.csv`
- `data/processed/startup_training.metadata.json`

They are generated artifacts and excluded from Git. The metadata records the
source checksums, derivation summary, columns, source-condition counts, row
count, curve count, and augmented hours.

## 9. Model inputs and non-model columns

The five model features, in fixed order, are:

```text
sun_intensity_w_m2
panel_voltage_v
panel_current_a
ambient_temperature_c
time_of_day_hour
```

The two targets, in fixed order, are:

```text
max_power_voltage_v
max_power_current_a
```

These processed columns are not given to the estimator:

- `condition_id`: identifies all augmented rows from the same physical curve
- `source_condition`: identifies uniform or partial-shading source class
- `max_power_w`: publisher reference power for the selected global peak

In particular, the estimator does not receive the answer power, the
partial-shading fraction, the source class, or the curve identifier.

## 10. Leakage-resistant holdout

Randomly splitting individual rows would leak near-identical augmented copies
of the same curve into both training and testing. The evaluation instead uses
`GroupShuffleSplit` with:

```text
test_size = 0.20
random_state = 42
group = condition_id
```

Consequently, every operating point and every time copy belonging to a source
curve stays entirely on one side of the split.

The resulting evaluation holdout contains:

- 100 complete source curves
- 3,055 prepared startup rows

The remaining 396 curves and 11,670 rows train the evaluation model. The split
is deterministic for the committed data and scikit-learn version.

## 11. Estimator and hyperparameters

The estimator is a scikit-learn `MultiOutputRegressor`. It fits one
`HistGradientBoostingRegressor` for MPP voltage and another for MPP current.

Each underlying regressor uses:

```text
learning_rate = 0.1
max_iter = 160
max_leaf_nodes = 31
l2_regularization = 1.0
random_state = 42
```

All other parameters use the scikit-learn 1.3.0 defaults.

Histogram gradient boosting was selected because it models nonlinear
relationships and interactions among irradiance, the current operating point,
and temperature while producing a compact CPU inference artifact. The
committed compressed model is 336,807 bytes.

The model consumes the raw numeric features. No scaler or preprocessing
pipeline is needed at runtime.

## 12. Evaluation and final refit

Training is performed in two distinct passes.

### Evaluation pass

1. Fit the estimator on the grouped 80% development partition.
2. Predict MPP voltage and current for the grouped 20% holdout.
3. Calculate voltage and current MAE.
4. Calculate voltage and current R².
5. Multiply predicted voltage by predicted current.
6. Compare that product with actual target voltage multiplied by actual target
   current to calculate power MAE.

The committed holdout results are:

| Metric | Result |
|---|---:|
| MPP voltage MAE | 0.563393 V |
| MPP current MAE | 0.021849 A |
| MPP voltage R² | 0.951568 |
| MPP current R² | 0.999206 |
| Derived maximum-power MAE | 4.695585 W |

### Final artifact pass

After the holdout metrics are captured, a fresh estimator with the same
hyperparameters is fitted on all 14,725 rows. This final refit is what is
persisted for application inference. The holdout model itself is not shipped.

This is why the committed artifact benefits from all available source curves
while the report still contains an evaluation obtained from unseen complete
curves.

## 13. Saved model artifact and report

The final artifact is serialized with joblib compression level 3 to:

```text
models/smart_mppt.joblib
```

The artifact contains:

- Artifact format version
- Both fitted regressors
- Ordered feature names
- Ordered target names
- Minimum and maximum training value for each feature
- Source dataset DOI

Training first writes a temporary file and then replaces the destination, so
an interrupted write does not leave the expected model path half-written.

`models/training_report.json` records:

- Dataset and model SHA-256 checksums
- Dataset DOI
- Model type
- Random seed
- scikit-learn version
- Training and holdout sizes
- Split policy
- Feature and target order
- Training ranges
- Holdout metrics

The committed artifact SHA-256 is:

```text
34d3b6d26eba37a355fab8ceee73ee82506e56b8baf25a9e6ecc2d4ba4188926
```

## 14. Runtime inference

Normal API operation does not download or retrain anything.

For `POST /predict`:

1. Pydantic verifies the request types and broad physical bounds.
2. The supplied time is converted to decimal hour.
3. The request fields are renamed and ordered to match the five training
   features.
4. The cached joblib artifact predicts MPP voltage and MPP current.
5. Negative model outputs, if any, are clamped to zero.
6. Power is calculated from the unrounded predicted voltage and current.
7. Voltage, current, and power are rounded to three decimal places.
8. Each input is compared with the model's recorded training range.
9. The API returns the MPP plus `within_training_range` and any warnings.

The model is loaded once per application process and cached for subsequent
requests.

The training-data ranges used for warnings are:

| Feature | Minimum | Maximum |
|---|---:|---:|
| Sun intensity | 500 W/m² | 1,000 W/m² |
| Startup panel voltage | 0 V | 36.585 V |
| Startup panel current | 0 A | 9 A |
| Temperature | 20 °C | 50 °C |
| Time of day | 08:00 | 16:00 |

Requests outside these ranges are accepted when they remain within the API's
broad validation bounds, but they are explicitly marked as outside the model's
training range.

## 15. Complete reproduction

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/download_dataset.py
python scripts/prepare_dataset.py
python scripts/train_model.py
pytest
```

The download is optional when the committed source summaries are present. It
is included above to verify the original public archive and reproduce the
extraction itself.

The scripts can also be run separately:

```bash
# Verify/extract the pinned source archive
python scripts/download_dataset.py

# Rebuild processed rows and metadata
python scripts/prepare_dataset.py

# Evaluate, refit, and replace model/report
python scripts/train_model.py

# Start the prediction service
smart-mppt-api
```

To use a model artifact at another location:

```bash
export SMART_MPPT_MODEL_PATH=/absolute/path/to/smart_mppt.joblib
smart-mppt-api
```

## 16. Automated verification

The test suite checks:

- The health endpoint loads the packaged model.
- The documented sample request produces the documented sample response.
- Both supported time formats are accepted.
- Invalid physical input is rejected.
- Out-of-training-range input returns warnings.
- The packaged model returns a positive MPP.
- Dataset preparation reproduces 14,725 rows from 496 curves.
- All prepared targets are positive.
- Dataset DOI and generated metadata counts are correct.

Run:

```bash
pytest
```

## 17. Scope and current data boundaries

This implementation satisfies the requested startup request/response workflow
and produces a predicted global MPP from the specified five fields.

The following boundaries should be considered when connecting physical
hardware:

- The public dataset represents its source panel and curve-generation setup.
  A materially different panel or series/parallel array should be validated and
  preferably fine-tuned using measurements from that hardware.
- Partial-shading source curves are available at 1,000 W/m² only.
- Time of day is an interface-compatible, label-preserving augmentation rather
  than a measured source variable.
- The source `Temperature` value is mapped directly to the requested ambient
  temperature field; sensor placement and calibration should be checked on the
  actual controller.
- Predictions outside the recorded training ranges are extrapolations and are
  flagged in the response.

These boundaries do not change the API contract. They define where additional
hardware-specific data would improve production accuracy without requiring a
redesign of the service.
