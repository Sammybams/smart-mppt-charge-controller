# How training and prediction work

This document explains the full process in simple steps.

## 1. The input and output

The deployed input stays exactly as requested:

| API input | Meaning |
| --- | --- |
| `light_lux` | Lux from the GY-302/BH1750 sensor |
| `temperature_c` | Ambient temperature |
| `timestamp` | Date and time of the reading |

The output is:

| API output | Meaning |
| --- | --- |
| `voltage_v` | Expected maximum-power panel voltage |
| `current_a` | Expected maximum-power panel current |
| `power_w` | Predicted voltage multiplied by predicted current |

The API does not ask for W/m2. Lux is used from beginning to end.

## 2. The real Lagos data

The original file is `data/Manual_Collection.csv`. It contains 12,477 rows
from a 30 W panel measured in Lagos on four days.

The owner confirmed this mapping:

```text
LIGHT         -> input lux
TEMPERATURE   -> input temperature
TIME          -> input timestamp
PANEL_VOLTAGE -> target MPP voltage
PANEL_CURRENT -> target MPP current
```

Battery fields and unnamed columns are not used.

The original file is kept unchanged. Its SHA-256 is:

```text
09ce21869b7133c6319af2a574daf190a3049abcf0566d383d92a755c7b9b734
```

## 3. Cleaning the real data

Run:

```bash
python scripts/prepare_manual_dataset.py
```

The script:

1. keeps only the five fields listed above;
2. removes 294 exact duplicates;
3. joins conflicting readings from the same second by taking their median;
4. sorts everything by time;
5. treats the timestamps as Lagos local time; and
6. calculates the time and sunlight features used by the model.

The result contains 11,581 rows.

No real target is silently replaced, clipped, or deleted. Output limits are
applied later when the API makes a prediction.

### Voltage calibration from the panel label

The Sunshine Solar AP-PM-30W label says Vmp is 19.3 V and Voc is 23.16 V.
However, 7,785 prepared field voltage labels are above 23.16 V. The maximum is
26.11 V, which cannot be the physical MPP voltage of this panel on the same
voltage scale.

The raw file remains unchanged. Before fitting voltage, training:

1. takes the brightest 25% of the field readings;
2. finds their median voltage, which is 24.18 V;
3. calculates `19.3 / 24.18 = 0.7981803143`;
4. multiplies every field voltage label by that factor; and
5. caps calibrated labels at the 23.16 V nameplate Voc.

This is one global sensor-scale correction. It is saved inside the model and
training report.

## 4. Features made from lux and time

The model receives the original lux and temperature. It also receives useful
values calculated from them:

- `log(1 + lux)`, which makes low-light differences easier to learn;
- time of day represented as sine and cosine;
- day of year represented as sine and cosine;
- approximate sun height above Lagos;
- a daylight factor; and
- lux compared with the light expected at that sun height.

Sine and cosine are used because time is circular. For example, 23:59 and
00:01 should be close together.

Sun height is calculated from the date, local time, and Lagos coordinates. It
does not require an internet connection.

The lux comparison is only a context feature. It is not presented as an exact
conversion from lux to W/m2.

## 5. Why generated data is needed

The real voltage has a useful relationship with the inputs. The real current
column does not. It contains values from 0.80 A to 1.49 A, but those values are
almost unrelated to lux, temperature, or time.

A model trained only on that current column learned no useful rule. The safest
old model returned the same median current for every request.

The new version adds generated examples so current can respond to light in a
reasonable way.

## 6. How generated data is made

Run:

```bash
python scripts/generate_augmented_dataset.py
```

This creates 30,000 repeatable examples. Random seed 42 is fixed, so running
the script again creates the same data.

Each example contains:

- a Lagos timestamp;
- sun height at that time;
- clear, cloudy, or strongly shaded light;
- simulated BH1750 measurement variation;
- a temperature between 24 C and 58 C;
- a possible bypass-diode voltage region; and
- expected MPP voltage, current, and power.

The generator uses lux as the saved input. It does not change the deployed API
to W/m2.

The supplied panel label gives these exact nameplate values:

| Assumption | Value |
| --- | ---: |
| Rated power | 30 W |
| Nameplate MPP voltage | 19.3 V |
| Nameplate MPP current | 1.56 A |
| Open-circuit voltage limit | 23.16 V |
| Short-circuit current limit | 1.67 A |
| Maximum allowed generated power | 33 W |

The panel is a Sunshine Solar AP-PM-30W made in the Lekki Free Zone, Nigeria.
Only the temperature coefficients remain assumptions because they are not
printed on the supplied label.

The BH1750 input is varied by up to about 20% to make the model less dependent
on one perfectly mounted sensor.

Partial shade is represented with full, two-thirds, and one-third voltage
regions. This helps the model see several possible peak areas. It still cannot
know the exact shadow shape from one lux value.

## 7. The two final models

### Voltage

Voltage uses a random forest with 40 trees. It is trained on the calibrated
Lagos voltage labels. Generated voltage does not replace the measured shape;
the nameplate calibration only corrects its physical scale.

### Current

A monotonic gradient-boosting model learns current from the generated data.
“Monotonic” means that its main lux relationship is not allowed to randomly
fall when lux increases.

At prediction time, current is blended like this:

```text
final current = 80% Lagos median anchor + 20% generated-data current
```

The Lagos anchor is 1.15 A. A low-light gate then moves the result toward zero
when lux is very small.

This is the compromise:

- current changes with lux;
- real Lagos data remains the main anchor; and
- assumptions do not control the whole result.

## 8. Testing without time leakage

The real-data test holds out one complete date at a time. The model trains on
the other dates and predicts the missing date.

This prevents neighbouring readings from the same day appearing in both the
training and test sets.

The latest results are:

| Real Lagos check | Result |
| --- | ---: |
| Calibrated voltage MAE | 1.362625 V |
| Calibrated voltage R2 | 0.776509 |
| Raw uncalibrated voltage MAE | 5.467041 V |
| Current MAE | 0.238208 A |
| Current R2 | -1.196824 |
| Derived calibrated power MAE | 4.754252 W |

The negative current R2 is important. It says the hybrid current does not
match the questionable current column well on unseen days. That is expected
because the new model follows light while the recorded current mostly does not.

Twenty percent of generated current examples are also held out:

| Generated-data check | Result |
| --- | ---: |
| Current MAE | 0.046475 A |
| Current R2 | 0.978523 |

This proves that the code learned the generated rule. It does not prove the
same accuracy on the physical panel.

All exact per-day results are in `models/training_report.json`.

## 9. Building the saved model

Run:

```bash
python scripts/train_model.py
```

Training fits the final voltage model on all 11,581 real rows and the current
physics model on all 30,000 generated rows. It saves:

```text
models/smart_mppt.joblib
models/training_report.json
```

Artifact version 4 stores both models, feature order, supported input ranges,
blend values, the complete panel nameplate, voltage calibration, safety limits,
timezone, panel rating, and dataset names.

Normal API startup loads this saved file. It does not train again.

## 10. What happens during a prediction

For one `POST /predict` request:

1. The API checks the three input values.
2. It converts the timestamp to Lagos time.
3. It calculates time, sun-height, and BH1750 context features.
4. The real-data model predicts voltage.
5. The generated-data model predicts a physics current.
6. The physics current is blended with the 1.15 A Lagos anchor.
7. The low-light gate reduces current when lux is very small.
8. Voltage is limited to the 23.16 V nameplate Voc.
9. Current is limited to the 1.67 A nameplate Isc.
10. If voltage times current exceeds 33 W, current is reduced so power is 33 W.
11. The API returns voltage, current, power, and warnings.

Lux of 62,258 or more triggers a message that the BH1750 is near its standard
range. The model supports generated values up to 100,000 lux, but the hardware
must use a suitable sensor setting to measure them correctly.

## 11. What this can and cannot do

This model gives a reasonable place for the controller to start searching. It
does not guarantee the true global maximum from only one lux reading.

One BH1750 measures light at one position. It cannot tell whether one cell,
half the panel, or the whole panel is shaded. Those cases can have different
power curves even when the sensor reports the same lux.

The controller should:

1. move near the predicted voltage;
2. check power on both sides; and
3. try another bypass-diode voltage region if measured power is unexpectedly
   low.

## 12. Best future improvement

The most useful next data collection is a fast I-V sweep. For each sweep, save:

- timestamp;
- BH1750 lux;
- ambient and panel temperature;
- every voltage and current point; and
- the voltage/current pair with the highest measured power.

Also record voltage and current with calibrated reference meters. With enough
complete days and real curve sweeps, the global voltage correction and
synthetic current blend can be reduced or removed.
