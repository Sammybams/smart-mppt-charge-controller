# Lagos 30 W model: training and runtime pipeline

This document records every material transformation from the supplied manual
CSV to the prediction returned by the API.

## 1. Objective and scope

The current owner-confirmed contract is:

| Meaning | Source/API field | Unit | Role |
| --- | --- | --- | --- |
| Light sensor value | `LIGHT` / `light_lux` | lux | Input |
| Ambient temperature | `TEMPERATURE` / `temperature_c` | degrees Celsius | Input |
| Lagos measurement time | `TIME` / `timestamp` | datetime | Input |
| Expected MPP panel voltage | `PANEL_VOLTAGE` / `voltage_v` | volts | Target/output |
| Expected MPP panel current | `PANEL_CURRENT` / `current_a` | amperes | Target/output |

Battery columns, unnamed columns, and free-text explanation are excluded. The
model is specific to the measured 30 W panel and local sensor arrangement.

## 2. Raw data audit

The immutable source is `data/Manual_Collection.csv` with SHA-256:

```text
09ce21869b7133c6319af2a574daf190a3049abcf0566d383d92a755c7b9b734
```

It contains 12,477 received rows on four dates:

| Date | Raw rows | Approximate period represented |
| --- | ---: | --- |
| 2026-06-22 | 15 | 19:40-19:43 |
| 2026-06-23 | 1,623 | 12:53-19:13 |
| 2026-07-21 | 4,434 | 13:10-19:02 |
| 2026-07-22 | 6,405 | 08:08-15:44 |

Observed selected-column ranges before modeling are approximately:

| Field | Minimum | Maximum |
| --- | ---: | ---: |
| Light | 10.83 lux | 54,612.50 lux |
| Temperature | 28.5 C | 53.8 C |
| Target voltage | 2.71 V | 26.11 V |
| Target current | 0.80 A | 1.49 A |

The collection has duplicate rows, repeated timestamps, and occasional input
order reversals of one or two seconds. It has no missing numeric values in the
selected columns.

The target current has exactly 70 unique values at 0.01 A increments. Its
distribution and weak feature correlations are why current performance is
reported separately from voltage rather than hidden inside one average score.

## 3. Deterministic preparation

`scripts/prepare_manual_dataset.py` calls
`smart_mppt.manual_dataset.prepare_manual_dataset`.

The steps are:

1. Read the original CSV without editing it.
2. Select only `TIME`, `LIGHT`, `TEMPERATURE`, `PANEL_VOLTAGE`, and
   `PANEL_CURRENT`.
3. Rename them to explicit unit-bearing training names.
4. Remove exact duplicates across those five selected fields.
5. Parse `TIME` with the exact `%m/%d/%Y %H:%M:%S` format.
6. Parse all four physical fields as numeric and reject non-finite or negative
   values.
7. Group records sharing the same second and take the median of each numeric
   field. Median aggregation is robust to conflicting duplicate readings and
   creates one unambiguous label per timestamp.
8. Sort ascending by timestamp; this also corrects minor logger-order
   reversals.
9. Start a new session after any gap longer than five minutes.
10. Generate time and light features described below.
11. Calculate reference target power as target voltage times target current.

Exact row accounting:

| Stage | Rows |
| --- | ---: |
| Received | 12,477 |
| Exact selected-field duplicates removed | 294 |
| Rows after exact deduplication | 12,183 |
| Surplus same-second rows median-aggregated | 602 |
| Final prepared rows | 11,581 |

There are 601 timestamp groups containing multiple non-identical selected
records. No target clipping, 30 W clipping, mean imputation, random noise, or
synthetic oversampling is performed.

## 4. Time and input feature engineering

The final feature order is:

```text
light_lux
log_light_lux
temperature_c
hour_sin
hour_cos
day_of_year_sin
day_of_year_cos
```

`log_light_lux = log(1 + light_lux)` gives the tree model both the original
sensor scale and a compressed representation of changes at lower light.

Clock time is cyclic. Seconds since midnight are converted to an angle and
then represented by sine and cosine:

```text
daily_angle = 2 * pi * seconds_since_midnight / 86400
hour_sin = sin(daily_angle)
hour_cos = cos(daily_angle)
```

This makes 23:59 close to 00:01 instead of opposite ends of a linear scale.
Day of year is encoded the same way using 365.2425 days and includes the
fractional day. This allows a June morning and a December morning to carry
different seasonal context without treating December 31 and January 1 as far
apart.

Training timestamps are naive values explicitly interpreted as Lagos local
time. Runtime timestamps with an offset are converted to `Africa/Lagos`; naive
runtime values are assigned that timezone.

Timestamp is contextual, not a substitute for light. Measurements at similar
times on different days show very different light patterns, as expected from
clouds and shade. Lux supplies current conditions; clock and season supply
broad solar-cycle context.

## 5. Why no rolling history or LSTM is used

The raw sampling intervals are irregular and differ substantially by day. More
importantly, 11,581 rows come from only four dates. A sequence model trained on
overlapping windows would see many nearly identical samples from the same few
trajectories. A random row/window split would then leak the same day into both
sides and produce an optimistic score.

The device requirement is also a one-time startup call. Requiring a historical
window would change that contract and complicate cold-start operation.

For those reasons, this version uses time-aware tabular features and validates
on whole unseen days. An LSTM becomes reasonable only after collecting many
independent days across seasons and operating conditions, with a defined
history buffer available on the controller.

## 6. Treatment of public data

The legacy UCP source and reproducible downloader are retained. UCP uses solar
irradiance in W/m2 and represents a simulated 250 W, 60-cell panel. The Lagos
sensor reports illuminance in lux for a physical 30 W panel.

Lux describes human-visible illuminance and W/m2 describes incident radiant
power. Their ratio changes with spectrum, clouds, sun angle, sensor response,
and calibration. Nameplate normalization can make voltage/current/power
dimensionless, but it cannot create the missing lux-to-irradiance calibration
or make the panel electrical characteristics identical. Therefore UCP is not
concatenated with the production training table.

A 2026 Scientific Reports study is closer in rating: it used an ET-M53630WW
30 W panel and recorded 55 I-V curves at five-minute intervals. However, it was
a different, aged panel in Oujda, Morocco, used W/m2 irradiance, and studied
uniform conditions. It is a good pattern for future data collection, not a
drop-in source of labels for this exact sensor and panel.

## 7. Leakage-resistant evaluation

`LeaveOneGroupOut` uses `collection_date` as its group. Four evaluation fits
are made. Each holds out one complete date and trains on the other three. No
record from the held-out date, including immediately adjacent sensor readings,
can enter that fold's training data.

The aggregate metric concatenates every fold's out-of-fold prediction before
calculation, so larger days contribute in proportion to their row count. The
per-day metrics remain in `models/training_report.json`. The 2026-06-22 fold
contains only 15 late-evening readings; its R2 is mathematically valid but not
a stable standalone estimate.

## 8. Model comparison and selection

Candidate voltage estimators included random forest, Extra Trees, histogram
gradient boosting, and a constant baseline. Random forest had the best overall
day-isolated voltage MAE in the comparison.

The voltage estimator is `RandomForestRegressor` with:

```text
n_estimators = 80
min_samples_leaf = 8
max_features = 0.9
n_jobs = 1
random_state = 42
```

The current candidates did not beat a median baseline under leave-one-day-out
validation. The final current estimator is consequently a
`DummyRegressor(strategy="median")`. It still produces the requested current,
but accurately represents the evidence: the current sensor target currently
has no demonstrated generalizable mapping from lux, temperature, and time.

This per-target choice is safer than a multi-output neural model whose good
voltage behavior could obscure uninformative current predictions.

## 9. Evaluation results

Out-of-fold metrics across all complete-day holdouts are:

| Metric | Result |
| --- | ---: |
| Voltage MAE | 1.882936 V |
| Voltage R2 | 0.649073 |
| Current MAE | 0.173991 A |
| Current R2 | -0.000220 |
| Derived power MAE | 4.684770 W |

Power for evaluation is calculated consistently as:

```text
actual_power = target_voltage * target_current
predicted_power = predicted_voltage * predicted_current
```

The negative-near-zero current R2 means the median does not explain current
variation. It is not evidence that current prediction is solved. The voltage
result is useful but also varies by day; the latest-day holdout has 2.132577 V
MAE and negative R2 because its voltage variance is narrow relative to its
cross-day shift.

## 10. Final refit and artifact

After validation metrics are produced, fresh voltage and current estimators are
fitted on all 11,581 prepared rows. The compressed artifact is written to a
temporary file and atomically moved to `models/smart_mppt.joblib`.

Artifact version 2 contains:

- fitted voltage and current estimators;
- ordered feature and target names;
- observed lux, temperature, local-hour, and day-of-year ranges;
- source dataset name, panel rating, light unit, and timezone; and
- artifact format version.

`models/training_report.json` records dataset/model checksums, software
version, model selection rationale, dates, features, targets, validation
policy, aggregate and per-day metrics, and known limitations.

The model is fitted directly to numeric features. Tree models do not require a
standard scaler, so no z-score normalization is used. `log1p(lux)` is the only
scale transform.

## 11. Runtime transformation

For `POST /predict`:

1. Pydantic checks types and broad physical limits.
2. The timestamp is interpreted or converted to Lagos local time.
3. Lux is transformed with the same `log1p` formula used in training.
4. The daily and annual cyclic features are generated by the shared
   `time_features.py` implementation.
5. Features are ordered exactly as stored in the artifact.
6. The voltage forest and current median regressor predict independently.
7. Negative predictions are clamped to zero.
8. Power is computed from the unrounded predictions.
9. Outputs are rounded to three decimals.
10. Lux, temperature, local time, and day of year are checked against observed
    collection ranges and warnings are returned for extrapolation.

The artifact loads once per API process and is cached. Normal requests neither
download data nor retrain the model, and no LLM or external API is involved.

## 12. What the target does and does not establish

The owner confirmed `PANEL_VOLTAGE` and `PANEL_CURRENT` as the expected maximum
voltage/current outputs, so they are used as supervised labels.

Nevertheless, one voltage/current pair at a timestamp cannot mathematically
demonstrate a global maximum on a multi-peak P-V curve. A global label requires
a contemporaneous curve sweep or another trusted controller/reference that
searched all relevant peaks. The model therefore estimates the supplied target
definition and is appropriate for choosing an initial search neighborhood. It
should not replace the firmware's local verification/search.

## 13. Recommended next collection

For a rigorous partial-shading global-MPP model, collect many independent days
and perform fast sweeps that minimize environmental change during each curve.
For every sweep store:

- unique sweep ID and precise Lagos timestamp;
- raw lux and, ideally, a calibrated pyranometer irradiance in W/m2;
- ambient and rear-panel/cell temperature separately;
- shade configuration or image-derived shade descriptor when available;
- every swept panel voltage and current pair;
- computed power for every point; and
- the global peak voltage, current, and power selected from the complete sweep.

Also record panel model/specification, sensor model and calibration, converter
duty cycle, battery/load state, and firmware version. Split future evaluation
by complete day and, where possible, by shade experiment. Once there are many
independent days, compare lag/rolling features and sequence models against this
tabular baseline without changing the test days.
