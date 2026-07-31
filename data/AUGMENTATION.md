# Generated 30 W lux data

The real Lagos file is small in one important way: it covers only four days,
and its current column has almost no clear relationship with lux.

This project therefore creates extra examples for the current model.

## What stays the same

The model input is still:

```text
lux + temperature + timestamp
```

The generated CSV stores lux. It does not require the device to measure W/m2.

## How to create it

```bash
python scripts/generate_augmented_dataset.py
```

This writes:

```text
data/processed/lagos_30w_augmented.csv
data/processed/lagos_30w_augmented.metadata.json
```

These files are generated and ignored by Git. Seed 42 makes the same 30,000
rows every time.

## What is simulated

The generator varies:

- Lagos date and daylight time;
- clear sky, clouds, and strong shade;
- BH1750 reading error and mounting differences;
- temperature;
- low and high light; and
- possible bypass-diode voltage regions.

It calculates expected MPP voltage and current for each case. Maximum generated
power is 33 W.

## Panel assumptions

The exact panel datasheet has not been supplied. The generator currently uses:

| Value | Assumption |
| --- | ---: |
| Rated power | 30 W |
| MPP voltage | 24.0 V |
| MPP current | 1.25 A |
| Open-circuit voltage | 26.5 V |
| Short-circuit current | 1.35 A |

These are surrogate values based on the 30 W rating and the Lagos voltage
scale. They are not claimed to be the manufacturer's exact specifications.

Replace them in `src/smart_mppt/augmentation.py` when the real panel datasheet
is available, regenerate the CSV, and retrain.

## How much generated data is trusted

Generated data teaches the shape of the current response, but it does not fully
replace the real data. The returned current uses:

```text
80% Lagos current anchor + 20% generated-data current
```

This small blend is intentional. It makes current respond to lux without
pretending that the assumed panel is exactly the same as the physical panel.

## Main limitation

One lux sensor cannot describe the shape of a shadow across the whole panel.
The generated shade cases therefore teach an expected starting point, not a
guaranteed global maximum. The controller must still verify nearby power.
