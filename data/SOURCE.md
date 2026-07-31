# Legacy UCP photovoltaic dataset

This source is retained as a reproducible partial-shading benchmark. It is not
used by the packaged Lagos 30 W production model. The production source and
contract are documented in `data/MANUAL_COLLECTION.md`.

This project uses version 1 of:

> kalaimohan T S (2021), “PV Panel: Irradiance, Temperature, Partial
> Shading - IV Curves”, Mendeley Data, V1,
> <https://doi.org/10.17632/z93gzbptf7.1>

The source dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The original archive is downloaded from Mendeley Data by
`scripts/download_dataset.py`. Its expected SHA-256 checksum is:

```text
22e39cc0b074d9ffd09459851c34898a54652ae9113661a118e2cc6270a08ae8
```

Only the two small inferred/summary CSV files are extracted. They contain the
operating conditions, I-V sample points, and identified local/maximum power
points needed by this project. The full raw archive and generated datasets are
intentionally excluded from Git.

No source measurements are modified by the downloader. The preparation stage
derives model-training rows from these source files and records that derivation
in its generated metadata.

UCP's light-related variable is solar irradiance in W/m2, not lux. The modeled
panel is approximately 250 W at its published maximum-power point. Those facts
prevent direct concatenation with the local 30 W lux collection without a
sensor calibration and a defensible cross-panel domain adaptation method.
