# Physical panel nameplate

The project owner supplied a photograph of the label on the physical panel.
These values were read directly from that photograph.

## Identity

| Item | Value |
| --- | --- |
| Brand | Sunshine Solar |
| Model | AP-PM-30W |
| Origin printed on label | Lekki Free Zone, Nigeria |

## Electrical values at standard test conditions

The label defines standard test conditions as air mass 1.5, irradiance
1,000 W/m2, and cell temperature 25 C.

| Label item | Symbol | Value |
| --- | --- | ---: |
| Maximum power | Pmax | 30 W |
| Maximum-power voltage | Vmp | 19.3 V |
| Maximum-power current | Imp | 1.56 A |
| Open-circuit voltage | Voc | 23.16 V |
| Short-circuit current | Isc | 1.67 A |

The rounded Vmp and Imp values multiply to 30.108 W. The nameplate Pmax of
30 W remains the official rating.

## Other label values

| Item | Value |
| --- | ---: |
| Maximum system voltage | 1,500 V |
| Maximum series fuse | 20 A |
| Operating temperature | -40 C to 85 C |
| Application class | Class A |
| Service statement | More than 90% after 10 years; more than 80% after 20 years |

## Important data check

The supplied Lagos CSV contains 7,785 prepared voltage labels above the
nameplate Voc of 23.16 V. That is 67.22% of the prepared rows. The largest
recorded target is 26.11 V.

For the stated panel, an MPP voltage cannot be higher than its open-circuit
voltage under the same condition. This points to voltage-sensor gain/offset,
column meaning, or logging calibration differences.

The training pipeline therefore keeps the raw CSV unchanged but calibrates its
voltage labels to the panel nameplate before fitting the production voltage
model. Runtime voltage is always limited to the nameplate Voc.
