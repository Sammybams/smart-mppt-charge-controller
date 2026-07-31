"""Nameplate values for the project's physical solar panel."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PanelSpecifications:
    manufacturer: str = "Sunshine Solar"
    model: str = "AP-PM-30W"
    maximum_power_w: float = 30.0
    maximum_power_voltage_v: float = 19.3
    maximum_power_current_a: float = 1.56
    open_circuit_voltage_v: float = 23.16
    short_circuit_current_a: float = 1.67
    maximum_system_voltage_v: float = 1500.0
    maximum_series_fuse_a: float = 20.0
    minimum_operating_temperature_c: float = -40.0
    maximum_operating_temperature_c: float = 85.0
    application_class: str = "Class A"
    stc_air_mass: float = 1.5
    stc_irradiance_w_m2: float = 1000.0
    stc_cell_temperature_c: float = 25.0
    origin: str = "Lekki Free Zone, Nigeria"

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


PANEL = PanelSpecifications()
