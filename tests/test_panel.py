import pytest

from smart_mppt.panel import PANEL


def test_panel_nameplate_values_are_internally_reasonable() -> None:
    assert PANEL.model == "AP-PM-30W"
    assert PANEL.maximum_power_w == 30
    assert PANEL.maximum_power_voltage_v == 19.3
    assert PANEL.maximum_power_current_a == 1.56
    assert PANEL.open_circuit_voltage_v == 23.16
    assert PANEL.short_circuit_current_a == 1.67
    assert PANEL.maximum_power_voltage_v * PANEL.maximum_power_current_a == pytest.approx(
        30.108
    )
    assert PANEL.maximum_power_voltage_v < PANEL.open_circuit_voltage_v
    assert PANEL.maximum_power_current_a < PANEL.short_circuit_current_a
