from __future__ import annotations

import pytest

from h3c.experiments.profiles import load_profile
from h3c.runtime.occupancy import effective_count
from h3c.runtime.protocol import forecast_points
from h3c_baselines.mpc.forecast import build_mpc_forecast_window


@pytest.mark.parametrize("case", ["SZ_Air", "MZ_Hydro", "MZ_Air"])
def test_current_plus_four_projection_is_exact_for_each_case(case: str) -> None:
    profile = load_profile(case)
    length = 120
    forecast: dict[str, list[float]] = {}
    for point in forecast_points(profile):
        if point == profile["global_inputs"]["outdoor_temperature"]:
            forecast[point] = [273.15 + index for index in range(length)]
        elif point == profile["global_inputs"]["solar_irradiance"]:
            forecast[point] = [100.0 + index for index in range(length)]
        elif point == profile["global_inputs"]["electricity_price"]:
            forecast[point] = [0.01 + index for index in range(length)]
        else:
            forecast[point] = [float(index + 1) for index in range(length)]

    zones = tuple(profile["zones"])
    start = int(profile["evaluation_start_day"]) * 86400
    step = 2
    action_time = start + step * 900
    window = build_mpc_forecast_window(
        profile,
        forecast,
        zones,
        step=step,
        action_time=action_time,
        horizon_steps=4,
    )
    evidence = window.evidence()

    assert evidence["available_offsets_steps"] == [0, 1, 2, 3, 4]
    assert evidence["optimizer_offsets_steps"] == [0, 1, 2, 3]
    assert evidence["terminal_offset_steps"] == 4
    assert evidence["outdoor_temperature_c"] == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0])
    assert evidence["solar_irradiance_w_m2"] == pytest.approx([102.0, 103.0, 104.0, 105.0, 106.0])
    assert evidence["electricity_price"] == pytest.approx([2.01, 3.01, 4.01, 5.01, 6.01])
    for zone in zones:
        point = profile["zones"][zone]["occupancy_forecast"]
        expected = [
            effective_count(
                profile["occupancy"],
                action_time + offset * 900,
                forecast[point][step + offset],
            )
            for offset in range(5)
        ]
        assert evidence["effective_occupancy"][zone] == expected
    assert window.optimizer_disturbances.shape == (4, len(zones) + 4)
    assert window.optimizer_prices.shape == (4,)
    assert window.optimizer_occupancy.shape == (4, len(zones))
    assert window.terminal_occupancy.tolist() == pytest.approx(
        [evidence["effective_occupancy"][zone][4] for zone in zones]
    )
