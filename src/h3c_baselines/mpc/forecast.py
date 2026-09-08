"""Single owner for the hierarchical MPC current-plus-four forecast window."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from h3c.runtime.occupancy import effective_count


@dataclass(frozen=True)
class MpcForecastWindow:
    """Five-point available forecast and its four-stage optimizer projection."""

    step: int
    zones: tuple[str, ...]
    horizon_steps: int
    disturbances: NDArray[np.float64]
    prices: NDArray[np.float64]
    occupancy: NDArray[np.float64]
    action_times_seconds: tuple[int, ...]
    daily_outdoor_means_c: tuple[float, ...]

    @property
    def optimizer_disturbances(self) -> NDArray[np.float64]:
        return self.disturbances[: self.horizon_steps]

    @property
    def optimizer_prices(self) -> NDArray[np.float64]:
        return self.prices[: self.horizon_steps]

    @property
    def optimizer_occupancy(self) -> NDArray[np.float64]:
        return self.occupancy[: self.horizon_steps]

    @property
    def optimizer_action_times(self) -> tuple[int, ...]:
        return self.action_times_seconds[: self.horizon_steps]

    @property
    def terminal_occupancy(self) -> NDArray[np.float64]:
        return np.asarray(self.occupancy[self.horizon_steps], dtype=np.float64)

    def evidence(self) -> dict[str, Any]:
        return {
            "schema": "h3c_mpc_forecast_window",
            "schema_version": 1,
            "step": self.step,
            "zones": list(self.zones),
            "available_offsets_steps": list(range(self.horizon_steps + 1)),
            "optimizer_offsets_steps": list(range(self.horizon_steps)),
            "terminal_offset_steps": self.horizon_steps,
            "action_times_seconds": list(self.action_times_seconds),
            "outdoor_temperature_c": self.disturbances[:, 0].tolist(),
            "solar_irradiance_w_m2": self.disturbances[:, 1].tolist(),
            "effective_occupancy": {
                zone: self.occupancy[:, index].tolist() for index, zone in enumerate(self.zones)
            },
            "electricity_price": self.prices.tolist(),
        }


def build_mpc_forecast_window(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    zones: tuple[str, ...],
    *,
    step: int,
    action_time: int,
    horizon_steps: int,
    step_seconds: int = 900,
) -> MpcForecastWindow:
    """Build offsets 0..N once; the N-step QP consumes only offsets 0..N-1."""

    disturbances: list[list[float]] = []
    occupancy_rows: list[list[float]] = []
    prices: list[float] = []
    action_times: list[int] = []
    daily_means: list[float] = []
    outdoor_point = str(profile["global_inputs"]["outdoor_temperature"])
    solar_point = str(profile["global_inputs"]["solar_irradiance"])
    price_point = str(profile["global_inputs"]["electricity_price"])

    for offset in range(horizon_steps + 1):
        time_seconds = action_time + offset * step_seconds
        fraction = (time_seconds % 86400) / 86400.0
        occupancy = [
            effective_count(
                profile["occupancy"],
                time_seconds,
                float(forecast[profile["zones"][zone]["occupancy_forecast"]][step + offset]),
            )
            for zone in zones
        ]
        disturbances.append(
            [
                float(forecast[outdoor_point][step + offset]) - 273.15,
                float(forecast[solar_point][step + offset]),
                *occupancy,
                math.sin(2.0 * math.pi * fraction),
                math.cos(2.0 * math.pi * fraction),
            ]
        )
        occupancy_rows.append(occupancy)
        prices.append(float(forecast[price_point][step + offset]))
        action_times.append(time_seconds)
        if offset < horizon_steps:
            daily = forecast[outdoor_point][step + offset : step + offset + 97 : 4]
            daily_means.append(sum(float(value) - 273.15 for value in daily) / len(daily))

    window = MpcForecastWindow(
        step=step,
        zones=zones,
        horizon_steps=horizon_steps,
        disturbances=np.asarray(disturbances, dtype=np.float64),
        prices=np.asarray(prices, dtype=np.float64),
        occupancy=np.asarray(occupancy_rows, dtype=np.float64),
        action_times_seconds=tuple(action_times),
        daily_outdoor_means_c=tuple(daily_means),
    )
    expected_rows = horizon_steps + 1
    if (
        horizon_steps <= 0
        or window.disturbances.shape != (expected_rows, len(zones) + 4)
        or window.prices.shape != (expected_rows,)
        or window.occupancy.shape != (expected_rows, len(zones))
        or len(window.action_times_seconds) != expected_rows
        or len(window.daily_outdoor_means_c) != horizon_steps
        or not np.all(np.isfinite(window.disturbances))
        or not np.all(np.isfinite(window.prices))
        or not np.all(np.isfinite(window.occupancy))
        or np.any(window.occupancy < 0.0)
    ):
        raise ValueError("MPC current-plus-four forecast window is invalid")
    return window
