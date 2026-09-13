# Experiment Protocol and Frozen Configuration

This document is a concise, human-readable index of the committed configuration files. The JSON
files remain authoritative. Values below are derived only from
[`configs/baselines/hierarchical_mpc.json`](../configs/baselines/hierarchical_mpc.json) and the
three case profiles under [`configs/cases/`](../configs/cases/).

## Cases and calendar windows

All cases use a 900-second control interval. The shared configuration reserves the seven calendar
days immediately preceding each evaluation start for data collection. Day indices are BOPTEST
simulation-day indices, and interval endpoints are exclusive.

| Case | BOPTEST testcase | Zones | Identification window | Formal evaluation window |
|---|---|---:|---:|---:|
| SZ Air | `bestest_air` | 1 | `[196, 203)` (7 days) | `[203, 210)` (7 days) |
| MZ Hydro | `multizone_office_simple_hydronic` | 2 | `[213, 220)` (7 days) | `[220, 225)` (5 days) |
| MZ Air | `multizone_office_simple_air` | 5 | `[192, 199)` (7 days) | `[199, 206)` (7 days) |

The zone orders used by controls, outputs, and model layouts are:

- SZ Air: `zone1`.
- MZ Hydro: `NZ`, `SZ`.
- MZ Air: `cor`, `eas`, `nor`, `sou`, `wes`.

The versioned model release included in this repository was produced through the separately
preregistered historical refit route. Its evidence and any refit-specific window alignment are
recorded in the release manifests and preregistration documents; the table above reports the
committed case and acquisition configuration only.

## Identification model and controller horizon

| Setting | Committed value |
|---|---:|
| ARX lag count | 5 (the current record plus four completed records) |
| MPC prediction/control horizon | 4 steps |
| Future forecast steps declared by each case | 4 |
| Physical step | 900 s (15 min) |
| Horizon duration | 3,600 s (1 h) |
| Physical cooling-setpoint envelope | 20–30 °C |
| Occupied controller support | 23.5–26.5 °C |
| Unoccupied controller support | 20–30 °C |
| Building-coordinator period | 4 physical steps (1 h) |
| Maximum feedback iterations | 1 |
| Ridge alpha candidates | `1e-6`, `1e-4`, `1e-2`, `1`, `100` |

The output vector contains all zone temperatures followed by whole-building power. Global inputs
in every profile include dry-bulb outdoor temperature, global horizontal solar irradiance, dynamic
electricity price, and case-specific power-meter signals. Each zone declares its own occupancy
forecast and cooling-setpoint actuator.

## Objective

The objective weights are common across cases: energy `1.0`, comfort `20.0`, and setpoint
smoothness `0.1`. Case-specific scales are frozen as follows.

| Case | Energy scale | Comfort scale | Smoothness scale |
|---|---:|---:|---:|
| SZ Air | 120.351459 | 10.0 | 0.179875 |
| MZ Hydro | 3.0 | 10.0 | 0.1755995 |
| MZ Air | 18.532822 | 10.0 | 0.386932 |

## Occupancy contracts

- **SZ Air:** occupancy mode `raw_count_positive`, sourced from the BOPTEST occupancy forecast.
- **MZ Air:** occupancy mode `official_hvac_window`, using the half-open interval
  `[06:00, 19:00)`.
- **MZ Hydro:** occupancy mode `raw_count_positive`, sourced from the BOPTEST occupancy forecast.
  Its missing-value contract uses the calendar origin `2021-01-01T00:00:00+00:00`, Monday–Friday
  occupancy, and the half-open interval `[07:00, 19:00)`. Documented non-occupancy values resolve
  to zero, while documented occupancy values use the previous step. The configured holidays are
  January 1, April 17–18, May 1 and 26, June 5–6, July 21, August 15, November 1 and 11, and
  December 25.

## Thermal-comfort configuration

All cases specify a metabolic rate of 1.1 met, relative humidity of 50%, and air speed of 0.1 m/s.
The configured clothing endpoints are 1.0 clo in winter and 0.5 clo in summer, with transition
temperatures of 10 °C and 26 °C.

| Case | Dynamic clothing |
|---|---|
| SZ Air | Enabled |
| MZ Hydro | Disabled |
| MZ Air | Enabled |

## Excitation, sampling, and resource limits

| Setting | Committed value |
|---|---:|
| Excitation dwell times | 30, 60, and 120 min |
| Occupied excitation bounds | 23.5–26.5 °C |
| Unoccupied excitation bounds | 20–30 °C |
| Comfort-recovery trigger | occupied absolute PMV greater than 0.70 |
| Comfort-recovery release | absolute PMV at most 0.50 |
| Fit checkpoints | 8, 16, 32, and 64 fit episodes |
| Holdout episodes | 4 |
| Closed-loop validation episodes | 1 per checkpoint |
| Basic-RBC reference episodes | 1 |
| Maximum concurrent training lanes | 4 |
| Configured BOPTEST service workers | 6 |
| Reserved workers | 2 |
| Internal warm-up | 7 days per episode |
| Campaign wall-clock limit | 6 h |

The JSON profiles also freeze all physical point names, static actuator overrides, power-meter
lists, graph-contract paths, and the canonical cooling-program path. Those exact machine-readable
values are intentionally not duplicated here.
