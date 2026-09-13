# Data Contract and Reproducibility Boundary

## Materials included in this repository

This repository provides the MPC source code, three case profiles, locked numerical dependencies,
frozen model coefficients, model cards, summary training evidence, and the associated
preregistration documents. Historical Git commits are retained only as provenance fields; the
upstream Git history was not imported into this standalone repository.

The repository does not include the BOPTEST service image, raw weather data, the training bank,
refit or validation workspaces, formal evaluation trajectories, API keys, or private endpoints.
Installing the dependencies is sufficient to verify the supplied models and run the offline test
suite; it does not provide access to the training data.

## Episode format

Each episode collected by the training pipeline is a self-contained unit comprising
`manifest.json` and `trajectory.npz`. The training owner generates the timestamps, inputs,
outputs, and disturbances. Episodes or test instances must not be combined manually.

The principal arrays in `trajectory.npz` are:

| Array | Meaning |
|---|---|
| `times` | Simulation time in seconds, including state boundaries |
| `outputs` | Zone temperatures (°C), followed by whole-building power (W) in the final column |
| `controls` | Cooling setpoint temperatures (°C), ordered according to the zones in the case profile |
| `disturbances` | Outdoor temperature, solar irradiance, effective occupancy, and time sine/cosine features; the model layout defines the authoritative column order |

The production checks in `training.py` and `refit.py` are authoritative for array lengths, time
alignment, and terminal rows. Missing or invalid rows must not be bypassed by zero-padding or
fabrication. Four-step prediction errors are aggregated over rolling origins and must be reported
alongside persistence errors computed from the same origins.

## Input constraints for historical refitting

`refit` accepts only a preserved bank beneath the current checkout's
`outputs/baselines/mpc/training/` directory. The bank must contain a consistent
`resolved_plan.json`, the original `failure.json`, case/episode/lane evidence, and no successful
`completion.json`. The historical implementation enforces a fixed contract covering four
workers, seven-day source episodes, case order, and episode budgets.

The following are therefore not valid substitutes: arbitrary CSV files, a directory containing
only model coefficients, an arbitrary failure screenshot, a successful training directory,
episodes assembled from different test instances, or a newly created empty `failure.json`.

When an original bank is available, keep its directory read-only. If an authorized copy is moved
to another machine, transfer the complete bank and its required evidence without modifying its
contents; write newly generated refit and validation artifacts to new output directories. When no
bank is available, use the fresh data-collection route documented in the README. That route is not
expected to reproduce the same model identity.

## Results and terminal states

- `completion.json`: terminal evidence that a stage completed normally; the applicable verifier
  must still pass.
- `failure.json`: immutable failure evidence that must be retained rather than overwritten.
- `candidate_model/`: a refit candidate; its presence does not imply that fresh validation has
  completed.
- `models/mpc_releases/<freeze_identity>/`: a versioned, joint release for all three cases. Check
  the parent manifest; copying a single case does not constitute the complete release.

Release summaries support consistency checks between model files and their recorded claims. A
fully independent audit of historical measured results additionally requires the raw trajectories;
summary evidence is not a substitute for those artifacts.
