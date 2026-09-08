# Hierarchical MPC baseline preregistration

Date: 2026-08-29

## Research question and fixed scope

This experiment adds one representative hierarchical model-predictive-control baseline per H3C
case.  It asks whether a two-layer, data-driven MPC using the same observable variables, physical
windows, public reward owner, and actuator limits as the existing comparison can outperform Basic
RBC and approach the frozen DRL policies.

Only the independent baseline package changes.  The online H3C agents, prompts, causal admission,
program interpreter, action assurance, existing baseline trajectories, paper, and historical
evidence remain unchanged.

## Controller

All cases use the same four-lag, four-step vector ARX structure.  The output vector contains all
zone temperatures and site power; controls are zone cooling setpoints; disturbances are outdoor
temperature, solar irradiance, public occupancy, and time sine/cosine.  Electricity price enters
only the objective.  Predicted PMV is a deterministic affine approximation produced through the
shared comfort model.

The upper building coordinator updates hourly and chooses a four-step site trajectory and zone
references.  Zone MPCs update every 15 minutes and choose setpoints in `[20, 30]` degrees Celsius.
They return predicted comfort loss or infeasibility to the coordinator, which may perform exactly
one revision.  The objective uses the existing cost, occupied-PMV-exceedance, and setpoint-
smoothness weights without case-specific tuning.  OSQP solves the resulting convex QPs.  Only the
first setpoint is applied.

## Identification data and lifecycle

Identification uses only the seven calendar days immediately before formal evaluation:

| Case | Identification week | Formal evaluation |
|---|---|---|
| SZ_Air | days 196--202 | seven days from day 203 |
| MZ_Hydro | days 213--219 | five days from day 220 |
| MZ_Air | days 192--198 | seven days from day 199 |

Each physical lane selects one test case once and keeps that test ID.  Every episode calls
`initialize` again on the same test ID with the same training-week start and a full seven-day
internal warm-up.  The FMU is therefore reset and physically conditioned for every episode; a
zero-warm-up reset is forbidden.  A lane stops its test ID once after its final episode.

At most four MPC lanes run concurrently.  One BOPTEST worker remains available to the current H3C
main experiment and one remains spare.  The scheduler uses
`min(4, healthy_workers - active_h3c_tests - 1)` and never exceeds five active test IDs.

Per case the maximum budget is 64 fit episodes, four fixed-seed whole-episode holdouts, closed-loop
validation after 8, 16, 32, and 64 fit episodes, and one training-week Basic-RBC reference.  Fit
and holdout histories never cross an episode boundary.  No query, target, or optimization horizon
may cross the formal evaluation boundary.

Fit trajectories use deterministic, zone-phased PRBS/GBN setpoints with 30-, 60-, and 120-minute
dwell times.  Occupied excitation is bounded to 23.5--26.5 degrees Celsius and unoccupied
excitation to 20--30 degrees Celsius.  When occupied absolute PMV exceeds 0.70 the canonical P0
comfort recovery acts until absolute PMV is at most 0.50, after which excitation resumes.  The
event is logged and the resulting physical data remains part of the episode.

## Model selection and formal evaluation

Models are fitted after 8, 16, 32, and 64 fit episodes using the existing fixed ridge grid.  A
candidate must be finite, beat the registered persistence predictor on whole-episode holdout data,
complete closed-loop validation without solver fallback, and keep occupied peak absolute PMV at or
below 0.70.  Among eligible candidates, training-week public reward selects the model.  Collection
stops early after two consecutive eligible checkpoints improve reward by less than one percent.
If no candidate beats Basic RBC, the best otherwise eligible candidate is still frozen and its
formal result is reported as adverse evidence; formal results never select or tune a model.

After freezing all models, exactly three fresh formal arms run once and serially within the MPC
suite: SZ_Air seven days, MZ_Hydro five days, and MZ_Air seven days.  Existing RBC and DRL arms are
not rerun.  An auditable controller fallback yields `METHOD-DEGRADED`; infrastructure, identity,
model, secret, or evidence failure yields `RUN-INVALID`; poor performance alone remains a valid
result.

## Stop rules and deliverables

The complete training and formal-evaluation wall-clock budget is six hours.  At the limit, the
last eligible checkpoint is frozen and no new episode starts.  BOPTEST initialization or advance
failure stops all MPC lanes; a fit failure rejects only that checkpoint.  No episode is resumed or
rerun for a favorable result.

Deliverables are the three frozen model cards and coefficient files, prediction and validation
reports, three formal trajectories, the common metric table against existing Basic RBC, P0/eRBC,
C-DRL, and H-DRL results, and cost/power/temperature/PMV/setpoint figures.  This task does not push
or modify the paper.
