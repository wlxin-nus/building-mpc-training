# MPC current-plus-four forecast availability preregistration

Date: 2026-09-05

## Question

Can the hierarchical MPC baseline expose the same current-plus-four forecast
information boundary as PPO/MAPPO and the proposed framework without changing its
four-control-move receding-horizon problem?

## Registered change

The runtime will construct one deterministic five-point MPC forecast window at every
decision time.  It contains outdoor temperature, solar irradiance, effective zone
occupancy, and electricity price for offsets `0, 1, 2, 3, 4`.  The complete window
will be passed to the MPC controller and recorded in a dedicated evidence stream.

The controller will continue to optimize exactly four moves, `u[t:t+3]`, and predict
the four resulting states, `y[t+1:t+4]`.  It will derive the existing four-stage QP
arrays from offsets `0:3` and the existing terminal occupancy boundary from offset
`4`.  Offset-4 outdoor temperature, solar irradiance, and price are available and
auditable but do not create a fifth stage or enter the frozen objective.

## Frozen elements

- four-step prediction and control horizon;
- ARX coefficients, scaling, lag count, and feature order;
- coordinator, zone QP, reconciliation, objective, bounds, and fallback;
- BOPTEST profiles, evaluation windows, KPI/reward owners, and baseline acceptance;
- all H3C prompts, Agents, causal, Budget, and Safety behavior.

This change is input/evidence wiring only.  It must leave the control action unchanged
when offsets `0:3` and terminal occupancy at offset `4` are fixed.

## Evidence and gates

- one shared forecast-window builder owns both the runtime projection and verifier
  reconstruction;
- every MPC physical step records exactly one five-point projection with explicit
  available, optimizer, and terminal offsets;
- the verifier rejects a missing, duplicated, reordered, or modified projection;
- tests prove the QP still receives four stages and that changing only offset-4
  weather, solar, or price does not change the control decision;
- tests prove offset-4 occupancy retains its existing terminal-boundary effect;
- targeted tests, full pytest, Ruff, format check, and strict mypy must pass;
- no API call, BOPTEST run, model refit, paper edit, push, or result replacement is
  part of this task.

## Explicit exclusion

The independently identified ARX control-history alignment defect is not repaired in
this change.  It remains a separate method-correctness item requiring its own scoped
repair and fresh closed-loop validation.
