# MPC common-observation offline refit preregistration

Date: 2026-09-05

## Objective

Refit the three hierarchical MPC predictors under the common observation contract selected for
Revision 1:

- current state: `x[t]`;
- four completed 15-minute records: `x[t-4:t-1]`;
- available exogenous trajectory: `d[t:t+4]`.

The one-step multi-output ARX regressor therefore uses state terms
`y[t], ..., y[t-4]`, the candidate action `u[t]`, four completed actions
`u[t-1], ..., u[t-4]`, and the current disturbance `d[t]`. The four-stage
receding-horizon controller continues to optimize `u[t:t+3]`; offset `t+4`
remains terminal forecast information and is not a fifth action.

## Data boundary

Use only the preserved source bank
`outputs/baselines/mpc/training/20260829T155735636396Z-3d327838`.
The two Air cases use their complete seven-day development windows. `MZ_Hydro`
uses the five occupied weekdays at the start of the preserved seven-day source
episodes. Existing fit, holdout, calibration, and excluded-episode roles remain
unchanged. No BOPTEST or model API call is permitted.

## Frozen elements

The ridge grid, episode roles, persistence comparison, PMV residual calibration,
four-stage hierarchy, objective, constraints, OSQP settings, fallback, public
reward, evaluation windows, and all Agent-side methods remain unchanged. Existing
models and earlier refit workspaces remain immutable; this task creates a new
unpromoted candidate workspace.

## Acceptance

- dataset rows contain `y[t:t-4]` and `u[t:t-4]` in newest-first order;
- runtime rollouts receive exactly four completed actions and prepend, rather
  than overwrite with, each candidate action;
- no future state, completed action, or evaluation-window sample enters fitting;
- all three candidates are finite and beat persistence on their frozen holdouts;
- the refit workspace rebuilds from source and verifies without external calls;
- targeted tests, full pytest, Ruff, format, strict mypy, and the staged secret
  scan pass before the implementation commit.

Fresh closed-loop MPC evaluation and DRL retraining are outside this task.
