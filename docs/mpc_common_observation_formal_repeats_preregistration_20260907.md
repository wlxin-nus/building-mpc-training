# Common-observation MPC three-repeat formal evaluation preregistration

Date: 2026-09-07

## Question and immutable candidate

This campaign evaluates the corrected common-observation hierarchical MPC candidate on the same
three case profiles, physical windows, occupancy rules, reward owner, and KPI owner used by the
registered H3C R01/R02/R03 results.  It is a baseline performance measurement, not another model
selection or tuning exercise.

The implementation baseline is `6618cf898c7d3e72951e4b84d601e0aaab254d21`.  The offline refit was
produced by `c882b95e58194e7a734ac878dc695c70b1d70e70` under preserved workspace
`outputs/baselines/mpc/refit/20260905T063755858960Z-c882b95e`.  The only admissible candidate model
identities are:

| Case | Candidate model identity |
|---|---|
| SZ Air | `62b361917c657bbbec4e0de4fbb4cc47d3457c95174e0706745e478bb7791fd1` |
| MZ Hydro | `29c1041c28a885310217777173ba7a728e24692181f06a292d861aa6dfa85413` |
| MZ Air | `4e38b27be0332fa0cff90252c091ddce03edc8f8a2b0a6e7f943feeeba991529` |

The preceding frozen suite from `53983f2411fe536435c708c88f3adada3080d393` and its three `n=1`
formal trajectories remain immutable historical evidence.  They are not observations in the new
three-repeat sample and must not contribute to its means or standard deviations.

## Frozen method and physical protocol

No ARX coefficient, source trajectory, feature order, lag count, four-move horizon, objective,
weight, constraint, robust margin, OSQP option, fallback controller, profile, occupancy rule,
initial state, evaluation boundary, reward, or KPI definition may change.  Mechanical support for
versioned publication, repeated-run identity, evidence verification, and reporting is allowed only
if it cannot change the controller decision for a fixed input.

All arms use one fresh BOPTEST test identity, one seven-day internal warm-up, no explicit control
prefix, a 900-second physical step, and one stop:

| Case | Evaluation start | Formal window | Occupancy owner |
|---|---:|---:|---|
| SZ Air | day 203 | 168 h | raw positive occupancy |
| MZ Hydro | day 220 | 120 h | registered raw/missing-value contract |
| MZ Air | day 199 | 168 h | official `[06:00,19:00)` HVAC window and raw positive occupancy |

The fixed canonical interpreter is
`D:\NUS\Paper\01-Heriachical Control\H3C_CAOL_Final_Worktree\.venv\Scripts\python.exe`, with
`PYTHONPATH` bound to this campaign worktree.  No new environment is created.  MPC makes zero
Baseten or other model-provider requests; Prompt tokens, Completion tokens, and LLM charges are
therefore exactly zero rather than missing.

## Validation, publication, and admission

Before formal evaluation, exactly one concurrent three-case fresh closed-loop validation suite is
run from the candidate above.  Every case must have a unique fresh test, complete all registered
steps, stop its test, and pass source, model, refit, trajectory, forecast, controller replay,
lifecycle, secret, and terminal-evidence verification.

Strict admission is `BASELINE-READY` only when every case has zero controller fallback and occupied
peak absolute PMV no greater than `0.70`.  The user prospectively authorized one transparent
`METHOD-DEGRADED` admission when, and only when, all execution-integrity checks pass and the failed
validation gates consist solely of one or both of:

- one or more recorded controller fallbacks;
- occupied peak absolute PMV greater than `0.70`.

Reward, cost, energy, zone-hours, PMV-hours, total variation, reversals, or rankings do not gate
admission and cannot trigger a rerun.  Any source/model/refit identity mismatch, missing or
non-finite trajectory, forecast/replay mismatch, secret exposure, invalid test lifecycle,
incomplete future, or corrupt terminal artifact is a hard stop with no model publication or formal
arm.

The new suite is published all-or-none under immutable
`models/mpc_releases/<freeze_identity>/`.  Existing `models/mpc/` and every prior release or run are
never removed, moved, overwritten, or used implicitly.  Formal runs record and verify the exact
release path, freeze identity, and per-case model identity.

## Formal sample and ordering

The formal sample contains exactly nine logical arms:

- replicate `R01`: SZ Air, MZ Hydro, and MZ Air concurrently;
- replicate `R02`: the same three cases concurrently, released only after all R01 arms terminate
  and pass execution-integrity verification;
- replicate `R03`: the same three cases concurrently, released only after all R02 arms terminate
  and pass execution-integrity verification.

There are never more than three active arms.  Every logical arm starts as attempt `A01` and binds
replicate and attempt to the suite and run identities.  Reusing a replicate/attempt identity,
release identity, run directory, or BOPTEST test identity fails closed.

A failed arm is replaceable only for an evidenced infrastructure interruption that is independent
of controller performance and leaves the original run immutable.  The replacement uses a fresh
test and directory, the next attempt label, and an explicit `replacement_for` identity; only the
failed logical arm is rerun.  Controller fallback, comfort degradation, poor KPI, solver behavior,
or unfavorable comparison is not infrastructure failure and never authorizes a replacement.

## Monitoring and batch boundaries

Before validation and each formal replicate, the launcher verifies the committed source, candidate
or release identities, import owner, dependency compatibility, empty relevant execution locks,
host-context BOPTEST TCP reachability, and at least 1 GiB free on drive D.  It must not select a
test or create active-run evidence before those checks pass.

Each active arm receives an independent ten-minute native heartbeat.  A heartbeat reads only the
registered PID/lock, immutable failure marker, and unique completion marker; it does not inspect
partial trajectories.  It is removed at arm terminal state.  A systemic execution-integrity
failure stops release of the next replicate.

## Metrics, statistics, and report

Only runs with complete production verification and an independent KPI recomputation enter the
formal sample.  For each case, the three admitted logical arms are summarized as mean plus or minus
sample standard deviation, using the `n-1` denominator, for:

- reward, cost, energy, discomfort zone-hours, PMV-hours, occupied peak absolute PMV;
- setpoint total variation, reversals, comfort-band crossings, and controller fallbacks;
- MPC solve seconds per physical step and end-to-end arm wall time;
- model calls, Prompt tokens, Completion tokens, and LLM charge, all fixed at zero.

The new MPC distribution is compared within case against the H3C Standard R01/R02/R03 distribution
and the frozen same-window eRBC point estimate.  A comparator with only one trajectory is labelled
`n=1; SD not estimable`; it is never displayed as zero variance.  The historical `53983f2` MPC
result is shown only in a clearly separated appendix.

The durable outputs are a dedicated Markdown report plus machine-readable CSV and JSON under
`Revision1/000Revise/Analysis/`, and a narrow MPC addition to the existing H3C R01/R02/R03 summary.
Raw physical evidence remains ignored and local.  No paper, LaTeX, figure, Prompt, Agent method, or
unrelated dirty file is modified, and nothing is pushed.
