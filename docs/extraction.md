# Standalone Repository Provenance and Verification

## Source

- Extraction date: 2026-09-08.
- Upstream source commit: `32524f3b3fd3861a9323226612c776b27f4b8a85`.
- Included model release: `b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c`.
- This standalone repository starts from an independent initial commit. It does not import the
  upstream paper, Agent, experiment, or Git history.

## Preserved and changed components

The seven MPC algorithm, training, and publication modules; the case and MPC configurations; and
the PMV and occupancy calculations retain their upstream content. `provenance.json` records the
source of every file. Text files permit only the ordinary line-ending normalization performed by
Git. Frozen release files are not line-ending-normalized, preserving their internal evidence
checksums.

Only the software boundary changed during extraction:

1. The CLI retains only MPC commands. DRL, formal-suite, whole-project reporting, and LLM commands
   were removed.
2. The HTTP module retains the upstream BOPTEST class and its required helpers; the LLM client was
   removed. Historical names used by JSON serialization helpers do not indicate a model-provider
   request path.
3. The reference RBC interpreter retains only loading, validation, execution, and the necessary
   definitions. Agent patch application and candidate derivation are not included.
4. Secret scanning no longer depends on LLM runtime configuration. It still checks any configured
   values of known secret variables and never prints those values.
5. Launch preflight no longer uses the paper workstation's hard-coded interpreter path. It checks
   the current checkout, clean Git state, locks, disk space, numerical dependencies, and same-process
   TCP connectivity. The original workstation continued to use its canonical environment.
6. Torch, DRL, Agent Framework, and other non-MPC dependencies were removed. Numerical library
   versions are pinned to those used in the existing canonical environment.

The extraction did not change ARX coefficients, training data, ridge selection, scaling, objective
weights, constraints, horizon, OSQP settings, fallback behavior, case profiles, PMV, or reward.
It did not perform physical retraining, additional diagnostics, or BOPTEST calls.

`configs/graphs/` contains compatibility files required by the case-profile contract. The MPC
runtime does not consume causal graphs. Retaining these files avoids rewriting profile validation
solely for the extraction.

## Verification scope

Release checks covered the existing offline MPC tests, extraction regressions, Ruff, formatting,
strict mypy, the dependency lock, a side-effect-free dry plan, consistency of the three-case frozen
model suite, source consistency, and pre-publication secret and file-inventory checks. No new
physical trajectories were collected from this repository; these offline checks must not be
described as a recertification of the nine formal results.

Local verification on 2026-09-08 recorded:

- Python 3.12.2 in the canonical interpreter, with NumPy 2.2.6, SciPy 1.15.3, OSQP 1.1.3, and
  pythermalcomfort 3.9.8.
- **106 tests passed**, with the fixture that prohibits test-network egress enabled.
- Ruff linting, the format check for 36 files, and strict mypy for 30 source files passed.
- `uv lock --check` and wheel construction passed without modifying the canonical environment.
- All three cases in the included frozen suite reported `case_checks=true` and `valid=true`; the
  original `METHOD-DEGRADED` classifications were preserved.
- Fifty-seven retained files matched their upstream counterparts. All retained functions and
  classes in the two extracted modules had ASTs identical to upstream.
- The dry training plan did not access BOPTEST or create runtime outputs.

## Historical research records

- [Original hierarchical MPC preregistration](hierarchical_mpc_preregistration.md)
- [Common-observation refit preregistration](mpc_common_observation_refit_preregistration_20260905.md)
- [Fourth-step forecast-availability preregistration](mpc_t_plus_4_forecast_availability_preregistration_20260905.md)
- [Three-repeat evaluation and versioned-publication preregistration](mpc_common_observation_formal_repeats_preregistration_20260907.md)

These documents preserve the actual sequence and approved scope of the upstream research. They may
contain upstream paths or commands and are not operational instructions for this standalone
repository. Follow this repository's README for installation and current commands.
