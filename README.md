# Building MPC Training

Reproducible system identification, hierarchical model predictive control (MPC), closed-loop validation, and immutable model publication for three BOPTEST cooling cases.

This repository is a focused extraction of the MPC baseline used by the H3C research project. It retains the numerical MPC implementation, case profiles, preregistered protocol, and a frozen three-case model release. It deliberately excludes the H3C agent, LLM providers and prompts, DRL baselines, paper source, and raw experiment workspaces.

> [!IMPORTANT]
> This is a software-and-model artifact, not a complete raw-data reproduction package. The bundled coefficients were produced before this standalone repository was created. The original episode bank and formal evaluation trajectories are not distributed here, so the frozen files can be integrity-checked against the repository manifests but cannot be refitted independently from the contents of this repository alone.

> [!WARNING]
> Two models in the bundled release are intentionally classified as `METHOD-DEGRADED`; only `MZ_Hydro` is `BASELINE-READY`. These labels describe the recorded closed-loop evidence, not file corruption. The release is not claimed to be globally optimal, fully tuned, or superior to an enhanced rule-based controller.

## Contents

- [Quick start](#quick-start)
- [Connect BOPTEST](#connect-boptest)
- [Supported workflows](#supported-workflows)
- [Method and experiment protocol](#method-and-experiment-protocol)
- [Frozen reference release](#frozen-reference-release)
- [Outputs and failure recovery](#outputs-and-failure-recovery)
- [Reproducibility boundaries](#reproducibility-boundaries)
- [Development and verification](#development-and-verification)
- [Troubleshooting](#troubleshooting)
- [Citation and license](#citation-and-license)

Detailed references:

- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Data contract and reproducibility boundary](docs/data_contract.md)
- [Extraction and verification record](docs/extraction.md)
- [Original hierarchical MPC preregistration](docs/hierarchical_mpc_preregistration.md)

## Quick start

### Requirements

For offline inspection and model verification:

- Git;
- Python 3.11–3.13 (Python 3.12 is recommended);
- Windows, Linux, or macOS on a platform supported by the pinned numerical packages.

For physical data collection or closed-loop validation, you additionally need an external BOPTEST deployment with the required test cases and at least six workers. A GPU, Conda, W&B, and API keys are not required.

### Option A: install with `uv` (recommended)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) with a trusted package manager or the official installer.

Windows PowerShell:

```powershell
winget install --id astral-sh.uv -e
uv --version
```

Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Then create the locked project environment:

```text
git clone https://github.com/wlxin-nus/building-mpc-training.git
cd building-mpc-training
uv python install 3.12
uv sync --locked --extra dev
uv run --no-sync building-mpc --help
```

`uv python install` can provision Python, so a separate manual Python download is not required. The committed lockfile is the canonical dependency resolution.

### Option B: standard virtual environment and `pip`

Conda is not required. If Python 3.11–3.13 is already installed, use a conventional virtual environment.

Windows PowerShell:

```powershell
git clone https://github.com/wlxin-nus/building-mpc-training.git
Set-Location building-mpc-training
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
building-mpc --help
```

Linux or macOS:

```bash
git clone https://github.com/wlxin-nus/building-mpc-training.git
cd building-mpc-training
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
building-mpc --help
```

The direct numerical dependencies are pinned in `pyproject.toml`; `uv sync --locked` additionally reproduces the complete transitive lock. Use the `uv` path for archival reproduction.

This project intentionally uses a full checkout with an editable installation because runtime profiles and frozen releases live outside the Python package. Do not install it into an environment that already contains the full H3C repository: both expose the `h3c` and `h3c_baselines` namespaces.

### Verify the bundled model release

The release identifier is:

```text
b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c
```

With `uv`:

```powershell
$release = 'b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c'
uv run --no-sync building-mpc verify-frozen-suite --mpc-release $release
uv run --no-sync building-mpc verify-model --case SZ_Air --mpc-release $release
```

With an activated standard virtual environment, omit `uv run --no-sync`:

```text
building-mpc verify-frozen-suite --mpc-release b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c
```

A successful command exits with code 0 and prints JSON containing `"valid": true`. This verifies the internal identities and hashes of the published suite. It does not replay the unavailable source trajectories.

### Preview the training plan safely

```text
uv run --no-sync building-mpc train --case all --workers 4 --max-fit-episodes 64
```

Without `--execute`, the command is a dry plan: it does not contact BOPTEST, create a test, or write runtime output. The returned JSON contains `"execution": false`.

## Connect BOPTEST

BOPTEST is an external dependency and is not vendored in this repository. Follow the [official BOPTEST deployment guide](https://ibpsa.github.io/project1-boptest/docs-userguide/getting_started.html). The official service supports multiple concurrent test instances by scaling its worker service.

The complete training protocol uses four active collection lanes and reserves two workers, so provision at least six workers. From a compatible BOPTEST checkout, the deployment is typically started with:

```text
docker compose up --scale worker=6 web worker provision
```

The exact compose command and exposed port depend on the BOPTEST revision and local deployment. Do not change Docker solely to match an example URL. Set `BOPTEST_URL` to the credential-free HTTP(S) origin that is reachable from the Python process; ports 8000, 5000, and 80 are all possible deployment choices.

Windows PowerShell example:

```powershell
$env:BOPTEST_URL = 'http://127.0.0.1:8000'
Invoke-RestMethod "$env:BOPTEST_URL/version"
Invoke-RestMethod "$env:BOPTEST_URL/testcases"
```

Linux or macOS example:

```bash
export BOPTEST_URL='http://127.0.0.1:8000'
curl -fsS "$BOPTEST_URL/version"
curl -fsS "$BOPTEST_URL/testcases"
```

The following testcase identifiers must be available:

| Repository case | BOPTEST testcase |
|---|---|
| `SZ_Air` | `bestest_air` |
| `MZ_Hydro` | `multizone_office_simple_hydronic` |
| `MZ_Air` | `multizone_office_simple_air` |

The physical launcher validates the URL shape, pinned numerical dependencies, free disk space, clean committed source, execution locks, and TCP reachability. It does **not** currently certify the BOPTEST version, enumerate testcases, or measure available worker capacity. Run the API checks above before a long campaign and record the returned BOPTEST version with your results. A different BOPTEST build is a different experimental environment, even if the testcase model name is unchanged.

`.env.example` is documentation only; the CLI does not automatically load `.env`. Export the variable in the process that launches the command.

## Supported workflows

The command surface has three deliberately separate workflows. They are not interchangeable.

### 1. Inspect or consume the frozen models

Use `verify-frozen-suite` and `verify-model` as shown above. The immutable coefficients, model cards, training manifests, and suite manifest are under:

```text
models/mpc_releases/<freeze-identity>/
```

This is the only workflow that is fully self-contained in the repository.

### 2. Collect a new episode bank and fit new legacy models

This creates a new experiment. It will not reproduce the identity of the bundled release.

```powershell
$env:BOPTEST_URL = 'http://127.0.0.1:8000'  # replace with your actual origin
uv run --no-sync building-mpc train --case all --workers 4 --max-fit-episodes 64 --execute
```

Operational behavior:

- cases run sequentially in the fixed order `SZ_Air` → `MZ_Hydro` → `MZ_Air`;
- each case uses four independent BOPTEST lanes concurrently;
- fit evidence is evaluated at 8, 16, 32, and 64 fit episodes, with separate holdout, reference, and closed-loop episodes;
- every lane selects its own test and reinitializes it for each episode after a seven-day internal warm-up;
- registered BOPTEST lanes are stopped on ordinary completion and handled `Exception` paths; a failure during test selection or configuration before lane registration may require manual TestID cleanup;
- the nominal six-hour campaign budget is checked between batches and is not a hard interrupt for an in-flight HTTP request or episode;
- new models are written to `models/mpc/<case>/`, and an existing target is never overwritten.

The current CLI accepts `--case all` only. It does not launch individual cases, stream per-episode progress, or resume an interrupted physical campaign. The files named `checkpoint-008.json` through `checkpoint-064.json` are checkpoint-selection reports—not restartable model/process checkpoints.

Successful runs end with:

```text
outputs/baselines/mpc/training/<run-id>/completion.json
```

Handled training failures retain `failure.json`. A `KeyboardInterrupt`, hard process termination, or failure before the outer exception handler may not produce that file. Preserve every incomplete run directory as experimental evidence.

> [!NOTE]
> The public CLI does not currently connect a successful from-scratch training directory directly to the versioned `validate`/publication workflow. It produces the legacy `models/mpc/<case>` targets. Do not pass that successful directory to `refit`; the refit verifier intentionally rejects it.

### 3. Refit an authorized preserved source bank, validate, and publish

This advanced recovery workflow is how the bundled versioned models were produced. Its input is a preserved historical **failed** training bank with the expected evidence contract. That bank is not included.

Place an authorized, byte-preserved copy under `outputs/baselines/mpc/training/<source-run>/`, then preview and execute:

```powershell
$source = 'outputs/baselines/mpc/training/<source-run>'
uv run --no-sync building-mpc refit --source-run $source --align-training-windows
uv run --no-sync building-mpc refit --source-run $source --align-training-windows --execute
```

Use the returned workspace path in the subsequent commands; placeholders below are not literal paths:

```powershell
$refit = 'outputs/baselines/mpc/refit/<refit-run>'
uv run --no-sync building-mpc verify-refit $refit
uv run --no-sync building-mpc validate --refit-workspace $refit --publication versioned
uv run --no-sync building-mpc validate --refit-workspace $refit --publication versioned --execute
uv run --no-sync building-mpc verify-validation 'outputs/baselines/mpc/validation/<validation-run>'
```

Fresh validation may run the three cases concurrently, each with an independent test. It does not retune coefficients, objective weights, or the four-step horizon. A validated suite is published atomically to `models/mpc_releases/<freeze-identity>/`; an existing identity cannot be replaced.

| Classification | Meaning |
|---|---|
| `BASELINE-READY` | Evidence is valid, fallback count is zero, and occupied peak `|PMV| ≤ 0.70`. |
| `METHOD-DEGRADED` | Evidence is valid, but fallback occurred or the comfort boundary was exceeded. The adverse result is retained and disclosed. |
| Integrity failure | Identity, lifecycle, trajectory, finite-value, or secret checks failed. Publication is blocked. |

`freeze-adverse` is an explicit publication route for already verified method degradation. It does not waive integrity checks.

## Method and experiment protocol

In this repository, “training” means identifying the predictive model used by MPC. It is not reinforcement learning, and the optimizer does not learn objective weights online.

| Component | Frozen implementation |
|---|---|
| Predictive model | Multi-output vector ARX for zone air temperatures and whole-building power |
| History | Current output plus four completed output records; candidate control plus four completed controls |
| Identification | Standardized ridge regression; alpha candidates `1e-6, 1e-4, 1e-2, 1, 100` |
| Horizon | Four 900-second steps (1 hour) |
| Hierarchy | Hourly building coordinator and 15-minute local zone MPC, with one feedback iteration |
| Comfort | Shared PMV owner, internal linear approximation, and a calibration-derived residual margin |
| Solver | OSQP 1.1.3 with the frozen tolerances, iteration budget, polishing, and fallback logic |
| Physical setpoint envelope | 20–30 °C |
| Occupied controller support | 23.5–26.5 °C |
| Unoccupied controller support | 20–30 °C |

The algorithm, objective weights, and core constraints are common across cases; physical point names, occupancy interpretation, power scales, and case-specific comfort treatment are declared in JSON profiles. See [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for the exact values and source files.

### Calendar windows

Day indices are BOPTEST simulation-day indices and intervals are half-open; they are not host computer dates.

| Case | Fit/holdout/calibration development window | Fresh validation | Historical formal evaluation |
|---|---|---|---|
| `SZ_Air` | `[196, 203)`, 7 days | day 196, 167 h | `[203, 210)`, 168 h |
| `MZ_Hydro` | `[213, 218)`, 5 days after aligned refit | day 213, 167 h | `[220, 225)`, 120 h |
| `MZ_Air` | `[192, 199)`, 7 days | day 192, 167 h | `[199, 206)`, 168 h |

Source collection episodes are seven days. `--align-training-windows` takes the five-day prefix for Hydronic refit and retains seven days for both air cases. Holdout trajectories are excluded from coefficient fitting but are used for ridge selection and persistence comparison; they are therefore not untouched final test data. Fresh validation uses 668 × 900 s = 167 h, leaving the final forecast hour outside the controlled interval.

## Frozen reference release

Bundled release:

```text
b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c
```

| Case | Model identity (abbreviated) | Fresh-validation fallbacks | Occupied peak `|PMV|` | Classification |
|---|---|---:|---:|---|
| `SZ_Air` | `62b361…7791fd1` | 1 | 0.61 | `METHOD-DEGRADED` |
| `MZ_Hydro` | `29c104…a85413` | 0 | 0.63 | `BASELINE-READY` |
| `MZ_Air` | `4e38b2…991529` | 2 | 0.90 | `METHOD-DEGRADED` |

Read the model cards and suite manifest in the release directory before reporting these models. A low open-loop prediction error alone is not evidence of optimal closed-loop control.

## Outputs and failure recovery

Runtime outputs are intentionally Git-ignored:

```text
outputs/baselines/mpc/
  training/<run-id>/
  refit/<run-id>/
  validation/<run-id>/
models/mpc/                       # newly trained legacy targets; ignored
models/mpc_releases/<identity>/   # reviewed immutable releases
```

### Ordinary interruption or exception

Training stops lanes that have been registered, and validation attempts to stop the TestID visible on its client, on ordinary completion and handled `Exception` paths. Handled failures normally write `failure.json`, but `KeyboardInterrupt`, a hard kill, or a selection/configuration failure before ownership is registered can bypass some evidence or cleanup. Retain the incomplete directory, inspect BOPTEST for an orphan TestID owned by this run, correct the external cause, and begin a new run. In-place resume is not implemented.

### Power loss or forced process termination

A hard kill can bypass cleanup, leaving BOPTEST tests and an execution lock. Before another run:

1. confirm that no original Python process is still active;
2. inspect BOPTEST and explicitly stop only the orphan TestIDs;
3. inspect `.training.lock`, `.validation.lock`, or `.execution.lock` under `outputs/baselines/mpc/`;
4. remove a lock only after verifying that its recorded PID no longer owns the campaign;
5. preserve the incomplete run directory and start a new run.

Do not blindly delete locks or stop all tests on a shared BOPTEST service. The current implementation has no persistent active-ID registry and cannot guarantee automatic recovery after a hard kill.

## Reproducibility boundaries

What is included:

- the MPC implementation and exact direct numerical dependency pins;
- three case profiles, common program, and solver/training configuration;
- immutable coefficients, model cards, manifests, and internal hashes for one release;
- preregistration records and file-level upstream provenance;
- offline unit and fake-physical tests.

What is not included:

- the BOPTEST service, testcase images, or weather assets;
- raw episode banks, refit workspaces, validation workspaces, or formal trajectories;
- private endpoints, credentials, logs, or machine-specific environments;
- the H3C controller, DRL baselines, or paper source.

Consequently, this artifact supports source inspection, environment reconstruction, dry planning, tests, model loading, and integrity verification. Reproducing the bundled coefficients or auditing every historical KPI requires the separately governed raw evidence. See [the data contract](docs/data_contract.md).

`provenance.json` binds verbatim scientific files to upstream commit `32524f3b3fd3861a9323226612c776b27f4b8a85`. The archival preregistration files preserve the original research chronology and may contain obsolete workstation commands; they are evidence, not current operating instructions. Use this README for all commands.

## Repository layout

```text
configs/                         case profiles, MPC settings, common control program
docs/                            protocol, data contract, extraction record, preregistration
models/mpc_releases/             immutable bundled model suite
src/h3c_baselines/mpc/
  vector_arx.py                  feature construction, fit, prediction, serialization
  training.py                    BOPTEST collection and candidate selection
  refit.py                       preserved-bank recovery and candidate verification
  forecast.py                    common observation and four-step forecast alignment
  optimizer.py                   coordinator, local MPC, and fallback
  validation.py                  fresh closed-loop validation and evidence checks
  registry.py                    model verification and immutable publication
src/h3c_baselines/cli.py         command-line interface
src/h3c/                         minimal shared physics, occupancy, PMV, and program logic
tests/                           unit, fake-physical, integrity, and extraction regressions
provenance.json                  upstream commit and verbatim-file hashes
CHECKSUMS.sha256                 repository-wide release integrity manifest
```

`configs/graphs/` is retained to satisfy the original case-profile contract. The MPC runtime does not use those graphs for causal control.

## Development and verification

Run the complete offline quality gate from a synchronized checkout:

```text
uv lock --check
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy --strict src/h3c src/h3c_baselines
uv build --no-sources
uv run --no-sync python scripts/verify_checksums.py
uv run --no-sync building-mpc verify-frozen-suite --mpc-release b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c
```

The test suite uses fake physical clients and blocks outbound network access. Passing it does not claim that a new physical training campaign has been run. See [the extraction record](docs/extraction.md) for the recorded extraction-time checks.

Contributions should not silently change a scientific constant. Any method, profile, dependency, or evidence-contract change requires an explicit rationale, new identity, and appropriate tests. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `BOPTEST_URL is required` | The environment variable is absent from the launching process. | Export it in the same shell; `.env` is not auto-loaded. |
| URL or TCP preflight fails | Wrong host/port, credentials/path in URL, Docker networking, firewall, or stopped service. | Probe `/version` and `/testcases` from the same shell and use only the origin, such as `http://127.0.0.1:8000`. |
| Test selection queues indefinitely | Insufficient or occupied BOPTEST workers. | Provision at least six workers and stop only confirmed orphan tests. |
| Numerical dependency mismatch | The active environment does not match the exact direct pins. | Run `uv sync --locked --extra dev` in a clean environment. |
| First CLI or test command starts slowly | `pythermalcomfort` may compile and cache Numba kernels on a new machine. | Allow the first import to complete; subsequent starts in the same environment should be faster. |
| Dirty-source preflight failure | Physical runs require a committed source identity. | Review and commit intended changes; do not bypass the identity gate. |
| Existing execution lock | Another campaign may be running, or a hard kill left a stale lock. | Verify the recorded PID and BOPTEST ownership before removing anything. |
| Existing `models/mpc/<case>` target | The launcher protects previous fitted models. | Archive the entire result with provenance or use a fresh checkout; do not overwrite it. |
| `METHOD-DEGRADED` with `valid: true` | The release evidence is intact but the recorded controller crossed a method-performance gate. | Report the classification; do not relabel it as corruption or success. |

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). It intentionally contains no paper citation or DOI that has not yet been assigned. Add the final bibliographic record only when it is authoritative.

The source code is distributed under the [MIT License](LICENSE). External projects and numerical libraries retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). BOPTEST software and testcase data are not bundled and remain subject to their upstream terms.

For security concerns, follow [SECURITY.md](SECURITY.md). For scientific or usability issues, use the repository issue tracker and include the source commit, operating system, Python version, BOPTEST version, testcase, command, and redacted failure evidence.
