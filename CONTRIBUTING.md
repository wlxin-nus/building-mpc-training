# Contributing

Thank you for helping improve Building MPC Training. Contributions should preserve the distinction
between software maintenance and changes to the registered scientific protocol.

## Set up a development environment

Install Python 3.11, 3.12, or 3.13 and
[uv](https://docs.astral.sh/uv/), then run:

```console
git clone https://github.com/wlxin-nus/building-mpc-training.git
cd building-mpc-training
uv sync --frozen --extra dev
```

The offline test suite does not contact BOPTEST. A live BOPTEST service is required only for
physical data collection and closed-loop validation.

## Make a change

1. Open an issue before changing a scientific assumption, training window, model structure,
   objective, constraint, or validation rule.
2. Create a focused branch and keep unrelated changes separate.
3. Add or update tests for every behavioral change.
4. Do not overwrite frozen models, source episode banks, completed outputs, or failure evidence.
5. Do not commit credentials, local endpoints, raw run directories, or private data.
6. Update the documentation and changelog when user-visible behavior changes.

A scientific-method change must receive a new protocol identity and must not reuse an existing
freeze identity. Documentation-only and engineering changes must not claim to reproduce physical
experiments that were not rerun.

## Run the checks

Before opening a pull request, run:

```console
uv lock --check
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy --strict src/h3c src/h3c_baselines
uv build --no-sources
uv run --no-sync python scripts/verify_checksums.py --write
uv run --no-sync python scripts/verify_checksums.py
```

Also verify the bundled frozen suite with the command documented in the README. Pull requests must
pass the same checks in CI. Regenerate `CHECKSUMS.sha256` only after all intended files are final;
review its diff before committing it.

## Pull requests

Describe the motivation, implementation, tests, and any reproducibility impact. Explicitly state
whether the change affects scientific configuration, model identity, stored artifacts, or only
documentation and tooling. Small, reviewable pull requests are preferred.

By contributing, you agree that your contribution is distributed under the repository's MIT
License.
