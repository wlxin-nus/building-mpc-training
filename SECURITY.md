# Security Policy

## Supported versions

Security fixes are applied to the current `main` branch. Historical commits, archived experiment
outputs, and externally operated BOPTEST deployments are not maintained by this repository.

## Report a vulnerability

Please use GitHub's private security-advisory workflow for this repository. Do not disclose a
suspected vulnerability in a public issue before the maintainers have had an opportunity to assess
it. Include the affected revision, reproduction steps, expected impact, and any suggested
mitigation. Do not include live credentials or sensitive experiment data in the report.

Model accuracy, control performance, or a scientifically unfavorable result is not by itself a
security vulnerability. Integrity failures, unsafe file handling, credential exposure, dependency
vulnerabilities, and unauthorized control or network behavior are within scope.

## Deployment guidance

- Run the code and BOPTEST only on trusted machines and networks.
- Use a dedicated virtual environment and review dependency-lock changes before installation.
- Keep `BOPTEST_URL` free of embedded credentials; the CLI rejects credential-bearing endpoints.
- Do not expose a BOPTEST service directly to an untrusted network.
- Treat downloaded model, trajectory, and configuration files as untrusted until their provenance
  and checksums have been verified.
- Never commit `.env` files, API keys, tokens, private endpoints, or raw private experiment data.

The software is research code. Review all constraints, fallbacks, and safety controls before using
it with a physical building or any other safety-critical system.
