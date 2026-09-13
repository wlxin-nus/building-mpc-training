# Third-Party Notices

This repository is distributed under the MIT License. It depends on or interoperates with the
third-party projects listed below. Those projects remain subject to their own licenses and terms.
This notice is informational and does not replace the license text supplied by each project.

## Runtime Python dependencies

| Project | Pinned version | License | Project site |
|---|---:|---|---|
| NumPy | 2.2.6 | BSD-3-Clause | <https://numpy.org/> |
| SciPy | 1.15.3 | BSD-3-Clause | <https://scipy.org/> |
| OSQP | 1.1.3 | Apache-2.0 | <https://osqp.org/> |
| pythermalcomfort | 3.9.8 | MIT | <https://pythermalcomfort.readthedocs.io/> |

Development and build tools are resolved by `uv.lock`; consult each package distribution for its
license text. The lock file records exact artifacts and hashes but does not relicense them.

## BOPTEST

Live data collection and closed-loop validation require an independently installed
[IBPSA Project 1 BOPTEST](https://github.com/ibpsa/project1-boptest) service and compatible test
cases. BOPTEST is not bundled in this repository. BOPTEST is distributed under its own revised
three-clause BSD license with an additional enhancement paragraph; consult the upstream project for
the authoritative license and notices.

The case-profile files in this repository link to the relevant upstream BOPTEST case
documentation. Weather files, FMUs, Docker images, and raw BOPTEST datasets are not redistributed
here.

## Frozen model artifacts

The bundled coefficient arrays and manifests were produced from BOPTEST-based research experiments.
They are distributed with this repository as small, immutable reference artifacts. Their model
cards and the top-level provenance manifest record their scientific source and identity.
