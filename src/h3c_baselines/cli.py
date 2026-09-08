"""MPC-only command surface; all mutations require an explicit --execute."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import socket
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from h3c.experiments.profiles import repository_root
from h3c.runtime.source_identity import committed_source_identity
from h3c_baselines.mpc.refit import (
    refit_hierarchical_mpc,
    resolved_refit_plan,
    verify_refit_workspace,
)
from h3c_baselines.mpc.registry import (
    freeze_method_degraded_mpc_suite,
    freeze_validated_mpc_suite,
    resolved_method_degraded_freeze_plan,
    verify_frozen_mpc_suite,
)
from h3c_baselines.mpc.training import (
    resolved_training_plan,
    train_hierarchical_mpc,
    verify_frozen_mpc_model,
)
from h3c_baselines.mpc.validation import (
    execute_fresh_validation,
    resolved_validation_plan,
    verify_validation_workspace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m h3c_baselines.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    mpc = commands.add_parser("mpc", help="identify, refit, validate, and freeze MPC")
    commands = mpc.add_subparsers(dest="mpc_command", required=True)
    train = commands.add_parser("train", help="collect BOPTEST episodes and fit candidates")
    train.add_argument("--case", choices=("all",), default="all")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--max-fit-episodes", type=int, default=64)
    train.add_argument("--execute", action="store_true")
    refit = commands.add_parser("refit", help="offline refit of a preserved failed source bank")
    refit.add_argument("--source-run", type=Path, required=True)
    refit.add_argument("--align-training-windows", action="store_true")
    refit.add_argument("--execute", action="store_true")
    verify_model = commands.add_parser("verify-model")
    verify_model.add_argument("--case", required=True, choices=("SZ_Air", "MZ_Hydro", "MZ_Air"))
    verify_model.add_argument("--mpc-release", type=Path)
    for name in ("verify-refit", "verify-validation"):
        commands.add_parser(name).add_argument("workspace", type=Path)
    validate = commands.add_parser("validate", help="fresh closed-loop validation and publication")
    validate.add_argument("--refit-workspace", type=Path, required=True)
    validate.add_argument("--publication", choices=("default", "versioned"), default="default")
    validate.add_argument("--execute", action="store_true")
    freeze = commands.add_parser("freeze-adverse", help="explicit method-degraded publication")
    freeze.add_argument("--validation-workspace", type=Path, required=True)
    freeze.add_argument("--publication", choices=("default", "versioned"), default="default")
    freeze.add_argument("--execute", action="store_true")
    commands.add_parser("verify-frozen-suite").add_argument("--mpc-release", type=Path)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def _release_argument(value: Path | None) -> Path | None:
    if value is None:
        return None
    if re.fullmatch(r"[0-9a-f]{64}", str(value)):
        return repository_root() / "models" / "mpc_releases" / str(value)
    return value


def _physical_campaign_preflight(endpoint: str) -> dict[str, Any]:
    """Portable launcher checks only; no test selection or physical advance."""
    root = repository_root().resolve()
    source = committed_source_identity(root)
    Path(__file__).resolve().relative_to(root / "src" / "h3c_baselines")
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected = dict(requirement.split("==") for requirement in project["project"]["dependencies"])
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError("MPC numerical dependencies differ from the frozen versions")
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 1024**3:
        raise RuntimeError("MPC requires at least 1 GiB free on the repository volume")
    if any(
        path.name in {".training.lock", ".validation.lock", ".execution.lock"}
        for path in (root / "outputs").rglob("*")
    ):
        raise RuntimeError("An existing MPC execution/training/validation lock blocks launch")
    parsed = urlparse(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("BOPTEST_URL must be a credential-free HTTP(S) origin")
    with socket.create_connection(
        (parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)), timeout=5.0
    ):
        pass
    return {
        "source_commit": source,
        "versions": versions,
        "disk_free_bytes": free_bytes,
        "boptest_tcp_reachable": True,
    }


def _endpoint() -> str:
    endpoint = os.environ.get("BOPTEST_URL", "").rstrip("/")
    if not endpoint:
        raise ValueError("BOPTEST_URL is required for physical MPC execution")
    return endpoint


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.mpc_command
    if command in {"verify-refit", "verify-validation", "verify-model", "verify-frozen-suite"}:
        if command == "verify-refit":
            result = verify_refit_workspace(args.workspace)
        elif command == "verify-validation":
            result = verify_validation_workspace(args.workspace)
        elif command == "verify-model":
            result = verify_frozen_mpc_model(args.case, _release_argument(args.mpc_release))
        else:
            result = verify_frozen_mpc_suite(_release_argument(args.mpc_release))
        _print(result)
        return 0 if result["valid"] else 1
    if command == "train":
        plan = resolved_training_plan(workers=args.workers, max_fit_episodes=args.max_fit_episodes)
        if not args.execute:
            _print(plan)
            return 0
        endpoint = _endpoint()
        _physical_campaign_preflight(endpoint)
        _print(
            train_hierarchical_mpc(
                endpoint=endpoint, workers=args.workers, maximum_fit_episodes=args.max_fit_episodes
            )
        )
        return 0
    if command == "refit":
        plan = resolved_refit_plan(
            args.source_run, align_training_windows=args.align_training_windows
        )
        if not args.execute:
            _print(plan)
            return 0
        _print(
            refit_hierarchical_mpc(
                args.source_run, align_training_windows=args.align_training_windows
            )
        )
        return 0
    if command == "validate":
        plan = (
            resolved_validation_plan(args.refit_workspace)
            if args.publication == "default"
            else resolved_validation_plan(args.refit_workspace, publication=args.publication)
        )
        if not args.execute:
            _print(plan)
            return 0
        endpoint = _endpoint()
        _physical_campaign_preflight(endpoint)
        result = execute_fresh_validation(
            args.refit_workspace, endpoint=endpoint, publication=args.publication
        )
        freeze = (
            freeze_method_degraded_mpc_suite
            if result.get("classification") == "METHOD-DEGRADED"
            else freeze_validated_mpc_suite
        )
        _print(
            {
                "validation": result,
                "freeze": freeze(Path(result["run_dir"]), publication=args.publication),
            }
        )
        return 0
    if command == "freeze-adverse":
        plan = resolved_method_degraded_freeze_plan(
            args.validation_workspace, publication=args.publication
        )
        if not args.execute:
            _print(plan)
            return 0 if plan["validation_valid_for_admission"] else 1
        _print(
            freeze_method_degraded_mpc_suite(
                args.validation_workspace, publication=args.publication
            )
        )
        return 0
    raise AssertionError("unreachable MPC command")


def mpc_main() -> int:
    """Short console entry point: building-mpc train (without the 'mpc' group)."""
    return main(["mpc", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
