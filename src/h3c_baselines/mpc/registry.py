"""All-or-none publication of a three-case fresh-validated MPC model suite."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from h3c.experiments.profiles import load_profile, repository_root
from h3c_baselines.configuration import load_hierarchical_mpc_config
from h3c_baselines.mpc.optimizer import comfort_target_met
from h3c_baselines.mpc.validation import (
    _read_json,
    _sha256,
    _suite_evidence_checks,
    _write_json,
    verify_validation_workspace,
)
from h3c_baselines.mpc.vector_arx import FittedArxModel, expected_model_identity
from h3c_baselines.outputs.integrity import secret_occurrences


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _comfort_target_attestation_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    peak = value.get("occupied_peak_absolute_pmv")
    if isinstance(peak, bool) or not isinstance(peak, (int, float)):
        return False
    peak_value = float(peak)
    return bool(
        math.isfinite(peak_value)
        and isinstance(value.get("comfort_target_met"), bool)
        and value["comfort_target_met"] is comfort_target_met(peak_value)
    )


def _resolve_validation(workspace: Path) -> tuple[Path, dict[str, Any]]:
    root = repository_root().resolve()
    target = workspace if workspace.is_absolute() else root / workspace
    target = target.resolve()
    verification = verify_validation_workspace(target)
    if verification.get("valid") is not True:
        raise ValueError("only a complete verified MPC validation suite can be frozen")
    return target, verification


_STRICT_ADMISSION = "strict_zero_fallback"
_DEGRADED_ADMISSION = "post_result_method_degraded"
_PREREGISTERED_DEGRADED_ADMISSION = "preregistered_method_degraded"
_DEGRADED_ADMISSIONS = {_DEGRADED_ADMISSION, _PREREGISTERED_DEGRADED_ADMISSION}
_DEFAULT_PUBLICATION = "default"
_VERSIONED_PUBLICATION = "versioned"
_LEGACY_DEGRADED_DECISION = (
    "docs/hierarchical_mpc_formal_evaluation_post_result_preregistration_20260831.md"
)
_PREREGISTERED_DEGRADED_DECISION = (
    "docs/mpc_common_observation_formal_repeats_preregistration_20260907.md"
)
_VERSIONED_PUBLICATION_VALUE = "versioned_identity_directory_replace"
_V3_PAYLOAD_KEYS = (
    "publication",
    "admission_mode",
    "decision_reference",
    "validation_workspace",
    "validation_terminal_kind",
    "validation_terminal_sha256",
    "validation_source_commit",
    "refit_verification_identity",
    "refit_workspace",
    "case_order",
    "model_identities",
    "validation_results",
    "source_bundles",
    "refit_candidate_gates",
)
_V3_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "freeze_identity",
        *_V3_PAYLOAD_KEYS,
        "cases",
        "secret_exposure_count",
    }
)
_CASE_ROW_KEYS = frozenset(
    {
        "case",
        "model_identity",
        "coefficient_sha256",
        "model_card_sha256",
        "training_manifest_sha256",
    }
)


def _fallback_count(value: Mapping[str, Any]) -> int | None:
    raw = value.get("fallback_count")
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None


def _method_degraded(value: Mapping[str, Any]) -> bool:
    fallback_count = _fallback_count(value)
    return fallback_count is not None and (
        fallback_count > 0 or value.get("comfort_target_met") is False
    )


def _preregistered_case_eligible(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("evidence_valid") is True
        and isinstance(value.get("checks"), Mapping)
        and all(value["checks"].values())
        and _fallback_count(value) == 0
        and value.get("comfort_target_met") is True
        and _comfort_target_attestation_valid(value)
    )


def verify_method_degraded_validation_workspace(
    workspace: Path,
    *,
    admission_mode: str = _DEGRADED_ADMISSION,
) -> dict[str, Any]:
    """Verify a terminal suite whose only failed gate is controller fallback."""

    try:
        if admission_mode not in _DEGRADED_ADMISSIONS:
            raise ValueError("unknown method-degraded admission mode")
        root = repository_root().resolve()
        target = workspace if workspace.is_absolute() else root / workspace
        target = target.resolve()
        target.relative_to((root / "outputs" / "baselines" / "mpc" / "validation").resolve())
        if (target / "completion.json").exists():
            raise ValueError("method-degraded admission requires preserved failure evidence")
        failure = _read_json(target / "failure.json")
        plan = _read_json(target / "resolved_plan.json")
        evidence = _read_json(target / "suite_evidence.json")
        checks, recomputed = _suite_evidence_checks(target, evidence)
        false_checks = {name for name, passed in checks.items() if passed is not True}
        case_order = list(load_hierarchical_mpc_config()["case_order"])
        fallback_count = sum(int(row.get("fallback_count", -1)) for row in recomputed)
        preregistered = admission_mode == _PREREGISTERED_DEGRADED_ADMISSION
        case_evidence_healthy = len(recomputed) == len(case_order) and all(
            row.get("case") == case
            and row.get("evidence_valid") is True
            and isinstance(row.get("checks"), Mapping)
            and all(row["checks"].values())
            and isinstance(row.get("fallback_count"), int)
            and int(row["fallback_count"]) >= 0
            and row.get("eligible")
            is (
                int(row["fallback_count"]) == 0
                and (row.get("comfort_target_met") is True or not preregistered)
            )
            and isinstance(row.get("reward"), (int, float))
            and not isinstance(row.get("reward"), bool)
            and math.isfinite(float(row["reward"]))
            and _comfort_target_attestation_valid(row)
            for case, row in zip(case_order, recomputed, strict=True)
        )
        expected_failure = {
            "schema": "h3c_hierarchical_mpc_validation_failure",
            "schema_version": 1,
            "validation_source_commit": evidence.get("validation_source_commit"),
            "checks": checks,
            "cases": recomputed,
            "candidate_promotion": False,
            "secret_exposure_count": evidence.get("secret_exposure_count"),
        }
        plan_payload = {
            key: value
            for key, value in plan.items()
            if key not in {"validation_source_commit", "plan_identity"}
        }
        plan_payload["execution"] = False
        admission_checks = {
            "preserved_registered_failure": failure == expected_failure,
            "only_zero_fallback_aggregate_failed": false_checks == {"all_arms_recomputed"},
            "case_evidence_healthy": case_evidence_healthy,
            "registered_method_degradation_present": fallback_count > 0
            or (
                preregistered and any(row.get("comfort_target_met") is False for row in recomputed)
            ),
            "plan_identity": plan.get("schema") == "h3c_hierarchical_mpc_fresh_validation_plan"
            and plan.get("execution") is True
            and _identity(plan_payload) == plan.get("plan_identity"),
            "source_identity": plan.get("validation_source_commit")
            == evidence.get("validation_source_commit")
            == failure.get("validation_source_commit")
            and isinstance(evidence.get("validation_source_commit"), str)
            and target.name.endswith(f"-{str(evidence.get('validation_source_commit'))[:8]}"),
            "refit_identity": plan.get("refit_workspace") == evidence.get("refit_workspace")
            and plan.get("refit_verification_identity")
            == evidence.get("refit_verification_identity"),
            "candidate_gates": isinstance(plan.get("candidate_gates"), list)
            and len(plan["candidate_gates"]) == len(case_order)
            and all(row.get("valid") is True for row in plan["candidate_gates"]),
        }
        return {
            "schema": "h3c_hierarchical_mpc_method_degraded_validation_verification",
            "schema_version": 1,
            "workspace": str(target),
            "checks": checks,
            "admission_checks": admission_checks,
            "cases": recomputed,
            "fallback_count": fallback_count,
            "admission_mode": admission_mode,
            "validation_source_commit": evidence.get("validation_source_commit"),
            "refit_workspace": evidence.get("refit_workspace"),
            "refit_verification_identity": evidence.get("refit_verification_identity"),
            "valid": all(admission_checks.values()),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "schema": "h3c_hierarchical_mpc_method_degraded_validation_verification",
            "schema_version": 1,
            "workspace": str(workspace),
            "error": f"{type(error).__name__}: {error}",
            "valid": False,
        }


def resolved_method_degraded_freeze_plan(
    validation_workspace: Path,
    *,
    admission_mode: str = _DEGRADED_ADMISSION,
    publication: str = _DEFAULT_PUBLICATION,
) -> dict[str, Any]:
    if publication == _VERSIONED_PUBLICATION:
        admission_mode = _PREREGISTERED_DEGRADED_ADMISSION
        verification = verify_validation_workspace(validation_workspace)
        valid_for_admission = bool(
            verification.get("valid") is True
            and verification.get("publication") == _VERSIONED_PUBLICATION
            and verification.get("classification") == "METHOD-DEGRADED"
        )
    elif publication == _DEFAULT_PUBLICATION:
        verification = (
            verify_method_degraded_validation_workspace(validation_workspace)
            if admission_mode == _DEGRADED_ADMISSION
            else verify_method_degraded_validation_workspace(
                validation_workspace,
                admission_mode=admission_mode,
            )
        )
        valid_for_admission = verification.get("valid") is True
    else:
        raise ValueError("unknown MPC publication mode")
    decision_reference = (
        _PREREGISTERED_DEGRADED_DECISION
        if admission_mode == _PREREGISTERED_DEGRADED_ADMISSION
        else _LEGACY_DEGRADED_DECISION
    )
    return {
        "schema": "h3c_hierarchical_mpc_method_degraded_freeze_plan",
        "schema_version": 1,
        "execution": False,
        "publication": publication,
        "admission_mode": admission_mode,
        "decision_reference": decision_reference,
        "validation_workspace": verification.get("workspace", str(validation_workspace)),
        "validation_valid_for_admission": valid_for_admission,
        "fallback_count": verification.get("fallback_count"),
        "cases": verification.get("cases", []),
    }


def _path_present(path: Path) -> bool:
    return os.path.lexists(path)


@contextmanager
def _publication_lock(path: Path) -> Any:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("another MPC suite publication owns the registry lock") from error
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def _is_reparse(path: Path) -> bool:
    if not _path_present(path):
        return False
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _models_root(root: Path) -> Path:
    target = root / "models"
    if not target.is_dir() or _is_reparse(target) or target.resolve() != target:
        raise ValueError("registered models root must be one real repository directory")
    return target


def _mpc_releases_root(root: Path, *, create: bool) -> Path:
    models = _models_root(root)
    target = models / "mpc_releases"
    if not _path_present(target):
        if not create:
            raise ValueError("registered MPC releases root does not exist")
        target.mkdir(parents=False, exist_ok=False)
    if (
        not target.is_dir()
        or _is_reparse(target)
        or target.resolve() != target
        or target.parent.resolve() != models
    ):
        raise ValueError("registered MPC releases root must be one real models directory")
    return target


_ROBUST_MARGIN_KEYS = {
    "scope",
    "estimator",
    "quantile",
    "order_statistic",
    "calibration_episode",
    "sample_count",
    "pmv_margin",
    "internal_comfort_band",
    "application",
}
_TERMINAL_REFERENCE_KEYS = {
    "target_offset_steps",
    "calibration_occupancy_source",
    "runtime_owner",
    "terminal_zone_targets",
    "terminal_occupied_zone_targets",
}


def _robust_attestation_valid(value: Any, model: FittedArxModel) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"robust_margin", "terminal_reference"}:
        return False
    robust = value.get("robust_margin")
    terminal = value.get("terminal_reference")
    if not isinstance(robust, Mapping) or not isinstance(terminal, Mapping):
        return False
    try:
        return bool(
            set(robust) == _ROBUST_MARGIN_KEYS
            and set(terminal) == _TERMINAL_REFERENCE_KEYS
            and robust.get("scope") == "case_specific_estimate_common_formula"
            and robust.get("estimator") == "p95_absolute_occupied_pmv_prediction_residual"
            and float(robust.get("quantile", -1.0)) == 0.95
            and robust.get("order_statistic") == "higher"
            and isinstance(robust.get("calibration_episode"), str)
            and bool(robust.get("calibration_episode"))
            and int(robust.get("sample_count", 0)) > 0
            and float(robust.get("pmv_margin", -1.0)) == model.pmv_robust_margin
            and float(robust.get("internal_comfort_band", -1.0)) == 0.5 - model.pmv_robust_margin
            and robust.get("application") == "internal_soft_comfort_band_only"
            and int(terminal.get("target_offset_steps", -1)) == 4
            and terminal.get("calibration_occupancy_source")
            == "basic_reference.disturbances[timestamp=origin+4]"
            and terminal.get("runtime_owner") == "effective_occupancy_forecast(step+4)"
            and int(terminal.get("terminal_zone_targets", 0)) > 0
            and int(terminal.get("terminal_occupied_zone_targets", -1)) >= 0
        )
    except (TypeError, ValueError):
        return False


def _source_bundle(refit_workspace: Path, case: str) -> dict[str, Any]:
    source = refit_workspace / case / "candidate_model"
    model = FittedArxModel.load(source / "model_coefficients.npz")
    source_card = _read_json(source / "model_card.json")
    source_report = _read_json(source / "candidate_report.json")
    robust_margin = source_card.get("robust_margin")
    calibration = source_report.get("calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    terminal_reference = calibration.get("terminal_reference")
    attestation = {
        "robust_margin": dict(robust_margin) if isinstance(robust_margin, Mapping) else {},
        "terminal_reference": (
            dict(terminal_reference) if isinstance(terminal_reference, Mapping) else {}
        ),
    }
    if not _robust_attestation_valid(attestation, model):
        raise ValueError(f"{case} refit model lacks robust-margin provenance")
    return {
        "case": case,
        "model_identity": model.identity,
        "coefficient_sha256": _sha256(source / "model_coefficients.npz"),
        "refit_model_card_sha256": _sha256(source / "model_card.json"),
        "refit_candidate_report_sha256": _sha256(source / "candidate_report.json"),
        "robust_calibration_attestation": attestation,
    }


def _expected_pending_files(case_order: list[str]) -> set[str]:
    return {
        "freeze_manifest.json",
        *{
            f"{case}/{name}"
            for case in case_order
            for name in (
                "model_coefficients.npz",
                "model_card.json",
                "training_manifest.json",
            )
        },
    }


def _frozen_model_card(
    *,
    case: str,
    model: FittedArxModel,
    validation_result: Mapping[str, Any],
    validation_source_commit: str,
    freeze_identity: str,
    refit_workspace: Path,
    validation_workspace: Path,
    source_bundle: Mapping[str, Any],
    admission_mode: str = _STRICT_ADMISSION,
    release_path: str | None = None,
) -> dict[str, Any]:
    profile = load_profile(case)
    config = load_hierarchical_mpc_config()
    value = {
        "schema": "h3c_hierarchical_mpc_model_card",
        "schema_version": 3 if release_path is not None else 2,
        "case": case,
        "controller": "hierarchical-mpc",
        "model_identity": model.identity,
        "pmv_robust_margin": model.pmv_robust_margin,
        "robust_calibration_attestation": source_bundle["robust_calibration_attestation"],
        "source_commit": validation_source_commit,
        "training_week_start_day": int(profile["evaluation_start_day"]) - 7,
        "training_week_days": 7,
        "model_structure": "vector_arx_current_plus_four_history_four_step",
        "hierarchy": "hourly_building_coordinator_and_15_minute_zone_mpc",
        "control_support_bounds_c": {
            key: config["excitation"][key] for key in ("occupied_bounds_c", "unoccupied_bounds_c")
        },
        "physical_validation": (
            "fresh_validation_passed"
            if admission_mode == _STRICT_ADMISSION
            else "fresh_validation_method_degraded"
        ),
        "freeze_identity": freeze_identity,
        "provenance": {
            "refit_workspace": refit_workspace.relative_to(repository_root()).as_posix(),
            "validation_workspace": validation_workspace.relative_to(repository_root()).as_posix(),
            "source_coefficient_sha256": source_bundle["coefficient_sha256"],
            "refit_model_card_sha256": source_bundle["refit_model_card_sha256"],
            "refit_candidate_report_sha256": source_bundle["refit_candidate_report_sha256"],
        },
        "validation": dict(validation_result),
    }
    if admission_mode in _DEGRADED_ADMISSIONS:
        value["admission_mode"] = admission_mode
        value["validation_classification"] = (
            "METHOD-DEGRADED-VALIDATION"
            if (
                int(validation_result["fallback_count"]) > 0
                if admission_mode == _DEGRADED_ADMISSION
                else _method_degraded(validation_result)
            )
            else "BASELINE-READY"
        )
    if release_path is not None:
        value["release_path"] = release_path
    return value


def _frozen_training_manifest(
    *,
    case: str,
    model: FittedArxModel,
    validation_result: Mapping[str, Any],
    validation_source_commit: str,
    freeze_identity: str,
    refit_workspace: Path,
    validation_workspace: Path,
    source_bundle: Mapping[str, Any],
    admission_mode: str = _STRICT_ADMISSION,
    release_path: str | None = None,
) -> dict[str, Any]:
    value = {
        "schema": "h3c_hierarchical_mpc_training_manifest",
        "schema_version": 3 if release_path is not None else 2,
        "case": case,
        "source_commit": validation_source_commit,
        "model_identity": model.identity,
        "training_output": refit_workspace.relative_to(repository_root()).as_posix(),
        "validation_output": validation_workspace.relative_to(repository_root()).as_posix(),
        "identification": "preserved_real_episode_bank_and_offline_refit",
        "fresh_validation": {
            "test_id": validation_result["test_id"],
            "select_count": 1,
            "initialize_count": 1,
            "stop_count": 1,
            "warmup_days": 7,
            "steps": validation_result["steps"],
            "fallback_count": validation_result["fallback_count"],
            "occupied_peak_absolute_pmv": validation_result["occupied_peak_absolute_pmv"],
            "comfort_target_met": validation_result["comfort_target_met"],
        },
        "freeze_identity": freeze_identity,
        "validation_attestation": dict(validation_result),
        "source_bundle": dict(source_bundle),
        "robust_calibration_attestation": source_bundle["robust_calibration_attestation"],
    }
    if admission_mode in _DEGRADED_ADMISSIONS:
        value["admission_mode"] = admission_mode
        value["validation_classification"] = (
            "METHOD-DEGRADED-VALIDATION"
            if (
                int(validation_result["fallback_count"]) > 0
                if admission_mode == _DEGRADED_ADMISSION
                else _method_degraded(validation_result)
            )
            else "BASELINE-READY"
        )
    if release_path is not None:
        value["release_path"] = release_path
    return value


def _stage_case(
    *,
    case: str,
    pending: Path,
    refit_workspace: Path,
    validation_workspace: Path,
    validation_result: Mapping[str, Any],
    validation_source_commit: str,
    freeze_identity: str,
    source_bundle: Mapping[str, Any],
    admission_mode: str = _STRICT_ADMISSION,
    release_path: str | None = None,
) -> dict[str, Any]:
    source = refit_workspace / case / "candidate_model"
    target = pending / case
    target.mkdir(parents=True, exist_ok=False)
    source_coefficients = source / "model_coefficients.npz"
    model = FittedArxModel.load(source_coefficients)
    if model.identity != expected_model_identity(model):
        raise ValueError(f"{case} candidate bundle identity is invalid")
    if validation_result.get("model_identity") != model.identity:
        raise ValueError(f"{case} validation/model identity mismatch")
    shutil.copy2(source_coefficients, target / "model_coefficients.npz")
    card = _frozen_model_card(
        case=case,
        model=model,
        validation_result=validation_result,
        validation_source_commit=validation_source_commit,
        freeze_identity=freeze_identity,
        refit_workspace=refit_workspace,
        validation_workspace=validation_workspace,
        source_bundle=source_bundle,
        admission_mode=admission_mode,
        release_path=release_path,
    )
    _write_json(target / "model_card.json", card)
    _write_json(
        target / "training_manifest.json",
        _frozen_training_manifest(
            case=case,
            model=model,
            validation_result=validation_result,
            validation_source_commit=validation_source_commit,
            freeze_identity=freeze_identity,
            refit_workspace=refit_workspace,
            validation_workspace=validation_workspace,
            source_bundle=source_bundle,
            admission_mode=admission_mode,
            release_path=release_path,
        ),
    )
    return {
        "case": case,
        "model_identity": model.identity,
        "coefficient_sha256": _sha256(target / "model_coefficients.npz"),
        "model_card_sha256": _sha256(target / "model_card.json"),
        "training_manifest_sha256": _sha256(target / "training_manifest.json"),
    }


def _verify_staged_suite(
    pending: Path,
    validation_workspace: Path | None,
    expected_freeze_identity: str,
) -> dict[str, Any]:
    try:
        config = load_hierarchical_mpc_config()
        case_order = list(config["case_order"])
        manifest = _read_json(pending / "freeze_manifest.json")
        manifest_version = manifest.get("schema_version")
        admission_mode = (
            str(manifest.get("admission_mode")) if manifest_version in {2, 3} else _STRICT_ADMISSION
        )
        versioned = manifest_version == 3
        legacy_degraded = manifest_version == 2 and admission_mode == _DEGRADED_ADMISSION
        legacy_payload_keys = (
            (
                "validation_workspace",
                "validation_failure_sha256",
                "validation_source_commit",
                "refit_verification_identity",
                "refit_workspace",
                "case_order",
                "model_identities",
                "validation_results",
                "source_bundles",
                "refit_candidate_gates",
                "admission_mode",
                "decision_reference",
                "original_failure_preserved",
            )
            if legacy_degraded
            else (
                "validation_workspace",
                "validation_completion_sha256",
                "validation_source_commit",
                "refit_verification_identity",
                "refit_workspace",
                "case_order",
                "model_identities",
                "validation_results",
                "source_bundles",
                "refit_candidate_gates",
            )
        )
        payload_keys = _V3_PAYLOAD_KEYS if versioned else legacy_payload_keys
        freeze_payload = {key: manifest.get(key) for key in payload_keys}
        recomputed_freeze_identity = _identity(freeze_payload)
        validation_cases = manifest.get("validation_results")
        validated_by_case = (
            {str(row.get("case")): row for row in validation_cases if isinstance(row, Mapping)}
            if isinstance(validation_cases, list)
            else {}
        )
        source_bundles = manifest.get("source_bundles")
        source_by_case = (
            {str(row.get("case")): row for row in source_bundles if isinstance(row, Mapping)}
            if isinstance(source_bundles, list)
            else {}
        )
        root = repository_root().resolve()
        refit_workspace = (root / str(manifest.get("refit_workspace", ""))).resolve()
        refit_workspace.relative_to((root / "outputs" / "baselines" / "mpc" / "refit").resolve())
        attested_validation_workspace = (
            root / str(manifest.get("validation_workspace", ""))
        ).resolve()
        attested_validation_workspace.relative_to(
            (root / "outputs" / "baselines" / "mpc" / "validation").resolve()
        )
        external_validation = True
        if validation_workspace is not None:
            validation = (
                verify_method_degraded_validation_workspace(validation_workspace)
                if legacy_degraded
                else verify_validation_workspace(validation_workspace)
            )
            terminal_name = "failure.json" if legacy_degraded else "completion.json"
            validation_terminal = _read_json(validation_workspace / terminal_name)
            validation_refit_identity = (
                validation.get("refit_verification_identity")
                if legacy_degraded
                else validation_terminal.get("refit_verification_identity")
            )
            validation_refit_workspace = (
                validation.get("refit_workspace")
                if legacy_degraded
                else validation_terminal.get("refit_workspace")
            )
            validation_plan = _read_json(validation_workspace / "resolved_plan.json")
            external_validation = bool(
                validation.get("valid") is True
                and validation_workspace.relative_to(root).as_posix()
                == manifest.get("validation_workspace")
                and _sha256(validation_workspace / terminal_name)
                == manifest.get(
                    "validation_terminal_sha256"
                    if versioned
                    else (
                        "validation_failure_sha256"
                        if legacy_degraded
                        else "validation_completion_sha256"
                    )
                )
                and (
                    not versioned
                    or manifest.get("validation_terminal_kind") == "completion"
                    and validation.get("publication") == _VERSIONED_PUBLICATION
                )
                and validation_terminal.get("validation_source_commit")
                == manifest.get("validation_source_commit")
                and validation_refit_identity == manifest.get("refit_verification_identity")
                and validation_refit_workspace == manifest.get("refit_workspace")
                and validation.get("cases") == validation_cases
                and validation_plan.get("candidate_gates") == manifest.get("refit_candidate_gates")
            )
        files = {
            path.relative_to(pending).as_posix() for path in pending.rglob("*") if path.is_file()
        }
        directories = {
            path.relative_to(pending).as_posix() for path in pending.rglob("*") if path.is_dir()
        }
        no_reparse_entries = not _is_reparse(pending) and all(
            not _is_reparse(path) for path in pending.rglob("*")
        )
        recorded_cases = manifest.get("cases")
        by_case = (
            {str(row.get("case")): row for row in recorded_cases if isinstance(row, Mapping)}
            if isinstance(recorded_cases, list)
            else {}
        )
        case_checks: dict[str, bool] = {}
        expected_case_rows: list[dict[str, Any]] = []
        for case in case_order:
            row = by_case.get(case, {})
            target = pending / case
            model = FittedArxModel.load(target / "model_coefficients.npz")
            card = _read_json(target / "model_card.json")
            training = _read_json(target / "training_manifest.json")
            provenance = card.get("provenance")
            provenance = provenance if isinstance(provenance, Mapping) else {}
            validated = validated_by_case.get(case, {})
            source_bundle = source_by_case.get(case, {})
            fresh_validation = training.get("fresh_validation")
            fresh_validation = fresh_validation if isinstance(fresh_validation, Mapping) else {}
            release_path = f"models/mpc_releases/{expected_freeze_identity}" if versioned else None
            expected_card = _frozen_model_card(
                case=case,
                model=model,
                validation_result=validated,
                validation_source_commit=str(manifest.get("validation_source_commit", "")),
                freeze_identity=expected_freeze_identity,
                refit_workspace=refit_workspace,
                validation_workspace=attested_validation_workspace,
                source_bundle=source_bundle,
                admission_mode=admission_mode,
                release_path=release_path,
            )
            expected_training = _frozen_training_manifest(
                case=case,
                model=model,
                validation_result=validated,
                validation_source_commit=str(manifest.get("validation_source_commit", "")),
                freeze_identity=expected_freeze_identity,
                refit_workspace=refit_workspace,
                validation_workspace=attested_validation_workspace,
                source_bundle=source_bundle,
                admission_mode=admission_mode,
                release_path=release_path,
            )
            expected_case_rows.append(
                {
                    "case": case,
                    "model_identity": model.identity,
                    "coefficient_sha256": _sha256(target / "model_coefficients.npz"),
                    "model_card_sha256": _sha256(target / "model_card.json"),
                    "training_manifest_sha256": _sha256(target / "training_manifest.json"),
                }
            )
            legacy_expected_card = {
                **expected_card,
                "model_structure": "vector_arx_four_lag_four_step",
            }
            card_matches = card == expected_card or (
                manifest_version == 2 and card == legacy_expected_card
            )
            fallback_count = validated.get("fallback_count")
            degraded_case_admitted = bool(
                admission_mode in _DEGRADED_ADMISSIONS
                and validated.get("evidence_valid") is True
                and isinstance(validated.get("checks"), Mapping)
                and all(validated["checks"].values())
                and isinstance(fallback_count, int)
                and not isinstance(fallback_count, bool)
                and fallback_count >= 0
                and validated.get("eligible")
                is (
                    fallback_count == 0
                    and (
                        validated.get("comfort_target_met") is True
                        or admission_mode == _DEGRADED_ADMISSION
                    )
                )
            )
            strict_case_admitted = bool(
                admission_mode == _STRICT_ADMISSION
                and validated.get("eligible") is True
                and fallback_count == 0
                and (not versioned or validated.get("comfort_target_met") is True)
            )
            case_checks[case] = (
                model.identity == expected_model_identity(model)
                and card_matches
                and training == expected_training
                and validated.get("model_identity") == model.identity
                and (strict_case_admitted or degraded_case_admitted)
                and isinstance(validated.get("reward"), (int, float))
                and not isinstance(validated.get("reward"), bool)
                and _comfort_target_attestation_valid(validated)
                and row.get("model_identity") == model.identity
                and row.get("coefficient_sha256") == _sha256(target / "model_coefficients.npz")
                and row.get("model_card_sha256") == _sha256(target / "model_card.json")
                and row.get("training_manifest_sha256")
                == _sha256(target / "training_manifest.json")
                and card.get("schema") == "h3c_hierarchical_mpc_model_card"
                and card.get("schema_version") == (3 if versioned else 2)
                and card.get("case") == case
                and card.get("model_identity") == model.identity
                and card.get("physical_validation")
                == (
                    "fresh_validation_passed"
                    if admission_mode == _STRICT_ADMISSION
                    else "fresh_validation_method_degraded"
                )
                and card.get("freeze_identity") == expected_freeze_identity
                and card.get("source_commit") == manifest.get("validation_source_commit")
                and card.get("validation") == validated
                and card.get("robust_calibration_attestation")
                == source_bundle.get("robust_calibration_attestation")
                and training.get("robust_calibration_attestation")
                == source_bundle.get("robust_calibration_attestation")
                and _robust_attestation_valid(
                    source_bundle.get("robust_calibration_attestation"), model
                )
                and provenance.get("refit_workspace")
                == refit_workspace.relative_to(root).as_posix()
                and provenance.get("validation_workspace") == manifest.get("validation_workspace")
                and provenance.get("source_coefficient_sha256")
                == source_bundle.get("coefficient_sha256")
                == _sha256(target / "model_coefficients.npz")
                and provenance.get("refit_model_card_sha256")
                == source_bundle.get("refit_model_card_sha256")
                and provenance.get("refit_candidate_report_sha256")
                == source_bundle.get("refit_candidate_report_sha256")
                and training.get("schema") == "h3c_hierarchical_mpc_training_manifest"
                and training.get("schema_version") == (3 if versioned else 2)
                and training.get("case") == case
                and training.get("model_identity") == model.identity
                and training.get("freeze_identity") == expected_freeze_identity
                and training.get("source_commit") == manifest.get("validation_source_commit")
                and training.get("validation_attestation") == validated
                and training.get("source_bundle") == source_bundle
                and fresh_validation.get("test_id") == validated.get("test_id")
                and fresh_validation.get("fallback_count") == fallback_count
                and fresh_validation.get("steps") == 668
                and _comfort_target_attestation_valid(fresh_validation)
                and source_bundle.get("case") == case
                and source_bundle.get("model_identity") == model.identity
                and (
                    not versioned
                    or card.get("release_path") == training.get("release_path") == release_path
                )
            )
        model_identities = manifest.get("model_identities")
        candidate_gates = manifest.get("refit_candidate_gates")
        candidate_gate_by_case = (
            {str(row.get("case")): row for row in candidate_gates if isinstance(row, Mapping)}
            if isinstance(candidate_gates, list)
            else {}
        )
        checks = {
            "external_validation": external_validation,
            "manifest_schema": manifest.get("schema") == "h3c_hierarchical_mpc_freeze_manifest"
            and (
                (manifest_version == 1 and admission_mode == _STRICT_ADMISSION)
                or (
                    manifest_version == 2
                    and admission_mode == _DEGRADED_ADMISSION
                    and manifest.get("decision_reference")
                    == (
                        "docs/hierarchical_mpc_formal_evaluation_post_result_"
                        "preregistration_20260831.md"
                    )
                    and manifest.get("original_failure_preserved") is True
                )
                or (
                    manifest_version == 3
                    and admission_mode in {_STRICT_ADMISSION, _PREREGISTERED_DEGRADED_ADMISSION}
                    and set(manifest) == _V3_MANIFEST_KEYS
                    and manifest.get("decision_reference") == _PREREGISTERED_DEGRADED_DECISION
                    and manifest.get("validation_terminal_kind") == "completion"
                )
            )
            and manifest.get("publication")
            == (_VERSIONED_PUBLICATION_VALUE if versioned else "single_directory_replace"),
            "freeze_identity": manifest.get("freeze_identity")
            == expected_freeze_identity
            == recomputed_freeze_identity,
            "case_order": manifest.get("case_order") == case_order,
            "validation_attestation": isinstance(validation_cases, list)
            and [row.get("case") for row in validation_cases if isinstance(row, Mapping)]
            == case_order
            and set(validated_by_case) == set(case_order),
            "source_attestation": isinstance(source_bundles, list)
            and [row.get("case") for row in source_bundles if isinstance(row, Mapping)]
            == case_order
            and set(source_by_case) == set(case_order),
            "model_identities": isinstance(model_identities, Mapping)
            and dict(model_identities)
            == {case: validated_by_case[case].get("model_identity") for case in case_order},
            "recomputed_persistence_attestation": isinstance(candidate_gates, list)
            and [row.get("case") for row in candidate_gates if isinstance(row, Mapping)]
            == case_order
            and all(
                candidate_gate_by_case[case].get("valid") is True
                and candidate_gate_by_case[case].get("model_identity")
                == validated_by_case[case].get("model_identity")
                and isinstance(candidate_gate_by_case[case].get("checks"), Mapping)
                and candidate_gate_by_case[case]["checks"].get("recomputed_persistence_gate")
                is True
                and candidate_gate_by_case[case]["checks"].get("source_prediction_finite") is True
                and candidate_gate_by_case[case]["checks"].get("source_beats_persistence") is True
                for case in case_order
            ),
            "source_identity": isinstance(manifest.get("validation_source_commit"), str)
            and len(str(manifest["validation_source_commit"])) == 40
            and isinstance(
                manifest.get(
                    "validation_terminal_sha256"
                    if versioned
                    else (
                        "validation_failure_sha256"
                        if legacy_degraded
                        else "validation_completion_sha256"
                    )
                ),
                str,
            )
            and len(
                str(
                    manifest.get(
                        "validation_terminal_sha256"
                        if versioned
                        else (
                            "validation_failure_sha256"
                            if legacy_degraded
                            else "validation_completion_sha256"
                        )
                    )
                )
            )
            == 64
            and isinstance(manifest.get("refit_verification_identity"), str)
            and len(str(manifest["refit_verification_identity"])) == 64,
            "admission_policy": (
                admission_mode == _STRICT_ADMISSION
                and all(
                    row.get("eligible") is True
                    and (not versioned or row.get("comfort_target_met") is True)
                    for row in validated_by_case.values()
                )
                or (
                    admission_mode == _DEGRADED_ADMISSION
                    and sum(int(row.get("fallback_count", 0)) for row in validated_by_case.values())
                    > 0
                )
                or (
                    admission_mode == _PREREGISTERED_DEGRADED_ADMISSION
                    and any(_method_degraded(row) for row in validated_by_case.values())
                )
            ),
            "case_rows": (
                isinstance(recorded_cases, list)
                and len(recorded_cases) == len(case_order)
                and all(
                    isinstance(row, Mapping) and set(row) == _CASE_ROW_KEYS
                    for row in recorded_cases
                )
                and recorded_cases == expected_case_rows
                if versioned
                else set(by_case) == set(case_order)
            ),
            "exact_file_set": files == _expected_pending_files(case_order),
            "exact_directory_set": directories == set(case_order),
            "no_reparse_entries": no_reparse_entries,
            "all_cases": all(case_checks.values()),
            "secret_scan": manifest.get("secret_exposure_count") == 0
            and secret_occurrences(pending) == 0,
        }
        return {
            "schema": "h3c_hierarchical_mpc_staged_suite_verification",
            "schema_version": 1,
            "checks": checks,
            "case_checks": case_checks,
            "valid": all(checks.values()),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "schema": "h3c_hierarchical_mpc_staged_suite_verification",
            "schema_version": 1,
            "error": f"{type(error).__name__}: {error}",
            "valid": False,
        }


def _publish_mpc_suite(
    *,
    workspace: Path,
    verification: Mapping[str, Any],
    terminal: Mapping[str, Any],
    admission_mode: str,
    publication: str = _DEFAULT_PUBLICATION,
) -> dict[str, Any]:
    if publication not in {_DEFAULT_PUBLICATION, _VERSIONED_PUBLICATION}:
        raise ValueError("unknown MPC publication mode")
    versioned = publication == _VERSIONED_PUBLICATION
    if versioned and admission_mode not in {
        _STRICT_ADMISSION,
        _PREREGISTERED_DEGRADED_ADMISSION,
    }:
        raise ValueError("versioned publication requires the preregistered admission policy")
    if not versioned and admission_mode not in {_STRICT_ADMISSION, _DEGRADED_ADMISSION}:
        raise ValueError("default publication requires the legacy admission policy")
    root = repository_root().resolve()
    validation_plan = _read_json(workspace / "resolved_plan.json")
    legacy_degraded = admission_mode == _DEGRADED_ADMISSION
    refit_workspace_value = (
        verification.get("refit_workspace") if legacy_degraded else terminal.get("refit_workspace")
    )
    refit_identity = (
        verification.get("refit_verification_identity")
        if legacy_degraded
        else terminal.get("refit_verification_identity")
    )
    refit_workspace = (root / str(refit_workspace_value)).resolve()
    refit_workspace.relative_to((root / "outputs" / "baselines" / "mpc" / "refit").resolve())
    config = load_hierarchical_mpc_config()
    case_order = list(config["case_order"])
    validation_cases = verification.get("cases")
    by_case = (
        {str(row.get("case")): row for row in validation_cases if isinstance(row, Mapping)}
        if isinstance(validation_cases, list)
        else {}
    )
    strict_ready = set(by_case) == set(case_order) and all(
        row.get("eligible") is True
        and _fallback_count(row) == 0
        and _comfort_target_attestation_valid(row)
        and (not versioned or _preregistered_case_eligible(row))
        for row in by_case.values()
    )
    legacy_degraded_ready = (
        set(by_case) == set(case_order)
        and all(
            row.get("evidence_valid") is True
            and isinstance(row.get("checks"), Mapping)
            and all(row["checks"].values())
            and _fallback_count(row) is not None
            and row.get("eligible") is (_fallback_count(row) == 0)
            and _comfort_target_attestation_valid(row)
            for row in by_case.values()
        )
        and sum(_fallback_count(row) or 0 for row in by_case.values()) > 0
    )
    preregistered_degraded_ready = (
        set(by_case) == set(case_order)
        and all(
            row.get("evidence_valid") is True
            and isinstance(row.get("checks"), Mapping)
            and all(row["checks"].values())
            and _fallback_count(row) is not None
            and row.get("eligible")
            is (_fallback_count(row) == 0 and row.get("comfort_target_met") is True)
            and _comfort_target_attestation_valid(row)
            for row in by_case.values()
        )
        and any(_method_degraded(row) for row in by_case.values())
    )
    if not (
        (admission_mode == _STRICT_ADMISSION and strict_ready)
        or (admission_mode == _DEGRADED_ADMISSION and legacy_degraded_ready)
        or (admission_mode == _PREREGISTERED_DEGRADED_ADMISSION and preregistered_degraded_ready)
    ):
        if admission_mode == _STRICT_ADMISSION:
            raise ValueError("all three validation cases must be eligible before freeze")
        raise ValueError("validation cases do not satisfy method-degraded admission")
    source_bundles = [_source_bundle(refit_workspace, case) for case in case_order]
    source_by_case = {str(row["case"]): row for row in source_bundles}
    models_root = _models_root(root)
    common_payload: dict[str, Any] = {
        "validation_workspace": workspace.relative_to(root).as_posix(),
        "validation_source_commit": terminal["validation_source_commit"],
        "refit_verification_identity": refit_identity,
        "refit_workspace": refit_workspace.relative_to(root).as_posix(),
        "case_order": case_order,
        "model_identities": {case: by_case[case]["model_identity"] for case in case_order},
        "validation_results": [dict(by_case[case]) for case in case_order],
        "source_bundles": source_bundles,
        "refit_candidate_gates": validation_plan["candidate_gates"],
    }
    if versioned:
        freeze_payload = {
            "publication": _VERSIONED_PUBLICATION_VALUE,
            "admission_mode": admission_mode,
            "decision_reference": _PREREGISTERED_DEGRADED_DECISION,
            "validation_terminal_kind": "completion",
            "validation_terminal_sha256": _sha256(workspace / "completion.json"),
            **common_payload,
        }
    elif legacy_degraded:
        freeze_payload = dict(common_payload)
        freeze_payload.update(
            {
                "validation_failure_sha256": _sha256(workspace / "failure.json"),
                "admission_mode": admission_mode,
                "decision_reference": (_LEGACY_DEGRADED_DECISION),
                "original_failure_preserved": True,
            }
        )
    else:
        freeze_payload = dict(common_payload)
        freeze_payload["validation_completion_sha256"] = _sha256(workspace / "completion.json")
    freeze_identity = _identity(freeze_payload)
    if versioned:
        releases_root = _mpc_releases_root(root, create=True)
        target = releases_root / freeze_identity
        pending = releases_root / f".{freeze_identity}.pending"
        release_path = target.relative_to(root).as_posix()
    else:
        releases_root = None
        target = models_root / "mpc"
        pending = models_root / f".mpc-{freeze_identity[:12]}.pending"
        release_path = None
    if _path_present(target):
        raise ValueError("registered MPC model target already exists; overwrite is forbidden")
    if _path_present(pending):
        raise ValueError("registered MPC pending suite already exists; overwrite is forbidden")
    pending.mkdir(parents=True, exist_ok=False)
    case_rows = [
        _stage_case(
            case=case,
            pending=pending,
            refit_workspace=refit_workspace,
            validation_workspace=workspace,
            validation_result=by_case[case],
            validation_source_commit=str(terminal["validation_source_commit"]),
            freeze_identity=freeze_identity,
            source_bundle=source_by_case[case],
            admission_mode=admission_mode,
            release_path=release_path,
        )
        for case in case_order
    ]
    _write_json(
        pending / "freeze_manifest.json",
        {
            "schema": "h3c_hierarchical_mpc_freeze_manifest",
            "schema_version": (
                3 if versioned else (2 if admission_mode == _DEGRADED_ADMISSION else 1)
            ),
            "freeze_identity": freeze_identity,
            **freeze_payload,
            "cases": case_rows,
            "publication": (
                _VERSIONED_PUBLICATION_VALUE if versioned else "single_directory_replace"
            ),
            "secret_exposure_count": secret_occurrences(pending),
        },
    )
    staged = _verify_staged_suite(pending, workspace, freeze_identity)
    if staged.get("valid") is not True:
        raise ValueError("staged MPC suite verification failed; pending evidence was preserved")
    with _publication_lock(models_root / ".mpc-publication.lock"):
        if versioned and _mpc_releases_root(root, create=False) != releases_root:
            raise ValueError("registered MPC releases root changed before publication")
        if _path_present(target):
            raise ValueError("registered MPC model target appeared before publication")
        if not pending.is_dir() or _is_reparse(pending) or pending.resolve() != pending:
            raise ValueError("staged MPC suite changed before publication")
        locked_staged = _verify_staged_suite(pending, workspace, freeze_identity)
        if locked_staged.get("valid") is not True:
            raise ValueError(
                "staged MPC suite changed before publication; pending evidence was preserved"
            )
        os.replace(pending, target)
        published = verify_frozen_mpc_suite(target)
        if published.get("valid") is not True:
            raise ValueError(
                "published MPC suite verification failed; target evidence was preserved"
            )
    return {
        "schema": "h3c_hierarchical_mpc_freeze",
        "schema_version": 1,
        "admission_mode": admission_mode,
        "classification": (
            "METHOD-DEGRADED" if admission_mode in _DEGRADED_ADMISSIONS else "BASELINE-READY"
        ),
        "freeze_identity": freeze_identity,
        "target": str(target),
        "cases": case_rows,
        "publication": (_VERSIONED_PUBLICATION_VALUE if versioned else "single_directory_replace"),
        "valid": True,
    }


def freeze_validated_mpc_suite(
    validation_workspace: Path,
    *,
    publication: str = _DEFAULT_PUBLICATION,
) -> dict[str, Any]:
    workspace, verification = _resolve_validation(validation_workspace)
    if publication == _VERSIONED_PUBLICATION and not (
        verification.get("publication") == _VERSIONED_PUBLICATION
        and verification.get("classification") == "BASELINE-READY"
    ):
        raise ValueError("versioned strict publication requires BASELINE-READY validation")
    return _publish_mpc_suite(
        workspace=workspace,
        verification=verification,
        terminal=_read_json(workspace / "completion.json"),
        admission_mode=_STRICT_ADMISSION,
        publication=publication,
    )


def freeze_method_degraded_mpc_suite(
    validation_workspace: Path,
    *,
    publication: str = _DEFAULT_PUBLICATION,
) -> dict[str, Any]:
    admission_mode = (
        _PREREGISTERED_DEGRADED_ADMISSION
        if publication == _VERSIONED_PUBLICATION
        else _DEGRADED_ADMISSION
    )
    verification = (
        verify_validation_workspace(validation_workspace)
        if publication == _VERSIONED_PUBLICATION
        else verify_method_degraded_validation_workspace(validation_workspace)
    )
    if verification.get("valid") is not True:
        raise ValueError("validation failure is not eligible for method-degraded admission")
    if publication == _VERSIONED_PUBLICATION and not (
        verification.get("publication") == _VERSIONED_PUBLICATION
        and verification.get("classification") == "METHOD-DEGRADED"
    ):
        raise ValueError("versioned degraded publication requires METHOD-DEGRADED validation")
    workspace = Path(str(verification["workspace"])).resolve()
    return _publish_mpc_suite(
        workspace=workspace,
        verification=verification,
        terminal=_read_json(
            workspace
            / ("completion.json" if publication == _VERSIONED_PUBLICATION else "failure.json")
        ),
        admission_mode=admission_mode,
        publication=publication,
    )


def verify_frozen_mpc_suite(target: Path | None = None) -> dict[str, Any]:
    root = repository_root().resolve()
    requested = target or root / "models" / "mpc"
    resolved = requested
    try:
        if ".." in requested.parts:
            raise ValueError("frozen MPC suite path must be canonical")
        if not requested.is_absolute():
            requested = root / requested
        lexical = requested.absolute()
        models = _models_root(root)
        lexical.relative_to(models)
        if not _path_present(lexical) or not lexical.is_dir():
            raise ValueError("frozen MPC suite must be one real directory")
        relative_parts = lexical.relative_to(models).parts
        cursor = models
        for part in relative_parts:
            cursor = cursor / part
            if _is_reparse(cursor):
                raise ValueError("frozen MPC suite path must not contain links or reparse points")
        resolved = lexical.resolve(strict=True)
        if lexical != resolved:
            raise ValueError("frozen MPC suite path must be canonical")
        manifest = _read_json(resolved / "freeze_manifest.json")
        manifest_version = manifest.get("schema_version")
        if manifest_version in {1, 2}:
            published_location = resolved == (root / "models" / "mpc").resolve()
        elif manifest_version == 3:
            freeze_identity = manifest.get("freeze_identity")
            releases_root = _mpc_releases_root(root, create=False)
            published_location = bool(
                isinstance(freeze_identity, str)
                and len(freeze_identity) == 64
                and all(character in "0123456789abcdef" for character in freeze_identity)
                and resolved == releases_root / freeze_identity
            )
        else:
            published_location = False
        result = _verify_staged_suite(
            resolved,
            None,
            str(manifest.get("freeze_identity", "")),
        )
        checks = {**result.get("checks", {}), "published_location": published_location}
        return {
            **result,
            "checks": checks,
            "valid": result.get("valid") is True and published_location,
            "target": str(resolved),
            "admission_mode": (manifest.get("admission_mode", _STRICT_ADMISSION)),
            "classification": (
                "METHOD-DEGRADED"
                if manifest.get("admission_mode") in _DEGRADED_ADMISSIONS
                else "BASELINE-READY"
            ),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "schema": "h3c_hierarchical_mpc_staged_suite_verification",
            "schema_version": 1,
            "target": str(resolved),
            "error": f"{type(error).__name__}: {error}",
            "valid": False,
        }


def frozen_mpc_validation_test_identities(target: Path) -> dict[str, str]:
    """Return the release-bound fresh-validation test IDs, failing closed."""

    verification = verify_frozen_mpc_suite(target)
    if verification.get("valid") is not True:
        raise ValueError("frozen MPC suite verification failed")
    release = Path(str(verification["target"]))
    manifest = _read_json(release / "freeze_manifest.json")
    case_order = manifest.get("case_order")
    if not isinstance(case_order, list) or not case_order:
        raise ValueError("frozen MPC suite case order is invalid")
    identities: dict[str, str] = {}
    for case in case_order:
        if not isinstance(case, str):
            raise ValueError("frozen MPC suite case identity is invalid")
        training = _read_json(release / case / "training_manifest.json")
        fresh_validation = training.get("fresh_validation")
        test_id = fresh_validation.get("test_id") if isinstance(fresh_validation, Mapping) else None
        if not isinstance(test_id, str) or not test_id or test_id in identities.values():
            raise ValueError("frozen MPC validation test identity is invalid or reused")
        identities[case] = test_id
    return identities
