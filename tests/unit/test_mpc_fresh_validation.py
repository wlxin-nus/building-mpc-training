from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, Lock
from typing import Any

import numpy as np
import pytest

import h3c_baselines.cli as baseline_cli
import h3c_baselines.mpc.training as training
import h3c_baselines.mpc.validation as validation
from h3c.runtime.comfort import ComfortModel, step_reward


def _configuration() -> dict[str, Any]:
    return {
        "case_order": ["CaseA", "CaseB", "CaseC"],
        "excitation": {
            "occupied_bounds_c": [23.5, 26.5],
            "unoccupied_bounds_c": [20.0, 30.0],
        },
    }


def _profile() -> dict[str, Any]:
    return {
        "testcase": "case",
        "evaluation_start_day": 10,
        "zones": {"zone": {}},
        "protocol": {"initial_setpoint_c": 25.0},
        "comfort": {
            "air_velocity_m_s": 0.1,
            "relative_humidity_percent": 50.0,
            "metabolic_rate": 1.1,
            "dynamic_clothing": False,
            "summer_clothing_insulation": 0.5,
            "winter_clothing_insulation": 1.0,
            "clothing_transition_low_c": 10.0,
            "clothing_transition_high_c": 20.0,
        },
        "objective": {
            "energy_weight": 1.0,
            "energy_scale": 1.0,
            "comfort_weight": 1.0,
            "comfort_scale": 1.0,
            "smoothness_weight": 1.0,
            "smoothness_scale": 1.0,
        },
    }


def _step_row(
    step: int,
    *,
    degraded: bool = False,
    outcome_temperature_c: float = 24.8,
) -> dict[str, Any]:
    price = 0.2
    power = 1000.0
    cost = power * 0.25 / 1000.0 * price
    previous = 25.0 if step == 0 else 24.5
    action_temperature = 25.0 if step == 0 else 24.8
    comfort = ComfortModel(_profile()["comfort"])
    comfort.update_clothing((10 - 7) * 86400 + step * 900, 20.0)
    action_pmv = comfort.pmv(action_temperature)
    outcome_pmv = comfort.pmv(outcome_temperature_c)
    reward = step_reward(
        cost=cost,
        pmv=[outcome_pmv],
        occupancy=[1.0],
        setpoints_c=[24.5],
        previous_setpoints_c=[previous],
        objective=_profile()["objective"],
    )
    start = (10 - 7) * 86400
    return {
        "schema": "h3c_hierarchical_mpc_validation_step",
        "schema_version": 1,
        "case": "CaseA",
        "step": step,
        "test_id": "fresh-test",
        "model_identity": "model-id",
        "action_time_seconds": start + step * 900,
        "outcome_time_seconds": start + (step + 1) * 900,
        "occupancy": {"zone": 1.0},
        "daily_outdoor_mean_c": 20.0,
        "clothing_insulation": 0.5,
        "action_zone_temperature_c": {"zone": action_temperature},
        "action_pmv": {"zone": action_pmv},
        "setpoints_c": {"zone": 24.5},
        "electricity_price": price,
        "outcome_site_power_w": power,
        "step_cost": cost,
        "outcome_zone_temperature_c": {"zone": outcome_temperature_c},
        "outcome_pmv": {"zone": outcome_pmv},
        "step_reward": reward,
        "controller_diagnostics": {
            "status": "fallback" if degraded else "optimized",
            "method_degraded": degraded,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _lifecycle() -> list[dict[str, Any]]:
    events: list[tuple[str, str | None]] = [
        ("selected", None),
        ("status_changed", "Running"),
        ("configured", "Running"),
        ("status_changed", "Running"),
        ("initialized", "Running"),
        ("stopped", None),
    ]
    return [
        {
            "sequence": index,
            "case": "CaseA",
            "phase": "physical_dispatch",
            "event": event,
            "dispatch_mode": "auto",
            "test_id": "fresh-test",
            "testcase": "case",
            **({"status": status} if status is not None else {}),
        }
        for index, (event, status) in enumerate(events)
    ]


def _write_episode_artifacts(path: Path, rows: list[dict[str, Any]]) -> None:
    episode = path / "episodes" / "validation-000-lane-0"
    episode.mkdir(parents=True)
    start = (10 - 7) * 86400
    outputs = [
        [float(rows[0]["action_zone_temperature_c"]["zone"]), 1000.0],
        *[[float(row["outcome_zone_temperature_c"]["zone"]), 1000.0] for row in rows],
    ]
    controls = [[float(row["setpoints_c"]["zone"])] for row in rows]
    controls.append(controls[-1].copy())
    disturbances = []
    for row in rows:
        day_fraction = (int(row["action_time_seconds"]) % 86400) / 86400.0
        disturbances.append(
            [
                30.0,
                0.0,
                float(row["occupancy"]["zone"]),
                np.sin(2.0 * np.pi * day_fraction),
                np.cos(2.0 * np.pi * day_fraction),
            ]
        )
    disturbances.append(disturbances[-1].copy())
    np.savez_compressed(
        episode / "trajectory.npz",
        times=np.asarray([start, start + 900, start + 1800], dtype=np.int64),
        outputs=np.asarray(outputs, dtype=np.float64),
        controls=np.asarray(controls, dtype=np.float64),
        disturbances=np.asarray(disturbances, dtype=np.float64),
    )
    _write_jsonl(path / "step_diagnostics.jsonl", rows)
    _write_jsonl(path / "lifecycle.jsonl", _lifecycle())
    (episode / "manifest.json").write_text(
        json.dumps(
            {
                "role": "validation",
                "episode": 0,
                "lane": 0,
                "test_id": "fresh-test",
                "model_identity": "model-id",
                "start_time_seconds": start,
                "warmup_period_seconds": 7 * 86400,
                "steps": 2,
                "reward": sum(float(row["step_reward"]) for row in rows),
                "peak_occupied_absolute_pmv": max(
                    abs(float(row["outcome_pmv"]["zone"])) for row in rows
                ),
                "fallback_count": sum(
                    row["controller_diagnostics"]["method_degraded"] is True for row in rows
                ),
                "recovery_step_count": 0,
                "forecast_phase": "mpc_fresh_validation",
            }
        ),
        encoding="utf-8",
    )


def _refresh_rewards(rows: list[dict[str, Any]]) -> None:
    previous = 25.0
    for row in rows:
        setpoint = float(row["setpoints_c"]["zone"])
        row["step_reward"] = step_reward(
            cost=float(row["step_cost"]),
            pmv=[float(row["outcome_pmv"]["zone"])],
            occupancy=[float(row["occupancy"]["zone"])],
            setpoints_c=[setpoint],
            previous_setpoints_c=[previous],
            objective=_profile()["objective"],
        )
        previous = setpoint


def test_arm_verifier_recomputes_timeline_reward_fallback_and_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "VALIDATION_STEPS", 2)
    monkeypatch.setattr(validation, "load_profile", lambda _case: _profile())
    _write_episode_artifacts(tmp_path, [_step_row(0), _step_row(1)])

    result = validation._recompute_arm(
        case="CaseA",
        arm_dir=tmp_path,
        model_identity="model-id",
        expected_test_id="fresh-test",
    )

    assert result["evidence_valid"] is True
    assert result["eligible"] is True
    assert result["comfort_target_met"] is True
    assert result["fallback_count"] == 0
    assert result["occupied_peak_absolute_pmv"] == pytest.approx(
        abs(ComfortModel(_profile()["comfort"]).pmv(24.8))
    )

    rows = [_step_row(0), _step_row(1)]
    rows[1]["step_reward"] += 1.0
    _write_jsonl(tmp_path / "tampered.jsonl", rows)
    (tmp_path / "step_diagnostics.jsonl").write_text(
        (tmp_path / "tampered.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    tampered = validation._recompute_arm(
        case="CaseA",
        arm_dir=tmp_path,
        model_identity="model-id",
        expected_test_id="fresh-test",
    )
    assert tampered["evidence_valid"] is False
    assert tampered["eligible"] is False


def test_completed_fallback_is_adverse_while_peak_is_reported_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "VALIDATION_STEPS", 2)
    monkeypatch.setattr(validation, "load_profile", lambda _case: _profile())
    _write_episode_artifacts(
        tmp_path,
        [_step_row(0, degraded=True), _step_row(1, outcome_temperature_c=28.0)],
    )

    result = validation._recompute_arm(
        case="CaseA",
        arm_dir=tmp_path,
        model_identity="model-id",
        expected_test_id="fresh-test",
    )

    assert result["evidence_valid"] is True
    assert result["fallback_count"] == 1
    assert result["occupied_peak_absolute_pmv"] > 0.70
    assert result["comfort_target_met"] is False
    assert result["eligible"] is False


def test_peak_only_miss_is_performance_evidence_not_a_freeze_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "VALIDATION_STEPS", 2)
    monkeypatch.setattr(validation, "load_profile", lambda _case: _profile())
    _write_episode_artifacts(
        tmp_path,
        [_step_row(0), _step_row(1, outcome_temperature_c=28.0)],
    )

    result = validation._recompute_arm(
        case="CaseA",
        arm_dir=tmp_path,
        model_identity="model-id",
        expected_test_id="fresh-test",
    )

    assert result["evidence_valid"] is True
    assert result["fallback_count"] == 0
    assert result["occupied_peak_absolute_pmv"] > 0.70
    assert result["comfort_target_met"] is False
    assert result["eligible"] is True


def test_versioned_peak_only_miss_is_preregistered_method_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "VALIDATION_STEPS", 2)
    monkeypatch.setattr(validation, "load_profile", lambda _case: _profile())
    _write_episode_artifacts(
        tmp_path,
        [_step_row(0), _step_row(1, outcome_temperature_c=28.0)],
    )

    result = validation._recompute_arm(
        case="CaseA",
        arm_dir=tmp_path,
        model_identity="model-id",
        expected_test_id="fresh-test",
        require_comfort=True,
    )

    assert result["evidence_valid"] is True
    assert result["fallback_count"] == 0
    assert result["comfort_target_met"] is False
    assert result["eligible"] is False
    assert result["validation_classification"] == "METHOD-DEGRADED"


def test_versioned_invalid_evidence_is_run_invalid_not_method_degraded() -> None:
    assert (
        validation._validation_classification(
            {
                "evidence_valid": False,
                "fallback_count": 0,
                "comfort_target_met": True,
            }
        )
        == "RUN-INVALID"
    )


def test_versioned_validation_plan_does_not_reuse_or_block_on_legacy_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "outputs" / "baselines" / "mpc" / "refit" / "source"
    source.mkdir(parents=True)
    legacy = tmp_path / "models" / "mpc"
    legacy.mkdir(parents=True)
    (legacy / "historical.txt").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(validation, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(validation, "_resolve_refit_workspace", lambda _path: (source, {}))
    monkeypatch.setattr(validation, "load_hierarchical_mpc_config", _configuration)
    monkeypatch.setattr(
        validation,
        "_refit_case_gate",
        lambda _source, _verification, case: {
            "case": case,
            "model_identity": f"model-{case}",
            "valid": True,
        },
    )

    plan = validation.resolved_validation_plan(source, publication="versioned")

    assert plan["publication"] == "versioned"
    assert plan["schema_version"] == 2
    assert plan["physical_gate"]["occupied_peak_absolute_pmv_max"] == 0.70
    assert (legacy / "historical.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("fallback_setpoint_c", "expected_evidence_valid"),
    [(26.6, True), (30.1, False)],
)
def test_fallback_uses_physical_bounds_not_optimized_fit_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fallback_setpoint_c: float,
    expected_evidence_valid: bool,
) -> None:
    monkeypatch.setattr(validation, "VALIDATION_STEPS", 2)
    monkeypatch.setattr(validation, "load_profile", lambda _case: _profile())
    monkeypatch.setattr(validation, "load_hierarchical_mpc_config", _configuration)
    rows = [_step_row(0, degraded=True), _step_row(1)]
    rows[0]["setpoints_c"]["zone"] = fallback_setpoint_c
    _refresh_rewards(rows)
    _write_episode_artifacts(tmp_path, rows)

    result = validation._recompute_arm(
        case="CaseA",
        arm_dir=tmp_path,
        model_identity="model-id",
        expected_test_id="fresh-test",
    )

    assert result["evidence_valid"] is expected_evidence_valid
    if expected_evidence_valid:
        assert result["fallback_count"] == 1
    assert result["eligible"] is False


def test_arm_verifier_rejects_lifecycle_testcase_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "VALIDATION_STEPS", 2)
    monkeypatch.setattr(validation, "load_profile", lambda _case: _profile())
    _write_episode_artifacts(tmp_path, [_step_row(0), _step_row(1)])
    lifecycle_path = tmp_path / "lifecycle.jsonl"
    lifecycle = _lifecycle()
    lifecycle[2]["testcase"] = "wrong-testcase"
    _write_jsonl(lifecycle_path, lifecycle)

    result = validation._recompute_arm(
        case="CaseA",
        arm_dir=tmp_path,
        model_identity="model-id",
        expected_test_id="fresh-test",
    )

    assert result["checks"]["lifecycle"] is False
    assert result["eligible"] is False


@pytest.mark.parametrize("tamper", ["pmv", "negative_occupancy", "occupied_support"])
def test_arm_verifier_rejects_coordinated_semantic_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    monkeypatch.setattr(validation, "VALIDATION_STEPS", 2)
    monkeypatch.setattr(validation, "load_profile", lambda _case: _profile())
    monkeypatch.setattr(validation, "load_hierarchical_mpc_config", _configuration)
    rows = [_step_row(0), _step_row(1)]
    if tamper == "pmv":
        for row in rows:
            row["action_pmv"]["zone"] += 0.5
            row["outcome_pmv"]["zone"] += 0.5
    elif tamper == "negative_occupancy":
        for row in rows:
            row["occupancy"]["zone"] = -1.0
    else:
        for row in rows:
            row["setpoints_c"]["zone"] = 22.0
    _refresh_rewards(rows)
    _write_episode_artifacts(tmp_path, rows)

    result = validation._recompute_arm(
        case="CaseA",
        arm_dir=tmp_path,
        model_identity="model-id",
        expected_test_id="fresh-test",
    )

    assert result["evidence_valid"] is False
    assert result["eligible"] is False


def test_dynamic_lifecycle_client_is_required_before_physics() -> None:
    class OldClient:
        pass

    with pytest.raises(ValueError, match="dynamic BOPTEST lifecycle"):
        validation._install_lifecycle_sink(OldClient(), lambda _row: None)  # type: ignore[arg-type]


def test_episode_identity_change_after_initialize_stops_before_advance(tmp_path: Path) -> None:
    class IdentityChangingClient:
        test_id: str | None = "frozen-test"
        advance_count = 0

        def initialize_selected(
            self, _start_time_seconds: int, _warmup_period_seconds: int
        ) -> dict[str, Any]:
            self.test_id = "changed-test"
            return {"time": 0.0}

        def advance(self, _controls: dict[str, float]) -> dict[str, Any]:
            self.advance_count += 1
            return {}

    client = IdentityChangingClient()
    lane = training._Lane(0, client, "frozen-test")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="identity changed during MPC episode initialize"):
        training._collect_episode(
            lane=lane,
            profile={"evaluation_start_day": 10, "zones": {"zone": {}}},
            role="fit",
            episode=0,
            excitation_config={},
            output_dir=tmp_path,
        )

    assert client.advance_count == 0


def test_refit_preflight_failure_occurs_before_client_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validation,
        "resolved_validation_plan",
        lambda _workspace: (_ for _ in ()).throw(ValueError("refit preflight failed")),
    )

    with pytest.raises(ValueError, match="refit preflight failed"):
        validation.execute_fresh_validation(
            Path("invalid-refit"),
            endpoint="http://unused",
            physical_factory=lambda _endpoint: pytest.fail("physical client was selected"),
        )


def test_workspace_verifier_binds_arm_terminal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "VALIDATION_STEPS", 2)
    monkeypatch.setattr(validation, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(validation, "load_profile", lambda _case: _profile())
    configuration = {
        **_configuration(),
        "case_order": ["CaseA"],
    }
    monkeypatch.setattr(validation, "load_hierarchical_mpc_config", lambda: configuration)
    refit_verification = {
        "valid": True,
        "cases": [
            {
                "case": "CaseA",
                "model_identity": "model-id",
                "checks": {"source_persistence_gate": True},
                "valid": True,
            }
        ],
    }
    monkeypatch.setattr(
        validation,
        "verify_refit_workspace",
        lambda _workspace: refit_verification,
    )
    monkeypatch.setattr(
        validation,
        "_refit_case_gate",
        lambda *_args: {"case": "CaseA", "model_identity": "model-id", "valid": True},
    )
    monkeypatch.setattr(validation, "secret_occurrences", lambda _path: 0)
    source_commit = "a" * 40
    refit_relative = "outputs/baselines/mpc/refit/refit"
    run_dir = (
        tmp_path
        / "outputs"
        / "baselines"
        / "mpc"
        / "validation"
        / f"validation-{source_commit[:8]}"
    )
    arm_dir = run_dir / "CaseA"
    rows = [_step_row(0), _step_row(1)]
    _write_episode_artifacts(arm_dir, rows)
    (arm_dir / "execution_owner.json").write_text(
        json.dumps(
            {
                "schema": "h3c_hierarchical_mpc_validation_execution_owner",
                "schema_version": 1,
                "case": "CaseA",
                "validation_source_commit": source_commit,
                "pid": 1,
                "lock_scope": "independent_case_arm",
            }
        ),
        encoding="utf-8",
    )
    (arm_dir / "resolved_arm.json").write_text(
        json.dumps(
            {
                "schema": "h3c_hierarchical_mpc_validation_arm",
                "schema_version": 1,
                "case": "CaseA",
                "testcase": "case",
                "model_identity": "model-id",
                "validation_source_commit": source_commit,
                "refit_workspace": refit_relative,
                "warmup_period_seconds": 7 * 86400,
                "steps": 2,
                "dispatch_mode": "auto",
            }
        ),
        encoding="utf-8",
    )
    recomputed = validation._recompute_arm(
        case="CaseA",
        arm_dir=arm_dir,
        model_identity="model-id",
        expected_test_id="fresh-test",
    )
    arm_completion = {
        "schema": "h3c_hierarchical_mpc_validation_arm_completion",
        "schema_version": 1,
        **recomputed,
        "secret_exposure_count": 0,
        "artifact_sha256": validation._artifact_hashes(arm_dir),
    }
    arm_completion_path = arm_dir / "completion.json"
    arm_completion_path.write_text(json.dumps(arm_completion), encoding="utf-8")
    arm_result = {
        "status": "completed",
        "arm_completion_sha256": validation._sha256(arm_completion_path),
        **arm_completion,
    }
    evidence = {
        "schema": "h3c_hierarchical_mpc_validation_evidence",
        "schema_version": 1,
        "validation_source_commit": source_commit,
        "refit_workspace": refit_relative,
        "refit_verification_identity": validation._identity(refit_verification),
        "case_order": ["CaseA"],
        "submitted_case_count": 1,
        "awaited_case_count": 1,
        "future_cancel_count": 0,
        "future_completion_order": ["CaseA"],
        "cases": [arm_result],
        "model_api_calls": 0,
        "secret_exposure_count": 0,
    }
    (run_dir / "suite_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    dry_plan = {
        "schema": "h3c_hierarchical_mpc_fresh_validation_plan",
        "schema_version": 1,
        "execution": False,
        "refit_workspace": refit_relative,
        "refit_verification_identity": validation._identity(refit_verification),
        "case_order": ["CaseA"],
    }
    plan_identity = validation._identity(dry_plan)
    executed_plan = {
        **dry_plan,
        "execution": True,
        "validation_source_commit": source_commit,
        "plan_identity": plan_identity,
    }
    (run_dir / "resolved_plan.json").write_text(json.dumps(executed_plan), encoding="utf-8")
    completion = {
        "schema": "h3c_hierarchical_mpc_validation_completion",
        "schema_version": 1,
        "validation_source_commit": source_commit,
        "plan_identity": plan_identity,
        "refit_workspace": refit_relative,
        "refit_verification_identity": validation._identity(refit_verification),
        "case_order": ["CaseA"],
        "cases": [recomputed],
        "validation_evidence_sha256": validation._sha256(run_dir / "suite_evidence.json"),
        "candidate_promotion": False,
        "secret_exposure_count": 0,
    }
    (run_dir / "completion.json").write_text(json.dumps(completion), encoding="utf-8")

    assert validation.verify_validation_workspace(run_dir)["valid"] is True
    arm_completion_path.rename(arm_completion_path.with_suffix(".missing"))
    assert validation.verify_validation_workspace(run_dir)["valid"] is False


def _patch_suite_execution(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    execute_case: Any,
    *,
    suite_valid: bool,
) -> None:
    monkeypatch.setattr(validation, "repository_root", lambda: root)
    monkeypatch.setattr(validation, "load_hierarchical_mpc_config", _configuration)
    monkeypatch.setattr(validation, "committed_source_identity", lambda: "a" * 40)
    monkeypatch.setattr(
        validation,
        "resolved_validation_plan",
        lambda _workspace: {
            "execution": False,
            "schema": "h3c_hierarchical_mpc_fresh_validation_plan",
            "schema_version": 1,
            "refit_workspace": "outputs/baselines/mpc/refit/source",
            "case_order": _configuration()["case_order"],
            "plan_identity": "plan-id",
        },
    )
    monkeypatch.setattr(
        validation,
        "verify_refit_workspace",
        lambda _workspace: {"valid": True, "cases": []},
    )
    monkeypatch.setattr(validation, "_execute_case_validation", execute_case)
    monkeypatch.setattr(validation, "secret_occurrences", lambda _path: 0)
    monkeypatch.setattr(
        validation,
        "_suite_evidence_checks",
        lambda _path, evidence: (
            {"registered_suite": suite_valid},
            [
                {
                    "case": case,
                    "test_id": f"test-{case}",
                    "model_identity": f"model-{case}",
                    "steps": 668,
                    "reward": -1.0,
                    "fallback_count": 0,
                    "occupied_peak_absolute_pmv": 0.6,
                    "eligible": True,
                }
                for case in _configuration()["case_order"]
            ],
        ),
    )


def test_three_natural_case_futures_overlap_and_all_are_awaited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = Barrier(3)
    active = 0
    maximum_active = 0
    guard = Lock()

    def execute_case(**kwargs: Any) -> dict[str, Any]:
        nonlocal active, maximum_active
        case = str(kwargs["case"])
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait(timeout=2)
        with guard:
            active -= 1
        return {
            "case": case,
            "status": "completed",
            "test_id": f"test-{case}",
            "model_identity": f"model-{case}",
            "eligible": True,
        }

    _patch_suite_execution(monkeypatch, tmp_path, execute_case, suite_valid=True)

    result = validation.execute_fresh_validation(
        Path("unused"),
        endpoint="http://unused",
        physical_factory=lambda _endpoint: None,  # type: ignore[arg-type,return-value]
    )

    assert maximum_active == 3
    assert result["case_order"] == _configuration()["case_order"]
    run_dir = Path(result["run_dir"])
    evidence = json.loads((run_dir / "suite_evidence.json").read_text(encoding="utf-8"))
    assert evidence["submitted_case_count"] == 3
    assert evidence["awaited_case_count"] == 3
    assert evidence["future_cancel_count"] == 0


def test_one_failed_future_still_awaits_every_case_and_never_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = Barrier(3)
    finished: list[str] = []
    guard = Lock()

    def execute_case(**kwargs: Any) -> dict[str, Any]:
        case = str(kwargs["case"])
        barrier.wait(timeout=2)
        with guard:
            finished.append(case)
        if case == "CaseA":
            return {"case": case, "status": "failed", "eligible": False}
        return {"case": case, "status": "completed", "eligible": True}

    _patch_suite_execution(monkeypatch, tmp_path, execute_case, suite_valid=False)

    with pytest.raises(ValueError, match="no model was promoted"):
        validation.execute_fresh_validation(
            Path("unused"),
            endpoint="http://unused",
            physical_factory=lambda _endpoint: None,  # type: ignore[arg-type,return-value]
        )

    assert set(finished) == set(_configuration()["case_order"])
    runs = list((tmp_path / "outputs" / "baselines" / "mpc" / "validation").iterdir())
    runs = [path for path in runs if path.is_dir()]
    assert len(runs) == 1
    assert (runs[0] / "failure.json").is_file()
    assert not (runs[0] / "completion.json").exists()
    assert not (tmp_path / "models" / "mpc").exists()


def test_validation_cli_is_dry_without_execute(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        baseline_cli,
        "resolved_validation_plan",
        lambda source: {"execution": False, "refit_workspace": str(source)},
    )
    monkeypatch.setattr(
        baseline_cli,
        "execute_fresh_validation",
        lambda *_args, **_kwargs: pytest.fail("dry validation touched physics"),
    )
    monkeypatch.setattr(
        baseline_cli,
        "freeze_validated_mpc_suite",
        lambda *_args, **_kwargs: pytest.fail("dry validation promoted models"),
    )

    assert baseline_cli.main(["mpc", "validate", "--refit-workspace", "refit-workspace"]) == 0
    assert '"execution": false' in capsys.readouterr().out
