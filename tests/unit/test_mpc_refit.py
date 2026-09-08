from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import h3c_baselines.cli as baseline_cli
import h3c_baselines.mpc.refit as refit
from h3c_baselines.mpc.training import EpisodeData
from h3c_baselines.mpc.vector_arx import (
    ArxLayout,
    FittedArxModel,
    Scaling,
    expected_model_identity,
    with_pmv_robust_margin,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _configuration() -> dict[str, Any]:
    return {
        "case_order": ["Case"],
        "fit_checkpoints": [8, 16, 32, 64],
        "holdout_episodes": 4,
        "ridge_alpha_candidates": [1e-6, 1.0],
    }


def _profile() -> dict[str, Any]:
    return {
        "evaluation_start_day": 10,
        "protocol": {"formal_evaluation_days": 5},
        "zones": {"z": {}},
        "comfort": {
            "dynamic_clothing": False,
            "metabolic_rate": 1.1,
            "relative_humidity_percent": 50.0,
            "air_velocity_m_s": 0.1,
            "winter_clothing_insulation": 1.0,
            "summer_clothing_insulation": 0.5,
            "clothing_transition_low_c": 10.0,
            "clothing_transition_high_c": 26.0,
        },
    }


def _episode(
    case_dir: Path,
    *,
    role: str,
    episode: int,
    lane: int,
    fallback_count: int = 0,
) -> Path:
    path = case_dir / "episodes" / f"{role}-{episode:03d}-lane-{lane}"
    path.mkdir(parents=True)
    start = 3 * 86400
    steps = 668 if role == "validation" else 672
    rows = steps + 1
    np.savez_compressed(
        path / "trajectory.npz",
        times=np.arange(rows, dtype=np.int64) * 900 + start,
        outputs=np.column_stack((np.linspace(24.0, 25.0, rows), np.linspace(100.0, 120.0, rows))),
        controls=np.full((rows, 1), 25.0),
        disturbances=np.zeros((rows, 5)),
    )
    _write_json(
        path / "manifest.json",
        {
            "role": role,
            "episode": episode,
            "lane": lane,
            "test_id": f"test-lane-{lane}",
            "start_time_seconds": start,
            "warmup_period_seconds": 7 * 86400,
            "steps": steps,
            "reward": -1.0,
            "peak_occupied_absolute_pmv": 0.6,
            "fallback_count": fallback_count,
            "recovery_step_count": 0,
        },
    )
    return path


def _source_bank(
    root: Path,
    *,
    frozen_manifest: bool = False,
) -> tuple[Path, Path]:
    source = root / "outputs" / "baselines" / "mpc" / "training" / "failed"
    case_dir = source / "Case"
    source.mkdir(parents=True)
    source_commit = "a" * 40
    _write_json(
        source / "resolved_plan.json",
        {
            "schema": "h3c_hierarchical_mpc_training_plan",
            "schema_version": 1,
            "execution": True,
            "source_commit": source_commit,
            "case_order": ["Case"],
            "fit_checkpoints": [8, 16, 32, 64],
            "holdout_episodes": 4,
            "workers": 4,
            "warmup_days_per_episode": 7,
            "training_week_days_before_evaluation": 7,
        },
    )
    _write_json(
        source / "failure.json",
        {
            "schema": "h3c_hierarchical_mpc_training_failure",
            "schema_version": 1,
            "source_commit": source_commit,
            "secret_exposure_count": 0,
        },
    )
    episode_paths = [
        *[_episode(case_dir, role="fit", episode=index, lane=index % 4) for index in range(32)],
        _episode(case_dir, role="basic_reference", episode=0, lane=0),
        *[_episode(case_dir, role="holdout", episode=index, lane=index) for index in range(4)],
    ]
    fallback = {8: 0, 16: 0, 32: 0}
    for checkpoint, count in fallback.items():
        episode_path = _episode(
            case_dir,
            role="validation",
            episode=checkpoint,
            lane=0,
            fallback_count=count,
        )
        episode_paths.append(episode_path)
        _write_json(
            case_dir / "checkpoints" / f"checkpoint-{checkpoint:03d}.json",
            {
                "checkpoint_fit_episodes": checkpoint,
                "closed_loop_validation": {
                    "fallback_count": count,
                    "peak_occupied_absolute_pmv": 0.6,
                    "reward": -1.0,
                },
            },
        )
    if frozen_manifest:
        counts = {
            lane: sum(
                json.loads((path / "manifest.json").read_text(encoding="utf-8"))["lane"] == lane
                for path in episode_paths
            )
            for lane in range(4)
        }
        _write_json(
            case_dir / "frozen_model" / "training_manifest.json",
            {
                "schema": "h3c_hierarchical_mpc_training_manifest",
                "schema_version": 1,
                "case": "Case",
                "source_commit": source_commit,
                "training_output": case_dir.relative_to(root).as_posix(),
                "lane_lifecycle": [
                    {
                        "lane": lane,
                        "test_id": f"test-lane-{lane}",
                        "select_count": 1,
                        "initialize_count": counts[lane],
                        "stop_count": 1,
                        "warmup_days_per_initialize": 7,
                    }
                    for lane in range(4)
                ],
            },
        )
    return source, case_dir


def _patch_source_owners(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(refit, "repository_root", lambda: root)
    monkeypatch.setattr(refit, "load_hierarchical_mpc_config", _configuration)
    monkeypatch.setattr(refit, "load_profile", lambda _case: _profile())


def test_source_roles_reconstruct_exact_partition_and_lane_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_bank(tmp_path, frozen_manifest=True)
    _patch_source_owners(monkeypatch, tmp_path)

    roles = refit._case_roles("Case", source)
    summary = refit._role_summary(roles)

    assert summary["episodes"]["adaptive_validation"] == [
        "episodes/validation-008-lane-0",
        "episodes/validation-016-lane-0",
    ]
    assert summary["episodes"]["calibration"] == "episodes/validation-032-lane-0"
    assert summary["episodes"]["excluded_fallback"] == []
    assert len(summary["episodes"]["holdout"]) == 4
    assert all(row["test_identity_count"] == 1 for row in summary["lane_lifecycle"])
    assert all(row["frozen_manifest_cross_checked"] is True for row in summary["lane_lifecycle"])


def test_aligned_training_window_uses_first_five_days_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    source_roles = refit._case_roles("Case", source)

    roles, window = refit._refit_roles("Case", source, align_training_windows=True)

    assert window is not None
    assert window["selected_days"] == 5
    assert window["source_start_day"] == 3
    assert window["selected_end_day_exclusive"] == 8
    assert window["physical_transitions"] == 480
    assert window["state_rows"] == 481
    assert window["selection"] == "source_prefix"
    selected = [*roles.fit, *roles.holdout, roles.calibration]
    assert all(len(episode.data.times) == 481 for episode in selected)
    assert all(episode.data.times[-1] == 8 * 86400 for episode in selected)
    assert len(source_roles.fit[0].data.times) == 673


def test_aligned_seven_day_training_window_preserves_complete_source_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    profile = _profile()
    profile["protocol"]["formal_evaluation_days"] = 7
    monkeypatch.setattr(refit, "load_profile", lambda _case: profile)
    source_roles = refit._case_roles("Case", source)

    roles, window = refit._refit_roles("Case", source, align_training_windows=True)

    assert window is not None
    assert window["selected_days"] == 7
    assert window["selection"] == "complete_source_episode"
    assert roles is not source_roles
    assert np.array_equal(roles.fit[0].data.outputs, source_roles.fit[0].data.outputs)


def test_aligned_refit_plan_records_each_selected_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)

    plan = refit.resolved_refit_plan(source, align_training_windows=True)

    assert plan["training_window_policy"] == "case_formal_evaluation_days_from_source_start"
    assert plan["training_windows"]["Case"]["selected_days"] == 5
    assert plan["training_windows"]["Case"]["fit_holdout_calibration_array_use"] == (
        "selected_window_only"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("test_id", "different-test", "test identity is unstable"),
        ("reward", float("nan"), "episode metrics are invalid"),
        ("fallback_count", -1, "episode metrics are invalid"),
        ("steps", 672, "episode protocol is invalid"),
    ],
)
def test_source_episode_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    source, case_dir = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    manifest_path = case_dir / "episodes" / "validation-032-lane-0" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        refit._case_roles("Case", source)


def test_source_plan_and_failure_commit_must_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    failure_path = source / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["source_commit"] = "b" * 40
    failure_path.write_text(json.dumps(failure), encoding="utf-8")

    with pytest.raises(ValueError, match="plan/failure identity is inconsistent"):
        refit._resolve_source_run(source)


def test_source_episode_timeline_must_match_registered_role_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, case_dir = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    trajectory_path = case_dir / "episodes" / "validation-032-lane-0" / "trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as value:
        arrays = {name: np.asarray(value[name]).copy() for name in value.files}
    arrays["times"] += 900
    np.savez_compressed(trajectory_path, **arrays)

    with pytest.raises(ValueError, match="source trajectory is invalid"):
        refit._case_roles("Case", source)


def test_source_fit_ids_and_registered_lane_assignment_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, case_dir = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    original = case_dir / "episodes" / "fit-003-lane-3"
    renumbered = case_dir / "episodes" / "fit-040-lane-3"
    original.rename(renumbered)
    manifest_path = renumbered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode"] = 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fit episode IDs are not contiguous"):
        refit._case_roles("Case", source)

    renumbered.rename(original)
    manifest["episode"] = 3
    manifest_path = original / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong_lane = case_dir / "episodes" / "fit-001-lane-1"
    renamed = case_dir / "episodes" / "fit-001-lane-0"
    wrong_lane.rename(renamed)
    manifest_path = renamed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lane"] = 0
    manifest["test_id"] = "test-lane-0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="episode lane assignment is invalid"):
        refit._case_roles("Case", source)


def test_source_validation_episode_set_cannot_omit_an_eligible_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, case_dir = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    missing = case_dir / "episodes" / "validation-016-lane-0"
    missing.rename(tmp_path / missing.name)

    with pytest.raises(ValueError, match="validation IDs do not exactly match"):
        refit._case_roles("Case", source)


def test_source_checkpoint_report_set_cannot_delete_an_eligible_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, case_dir = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    missing = case_dir / "checkpoints" / "checkpoint-016.json"
    missing.rename(tmp_path / missing.name)

    with pytest.raises(ValueError, match="checkpoint reports do not exactly match"):
        refit._case_roles("Case", source)


def test_source_validation_cannot_overreach_fit_episode_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, case_dir = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    _episode(case_dir, role="validation", episode=64, lane=0)
    _write_json(
        case_dir / "checkpoints" / "checkpoint-064.json",
        {
            "checkpoint_fit_episodes": 64,
            "closed_loop_validation": {
                "fallback_count": 0,
                "peak_occupied_absolute_pmv": 0.6,
                "reward": -1.0,
            },
        },
    )

    with pytest.raises(ValueError, match="validation IDs do not exactly match"):
        refit._case_roles("Case", source)


def _constant_model(layout: ArxLayout) -> FittedArxModel:
    base = FittedArxModel(
        layout=layout,
        intercept=np.asarray([24.0, 100.0]),
        coefficients=np.zeros((layout.feature_dimension, layout.output_dimension)),
        scaling=Scaling(
            feature_mean=np.zeros(layout.feature_dimension),
            feature_scale=np.ones(layout.feature_dimension),
            output_mean=np.zeros(layout.output_dimension),
            output_scale=np.ones(layout.output_dimension),
        ),
        ridge_alpha=1.0,
        identity="placeholder",
    )
    return FittedArxModel(
        base.layout,
        base.intercept,
        base.coefficients,
        base.scaling,
        base.ridge_alpha,
        expected_model_identity(base),
    )


def test_calibration_uses_real_basic_reference_occupancy_at_k_plus_4() -> None:
    layout = ArxLayout(("z",), ("outdoor", "solar", "occupancy", "sin", "cos"))
    model = _constant_model(layout)
    times = np.arange(9, dtype=np.int64) * 900
    outputs = np.column_stack((np.full(9, 25.0), np.full(9, 100.0)))
    controls = np.full((9, 1), 25.0)
    calibration_disturbances = np.zeros((9, 5))
    reference_times = np.arange(9, dtype=np.int64) * 900
    reference_disturbances = np.zeros((9, 5))
    # For origin=4, k+3 is unoccupied while the real k+4 target is occupied.
    calibration_disturbances[8, 2] = 1.0
    reference_disturbances[8, 2] = 1.0
    calibration = EpisodeData(
        "validation",
        64,
        0,
        "test",
        times,
        outputs,
        controls,
        calibration_disturbances,
        -1.0,
        0.6,
        0,
        0,
    )
    basic = EpisodeData(
        "basic_reference",
        0,
        0,
        "test",
        reference_times,
        np.column_stack((np.full(9, 25.0), np.full(9, 100.0))),
        np.full((9, 1), 25.0),
        reference_disturbances,
        -1.0,
        0.6,
        0,
        0,
    )

    report = refit._calibration_report(model, calibration, basic, _profile(), {0: 20.0})

    assert report["terminal_reference"]["target_offset_steps"] == 4
    assert report["terminal_reference"]["terminal_occupied_zone_targets"] == 1
    assert report["residual_count"] == 1
    assert report["scope"] == "case_specific_estimate_common_formula"

    reference_disturbances[5, 0] = 1.0
    with pytest.raises(ValueError, match="exogenous disturbances do not align"):
        refit._calibration_report(model, calibration, basic, _profile(), {0: 20.0})


def test_calibration_respects_half_open_training_window_at_midnight() -> None:
    layout = ArxLayout(("z",), ("outdoor", "solar", "occupancy", "sin", "cos"))
    model = _constant_model(layout)
    times = np.arange(10, dtype=np.int64) * 900 + (86400 - 9 * 900)
    outputs = np.column_stack((np.full(10, 25.0), np.full(10, 100.0)))
    controls = np.full((10, 1), 25.0)
    disturbances = np.zeros((10, 5))
    disturbances[8, 2] = 1.0
    calibration = EpisodeData(
        "validation", 64, 0, "test", times, outputs, controls, disturbances, -1.0, 0.6, 0, 0
    )
    basic = EpisodeData(
        "basic_reference",
        0,
        0,
        "test",
        times,
        outputs,
        controls,
        disturbances,
        -1.0,
        0.6,
        0,
        0,
    )

    report = refit._calibration_report(
        model,
        calibration,
        basic,
        _profile(),
        {0: 20.0},
        target_end_time_exclusive=86400,
    )

    assert report["target_end_time_exclusive"] == 86400
    assert report["terminal_reference"]["terminal_zone_targets"] == 1


def test_margin_changes_bundle_identity_but_zero_preserves_it() -> None:
    model = _constant_model(ArxLayout(("z",), ("outdoor", "solar", "occupancy", "sin", "cos")))

    assert with_pmv_robust_margin(model, 0.0).identity == model.identity
    calibrated = with_pmv_robust_margin(model, 0.1)
    assert calibrated.identity != model.identity
    assert calibrated.identity == expected_model_identity(calibrated)


def test_coordinated_candidate_artifact_tamper_cannot_bypass_source_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refit, "load_profile", lambda _case: _profile())
    source_model = with_pmv_robust_margin(_constant_model(refit._arx_layout(_profile())), 0.1)
    target = tmp_path / "candidate"
    target.mkdir()
    source_model.save(target / "model_coefficients.npz")
    episode_roles = {
        "episodes": {
            "fit": ["fit", "adaptive"],
            "adaptive_validation": ["adaptive"],
            "calibration": "calibration",
            "holdout": ["holdout"],
            "excluded_fallback": [],
            "unused_validation": [],
        }
    }
    calibration = {
        "pmv_robust_margin": 0.1,
        "internal_comfort_band": 0.4,
        "residual_count": 4,
    }
    fit_report = {"final_fit_includes_holdout": False}
    quality = {"finite": True, "beats_persistence": True}
    report = {
        "schema": "h3c_hierarchical_mpc_refit_candidate",
        "case": "Case",
        "model_identity": source_model.identity,
        "eligible": True,
        "checks": {"registered": True},
        "episode_roles": episode_roles,
        "fit_rows": 10,
        "holdout_rows": 4,
        "calibration_one_step_rows": 5,
        "fit_report": fit_report,
        "holdout_use": "alpha_selection_and_persistence_gate_only",
        "final_fit_includes_holdout": False,
        "prediction_quality": quality,
        "calibration": calibration,
        "physical_validation": "pending_fresh_validation",
    }
    card = {
        "schema": "h3c_hierarchical_mpc_refit_model_card",
        "case": "Case",
        "model_identity": source_model.identity,
        "holdout_use": "alpha_selection_and_persistence_gate_only",
        "final_fit_includes_holdout": False,
        "robust_margin": {
            "scope": "case_specific_estimate_common_formula",
            "estimator": "p95_absolute_occupied_pmv_prediction_residual",
            "quantile": 0.95,
            "order_statistic": "higher",
            "calibration_episode": "calibration",
            "sample_count": 4,
            "pmv_margin": 0.1,
            "internal_comfort_band": 0.4,
            "application": "internal_soft_comfort_band_only",
        },
        "physical_validation": "pending_fresh_validation",
    }
    source_evidence = {
        "episode_roles": episode_roles,
        "fit_rows": 10,
        "holdout_rows": 4,
        "calibration_one_step_rows": 5,
        "fit_report": fit_report,
        "holdout_use": "alpha_selection_and_persistence_gate_only",
        "final_fit_includes_holdout": False,
        "prediction_quality": quality,
        "calibration": calibration,
        "candidate_model_identity": source_model.identity,
    }
    _write_json(target / "candidate_report.json", report)
    _write_json(target / "model_card.json", card)
    pristine_verification = refit._verify_candidate(
        "Case",
        target,
        source_evidence=source_evidence,
        source_model=source_model,
    )
    assert pristine_verification["valid"] is True
    json.dumps(pristine_verification, allow_nan=False)
    assert type(pristine_verification["valid"]) is bool
    assert all(type(value) is bool for value in pristine_verification["checks"].values())

    changed_coefficients = source_model.coefficients.copy()
    changed_coefficients[0, 0] += 1.0
    changed = FittedArxModel(
        source_model.layout,
        source_model.intercept,
        changed_coefficients,
        source_model.scaling,
        source_model.ridge_alpha,
        "placeholder",
        source_model.pmv_robust_margin,
    )
    changed = FittedArxModel(
        changed.layout,
        changed.intercept,
        changed.coefficients,
        changed.scaling,
        changed.ridge_alpha,
        expected_model_identity(changed),
        changed.pmv_robust_margin,
    )
    changed.save(target / "model_coefficients.npz")
    report["model_identity"] = changed.identity
    card["model_identity"] = changed.identity
    _write_json(target / "candidate_report.json", report)
    _write_json(target / "model_card.json", card)

    verification = refit._verify_candidate(
        "Case",
        target,
        source_evidence=source_evidence,
        source_model=source_model,
    )
    assert verification["valid"] is False
    assert verification["checks"]["source_model_exact"] is False
    assert verification["checks"]["source_coefficients"] is False
    assert verification["checks"]["source_candidate_identity"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantile", 0.9),
        ("application", "public_reward_band"),
        ("calibration_episode", "different-calibration"),
    ],
)
def test_candidate_card_robust_margin_contract_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    monkeypatch.setattr(refit, "load_profile", lambda _case: _profile())
    model = with_pmv_robust_margin(_constant_model(refit._arx_layout(_profile())), 0.1)
    target = tmp_path / "candidate"
    target.mkdir()
    model.save(target / "model_coefficients.npz")
    roles = {
        "episodes": {
            "fit": ["fit", "adaptive"],
            "adaptive_validation": ["adaptive"],
            "calibration": "calibration",
            "holdout": ["holdout"],
        }
    }
    calibration = {"pmv_robust_margin": 0.1, "residual_count": 4}
    fit_report = {"final_fit_includes_holdout": False}
    quality = {"finite": True, "beats_persistence": True}
    _write_json(
        target / "candidate_report.json",
        {
            "schema": "h3c_hierarchical_mpc_refit_candidate",
            "case": "Case",
            "model_identity": model.identity,
            "eligible": True,
            "checks": {"registered": True},
            "episode_roles": roles,
            "fit_rows": 10,
            "holdout_rows": 4,
            "calibration_one_step_rows": 5,
            "fit_report": fit_report,
            "holdout_use": "alpha_selection_and_persistence_gate_only",
            "final_fit_includes_holdout": False,
            "prediction_quality": quality,
            "calibration": calibration,
            "physical_validation": "pending_fresh_validation",
        },
    )
    robust = {
        "scope": "case_specific_estimate_common_formula",
        "estimator": "p95_absolute_occupied_pmv_prediction_residual",
        "quantile": 0.95,
        "order_statistic": "higher",
        "calibration_episode": "calibration",
        "sample_count": 4,
        "pmv_margin": 0.1,
        "internal_comfort_band": 0.4,
        "application": "internal_soft_comfort_band_only",
    }
    robust[field] = value
    _write_json(
        target / "model_card.json",
        {
            "schema": "h3c_hierarchical_mpc_refit_model_card",
            "case": "Case",
            "model_identity": model.identity,
            "holdout_use": "alpha_selection_and_persistence_gate_only",
            "final_fit_includes_holdout": False,
            "robust_margin": robust,
            "physical_validation": "pending_fresh_validation",
        },
    )
    source_evidence = {
        "episode_roles": roles,
        "fit_rows": 10,
        "holdout_rows": 4,
        "calibration_one_step_rows": 5,
        "fit_report": fit_report,
        "holdout_use": "alpha_selection_and_persistence_gate_only",
        "final_fit_includes_holdout": False,
        "prediction_quality": quality,
        "calibration": calibration,
        "candidate_model_identity": model.identity,
    }

    verification = refit._verify_candidate(
        "Case", target, source_evidence=source_evidence, source_model=model
    )

    assert verification["valid"] is False
    assert verification["checks"]["model_card"] is False


def test_source_evidence_recomputes_persistence_instead_of_reading_candidate_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    observed: dict[str, int] = {}

    def quality(_model: FittedArxModel, episodes: list[EpisodeData]) -> dict[str, Any]:
        observed["holdout_episode_count"] = len(episodes)
        return {"finite": True, "beats_persistence": True, "owner": "recomputed"}

    monkeypatch.setattr(refit, "open_loop_prediction_quality", quality)
    monkeypatch.setattr(
        refit,
        "_calibration_report",
        lambda *_args, **_kwargs: {
            "owner": "recomputed",
            "pmv_robust_margin": 0.1,
        },
    )
    monkeypatch.setattr(refit, "_daily_outdoor_means", lambda _episode: {3: 20.0})

    evidence = refit.recompute_candidate_source_evidence("Case", source)

    assert observed["holdout_episode_count"] == 4
    assert evidence["prediction_quality"] == {
        "finite": True,
        "beats_persistence": True,
        "owner": "recomputed",
    }
    assert evidence["holdout_use"] == "alpha_selection_and_persistence_gate_only"
    assert evidence["final_fit_includes_holdout"] is False


def test_refit_cli_is_dry_without_explicit_execute(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        baseline_cli,
        "resolved_refit_plan",
        lambda source, **_kwargs: {"execution": False, "source_run": str(source)},
    )

    def forbidden_execute(_source: Path, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("dry refit must not execute")

    monkeypatch.setattr(baseline_cli, "refit_hierarchical_mpc", forbidden_execute)

    assert baseline_cli.main(["mpc", "refit", "--source-run", "preserved-failure"]) == 0
    assert '"execution": false' in capsys.readouterr().out


def test_refit_cli_forwards_aligned_training_window_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, bool] = {}

    def plan(_source: Path, *, align_training_windows: bool) -> dict[str, Any]:
        observed["aligned"] = align_training_windows
        return {"execution": False, "aligned": align_training_windows}

    monkeypatch.setattr(baseline_cli, "resolved_refit_plan", plan)
    monkeypatch.setattr(
        baseline_cli,
        "refit_hierarchical_mpc",
        lambda *_args, **_kwargs: pytest.fail("dry refit must not execute"),
    )

    assert (
        baseline_cli.main(
            [
                "mpc",
                "refit",
                "--source-run",
                "preserved-failure",
                "--align-training-windows",
            ]
        )
        == 0
    )
    assert observed == {"aligned": True}
    assert '"aligned": true' in capsys.readouterr().out


def test_executed_plan_is_exactly_derived_from_fresh_dry_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_bank(tmp_path)
    _patch_source_owners(monkeypatch, tmp_path)
    commit = "c" * 40

    expected = {**refit.resolved_refit_plan(source), "execution": True}
    expected["refit_source_commit"] = commit

    assert refit._executed_refit_plan(source, commit) == expected
    assert {**expected, "unexpected": True} != refit._executed_refit_plan(source, commit)


def test_execute_failure_on_later_case_is_atomic_and_never_promotes_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "outputs" / "baselines" / "mpc" / "training" / "failed"
    _write_json(source / "failure.json", {"source_commit": "a" * 40})
    _write_json(source / "resolved_plan.json", {"schema": "source"})
    model = _constant_model(refit._arx_layout(_profile()))
    monkeypatch.setattr(refit, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        refit,
        "load_hierarchical_mpc_config",
        lambda: {"case_order": ["First", "Later", "Never"]},
    )
    monkeypatch.setattr(refit, "_resolve_source_run", lambda _source: source)
    monkeypatch.setattr(
        refit,
        "resolved_refit_plan",
        lambda _source, **_kwargs: {
            "schema": "h3c_hierarchical_mpc_refit_plan",
            "execution": False,
        },
    )
    monkeypatch.setattr(refit, "committed_source_identity", lambda: "c" * 40)
    monkeypatch.setattr(refit, "_source_file_manifest", lambda _source: [])
    monkeypatch.setattr(refit, "secret_occurrences", lambda _path: 0)

    def fit_candidate(case: str, _source: Path, _target: Path, **_kwargs: object) -> dict[str, Any]:
        if case == "Later":
            raise RuntimeError("registered later-case failure")
        return {"case": case}

    monkeypatch.setattr(refit, "_fit_candidate", fit_candidate)
    monkeypatch.setattr(
        refit,
        "_rebuild_candidate_from_source",
        lambda _case, _source, **_kwargs: refit._RebuiltCandidate(model=model, evidence={}),
    )
    monkeypatch.setattr(
        refit,
        "_verify_candidate",
        lambda *_args, **_kwargs: {"valid": True},
    )

    with pytest.raises(RuntimeError, match="registered later-case failure"):
        refit.refit_hierarchical_mpc(source)

    run_root = tmp_path / "outputs" / "baselines" / "mpc" / "refit"
    workspaces = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(workspaces) == 1
    workspace = workspaces[0]
    failure = json.loads((workspace / "failure.json").read_text(encoding="utf-8"))
    assert failure["completed_cases"] == ["First"]
    assert not (workspace / "completion.json").exists()
    assert not (tmp_path / "models" / "mpc").exists()
