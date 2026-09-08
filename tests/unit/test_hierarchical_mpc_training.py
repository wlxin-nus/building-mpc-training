from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, cast

import numpy as np
import pytest

import h3c_baselines.mpc.training as training
from h3c.experiments.profiles import load_profile
from h3c_baselines.configuration import load_hierarchical_mpc_config
from h3c_baselines.mpc.training import (
    ActiveExcitationController,
    EpisodeData,
    _collect_episode,
    _episode_dataset,
    _Lane,
    _training_lock,
    _write_completion,
    resolved_training_plan,
)
from h3c_baselines.mpc.vector_arx import ArxLayout, FittedArxModel, Scaling


def _episode(index: int, *, rows: int = 12) -> EpisodeData:
    times = np.arange(rows, dtype=np.int64) * 900
    outputs = np.column_stack(
        (
            np.linspace(23.0 + index, 24.0 + index, rows),
            np.linspace(100.0, 200.0, rows),
        )
    )
    return EpisodeData(
        role="fit",
        episode=index,
        lane=index,
        test_id=f"test-{index}",
        times=times,
        outputs=outputs,
        controls=np.full((rows, 1), 25.0 + index),
        disturbances=np.column_stack(
            (
                np.full(rows, 30.0),
                np.linspace(0.0, 400.0, rows),
                np.ones(rows),
                np.zeros(rows),
                np.ones(rows),
            )
        ),
        reward=-float(index),
        peak_occupied_absolute_pmv=0.4,
        fallback_count=0,
        recovery_step_count=0,
    )


def test_episode_dataset_never_creates_cross_episode_history() -> None:
    layout = ArxLayout(("zone",), ("outdoor", "solar", "occupancy", "sin", "cos"))
    first = _episode(0)
    second = _episode(1)

    features, targets = _episode_dataset(layout, (first, second))

    rows_per_episode = len(first.times) - layout.lag_count
    assert len(features) == len(targets) == 2 * rows_per_episode
    assert set(targets[:rows_per_episode, 0]) <= set(first.outputs[:, 0])
    assert set(targets[rows_per_episode:, 0]) <= set(second.outputs[:, 0])


def test_cancelled_parallel_episode_never_initializes_physics(tmp_path: Path) -> None:
    class MustNotInitialize:
        test_id = "reserved"

        def initialize_selected(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
            raise AssertionError("cancelled episode touched BOPTEST")

    cancelled = Event()
    cancelled.set()
    lane = _Lane(0, MustNotInitialize(), "reserved")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="cancelled"):
        _collect_episode(
            lane=lane,
            profile={},
            role="fit",
            episode=0,
            excitation_config={},
            output_dir=tmp_path,
            cancel_event=cancelled,
        )
    assert lane.initialize_count == 0


def test_training_lock_and_completion_sentinel_are_single_owner(tmp_path: Path) -> None:
    lock = tmp_path / ".training.lock"
    with _training_lock(lock):
        assert lock.is_file()
        with (
            pytest.raises(ValueError, match="another hierarchical MPC training task"),
            _training_lock(lock),
        ):
            pass
    assert not lock.exists()
    with _training_lock(lock):
        assert lock.is_file()

    completion = tmp_path / "completion.json"
    _write_completion(completion, {"status": "complete"})
    assert completion.read_text(encoding="utf-8").endswith("\n")
    assert not (tmp_path / ".completion.json.pending").exists()
    with pytest.raises(ValueError, match="already exists"):
        _write_completion(completion, {"status": "duplicate"})


def test_excitation_respects_occupancy_bounds_and_comfort_recovery() -> None:
    config = {
        "dwell_minutes": [30, 60, 120],
        "occupied_bounds_c": [23.5, 26.5],
        "unoccupied_bounds_c": [20.0, 30.0],
        "comfort_recovery_trigger_absolute_pmv": 0.70,
        "comfort_recovery_release_absolute_pmv": 0.50,
    }
    first = ActiveExcitationController(("zone",), config, seed=17)
    second = ActiveExcitationController(("zone",), config, seed=17)
    first_trace: list[float] = []
    second_trace: list[float] = []
    for _ in range(16):
        first_trace.append(
            first.decide(
                occupancy={"zone": 1.0},
                pmv={"zone": 0.0},
                recovery_setpoints={"zone": 25.0},
            )[0]["zone"]
        )
        second_trace.append(
            second.decide(
                occupancy={"zone": 1.0},
                pmv={"zone": 0.0},
                recovery_setpoints={"zone": 25.0},
            )[0]["zone"]
        )
    assert first_trace == second_trace
    assert all(23.5 <= value <= 26.5 for value in first_trace)

    recovery, count = first.decide(
        occupancy={"zone": 1.0},
        pmv={"zone": 0.71},
        recovery_setpoints={"zone": 24.7},
    )
    assert recovery == {"zone": 24.7}
    assert count == 1
    still_recovering, count = first.decide(
        occupancy={"zone": 1.0},
        pmv={"zone": 0.51},
        recovery_setpoints={"zone": 24.8},
    )
    assert still_recovering == {"zone": 24.8}
    assert count == 1
    released, count = first.decide(
        occupancy={"zone": 1.0},
        pmv={"zone": 0.50},
        recovery_setpoints={"zone": 24.9},
    )
    assert 23.5 <= released["zone"] <= 26.5
    assert count == 0


@pytest.mark.parametrize(
    ("workers", "episodes"),
    ((0, 64), (5, 64), (4, 7), (4, 12), (4, 65)),
)
def test_training_plan_rejects_capacity_and_unregistered_checkpoints(
    workers: int, episodes: int
) -> None:
    with pytest.raises(ValueError):
        resolved_training_plan(workers=workers, max_fit_episodes=episodes)


def test_failed_multicase_training_never_promotes_partial_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = dict(load_hierarchical_mpc_config())
    configuration["case_order"] = ["SZ_Air", "MZ_Hydro"]
    monkeypatch.setattr(training, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(training, "load_hierarchical_mpc_config", lambda: configuration)
    monkeypatch.setattr(training, "committed_source_identity", lambda: "source-commit")

    class FakePhysical:
        test_id: str | None = None

        def select_testcase(self, testcase: str) -> str:
            self.test_id = f"test-{testcase}"
            return self.test_id

        def stop(self) -> None:
            self.test_id = None

    def fake_fit_case(**kwargs: object) -> tuple[FittedArxModel, dict[str, object]]:
        case = str(kwargs["case"])
        lanes = cast(list[_Lane], kwargs["lanes"])
        for lane in lanes:
            lane.initialize_count = 1
        if case == "MZ_Hydro":
            raise ValueError("registered Hydro candidate failure")
        zones = tuple(load_profile(case)["zones"])
        layout = ArxLayout(
            zones,
            (
                "outdoor_temperature_c",
                "solar_irradiance_w_m2",
                *[f"effective_occupancy_{zone}" for zone in zones],
                "time_sine",
                "time_cosine",
            ),
        )
        model = FittedArxModel(
            layout=layout,
            intercept=np.zeros(layout.output_dimension),
            coefficients=np.zeros((layout.feature_dimension, layout.output_dimension)),
            scaling=Scaling(
                feature_mean=np.zeros(layout.feature_dimension),
                feature_scale=np.ones(layout.feature_dimension),
                output_mean=np.zeros(layout.output_dimension),
                output_scale=np.ones(layout.output_dimension),
            ),
            ridge_alpha=1.0,
            identity="staged-sz-model",
        )
        return model, {"case": case, "eligible_checkpoint_count": 1}

    monkeypatch.setattr(training, "_fit_case", fake_fit_case)
    with pytest.raises(ValueError, match="registered Hydro candidate failure"):
        training.train_hierarchical_mpc(
            endpoint="http://unused",
            workers=4,
            maximum_fit_episodes=8,
            physical_factory=cast(training.PhysicalFactory, lambda _endpoint: FakePhysical()),
        )

    runs = list((tmp_path / "outputs" / "baselines" / "mpc" / "training").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "failure.json").is_file()
    assert (runs[0] / "SZ_Air" / "frozen_model" / "model_card.json").is_file()
    assert not (tmp_path / "models" / "mpc" / "SZ_Air").exists()
