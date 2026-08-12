from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from crazyflow.control.mellinger.research import (
    DomainRandomizationConfig,
    ManifestEpisode,
    OptimizerConfig,
    ResearchConfig,
    SimulationConfig,
    Split,
    SplitManifest,
    TrajectoryConfig,
    build_episode_batch,
    build_training_pipelines,
    raw_from_data,
    research_objective,
    run_training_validation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def small_simulation() -> SimulationConfig:
    return SimulationConfig(n_worlds=2, horizon=6, sim_freq_hz=500, control_freq_hz=100)


def technical_trajectory() -> TrajectoryConfig:
    return replace(
        TrajectoryConfig(),
        harmonics=2,
        amplitude_m=(0.0001, 0.0001, 0.0001),
        max_acceleration_m_s2=50.0,
        max_jerk_m_s3=10000.0,
        max_tilt_rad=1.4,
    )


def validation_manifest() -> SplitManifest:
    return SplitManifest(
        schema_version="crazyflow.mellinger_split_manifest.v1",
        manifest_id="synthetic-validation-fixture",
        split=Split.VALIDATION,
        episodes=(
            ManifestEpisode("validation-fixture-0", 9101),
            ManifestEpisode("validation-fixture-1", 9103),
        ),
    )


def runner_config(*, run_id: str, steps: int, checkpoint_interval: int) -> ResearchConfig:
    simulation = replace(small_simulation(), n_worlds=1)
    return ResearchConfig(
        schema_version="crazyflow.mellinger_research_config.v1",
        run_id=run_id,
        root_seed=73,
        train=simulation,
        validation=simulation,
        trajectory=technical_trajectory(),
        domain_randomization=DomainRandomizationConfig(mass_enabled=True),
        optimizer=OptimizerConfig(
            steps=steps, learning_rate=0.001, gain_stage=1, checkpoint_interval=checkpoint_interval
        ),
        validation_manifest="configs/research/mellinger/validation_manifest.v1.json",
        test_manifest="configs/research/mellinger/does-not-exist-and-must-not-open.json",
    )


@pytest.mark.integration
def test_split_pipelines_and_fixed_validation_are_structurally_independent() -> None:
    simulation = small_simulation()
    pipelines = build_training_pipelines(simulation, simulation, root_seed=73)
    domain = DomainRandomizationConfig(mass_enabled=True)
    train_0 = build_episode_batch(pipelines.train, 73, 0, technical_trajectory(), domain)
    train_1 = build_episode_batch(pipelines.train, 73, 1, technical_trajectory(), domain)
    fixed_a = build_episode_batch(
        pipelines.validation, 73, 0, technical_trajectory(), domain, validation_manifest()
    )
    fixed_b = build_episode_batch(
        pipelines.validation, 999, 0, technical_trajectory(), domain, validation_manifest()
    )

    assert pipelines.train.sim is not pipelines.validation.sim
    assert pipelines.train.sim.data is not pipelines.validation.sim.data
    assert not hasattr(pipelines, "test")
    assert not np.array_equal(train_0.commands, train_1.commands)
    assert np.array_equal(fixed_a.commands, fixed_b.commands)
    assert np.array_equal(fixed_a.masses_kg, fixed_b.masses_kg)
    assert set(train_0.episode_ids).isdisjoint(fixed_a.episode_ids)


@pytest.mark.integration
def test_real_multiworld_objective_has_shared_finite_gain_gradient_and_metrics() -> None:
    simulation = small_simulation()
    pipelines = build_training_pipelines(simulation, simulation, root_seed=79)
    domain = DomainRandomizationConfig(
        mass_enabled=True,
        delay_enabled=True,
        delay_max_control_steps=1,
        wrench_enabled=True,
        force_std_n=0.0001,
        torque_std_nm=0.000001,
        wrench_correlation_time_s=0.05,
    )
    batch = build_episode_batch(pipelines.train, 79, 0, technical_trajectory(), domain)
    raw = raw_from_data(batch.initial_data, stage=1)

    def objective(gains: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
        return research_objective(
            gains,
            batch,
            step_fn=pipelines.train.step_fn,
            steps_per_command=simulation.sim_freq_hz // simulation.control_freq_hz,
            gain_stage=1,
        )

    (loss, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(raw)

    assert raw.ndim == 1
    assert gradient.shape == raw.shape
    assert np.all(np.isfinite(gradient))
    assert np.any(np.abs(np.asarray(gradient)) > 0.0)
    assert np.isfinite(loss)
    assert auxiliary["per_case_loss"].shape == (simulation.n_worlds,)
    assert auxiliary["per_case_metrics"]["p95_position_error_m"].shape == (simulation.n_worlds,)
    assert "terminal_velocity_error_m_s" in auxiliary["mean_metrics"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("steps", "checkpoint_interval", "expected_steps"),
    ((3, 1, [1, 2, 3]), (4, 2, [2, 4]), (5, 2, [2, 4, 5]), (3, 10, [3])),
)
def test_sparse_checkpoint_schedule_and_complete_metrics(
    tmp_path: Path, steps: int, checkpoint_interval: int, expected_steps: list[int]
) -> None:
    config = runner_config(
        run_id=f"sparse-schedule-{steps}-{checkpoint_interval}",
        steps=steps,
        checkpoint_interval=checkpoint_interval,
    )
    output_dir = tmp_path / "run"
    summary = run_training_validation(
        config,
        config_path=Path("synthetic-sparse-config.json"),
        output_dir=output_dir,
        repository_root=REPOSITORY_ROOT,
        command="sparse-schedule-fixture",
    )

    actual_steps = [
        int(path.stem.rsplit("-", 1)[1])
        for path in sorted(output_dir.glob("checkpoint-step-*.json"))
    ]
    metrics = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text().splitlines()]
    assert summary["status"] == "success"
    assert actual_steps == expected_steps
    assert len(actual_steps) == len(set(actual_steps))
    assert [row["step"] for row in metrics] == [
        step for step in range(1, steps + 1) for _ in range(2)
    ]
    assert [row["split"] for row in metrics] == ["train", "validation"] * steps
    assert [item["step"] for item in summary["timing"]["checkpoint_writes"]] == expected_steps
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["checkpoint_policy"]["interval_completed_updates"] == checkpoint_interval
    assert provenance["frozen_test_manifest_reference_not_opened"].endswith(
        "does-not-exist-and-must-not-open.json"
    )
    assert summary["test_manifest_opened"] is False


@pytest.mark.integration
def test_runner_resume_is_exact_and_never_opens_test_manifest(tmp_path: Path) -> None:
    config = runner_config(run_id="resume-integration-fixture", steps=4, checkpoint_interval=2)
    uninterrupted_dir = tmp_path / "uninterrupted"
    paused_dir = tmp_path / "paused"
    resumed_dir = tmp_path / "resumed"

    uninterrupted = run_training_validation(
        config,
        config_path=Path("synthetic-resume-config.json"),
        output_dir=uninterrupted_dir,
        repository_root=REPOSITORY_ROOT,
        command="uninterrupted-fixture",
    )
    paused = run_training_validation(
        config,
        config_path=Path("synthetic-resume-config.json"),
        output_dir=paused_dir,
        repository_root=REPOSITORY_ROOT,
        command="paused-fixture",
        max_updates_this_process=2,
    )
    resumed = run_training_validation(
        config,
        config_path=Path("synthetic-resume-config.json"),
        output_dir=resumed_dir,
        repository_root=REPOSITORY_ROOT,
        command="resumed-fixture",
        resume_path=paused_dir / "checkpoint-step-000002.json",
    )

    assert uninterrupted["status"] == resumed["status"] == "success"
    assert paused["status"] == "paused"
    assert paused["resume_checkpoint"] == "checkpoint-step-000002.json"
    reference = json.loads((uninterrupted_dir / "checkpoint-step-000004.json").read_text())
    actual = json.loads((resumed_dir / "checkpoint-step-000004.json").read_text())
    scientific_fields = (
        "config_fingerprint",
        "gain_registry_fingerprint",
        "source_fingerprint",
        "runtime_fingerprint",
        "step",
        "episode_index",
        "root_seed",
        "raw_gains",
        "optimizer_state",
        "history",
        "metrics_records",
        "selection",
        "selected_raw_gains",
        "validation_manifest_id",
        "validation_manifest_fingerprint",
        "frozen_test_manifest_reference",
    )
    assert {field: reference[field] for field in scientific_fields} == {
        field: actual[field] for field in scientific_fields
    }
    resumed_metrics = [
        json.loads(line) for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in resumed_metrics] == [
        step for step in range(1, 5) for _ in range(2)
    ]
    assert [row["split"] for row in resumed_metrics] == ["train", "validation"] * 4
    assert [path.name for path in resumed_dir.glob("checkpoint-step-*.json")] == [
        "checkpoint-step-000004.json"
    ]
    provenance = json.loads((resumed_dir / "provenance.json").read_text())
    assert provenance["resume_lineage"]["parent_step"] == 2
    assert provenance["frozen_test_manifest_reference_not_opened"].endswith(
        "does-not-exist-and-must-not-open.json"
    )
    assert resumed["test_manifest_opened"] is False
