from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control
from crazyflow.control.mellinger.research import (
    FIXED_SUPPORT_PREFIX_DISTRIBUTION,
    GAIN_REGISTRY,
    LEGACY_TRAJECTORY_DISTRIBUTION,
    OptimizerConfig,
    Split,
    TrajectoryConfig,
    apply_action_delay,
    apply_raw_gains,
    apply_true_mass,
    diagnose_trajectory_attempts,
    fingerprint,
    generate_trajectory,
    load_config,
    load_manifest,
    physical_from_raw,
    raw_from_data,
    registry_fingerprint,
    sample_correlated_wrenches,
    sample_delay_steps,
    sample_world_masses,
    seed_key,
    specs_for_stage,
)
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim


def component_keys(root_seed: int, split: Split, component: str, worlds: int = 3) -> jax.Array:
    return jnp.stack(
        [
            seed_key(root_seed, split, episode=2, world=world, component=component)
            for world in range(worlds)
        ]
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_configs_and_validation_manifest_validate_while_test_stays_opaque() -> None:
    config_root = REPOSITORY_ROOT / "configs/research/mellinger"
    smoke = load_config(config_root / "smoke.json")
    workstation = load_config(config_root / "workstation.json")
    validation = load_manifest(
        config_root / "validation_manifest.v1.json", expected_split=Split.VALIDATION
    )

    assert len(fingerprint(smoke)) == 64
    assert smoke.optimizer.checkpoint_interval == 1
    assert workstation.domain_randomization.mass_half_width_kg == pytest.approx(0.0002)
    assert not workstation.domain_randomization.delay_enabled
    assert not workstation.domain_randomization.wrench_enabled
    assert validation.episodes
    assert workstation.test_manifest == "configs/research/mellinger/test_manifest.v1.json"
    with pytest.raises(ValueError, match="expected test"):
        load_manifest(config_root / "validation_manifest.v1.json", expected_split=Split.TEST)


@pytest.mark.unit
def test_checkpoint_interval_validation_and_legacy_fingerprint(tmp_path: Path) -> None:
    sprint9_path = REPOSITORY_ROOT / "artifacts/day9-long-pilot/pilot_config.json"
    sprint9 = load_config(sprint9_path)
    assert sprint9.optimizer.checkpoint_interval == 1
    assert fingerprint(sprint9) == (
        "23ee0f46f581b6453ec51915a8cd5410e7cd77d48c9d6a07caadee74a98c5fb5"
    )

    raw = json.loads(sprint9_path.read_text())
    raw["optimizer"]["checkpoint_interval"] = 100
    sparse_path = tmp_path / "sparse.json"
    sparse_path.write_text(json.dumps(raw))
    sparse = load_config(sparse_path)
    assert sparse.optimizer.checkpoint_interval == 100
    assert fingerprint(sparse) != fingerprint(sprint9)

    for invalid in (0, -1, 1.5, True, "100"):
        with pytest.raises(ValueError, match="checkpoint_interval must be a positive integer"):
            replace(OptimizerConfig(), checkpoint_interval=invalid).validate()


@pytest.mark.unit
def test_seed_tree_is_deterministic_and_split_separated() -> None:
    first = component_keys(17, Split.TRAIN, "trajectory")
    second = component_keys(17, Split.TRAIN, "trajectory")
    validation = component_keys(17, Split.VALIDATION, "trajectory")
    changed_train = component_keys(18, Split.TRAIN, "trajectory")

    key_data = jax.random.key_data
    assert np.array_equal(key_data(first), key_data(second))
    assert len({tuple(value) for value in np.asarray(key_data(first))}) == 3
    assert not np.array_equal(key_data(first), key_data(validation))
    assert not np.array_equal(key_data(first), key_data(changed_train))
    repeated_validation = component_keys(17, Split.VALIDATION, "trajectory")
    assert np.array_equal(key_data(validation), key_data(repeated_validation))


@pytest.mark.unit
def test_mass_randomization_is_bounded_per_world_and_controller_mass_stays_fixed() -> None:
    sim = Sim(
        n_worlds=3,
        n_drones=1,
        drone="cf2x_L250",
        dynamics=Dynamics.first_principles,
        control=Control.state,
        device="cpu",
    )
    nominal = float(sim.data.params.mass[0, 0, 0])
    controller_mass = np.asarray(sim.data.controls.state.params["mass"]).copy()
    keys = component_keys(31, Split.TRAIN, "mass")
    masses = sample_world_masses(keys, nominal, 0.0002)
    randomized = apply_true_mass(sim.data, masses)

    assert masses.shape == (3, 1, 1)
    assert np.all(np.asarray(masses) >= nominal - 0.0002)
    assert np.all(np.asarray(masses) <= nominal + 0.0002)
    assert np.array_equal(randomized.controls.state.params["mass"], controller_mass)
    assert not np.shares_memory(
        np.asarray(randomized.params.mass), np.asarray(sim.data.params.mass)
    )


@pytest.mark.unit
def test_action_delay_zero_identity_and_exact_interval_semantics() -> None:
    commands = jnp.arange(5 * 2 * 1 * 13).reshape(5, 2, 1, 13)
    delayed = apply_action_delay(commands, jnp.array([0, 2]))

    assert np.array_equal(delayed[:, 0], commands[:, 0])
    assert np.array_equal(delayed[:3, 1], jnp.broadcast_to(commands[0:1, 1], (3, 1, 13)))
    assert np.array_equal(delayed[3, 1], commands[1, 1])
    keys = component_keys(7, Split.TRAIN, "delay", worlds=2)
    assert np.array_equal(sample_delay_steps(keys, 0), np.zeros(2, dtype=np.int32))
    with pytest.raises(ValueError, match="nonnegative"):
        apply_action_delay(commands, jnp.array([0, -1]))


@pytest.mark.unit
def test_correlated_wrenches_are_deterministic_independent_and_disable_to_zero() -> None:
    keys = component_keys(29, Split.TRAIN, "wrench", worlds=2)
    arguments = (keys, 12, 1, 0.01, 0.002, 0.00001, 0.05)
    force_a, torque_a = sample_correlated_wrenches(*arguments)
    force_b, torque_b = sample_correlated_wrenches(*arguments)
    zero_force, zero_torque = sample_correlated_wrenches(keys, 12, 1, 0.01, 0.0, 0.0, 0.05)

    assert force_a.shape == torque_a.shape == (12, 2, 1, 3)
    assert np.array_equal(force_a, force_b)
    assert np.array_equal(torque_a, torque_b)
    assert not np.array_equal(force_a[:, 0], force_a[:, 1])
    assert np.count_nonzero(zero_force) == np.count_nonzero(zero_torque) == 0


@pytest.mark.unit
def test_gain_registry_roundtrip_and_exclusions() -> None:
    sim = Sim(
        n_worlds=2,
        n_drones=1,
        drone="cf2x_L250",
        dynamics=Dynamics.first_principles,
        control=Control.state,
        device="cpu",
    )
    raw = raw_from_data(sim.data, stage=3)
    physical = physical_from_raw(raw, stage=3)
    updated = apply_raw_gains(sim.data, raw, stage=3)

    assert raw.shape == (len(specs_for_stage(3)),)
    assert len(registry_fingerprint()) == 64
    assert {spec.name for spec in GAIN_REGISTRY if spec.decision == "exclude"} == {"kd_omega_z"}
    assert "ki_m_xy" not in physical
    for spec in specs_for_stage(3):
        controller = getattr(updated.controls, spec.controller)
        values = np.asarray(controller.params[spec.parameter])[list(spec.axes)]
        assert np.allclose(values, float(physical[spec.name]), rtol=1e-5)
    assert updated.controls.state.params["kp"].shape == (3,)


@pytest.mark.unit
def test_random_fourier_trajectory_is_deterministic_bounded_and_c3_at_endpoints() -> None:
    config = replace(
        TrajectoryConfig(),
        amplitude_m=(0.04, 0.03, 0.02),
        max_speed_m_s=3.0,
        max_acceleration_m_s2=20.0,
        max_jerk_m_s3=500.0,
        max_tilt_rad=1.3,
    )
    key = seed_key(41, Split.VALIDATION, 0, 0, "trajectory")
    first = generate_trajectory(key, horizon=400, control_freq_hz=100, config=config)
    second = generate_trajectory(key, horizon=400, control_freq_hz=100, config=config)

    assert np.array_equal(first.trajectory.pos, second.trajectory.pos)
    assert np.array_equal(first.jerk, second.jerk)
    assert first.trajectory.pos.shape == (401, 3)
    positions = np.asarray(first.trajectory.pos)
    velocities = np.asarray(first.trajectory.vel)
    accelerations = np.asarray(first.trajectory.acc)
    jerks = np.asarray(first.jerk)
    assert np.allclose(positions[[0, -1]], config.center_m, atol=2e-6)
    assert np.allclose(velocities[[0, -1]], 0.0, atol=2e-5)
    assert np.allclose(accelerations[[0, -1]], 0.0, atol=2e-4)
    assert np.allclose(jerks[[0, -1]], 0.0, atol=2e-3)
    assert first.diagnostics.max_jerk_m_s3 <= config.max_jerk_m_s3


@pytest.mark.unit
def test_legacy_trajectory_default_preserves_golden_h400_candidate() -> None:
    config = load_config(REPOSITORY_ROOT / "configs/research/mellinger/workstation.json")
    key = seed_key(config.root_seed, Split.TRAIN, 0, 0, "trajectory")
    report = diagnose_trajectory_attempts(key, 400, 100, config.trajectory)[-1]

    assert config.trajectory.distribution_version == LEGACY_TRAJECTORY_DISTRIBUTION
    assert report.accepted
    assert report.attempt_index == 0
    assert report.array_digests == {
        "position": {
            "dtype": "float32",
            "shape": [401, 3],
            "sha256": "099069834cad644776ce6d181b57fc984c42ec4c4a510eda10eb6a65e782f181",
        },
        "velocity": {
            "dtype": "float32",
            "shape": [401, 3],
            "sha256": "8bf3f28ce104f7e0fe9d22d60f8d21f4aee1a8489e7fd8a1fde82548b8755fe1",
        },
        "acceleration": {
            "dtype": "float32",
            "shape": [401, 3],
            "sha256": "5107f30e74dd32c1e1c6450f7025e405aacd17e65f5dabbdcf94d103206e777c",
        },
        "jerk": {
            "dtype": "float32",
            "shape": [401, 3],
            "sha256": "4b50d0ea8aecdab40026c1a5c42ba14a80f3ea2b5065e05b89e569a3e499fa73",
        },
    }


@pytest.mark.unit
def test_fixed_support_distribution_is_prefix_consistent_and_h400_legacy_exact() -> None:
    config = load_config(REPOSITORY_ROOT / "configs/research/mellinger/sprint7_pilot.json")
    key = seed_key(config.root_seed, Split.TRAIN, 3, 0, "trajectory")
    generated = {
        horizon: generate_trajectory(key, horizon, 100, config.trajectory)
        for horizon in (100, 200, 400)
    }
    reports = {
        horizon: diagnose_trajectory_attempts(key, horizon, 100, config.trajectory)[-1]
        for horizon in (100, 200, 400)
    }
    repeated = generate_trajectory(key, 100, 100, config.trajectory)
    different_key = seed_key(config.root_seed, Split.TRAIN, 4, 0, "trajectory")
    different = generate_trajectory(different_key, 100, 100, config.trajectory)
    legacy = replace(
        config.trajectory,
        distribution_version=LEGACY_TRAJECTORY_DISTRIBUTION,
        support_duration_s=None,
        minimum_duration_s=None,
    )
    legacy_h400 = generate_trajectory(key, 400, 100, legacy)

    assert config.trajectory.distribution_version == FIXED_SUPPORT_PREFIX_DISTRIBUTION
    assert generated[100].attempt == generated[200].attempt == generated[400].attempt
    assert all(report.accepted for report in reports.values())
    assert all(report.validation_horizon_control_intervals == 400 for report in reports.values())
    assert all(all(item.passed for item in report.constraints) for report in reports.values())
    assert np.array_equal(generated[100].trajectory.pos, generated[400].trajectory.pos[:101])
    assert np.array_equal(generated[200].trajectory.vel, generated[400].trajectory.vel[:201])
    assert np.array_equal(generated[100].jerk, generated[400].jerk[:101])
    assert np.array_equal(generated[100].trajectory.pos, repeated.trajectory.pos)
    assert not np.array_equal(generated[100].trajectory.pos, different.trajectory.pos)
    assert np.array_equal(generated[400].trajectory.pos, legacy_h400.trajectory.pos)
    assert np.array_equal(generated[400].jerk, legacy_h400.jerk)
    assert generated[400].attempt == legacy_h400.attempt


@pytest.mark.unit
def test_trajectory_distribution_versions_reject_invalid_or_out_of_range_requests() -> None:
    base = TrajectoryConfig()
    with pytest.raises(ValueError, match="unsupported trajectory distribution_version"):
        replace(base, distribution_version="silent-fallback-forbidden").validate()
    with pytest.raises(ValueError, match="requires finite positive"):
        replace(base, distribution_version=FIXED_SUPPORT_PREFIX_DISTRIBUTION).validate()

    config = load_config(REPOSITORY_ROOT / "configs/research/mellinger/sprint7_pilot.json")
    key = jax.random.key(7)
    with pytest.raises(ValueError, match="below minimum_duration_s"):
        generate_trajectory(key, 50, 100, config.trajectory)
    with pytest.raises(ValueError, match="exceeds support_duration_s"):
        generate_trajectory(key, 401, 100, config.trajectory)


@pytest.mark.unit
def test_random_trajectory_rejection_is_finite() -> None:
    impossible = replace(TrajectoryConfig(), max_speed_m_s=1.0e-9, max_attempts=2)
    key = jax.random.key(1)
    with pytest.raises(ValueError, match="2 deterministic attempts"):
        generate_trajectory(key, horizon=100, control_freq_hz=100, config=impossible)


@pytest.mark.unit
def test_trajectory_attempt_diagnostics_preserve_rejection_and_signed_margins() -> None:
    impossible = replace(TrajectoryConfig(), max_speed_m_s=1.0e-9, max_attempts=2)
    key = jax.random.key(1)
    first = diagnose_trajectory_attempts(key, 100, 100, impossible)
    second = diagnose_trajectory_attempts(key, 100, 100, impossible)

    assert len(first) == 2
    assert first == second
    assert not any(attempt.accepted for attempt in first)
    for attempt in first:
        speed = next(item for item in attempt.constraints if item.name == "speed_m_s")
        assert not speed.passed
        assert speed.observed_minus_limit > 0.0
        assert speed.acceptance_margin < 0.0
        assert speed.violation_magnitude == pytest.approx(-speed.acceptance_margin)
        assert speed.sample_index is not None
        assert speed.time_seconds == pytest.approx(speed.sample_index / 100)
        assert len(attempt.array_digests["position"]["sha256"]) == 64
