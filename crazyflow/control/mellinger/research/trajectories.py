"""Bounded random Fourier trajectories with analytic derivatives through jerk."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.control.mellinger.research.config import (
    FIXED_SUPPORT_PREFIX_DISTRIBUTION,
    LEGACY_TRAJECTORY_DISTRIBUTION,
)
from crazyflow.trajectory import Trajectory

if TYPE_CHECKING:
    from crazyflow.control.mellinger.research.config import TrajectoryConfig


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    max_speed_m_s: float
    max_acceleration_m_s2: float
    max_jerk_m_s3: float
    max_yaw_rate_rad_s: float
    max_tilt_rad: float
    min_specific_force_m_s2: float


@dataclass(frozen=True)
class GeneratedTrajectory:
    trajectory: Trajectory
    jerk: jax.Array
    diagnostics: TrajectoryDiagnostics
    attempt: int


@dataclass(frozen=True)
class ConstraintEvaluation:
    """One signed trajectory-bound evaluation at its extremal sample."""

    name: str
    relation: str
    limit: float | None
    observed: float | None
    observed_minus_limit: float | None
    acceptance_margin: float | None
    violation_magnitude: float
    sample_index: int | None
    time_seconds: float | None
    passed: bool


@dataclass(frozen=True)
class TrajectoryAttemptDiagnostics:
    """Construction-only evidence for one deterministic rejection attempt."""

    attempt_index: int
    attempt_count: int
    accepted: bool
    validation_scope: str
    returned_horizon_control_intervals: int
    validation_horizon_control_intervals: int
    constraints: tuple[ConstraintEvaluation, ...]
    statistics: dict[str, object]
    array_digests: dict[str, dict[str, object]]
    validation_array_digests: dict[str, dict[str, object]]


def _smoothstep7(u: jax.Array) -> tuple[jax.Array, ...]:
    s = 35 * u**4 - 84 * u**5 + 70 * u**6 - 20 * u**7
    ds = 140 * u**3 - 420 * u**4 + 420 * u**5 - 140 * u**6
    d2s = 420 * u**2 - 1680 * u**3 + 2100 * u**4 - 840 * u**5
    d3s = 840 * u - 5040 * u**2 + 8400 * u**3 - 4200 * u**4
    return s, ds, d2s, d3s


def _envelope(time: jax.Array, duration: float) -> tuple[jax.Array, ...]:
    u = time / duration
    left = _smoothstep7(u)
    right_raw = _smoothstep7(1.0 - u)
    right = (right_raw[0], -right_raw[1], right_raw[2], -right_raw[3])
    e0 = 4.0 * left[0] * right[0]
    e1 = 4.0 * (left[1] * right[0] + left[0] * right[1]) / duration
    e2 = 4.0 * (left[2] * right[0] + 2 * left[1] * right[1] + left[0] * right[2]) / duration**2
    e3 = (
        4.0
        * (
            left[3] * right[0]
            + 3 * left[2] * right[1]
            + 3 * left[1] * right[2]
            + left[0] * right[3]
        )
        / duration**3
    )
    return e0, e1, e2, e3


def _candidate(
    key: jax.Array, horizon: int, control_freq_hz: int, config: TrajectoryConfig
) -> tuple[Trajectory, jax.Array]:
    time = jnp.arange(horizon + 1, dtype=jnp.float32) / control_freq_hz
    duration = horizon / control_freq_hz
    coefficient_key, phase_key = jax.random.split(key)
    harmonics = jnp.arange(1, config.harmonics + 1, dtype=time.dtype)
    amplitude = jnp.asarray(config.amplitude_m, dtype=time.dtype)
    coefficients = (
        jax.random.uniform(coefficient_key, (3, config.harmonics), minval=-1.0, maxval=1.0)
        * amplitude[:, None]
        / jnp.sqrt(config.harmonics)
    )
    phases = jax.random.uniform(phase_key, (3, config.harmonics), minval=-jnp.pi, maxval=jnp.pi)
    omega = 2.0 * jnp.pi * harmonics / duration
    angle = time[:, None, None] * omega[None, None, :] + phases[None, :, :]
    sin_angle, cos_angle = jnp.sin(angle), jnp.cos(angle)
    b0 = jnp.sum(coefficients[None] * sin_angle, axis=-1)
    b1 = jnp.sum(coefficients[None] * omega[None, None] * cos_angle, axis=-1)
    b2 = jnp.sum(-coefficients[None] * omega[None, None] ** 2 * sin_angle, axis=-1)
    b3 = jnp.sum(-coefficients[None] * omega[None, None] ** 3 * cos_angle, axis=-1)
    e0, e1, e2, e3 = (value[:, None] for value in _envelope(time, duration))
    center = jnp.asarray(config.center_m, dtype=time.dtype)
    pos = center + e0 * b0
    vel = e1 * b0 + e0 * b1
    acc = e2 * b0 + 2 * e1 * b1 + e0 * b2
    jerk = e3 * b0 + 3 * e2 * b1 + 3 * e1 * b2 + e0 * b3
    zeros = jnp.zeros_like(time)
    return Trajectory(time=time, pos=pos, vel=vel, acc=acc, yaw=zeros, yaw_rate=zeros), jerk


def _duration_horizon(duration_s: float, control_freq_hz: int, name: str) -> int:
    samples = duration_s * control_freq_hz
    rounded = round(samples)
    if not math.isclose(samples, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must be an integer number of control intervals")
    return int(rounded)


def _candidate_and_validation_parent(
    key: jax.Array, horizon: int, control_freq_hz: int, config: TrajectoryConfig
) -> tuple[Trajectory, jax.Array, Trajectory, jax.Array, str, int]:
    if config.distribution_version == LEGACY_TRAJECTORY_DISTRIBUTION:
        trajectory, jerk = _candidate(key, horizon, control_freq_hz, config)
        return trajectory, jerk, trajectory, jerk, "returned_trajectory", horizon
    if config.distribution_version != FIXED_SUPPORT_PREFIX_DISTRIBUTION:
        raise ValueError(
            f"unsupported trajectory distribution_version: {config.distribution_version!r}"
        )
    requested_duration_s = horizon / control_freq_hz
    if requested_duration_s < config.minimum_duration_s:
        raise ValueError(
            "fixed_support_prefix_v2 requested duration "
            f"{requested_duration_s:g}s is below minimum_duration_s={config.minimum_duration_s:g}s"
        )
    if requested_duration_s > config.support_duration_s:
        raise ValueError(
            "fixed_support_prefix_v2 requested duration "
            f"{requested_duration_s:g}s exceeds support_duration_s={config.support_duration_s:g}s"
        )
    support_horizon = _duration_horizon(
        config.support_duration_s, control_freq_hz, "support_duration_s"
    )
    parent, parent_jerk = _candidate(key, support_horizon, control_freq_hz, config)
    prefix_length = horizon + 1
    trajectory = jax.tree.map(lambda value: value[:prefix_length], parent)
    jerk = parent_jerk[:prefix_length]
    return (
        trajectory,
        jerk,
        parent,
        parent_jerk,
        "complete_fixed_support_parent_before_prefix",
        support_horizon,
    )


def trajectory_diagnostics(trajectory: Trajectory, jerk: jax.Array) -> TrajectoryDiagnostics:
    """Compute host-side physical filter diagnostics."""
    acceleration = np.asarray(trajectory.acc)
    specific_force = acceleration + np.array([0.0, 0.0, 9.81])
    horizontal = np.linalg.norm(specific_force[:, :2], axis=-1)
    tilt = np.arctan2(horizontal, specific_force[:, 2])
    return TrajectoryDiagnostics(
        max_speed_m_s=float(np.max(np.linalg.norm(np.asarray(trajectory.vel), axis=-1))),
        max_acceleration_m_s2=float(np.max(np.linalg.norm(acceleration, axis=-1))),
        max_jerk_m_s3=float(np.max(np.linalg.norm(np.asarray(jerk), axis=-1))),
        max_yaw_rate_rad_s=float(np.max(np.abs(np.asarray(trajectory.yaw_rate)))),
        max_tilt_rad=float(np.max(np.abs(tilt))),
        min_specific_force_m_s2=float(np.min(np.linalg.norm(specific_force, axis=-1))),
    )


def _constraint(
    *, name: str, relation: str, limit: float, observed: float, sample_index: int, time: np.ndarray
) -> ConstraintEvaluation:
    if relation == "<=":
        acceptance_margin = limit - observed
    elif relation == ">=":
        acceptance_margin = observed - limit
    else:  # pragma: no cover - all callers use one of the two explicit relations
        raise ValueError(f"unsupported constraint relation: {relation}")
    return ConstraintEvaluation(
        name=name,
        relation=relation,
        limit=float(limit),
        observed=float(observed),
        observed_minus_limit=float(observed - limit),
        acceptance_margin=float(acceptance_margin),
        violation_magnitude=float(max(0.0, -acceptance_margin)),
        sample_index=int(sample_index),
        time_seconds=float(time[sample_index]),
        passed=bool(acceptance_margin >= 0.0),
    )


def trajectory_constraint_evaluations(
    trajectory: Trajectory, jerk: jax.Array, config: TrajectoryConfig
) -> tuple[ConstraintEvaluation, ...]:
    """Return all legacy-v1 checks with signed margins and extremal sample indices."""
    time = np.asarray(trajectory.time)
    position = np.asarray(trajectory.pos)
    velocity = np.asarray(trajectory.vel)
    acceleration = np.asarray(trajectory.acc)
    jerk_array = np.asarray(jerk)
    yaw_rate = np.asarray(trajectory.yaw_rate)
    specific_force = acceleration + np.array([0.0, 0.0, 9.81])
    specific_force_norm = np.linalg.norm(specific_force, axis=-1)
    horizontal = np.linalg.norm(specific_force[:, :2], axis=-1)
    tilt = np.abs(np.arctan2(horizontal, specific_force[:, 2]))
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    jerk_norm = np.linalg.norm(jerk_array, axis=-1)
    abs_yaw_rate = np.abs(yaw_rate)
    finite_values = (position, velocity, acceleration, yaw_rate)
    finite = all(np.all(np.isfinite(value)) for value in finite_values)
    constraints = [
        ConstraintEvaluation(
            name="legacy_finite_position_velocity_acceleration_yaw_rate",
            relation="finite",
            limit=None,
            observed=None,
            observed_minus_limit=None,
            acceptance_margin=None,
            violation_magnitude=0.0 if finite else float("inf"),
            sample_index=None,
            time_seconds=None,
            passed=finite,
        )
    ]
    axis_names = ("x", "y", "z")
    for axis, axis_name in enumerate(axis_names):
        lower_index = int(np.argmin(position[:, axis]))
        upper_index = int(np.argmax(position[:, axis]))
        constraints.extend(
            (
                _constraint(
                    name=f"workspace_{axis_name}_minimum_m",
                    relation=">=",
                    limit=config.workspace_min_m[axis],
                    observed=position[lower_index, axis],
                    sample_index=lower_index,
                    time=time,
                ),
                _constraint(
                    name=f"workspace_{axis_name}_maximum_m",
                    relation="<=",
                    limit=config.workspace_max_m[axis],
                    observed=position[upper_index, axis],
                    sample_index=upper_index,
                    time=time,
                ),
            )
        )
    maximum_checks = (
        ("speed_m_s", config.max_speed_m_s, speed),
        ("acceleration_m_s2", config.max_acceleration_m_s2, acceleration_norm),
        ("jerk_m_s3", config.max_jerk_m_s3, jerk_norm),
        ("yaw_rate_rad_s", config.max_yaw_rate_rad_s, abs_yaw_rate),
        ("tilt_rad", config.max_tilt_rad, tilt),
    )
    for name, limit, values in maximum_checks:
        sample_index = int(np.argmax(values))
        constraints.append(
            _constraint(
                name=name,
                relation="<=",
                limit=limit,
                observed=values[sample_index],
                sample_index=sample_index,
                time=time,
            )
        )
    minimum_specific_force_index = int(np.argmin(specific_force_norm))
    constraints.append(
        _constraint(
            name="specific_force_m_s2",
            relation=">=",
            limit=config.min_specific_force_m_s2,
            observed=specific_force_norm[minimum_specific_force_index],
            sample_index=minimum_specific_force_index,
            time=time,
        )
    )
    return tuple(constraints)


def _array_digest(value: jax.Array | np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": digest.hexdigest()}


def _vector_statistics(value: np.ndarray) -> dict[str, object]:
    norms = np.linalg.norm(value, axis=-1)
    return {
        "component_rms": np.sqrt(np.mean(np.square(value), axis=0)).tolist(),
        "component_abs_max": np.max(np.abs(value), axis=0).tolist(),
        "vector_norm_rms": float(np.sqrt(np.mean(np.square(norms)))),
        "vector_norm_max": float(np.max(norms)),
    }


def _spectral_statistics(position_displacement: np.ndarray, sample_frequency_hz: int) -> dict:
    centered = position_displacement - np.mean(position_displacement, axis=0, keepdims=True)
    frequencies = np.fft.rfftfreq(centered.shape[0], d=1.0 / sample_frequency_hz)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequencies = frequencies[1:]
    power = power[1:]
    per_axis = []
    for axis in range(position_displacement.shape[1]):
        axis_power = power[:, axis]
        total_power = float(np.sum(axis_power))
        if total_power == 0.0 or not np.isfinite(total_power):
            per_axis.append(
                {"dominant_frequency_hz": None, "spectral_centroid_hz": None, "power": 0.0}
            )
            continue
        per_axis.append(
            {
                "dominant_frequency_hz": float(frequencies[int(np.argmax(axis_power))]),
                "spectral_centroid_hz": float(np.sum(frequencies * axis_power) / total_power),
                "power": total_power,
            }
        )
    return {
        "method": "one_sided_dft_of_mean_removed_enveloped_position",
        "frequency_resolution_hz": (float(frequencies[0]) if frequencies.size else None),
        "per_axis": per_axis,
    }


def trajectory_statistics(
    trajectory: Trajectory,
    jerk: jax.Array,
    *,
    center_m: tuple[float, float, float],
    control_freq_hz: int,
) -> dict[str, object]:
    """Return construction-only magnitude, span, and descriptive spectral statistics."""
    position = np.asarray(trajectory.pos)
    displacement = position - np.asarray(center_m)
    velocity = np.asarray(trajectory.vel)
    acceleration = np.asarray(trajectory.acc)
    jerk_array = np.asarray(jerk)
    return {
        "position_m": {
            "component_min": np.min(position, axis=0).tolist(),
            "component_max": np.max(position, axis=0).tolist(),
            "component_span": np.ptp(position, axis=0).tolist(),
            **_vector_statistics(position),
        },
        "position_displacement_m": _vector_statistics(displacement),
        "velocity_m_s": _vector_statistics(velocity),
        "acceleration_m_s2": _vector_statistics(acceleration),
        "jerk_m_s3": _vector_statistics(jerk_array),
        "spectrum": _spectral_statistics(displacement, control_freq_hz),
    }


def diagnose_trajectory_attempts(
    key: jax.Array,
    horizon: int,
    control_freq_hz: int,
    config: TrajectoryConfig,
    *,
    max_attempts: int | None = None,
) -> tuple[TrajectoryAttemptDiagnostics, ...]:
    """Construct candidates only and report each attempt through the first acceptance."""
    config.validate()
    if horizon < 2 or control_freq_hz < 1:
        raise ValueError("horizon must be at least two and control frequency positive")
    attempt_limit = config.max_attempts if max_attempts is None else max_attempts
    if attempt_limit < 1:
        raise ValueError("diagnostic max_attempts must be positive")
    reports = []
    for attempt in range(attempt_limit):
        attempt_key = jax.random.fold_in(key, attempt)
        trajectory, jerk, validation, validation_jerk, validation_scope, validation_horizon = (
            _candidate_and_validation_parent(attempt_key, horizon, control_freq_hz, config)
        )
        constraints = trajectory_constraint_evaluations(validation, validation_jerk, config)
        accepted = all(item.passed for item in constraints)
        reports.append(
            TrajectoryAttemptDiagnostics(
                attempt_index=attempt,
                attempt_count=attempt + 1,
                accepted=accepted,
                validation_scope=validation_scope,
                returned_horizon_control_intervals=horizon,
                validation_horizon_control_intervals=validation_horizon,
                constraints=constraints,
                statistics=trajectory_statistics(
                    trajectory, jerk, center_m=config.center_m, control_freq_hz=control_freq_hz
                ),
                array_digests={
                    "position": _array_digest(trajectory.pos),
                    "velocity": _array_digest(trajectory.vel),
                    "acceleration": _array_digest(trajectory.acc),
                    "jerk": _array_digest(jerk),
                },
                validation_array_digests={
                    "position": _array_digest(validation.pos),
                    "velocity": _array_digest(validation.vel),
                    "acceleration": _array_digest(validation.acc),
                    "jerk": _array_digest(validation_jerk),
                },
            )
        )
        if accepted:
            break
    return tuple(reports)


def _valid(
    trajectory: Trajectory, diagnostics: TrajectoryDiagnostics, config: TrajectoryConfig
) -> bool:
    position = np.asarray(trajectory.pos)
    finite = all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (trajectory.pos, trajectory.vel, trajectory.acc, trajectory.yaw_rate)
    )
    in_workspace = np.all(position >= np.asarray(config.workspace_min_m)) and np.all(
        position <= np.asarray(config.workspace_max_m)
    )
    return bool(
        finite
        and in_workspace
        and diagnostics.max_speed_m_s <= config.max_speed_m_s
        and diagnostics.max_acceleration_m_s2 <= config.max_acceleration_m_s2
        and diagnostics.max_jerk_m_s3 <= config.max_jerk_m_s3
        and diagnostics.max_yaw_rate_rad_s <= config.max_yaw_rate_rad_s
        and diagnostics.max_tilt_rad <= config.max_tilt_rad
        and diagnostics.min_specific_force_m_s2 >= config.min_specific_force_m_s2
    )


def generate_trajectory(
    key: jax.Array, horizon: int, control_freq_hz: int, config: TrajectoryConfig
) -> GeneratedTrajectory:
    """Generate a valid sample with finite, deterministic host rejection."""
    config.validate()
    if horizon < 2 or control_freq_hz < 1:
        raise ValueError("horizon must be at least two and control frequency positive")
    for attempt in range(config.max_attempts):
        attempt_key = jax.random.fold_in(key, attempt)
        trajectory, jerk, validation, validation_jerk, _, _ = _candidate_and_validation_parent(
            attempt_key, horizon, control_freq_hz, config
        )
        validation_diagnostics = trajectory_diagnostics(validation, validation_jerk)
        if _valid(validation, validation_diagnostics, config):
            diagnostics = trajectory_diagnostics(trajectory, jerk)
            return GeneratedTrajectory(trajectory, jerk, diagnostics, attempt)
    raise ValueError(f"no valid trajectory found in {config.max_attempts} deterministic attempts")


def stack_trajectories(trajectories: tuple[Trajectory, ...]) -> Trajectory:
    """Stack one trajectory per world into ``(T, N, 1, ...)`` layout."""
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    return jax.tree.map(lambda *values: jnp.stack(values, axis=1)[:, :, None, ...], *trajectories)
