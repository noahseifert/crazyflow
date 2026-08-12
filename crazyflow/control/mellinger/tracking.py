"""Differentiable Mellinger tracking rollouts and losses."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass

import crazyflow.sim.functional as F
from crazyflow.control.transform import motor_force2rotor_vel

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData
    from crazyflow.trajectory import Trajectory


# The bounds are intentionally conservative around Crazyflow's cf2x defaults.
# kp has units N/m and kd has units N s/m because both terms contribute force.
KP_XY_RANGE = (0.10, 1.20)
KP_Z_RANGE = (0.30, 2.50)
KD_XY_RANGE = (0.05, 0.80)
KD_Z_RANGE = (0.10, 1.20)
GAIN_BOUNDS = {"kp_xy": KP_XY_RANGE, "kp_z": KP_Z_RANGE, "kd_xy": KD_XY_RANGE, "kd_z": KD_Z_RANGE}
TRACKING_LOSS_TERM_NAMES = ("position", "velocity", "effort", "smoothness", "terminal", "altitude")


@dataclass
class GainVariables:
    """Unconstrained scalar optimization variables.

    Every leaf is a floating-point scalar. Shared x/y variables preserve the
    symmetry of the Crazyflie firmware gains.
    """

    kp_xy: Array
    kp_z: Array
    kd_xy: Array
    kd_z: Array


@dataclass
class PositionVelocityGains:
    """Physical Mellinger gains after the smooth bounded transform."""

    kp: Array
    kd: Array


@dataclass
class RolloutTrace:
    """Selected per-step outputs from an actual Crazyflow rollout."""

    pos: Array
    quat: Array
    vel: Array
    ang_vel: Array
    rotor_vel: Array
    commanded_rotor_vel: Array
    state_command: Array
    attitude_command: Array
    force_torque_command: Array


@dataclass
class TrackingLossConfig:
    """Dimensionless tracking-loss weights and physical normalization scales."""

    position_weight: float = 1.0
    velocity_weight: float = 0.10
    effort_weight: float = 1.0e-3
    smoothness_weight: float = 1.0e-3
    terminal_weight: float = 0.10
    altitude_weight: float = 0.05
    position_scale: float = 0.25
    velocity_scale: float = 1.0
    altitude_margin: float = 0.15
    altitude_softness: float = 0.05


@dataclass
class TrackingLossTerms:
    """Per-case decomposition of the unchanged scalar tracking loss.

    ``raw`` retains the physical or controller-normalized quantity before the
    configured loss normalization and weight. ``normalization_divisor`` makes
    the normalization explicit, while ``normalized`` and ``weighted`` expose
    the two successive values used for Stage-2 diagnostics.
    """

    raw: dict[str, Array]
    normalization_divisor: dict[str, Array]
    normalized: dict[str, Array]
    weight: dict[str, Array]
    weighted: dict[str, Array]


def _bounded(raw: Array, bounds: tuple[float, float]) -> Array:
    lower, upper = bounds
    return lower + (upper - lower) * jax.nn.sigmoid(raw)


def _inverse_bounded(value: Array, bounds: tuple[float, float]) -> Array:
    lower, upper = bounds
    unit_value = (value - lower) / (upper - lower)
    # This safeguard only initializes variables; it is not in the differentiated objective.
    unit_value = jnp.clip(unit_value, 1.0e-6, 1.0 - 1.0e-6)
    return jnp.log(unit_value) - jnp.log1p(-unit_value)


def gain_variables_from_params(params: dict[str, Array]) -> GainVariables:
    """Initialize unconstrained variables from a Mellinger parameter dictionary."""
    kp, kd = params["kp"], params["kd"]
    return GainVariables(
        kp_xy=_inverse_bounded(kp[0], KP_XY_RANGE),
        kp_z=_inverse_bounded(kp[2], KP_Z_RANGE),
        kd_xy=_inverse_bounded(kd[0], KD_XY_RANGE),
        kd_z=_inverse_bounded(kd[2], KD_Z_RANGE),
    )


def physical_gains(variables: GainVariables) -> PositionVelocityGains:
    """Map unconstrained variables to positive, bounded, symmetric gains."""
    kp_xy = _bounded(variables.kp_xy, KP_XY_RANGE)
    kp_z = _bounded(variables.kp_z, KP_Z_RANGE)
    kd_xy = _bounded(variables.kd_xy, KD_XY_RANGE)
    kd_z = _bounded(variables.kd_z, KD_Z_RANGE)
    return PositionVelocityGains(
        kp=jnp.stack((kp_xy, kp_xy, kp_z)), kd=jnp.stack((kd_xy, kd_xy, kd_z))
    )


def apply_gain_variables(data: SimData, variables: GainVariables) -> SimData:
    """Return ``data`` with experiment-local Mellinger kp/kd values."""
    state_control = data.controls.state
    if state_control is None:
        raise ValueError("Mellinger gain variables require Control.state")
    gains = physical_gains(variables)
    params = state_control.params | {"kp": gains.kp, "kd": gains.kd}
    return data.replace(controls=data.controls.replace(state=state_control.replace(params=params)))


def hover_rotor_velocity(data: SimData) -> Array:
    """Return physical hover RPM for every world and drone."""
    force_torque_control = data.controls.force_torque
    if force_torque_control is None:
        raise ValueError("Hover RPM requires the first-principles force/torque controller")
    hover_force = data.params.mass * (-data.params.gravity_vec[..., [2]]) / 4.0
    motor_forces = jnp.broadcast_to(hover_force, (*hover_force.shape[:-1], 4))
    return motor_force2rotor_vel(motor_forces, force_torque_control.params["rpm2thrust"])


def initialize_tracking_state(
    data: SimData, initial_position: Array, initial_velocity: Array
) -> SimData:
    """Set a deterministic trajectory-matched state and physical hover rotor RPM."""
    initial_position = jnp.broadcast_to(initial_position, data.states.pos.shape)
    initial_velocity = jnp.broadcast_to(initial_velocity, data.states.vel.shape)
    states = data.states.replace(
        pos=initial_position, vel=initial_velocity, rotor_vel=hover_rotor_velocity(data)
    )
    return data.replace(states=states)


def rollout_state_commands(
    initial_data: SimData,
    commands: Array,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
) -> tuple[SimData, RolloutTrace]:
    """Roll out state commands through Crazyflow with ``jax.lax.scan``.

    Args:
        initial_data: Full ``SimData`` carry. It includes rigid-body and rotor
            state, Mellinger position/attitude integrators, previous angular
            velocity, control staging buffers, and controller step counters.
        commands: State commands with shape ``(T, n_worlds, n_drones, 13)``.
        step_fn: Pure function returned by ``Sim.build_step_fn()``.
        steps_per_command: Fixed number of dynamics ticks per command.

    Returns:
        Final ``SimData`` and a compact time-major trace.

    A Python loop inside ``jax.jit`` is normally unrolled while tracing, making
    compilation and the generated program grow with the horizon. ``lax.scan``
    represents a fixed-horizon recurrence as one loop primitive and keeps the
    carry's PyTree structure, shapes, and dtypes static. Reverse-mode autodiff
    can still retain residuals from every step. If long-horizon memory becomes
    limiting, a future ``jax.checkpoint``/``jax.remat`` boundary can recompute
    selected steps during the backward pass in exchange for extra compute.
    """
    expected_shape = (commands.shape[0], initial_data.core.n_worlds, initial_data.core.n_drones, 13)
    if commands.shape != expected_shape:
        raise ValueError(f"Expected commands with shape {expected_shape}, got {commands.shape}")
    if steps_per_command < 1:
        raise ValueError("steps_per_command must be positive")

    def scan_step(data: SimData, command: Array) -> tuple[SimData, RolloutTrace]:
        data = F.state_control(data, command)
        data = step_fn(data, steps_per_command)
        state_control = data.controls.state
        attitude_control = data.controls.attitude
        force_torque_control = data.controls.force_torque
        assert state_control is not None
        assert attitude_control is not None
        assert force_torque_control is not None
        trace = RolloutTrace(
            pos=data.states.pos,
            quat=data.states.quat,
            vel=data.states.vel,
            ang_vel=data.states.ang_vel,
            rotor_vel=data.states.rotor_vel,
            commanded_rotor_vel=data.controls.rotor_vel,
            state_command=state_control.cmd,
            attitude_command=attitude_control.cmd,
            force_torque_command=force_torque_control.cmd,
        )
        return data, trace

    return jax.lax.scan(scan_step, initial_data, commands)


def rotor_velocity_limits(data: SimData) -> tuple[Array, Array]:
    """Return controller motor-clipping limits in RPM."""
    force_torque_control = data.controls.force_torque
    if force_torque_control is None:
        raise ValueError("Rotor limits require the first-principles force/torque controller")
    params = force_torque_control.params
    minimum = motor_force2rotor_vel(params["thrust_min"], params["rpm2thrust"])
    maximum = motor_force2rotor_vel(params["thrust_max"], params["rpm2thrust"])
    return minimum, maximum


def tracking_loss(
    trace: RolloutTrace,
    reference: Trajectory,
    hover_rpm: Array,
    rotor_limits: tuple[Array, Array],
    config: TrackingLossConfig,
) -> tuple[Array, dict[str, Array]]:
    """Compute the mean normalized loss over worlds and aggregate diagnostics."""
    per_case_loss, per_case_metrics = tracking_loss_per_case(
        trace, reference, hover_rpm, rotor_limits, config
    )
    return jnp.mean(per_case_loss), aggregate_tracking_metrics(per_case_metrics)


def _batch_reference_field(value: Array, target_shape: tuple[int, ...], name: str) -> Array:
    """Return a reference field in time/world/drone layout."""
    if value.ndim == 2:
        value = value[:, None, None, :]
    try:
        value = jnp.broadcast_to(value, target_shape)
    except ValueError:
        raise ValueError(f"Expected reference {name} with shape {target_shape}, got {value.shape}")
    return value


def tracking_loss_per_case(
    trace: RolloutTrace,
    reference: Trajectory,
    hover_rpm: Array,
    rotor_limits: tuple[Array, Array],
    config: TrackingLossConfig,
) -> tuple[Array, dict[str, Array]]:
    """Compute the unchanged normalized tracking loss separately for each world.

    A conventional trajectory with fields shaped ``(T, 3)`` is accepted for
    the existing single-reference behavior. A batched reference uses
    ``(T, N, M, 3)`` position and velocity fields matching the rollout.
    Reductions retain the world axis and average over time and drones.
    """
    reference_pos = _batch_reference_field(reference.pos, trace.pos.shape, "position")
    reference_vel = _batch_reference_field(reference.vel, trace.vel.shape, "velocity")
    pos_error = trace.pos - reference_pos
    vel_error = trace.vel - reference_vel

    loss_terms = tracking_loss_terms_per_case(trace, reference, hover_rpm, config)
    loss_position = loss_terms.weighted["position"]
    loss_velocity = loss_terms.weighted["velocity"]
    loss_effort = loss_terms.weighted["effort"]
    loss_smoothness = loss_terms.weighted["smoothness"]
    loss_terminal = loss_terms.weighted["terminal"]
    loss_altitude = loss_terms.weighted["altitude"]
    total = (
        loss_position
        + loss_velocity
        + loss_effort
        + loss_smoothness
        + loss_terminal
        + loss_altitude
    )

    position_mse = jnp.mean(jnp.sum(pos_error**2, axis=-1), axis=(0, 2))
    velocity_mse = jnp.mean(jnp.sum(vel_error**2, axis=-1), axis=(0, 2))
    control_effort = loss_terms.raw["effort"]
    control_smoothness = loss_terms.raw["smoothness"]

    minimum_rpm, maximum_rpm = rotor_limits
    tolerance = 1.0e-4
    lower_saturation = (trace.commanded_rotor_vel > 0.5 * minimum_rpm) & (
        trace.commanded_rotor_vel <= minimum_rpm * (1.0 + tolerance)
    )
    upper_saturation = trace.commanded_rotor_vel >= maximum_rpm * (1.0 - tolerance)
    saturation_fraction = jnp.mean(lower_saturation | upper_saturation, axis=(0, 2, 3))

    finite_arrays = (
        trace.pos,
        trace.quat,
        trace.vel,
        trace.ang_vel,
        trace.rotor_vel,
        trace.commanded_rotor_vel,
    )
    nonfinite_count = sum(
        jnp.sum(~jnp.isfinite(value), axis=(0, *range(2, value.ndim))) for value in finite_arrays
    )
    value_count = sum(
        value.shape[0] * value.shape[2] * prod(value.shape[3:]) for value in finite_arrays
    )

    metrics = {
        "loss_total": total,
        "loss_position": loss_position,
        "loss_velocity": loss_velocity,
        "loss_effort": loss_effort,
        "loss_smoothness": loss_smoothness,
        "loss_terminal": loss_terminal,
        "loss_altitude": loss_altitude,
        "position_rmse_m": jnp.sqrt(position_mse),
        "velocity_rmse_m_s": jnp.sqrt(velocity_mse),
        "max_position_error_m": jnp.max(jnp.linalg.vector_norm(pos_error, axis=-1), axis=(0, 2)),
        "control_effort": control_effort,
        "control_smoothness": control_smoothness,
        "motor_saturation_fraction": saturation_fraction,
        "zero_thrust_gate_fraction": jnp.mean(trace.attitude_command[..., 3] <= 0.0, axis=(0, 2)),
        "floor_clip_fraction": jnp.mean(trace.pos[..., 2] <= -0.001, axis=(0, 2)),
        "nonfinite_state_fraction": nonfinite_count / value_count,
    }
    return total, metrics


def tracking_loss_terms_per_case(
    trace: RolloutTrace, reference: Trajectory, hover_rpm: Array, config: TrackingLossConfig
) -> TrackingLossTerms:
    """Return the six existing loss terms without changing their arithmetic."""
    reference_pos = _batch_reference_field(reference.pos, trace.pos.shape, "position")
    reference_vel = _batch_reference_field(reference.vel, trace.vel.shape, "velocity")
    pos_error = trace.pos - reference_pos
    vel_error = trace.vel - reference_vel

    position_mse = jnp.mean(jnp.sum(pos_error**2, axis=-1), axis=(0, 2))
    velocity_mse = jnp.mean(jnp.sum(vel_error**2, axis=-1), axis=(0, 2))
    terminal_mse = jnp.mean(jnp.sum(pos_error[-1] ** 2, axis=-1), axis=1)

    normalized_rpm = (trace.commanded_rotor_vel - hover_rpm) / hover_rpm
    control_effort = jnp.mean(normalized_rpm**2, axis=(0, 2, 3))
    if trace.commanded_rotor_vel.shape[0] > 1:
        rpm_delta = jnp.diff(trace.commanded_rotor_vel, axis=0) / hover_rpm
        control_smoothness = jnp.mean(rpm_delta**2, axis=(0, 2, 3))
    else:
        control_smoothness = jnp.zeros(trace.pos.shape[1], dtype=trace.pos.dtype)

    # This is a smooth alternative to a hard altitude/floor violation cost.
    altitude_argument = (config.altitude_margin - trace.pos[..., 2]) / config.altitude_softness
    altitude_penalty = jnp.mean(jax.nn.softplus(altitude_argument) ** 2, axis=(0, 2))

    raw = {
        "position": position_mse,
        "velocity": velocity_mse,
        "effort": control_effort,
        "smoothness": control_smoothness,
        "terminal": terminal_mse,
        "altitude": altitude_penalty,
    }
    dtype = trace.pos.dtype
    one = jnp.asarray(1.0, dtype=dtype)
    normalization_divisor = {
        "position": jnp.asarray(config.position_scale**2, dtype=dtype),
        "velocity": jnp.asarray(config.velocity_scale**2, dtype=dtype),
        "effort": one,
        "smoothness": one,
        "terminal": jnp.asarray(config.position_scale**2, dtype=dtype),
        "altitude": one,
    }
    weight = {
        "position": jnp.asarray(config.position_weight, dtype=dtype),
        "velocity": jnp.asarray(config.velocity_weight, dtype=dtype),
        "effort": jnp.asarray(config.effort_weight, dtype=dtype),
        "smoothness": jnp.asarray(config.smoothness_weight, dtype=dtype),
        "terminal": jnp.asarray(config.terminal_weight, dtype=dtype),
        "altitude": jnp.asarray(config.altitude_weight, dtype=dtype),
    }
    normalized = {
        name: raw[name] / normalization_divisor[name] for name in TRACKING_LOSS_TERM_NAMES
    }
    # Keep these expressions identical to the historical scalar objective.
    weighted = {
        "position": config.position_weight * position_mse / config.position_scale**2,
        "velocity": config.velocity_weight * velocity_mse / config.velocity_scale**2,
        "effort": config.effort_weight * control_effort,
        "smoothness": config.smoothness_weight * control_smoothness,
        "terminal": config.terminal_weight * terminal_mse / config.position_scale**2,
        "altitude": config.altitude_weight * altitude_penalty,
    }
    return TrackingLossTerms(raw, normalization_divisor, normalized, weight, weighted)


def aggregate_tracking_metrics(
    per_case_metrics: dict[str, Array], weights: Array | None = None
) -> dict[str, Array]:
    """Aggregate per-world diagnostics with optional nonnegative case weights."""
    case_count = per_case_metrics["loss_total"].shape[0]
    if weights is None:
        weights = jnp.ones(case_count, dtype=per_case_metrics["loss_total"].dtype)
    weights = jnp.asarray(weights, dtype=per_case_metrics["loss_total"].dtype)
    if weights.shape != (case_count,):
        raise ValueError(f"Expected weights with shape {(case_count,)}, got {weights.shape}")
    normalized_weights = weights / jnp.sum(weights)

    aggregated = {}
    for name, values in per_case_metrics.items():
        if name in {"position_rmse_m", "velocity_rmse_m_s"}:
            aggregated[name] = jnp.sqrt(jnp.sum(normalized_weights * values**2))
        elif name == "max_position_error_m":
            aggregated[name] = jnp.max(jnp.where(weights > 0.0, values, -jnp.inf))
        else:
            aggregated[name] = jnp.sum(normalized_weights * values)
    return aggregated


def tracking_objective_per_case(
    variables: GainVariables,
    initial_data: SimData,
    commands: Array,
    reference: Trajectory,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
    config: TrackingLossConfig,
) -> tuple[Array, dict[str, Array]]:
    """Run the Crazyflow pipeline and retain one loss and metrics row per world."""
    data = apply_gain_variables(initial_data, variables)
    _, trace = rollout_state_commands(data, commands, step_fn, steps_per_command)
    return tracking_loss_per_case(
        trace,
        reference,
        hover_rotor_velocity(initial_data),
        rotor_velocity_limits(initial_data),
        config,
    )


def tracking_objective(
    variables: GainVariables,
    initial_data: SimData,
    commands: Array,
    reference: Trajectory,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
    config: TrackingLossConfig,
) -> tuple[Array, dict[str, Array]]:
    """Run the real Crazyflow pipeline and return loss plus auxiliary metrics."""
    per_case_loss, per_case_metrics = tracking_objective_per_case(
        variables, initial_data, commands, reference, step_fn, steps_per_command, config
    )
    return jnp.mean(per_case_loss), aggregate_tracking_metrics(per_case_metrics)
