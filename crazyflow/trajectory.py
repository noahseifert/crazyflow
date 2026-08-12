"""Analytic, JAX-compatible reference trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax.struct import dataclass

if TYPE_CHECKING:
    from jax import Array


@dataclass
class Trajectory:
    """Sampled reference trajectory in SI units.

    Every field uses the same leading time shape ``(T,)``. Position, velocity,
    and acceleration have a final Cartesian axis of length three.
    """

    time: Array
    pos: Array
    vel: Array
    acc: Array
    yaw: Array
    yaw_rate: Array


@dataclass
class CircleTrajectoryParams:
    """Parameters for a horizontal circle.

    Attributes:
        center: Circle center ``[x, y, z]`` in metres.
        radius: Circle radius in metres.
        period: Steady-state revolution period in seconds.
        ramp_duration: Duration in seconds over which angular speed ramps smoothly from zero.
        yaw: Constant yaw reference in radians.
    """

    center: Array
    radius: Array
    period: Array
    ramp_duration: Array
    yaw: Array


@dataclass
class Figure8TrajectoryParams:
    """Parameters for a horizontal figure-eight.

    Attributes:
        center: Figure-eight center ``[x, y, z]`` in metres.
        amplitude: Positive ``[x, y]`` amplitudes in metres.
        period: Period of the x component in seconds.
        ramp_duration: Duration in seconds over which phase rate ramps smoothly from zero.
        yaw: Constant yaw reference in radians.
    """

    center: Array
    amplitude: Array
    period: Array
    ramp_duration: Array
    yaw: Array


def _smooth_phase(time: Array, period: Array, ramp_duration: Array) -> tuple[Array, Array, Array]:
    """Return phase and its first two analytic time derivatives.

    A quintic smoothstep ramps angular speed from zero to ``2π / period``. Its
    analytic integral defines phase, so phase, phase rate, and phase
    acceleration are continuous at both ends of the ramp. The piecewise
    ``where`` is therefore smooth through second order, despite being an
    explicit branch.
    """
    omega = 2.0 * jnp.pi / period
    u = time / ramp_duration
    u_ramp = jnp.clip(u, 0.0, 1.0)

    rate_scale = 10.0 * u_ramp**3 - 15.0 * u_ramp**4 + 6.0 * u_ramp**5
    rate_scale_derivative = 30.0 * u_ramp**2 - 60.0 * u_ramp**3 + 30.0 * u_ramp**4
    integrated_rate_scale = 2.5 * u_ramp**4 - 3.0 * u_ramp**5 + u_ramp**6

    in_ramp = time < ramp_duration
    phase = jnp.where(
        in_ramp, omega * ramp_duration * integrated_rate_scale, omega * (time - 0.5 * ramp_duration)
    )
    phase_rate = jnp.where(in_ramp, omega * rate_scale, omega)
    phase_acc = jnp.where(in_ramp, omega * rate_scale_derivative / ramp_duration, 0.0)
    return phase, phase_rate, phase_acc


def circle_trajectory(time: Array, params: CircleTrajectoryParams) -> Trajectory:
    """Generate a smooth horizontal circle with analytic derivatives.

    Args:
        time: One-dimensional sample times in seconds, starting at or after zero.
        params: Circle geometry and timing.

    Returns:
        Position in metres, velocity in m/s, acceleration in m/s², yaw in
        radians, and yaw rate in rad/s.
    """
    time = jnp.asarray(time)
    dtype = time.dtype
    center = jnp.asarray(params.center, dtype=dtype)
    radius = jnp.asarray(params.radius, dtype=dtype)
    phase, phase_rate, phase_acc = _smooth_phase(
        time,
        jnp.asarray(params.period, dtype=dtype),
        jnp.asarray(params.ramp_duration, dtype=dtype),
    )

    cos_phase, sin_phase = jnp.cos(phase), jnp.sin(phase)
    zeros = jnp.zeros_like(time)
    pos = center + jnp.stack((radius * cos_phase, radius * sin_phase, zeros), axis=-1)
    vel = jnp.stack(
        (-radius * sin_phase * phase_rate, radius * cos_phase * phase_rate, zeros), axis=-1
    )
    acc = jnp.stack(
        (
            -radius * (cos_phase * phase_rate**2 + sin_phase * phase_acc),
            radius * (-sin_phase * phase_rate**2 + cos_phase * phase_acc),
            zeros,
        ),
        axis=-1,
    )
    yaw = jnp.broadcast_to(jnp.asarray(params.yaw, dtype=dtype), time.shape)
    return Trajectory(time=time, pos=pos, vel=vel, acc=acc, yaw=yaw, yaw_rate=zeros)


def figure8_trajectory(time: Array, params: Figure8TrajectoryParams) -> Trajectory:
    """Generate a smooth horizontal figure-eight with analytic derivatives.

    The curve is ``x = ax sin(phase)``, ``y = ay sin(2 phase)``. The phase-rate
    ramp makes position, velocity, and acceleration start without a jump.
    """
    time = jnp.asarray(time)
    dtype = time.dtype
    center = jnp.asarray(params.center, dtype=dtype)
    amplitude = jnp.asarray(params.amplitude, dtype=dtype)
    phase, phase_rate, phase_acc = _smooth_phase(
        time,
        jnp.asarray(params.period, dtype=dtype),
        jnp.asarray(params.ramp_duration, dtype=dtype),
    )

    sin_phase, cos_phase = jnp.sin(phase), jnp.cos(phase)
    sin_2phase, cos_2phase = jnp.sin(2.0 * phase), jnp.cos(2.0 * phase)
    zeros = jnp.zeros_like(time)
    pos = center + jnp.stack((amplitude[0] * sin_phase, amplitude[1] * sin_2phase, zeros), axis=-1)
    vel = jnp.stack(
        (
            amplitude[0] * cos_phase * phase_rate,
            2.0 * amplitude[1] * cos_2phase * phase_rate,
            zeros,
        ),
        axis=-1,
    )
    acc = jnp.stack(
        (
            amplitude[0] * (-sin_phase * phase_rate**2 + cos_phase * phase_acc),
            2.0 * amplitude[1] * (-2.0 * sin_2phase * phase_rate**2 + cos_2phase * phase_acc),
            zeros,
        ),
        axis=-1,
    )
    yaw = jnp.broadcast_to(jnp.asarray(params.yaw, dtype=dtype), time.shape)
    return Trajectory(time=time, pos=pos, vel=vel, acc=acc, yaw=yaw, yaw_rate=zeros)


def state_commands(trajectory: Trajectory) -> Array:
    """Convert a trajectory to Crazyflow state commands with shape ``(T, 13)``.

    Crazyflow currently ignores desired body rates in the Mellinger state
    stage. Roll and pitch rates are therefore zero; the analytic yaw rate is
    nevertheless populated for forward compatibility.
    """
    zero_rates = jnp.zeros((*trajectory.time.shape, 2), dtype=trajectory.time.dtype)
    return jnp.concat(
        (
            trajectory.pos,
            trajectory.vel,
            trajectory.acc,
            trajectory.yaw[..., None],
            zero_rates,
            trajectory.yaw_rate[..., None],
        ),
        axis=-1,
    )


def broadcast_state_commands(commands: Array, n_worlds: int, n_drones: int) -> Array:
    """Broadcast ``(T, 13)`` commands to Crazyflow's ``(T, N, M, 13)`` layout."""
    if commands.ndim != 2 or commands.shape[-1] != 13:
        raise ValueError(f"Expected commands with shape (T, 13), got {commands.shape}")
    return jnp.broadcast_to(commands[:, None, None, :], (commands.shape[0], n_worlds, n_drones, 13))
