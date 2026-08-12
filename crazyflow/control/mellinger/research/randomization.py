"""Episode-scoped mass, action-delay, and correlated-wrench components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData


def sample_world_masses(
    keys: Array, nominal_mass_kg: float, half_width_kg: float, n_drones: int = 1
) -> Array:
    """Sample one constant true mass per world, broadcast over drones."""
    if nominal_mass_kg <= 0.0 or half_width_kg < 0.0 or half_width_kg >= nominal_mass_kg:
        raise ValueError("mass interval must be positive")
    samples = jax.vmap(
        lambda key: jax.random.uniform(
            key, (), minval=nominal_mass_kg - half_width_kg, maxval=nominal_mass_kg + half_width_kg
        )
    )(keys)
    return jnp.broadcast_to(samples[:, None, None], (keys.shape[0], n_drones, 1))


def apply_true_mass(data: SimData, masses_kg: Array) -> SimData:
    """Replace only dynamics mass; controller mass remains unchanged."""
    if masses_kg.shape != data.params.mass.shape:
        raise ValueError(f"expected mass shape {data.params.mass.shape}, got {masses_kg.shape}")
    if not bool(jnp.all(masses_kg > 0.0)):
        raise ValueError("all masses must be positive")
    return data.replace(params=data.params.replace(mass=masses_kg))


def sample_delay_steps(keys: Array, maximum: int) -> Array:
    """Sample integer control-interval delays independently per world."""
    if maximum < 0:
        raise ValueError("maximum delay must be nonnegative")
    if maximum == 0:
        return jnp.zeros((keys.shape[0],), dtype=jnp.int32)
    return jax.vmap(lambda key: jax.random.randint(key, (), 0, maximum + 1))(keys)


def apply_action_delay(commands: Array, delay_steps: Array) -> Array:
    """Delay commands per world with the episode's first command as buffer fill.

    A delay of ``d`` means output interval ``t`` uses input ``max(t-d, 0)``.
    Therefore zero delay is exactly the identity and no stale state crosses an
    episode boundary.
    """
    if commands.ndim != 4 or commands.shape[-1] != 13:
        raise ValueError("commands must have shape (T, N, M, 13)")
    if delay_steps.shape != (commands.shape[1],):
        raise ValueError(f"expected delay shape {(commands.shape[1],)}, got {delay_steps.shape}")
    if not bool(jnp.all(delay_steps >= 0)):
        raise ValueError("delays must be nonnegative")
    indices = jnp.maximum(jnp.arange(commands.shape[0])[:, None] - delay_steps[None, :], 0)
    indices = jnp.broadcast_to(indices[:, :, None, None], commands.shape)
    return jnp.take_along_axis(commands, indices, axis=0)


def sample_correlated_wrenches(
    keys: Array,
    horizon: int,
    n_drones: int,
    control_dt_s: float,
    force_std_n: float,
    torque_std_nm: float,
    correlation_time_s: float,
) -> tuple[Array, Array]:
    """Sample stationary AR(1) world-frame forces [N] and torques [N m]."""
    if horizon < 1 or n_drones < 1 or control_dt_s <= 0.0 or correlation_time_s <= 0.0:
        raise ValueError("horizon/drones/durations must be positive")
    if force_std_n < 0.0 or torque_std_nm < 0.0:
        raise ValueError("wrench standard deviations must be nonnegative")
    world_count = keys.shape[0]
    if force_std_n == 0.0 and torque_std_nm == 0.0:
        zeros = jnp.zeros((horizon, world_count, n_drones, 3), dtype=jnp.float32)
        return zeros, zeros
    rho = jnp.exp(-control_dt_s / correlation_time_s)
    innovation_scale = jnp.sqrt(1.0 - rho**2)

    def one_world(key: Array) -> tuple[Array, Array]:
        force_key, torque_key = jax.random.split(key)
        force_noise = jax.random.normal(force_key, (horizon, n_drones, 3))
        torque_noise = jax.random.normal(torque_key, (horizon, n_drones, 3))

        def ar1(noise: Array, scale: float) -> Array:
            def step(previous: Array, sample: Array) -> tuple[Array, Array]:
                current = rho * previous + innovation_scale * scale * sample
                return current, current

            initial = scale * noise[0]
            _, tail = jax.lax.scan(step, initial, noise[1:])
            return jnp.concatenate((initial[None], tail), axis=0)

        return ar1(force_noise, force_std_n), ar1(torque_noise, torque_std_nm)

    force, torque = jax.vmap(one_world)(keys)
    return jnp.swapaxes(force, 0, 1), jnp.swapaxes(torque, 0, 1)
