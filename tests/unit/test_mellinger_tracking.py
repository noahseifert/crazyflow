from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control
from crazyflow.control.mellinger import (
    TrackingLossConfig,
    apply_gain_variables,
    gain_variables_from_params,
    initialize_tracking_state,
    rollout_state_commands,
    tracking_objective,
)
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.trajectory import (
    CircleTrajectoryParams,
    Figure8TrajectoryParams,
    broadcast_state_commands,
    circle_trajectory,
    figure8_trajectory,
    state_commands,
)


def _circle_params() -> CircleTrajectoryParams:
    return CircleTrajectoryParams(
        center=jnp.array([0.0, 0.0, 0.75], dtype=jnp.float32),
        radius=jnp.asarray(0.25, dtype=jnp.float32),
        period=jnp.asarray(3.0, dtype=jnp.float32),
        ramp_duration=jnp.asarray(0.5, dtype=jnp.float32),
        yaw=jnp.asarray(0.0, dtype=jnp.float32),
    )


def _figure8_params() -> Figure8TrajectoryParams:
    return Figure8TrajectoryParams(
        center=jnp.array([0.0, 0.0, 0.75], dtype=jnp.float32),
        amplitude=jnp.array([0.30, 0.15], dtype=jnp.float32),
        period=jnp.asarray(3.0, dtype=jnp.float32),
        ramp_duration=jnp.asarray(0.5, dtype=jnp.float32),
        yaw=jnp.asarray(0.0, dtype=jnp.float32),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("generator", "params"),
    ((circle_trajectory, _circle_params()), (figure8_trajectory, _figure8_params())),
)
def test_trajectory_shapes_dtypes_and_analytic_derivatives(generator: Any, params: Any) -> None:
    time_vector = jnp.linspace(0.0, 1.0, 21, dtype=jnp.float32)
    trajectory = generator(time_vector, params)
    assert trajectory.pos.shape == (21, 3)
    assert trajectory.vel.shape == (21, 3)
    assert trajectory.acc.shape == (21, 3)
    assert trajectory.yaw.shape == (21,)
    assert trajectory.yaw_rate.shape == (21,)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(trajectory))
    assert jnp.allclose(trajectory.vel[0], 0.0)
    assert jnp.allclose(trajectory.acc[0], 0.0)

    check_times = jnp.array([0.1, 0.35, 0.5, 0.8], dtype=jnp.float32)

    def position_at_time(time: jax.Array) -> jax.Array:
        return generator(time[None], params).pos[0]

    def velocity_at_time(time: jax.Array) -> jax.Array:
        return generator(time[None], params).vel[0]

    autodiff_velocity = jax.vmap(jax.jacfwd(position_at_time))(check_times)
    autodiff_acceleration = jax.vmap(jax.jacfwd(velocity_at_time))(check_times)
    analytic = generator(check_times, params)
    assert jnp.allclose(autodiff_velocity, analytic.vel, rtol=2.0e-5, atol=2.0e-6)
    assert jnp.allclose(autodiff_acceleration, analytic.acc, rtol=2.0e-5, atol=2.0e-6)


@pytest.mark.unit
def test_state_command_shape_and_values() -> None:
    trajectory = figure8_trajectory(jnp.arange(5, dtype=jnp.float32) / 100.0, _figure8_params())
    commands = state_commands(trajectory)
    batched = broadcast_state_commands(commands, n_worlds=2, n_drones=3)
    assert commands.shape == (5, 13)
    assert batched.shape == (5, 2, 3, 13)
    assert jnp.allclose(commands[:, :3], trajectory.pos)
    assert jnp.allclose(commands[:, 3:6], trajectory.vel)
    assert jnp.allclose(commands[:, 6:9], trajectory.acc)
    assert jnp.allclose(commands[:, 9], trajectory.yaw)
    assert jnp.allclose(commands[:, 12], trajectory.yaw_rate)


@pytest.fixture(scope="module")
def tracking_case() -> dict[str, Any]:
    sim_frequency = 500
    control_frequency = 100
    horizon = 20
    time_vector = jnp.arange(horizon + 1, dtype=jnp.float32) / control_frequency
    full_reference = figure8_trajectory(time_vector, _figure8_params())
    reference = jax.tree.map(lambda value: value[1:], full_reference)
    sim = Sim(
        drone="cf2x_L250",
        dynamics=Dynamics.first_principles,
        control=Control.state,
        freq=sim_frequency,
        state_freq=control_frequency,
        attitude_freq=sim_frequency,
        force_torque_freq=sim_frequency,
        rng_key=7,
    )
    initial_data = initialize_tracking_state(sim.data, full_reference.pos[0], full_reference.vel[0])
    commands = broadcast_state_commands(state_commands(reference), 1, 1)
    state_control = initial_data.controls.state
    assert state_control is not None
    variables = gain_variables_from_params(state_control.params)
    step_fn = sim.build_step_fn()
    steps_per_command = sim_frequency // control_frequency
    objective = partial(
        tracking_objective,
        initial_data=initial_data,
        commands=commands,
        reference=reference,
        step_fn=step_fn,
        steps_per_command=steps_per_command,
        config=TrackingLossConfig(),
    )
    loss_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    (loss, metrics), gradient = loss_and_grad(variables)
    jax.block_until_ready((loss, metrics, gradient))
    return {
        "sim": sim,
        "initial_data": initial_data,
        "commands": commands,
        "reference": reference,
        "variables": variables,
        "step_fn": step_fn,
        "steps_per_command": steps_per_command,
        "objective": objective,
        "loss": loss,
        "metrics": metrics,
        "gradient": gradient,
    }


@pytest.mark.unit
def test_real_crazyflow_rollout_shapes(tracking_case: dict[str, Any]) -> None:
    final_data, trace = rollout_state_commands(
        apply_gain_variables(tracking_case["initial_data"], tracking_case["variables"]),
        tracking_case["commands"],
        tracking_case["step_fn"],
        tracking_case["steps_per_command"],
    )
    horizon = tracking_case["commands"].shape[0]
    assert trace.pos.shape == (horizon, 1, 1, 3)
    assert trace.quat.shape == (horizon, 1, 1, 4)
    assert trace.vel.shape == (horizon, 1, 1, 3)
    assert trace.commanded_rotor_vel.shape == (horizon, 1, 1, 4)
    assert final_data.core.steps.shape == (1, 1)
    assert int(final_data.core.steps[0, 0]) == horizon * tracking_case["steps_per_command"]


@pytest.mark.unit
def test_tracking_loss_and_gradient(tracking_case: dict[str, Any]) -> None:
    loss = tracking_case["loss"]
    metrics = tracking_case["metrics"]
    gradient = tracking_case["gradient"]
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert metrics["nonfinite_state_fraction"] == 0.0
    assert jax.tree.structure(gradient) == jax.tree.structure(tracking_case["variables"])
    assert all(
        jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.shape == ()
        for leaf in jax.tree.leaves(tracking_case["variables"])
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    gradient_norm = jnp.sqrt(sum(jnp.vdot(leaf, leaf) for leaf in jax.tree.leaves(gradient)))
    assert 1.0e-8 < gradient_norm < 1.0e4


@pytest.mark.unit
def test_rollout_recurrence_is_a_scan(tracking_case: dict[str, Any]) -> None:
    jaxpr = jax.make_jaxpr(lambda variables: tracking_case["objective"](variables)[0])(
        tracking_case["variables"]
    )
    assert "scan[" in str(jaxpr)


@pytest.mark.unit
def test_jit_eager_determinism_and_directional_derivative(tracking_case: dict[str, Any]) -> None:
    objective = tracking_case["objective"]
    variables = tracking_case["variables"]
    loss = tracking_case["loss"]
    gradient = tracking_case["gradient"]

    eager_loss, _ = objective(variables)
    assert jnp.allclose(eager_loss, loss, rtol=1.0e-5, atol=1.0e-6)
    assert np.array_equal(np.asarray(objective(variables)[0]), np.asarray(eager_loss))

    gradient_norm = jnp.sqrt(sum(jnp.vdot(leaf, leaf) for leaf in jax.tree.leaves(gradient)))
    direction = jax.tree.map(lambda leaf: leaf / gradient_norm, gradient)
    epsilon = 1.0e-2
    plus = jax.tree.map(lambda value, delta: value + epsilon * delta, variables, direction)
    minus = jax.tree.map(lambda value, delta: value - epsilon * delta, variables, direction)
    finite_difference = (objective(plus)[0] - objective(minus)[0]) / (2.0 * epsilon)
    relative_error = jnp.abs(finite_difference - gradient_norm) / (
        jnp.abs(finite_difference) + gradient_norm + 1.0e-8
    )
    assert relative_error < 5.0e-2

    trial = jax.tree.map(lambda value, delta: value - 1.0e-2 * delta, variables, direction)
    assert objective(trial)[0] < loss
