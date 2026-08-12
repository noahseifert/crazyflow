"""Differentiate a trajectory-tracking loss through Crazyflow's JAX Mellinger pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crazyflow.control import Control
from crazyflow.control.mellinger import (
    GainVariables,
    TrackingLossConfig,
    apply_gain_variables,
    gain_variables_from_params,
    initialize_tracking_state,
    physical_gains,
    rollout_state_commands,
    tracking_objective,
)
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.trajectory import (
    CircleTrajectoryParams,
    Figure8TrajectoryParams,
    Trajectory,
    broadcast_state_commands,
    circle_trajectory,
    figure8_trajectory,
    state_commands,
)

RUN_IN_INTEGRATION_TEST = False


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", choices=("figure8", "circle"), default="figure8")
    parser.add_argument("--horizon", type=int, default=200, help="Number of control intervals")
    parser.add_argument(
        "--control-freq", type=int, default=100, help="State-control frequency in Hz"
    )
    parser.add_argument("--sim-freq", type=int, default=500, help="Dynamics frequency in Hz")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-2)
    parser.add_argument("--descent-step", type=float, default=1.0e-2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mellinger_tracking"))
    return parser.parse_args(argv)


def make_trajectory(kind: str, horizon: int, control_freq: int) -> tuple[Trajectory, Trajectory]:
    """Return the full trajectory including t=0 and the rollout-aligned slice."""
    time_vector = jnp.arange(horizon + 1, dtype=jnp.float32) / control_freq
    if kind == "circle":
        params = CircleTrajectoryParams(
            center=jnp.array([0.0, 0.0, 0.75], dtype=jnp.float32),
            radius=jnp.asarray(0.25, dtype=jnp.float32),
            period=jnp.asarray(3.0, dtype=jnp.float32),
            ramp_duration=jnp.asarray(0.5, dtype=jnp.float32),
            yaw=jnp.asarray(0.0, dtype=jnp.float32),
        )
        full = circle_trajectory(time_vector, params)
    else:
        params = Figure8TrajectoryParams(
            center=jnp.array([0.0, 0.0, 0.75], dtype=jnp.float32),
            amplitude=jnp.array([0.30, 0.15], dtype=jnp.float32),
            period=jnp.asarray(3.0, dtype=jnp.float32),
            ramp_duration=jnp.asarray(0.5, dtype=jnp.float32),
            yaw=jnp.asarray(0.0, dtype=jnp.float32),
        )
        full = figure8_trajectory(time_vector, params)
    reference = jax.tree.map(lambda value: value[1:], full)
    return full, reference


def tree_l2_norm(tree: Any) -> jax.Array:
    """Return the Euclidean norm of all floating-point PyTree leaves."""
    return jnp.sqrt(sum(jnp.vdot(leaf, leaf) for leaf in jax.tree.leaves(tree)))


def add_scaled(tree: Any, direction: Any, scale: float | jax.Array) -> Any:
    """Return ``tree + scale * direction`` with an unchanged PyTree structure."""
    return jax.tree.map(lambda value, delta: value + scale * delta, tree, direction)


def normalized_direction(values: tuple[float, float, float, float]) -> GainVariables:
    """Build a unit-norm deterministic direction."""
    direction = GainVariables(*(jnp.asarray(value, dtype=jnp.float32) for value in values))
    norm = tree_l2_norm(direction)
    return jax.tree.map(lambda value: value / norm, direction)


def directional_derivative_checks(
    objective_value: Any, variables: GainVariables, gradient: GainVariables, epsilon: float
) -> list[dict[str, float]]:
    """Compare central differences with autodiff along three fixed directions."""
    directions = (
        normalized_direction((1.0, 0.0, 0.0, 0.0)),
        normalized_direction((0.5, -0.5, 0.5, -0.5)),
        normalized_direction((-0.25, 0.75, 0.50, -0.35)),
    )
    checks = []
    for index, direction in enumerate(directions):
        plus = objective_value(add_scaled(variables, direction, epsilon))
        minus = objective_value(add_scaled(variables, direction, -epsilon))
        finite_difference = (plus - minus) / (2.0 * epsilon)
        autodiff = sum(
            jnp.vdot(gradient_leaf, direction_leaf)
            for gradient_leaf, direction_leaf in zip(
                jax.tree.leaves(gradient), jax.tree.leaves(direction), strict=True
            )
        )
        denominator = jnp.abs(finite_difference) + jnp.abs(autodiff) + 1.0e-8
        relative_error = jnp.abs(finite_difference - autodiff) / denominator
        checks.append(
            {
                "direction": index,
                "finite_difference": float(finite_difference),
                "autodiff": float(autodiff),
                "relative_error": float(relative_error),
            }
        )
    return checks


def _float_dict(values: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(value) for name, value in values.items()}


def _gain_dict(variables: GainVariables) -> dict[str, float]:
    return {name: float(value) for name, value in variables.__dict__.items()}


def _git_metadata() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    state_hash = hashlib.sha256()
    state_hash.update(subprocess.check_output(["git", "diff", "--binary"], text=False))
    untracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "crazyflow",
            "docs",
            "examples",
            "tests",
        ],
        text=True,
    )
    for relative_path in sorted(filter(None, untracked.splitlines())):
        state_hash.update(relative_path.encode())
        state_hash.update(Path(relative_path).read_bytes())
    return commit, state_hash.hexdigest()


def save_plot(path: Path, trace: Any, reference: Trajectory, metrics: dict[str, jax.Array]) -> None:
    """Save reference/actual tracking and weighted loss components."""
    actual_pos = np.asarray(trace.pos[:, 0, 0])
    reference_pos = np.asarray(reference.pos)
    time_vector = np.asarray(reference.time)
    error = np.linalg.norm(actual_pos - reference_pos, axis=-1)

    components = (
        "loss_position",
        "loss_velocity",
        "loss_effort",
        "loss_smoothness",
        "loss_terminal",
        "loss_altitude",
    )
    labels = ("position", "velocity", "effort", "smoothness", "terminal", "altitude")
    values = [float(metrics[name]) for name in components]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(reference_pos[:, 0], reference_pos[:, 1], label="reference")
    axes[0, 0].plot(actual_pos[:, 0], actual_pos[:, 1], label="Crazyflow")
    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", title="Horizontal path")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()

    axes[0, 1].plot(time_vector, reference_pos[:, 2], label="reference")
    axes[0, 1].plot(time_vector, actual_pos[:, 2], label="Crazyflow")
    axes[0, 1].set(xlabel="time [s]", ylabel="z [m]", title="Altitude")
    axes[0, 1].legend()

    axes[1, 0].plot(time_vector, error)
    axes[1, 0].set(xlabel="time [s]", ylabel="position error [m]", title="Tracking error")

    axes[1, 1].bar(labels, values)
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].set(ylabel="weighted loss", title="Loss components")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(args: argparse.Namespace | None = None) -> None:
    """Run and validate a differentiable Crazyflow tracking rollout."""
    args = parse_args(()) if args is None else args
    if args.horizon < 2:
        raise ValueError("horizon must be at least two")
    if args.sim_freq % args.control_freq:
        raise ValueError("sim-freq must be divisible by control-freq")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_reference, reference = make_trajectory(args.trajectory, args.horizon, args.control_freq)
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone="cf2x_L250",
        dynamics=Dynamics.first_principles,
        control=Control.state,
        freq=args.sim_freq,
        state_freq=args.control_freq,
        attitude_freq=args.sim_freq,
        force_torque_freq=args.sim_freq,
        device="cpu",
        rng_key=args.seed,
    )
    step_fn = sim.build_step_fn()
    initial_data = initialize_tracking_state(sim.data, full_reference.pos[0], full_reference.vel[0])
    commands = broadcast_state_commands(state_commands(reference), sim.n_worlds, sim.n_drones)
    state_control = initial_data.controls.state
    assert state_control is not None
    variables = gain_variables_from_params(state_control.params)
    config = TrackingLossConfig()
    steps_per_command = args.sim_freq // args.control_freq

    objective = partial(
        tracking_objective,
        initial_data=initial_data,
        commands=commands,
        reference=reference,
        step_fn=step_fn,
        steps_per_command=steps_per_command,
        config=config,
    )
    eager_loss_and_grad = jax.value_and_grad(objective, has_aux=True)
    loss_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    objective_value = jax.jit(lambda candidate: objective(candidate)[0])

    compile_start = time.perf_counter()
    compiled_result = loss_and_grad(variables)
    jax.block_until_ready(compiled_result)
    compile_seconds = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    (loss, metrics), gradient = loss_and_grad(variables)
    jax.block_until_ready((loss, metrics, gradient))
    run_seconds = time.perf_counter() - run_start

    repeated_result = loss_and_grad(variables)
    jax.block_until_ready(repeated_result)
    (repeated_loss, repeated_metrics), repeated_gradient = repeated_result
    deterministic = bool(
        np.array_equal(np.asarray(loss), np.asarray(repeated_loss))
        and all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree.leaves((metrics, gradient)),
                jax.tree.leaves((repeated_metrics, repeated_gradient)),
                strict=True,
            )
        )
    )

    (eager_loss, eager_metrics), _ = eager_loss_and_grad(variables)
    jax.block_until_ready((eager_loss, eager_metrics))
    jit_eager_abs_difference = float(jnp.abs(loss - eager_loss))

    gradient_norm = tree_l2_norm(gradient)
    finite = bool(
        jnp.isfinite(loss) & all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    )
    nontrivial_gradient = bool((gradient_norm > 1.0e-8) & (gradient_norm < 1.0e4))
    direction_checks = directional_derivative_checks(
        objective_value, variables, gradient, args.fd_epsilon
    )
    max_directional_error = max(check["relative_error"] for check in direction_checks)

    unit_gradient = jax.tree.map(lambda value: value / gradient_norm, gradient)
    trial_variables = add_scaled(variables, unit_gradient, -args.descent_step)
    trial_loss = objective_value(trial_variables)
    trial_loss.block_until_ready()
    descent_passed = bool(trial_loss < loss)

    rollout_fn = jax.jit(
        lambda candidate: rollout_state_commands(
            apply_gain_variables(initial_data, candidate), commands, step_fn, steps_per_command
        )
    )
    _, trace = rollout_fn(variables)
    jax.block_until_ready(trace)

    plot_path = args.output_dir / f"{args.trajectory}_tracking.png"
    result_path = args.output_dir / f"{args.trajectory}_result.json"
    save_plot(plot_path, trace, reference, metrics)

    commit, source_state_sha256 = _git_metadata()
    gains = physical_gains(variables)
    validations = {
        "finite_loss_and_gradient": finite,
        "nontrivial_gradient": nontrivial_gradient,
        "deterministic_replay": deterministic,
        "jit_eager_match": jit_eager_abs_difference <= 1.0e-5,
        "directional_derivatives": max_directional_error <= 5.0e-2,
        "negative_gradient_step_decreases_loss": descent_passed,
    }
    result = {
        "commit": commit,
        "source_state_sha256": source_state_sha256,
        "seed": args.seed,
        "trajectory": args.trajectory,
        "horizon": args.horizon,
        "sim_frequency_hz": args.sim_freq,
        "control_frequency_hz": args.control_freq,
        "duration_s": args.horizon / args.control_freq,
        "loss": float(loss),
        "trial_loss": float(trial_loss),
        "metrics": _float_dict(metrics),
        "optimization_variables": _gain_dict(variables),
        "physical_gains": {
            "kp_N_per_m": np.asarray(gains.kp).tolist(),
            "kd_N_s_per_m": np.asarray(gains.kd).tolist(),
        },
        "gradient": _gain_dict(gradient),
        "gradient_norm": float(gradient_norm),
        "directional_derivative_checks": direction_checks,
        "max_directional_relative_error": max_directional_error,
        "jit_eager_loss_abs_difference": jit_eager_abs_difference,
        "compile_and_first_run_seconds": compile_seconds,
        "cached_forward_backward_seconds": run_seconds,
        "validations": validations,
        "all_validations_passed": all(validations.values()),
        "versions": {"jax": jax.__version__, "numpy": np.__version__},
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"loss={float(loss):.8f}")
    print(f"gradient_norm={float(gradient_norm):.8e}")
    print(f"trial_loss={float(trial_loss):.8f}")
    print(f"max_directional_relative_error={max_directional_error:.3e}")
    print(f"validations={validations}")
    print(f"plot={plot_path}")
    print(f"result={result_path}")
    if not all(validations.values()):
        raise RuntimeError("One or more rollout validations failed; inspect the JSON result")


if __name__ == "__main__":
    main(parse_args())
