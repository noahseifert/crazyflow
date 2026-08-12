"""Run deterministic two-world Mellinger batch and gradient diagnostics.

The fixed batch maps world 0 to the Figure-8 training case and world 1 to the
circle validation case. Validation is diagnostic and is never included in the
training loss. This script evaluates local sensitivities only; it does not
implement or persist an optimization update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crazyflow.control import Control
from crazyflow.control.mellinger import (
    GAIN_BOUNDS,
    GainVariables,
    TrackingLossConfig,
    aggregate_tracking_metrics,
    apply_gain_variables,
    gain_variables_from_params,
    initialize_tracking_state,
    physical_gains,
    rollout_state_commands,
    tracking_objective_per_case,
)
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.trajectory import (
    CircleTrajectoryParams,
    Figure8TrajectoryParams,
    Trajectory,
    circle_trajectory,
    figure8_trajectory,
    state_commands,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "crazyflow.mellinger_batch_diagnostics.v1"
CASE_NAMES = ("figure8", "circle")
CASE_SPLITS = ("train", "validation")
TRAIN_MASK = jnp.asarray((1.0, 0.0), dtype=jnp.float32)
VALIDATION_MASK = jnp.asarray((0.0, 1.0), dtype=jnp.float32)
COMBINED_MASK = jnp.asarray((1.0, 1.0), dtype=jnp.float32)
LOSS_COMPONENT_NAMES = (
    "loss_position",
    "loss_velocity",
    "loss_effort",
    "loss_smoothness",
    "loss_terminal",
    "loss_altitude",
)
DAY1_H200_BASELINES = {
    "figure8": {
        "loss": 0.07310394197702408,
        "gradient": {
            "kp_xy": 0.020742792636156082,
            "kp_z": -0.010739986784756184,
            "kd_xy": -0.050698306411504745,
            "kd_z": -0.0047681089490652084,
        },
    },
    "circle": {
        "loss": 0.02409282885491848,
        "gradient": {
            "kp_xy": 0.00048197622527368367,
            "kp_z": -0.009044521488249302,
            "kd_xy": -0.011416671797633171,
            "kd_z": -0.002544915769249201,
        },
    },
}
RUN_IN_INTEGRATION_TEST = False


@dataclass(frozen=True)
class ExperimentInputs:
    """Static builder outputs and immutable arrays for one world batch."""

    sim: Sim
    initial_data: Any
    commands: jax.Array
    full_reference: Trajectory
    reference: Trajectory
    variables: GainVariables
    step_fn: Callable[..., Any]
    steps_per_command: int
    case_names: tuple[str, ...]


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse the documented Day-2 command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=200, help="Number of control intervals")
    parser.add_argument(
        "--control-freq", type=int, default=100, help="State-control frequency [Hz]"
    )
    parser.add_argument("--sim-freq", type=int, default=500, help="Dynamics frequency [Hz]")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-2)
    parser.add_argument("--gain-perturbation-fraction", type=float, default=0.10)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day2-audit/batch-h200"))
    return parser.parse_args(argv)


def _trajectory_params(kind: str) -> CircleTrajectoryParams | Figure8TrajectoryParams:
    if kind == "figure8":
        return Figure8TrajectoryParams(
            center=jnp.array([0.0, 0.0, 0.75], dtype=jnp.float32),
            amplitude=jnp.array([0.30, 0.15], dtype=jnp.float32),
            period=jnp.asarray(3.0, dtype=jnp.float32),
            ramp_duration=jnp.asarray(0.5, dtype=jnp.float32),
            yaw=jnp.asarray(0.0, dtype=jnp.float32),
        )
    if kind == "circle":
        return CircleTrajectoryParams(
            center=jnp.array([0.0, 0.0, 0.75], dtype=jnp.float32),
            radius=jnp.asarray(0.25, dtype=jnp.float32),
            period=jnp.asarray(3.0, dtype=jnp.float32),
            ramp_duration=jnp.asarray(0.5, dtype=jnp.float32),
            yaw=jnp.asarray(0.0, dtype=jnp.float32),
        )
    raise ValueError(f"Unknown trajectory kind: {kind}")


def make_trajectory(kind: str, horizon: int, control_freq: int) -> tuple[Trajectory, Trajectory]:
    """Return a Day-1-identical full and rollout-aligned trajectory."""
    time_vector = jnp.arange(horizon + 1, dtype=jnp.float32) / control_freq
    params = _trajectory_params(kind)
    if kind == "figure8":
        full = figure8_trajectory(time_vector, params)
    else:
        full = circle_trajectory(time_vector, params)
    return full, jax.tree.map(lambda value: value[1:], full)


def _stack_trajectories(trajectories: tuple[Trajectory, ...]) -> Trajectory:
    """Stack trajectories into time/world/drone layout."""
    return jax.tree.map(lambda *values: jnp.stack(values, axis=1)[:, :, None, ...], *trajectories)


def make_batch_references(
    horizon: int, control_freq: int, case_names: tuple[str, ...] = CASE_NAMES
) -> tuple[Trajectory, Trajectory, jax.Array]:
    """Build batched references and commands with shape ``(T, N, 1, 13)``."""
    pairs = tuple(make_trajectory(name, horizon, control_freq) for name in case_names)
    full = _stack_trajectories(tuple(pair[0] for pair in pairs))
    reference = _stack_trajectories(tuple(pair[1] for pair in pairs))
    commands = jnp.stack(tuple(state_commands(pair[1]) for pair in pairs), axis=1)[:, :, None, :]
    return full, reference, commands


def build_experiment(
    horizon: int,
    control_freq: int,
    sim_freq: int,
    seed: int,
    case_names: tuple[str, ...] = CASE_NAMES,
) -> ExperimentInputs:
    """Construct a real Crazyflow batch without putting ``Sim`` in the objective."""
    if horizon < 2:
        raise ValueError("horizon must be at least two")
    if control_freq <= 0 or sim_freq <= 0 or sim_freq % control_freq:
        raise ValueError("sim-freq must be positive and divisible by control-freq")
    full_reference, reference, commands = make_batch_references(horizon, control_freq, case_names)
    sim = Sim(
        n_worlds=len(case_names),
        n_drones=1,
        drone="cf2x_L250",
        dynamics=Dynamics.first_principles,
        control=Control.state,
        freq=sim_freq,
        state_freq=control_freq,
        attitude_freq=sim_freq,
        force_torque_freq=sim_freq,
        device="cpu",
        rng_key=seed,
    )
    initial_data = initialize_tracking_state(sim.data, full_reference.pos[0], full_reference.vel[0])
    state_control = initial_data.controls.state
    assert state_control is not None
    return ExperimentInputs(
        sim=sim,
        initial_data=initial_data,
        commands=commands,
        full_reference=full_reference,
        reference=reference,
        variables=gain_variables_from_params(state_control.params),
        step_fn=sim.build_step_fn(),
        steps_per_command=sim_freq // control_freq,
        case_names=case_names,
    )


def masked_tracking_objective(
    variables: GainVariables,
    initial_data: Any,
    commands: jax.Array,
    reference: Trajectory,
    mask: jax.Array,
    *,
    step_fn: Callable[..., Any],
    steps_per_command: int,
    config: TrackingLossConfig,
) -> tuple[jax.Array, dict[str, Any]]:
    """Return one numerically masked objective and both selected/per-case metrics."""
    per_case_loss, per_case_metrics = tracking_objective_per_case(
        variables, initial_data, commands, reference, step_fn, steps_per_command, config
    )
    normalized_mask = mask / jnp.sum(mask)
    loss = jnp.sum(normalized_mask * per_case_loss)
    return loss, {
        "per_case_loss": per_case_loss,
        "per_case_metrics": per_case_metrics,
        "selected_metrics": aggregate_tracking_metrics(per_case_metrics, mask),
    }


def tree_l2_norm(tree: Any) -> jax.Array:
    """Return the Euclidean norm of all PyTree leaves."""
    return jnp.sqrt(sum(jnp.vdot(leaf, leaf) for leaf in jax.tree.leaves(tree)))


def tree_add_scaled(tree: Any, direction: Any, scale: float | jax.Array) -> Any:
    """Return ``tree + scale * direction``."""
    return jax.tree.map(lambda value, delta: value + scale * delta, tree, direction)


def normalized_direction(values: tuple[float, float, float, float]) -> GainVariables:
    """Create one deterministic unit direction in raw gain space."""
    direction = GainVariables(*(jnp.asarray(value, dtype=jnp.float32) for value in values))
    norm = tree_l2_norm(direction)
    return jax.tree.map(lambda value: value / norm, direction)


def fixed_directions() -> tuple[GainVariables, GainVariables, GainVariables]:
    """Return the three Day-1-compatible deterministic raw-space directions."""
    return (
        normalized_direction((1.0, 0.0, 0.0, 0.0)),
        normalized_direction((0.5, -0.5, 0.5, -0.5)),
        normalized_direction((-0.25, 0.75, 0.50, -0.35)),
    )


def gain_dict(values: GainVariables) -> dict[str, float]:
    """Convert the four raw leaves to JSON-native floats."""
    return {name: float(value) for name, value in values.__dict__.items()}


def gradient_max_abs_difference(left: GainVariables, right: GainVariables) -> float:
    """Return the largest absolute difference between matching gain leaves."""
    return max(
        float(jnp.abs(left_leaf - right_leaf))
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def gradient_allclose(
    left: GainVariables, right: GainVariables, rtol: float = 1.0e-5, atol: float = 1.0e-7
) -> bool:
    """Check every matching gradient leaf."""
    return all(
        bool(jnp.allclose(left_leaf, right_leaf, rtol=rtol, atol=atol))
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _objective_value(
    forward: Callable[..., Any],
    variables: GainVariables,
    experiment: ExperimentInputs,
    mask: jax.Array,
) -> jax.Array:
    return forward(
        variables, experiment.initial_data, experiment.commands, experiment.reference, mask
    )[0]


def directional_derivative_checks(
    forward: Callable[..., Any],
    variables: GainVariables,
    gradient: GainVariables,
    experiment: ExperimentInputs,
    mask: jax.Array,
    epsilon: float,
) -> list[dict[str, Any]]:
    """Compare central differences and JAX directional gradients."""
    checks = []
    for index, direction in enumerate(fixed_directions()):
        plus = _objective_value(
            forward, tree_add_scaled(variables, direction, epsilon), experiment, mask
        )
        minus = _objective_value(
            forward, tree_add_scaled(variables, direction, -epsilon), experiment, mask
        )
        finite_difference = (plus - minus) / (2.0 * epsilon)
        autodiff = sum(
            jnp.vdot(gradient_leaf, direction_leaf)
            for gradient_leaf, direction_leaf in zip(
                jax.tree.leaves(gradient), jax.tree.leaves(direction), strict=True
            )
        )
        denominator = jnp.abs(finite_difference) + jnp.abs(autodiff) + 1.0e-8
        relative_error = jnp.abs(finite_difference - autodiff) / denominator
        jax.block_until_ready((finite_difference, autodiff, relative_error))
        checks.append(
            {
                "direction_index": index,
                "raw_direction": gain_dict(direction),
                "finite_difference": float(finite_difference),
                "autodiff": float(autodiff),
                "relative_error": float(relative_error),
            }
        )
    return checks


def _timed_call(function: Callable[..., Any], *arguments: Any) -> tuple[float, Any]:
    start_ns = time.perf_counter_ns()
    result = function(*arguments)
    jax.block_until_ready(result)
    elapsed = (time.perf_counter_ns() - start_ns) / 1.0e9
    return elapsed, result


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    return {
        "raw_seconds": samples,
        "minimum_seconds": min(samples),
        "median_seconds": median(samples),
        "maximum_seconds": max(samples),
    }


def profile_computation(
    forward: Callable[..., Any],
    value_and_grad: Callable[..., Any],
    batch: ExperimentInputs,
    single_cases: tuple[ExperimentInputs, ExperimentInputs],
    repeats: int,
) -> tuple[dict[str, Any], Any]:
    """Profile synchronized computation while excluding serialization and plots."""
    compile_seconds, first_result = _timed_call(
        value_and_grad,
        batch.variables,
        batch.initial_data,
        batch.commands,
        batch.reference,
        COMBINED_MASK,
    )

    _timed_call(
        forward, batch.variables, batch.initial_data, batch.commands, batch.reference, COMBINED_MASK
    )
    batch_forward_samples = [
        _timed_call(
            forward,
            batch.variables,
            batch.initial_data,
            batch.commands,
            batch.reference,
            COMBINED_MASK,
        )[0]
        for _ in range(repeats)
    ]
    batch_forward_backward_samples = [
        _timed_call(
            value_and_grad,
            batch.variables,
            batch.initial_data,
            batch.commands,
            batch.reference,
            COMBINED_MASK,
        )[0]
        for _ in range(repeats)
    ]

    single_mask = jnp.ones((1,), dtype=jnp.float32)
    for experiment in single_cases:
        _timed_call(
            forward,
            batch.variables,
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
            single_mask,
        )
        _timed_call(
            value_and_grad,
            batch.variables,
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
            single_mask,
        )

    sequential_forward_samples = []
    sequential_forward_backward_samples = []
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        forward_results = tuple(
            forward(
                batch.variables,
                experiment.initial_data,
                experiment.commands,
                experiment.reference,
                single_mask,
            )
            for experiment in single_cases
        )
        jax.block_until_ready(forward_results)
        sequential_forward_samples.append((time.perf_counter_ns() - start_ns) / 1.0e9)

        start_ns = time.perf_counter_ns()
        forward_backward_results = tuple(
            value_and_grad(
                batch.variables,
                experiment.initial_data,
                experiment.commands,
                experiment.reference,
                single_mask,
            )
            for experiment in single_cases
        )
        jax.block_until_ready(forward_backward_results)
        sequential_forward_backward_samples.append((time.perf_counter_ns() - start_ns) / 1.0e9)

    batch_forward = _timing_summary(batch_forward_samples)
    batch_forward_backward = _timing_summary(batch_forward_backward_samples)
    sequential_forward = _timing_summary(sequential_forward_samples)
    sequential_forward_backward = _timing_summary(sequential_forward_backward_samples)
    ratios = {
        "sequential_over_batch_forward": (
            sequential_forward["median_seconds"] / batch_forward["median_seconds"]
        ),
        "batch_over_sequential_forward": (
            batch_forward["median_seconds"] / sequential_forward["median_seconds"]
        ),
        "sequential_over_batch_forward_backward": (
            sequential_forward_backward["median_seconds"] / batch_forward_backward["median_seconds"]
        ),
        "batch_over_sequential_forward_backward": (
            batch_forward_backward["median_seconds"] / sequential_forward_backward["median_seconds"]
        ),
    }
    device = jax.devices()[0]
    profile = {
        "timer": "time.perf_counter_ns",
        "synchronization": "jax.block_until_ready",
        "compile_and_first_batch_forward_backward_seconds": compile_seconds,
        "cached_batch_forward": batch_forward,
        "cached_batch_forward_backward": batch_forward_backward,
        "cached_sequential_two_case_forward": sequential_forward,
        "cached_sequential_two_case_forward_backward": sequential_forward_backward,
        "ratios": ratios,
        "repeats": repeats,
        "backend": jax.default_backend(),
        "platform": device.platform,
        "device_kind": getattr(device, "device_kind", str(device)),
        "jax_version": jax.__version__,
    }
    return profile, first_result


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            "crazyflow",
            "docs",
            "examples",
            "tests",
        ],
        text=True,
    ).splitlines()
    state_hash = hashlib.sha256()
    state_hash.update(
        subprocess.check_output(
            ["git", "diff", "--binary", "--", "crazyflow", "docs", "examples", "tests"]
        )
    )
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
        state_hash.update((REPOSITORY_ROOT / relative_path).read_bytes())
    return {
        "repository": str(REPOSITORY_ROOT),
        "branch": branch,
        "commit": commit,
        "source_dirty": bool(status),
        "source_status_short": status,
        "source_state_sha256": state_hash.hexdigest(),
    }


def _metrics_for_case(per_case_metrics: dict[str, jax.Array], index: int) -> dict[str, float]:
    return {name: float(values[index]) for name, values in per_case_metrics.items()}


def _metrics_dict(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(value) for name, value in metrics.items()}


def _physical_gain_dict(variables: GainVariables) -> dict[str, float]:
    gains = physical_gains(variables)
    return {
        "kp_xy": float(gains.kp[0]),
        "kp_z": float(gains.kp[2]),
        "kd_xy": float(gains.kd[0]),
        "kd_z": float(gains.kd[2]),
    }


def _variables_from_physical(values: dict[str, float]) -> GainVariables:
    return gain_variables_from_params(
        {
            "kp": jnp.asarray(
                (values["kp_xy"], values["kp_xy"], values["kp_z"]), dtype=jnp.float32
            ),
            "kd": jnp.asarray(
                (values["kd_xy"], values["kd_xy"], values["kd_z"]), dtype=jnp.float32
            ),
        }
    )


def gain_sensitivity(
    forward: Callable[..., Any],
    batch: ExperimentInputs,
    baseline_losses: dict[str, float],
    fraction: float,
) -> tuple[dict[str, Any], bool]:
    """Evaluate independent physical ±fraction perturbations without updating gains."""
    baseline_physical = _physical_gain_dict(batch.variables)
    results = {}
    all_in_bounds = True
    for gain_name, baseline_value in baseline_physical.items():
        minus_physical = baseline_physical | {gain_name: baseline_value * (1.0 - fraction)}
        plus_physical = baseline_physical | {gain_name: baseline_value * (1.0 + fraction)}
        lower, upper = GAIN_BOUNDS[gain_name]
        all_in_bounds &= (
            lower < minus_physical[gain_name] < upper and lower < plus_physical[gain_name] < upper
        )
        minus_variables = _variables_from_physical(minus_physical)
        plus_variables = _variables_from_physical(plus_physical)
        minus_result = forward(
            minus_variables, batch.initial_data, batch.commands, batch.reference, COMBINED_MASK
        )
        plus_result = forward(
            plus_variables, batch.initial_data, batch.commands, batch.reference, COMBINED_MASK
        )
        jax.block_until_ready((minus_result, plus_result))
        minus_per_case = minus_result[1]["per_case_loss"]
        plus_per_case = plus_result[1]["per_case_loss"]
        minus_losses = {
            "train": float(minus_per_case[0]),
            "validation": float(minus_per_case[1]),
            "combined": float(minus_result[0]),
        }
        plus_losses = {
            "train": float(plus_per_case[0]),
            "validation": float(plus_per_case[1]),
            "combined": float(plus_result[0]),
        }
        denominator = plus_physical[gain_name] - minus_physical[gain_name]
        results[gain_name] = {
            "bounds": [lower, upper],
            "baseline": {
                "raw_gains": gain_dict(batch.variables),
                "physical_gains": baseline_physical,
                "losses": baseline_losses,
            },
            "minus": {
                "raw_gains": gain_dict(minus_variables),
                "physical_gains": _physical_gain_dict(minus_variables),
                "losses": minus_losses,
                "loss_differences_from_baseline": {
                    name: minus_losses[name] - baseline_losses[name] for name in baseline_losses
                },
            },
            "plus": {
                "raw_gains": gain_dict(plus_variables),
                "physical_gains": _physical_gain_dict(plus_variables),
                "losses": plus_losses,
                "loss_differences_from_baseline": {
                    name: plus_losses[name] - baseline_losses[name] for name in baseline_losses
                },
            },
            "central_sensitivity_per_physical_unit": {
                name: (plus_losses[name] - minus_losses[name]) / denominator
                for name in baseline_losses
            },
        }
    return results, all_in_bounds


def _cosine_similarity(left: GainVariables, right: GainVariables) -> tuple[float | None, str]:
    left_norm = tree_l2_norm(left)
    right_norm = tree_l2_norm(right)
    jax.block_until_ready((left_norm, right_norm))
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return None, "undefined_zero_norm"
    dot = sum(
        jnp.vdot(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )
    return float(dot / (left_norm * right_norm)), "defined"


def _day1_regression(
    args: argparse.Namespace,
    separate_losses: dict[str, float],
    separate_gradients: dict[str, GainVariables],
) -> dict[str, Any]:
    applicable = (
        args.horizon == 200
        and args.control_freq == 100
        and args.sim_freq == 500
        and args.seed == 20260724
    )
    comparisons = {}
    passed = True
    if applicable:
        for case_name in CASE_NAMES:
            baseline = DAY1_H200_BASELINES[case_name]
            loss_difference = separate_losses[case_name] - baseline["loss"]
            gradient_differences = {
                name: gain_dict(separate_gradients[case_name])[name] - baseline["gradient"][name]
                for name in baseline["gradient"]
            }
            case_passed = bool(
                np.isclose(separate_losses[case_name], baseline["loss"], rtol=1.0e-5, atol=1.0e-7)
                and all(
                    np.isclose(
                        gain_dict(separate_gradients[case_name])[name],
                        baseline["gradient"][name],
                        rtol=1.0e-5,
                        atol=1.0e-7,
                    )
                    for name in baseline["gradient"]
                )
            )
            passed &= case_passed
            comparisons[case_name] = {
                "expected": baseline,
                "actual": {
                    "loss": separate_losses[case_name],
                    "gradient": gain_dict(separate_gradients[case_name]),
                },
                "loss_difference": loss_difference,
                "gradient_differences": gradient_differences,
                "passed": case_passed,
            }
    return {"applicable": applicable, "passed": passed, "comparisons": comparisons}


def _all_finite_tree(tree: Any) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(tree))


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (jax.Array, np.generic)):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    return value


def save_batch_tracking_plot(
    path: Path, trace: Any, reference: Trajectory, case_names: tuple[str, ...]
) -> None:
    """Plot reference/actual paths and errors separately for both cases."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for index, (case_name, split) in enumerate(zip(case_names, CASE_SPLITS, strict=True)):
        actual = np.asarray(trace.pos[:, index, 0])
        target = np.asarray(reference.pos[:, index, 0])
        times = np.asarray(reference.time[:, index, 0])
        error = np.linalg.norm(actual - target, axis=-1)
        axes[index, 0].plot(target[:, 0], target[:, 1], label="reference")
        axes[index, 0].plot(actual[:, 0], actual[:, 1], label="Crazyflow")
        axes[index, 0].set(
            xlabel="x [m]", ylabel="y [m]", title=f"{case_name} / {split}: horizontal path"
        )
        axes[index, 0].axis("equal")
        axes[index, 0].legend()
        axes[index, 1].plot(times, error, label="position error")
        axes[index, 1].plot(times, target[:, 2] - actual[:, 2], label="altitude error")
        axes[index, 1].set(
            xlabel="time [s]", ylabel="error [m]", title=f"{case_name} / {split}: errors"
        )
        axes[index, 1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_loss_sensitivity_plot(
    path: Path, per_case_metrics: dict[str, jax.Array], sensitivity: dict[str, Any]
) -> None:
    """Plot weighted loss components and controlled one-gain sensitivities."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_values = np.arange(len(LOSS_COMPONENT_NAMES))
    width = 0.35
    axes[0].bar(
        x_values - width / 2,
        [float(per_case_metrics[name][0]) for name in LOSS_COMPONENT_NAMES],
        width,
        label="Figure-8 / train",
    )
    axes[0].bar(
        x_values + width / 2,
        [float(per_case_metrics[name][1]) for name in LOSS_COMPONENT_NAMES],
        width,
        label="Circle / validation",
    )
    axes[0].set_xticks(x_values, [name.removeprefix("loss_") for name in LOSS_COMPONENT_NAMES])
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set(ylabel="weighted loss", title="Per-case loss components")
    axes[0].legend()

    gain_names = tuple(sensitivity)
    x_values = np.arange(len(gain_names))
    for offset, objective_name in zip((-0.25, 0.0, 0.25), ("train", "validation", "combined")):
        axes[1].bar(
            x_values + offset,
            [
                sensitivity[name]["central_sensitivity_per_physical_unit"][objective_name]
                for name in gain_names
            ],
            0.25,
            label=objective_name,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x_values, gain_names)
    axes[1].set(
        ylabel="central loss sensitivity / physical gain unit",
        title="Independent physical ±10% sensitivity",
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_gradient_plot(
    path: Path, gradients: dict[str, GainVariables], directional: dict[str, list[dict[str, Any]]]
) -> None:
    """Plot objective gradients and directional autodiff/finite differences."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    gain_names = tuple(gain_dict(gradients["train"]))
    x_values = np.arange(len(gain_names))
    for offset, objective_name in zip((-0.25, 0.0, 0.25), gradients, strict=True):
        values = gain_dict(gradients[objective_name])
        axes[0].bar(
            x_values + offset, [values[name] for name in gain_names], 0.25, label=objective_name
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x_values, gain_names)
    axes[0].set(ylabel="raw-space gradient", title="Gradient leaves")
    axes[0].legend()

    labels = []
    autodiff_values = []
    finite_difference_values = []
    for objective_name, checks in directional.items():
        for check in checks:
            labels.append(f"{objective_name} d{check['direction_index']}")
            autodiff_values.append(check["autodiff"])
            finite_difference_values.append(check["finite_difference"])
    x_values = np.arange(len(labels))
    axes[1].bar(x_values - 0.18, autodiff_values, 0.36, label="autodiff")
    axes[1].bar(x_values + 0.18, finite_difference_values, 0.36, label="central difference")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x_values, labels, rotation=35)
    axes[1].set(ylabel="directional derivative", title="Autodiff vs finite difference")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_diagnostics(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute Day-2 computation and return JSON data plus plot inputs."""
    if args.fd_epsilon <= 0.0:
        raise ValueError("fd-epsilon must be positive")
    if not 0.0 < args.gain_perturbation_fraction < 1.0:
        raise ValueError("gain-perturbation-fraction must be in (0, 1)")
    if args.profile_repeats < 1:
        raise ValueError("profile-repeats must be positive")

    batch = build_experiment(args.horizon, args.control_freq, args.sim_freq, args.seed, CASE_NAMES)
    single_cases = tuple(
        build_experiment(args.horizon, args.control_freq, args.sim_freq, args.seed, (case_name,))
        for case_name in CASE_NAMES
    )
    reordered = build_experiment(
        args.horizon, args.control_freq, args.sim_freq, args.seed, tuple(reversed(CASE_NAMES))
    )
    config = TrackingLossConfig()
    objective = partial(
        masked_tracking_objective,
        step_fn=batch.step_fn,
        steps_per_command=batch.steps_per_command,
        config=config,
    )
    forward = jax.jit(objective)
    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    profiling, first_combined_result = profile_computation(
        forward, value_and_grad, batch, single_cases, args.profile_repeats
    )
    objective_results = {}
    for name, mask in (
        ("train", TRAIN_MASK),
        ("validation", VALIDATION_MASK),
        ("combined", COMBINED_MASK),
    ):
        if name == "combined":
            result = first_combined_result
        else:
            result = value_and_grad(
                batch.variables, batch.initial_data, batch.commands, batch.reference, mask
            )
            jax.block_until_ready(result)
        objective_results[name] = result

    single_mask = jnp.ones((1,), dtype=jnp.float32)
    separate_results = {}
    for case_name, experiment in zip(CASE_NAMES, single_cases, strict=True):
        result = value_and_grad(
            batch.variables,
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
            single_mask,
        )
        jax.block_until_ready(result)
        separate_results[case_name] = result

    reordered_result = value_and_grad(
        batch.variables,
        reordered.initial_data,
        reordered.commands,
        reordered.reference,
        COMBINED_MASK,
    )
    jax.block_until_ready(reordered_result)

    gradients = {name: result[1] for name, result in objective_results.items()}
    gradient_norms = {name: float(tree_l2_norm(value)) for name, value in gradients.items()}
    combined_loss = float(objective_results["combined"][0][0])
    per_case_loss = objective_results["combined"][0][1]["per_case_loss"]
    per_case_metrics = objective_results["combined"][0][1]["per_case_metrics"]
    selected_metrics = {
        name: _metrics_dict(result[0][1]["selected_metrics"])
        for name, result in objective_results.items()
    }
    baseline_losses = {
        "train": float(objective_results["train"][0][0]),
        "validation": float(objective_results["validation"][0][0]),
        "combined": combined_loss,
    }
    separate_losses = {
        case_name: float(result[0][0]) for case_name, result in separate_results.items()
    }
    separate_gradients = {case_name: result[1] for case_name, result in separate_results.items()}
    mean_separate_gradient = jax.tree.map(
        lambda figure8, circle: (figure8 + circle) / 2.0,
        separate_gradients["figure8"],
        separate_gradients["circle"],
    )
    separate_loss_max_abs_difference = max(
        abs(float(per_case_loss[index]) - separate_losses[case_name])
        for index, case_name in enumerate(CASE_NAMES)
    )
    combined_vs_separate_loss_difference = (
        combined_loss - (separate_losses["figure8"] + separate_losses["circle"]) / 2.0
    )
    combined_vs_separate_gradient_max_abs_difference = gradient_max_abs_difference(
        gradients["combined"], mean_separate_gradient
    )
    reordered_loss_difference = float(reordered_result[0][0]) - combined_loss
    reordered_gradient_max_abs_difference = gradient_max_abs_difference(
        reordered_result[1], gradients["combined"]
    )

    eager_result = jax.value_and_grad(objective, has_aux=True)(
        batch.variables, batch.initial_data, batch.commands, batch.reference, COMBINED_MASK
    )
    jax.block_until_ready(eager_result)
    eager_loss_difference = float(eager_result[0][0] - objective_results["combined"][0][0])
    eager_gradient_max_abs_difference = gradient_max_abs_difference(
        eager_result[1], gradients["combined"]
    )
    repeated_result = value_and_grad(
        batch.variables, batch.initial_data, batch.commands, batch.reference, COMBINED_MASK
    )
    jax.block_until_ready(repeated_result)
    deterministic = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(objective_results["combined"]),
            jax.tree.leaves(repeated_result),
            strict=True,
        )
    )

    directional = {
        "train": directional_derivative_checks(
            forward, batch.variables, gradients["train"], batch, TRAIN_MASK, args.fd_epsilon
        ),
        "combined": directional_derivative_checks(
            forward, batch.variables, gradients["combined"], batch, COMBINED_MASK, args.fd_epsilon
        ),
    }
    maximum_directional_errors = {
        name: max(check["relative_error"] for check in checks)
        for name, checks in directional.items()
    }

    descent = {}
    for name, mask in (("train", TRAIN_MASK), ("combined", COMBINED_MASK)):
        unit_gradient = jax.tree.map(lambda value: value / gradient_norms[name], gradients[name])
        trial_variables = tree_add_scaled(batch.variables, unit_gradient, -1.0e-2)
        trial_loss = _objective_value(forward, trial_variables, batch, mask)
        trial_loss.block_until_ready()
        descent[name] = {
            "step_length_raw": 1.0e-2,
            "baseline_loss": baseline_losses[name],
            "trial_loss": float(trial_loss),
            "loss_difference": float(trial_loss) - baseline_losses[name],
            "decreased": bool(trial_loss < baseline_losses[name]),
            "trial_raw_gains": gain_dict(trial_variables),
            "trial_physical_gains": _physical_gain_dict(trial_variables),
        }

    sensitivity, sensitivity_in_bounds = gain_sensitivity(
        forward, batch, baseline_losses, args.gain_perturbation_fraction
    )
    cosine_value, cosine_status = _cosine_similarity(gradients["train"], gradients["validation"])
    day1_regression = _day1_regression(args, separate_losses, separate_gradients)

    rollout_fn = jax.jit(
        lambda variables: rollout_state_commands(
            apply_gain_variables(batch.initial_data, variables),
            batch.commands,
            batch.step_fn,
            batch.steps_per_command,
        )
    )
    _, trace = rollout_fn(batch.variables)
    jax.block_until_ready(trace)

    masks_valid = bool(
        jnp.all((TRAIN_MASK * VALIDATION_MASK) == 0.0)
        and jnp.sum(TRAIN_MASK) > 0.0
        and jnp.sum(VALIDATION_MASK) > 0.0
        and jnp.all((TRAIN_MASK + VALIDATION_MASK) == 1.0)
    )
    gradients_finite = all(_all_finite_tree(gradient) for gradient in gradients.values())
    gradient_norms_valid = all(1.0e-8 < gradient_norms[name] < 1.0e4 for name in gradients)
    profiling_values = [
        profiling["compile_and_first_batch_forward_backward_seconds"],
        *profiling["cached_batch_forward"]["raw_seconds"],
        *profiling["cached_batch_forward_backward"]["raw_seconds"],
        *profiling["cached_sequential_two_case_forward"]["raw_seconds"],
        *profiling["cached_sequential_two_case_forward_backward"]["raw_seconds"],
        *profiling["ratios"].values(),
    ]
    validations = {
        "command_shape_t_2_1_13": batch.commands.shape == (args.horizon, 2, 1, 13),
        "world_commands_distinct": not bool(
            jnp.array_equal(batch.commands[:, 0], batch.commands[:, 1])
        ),
        "split_masks_disjoint_nonempty_complete": masks_valid,
        "finite_losses_metrics_states_gradients": bool(
            _all_finite_tree(objective_results)
            and _all_finite_tree(trace)
            and gradients_finite
            and all(value == 0.0 for value in per_case_metrics["nonfinite_state_fraction"])
        ),
        "gradient_norms_in_range": gradient_norms_valid,
        "batch_per_case_losses_match_separate": bool(
            all(
                np.isclose(
                    float(per_case_loss[index]),
                    separate_losses[case_name],
                    rtol=1.0e-5,
                    atol=1.0e-7,
                )
                for index, case_name in enumerate(CASE_NAMES)
            )
        ),
        "combined_loss_matches_separate_mean": bool(
            np.isclose(combined_vs_separate_loss_difference, 0.0, rtol=1.0e-5, atol=1.0e-7)
        ),
        "combined_gradient_matches_separate_mean": gradient_allclose(
            gradients["combined"], mean_separate_gradient
        ),
        "case_order_invariant_loss": bool(
            np.isclose(reordered_loss_difference, 0.0, rtol=1.0e-5, atol=1.0e-7)
        ),
        "case_order_invariant_gradient": gradient_allclose(
            reordered_result[1], gradients["combined"]
        ),
        "jit_eager_match": bool(
            np.isclose(eager_loss_difference, 0.0, rtol=1.0e-5, atol=1.0e-7)
            and eager_gradient_max_abs_difference <= 1.0e-5
        ),
        "deterministic_replay": deterministic,
        "train_directional_derivatives": maximum_directional_errors["train"] <= 5.0e-2,
        "combined_directional_derivatives": maximum_directional_errors["combined"] <= 5.0e-2,
        "train_negative_gradient_step_decreases_loss": descent["train"]["decreased"],
        "combined_negative_gradient_step_decreases_loss": descent["combined"]["decreased"],
        "gain_sensitivity_perturbations_in_bounds": sensitivity_in_bounds,
        "profiling_times_positive_finite": all(
            np.isfinite(value) and value > 0.0 for value in profiling_values
        ),
        "day1_h200_case_regression": day1_regression["passed"],
    }
    repository = _git_metadata()
    physical = physical_gains(batch.variables)
    result = {
        "schema_version": SCHEMA_VERSION,
        "repository_state": repository,
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "experiment": {
            "seed": args.seed,
            "horizon_control_intervals": args.horizon,
            "duration_seconds": args.horizon / args.control_freq,
            "control_frequency_hz": args.control_freq,
            "simulation_frequency_hz": args.sim_freq,
            "steps_per_control_interval": batch.steps_per_command,
            "worlds": 2,
            "drones_per_world": 1,
            "command_shape": list(batch.commands.shape),
            "trace_shapes": {name: list(value.shape) for name, value in trace.__dict__.items()},
            "split": [
                {
                    "batch_index": index,
                    "world": index,
                    "trajectory": case_name,
                    "split": CASE_SPLITS[index],
                    "drones": 1,
                }
                for index, case_name in enumerate(CASE_NAMES)
            ],
            "train_mask": np.asarray(TRAIN_MASK).tolist(),
            "validation_mask": np.asarray(VALIDATION_MASK).tolist(),
            "combined_mask": np.asarray(COMBINED_MASK).tolist(),
            "finite_difference_epsilon_raw": args.fd_epsilon,
            "gain_perturbation_fraction": args.gain_perturbation_fraction,
            "local_descent_step_length_raw": 1.0e-2,
            "profile_repeats": args.profile_repeats,
            "controller_mass_kg": 0.029,
            "dynamics_mass_kg": 0.0319,
            "mass_is_experiment_variable": False,
        },
        "gain_parameterization": {
            "transform": "lower + (upper - lower) * sigmoid(raw)",
            "bounds": {name: list(bounds) for name, bounds in GAIN_BOUNDS.items()},
            "baseline_raw_gains": gain_dict(batch.variables),
            "baseline_physical_gains": _physical_gain_dict(batch.variables),
            "baseline_kp_N_per_m": np.asarray(physical.kp).tolist(),
            "baseline_kd_N_s_per_m": np.asarray(physical.kd).tolist(),
        },
        "losses": {
            "per_case": {
                f"{CASE_NAMES[index]}_{CASE_SPLITS[index]}": _metrics_for_case(
                    per_case_metrics, index
                )
                for index in range(2)
            },
            **{
                name: {
                    "loss_total": baseline_losses[name],
                    "components": {
                        component: selected_metrics[name][component]
                        for component in LOSS_COMPONENT_NAMES
                    },
                }
                for name in ("train", "validation", "combined")
            },
        },
        "diagnostics": {
            "per_case": {
                f"{CASE_NAMES[index]}_{CASE_SPLITS[index]}": _metrics_for_case(
                    per_case_metrics, index
                )
                for index in range(2)
            },
            "train": selected_metrics["train"],
            "validation": selected_metrics["validation"],
            "combined": selected_metrics["combined"],
        },
        "gradients": {
            name: {"leaves": gain_dict(gradient), "l2_norm": gradient_norms[name]}
            for name, gradient in gradients.items()
        },
        "train_validation_gradient_cosine": {
            "value": cosine_value,
            "status": cosine_status,
            "zero_norm_behavior": "value is null when either exact L2 norm is zero",
        },
        "directional_derivatives": directional,
        "maximum_directional_relative_errors": maximum_directional_errors,
        "local_descent": descent,
        "gain_sensitivity": sensitivity,
        "aggregation_checks": {
            "separate_n1_losses": separate_losses,
            "separate_n1_gradients": {
                name: gain_dict(gradient) for name, gradient in separate_gradients.items()
            },
            "batch_per_case_loss_max_abs_difference": separate_loss_max_abs_difference,
            "combined_vs_separate_mean_loss_difference": (combined_vs_separate_loss_difference),
            "combined_vs_separate_mean_gradient_max_abs_difference": (
                combined_vs_separate_gradient_max_abs_difference
            ),
            "reordered_combined_loss_difference": reordered_loss_difference,
            "reordered_combined_gradient_max_abs_difference": (
                reordered_gradient_max_abs_difference
            ),
            "jit_eager_loss_difference": eager_loss_difference,
            "jit_eager_gradient_max_abs_difference": eager_gradient_max_abs_difference,
            "deterministic_replay": deterministic,
            "day1_h200_regression": day1_regression,
        },
        "profiling": {
            **profiling,
            "horizon_control_intervals": args.horizon,
            "batch_command_shape": list(batch.commands.shape),
            "single_command_shapes": [
                list(experiment.commands.shape) for experiment in single_cases
            ],
        },
        "validations": validations,
        "all_validations_passed": all(validations.values()),
    }
    return _json_native(result), {
        "trace": trace,
        "reference": batch.reference,
        "per_case_metrics": per_case_metrics,
        "sensitivity": sensitivity,
        "gradients": gradients,
        "directional": directional,
    }


def main(args: argparse.Namespace | None = None) -> None:
    """Run diagnostics and write exactly four deterministic artifact files."""
    args = parse_args(()) if args is None else args
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    result, plot_inputs = run_diagnostics(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "batch_diagnostics.json"
    tracking_path = output_dir / "batch_tracking.png"
    loss_path = output_dir / "loss_and_sensitivity.png"
    gradient_path = output_dir / "gradient_diagnostics.png"
    save_batch_tracking_plot(
        tracking_path, plot_inputs["trace"], plot_inputs["reference"], CASE_NAMES
    )
    save_loss_sensitivity_plot(
        loss_path, plot_inputs["per_case_metrics"], plot_inputs["sensitivity"]
    )
    save_gradient_plot(gradient_path, plot_inputs["gradients"], plot_inputs["directional"])
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    print(f"train_loss={result['losses']['train']['loss_total']:.10f}")
    print(f"validation_loss={result['losses']['validation']['loss_total']:.10f}")
    print(f"combined_loss={result['losses']['combined']['loss_total']:.10f}")
    print(
        "gradient_norms="
        + str({name: values["l2_norm"] for name, values in result["gradients"].items()})
    )
    print(
        "maximum_directional_relative_errors=" + str(result["maximum_directional_relative_errors"])
    )
    print(f"validations={result['validations']}")
    print(f"result={result_path}")
    if not result["all_validations_passed"]:
        raise RuntimeError("One or more Day-2 validations failed; inspect the preserved JSON")


if __name__ == "__main__":
    main(parse_args())
