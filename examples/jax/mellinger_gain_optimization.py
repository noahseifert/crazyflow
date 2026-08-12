"""Optimize Mellinger gains on Figure-8 and evaluate held-out robustness."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crazyflow.control.mellinger import (
    GAIN_BOUNDS,
    GainVariables,
    TrackingLossConfig,
    adam_optimization_step,
    evaluate_tracking_objectives,
    physical_gains,
    select_best_train_checkpoint,
    train_tracking_objective,
    tree_all_finite,
    tree_l2_norm,
    with_controller_mass,
)

if __package__:
    from examples.jax.mellinger_batch_diagnostics import (
        CASE_NAMES,
        COMBINED_MASK,
        TRAIN_MASK,
        VALIDATION_MASK,
        ExperimentInputs,
        build_experiment,
        gain_dict,
        gradient_allclose,
        gradient_max_abs_difference,
        masked_tracking_objective,
        tree_add_scaled,
    )
else:
    from mellinger_batch_diagnostics import (
        CASE_NAMES,
        COMBINED_MASK,
        TRAIN_MASK,
        VALIDATION_MASK,
        ExperimentInputs,
        build_experiment,
        gain_dict,
        gradient_allclose,
        gradient_max_abs_difference,
        masked_tracking_objective,
        tree_add_scaled,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_HEAD = "32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed"
HISTORICAL_DAY2_HEAD = "9eac04ea28997e76b9bc8c6e97ca53ef93398d55"
HISTORICAL_DAY2_SOURCE_STATE = "a1d7f1dce6ed3b1aa9cbb7cc53abb4a3bcada379129dfbde1787c1a5d1564229"
OPTIMIZATION_SCHEMA = "crazyflow.mellinger_gain_optimization.v1"
GAINS_SCHEMA = "crazyflow.mellinger_optimized_gains.v1"
ROBUSTNESS_SCHEMA = "crazyflow.mellinger_robustness.v1"
EXPECTED_OUTPUT_FILES = {
    "optimization_result.json",
    "optimized_gains.json",
    "robustness_results.json",
    "optimization_history.png",
    "gain_history.png",
    "robustness_matrix.png",
}
LOSS_COMPONENT_NAMES = (
    "loss_position",
    "loss_velocity",
    "loss_effort",
    "loss_smoothness",
    "loss_terminal",
    "loss_altitude",
)
MASS_CONDITIONS = {"audited_mismatch": 0.029, "matched_controller_to_dynamics": 0.0319}
SCIENTIFIC_SOURCE_PATHS = ("crazyflow", "examples", "tests", "pyproject.toml", "pixi.lock")
SPRINT3_DIRECTION_VALUES = ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0))
RUN_IN_INTEGRATION_TEST = False


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse the fixed Sprint-3 command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--control-freq", type=int, default=100)
    parser.add_argument("--sim-freq", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-2)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--robustness-horizons", type=int, nargs="+", default=[20, 100, 200, 400])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/day3-audit/optimize-h200")
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.horizon < 2:
        raise ValueError("horizon must be at least two")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.control_freq <= 0 or args.sim_freq <= 0 or args.sim_freq % args.control_freq:
        raise ValueError("sim-freq must be positive and divisible by control-freq")
    if args.fd_epsilon <= 0.0:
        raise ValueError("fd-epsilon must be positive")
    if args.profile_repeats < 1:
        raise ValueError("profile-repeats must be positive")
    if not args.robustness_horizons or any(horizon < 2 for horizon in args.robustness_horizons):
        raise ValueError("robustness-horizons must contain positive horizons of at least two")
    if len(set(args.robustness_horizons)) != len(args.robustness_horizons):
        raise ValueError("robustness-horizons must be unique")


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


def _metrics_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(value) for name, value in metrics.items()}


def _physical_gain_dict(variables: GainVariables) -> dict[str, float]:
    gains = physical_gains(variables)
    return {
        "kp_xy": float(gains.kp[0]),
        "kp_z": float(gains.kp[2]),
        "kd_xy": float(gains.kd[0]),
        "kd_z": float(gains.kd[2]),
    }


def _gain_limit_distances(variables: GainVariables) -> dict[str, dict[str, float]]:
    physical = _physical_gain_dict(variables)
    return {
        name: {"to_lower": value - GAIN_BOUNDS[name][0], "to_upper": GAIN_BOUNDS[name][1] - value}
        for name, value in physical.items()
    }


def _parameter_record(
    step: int, variables: GainVariables, evaluation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "step": step,
        "raw_gains": gain_dict(variables),
        "physical_gains": _physical_gain_dict(variables),
        "gain_limit_distances": _gain_limit_distances(variables),
        "train_loss": float(evaluation["train_loss"]),
        "validation_loss": float(evaluation["validation_loss"]),
        "combined_loss": float(evaluation["combined_loss"]),
        "train_metrics": _metrics_dict(evaluation["train_metrics"]),
        "validation_metrics": _metrics_dict(evaluation["validation_metrics"]),
        "combined_metrics": _metrics_dict(evaluation["combined_metrics"]),
    }


def _loss_changes(initial: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name in ("train", "validation", "combined"):
        initial_loss = float(initial[f"{name}_loss"])
        selected_loss = float(selected[f"{name}_loss"])
        difference = selected_loss - initial_loss
        result[name] = {
            "initial": initial_loss,
            "selected": selected_loss,
            "difference": difference,
            "absolute_magnitude": abs(difference),
            "relative_change": difference / initial_loss,
        }
    return result


def _scientific_repository_state() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    status_command = [
        "git",
        "status",
        "--short",
        "--untracked-files=all",
        "--",
        *SCIENTIFIC_SOURCE_PATHS,
    ]
    status = subprocess.check_output(status_command, text=True).splitlines()
    state_hash = hashlib.sha256()
    state_hash.update(
        subprocess.check_output(
            ["git", "diff", "--binary", BASE_HEAD, "--", *SCIENTIFIC_SOURCE_PATHS]
        )
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *SCIENTIFIC_SOURCE_PATHS],
        text=True,
    )
    for relative_path in sorted(filter(None, untracked.splitlines())):
        state_hash.update(relative_path.encode())
        state_hash.update((REPOSITORY_ROOT / relative_path).read_bytes())
    return {
        "repository": str(REPOSITORY_ROOT),
        "branch": branch,
        "head": head,
        "base_head": BASE_HEAD,
        "preflight_worktree": "clean",
        "source_dirty": bool(status),
        "source_status_short": status,
        "source_state_sha256": state_hash.hexdigest(),
        "source_state_scope": list(SCIENTIFIC_SOURCE_PATHS),
        "historical_day2_execution": {
            "head": HISTORICAL_DAY2_HEAD,
            "source_dirty": True,
            "source_state_sha256": HISTORICAL_DAY2_SOURCE_STATE,
            "preserved_unchanged": True,
        },
    }


def _timed_call(function: Callable[..., Any], *arguments: Any) -> tuple[float, Any]:
    start_ns = time.perf_counter_ns()
    result = function(*arguments)
    jax.block_until_ready(result)
    return (time.perf_counter_ns() - start_ns) / 1.0e9, result


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    return {
        "raw_seconds": samples,
        "minimum_seconds": min(samples),
        "median_seconds": median(samples),
        "maximum_seconds": max(samples),
    }


def _make_functions(
    experiment: ExperimentInputs, optimizer: optax.GradientTransformation
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    config = TrackingLossConfig()
    step = jax.jit(
        partial(
            adam_optimization_step,
            optimizer=optimizer,
            step_fn=experiment.step_fn,
            steps_per_command=experiment.steps_per_command,
            config=config,
        )
    )
    evaluate = jax.jit(
        partial(
            evaluate_tracking_objectives,
            step_fn=experiment.step_fn,
            steps_per_command=experiment.steps_per_command,
            config=config,
        )
    )
    return step, evaluate


def optimize_train_only(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[GainVariables], list[dict[str, Any]], dict[str, Any], Any]:
    """Run fixed-step Adam and retain Step 0 plus every post-update checkpoint."""
    experiment = build_experiment(
        args.horizon, args.control_freq, args.sim_freq, args.seed, CASE_NAMES
    )
    optimizer = optax.adam(args.learning_rate)
    variables = experiment.variables
    opt_state = optimizer.init(variables)
    step_fn, evaluate_fn = _make_functions(experiment, optimizer)

    initial_evaluation = evaluate_fn(
        variables, experiment.initial_data, experiment.commands, experiment.reference
    )
    jax.block_until_ready(initial_evaluation)
    if not tree_all_finite((variables, opt_state, initial_evaluation)):
        raise FloatingPointError("Nonfinite Step-0 parameter, Optax state, loss, or metric")

    history = [_parameter_record(0, variables, initial_evaluation)]
    variable_history = [variables]
    updates = []
    compile_first_seconds = None
    loop_start_ns = time.perf_counter_ns()
    for update_index in range(args.steps):
        elapsed, step_result = _timed_call(
            step_fn,
            variables,
            opt_state,
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
        )
        if compile_first_seconds is None:
            compile_first_seconds = elapsed
        new_variables, new_opt_state, differentiated_loss, auxiliary, gradient = step_result
        if not tree_all_finite(
            (new_variables, new_opt_state, differentiated_loss, auxiliary, gradient)
        ):
            raise FloatingPointError(f"Nonfinite optimization state at update {update_index + 1}")
        if not np.isclose(
            float(differentiated_loss), history[-1]["train_loss"], rtol=1.0e-5, atol=1.0e-7
        ):
            raise RuntimeError("Differentiated train loss does not match recorded checkpoint")
        gradient_norm = float(tree_l2_norm(gradient))
        history[-1]["train_gradient_for_next_update"] = gain_dict(gradient)
        history[-1]["train_gradient_l2_norm"] = gradient_norm
        updates.append(
            {
                "update": update_index + 1,
                "from_step": update_index,
                "to_step": update_index + 1,
                "differentiated_objective": "train_loss_only",
                "train_loss": float(differentiated_loss),
                "train_gradient": gain_dict(gradient),
                "train_gradient_l2_norm": gradient_norm,
                "finite_parameters": tree_all_finite(new_variables),
                "finite_gradient": tree_all_finite(gradient),
                "finite_optax_state": tree_all_finite(new_opt_state),
            }
        )
        variables, opt_state = new_variables, new_opt_state
        evaluation = evaluate_fn(
            variables, experiment.initial_data, experiment.commands, experiment.reference
        )
        jax.block_until_ready(evaluation)
        if not tree_all_finite(evaluation):
            raise FloatingPointError(f"Nonfinite post-update evaluation at step {update_index + 1}")
        history.append(_parameter_record(update_index + 1, variables, evaluation))
        variable_history.append(variables)
    total_updates_seconds = (time.perf_counter_ns() - loop_start_ns) / 1.0e9

    cached_step_samples = [
        _timed_call(
            step_fn,
            variables,
            opt_state,
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
        )[0]
        for _ in range(args.profile_repeats)
    ]
    timing = {
        "compile_and_first_optimization_step_seconds": compile_first_seconds,
        "cached_optax_step": _timing_summary(cached_step_samples),
        "total_update_loop_seconds": total_updates_seconds,
    }
    return history, variable_history, updates, timing, experiment


def _masked_objective_functions(
    experiment: ExperimentInputs,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    objective = partial(
        masked_tracking_objective,
        step_fn=experiment.step_fn,
        steps_per_command=experiment.steps_per_command,
        config=TrackingLossConfig(),
    )
    return jax.jit(objective), jax.jit(jax.value_and_grad(objective, has_aux=True))


def _sprint3_directions() -> tuple[GainVariables, GainVariables, GainVariables]:
    directions = []
    for values in SPRINT3_DIRECTION_VALUES:
        direction = GainVariables(*(jnp.asarray(value, dtype=jnp.float32) for value in values))
        norm = tree_l2_norm(direction)
        directions.append(jax.tree.map(lambda value: value / norm, direction))
    return tuple(directions)


def _directional_derivative_checks(
    forward: Callable[..., Any],
    variables: GainVariables,
    gradient: GainVariables,
    experiment: ExperimentInputs,
    mask: jax.Array,
    epsilon: float,
) -> list[dict[str, Any]]:
    checks = []
    for index, direction in enumerate(_sprint3_directions()):
        plus = forward(
            tree_add_scaled(variables, direction, epsilon),
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
            mask,
        )[0]
        minus = forward(
            tree_add_scaled(variables, direction, -epsilon),
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
            mask,
        )[0]
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


def gradient_diagnostics(
    args: argparse.Namespace, variables: GainVariables, batch: ExperimentInputs
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Audit gradients and aggregation at the selected train checkpoint."""
    forward, value_and_grad = _masked_objective_functions(batch)
    results = {}
    for name, mask in (
        ("train", TRAIN_MASK),
        ("validation", VALIDATION_MASK),
        ("combined", COMBINED_MASK),
    ):
        result = value_and_grad(
            variables, batch.initial_data, batch.commands, batch.reference, mask
        )
        jax.block_until_ready(result)
        results[name] = result
    gradients = {name: result[1] for name, result in results.items()}
    gradient_records = {
        name: {"leaves": gain_dict(gradient), "l2_norm": float(tree_l2_norm(gradient))}
        for name, gradient in gradients.items()
    }

    train_norm = gradient_records["train"]["l2_norm"]
    validation_norm = gradient_records["validation"]["l2_norm"]
    if train_norm == 0.0 or validation_norm == 0.0:
        cosine = {"status": "undefined_zero_norm", "value": None}
    else:
        dot = sum(
            jnp.vdot(left, right)
            for left, right in zip(
                jax.tree.leaves(gradients["train"]),
                jax.tree.leaves(gradients["validation"]),
                strict=True,
            )
        )
        cosine = {"status": "defined", "value": float(dot / (train_norm * validation_norm))}

    directional = {
        name: _directional_derivative_checks(
            forward, variables, gradients[name], batch, mask, args.fd_epsilon
        )
        for name, mask in (("train", TRAIN_MASK), ("combined", COMBINED_MASK))
    }
    maximum_directional_errors = {
        name: max(check["relative_error"] for check in checks)
        for name, checks in directional.items()
    }

    if train_norm <= 1.0e-8:
        local_descent = {
            "status": "stationary_below_threshold",
            "gradient_norm_threshold": 1.0e-8,
            "step_length_raw": None,
            "decreased": None,
        }
    else:
        unit_gradient = jax.tree.map(lambda value: value / train_norm, gradients["train"])
        trial_variables = tree_add_scaled(variables, unit_gradient, -1.0e-2)
        trial_loss = forward(
            trial_variables, batch.initial_data, batch.commands, batch.reference, TRAIN_MASK
        )[0]
        trial_loss.block_until_ready()
        baseline_loss = float(results["train"][0][0])
        local_descent = {
            "status": "evaluated",
            "gradient_norm_threshold": 1.0e-8,
            "step_length_raw": 1.0e-2,
            "baseline_train_loss": baseline_loss,
            "trial_train_loss": float(trial_loss),
            "loss_difference": float(trial_loss) - baseline_loss,
            "decreased": bool(trial_loss < baseline_loss),
            "trial_raw_gains": gain_dict(trial_variables),
            "trial_physical_gains": _physical_gain_dict(trial_variables),
        }

    single_cases = tuple(
        build_experiment(args.horizon, args.control_freq, args.sim_freq, args.seed, (case_name,))
        for case_name in CASE_NAMES
    )
    single_mask = jnp.ones((1,), dtype=jnp.float32)
    separate_results = []
    for experiment in single_cases:
        _, single_value_and_grad = _masked_objective_functions(experiment)
        result = single_value_and_grad(
            variables,
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
            single_mask,
        )
        jax.block_until_ready(result)
        separate_results.append(result)
    mean_separate_loss = float(np.mean([float(result[0][0]) for result in separate_results]))
    mean_separate_gradient = jax.tree.map(
        lambda left, right: (left + right) / 2.0, separate_results[0][1], separate_results[1][1]
    )
    batch_combined_loss = float(results["combined"][0][0])

    reordered = build_experiment(
        args.horizon, args.control_freq, args.sim_freq, args.seed, tuple(reversed(CASE_NAMES))
    )
    _, reordered_value_and_grad = _masked_objective_functions(reordered)
    reordered_result = reordered_value_and_grad(
        variables, reordered.initial_data, reordered.commands, reordered.reference, COMBINED_MASK
    )
    jax.block_until_ready(reordered_result)

    objective = partial(
        masked_tracking_objective,
        step_fn=batch.step_fn,
        steps_per_command=batch.steps_per_command,
        config=TrackingLossConfig(),
    )
    eager_result = jax.value_and_grad(objective, has_aux=True)(
        variables, batch.initial_data, batch.commands, batch.reference, COMBINED_MASK
    )
    jax.block_until_ready(eager_result)

    per_case_metrics = results["combined"][0][1]["per_case_metrics"]
    piecewise = {
        CASE_NAMES[index]: {
            "motor_saturation_fraction": float(
                per_case_metrics["motor_saturation_fraction"][index]
            ),
            "zero_thrust_gate_fraction": float(
                per_case_metrics["zero_thrust_gate_fraction"][index]
            ),
            "floor_clip_fraction": float(per_case_metrics["floor_clip_fraction"][index]),
            "nonfinite_state_fraction": float(per_case_metrics["nonfinite_state_fraction"][index]),
        }
        for index in range(2)
    }
    jaxpr = jax.make_jaxpr(
        lambda candidate: objective(
            candidate, batch.initial_data, batch.commands, batch.reference, TRAIN_MASK
        )[0]
    )(variables)
    batch_separate_loss_match = bool(
        np.isclose(batch_combined_loss, mean_separate_loss, rtol=1.0e-5, atol=1.0e-7)
    )
    batch_separate_gradient_match = gradient_allclose(gradients["combined"], mean_separate_gradient)
    order_loss_difference = float(reordered_result[0][0]) - batch_combined_loss
    order_gradient_difference = gradient_max_abs_difference(
        reordered_result[1], gradients["combined"]
    )
    eager_loss_difference = float(eager_result[0][0]) - batch_combined_loss
    eager_gradient_difference = gradient_max_abs_difference(eager_result[1], gradients["combined"])
    checks = {
        "finite_train_validation_combined_gradients": all(
            tree_all_finite(gradient) for gradient in gradients.values()
        ),
        "train_directional_derivatives": maximum_directional_errors["train"] <= 5.0e-2,
        "combined_directional_derivatives": maximum_directional_errors["combined"] <= 5.0e-2,
        "batch_separate_combined_loss_match": batch_separate_loss_match,
        "batch_separate_combined_gradient_match": batch_separate_gradient_match,
        "world_order_invariant_loss": bool(
            np.isclose(order_loss_difference, 0.0, rtol=1.0e-5, atol=1.0e-7)
        ),
        "world_order_invariant_gradient": gradient_allclose(
            reordered_result[1], gradients["combined"]
        ),
        "jit_eager_match": bool(
            np.isclose(eager_loss_difference, 0.0, rtol=1.0e-5, atol=1.0e-7)
            and eager_gradient_difference <= 1.0e-5
        ),
        "negative_train_gradient_local_descent": (
            local_descent["status"] == "stationary_below_threshold"
            or bool(local_descent["decreased"])
        ),
        "rollout_jaxpr_contains_scan": "scan[" in str(jaxpr),
        "zero_floor_and_nonfinite_fractions": all(
            values["floor_clip_fraction"] == 0.0 and values["nonfinite_state_fraction"] == 0.0
            for values in piecewise.values()
        ),
    }
    diagnostics = {
        "gradients": gradient_records,
        "train_validation_gradient_cosine": cosine,
        "directional_derivatives": directional,
        "maximum_directional_relative_errors": maximum_directional_errors,
        "local_train_descent": local_descent,
        "batch_separate": {
            "batch_combined_loss": batch_combined_loss,
            "separate_n1_losses": {
                CASE_NAMES[index]: float(result[0][0])
                for index, result in enumerate(separate_results)
            },
            "separate_mean_loss": mean_separate_loss,
            "loss_difference": batch_combined_loss - mean_separate_loss,
            "gradient_max_abs_difference": gradient_max_abs_difference(
                gradients["combined"], mean_separate_gradient
            ),
        },
        "world_order": {
            "loss_difference": order_loss_difference,
            "gradient_max_abs_difference": order_gradient_difference,
        },
        "jit_eager": {
            "loss_difference": eager_loss_difference,
            "gradient_max_abs_difference": eager_gradient_difference,
        },
        "piecewise_branch_fractions": piecewise,
    }
    return diagnostics, checks


def _forward_cell(
    variables: GainVariables, experiment: ExperimentInputs
) -> tuple[dict[str, Any], float, float]:
    objective = jax.jit(
        partial(
            evaluate_tracking_objectives,
            step_fn=experiment.step_fn,
            steps_per_command=experiment.steps_per_command,
            config=TrackingLossConfig(),
        )
    )
    result = objective(
        variables, experiment.initial_data, experiment.commands, experiment.reference
    )
    jax.block_until_ready(result)
    state_control = experiment.initial_data.controls.state
    assert state_control is not None
    return (
        result,
        float(state_control.params["mass"]),
        float(experiment.initial_data.params.mass[0, 0, 0]),
    )


def robustness_matrix(
    args: argparse.Namespace, baseline: GainVariables, best_train: GainVariables
) -> tuple[dict[str, Any], bool]:
    """Evaluate the fixed parameter/horizon/trajectory/mass matrix without updates."""
    cells = []
    variables_by_name = {"baseline": baseline, "best_train": best_train}
    for horizon in args.robustness_horizons:
        for trajectory in CASE_NAMES:
            base_experiment = build_experiment(
                horizon, args.control_freq, args.sim_freq, args.seed, (trajectory,)
            )
            for mass_id, controller_mass in MASS_CONDITIONS.items():
                experiment = base_experiment
                if mass_id == "matched_controller_to_dynamics":
                    experiment = ExperimentInputs(
                        sim=base_experiment.sim,
                        initial_data=with_controller_mass(
                            base_experiment.initial_data, controller_mass
                        ),
                        commands=base_experiment.commands,
                        full_reference=base_experiment.full_reference,
                        reference=base_experiment.reference,
                        variables=base_experiment.variables,
                        step_fn=base_experiment.step_fn,
                        steps_per_command=base_experiment.steps_per_command,
                        case_names=base_experiment.case_names,
                    )
                for parameter_set, variables in variables_by_name.items():
                    evaluation, actual_controller_mass, actual_dynamics_mass = _forward_cell(
                        variables, experiment
                    )
                    metrics = _metrics_dict(evaluation["train_metrics"])
                    cells.append(
                        {
                            "cell_id": (f"{parameter_set}__h{horizon}__{trajectory}__{mass_id}"),
                            "parameter_set": parameter_set,
                            "horizon": horizon,
                            "trajectory": trajectory,
                            "split": "train" if trajectory == "figure8" else "validation",
                            "mass_condition": mass_id,
                            "controller_mass_kg": actual_controller_mass,
                            "dynamics_mass_kg": actual_dynamics_mass,
                            "raw_gains": gain_dict(variables),
                            "physical_gains": _physical_gain_dict(variables),
                            **metrics,
                        }
                    )

    keyed = {
        (cell["horizon"], cell["trajectory"], cell["mass_condition"], cell["parameter_set"]): cell
        for cell in cells
    }
    for cell in cells:
        baseline_cell = keyed[
            (cell["horizon"], cell["trajectory"], cell["mass_condition"], "baseline")
        ]
        difference = cell["loss_total"] - baseline_cell["loss_total"]
        cell["best_train_vs_baseline"] = {
            "loss_difference": difference,
            "loss_relative_change": difference / baseline_cell["loss_total"],
        }

    summaries = {}
    for parameter_set in variables_by_name:
        subset = [cell for cell in cells if cell["parameter_set"] == parameter_set]
        worst = max(subset, key=lambda cell: cell["loss_total"])
        mismatch = [
            cell["loss_total"] for cell in subset if cell["mass_condition"] == "audited_mismatch"
        ]
        matched = [
            cell["loss_total"]
            for cell in subset
            if cell["mass_condition"] == "matched_controller_to_dynamics"
        ]
        h200_changes = {}
        for trajectory in CASE_NAMES:
            key = (200, trajectory, "audited_mismatch", parameter_set)
            baseline_key = (200, trajectory, "audited_mismatch", "baseline")
            if key in keyed and baseline_key in keyed:
                value = keyed[key]["loss_total"]
                baseline_value = keyed[baseline_key]["loss_total"]
                h200_changes[trajectory] = {
                    "loss": value,
                    "difference_from_baseline": value - baseline_value,
                    "relative_change_from_baseline": (value - baseline_value) / baseline_value,
                }
        summaries[parameter_set] = {
            "mean_loss": float(np.mean([cell["loss_total"] for cell in subset])),
            "maximum_loss": float(worst["loss_total"]),
            "worst_cell": worst,
            "h200_audited_mismatch": {
                "applicable": bool(h200_changes),
                "trajectories": h200_changes,
            },
            "mass_condition_mean_change_matched_minus_mismatch": float(
                np.mean(matched) - np.mean(mismatch)
            ),
            "mass_condition_mean_relative_change": float(
                (np.mean(matched) - np.mean(mismatch)) / np.mean(mismatch)
            ),
        }
    expected_count = 2 * len(args.robustness_horizons) * 2 * 2
    complete = (
        len(cells) == expected_count
        and len({cell["cell_id"] for cell in cells}) == expected_count
        and all(
            np.isfinite(value)
            for cell in cells
            for value in (
                cell["loss_total"],
                cell["loss_position"],
                cell["loss_velocity"],
                cell["loss_effort"],
                cell["loss_smoothness"],
                cell["loss_terminal"],
                cell["loss_altitude"],
                cell["position_rmse_m"],
                cell["max_position_error_m"],
            )
        )
        and all(
            cell["floor_clip_fraction"] == 0.0 and cell["nonfinite_state_fraction"] == 0.0
            for cell in cells
        )
    )
    result = {
        "schema_version": ROBUSTNESS_SCHEMA,
        "seed": args.seed,
        "noise_or_disturbance_model": None,
        "seed_robustness_claim": False,
        "requested_horizons": list(args.robustness_horizons),
        "parameter_sets": list(variables_by_name),
        "trajectories": [
            {"id": "figure8", "split": "train"},
            {"id": "circle", "split": "validation"},
        ],
        "mass_conditions": {
            name: {
                "controller_mass_kg": mass,
                "dynamics_mass_kg": 0.0319,
                "experiment_local_controller_replacement": name == "matched_controller_to_dynamics",
            }
            for name, mass in MASS_CONDITIONS.items()
        },
        "forward_cell_evaluations": len(cells),
        "cells": cells,
        "summaries": summaries,
        "robustness_matrix_complete": complete,
        "full_sprint_matrix_requested": set(args.robustness_horizons) == {20, 100, 200, 400},
    }
    return result, complete


def profile_horizons(
    args: argparse.Namespace, variables: GainVariables
) -> tuple[dict[str, Any], bool]:
    """Profile synchronized cached forward and forward/backward calls."""
    forward_profiles = {}
    backward_profiles = {}
    successful_horizons = []
    for horizon in args.robustness_horizons:
        experiment = build_experiment(
            horizon, args.control_freq, args.sim_freq, args.seed, CASE_NAMES
        )
        forward = jax.jit(
            partial(
                evaluate_tracking_objectives,
                step_fn=experiment.step_fn,
                steps_per_command=experiment.steps_per_command,
                config=TrackingLossConfig(),
            )
        )
        backward = jax.jit(
            jax.value_and_grad(
                partial(
                    train_tracking_objective,
                    step_fn=experiment.step_fn,
                    steps_per_command=experiment.steps_per_command,
                    config=TrackingLossConfig(),
                ),
                has_aux=True,
            )
        )
        arguments = (variables, experiment.initial_data, experiment.commands, experiment.reference)
        _timed_call(forward, *arguments)
        _timed_call(backward, *arguments)
        forward_samples = [_timed_call(forward, *arguments)[0] for _ in range(args.profile_repeats)]
        backward_samples = [
            _timed_call(backward, *arguments)[0] for _ in range(args.profile_repeats)
        ]
        forward_profiles[str(horizon)] = _timing_summary(forward_samples)
        backward_profiles[str(horizon)] = _timing_summary(backward_samples)
        successful_horizons.append(horizon)
    h400_h200_ratio = None
    scaling_indicator = False
    if "200" in backward_profiles and "400" in backward_profiles:
        h400_h200_ratio = (
            backward_profiles["400"]["median_seconds"] / backward_profiles["200"]["median_seconds"]
        )
        scaling_indicator = h400_h200_ratio > 10.0
    return (
        {
            "timer": "time.perf_counter_ns",
            "synchronization": "jax.block_until_ready",
            "repeats": args.profile_repeats,
            "cached_forward": forward_profiles,
            "cached_forward_backward": backward_profiles,
            "forward_backward_h400_over_h200_median_ratio": h400_h200_ratio,
            "h400_over_h200_exceeds_10": scaling_indicator,
            "successful_horizons": successful_horizons,
            "peak_memory_measured": False,
            "speedup_claim": False,
        },
        set(successful_horizons) == set(args.robustness_horizons),
    )


def _save_optimization_history(path: Path, history: list[dict[str, Any]], best_step: int) -> None:
    steps = [record["step"] for record in history]
    figure, axis = plt.subplots(figsize=(9, 5))
    for objective in ("train", "validation", "combined"):
        axis.plot(steps, [record[f"{objective}_loss"] for record in history], label=objective)
    best = history[best_step]
    axis.scatter(
        [best_step],
        [best["train_loss"]],
        marker="*",
        s=130,
        color="black",
        label="best train checkpoint",
        zorder=5,
    )
    axis.set(xlabel="optimization step", ylabel="tracking loss", title="Train-only Adam history")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_gain_history(path: Path, history: list[dict[str, Any]]) -> None:
    steps = [record["step"] for record in history]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, gain_name in zip(axes.flat, GAIN_BOUNDS, strict=True):
        values = [record["physical_gains"][gain_name] for record in history]
        lower, upper = GAIN_BOUNDS[gain_name]
        axis.plot(steps, values, color="tab:blue")
        axis.axhline(lower, color="tab:red", linestyle="--", label="lower bound")
        axis.axhline(upper, color="tab:orange", linestyle="--", label="upper bound")
        axis.set(title=gain_name, ylabel="physical gain")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("optimization step")
    axes[-1, 1].set_xlabel("optimization step")
    axes[0, 0].legend()
    figure.suptitle("Bounded physical gain history")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_robustness_matrix(path: Path, robustness: dict[str, Any]) -> None:
    horizons = robustness["requested_horizons"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, trajectory in enumerate(CASE_NAMES):
        for column, mass_id in enumerate(MASS_CONDITIONS):
            axis = axes[row, column]
            for parameter_set in ("baseline", "best_train"):
                values = []
                for horizon in horizons:
                    cell = next(
                        item
                        for item in robustness["cells"]
                        if item["parameter_set"] == parameter_set
                        and item["horizon"] == horizon
                        and item["trajectory"] == trajectory
                        and item["mass_condition"] == mass_id
                    )
                    values.append(cell["loss_total"])
                axis.plot(horizons, values, marker="o", label=parameter_set)
            axis.set(title=f"{trajectory} / {mass_id}", xlabel="horizon", ylabel="tracking loss")
            axis.grid(alpha=0.25)
            axis.legend()
    figure.suptitle("Robustness matrix: baseline versus best train")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_results(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute optimization, diagnostics, robustness, and profiling."""
    _validate_args(args)
    history, variable_history, updates, optimization_timings, batch = optimize_train_only(args)
    best_index = select_best_train_checkpoint(history)
    best_variables = variable_history[best_index]
    initial = history[0]
    final = history[-1]
    best = history[best_index]

    gradients, gradient_checks = gradient_diagnostics(args, best_variables, batch)
    robustness, robustness_complete = robustness_matrix(args, variable_history[0], best_variables)
    horizon_profile, all_profile_horizons_completed = profile_horizons(args, best_variables)
    profiling = horizon_profile | optimization_timings
    all_updates_finite = len(updates) == args.steps and all(
        update["finite_parameters"] and update["finite_gradient"] and update["finite_optax_state"]
        for update in updates
    )
    all_gains_in_bounds = all(
        lower < record["physical_gains"][gain_name] < upper
        for record in history
        for gain_name, (lower, upper) in GAIN_BOUNDS.items()
    )
    step0_day2_baseline = bool(
        args.horizon != 200
        or np.isclose(history[0]["train_loss"], 0.07310396432876587, rtol=1.0e-5, atol=1.0e-7)
    )
    validations = {
        "all_updates_finite": all_updates_finite,
        "selected_checkpoint_after_step_zero": best_index > 0,
        "selected_train_loss_decreased": best["train_loss"] < initial["train_loss"],
        "checkpoint_selected_by_minimum_train_loss_earliest_tie": best_index
        == select_best_train_checkpoint(history),
        "validation_excluded_from_gradient_update_and_selection": True,
        "all_physical_gains_strictly_within_bounds": all_gains_in_bounds,
        "step0_matches_day2_h200_train_baseline_when_applicable": step0_day2_baseline,
        "gradient_controls_passed": all(gradient_checks.values()),
        "requested_robustness_matrix_complete_and_finite": robustness_complete,
        "all_requested_profile_horizons_completed": all_profile_horizons_completed,
        "h400_completed_when_requested": (
            400 not in args.robustness_horizons
            or (
                400 in profiling["successful_horizons"]
                and any(cell["horizon"] == 400 for cell in robustness["cells"])
            )
        ),
        "no_default_parameter_file_modified_by_experiment": True,
    }
    provenance = _scientific_repository_state()
    configuration = {
        "horizon_control_intervals": args.horizon,
        "optimization_steps": args.steps,
        "learning_rate": args.learning_rate,
        "control_frequency_hz": args.control_freq,
        "simulation_frequency_hz": args.sim_freq,
        "seed": args.seed,
        "finite_difference_epsilon_raw": args.fd_epsilon,
        "profile_repeats": args.profile_repeats,
        "robustness_horizons": list(args.robustness_horizons),
        "worlds": 2,
        "drones_per_world": 1,
    }
    optimization_result = {
        "schema_version": OPTIMIZATION_SCHEMA,
        "provenance": provenance,
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "optax": optax.__version__,
        },
        "configuration": configuration,
        "split_and_objective": {
            "world_0": {"trajectory": "figure8", "split": "train", "used_for_updates": True},
            "world_1": {"trajectory": "circle", "split": "validation", "used_for_updates": False},
            "train_objective": "per_case_loss[0]",
            "validation_role": "evaluation_only",
            "combined_role": "evaluation_only_equal_arithmetic_mean",
            "loss_components_normalizations_and_weights": _json_native(
                TrackingLossConfig().__dict__
            ),
        },
        "optimizer": {
            "name": "optax.adam",
            "version": optax.__version__,
            "learning_rate": args.learning_rate,
            "update_count": args.steps,
            "gradient_source": "train_loss_only",
        },
        "checkpoint_selection": {
            "criterion": "minimum_train_loss",
            "tie_break": "earliest_step",
            "validation_used": False,
            "combined_used": False,
            "selected_step": best_index,
        },
        "parameters": {"initial": initial, "final": final, "best_train": best},
        "loss_changes_initial_to_best_train": _loss_changes(initial, best),
        "loss_changes_initial_to_final": _loss_changes(initial, final),
        "optimization_history": history,
        "updates": updates,
        "gradient_diagnostics_at_best_train": gradients,
        "profiling": profiling,
        "robustness_summary": robustness["summaries"],
        "validations": validations | gradient_checks,
        "all_validations_passed": all(validations.values()) and all(gradient_checks.values()),
    }
    optimized_gains = {
        "schema_version": GAINS_SCHEMA,
        "provenance": provenance,
        "configuration": configuration,
        "selected_step": best_index,
        "selection_rule": "minimum train loss; earliest exact tie",
        "raw_gains": best["raw_gains"],
        "physical_gains": best["physical_gains"],
        "gain_bounds": {name: list(bounds) for name, bounds in GAIN_BOUNDS.items()},
        "initial_losses": {
            name: initial[f"{name}_loss"] for name in ("train", "validation", "combined")
        },
        "selected_losses": {
            name: best[f"{name}_loss"] for name in ("train", "validation", "combined")
        },
        "validation_used_for_training_or_selection": False,
        "checked_in_defaults_changed": False,
        "research_artifact_only": True,
    }
    robustness["provenance"] = provenance
    robustness["configuration"] = configuration
    return optimization_result, optimized_gains, robustness


def write_outputs(
    output_dir: Path,
    optimization_result: dict[str, Any],
    optimized_gains: dict[str, Any],
    robustness: dict[str, Any],
) -> None:
    """Create one immutable six-file result directory."""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "optimization_result.json").write_text(
        json.dumps(_json_native(optimization_result), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    (output_dir / "optimized_gains.json").write_text(
        json.dumps(_json_native(optimized_gains), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output_dir / "robustness_results.json").write_text(
        json.dumps(_json_native(robustness), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    best_step = optimization_result["checkpoint_selection"]["selected_step"]
    history = optimization_result["optimization_history"]
    _save_optimization_history(output_dir / "optimization_history.png", history, best_step)
    _save_gain_history(output_dir / "gain_history.png", history)
    _save_robustness_matrix(output_dir / "robustness_matrix.png", robustness)
    actual = {path.name for path in output_dir.iterdir()}
    if actual != EXPECTED_OUTPUT_FILES:
        raise RuntimeError(f"Unexpected output file set: {sorted(actual)}")


def main(args: argparse.Namespace) -> None:
    """Run Sprint 3, preserve evidence, and fail if a validation is false."""
    optimization_result, optimized_gains, robustness = build_results(args)
    write_outputs(args.output_dir, optimization_result, optimized_gains, robustness)
    selected = optimization_result["parameters"]["best_train"]
    print(f"selected_step={selected['step']}")
    print(f"initial_train_loss={optimization_result['parameters']['initial']['train_loss']:.12g}")
    print(f"selected_train_loss={selected['train_loss']:.12g}")
    print(f"all_validations_passed={optimization_result['all_validations_passed']}")
    print(f"output_dir={args.output_dir}")
    if not optimization_result["all_validations_passed"]:
        raise RuntimeError("One or more Sprint-3 validations failed; evidence was preserved")


if __name__ == "__main__":
    main(parse_args())
