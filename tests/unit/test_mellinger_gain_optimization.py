from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from crazyflow.control.mellinger import (
    GAIN_BOUNDS,
    TrackingLossConfig,
    adam_optimization_step,
    evaluate_tracking_objectives,
    physical_gains,
    select_best_train_checkpoint,
    train_tracking_objective,
    tree_all_finite,
    with_controller_mass,
)
from examples.jax.mellinger_batch_diagnostics import CASE_NAMES, build_experiment
from examples.jax.mellinger_gain_optimization import (
    EXPECTED_OUTPUT_FILES,
    build_results,
    robustness_matrix,
    write_outputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _args(
    *, horizon: int = 20, steps: int = 3, robustness_horizons: list[int] | None = None
) -> Namespace:
    return Namespace(
        horizon=horizon,
        steps=steps,
        learning_rate=1.0e-2,
        control_freq=100,
        sim_freq=500,
        seed=20260724,
        fd_epsilon=1.0e-2,
        profile_repeats=1,
        robustness_horizons=robustness_horizons or [20],
        output_dir=Path("unused"),
    )


def _make_step(experiment: Any) -> tuple[Any, Any]:
    optimizer = optax.adam(1.0e-2)
    step = jax.jit(
        partial(
            adam_optimization_step,
            optimizer=optimizer,
            step_fn=experiment.step_fn,
            steps_per_command=experiment.steps_per_command,
            config=TrackingLossConfig(),
        )
    )
    return optimizer, step


def _run_three_steps(experiment: Any) -> tuple[list[Any], list[Any], list[Any]]:
    optimizer, step = _make_step(experiment)
    variables = experiment.variables
    opt_state = optimizer.init(variables)
    variables_history = [variables]
    state_history = [opt_state]
    gradients = []
    for _ in range(3):
        variables, opt_state, _, _, gradient = step(
            variables, opt_state, experiment.initial_data, experiment.commands, experiment.reference
        )
        jax.block_until_ready((variables, opt_state, gradient))
        variables_history.append(variables)
        state_history.append(opt_state)
        gradients.append(gradient)
    return variables_history, state_history, gradients


@pytest.fixture(scope="module")
def short_case() -> dict[str, Any]:
    experiment = build_experiment(20, 100, 500, 20260724, CASE_NAMES)
    variables, states, gradients = _run_three_steps(experiment)
    return {
        "experiment": experiment,
        "variables": variables,
        "states": states,
        "gradients": gradients,
    }


@pytest.fixture(scope="module")
def smoke_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return build_results(_args())


@pytest.fixture(scope="module")
def small_full_matrix(short_case: dict[str, Any]) -> dict[str, Any]:
    result, complete = robustness_matrix(
        _args(robustness_horizons=[2, 3, 4, 5]),
        short_case["variables"][0],
        short_case["variables"][-1],
    )
    assert complete
    return result


@pytest.mark.unit
def test_optax_state_raw_parameters_and_gradients_are_finite_pytrees(
    short_case: dict[str, Any],
) -> None:
    experiment = short_case["experiment"]
    for variables, state in zip(short_case["variables"], short_case["states"], strict=True):
        assert jax.tree.structure(variables) == jax.tree.structure(experiment.variables)
        assert jax.tree.structure(state) == jax.tree.structure(short_case["states"][0])
        assert tree_all_finite((variables, state))
    assert all(
        jax.tree.structure(gradient) == jax.tree.structure(experiment.variables)
        and tree_all_finite(gradient)
        for gradient in short_case["gradients"]
    )


@pytest.mark.unit
def test_train_step_gradient_ignores_validation_only_reference_change(
    short_case: dict[str, Any],
) -> None:
    experiment = short_case["experiment"]
    objective = partial(
        train_tracking_objective,
        step_fn=experiment.step_fn,
        steps_per_command=experiment.steps_per_command,
        config=TrackingLossConfig(),
    )
    changed_reference = experiment.reference.replace(
        pos=experiment.reference.pos.at[:, 1].add(jnp.asarray((0.4, -0.2, 0.1)))
    )
    baseline = jax.value_and_grad(objective, has_aux=True)(
        experiment.variables, experiment.initial_data, experiment.commands, experiment.reference
    )
    changed = jax.value_and_grad(objective, has_aux=True)(
        experiment.variables, experiment.initial_data, experiment.commands, changed_reference
    )
    assert np.array_equal(np.asarray(baseline[0][0]), np.asarray(changed[0][0]))
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(baseline[1]), jax.tree.leaves(changed[1]), strict=True
        )
    )


@pytest.mark.unit
def test_one_update_changes_raw_and_keeps_all_physical_gains_inside_bounds(
    short_case: dict[str, Any],
) -> None:
    before, after = short_case["variables"][:2]
    assert all(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
    )
    gains = physical_gains(after)
    physical = {
        "kp_xy": float(gains.kp[0]),
        "kp_z": float(gains.kp[2]),
        "kd_xy": float(gains.kd[0]),
        "kd_z": float(gains.kd[2]),
    }
    assert all(
        GAIN_BOUNDS[name][0] < value < GAIN_BOUNDS[name][1] for name, value in physical.items()
    )


@pytest.mark.unit
def test_step_zero_reproduces_day2_h200_train_baseline() -> None:
    experiment = build_experiment(200, 100, 500, 20260724, CASE_NAMES)
    evaluate = jax.jit(
        partial(
            evaluate_tracking_objectives,
            step_fn=experiment.step_fn,
            steps_per_command=experiment.steps_per_command,
            config=TrackingLossConfig(),
        )
    )
    result = evaluate(
        experiment.variables, experiment.initial_data, experiment.commands, experiment.reference
    )
    result["train_loss"].block_until_ready()
    assert np.isclose(float(result["train_loss"]), 0.07310396432876587, rtol=1.0e-5, atol=1.0e-7)


@pytest.mark.unit
def test_fixed_inputs_produce_deterministic_optimization_history(
    short_case: dict[str, Any],
) -> None:
    replay_variables, replay_states, replay_gradients = _run_three_steps(short_case["experiment"])
    for expected_tree, actual_tree in (
        *zip(short_case["variables"], replay_variables, strict=True),
        *zip(short_case["states"], replay_states, strict=True),
        *zip(short_case["gradients"], replay_gradients, strict=True),
    ):
        assert all(
            np.array_equal(np.asarray(expected), np.asarray(actual))
            for expected, actual in zip(
                jax.tree.leaves(expected_tree), jax.tree.leaves(actual_tree), strict=True
            )
        )


@pytest.mark.unit
def test_checkpoint_selection_uses_train_loss_and_earliest_tie_only() -> None:
    history = [
        {"step": 0, "train_loss": 2.0, "validation_loss": -100.0, "combined_loss": -100.0},
        {"step": 1, "train_loss": 1.0, "validation_loss": 100.0, "combined_loss": 100.0},
        {"step": 2, "train_loss": 1.0, "validation_loss": -200.0, "combined_loss": -200.0},
    ]
    assert select_best_train_checkpoint(history) == 1


@pytest.mark.unit
def test_validation_and_combined_values_do_not_change_optax_update(
    short_case: dict[str, Any],
) -> None:
    experiment = short_case["experiment"]
    optimizer, step = _make_step(experiment)
    state = optimizer.init(experiment.variables)
    changed_reference = experiment.reference.replace(
        pos=experiment.reference.pos.at[:, 1].multiply(3.0)
    )
    baseline = step(
        experiment.variables,
        state,
        experiment.initial_data,
        experiment.commands,
        experiment.reference,
    )
    changed = step(
        experiment.variables, state, experiment.initial_data, experiment.commands, changed_reference
    )
    jax.block_until_ready((baseline, changed))
    for baseline_tree, changed_tree in ((baseline[0], changed[0]), (baseline[1], changed[1])):
        assert all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree.leaves(baseline_tree), jax.tree.leaves(changed_tree), strict=True
            )
        )


@pytest.mark.unit
def test_selected_checkpoint_batch_separate_and_order_checks_pass(
    smoke_results: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    checks = smoke_results[0]["validations"]
    assert checks["batch_separate_combined_loss_match"]
    assert checks["batch_separate_combined_gradient_match"]
    assert checks["world_order_invariant_loss"]
    assert checks["world_order_invariant_gradient"]


@pytest.mark.unit
def test_selected_checkpoint_directional_derivatives_pass(
    smoke_results: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    diagnostic = smoke_results[0]["gradient_diagnostics_at_best_train"]
    assert diagnostic["maximum_directional_relative_errors"]["train"] <= 5.0e-2
    assert diagnostic["maximum_directional_relative_errors"]["combined"] <= 5.0e-2
    assert len(diagnostic["directional_derivatives"]["train"]) == 3
    assert len(diagnostic["directional_derivatives"]["combined"]) == 3


@pytest.mark.unit
def test_jit_eager_match_and_rollout_jaxpr_still_contains_scan(
    smoke_results: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    checks = smoke_results[0]["validations"]
    assert checks["jit_eager_match"]
    assert checks["rollout_jaxpr_contains_scan"]


@pytest.mark.unit
def test_experiment_local_mass_replacement_does_not_change_defaults(
    short_case: dict[str, Any],
) -> None:
    path = REPOSITORY_ROOT / "crazyflow/control/mellinger/params.toml"
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    initial_data = short_case["experiment"].initial_data
    replaced = with_controller_mass(initial_data, 0.0319)
    assert float(initial_data.controls.state.params["mass"]) == pytest.approx(0.029)
    assert float(replaced.controls.state.params["mass"]) == pytest.approx(0.0319)
    assert float(initial_data.params.mass[0, 0, 0]) == pytest.approx(0.0319)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash


@pytest.mark.unit
def test_robustness_matrix_has_exactly_32_unique_cells(small_full_matrix: dict[str, Any]) -> None:
    cells = small_full_matrix["cells"]
    assert len(cells) == 32
    assert len({cell["cell_id"] for cell in cells}) == 32


@pytest.mark.unit
def test_all_robustness_values_are_finite_with_zero_floor_and_nonfinite(
    small_full_matrix: dict[str, Any],
) -> None:
    for cell in small_full_matrix["cells"]:
        assert all(
            np.isfinite(cell[name])
            for name in (
                "loss_total",
                "loss_position",
                "loss_velocity",
                "loss_effort",
                "loss_smoothness",
                "loss_terminal",
                "loss_altitude",
                "position_rmse_m",
                "max_position_error_m",
                "motor_saturation_fraction",
                "zero_thrust_gate_fraction",
            )
        )
        assert cell["floor_clip_fraction"] == 0.0
        assert cell["nonfinite_state_fraction"] == 0.0


@pytest.mark.unit
def test_json_schemas_have_required_fields_and_are_serializable(
    smoke_results: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    optimization, gains, robustness = smoke_results
    assert {
        "schema_version",
        "provenance",
        "versions",
        "configuration",
        "split_and_objective",
        "optimizer",
        "checkpoint_selection",
        "parameters",
        "optimization_history",
        "gradient_diagnostics_at_best_train",
        "profiling",
        "validations",
        "all_validations_passed",
    } <= optimization.keys()
    assert {
        "schema_version",
        "provenance",
        "selected_step",
        "raw_gains",
        "physical_gains",
        "gain_bounds",
        "initial_losses",
        "selected_losses",
    } <= gains.keys()
    assert {
        "schema_version",
        "provenance",
        "cells",
        "summaries",
        "robustness_matrix_complete",
    } <= robustness.keys()
    assert json.dumps(smoke_results, allow_nan=False)


@pytest.mark.unit
def test_h20_three_step_smoke_writes_exactly_six_files(
    tmp_path: Path, smoke_results: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    output_dir = tmp_path / "smoke"
    write_outputs(output_dir, *smoke_results)
    assert {path.name for path in output_dir.iterdir()} == EXPECTED_OUTPUT_FILES
