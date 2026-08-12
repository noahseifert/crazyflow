from __future__ import annotations

import json
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control.mellinger import TrackingLossConfig
from examples.jax.mellinger_batch_diagnostics import (
    CASE_NAMES,
    COMBINED_MASK,
    GAIN_BOUNDS,
    SCHEMA_VERSION,
    TRAIN_MASK,
    VALIDATION_MASK,
    build_experiment,
    masked_tracking_objective,
    run_diagnostics,
)


@pytest.fixture(scope="module")
def batch_diagnostics() -> dict[str, Any]:
    args = Namespace(
        horizon=20,
        control_freq=100,
        sim_freq=500,
        seed=20260724,
        fd_epsilon=1.0e-2,
        gain_perturbation_fraction=0.10,
        profile_repeats=1,
        output_dir=Path("unused"),
    )
    result, runtime = run_diagnostics(args)
    experiment = build_experiment(
        args.horizon, args.control_freq, args.sim_freq, args.seed, CASE_NAMES
    )
    return {"args": args, "result": result, "runtime": runtime, "experiment": experiment}


@pytest.mark.unit
def test_figure8_and_circle_share_two_world_command_batch(
    batch_diagnostics: dict[str, Any],
) -> None:
    experiment = batch_diagnostics["experiment"]
    assert experiment.commands.shape == (20, 2, 1, 13)
    assert not np.array_equal(
        np.asarray(experiment.commands[:, 0]), np.asarray(experiment.commands[:, 1])
    )


@pytest.mark.unit
def test_batch_rollout_trace_axes_and_finite_states(batch_diagnostics: dict[str, Any]) -> None:
    trace = batch_diagnostics["runtime"]["trace"]
    assert trace.pos.shape == (20, 2, 1, 3)
    assert trace.quat.shape == (20, 2, 1, 4)
    assert trace.vel.shape == (20, 2, 1, 3)
    assert trace.commanded_rotor_vel.shape == (20, 2, 1, 4)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(trace))


@pytest.mark.unit
def test_train_validation_masks_are_disjoint_nonempty_and_complete() -> None:
    assert jnp.sum(TRAIN_MASK) > 0
    assert jnp.sum(VALIDATION_MASK) > 0
    assert jnp.all(TRAIN_MASK * VALIDATION_MASK == 0)
    assert jnp.array_equal(TRAIN_MASK + VALIDATION_MASK, COMBINED_MASK)


@pytest.mark.unit
def test_per_case_loss_shape_and_explicit_combined_mean(batch_diagnostics: dict[str, Any]) -> None:
    losses = batch_diagnostics["result"]["losses"]
    per_case = jnp.asarray(
        (
            losses["per_case"]["figure8_train"]["loss_total"],
            losses["per_case"]["circle_validation"]["loss_total"],
        )
    )
    assert per_case.shape == (2,)
    assert jnp.allclose(losses["combined"]["loss_total"], jnp.mean(per_case))
    assert losses["train"]["loss_total"] == per_case[0]
    assert losses["validation"]["loss_total"] == per_case[1]


@pytest.mark.unit
def test_batch_combined_loss_matches_separate_n1_mean(batch_diagnostics: dict[str, Any]) -> None:
    checks = batch_diagnostics["result"]["aggregation_checks"]
    expected = np.mean(tuple(checks["separate_n1_losses"].values()))
    actual = batch_diagnostics["result"]["losses"]["combined"]["loss_total"]
    assert np.isclose(actual, expected, rtol=1.0e-5, atol=1.0e-7)
    assert checks["batch_per_case_loss_max_abs_difference"] <= 1.0e-7


@pytest.mark.unit
def test_batch_combined_gradient_matches_separate_n1_mean(
    batch_diagnostics: dict[str, Any],
) -> None:
    result = batch_diagnostics["result"]
    combined = result["gradients"]["combined"]["leaves"]
    separate = result["aggregation_checks"]["separate_n1_gradients"]
    for gain_name, value in combined.items():
        expected = (separate["figure8"][gain_name] + separate["circle"][gain_name]) / 2.0
        assert np.isclose(value, expected, rtol=1.0e-5, atol=1.0e-7)


@pytest.mark.unit
def test_all_gain_gradients_are_finite_and_norms_are_relevant(
    batch_diagnostics: dict[str, Any],
) -> None:
    gradients = batch_diagnostics["result"]["gradients"]
    for objective in ("train", "validation", "combined"):
        leaves = gradients[objective]["leaves"]
        assert set(leaves) == {"kp_xy", "kp_z", "kd_xy", "kd_z"}
        assert all(np.isfinite(value) for value in leaves.values())
        assert 1.0e-8 < gradients[objective]["l2_norm"] < 1.0e4


@pytest.mark.unit
def test_eager_jit_and_fixed_input_replay_match(batch_diagnostics: dict[str, Any]) -> None:
    checks = batch_diagnostics["result"]["aggregation_checks"]
    assert abs(checks["jit_eager_loss_difference"]) <= 1.0e-5
    assert checks["jit_eager_gradient_max_abs_difference"] <= 1.0e-5
    assert checks["deterministic_replay"]


@pytest.mark.unit
def test_batch_objective_jaxpr_contains_scan(batch_diagnostics: dict[str, Any]) -> None:
    experiment = batch_diagnostics["experiment"]
    objective = partial(
        masked_tracking_objective,
        step_fn=experiment.step_fn,
        steps_per_command=experiment.steps_per_command,
        config=TrackingLossConfig(),
    )
    jaxpr = jax.make_jaxpr(
        lambda variables: objective(
            variables,
            experiment.initial_data,
            experiment.commands,
            experiment.reference,
            COMBINED_MASK,
        )[0]
    )(experiment.variables)
    assert "scan[" in str(jaxpr)


@pytest.mark.unit
def test_three_train_and_combined_directional_derivatives_pass(
    batch_diagnostics: dict[str, Any],
) -> None:
    result = batch_diagnostics["result"]
    for objective in ("train", "combined"):
        checks = result["directional_derivatives"][objective]
        assert len(checks) == 3
        assert max(check["relative_error"] for check in checks) <= 5.0e-2


@pytest.mark.unit
def test_negative_gradient_step_reduces_train_and_combined_losses(
    batch_diagnostics: dict[str, Any],
) -> None:
    descent = batch_diagnostics["result"]["local_descent"]
    for objective in ("train", "combined"):
        assert descent[objective]["step_length_raw"] == 1.0e-2
        assert descent[objective]["decreased"]
        assert descent[objective]["trial_loss"] < descent[objective]["baseline_loss"]


@pytest.mark.unit
def test_case_order_does_not_change_combined_loss_or_gradient(
    batch_diagnostics: dict[str, Any],
) -> None:
    checks = batch_diagnostics["result"]["aggregation_checks"]
    assert np.isclose(checks["reordered_combined_loss_difference"], 0.0, rtol=1.0e-5, atol=1.0e-7)
    assert checks["reordered_combined_gradient_max_abs_difference"] <= 1.0e-7


@pytest.mark.unit
def test_physical_gain_perturbations_stay_inside_logistic_bounds(
    batch_diagnostics: dict[str, Any],
) -> None:
    sensitivity = batch_diagnostics["result"]["gain_sensitivity"]
    for varied_gain, diagnostic in sensitivity.items():
        assert diagnostic["bounds"] == list(GAIN_BOUNDS[varied_gain])
        for variant in ("minus", "plus"):
            physical = diagnostic[variant]["physical_gains"]
            for gain_name, value in physical.items():
                lower, upper = GAIN_BOUNDS[gain_name]
                assert lower < value < upper


@pytest.mark.unit
def test_smoke_result_schema_is_json_serializable_and_finite(
    batch_diagnostics: dict[str, Any],
) -> None:
    result = batch_diagnostics["result"]
    assert result["schema_version"] == SCHEMA_VERSION
    required = {
        "repository_state",
        "versions",
        "experiment",
        "gain_parameterization",
        "losses",
        "diagnostics",
        "gradients",
        "train_validation_gradient_cosine",
        "directional_derivatives",
        "local_descent",
        "gain_sensitivity",
        "aggregation_checks",
        "profiling",
        "validations",
        "all_validations_passed",
    }
    assert required <= result.keys()
    serialized = json.dumps(result, allow_nan=False)
    assert serialized
    assert result["all_validations_passed"]
