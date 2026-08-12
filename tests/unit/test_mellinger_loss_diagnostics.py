from __future__ import annotations

import hashlib
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import pytest

from crazyflow.control.mellinger import (
    GAIN_VARIABLE_NAMES,
    TRACKING_LOSS_TERM_NAMES,
    TrackingLossConfig,
    tracking_loss_diagnostics_per_case,
    tracking_objective_per_case,
)
from examples.jax.mellinger_batch_diagnostics import CASE_NAMES, build_experiment
from examples.jax.mellinger_loss_diagnostics import build_report, write_artifacts

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE_METRICS = {
    "loss_total": (0.0022911163978278637, 0.0011237584985792637),
    "loss_position": (0.0006717538926750422, 0.0003500662569422275),
    "loss_velocity": (0.0012258307542651892, 0.0005780282081104815),
    "loss_effort": (2.1055286651971983e-06, 1.560200303174497e-06),
    "loss_smoothness": (6.09204420243259e-08, 3.042393359464768e-08),
    "loss_terminal": (0.0003913654072675854, 0.00019407337822485715),
    "loss_altitude": (2.085060422929441e-12, 2.084624139975233e-12),
    "position_rmse_m": (0.0064795538783073425, 0.004677514545619488),
    "velocity_rmse_m_s": (0.11071723699569702, 0.07602816820144653),
    "max_position_error_m": (0.01563980057835579, 0.011013439856469631),
    "control_effort": (0.00210552872158587, 0.0015602002386003733),
    "control_smoothness": (6.092043986427598e-05, 3.042393109353725e-05),
    "motor_saturation_fraction": (0.0, 0.0),
    "zero_thrust_gate_fraction": (0.0, 0.0),
    "floor_clip_fraction": (0.0, 0.0),
    "nonfinite_state_fraction": (0.0, 0.0),
}


@pytest.fixture(scope="module")
def diagnostic_case() -> dict[str, Any]:
    batch = build_experiment(20, 100, 500, 20260724, CASE_NAMES)
    config = TrackingLossConfig()
    diagnostic_fn = jax.jit(
        partial(
            tracking_loss_diagnostics_per_case,
            initial_data=batch.initial_data,
            commands=batch.commands,
            reference=batch.reference,
            step_fn=batch.step_fn,
            steps_per_command=batch.steps_per_command,
            config=config,
        )
    )
    reference_fn = partial(
        tracking_objective_per_case,
        initial_data=batch.initial_data,
        commands=batch.commands,
        reference=batch.reference,
        step_fn=batch.step_fn,
        steps_per_command=batch.steps_per_command,
        config=config,
    )
    diagnostics = diagnostic_fn(batch.variables)
    reference_loss, reference_metrics = jax.jit(reference_fn)(batch.variables)
    reference_jacobian = jax.jit(jax.jacrev(lambda variables: reference_fn(variables)[0]))(
        batch.variables
    )
    jax.block_until_ready((diagnostics, reference_loss, reference_metrics, reference_jacobian))
    return {
        "diagnostics": diagnostics,
        "reference_loss": reference_loss,
        "reference_metrics": reference_metrics,
        "reference_jacobian": reference_jacobian,
    }


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return build_report(20, 20260724)


@pytest.mark.unit
def test_loss_terms_reconstruct_unchanged_reference(diagnostic_case: dict[str, Any]) -> None:
    diagnostics = diagnostic_case["diagnostics"]
    contributions = np.stack(
        [np.asarray(diagnostics.terms.weighted[name]) for name in TRACKING_LOSS_TERM_NAMES], axis=-1
    )
    assert set(diagnostics.terms.raw) == set(TRACKING_LOSS_TERM_NAMES)
    assert np.allclose(
        np.sum(contributions, axis=-1),
        np.asarray(diagnostic_case["reference_loss"]),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert np.allclose(
        np.asarray(diagnostics.total_loss),
        np.asarray(diagnostic_case["reference_loss"]),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert set(diagnostics.per_case_metrics) == set(REFERENCE_METRICS)
    for name, expected in REFERENCE_METRICS.items():
        assert np.allclose(
            np.asarray(diagnostics.per_case_metrics[name]), expected, rtol=1.0e-6, atol=1.0e-8
        )
        assert np.allclose(
            np.asarray(diagnostic_case["reference_metrics"][name]),
            expected,
            rtol=1.0e-6,
            atol=1.0e-8,
        )


@pytest.mark.unit
def test_term_gain_jacobians_sum_to_total_gradient(diagnostic_case: dict[str, Any]) -> None:
    diagnostics = diagnostic_case["diagnostics"]
    jacobian = diagnostics.weighted_gradient_by_bound_variable
    reference = diagnostic_case["reference_jacobian"]
    for gain_name in GAIN_VARIABLE_NAMES:
        term_values = np.asarray(getattr(jacobian, gain_name))
        assert term_values.shape == (2, 6)
        assert np.all(np.isfinite(term_values))
        assert np.allclose(
            np.sum(term_values, axis=-1),
            np.asarray(getattr(reference, gain_name)),
            rtol=1.0e-5,
            atol=1.0e-7,
        )


@pytest.mark.unit
def test_report_is_narrow_complete_and_clears_smoke_gates(report: dict[str, Any]) -> None:
    assert report["schema_version"] == "crazyflow.mellinger_loss_diagnostics.v1"
    assert report["claim_boundary"] == {
        "classification": "infrastructure_smoke",
        "optimizer_updates": 0,
        "evaluated_gain_state": "crazyflow_default",
        "supports_convergence_claim": False,
        "supports_generalization_claim": False,
        "supports_hardware_claim": False,
        "supports_controller_superiority_claim": False,
    }
    assert report["experiment"]["platform"] == "cf2x_L250"
    assert report["experiment"]["seed"] == 20260724
    assert report["experiment"]["horizon_control_intervals"] == 20
    assert [case["split"] for case in report["cases"]] == ["train", "validation"]
    assert [case["trajectory_type"] for case in report["cases"]] == ["figure8", "circle"]
    assert report["loss_contract"]["term_order"] == list(TRACKING_LOSS_TERM_NAMES)
    assert report["gains"]["bound_variable_names"] == list(GAIN_VARIABLE_NAMES)
    assert report["all_validations_passed"]
    assert all(report["validations"].values())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.unit
def test_artifacts_are_json_driven_and_deterministic(
    report: dict[str, Any], tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_paths = write_artifacts(report, first)
    second_paths = write_artifacts(report, second)
    expected_names = {
        "loss_diagnostics.json",
        "loss_contributions.png",
        "loss_term_gain_gradients.png",
        "SHA256SUMS",
    }
    assert {path.name for path in first_paths} == expected_names
    assert {path.name for path in first.iterdir()} == expected_names
    assert {path.name for path in second.iterdir()} == expected_names
    assert {_sha256(path): path.name for path in first_paths} == {
        _sha256(path): path.name for path in second_paths
    }
    assert (first / "loss_contributions.png").stat().st_size > 0
    assert (first / "loss_term_gain_gradients.png").stat().st_size > 0
    assert (first / "SHA256SUMS").read_text() == (second / "SHA256SUMS").read_text()
