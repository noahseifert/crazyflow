# ruff: noqa: E501 -- contract and presentation statements are intentionally explicit.
"""Run the frozen vertical ``ki_z`` excitation and local-sensitivity pilot.

This is a deterministic diagnostic, not optimization.  It compares one
matched controller/dynamics mass case with the audited default mass mismatch,
varies only the bounded raw ``ki_z`` variable, and applies predeclared gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.control.mellinger import (
    RolloutTrace,
    TrackingLossConfig,
    hover_rotor_velocity,
    initialize_tracking_state,
    rotor_velocity_limits,
    tracking_loss_per_case,
    tracking_loss_terms_per_case,
    with_controller_mass,
)
from crazyflow.control.mellinger.research import (
    apply_raw_gains,
    physical_from_raw,
    raw_from_data,
    specs_for_stage,
)
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator
from crazyflow.trajectory import Trajectory, state_commands

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "853b3d738233436d438d821e50198d9742965697"
BRANCH = "codex/wo-gr-s2-002"
WORKER = "WO-GR-S2-002-WORKER-01 /root/wo_gr_s2_002_worker"
WORKTREE = "/tmp/crazyflow-gradient-research-wo-gr-s2-002"
SCHEMA_VERSION = "crazyflow.ki_z_sensitivity.v1"
GO_STATUS = "GO_FOR_FUTURE_KI_Z_OPTIMIZATION"
NO_GO_STATUS = "NO_GO_KI_Z_NOT_YET_IDENTIFIABLE"
EVIDENCE_LABEL = "DIAGNOSTIC_ONLY_NOT_OPTIMIZATION"

PLATFORM = "cf2x_L250"
SEED = 20260724
SIMULATION_FREQUENCY_HZ = 500
CONTROL_FREQUENCY_HZ = 100
STEPS_PER_COMMAND = SIMULATION_FREQUENCY_HZ // CONTROL_FREQUENCY_HZ
HORIZON_CONTROL_INTERVALS = 500
DURATION_SECONDS = 5.0
REFERENCE_ALTITUDE_M = 0.75
MATCHED_MASS_KG = 0.0319
MISMATCH_CONTROLLER_MASS_KG = 0.029
GRAVITY_M_S2 = 9.81
INTEGRAL_LIMIT_M_S = 0.4
SATURATION_TOLERANCE = 1.0e-4

KI_Z_LOWER_BOUND = 0.025
KI_Z_UPPER_BOUND = 0.10
DEFAULT_KI_Z = 0.05000000074505806
DEFAULT_RAW_KI_Z = -0.6931471824645996
RAW_DELTA = 0.25
LOWER_RAW_KI_Z = -0.9431471824645996
UPPER_RAW_KI_Z = -0.4431471824645996
LOWER_PHYSICAL_KI_Z = 0.04601987823843956
UPPER_PHYSICAL_KI_Z = 0.05432434752583504
DEFAULT_TRANSFORM_SLOPE = 0.01666666753590107
FLOAT32_EPS = 1.1920928955078125e-7

EXCITATION_ABSOLUTE_THRESHOLD_M_S = 0.05
EXCITATION_CONTRAST_THRESHOLD_M_S = 0.025
INTEGRAL_CLIP_SAFE_MAX_M_S = 0.36
MOTOR_MARGIN_THRESHOLD = 0.01
AD_FD_RELATIVE_TOLERANCE = 0.10
BIAS_GRADIENT_RATIO = 1.5
BIAS_GRADIENT_DIFFERENCE_MULTIPLIER = 5.0
NONTRIVIAL_MULTIPLIER = 10.0
TRANSFORM_CONDITIONING_THRESHOLD = 0.50

CASE_NAMES = ("matched_controller_to_dynamics", "audited_controller_dynamics_mismatch")
VARIANT_NAMES = ("lower", "default", "upper")
OUTPUT_NAMES = (
    "PRESENTATION_HANDOFF.md",
    "ki_z_sensitivity.json",
    "ki_z_sensitivity.png",
    "provenance.json",
)
INPUT_SOURCE_PATHS = (
    "crazyflow/control/mellinger/research/gains.py",
    "crazyflow/control/mellinger/tracking.py",
    "crazyflow/control/mellinger/optimization.py",
    "crazyflow/control/mellinger/params.toml",
    "crazyflow/drones/params.toml",
    "examples/jax/mellinger_ki_z_sensitivity.py",
)


class PilotContractError(RuntimeError):
    """Raised before output when the frozen scientific contract is violated."""


class PilotWithheldError(RuntimeError):
    """Raised before output when a technical acceptance gate fails."""


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse the deliberately narrow pilot command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ki-z-identifiability-pilot", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day23-ki-z-sensitivity"))
    return parser.parse_args(argv)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _generator_commit() -> str:
    commit = _git("log", "-1", "--format=%H", "--", "examples/jax/mellinger_ki_z_sensitivity.py")
    if len(commit) != 40:
        raise PilotContractError("generator source must be committed before evidence generation")
    return commit


def _float(value: Any) -> float:
    result = float(np.asarray(value))
    if not math.isfinite(result):
        raise PilotWithheldError("nonfinite scalar encountered")
    return result


def _list(value: Any) -> list[Any]:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise PilotWithheldError("nonfinite trace encountered")
    return array.tolist()


def _numpy_leaf(value: Any) -> np.ndarray:
    """Expose ordinary arrays and typed PRNG keys for exact contract comparison."""
    try:
        return np.asarray(value)
    except TypeError:
        return np.asarray(jax.random.key_data(value))


def build_reference() -> tuple[Trajectory, Trajectory, jax.Array]:
    """Build the exact 5 s constant-altitude reference and state commands."""
    time = jnp.arange(HORIZON_CONTROL_INTERVALS + 1, dtype=jnp.float32) / CONTROL_FREQUENCY_HZ
    pos = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, REFERENCE_ALTITUDE_M), dtype=jnp.float32),
        (HORIZON_CONTROL_INTERVALS + 1, 3),
    )
    zeros_3 = jnp.zeros_like(pos)
    zeros_1 = jnp.zeros((HORIZON_CONTROL_INTERVALS + 1,), dtype=jnp.float32)
    full = Trajectory(time=time, pos=pos, vel=zeros_3, acc=zeros_3, yaw=zeros_1, yaw_rate=zeros_1)
    reference = jax.tree.map(lambda value: value[1:], full)
    commands = state_commands(reference)[:, None, None, :]
    return full, reference, commands


def build_simulation() -> tuple[Sim, Any, Any, Any, jax.Array]:
    """Build the shared simulation and both only-mass-different initial states."""
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=PLATFORM,
        dynamics=Dynamics.first_principles,
        control=Control.state,
        integrator=Integrator.euler,
        freq=SIMULATION_FREQUENCY_HZ,
        state_freq=CONTROL_FREQUENCY_HZ,
        attitude_freq=SIMULATION_FREQUENCY_HZ,
        force_torque_freq=SIMULATION_FREQUENCY_HZ,
        device="cpu",
        rng_key=SEED,
    )
    full, reference, commands = build_reference()
    initial = initialize_tracking_state(sim.data, full.pos[0], full.vel[0])
    state = initial.controls.state
    if state is None:
        raise PilotContractError("Control.state is unavailable")
    dynamics_mass = _float(initial.params.mass[0, 0, 0])
    default_controller_mass = _float(state.params["mass"])
    if not np.isclose(dynamics_mass, MATCHED_MASS_KG, rtol=0.0, atol=1.0e-7):
        raise PilotContractError(f"unexpected dynamics mass: {dynamics_mass}")
    if not np.isclose(default_controller_mass, MISMATCH_CONTROLLER_MASS_KG, rtol=0.0, atol=1.0e-7):
        raise PilotContractError(f"unexpected controller mass: {default_controller_mass}")
    if not np.array_equal(np.asarray(initial.states.quat[0, 0]), np.asarray((0.0, 0.0, 0.0, 1.0))):
        raise PilotContractError("initial quaternion is not identity")
    matched = with_controller_mass(initial, MATCHED_MASS_KG)
    mismatch = with_controller_mass(initial, MISMATCH_CONTROLLER_MASS_KG)
    normalized_matched = with_controller_mass(matched, 0.0)
    normalized_mismatch = with_controller_mass(mismatch, 0.0)
    if jax.tree.structure(normalized_matched) != jax.tree.structure(normalized_mismatch):
        raise PilotContractError("mass-case PyTree structures differ")
    if not all(
        np.array_equal(_numpy_leaf(left), _numpy_leaf(right))
        for left, right in zip(
            jax.tree.leaves(normalized_matched), jax.tree.leaves(normalized_mismatch), strict=True
        )
    ):
        raise PilotContractError("matched and mismatch cases differ outside controller mass")
    return sim, matched, mismatch, reference, commands


def stage2_gain_contract(initial_data: Any) -> dict[str, Any]:
    """Validate and return the unchanged Registry transform contract."""
    specs = specs_for_stage(2)
    names = tuple(spec.name for spec in specs)
    if names != ("kp_xy", "kp_z", "kd_xy", "kd_z", "ki_xy", "ki_z"):
        raise PilotContractError(f"unexpected Stage-2 registry order: {names}")
    index = names.index("ki_z")
    spec = specs[index]
    if (spec.lower, spec.upper, spec.stage) != (KI_Z_LOWER_BOUND, KI_Z_UPPER_BOUND, 2):
        raise PilotContractError("ki_z registry metadata changed")
    raw = raw_from_data(initial_data, stage=2)
    values = {
        "lower": raw.at[index].set(jnp.asarray(LOWER_RAW_KI_Z, dtype=raw.dtype)),
        "default": raw,
        "upper": raw.at[index].set(jnp.asarray(UPPER_RAW_KI_Z, dtype=raw.dtype)),
    }
    expected_raw = {"lower": LOWER_RAW_KI_Z, "default": DEFAULT_RAW_KI_Z, "upper": UPPER_RAW_KI_Z}
    expected_physical = {
        "lower": LOWER_PHYSICAL_KI_Z,
        "default": DEFAULT_KI_Z,
        "upper": UPPER_PHYSICAL_KI_Z,
    }
    physical: dict[str, float] = {}
    for name, candidate in values.items():
        changed = np.flatnonzero(np.asarray(candidate) != np.asarray(raw)).tolist()
        if name == "default" and changed:
            raise PilotContractError("default raw vector changed")
        if name != "default" and changed != [index]:
            raise PilotContractError(f"{name} changes fields other than ki_z: {changed}")
        if not np.isclose(_float(candidate[index]), expected_raw[name], rtol=0.0, atol=2.0e-7):
            raise PilotContractError(f"unexpected {name} raw ki_z")
        mapped = physical_from_raw(candidate, stage=2)
        physical[name] = _float(mapped["ki_z"])
        if not np.isclose(physical[name], expected_physical[name], rtol=0.0, atol=2.0e-7):
            raise PilotContractError(f"unexpected {name} physical ki_z")
    transform_slope = (KI_Z_UPPER_BOUND - KI_Z_LOWER_BOUND) * float(
        jax.nn.sigmoid(jnp.asarray(DEFAULT_RAW_KI_Z, dtype=jnp.float32))
        * (1.0 - jax.nn.sigmoid(jnp.asarray(DEFAULT_RAW_KI_Z, dtype=jnp.float32)))
    )
    if not np.isclose(transform_slope, DEFAULT_TRANSFORM_SLOPE, rtol=0.0, atol=2.0e-9):
        raise PilotContractError("unexpected default transform slope")
    conditioning_ratio = transform_slope / ((KI_Z_UPPER_BOUND - KI_Z_LOWER_BOUND) / 4.0)
    return {
        "index": index,
        "names": list(names),
        "base_raw_vector": _list(raw),
        "raw_vectors": {name: _list(value) for name, value in values.items()},
        "raw_values": {name: _float(value[index]) for name, value in values.items()},
        "physical_values_N_per_m_s": physical,
        "bounds_N_per_m_s": [KI_Z_LOWER_BOUND, KI_Z_UPPER_BOUND],
        "raw_delta": RAW_DELTA,
        "transform_slope_N_per_m_s_per_raw": transform_slope,
        "transform_conditioning_ratio": conditioning_ratio,
        "only_ki_z_changes": True,
    }


def rollout_with_integral(
    initial_data: Any, commands: jax.Array, step_fn: Callable[[Any, int], Any]
) -> tuple[Any, RolloutTrace, jax.Array]:
    """Roll out state commands while recording the existing Z-integrator state."""

    def scan_step(data: Any, command: jax.Array) -> tuple[Any, tuple[RolloutTrace, jax.Array]]:
        data = F.state_control(data, command)
        data = step_fn(data, STEPS_PER_COMMAND)
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
        return data, (trace, state_control.pos_err_i)

    final, (trace, integral) = jax.lax.scan(scan_step, initial_data, commands)
    return final, trace, integral


def _variant_data(initial_data: Any, raw_vector: jax.Array) -> Any:
    return apply_raw_gains(initial_data, raw_vector, stage=2)


def _loss_for_raw_ki_z(
    raw_ki_z: jax.Array,
    initial_data: Any,
    base_raw: jax.Array,
    ki_z_index: int,
    commands: jax.Array,
    reference: Trajectory,
    step_fn: Callable[[Any, int], Any],
) -> jax.Array:
    raw = base_raw.at[ki_z_index].set(raw_ki_z)
    data = _variant_data(initial_data, raw)
    _, trace, _ = rollout_with_integral(data, commands, step_fn)
    losses, _ = tracking_loss_per_case(
        trace,
        reference,
        hover_rotor_velocity(initial_data),
        rotor_velocity_limits(initial_data),
        TrackingLossConfig(),
    )
    return losses[0]


def _term_record(terms: Any) -> dict[str, dict[str, float]]:
    return {
        name: {
            "raw": _float(terms.raw[name][0]),
            "normalization_divisor": _float(terms.normalization_divisor[name]),
            "normalized": _float(terms.normalized[name][0]),
            "weight": _float(terms.weight[name]),
            "weighted": _float(terms.weighted[name][0]),
        }
        for name in ("position", "velocity", "effort", "smoothness", "terminal", "altitude")
    }


def _rollout_record(
    initial_data: Any,
    raw_vector: jax.Array,
    raw_ki_z: float,
    physical_ki_z: float,
    commands: jax.Array,
    reference: Trajectory,
    step_fn: Callable[[Any, int], Any],
) -> dict[str, Any]:
    data = _variant_data(initial_data, raw_vector)
    _, trace, integral = rollout_with_integral(data, commands, step_fn)
    limits = rotor_velocity_limits(initial_data)
    hover = hover_rotor_velocity(initial_data)
    losses, metrics = tracking_loss_per_case(trace, reference, hover, limits, TrackingLossConfig())
    terms = tracking_loss_terms_per_case(trace, reference, hover, TrackingLossConfig())
    jax.block_until_ready((trace, integral, losses, metrics, terms))

    actual_z = np.asarray(trace.pos[:, 0, 0, 2])
    reference_z = np.asarray(reference.pos[:, 2])
    signed_error = actual_z - reference_z
    controller_error = -signed_error
    vertical_velocity = np.asarray(trace.vel[:, 0, 0, 2])
    integral_z = np.asarray(integral[:, 0, 0, 2])
    commanded = np.asarray(trace.commanded_rotor_vel[:, 0, 0, :])
    minimum = float(np.asarray(limits[0]))
    maximum = float(np.asarray(limits[1]))
    lower_margin_rpm = commanded - minimum
    upper_margin_rpm = maximum - commanded
    nearest_margin_rpm = np.minimum(lower_margin_rpm, upper_margin_rpm)
    lower_margin = lower_margin_rpm / (maximum - minimum)
    upper_margin = upper_margin_rpm / (maximum - minimum)
    normalized_margin = nearest_margin_rpm / (maximum - minimum)
    lower_saturation = (commanded > 0.5 * minimum) & (
        commanded <= minimum * (1.0 + SATURATION_TOLERANCE)
    )
    upper_saturation = commanded >= maximum * (1.0 - SATURATION_TOLERANCE)
    saturated = lower_saturation | upper_saturation
    metric_values = {name: _float(value[0]) for name, value in metrics.items()}
    finite = all(
        np.all(np.isfinite(value))
        for value in (
            actual_z,
            reference_z,
            signed_error,
            vertical_velocity,
            integral_z,
            commanded,
            normalized_margin,
        )
    ) and all(math.isfinite(value) for value in metric_values.values())
    if not finite:
        raise PilotWithheldError("nonfinite rollout value")
    max_abs_integral = float(np.max(np.abs(integral_z)))
    return {
        "raw_ki_z": raw_ki_z,
        "physical_ki_z_N_per_m_s": physical_ki_z,
        "state_time_s": _list(reference.time),
        "reference_altitude_m": _list(reference_z),
        "actual_altitude_m": _list(actual_z),
        "signed_altitude_error_actual_minus_reference_m": _list(signed_error),
        "controller_altitude_error_reference_minus_actual_m": _list(controller_error),
        "vertical_velocity_m_s": _list(vertical_velocity),
        "integral_state_z_m_s": _list(integral_z),
        "integral_contribution_z_N": _list(physical_ki_z * integral_z),
        "commanded_rotor_rpm": _list(commanded),
        "rotor_limits_rpm": {"minimum": minimum, "maximum": maximum},
        "lower_motor_margin_rpm": _list(lower_margin_rpm),
        "upper_motor_margin_rpm": _list(upper_margin_rpm),
        "nearest_motor_bound_margin_rpm": _list(nearest_margin_rpm),
        "lower_motor_margin_normalized": _list(lower_margin),
        "upper_motor_margin_normalized": _list(upper_margin),
        "nearest_motor_bound_margin_normalized": _list(normalized_margin),
        "lower_saturation": lower_saturation.tolist(),
        "upper_saturation": upper_saturation.tolist(),
        "metrics": metric_values,
        "loss_v1_terms": _term_record(terms),
        "descriptive_metrics": {
            "terminal_signed_error_m": float(signed_error[-1]),
            "terminal_absolute_error_m": float(abs(signed_error[-1])),
            "rmse_altitude_error_m": float(np.sqrt(np.mean(signed_error**2))),
            "p95_absolute_altitude_error_m": float(np.percentile(np.abs(signed_error), 95)),
            "maximum_absolute_altitude_error_m": float(np.max(np.abs(signed_error))),
            "max_abs_integral_z_m_s": max_abs_integral,
            "integrator_margin_normalized": (INTEGRAL_LIMIT_M_S - max_abs_integral)
            / INTEGRAL_LIMIT_M_S,
            "minimum_motor_bound_margin_rpm": float(np.min(nearest_margin_rpm)),
            "minimum_motor_bound_margin_normalized": float(np.min(normalized_margin)),
            "motor_saturated_sample_count": int(np.sum(saturated)),
            "motor_saturation_fraction": float(np.mean(saturated, dtype=np.float32)),
            "lower_saturation_count": int(np.sum(lower_saturation)),
            "upper_saturation_count": int(np.sum(upper_saturation)),
            "zero_thrust_fraction": metric_values["zero_thrust_gate_fraction"],
            "ground_floor_fraction": metric_values["floor_clip_fraction"],
            "nonfinite_fraction": metric_values["nonfinite_state_fraction"],
        },
    }


def _gradient_record(losses: dict[str, float], gradient: float) -> dict[str, Any]:
    loss_scale = max(1.0, *(abs(value) for value in losses.values()))
    loss_floor = 64.0 * FLOAT32_EPS * loss_scale
    gradient_floor = loss_floor / RAW_DELTA
    central_fd = (losses["upper"] - losses["lower"]) / (2.0 * RAW_DELTA)
    lower_secant = (losses["default"] - losses["lower"]) / RAW_DELTA
    upper_secant = (losses["upper"] - losses["default"]) / RAW_DELTA
    tolerance = max(gradient_floor, AD_FD_RELATIVE_TOLERANCE * max(abs(gradient), abs(central_fd)))
    sign_required = max(abs(gradient), abs(central_fd)) > gradient_floor
    same_sign = (gradient == 0.0 and central_fd == 0.0) or (gradient * central_fd > 0.0)
    consistency = abs(gradient - central_fd) <= tolerance and (not sign_required or same_sign)
    return {
        "jax_ad_d_loss_v1_d_raw_ki_z": gradient,
        "central_finite_difference": central_fd,
        "lower_one_sided_secant": lower_secant,
        "upper_one_sided_secant": upper_secant,
        "losses": losses,
        "loss_scale": loss_scale,
        "loss_roundoff_floor": loss_floor,
        "gradient_roundoff_floor": gradient_floor,
        "absolute_ad_fd_difference": abs(gradient - central_fd),
        "ad_fd_tolerance": tolerance,
        "same_sign_required": sign_required,
        "same_sign": same_sign,
        "ad_fd_consistency_pass": consistency,
    }


def build_evidence() -> dict[str, Any]:
    """Execute the frozen six rollouts, gradients, gates, and recommendation."""
    sim, matched, mismatch, reference, commands = build_simulation()
    gain_contract = stage2_gain_contract(matched)
    raw_vectors = {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in gain_contract["raw_vectors"].items()
    }
    step_fn = sim.build_step_fn()
    case_initial = {
        "matched_controller_to_dynamics": matched,
        "audited_controller_dynamics_mismatch": mismatch,
    }
    controller_masses = {
        "matched_controller_to_dynamics": MATCHED_MASS_KG,
        "audited_controller_dynamics_mismatch": MISMATCH_CONTROLLER_MASS_KG,
    }
    cases: dict[str, Any] = {}
    for case_name in CASE_NAMES:
        initial = case_initial[case_name]
        variants = {
            variant: _rollout_record(
                initial,
                raw_vectors[variant],
                gain_contract["raw_values"][variant],
                gain_contract["physical_values_N_per_m_s"][variant],
                commands,
                reference,
                step_fn,
            )
            for variant in VARIANT_NAMES
        }
        objective = lambda raw: _loss_for_raw_ki_z(  # noqa: E731
            raw,
            initial,
            raw_vectors["default"],
            gain_contract["index"],
            commands,
            reference,
            step_fn,
        )
        default_loss, gradient = jax.value_and_grad(objective)(
            jnp.asarray(gain_contract["raw_values"]["default"], dtype=jnp.float32)
        )
        jax.block_until_ready((default_loss, gradient))
        losses = {name: variants[name]["metrics"]["loss_total"] for name in VARIANT_NAMES}
        if not np.isclose(_float(default_loss), losses["default"], rtol=0.0, atol=2.0e-7):
            raise PilotContractError(f"{case_name} objective/recorded default loss mismatch")
        gradient_record = _gradient_record(losses, _float(gradient))
        cases[case_name] = {
            "controller_mass_kg": controller_masses[case_name],
            "dynamics_mass_kg": MATCHED_MASS_KG,
            "variants": variants,
            "local_sensitivity_at_default": gradient_record,
        }

    technical_failures: list[str] = []
    for case_name, case in cases.items():
        if not case["local_sensitivity_at_default"]["ad_fd_consistency_pass"]:
            technical_failures.append(f"{case_name}:ad_fd_consistency")
        for variant_name, variant in case["variants"].items():
            desc = variant["descriptive_metrics"]
            prefix = f"{case_name}:{variant_name}"
            if desc["motor_saturated_sample_count"] != 0:
                technical_failures.append(f"{prefix}:motor_saturation")
            if desc["max_abs_integral_z_m_s"] > INTEGRAL_CLIP_SAFE_MAX_M_S:
                technical_failures.append(f"{prefix}:integrator_clip_margin")
            if desc["minimum_motor_bound_margin_normalized"] < MOTOR_MARGIN_THRESHOLD:
                technical_failures.append(f"{prefix}:motor_margin")
            if desc["zero_thrust_fraction"] != 0.0:
                technical_failures.append(f"{prefix}:zero_thrust")
            if desc["ground_floor_fraction"] != 0.0:
                technical_failures.append(f"{prefix}:ground_floor")
            if desc["nonfinite_fraction"] != 0.0:
                technical_failures.append(f"{prefix}:nonfinite")
    if technical_failures:
        raise PilotWithheldError("technical gates failed: " + ", ".join(technical_failures))

    matched_default = cases["matched_controller_to_dynamics"]["variants"]["default"]
    mismatch_default = cases["audited_controller_dynamics_mismatch"]["variants"]["default"]
    matched_gradient = cases["matched_controller_to_dynamics"]["local_sensitivity_at_default"]
    mismatch_gradient = cases["audited_controller_dynamics_mismatch"][
        "local_sensitivity_at_default"
    ]
    mismatch_integral = mismatch_default["descriptive_metrics"]["max_abs_integral_z_m_s"]
    matched_integral = matched_default["descriptive_metrics"]["max_abs_integral_z_m_s"]
    excitation_absolute = mismatch_integral >= EXCITATION_ABSOLUTE_THRESHOLD_M_S
    excitation_contrast = mismatch_integral - matched_integral >= EXCITATION_CONTRAST_THRESHOLD_M_S
    mismatch_ad = mismatch_gradient["jax_ad_d_loss_v1_d_raw_ki_z"]
    matched_ad = matched_gradient["jax_ad_d_loss_v1_d_raw_ki_z"]
    mismatch_loss_delta = max(
        abs(mismatch_gradient["losses"]["lower"] - mismatch_gradient["losses"]["default"]),
        abs(mismatch_gradient["losses"]["upper"] - mismatch_gradient["losses"]["default"]),
    )
    nontrivial_gradient = (
        abs(mismatch_ad) >= NONTRIVIAL_MULTIPLIER * mismatch_gradient["gradient_roundoff_floor"]
    )
    nontrivial_loss = (
        mismatch_loss_delta >= NONTRIVIAL_MULTIPLIER * mismatch_gradient["loss_roundoff_floor"]
    )
    bias_ratio = abs(mismatch_ad) >= BIAS_GRADIENT_RATIO * max(
        abs(matched_ad), matched_gradient["gradient_roundoff_floor"]
    )
    shared_gradient_floor = max(
        matched_gradient["gradient_roundoff_floor"], mismatch_gradient["gradient_roundoff_floor"]
    )
    bias_difference = abs(mismatch_ad - matched_ad) >= (
        BIAS_GRADIENT_DIFFERENCE_MULTIPLIER * shared_gradient_floor
    )
    transform_conditioned = (
        gain_contract["transform_conditioning_ratio"] >= TRANSFORM_CONDITIONING_THRESHOLD
    ) and all(
        KI_Z_LOWER_BOUND < value < KI_Z_UPPER_BOUND
        for value in gain_contract["physical_values_N_per_m_s"].values()
    )
    scientific_gates = {
        "excitation_absolute": {
            "pass": excitation_absolute,
            "value_m_s": mismatch_integral,
            "threshold_m_s": EXCITATION_ABSOLUTE_THRESHOLD_M_S,
        },
        "excitation_contrast": {
            "pass": excitation_contrast,
            "value_m_s": mismatch_integral - matched_integral,
            "threshold_m_s": EXCITATION_CONTRAST_THRESHOLD_M_S,
        },
        "nontrivial_gradient": {
            "pass": nontrivial_gradient,
            "absolute_gradient": abs(mismatch_ad),
            "threshold": NONTRIVIAL_MULTIPLIER * mismatch_gradient["gradient_roundoff_floor"],
        },
        "nontrivial_loss_perturbation": {
            "pass": nontrivial_loss,
            "maximum_absolute_loss_change": mismatch_loss_delta,
            "threshold": NONTRIVIAL_MULTIPLIER * mismatch_gradient["loss_roundoff_floor"],
        },
        "bias_gradient_ratio": {
            "pass": bias_ratio,
            "mismatch_abs_gradient": abs(mismatch_ad),
            "matched_or_floor_reference": max(
                abs(matched_ad), matched_gradient["gradient_roundoff_floor"]
            ),
            "required_ratio": BIAS_GRADIENT_RATIO,
        },
        "bias_gradient_difference": {
            "pass": bias_difference,
            "absolute_gradient_difference": abs(mismatch_ad - matched_ad),
            "threshold": BIAS_GRADIENT_DIFFERENCE_MULTIPLIER * shared_gradient_floor,
        },
        "transform_conditioning": {
            "pass": transform_conditioned,
            "ratio": gain_contract["transform_conditioning_ratio"],
            "threshold": TRANSFORM_CONDITIONING_THRESHOLD,
            "all_values_strictly_within_bounds": True,
        },
    }
    failed_scientific = [name for name, gate in scientific_gates.items() if not gate["pass"]]
    recommendation = GO_STATUS if not failed_scientific else NO_GO_STATUS
    nominal_force_difference = (MATCHED_MASS_KG - MISMATCH_CONTROLLER_MASS_KG) * GRAVITY_M_S2
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": recommendation,
        "evidence_label": EVIDENCE_LABEL,
        "claim_boundary": {
            "proves": "whether default ki_z is locally identifiable under this one frozen simulated vertical hold and mass-bias contrast",
            "does_not_prove": "optimization, improvement, superiority, convergence, robustness, hardware transfer, or flight readiness",
            "presentation_use": "backup or next-step evidence only; the Friday deck does not depend on this pilot",
        },
        "contract": {
            "platform": PLATFORM,
            "dynamics": "first_principles",
            "control": "state",
            "integrator": "explicit_euler",
            "device": "cpu",
            "n_worlds": 1,
            "n_drones": 1,
            "seed": SEED,
            "simulation_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "state_control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "attitude_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "force_torque_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "steps_per_state_command": STEPS_PER_COMMAND,
            "horizon_control_intervals": HORIZON_CONTROL_INTERVALS,
            "duration_seconds": DURATION_SECONDS,
            "reference_position_m": [0.0, 0.0, REFERENCE_ALTITUDE_M],
            "reference_velocity_m_s": [0.0, 0.0, 0.0],
            "reference_acceleration_m_s2": [0.0, 0.0, 0.0],
            "reference_yaw_rad": 0.0,
            "initial_position_m": [0.0, 0.0, REFERENCE_ALTITUDE_M],
            "initial_velocity_m_s": [0.0, 0.0, 0.0],
            "initial_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "initial_rotor_velocity_basis": "physical hover from unchanged dynamics mass",
            "dynamics_mass_kg": MATCHED_MASS_KG,
            "cases": {
                "matched_controller_to_dynamics": {"controller_mass_kg": MATCHED_MASS_KG},
                "audited_controller_dynamics_mismatch": {
                    "controller_mass_kg": MISMATCH_CONTROLLER_MASS_KG
                },
            },
            "only_case_difference": "controller mass",
            "only_case_difference_programmatically_verified": True,
            "nominal_force_difference_N": nominal_force_difference,
            "pd_error_scale_at_default_kp_z_m": nominal_force_difference / 1.25,
            "integral_limit_z_m_s": INTEGRAL_LIMIT_M_S,
            "loss": "unchanged TrackingLossConfig loss v1",
            "horizon_justification": "5 s supplies 500 integral updates under a 0.0029 kg audited mass bias while remaining a bounded diagnostic well below a deliberately clipped long run",
        },
        "gain_contract": gain_contract,
        "predeclared_thresholds": {
            "float32_epsilon": FLOAT32_EPS,
            "loss_roundoff_floor_formula": "64*eps*max(1,abs(L_lower),abs(L_default),abs(L_upper))",
            "gradient_roundoff_floor_formula": "loss_roundoff_floor/0.25",
            "ad_fd_tolerance_formula": "max(gradient_roundoff_floor,0.10*max(abs(g_ad),abs(g_fd)))",
            "excitation_absolute_m_s": EXCITATION_ABSOLUTE_THRESHOLD_M_S,
            "excitation_contrast_m_s": EXCITATION_CONTRAST_THRESHOLD_M_S,
            "integrator_max_abs_m_s": INTEGRAL_CLIP_SAFE_MAX_M_S,
            "nontrivial_multiplier_over_roundoff": NONTRIVIAL_MULTIPLIER,
            "bias_gradient_ratio": BIAS_GRADIENT_RATIO,
            "bias_gradient_difference_multiplier": BIAS_GRADIENT_DIFFERENCE_MULTIPLIER,
            "transform_conditioning_ratio": TRANSFORM_CONDITIONING_THRESHOLD,
            "motor_bound_margin_normalized": MOTOR_MARGIN_THRESHOLD,
            "motor_saturation_tolerance": SATURATION_TOLERANCE,
        },
        "cases": cases,
        "technical_gates": {
            "pass": True,
            "all_values_and_gradients_finite": True,
            "all_six_rollouts_zero_motor_saturation": True,
            "all_six_rollouts_integrator_margin_at_least_10_percent": True,
            "all_six_rollouts_motor_margin_at_least_0_01": True,
            "all_six_rollouts_zero_ground_floor_and_zero_thrust": True,
            "both_ad_fd_checks_pass": True,
            "contract_only_mass_difference": True,
            "only_ki_z_varied": True,
        },
        "scientific_gates": scientific_gates,
        "failed_scientific_criteria": failed_scientific,
        "recommendation": recommendation,
        "recommendation_scope": "future ki_z optimization Work Order consideration only; no optimization is authorized",
    }
    return result


def _plot_from_json(path: Path, result: dict[str, Any]) -> None:
    """Render the single diagnostic figure solely from serialized JSON values."""
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    colors = {
        "matched_controller_to_dynamics": "#1f77b4",
        "audited_controller_dynamics_mismatch": "#d62728",
    }
    labels = {
        "matched_controller_to_dynamics": "matched mass",
        "audited_controller_dynamics_mismatch": "audited mismatch",
    }
    for case_name in CASE_NAMES:
        default = result["cases"][case_name]["variants"]["default"]
        time = default["state_time_s"]
        axes[0, 0].plot(
            time, default["actual_altitude_m"], color=colors[case_name], label=labels[case_name]
        )
        axes[0, 1].plot(
            time,
            default["controller_altitude_error_reference_minus_actual_m"],
            color=colors[case_name],
            label=labels[case_name],
        )
        axes[1, 0].plot(
            time, default["integral_state_z_m_s"], color=colors[case_name], label=labels[case_name]
        )
    axes[0, 0].axhline(
        REFERENCE_ALTITUDE_M, color="black", linestyle="--", linewidth=1.0, label="reference"
    )
    axes[0, 0].set(title="Default ki_z: altitude", ylabel="z [m]")
    axes[0, 1].set(title="Default ki_z: controller error", ylabel="reference - actual [m]")
    axes[1, 0].axhline(
        EXCITATION_ABSOLUTE_THRESHOLD_M_S, color="#777777", linestyle=":", linewidth=1.0
    )
    axes[1, 0].axhline(
        -EXCITATION_ABSOLUTE_THRESHOLD_M_S, color="#777777", linestyle=":", linewidth=1.0
    )
    axes[1, 0].set(
        title="Default ki_z: integral excitation", xlabel="time [s]", ylabel="integral z [m s]"
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for case_name in CASE_NAMES:
        local = result["cases"][case_name]["local_sensitivity_at_default"]
        raw_values = [result["gain_contract"]["raw_values"][name] for name in VARIANT_NAMES]
        losses = [local["losses"][name] for name in VARIANT_NAMES]
        axes[1, 1].plot(
            raw_values, losses, marker="o", color=colors[case_name], label=labels[case_name]
        )
        default_raw = result["gain_contract"]["raw_values"]["default"]
        default_loss = local["losses"]["default"]
        gradient = local["jax_ad_d_loss_v1_d_raw_ki_z"]
        tangent_x = np.asarray((raw_values[0], raw_values[-1]))
        axes[1, 1].plot(
            tangent_x,
            default_loss + gradient * (tangent_x - default_raw),
            color=colors[case_name],
            linestyle="--",
            linewidth=1.0,
        )
    axes[1, 1].set(
        title="Loss v1 sensitivity (points) and local AD tangent",
        xlabel="bounded raw ki_z variable",
        ylabel="loss v1",
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    failed = result["failed_scientific_criteria"]
    subtitle = f"{result['recommendation']} | zero saturation | failed scientific criteria: {', '.join(failed) if failed else 'none'}"
    figure.suptitle(
        "Vertical ki_z excitation and local-sensitivity pilot\n" + subtitle, fontsize=12
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(
        path, dpi=180, metadata={"Software": "Crazyflow deterministic research diagnostic"}
    )
    plt.close(figure)


def _presentation_handoff(result: dict[str, Any]) -> bytes:
    mismatch = result["cases"]["audited_controller_dynamics_mismatch"]
    matched = result["cases"]["matched_controller_to_dynamics"]
    mismatch_integral = mismatch["variants"]["default"]["descriptive_metrics"][
        "max_abs_integral_z_m_s"
    ]
    matched_integral = matched["variants"]["default"]["descriptive_metrics"][
        "max_abs_integral_z_m_s"
    ]
    mismatch_gradient = mismatch["local_sensitivity_at_default"]["jax_ad_d_loss_v1_d_raw_ki_z"]
    failed = result["failed_scientific_criteria"]
    text = f"""# Presentation handoff — vertical ki_z sensitivity pilot

- Status: `{result["recommendation"]}`
- Evidence label: `{EVIDENCE_LABEL}`
- Presentation role: backup/next-step evidence only; the Friday deck is independent.

## What the figure proves

For the one frozen 5 s `cf2x_L250` simulation hold, it shows the matched-mass reference and the audited 0.029 kg controller versus 0.0319 kg dynamics mismatch, the default integral response, and the predeclared local Loss-v1 sensitivity check for `ki_z`. All six Lower/Default/Upper rollouts are finite, saturation-free, clear the motor/integrator/ground gates, and both AD/central-FD checks pass.

## Exact evidence

- Mismatch default maximum absolute integral state: `{mismatch_integral:.10g} m s`.
- Matched default maximum absolute integral state: `{matched_integral:.10g} m s`.
- Mismatch local `d loss_v1 / d raw_ki_z`: `{mismatch_gradient:.10g}`.
- Failed predeclared scientific criteria: `{", ".join(failed) if failed else "none"}`.

## What it does not prove

No optimizer step occurred. This does not prove improvement, superiority, convergence, robustness, hardware transfer, flight readiness, or that a future optimization will succeed. It changes no default, registry bound, loss, platform, firmware, or accepted Friday artifact.

## One-sentence backup-slide takeaway

`{result["recommendation"]}: the frozen vertical mass-bias hold {"meets" if not failed else "does not meet"} every predeclared ki_z identifiability threshold, with zero saturation; this is a diagnostic authorization signal only, not an optimized result.`

## Exact verbal caveat

“This is one deterministic simulation sensitivity pilot under the known cf2x_L250 mass mismatch; it contains no optimization and makes no flight or hardware-transfer claim.”
"""
    return text.encode()


def _provenance(result: dict[str, Any], generator_commit: str) -> dict[str, Any]:
    commands = [
        "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-s2-002 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/crazyflow-gradient-s2-002-matplotlib /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python examples/jax/mellinger_ki_z_sensitivity.py --run-ki-z-identifiability-pilot --output-dir artifacts/day23-ki-z-sensitivity",
        "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-s2-002 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/crazyflow-gradient-s2-002-matplotlib /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_mellinger_ki_z_sensitivity.py tests/unit/test_mellinger_research.py tests/unit/test_mellinger_tracking.py tests/unit/test_mellinger_gain_optimization.py",
        "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-s2-002 PYTHONDONTWRITEBYTECODE=1 RUFF_CACHE_DIR=/tmp/crazyflow-gradient-s2-002-ruff /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m ruff check examples/jax/mellinger_ki_z_sensitivity.py tests/unit/test_mellinger_ki_z_sensitivity.py",
        "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-s2-002 PYTHONDONTWRITEBYTECODE=1 /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m ruff format --check examples/jax/mellinger_ki_z_sensitivity.py tests/unit/test_mellinger_ki_z_sensitivity.py",
        "(cd artifacts/day23-ki-z-sensitivity && sha256sum --quiet -c SHA256SUMS)",
    ]
    return {
        "schema_version": "crazyflow.ki_z_sensitivity.provenance.v1",
        "base_commit": BASE_COMMIT,
        "generator_commit": generator_commit,
        "branch": BRANCH,
        "worker": WORKER,
        "worktree": WORKTREE,
        "status": result["recommendation"],
        "frozen_contract_identity": _sha256_bytes(
            _json_bytes(
                {
                    "contract": result["contract"],
                    "gain_contract": result["gain_contract"],
                    "predeclared_thresholds": result["predeclared_thresholds"],
                }
            )
        ),
        "input_sources": [
            {"path": path, "sha256": _sha256_path(REPOSITORY_ROOT / path)}
            for path in INPUT_SOURCE_PATHS
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "exact_commands": commands,
        "protected_inputs_opened": [],
        "optimizer_steps": 0,
        "output_inventory": [*OUTPUT_NAMES, "SHA256SUMS"],
        "sha256_indexed_payloads": list(OUTPUT_NAMES),
        "determinism_rule": "two fresh packages from generator commit must be byte-identical",
        "presentation_dependency": "none; backup/next-step evidence only",
    }


def emit_package(output_dir: Path) -> dict[str, Any]:
    """Build evidence in memory, then atomically populate a new output directory."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PilotContractError(f"refusing to overwrite nonempty output directory: {output_dir}")
    result = build_evidence()
    generator_commit = _generator_commit()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "ki_z_sensitivity.json"
    result_path.write_bytes(_json_bytes(result))
    _plot_from_json(output_dir / "ki_z_sensitivity.png", json.loads(result_path.read_text()))
    (output_dir / "PRESENTATION_HANDOFF.md").write_bytes(_presentation_handoff(result))
    (output_dir / "provenance.json").write_bytes(_json_bytes(_provenance(result, generator_commit)))
    checksum_lines = [f"{_sha256_path(output_dir / name)}  {name}" for name in OUTPUT_NAMES]
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    return result


def main(argv: list[str] | tuple[str, ...] | None = None) -> None:
    args = parse_args(argv)
    result = emit_package(args.output_dir)
    print(f"recommendation={result['recommendation']}")
    print(f"failed_scientific_criteria={result['failed_scientific_criteria']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
