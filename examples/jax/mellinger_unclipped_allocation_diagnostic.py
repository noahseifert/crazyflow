# ruff: noqa: E501, I001 -- exact lines and required Crazyflow-before-SciPy import order.
"""Replay the frozen Day-22 case with diagnostic-local unclipped allocation traces.

The production controller, allocator, defaults, and acceptance gate remain unchanged.  This
command duplicates only transparent Float32 arithmetic immediately around the two production
allocation calls, validates the duplicate against production and accepted Day-22 evidence, and
then emits a failed-gate diagnostic package.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import array_api_extra as xpx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from array_api_compat import array_namespace

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import crazyflow.sim.functional as F
from scipy.spatial.transform import Rotation as R
from crazyflow.control import Control
from crazyflow.control.core import controllable
from crazyflow.control.mellinger import (
    RolloutTrace,
    hover_rotor_velocity,
    initialize_tracking_state,
)
from crazyflow.control.mellinger.control import force_torque_pwms2pwms
from crazyflow.control.mellinger.research import apply_raw_gains
from crazyflow.control.transform import force2pwm, motor_force2rotor_vel, pwm2force
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator
from crazyflow.trajectory import state_commands
from crazyflow.utils import leaf_replace
from examples.jax import mellinger_friday_evidence as friday
from examples.jax import mellinger_saturation_diagnostic as saturation
from examples.jax.mellinger_batch_diagnostics import make_trajectory

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "bcc8b0d57c9f88dd1279ec94a05e14b45776ed0b"
BRANCH = "codex/wo-gr-f0-003"
WORKER = "WO-GR-F0-003-WORKER-01 /root/wo_gr_f0_003_worker"
WORKTREE = "/tmp/crazyflow-gradient-research-wo-gr-f0-003"
SCHEMA_VERSION = "crazyflow.unclipped_allocation_diagnostic.v1"
STATUS = "DIAGNOSTIC_ONLY_SATURATION_PRESENT"
GATE_FAILURE_STATUS = "FAIL_ZERO_MOTOR_SATURATION_GATE"
SATURATION_DETECTION_THRESHOLD = 1.0e-4
FLOAT32_EPS = 1.1920928955078125e-7
RECONSTRUCTION_FACTOR = 64.0
WRENCH_PLATFORM_SCALE = np.asarray(
    [0.48, 0.0156144, 0.0156144, 0.0035284141711519001], dtype=np.float64
)
WRENCH_RELATIVE_FLOOR = 1024.0 * FLOAT32_EPS * WRENCH_PLATFORM_SCALE
CONFIGURED_PWM_LIMITS = (7000.0, 65535.0)
CONFIGURED_FORCE_LIMITS_N = (0.012817578393224994, 0.12)
CONFIGURED_RPM2THRUST = (0.0, -5.382196214637237e-7, 2.4582929831265485e-10)
FIGURE_WINDOW_START_INDEX = 44
FIGURE_WINDOW_END_INDEX = 62
EXPECTED_PIPELINE = (
    "state_controller",
    "attitude_controller",
    "force_torque_controller",
    "integration",
    "increment_steps",
    "clip_floor_pos",
)
TRACE_FIELDS = (
    "pos",
    "quat",
    "vel",
    "ang_vel",
    "rotor_vel",
    "commanded_rotor_vel",
    "state_command",
    "attitude_command",
    "force_torque_command",
)
DAY19_DIR = REPOSITORY_ROOT / "artifacts/day19-h100-analysis"
DAY22_DIR = REPOSITORY_ROOT / "artifacts/day22-saturation-diagnostic"
DAY22_OUTPUT_NAMES = (
    "PRESENTATION_HANDOFF.md",
    "motor_saturation_detail.png",
    "provenance.json",
    "saturation_diagnostic.json",
    "trajectory_error_saturation.png",
)
OUTPUT_NAMES = (
    "PRESENTATION_HANDOFF.md",
    "provenance.json",
    "unclipped_allocation_diagnostic.json",
    "unclipped_allocation_diagnostic.png",
)
INDEX_PATTERN = re.compile(r"([0-9a-f]{64})  (?:\./)?(.+)")


class DiagnosticError(RuntimeError):
    """Raised before output when a frozen identity or numerical gate fails."""


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse the deliberately separate diagnostic-only command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-only-unclipped-allocation",
        action="store_true",
        required=True,
        help="acknowledge failed-gate diagnostic scope without changing allocation",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/day24-unclipped-allocation")
    )
    return parser.parse_args(argv)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise DiagnosticError(f"expected JSON object: {path}")
    return value


def _parse_checksum_index(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = INDEX_PATTERN.fullmatch(line)
        if match is None:
            raise DiagnosticError(f"invalid checksum line in {path}: {line!r}")
        digest, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in entries:
            raise DiagnosticError(f"unsafe or duplicate checksum path in {path}: {relative}")
        entries[relative] = digest
    return entries


def _verify_index(directory: Path, expected_names: tuple[str, ...]) -> list[dict[str, str]]:
    index_path = directory / "SHA256SUMS"
    entries = _parse_checksum_index(index_path)
    if tuple(sorted(entries)) != tuple(sorted(expected_names)):
        raise DiagnosticError(f"checksum inventory mismatch: {index_path}")
    identities = []
    for name in sorted(expected_names):
        path = directory / name
        actual = _sha256_path(path)
        if actual != entries[name]:
            raise DiagnosticError(f"accepted payload identity mismatch: {path}")
        identities.append({"path": path.relative_to(REPOSITORY_ROOT).as_posix(), "sha256": actual})
    return identities


def load_day22() -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Verify the accepted Day-22 package and load its derived JSON only."""
    identities = _verify_index(DAY22_DIR, DAY22_OUTPUT_NAMES)
    report = _load_object(DAY22_DIR / "saturation_diagnostic.json")
    if report.get("status") != STATUS:
        raise DiagnosticError("accepted Day-22 diagnostic status changed")
    if report.get("normal_acceptance_gate", {}).get("status") != GATE_FAILURE_STATUS:
        raise DiagnosticError("accepted Day-22 zero-saturation gate changed")
    return report, identities


def load_data_audit() -> dict[str, Any]:
    """Crosscheck DATA-AUDIT-01 from the permitted derivative and analyzer source."""
    identities = _verify_index(
        DAY19_DIR,
        (
            "SOURCE_RUN_INDEXES.sha256",
            "analyze_h100_replication.py",
            "h100_replication_analysis.json",
        ),
    )
    checksum_index_sha256 = _sha256_path(DAY19_DIR / "SHA256SUMS")
    expected_checksum_index = "fe607e7ee7b129e6e9741d2cc62e40625080b408388fcf4d584aca47789b8b17"
    if checksum_index_sha256 != expected_checksum_index:
        raise DiagnosticError("accepted Day-19 checksum-index identity changed")
    analysis = _load_object(DAY19_DIR / "h100_replication_analysis.json")
    seeds = analysis.get("seeds", [])
    if analysis.get("status") != "PASS" or len(seeds) != 10:
        raise DiagnosticError("Day-19 derivative seed/status contract changed")
    expected_rows = {"total": 10000, "train": 5000, "validation": 5000, "test": 0}
    if any(seed.get("metric_rows") != expected_rows for seed in seeds):
        raise DiagnosticError("Day-19 per-seed metric-row contract changed")
    if analysis["aggregates"].get("total_metric_rows") != 100000:
        raise DiagnosticError("Day-19 total Train/Validation row count changed")
    if analysis["aggregates"].get("total_test_metric_rows") != 0:
        raise DiagnosticError("Day-19 derivative reports Test metric rows")
    per_seed_maxima = [
        seed["technical_metric_maxima"]["motor_saturation_fraction"] for seed in seeds
    ]
    panel_maximum = analysis["aggregates"]["panel_technical_metric_maxima"][
        "motor_saturation_fraction"
    ]
    if per_seed_maxima != [0.0] * 10 or panel_maximum != 0.0:
        raise DiagnosticError("Day-19 registered saturation maximum changed")

    source = (DAY19_DIR / "analyze_h100_replication.py").read_text()
    required_source_fragments = (
        'if any(row.get("split") == "test" for row in metrics):',
        'for split in ("train", "validation")',
        "if observed_order != expected_order:",
        'name: max(float(row["metrics"][name]) for row in metrics)',
        'name: max(seed["technical_metric_maxima"][name] for seed in seeds)',
    )
    if any(fragment not in source for fragment in required_source_fragments):
        raise DiagnosticError("Day-19 analyzer source logic crosscheck changed")
    return {
        "status": "PASS_DERIVED_DATA_ONLY",
        "day19_checksum_index_sha256": checksum_index_sha256,
        "verified_payloads": identities,
        "seed_count": 10,
        "rows_per_seed": expected_rows,
        "total_train_validation_rows": 100000,
        "total_test_rows": 0,
        "per_seed_motor_saturation_fraction_maxima": per_seed_maxima,
        "panel_motor_saturation_fraction_maximum": panel_maximum,
        "analyzer_source_crosscheck": {
            "rejects_test_rows": True,
            "requires_complete_alternating_train_validation_order": True,
            "per_seed_maximum_over_all_metric_rows": True,
            "panel_maximum_over_per_seed_maxima": True,
            "analyzer_executed": False,
        },
        "claim_boundary": (
            "This proves only that no saturation was registered in stored H100 Train/Validation "
            "objective rows; it provides no motor trace, reserve, or allocation-distortion evidence."
        ),
    }


def _wrench_from_motor_forces(motor_forces: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    xp = array_namespace(motor_forces)
    torque = (params["mixing_matrix"] @ motor_forces[..., None])[..., 0]
    torque = torque * xp.stack([params["L"], params["L"], params["thrust2torque"]])
    collective = xp.sum(motor_forces, axis=-1)[..., None]
    return xp.concat((collective, torque), axis=-1)


def stage_a_diagnostic(data: Any) -> dict[str, jax.Array]:
    """Repeat transparent Stage-A Float32 arithmetic immediately before production."""
    states = data.states
    control = data.controls.attitude
    if control is None:
        raise DiagnosticError("Stage A requires the attitude controller")
    mask = controllable(data.core.steps, data.core.freq, control.steps, control.freq)
    control = leaf_replace(control, mask, cmd=control.staged_cmd)
    params = control.params
    xp = array_namespace(states.quat)
    force_des = control.cmd[..., 3]
    rpy_des = control.cmd[..., :3]
    dt = 1 / control.freq

    rot = R.from_quat(states.quat)
    rot_des = R.from_euler("xyz", rpy_des, degrees=False)
    rotation_delta = (rot_des.inv() * rot).as_matrix()
    error_matrix = rotation_delta - rotation_delta.mT
    rotation_error = xp.stack(
        (error_matrix[..., 2, 1], error_matrix[..., 0, 2], error_matrix[..., 1, 0]), axis=-1
    )
    angular_velocity_error = -states.ang_vel
    angular_velocity_derivative_error = -(states.ang_vel - control.last_ang_vel) / dt
    angular_velocity_derivative_error = xpx.at(angular_velocity_derivative_error)[..., 2].set(0)
    integral_error = xp.clip(
        control.r_int_error - rotation_error * dt, -params["int_err_max"], params["int_err_max"]
    )
    raw_torque_pwm = (
        -params["kR"] * rotation_error
        + params["kw"] * angular_velocity_error
        + params["ki_m"] * integral_error
        + params["kd_omega"] * angular_velocity_derivative_error
    )
    torque_pwm = xp.clip(raw_torque_pwm, -params["torque_pwm_max"], params["torque_pwm_max"])
    torque_pwm = xp.where((force_des > 0)[..., None], torque_pwm, 0.0)
    force_des_pwm = force2pwm(force_des / 4, params["thrust_max"], params["pwm_max"])
    motor_pwm_preclip = force_torque_pwms2pwms(force_des_pwm, torque_pwm, params["mixing_matrix"])
    all_zero = xp.all(motor_pwm_preclip == 0, axis=-1, keepdims=True)
    motor_pwm_postclip = xp.where(
        all_zero, 0.0, xp.clip(motor_pwm_preclip, params["pwm_min"], params["pwm_max"])
    )
    motor_force_preclip = pwm2force(motor_pwm_preclip, params["thrust_max"], params["pwm_max"])
    motor_force_postclip = pwm2force(motor_pwm_postclip, params["thrust_max"], params["pwm_max"])
    rpm2thrust = data.controls.force_torque.params["rpm2thrust"]
    motor_speed_preclip = motor_force2rotor_vel(motor_force_preclip, rpm2thrust)
    motor_speed_postclip = motor_force2rotor_vel(motor_force_postclip, rpm2thrust)

    pwm_lower = params["pwm_min"]
    pwm_upper = params["pwm_max"]
    force_lower = pwm2force(pwm_lower, params["thrust_max"], pwm_upper)
    force_upper = params["thrust_max"]
    speed_lower = motor_force2rotor_vel(force_lower, rpm2thrust)
    speed_upper = motor_force2rotor_vel(force_upper, rpm2thrust)
    pwm_lower_exceedance = xp.maximum(pwm_lower - motor_pwm_preclip, 0)
    pwm_upper_exceedance = xp.maximum(motor_pwm_preclip - pwm_upper, 0)
    force_lower_exceedance = xp.maximum(force_lower - motor_force_preclip, 0)
    force_upper_exceedance = xp.maximum(motor_force_preclip - force_upper, 0)
    speed_lower_exceedance = xp.maximum(speed_lower - motor_speed_preclip, 0)
    speed_upper_exceedance = xp.maximum(motor_speed_preclip - speed_upper, 0)
    wrench_preclip = _wrench_from_motor_forces(motor_force_preclip, params)
    wrench_postclip = _wrench_from_motor_forces(motor_force_postclip, params)
    wrench_distortion = wrench_postclip - wrench_preclip
    return {
        "collective_force_request_N": force_des,
        "force_des_pwm_per_motor": force_des_pwm,
        "raw_torque_pwm": raw_torque_pwm,
        "postclip_torque_pwm": torque_pwm,
        "torque_lower_clip_mask": raw_torque_pwm < -params["torque_pwm_max"],
        "torque_upper_clip_mask": raw_torque_pwm > params["torque_pwm_max"],
        "torque_any_clip_mask": xp.abs(raw_torque_pwm) > params["torque_pwm_max"],
        "motor_pwm_preclip": motor_pwm_preclip,
        "motor_pwm_postclip": motor_pwm_postclip,
        "motor_pwm_lower_clip_mask": motor_pwm_preclip < pwm_lower,
        "motor_pwm_upper_clip_mask": motor_pwm_preclip > pwm_upper,
        "motor_pwm_any_clip_mask": (motor_pwm_preclip < pwm_lower)
        | (motor_pwm_preclip > pwm_upper),
        "motor_force_preclip_N": motor_force_preclip,
        "motor_force_postclip_N": motor_force_postclip,
        "motor_speed_preclip": motor_speed_preclip,
        "motor_speed_postclip": motor_speed_postclip,
        "pwm_lower_exceedance": pwm_lower_exceedance,
        "pwm_upper_exceedance": pwm_upper_exceedance,
        "pwm_lower_exceedance_normalized": pwm_lower_exceedance / (pwm_upper - pwm_lower),
        "pwm_upper_exceedance_normalized": pwm_upper_exceedance / (pwm_upper - pwm_lower),
        "force_lower_exceedance_N": force_lower_exceedance,
        "force_upper_exceedance_N": force_upper_exceedance,
        "force_lower_exceedance_normalized": force_lower_exceedance / (force_upper - force_lower),
        "force_upper_exceedance_normalized": force_upper_exceedance / (force_upper - force_lower),
        "speed_lower_exceedance": speed_lower_exceedance,
        "speed_upper_exceedance": speed_upper_exceedance,
        "speed_lower_exceedance_normalized": speed_lower_exceedance / (speed_upper - speed_lower),
        "speed_upper_exceedance_normalized": speed_upper_exceedance / (speed_upper - speed_lower),
        "wrench_preclip": wrench_preclip,
        "wrench_postclip": wrench_postclip,
        "wrench_distortion_signed": wrench_distortion,
        "wrench_distortion_absolute": xp.abs(wrench_distortion),
        "wrench_distortion_platform_normalized": xp.abs(wrench_distortion)
        / xp.asarray(WRENCH_PLATFORM_SCALE, dtype=wrench_distortion.dtype),
        "speed_inverse_discriminant": rpm2thrust[1] ** 2
        - 4 * rpm2thrust[2] * (rpm2thrust[0] - motor_force_preclip),
    }


def stage_b_diagnostic(data: Any) -> dict[str, jax.Array]:
    """Repeat transparent Stage-B Float32 arithmetic immediately before production."""
    control = data.controls.force_torque
    if control is None:
        raise DiagnosticError("Stage B requires the force/torque controller")
    mask = controllable(data.core.steps, data.core.freq, control.steps, control.freq)
    control = leaf_replace(control, mask, cmd=control.staged_cmd)
    params = control.params
    xp = array_namespace(control.cmd)
    force = control.cmd[..., [0]]
    torque = control.cmd[..., 1:]
    torque_forces = (
        torque * xp.asarray([1 / params["L"], 1 / params["L"], 1 / params["thrust2torque"]])
    ) @ params["mixing_matrix"]
    motor_force_preclip = (torque_forces + force) / 4
    force_is_zero = xp.all(force == 0, axis=-1, keepdims=True)
    motor_force_postclip = xp.where(
        force_is_zero, 0.0, xp.clip(motor_force_preclip, params["thrust_min"], params["thrust_max"])
    )
    commanded_speed = motor_force2rotor_vel(motor_force_postclip, params["rpm2thrust"])
    lower_exceedance = xp.maximum(params["thrust_min"] - motor_force_preclip, 0)
    upper_exceedance = xp.maximum(motor_force_preclip - params["thrust_max"], 0)
    wrench_preclip = _wrench_from_motor_forces(motor_force_preclip, params)
    wrench_postclip = _wrench_from_motor_forces(motor_force_postclip, params)
    distortion = wrench_postclip - wrench_preclip
    return {
        "input_stage_a_production_wrench": control.cmd,
        "motor_force_preclip_N": motor_force_preclip,
        "motor_force_postclip_N": motor_force_postclip,
        "commanded_motor_speed": commanded_speed,
        "motor_force_lower_clip_mask": motor_force_preclip < params["thrust_min"],
        "motor_force_upper_clip_mask": motor_force_preclip > params["thrust_max"],
        "motor_force_any_clip_mask": (motor_force_preclip < params["thrust_min"])
        | (motor_force_preclip > params["thrust_max"]),
        "force_lower_exceedance_N": lower_exceedance,
        "force_upper_exceedance_N": upper_exceedance,
        "force_lower_exceedance_normalized": lower_exceedance
        / (params["thrust_max"] - params["thrust_min"]),
        "force_upper_exceedance_normalized": upper_exceedance
        / (params["thrust_max"] - params["thrust_min"]),
        "wrench_preclip": wrench_preclip,
        "wrench_postclip": wrench_postclip,
        "wrench_distortion_signed": distortion,
        "wrench_distortion_absolute": xp.abs(distortion),
        "wrench_distortion_platform_normalized": xp.abs(distortion)
        / xp.asarray(WRENCH_PLATFORM_SCALE, dtype=distortion.dtype),
    }


def _make_instrumented_rollout(sim: Sim) -> Any:
    pipeline = tuple(sim.step_pipeline.items())
    if tuple(name for name, _ in pipeline) != EXPECTED_PIPELINE:
        raise DiagnosticError("production step-pipeline identity changed")

    def single_step(data: Any, _: None) -> tuple[Any, dict[str, dict[str, jax.Array]]]:
        stage_a: dict[str, jax.Array] | None = None
        stage_b: dict[str, jax.Array] | None = None
        for name, function in pipeline:
            if name == "attitude_controller":
                stage_a = stage_a_diagnostic(data)
            if name == "force_torque_controller":
                stage_b = stage_b_diagnostic(data)
            data = function(data)
            if name == "attitude_controller":
                assert stage_a is not None
                stage_a["production_postclip_wrench"] = data.controls.force_torque.staged_cmd
            if name == "force_torque_controller":
                assert stage_b is not None
                stage_b["production_commanded_motor_speed"] = data.controls.rotor_vel
        assert stage_a is not None and stage_b is not None
        return data, {"stage_a": stage_a, "stage_b": stage_b}

    def interval_step(data: Any, command: jax.Array) -> tuple[Any, dict[str, Any]]:
        data = F.state_control(data, command)
        data, substeps = jax.lax.scan(single_step, data, None, length=5, unroll=1)
        data = data.replace(core=data.core.replace(mjx_synced=False))
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
        fifth = jax.tree.map(lambda value: value[-1], substeps)
        return data, {"trace": trace, "diagnostic": fifth}

    @jax.jit
    def rollout(initial_data: Any, commands: jax.Array) -> tuple[Any, Any]:
        return jax.lax.scan(interval_step, initial_data, commands)

    return rollout


def _new_frozen_sim() -> Sim:
    return Sim(
        n_worlds=1,
        n_drones=1,
        drone="cf2x_L250",
        dynamics=Dynamics.first_principles,
        control=Control.state,
        integrator=Integrator.euler,
        freq=friday.SIMULATION_FREQUENCY_HZ,
        state_freq=friday.CONTROL_FREQUENCY_HZ,
        attitude_freq=friday.SIMULATION_FREQUENCY_HZ,
        force_torque_freq=friday.SIMULATION_FREQUENCY_HZ,
        device="cpu",
        rng_key=friday.ROLLOUT_SEED,
    )


def _all_finite(label: str, value: Any) -> None:
    for leaf in jax.tree.leaves(value):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise DiagnosticError(f"nonfinite numeric value in {label}")


def _require_array_equal(label: str, left: Any, right: Any) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (
        left_array.shape != right_array.shape
        or left_array.dtype != right_array.dtype
        or not np.array_equal(left_array, right_array)
    ):
        maximum_error = (
            float(np.max(np.abs(left_array.astype(float) - right_array.astype(float))))
            if left_array.shape == right_array.shape and left_array.size
            else math.inf
        )
        raise DiagnosticError(
            f"exact array identity failed for {label}: "
            f"{left_array.shape}/{left_array.dtype} versus {right_array.shape}/{right_array.dtype}, "
            f"maximum error {maximum_error}"
        )


def _reconstruction_check(label: str, left: Any, right: Any, scale: Any) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise DiagnosticError(f"reconstruction shape mismatch for {label}")
    error = np.abs(left_array - right_array)
    scale_array = np.asarray(scale, dtype=np.float64)
    bound = (
        RECONSTRUCTION_FACTOR
        * FLOAT32_EPS
        * np.maximum(scale_array, np.maximum(np.abs(left_array), np.abs(right_array)))
    )
    if np.any(error > bound):
        raise DiagnosticError(
            f"Float32 reconstruction tolerance exceeded for {label}: "
            f"{float(error.max())} > {float(bound.max())}"
        )
    return {
        "maximum_absolute_error": float(error.max(initial=0.0)),
        "maximum_allowed_error": float(bound.max(initial=0.0)),
        "factor_times_float32_eps": RECONSTRUCTION_FACTOR * FLOAT32_EPS,
        "rtol": 0.0,
        "passed": True,
    }


def _squeeze_case(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[0] != friday.HORIZON or array.shape[1:3] != (1, 1):
        raise DiagnosticError(f"unexpected diagnostic array shape: {array.shape}")
    return array[:, 0, 0]


def _intervals(mask: np.ndarray) -> list[dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    result = []
    index = 0
    while index < friday.HORIZON:
        if not bool(mask[index]):
            index += 1
            continue
        start = index
        while index + 1 < friday.HORIZON and bool(mask[index + 1]):
            index += 1
        end = index
        count = end - start + 1
        result.append(
            {
                "start_index_inclusive": start,
                "end_index_inclusive": end,
                "start_time_s": start / friday.CONTROL_FREQUENCY_HZ,
                "end_time_s": (end + 1) / friday.CONTROL_FREQUENCY_HZ,
                "sample_count": count,
                "duration_s": count / friday.CONTROL_FREQUENCY_HZ,
            }
        )
        index += 1
    return result


def _mask_summary(mask: np.ndarray, channel_label: str) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (friday.HORIZON, mask.shape[1]):
        raise DiagnosticError(f"unexpected mask shape: {mask.shape}")
    channels = []
    for channel in range(mask.shape[1]):
        intervals = _intervals(mask[:, channel])
        channels.append(
            {
                channel_label: channel,
                "count": int(mask[:, channel].sum()),
                "indices_zero_based": np.flatnonzero(mask[:, channel]).astype(int).tolist(),
                "state_times_s": (
                    (np.flatnonzero(mask[:, channel]) + 1) / friday.CONTROL_FREQUENCY_HZ
                ).tolist(),
                "intervals": intervals,
                "longest_duration_s": max(
                    (interval["duration_s"] for interval in intervals), default=0.0
                ),
            }
        )
    return {
        "count_all_channels": int(mask.sum()),
        "sample_count_all_channels": int(mask.size),
        "any": bool(mask.any()),
        "by_channel": channels,
    }


def _minmax_by_channel(values: np.ndarray, channel_label: str) -> list[dict[str, Any]]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    result = []
    for channel in range(values.shape[1]):
        column = values[:, channel]
        minimum_index = int(np.argmin(column))
        maximum_index = int(np.argmax(column))
        result.append(
            {
                channel_label: channel,
                "minimum": float(column[minimum_index]),
                "minimum_index": minimum_index,
                "minimum_state_time_s": (minimum_index + 1) / friday.CONTROL_FREQUENCY_HZ,
                "maximum": float(column[maximum_index]),
                "maximum_index": maximum_index,
                "maximum_state_time_s": (maximum_index + 1) / friday.CONTROL_FREQUENCY_HZ,
            }
        )
    return result


def _maximum_record(values: np.ndarray, channel_label: str) -> dict[str, Any]:
    values = np.asarray(values)
    flat_index = int(np.argmax(values))
    index = np.unravel_index(flat_index, values.shape)
    time_index = int(index[0])
    channel = int(index[1]) if values.ndim > 1 else 0
    return {
        "value": float(values[index]),
        channel_label: channel,
        "index_zero_based": time_index,
        "state_time_s": (time_index + 1) / friday.CONTROL_FREQUENCY_HZ,
        "interval_start_s": time_index / friday.CONTROL_FREQUENCY_HZ,
        "interval_end_s": (time_index + 1) / friday.CONTROL_FREQUENCY_HZ,
    }


def _relative_wrench_payload(preclip: np.ndarray, distortion: np.ndarray) -> dict[str, Any]:
    defined = np.abs(preclip) >= WRENCH_RELATIVE_FLOOR
    values: list[list[float | None]] = []
    for row, row_mask, desired in zip(distortion, defined, preclip, strict=True):
        values.append(
            [
                float(value / baseline) if active else None
                for value, active, baseline in zip(row, row_mask, desired, strict=True)
            ]
        )
    return {
        "definition": "signed_distortion / desired_preclip_axis only where floor is met",
        "floors": WRENCH_RELATIVE_FLOOR.tolist(),
        "defined_mask": defined.tolist(),
        "values_or_null_when_undefined": values,
    }


def _wrench_summary(
    preclip: np.ndarray,
    postclip: np.ndarray,
    distortion: np.ndarray,
    motor_mask: np.ndarray,
    torque_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    names = ("collective_N", "roll_Nm", "pitch_Nm", "yaw_Nm")
    records = []
    for axis, name in enumerate(names):
        index = int(np.argmax(np.abs(distortion[:, axis])))
        desired = float(preclip[index, axis])
        signed = float(distortion[index, axis])
        relative = signed / desired if abs(desired) >= WRENCH_RELATIVE_FLOOR[axis] else None
        record = {
            "axis_index": axis,
            "axis": name,
            "maximum_absolute_distortion": abs(signed),
            "signed_distortion_at_maximum": signed,
            "desired_preclip_at_maximum": desired,
            "realized_postclip_at_maximum": float(postclip[index, axis]),
            "platform_normalized_absolute_distortion": abs(signed) / WRENCH_PLATFORM_SCALE[axis],
            "relative_distortion_when_defined": relative,
            "relative_floor": float(WRENCH_RELATIVE_FLOOR[axis]),
            "index_zero_based": index,
            "state_time_s": (index + 1) / friday.CONTROL_FREQUENCY_HZ,
            "contributing_clipped_motors": np.flatnonzero(motor_mask[index]).astype(int).tolist(),
        }
        if torque_mask is not None:
            record["contributing_clipped_torque_axes"] = (
                np.flatnonzero(torque_mask[index]).astype(int).tolist()
            )
        records.append(record)
    return records


def _array_section(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "contracts": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "values": {name: value.tolist() for name, value in arrays.items()},
    }


def _stage_a_section(arrays: dict[str, np.ndarray], params: dict[str, Any]) -> dict[str, Any]:
    motor_mask = arrays["motor_pwm_any_clip_mask"]
    torque_mask = arrays["torque_any_clip_mask"]
    pre_wrench = arrays["wrench_preclip"]
    post_wrench = arrays["wrench_postclip"]
    distortion = arrays["wrench_distortion_signed"]
    summary = {
        "torque_axis_clipping": _mask_summary(torque_mask, "axis_index"),
        "motor_clipping": {
            "lower": _mask_summary(arrays["motor_pwm_lower_clip_mask"], "motor_index"),
            "upper": _mask_summary(arrays["motor_pwm_upper_clip_mask"], "motor_index"),
            "any": _mask_summary(motor_mask, "motor_index"),
        },
        "minmax": {
            name: _minmax_by_channel(arrays[name], "motor_index")
            for name in (
                "motor_pwm_preclip",
                "motor_pwm_postclip",
                "motor_force_preclip_N",
                "motor_force_postclip_N",
                "motor_speed_preclip",
                "motor_speed_postclip",
            )
        },
        "maximum_exceedance": {
            name: _maximum_record(arrays[name], "motor_index")
            for name in (
                "pwm_lower_exceedance",
                "pwm_upper_exceedance",
                "force_lower_exceedance_N",
                "force_upper_exceedance_N",
                "speed_lower_exceedance",
                "speed_upper_exceedance",
            )
        },
        "maximum_wrench_distortion_by_axis": _wrench_summary(
            pre_wrench, post_wrench, distortion, motor_mask, torque_mask
        ),
    }
    return {
        "limits": params,
        "arrays": _array_section(arrays),
        "relative_wrench_distortion": _relative_wrench_payload(pre_wrench, distortion),
        "summary": summary,
    }


def _stage_b_section(arrays: dict[str, np.ndarray], params: dict[str, Any]) -> dict[str, Any]:
    motor_mask = arrays["motor_force_any_clip_mask"]
    pre_wrench = arrays["wrench_preclip"]
    post_wrench = arrays["wrench_postclip"]
    distortion = arrays["wrench_distortion_signed"]
    summary = {
        "additional_clip_created": bool(motor_mask.any()),
        "motor_clipping": {
            "lower": _mask_summary(arrays["motor_force_lower_clip_mask"], "motor_index"),
            "upper": _mask_summary(arrays["motor_force_upper_clip_mask"], "motor_index"),
            "any": _mask_summary(motor_mask, "motor_index"),
        },
        "minmax": {
            name: _minmax_by_channel(arrays[name], "motor_index")
            for name in ("motor_force_preclip_N", "motor_force_postclip_N", "commanded_motor_speed")
        },
        "maximum_exceedance": {
            name: _maximum_record(arrays[name], "motor_index")
            for name in ("force_lower_exceedance_N", "force_upper_exceedance_N")
        },
        "maximum_wrench_distortion_by_axis": _wrench_summary(
            pre_wrench, post_wrench, distortion, motor_mask
        ),
    }
    return {
        "limits": params,
        "arrays": _array_section(arrays),
        "relative_wrench_distortion": _relative_wrench_payload(pre_wrench, distortion),
        "summary": summary,
    }


def _trace_exact_checks(custom: RolloutTrace, production: RolloutTrace, label: str) -> None:
    for field in TRACE_FIELDS:
        _require_array_equal(
            f"{label} production trace {field}", getattr(custom, field), getattr(production, field)
        )


def _accepted_day22_checks(
    label: str,
    production_trace: RolloutTrace,
    production_rollout: dict[str, Any],
    day22: dict[str, Any],
) -> dict[str, Any]:
    accepted = day22["variants"][label]
    current = production_rollout["variants"][label]
    commands = np.asarray(production_trace.commanded_rotor_vel)
    accepted_commands = np.asarray(
        accepted["saturation"]["commanded_rotor_velocity_rpm"], dtype=commands.dtype
    )[:, None, None, :]
    _require_array_equal(f"{label} accepted Day-22 motor commands", commands, accepted_commands)
    for key in ("actual_position_m", "position_error_m", "position_error_norm_m"):
        current_array = np.asarray(current[key])
        accepted_array = np.asarray(accepted[key], dtype=current_array.dtype)
        _require_array_equal(f"{label} accepted Day-22 {key}", current_array, accepted_array)
    current_metrics = current["metrics"]
    accepted_metrics = accepted["tracking_metrics"]["unchanged_tracking_loss_metrics"]
    if current_metrics != accepted_metrics:
        raise DiagnosticError(f"{label} accepted Day-22 tracking metrics changed")
    lower, upper = saturation.saturation_decisions(
        commands,
        np.asarray(accepted["saturation"]["limits_rpm"]["minimum"], dtype=commands.dtype),
        np.asarray(accepted["saturation"]["limits_rpm"]["maximum"], dtype=commands.dtype),
    )
    accepted_lower = np.asarray(accepted["saturation"]["decision_arrays"]["lower"], dtype=bool)[
        :, None, None, :
    ]
    accepted_upper = np.asarray(accepted["saturation"]["decision_arrays"]["upper"], dtype=bool)[
        :, None, None, :
    ]
    _require_array_equal(f"{label} accepted lower decisions", lower, accepted_lower)
    _require_array_equal(f"{label} accepted upper decisions", upper, accepted_upper)
    total = accepted["saturation"]["total"]
    return {
        "tracking_metrics": current_metrics,
        "saturated_count": total["saturated_count"],
        "sample_count_all_motors": total["sample_count_all_motors"],
        "lower_saturated_count": total["lower_saturated_count"],
        "upper_saturated_count": total["upper_saturated_count"],
        "motor_saturation_fraction": total["tracking_loss_metric_fraction"],
        "commands_array_equal": True,
        "tracking_arrays_and_metrics_equal": True,
        "detection_decisions_array_equal": True,
    }


def _limits(initial_data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    attitude = initial_data.controls.attitude.params
    force_torque = initial_data.controls.force_torque.params
    pwm_min = float(attitude["pwm_min"])
    pwm_max = float(attitude["pwm_max"])
    thrust_max = float(attitude["thrust_max"])
    force_min = float(pwm2force(pwm_min, thrust_max, pwm_max))
    rpm2thrust = force_torque["rpm2thrust"]
    _require_array_equal(
        "configured/runtime Float32 rpm2thrust",
        np.asarray(rpm2thrust),
        np.asarray(CONFIGURED_RPM2THRUST, dtype=np.float32),
    )
    if (pwm_min, pwm_max) != CONFIGURED_PWM_LIMITS:
        raise DiagnosticError("configured PWM limits changed")
    if not np.array_equal(
        np.asarray([force_torque["thrust_min"], force_torque["thrust_max"]]),
        np.asarray(CONFIGURED_FORCE_LIMITS_N, dtype=np.float32),
    ):
        raise DiagnosticError("configured/runtime Float32 force limits changed")
    speed_min = float(motor_force2rotor_vel(jnp.asarray(force_min), rpm2thrust))
    speed_max = float(motor_force2rotor_vel(jnp.asarray(thrust_max), rpm2thrust))
    return (
        {
            "torque_pwm_max": np.asarray(attitude["torque_pwm_max"]).tolist(),
            "configured_pwm_limits": list(CONFIGURED_PWM_LIMITS),
            "configured_motor_force_limits_N": list(CONFIGURED_FORCE_LIMITS_N),
            "pwm_min": pwm_min,
            "pwm_max": pwm_max,
            "motor_force_min_N": force_min,
            "motor_force_max_N": thrust_max,
            "motor_speed_min": speed_min,
            "motor_speed_max": speed_max,
            "pwm_span": pwm_max - pwm_min,
            "motor_force_span_N": thrust_max - force_min,
            "motor_speed_span": speed_max - speed_min,
        },
        {
            "configured_motor_force_limits_N": list(CONFIGURED_FORCE_LIMITS_N),
            "motor_force_min_N": float(force_torque["thrust_min"]),
            "motor_force_max_N": float(force_torque["thrust_max"]),
            "configured_rpm2thrust": list(CONFIGURED_RPM2THRUST),
            "runtime_float32_rpm2thrust": np.asarray(force_torque["rpm2thrust"]).tolist(),
            "existing_motor_speed_convention": "Crazyflow controller rotor-velocity convention",
        },
    )


def _hover_summary(initial_data: Any, stage_b_limits: dict[str, Any]) -> dict[str, Any]:
    speed = float(np.asarray(hover_rotor_velocity(initial_data))[0, 0, 0])
    coefficients = np.asarray(initial_data.controls.force_torque.params["rpm2thrust"])
    calibrated_force = float(coefficients[0] + coefficients[1] * speed + coefficients[2] * speed**2)
    maximum_speed = float(
        motor_force2rotor_vel(
            initial_data.controls.force_torque.params["thrust_max"],
            initial_data.controls.force_torque.params["rpm2thrust"],
        )
    )
    return {
        "hover_motor_speed": speed,
        "configured_maximum_motor_speed": maximum_speed,
        "speed_fraction_of_configured_maximum": speed / maximum_speed,
        "calibrated_hover_thrust_per_motor_N": calibrated_force,
        "thrust_fraction_of_configured_maximum": calibrated_force
        / stage_b_limits["motor_force_max_N"],
        "quadratic_calibration_configured": list(CONFIGURED_RPM2THRUST),
        "quadratic_calibration_runtime_float32": coefficients.tolist(),
        "boundary": (
            "Pure Crazyflow simulation convention only; not a hardware reserve, safety, flight, "
            "firmware, or motor-characterization claim."
        ),
    }


def _seed05_motor0_fact(arrays: dict[str, np.ndarray], limits: dict[str, Any]) -> dict[str, Any]:
    pwm = arrays["motor_pwm_preclip"][:, 0]
    index = int(np.argmax(pwm))
    return {
        "motor_index": 0,
        "index_zero_based": index,
        "state_time_s": (index + 1) / friday.CONTROL_FREQUENCY_HZ,
        "interval_start_s": index / friday.CONTROL_FREQUENCY_HZ,
        "interval_end_s": (index + 1) / friday.CONTROL_FREQUENCY_HZ,
        "requested_pwm": float(arrays["motor_pwm_preclip"][index, 0]),
        "upper_limit_pwm": limits["pwm_max"],
        "upper_exceedance_pwm": float(arrays["pwm_upper_exceedance"][index, 0]),
        "requested_motor_force_N": float(arrays["motor_force_preclip_N"][index, 0]),
        "upper_limit_motor_force_N": limits["motor_force_max_N"],
        "upper_exceedance_motor_force_N": float(arrays["force_upper_exceedance_N"][index, 0]),
        "requested_motor_speed": float(arrays["motor_speed_preclip"][index, 0]),
        "upper_limit_motor_speed": limits["motor_speed_max"],
        "upper_exceedance_motor_speed": float(arrays["speed_upper_exceedance"][index, 0]),
        "speed_convention": "existing Crazyflow controller rotor-velocity convention",
    }


def build_diagnostic() -> dict[str, Any]:
    """Build and validate the full report in memory before any output is written."""
    day22, day22_identities = load_day22()
    data_audit = load_data_audit()
    candidate = day22["source_identity"]["candidate"]
    production_rollout, production_traces, production_initial = friday.build_rollout_trace_bundle(
        candidate
    )
    if production_rollout["contract"] != day22["frozen_rollout_contract"]:
        raise DiagnosticError("fresh production rollout contract differs from accepted Day 22")
    if production_rollout["reference_time_s"] != day22["reference_time_s"]:
        raise DiagnosticError("fresh reference time differs from accepted Day 22")
    if production_rollout["state_time_s"] != day22["state_time_s"]:
        raise DiagnosticError("fresh state time differs from accepted Day 22")
    if production_rollout["reference_position_m"] != day22["reference_position_m"]:
        raise DiagnosticError("fresh reference positions differ from accepted Day 22")
    if production_rollout["target_position_m"] != day22["target_position_m"]:
        raise DiagnosticError("fresh target positions differ from accepted Day 22")

    full_reference, reference = make_trajectory(
        "figure8", friday.HORIZON, friday.CONTROL_FREQUENCY_HZ
    )
    commands = state_commands(reference)[:, None, None, :]
    sim = _new_frozen_sim()
    diagnostic_initial = initialize_tracking_state(
        sim.data, full_reference.pos[0], full_reference.vel[0]
    )
    for field in ("pos", "vel", "quat", "rotor_vel"):
        _require_array_equal(
            f"initial state {field}",
            getattr(diagnostic_initial.states, field),
            getattr(production_initial.states, field),
        )
    rollout = _make_instrumented_rollout(sim)
    stage_a_limits, stage_b_limits = _limits(diagnostic_initial)
    variants: dict[str, Any] = {}
    validation_records: dict[str, Any] = {}
    raw_by_label = {
        "default": production_rollout["variants"]["default"]["raw_gains"],
        friday.CANDIDATE_LABEL: candidate["selected_raw_gains"],
    }
    for label, raw_gains in raw_by_label.items():
        initial = apply_raw_gains(
            diagnostic_initial, jnp.asarray(raw_gains, dtype=jnp.float32), stage=1
        )
        _, output = rollout(initial, commands)
        jax.block_until_ready(output)
        custom_trace = output["trace"]
        diagnostics = output["diagnostic"]
        _all_finite(f"{label} custom trace", custom_trace)
        _all_finite(f"{label} Stage A", diagnostics["stage_a"])
        _all_finite(f"{label} Stage B", diagnostics["stage_b"])
        _trace_exact_checks(custom_trace, production_traces[label], label)
        day22_check = _accepted_day22_checks(
            label, production_traces[label], production_rollout, day22
        )
        stage_a_arrays = {
            name: _squeeze_case(value) for name, value in diagnostics["stage_a"].items()
        }
        stage_b_arrays = {
            name: _squeeze_case(value) for name, value in diagnostics["stage_b"].items()
        }
        if np.any(stage_a_arrays["speed_inverse_discriminant"] < 0):
            raise DiagnosticError(f"{label} Stage-A preclip speed inverse is non-real")
        stage_a_identity = _reconstruction_check(
            f"{label} Stage-A postclip wrench",
            stage_a_arrays["wrench_postclip"],
            stage_a_arrays["production_postclip_wrench"],
            WRENCH_PLATFORM_SCALE,
        )
        stage_b_input_identity = _reconstruction_check(
            f"{label} Stage-B input from Stage A",
            stage_b_arrays["input_stage_a_production_wrench"],
            stage_a_arrays["production_postclip_wrench"],
            WRENCH_PLATFORM_SCALE,
        )
        stage_b_command_identity = _reconstruction_check(
            f"{label} Stage-B production command",
            stage_b_arrays["commanded_motor_speed"],
            stage_b_arrays["production_commanded_motor_speed"],
            stage_a_limits["motor_speed_max"],
        )
        _require_array_equal(
            f"{label} Stage-B custom/trace commands",
            stage_b_arrays["production_commanded_motor_speed"],
            _squeeze_case(custom_trace.commanded_rotor_vel),
        )
        variants[label] = {
            "status": "NO_SATURATION_PRESENT" if label == "default" else STATUS,
            "raw_gains": raw_gains,
            "physical_gains": production_rollout["variants"][label]["physical_gains"],
            "day22_replay": day22_check,
            "stage_a_legacy_pwm": _stage_a_section(stage_a_arrays, stage_a_limits),
            "stage_b_si_force": _stage_b_section(stage_b_arrays, stage_b_limits),
        }
        validation_records[label] = {
            "all_diagnostic_arrays_finite": True,
            "stage_a_preclip_speed_inverse_real_and_finite": True,
            "custom_trace_exactly_equals_production_trace": True,
            "stage_a_local_postclip_wrench_matches_production": stage_a_identity,
            "stage_b_input_exactly_stage_a_production_wrench": stage_b_input_identity,
            "stage_b_local_command_matches_production": stage_b_command_identity,
            "production_trace_and_tracking_equal_accepted_day22": True,
        }

    default_count = variants["default"]["day22_replay"]["saturated_count"]
    candidate_count = variants[friday.CANDIDATE_LABEL]["day22_replay"]["saturated_count"]
    if default_count != 0 or candidate_count != 11:
        raise DiagnosticError("accepted Day-22 0/400 and 11/400 counts changed")
    candidate_stage_a = variants[friday.CANDIDATE_LABEL]["stage_a_legacy_pwm"]
    candidate_stage_a_arrays = {
        name: np.asarray(value) for name, value in candidate_stage_a["arrays"]["values"].items()
    }
    motor_clip = candidate_stage_a_arrays["motor_pwm_any_clip_mask"].astype(bool)
    expected_motor_clip = np.zeros((friday.HORIZON, 4), dtype=bool)
    expected_motor_clip[48:59, 0] = True
    if not np.array_equal(motor_clip, expected_motor_clip):
        raise DiagnosticError(
            "Stage-A physical clipping does not match Day-22 motor/time attribution"
        )
    maximum_seed05 = _seed05_motor0_fact(candidate_stage_a_arrays, stage_a_limits)
    hover = _hover_summary(diagnostic_initial, stage_b_limits)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "work_order": "WO-GR-F0-003",
        "purpose": (
            "diagnostic-local exposure of pre/post values at both unchanged allocation stages for "
            "the accepted Day-22 Default/Seed-05 Figure-8 simulation"
        ),
        "source_identity": {
            "accepted_day22_payloads_verified": day22_identities,
            "accepted_day22_checksum_index": {
                "path": "artifacts/day22-saturation-diagnostic/SHA256SUMS",
                "sha256": _sha256_path(DAY22_DIR / "SHA256SUMS"),
            },
            "candidate": candidate,
        },
        "frozen_rollout_contract": production_rollout["contract"],
        "sampling_contract": {
            "control_intervals": 100,
            "simulation_substeps_per_interval": 5,
            "stored_substep_one_based": 5,
            "state_time_definition_s": "(index + 1) / 100",
            "interval_definition_s": "[index / 100, (end_index + 1) / 100)",
        },
        "numerical_contract": {
            "float32_eps": FLOAT32_EPS,
            "reconstruction_factor": RECONSTRUCTION_FACTOR,
            "rtol": 0.0,
            "wrench_platform_scales": WRENCH_PLATFORM_SCALE.tolist(),
            "wrench_relative_floors": WRENCH_RELATIVE_FLOOR.tolist(),
            "saturation_detection_threshold": SATURATION_DETECTION_THRESHOLD,
            "threshold_role": (
                "Day-22 classification detection threshold only; never subtracted from or used "
                "as a physical exceedance amount"
            ),
            "exceedance_definition": "max(raw - upper, 0) or max(lower - raw, 0)",
        },
        "variants": variants,
        "maximum_seed05_motor0_demand": maximum_seed05,
        "hover_operating_point": hover,
        "data_audit_01": data_audit,
        "normal_acceptance_gate": {
            "status": GATE_FAILURE_STATUS,
            "criterion": "motor_saturation_fraction == 0.0 for both variants",
            "criterion_unchanged": True,
            "default_saturated_count": default_count,
            "default_sample_count": 400,
            "seed05_saturated_count": candidate_count,
            "seed05_sample_count": 400,
            "accepted_candidate": False,
            "flight_ready": False,
        },
        "validations": {
            "pipeline_exactly_unchanged_six_stage_order": True,
            "fifth_substep_only_stored": True,
            "variants": validation_records,
            "default_exactly_0_of_400": True,
            "seed05_exactly_11_of_400_motor0_upper_indices_48_through_58": True,
            "data_audit_derived_only": True,
            "protected_day10_day13_read": False,
            "test_manifest_or_split_opened": False,
            "training_or_optimization_runs_started": 0,
        },
        "claim_boundary": (
            "Pure simulation diagnostic only. The unchanged zero-saturation gate remains FAIL; "
            "no controller, allocator, threshold, gain, default, candidate, superiority, safety, "
            "firmware, hardware, flight, or Sim2Real claim is made."
        ),
    }
    _all_finite("complete diagnostic report", report)
    validate_report_summaries(report)
    return report


def _summary_arrays(section: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = {}
    for name, values in section["arrays"]["values"].items():
        dtype = np.dtype(section["arrays"]["contracts"][name]["dtype"])
        arrays[name] = np.asarray(values, dtype=dtype)
        if list(arrays[name].shape) != section["arrays"]["contracts"][name]["shape"]:
            raise DiagnosticError(f"JSON array contract mismatch: {name}")
    return arrays


def validate_report_summaries(report: dict[str, Any]) -> None:
    """Recompute every stored array-derived summary from JSON-compatible values."""
    for label, variant in report["variants"].items():
        stage_a = variant["stage_a_legacy_pwm"]
        arrays_a = _summary_arrays(stage_a)
        recomputed_a = _stage_a_section(arrays_a, stage_a["limits"])["summary"]
        if recomputed_a != stage_a["summary"]:
            raise DiagnosticError(f"{label} Stage-A summary is not array-derived")
        stage_b = variant["stage_b_si_force"]
        arrays_b = _summary_arrays(stage_b)
        recomputed_b = _stage_b_section(arrays_b, stage_b["limits"])["summary"]
        if recomputed_b != stage_b["summary"]:
            raise DiagnosticError(f"{label} Stage-B summary is not array-derived")
    candidate = _summary_arrays(report["variants"][friday.CANDIDATE_LABEL]["stage_a_legacy_pwm"])
    expected_fact = _seed05_motor0_fact(
        candidate, report["variants"][friday.CANDIDATE_LABEL]["stage_a_legacy_pwm"]["limits"]
    )
    if expected_fact != report["maximum_seed05_motor0_demand"]:
        raise DiagnosticError("Seed-05 motor-0 maximum fact is not array-derived")


def _figure_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Derive the fixed four-panel presentation contract only from report values."""
    default_a = _summary_arrays(report["variants"]["default"]["stage_a_legacy_pwm"])
    candidate = report["variants"][friday.CANDIDATE_LABEL]
    seed_a = _summary_arrays(candidate["stage_a_legacy_pwm"])
    seed_b = _summary_arrays(candidate["stage_b_si_force"])
    time = (np.arange(friday.HORIZON) + 1) / friday.CONTROL_FREQUENCY_HZ
    window = slice(FIGURE_WINDOW_START_INDEX, FIGURE_WINDOW_END_INDEX + 1)
    platform_scales = np.asarray(report["numerical_contract"]["wrench_platform_scales"])
    signed_wrench_normalized = seed_a["wrench_distortion_signed"] / platform_scales
    fact = report["maximum_seed05_motor0_demand"]
    default_torque_count = int(default_a["torque_any_clip_mask"].sum())
    default_motor_count = int(default_a["motor_pwm_any_clip_mask"].sum())
    seed_torque_count = int(seed_a["torque_any_clip_mask"].sum())
    seed_motor_count = int(seed_a["motor_pwm_any_clip_mask"].sum())
    seed_stage_b_count = int(seed_b["motor_force_any_clip_mask"].sum())
    wrench_maxima = candidate["stage_a_legacy_pwm"]["summary"]["maximum_wrench_distortion_by_axis"]
    exact_text = "\n".join(
        (
            "EXACT MAXIMUM — SEED 05 MOTOR 0",
            f"Index {fact['index_zero_based']} | state time {fact['state_time_s']} s",
            f"Request: {fact['requested_pwm']} PWM | {fact['requested_motor_force_N']} N",
            f"         {fact['requested_motor_speed']} (Crazyflow motor-speed convention)",
            f"Exceed:  {fact['upper_exceedance_pwm']} PWM | "
            f"{fact['upper_exceedance_motor_force_N']} N",
            f"         {fact['upper_exceedance_motor_speed']} (same convention)",
            "",
            "CLIP COUNTS",
            f"Default Stage A: torque {default_torque_count}/300 | motor {default_motor_count}/400",
            f"Seed 05 Stage A: torque {seed_torque_count}/300 | motor {seed_motor_count}/400",
            f"Seed 05 Stage B added: motor {seed_stage_b_count}/400",
            "",
            "MAX |STAGE-A WRENCH DISTORTION|",
            f"Collective: {wrench_maxima[0]['maximum_absolute_distortion']} N",
            f"Roll: {wrench_maxima[1]['maximum_absolute_distortion']} N m",
            f"Pitch: {wrench_maxima[2]['maximum_absolute_distortion']} N m",
            f"Yaw: {wrench_maxima[3]['maximum_absolute_distortion']} N m",
            "",
            f"STATUS: {STATUS}",
            f"GATE: {GATE_FAILURE_STATUS}",
            "PURE SIMULATION — NO HARDWARE OR FLIGHT CLAIM",
        )
    )
    return {
        "time_s": time,
        "window_start_index": FIGURE_WINDOW_START_INDEX,
        "window_end_index": FIGURE_WINDOW_END_INDEX,
        "window_time_s": time[window],
        "seed05_motor_pwm_preclip_window": seed_a["motor_pwm_preclip"][window],
        "seed05_motor_pwm_postclip_window": seed_a["motor_pwm_postclip"][window],
        "seed05_motor0_pwm_preclip": seed_a["motor_pwm_preclip"][:, 0],
        "seed05_motor0_pwm_postclip": seed_a["motor_pwm_postclip"][:, 0],
        "pwm_upper_limit": candidate["stage_a_legacy_pwm"]["limits"]["pwm_max"],
        "signed_wrench_platform_normalized": signed_wrench_normalized,
        "exact_text_panel": exact_text,
    }


def _figure_bytes(report: dict[str, Any]) -> bytes:
    """Render the only figure exclusively from the validated JSON report."""
    contract = _figure_contract(report)
    time = contract["time_s"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(time, contract["seed05_motor0_pwm_preclip"], label="M0 unclipped")
    axes[0, 0].plot(time, contract["seed05_motor0_pwm_postclip"], label="M0 clipped")
    axes[0, 0].axhline(
        contract["pwm_upper_limit"], color="black", linestyle="--", label="true PWM upper limit"
    )
    axes[0, 0].axvspan(0.48, 0.59, color="#E45756", alpha=0.10, label="clip interval")
    axes[0, 0].set(
        title="Stage A: unclipped vs clipped Seed-05 motor-0 demand",
        xlabel="state time [s]",
        ylabel="PWM",
    )
    axes[0, 0].legend(fontsize=8)

    motor_colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    for motor, color in enumerate(motor_colors):
        axes[0, 1].plot(
            contract["window_time_s"],
            contract["seed05_motor_pwm_preclip_window"][:, motor],
            color=color,
            linestyle="-",
            linewidth=1.8,
            label=f"M{motor} preclip",
        )
        axes[0, 1].plot(
            contract["window_time_s"],
            contract["seed05_motor_pwm_postclip_window"][:, motor],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"M{motor} postclip",
        )
    axes[0, 1].axhline(
        contract["pwm_upper_limit"], color="black", linestyle=":", label="upper limit"
    )
    axes[0, 1].set(
        title="Stage A: all motors around indices 48–58 (color=motor, line=pre/post)",
        xlabel="state time [s]",
        ylabel="PWM",
    )
    axes[0, 1].legend(ncol=3, fontsize=7)

    wrench_names = ("collective", "roll", "pitch", "yaw")
    for axis, (name, color) in enumerate(zip(wrench_names, motor_colors, strict=True)):
        axes[1, 0].plot(
            time, contract["signed_wrench_platform_normalized"][:, axis], label=name, color=color
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(
        title="Stage-A signed wrench distortion (postclip − preclip)",
        xlabel="state time [s]",
        ylabel="signed distortion / platform scale [1]",
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        1.0,
        contract["exact_text_panel"],
        ha="left",
        va="top",
        fontsize=8.1,
        family="monospace",
        linespacing=1.12,
        transform=axes[1, 1].transAxes,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F7F7F7", "edgecolor": "#777777"},
    )

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Day 24 diagnostic-local two-stage allocation replay — zero-saturation gate remains FAIL"
    )
    figure.text(
        0.5,
        0.015,
        "Pure cf2x_L250 simulation diagnostic; no controller change, candidate acceptance, safety, hardware, or flight claim.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    payload = io.BytesIO()
    figure.savefig(payload, format="png", dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)
    return payload.getvalue()


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _generator_commit() -> str:
    subject = "research: add unclipped allocation diagnostic"
    records = _git_output("log", "--format=%H%x00%s", f"{BASE_COMMIT}..HEAD").splitlines()
    matches = [
        record.split("\0", maxsplit=1)[0]
        for record in records
        if record.split("\0", maxsplit=1)[1] == subject
    ]
    if len(matches) != 1 or not re.fullmatch(r"[0-9a-f]{40}", matches[0]):
        raise DiagnosticError("could not identify the unique committed diagnostic generator")
    return matches[0]


def _provenance(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "crazyflow.unclipped_allocation_provenance.v1",
        "work_order": "WO-GR-F0-003",
        "base_commit": BASE_COMMIT,
        "result_commit_generator_and_tests": _generator_commit(),
        "package_commit": "recorded externally in worker handoff and independent review",
        "branch": BRANCH,
        "worker": WORKER,
        "worktree": WORKTREE,
        "inputs": {
            "accepted_day22_checksum_index": report["source_identity"][
                "accepted_day22_checksum_index"
            ],
            "accepted_day22_payloads": report["source_identity"][
                "accepted_day22_payloads_verified"
            ],
            "data_audit_01": report["data_audit_01"],
            "protected_day10_day13_read": False,
            "test_manifest_or_split_opened": False,
        },
        "configuration": {
            **report["frozen_rollout_contract"],
            **report["sampling_contract"],
            "diagnostic_status": STATUS,
            "normal_zero_saturation_gate_unchanged": True,
        },
        "commands": {
            "generate": "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-003-matplotlib /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python examples/jax/mellinger_unclipped_allocation_diagnostic.py --diagnostic-only-unclipped-allocation --output-dir artifacts/day24-unclipped-allocation",
            "focused_pytest": "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-003-matplotlib /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_mellinger_unclipped_allocation_diagnostic.py tests/unit/test_mellinger_saturation_diagnostic.py tests/unit/test_mellinger_friday_evidence.py tests/unit/test_mellinger_tracking.py tests/unit/control/test_mellinger.py tests/unit/control/test_transform.py",
            "ruff_check": "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 PYTHONDONTWRITEBYTECODE=1 RUFF_CACHE_DIR=/tmp/crazyflow-gradient-f0-003-ruff /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m ruff check examples/jax/mellinger_unclipped_allocation_diagnostic.py tests/unit/test_mellinger_unclipped_allocation_diagnostic.py",
            "ruff_format_check": "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-003 PYTHONDONTWRITEBYTECODE=1 /home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m ruff format --check examples/jax/mellinger_unclipped_allocation_diagnostic.py tests/unit/test_mellinger_unclipped_allocation_diagnostic.py",
            "accepted_checksums": "sha256sum --quiet -c for Day 19, Day 21, Day 22, and Day 23 accepted packages; analyzer not executed",
            "deterministic_reproduction": "two fresh /tmp packages from the generator/test commit; 4/4 checksums and byte identity to each other and final package",
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "output_inventory": list(OUTPUT_NAMES) + ["SHA256SUMS"],
        "output_checksum_rule": "SHA256SUMS covers the four other files, not itself",
    }


def _presentation_handoff(report: dict[str, Any]) -> str:
    fact = report["maximum_seed05_motor0_demand"]
    default = report["variants"]["default"]
    candidate = report["variants"][friday.CANDIDATE_LABEL]
    stage_a = candidate["stage_a_legacy_pwm"]["summary"]
    stage_b = candidate["stage_b_si_force"]["summary"]
    wrench = stage_a["maximum_wrench_distortion_by_axis"]
    wrench_text = ", ".join(
        f"{item['axis']}={item['maximum_absolute_distortion']:.12g}" for item in wrench
    )
    return f"""# Unclipped two-stage allocation presentation handoff

## Fixed status and gate

This is a review-candidate worker result with status `{STATUS}`. The unchanged gate remains
`{GATE_FAILURE_STATUS}`: Default is {default["day22_replay"]["saturated_count"]}/400 and Seed 05 is
{candidate["day22_replay"]["saturated_count"]}/400, only motor 0 upper at indices 48–58 and
interval [0.48, 0.59) s. Nothing here accepts a candidate or changes the controller.

## Exact two-stage result

- Stage A motor clipping: {stage_a["motor_clipping"]["any"]["count_all_channels"]} Seed-05 samples.
- Stage B created additional clipping: {stage_b["additional_clip_created"]} with
  {stage_b["motor_clipping"]["any"]["count_all_channels"]} samples.
- Maximum absolute Stage-A wrench distortion by axis: {wrench_text}.
- The maximum Seed-05 motor-0 request is {fact["requested_pwm"]:.12g} PWM,
  {fact["requested_motor_force_N"]:.12g} N and {fact["requested_motor_speed"]:.12g} in the existing
  Crazyflow motor-speed convention. It exceeds the corresponding upper limits by
  {fact["upper_exceedance_pwm"]:.12g} PWM, {fact["upper_exceedance_motor_force_N"]:.12g} N and
  {fact["upper_exceedance_motor_speed"]:.12g}, at index {fact["index_zero_based"]}.

## Figure use

`unclipped_allocation_diagnostic.png` is generated only from the validated JSON. It shows the raw
and clipped Seed-05 motor-0 request against the true limit, all four motors' Stage-A pre/postclip
commands around indices 48–58, signed platform-normalized Stage-A wrench distortion, and a compact
panel containing exact maximum request/exceedance, clip counts, failed status and claim boundary.
Use it only with the failed-gate status visible.

## DATA-AUDIT-01 boundary

The checksum-verified Day-19 derivative has ten seeds with 5000 Train, 5000 Validation and zero
Test rows each (100000 Train/Validation total), and per-seed/panel registered saturation maxima of
0.0. The analyzer source rejects Test rows, requires complete alternating order and aggregates
per-seed then panel maxima. The analyzer was not executed and protected Day-13/Test data were not
opened. This does not provide motor traces, reserve or allocation distortion for those runs.

## Exact verbal caveat

> Der unveränderte Day-22-Simulationsfall wurde nur diagnostisch bis vor beide Clippingstufen zurückverfolgt. Seed 05 verlangt zeitweise mehr als die konfigurierte Motor-0-Grenze; das Null-Sättigungs-Gate bleibt deshalb FAIL. Die Zahlen sind weder Controllerfreigabe noch Sicherheits-, Hardware- oder Flugnachweis.

## Claim boundary

Pure `cf2x_L250` simulation diagnostic only. No Test, tuning, optimization, candidate acceptance,
superiority, safety, firmware, hardware, real-flight or Sim2Real claim is made.
"""


def build_package_payloads(report: dict[str, Any]) -> dict[str, bytes]:
    """Build all four checksummed payloads deterministically in memory."""
    validate_report_summaries(report)
    payloads = {
        "unclipped_allocation_diagnostic.json": _json_bytes(report),
        "unclipped_allocation_diagnostic.png": _figure_bytes(report),
        "provenance.json": _json_bytes(_provenance(report)),
        "PRESENTATION_HANDOFF.md": _presentation_handoff(report).encode(),
    }
    if tuple(sorted(payloads)) != tuple(sorted(OUTPUT_NAMES)):
        raise DiagnosticError("diagnostic payload inventory mismatch")
    return payloads


def write_package(output_dir: Path, payloads: dict[str, bytes]) -> None:
    """Write a complete package once and refuse every overwrite."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in OUTPUT_NAMES:
        (output_dir / name).write_bytes(payloads[name])
    checksums = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n" for name in sorted(OUTPUT_NAMES)
    )
    (output_dir / "SHA256SUMS").write_text(checksums)


def main() -> None:
    """Generate the failed-gate allocation diagnostic after all validations pass."""
    arguments = parse_args()
    if arguments.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {arguments.output_dir}")
    report = build_diagnostic()
    payloads = build_package_payloads(report)
    write_package(arguments.output_dir, payloads)
    fact = report["maximum_seed05_motor0_demand"]
    candidate = report["variants"][friday.CANDIDATE_LABEL]
    print(f"diagnostic_status={STATUS}")
    print(f"normal_acceptance_gate={GATE_FAILURE_STATUS}")
    print(f"seed05_saturated_count={candidate['day22_replay']['saturated_count']}")
    print(f"seed05_motor0_maximum_requested_pwm={fact['requested_pwm']}")
    print(f"seed05_motor0_maximum_upper_exceedance_pwm={fact['upper_exceedance_pwm']}")
    print("test_opened=False")
    print("training_runs_started=0")


if __name__ == "__main__":
    main()
