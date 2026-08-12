"""Frozen cf21B_500 robust-episode construction and evaluation helpers.

This module is deliberately experiment-local.  It calls the unchanged Crazyflow
Mellinger controller and first-principles dynamics, retains the complete ``SimData``
carry for a six-second rollout, and applies the two-second score mask only after the
rollout.  The diagnostic allocation arithmetic repeats transparent Float32 operations
immediately before the production stages; it does not replace either stage.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Iterable

import array_api_extra as xpx
import jax
import jax.numpy as jnp
import numpy as np
from array_api_compat import array_namespace
from scipy.spatial.transform import Rotation as R

import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.control.core import controllable
from crazyflow.control.mellinger.control import force_torque_pwms2pwms
from crazyflow.control.mellinger.research.gains import (
    apply_raw_gains,
    raw_from_data,
    registry_fingerprint,
)
from crazyflow.control.mellinger.tracking import (
    RolloutTrace,
    TrackingLossConfig,
    hover_rotor_velocity,
    initialize_tracking_state,
    rollout_state_commands,
    rotor_velocity_limits,
    tracking_loss_per_case,
    tracking_loss_terms_per_case,
)
from crazyflow.control.transform import force2pwm, motor_force2rotor_vel, pwm2force
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator
from crazyflow.trajectory import Trajectory, state_commands
from crazyflow.utils import leaf_replace

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData


SCHEMA_VERSION = "crazyflow.cf21b_robust_foundation.v1"
SOURCE_BASE_COMMIT = "a4e4136f8312851790555e3de6cdf23065252edd"
PRE_CORRECTION_RESULT_COMMIT = "c4bbb57443836aecba52d72cb73a11ee4cbdb5dd"
PRE_CORRECTION_REPORT_SHA256 = "31d4b468a31b140c1dfe14bd2f1bb3d72ecdf648d4b42094d707d37eb0e74afb"
PRE_CORRECTION_CONTRACT_SHA256 = "d34f08a9053e489bbf51c91de1791057ebb468bd562a34b6ee1b6976dbb466d6"
PRE_CORRECTION_BENCHMARK_SHA256 = "9ca5d2e07da2e41b1144117c744cd36a6a908e078bd4f7e21c06d44323b2e8d0"
PRE_CORRECTION_PARENT_PIN_SHA256 = (
    "bcb40e132f4b85e99f814f0a9db4d90244523b488697868ece8597d3654de82c"
)
PRE_CORRECTION_NUMERIC_PROJECTION_SHA256 = (
    "fd1195f9d10c88a2d550090631f6b21d0d4f80fd47a2bbe09a52e7c9a0975a61"
)
PRE_CORRECTION_NUMERIC_PATH_COUNT = 5_465_162
PRE_CORRECTION_NUMERIC_PROJECTION_BYTES = 644_397_249
PLATFORM = "cf21B_500"
SIMULATION_FREQUENCY_HZ = 500
CONTROL_FREQUENCY_HZ = 100
STEPS_PER_CONTROL = 5
PARENT_DURATION_S = 8.0
PARENT_INTERVALS = 800
ROLLOUT_DURATION_S = 6.0
ROLLOUT_INTERVALS = 600
SCORE_DURATION_S = 2.0
SCORE_INTERVALS = 200
DYNAMICS_MASS_KG = 0.04338
MISMATCH_CONTROLLER_MASS_KG = 0.0393
MATCHED_CONTROLLER_MASS_KG = 0.04338
MAX_ATTEMPTS = 64
FLOAT32_EPS = 1.1920928955078125e-7
RECONSTRUCTION_FACTOR = 64.0
BENCHMARK_CONSISTENCY_FACTOR = 128.0
INTEGRAL_DETECTION_THRESHOLD = 1.0e-4
WRENCH_PLATFORM_SCALE = np.asarray(
    [0.8, 0.028284, 0.028284, 0.004751147148794944], dtype=np.float64
)
WRENCH_RELATIVE_FLOOR = 1024.0 * FLOAT32_EPS * WRENCH_PLATFORM_SCALE
WORKSPACE_MIN_M = np.asarray([-0.60, -0.60, 0.35], dtype=np.float64)
WORKSPACE_MAX_M = np.asarray([0.60, 0.60, 1.20], dtype=np.float64)
GLOBAL_LIMITS = {
    "speed_m_s": 1.50,
    "acceleration_m_s2": 5.00,
    "jerk_m_s3": 35.0,
    "yaw_rate_rad_s": 1.00,
    "tilt_rad": 0.70,
    "specific_force_m_s2": 2.00,
}
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
MASS_VARIANTS = {
    "repository_mismatch": MISMATCH_CONTROLLER_MASS_KG,
    "matched": MATCHED_CONTROLLER_MASS_KG,
}


class RobustContractError(RuntimeError):
    """Raised before evidence output when a frozen contract or technical gate fails."""


class Split(StrEnum):
    """The only two splits available to this package."""

    TRAIN = "train"
    VALIDATION = "validation"


class MotionClass(StrEnum):
    """Predeclared reference-envelope strata."""

    SOFT = "soft"
    NOMINAL = "nominal"
    DYNAMIC = "dynamic"
    NEAR_LIMIT = "near_limit"


@dataclass(frozen=True)
class ClassContract:
    lower_open: float
    upper_closed: float
    target_usage: float
    max_abs_yaw_rad: float


CLASS_CONTRACTS = {
    MotionClass.SOFT: ClassContract(0.05, 0.25, 0.15, 0.15),
    MotionClass.NOMINAL: ClassContract(0.25, 0.50, 0.38, 0.30),
    MotionClass.DYNAMIC: ClassContract(0.50, 0.75, 0.62, 0.50),
    MotionClass.NEAR_LIMIT: ClassContract(0.75, 0.95, 0.82, 0.70),
}


EPISODE_SEEDS = {
    Split.TRAIN: {
        MotionClass.SOFT: (8101001, 8101002),
        MotionClass.NOMINAL: (8101101, 8101102),
        MotionClass.DYNAMIC: (8101201, 8101202),
        MotionClass.NEAR_LIMIT: (8101301, 8101302),
    },
    Split.VALIDATION: {
        MotionClass.SOFT: (8201001, 8201002),
        MotionClass.NOMINAL: (8201101, 8201102),
        MotionClass.DYNAMIC: (8201201, 8201202),
        MotionClass.NEAR_LIMIT: (8201301, 8201302),
    },
}

SPLIT_NAMESPACE = {Split.TRAIN: 101, Split.VALIDATION: 211}
CLASS_NAMESPACE = {
    MotionClass.SOFT: 1001,
    MotionClass.NOMINAL: 1101,
    MotionClass.DYNAMIC: 1201,
    MotionClass.NEAR_LIMIT: 1301,
}
COMPONENT_NAMESPACE = {"position": 401, "yaw": 409, "acceptance": 419}


@dataclass(frozen=True)
class EpisodeSpec:
    split: Split
    motion_class: MotionClass
    ordinal: int
    seed: int
    episode_id: str
    pair_id: str
    warmup_s: float
    score_start: int
    score_stop: int


@dataclass(frozen=True)
class ReferenceCandidate:
    trajectory: Trajectory
    jerk: Array
    attempt: int
    attempts: tuple[dict[str, Any], ...]
    score_statistics: dict[str, Any]
    parent_statistics: dict[str, Any]
    array_digests: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class EvaluationInputs:
    specs: tuple[EpisodeSpec, ...]
    candidates: tuple[ReferenceCandidate, ...]
    commands: Array
    reference: Trajectory


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic, finite-only JSON bytes."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def array_digest(value: Any) -> dict[str, Any]:
    """Hash an array including its dtype and shape contract."""
    if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": digest.hexdigest()}


def vertical_excitation_machine_contract() -> dict[str, Any]:
    """Return the D-047 disclosure for the unchanged retained vertical excitation."""
    return {
        "name": "retained_fixed_positive_z_central_climb",
        "axis": {"name": "z", "index": 2, "direction": "positive"},
        "central_rate": {
            "literal_m_s": "0.06",
            "dtype": "float32",
            "float32_value_m_s": float(np.float32(0.06)),
        },
        "base_before_envelope": {
            "time_origin_s": 2.0,
            "displacement_expression": "float32(0.06) * (time_s - float32(2.0))",
            "velocity_expression": "float32(0.06)",
            "acceleration_expression": "float32(0.0)",
            "jerk_expression": "float32(0.0)",
        },
        "envelope": {
            "function": "smoothstep7",
            "continuity": "C3",
            "entry_duration_s": 2.0,
            "unit_plateau_duration_s": 4.0,
            "exit_duration_s": 2.0,
            "support_s": 8.0,
            "same_envelope_as_bounded_fourier_parent": True,
        },
        "derivative_construction": {
            "method": "exact_product_rule_from_unchanged__enveloped_implementation",
            "orders": ["displacement", "velocity", "acceleration", "jerk"],
            "highest_derivative_order": 3,
        },
        "class_independent": True,
        "composition_order": [
            "scale_bounded_three_axis_fourier_parent_to_class_target_usage",
            "add_enveloped_vertical_excitation_to_z_component",
            "add_fixed_center_[0.0,0.0,0.75]_m",
        ],
        "identical_in_both_mass_variants": True,
        "causal_boundary": (
            "No separate climb or controller-mass causal attribution without a controlled "
            "ablation; absolute Z metrics apply only to trajectories including this component."
        ),
        "future_distribution_boundary": (
            "This fixed positive climb is not a future training distribution; a future G3 must "
            "cover hold/climb/descent diversity or explicitly justify another design."
        ),
    }


def duration_scope_contract() -> dict[str, Any]:
    """Return the exact parent, rollout, scored-window, and gate measurement scopes."""
    return {
        "parent_reference_support": {
            "duration_s": 8.0,
            "control_intervals": 800,
            "reference_samples": 801,
            "simulation_rollout_claim": False,
        },
        "continuous_rollout_from_parent_start": {
            "duration_s": 6.0,
            "control_intervals": 600,
            "parent_start_time_s": 0.0,
            "reset_at_scored_window": False,
        },
        "scored_window": {
            "duration_s": 2.0,
            "control_intervals": 200,
            "warmup_options_s": [2.0, 3.0, 4.0],
            "interval_semantics": "half-open [warmup_s,warmup_s+2.0s)",
        },
        "full_six_second_rollout_gates": [
            "all_arrays_finite",
            "ground_or_floor_activation",
            "zero_thrust_activation",
        ],
        "scored_window_only_measurements": [
            "integral_boundary_contact",
            "stage_a_torque_clipping",
            "stage_a_motor_clipping",
            "stage_b_additional_clipping",
            "tracking",
            "loss_v1",
            "reserve",
            "wrench",
        ],
    }


def vertical_excitation_components(time: Array) -> tuple[Array, Array, Array, Array]:
    """Reproduce the unchanged Float32 vertical component for semantic tests and metadata."""
    envelope = _c3_envelope(time)
    climb_rate = jnp.asarray(0.06, dtype=time.dtype)
    climb_base = (
        (climb_rate * (time - 2.0))[:, None],
        jnp.broadcast_to(climb_rate, time.shape)[:, None],
        jnp.zeros_like(time)[:, None],
        jnp.zeros_like(time)[:, None],
    )
    return _enveloped(climb_base, envelope)


def parent_contract_pin_identity(contract: dict[str, Any]) -> dict[str, Any]:
    """Project the exact 16 parent identities and full-array digests used by review."""
    records = []
    for episode in contract["episodes"]:
        records.append(
            {
                "episode_id": episode["episode_id"],
                "seed": episode["seed"],
                "accepted_attempt_index": episode["accepted_attempt_index"],
                "executed_attempt_count": episode["executed_attempt_count"],
                "score_start_interval": episode["score_start_interval"],
                "score_stop_interval_exclusive": episode["score_stop_interval_exclusive"],
                "warmup_s": episode["warmup_s"],
                "split": episode["split"],
                "motion_class": episode["motion_class"],
                "array_digests": episode["arrays"]["digests"],
            }
        )
    serialized = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "record_count": len(records),
        "serialized_bytes": len(serialized),
        "sha256": sha256_bytes(serialized),
    }


def _numeric_projection_walk(
    reference: Any, candidate: Any, path: str = ""
) -> Iterable[tuple[str, int | float]]:
    if isinstance(candidate, np.generic):
        candidate = candidate.item()
    if isinstance(reference, bool):
        return
    if isinstance(reference, (int, float)):
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise RobustContractError(f"numeric projection type changed at {path}")
        if type(candidate) is not type(reference) or candidate != reference:
            raise RobustContractError(f"numeric projection value changed at {path}")
        yield path, candidate
        return
    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            raise RobustContractError(f"numeric projection container changed at {path}")
        for key in sorted(reference):
            if key not in candidate:
                raise RobustContractError(f"numeric projection path missing: {path}/{key}")
            yield from _numeric_projection_walk(reference[key], candidate[key], f"{path}/{key}")
        return
    if isinstance(reference, list):
        if not isinstance(candidate, list) or len(candidate) != len(reference):
            raise RobustContractError(f"numeric projection list changed at {path}")
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            yield from _numeric_projection_walk(left, right, f"{path}/{index}")


def numeric_projection_identity(
    reference: dict[str, Any], candidate: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Hash candidate values at every pre-existing numeric reference path without materializing."""
    if candidate is None:
        candidate = reference
    digest = hashlib.sha256()
    digest.update(b"[")
    serialized_bytes = 1
    count = 0
    for path, value in _numeric_projection_walk(reference, candidate):
        record = json.dumps((path, value), separators=(",", ":"), allow_nan=False).encode()
        if count:
            digest.update(b",")
            serialized_bytes += 1
        digest.update(record)
        serialized_bytes += len(record)
        count += 1
    digest.update(b"]")
    serialized_bytes += 1
    return {"path_count": count, "serialized_bytes": serialized_bytes, "sha256": digest.hexdigest()}


def compare_preexisting_tree(
    reference: Any, candidate: Any, *, excluded_paths: frozenset[str] = frozenset(), path: str = ""
) -> int:
    """Require every old field outside declared prose/provenance exclusions to remain exact."""
    if isinstance(candidate, np.generic):
        candidate = candidate.item()
    if path in excluded_paths:
        return 0
    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            raise RobustContractError(f"pre-correction container changed at {path}")
        total = 0
        for key in sorted(reference):
            child = f"{path}/{key}"
            if key not in candidate:
                raise RobustContractError(f"pre-correction field missing: {child}")
            total += compare_preexisting_tree(
                reference[key], candidate[key], excluded_paths=excluded_paths, path=child
            )
        return total
    if isinstance(reference, list):
        if not isinstance(candidate, list) or len(candidate) != len(reference):
            raise RobustContractError(f"pre-correction list changed at {path}")
        return sum(
            compare_preexisting_tree(
                left, right, excluded_paths=excluded_paths, path=f"{path}/{index}"
            )
            for index, (left, right) in enumerate(zip(reference, candidate, strict=True))
        )
    if type(candidate) is not type(reference) or candidate != reference:
        raise RobustContractError(f"pre-correction field changed at {path}")
    return 1


def pre_correction_invariance_evidence(
    *,
    old_report_bytes: bytes,
    old_contract_bytes: bytes,
    old_benchmark_bytes: bytes,
    new_report: dict[str, Any],
    new_contract: dict[str, Any],
    benchmark_bytes: bytes,
) -> dict[str, Any]:
    """Compare the explicit old c4 result with the corrected package before any write."""
    pinned = {
        "report": (old_report_bytes, PRE_CORRECTION_REPORT_SHA256),
        "episode_contract": (old_contract_bytes, PRE_CORRECTION_CONTRACT_SHA256),
        "benchmark": (old_benchmark_bytes, PRE_CORRECTION_BENCHMARK_SHA256),
    }
    for name, (payload, expected) in pinned.items():
        if sha256_bytes(payload) != expected:
            raise RobustContractError(f"pre-correction {name} pin changed")
    if benchmark_bytes != old_benchmark_bytes:
        raise RobustContractError("benchmark bytes changed from pre-correction result")
    old_report = json.loads(old_report_bytes)
    old_contract = json.loads(old_contract_bytes)
    old_numeric = numeric_projection_identity(old_report)
    new_at_old_numeric_paths = numeric_projection_identity(old_report, new_report)
    if old_numeric != new_at_old_numeric_paths or old_numeric != {
        "path_count": PRE_CORRECTION_NUMERIC_PATH_COUNT,
        "serialized_bytes": PRE_CORRECTION_NUMERIC_PROJECTION_BYTES,
        "sha256": PRE_CORRECTION_NUMERIC_PROJECTION_SHA256,
    }:
        raise RobustContractError("pre-correction numeric report projection changed")
    old_parent = parent_contract_pin_identity(old_contract)
    new_parent = parent_contract_pin_identity(new_contract)
    if old_parent != new_parent or old_parent["sha256"] != PRE_CORRECTION_PARENT_PIN_SHA256:
        raise RobustContractError("pre-correction parent/seed/attempt/window/array pin changed")
    report_leaf_count = compare_preexisting_tree(
        old_report, new_report, excluded_paths=frozenset({"/generator_commit"})
    )
    contract_leaf_count = compare_preexisting_tree(
        old_contract, new_contract, excluded_paths=frozenset({"/contract/position_construction"})
    )
    return {
        "status": "PASS_PRE_CORRECTION_C4_INVARIANCE",
        "old_side_result_commit": PRE_CORRECTION_RESULT_COMMIT,
        "old_payload_sha256": {name: expected for name, (_, expected) in pinned.items()},
        "numeric_report_projection": {
            "old": old_numeric,
            "new_values_at_all_old_paths": new_at_old_numeric_paths,
            "coverage": (
                "Every pre-existing numeric report path, including raw arrays, summaries, "
                "paired deltas, masses, and benchmark metadata."
            ),
        },
        "parent_identity_projection": {"old": old_parent, "new": new_parent},
        "preexisting_tree_comparison": {
            "report_scalar_leaf_count": report_leaf_count,
            "contract_scalar_leaf_count": contract_leaf_count,
            "report_excluded_existing_paths": ["/generator_commit"],
            "contract_excluded_existing_paths": ["/contract/position_construction"],
            "new_keys_allowed_only_for": [
                "D-047 trajectory metadata",
                "derived split/class aggregates",
                "duration/measurement scope",
                "invariance evidence",
            ],
        },
        "benchmark_bytes_equal": True,
        "permitted_difference_boundary": (
            "Only new derived aggregates/metadata, corrected prose, stable truthful generator "
            "commit provenance, and resulting checksums may differ from c4."
        ),
    }


def episode_specs() -> tuple[EpisodeSpec, ...]:
    """Return the exact 16 disjoint Train/Validation episode specifications."""
    result = []
    for split in (Split.TRAIN, Split.VALIDATION):
        for motion_class in MotionClass:
            for ordinal, seed in enumerate(EPISODE_SEEDS[split][motion_class]):
                warmup_s = float(2 + seed % 3)
                score_start = int(warmup_s * CONTROL_FREQUENCY_HZ)
                episode_id = f"{split.value}-{motion_class.value}-{ordinal + 1:02d}-{seed}"
                result.append(
                    EpisodeSpec(
                        split=split,
                        motion_class=motion_class,
                        ordinal=ordinal,
                        seed=seed,
                        episode_id=episode_id,
                        pair_id=f"pair-{split.value}-{motion_class.value}-{ordinal + 1:02d}",
                        warmup_s=warmup_s,
                        score_start=score_start,
                        score_stop=score_start + SCORE_INTERVALS,
                    )
                )
    ids = [item.episode_id for item in result]
    seeds = [item.seed for item in result]
    if len(result) != 16 or len(set(ids)) != 16 or len(set(seeds)) != 16:
        raise RobustContractError("episode ID/seed split contract is not exactly disjoint")
    return tuple(result)


def component_key(spec: EpisodeSpec, attempt: int, component: str) -> Array:
    """Derive explicitly separated split/class/episode/attempt/component keys."""
    if attempt < 0 or component not in COMPONENT_NAMESPACE:
        raise ValueError("invalid attempt/component key coordinate")
    key = jax.random.key(spec.seed)
    for coordinate in (
        SPLIT_NAMESPACE[spec.split],
        CLASS_NAMESPACE[spec.motion_class],
        spec.ordinal,
        attempt,
        COMPONENT_NAMESPACE[component],
    ):
        key = jax.random.fold_in(key, coordinate)
    return key


def _smoothstep7(u: Array) -> tuple[Array, Array, Array, Array]:
    value = 35 * u**4 - 84 * u**5 + 70 * u**6 - 20 * u**7
    first = 140 * u**3 - 420 * u**4 + 420 * u**5 - 140 * u**6
    second = 420 * u**2 - 1680 * u**3 + 2100 * u**4 - 840 * u**5
    third = 840 * u - 5040 * u**2 + 8400 * u**3 - 4200 * u**4
    return value, first, second, third


def _c3_envelope(time: Array) -> tuple[Array, Array, Array, Array]:
    """Two-second C3 ramps around a four-second unit plateau."""
    ramp_s = jnp.asarray(2.0, dtype=time.dtype)
    left_u = jnp.clip(time / ramp_s, 0.0, 1.0)
    right_u = jnp.clip((PARENT_DURATION_S - time) / ramp_s, 0.0, 1.0)
    left = _smoothstep7(left_u)
    right_raw = _smoothstep7(right_u)
    right = (right_raw[0], -right_raw[1], right_raw[2], -right_raw[3])
    left_active = time < ramp_s
    right_active = time > PARENT_DURATION_S - ramp_s
    one = jnp.ones_like(time)
    zero = jnp.zeros_like(time)
    envelope = jnp.where(left_active, left[0], jnp.where(right_active, right[0], one))
    first = jnp.where(
        left_active, left[1] / ramp_s, jnp.where(right_active, right[1] / ramp_s, zero)
    )
    second = jnp.where(
        left_active, left[2] / ramp_s**2, jnp.where(right_active, right[2] / ramp_s**2, zero)
    )
    third = jnp.where(
        left_active, left[3] / ramp_s**3, jnp.where(right_active, right[3] / ramp_s**3, zero)
    )
    return envelope, first, second, third


def _fourier_series(
    time: Array, key: Array, axes: int, *, base_period_s: float = 2.0
) -> tuple[Array, Array, Array, Array]:
    """Return deterministic bounded Fourier values and derivatives through jerk."""
    coefficient_key, phase_key = jax.random.split(key)
    harmonics = jnp.arange(1, 4, dtype=time.dtype)
    omega = 2.0 * jnp.pi * harmonics / base_period_s
    coefficients = (
        jax.random.uniform(coefficient_key, (axes, 3), minval=-1.0, maxval=1.0, dtype=time.dtype)
        / harmonics[None, :] ** 2.5
    )
    axis_scale = jnp.asarray([1.0, 0.78, 0.65][:axes], dtype=time.dtype)[:, None]
    coefficients = coefficients * axis_scale
    phases = jax.random.uniform(
        phase_key, (axes, 3), minval=-jnp.pi, maxval=jnp.pi, dtype=time.dtype
    )
    angle = time[:, None, None] * omega[None, None, :] + phases[None, :, :]
    sin_angle = jnp.sin(angle)
    cos_angle = jnp.cos(angle)
    value = jnp.sum(coefficients[None] * sin_angle, axis=-1)
    first = jnp.sum(coefficients[None] * omega[None, None] * cos_angle, axis=-1)
    second = jnp.sum(-coefficients[None] * omega[None, None] ** 2 * sin_angle, axis=-1)
    third = jnp.sum(-coefficients[None] * omega[None, None] ** 3 * cos_angle, axis=-1)
    return value, first, second, third


def _enveloped(
    base: tuple[Array, Array, Array, Array], envelope: tuple[Array, Array, Array, Array]
) -> tuple[Array, Array, Array, Array]:
    b0, b1, b2, b3 = base
    e0, e1, e2, e3 = (value[:, None] for value in envelope)
    return (
        e0 * b0,
        e1 * b0 + e0 * b1,
        e2 * b0 + 2 * e1 * b1 + e0 * b2,
        e3 * b0 + 3 * e2 * b1 + 3 * e1 * b2 + e0 * b3,
    )


def _interval_indices(spec: EpisodeSpec) -> np.ndarray:
    """State/reference samples reached after each half-open scored control interval."""
    return np.arange(spec.score_start + 1, spec.score_stop + 1, dtype=np.int64)


def reference_statistics(
    trajectory: Trajectory, jerk: Array, indices: Iterable[int] | slice
) -> dict[str, Any]:
    """Compute the complete pre-controller reference envelope for selected samples."""
    position = np.asarray(trajectory.pos)[indices]
    velocity = np.asarray(trajectory.vel)[indices]
    acceleration = np.asarray(trajectory.acc)[indices]
    jerk_array = np.asarray(jerk)[indices]
    yaw = np.asarray(trajectory.yaw)[indices]
    yaw_rate = np.asarray(trajectory.yaw_rate)[indices]
    specific_force = acceleration + np.asarray([0.0, 0.0, 9.81])
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    jerk_norm = np.linalg.norm(jerk_array, axis=-1)
    specific_force_norm = np.linalg.norm(specific_force, axis=-1)
    tilt = np.abs(np.arctan2(np.linalg.norm(specific_force[:, :2], axis=-1), specific_force[:, 2]))
    translational_terms = {
        "speed": speed / GLOBAL_LIMITS["speed_m_s"],
        "acceleration": acceleration_norm / GLOBAL_LIMITS["acceleration_m_s2"],
        "jerk": jerk_norm / GLOBAL_LIMITS["jerk_m_s3"],
        "tilt": tilt / GLOBAL_LIMITS["tilt_rad"],
        "specific_force": np.maximum(
            (9.81 - specific_force_norm) / (9.81 - GLOBAL_LIMITS["specific_force_m_s2"]), 0.0
        ),
    }
    utilization_by_term = {
        name: float(np.max(value)) for name, value in translational_terms.items()
    }
    u_trans = max(utilization_by_term.values())
    u_yaw = float(np.max(np.abs(yaw_rate)) / GLOBAL_LIMITS["yaw_rate_rad_s"])

    def maximum_record(values: np.ndarray) -> dict[str, Any]:
        index = int(np.argmax(values))
        return {"value": float(values[index]), "local_index": index}

    def minimum_record(values: np.ndarray) -> dict[str, Any]:
        index = int(np.argmin(values))
        return {"value": float(values[index]), "local_index": index}

    return {
        "position_m": {
            "minimum_by_axis": np.min(position, axis=0).tolist(),
            "maximum_by_axis": np.max(position, axis=0).tolist(),
        },
        "max_speed_m_s": maximum_record(speed),
        "max_acceleration_m_s2": maximum_record(acceleration_norm),
        "max_jerk_m_s3": maximum_record(jerk_norm),
        "max_abs_yaw_rad": maximum_record(np.abs(yaw)),
        "max_abs_yaw_rate_rad_s": maximum_record(np.abs(yaw_rate)),
        "max_tilt_rad": maximum_record(tilt),
        "min_specific_force_m_s2": minimum_record(specific_force_norm),
        "utilization_by_term": utilization_by_term,
        "u_trans": float(u_trans),
        "u_yaw": float(u_yaw),
        "all_finite": bool(
            all(
                np.all(np.isfinite(value))
                for value in (position, velocity, acceleration, jerk_array, yaw, yaw_rate)
            )
        ),
        "sample_count": int(position.shape[0]),
    }


def _candidate_attempt(spec: EpisodeSpec, attempt: int) -> tuple[Trajectory, Array]:
    time = jnp.arange(PARENT_INTERVALS + 1, dtype=jnp.float32) / CONTROL_FREQUENCY_HZ
    envelope = _c3_envelope(time)
    position_base = _enveloped(
        _fourier_series(time, component_key(spec, attempt, "position"), 3), envelope
    )
    score_indices = _interval_indices(spec)
    raw_speed = np.linalg.norm(np.asarray(position_base[1])[score_indices], axis=-1)
    if float(np.max(raw_speed)) <= 0.0:
        raise RobustContractError("degenerate deterministic position candidate")
    position_scale = (
        CLASS_CONTRACTS[spec.motion_class].target_usage
        * GLOBAL_LIMITS["speed_m_s"]
        / float(np.max(raw_speed))
    )
    displacement, velocity, acceleration, jerk = (
        value * jnp.asarray(position_scale, dtype=time.dtype) for value in position_base
    )
    # A predeclared, class-independent six-centimetre-per-second central climb keeps the long
    # mismatch rollout persistently exciting in z while remaining inside every numerical class
    # band.  It is part of the analytic reference construction, not a controller-result filter.
    climb_rate = jnp.asarray(0.06, dtype=time.dtype)
    climb_base = (
        (climb_rate * (time - 2.0))[:, None],
        jnp.broadcast_to(climb_rate, time.shape)[:, None],
        jnp.zeros_like(time)[:, None],
        jnp.zeros_like(time)[:, None],
    )
    climb = _enveloped(climb_base, envelope)
    z_axis = jnp.asarray([0.0, 0.0, 1.0], dtype=time.dtype)
    displacement = displacement + climb[0] * z_axis
    velocity = velocity + climb[1] * z_axis
    acceleration = acceleration + climb[2] * z_axis
    jerk = jerk + climb[3] * z_axis
    center = jnp.asarray([0.0, 0.0, 0.75], dtype=time.dtype)
    position = center + displacement

    yaw_base = _enveloped(_fourier_series(time, component_key(spec, attempt, "yaw"), 1), envelope)
    raw_yaw_rate = np.abs(np.asarray(yaw_base[1])[:, 0][score_indices])
    if float(np.max(raw_yaw_rate)) <= 0.0:
        raise RobustContractError("degenerate deterministic yaw candidate")
    yaw_scale = CLASS_CONTRACTS[spec.motion_class].target_usage / float(np.max(raw_yaw_rate))
    yaw = yaw_base[0][:, 0] * jnp.asarray(yaw_scale, dtype=time.dtype)
    yaw_rate = yaw_base[1][:, 0] * jnp.asarray(yaw_scale, dtype=time.dtype)
    return (
        Trajectory(
            time=time, pos=position, vel=velocity, acc=acceleration, yaw=yaw, yaw_rate=yaw_rate
        ),
        jerk,
    )


def _acceptance_reasons(
    spec: EpisodeSpec, trajectory: Trajectory, score: dict[str, Any], parent: dict[str, Any]
) -> list[str]:
    contract = CLASS_CONTRACTS[spec.motion_class]
    reasons = []
    position = np.asarray(trajectory.pos)
    if not score["all_finite"] or not parent["all_finite"]:
        reasons.append("nonfinite_reference")
    if np.any(position < WORKSPACE_MIN_M) or np.any(position > WORKSPACE_MAX_M):
        reasons.append("parent_workspace")
    for name in ("u_trans", "u_yaw"):
        if not (contract.lower_open < score[name] <= contract.upper_closed):
            reasons.append(f"score_{name}_outside_class_band")
        if parent[name] > 0.95:
            reasons.append(f"parent_{name}_above_global_0p95")
    if score["max_abs_yaw_rad"]["value"] > contract.max_abs_yaw_rad:
        reasons.append("score_abs_yaw_above_class_limit")
    if parent["max_abs_yaw_rad"]["value"] > contract.max_abs_yaw_rad:
        reasons.append("parent_abs_yaw_above_class_limit")
    if score["max_abs_yaw_rate_rad_s"]["value"] <= 0.0:
        reasons.append("score_yaw_rate_not_nonzero")
    return reasons


def construct_episode(spec: EpisodeSpec) -> ReferenceCandidate:
    """Construct one accepted parent using reference-only deterministic rejection."""
    attempts = []
    for attempt in range(MAX_ATTEMPTS):
        trajectory, jerk = _candidate_attempt(spec, attempt)
        score = reference_statistics(trajectory, jerk, _interval_indices(spec))
        parent = reference_statistics(trajectory, jerk, slice(None))
        reasons = _acceptance_reasons(spec, trajectory, score, parent)
        attempts.append(
            {
                "attempt_index": attempt,
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "score_statistics": score,
                "parent_statistics": parent,
                "key_digests": {
                    component: array_digest(component_key(spec, attempt, component))
                    for component in COMPONENT_NAMESPACE
                },
            }
        )
        if not reasons:
            arrays = {
                "time": trajectory.time,
                "position": trajectory.pos,
                "velocity": trajectory.vel,
                "acceleration": trajectory.acc,
                "jerk": jerk,
                "yaw": trajectory.yaw,
                "yaw_rate": trajectory.yaw_rate,
            }
            return ReferenceCandidate(
                trajectory=trajectory,
                jerk=jerk,
                attempt=attempt,
                attempts=tuple(attempts),
                score_statistics=score,
                parent_statistics=parent,
                array_digests={name: array_digest(value) for name, value in arrays.items()},
            )
    raise RobustContractError(
        f"{spec.episode_id} has no accepted construction in {MAX_ATTEMPTS} attempts"
    )


def construct_all_episodes() -> tuple[tuple[EpisodeSpec, ReferenceCandidate], ...]:
    """Construct and validate the exact frozen 16-parent registry."""
    constructed = tuple((spec, construct_episode(spec)) for spec in episode_specs())
    if len(constructed) != 16:
        raise RobustContractError("exactly 16 parents are required")
    return constructed


def stack_evaluation_inputs(
    constructed: tuple[tuple[EpisodeSpec, ReferenceCandidate], ...],
) -> EvaluationInputs:
    """Stack six-second commands while preserving each complete eight-second parent."""
    specs = tuple(spec for spec, _ in constructed)
    candidates = tuple(candidate for _, candidate in constructed)
    commands = jnp.stack(
        tuple(
            state_commands(candidate.trajectory)[1 : ROLLOUT_INTERVALS + 1]
            for candidate in candidates
        ),
        axis=1,
    )[:, :, None, :]
    reference = jax.tree.map(
        lambda *values: jnp.stack(values, axis=1)[:, :, None, ...],
        *(
            jax.tree.map(lambda value: value[1 : ROLLOUT_INTERVALS + 1], item.trajectory)
            for item in candidates
        ),
    )
    expected = (ROLLOUT_INTERVALS, len(specs), 1, 13)
    if commands.shape != expected:
        raise RobustContractError(f"stacked command shape changed: {commands.shape} != {expected}")
    return EvaluationInputs(specs, candidates, commands, reference)


def build_simulation(n_worlds: int, *, rng_seed: int = 2501) -> Sim:
    """Build the frozen CPU/Float32 cf21B_500 simulation."""
    if jax.config.jax_enable_x64:
        raise RobustContractError("the frozen contract requires JAX x64 to be disabled")
    sim = Sim(
        n_worlds=n_worlds,
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
        rng_key=rng_seed,
    )
    if tuple(sim.step_pipeline) != EXPECTED_PIPELINE:
        raise RobustContractError("production six-stage step pipeline identity changed")
    if sim.device.platform != "cpu":
        raise RobustContractError("simulation backend is not CPU")
    return sim


def with_controller_mass(data: SimData, mass_kg: float) -> SimData:
    """Replace only the experiment-local state-controller mass."""
    state = data.controls.state
    if state is None:
        raise RobustContractError("controller mass requires the state controller")
    mass = jnp.asarray(mass_kg, dtype=state.params["mass"].dtype)
    return data.replace(
        controls=data.controls.replace(state=state.replace(params=state.params | {"mass": mass}))
    )


def initialize_inputs(data: SimData, inputs: EvaluationInputs) -> SimData:
    """Initialize t=0 state while retaining all untouched zero controller state."""
    initial_position = jnp.stack(
        tuple(candidate.trajectory.pos[0] for candidate in inputs.candidates), axis=0
    )[:, None, :]
    initial_velocity = jnp.stack(
        tuple(candidate.trajectory.vel[0] for candidate in inputs.candidates), axis=0
    )[:, None, :]
    initialized = initialize_tracking_state(data, initial_position, initial_velocity)
    dynamics_mass = np.asarray(initialized.params.mass)
    if not np.array_equal(
        dynamics_mass, np.full(dynamics_mass.shape, DYNAMICS_MASS_KG, dtype=dynamics_mass.dtype)
    ):
        raise RobustContractError(f"dynamics mass is not exactly {DYNAMICS_MASS_KG} kg")
    quaternion = np.asarray(initialized.states.quat)
    expected_quaternion = np.zeros_like(quaternion)
    expected_quaternion[..., -1] = 1.0
    if not np.array_equal(quaternion, expected_quaternion):
        raise RobustContractError("initial quaternion is not identity")
    if not np.all(np.asarray(initialized.states.ang_vel) == 0.0):
        raise RobustContractError("initial angular velocity is not zero")
    return initialized


def _leaf_paths(tree: Any) -> tuple[tuple[str, np.ndarray], ...]:
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)

    def to_numpy(leaf: Any) -> np.ndarray:
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        return np.asarray(leaf)

    return tuple((jax.tree_util.keystr(path), to_numpy(leaf)) for path, leaf in leaves_with_paths)


def mass_only_pytree_diff(mismatch: SimData, matched: SimData) -> dict[str, Any]:
    """Prove the two complete initial carries differ only in controller mass."""
    left = _leaf_paths(mismatch)
    right = _leaf_paths(matched)
    if len(left) != len(right):
        raise RobustContractError("mass-pair PyTree leaf count changed")
    differences = []
    for (left_path, left_value), (right_path, right_value) in zip(left, right, strict=True):
        if left_path != right_path:
            raise RobustContractError("mass-pair PyTree paths changed")
        if (
            left_value.shape != right_value.shape
            or left_value.dtype != right_value.dtype
            or not np.array_equal(left_value, right_value)
        ):
            differences.append(
                {
                    "path": left_path,
                    "left": left_value.tolist(),
                    "right": right_value.tolist(),
                    "dtype": str(left_value.dtype),
                    "shape": list(left_value.shape),
                }
            )
    if len(differences) != 1 or "['mass']" not in differences[0]["path"]:
        raise RobustContractError(
            f"mass pair differs outside the sole controller-mass leaf: {differences}"
        )
    difference = differences[0]
    if not np.array_equal(
        np.asarray(difference["left"]), np.asarray(MISMATCH_CONTROLLER_MASS_KG, dtype=np.float32)
    ) or not np.array_equal(
        np.asarray(difference["right"]), np.asarray(MATCHED_CONTROLLER_MASS_KG, dtype=np.float32)
    ):
        raise RobustContractError("mass-pair values changed")
    return {
        "status": "PASS_ONLY_CONTROLS_STATE_PARAMS_MASS_DIFFERS",
        "leaf_count": len(left),
        "differences": differences,
    }


def _wrench_from_motor_forces(motor_forces: Array, params: dict[str, Array]) -> Array:
    xp = array_namespace(motor_forces)
    torque = (params["mixing_matrix"] @ motor_forces[..., None])[..., 0]
    torque = torque * xp.stack([params["L"], params["L"], params["thrust2torque"]])
    return xp.concat((xp.sum(motor_forces, axis=-1)[..., None], torque), axis=-1)


def stage_a_diagnostic(data: SimData) -> dict[str, Array]:
    """Repeat Stage-A Float32 arithmetic immediately before production."""
    states = data.states
    control = data.controls.attitude
    force_torque_control = data.controls.force_torque
    if control is None or force_torque_control is None:
        raise RobustContractError("Stage A diagnostic requires both downstream controllers")
    mask = controllable(data.core.steps, data.core.freq, control.steps, control.freq)
    control = leaf_replace(control, mask, cmd=control.staged_cmd)
    params = control.params
    xp = array_namespace(states.quat)
    force_des = control.cmd[..., 3]
    rpy_des = control.cmd[..., :3]
    dt = 1 / control.freq
    rot = R.from_quat(states.quat)
    rot_des = R.from_euler("xyz", rpy_des, degrees=False)
    delta = (rot_des.inv() * rot).as_matrix()
    error_matrix = delta - delta.mT
    rotation_error = xp.stack(
        (error_matrix[..., 2, 1], error_matrix[..., 0, 2], error_matrix[..., 1, 0]), axis=-1
    )
    derivative_error = -(states.ang_vel - control.last_ang_vel) / dt
    derivative_error = xpx.at(derivative_error)[..., 2].set(0)
    integral_error = xp.clip(
        control.r_int_error - rotation_error * dt, -params["int_err_max"], params["int_err_max"]
    )
    raw_torque_pwm = (
        -params["kR"] * rotation_error
        - params["kw"] * states.ang_vel
        + params["ki_m"] * integral_error
        + params["kd_omega"] * derivative_error
    )
    torque_pwm = xp.clip(raw_torque_pwm, -params["torque_pwm_max"], params["torque_pwm_max"])
    torque_pwm = xp.where((force_des > 0)[..., None], torque_pwm, 0.0)
    force_pwm = force2pwm(force_des / 4, params["thrust_max"], params["pwm_max"])
    motor_pwm_preclip = force_torque_pwms2pwms(force_pwm, torque_pwm, params["mixing_matrix"])
    all_zero = xp.all(motor_pwm_preclip == 0, axis=-1, keepdims=True)
    motor_pwm_postclip = xp.where(
        all_zero, 0.0, xp.clip(motor_pwm_preclip, params["pwm_min"], params["pwm_max"])
    )
    motor_force_preclip = pwm2force(motor_pwm_preclip, params["thrust_max"], params["pwm_max"])
    motor_force_postclip = pwm2force(motor_pwm_postclip, params["thrust_max"], params["pwm_max"])
    rpm2thrust = force_torque_control.params["rpm2thrust"]
    motor_speed_preclip = motor_force2rotor_vel(motor_force_preclip, rpm2thrust)
    motor_speed_postclip = motor_force2rotor_vel(motor_force_postclip, rpm2thrust)
    force_lower = pwm2force(params["pwm_min"], params["thrust_max"], params["pwm_max"])
    speed_lower = motor_force2rotor_vel(force_lower, rpm2thrust)
    speed_upper = motor_force2rotor_vel(params["thrust_max"], rpm2thrust)
    wrench_preclip = _wrench_from_motor_forces(motor_force_preclip, params)
    wrench_postclip = _wrench_from_motor_forces(motor_force_postclip, params)
    distortion = wrench_postclip - wrench_preclip
    return {
        "collective_force_request_n": force_des,
        "raw_torque_pwm": raw_torque_pwm,
        "postclip_torque_pwm": torque_pwm,
        "torque_lower_clip_mask": raw_torque_pwm < -params["torque_pwm_max"],
        "torque_upper_clip_mask": raw_torque_pwm > params["torque_pwm_max"],
        "torque_any_clip_mask": xp.abs(raw_torque_pwm) > params["torque_pwm_max"],
        "motor_pwm_preclip": motor_pwm_preclip,
        "motor_pwm_postclip": motor_pwm_postclip,
        "motor_pwm_lower_clip_mask": motor_pwm_preclip < params["pwm_min"],
        "motor_pwm_upper_clip_mask": motor_pwm_preclip > params["pwm_max"],
        "motor_pwm_any_clip_mask": (motor_pwm_preclip < params["pwm_min"])
        | (motor_pwm_preclip > params["pwm_max"]),
        "motor_force_preclip_n": motor_force_preclip,
        "motor_force_postclip_n": motor_force_postclip,
        "motor_speed_preclip": motor_speed_preclip,
        "motor_speed_postclip": motor_speed_postclip,
        "force_lower_reserve_n": motor_force_postclip - force_lower,
        "force_upper_reserve_n": params["thrust_max"] - motor_force_postclip,
        "force_lower_reserve_normalized": (motor_force_postclip - force_lower)
        / (params["thrust_max"] - force_lower),
        "force_upper_reserve_normalized": (params["thrust_max"] - motor_force_postclip)
        / (params["thrust_max"] - force_lower),
        "speed_lower_reserve": motor_speed_postclip - speed_lower,
        "speed_upper_reserve": speed_upper - motor_speed_postclip,
        "speed_lower_reserve_normalized": (motor_speed_postclip - speed_lower)
        / (speed_upper - speed_lower),
        "speed_upper_reserve_normalized": (speed_upper - motor_speed_postclip)
        / (speed_upper - speed_lower),
        "wrench_preclip": wrench_preclip,
        "wrench_postclip": wrench_postclip,
        "wrench_distortion_signed": distortion,
        "wrench_distortion_absolute": xp.abs(distortion),
        "wrench_distortion_platform_normalized": xp.abs(distortion)
        / xp.asarray(WRENCH_PLATFORM_SCALE, dtype=distortion.dtype),
    }


def stage_b_diagnostic(data: SimData) -> dict[str, Array]:
    """Repeat Stage-B Float32 arithmetic immediately before production."""
    control = data.controls.force_torque
    if control is None:
        raise RobustContractError("Stage B diagnostic requires the force/torque controller")
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
    wrench_preclip = _wrench_from_motor_forces(motor_force_preclip, params)
    wrench_postclip = _wrench_from_motor_forces(motor_force_postclip, params)
    distortion = wrench_postclip - wrench_preclip
    return {
        "input_stage_a_production_wrench": control.cmd,
        "motor_force_preclip_n": motor_force_preclip,
        "motor_force_postclip_n": motor_force_postclip,
        "commanded_motor_speed": commanded_speed,
        "motor_force_lower_clip_mask": motor_force_preclip < params["thrust_min"],
        "motor_force_upper_clip_mask": motor_force_preclip > params["thrust_max"],
        "motor_force_any_clip_mask": (motor_force_preclip < params["thrust_min"])
        | (motor_force_preclip > params["thrust_max"]),
        "wrench_preclip": wrench_preclip,
        "wrench_postclip": wrench_postclip,
        "wrench_distortion_signed": distortion,
        "wrench_distortion_absolute": xp.abs(distortion),
        "wrench_distortion_platform_normalized": xp.abs(distortion)
        / xp.asarray(WRENCH_PLATFORM_SCALE, dtype=distortion.dtype),
    }


def _trace_from_data(data: SimData) -> RolloutTrace:
    state = data.controls.state
    attitude = data.controls.attitude
    force_torque = data.controls.force_torque
    if state is None or attitude is None or force_torque is None:
        raise RobustContractError("complete state-control trace is unavailable")
    return RolloutTrace(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        rotor_vel=data.states.rotor_vel,
        commanded_rotor_vel=data.controls.rotor_vel,
        state_command=state.cmd,
        attitude_command=attitude.cmd,
        force_torque_command=force_torque.cmd,
    )


def make_instrumented_rollout(
    sim: Sim,
) -> Callable[[SimData, Array], tuple[SimData, tuple[SimData, SimData, SimData], Any]]:
    """Return a JIT rollout that keeps production phases and captures local diagnostics."""
    pipeline = tuple(sim.step_pipeline.items())
    if tuple(name for name, _ in pipeline) != EXPECTED_PIPELINE:
        raise RobustContractError("production six-stage pipeline identity changed")

    def one_simulation_step(data: SimData, _: None) -> tuple[SimData, dict[str, Any]]:
        stage_a: dict[str, Array] | None = None
        stage_b: dict[str, Array] | None = None
        for name, function in pipeline:
            if name == "attitude_controller":
                stage_a = stage_a_diagnostic(data)
            elif name == "force_torque_controller":
                stage_b = stage_b_diagnostic(data)
            data = function(data)
            if name == "attitude_controller":
                assert stage_a is not None
                stage_a["production_postclip_wrench"] = data.controls.force_torque.staged_cmd
            elif name == "force_torque_controller":
                assert stage_b is not None
                stage_b["production_commanded_motor_speed"] = data.controls.rotor_vel
        assert stage_a is not None and stage_b is not None
        return data, {"stage_a": stage_a, "stage_b": stage_b}

    boundary_counts = (200, 300, 400)

    def interval_step(
        carry: tuple[SimData, SimData, SimData, SimData, Array], command: Array
    ) -> tuple[tuple[SimData, SimData, SimData, SimData, Array], dict[str, Any]]:
        data, boundary_200, boundary_300, boundary_400, index = carry
        data = F.state_control(data, command)
        data, substeps = jax.lax.scan(
            one_simulation_step, data, None, length=STEPS_PER_CONTROL, unroll=1
        )
        data = data.replace(core=data.core.replace(mjx_synced=False))
        completed = index + 1
        snapshots = (boundary_200, boundary_300, boundary_400)
        snapshots = tuple(
            jax.tree.map(
                lambda previous, current, target=target: jnp.where(
                    completed == target, current, previous
                ),
                previous,
                data,
            )
            for previous, target in zip(snapshots, boundary_counts, strict=True)
        )
        state = data.controls.state
        assert state is not None
        output = {
            "trace": _trace_from_data(data),
            "pos_err_i": state.pos_err_i,
            "diagnostic": substeps,
        }
        return (data, *snapshots, completed), output

    @jax.jit
    def rollout(
        initial_data: SimData, commands: Array
    ) -> tuple[SimData, tuple[SimData, SimData, SimData], Any]:
        carry = (initial_data, initial_data, initial_data, initial_data, jnp.asarray(0))
        carry, outputs = jax.lax.scan(interval_step, carry, commands)
        final, boundary_200, boundary_300, boundary_400, _ = carry
        return final, (boundary_200, boundary_300, boundary_400), outputs

    return rollout


def _numeric_leaves_finite(value: Any) -> bool:
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            return False
    return True


def require_array_equal(label: str, left: Any, right: Any) -> None:
    """Require shape/dtype/value identity under the predeclared exact contract."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (
        left_array.shape != right_array.shape
        or left_array.dtype != right_array.dtype
        or not np.array_equal(left_array, right_array)
    ):
        maximum_error = math.inf
        if left_array.shape == right_array.shape and left_array.size:
            maximum_error = float(
                np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64)))
            )
        raise RobustContractError(f"{label} exact identity failed; max error {maximum_error}")


def reconstruction_check(label: str, left: Any, right: Any, scale: Any) -> dict[str, Any]:
    """Apply the frozen transparent-Float32 absolute-error bound."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise RobustContractError(f"{label} reconstruction shape mismatch")
    error = np.abs(left_array - right_array)
    bound = (
        RECONSTRUCTION_FACTOR
        * FLOAT32_EPS
        * np.maximum(
            np.asarray(scale, dtype=np.float64), np.maximum(np.abs(left_array), np.abs(right_array))
        )
    )
    if np.any(error > bound):
        raise RobustContractError(
            f"{label} exceeds Float32 reconstruction bound: "
            f"{float(error.max(initial=0.0))} > {float(bound.max(initial=0.0))}"
        )
    return {
        "passed": True,
        "maximum_absolute_error": float(error.max(initial=0.0)),
        "maximum_allowed_error": float(bound.max(initial=0.0)),
        "factor_times_float32_eps": RECONSTRUCTION_FACTOR * FLOAT32_EPS,
        "rtol": 0.0,
    }


def require_pytree_equal(label: str, left: Any, right: Any) -> None:
    """Require exact path, shape, dtype, and value identity for every PyTree leaf."""
    left_leaves = _leaf_paths(left)
    right_leaves = _leaf_paths(right)
    if len(left_leaves) != len(right_leaves):
        raise RobustContractError(f"{label} PyTree leaf-count mismatch")
    for (left_path, left_value), (right_path, right_value) in zip(
        left_leaves, right_leaves, strict=True
    ):
        if left_path != right_path:
            raise RobustContractError(f"{label} PyTree path mismatch")
        require_array_equal(f"{label} {left_path}", left_value, right_value)


def carry_replay_checks(
    initial_data: SimData,
    commands: Array,
    step_fn: Callable[[SimData, int], SimData],
    captured_boundaries: tuple[SimData, SimData, SimData],
) -> dict[str, Any]:
    """Compare each uncut-rollout boundary carry with an independent prefix replay."""
    records = []
    for boundary, captured in zip((200, 300, 400), captured_boundaries, strict=True):
        replay_final, _ = rollout_state_commands(
            initial_data, commands[:boundary], step_fn, STEPS_PER_CONTROL
        )
        jax.block_until_ready(replay_final)
        require_pytree_equal(f"complete carry at interval {boundary}", replay_final, captured)
        records.append(
            {
                "boundary_interval": boundary,
                "boundary_time_s": boundary / CONTROL_FREQUENCY_HZ,
                "all_pytree_leaves_array_equal": True,
                "reset_or_reinitialization": False,
            }
        )
    return {
        "status": "PASS_COMPLETE_SIMDATA_CARRY_CONTINUOUS",
        "comparisons": records,
        "scoring_applied_after_full_rollout": True,
        "gradient_prefix_detached": False,
    }


def _trace_identity(custom: RolloutTrace, production: RolloutTrace, label: str) -> None:
    for field in TRACE_FIELDS:
        require_array_equal(
            f"{label} production trace {field}", getattr(custom, field), getattr(production, field)
        )


def validate_diagnostic_identity(outputs: Any) -> dict[str, Any]:
    """Validate local Stage-A/B reconstructions against same-step production results."""
    diagnostics = outputs["diagnostic"]
    stage_a = diagnostics["stage_a"]
    stage_b = diagnostics["stage_b"]
    stage_a_check = reconstruction_check(
        "Stage-A postclip wrench",
        stage_a["wrench_postclip"],
        stage_a["production_postclip_wrench"],
        WRENCH_PLATFORM_SCALE,
    )
    stage_b_check = reconstruction_check(
        "Stage-B commanded motor speed",
        stage_b["commanded_motor_speed"],
        stage_b["production_commanded_motor_speed"],
        65535.0,
    )
    stage_b_input_check = reconstruction_check(
        "Stage-B input wrench",
        stage_b["wrench_preclip"],
        stage_b["input_stage_a_production_wrench"],
        WRENCH_PLATFORM_SCALE,
    )
    return {
        "stage_a_postclip_wrench_vs_production": stage_a_check,
        "stage_b_command_vs_production": stage_b_check,
        "stage_b_preclip_wrench_vs_stage_a_production": stage_b_input_check,
    }


def _gather_per_world(value: Array, starts: Array, length: int) -> Array:
    """Differentiably gather one fixed-length time window per world."""
    value = jnp.asarray(value)
    world_count = value.shape[1]
    if starts.shape != (world_count,):
        raise ValueError(f"expected starts shape {(world_count,)}, got {starts.shape}")
    indices = jnp.arange(length, dtype=starts.dtype)[:, None] + starts[None, :]
    index_shape = (length, world_count) + (1,) * (value.ndim - 2)
    indices = jnp.reshape(indices, index_shape)
    indices = jnp.broadcast_to(indices, (length, world_count, *value.shape[2:]))
    return jnp.take_along_axis(value, indices, axis=0)


def scored_trace(trace: RolloutTrace, specs: tuple[EpisodeSpec, ...]) -> RolloutTrace:
    """Gather each world's frozen 200-interval interior window from a complete trace."""
    starts = jnp.asarray([spec.score_start for spec in specs], dtype=jnp.int32)
    return jax.tree.map(lambda value: _gather_per_world(value, starts, SCORE_INTERVALS), trace)


def scored_reference(reference: Trajectory, specs: tuple[EpisodeSpec, ...]) -> Trajectory:
    """Gather interval-aligned references for each frozen score window."""
    starts = jnp.asarray([spec.score_start for spec in specs], dtype=jnp.int32)
    return jax.tree.map(lambda value: _gather_per_world(value, starts, SCORE_INTERVALS), reference)


def scored_interval_value(value: Array, specs: tuple[EpisodeSpec, ...]) -> Array:
    """Gather a generic 100-Hz interval value for each frozen score window."""
    starts = jnp.asarray([spec.score_start for spec in specs], dtype=jnp.int32)
    return _gather_per_world(value, starts, SCORE_INTERVALS)


def scored_diagnostic_value(value: Array, specs: tuple[EpisodeSpec, ...]) -> Array:
    """Gather per-world interval windows from ``(interval, substep, world, ...)`` data."""
    value = jnp.asarray(value)
    starts = jnp.asarray([spec.score_start for spec in specs], dtype=jnp.int32)
    world_count = value.shape[2]
    indices = jnp.arange(SCORE_INTERVALS, dtype=starts.dtype)[:, None] + starts[None, :]
    index_shape = (SCORE_INTERVALS, 1, world_count) + (1,) * (value.ndim - 3)
    indices = jnp.reshape(indices, index_shape)
    indices = jnp.broadcast_to(
        indices, (SCORE_INTERVALS, value.shape[1], world_count, *value.shape[3:])
    )
    return jnp.take_along_axis(value, indices, axis=0)


def _runs(mask: np.ndarray, start_time_s: float, frequency_hz: int) -> list[dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    intervals = []
    cursor = 0
    while cursor < mask.size:
        if not mask[cursor]:
            cursor += 1
            continue
        start = cursor
        while cursor + 1 < mask.size and mask[cursor + 1]:
            cursor += 1
        stop = cursor + 1
        intervals.append(
            {
                "start_index_inclusive": start,
                "stop_index_exclusive": stop,
                "start_time_s": start_time_s + start / frequency_hz,
                "end_time_s": start_time_s + stop / frequency_hz,
                "sample_count": stop - start,
                "duration_s": (stop - start) / frequency_hz,
            }
        )
        cursor += 1
    return intervals


def mask_summary(
    mask: Any, *, start_time_s: float, frequency_hz: int, channel_name: str
) -> dict[str, Any]:
    """Summarize exact clip/detection decisions with half-open intervals."""
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 1:
        array = array[:, None]
    channels = []
    for channel in range(array.shape[1]):
        indices = np.flatnonzero(array[:, channel])
        intervals = _runs(array[:, channel], start_time_s, frequency_hz)
        channels.append(
            {
                channel_name: channel,
                "count": int(indices.size),
                "indices_zero_based": indices.astype(int).tolist(),
                "times_s": (start_time_s + indices / frequency_hz).tolist(),
                "intervals": intervals,
                "longest_duration_s": max(
                    (interval["duration_s"] for interval in intervals), default=0.0
                ),
            }
        )
    return {
        "count": int(array.sum()),
        "sample_count": int(array.size),
        "fraction": float(array.mean()),
        "any": bool(array.any()),
        "by_channel": channels,
        "longest_duration_s": max(
            (channel["longest_duration_s"] for channel in channels), default=0.0
        ),
    }


def _error_metrics(position_error: np.ndarray, velocity_error: np.ndarray) -> dict[str, float]:
    position_error = np.asarray(position_error, dtype=np.float64)
    velocity_error = np.asarray(velocity_error, dtype=np.float64)
    total = np.linalg.norm(position_error, axis=-1)
    xy = np.linalg.norm(position_error[..., :2], axis=-1)
    z = np.abs(position_error[..., 2])
    velocity_total = np.linalg.norm(velocity_error, axis=-1)
    velocity_xy = np.linalg.norm(velocity_error[..., :2], axis=-1)
    velocity_z = velocity_error[..., 2]
    return {
        "position_rmse_total_m": float(np.sqrt(np.mean(total**2))),
        "position_rmse_xy_m": float(np.sqrt(np.mean(xy**2))),
        "position_rmse_z_m": float(np.sqrt(np.mean(z**2))),
        "position_p95_total_m": float(np.quantile(total, 0.95)),
        "position_p95_xy_m": float(np.quantile(xy, 0.95)),
        "position_p95_abs_z_m": float(np.quantile(z, 0.95)),
        "position_max_total_m": float(np.max(total)),
        "position_max_xy_m": float(np.max(xy)),
        "position_max_abs_z_m": float(np.max(z)),
        "position_terminal_total_m": float(total[-1]),
        "position_terminal_xy_m": float(xy[-1]),
        "position_terminal_abs_z_m": float(z[-1]),
        "position_terminal_signed_z_m": float(position_error[-1, 2]),
        "position_mean_signed_z_m": float(np.mean(position_error[..., 2])),
        "velocity_rmse_total_m_s": float(np.sqrt(np.mean(velocity_total**2))),
        "velocity_rmse_xy_m_s": float(np.sqrt(np.mean(velocity_xy**2))),
        "velocity_rmse_z_m_s": float(np.sqrt(np.mean(velocity_z**2))),
    }


def _loss_term_records(terms: Any, world: int) -> tuple[dict[str, Any], float]:
    records = {}
    weighted_sum = 0.0
    for name in ("position", "velocity", "effort", "smoothness", "terminal", "altitude"):
        record = {
            "raw": float(np.asarray(terms.raw[name])[world]),
            "normalization_divisor": float(np.asarray(terms.normalization_divisor[name])),
            "weight": float(np.asarray(terms.weight[name])),
            "normalized": float(np.asarray(terms.normalized[name])[world]),
            "weighted": float(np.asarray(terms.weighted[name])[world]),
        }
        records[name] = record
        weighted_sum += record["weighted"]
    return records, weighted_sum


def _integral_summary(
    values: np.ndarray, limits: np.ndarray, start_time_s: float
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    threshold = limits * (1.0 - INTEGRAL_DETECTION_THRESHOLD)
    lower_detection = values <= -threshold
    upper_detection = values >= threshold
    detection = lower_detection | upper_detection
    combined = mask_summary(
        detection,
        start_time_s=start_time_s,
        frequency_hz=CONTROL_FREQUENCY_HZ,
        channel_name="axis_index",
    )
    lower = mask_summary(
        lower_detection,
        start_time_s=start_time_s,
        frequency_hz=CONTROL_FREQUENCY_HZ,
        channel_name="axis_index",
    )
    upper = mask_summary(
        upper_detection,
        start_time_s=start_time_s,
        frequency_hz=CONTROL_FREQUENCY_HZ,
        channel_name="axis_index",
    )
    axis_names = ("x", "y", "z")
    by_axis = []
    for axis, axis_name in enumerate(axis_names):
        axis_mask = detection[:, axis]
        hit_indices = np.flatnonzero(axis_mask)
        sides = []
        if np.any(lower_detection[:, axis]):
            sides.append("lower_negative")
        if np.any(upper_detection[:, axis]):
            sides.append("upper_positive")
        by_axis.append(
            {
                "axis_index": axis,
                "axis": axis_name,
                "bound_sides": sides,
                "hit_count": int(hit_indices.size),
                "sample_count": int(axis_mask.size),
                "hit_fraction": float(axis_mask.mean()),
                "total_duration_s": float(hit_indices.size / CONTROL_FREQUENCY_HZ),
                "first_contact_time_s": (
                    float(start_time_s + hit_indices[0] / CONTROL_FREQUENCY_HZ)
                    if hit_indices.size
                    else None
                ),
                "last_contact_time_s": (
                    float(start_time_s + hit_indices[-1] / CONTROL_FREQUENCY_HZ)
                    if hit_indices.size
                    else None
                ),
                "maximum_absolute_state": float(np.max(np.abs(values[:, axis]))),
                "terminal_state": float(values[-1, axis]),
                "limit": float(limits[axis]),
                "lower": lower["by_channel"][axis],
                "upper": upper["by_channel"][axis],
                "combined": combined["by_channel"][axis],
            }
        )
    return {
        "max_abs_by_axis": np.max(np.abs(values), axis=0).tolist(),
        "end_by_axis": values[-1].tolist(),
        "max_fraction_of_limit_by_axis": np.max(np.abs(values) / limits, axis=0).tolist(),
        "limits": limits.tolist(),
        "detection_threshold_fraction": INTEGRAL_DETECTION_THRESHOLD,
        "clip_detection": combined,
        "lower_contact": lower,
        "upper_contact": upper,
        "contact_by_axis": by_axis,
        "contact_status": "CONTACT" if combined["any"] else "NO_CONTACT",
    }


def _reserve_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    fields = (
        "force_lower_reserve_n",
        "force_upper_reserve_n",
        "force_lower_reserve_normalized",
        "force_upper_reserve_normalized",
        "speed_lower_reserve",
        "speed_upper_reserve",
        "speed_lower_reserve_normalized",
        "speed_upper_reserve_normalized",
    )
    by_field = {}
    for name in fields:
        value = arrays[name]
        by_field[name] = {
            "minimum_by_motor": np.min(value, axis=0).tolist(),
            "maximum_by_motor": np.max(value, axis=0).tolist(),
            "minimum_all_motors": float(np.min(value)),
        }
    normalized_minimum = np.minimum(
        arrays["force_lower_reserve_normalized"], arrays["force_upper_reserve_normalized"]
    )
    speed_normalized_minimum = np.minimum(
        arrays["speed_lower_reserve_normalized"], arrays["speed_upper_reserve_normalized"]
    )
    return {
        "by_field": by_field,
        "minimum_normalized_force_reserve": float(np.min(normalized_minimum)),
        "minimum_normalized_speed_reserve": float(np.min(speed_normalized_minimum)),
    }


def _wrench_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    preclip = arrays["wrench_preclip"]
    postclip = arrays["wrench_postclip"]
    distortion = arrays["wrench_distortion_signed"]
    axes = ("collective", "roll", "pitch", "yaw")
    by_axis = []
    for axis, name in enumerate(axes):
        index = int(np.argmax(np.abs(distortion[:, axis])))
        desired = float(preclip[index, axis])
        signed = float(distortion[index, axis])
        relative = signed / desired if abs(desired) >= WRENCH_RELATIVE_FLOOR[axis] else None
        by_axis.append(
            {
                "axis": name,
                "maximum_absolute_distortion": abs(signed),
                "signed_distortion_at_maximum": signed,
                "preclip_at_maximum": desired,
                "postclip_at_maximum": float(postclip[index, axis]),
                "platform_normalized_absolute_distortion": abs(signed)
                / WRENCH_PLATFORM_SCALE[axis],
                "relative_distortion_when_defined": relative,
                "relative_floor": float(WRENCH_RELATIVE_FLOOR[axis]),
                "sample_index": index,
            }
        )
    return {"by_axis": by_axis}


def _episode_arrays(
    world: int,
    score_trace_value: RolloutTrace,
    score_reference_value: Trajectory,
    score_pos_err_i: Array,
    score_diagnostic: Any,
) -> dict[str, np.ndarray]:
    reference_position = np.asarray(score_reference_value.pos)[:, world, 0]
    reference_velocity = np.asarray(score_reference_value.vel)[:, world, 0]
    trace_position = np.asarray(score_trace_value.pos)[:, world, 0]
    trace_velocity = np.asarray(score_trace_value.vel)[:, world, 0]
    stage_a = score_diagnostic["stage_a"]
    stage_b = score_diagnostic["stage_b"]

    def flatten_substeps(value: Any) -> np.ndarray:
        array = np.asarray(value)[:, :, world, 0]
        return array.reshape((-1, *array.shape[2:]))

    result = {
        "reference_position_m": reference_position,
        "reference_velocity_m_s": reference_velocity,
        "actual_position_m": trace_position,
        "actual_velocity_m_s": trace_velocity,
        "position_error_m": trace_position - reference_position,
        "velocity_error_m_s": trace_velocity - reference_velocity,
        "pos_err_i": np.asarray(score_pos_err_i)[:, world, 0],
        "commanded_motor_speed": np.asarray(score_trace_value.commanded_rotor_vel)[:, world, 0],
    }
    for name in (
        "raw_torque_pwm",
        "postclip_torque_pwm",
        "torque_lower_clip_mask",
        "torque_upper_clip_mask",
        "torque_any_clip_mask",
        "motor_pwm_preclip",
        "motor_pwm_postclip",
        "motor_pwm_lower_clip_mask",
        "motor_pwm_upper_clip_mask",
        "motor_pwm_any_clip_mask",
        "motor_force_preclip_n",
        "motor_force_postclip_n",
        "motor_speed_preclip",
        "motor_speed_postclip",
        "force_lower_reserve_n",
        "force_upper_reserve_n",
        "force_lower_reserve_normalized",
        "force_upper_reserve_normalized",
        "speed_lower_reserve",
        "speed_upper_reserve",
        "speed_lower_reserve_normalized",
        "speed_upper_reserve_normalized",
        "wrench_preclip",
        "wrench_postclip",
        "wrench_distortion_signed",
        "wrench_distortion_absolute",
        "wrench_distortion_platform_normalized",
        "production_postclip_wrench",
        "collective_force_request_n",
    ):
        result[f"stage_a_{name}"] = flatten_substeps(stage_a[name])
    for name in (
        "motor_force_preclip_n",
        "motor_force_postclip_n",
        "commanded_motor_speed",
        "motor_force_lower_clip_mask",
        "motor_force_upper_clip_mask",
        "motor_force_any_clip_mask",
        "wrench_preclip",
        "wrench_postclip",
        "wrench_distortion_signed",
        "wrench_distortion_absolute",
        "wrench_distortion_platform_normalized",
        "input_stage_a_production_wrench",
        "production_commanded_motor_speed",
    ):
        result[f"stage_b_{name}"] = flatten_substeps(stage_b[name])
    return result


def _mechanism_window_arrays(
    world: int, start: int, stop: int, inputs: EvaluationInputs, outputs: Any
) -> dict[str, np.ndarray]:
    """Return measured 100-Hz and 500-Hz mechanism-boundary time series."""
    trace = outputs["trace"]
    diagnostic = outputs["diagnostic"]
    actual_z = np.asarray(trace.pos)[start:stop, world, 0, 2]
    reference_z = np.asarray(inputs.reference.pos)[start:stop, world, 0, 2]
    integral_z = np.asarray(outputs["pos_err_i"])[start:stop, world, 0, 2]

    def flatten(value: Any) -> np.ndarray:
        array = np.asarray(value)[start:stop, :, world, 0]
        return array.reshape((-1, *array.shape[2:]))

    lower_reserve = flatten(diagnostic["stage_a"]["force_lower_reserve_normalized"])
    upper_reserve = flatten(diagnostic["stage_a"]["force_upper_reserve_normalized"])
    return {
        "state_time_s": (np.arange(start, stop, dtype=np.float64) + 1) / CONTROL_FREQUENCY_HZ,
        "reference_z_m": reference_z,
        "actual_z_m": actual_z,
        "z_error_m": actual_z - reference_z,
        "z_integral_state": integral_z,
        "diagnostic_time_s": np.arange(
            start * STEPS_PER_CONTROL, stop * STEPS_PER_CONTROL, dtype=np.float64
        )
        / SIMULATION_FREQUENCY_HZ,
        "collective_target_thrust_equivalent_request_n": flatten(
            diagnostic["stage_a"]["collective_force_request_n"]
        ),
        "realized_wrench": flatten(diagnostic["stage_b"]["wrench_postclip"]),
        "production_motor_command": flatten(
            diagnostic["stage_b"]["production_commanded_motor_speed"]
        ),
        "normalized_motor_force_reserve": np.minimum(lower_reserve, upper_reserve),
    }


def mechanism_episode_section(
    world: int, spec: EpisodeSpec, inputs: EvaluationInputs, outputs: Any
) -> dict[str, Any]:
    """Separate measured early-transient/scored arrays from inference and transfer claims."""
    early = _mechanism_window_arrays(world, 0, spec.score_start, inputs, outputs)
    scored = _mechanism_window_arrays(world, spec.score_start, spec.score_stop, inputs, outputs)
    return {
        "measured": {
            "early_transient": _json_array_section(early),
            "scored_window": _json_array_section(scored),
            "boundary": (
                "Direct simulation arrays at existing controller/diagnostic boundaries; "
                "no isolated causal attribution."
            ),
        },
        "code_path_inference": {
            "expression": (
                "state2attitude uses mass * (setpoint_acc - gravity_vec) + feedback, "
                "followed by mass_thrust=132000 and nonlinear PWM-to-force conversion"
            ),
            "status": "implementation-path explanation, not experimental causal proof",
        },
        "transfer_boundary": (
            "Controller mass acts as a thrust-feedforward parameter inside this fixed chain. "
            "It is not a physical-mass estimate, firmware parameter, or flight value without "
            "a separate parameter-semantics and transfer gate."
        ),
    }


def _json_array_section(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "contracts": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(arrays.items())
        },
        "digests": {name: array_digest(value) for name, value in sorted(arrays.items())},
        "values": {name: value.tolist() for name, value in sorted(arrays.items())},
    }


def _named_subset(
    arrays: dict[str, np.ndarray], prefix: str, names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    return {name: arrays[f"{prefix}{name}"] for name in names}


def summary_from_episode_arrays(
    arrays: dict[str, np.ndarray],
    *,
    start_time_s: float,
    integral_limits: np.ndarray,
    loss_terms: dict[str, Any],
    loss_total: float,
) -> dict[str, Any]:
    """Recompute every per-episode report summary from its stored arrays."""
    stage_a_motor = arrays["stage_a_motor_pwm_any_clip_mask"]
    stage_a_torque = arrays["stage_a_torque_any_clip_mask"]
    stage_b_motor = arrays["stage_b_motor_force_any_clip_mask"]
    reserves = _named_subset(
        arrays,
        "stage_a_",
        (
            "force_lower_reserve_n",
            "force_upper_reserve_n",
            "force_lower_reserve_normalized",
            "force_upper_reserve_normalized",
            "speed_lower_reserve",
            "speed_upper_reserve",
            "speed_lower_reserve_normalized",
            "speed_upper_reserve_normalized",
        ),
    )
    stage_a_wrench = _named_subset(
        arrays, "stage_a_", ("wrench_preclip", "wrench_postclip", "wrench_distortion_signed")
    )
    stage_b_wrench = _named_subset(
        arrays, "stage_b_", ("wrench_preclip", "wrench_postclip", "wrench_distortion_signed")
    )
    return {
        "tracking": _error_metrics(arrays["position_error_m"], arrays["velocity_error_m_s"]),
        "loss_v1": {
            "terms": loss_terms,
            "exact_weighted_sum": float(sum(item["weighted"] for item in loss_terms.values())),
            "reported_total": float(loss_total),
            "arithmetic_absolute_error": abs(
                float(sum(item["weighted"] for item in loss_terms.values())) - loss_total
            ),
        },
        "integral": _integral_summary(arrays["pos_err_i"], integral_limits, start_time_s),
        "motor_reserve": _reserve_summary(reserves),
        "clipping": {
            "stage_a_torque": mask_summary(
                stage_a_torque,
                start_time_s=start_time_s,
                frequency_hz=SIMULATION_FREQUENCY_HZ,
                channel_name="axis_index",
            ),
            "stage_a_motor_lower": mask_summary(
                arrays["stage_a_motor_pwm_lower_clip_mask"],
                start_time_s=start_time_s,
                frequency_hz=SIMULATION_FREQUENCY_HZ,
                channel_name="motor_index",
            ),
            "stage_a_motor_upper": mask_summary(
                arrays["stage_a_motor_pwm_upper_clip_mask"],
                start_time_s=start_time_s,
                frequency_hz=SIMULATION_FREQUENCY_HZ,
                channel_name="motor_index",
            ),
            "stage_a_motor": mask_summary(
                stage_a_motor,
                start_time_s=start_time_s,
                frequency_hz=SIMULATION_FREQUENCY_HZ,
                channel_name="motor_index",
            ),
            "stage_b_additional_lower": mask_summary(
                arrays["stage_b_motor_force_lower_clip_mask"],
                start_time_s=start_time_s,
                frequency_hz=SIMULATION_FREQUENCY_HZ,
                channel_name="motor_index",
            ),
            "stage_b_additional_upper": mask_summary(
                arrays["stage_b_motor_force_upper_clip_mask"],
                start_time_s=start_time_s,
                frequency_hz=SIMULATION_FREQUENCY_HZ,
                channel_name="motor_index",
            ),
            "stage_b_additional": mask_summary(
                stage_b_motor,
                start_time_s=start_time_s,
                frequency_hz=SIMULATION_FREQUENCY_HZ,
                channel_name="motor_index",
            ),
        },
        "wrench": {
            "stage_a": _wrench_summary(stage_a_wrench),
            "stage_b_additional": _wrench_summary(stage_b_wrench),
            "platform_scales": WRENCH_PLATFORM_SCALE.tolist(),
            "relative_floors": WRENCH_RELATIVE_FLOOR.tolist(),
        },
        "postclip_motor": {
            "force_min_by_motor_n": np.min(
                arrays["stage_a_motor_force_postclip_n"], axis=0
            ).tolist(),
            "force_max_by_motor_n": np.max(
                arrays["stage_a_motor_force_postclip_n"], axis=0
            ).tolist(),
            "speed_min_by_motor": np.min(arrays["stage_a_motor_speed_postclip"], axis=0).tolist(),
            "speed_max_by_motor": np.max(arrays["stage_a_motor_speed_postclip"], axis=0).tolist(),
        },
        "effort": float(loss_terms["effort"]["raw"]),
        "smoothness": float(loss_terms["smoothness"]["raw"]),
        "finite": bool(all(np.all(np.isfinite(value)) for value in arrays.values())),
    }


def _full_rollout_gate(
    specs: tuple[EpisodeSpec, ...], outputs: Any, variant: str, integral_limits: np.ndarray
) -> list[dict[str, Any]]:
    """Apply full six-second finite/ground/zero-thrust and class clip gates."""
    trace = outputs["trace"]
    diagnostics = outputs["diagnostic"]
    pos_err_i = np.asarray(outputs["pos_err_i"])
    position = np.asarray(trace.pos)
    stage_a = diagnostics["stage_a"]
    stage_b = diagnostics["stage_b"]
    records = []
    for world, spec in enumerate(specs):
        finite = all(
            np.all(np.isfinite(np.asarray(value)[:, :, world]))
            for value in jax.tree.leaves(diagnostics)
            if np.issubdtype(np.asarray(value).dtype, np.number)
        ) and all(
            np.all(np.isfinite(np.asarray(value)[:, world]))
            for value in jax.tree.leaves(trace)
            if np.issubdtype(np.asarray(value).dtype, np.number)
        )
        ground = bool(np.any(position[:, world, 0, 2] <= -0.001))
        zero_thrust = bool(
            np.any(np.asarray(stage_a["collective_force_request_n"])[:, :, world, 0] <= 0.0)
        )
        interval_score = slice(spec.score_start, spec.score_stop)
        stage_a_motor = np.asarray(stage_a["motor_pwm_any_clip_mask"])[interval_score, :, world, 0]
        stage_a_torque = np.asarray(stage_a["torque_any_clip_mask"])[interval_score, :, world, 0]
        stage_b_motor = np.asarray(stage_b["motor_force_any_clip_mask"])[
            interval_score, :, world, 0
        ]
        score_integral = pos_err_i[interval_score, world, 0]
        integral_hit = bool(
            np.any(
                np.abs(score_integral)
                >= np.asarray(integral_limits) * (1.0 - INTEGRAL_DETECTION_THRESHOLD)
            )
        )
        motor_summary = mask_summary(
            stage_a_motor.reshape(-1, 4),
            start_time_s=spec.warmup_s,
            frequency_hz=SIMULATION_FREQUENCY_HZ,
            channel_name="motor_index",
        )
        torque_count = int(stage_a_torque.sum())
        stage_b_count = int(stage_b_motor.sum())
        if not finite:
            raise RobustContractError(f"{variant}/{spec.episode_id} has nonfinite rollout data")
        if ground:
            raise RobustContractError(f"{variant}/{spec.episode_id} activates ground/floor gate")
        if zero_thrust:
            raise RobustContractError(f"{variant}/{spec.episode_id} activates zero thrust")
        if spec.motion_class in {MotionClass.SOFT, MotionClass.NOMINAL}:
            if motor_summary["count"] or torque_count or stage_b_count:
                raise RobustContractError(
                    f"{variant}/{spec.episode_id} violates soft/nominal zero-clip gate"
                )
        if spec.motion_class is MotionClass.DYNAMIC:
            if stage_b_count:
                raise RobustContractError(
                    f"{variant}/{spec.episode_id} has forbidden dynamic Stage-B clipping"
                )
            if motor_summary["fraction"] > 0.0025 or motor_summary["longest_duration_s"] > 0.02:
                raise RobustContractError(
                    f"{variant}/{spec.episode_id} exceeds dynamic Stage-A clip gate"
                )
        records.append(
            {
                "episode_id": spec.episode_id,
                "finite": finite,
                "ground_or_floor_activation": ground,
                "zero_thrust_activation": zero_thrust,
                "stage_a_motor_clip_count_scored": motor_summary["count"],
                "stage_a_motor_clip_fraction_scored": motor_summary["fraction"],
                "stage_a_motor_longest_interval_s_scored": motor_summary["longest_duration_s"],
                "stage_a_torque_clip_count_scored": torque_count,
                "stage_b_additional_clip_count_scored": stage_b_count,
                "integral_clip_detection_scored": integral_hit,
                "integral_contact_interpretation": (
                    "diagnostic_mass_mismatch_contact_not_feasibility"
                    if variant == "repository_mismatch" and integral_hit
                    else "descriptive_contact_not_candidate_pass"
                    if integral_hit
                    else "no_contact"
                ),
                "candidate_gate": (
                    "descriptive_only_not_candidate_pass"
                    if spec.motion_class is MotionClass.NEAR_LIMIT
                    else "technical_gate_pass"
                ),
            }
        )
    return records


def differentiated_score_objective(
    raw_gains: Array,
    initial_data: SimData,
    commands: Array,
    reference: Trajectory,
    score_starts: Array,
    step_fn: Callable[[SimData, int], SimData],
) -> Array:
    """The exact Stage-1-default raw-gain benchmark/evaluation path."""
    data = apply_raw_gains(initial_data, raw_gains, 1)
    _, trace = rollout_state_commands(data, commands, step_fn, STEPS_PER_CONTROL)
    gathered_trace = jax.tree.map(
        lambda value: _gather_per_world(value, score_starts, SCORE_INTERVALS), trace
    )
    gathered_reference = jax.tree.map(
        lambda value: _gather_per_world(value, score_starts, SCORE_INTERVALS), reference
    )
    per_case_loss, _ = tracking_loss_per_case(
        gathered_trace,
        gathered_reference,
        hover_rotor_velocity(initial_data),
        rotor_velocity_limits(initial_data),
        TrackingLossConfig(),
    )
    return jnp.mean(per_case_loss)


def _parameter_fingerprint(data: SimData) -> str:
    payload = []
    for path, value in _leaf_paths(
        {
            "dynamics": data.params,
            "state": data.controls.state.params,
            "attitude": data.controls.attitude.params,
            "force_torque": data.controls.force_torque.params,
        }
    ):
        payload.append((path, array_digest(value)))
    return sha256_bytes(canonical_json_bytes(payload))


def _variant_evaluation(
    sim: Sim, initial_data: SimData, inputs: EvaluationInputs, variant: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step_fn = sim.build_step_fn()
    instrumented = make_instrumented_rollout(sim)
    final, boundaries, outputs = instrumented(initial_data, inputs.commands)
    jax.block_until_ready((final, boundaries, outputs))
    if not _numeric_leaves_finite((final, boundaries, outputs)):
        raise RobustContractError(f"{variant} instrumented rollout contains nonfinite data")
    production_fn = jax.jit(
        lambda data, commands: rollout_state_commands(data, commands, step_fn, STEPS_PER_CONTROL)
    )
    production_final, production_trace = production_fn(initial_data, inputs.commands)
    jax.block_until_ready((production_final, production_trace))
    require_pytree_equal(f"{variant} final carry", final, production_final)
    _trace_identity(outputs["trace"], production_trace, variant)
    carry_checks = carry_replay_checks(initial_data, inputs.commands, step_fn, boundaries)
    reconstruction = validate_diagnostic_identity(outputs)
    integral_limits = np.asarray(initial_data.controls.state.params["int_err_max"])
    technical_records = _full_rollout_gate(inputs.specs, outputs, variant, integral_limits)

    score_trace_value = scored_trace(outputs["trace"], inputs.specs)
    score_reference_value = scored_reference(inputs.reference, inputs.specs)
    score_pos_err_i = scored_interval_value(outputs["pos_err_i"], inputs.specs)
    score_diagnostic = jax.tree.map(
        lambda value: scored_diagnostic_value(value, inputs.specs), outputs["diagnostic"]
    )
    loss_config = TrackingLossConfig()
    per_case_loss, _ = tracking_loss_per_case(
        score_trace_value,
        score_reference_value,
        hover_rotor_velocity(initial_data),
        rotor_velocity_limits(initial_data),
        loss_config,
    )
    loss_terms = tracking_loss_terms_per_case(
        score_trace_value, score_reference_value, hover_rotor_velocity(initial_data), loss_config
    )
    raw = raw_from_data(initial_data, 1)
    score_starts = jnp.asarray([spec.score_start for spec in inputs.specs], dtype=jnp.int32)
    value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda value: differentiated_score_objective(
                value, initial_data, inputs.commands, inputs.reference, score_starts, step_fn
            )
        )
    )
    differentiated_loss, gradient = value_and_grad(raw)
    jax.block_until_ready((differentiated_loss, gradient))
    if not np.all(np.isfinite(np.asarray(gradient))) or not np.isfinite(float(differentiated_loss)):
        raise RobustContractError(f"{variant} has nonfinite differentiated loss/gradient")
    mean_loss = float(np.mean(np.asarray(per_case_loss)))
    reconstruction_check(
        f"{variant} differentiated loss versus diagnostic rollout loss",
        differentiated_loss,
        mean_loss,
        1.0,
    )

    episodes = []
    for world, spec in enumerate(inputs.specs):
        arrays = _episode_arrays(
            world, score_trace_value, score_reference_value, score_pos_err_i, score_diagnostic
        )
        term_records, term_sum = _loss_term_records(loss_terms, world)
        loss_total = float(np.asarray(per_case_loss)[world])
        if not math.isclose(term_sum, loss_total, rel_tol=0.0, abs_tol=2.0e-6):
            raise RobustContractError(f"{variant}/{spec.episode_id} Loss-v1 arithmetic changed")
        summary = summary_from_episode_arrays(
            arrays,
            start_time_s=spec.warmup_s,
            integral_limits=integral_limits,
            loss_terms=term_records,
            loss_total=loss_total,
        )
        technical = technical_records[world]
        episodes.append(
            {
                "episode_id": spec.episode_id,
                "pair_id": spec.pair_id,
                "split": spec.split.value,
                "motion_class": spec.motion_class.value,
                "seed": spec.seed,
                "mass_variant": variant,
                "controller_mass_kg": MASS_VARIANTS[variant],
                "dynamics_mass_kg": DYNAMICS_MASS_KG,
                "window": {
                    "warmup_s": spec.warmup_s,
                    "score_start_interval": spec.score_start,
                    "score_stop_interval_exclusive": spec.score_stop,
                    "score_duration_s": SCORE_DURATION_S,
                    "interval_semantics": "half-open [warmup_s,warmup_s+2.0s)",
                },
                "arrays": _json_array_section(arrays),
                "summary": summary,
                "mechanism_audit": mechanism_episode_section(world, spec, inputs, outputs),
                "technical_gates": technical,
            }
        )
    return (
        {
            "mass_variant": variant,
            "controller_mass_kg": MASS_VARIANTS[variant],
            "dynamics_mass_kg": DYNAMICS_MASS_KG,
            "episodes": episodes,
            "carry_continuity": carry_checks,
            "diagnostic_reconstruction": reconstruction,
            "differentiation": {
                "gain_stage": 1,
                "raw_default_gain_vector": np.asarray(raw).tolist(),
                "mean_loss": float(differentiated_loss),
                "gradient": np.asarray(gradient).tolist(),
                "all_finite": True,
                "optimizer_updates": 0,
            },
        },
        technical_records,
    )


def _summary_vector(episode: dict[str, Any]) -> dict[str, float]:
    summary = episode["summary"]
    values = {
        "loss_v1": float(summary["loss_v1"]["reported_total"]),
        **{name: float(value) for name, value in summary["tracking"].items()},
        "minimum_normalized_force_reserve": float(
            summary["motor_reserve"]["minimum_normalized_force_reserve"]
        ),
        "minimum_normalized_speed_reserve": float(
            summary["motor_reserve"]["minimum_normalized_speed_reserve"]
        ),
        "stage_a_torque_clip_fraction": float(summary["clipping"]["stage_a_torque"]["fraction"]),
        "stage_a_motor_clip_fraction": float(summary["clipping"]["stage_a_motor"]["fraction"]),
        "stage_b_additional_clip_fraction": float(
            summary["clipping"]["stage_b_additional"]["fraction"]
        ),
        "integral_max_fraction": float(max(summary["integral"]["max_fraction_of_limit_by_axis"])),
        "integral_contact_count": float(summary["integral"]["clip_detection"]["count"]),
        "integral_contact_fraction": float(summary["integral"]["clip_detection"]["fraction"]),
        "effort": float(summary["effort"]),
        "smoothness": float(summary["smoothness"]),
    }
    for stage_name in ("stage_a", "stage_b_additional"):
        for record in summary["wrench"][stage_name]["by_axis"]:
            values[f"{stage_name}_{record['axis']}_maximum_absolute_distortion"] = float(
                record["maximum_absolute_distortion"]
            )
    return values


def _integral_contact_aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stored per-episode integral contacts without dropping individual records."""
    episode_records = []
    for episode in episodes:
        integral = episode["summary"]["integral"]
        tracking = episode["summary"]["tracking"]
        clipping = episode["summary"]["clipping"]
        episode_records.append(
            {
                "episode_id": episode["episode_id"],
                "pair_id": episode["pair_id"],
                "split": episode["split"],
                "motion_class": episode["motion_class"],
                "status": integral["contact_status"],
                "hit_count": integral["clip_detection"]["count"],
                "sample_count": integral["clip_detection"]["sample_count"],
                "hit_fraction": integral["clip_detection"]["fraction"],
                "total_duration_s": (integral["clip_detection"]["count"] / CONTROL_FREQUENCY_HZ),
                "by_axis": integral["contact_by_axis"],
                "maximum_absolute_state_by_axis": integral["max_abs_by_axis"],
                "terminal_state_by_axis": integral["end_by_axis"],
                "z_error": {
                    "rmse_m": tracking["position_rmse_z_m"],
                    "terminal_signed_m": tracking["position_terminal_signed_z_m"],
                    "terminal_absolute_m": tracking["position_terminal_abs_z_m"],
                    "mean_signed_m": tracking["position_mean_signed_z_m"],
                },
                "paired_clip_status": {
                    "stage_a_torque_count": clipping["stage_a_torque"]["count"],
                    "stage_a_motor_count": clipping["stage_a_motor"]["count"],
                    "stage_b_additional_count": clipping["stage_b_additional"]["count"],
                },
                "minimum_normalized_motor_reserve": episode["summary"]["motor_reserve"][
                    "minimum_normalized_force_reserve"
                ],
            }
        )
    hit_count = sum(item["hit_count"] for item in episode_records)
    sample_count = sum(item["sample_count"] for item in episode_records)
    return {
        "episode_count": len(episode_records),
        "episodes_with_contact": sum(item["status"] == "CONTACT" for item in episode_records),
        "hit_count": hit_count,
        "sample_count": sample_count,
        "hit_fraction": hit_count / sample_count,
        "total_duration_s": hit_count / CONTROL_FREQUENCY_HZ,
        "episodes": episode_records,
        "individual_episodes_retained": True,
    }


LOSS_TERM_FIELDS = ("raw", "normalization_divisor", "weight", "normalized", "weighted")


def _descriptive_aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _loss_and_trajectory_class_aggregates(
    variants: dict[str, dict[str, Any]], split: str, motion_class: str
) -> dict[str, Any]:
    """Derive explicit split/class aggregates only from retained episode summaries."""
    selected = {
        variant: [
            episode
            for episode in variants[variant]["episodes"]
            if episode["split"] == split and episode["motion_class"] == motion_class
        ]
        for variant in MASS_VARIANTS
    }
    if any(len(episodes) != 2 for episodes in selected.values()):
        raise RobustContractError(f"{split}/{motion_class} aggregate requires two episodes")
    term_names = tuple(sorted(selected["repository_mismatch"][0]["summary"]["loss_v1"]["terms"]))
    if term_names != ("altitude", "effort", "position", "smoothness", "terminal", "velocity"):
        raise RobustContractError("Loss-v1 term inventory changed")
    loss_terms = {}
    for term in term_names:
        variants_by_term = {}
        for variant, episodes in selected.items():
            variants_by_term[variant] = {
                field: _descriptive_aggregate(
                    [
                        float(episode["summary"]["loss_v1"]["terms"][term][field])
                        for episode in episodes
                    ]
                )
                for field in LOSS_TERM_FIELDS
            }
        variants_by_term["matched_minus_mismatch_mean"] = {
            field: variants_by_term["matched"][field]["mean"]
            - variants_by_term["repository_mismatch"][field]["mean"]
            for field in LOSS_TERM_FIELDS
        }
        loss_terms[term] = variants_by_term
    tracking_names = tuple(sorted(selected["repository_mismatch"][0]["summary"]["tracking"]))
    tracking = {}
    for name in tracking_names:
        by_variant = {
            variant: _descriptive_aggregate(
                [float(episode["summary"]["tracking"][name]) for episode in episodes]
            )
            for variant, episodes in selected.items()
        }
        by_variant["matched_minus_mismatch_mean"] = (
            by_variant["matched"]["mean"] - by_variant["repository_mismatch"]["mean"]
        )
        tracking[name] = by_variant
    return {
        "loss_v1_term_aggregates": {
            "episode_count_per_variant": 2,
            "aggregation": "mean/minimum/maximum across retained episode values",
            "paired_delta_definition": "matched mean - repository_mismatch mean",
            "fields": list(LOSS_TERM_FIELDS),
            "terms": loss_terms,
        },
        "trajectory_tracking_statistic_aggregates": {
            "episode_count_per_variant": 2,
            "aggregation": "mean/minimum/maximum across retained scored-window episode values",
            "paired_delta_definition": "matched mean - repository_mismatch mean",
            "statistics": tracking,
        },
    }


def _paired_and_class_summaries(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mismatch = {item["episode_id"]: item for item in variants["repository_mismatch"]["episodes"]}
    matched = {item["episode_id"]: item for item in variants["matched"]["episodes"]}
    if set(mismatch) != set(matched):
        raise RobustContractError("mass variants do not contain identical episode IDs")
    paired = []
    for episode_id in sorted(mismatch):
        left = mismatch[episode_id]
        right = matched[episode_id]
        for field in ("pair_id", "split", "motion_class", "seed", "window"):
            if left[field] != right[field]:
                raise RobustContractError(f"mass-pair field changed: {episode_id}/{field}")
        left_vector = _summary_vector(left)
        right_vector = _summary_vector(right)
        paired.append(
            {
                "episode_id": episode_id,
                "pair_id": left["pair_id"],
                "split": left["split"],
                "motion_class": left["motion_class"],
                "definition": "matched - repository_mismatch",
                "deltas": {
                    name: right_vector[name] - left_vector[name] for name in sorted(left_vector)
                },
            }
        )
    per_class = {}
    for split in (Split.TRAIN.value, Split.VALIDATION.value):
        per_class[split] = {}
        for motion_class in MotionClass:
            selected = [
                item
                for item in paired
                if item["split"] == split and item["motion_class"] == motion_class.value
            ]
            if len(selected) != 2:
                raise RobustContractError(f"{split}/{motion_class.value} pair count is not two")
            names = selected[0]["deltas"]
            class_deltas = {
                name: float(np.mean([item["deltas"][name] for item in selected])) for name in names
            }
            variant_means = {}
            for variant in MASS_VARIANTS:
                episodes = [
                    item
                    for item in variants[variant]["episodes"]
                    if item["split"] == split and item["motion_class"] == motion_class.value
                ]
                vectors = [_summary_vector(item) for item in episodes]
                variant_means[variant] = {
                    name: float(np.mean([vector[name] for vector in vectors]))
                    for name in vectors[0]
                }
            explicit_aggregates = _loss_and_trajectory_class_aggregates(
                variants, split, motion_class.value
            )
            per_class[split][motion_class.value] = {
                "episode_count": 2,
                "variant_means": variant_means,
                "delta_definition": "matched - repository_mismatch",
                "mean_deltas": class_deltas,
                "integral_contact": {
                    variant: _integral_contact_aggregate(
                        [
                            item
                            for item in variants[variant]["episodes"]
                            if item["split"] == split and item["motion_class"] == motion_class.value
                        ]
                    )
                    for variant in MASS_VARIANTS
                },
                "aggregation_boundary": (
                    "separate_descriptive_near_limit_stratum"
                    if motion_class is MotionClass.NEAR_LIMIT
                    else "separate_class_report_no_cross_class_training_objective"
                ),
                **explicit_aggregates,
            }
    variant_findings = {}
    for variant in MASS_VARIANTS:
        episodes = variants[variant]["episodes"]
        z_rmse = [episode["summary"]["tracking"]["position_rmse_z_m"] for episode in episodes]
        negative_z_contact = sum(
            episode["summary"]["integral"]["contact_by_axis"][2]["lower"]["count"]
            for episode in episodes
        )
        positive_z_contact = sum(
            episode["summary"]["integral"]["contact_by_axis"][2]["upper"]["count"]
            for episode in episodes
        )
        variant_findings[variant] = {
            "episode_count": len(episodes),
            "mean_z_rmse_m": float(np.mean(z_rmse)),
            "negative_z_integral_contact_count": negative_z_contact,
            "positive_z_integral_contact_count": positive_z_contact,
        }
    mismatch_finding = variant_findings["repository_mismatch"]
    matched_finding = variant_findings["matched"]
    negative_finding = {
        "status": "NEGATIVE_PHYSICAL_VALUE_MATCH_FINDING",
        "variants": variant_findings,
        "matched_minus_mismatch": {
            "mean_z_rmse_m": matched_finding["mean_z_rmse_m"] - mismatch_finding["mean_z_rmse_m"],
            "negative_z_integral_contact_count": matched_finding[
                "negative_z_integral_contact_count"
            ]
            - mismatch_finding["negative_z_integral_contact_count"],
        },
        "interpretation": (
            "Physical-value matching increases negative z-integral boundary contact and "
            "worsens mean Z-RMSE in this fixed simulation/controller conversion chain."
        ),
        "not_a_claim_of": [
            "improvement",
            "candidate_pass",
            "causal_isolation",
            "physical_mass_estimation",
            "firmware_transfer",
            "hardware_or_flight_readiness",
        ],
    }
    if (
        negative_finding["matched_minus_mismatch"]["mean_z_rmse_m"] <= 0.0
        or negative_finding["matched_minus_mismatch"]["negative_z_integral_contact_count"] <= 0
    ):
        raise RobustContractError("required negative physical-value-match finding changed")
    return {
        "paired_episode_deltas": paired,
        "per_split_class": per_class,
        "negative_physical_value_match_finding": negative_finding,
    }


def episode_contract_payload(
    constructed: tuple[tuple[EpisodeSpec, ReferenceCandidate], ...],
) -> dict[str, Any]:
    """Build the complete reference/window/seed contract, including stored parent arrays."""
    episodes = []
    for spec, candidate in constructed:
        arrays = {
            "time_s": np.asarray(candidate.trajectory.time),
            "position_m": np.asarray(candidate.trajectory.pos),
            "velocity_m_s": np.asarray(candidate.trajectory.vel),
            "acceleration_m_s2": np.asarray(candidate.trajectory.acc),
            "jerk_m_s3": np.asarray(candidate.jerk),
            "yaw_rad": np.asarray(candidate.trajectory.yaw),
            "yaw_rate_rad_s": np.asarray(candidate.trajectory.yaw_rate),
        }
        episodes.append(
            {
                "episode_id": spec.episode_id,
                "pair_id": spec.pair_id,
                "split": spec.split.value,
                "motion_class": spec.motion_class.value,
                "ordinal": spec.ordinal,
                "seed": spec.seed,
                "warmup_s": spec.warmup_s,
                "score_start_interval": spec.score_start,
                "score_stop_interval_exclusive": spec.score_stop,
                "accepted_attempt_index": candidate.attempt,
                "executed_attempt_count": len(candidate.attempts),
                "attempts": list(candidate.attempts),
                "score_statistics": candidate.score_statistics,
                "parent_statistics": candidate.parent_statistics,
                "arrays": _json_array_section(arrays),
            }
        )
    reference_aggregates = {}
    for split in (Split.TRAIN.value, Split.VALIDATION.value):
        reference_aggregates[split] = {}
        for motion_class in MotionClass:
            selected = [
                episode
                for episode in episodes
                if episode["split"] == split and episode["motion_class"] == motion_class.value
            ]
            if len(selected) != 2:
                raise RobustContractError(
                    f"{split}/{motion_class.value} reference aggregate requires two episodes"
                )
            scopes = {}
            for source_name, output_name in (
                ("parent_statistics", "parent_support_8s"),
                ("score_statistics", "scored_reference_window_2s"),
            ):
                numeric_maps = [
                    dict(_numeric_projection_walk(item[source_name], item[source_name]))
                    for item in selected
                ]
                if set(numeric_maps[0]) != set(numeric_maps[1]):
                    raise RobustContractError("reference trajectory statistic paths changed")
                scopes[output_name] = {
                    "episode_count": 2,
                    "statistics_by_json_pointer": {
                        path: _descriptive_aggregate(
                            [float(numeric_map[path]) for numeric_map in numeric_maps]
                        )
                        for path in sorted(numeric_maps[0])
                    },
                    "all_finite": all(item[source_name]["all_finite"] for item in selected),
                }
            reference_aggregates[split][motion_class.value] = scopes
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "platform": PLATFORM,
            "parent_support_s": PARENT_DURATION_S,
            "parent_control_intervals": PARENT_INTERVALS,
            "parent_reference_samples": PARENT_INTERVALS + 1,
            "rollout_s": ROLLOUT_DURATION_S,
            "rollout_control_intervals": ROLLOUT_INTERVALS,
            "score_s": SCORE_DURATION_S,
            "score_control_intervals": SCORE_INTERVALS,
            "window_semantics": "half-open [warmup_s,warmup_s+2.0s)",
            "warmup_rule": "2.0 + (episode_seed mod 3)",
            "full_rollout_before_scoring": True,
            "carry_reset_at_window": False,
            "gradient_flows_through_warmup": True,
            "position_construction": (
                "Class-scaled bounded three-axis Fourier parent plus the D-047-disclosed fixed "
                "positive-z central climb, both using the same C3 envelope, then fixed center."
            ),
            "vertical_excitation": vertical_excitation_machine_contract(),
            "duration_and_measurement_scope": duration_scope_contract(),
            "yaw_construction": "independent bounded Fourier yaw with matching analytic yaw_rate",
            "yaw_consumption": {
                "absolute_yaw_consumed_by_state2attitude": True,
                "yaw_rate_stored_in_13_field": True,
                "yaw_rate_feedforward_consumed_by_current_state_stage": False,
            },
            "class_contracts": {
                key.value: {
                    "lower_open": value.lower_open,
                    "upper_closed": value.upper_closed,
                    "target_usage": value.target_usage,
                    "max_abs_yaw_rad": value.max_abs_yaw_rad,
                }
                for key, value in CLASS_CONTRACTS.items()
            },
            "workspace_min_m": WORKSPACE_MIN_M.tolist(),
            "workspace_max_m": WORKSPACE_MAX_M.tolist(),
            "global_limits": GLOBAL_LIMITS,
            "maximum_attempts": MAX_ATTEMPTS,
            "test_constructed_or_loaded": False,
        },
        "episodes": episodes,
        "per_split_class_reference_trajectory_statistics": reference_aggregates,
    }


def split_manifest_payload(
    split: Split, constructed: tuple[tuple[EpisodeSpec, ReferenceCandidate], ...]
) -> dict[str, Any]:
    """Build one fixed Train or Validation manifest without constructing Test."""
    selected = [(spec, candidate) for spec, candidate in constructed if spec.split is split]
    classes = {}
    for motion_class in MotionClass:
        class_items = [item for item in selected if item[0].motion_class is motion_class]
        executed = sum(len(candidate.attempts) for _, candidate in class_items)
        accepted = len(class_items)
        classes[motion_class.value] = {
            "accepted_parent_count": accepted,
            "executed_attempt_count": executed,
            "acceptance_rate": accepted / executed,
            "episode_ids": [spec.episode_id for spec, _ in class_items],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": f"day25-{split.value}-cf21b-robust-foundation",
        "split": split.value,
        "test_manifest": None,
        "episodes": [
            {
                "episode_id": spec.episode_id,
                "pair_id": spec.pair_id,
                "seed": spec.seed,
                "motion_class": spec.motion_class.value,
                "ordinal": spec.ordinal,
                "warmup_s": spec.warmup_s,
                "score_start_interval": spec.score_start,
                "score_stop_interval_exclusive": spec.score_stop,
                "accepted_attempt_index": candidate.attempt,
                "parent_array_digests": candidate.array_digests,
            }
            for spec, candidate in selected
        ],
        "acceptance_by_class": classes,
    }


def validate_report_summaries(report: dict[str, Any]) -> None:
    """Recompute all per-episode summaries from the arrays embedded in the report."""
    for variant in report["mass_variants"].values():
        for episode in variant["episodes"]:
            contracts = episode["arrays"]["contracts"]
            arrays = {
                name: np.asarray(value, dtype=np.dtype(contracts[name]["dtype"]))
                for name, value in episode["arrays"]["values"].items()
            }
            summary = episode["summary"]
            rebuilt = summary_from_episode_arrays(
                arrays,
                start_time_s=episode["window"]["warmup_s"],
                integral_limits=np.asarray(summary["integral"]["limits"]),
                loss_terms=summary["loss_v1"]["terms"],
                loss_total=summary["loss_v1"]["reported_total"],
            )
            if canonical_json_bytes(rebuilt) != canonical_json_bytes(summary):
                raise RobustContractError(
                    f"stored-array summary recomputation failed for {episode['episode_id']}"
                )


def build_foundation_evidence() -> dict[str, Any]:
    """Run all reference, mass-isolation, replay, finite, and technical gates."""
    constructed = construct_all_episodes()
    inputs = stack_evaluation_inputs(constructed)
    sim = build_simulation(len(inputs.specs))
    base = initialize_inputs(sim.data, inputs)
    controller_mass = np.asarray(base.controls.state.params["mass"])
    if not np.array_equal(
        controller_mass, np.asarray(MISMATCH_CONTROLLER_MASS_KG, dtype=controller_mass.dtype)
    ):
        raise RobustContractError("repository controller mass default changed")
    mismatch = with_controller_mass(base, MISMATCH_CONTROLLER_MASS_KG)
    matched = with_controller_mass(base, MATCHED_CONTROLLER_MASS_KG)
    mass_isolation = mass_only_pytree_diff(mismatch, matched)
    variants = {}
    for name, data in (("repository_mismatch", mismatch), ("matched", matched)):
        variant, _ = _variant_evaluation(sim, data, inputs, name)
        variants[name] = variant
    paired = _paired_and_class_summaries(variants)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_ROBUST_FOUNDATION",
        "scope": {
            "simulation_only": True,
            "optimizer_updates": 0,
            "candidate_selected": False,
            "hardware_or_firmware_validation": False,
            "test_access": False,
            "near_limit": "separate descriptive boundary stratum",
            "decision_d045": (
                "prospective integral contact in both fixed mass variants is measured baseline "
                "evidence, never a technical pass or stop"
            ),
            "gr_g2_001_status": "BLOCKED_WITHHELD_REJECTED_UNCHANGED",
            "gr_g2_002_status": "BLOCKED_WITHHELD_REJECTED_UNCHANGED",
            "correction_d047": (
                "Sole transparency correction: disclose the unchanged positive-z trajectory "
                "component, add derived aggregates, and correct duration/measurement scopes."
            ),
            "duration_and_measurement_scope": duration_scope_contract(),
        },
        "configuration": {
            "platform": PLATFORM,
            "dynamics": Dynamics.first_principles.value,
            "integrator": Integrator.euler.value,
            "simulation_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "state_control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "attitude_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "force_torque_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "n_drones_per_world": 1,
            "device": "cpu",
            "jax_x64": False,
            "dynamics_mass_kg": DYNAMICS_MASS_KG,
            "controller_masses_kg": MASS_VARIANTS,
            "disturbance_delay_noise_wind_initial_randomization": False,
            "loss": "unchanged Loss v1",
            "gain_stage": 1,
            "gain_defaults_changed": False,
            "registry_fingerprint": registry_fingerprint(),
            "parameter_fingerprints": {
                "repository_mismatch": _parameter_fingerprint(mismatch),
                "matched": _parameter_fingerprint(matched),
            },
            "source_base_commit": SOURCE_BASE_COMMIT,
        },
        "input_identity": {
            "commands": array_digest(inputs.commands),
            "reference_position": array_digest(inputs.reference.pos),
            "reference_velocity": array_digest(inputs.reference.vel),
            "window_starts": [spec.score_start for spec in inputs.specs],
            "episode_ids": [spec.episode_id for spec in inputs.specs],
        },
        "mass_only_pytree_diff": mass_isolation,
        "mass_variants": variants,
        "mass_ablation": paired,
        "trajectory_contract": {
            "vertical_excitation": vertical_excitation_machine_contract(),
            "composition_before_center": True,
            "numerical_construction_changed_by_d047": False,
        },
        "mechanism_audit": {
            "measured": (
                "Per-episode early-transient and scored arrays record controller mass, "
                "collective target-thrust-equivalent request, realized wrench, Z error, "
                "Z integral, motor command, and reserve at existing boundaries."
            ),
            "code_path_inference": (
                "state2attitude uses mass * (setpoint_acc - gravity_vec) + feedback, then "
                "mass_thrust=132000 and nonlinear PWM-to-force conversion. This explains the "
                "implementation path but does not isolate causality."
            ),
            "transfer_boundary": (
                "A later optimized controller-mass value cannot be treated as physical mass, "
                "firmware configuration, or flight value before a separate semantics/transfer gate."
            ),
        },
    }
    validate_report_summaries(report)
    return {
        "report": report,
        "episode_contract": episode_contract_payload(constructed),
        "train_manifest": split_manifest_payload(Split.TRAIN, constructed),
        "validation_manifest": split_manifest_payload(Split.VALIDATION, constructed),
    }


def benchmark_block_inputs(
    world_count: int,
) -> tuple[Sim, SimData, EvaluationInputs, Array, dict[str, Any]]:
    """Build exact 4-world feasible-block replications for the differentiated path."""
    if world_count not in {4, 16, 32}:
        raise RobustContractError("benchmark world count must be one of 4, 16, or 32")
    constructed = construct_all_episodes()

    def select(
        split: Split, motion_class: MotionClass, ordinal: int
    ) -> tuple[EpisodeSpec, ReferenceCandidate]:
        matches = [
            item
            for item in constructed
            if item[0].split is split
            and item[0].motion_class is motion_class
            and item[0].ordinal == ordinal
        ]
        if len(matches) != 1:
            raise RobustContractError("benchmark block episode selection is not unique")
        return matches[0]

    block = (
        select(Split.TRAIN, MotionClass.SOFT, 0),
        select(Split.TRAIN, MotionClass.NOMINAL, 0),
        select(Split.TRAIN, MotionClass.NOMINAL, 1),
        select(Split.TRAIN, MotionClass.DYNAMIC, 0),
    )
    repeated = block * (world_count // 4)
    inputs = stack_evaluation_inputs(repeated)
    sim = build_simulation(world_count, rng_seed=2532 + world_count)
    initial = with_controller_mass(initialize_inputs(sim.data, inputs), MISMATCH_CONTROLLER_MASS_KG)
    raw = raw_from_data(initial, 1)
    commands = np.asarray(inputs.commands)
    reference_position = np.asarray(inputs.reference.pos)
    starts = np.asarray([spec.score_start for spec in inputs.specs], dtype=np.int32)
    for repeat in range(world_count // 4):
        block_slice = slice(repeat * 4, (repeat + 1) * 4)
        if not np.array_equal(commands[:, block_slice], commands[:, :4]):
            raise RobustContractError("benchmark command block replication is not bit-identical")
        if not np.array_equal(reference_position[:, block_slice], reference_position[:, :4]):
            raise RobustContractError("benchmark reference block replication is not bit-identical")
        if not np.array_equal(starts[block_slice], starts[:4]):
            raise RobustContractError("benchmark window block replication is not bit-identical")
    metadata = {
        "world_count": world_count,
        "block_repetitions": world_count // 4,
        "block_episode_ids": [spec.episode_id for spec, _ in block],
        "world_episode_ids": [spec.episode_id for spec in inputs.specs],
        "input_digests": {
            "commands": array_digest(inputs.commands),
            "reference_position": array_digest(inputs.reference.pos),
            "reference_velocity": array_digest(inputs.reference.vel),
            "score_starts": array_digest(starts),
            "initial_states": array_digest(initial.states.pos),
            "raw_default_stage1_gains": array_digest(raw),
            "four_world_command_block": array_digest(inputs.commands[:, :4]),
            "four_world_reference_block": array_digest(inputs.reference.pos[:, :4]),
        },
        "replication_array_equal": True,
        "near_limit_included": False,
        "mass_variant": "repository_mismatch",
    }
    return sim, initial, inputs, raw, metadata


def benchmark_value_and_grad(
    sim: Sim, initial: SimData, inputs: EvaluationInputs
) -> Callable[[Array], tuple[Array, Array]]:
    """Return the JIT-ready exact benchmark value-and-gradient function."""
    starts = jnp.asarray([spec.score_start for spec in inputs.specs], dtype=jnp.int32)
    step_fn = sim.build_step_fn()
    return jax.value_and_grad(
        lambda raw: differentiated_score_objective(
            raw, initial, inputs.commands, inputs.reference, starts, step_fn
        )
    )


def benchmark_consistency_check(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen replicated-block loss/gradient consistency bound."""
    records = sorted(results, key=lambda item: item["world_count"])
    if [record["world_count"] for record in records] != [4, 16, 32]:
        raise RobustContractError("benchmark consistency requires complete 4/16/32 results")
    baseline_loss = float(records[0]["loss"])
    baseline_gradient = np.asarray(records[0]["gradient"], dtype=np.float64)
    comparisons = []
    for record in records[1:]:
        loss = float(record["loss"])
        gradient = np.asarray(record["gradient"], dtype=np.float64)
        loss_error = abs(loss - baseline_loss)
        loss_bound = (
            BENCHMARK_CONSISTENCY_FACTOR * FLOAT32_EPS * max(1.0e-3, abs(loss), abs(baseline_loss))
        )
        gradient_error = np.abs(gradient - baseline_gradient)
        gradient_bound = (
            BENCHMARK_CONSISTENCY_FACTOR
            * FLOAT32_EPS
            * np.maximum(1.0e-3, np.maximum(np.abs(gradient), np.abs(baseline_gradient)))
        )
        if loss_error > loss_bound or np.any(gradient_error > gradient_bound):
            raise RobustContractError(
                f"benchmark replicated-block consistency failed at {record['world_count']} worlds"
            )
        comparisons.append(
            {
                "world_count": record["world_count"],
                "loss_absolute_error_vs_4": loss_error,
                "loss_allowed_error": loss_bound,
                "gradient_absolute_error_vs_4": gradient_error.tolist(),
                "gradient_allowed_error": gradient_bound.tolist(),
                "passed": True,
            }
        )
    return {
        "status": "PASS_REPLICATED_BLOCK_CONSISTENCY",
        "factor_times_float32_eps": BENCHMARK_CONSISTENCY_FACTOR * FLOAT32_EPS,
        "rtol": 0.0,
        "comparisons": comparisons,
    }
