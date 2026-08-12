"""Prospective G3.4 parameter eligibility audit for the cf21B_500 simulation.

The canonical controller remains untouched.  This experiment creates an immutable
``SimData`` copy whose sole relaxed leaf is the state-controller ``mass_thrust``
value, represented as Float32.  Continuous diagnostics and the separately rounded
integer response are deliberately different APIs and different report sections.
No function in this module initializes or updates an optimizer.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from mujoco.mjx import warp as mjx_warp

import crazyflow.sim.functional as F
from crazyflow.control.mellinger.control import state2attitude
from crazyflow.control.mellinger.research import robust_evaluation as robust
from crazyflow.control.mellinger.tracking import (
    TrackingLossConfig,
    hover_rotor_velocity,
    rotor_velocity_limits,
    tracking_loss_per_case,
    tracking_loss_terms_per_case,
)
from crazyflow.trajectory import Trajectory, state_commands

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData


SCHEMA_VERSION = "crazyflow.g3_identifiability_freeze_v2.v4"
SOURCE_BASE_COMMIT = "77e24a4768d77a5aec19c18cad37423848051f0f"
G3_003_SOURCE_BASE_COMMIT = "5f515f00d56c796bae780145e36a1987af080ec5"
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
MAX_ATTEMPTS = 64
DYNAMICS_MASS_KG = 0.04338
CANONICAL_MASS_THRUST = 132000
AD_FD_STEP = 0.01
EFFECT_STEP = 0.05
AD_FD_ABSOLUTE_TOLERANCE = 2.0e-4
AD_FD_RELATIVE_TOLERANCE = 0.05
LOSS_TERM_NAMES = ("position", "velocity", "effort", "smoothness", "terminal", "altitude")
PARAMETER_NAMES = ("kp_xy", "kp_z", "kd_xy", "kd_z", "ki_z", "mass", "mass_thrust")
FOCUSED_PARENT_IDS = ("train-soft-hold-8301001", "validation-nominal-descent-8401103")
WRENCH_PLATFORM_SCALE = robust.WRENCH_PLATFORM_SCALE
G3_003_DIAGNOSIS_VERSION = "gr-g3-003-v1"
G3_004_SMOKE_VERSION = "gr-g3-004-reverse-smoke-v1"
G3_004_EXECUTION_MODE = "six-batches-of-four"
G3_004_SELECTION_DECISION = "D-073"
G3_004_SELECTION_POLICY = "prospective_fixed_after_observed_local_primary_24_oom"
G3_004_BATCH_EXECUTION_SCHEMA = "crazyflow.g3_004.six_batches_of_four.d073.v1"
G3_004_BATCH_SLICES = (
    ("train-0-4", 0, 4, "train"),
    ("train-4-8", 4, 8, "train"),
    ("train-8-12", 8, 12, "train"),
    ("validation-12-16", 12, 16, "validation"),
    ("validation-16-20", 16, 20, "validation"),
    ("validation-20-24", 20, 24, "validation"),
)
G3_004_BATCH_PARENT_COUNTS = (4, 4, 4, 4, 4, 4)
G3_004_FEASIBLE_SPLIT_DENOMINATOR = 9
NONFINITE_NEAR_LIMIT_EFFECT_REASON = "NONFINITE_NEAR_LIMIT_EFFECT_ROLLOUT"
WITHHELD_TECHNICAL_ROBUSTNESS = "WITHHELD_TECHNICAL_ROBUSTNESS"
NONFINITE_EFFECT_SCHEMA = "crazyflow.g3_004.nonfinite_near_limit_effect.d074.v1"
LOSS_ONLY_NAMES = ("loss_total",) + tuple(f"loss_contribution_{term}" for term in LOSS_TERM_NAMES)
OPTIMIZER_OBJECTIVE_NAMES = LOSS_ONLY_NAMES
OPTIMIZER_OBJECTIVE_ROLE = "OPTIMIZER_OBJECTIVE_GRADIENT_CONTRACT"
DIAGNOSTIC_AD_ROLE = "DIAGNOSTIC_AD_NOT_OPTIMIZER_CONTRACT"
WITHHELD_PARAMETER_CATEGORY = "WITHHELD"
WITHHELD_OBJECTIVE_GRADIENT_INVALID = "WITHHELD_OBJECTIVE_GRADIENT_INVALID"
G3_004_FEATURE_COUNT = 143
G3_004_DIAGNOSTIC_FEATURE_COUNT = 136
G3_004_COMPLETION_STATUS = "COMPLETE_G3_4_PARAMETER_ELIGIBILITY_AUDIT"
G3_003_DRAFT_PATHS = (
    "crazyflow/control/mellinger/research/g3_freeze_v2.py",
    "examples/jax/mellinger_g3_freeze_v2.py",
    "tests/unit/test_mellinger_g3_freeze_v2.py",
)
G3_003_INPUT_DIGESTS = {
    G3_003_DRAFT_PATHS[0]: "3e7dd009a089f8a6d2a10aaf0af27ba3579a7b8e781bb40181258bf51a868b5f",
    G3_003_DRAFT_PATHS[1]: "6192dbaab3633f6940d0851b183a32fa0d5f929e131b3f805260ee78e8d3db95",
    G3_003_DRAFT_PATHS[2]: "17ab0e57ee8ab350fa29553b50dfdadce69dad244cd6d2ce08b7302fd676f2c8",
}
G3_003_FD_TOTAL_REFERENCES = {
    FOCUSED_PARENT_IDS[0]: 0.6146990060806274,
    FOCUSED_PARENT_IDS[1]: 0.5959756374359131,
}
G3_003_REPRODUCTION_COMMAND = (
    "CORR01_CACHE_DIR=$(mktemp -d /tmp/gr-g3-003-corr01-repro-XXXXXX) && "
    'test -z "$(ls -A "$CORR01_CACHE_DIR")" && '
    "CORR01_ENV_DIR=$(mktemp -d /tmp/gr-g3-003-corr01-repro-env-XXXXXX) && "
    'mkdir -p "$CORR01_ENV_DIR/xdg" "$CORR01_ENV_DIR/matplotlib" '
    '"$CORR01_ENV_DIR/tmp" && '
    "PYTHONDONTWRITEBYTECODE=1 "
    "PYTHONPATH=/home/noah3/bachelorarbeit/worktrees/"
    "crazyflow-gradient-research-wo-gr-g3-002 "
    "JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=false "
    "JAX_COMPILATION_CACHE_DIR=$CORR01_CACHE_DIR "
    "XDG_CACHE_HOME=$CORR01_ENV_DIR/xdg "
    "MPLCONFIGDIR=$CORR01_ENV_DIR/matplotlib "
    "TMPDIR=$CORR01_ENV_DIR/tmp "
    "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false "
    "/usr/bin/timeout --signal=TERM 600s "
    "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
    "examples/jax/mellinger_g3_freeze_v2.py --diagnose-g3-003 "
    "--output-file $CORR01_ENV_DIR/diagnosis.json"
)


class G3ContractError(RuntimeError):
    """Raised immediately when a prospective contract or technical gate fails."""


class G3ResourceFallbackRequired(G3ContractError):
    """Raised only when the primary 24-world batch reaches a frozen resource boundary."""


class _G3BatchDeadlineExceeded(TimeoutError):
    """Internal signal used to enforce the frozen per-batch deadline."""


@contextmanager
def _batch_deadline(seconds: int) -> Iterable[None]:
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_deadline(_signum: int, _frame: Any) -> None:
        raise _G3BatchDeadlineExceeded

    signal.signal(signal.SIGALRM, raise_deadline)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


Split = robust.Split
MotionClass = robust.MotionClass
CLASS_CONTRACTS = robust.CLASS_CONTRACTS


class Profile(StrEnum):
    """The three balanced vertical reference profiles."""

    HOLD = "hold"
    CLIMB = "climb"
    DESCENT = "descent"


PROFILE_RATES_M_S = {Profile.HOLD: 0.0, Profile.CLIMB: 0.06, Profile.DESCENT: -0.06}
PROFILE_INDEX = {profile: index for index, profile in enumerate(Profile)}
SPLIT_INDEX = {Split.TRAIN: 0, Split.VALIDATION: 1}
CLASS_INDEX = {motion_class: index for index, motion_class in enumerate(MotionClass)}
EPISODE_SEEDS = {
    Split.TRAIN: {
        MotionClass.SOFT: (8301001, 8301002, 8301003),
        MotionClass.NOMINAL: (8301101, 8301102, 8301103),
        MotionClass.DYNAMIC: (8301201, 8301202, 8301203),
        MotionClass.NEAR_LIMIT: (8301301, 8301302, 8301303),
    },
    Split.VALIDATION: {
        MotionClass.SOFT: (8401001, 8401002, 8401003),
        MotionClass.NOMINAL: (8401101, 8401102, 8401103),
        MotionClass.DYNAMIC: (8401201, 8401202, 8401203),
        MotionClass.NEAR_LIMIT: (8401301, 8401302, 8401303),
    },
}
KEY_NAMESPACES = {
    "pair": 307,
    "split": 311,
    "class": 313,
    "profile": 317,
    "attempt": 331,
    "position": 337,
    "yaw": 347,
}


@dataclass(frozen=True)
class EpisodeSpec:
    split: Split
    motion_class: MotionClass
    profile: Profile
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
    component_contracts: tuple[dict[str, Any], ...]
    parent_digest: str


@dataclass(frozen=True)
class EvaluationInputs:
    specs: tuple[EpisodeSpec, ...]
    candidates: tuple[ReferenceCandidate, ...]
    commands: Array
    reference: Trajectory


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    parameter: str
    axes: tuple[int, ...]
    default: float
    lower: float
    upper: float
    unit: str


PARAMETER_SPECS = {
    item.name: item
    for item in (
        ParameterSpec("kp_xy", "kp", (0, 1), 0.4, 0.10, 1.20, "N/m"),
        ParameterSpec("kp_z", "kp", (2,), 1.25, 0.30, 2.50, "N/m"),
        ParameterSpec("kd_xy", "kd", (0, 1), 0.2, 0.05, 0.80, "N s/m"),
        ParameterSpec("kd_z", "kd", (2,), 0.5, 0.10, 1.20, "N s/m"),
        ParameterSpec("ki_z", "ki", (2,), 0.05, 0.025, 0.10, "N/(m s)"),
        ParameterSpec("mass", "mass", (), 0.0393, 0.035, 0.047, "kg"),
        ParameterSpec("mass_thrust", "mass_thrust", (), 132000.0, 118800.0, 145200.0, "unresolved"),
    )
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable, newline-terminated JSON bytes."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest for raw bytes."""
    return hashlib.sha256(value).hexdigest()


def array_digest(value: Any) -> dict[str, Any]:
    """Return canonical digest metadata for an array."""
    return robust.array_digest(value)


def episode_specs() -> tuple[EpisodeSpec, ...]:
    """Return the exact 24 Train/Validation × class × profile parent registry."""
    result: list[EpisodeSpec] = []
    for split in (Split.TRAIN, Split.VALIDATION):
        for motion_class in MotionClass:
            for profile, seed in zip(Profile, EPISODE_SEEDS[split][motion_class], strict=True):
                warmup_s = float(
                    2
                    + (SPLIT_INDEX[split] + CLASS_INDEX[motion_class] + PROFILE_INDEX[profile]) % 3
                )
                score_start = int(warmup_s * CONTROL_FREQUENCY_HZ)
                episode_id = f"{split.value}-{motion_class.value}-{profile.value}-{seed}"
                result.append(
                    EpisodeSpec(
                        split=split,
                        motion_class=motion_class,
                        profile=profile,
                        seed=seed,
                        episode_id=episode_id,
                        pair_id=f"pair-{motion_class.value}-{profile.value}",
                        warmup_s=warmup_s,
                        score_start=score_start,
                        score_stop=score_start + SCORE_INTERVALS,
                    )
                )
    ids = [item.episode_id for item in result]
    seeds = [item.seed for item in result]
    if len(result) != 24 or len(set(ids)) != 24 or len(set(seeds)) != 24:
        raise G3ContractError("parent registry is not exactly 24 disjoint IDs and seeds")
    if any("test" in item.episode_id for item in result):
        raise G3ContractError("Test construction is forbidden")
    return tuple(result)


def component_key(spec: EpisodeSpec, attempt: int, component: str) -> Array:
    """Fold pair, split, class, profile, attempt, and component namespaces separately."""
    if attempt < 0 or component not in {"position", "yaw"}:
        raise ValueError("invalid component-key coordinate")
    key = jax.random.key(spec.seed)
    coordinates = (
        (KEY_NAMESPACES["pair"], CLASS_INDEX[spec.motion_class] * 3 + PROFILE_INDEX[spec.profile]),
        (KEY_NAMESPACES["split"], SPLIT_INDEX[spec.split]),
        (KEY_NAMESPACES["class"], CLASS_INDEX[spec.motion_class]),
        (KEY_NAMESPACES["profile"], PROFILE_INDEX[spec.profile]),
        (KEY_NAMESPACES["attempt"], attempt),
        (KEY_NAMESPACES[component], 0),
    )
    for namespace, coordinate in coordinates:
        key = jax.random.fold_in(key, namespace)
        key = jax.random.fold_in(key, coordinate)
    return key


def _interval_indices(spec: EpisodeSpec) -> np.ndarray:
    return np.arange(spec.score_start + 1, spec.score_stop + 1, dtype=np.int64)


def _profile_components(time: Array, profile: Profile) -> tuple[Array, Array, Array, Array]:
    rate = jnp.asarray(PROFILE_RATES_M_S[profile], dtype=time.dtype)
    base_time = time - jnp.asarray(2.0, dtype=time.dtype)
    base = (
        (rate * base_time)[:, None],
        jnp.broadcast_to(rate, time.shape)[:, None],
        jnp.zeros_like(time)[:, None],
        jnp.zeros_like(time)[:, None],
    )
    return robust._enveloped(base, robust._c3_envelope(time))


def _component_contracts(
    spec: EpisodeSpec,
    trajectory: Trajectory,
    jerk: Array,
    fourier: tuple[Array, Array, Array, Array],
    profile: tuple[Array, Array, Array, Array],
) -> tuple[dict[str, Any], ...]:
    items = (
        {
            "name": "bounded_fourier_translation",
            "axis": "xyz",
            "unit": "m",
            "formula": "three harmonics with seed/attempt coefficients and phases",
            "envelope": "C3 smoothstep7: 2 s entry, 4 s unit plateau, 2 s exit",
            "time_origin_s": 0.0,
            "highest_derivative_order": 3,
            "class_dependent": True,
            "profile_dependent": False,
            "composition_position": 1,
            "array_digests": {
                name: array_digest(value)
                for name, value in zip(
                    ("displacement", "velocity", "acceleration", "jerk"), fourier, strict=True
                )
            },
        },
        {
            "name": "vertical_profile",
            "axis": "+z",
            "unit": "m",
            "formula": "rate*(time-2 s), analytic product rule through jerk",
            "rate_m_s": PROFILE_RATES_M_S[spec.profile],
            "envelope": "C3 smoothstep7: 2 s entry, 4 s unit plateau, 2 s exit",
            "time_origin_s": 2.0,
            "highest_derivative_order": 3,
            "class_dependent": False,
            "profile_dependent": True,
            "composition_position": 2,
            "array_digests": {
                name: array_digest(value)
                for name, value in zip(
                    ("displacement", "velocity", "acceleration", "jerk"), profile, strict=True
                )
            },
        },
        {
            "name": "fixed_center",
            "axis": "xyz",
            "unit": "m",
            "formula": "[0,0,0.75] added last",
            "envelope": "none",
            "time_origin_s": 0.0,
            "highest_derivative_order": 0,
            "class_dependent": False,
            "profile_dependent": False,
            "composition_position": 3,
            "array_digest": array_digest(np.asarray([0.0, 0.0, 0.75], dtype=np.float32)),
        },
        {
            "name": "bounded_absolute_yaw",
            "axis": "yaw",
            "unit": "rad",
            "formula": "bounded Fourier angle; analytic yaw rate",
            "envelope": "C3 smoothstep7: 2 s entry, 4 s unit plateau, 2 s exit",
            "time_origin_s": 0.0,
            "highest_derivative_order": 1,
            "class_dependent": True,
            "profile_dependent": False,
            "composition_position": "separate",
            "consumption": "absolute yaw consumed; yaw-rate feedforward is not claimed",
            "array_digests": {
                "yaw": array_digest(trajectory.yaw),
                "yaw_rate": array_digest(trajectory.yaw_rate),
            },
        },
    )
    combined = {
        "position": trajectory.pos,
        "velocity": trajectory.vel,
        "acceleration": trajectory.acc,
        "jerk": jerk,
    }
    if not all(np.all(np.isfinite(np.asarray(value))) for value in combined.values()):
        raise G3ContractError(f"{spec.episode_id} component arrays are nonfinite")
    return items


def _candidate_attempt(
    spec: EpisodeSpec, attempt: int
) -> tuple[Trajectory, Array, tuple[dict[str, Any], ...]]:
    time = jnp.arange(PARENT_INTERVALS + 1, dtype=jnp.float32) / CONTROL_FREQUENCY_HZ
    envelope = robust._c3_envelope(time)
    raw_fourier = robust._enveloped(
        robust._fourier_series(time, component_key(spec, attempt, "position"), 3), envelope
    )
    indices = _interval_indices(spec)
    raw_speed = np.linalg.norm(np.asarray(raw_fourier[1])[indices], axis=-1)
    if float(np.max(raw_speed)) <= 0.0:
        raise G3ContractError("degenerate deterministic Fourier translation")
    scale = (
        CLASS_CONTRACTS[spec.motion_class].target_usage
        * robust.GLOBAL_LIMITS["speed_m_s"]
        / float(np.max(raw_speed))
    )
    fourier = tuple(value * jnp.asarray(scale, dtype=time.dtype) for value in raw_fourier)
    profile = _profile_components(time, spec.profile)
    z_axis = jnp.asarray([0.0, 0.0, 1.0], dtype=time.dtype)
    displacement = fourier[0] + profile[0] * z_axis
    velocity = fourier[1] + profile[1] * z_axis
    acceleration = fourier[2] + profile[2] * z_axis
    jerk = fourier[3] + profile[3] * z_axis
    position = displacement + jnp.asarray([0.0, 0.0, 0.75], dtype=time.dtype)

    yaw_base = robust._enveloped(
        robust._fourier_series(time, component_key(spec, attempt, "yaw"), 1), envelope
    )
    raw_yaw_rate = np.abs(np.asarray(yaw_base[1])[:, 0][indices])
    if float(np.max(raw_yaw_rate)) <= 0.0:
        raise G3ContractError("degenerate deterministic yaw")
    yaw_scale = CLASS_CONTRACTS[spec.motion_class].target_usage / float(np.max(raw_yaw_rate))
    yaw = yaw_base[0][:, 0] * jnp.asarray(yaw_scale, dtype=time.dtype)
    yaw_rate = yaw_base[1][:, 0] * jnp.asarray(yaw_scale, dtype=time.dtype)
    trajectory = Trajectory(
        time=time, pos=position, vel=velocity, acc=acceleration, yaw=yaw, yaw_rate=yaw_rate
    )
    contracts = _component_contracts(spec, trajectory, jerk, fourier, profile)
    return trajectory, jerk, contracts


def _acceptance_reasons(
    spec: EpisodeSpec, trajectory: Trajectory, score: dict[str, Any], parent: dict[str, Any]
) -> list[str]:
    contract = CLASS_CONTRACTS[spec.motion_class]
    reasons: list[str] = []
    position = np.asarray(trajectory.pos)
    if not score["all_finite"] or not parent["all_finite"]:
        reasons.append("nonfinite_reference")
    if np.any(position < robust.WORKSPACE_MIN_M) or np.any(position > robust.WORKSPACE_MAX_M):
        reasons.append("parent_workspace")
    for name in ("u_trans", "u_yaw"):
        if not (contract.lower_open < score[name] <= contract.upper_closed):
            reasons.append(f"score_{name}_outside_class_band")
        if parent[name] > 0.95:
            reasons.append(f"parent_{name}_above_global_0p95")
    if parent["max_abs_yaw_rad"]["value"] > contract.max_abs_yaw_rad:
        reasons.append("parent_abs_yaw_above_class_limit")
    indices = _interval_indices(spec)
    position_window = np.asarray(trajectory.pos)[indices]
    velocity_window = np.asarray(trajectory.vel)[indices]
    yaw_window = np.asarray(trajectory.yaw)[indices]
    yaw_rate_window = np.asarray(trajectory.yaw_rate)[indices]
    if np.ptp(position_window[:, 0]) < 0.01:
        reasons.append("score_x_excitation")
    if np.ptp(position_window[:, 1]) < 0.01:
        reasons.append("score_y_excitation")
    if not (
        np.ptp(position_window[:, 2]) >= 0.005
        or float(np.sqrt(np.mean(velocity_window[:, 2] ** 2))) >= 0.01
    ):
        reasons.append("score_z_excitation")
    if np.ptp(yaw_window) < 0.02:
        reasons.append("score_yaw_excitation")
    if float(np.sqrt(np.mean(yaw_rate_window**2))) < 0.01:
        reasons.append("score_yaw_rate_excitation")
    return reasons


def construct_episode(spec: EpisodeSpec) -> ReferenceCandidate:
    """Construct one parent using controller-blind deterministic rejection only."""
    attempts: list[dict[str, Any]] = []
    for attempt in range(MAX_ATTEMPTS):
        trajectory, jerk, contracts = _candidate_attempt(spec, attempt)
        score = robust.reference_statistics(trajectory, jerk, _interval_indices(spec))
        parent = robust.reference_statistics(trajectory, jerk, slice(None))
        reasons = _acceptance_reasons(spec, trajectory, score, parent)
        key_digests = {
            component: array_digest(jax.random.key_data(component_key(spec, attempt, component)))
            for component in ("position", "yaw")
        }
        attempts.append(
            {
                "attempt_index": attempt,
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "score_statistics": score,
                "parent_statistics": parent,
                "component_key_digests": key_digests,
            }
        )
        if reasons:
            continue
        arrays = {
            "time": trajectory.time,
            "position": trajectory.pos,
            "velocity": trajectory.vel,
            "acceleration": trajectory.acc,
            "jerk": jerk,
            "yaw": trajectory.yaw,
            "yaw_rate": trajectory.yaw_rate,
        }
        digests = {name: array_digest(value) for name, value in arrays.items()}
        parent_digest = sha256_bytes(canonical_json_bytes(digests))
        return ReferenceCandidate(
            trajectory=trajectory,
            jerk=jerk,
            attempt=attempt,
            attempts=tuple(attempts),
            score_statistics=score,
            parent_statistics=parent,
            array_digests=digests,
            component_contracts=contracts,
            parent_digest=parent_digest,
        )
    raise G3ContractError(f"{spec.episode_id} has no accepted attempt in {MAX_ATTEMPTS}")


@lru_cache(maxsize=1)
def construct_all_episodes() -> tuple[tuple[EpisodeSpec, ReferenceCandidate], ...]:
    """Construct and validate all registered reference episodes."""
    constructed = tuple((spec, construct_episode(spec)) for spec in episode_specs())
    if len(constructed) != 24:
        raise G3ContractError("exactly 24 parents are required")
    train = [candidate for spec, candidate in constructed if spec.split is Split.TRAIN]
    validation = [candidate for spec, candidate in constructed if spec.split is Split.VALIDATION]
    train_digests = {candidate.parent_digest for candidate in train}
    validation_digests = {candidate.parent_digest for candidate in validation}
    if train_digests & validation_digests:
        raise G3ContractError("Train/Validation full-parent digest leakage")
    return constructed


def stack_evaluation_inputs(
    constructed: Iterable[tuple[EpisodeSpec, ReferenceCandidate]],
) -> EvaluationInputs:
    """Stack constructed episodes into batched evaluation inputs."""
    items = tuple(constructed)
    specs = tuple(spec for spec, _ in items)
    candidates = tuple(candidate for _, candidate in items)
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
            jax.tree.map(lambda value: value[1 : ROLLOUT_INTERVALS + 1], candidate.trajectory)
            for candidate in candidates
        ),
    )
    expected = (ROLLOUT_INTERVALS, len(specs), 1, 13)
    if commands.shape != expected:
        raise G3ContractError(f"stacked commands changed shape: {commands.shape} != {expected}")
    return EvaluationInputs(specs, candidates, commands, reference)


def focused_inputs() -> EvaluationInputs:
    """Return evaluation inputs for the focused parent pair."""
    selected = tuple(
        item for item in construct_all_episodes() if item[0].episode_id in FOCUSED_PARENT_IDS
    )
    if tuple(item[0].episode_id for item in selected) != FOCUSED_PARENT_IDS:
        raise G3ContractError("focused parent order or identity changed")
    return stack_evaluation_inputs(selected)


def build_initial_data(inputs: EvaluationInputs) -> tuple[Any, SimData]:
    """Build the simulation and initialize its evaluation data."""
    sim = robust.build_simulation(len(inputs.specs), rng_seed=2602)
    if str(sim.mjx_model.impl) != "Impl.JAX" or str(sim.mjx_data.impl) != "Impl.JAX":
        raise G3ContractError("G3.4 requires MJX Impl.JAX for model and data")
    if mjx_warp.WARP_INSTALLED:
        raise G3ContractError("G3.4 requires WARP_INSTALLED=false")
    return sim, robust.initialize_inputs(sim.data, inputs)


def _state_params(data: SimData) -> dict[str, Array]:
    state = data.controls.state
    if state is None:
        raise G3ContractError("state controller is unavailable")
    return state.params


def with_mass_thrust_relaxation(data: SimData, theta: Any = 0.0) -> SimData:
    """Replace only canonical ``mass_thrust`` by Float32 ``132000*exp(theta)``."""
    state = data.controls.state
    if state is None:
        raise G3ContractError("mass_thrust relaxation requires the state controller")
    canonical = state.params["mass_thrust"]
    canonical_array = np.asarray(canonical)
    if canonical_array.shape != () or not np.issubdtype(canonical_array.dtype, np.integer):
        raise G3ContractError("canonical mass_thrust is not the original scalar integer leaf")
    if int(canonical_array) != CANONICAL_MASS_THRUST:
        raise G3ContractError("canonical mass_thrust value changed")
    theta_array = jnp.asarray(theta, dtype=jnp.float32)
    relaxed = jnp.asarray(CANONICAL_MASS_THRUST, dtype=jnp.float32) * jnp.exp(theta_array)
    return data.replace(
        controls=data.controls.replace(
            state=state.replace(params=state.params | {"mass_thrust": relaxed})
        )
    )


def _replace_parameter(data: SimData, name: str, theta: Any) -> SimData:
    """Apply exactly one positive log-coordinate to the relaxed continuous baseline."""
    if name not in PARAMETER_SPECS:
        raise ValueError(f"unknown parameter {name}")
    relaxed = with_mass_thrust_relaxation(data, 0.0)
    if name == "mass_thrust":
        return with_mass_thrust_relaxation(data, theta)
    state = relaxed.controls.state
    assert state is not None
    spec = PARAMETER_SPECS[name]
    value = jnp.asarray(spec.default, dtype=jnp.float32) * jnp.exp(
        jnp.asarray(theta, dtype=jnp.float32)
    )
    if spec.axes:
        parameter = state.params[spec.parameter]
        parameter = parameter.at[jnp.asarray(spec.axes)].set(value)
    else:
        parameter = value
    return relaxed.replace(
        controls=relaxed.controls.replace(
            state=state.replace(params=state.params | {spec.parameter: parameter})
        )
    )


def _leaf_records(tree: Any) -> tuple[tuple[str, np.ndarray], ...]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    records = []
    for path, leaf in leaves:
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        records.append((jax.tree_util.keystr(path), np.asarray(leaf)))
    return tuple(records)


def _relaxation_leaf_diff(canonical: SimData, relaxed: SimData) -> dict[str, Any]:
    left = _leaf_records(canonical)
    right = _leaf_records(relaxed)
    if len(left) != len(right):
        raise G3ContractError("relaxation changed the PyTree leaf count")
    differences = []
    for (left_path, left_value), (right_path, right_value) in zip(left, right, strict=True):
        if left_path != right_path:
            raise G3ContractError("relaxation changed PyTree path order")
        if (
            left_value.shape != right_value.shape
            or left_value.dtype != right_value.dtype
            or not np.array_equal(left_value, right_value)
        ):
            differences.append(
                {
                    "path": left_path,
                    "canonical_shape": list(left_value.shape),
                    "relaxed_shape": list(right_value.shape),
                    "canonical_dtype": str(left_value.dtype),
                    "relaxed_dtype": str(right_value.dtype),
                    "canonical_value": left_value.tolist(),
                    "relaxed_value": right_value.tolist(),
                }
            )
    if len(differences) != 1 or "['mass_thrust']" not in differences[0]["path"]:
        raise G3ContractError(f"relaxation changed leaves outside mass_thrust: {differences}")
    difference = differences[0]
    if difference["canonical_value"] != CANONICAL_MASS_THRUST:
        raise G3ContractError("canonical integer mass_thrust changed")
    if np.asarray(difference["relaxed_value"], dtype=np.float32).item() != np.float32(
        CANONICAL_MASS_THRUST
    ):
        raise G3ContractError("theta=0 relaxation is not exact Float32 132000")
    if difference["relaxed_dtype"] != "float32":
        raise G3ContractError("relaxed mass_thrust dtype is not Float32")
    return {
        "status": "PASS_ONLY_MASS_THRUST_DTYPE_RELAXED",
        "leaf_count": len(left),
        "difference": difference,
    }


def _require_relaxed_parity(label: str, canonical: Any, relaxed: Any) -> None:
    left = _leaf_records(canonical)
    right = _leaf_records(relaxed)
    if len(left) != len(right):
        raise G3ContractError(f"{label}: PyTree leaf count differs")
    for (left_path, left_value), (right_path, right_value) in zip(left, right, strict=True):
        if left_path != right_path or left_value.shape != right_value.shape:
            raise G3ContractError(f"{label}: path/shape differs at {left_path}")
        if "['mass_thrust']" in left_path:
            if int(left_value) != CANONICAL_MASS_THRUST:
                raise G3ContractError(f"{label}: canonical mass_thrust value differs")
            if right_value.dtype != np.dtype(np.float32) or float(right_value) != float(
                np.float32(CANONICAL_MASS_THRUST)
            ):
                raise G3ContractError(f"{label}: relaxed mass_thrust exception differs")
            continue
        if left_value.dtype != right_value.dtype or not np.array_equal(left_value, right_value):
            raise G3ContractError(f"{label}: exact parity failed at {left_path}")


def _parameter_isolation(data: SimData, name: str, theta: float = EFFECT_STEP) -> dict[str, Any]:
    baseline = with_mass_thrust_relaxation(data, 0.0)
    perturbed = _replace_parameter(data, name, theta)
    left = _leaf_records(baseline)
    right = _leaf_records(perturbed)
    changed = []
    spec = PARAMETER_SPECS[name]
    for (left_path, left_value), (right_path, right_value) in zip(left, right, strict=True):
        if left_path != right_path or left_value.shape != right_value.shape:
            raise G3ContractError(f"{name} isolation changed path or shape")
        if left_value.dtype != right_value.dtype:
            raise G3ContractError(f"{name} isolation changed dtype at {left_path}")
        different = np.argwhere(left_value != right_value)
        if different.size or (
            left_value.shape == () and not np.array_equal(left_value, right_value)
        ):
            changed.append((left_path, different.tolist()))
    if len(changed) != 1 or f"['{spec.parameter}']" not in changed[0][0]:
        raise G3ContractError(f"{name} isolation is ambiguous: {changed}")
    if spec.axes and sorted(index[0] for index in changed[0][1]) != sorted(spec.axes):
        raise G3ContractError(f"{name} changed indices do not match {spec.axes}")
    value = spec.default * math.exp(theta)
    if not spec.lower <= value <= spec.upper:
        raise G3ContractError(f"{name} effect perturbation leaves diagnostic bounds")
    return {
        "status": "PASS_EXACT_PARAMETER_ISOLATION",
        "parameter": name,
        "changed_leaf": changed[0][0],
        "changed_indices": changed[0][1],
        "theta": theta,
        "physical_value": value,
    }


def _parameter_default_parity(data: SimData, name: str) -> dict[str, Any]:
    """Require the declared theta-zero coordinate to preserve every relaxed baseline leaf."""
    baseline = with_mass_thrust_relaxation(data, 0.0)
    theta_zero = _replace_parameter(data, name, 0.0)
    left = _leaf_records(baseline)
    right = _leaf_records(theta_zero)
    if len(left) != len(right):
        raise G3ContractError(f"{name} theta-zero parity changed the leaf count")
    for (left_path, left_value), (right_path, right_value) in zip(left, right, strict=True):
        if left_path != right_path:
            raise G3ContractError(f"{name} theta-zero parity changed leaf order")
        if (
            left_value.shape != right_value.shape
            or left_value.dtype != right_value.dtype
            or not np.array_equal(left_value, right_value)
        ):
            raise G3ContractError(f"{name} theta-zero parity failed at {left_path}")
    return {
        "status": "PASS_THETA_ZERO_DEFAULT_PARITY",
        "parameter": name,
        "parent_independent_simdata_leaf_count": len(left),
        "all_leaves_array_equal": True,
    }


def _direct_state_controller_parity(
    canonical: SimData, relaxed: SimData, command: Array
) -> dict[str, Any]:
    canonical_state = canonical.controls.state
    relaxed_state = relaxed.controls.state
    if canonical_state is None or relaxed_state is None:
        raise G3ContractError("direct parity requires state controllers")
    args = (canonical.states.pos, canonical.states.quat, canonical.states.vel, command)
    original_outputs = state2attitude(
        *args,
        pos_err_i=canonical_state.pos_err_i,
        ctrl_freq=canonical_state.freq,
        **canonical_state.params,
    )
    relaxed_outputs = state2attitude(
        *args,
        pos_err_i=relaxed_state.pos_err_i,
        ctrl_freq=relaxed_state.freq,
        **relaxed_state.params,
    )
    for name, left, right in zip(
        ("command_rpyt", "pos_err_i"), original_outputs, relaxed_outputs, strict=True
    ):
        robust.require_array_equal(f"direct state-controller {name}", left, right)
        if not np.all(np.isfinite(np.asarray(left))):
            raise G3ContractError(f"direct state-controller {name} is nonfinite")
    return {
        "status": "PASS_DIRECT_STATE_CONTROLLER_DEFAULT_PARITY",
        "paths": ["command_rpyt", "pos_err_i"],
        "array_equal": True,
    }


def _scored_bundle(inputs: EvaluationInputs, initial_data: SimData, outputs: Any) -> dict[str, Any]:
    trace = robust.scored_trace(outputs["trace"], inputs.specs)
    reference = robust.scored_reference(inputs.reference, inputs.specs)
    pos_err_i = robust.scored_interval_value(outputs["pos_err_i"], inputs.specs)
    diagnostic = jax.tree.map(
        lambda value: robust.scored_diagnostic_value(value, inputs.specs), outputs["diagnostic"]
    )
    config = TrackingLossConfig()
    total, _ = tracking_loss_per_case(
        trace,
        reference,
        hover_rotor_velocity(initial_data),
        rotor_velocity_limits(initial_data),
        config,
    )
    terms = tracking_loss_terms_per_case(
        trace, reference, hover_rotor_velocity(initial_data), config
    )
    return {
        "trace": trace,
        "reference": reference,
        "pos_err_i": pos_err_i,
        "diagnostic": diagnostic,
        "loss_total": total,
        "loss_terms": terms,
    }


def _feature_matrix(scored: dict[str, Any], initial_data: SimData) -> tuple[tuple[str, ...], Array]:
    trace = scored["trace"]
    reference = scored["reference"]
    integral = scored["pos_err_i"]
    diagnostic = scored["diagnostic"]
    terms = scored["loss_terms"]
    pos_error = trace.pos - reference.pos
    vel_error = trace.vel - reference.vel
    state = initial_data.controls.state
    assert state is not None
    integral_limits = state.params["int_err_max"]
    stage_a = diagnostic["stage_a"]
    stage_b = diagnostic["stage_b"]

    values: dict[str, Array] = {}
    for term in LOSS_TERM_NAMES:
        values[f"loss_raw_{term}"] = terms.raw[term]
        values[f"loss_normalization_{term}"] = jnp.broadcast_to(
            terms.normalization_divisor[term], scored["loss_total"].shape
        )
        values[f"loss_normalized_{term}"] = terms.normalized[term]
        values[f"loss_weight_{term}"] = jnp.broadcast_to(
            terms.weight[term], scored["loss_total"].shape
        )
        values[f"loss_contribution_{term}"] = terms.weighted[term]
    values["loss_total"] = scored["loss_total"]
    values["position_mse_xy"] = jnp.mean(jnp.sum(pos_error[..., :2] ** 2, axis=-1), axis=(0, 2))
    values["position_mse_z"] = jnp.mean(pos_error[..., 2] ** 2, axis=(0, 2))
    values["position_rmse_xy"] = jnp.sqrt(values["position_mse_xy"])
    values["position_rmse_z"] = jnp.sqrt(values["position_mse_z"])
    values["velocity_rmse_xy"] = jnp.sqrt(
        jnp.mean(jnp.sum(vel_error[..., :2] ** 2, axis=-1), axis=(0, 2))
    )
    values["velocity_rmse_z"] = jnp.sqrt(jnp.mean(vel_error[..., 2] ** 2, axis=(0, 2)))
    values["mean_signed_z_error"] = jnp.mean(pos_error[..., 2], axis=(0, 2))
    values["terminal_signed_z_error"] = jnp.mean(pos_error[-1, ..., 2], axis=1)
    for axis, axis_name in enumerate(("x", "y", "z")):
        axis_value = integral[..., axis]
        values[f"integral_rms_{axis_name}"] = jnp.sqrt(jnp.mean(axis_value**2, axis=(0, 2)))
        values[f"integral_max_abs_{axis_name}"] = jnp.max(jnp.abs(axis_value), axis=(0, 2))
        values[f"integral_terminal_{axis_name}"] = jnp.mean(axis_value[-1], axis=1)
    normalized_integral_z = integral[..., 2] / integral_limits[2]
    values["integral_z_nearness_rms"] = jnp.sqrt(jnp.mean(normalized_integral_z**2, axis=(0, 2)))
    values["loss_v2_integral_z_candidate_v0"] = jnp.mean(normalized_integral_z**2, axis=(0, 2))
    lower = stage_a["force_lower_reserve_normalized"]
    upper = stage_a["force_upper_reserve_normalized"]
    values["minimum_normalized_motor_reserve"] = jnp.min(
        jnp.minimum(lower, upper), axis=(0, 1, 3, 4)
    )
    values["requested_collective_mean"] = jnp.mean(
        stage_a["collective_force_request_n"], axis=(0, 1, 3)
    )
    for axis, axis_name in enumerate(("collective", "roll", "pitch", "yaw")):
        requested = stage_a["wrench_preclip"][..., axis]
        realized = stage_b["wrench_postclip"][..., axis]
        distortion = stage_b["wrench_distortion_absolute"][..., axis]
        values[f"requested_wrench_rms_{axis_name}"] = jnp.sqrt(
            jnp.mean(requested**2, axis=(0, 1, 3))
        )
        values[f"realized_wrench_rms_{axis_name}"] = jnp.sqrt(jnp.mean(realized**2, axis=(0, 1, 3)))
        values[f"wrench_distortion_abs_mean_{axis_name}"] = jnp.mean(distortion, axis=(0, 1, 3))
        values[f"wrench_distortion_normalized_mean_{axis_name}"] = jnp.mean(
            distortion / jnp.asarray(WRENCH_PLATFORM_SCALE[axis], dtype=distortion.dtype),
            axis=(0, 1, 3),
        )
    names = tuple(values)
    return names, jnp.stack(tuple(values[name] for name in names), axis=1)


def _segment_loss_matrix(
    inputs: EvaluationInputs, initial_data: SimData, outputs: Any
) -> tuple[tuple[str, ...], Array]:
    """Return Loss-v1 values for every episode and required rollout segment."""
    trace = outputs["trace"]
    reference = inputs.reference
    hover = hover_rotor_velocity(initial_data)
    config = TrackingLossConfig()
    columns: dict[str, list[Array]] = {}
    for segment in ("warmup_prefix", "scored_window", "full_rollout"):
        for world, spec in enumerate(inputs.specs):
            start, stop = {
                "warmup_prefix": (0, spec.score_start),
                "scored_window": (spec.score_start, spec.score_stop),
                "full_rollout": (0, ROLLOUT_INTERVALS),
            }[segment]
            segment_trace = jax.tree.map(lambda value: value[start:stop, world : world + 1], trace)
            segment_reference = jax.tree.map(
                lambda value: value[start:stop, world : world + 1], reference
            )
            terms = tracking_loss_terms_per_case(
                segment_trace, segment_reference, hover[world : world + 1], config
            )
            total = jnp.zeros_like(terms.weighted[LOSS_TERM_NAMES[0]])
            for term in LOSS_TERM_NAMES:
                total = total + terms.weighted[term]
            values: dict[str, Array] = {}
            for term in LOSS_TERM_NAMES:
                values[f"{segment}_loss_raw_{term}"] = terms.raw[term]
                values[f"{segment}_loss_normalization_{term}"] = jnp.broadcast_to(
                    terms.normalization_divisor[term], total.shape
                )
                values[f"{segment}_loss_weight_{term}"] = jnp.broadcast_to(
                    terms.weight[term], total.shape
                )
                values[f"{segment}_loss_contribution_{term}"] = terms.weighted[term]
            values[f"{segment}_loss_total"] = total
            for name, value in values.items():
                columns.setdefault(name, []).append(value[0])
    names = tuple(columns)
    return names, jnp.stack(tuple(jnp.stack(columns[name]) for name in names), axis=1)


def _evaluation_feature_matrix(
    inputs: EvaluationInputs, initial_data: SimData, outputs: Any
) -> tuple[tuple[str, ...], Array]:
    scored_names, scored_matrix = _feature_matrix(
        _scored_bundle(inputs, initial_data, outputs), initial_data
    )
    segment_names, segment_matrix = _segment_loss_matrix(inputs, initial_data, outputs)
    return scored_names + segment_names, jnp.concatenate((scored_matrix, segment_matrix), axis=1)


def _instrumented_evaluation(
    sim: Any, data: SimData, inputs: EvaluationInputs
) -> tuple[Any, Any, Any, dict[str, Any], tuple[str, ...], Array]:
    rollout = robust.make_instrumented_rollout(sim)
    final, boundaries, outputs = rollout(data, inputs.commands)
    jax.block_until_ready((final, boundaries, outputs))
    scored = _scored_bundle(inputs, data, outputs)
    names, features = _evaluation_feature_matrix(inputs, data, outputs)
    jax.block_until_ready(features)
    return final, boundaries, outputs, scored, names, features


def _static_source_contract() -> dict[str, Any]:
    source = inspect.getsource(state2attitude)
    required = ("mass * (setpoint_acc - gravity_vec)", "mass_thrust * current_thrust", "pwm2force")
    if any(fragment not in source for fragment in required):
        raise G3ContractError("state2attitude source chain changed")
    if "rint" in source or "round" in source:
        raise G3ContractError("shared state2attitude unexpectedly rounds mass_thrust")
    return {
        "status": "PASS_STATIC_SOURCE_CHAIN",
        "source": "crazyflow/control/mellinger/control.py::state2attitude",
        "required_fragments": list(required),
        "shared_source_modified": False,
    }


@lru_cache(maxsize=1)
def focused_default_parity() -> dict[str, Any]:
    """Run static, direct-controller, and two-parent full default parity."""
    inputs = focused_inputs()
    sim, canonical = build_initial_data(inputs)
    relaxed = with_mass_thrust_relaxation(canonical, 0.0)
    leaf = _relaxation_leaf_diff(canonical, relaxed)
    direct = _direct_state_controller_parity(canonical, relaxed, inputs.commands[0])
    original = _instrumented_evaluation(sim, canonical, inputs)
    relaxed_result = _instrumented_evaluation(sim, relaxed, inputs)
    _require_relaxed_parity("focused final carry", original[0], relaxed_result[0])
    for index, (left, right) in enumerate(zip(original[1], relaxed_result[1], strict=True)):
        _require_relaxed_parity(f"focused boundary carry {index}", left, right)
    robust.require_pytree_equal("focused rollout outputs", original[2], relaxed_result[2])
    robust.require_pytree_equal("focused scored bundle", original[3], relaxed_result[3])
    if original[4] != relaxed_result[4]:
        raise G3ContractError("focused feature inventory changed under relaxation")
    robust.require_array_equal("focused feature parity", original[5], relaxed_result[5])
    return {
        "status": "PASS_FOCUSED_DEFAULT_PARITY",
        "parents": list(FOCUSED_PARENT_IDS),
        "static_source": _static_source_contract(),
        "leaf_isolation": leaf,
        "direct_state_controller": direct,
        "full_rollout_paths": [
            "state_command_and_pos_err_i",
            "attitude_rate_requested_collective_force_torque_preclip_wrench",
            "stage_a_pwm_motorforce_masks_reserve",
            "stage_b_motorforce_rotorspeed_masks_requested_realized_wrench",
            "rollout_states_and_complete_controller_carries",
            "loss_v1_six_terms_normalizations_weights_contributions_total",
            "continuous_feature_inventory",
            "ground_floor_zero_thrust_integral_reserve_clip_counters",
        ],
        "array_equal": True,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def _parameter_function(
    sim: Any, initial_data: SimData, inputs: EvaluationInputs, parameter: str
) -> tuple[Callable[[Array], Array], tuple[str, ...]]:
    rollout = robust.make_instrumented_rollout(sim)
    baseline = with_mass_thrust_relaxation(initial_data, 0.0)
    _, _, outputs = rollout(baseline, inputs.commands)
    names, _ = _evaluation_feature_matrix(inputs, baseline, outputs)

    def evaluate(theta: Array) -> Array:
        data = _replace_parameter(initial_data, parameter, theta)
        _, _, result = rollout(data, inputs.commands)
        evaluated_names, matrix = _evaluation_feature_matrix(inputs, data, result)
        assert evaluated_names == names
        return matrix

    return jax.jit(evaluate), names


def _reverse_fd_comparison(reverse: np.ndarray, fd: np.ndarray) -> dict[str, Any]:
    tolerance = AD_FD_ABSOLUTE_TOLERANCE + AD_FD_RELATIVE_TOLERANCE * np.maximum(
        np.abs(reverse), np.abs(fd)
    )
    absolute_error = np.abs(reverse - fd)
    significant = (np.abs(reverse) > 1.0e-3) & (np.abs(fd) > 1.0e-3)
    sign_equal = (~significant) | (np.signbit(reverse) == np.signbit(fd))
    passed = (absolute_error <= tolerance) & sign_equal
    return {
        "reverse": reverse.tolist(),
        "fd": fd.tolist(),
        "absolute_error": absolute_error.tolist(),
        "tolerance": tolerance.tolist(),
        "significant": significant.tolist(),
        "sign_equal": sign_equal.tolist(),
        "passed": passed.tolist(),
        "mismatch_count": int(np.size(passed) - np.count_nonzero(passed)),
    }


def _output_role_inventory(feature_names: tuple[str, ...]) -> dict[str, Any]:
    """Bind every frozen feature to its optimizer-objective or diagnostic role."""
    if len(feature_names) != G3_004_FEATURE_COUNT or len(set(feature_names)) != len(feature_names):
        raise G3ContractError("G3.4 feature inventory is not exactly 143 unique outputs")
    if any(feature_names.count(name) != 1 for name in OPTIMIZER_OBJECTIVE_NAMES):
        raise G3ContractError("optimizer-objective output inventory changed")
    diagnostic_names = tuple(
        name for name in feature_names if name not in set(OPTIMIZER_OBJECTIVE_NAMES)
    )
    if len(diagnostic_names) != G3_004_DIAGNOSTIC_FEATURE_COUNT:
        raise G3ContractError("diagnostic output inventory is not exactly 136 outputs")
    return {
        "optimizer_objective": {
            "role": OPTIMIZER_OBJECTIVE_ROLE,
            "feature_count": len(OPTIMIZER_OBJECTIVE_NAMES),
            "feature_names": list(OPTIMIZER_OBJECTIVE_NAMES),
            "loss_contract": "Loss-v1 total and exact weighted scored-window decomposition",
        },
        "diagnostic": {
            "role": DIAGNOSTIC_AD_ROLE,
            "feature_count": len(diagnostic_names),
            "feature_names": list(diagnostic_names),
            "optimizer_gradient_claim": False,
        },
        "partition_complete": set(OPTIMIZER_OBJECTIVE_NAMES).isdisjoint(diagnostic_names)
        and len(OPTIMIZER_OBJECTIVE_NAMES) + len(diagnostic_names) == len(feature_names),
    }


def _comparison_mismatch_records(
    comparison: dict[str, Any],
    feature_names: tuple[str, ...],
    feature_indices: tuple[int, ...],
    feasible_coordinates: tuple[tuple[int, EpisodeSpec], ...],
    role_by_feature: dict[str, str],
) -> list[dict[str, Any]]:
    """Map each finite comparison mismatch to its frozen parent and feature labels."""
    reverse = np.asarray(comparison["reverse"], dtype=np.float64)
    fd = np.asarray(comparison["fd"], dtype=np.float64)
    absolute_error = np.asarray(comparison["absolute_error"], dtype=np.float64)
    tolerance = np.asarray(comparison["tolerance"], dtype=np.float64)
    significant = np.asarray(comparison["significant"], dtype=bool)
    sign_equal = np.asarray(comparison["sign_equal"], dtype=bool)
    passed = np.asarray(comparison["passed"], dtype=bool)
    expected_shape = (len(feasible_coordinates), len(feature_names))
    if any(
        value.shape != expected_shape
        for value in (reverse, fd, absolute_error, tolerance, significant, sign_equal, passed)
    ):
        raise G3ContractError("Reverse/FD comparison disclosure shape changed")
    records = []
    for row, column in np.argwhere(~passed):
        parent_registry_index, spec = feasible_coordinates[int(row)]
        feature = feature_names[int(column)]
        reverse_value = float(reverse[row, column])
        fd_value = float(fd[row, column])
        false_zero = abs(fd_value) > 1.0e-3 and abs(reverse_value) <= 1.0e-8
        records.append(
            {
                "feasible_parent_index": int(row),
                "parent_registry_index": parent_registry_index,
                "parent_id": spec.episode_id,
                "split": spec.split.value,
                "motion_class": spec.motion_class.value,
                "profile": spec.profile.value,
                "feature_index": feature_indices[int(column)],
                "feature_name": feature,
                "output_role": role_by_feature[feature],
                "reverse": reverse_value,
                "fd": fd_value,
                "absolute_reverse": abs(reverse_value),
                "absolute_fd": abs(fd_value),
                "absolute_error": float(absolute_error[row, column]),
                "tolerance": float(tolerance[row, column]),
                "significant": bool(significant[row, column]),
                "sign_equal": bool(sign_equal[row, column]),
                "significant_sign_mismatch": bool(
                    significant[row, column] and not sign_equal[row, column]
                ),
                "false_zero": false_zero,
                "causal_claim": None,
            }
        )
    return records


def _output_role_comparison_payload(
    reverse: np.ndarray,
    fd: np.ndarray,
    feature_names: tuple[str, ...],
    specs: tuple[EpisodeSpec, ...],
) -> dict[str, Any]:
    """Compare all finite outputs while separating objective and diagnostic contracts."""
    inventory = _output_role_inventory(feature_names)
    expected_shape = (len(specs), len(feature_names))
    if reverse.shape != expected_shape or fd.shape != expected_shape:
        raise G3ContractError("Reverse/FD arrays do not match the frozen parent/output inventory")
    feasible_coordinates = tuple(
        (index, spec)
        for index, spec in enumerate(specs)
        if spec.motion_class is not MotionClass.NEAR_LIMIT
    )
    feasible = np.asarray(
        [spec.motion_class is not MotionClass.NEAR_LIMIT for spec in specs], dtype=bool
    )
    objective_indices = tuple(feature_names.index(name) for name in OPTIMIZER_OBJECTIVE_NAMES)
    diagnostic_names = tuple(inventory["diagnostic"]["feature_names"])
    diagnostic_indices = tuple(feature_names.index(name) for name in diagnostic_names)
    role_by_feature = {
        name: (
            OPTIMIZER_OBJECTIVE_ROLE if name in OPTIMIZER_OBJECTIVE_NAMES else DIAGNOSTIC_AD_ROLE
        )
        for name in feature_names
    }

    def section(names: tuple[str, ...], indices: tuple[int, ...], role: str) -> dict[str, Any]:
        comparison = _reverse_fd_comparison(reverse[feasible][:, indices], fd[feasible][:, indices])
        passed = np.asarray(comparison["passed"], dtype=bool)
        mismatch_records = _comparison_mismatch_records(
            comparison, names, indices, feasible_coordinates, role_by_feature
        )
        return {
            "role": role,
            "feature_names": list(names),
            "comparison": comparison,
            "mismatch_count": len(mismatch_records),
            "mismatch_strata": int(np.count_nonzero(np.any(~passed, axis=1))),
            "significant_sign_mismatch_count": sum(
                int(record["significant_sign_mismatch"]) for record in mismatch_records
            ),
            "false_zero_count": sum(int(record["false_zero"]) for record in mismatch_records),
            "mismatch_records": mismatch_records,
        }

    all_indices = tuple(range(len(feature_names)))
    return {
        "inventory": inventory,
        "all_features": section(feature_names, all_indices, "MIXED_OUTPUT_ROLE_AUDIT"),
        "optimizer_objective": section(
            OPTIMIZER_OBJECTIVE_NAMES, objective_indices, OPTIMIZER_OBJECTIVE_ROLE
        ),
        "diagnostic": section(diagnostic_names, diagnostic_indices, DIAGNOSTIC_AD_ROLE)
        | {
            "optimizer_gradient_claim": False,
            "finite_mismatch_effect_on_objective_validity": "NONE",
        },
    }


def _objective_gradient_contract_payload(
    objective_section: dict[str, Any],
    allowed_mismatch_strata: int,
    split_congruence_failures: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """Apply the frozen Objective gate to a complete finite comparison disclosure."""
    if type(allowed_mismatch_strata) is not int or allowed_mismatch_strata < 0:
        raise G3ContractError("Objective mismatch-stratum allowance is invalid")
    required = {
        "role",
        "comparison",
        "mismatch_count",
        "mismatch_strata",
        "significant_sign_mismatch_count",
        "false_zero_count",
        "mismatch_records",
    }
    if not required <= objective_section.keys() or objective_section["role"] != (
        OPTIMIZER_OBJECTIVE_ROLE
    ):
        raise G3ContractError("Objective comparison disclosure is incomplete")
    per_parent_gate_failed = (
        objective_section["mismatch_strata"] > allowed_mismatch_strata
        or objective_section["significant_sign_mismatch_count"] > 0
        or objective_section["false_zero_count"] > 0
    )
    failures = []
    if per_parent_gate_failed:
        failures.append(
            {
                "gate": "PER_PARENT_OBJECTIVE_REVERSE_FD",
                "finite": True,
                "mismatch_count": objective_section["mismatch_count"],
                "mismatch_strata": objective_section["mismatch_strata"],
                "allowed_mismatch_strata": allowed_mismatch_strata,
                "mismatch_strata_allowance_exceeded": (
                    objective_section["mismatch_strata"] > allowed_mismatch_strata
                ),
                "significant_sign_mismatch": (
                    objective_section["significant_sign_mismatch_count"] > 0
                ),
                "false_zero": objective_section["false_zero_count"] > 0,
                "comparison_disclosure": objective_section,
            }
        )
    for failure in split_congruence_failures:
        if failure.get("gate") != "SPLIT_AGGREGATE_OBJECTIVE_REVERSE_CONGRUENCE" or (
            failure.get("finite") is not True
        ):
            raise G3ContractError("Objective split-congruence failure record is invalid")
        failures.append(failure)
    eligible = not failures
    return {
        "role": OPTIMIZER_OBJECTIVE_ROLE,
        "eligible": eligible,
        "withholding_reason": (None if eligible else WITHHELD_OBJECTIVE_GRADIENT_INVALID),
        "finite_failures": failures,
        "comparison_disclosure": objective_section,
        "frozen_gate": {
            "mismatch_strata": objective_section["mismatch_strata"],
            "allowed_mismatch_strata": allowed_mismatch_strata,
            "significant_sign_mismatch": (objective_section["significant_sign_mismatch_count"] > 0),
            "false_zero": objective_section["false_zero_count"] > 0,
            "split_congruence_failure_count": len(split_congruence_failures),
        },
    }


def _with_discrete_mass_thrust(data: SimData, value: int) -> SimData:
    state = data.controls.state
    if state is None:
        raise G3ContractError("discrete response requires state controller")
    original = state.params["mass_thrust"]
    if not np.issubdtype(np.asarray(original).dtype, np.integer):
        raise G3ContractError("discrete response did not start from canonical integer dtype")
    discrete = jnp.asarray(value, dtype=original.dtype)
    return data.replace(
        controls=data.controls.replace(
            state=state.replace(params=state.params | {"mass_thrust": discrete})
        )
    )


def _loss_only(sim: Any, data: SimData, inputs: EvaluationInputs) -> np.ndarray:
    rollout = robust.make_instrumented_rollout(sim)
    _, _, outputs = rollout(data, inputs.commands)
    scored = _scored_bundle(inputs, data, outputs)
    jax.block_until_ready(scored["loss_total"])
    return np.asarray(scored["loss_total"], dtype=np.float64)


def _json_number(value: Any) -> float | str:
    scalar = float(value)
    if np.isnan(scalar):
        return "NaN"
    if np.isposinf(scalar):
        return "Infinity"
    if np.isneginf(scalar):
        return "-Infinity"
    return scalar


def _json_array(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return _json_number(array.item())
    return [_json_array(item) for item in array]


def _array_payload(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "isfinite": bool(np.all(np.isfinite(array))),
        "value": _json_array(array),
    }


def _array_fd_comparison(ad: Any, fd: Any) -> dict[str, Any]:
    ad_array = np.asarray(ad, dtype=np.float64)
    fd_array = np.asarray(fd, dtype=np.float64)
    tolerance = AD_FD_ABSOLUTE_TOLERANCE + AD_FD_RELATIVE_TOLERANCE * np.maximum(
        np.abs(ad_array), np.abs(fd_array)
    )
    absolute_error = np.abs(ad_array - fd_array)
    finite = np.isfinite(ad_array) & np.isfinite(fd_array)
    passed = finite & (absolute_error <= tolerance)
    return {
        "finite": bool(np.all(finite)),
        "passed": bool(np.all(passed)),
        "passed_elements": _json_array(passed),
        "absolute_error": _array_payload(absolute_error),
        "tolerance": _array_payload(tolerance),
    }


def _reverse_congruence_comparison(
    left: float, right: float, *, require_sign: bool
) -> dict[str, Any]:
    finite_left = math.isfinite(left)
    finite_right = math.isfinite(right)
    absolute_difference = abs(left - right)
    tolerance = AD_FD_ABSOLUTE_TOLERANCE + AD_FD_RELATIVE_TOLERANCE * max(abs(left), abs(right))
    sign_required = require_sign and abs(left) > 1.0e-3 and abs(right) > 1.0e-3
    sign_equal = (left == 0.0 and right == 0.0) or (
        bool(np.signbit(left)) == bool(np.signbit(right))
    )
    within_tolerance = finite_left and finite_right and absolute_difference <= tolerance
    return {
        "absolute_difference": _json_number(absolute_difference),
        "tolerance": _json_number(tolerance),
        "finite_left": finite_left,
        "finite_right": finite_right,
        "sign_equal": sign_equal,
        "sign_required": sign_required,
        "passed": within_tolerance and (sign_equal or not sign_required),
    }


def _reverse_congruence_gate(
    episode_id: str,
    scalar_value_and_grad: float,
    vector_reverse_total: float,
    central_fd_total: float,
) -> dict[str, Any]:
    scalar_vs_vector = _reverse_congruence_comparison(
        scalar_value_and_grad, vector_reverse_total, require_sign=False
    )
    scalar_vs_fd = _reverse_congruence_comparison(
        scalar_value_and_grad, central_fd_total, require_sign=True
    )
    vector_vs_fd = _reverse_congruence_comparison(
        vector_reverse_total, central_fd_total, require_sign=True
    )
    return {
        "parent_id": episode_id,
        "scalar_value_and_grad": _json_number(scalar_value_and_grad),
        "vector_reverse_total": _json_number(vector_reverse_total),
        "central_fd_total": _json_number(central_fd_total),
        "scalar_vs_vector": scalar_vs_vector,
        "scalar_vs_fd": scalar_vs_fd,
        "vector_vs_fd": vector_vs_fd,
        "passed": all(
            comparison["passed"] for comparison in (scalar_vs_vector, scalar_vs_fd, vector_vs_fd)
        ),
    }


def _focused_loss_vector_function(
    sim: Any, initial_data: SimData, inputs: EvaluationInputs, parameter: str
) -> Callable[[Array], Array]:
    """Return the seven-output Loss-v1 vector used only by the G3.4 reverse smoke."""
    rollout = robust.make_instrumented_rollout(sim)

    def evaluate(theta: Array) -> Array:
        data = _replace_parameter(initial_data, parameter, theta)
        _, _, outputs = rollout(data, inputs.commands)
        scored = _scored_bundle(inputs, data, outputs)
        return jnp.stack(
            (scored["loss_total"],)
            + tuple(scored["loss_terms"].weighted[name] for name in LOSS_TERM_NAMES),
            axis=1,
        )

    return jax.jit(evaluate)


def validate_g3_004_smoke(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen smoke assertions after the complete raw record exists."""
    failures: list[str] = []
    if raw.get("parameter_order") != list(PARAMETER_NAMES):
        failures.append("seven-parameter order changed")
    if raw.get("parent_order") != list(FOCUSED_PARENT_IDS):
        failures.append("focused parent order changed")
    if raw.get("default_parity", {}).get("status") != "PASS_FOCUSED_DEFAULT_PARITY":
        failures.append("focused default parity failed")
    records = raw.get("parameters", {})
    if set(records) != set(PARAMETER_NAMES):
        failures.append("a mandatory parameter is missing")
    for parameter in PARAMETER_NAMES:
        record = records.get(parameter)
        if record is None:
            continue
        if record["isolation"]["status"] != "PASS_EXACT_PARAMETER_ISOLATION":
            failures.append(f"{parameter}: leaf isolation failed")
        if record["theta_zero_default_parity"]["status"] != ("PASS_THETA_ZERO_DEFAULT_PARITY"):
            failures.append(f"{parameter}: theta-zero default parity failed")
        if record["optimizer_initialization_count"] or record["optimizer_update_count"]:
            failures.append(f"{parameter}: optimizer contract failed")
        parent_records = record.get("parents", [])
        if [item["parent_id"] for item in parent_records] != list(FOCUSED_PARENT_IDS):
            failures.append(f"{parameter}: parent inventory changed")
            continue
        for parent in parent_records:
            if not parent["all_primal_values_finite"]:
                failures.append(f"{parameter}/{parent['parent_id']}: nonfinite Loss-v1 primal")
            if not parent["scalar_reverse_gate"]["passed"]:
                failures.append(f"{parameter}/{parent['parent_id']}: scalar reverse gate failed")
            for component in parent["loss_vector"]:
                if not component["reverse_vs_fd"]["passed"]:
                    failures.append(
                        f"{parameter}/{parent['parent_id']}/{component['name']}: reverse/FD failed"
                    )
                if component["false_zero"]:
                    failures.append(
                        f"{parameter}/{parent['parent_id']}/{component['name']}: false zero"
                    )
    if raw.get("optimizer_initialization_count") or raw.get("optimizer_update_count"):
        failures.append("top-level optimizer contract failed")
    if failures:
        raise G3ContractError(
            "G3.4 reverse smoke failed after raw recording:\n"
            + canonical_json_bytes({"failures": failures, "raw": raw}).decode().rstrip()
        )
    validated = dict(raw)
    validated["status"] = "PASS_ALL_SEVEN_PARAMETER_REVERSE_SMOKE"
    validated["raw_recorded_before_assertions"] = True
    return validated


@lru_cache(maxsize=1)
def _g3_004_smoke_raw() -> dict[str, Any]:
    """Compute all smoke raw values without applying a scientific assertion."""
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise G3ContractError("G3.4 requires CPU with JAX x64 disabled")
    parity = focused_default_parity()
    inputs = focused_inputs()
    sim, initial_data = build_initial_data(inputs)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    plus_theta = jnp.asarray(AD_FD_STEP, dtype=jnp.float32)
    minus_theta = jnp.asarray(-AD_FD_STEP, dtype=jnp.float32)
    parameter_records: dict[str, Any] = {}
    for parameter in PARAMETER_NAMES:
        function = _focused_loss_vector_function(sim, initial_data, inputs, parameter)
        baseline = function(zero)
        plus = function(plus_theta)
        minus = function(minus_theta)
        vector_reverse = jax.jacrev(function)(zero)
        scalar_results = []
        for world in range(len(inputs.specs)):
            value, gradient = jax.value_and_grad(
                lambda theta, world=world: function(theta)[world, 0]
            )(zero)
            scalar_results.append((value, gradient))
        jax.block_until_ready((baseline, plus, minus, vector_reverse, scalar_results))
        baseline_np = np.asarray(baseline, dtype=np.float64)
        plus_np = np.asarray(plus, dtype=np.float64)
        minus_np = np.asarray(minus, dtype=np.float64)
        reverse_np = np.asarray(vector_reverse, dtype=np.float64)
        fd_np = (plus_np - minus_np) / (2.0 * AD_FD_STEP)
        parent_records = []
        for world, spec in enumerate(inputs.specs):
            scalar_value = float(np.asarray(scalar_results[world][0]))
            scalar_reverse = float(np.asarray(scalar_results[world][1]))
            vector_total = float(reverse_np[world, 0])
            fd_total = float(fd_np[world, 0])
            loss_vector = []
            for index, name in enumerate(LOSS_ONLY_NAMES):
                reverse_value = float(reverse_np[world, index])
                fd_value = float(fd_np[world, index])
                loss_vector.append(
                    {
                        "name": name,
                        "primal": _json_number(baseline_np[world, index]),
                        "theta_plus_h": _json_number(plus_np[world, index]),
                        "theta_minus_h": _json_number(minus_np[world, index]),
                        "vector_reverse": _json_number(reverse_value),
                        "central_fd": _json_number(fd_value),
                        "reverse_vs_fd": _reverse_congruence_comparison(
                            reverse_value, fd_value, require_sign=True
                        ),
                        "false_zero": abs(fd_value) > 1.0e-3 and abs(reverse_value) <= 1.0e-8,
                    }
                )
            parent_records.append(
                {
                    "parent_id": spec.episode_id,
                    "split": spec.split.value,
                    "motion_class": spec.motion_class.value,
                    "profile": spec.profile.value,
                    "scalar_loss_value": _json_number(scalar_value),
                    "scalar_reverse_gate": _reverse_congruence_gate(
                        spec.episode_id, scalar_reverse, vector_total, fd_total
                    ),
                    "loss_vector": loss_vector,
                    "all_primal_values_finite": bool(np.all(np.isfinite(baseline_np[world]))),
                }
            )
        parameter_records[parameter] = {
            "parameter": parameter,
            "coordinate": (
                "mass_thrust(theta)=float32(132000)*exp(theta)"
                if parameter == "mass_thrust"
                else "p(theta)=p_default*exp(theta)"
            ),
            "isolation": _parameter_isolation(initial_data, parameter),
            "theta_zero_default_parity": _parameter_default_parity(initial_data, parameter),
            "parents": parent_records,
            "optimizer_initialization_count": 0,
            "optimizer_update_count": 0,
        }
    return {
        "smoke_version": G3_004_SMOKE_VERSION,
        "status": "RAW_VALUES_RECORDED_BEFORE_ASSERTIONS",
        "work_order": "WO-GR-G3-004",
        "source_base_commit": SOURCE_BASE_COMMIT,
        "platform": PLATFORM,
        "backend": "cpu",
        "jax_enable_x64": False,
        "float_execution": "Float32",
        "h": AD_FD_STEP,
        "parameter_order": list(PARAMETER_NAMES),
        "parent_order": list(FOCUSED_PARENT_IDS),
        "loss_output_order": list(LOSS_ONLY_NAMES),
        "default_parity": parity,
        "productive_reverse_source": _runner_reverse_source_reference(),
        "scalar_reverse_operator": "jax.value_and_grad(loss_total)",
        "vector_reverse_operator": "jax.jacrev(loss_v1_total_plus_six_contributions)",
        "forward_mode_executed": False,
        "parameters": parameter_records,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
        "claim_boundary": (
            "optimizer-congruent reverse-mode simulation diagnostic only; no G4, optimizer, "
            "firmware, hardware, safety, transfer, or flight claim"
        ),
    }


def build_g3_004_smoke(*, validate: bool = True) -> dict[str, Any]:
    """Return the bounded all-seven reverse smoke, optionally before assertions."""
    raw = _g3_004_smoke_raw()
    return validate_g3_004_smoke(raw) if validate else raw


def _mode_rows(
    primal: dict[str, Any],
    plus: dict[str, Any],
    minus: dict[str, Any],
    jvp: dict[str, Any],
    reverse: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for path in primal:
        fd = (plus[path] - minus[path]) / (2.0 * AD_FD_STEP)
        rows.append(
            {
                "output_path": path,
                "primal": _array_payload(primal[path]),
                "theta_plus_h": _array_payload(plus[path]),
                "theta_minus_h": _array_payload(minus[path]),
                "fd": _array_payload(fd),
                "jvp": _array_payload(jvp[path]),
                "reverse": _array_payload(reverse[path]),
                "jvp_fd_comparison": _array_fd_comparison(jvp[path], fd),
                "reverse_fd_comparison": _array_fd_comparison(reverse[path], fd),
            }
        )
    return rows


def _four_modes(function: Callable[[Array], dict[str, Array]]) -> dict[str, Any]:
    evaluate = jax.jit(function)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    tangent = jnp.asarray(1.0, dtype=jnp.float32)
    primal, jvp = jax.jvp(evaluate, (zero,), (tangent,))
    plus = evaluate(jnp.asarray(AD_FD_STEP, dtype=jnp.float32))
    minus = evaluate(jnp.asarray(-AD_FD_STEP, dtype=jnp.float32))
    reverse = jax.jacrev(evaluate)(zero)
    jax.block_until_ready((primal, plus, minus, jvp, reverse))
    rows = _mode_rows(primal, plus, minus, jvp, reverse)
    if not all(row["primal"]["isfinite"] for row in rows):
        raise G3ContractError("G3.3 encountered a nonfinite theta=0 primal")
    if not all(
        row["theta_plus_h"]["isfinite"]
        and row["theta_minus_h"]["isfinite"]
        and row["fd"]["isfinite"]
        for row in rows
    ):
        raise G3ContractError("G3.3 encountered a nonfinite central-FD primal")
    return {
        "rows": rows,
        "primal": primal,
        "plus": plus,
        "minus": minus,
        "jvp": jvp,
        "reverse": reverse,
    }


def _single_focused_inputs(episode_id: str) -> EvaluationInputs:
    selected = tuple(item for item in construct_all_episodes() if item[0].episode_id == episode_id)
    if len(selected) != 1:
        raise G3ContractError(f"focused G3.3 parent {episode_id} is not unique")
    return stack_evaluation_inputs(selected)


def _loss_only_vector(
    sim: Any, data: SimData, inputs: EvaluationInputs, rollout: Callable[..., Any]
) -> Array:
    _, _, outputs = rollout(data, inputs.commands)
    scored = _scored_bundle(inputs, data, outputs)
    return jnp.stack(
        (scored["loss_total"],)
        + tuple(scored["loss_terms"].weighted[name] for name in LOSS_TERM_NAMES),
        axis=1,
    )


def _loss_only_modes(episode_id: str) -> dict[str, Any]:
    inputs = _single_focused_inputs(episode_id)
    sim, initial_data = build_initial_data(inputs)
    rollout = robust.make_instrumented_rollout(sim)

    def function(theta: Array) -> dict[str, Array]:
        data = with_mass_thrust_relaxation(initial_data, theta)
        vector = _loss_only_vector(sim, data, inputs, rollout)[0]
        return {name: vector[index] for index, name in enumerate(LOSS_ONLY_NAMES)}

    modes = _four_modes(function)
    expected_paths = set(LOSS_ONLY_NAMES)
    for mode_name in ("primal", "plus", "minus", "jvp", "reverse"):
        if set(modes[mode_name]) != expected_paths:
            raise G3ContractError(f"{episode_id} G3.3 {mode_name} loss-only keyset changed")
    row_paths = [row["output_path"] for row in modes["rows"]]
    if len(row_paths) != len(expected_paths) or set(row_paths) != expected_paths:
        raise G3ContractError(f"{episode_id} G3.3 loss-only row paths changed")
    rows_by_path = {row["output_path"]: row for row in modes["rows"]}
    modes["rows"] = [rows_by_path[path] for path in LOSS_ONLY_NAMES]
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    scalar_value, scalar_reverse = jax.value_and_grad(lambda theta: function(theta)["loss_total"])(
        zero
    )
    jax.block_until_ready((scalar_value, scalar_reverse))
    total_row = modes["rows"][0]
    total_fd = float(np.asarray(modes["plus"]["loss_total"] - modes["minus"]["loss_total"])) / (
        2.0 * AD_FD_STEP
    )
    vector_total = float(np.asarray(modes["reverse"]["loss_total"]))
    scalar_total = float(np.asarray(scalar_reverse))
    reverse_congruence_gate = _reverse_congruence_gate(
        episode_id, scalar_total, vector_total, total_fd
    )
    serialized_reverse_gate = canonical_json_bytes(reverse_congruence_gate).decode().rstrip()
    reference = G3_003_FD_TOTAL_REFERENCES[episode_id]
    reference_tolerance = AD_FD_ABSOLUTE_TOLERANCE + AD_FD_RELATIVE_TOLERANCE * max(
        abs(total_fd), abs(reference)
    )
    if abs(total_fd) <= 1.0e-3:
        raise G3ContractError(
            f"{episode_id} G3.3 total FD is not significantly excited\n{serialized_reverse_gate}"
        )
    if abs(total_fd - reference) > reference_tolerance:
        raise G3ContractError(
            f"{episode_id} G3.3 FD reference changed: {total_fd} != {reference}\n"
            f"{serialized_reverse_gate}"
        )
    reverse_paths_consistent = reverse_congruence_gate["scalar_vs_vector"]["passed"]
    if not reverse_congruence_gate["passed"]:
        raise G3ContractError(
            f"{episode_id} reverse congruence gate failed\n{serialized_reverse_gate}"
        )
    return {
        "episode_id": episode_id,
        "loss_output_order": list(LOSS_ONLY_NAMES),
        "feature_matrix_constructed": False,
        "rows": modes["rows"],
        "scalar_value_and_grad": {
            "value": _array_payload(scalar_value),
            "gradient": _array_payload(scalar_reverse),
            "fd_comparison": _array_fd_comparison(scalar_reverse, total_fd),
        },
        "reverse_congruence_gate": reverse_congruence_gate,
        "vector_reverse_total_matches_scalar": reverse_paths_consistent,
        "fd_total_reference": reference,
        "fd_total_observed": total_fd,
        "fd_total_reference_tolerance": reference_tolerance,
        "fd_total_significant": True,
        "jvp_all_rows_pass_fd": all(row["jvp_fd_comparison"]["passed"] for row in modes["rows"]),
        "reverse_all_rows_pass_fd": all(
            row["reverse_fd_comparison"]["passed"] for row in modes["rows"]
        )
        and total_row["reverse"]["isfinite"]
        and reverse_congruence_gate["vector_vs_fd"]["passed"],
    }


def _corr01_reverse_congruence_records() -> list[dict[str, Any]]:
    records = [
        _loss_only_modes(episode_id)["reverse_congruence_gate"] for episode_id in FOCUSED_PARENT_IDS
    ]
    if not all(record["passed"] for record in records):
        raise G3ContractError(canonical_json_bytes(records).decode().rstrip())
    return records


def _feature_modes(episode_id: str) -> dict[str, Any]:
    inputs = _single_focused_inputs(episode_id)
    sim, initial_data = build_initial_data(inputs)
    matrix_function, names = _parameter_function(sim, initial_data, inputs, "mass_thrust")

    def function(theta: Array) -> dict[str, Array]:
        matrix = matrix_function(theta)[0]
        return {name: matrix[index] for index, name in enumerate(names)}

    modes = _four_modes(function)
    forward_nonfinite = [row["output_path"] for row in modes["rows"] if not row["jvp"]["isfinite"]]
    reverse_nonfinite = [
        row["output_path"] for row in modes["rows"] if not row["reverse"]["isfinite"]
    ]
    return {
        "episode_id": episode_id,
        "feature_names": list(names),
        "rows": modes["rows"],
        "forward_nonfinite_rows": forward_nonfinite,
        "reverse_nonfinite_rows": reverse_nonfinite,
        "first_forward_nonfinite_row": forward_nonfinite[0] if forward_nonfinite else None,
        "first_reverse_nonfinite_row": reverse_nonfinite[0] if reverse_nonfinite else None,
        "forward_all_rows_pass_fd": all(
            row["jvp_fd_comparison"]["passed"] for row in modes["rows"]
        ),
        "reverse_all_rows_pass_fd": all(
            row["reverse_fd_comparison"]["passed"] for row in modes["rows"]
        ),
    }


def _carry_projection(data: SimData) -> dict[str, Array]:
    state = data.controls.state
    attitude = data.controls.attitude
    force_torque = data.controls.force_torque
    if state is None or attitude is None or force_torque is None:
        raise G3ContractError("G3.3 carry projection requires the complete controller pipeline")
    return {
        "states/pos": data.states.pos,
        "states/quat": data.states.quat,
        "states/vel": data.states.vel,
        "states/ang_vel": data.states.ang_vel,
        "states/rotor_vel": data.states.rotor_vel,
        "controls/rotor_vel": data.controls.rotor_vel,
        "controls/state/cmd": state.cmd,
        "controls/state/staged_cmd": state.staged_cmd,
        "controls/state/pos_err_i": state.pos_err_i,
        "controls/attitude/cmd": attitude.cmd,
        "controls/attitude/staged_cmd": attitude.staged_cmd,
        "controls/attitude/r_int_error": attitude.r_int_error,
        "controls/attitude/last_ang_vel": attitude.last_ang_vel,
        "controls/force_torque/cmd": force_torque.cmd,
        "controls/force_torque/staged_cmd": force_torque.staged_cmd,
    }


def _make_carry_trace(sim: Any) -> Callable[[SimData, Array], dict[str, Array]]:
    pipeline = tuple(sim.step_pipeline.items())
    if tuple(name for name, _ in pipeline) != robust.EXPECTED_PIPELINE:
        raise G3ContractError("G3.3 production pipeline identity changed")

    def one_simulation_step(data: SimData, _: None) -> tuple[SimData, None]:
        for _, function in pipeline:
            data = function(data)
        return data, None

    def interval_step(data: SimData, command: Array) -> tuple[SimData, dict[str, Array]]:
        data = F.state_control(data, command)
        data, _ = jax.lax.scan(one_simulation_step, data, None, length=STEPS_PER_CONTROL, unroll=1)
        data = data.replace(core=data.core.replace(mjx_synced=False))
        return data, _carry_projection(data)

    @jax.jit
    def rollout(initial_data: SimData, commands: Array) -> dict[str, Array]:
        _, trace = jax.lax.scan(interval_step, initial_data, commands)
        return trace

    return rollout


def _bisect_first_false(predicate: Callable[[int], bool], stop: int) -> dict[str, Any]:
    examined: list[int] = []
    if predicate(stop):
        return {
            "last_finite_prefix": stop,
            "first_nonfinite_prefix": None,
            "examined_prefixes": [stop],
        }
    low = 0
    high = stop
    while high - low > 1:
        middle = (low + high) // 2
        examined.append(middle)
        if predicate(middle):
            low = middle
        else:
            high = middle
    examined.extend((low, high))
    return {
        "last_finite_prefix": low,
        "first_nonfinite_prefix": high,
        "examined_prefixes": examined,
    }


def _prefix_path_sums(trace: dict[str, Array], prefix: Array) -> Array:
    index = prefix - jnp.asarray(1, dtype=prefix.dtype)
    return jnp.stack(tuple(jnp.sum(value[index]) for value in trace.values()))


def _first_nonfinite_record(
    *,
    episode_id: str,
    mode: str,
    boundary: dict[str, Any],
    paths: tuple[str, ...],
    primal_trace: dict[str, Any],
    plus_trace: dict[str, Any],
    minus_trace: dict[str, Any],
    jvp_trace: dict[str, Any],
    reverse_path_values: np.ndarray | None,
    pipeline_section: str,
) -> dict[str, Any] | None:
    prefix = boundary["first_nonfinite_prefix"]
    if prefix is None:
        return None
    index = prefix - 1
    derivative_values = (
        [np.asarray(jvp_trace[path])[index] for path in paths]
        if mode == "forward"
        else list(np.asarray(reverse_path_values))
    )
    affected = [
        path
        for path, value in zip(paths, derivative_values, strict=True)
        if not np.all(np.isfinite(value))
    ]
    output_path = affected[0] if affected else "aggregate_projection"
    path_index = paths.index(output_path) if output_path in paths else 0
    primal_value = np.asarray(primal_trace[paths[path_index]])[index]
    plus_value = np.asarray(plus_trace[paths[path_index]])[index]
    minus_value = np.asarray(minus_trace[paths[path_index]])[index]
    fd_value = (plus_value - minus_value) / (2.0 * AD_FD_STEP)
    derivative_value = derivative_values[path_index]
    record = {
        "episode_id": episode_id,
        "ad_mode": mode,
        "last_fully_finite_prefix_index": boundary["last_finite_prefix"],
        "last_fully_finite_time_s": boundary["last_finite_prefix"] / CONTROL_FREQUENCY_HZ,
        "first_nonfinite_prefix_index": prefix,
        "first_nonfinite_time_s": prefix / CONTROL_FREQUENCY_HZ,
        "examined_prefixes": boundary["examined_prefixes"],
        "narrowest_pipeline_section": pipeline_section,
        "output_path": output_path,
        "shape": list(np.asarray(primal_value).shape),
        "dtype": str(np.asarray(primal_value).dtype),
        "projection": "raw scalar" if np.asarray(primal_value).shape == () else "sum projection",
        "primal": _array_payload(primal_value),
        "fd": _array_payload(fd_value),
        "derivative": _array_payload(derivative_value),
        "reproduction_command": G3_003_REPRODUCTION_COMMAND,
    }
    record["run_digest"] = sha256_bytes(canonical_json_bytes(record))
    return record


def _carry_modes(episode_id: str) -> dict[str, Any]:
    inputs = _single_focused_inputs(episode_id)
    sim, initial_data = build_initial_data(inputs)
    rollout = _make_carry_trace(sim)

    def function(theta: Array) -> dict[str, Array]:
        return rollout(with_mass_thrust_relaxation(initial_data, theta), inputs.commands)

    evaluate = jax.jit(function)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    one = jnp.asarray(1.0, dtype=jnp.float32)
    primal, jvp = jax.jvp(evaluate, (zero,), (one,))
    plus = evaluate(jnp.asarray(AD_FD_STEP, dtype=jnp.float32))
    minus = evaluate(jnp.asarray(-AD_FD_STEP, dtype=jnp.float32))
    jax.block_until_ready((primal, jvp, plus, minus))
    paths = tuple(primal)
    if not all(np.all(np.isfinite(np.asarray(value))) for value in primal.values()):
        raise G3ContractError(f"{episode_id} carry primal is nonfinite")
    if not all(
        np.all(np.isfinite(np.asarray(value))) for tree in (plus, minus) for value in tree.values()
    ):
        raise G3ContractError(f"{episode_id} carry FD primal is nonfinite")
    forward_prefix_finite = np.ones(ROLLOUT_INTERVALS + 1, dtype=bool)
    for prefix in range(1, ROLLOUT_INTERVALS + 1):
        forward_prefix_finite[prefix] = all(
            np.all(np.isfinite(np.asarray(jvp[path])[prefix - 1])) for path in paths
        )
    cumulative_forward = np.logical_and.accumulate(forward_prefix_finite)
    forward_boundary = _bisect_first_false(
        lambda prefix: bool(cumulative_forward[prefix]), ROLLOUT_INTERVALS
    )

    def reverse_summary(theta: Array, prefix: Array) -> Array:
        return jnp.sum(_prefix_path_sums(evaluate(theta), prefix))

    reverse_value_and_grad = jax.jit(jax.value_and_grad(reverse_summary, argnums=0))
    reverse_cache: dict[int, bool] = {0: True}

    def reverse_finite(prefix: int) -> bool:
        if prefix not in reverse_cache:
            value, gradient = reverse_value_and_grad(zero, jnp.asarray(prefix, dtype=jnp.int32))
            jax.block_until_ready((value, gradient))
            reverse_cache[prefix] = bool(
                np.all(np.isfinite(np.asarray(value))) and np.all(np.isfinite(np.asarray(gradient)))
            )
        return reverse_cache[prefix]

    reverse_boundary = _bisect_first_false(reverse_finite, ROLLOUT_INTERVALS)

    def path_sums(theta: Array, prefix: Array) -> Array:
        return _prefix_path_sums(evaluate(theta), prefix)

    reverse_path_values = None
    reverse_prefix = reverse_boundary["first_nonfinite_prefix"]
    if reverse_prefix is not None:
        reverse_path_values = np.asarray(
            jax.jacrev(path_sums, argnums=0)(zero, jnp.asarray(reverse_prefix, dtype=jnp.int32))
        )
    forward_record = _first_nonfinite_record(
        episode_id=episode_id,
        mode="forward",
        boundary=forward_boundary,
        paths=paths,
        primal_trace=primal,
        plus_trace=plus,
        minus_trace=minus,
        jvp_trace=jvp,
        reverse_path_values=None,
        pipeline_section="pending interval pipeline isolation",
    )
    reverse_record = _first_nonfinite_record(
        episode_id=episode_id,
        mode="reverse",
        boundary=reverse_boundary,
        paths=paths,
        primal_trace=primal,
        plus_trace=plus,
        minus_trace=minus,
        jvp_trace=jvp,
        reverse_path_values=reverse_path_values,
        pipeline_section="pending interval pipeline isolation",
    )
    return {
        "episode_id": episode_id,
        "paths": list(paths),
        "forward_boundary": forward_boundary,
        "reverse_boundary": reverse_boundary,
        "reverse_prefix_finite_cache": {
            str(prefix): reverse_cache[prefix] for prefix in sorted(reverse_cache)
        },
        "forward_first_nonfinite": forward_record,
        "reverse_first_nonfinite": reverse_record,
        "optimizer_congruent_reverse_operator": "jax.value_and_grad",
        "_trace_values": (primal, plus, minus, jvp),
    }


def _prefix_carry(sim: Any, data: SimData, commands: Array) -> SimData:
    pipeline = tuple(sim.step_pipeline.items())

    def one_simulation_step(carry: SimData, _: None) -> tuple[SimData, None]:
        for _, function in pipeline:
            carry = function(carry)
        return carry, None

    def interval_step(carry: SimData, command: Array) -> tuple[SimData, None]:
        carry = F.state_control(carry, command)
        carry, _ = jax.lax.scan(
            one_simulation_step, carry, None, length=STEPS_PER_CONTROL, unroll=1
        )
        carry = carry.replace(core=carry.core.replace(mjx_synced=False))
        return carry, None

    if commands.shape[0] == 0:
        return data
    final, _ = jax.lax.scan(interval_step, data, commands)
    return final


def _direct_state_outputs(data: SimData, command: Array) -> dict[str, Array]:
    state = data.controls.state
    if state is None:
        raise G3ContractError("G3.3 direct state-controller output is unavailable")
    command_rpyt, pos_err_i = state2attitude(
        data.states.pos,
        data.states.quat,
        data.states.vel,
        command,
        pos_err_i=state.pos_err_i,
        ctrl_freq=state.freq,
        **state.params,
    )
    return {
        "direct_state_controller/command_rpyt": command_rpyt,
        "direct_state_controller/pos_err_i": pos_err_i,
    }


def _pipeline_interval_function(
    sim: Any, initial_data: SimData, inputs: EvaluationInputs, prefix: int
) -> Callable[[Array], dict[str, Array]]:
    if not 1 <= prefix <= ROLLOUT_INTERVALS:
        raise G3ContractError(f"invalid G3.3 pipeline prefix {prefix}")
    pipeline = tuple(sim.step_pipeline.items())
    if tuple(name for name, _ in pipeline) != robust.EXPECTED_PIPELINE:
        raise G3ContractError("G3.3 pipeline isolation identity changed")
    command = inputs.commands[prefix - 1]
    prior_commands = inputs.commands[: prefix - 1]

    def function(theta: Array) -> dict[str, Array]:
        data = with_mass_thrust_relaxation(initial_data, theta)
        data = _prefix_carry(sim, data, prior_commands)
        result = _direct_state_outputs(data, command)
        data = F.state_control(data, command)
        state_outputs: dict[str, list[Array]] = {
            "state_controller/attitude_staged_cmd": [],
            "state_controller/state_cmd": [],
            "state_controller/pos_err_i": [],
        }
        attitude_outputs: dict[str, list[Array]] = {
            "attitude_stage_a_diagnostic/wrench_preclip": [],
            "attitude_stage_a_diagnostic/motor_pwm_preclip": [],
            "attitude_controller/production_postclip_wrench": [],
        }
        force_torque_outputs: dict[str, list[Array]] = {
            "force_torque_stage_b_diagnostic/motor_force_preclip_n": [],
            "force_torque_stage_b_diagnostic/commanded_motor_speed": [],
            "force_torque_controller/production_commanded_motor_speed": [],
        }
        integration_outputs: dict[str, list[Array]] = {
            "dynamics_integration/states_pos": [],
            "dynamics_integration/states_quat": [],
            "dynamics_integration/states_vel": [],
            "dynamics_integration/states_ang_vel": [],
            "dynamics_integration/states_rotor_vel": [],
        }
        for _ in range(STEPS_PER_CONTROL):
            stage_a = None
            stage_b = None
            for name, pipeline_function in pipeline:
                if name == "attitude_controller":
                    stage_a = robust.stage_a_diagnostic(data)
                elif name == "force_torque_controller":
                    stage_b = robust.stage_b_diagnostic(data)
                data = pipeline_function(data)
                if name == "state_controller":
                    state = data.controls.state
                    attitude = data.controls.attitude
                    assert state is not None and attitude is not None
                    state_outputs["state_controller/attitude_staged_cmd"].append(
                        attitude.staged_cmd
                    )
                    state_outputs["state_controller/state_cmd"].append(state.cmd)
                    state_outputs["state_controller/pos_err_i"].append(state.pos_err_i)
                elif name == "attitude_controller":
                    assert stage_a is not None and data.controls.force_torque is not None
                    attitude_outputs["attitude_stage_a_diagnostic/wrench_preclip"].append(
                        stage_a["wrench_preclip"]
                    )
                    attitude_outputs["attitude_stage_a_diagnostic/motor_pwm_preclip"].append(
                        stage_a["motor_pwm_preclip"]
                    )
                    attitude_outputs["attitude_controller/production_postclip_wrench"].append(
                        data.controls.force_torque.staged_cmd
                    )
                elif name == "force_torque_controller":
                    assert stage_b is not None
                    force_torque_outputs[
                        "force_torque_stage_b_diagnostic/motor_force_preclip_n"
                    ].append(stage_b["motor_force_preclip_n"])
                    force_torque_outputs[
                        "force_torque_stage_b_diagnostic/commanded_motor_speed"
                    ].append(stage_b["commanded_motor_speed"])
                    force_torque_outputs[
                        "force_torque_controller/production_commanded_motor_speed"
                    ].append(data.controls.rotor_vel)
                elif name == "integration":
                    integration_outputs["dynamics_integration/states_pos"].append(data.states.pos)
                    integration_outputs["dynamics_integration/states_quat"].append(data.states.quat)
                    integration_outputs["dynamics_integration/states_vel"].append(data.states.vel)
                    integration_outputs["dynamics_integration/states_ang_vel"].append(
                        data.states.ang_vel
                    )
                    integration_outputs["dynamics_integration/states_rotor_vel"].append(
                        data.states.rotor_vel
                    )
        data = data.replace(core=data.core.replace(mjx_synced=False))
        for collection in (
            state_outputs,
            attitude_outputs,
            force_torque_outputs,
            integration_outputs,
        ):
            result.update({key: jnp.stack(values) for key, values in collection.items()})
        for key, value in _carry_projection(data).items():
            result[f"resulting_state_carry/{key}"] = value
        return result

    return function


def _pipeline_section(path: str) -> str:
    return path.split("/", 1)[0]


def _pipeline_modes(
    episode_id: str, forward_prefix: int | None, reverse_prefix: int | None
) -> dict[str, Any]:
    inputs = _single_focused_inputs(episode_id)
    sim, initial_data = build_initial_data(inputs)
    prefixes = sorted({prefix for prefix in (forward_prefix, reverse_prefix) if prefix is not None})
    if not prefixes:
        prefixes = [ROLLOUT_INTERVALS]
    records = {}
    for prefix in prefixes:
        modes = _four_modes(_pipeline_interval_function(sim, initial_data, inputs, prefix))
        forward_nonfinite = [
            row["output_path"] for row in modes["rows"] if not row["jvp"]["isfinite"]
        ]
        reverse_nonfinite = [
            row["output_path"] for row in modes["rows"] if not row["reverse"]["isfinite"]
        ]
        records[str(prefix)] = {
            "prefix_index": prefix,
            "time_s": prefix / CONTROL_FREQUENCY_HZ,
            "rows": modes["rows"],
            "forward_first_nonfinite_path": (forward_nonfinite[0] if forward_nonfinite else None),
            "forward_narrowest_section": (
                _pipeline_section(forward_nonfinite[0]) if forward_nonfinite else None
            ),
            "reverse_first_nonfinite_path": (reverse_nonfinite[0] if reverse_nonfinite else None),
            "reverse_narrowest_section": (
                _pipeline_section(reverse_nonfinite[0]) if reverse_nonfinite else None
            ),
            "operation_claim": None,
            "operation_claim_reason": (
                "G3.3 reports the narrowest stable pipeline/output boundary; it does not name "
                "an individual operation without two-run input/output isolation."
            ),
        }
    return {"episode_id": episode_id, "prefixes": records}


def _disclose_split_localization(
    record: dict[str, Any], interval: dict[str, Any], mode: str
) -> None:
    if mode not in {"forward", "reverse"}:
        raise G3ContractError(f"unsupported G3.3 localization mode {mode}")
    if record["derivative"]["isfinite"]:
        raise G3ContractError("G3.3 first-nonfinite record has a finite rollout derivative")
    derivative_name = "jvp" if mode == "forward" else "reverse"
    path_key = f"{mode}_first_nonfinite_path"
    section_key = f"{mode}_narrowest_section"
    pipeline_output_path = interval[path_key]
    split_interval_derivatives_all_finite = all(
        row[derivative_name]["isfinite"] for row in interval["rows"]
    )
    if pipeline_output_path is None:
        if not split_interval_derivatives_all_finite or interval[section_key] is not None:
            raise G3ContractError("G3.3 split interval did not reproduce a unique finite boundary")
        localization_status = "NOT_REPRODUCED_IN_SPLIT_INTERVAL"
        narrowest_pipeline_section = None
        localization_claim_boundary = "full_rollout_carry_output"
    else:
        if not isinstance(pipeline_output_path, str) or not pipeline_output_path:
            raise G3ContractError("G3.3 localized pipeline output path is empty")
        matching_rows = [
            row for row in interval["rows"] if row["output_path"] == pipeline_output_path
        ]
        expected_section = _pipeline_section(pipeline_output_path)
        if (
            len(matching_rows) != 1
            or matching_rows[0][derivative_name]["isfinite"]
            or interval[section_key] != expected_section
        ):
            raise G3ContractError(
                "G3.3 localized pipeline boundary lacks a nonfinite split derivative"
            )
        localization_status = "LOCALIZED"
        narrowest_pipeline_section = expected_section
        localization_claim_boundary = "split_interval_pipeline_output"
    record.update(
        {
            "localization_status": localization_status,
            "pipeline_output_path": pipeline_output_path,
            "narrowest_pipeline_section": narrowest_pipeline_section,
            "split_interval_derivatives_all_finite": (split_interval_derivatives_all_finite),
            "localization_claim_boundary": localization_claim_boundary,
            "individual_operation_claim": None,
            "individual_operation_claim_permitted": False,
        }
    )
    record["run_digest"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in record.items() if key != "run_digest"})
    )


def _runner_reverse_source_reference() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[4]
    runner_path = repository_root / "crazyflow/control/mellinger/research/runner.py"
    lines = runner_path.read_text().splitlines()
    expected = "jax.value_and_grad(objective, has_aux=True)(candidate)"
    matches = [index + 1 for index, line in enumerate(lines) if expected in line]
    if matches != [255]:
        raise G3ContractError(f"productive reverse source reference changed: {matches}")
    if not lines[162].startswith("def run_training_validation("):
        raise G3ContractError("productive reverse enclosing function changed")
    return {
        "file": "crazyflow/control/mellinger/research/runner.py",
        "function": "run_training_validation.train_step.objective",
        "line": 255,
        "source": "jax.value_and_grad(objective, has_aux=True)(candidate)",
        "diagnostic_scalar_operator": "jax.value_and_grad(loss_total)",
        "diagnostic_vector_operator": "jax.jacrev(loss_only_vector)",
        "optimizer_library_imported_by_diagnostic": False,
        "optimizer_state_created_by_diagnostic": False,
    }


def _current_draft_digests() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[4]
    return {
        path: sha256_bytes((repository_root / path).read_bytes()) for path in G3_003_DRAFT_PATHS
    }


def _parent_contract(episode_id: str) -> dict[str, Any]:
    inputs = _single_focused_inputs(episode_id)
    spec = inputs.specs[0]
    candidate = inputs.candidates[0]
    return {
        "episode_id": episode_id,
        "split": spec.split.value,
        "motion_class": spec.motion_class.value,
        "profile": spec.profile.value,
        "seed": spec.seed,
        "attempt": candidate.attempt,
        "warmup_s": spec.warmup_s,
        "rollout_intervals": ROLLOUT_INTERVALS,
        "scored_window": [spec.score_start, spec.score_stop],
        "parent_digest": candidate.parent_digest,
        "array_digests": candidate.array_digests,
    }


def _diagnosis_classification(loss: list[dict[str, Any]], feature: list[dict[str, Any]]) -> str:
    loss_forward_pass = all(item["jvp_all_rows_pass_fd"] for item in loss)
    loss_reverse_pass = all(item["reverse_all_rows_pass_fd"] for item in loss)
    feature_all_pass = all(
        item["forward_all_rows_pass_fd"] and item["reverse_all_rows_pass_fd"] for item in feature
    )
    if loss_reverse_pass and not loss_forward_pass:
        return "FORWARD_ONLY"
    if not loss_reverse_pass and not loss_forward_pass:
        return "REVERSE_AND_FORWARD"
    if loss_reverse_pass and loss_forward_pass and not feature_all_pass:
        return "FEATURE_ONLY"
    return "UNRESOLVED"


@lru_cache(maxsize=1)
def build_g3_003_diagnosis() -> dict[str, Any]:
    """Run the bounded two-parent G3.3 NaN diagnosis without optimizer activity."""
    parity = focused_default_parity()
    if parity["status"] != "PASS_FOCUSED_DEFAULT_PARITY":
        raise G3ContractError("G3.3 requires the retained theta=0 focused parity")
    loss = [_loss_only_modes(episode_id) for episode_id in FOCUSED_PARENT_IDS]
    feature = [_feature_modes(episode_id) for episode_id in FOCUSED_PARENT_IDS]
    carry = [_carry_modes(episode_id) for episode_id in FOCUSED_PARENT_IDS]
    pipeline = []
    for item in carry:
        forward = item["forward_boundary"]["first_nonfinite_prefix"]
        reverse = item["reverse_boundary"]["first_nonfinite_prefix"]
        localized = _pipeline_modes(item["episode_id"], forward, reverse)
        pipeline.append(localized)
        prefix_records = localized["prefixes"]
        if item["forward_first_nonfinite"] is not None:
            interval = prefix_records[str(forward)]
            _disclose_split_localization(item["forward_first_nonfinite"], interval, "forward")
        if item["reverse_first_nonfinite"] is not None:
            interval = prefix_records[str(reverse)]
            _disclose_split_localization(item["reverse_first_nonfinite"], interval, "reverse")
        item.pop("_trace_values")
    classification = _diagnosis_classification(loss, feature)
    payload = {
        "diagnosis_version": G3_003_DIAGNOSIS_VERSION,
        "work_order": "WO-GR-G3-003",
        "source_identity": {
            "base_commit": G3_003_SOURCE_BASE_COMMIT,
            "checkout_branch_included": False,
            "absolute_worktree_path_included": False,
            "input_draft_sha256": G3_003_INPUT_DIGESTS,
            "final_draft_sha256": _current_draft_digests(),
        },
        "platform": PLATFORM,
        "backend": "cpu",
        "jax_enable_x64": False,
        "float_execution": "Float32",
        "coordinate": "mass_thrust(theta)=float32(132000)*exp(theta)",
        "theta_default": 0.0,
        "h": AD_FD_STEP,
        "ad_fd_tolerance": {
            "formula": "abs(ad-fd) <= 2e-4 + 0.05*max(abs(ad),abs(fd))",
            "absolute": AD_FD_ABSOLUTE_TOLERANCE,
            "relative": AD_FD_RELATIVE_TOLERANCE,
        },
        "parents": [_parent_contract(episode_id) for episode_id in FOCUSED_PARENT_IDS],
        "default_parity": parity,
        "productive_reverse_source": _runner_reverse_source_reference(),
        "loss_only": loss,
        "feature_matrix": feature,
        "rollout_carry": carry,
        "controller_pipeline": pipeline,
        "classification": classification,
        "classification_is_diagnostic_only": True,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
        "resource_contract": {
            "maximum_runtime_seconds": 600,
            "maximum_rss_bytes": 24 * 1024**3,
            "focus_parent_count": 2,
            "theta_values": [-AD_FD_STEP, 0.0, AD_FD_STEP],
        },
        "claim_boundaries": [
            "Simulation-only localization of derivatives through the fixed Float32 CPU cases.",
            "No mathematical operation is declared physically wrong or corrected.",
            "Historical G3.2 focus gate 2 remains BLOCKED/WITHHELD.",
            "No G3.2 continuation, G4, candidate, firmware, hardware, or flight permission.",
        ],
        "reproduction_command": G3_003_REPRODUCTION_COMMAND,
    }
    if classification not in {"FORWARD_ONLY", "REVERSE_AND_FORWARD", "FEATURE_ONLY", "UNRESOLVED"}:
        raise G3ContractError(f"invalid G3.3 classification {classification}")
    if payload["optimizer_initialization_count"] or payload["optimizer_update_count"]:
        raise G3ContractError("G3.3 optimizer contract was violated")
    return payload


@lru_cache(maxsize=1)
def discrete_integer_response() -> dict[str, Any]:
    """Measure rounded integer responses only after the G3.4 reverse smoke passes."""
    smoke = build_g3_004_smoke()
    if smoke["status"] != "PASS_ALL_SEVEN_PARAMETER_REVERSE_SMOKE":
        raise G3ContractError("all-seven reverse smoke must pass before discrete response")
    inputs = focused_inputs()
    sim, initial_data = build_initial_data(inputs)
    original_dtype = np.asarray(_state_params(initial_data)["mass_thrust"]).dtype
    records = []
    for step in (AD_FD_STEP, EFFECT_STEP):
        minus_theta = np.float32(-step)
        plus_theta = np.float32(step)
        base = np.float32(CANONICAL_MASS_THRUST)
        q_minus = int(np.rint(base * np.exp(minus_theta)).astype(original_dtype))
        q_plus = int(np.rint(base * np.exp(plus_theta)).astype(original_dtype))
        loss_minus = _loss_only(sim, _with_discrete_mass_thrust(initial_data, q_minus), inputs)
        loss_plus = _loss_only(sim, _with_discrete_mass_thrust(initial_data, q_plus), inputs)
        integer_span = q_plus - q_minus
        records.append(
            {
                "theta_magnitude": step,
                "q_minus": q_minus,
                "q_plus": q_plus,
                "realized_log_step_minus": math.log(q_minus / CANONICAL_MASS_THRUST),
                "realized_log_step_plus": math.log(q_plus / CANONICAL_MASS_THRUST),
                "loss_minus": loss_minus.tolist(),
                "loss_plus": loss_plus.tolist(),
                "symmetric_quotient_per_integer_unit": (
                    (loss_plus - loss_minus) / integer_span
                ).tolist(),
                "theta_normalized_secant": ((loss_plus - loss_minus) / (2.0 * step)).tolist(),
            }
        )
    payload = {
        "status": "DISCRETE_INTEGER_RESPONSE",
        "category": "DISCRETE_DIAGNOSTIC_ONLY",
        "rounding": "IEEE/JAX round-to-nearest-even",
        "original_dtype": str(original_dtype),
        "records": records,
        "continuous_ad_validation": False,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }
    forbidden = {"ad", "gradient", "go_label"}
    if forbidden & payload.keys():
        raise G3ContractError("discrete response schema contains a continuous AD/GO field")
    return payload


def _full_default_parity(
    batch_label: str, sim: Any, initial_data: SimData, inputs: EvaluationInputs
) -> tuple[dict[str, Any], tuple[str, ...], np.ndarray, Any]:
    """Require original-integer and relaxed-Float32 identity for one complete batch."""
    relaxed = with_mass_thrust_relaxation(initial_data, 0.0)
    original = _instrumented_evaluation(sim, initial_data, inputs)
    relaxed_result = _instrumented_evaluation(sim, relaxed, inputs)
    _require_relaxed_parity(f"{batch_label} final carry", original[0], relaxed_result[0])
    for index, (left, right) in enumerate(zip(original[1], relaxed_result[1], strict=True)):
        _require_relaxed_parity(f"{batch_label} boundary carry {index}", left, right)
    robust.require_pytree_equal(f"{batch_label} rollout outputs", original[2], relaxed_result[2])
    robust.require_pytree_equal(f"{batch_label} scored bundle", original[3], relaxed_result[3])
    if original[4] != relaxed_result[4]:
        raise G3ContractError(f"{batch_label} feature inventory changed under relaxation")
    robust.require_array_equal(f"{batch_label} feature parity", original[5], relaxed_result[5])
    integral_limits = np.asarray(_state_params(initial_data)["int_err_max"])
    technical = robust._full_rollout_gate(
        inputs.specs, relaxed_result[2], f"{batch_label}/relaxed_default", integral_limits
    )
    return (
        {
            "status": "PASS_BATCH_DEFAULT_PARITY",
            "batch_label": batch_label,
            "parent_ids": [item.episode_id for item in inputs.specs],
            "parent_count": len(inputs.specs),
            "array_equal": True,
            "allowed_internal_exception": "mass_thrust integer-vs-Float32 dtype only",
            "technical_gates": technical,
            "technical_details": _technical_detail_records(inputs.specs, relaxed_result[2]),
        },
        relaxed_result[4],
        np.asarray(relaxed_result[5], dtype=np.float64),
        relaxed_result[2],
    )


def _technical_detail_records(specs: tuple[EpisodeSpec, ...], outputs: Any) -> list[dict[str, Any]]:
    """Describe clip side/axis/rate, reserve, and wrench without changing hard gates."""
    stage_a = outputs["diagnostic"]["stage_a"]
    stage_b = outputs["diagnostic"]["stage_b"]
    records = []
    for world, spec in enumerate(specs):
        interval = slice(spec.score_start, spec.score_stop)

        def flattened(tree: dict[str, Any], name: str) -> np.ndarray:
            value = np.asarray(tree[name])[interval, :, world, 0]
            return value.reshape((-1, *value.shape[2:]))

        stage_a_motor_lower = flattened(stage_a, "motor_pwm_lower_clip_mask")
        stage_a_motor_upper = flattened(stage_a, "motor_pwm_upper_clip_mask")
        stage_a_torque_lower = flattened(stage_a, "torque_lower_clip_mask")
        stage_a_torque_upper = flattened(stage_a, "torque_upper_clip_mask")
        stage_b_motor_lower = flattened(stage_b, "motor_force_lower_clip_mask")
        stage_b_motor_upper = flattened(stage_b, "motor_force_upper_clip_mask")
        lower_reserve = flattened(stage_a, "force_lower_reserve_normalized")
        upper_reserve = flattened(stage_a, "force_upper_reserve_normalized")
        stage_a_distortion = flattened(stage_a, "wrench_distortion_absolute")
        stage_b_distortion = flattened(stage_b, "wrench_distortion_absolute")
        records.append(
            {
                "episode_id": spec.episode_id,
                "split": spec.split.value,
                "motion_class": spec.motion_class.value,
                "profile": spec.profile.value,
                "stage_a_motor_lower": robust.mask_summary(
                    stage_a_motor_lower,
                    start_time_s=spec.warmup_s,
                    frequency_hz=SIMULATION_FREQUENCY_HZ,
                    channel_name="motor_index",
                ),
                "stage_a_motor_upper": robust.mask_summary(
                    stage_a_motor_upper,
                    start_time_s=spec.warmup_s,
                    frequency_hz=SIMULATION_FREQUENCY_HZ,
                    channel_name="motor_index",
                ),
                "stage_a_torque_lower": robust.mask_summary(
                    stage_a_torque_lower,
                    start_time_s=spec.warmup_s,
                    frequency_hz=SIMULATION_FREQUENCY_HZ,
                    channel_name="axis_index",
                ),
                "stage_a_torque_upper": robust.mask_summary(
                    stage_a_torque_upper,
                    start_time_s=spec.warmup_s,
                    frequency_hz=SIMULATION_FREQUENCY_HZ,
                    channel_name="axis_index",
                ),
                "stage_b_motor_lower": robust.mask_summary(
                    stage_b_motor_lower,
                    start_time_s=spec.warmup_s,
                    frequency_hz=SIMULATION_FREQUENCY_HZ,
                    channel_name="motor_index",
                ),
                "stage_b_motor_upper": robust.mask_summary(
                    stage_b_motor_upper,
                    start_time_s=spec.warmup_s,
                    frequency_hz=SIMULATION_FREQUENCY_HZ,
                    channel_name="motor_index",
                ),
                "minimum_normalized_motor_force_reserve": float(
                    np.min(np.minimum(lower_reserve, upper_reserve))
                ),
                "maximum_stage_a_wrench_distortion_by_axis": np.max(
                    stage_a_distortion, axis=0
                ).tolist(),
                "maximum_stage_b_wrench_distortion_by_axis": np.max(
                    stage_b_distortion, axis=0
                ).tolist(),
                "near_limit_is_descriptive_only": spec.motion_class is MotionClass.NEAR_LIMIT,
            }
        )
    return records


def _slice_effect_outputs_for_parent(outputs: Any, world: int) -> dict[str, Any]:
    """Slice exactly one world while preserving every time and substep axis."""
    return {
        "trace": jax.tree.map(lambda value: value[:, world : world + 1], outputs["trace"]),
        "diagnostic": jax.tree.map(
            lambda value: value[:, :, world : world + 1], outputs["diagnostic"]
        ),
        "pos_err_i": outputs["pos_err_i"][:, world : world + 1],
    }


def _scored_window_relation(index: int, spec: EpisodeSpec) -> str:
    if index < spec.score_start:
        return "BEFORE"
    if index < spec.score_stop:
        return "INSIDE"
    return "AFTER"


def _nonfinite_effect_leaf_records(
    outputs: Any, world: int, spec: EpisodeSpec
) -> list[dict[str, Any]]:
    records = []
    for root, world_axis in (("trace", 1), ("diagnostic", 2)):
        for path, array in _leaf_records(outputs[root]):
            if not np.issubdtype(array.dtype, np.number):
                continue
            single = np.take(array, (world,), axis=world_axis)
            affected = np.argwhere(~np.isfinite(single))
            if affected.size == 0:
                continue
            single_index = [int(value) for value in affected[0]]
            full_index = list(single_index)
            full_index[world_axis] = world
            control_index = single_index[0]
            simulation_substep_index = single_index[1] if root == "diagnostic" else None
            physical_time_seconds = (
                (control_index * STEPS_PER_CONTROL + simulation_substep_index)
                / SIMULATION_FREQUENCY_HZ
                if simulation_substep_index is not None
                else (control_index + 1) / CONTROL_FREQUENCY_HZ
            )
            records.append(
                {
                    "path": f"{root}{path}",
                    "full_batch_shape": list(array.shape),
                    "single_world_shape": list(single.shape),
                    "dtype": str(array.dtype),
                    "world_axis": world_axis,
                    "nonfinite_count": int(affected.shape[0]),
                    "first_nonfinite_full_array_index": full_index,
                    "first_nonfinite_single_world_array_index": single_index,
                    "control_interval_index": control_index,
                    "simulation_substep_index": simulation_substep_index,
                    "index_interpretation": (
                        "diagnostic control interval, simulation substep, single-world axis, "
                        "then canonical leaf axes"
                        if root == "diagnostic"
                        else "rollout control interval, single-world axis, then canonical leaf axes"
                    ),
                    "scored_window_relation": _scored_window_relation(control_index, spec),
                    "physical_time_seconds": physical_time_seconds,
                    "physical_time_mapping_status": ("PROVEN_BY_EXISTING_ROLLOUT_AXIS_CONTRACT"),
                }
            )
    records.sort(key=lambda record: record["path"])
    return records


def _require_effect_gate_prerequisites_finite(outputs: Any, variant: str) -> dict[str, Any]:
    trace = outputs["trace"]
    diagnostic = outputs["diagnostic"]
    required = {
        "trace.pos": np.asarray(trace.pos),
        "pos_err_i": np.asarray(outputs["pos_err_i"]),
        "diagnostic.stage_a.collective_force_request_n": np.asarray(
            diagnostic["stage_a"]["collective_force_request_n"]
        ),
        "diagnostic.stage_a.motor_pwm_any_clip_mask": np.asarray(
            diagnostic["stage_a"]["motor_pwm_any_clip_mask"]
        ),
        "diagnostic.stage_a.torque_any_clip_mask": np.asarray(
            diagnostic["stage_a"]["torque_any_clip_mask"]
        ),
        "diagnostic.stage_b.motor_force_any_clip_mask": np.asarray(
            diagnostic["stage_b"]["motor_force_any_clip_mask"]
        ),
    }
    nonfinite = [name for name, value in required.items() if not np.all(np.isfinite(value))]
    if nonfinite:
        raise G3ContractError(
            f"{variant} has nonfinite data required by an unchanged technical gate: {nonfinite}"
        )
    return {"all_required_arrays_finite": True, "required_paths": sorted(required)}


def _near_limit_nonfinite_gate_record(
    spec: EpisodeSpec, outputs: Any, variant: str, integral_limits: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    prerequisite = _require_effect_gate_prerequisites_finite(outputs, variant)
    trace = outputs["trace"]
    diagnostic = outputs["diagnostic"]
    position = np.asarray(trace.pos)
    stage_a = diagnostic["stage_a"]
    stage_b = diagnostic["stage_b"]
    ground = bool(np.any(position[:, 0, 0, 2] <= -0.001))
    zero_thrust = bool(np.any(np.asarray(stage_a["collective_force_request_n"])[:, :, 0, 0] <= 0.0))
    interval_score = slice(spec.score_start, spec.score_stop)
    stage_a_motor = np.asarray(stage_a["motor_pwm_any_clip_mask"])[interval_score, :, 0, 0]
    stage_a_torque = np.asarray(stage_a["torque_any_clip_mask"])[interval_score, :, 0, 0]
    stage_b_motor = np.asarray(stage_b["motor_force_any_clip_mask"])[interval_score, :, 0, 0]
    score_integral = np.asarray(outputs["pos_err_i"])[interval_score, 0, 0]
    integral_hit = bool(
        np.any(
            np.abs(score_integral)
            >= np.asarray(integral_limits) * (1.0 - robust.INTEGRAL_DETECTION_THRESHOLD)
        )
    )
    motor_summary = robust.mask_summary(
        stage_a_motor.reshape(-1, 4),
        start_time_s=spec.warmup_s,
        frequency_hz=SIMULATION_FREQUENCY_HZ,
        channel_name="motor_index",
    )
    torque_count = int(stage_a_torque.sum())
    stage_b_count = int(stage_b_motor.sum())
    if ground:
        raise G3ContractError(f"{variant}/{spec.episode_id} activates ground/floor gate")
    if zero_thrust:
        raise G3ContractError(f"{variant}/{spec.episode_id} activates zero thrust")
    if spec.motion_class in {MotionClass.SOFT, MotionClass.NOMINAL}:
        if motor_summary["count"] or torque_count or stage_b_count:
            raise G3ContractError(
                f"{variant}/{spec.episode_id} violates soft/nominal zero-clip gate"
            )
    if spec.motion_class is MotionClass.DYNAMIC:
        if stage_b_count:
            raise G3ContractError(
                f"{variant}/{spec.episode_id} has forbidden dynamic Stage-B clipping"
            )
        if motor_summary["fraction"] > 0.0025 or motor_summary["longest_duration_s"] > 0.02:
            raise G3ContractError(f"{variant}/{spec.episode_id} exceeds dynamic Stage-A clip gate")
    return (
        {
            "episode_id": spec.episode_id,
            "finite": False,
            "ground_or_floor_activation": ground,
            "zero_thrust_activation": zero_thrust,
            "stage_a_motor_clip_count_scored": motor_summary["count"],
            "stage_a_motor_clip_fraction_scored": motor_summary["fraction"],
            "stage_a_motor_longest_interval_s_scored": motor_summary["longest_duration_s"],
            "stage_a_torque_clip_count_scored": torque_count,
            "stage_b_additional_clip_count_scored": stage_b_count,
            "integral_clip_detection_scored": integral_hit,
            "integral_contact_interpretation": (
                "descriptive_contact_not_candidate_pass" if integral_hit else "no_contact"
            ),
            "candidate_gate": "withheld_technical_robustness",
            "technical_robustness_eligible": False,
            "reason": NONFINITE_NEAR_LIMIT_EFFECT_REASON,
        },
        prerequisite,
    )


def _effect_technical_parent_adapter(
    batch_label: str,
    specs: tuple[EpisodeSpec, ...],
    outputs: Any,
    variant: str,
    integral_limits: np.ndarray,
    parameter: str,
    theta_label: str,
    theta_value: float,
    physical_parameter_value: float,
) -> dict[str, Any]:
    """Preserve finite gates and report only the D-074 Near-limit effect outcome."""
    if theta_label not in {"minus_0p05", "plus_0p05"} or theta_value not in {
        -EFFECT_STEP,
        EFFECT_STEP,
    }:
        raise G3ContractError("effect technical adapter received an undeclared theta context")
    registry = {item.episode_id: index for index, item in enumerate(episode_specs())}
    gates = []
    details = []
    nonfinite_records = []
    for world, spec in enumerate(specs):
        sliced = _slice_effect_outputs_for_parent(outputs, world)
        affected = _nonfinite_effect_leaf_records(outputs, world, spec)
        pos_err_i_finite = bool(np.all(np.isfinite(np.asarray(sliced["pos_err_i"]))))
        if not affected and pos_err_i_finite:
            gates.extend(robust._full_rollout_gate((spec,), sliced, variant, integral_limits))
            details.extend(_technical_detail_records((spec,), sliced))
            continue
        if spec.motion_class is not MotionClass.NEAR_LIMIT:
            robust._full_rollout_gate((spec,), sliced, variant, integral_limits)
            raise AssertionError("finite gate unexpectedly accepted a nonfinite feasible parent")
        if not pos_err_i_finite:
            raise G3ContractError(f"{variant}/{spec.episode_id} has nonfinite pos_err_i")
        if not affected:
            raise G3ContractError(
                f"{variant}/{spec.episode_id} has nonfinite data outside trace/diagnostic leaves"
            )
        gate, prerequisite = _near_limit_nonfinite_gate_record(
            spec, sliced, variant, integral_limits
        )
        gates.append(gate)
        detail_inputs = (
            np.asarray(sliced["diagnostic"][stage][name])
            for stage, name in (
                ("stage_a", "motor_pwm_lower_clip_mask"),
                ("stage_a", "motor_pwm_upper_clip_mask"),
                ("stage_a", "torque_lower_clip_mask"),
                ("stage_a", "torque_upper_clip_mask"),
                ("stage_b", "motor_force_lower_clip_mask"),
                ("stage_b", "motor_force_upper_clip_mask"),
                ("stage_a", "force_lower_reserve_normalized"),
                ("stage_a", "force_upper_reserve_normalized"),
                ("stage_a", "wrench_distortion_absolute"),
                ("stage_b", "wrench_distortion_absolute"),
            )
        )
        if all(np.all(np.isfinite(value)) for value in detail_inputs):
            details.extend(_technical_detail_records((spec,), sliced))
            detail_status = "AVAILABLE_FROM_FINITE_DETAIL_INPUTS"
        else:
            details.append(
                {
                    "episode_id": spec.episode_id,
                    "split": spec.split.value,
                    "motion_class": spec.motion_class.value,
                    "profile": spec.profile.value,
                    "technical_detail_status": "UNAVAILABLE_NONFINITE_DEPENDENCY",
                    "affected_paths": [record["path"] for record in affected],
                }
            )
            detail_status = "UNAVAILABLE_NONFINITE_DEPENDENCY"
        nonfinite_records.append(
            {
                "schema_version": NONFINITE_EFFECT_SCHEMA,
                "parameter": parameter,
                "theta_label": theta_label,
                "theta_value": float(theta_value),
                "physical_parameter_value": float(physical_parameter_value),
                "physical_parameter_formula": (
                    f"{PARAMETER_SPECS[parameter].default}*exp({float(theta_value)})"
                ),
                "coordinate": (
                    "mass_thrust(theta)=float32(132000)*exp(theta)"
                    if parameter == "mass_thrust"
                    else "p(theta)=p_default*exp(theta)"
                ),
                "batch_label": batch_label,
                "parent_id": spec.episode_id,
                "split": spec.split.value,
                "motion_class": spec.motion_class.value,
                "profile": spec.profile.value,
                "batch_world_index": world,
                "registry_index": registry[spec.episode_id],
                "affected_leaves": affected,
                "technical_detail_status": detail_status,
                "unchanged_gate_prerequisites": prerequisite,
                "ground_or_floor_activation": gate["ground_or_floor_activation"],
                "zero_thrust_activation": gate["zero_thrust_activation"],
                "stage_a_motor_clip_count_scored": gate["stage_a_motor_clip_count_scored"],
                "stage_a_torque_clip_count_scored": gate["stage_a_torque_clip_count_scored"],
                "stage_b_additional_clip_count_scored": gate[
                    "stage_b_additional_clip_count_scored"
                ],
                "technical_robustness_eligible": False,
                "reason": NONFINITE_NEAR_LIMIT_EFFECT_REASON,
                "causal_claim": None,
                "individual_operation_claim": None,
                "claim_boundary": "full_effect_rollout_leaf_metadata_only",
            }
        )
    return {"gates": gates, "details": details, "nonfinite_effect_rollouts": nonfinite_records}


def _require_parameter_diagnostic_arrays_finite(
    name: str, batch_label: str, arrays: dict[str, np.ndarray]
) -> None:
    for context, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise G3ContractError(
                f"{name}/{batch_label}/{context} continuous diagnostics contain nonfinite values"
            )


def _parameter_batch_diagnostic(
    batch_label: str,
    sim: Any,
    initial_data: SimData,
    inputs: EvaluationInputs,
    name: str,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Measure one complete execution batch without applying global scientific gates."""
    isolation = _parameter_isolation(initial_data, name)
    theta_zero_default_parity = _parameter_default_parity(initial_data, name)
    spec = PARAMETER_SPECS[name]
    effect_physical_values = {
        label: spec.default * math.exp(theta)
        for label, theta in (("minus_0p05", -EFFECT_STEP), ("plus_0p05", EFFECT_STEP))
    }
    if not all(spec.lower <= value <= spec.upper for value in effect_physical_values.values()):
        raise G3ContractError(f"{name} effect perturbation leaves diagnostic bounds")
    function, names = _parameter_function(sim, initial_data, inputs, name)
    if names != feature_names:
        raise G3ContractError(f"{name} feature inventory changed")
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    baseline = function(zero)
    reverse = jax.jacrev(function)(zero)
    plus_h = function(jnp.asarray(AD_FD_STEP, dtype=jnp.float32))
    minus_h = function(jnp.asarray(-AD_FD_STEP, dtype=jnp.float32))
    plus_effect = function(jnp.asarray(EFFECT_STEP, dtype=jnp.float32))
    minus_effect = function(jnp.asarray(-EFFECT_STEP, dtype=jnp.float32))
    jax.block_until_ready((baseline, reverse, plus_h, minus_h, plus_effect, minus_effect))
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (baseline, reverse, plus_h, minus_h, plus_effect, minus_effect)
    )
    baseline_np, reverse_np, plus_h_np, minus_h_np, plus_effect_np, minus_effect_np = arrays
    fd_np = (plus_h_np - minus_h_np) / (2.0 * AD_FD_STEP)
    _require_parameter_diagnostic_arrays_finite(
        name,
        batch_label,
        {
            "baseline": baseline_np,
            "reverse": reverse_np,
            "plus_h": plus_h_np,
            "minus_h": minus_h_np,
            "fd": fd_np,
            "effect_plus": plus_effect_np,
            "effect_minus": minus_effect_np,
        },
    )
    total_index = feature_names.index("loss_total")
    scalar_split_aggregates = {}
    for split in (Split.TRAIN, Split.VALIDATION):
        mask = np.asarray(
            [
                item.split is split and item.motion_class is not MotionClass.NEAR_LIMIT
                for item in inputs.specs
            ]
        )
        if not np.any(mask):
            continue
        weights = jnp.asarray(mask, dtype=jnp.float32)
        scalar_value, scalar_reverse = jax.value_and_grad(
            lambda theta, weights=weights: (
                jnp.sum(function(theta)[:, total_index] * weights) / jnp.sum(weights)
            )
        )(zero)
        jax.block_until_ready((scalar_value, scalar_reverse))
        contributing_parent_ids = [
            item.episode_id
            for item, contributes in zip(inputs.specs, mask, strict=True)
            if contributes
        ]
        contributing_parent_count = len(contributing_parent_ids)
        if contributing_parent_count <= 0:
            raise G3ContractError(f"{name}/{batch_label}/{split.value} has no contributors")
        scalar_split_aggregates[split.value] = {
            "batch_label": batch_label,
            "contributing_parent_ids": contributing_parent_ids,
            "contributing_parent_count": contributing_parent_count,
            "batch_statistic_denominator": contributing_parent_count,
            "original_statistic_denominator_semantics": (
                "count of feasible parents in the complete frozen split"
            ),
            "loss_total_scalar_value": float(np.asarray(scalar_value)),
            "loss_total_scalar_reverse": float(np.asarray(scalar_reverse)),
        }
    technical = {}
    integral_limits = np.asarray(_state_params(initial_data)["int_err_max"])
    rollout = robust.make_instrumented_rollout(sim)
    for theta_label, theta in (("minus_0p05", -EFFECT_STEP), ("plus_0p05", EFFECT_STEP)):
        data = _replace_parameter(initial_data, name, theta)
        _, _, outputs = rollout(data, inputs.commands)
        jax.block_until_ready(outputs)
        technical[theta_label] = _effect_technical_parent_adapter(
            batch_label,
            inputs.specs,
            outputs,
            f"{name}/{batch_label}/{theta_label}",
            integral_limits,
            name,
            theta_label,
            theta,
            effect_physical_values[theta_label],
        )
    return {
        "batch_label": batch_label,
        "parent_ids": [item.episode_id for item in inputs.specs],
        "parent_count": len(inputs.specs),
        "parameter": name,
        "isolation": isolation,
        "theta_zero_default_parity": theta_zero_default_parity,
        "scalar_split_aggregates": scalar_split_aggregates,
        "technical_gates": technical,
        "baseline": baseline_np,
        "reverse": reverse_np,
        "fd": fd_np,
        "effect_plus": plus_effect_np,
        "effect_minus": minus_effect_np,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def _combine_count_weighted_scalar_aggregates(
    records: tuple[dict[str, Any], ...], value_key: str, expected_denominator: int
) -> float:
    """Reconstruct one whole-split mean from exact contributing-parent counts."""
    if not records or type(expected_denominator) is not int or expected_denominator <= 0:
        raise G3ContractError("scalar aggregate denominator contract is invalid")
    counts = tuple(record.get("contributing_parent_count") for record in records)
    if any(type(count) is not int or count <= 0 for count in counts):
        raise G3ContractError("scalar aggregate contributing-parent count is invalid")
    if sum(counts) != expected_denominator:
        raise G3ContractError("scalar aggregate denominator changed from the whole-split statistic")
    values = tuple(record.get(value_key) for record in records)
    if not all(type(value) in {float, int} and np.isfinite(value) for value in values):
        raise G3ContractError(f"scalar aggregate {value_key} contains a nonfinite value")
    return float(
        sum(float(value) * count for value, count in zip(values, counts, strict=True))
        / expected_denominator
    )


def _parameter_diagnostic(
    batch_records: tuple[dict[str, Any], ...],
    specs: tuple[EpisodeSpec, ...],
    name: str,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Concatenate all parents before global gates and combine exact split statistics."""
    observed_ids = tuple(parent_id for batch in batch_records for parent_id in batch["parent_ids"])
    expected_ids = tuple(item.episode_id for item in specs)
    if observed_ids != expected_ids or len(batch_records) not in {1, 2, 6}:
        raise G3ContractError(f"{name} batch aggregation changed parent identity or order")
    for batch in batch_records:
        if batch["parameter"] != name:
            raise G3ContractError(f"{name} batch aggregation mixed parameters")
    arrays = {
        key: np.concatenate(tuple(batch[key] for batch in batch_records), axis=0)
        for key in ("baseline", "reverse", "fd", "effect_plus", "effect_minus")
    }
    reverse_np = arrays["reverse"]
    fd_np = arrays["fd"]
    feasible = np.asarray(
        [item.motion_class is not MotionClass.NEAR_LIMIT for item in specs], dtype=bool
    )
    role_comparisons = _output_role_comparison_payload(reverse_np, fd_np, feature_names, specs)
    feature_section = role_comparisons["all_features"]
    objective_section = role_comparisons["optimizer_objective"]
    diagnostic_section = role_comparisons["diagnostic"]
    feature_comparison = feature_section["comparison"]
    feature_tolerance_mismatch_count = feature_section["mismatch_count"]
    feature_tolerance_mismatch_strata = feature_section["mismatch_strata"]
    feature_sign_mismatch = feature_section["significant_sign_mismatch_count"] > 0
    feature_false_zero = feature_section["false_zero_count"] > 0
    indices = [feature_names.index(output) for output in OPTIMIZER_OBJECTIVE_NAMES]
    comparison = objective_section["comparison"]
    mismatch_strata = objective_section["mismatch_strata"]
    allowed_mismatch_strata = int(math.floor(0.10 * int(np.count_nonzero(feasible))))
    significant_sign_mismatch = objective_section["significant_sign_mismatch_count"] > 0
    false_zero = objective_section["false_zero_count"] > 0
    split_congruence_failures = []
    total_index = feature_names.index("loss_total")
    split_aggregates = {}
    for split in (Split.TRAIN, Split.VALIDATION):
        mask = feasible & np.asarray([item.split is split for item in specs])
        matching = tuple(
            batch["scalar_split_aggregates"][split.value]
            for batch in batch_records
            if split.value in batch["scalar_split_aggregates"]
        )
        expected_denominator = int(np.count_nonzero(mask))
        if expected_denominator != G3_004_FEASIBLE_SPLIT_DENOMINATOR:
            raise G3ContractError(f"{name}/{split.value} feasible denominator changed")
        contributing_counts = [record["contributing_parent_count"] for record in matching]
        if len(batch_records) == 6 and contributing_counts != [4, 4, 1]:
            raise G3ContractError(f"{name}/{split.value} six-batch contributions changed")
        expected_contributing_ids = [
            item.episode_id for item, contributes in zip(specs, mask, strict=True) if contributes
        ]
        observed_contributing_ids = [
            parent_id for record in matching for parent_id in record["contributing_parent_ids"]
        ]
        if observed_contributing_ids != expected_contributing_ids:
            raise G3ContractError(f"{name}/{split.value} scalar contributor order changed")
        scalar_value = _combine_count_weighted_scalar_aggregates(
            matching, "loss_total_scalar_value", expected_denominator
        )
        scalar_reverse_value = _combine_count_weighted_scalar_aggregates(
            matching, "loss_total_scalar_reverse", expected_denominator
        )
        vector_reverse_mean = float(np.mean(reverse_np[mask, total_index]))
        fd_mean = float(np.mean(fd_np[mask, total_index]))
        gate = _reverse_congruence_gate(
            f"{name}/{split.value}-aggregate", scalar_reverse_value, vector_reverse_mean, fd_mean
        )
        if not gate["passed"]:
            split_congruence_failures.append(
                {
                    "gate": "SPLIT_AGGREGATE_OBJECTIVE_REVERSE_CONGRUENCE",
                    "split": split.value,
                    "finite": all(
                        np.isfinite(value)
                        for value in (scalar_reverse_value, vector_reverse_mean, fd_mean)
                    ),
                    "record": gate,
                }
            )
        if not all(
            np.isfinite(value) for value in (scalar_reverse_value, vector_reverse_mean, fd_mean)
        ):
            raise G3ContractError(f"{name}/{split.value} aggregate total derivative is nonfinite")
        split_aggregates[split.value] = {
            "batch_contributions": [
                {
                    "batch_label": record["batch_label"],
                    "contributing_parent_ids": record["contributing_parent_ids"],
                    "contributing_parent_count": record["contributing_parent_count"],
                    "batch_statistic_denominator": record["batch_statistic_denominator"],
                }
                for record in matching
            ],
            "contributing_parent_counts": contributing_counts,
            "original_statistic_denominator": expected_denominator,
            "original_statistic_denominator_semantics": (
                "count of feasible parents in the complete frozen split"
            ),
            "scalar_aggregation_formula": (
                "sum(batch_mean_i * contributing_parent_count_i) / sum(contributing_parent_count_i)"
            ),
            "loss_total_scalar_value": scalar_value,
            "loss_total_scalar_reverse": scalar_reverse_value,
            "loss_total_vector_reverse_mean": vector_reverse_mean,
            "loss_total_fd_mean": fd_mean,
            "reverse_congruence_gate": gate,
            "loss_outputs": {
                feature_names[index]: {
                    "reverse_mean": float(np.mean(reverse_np[mask, index])),
                    "fd_mean": float(np.mean(fd_np[mask, index])),
                }
                for index in indices
            },
        }
    objective_gradient_contract = _objective_gradient_contract_payload(
        objective_section, allowed_mismatch_strata, tuple(split_congruence_failures)
    )
    objective_gradient_valid = objective_gradient_contract["eligible"]
    spec = PARAMETER_SPECS[name]
    effect_physical_values = {
        label: spec.default * math.exp(theta)
        for label, theta in (("minus_0p05", -EFFECT_STEP), ("plus_0p05", EFFECT_STEP))
    }
    nonfinite_effect_rollouts = [
        record
        | {
            "prerequisite_scientific_gates": {
                "theta_zero_default_parity": "PASS_THETA_ZERO_DEFAULT_PARITY",
                "complete_baseline_and_effect_feature_vectors_finite": True,
                "scalar_and_vector_reverse_finite": True,
                "central_fd_finite": True,
                "optimizer_objective_gradient_contract_passed": objective_gradient_valid,
                "diagnostic_ad_role": DIAGNOSTIC_AD_ROLE,
            }
        }
        for theta_label in ("minus_0p05", "plus_0p05")
        for batch in batch_records
        for record in batch["technical_gates"][theta_label]["nonfinite_effect_rollouts"]
    ]
    return {
        "parameter": name,
        "coordinate": (
            "mass_thrust(theta)=float32(132000)*exp(theta)"
            if name == "mass_thrust"
            else "p(theta)=p_default*exp(theta)"
        ),
        "default": spec.default,
        "bounds": [spec.lower, spec.upper],
        "unit": spec.unit,
        "isolation": batch_records[0]["isolation"],
        "theta_zero_default_parity": batch_records[0]["theta_zero_default_parity"],
        "batch_isolation_and_parity": [
            {
                "batch_label": batch["batch_label"],
                "parent_ids": batch["parent_ids"],
                "isolation": batch["isolation"],
                "theta_zero_default_parity": batch["theta_zero_default_parity"],
            }
            for batch in batch_records
        ],
        "reverse_fd_h": AD_FD_STEP,
        "effect_theta": EFFECT_STEP,
        "effect_physical_values": effect_physical_values,
        "output_role_contract": role_comparisons["inventory"],
        "objective_gradient_contract": objective_gradient_contract,
        "diagnostic_ad_disclosure": diagnostic_section,
        "all_feature_ad_disclosure": feature_section,
        "reverse_fd_loss_comparison": comparison,
        "reverse_fd_feature_comparison": feature_comparison,
        "feature_tolerance_mismatch_count": feature_tolerance_mismatch_count,
        "feature_tolerance_mismatch_strata": feature_tolerance_mismatch_strata,
        "feature_significant_sign_mismatch": feature_sign_mismatch,
        "feature_false_zero": feature_false_zero,
        "mismatch_strata": mismatch_strata,
        "allowed_mismatch_strata": allowed_mismatch_strata,
        "significant_sign_mismatch": significant_sign_mismatch,
        "false_zero": false_zero,
        "split_aggregates": split_aggregates,
        "technical_gates": {
            theta_label: {
                "batches": [
                    {"batch_label": batch["batch_label"], **batch["technical_gates"][theta_label]}
                    for batch in batch_records
                ]
            }
            for theta_label in ("minus_0p05", "plus_0p05")
        },
        "technical_robustness": {
            "eligible": not nonfinite_effect_rollouts,
            "reason": (
                None if not nonfinite_effect_rollouts else NONFINITE_NEAR_LIMIT_EFFECT_REASON
            ),
            "nonfinite_effect_rollouts": nonfinite_effect_rollouts,
        },
        **arrays,
        "forward_mode_executed": False,
        "scalar_reverse_operator": "jax.value_and_grad",
        "vector_reverse_operator": "jax.jacrev",
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def _unsegmented_feature_name(name: str) -> str:
    for segment in ("warmup_prefix", "scored_window", "full_rollout"):
        prefix = f"{segment}_"
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def _feature_family(name: str) -> str:
    name = _unsegmented_feature_name(name)
    if name.startswith("loss_") or name.startswith("position_") or name.startswith("velocity_"):
        return "tracking"
    if name.startswith("integral_"):
        return "integral"
    if "wrench" in name or "collective" in name or "reserve" in name:
        return "wrench"
    return "other"


def _feature_scale(name: str, baseline: np.ndarray) -> np.ndarray:
    name = _unsegmented_feature_name(name)
    if name in {"loss_raw_position", "loss_raw_terminal"}:
        return np.full_like(baseline, 0.60**2)
    if name == "loss_raw_velocity":
        return np.full_like(baseline, robust.GLOBAL_LIMITS["speed_m_s"] ** 2)
    if name.startswith("position_mse_"):
        envelope = 0.60 if "xy" in name else 0.85
        return np.full_like(baseline, envelope**2)
    if name.startswith("position_") or "signed_z_error" in name:
        return np.full_like(baseline, 0.60 if "xy" in name else 0.85)
    if name.startswith("velocity_"):
        return np.full_like(baseline, robust.GLOBAL_LIMITS["speed_m_s"])
    if name.startswith("integral_"):
        if name.endswith("_x"):
            return np.full_like(baseline, 2.0)
        if name.endswith("_y"):
            return np.full_like(baseline, 2.0)
        if name.endswith("_z"):
            return np.full_like(baseline, 0.4)
        return np.ones_like(baseline)
    if name == "requested_collective_mean":
        return np.full_like(baseline, WRENCH_PLATFORM_SCALE[0])
    if "wrench" in name:
        for axis, axis_name in enumerate(("collective", "roll", "pitch", "yaw")):
            if name.endswith(axis_name):
                return np.full_like(baseline, WRENCH_PLATFORM_SCALE[axis])
    if "reserve" in name:
        return np.ones_like(baseline)
    return np.maximum(np.abs(baseline), 1.0e-6)


def _feature_scale_rule(name: str) -> str:
    name = _unsegmented_feature_name(name)
    if name in {"loss_raw_position", "loss_raw_terminal"}:
        return "physical_workspace_envelope_squared"
    if name == "loss_raw_velocity":
        return "physical_speed_envelope_squared"
    if name.startswith("position_") or "signed_z_error" in name:
        return "physical_workspace_envelope"
    if name.startswith("velocity_"):
        return "physical_speed_envelope_1p5_m_s"
    if name.startswith("integral_"):
        return "state_controller_int_err_max_or_unit_dimensionless"
    if "wrench" in name:
        return "day25_platform_wrench_scale"
    if "reserve" in name:
        return "dimensionless_unit_scale"
    return "max_abs_default_baseline_or_1e-6"


def _normalized_sensitivity(
    diagnostic: dict[str, Any], feature_names: tuple[str, ...]
) -> np.ndarray:
    numerator = diagnostic["effect_plus"] - diagnostic["effect_minus"]
    scales = np.stack(
        tuple(
            _feature_scale(name, diagnostic["baseline"][:, index])
            for index, name in enumerate(feature_names)
        ),
        axis=1,
    )
    result = numerator / (2.0 * EFFECT_STEP * scales)
    if not np.all(np.isfinite(result)):
        raise G3ContractError(f"{diagnostic['parameter']} normalized sensitivity is nonfinite")
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left.ravel(), right.ravel()) / denominator)


def _single_parameter_classification(
    name: str,
    matrix: np.ndarray,
    feature_names: tuple[str, ...],
    specs: tuple[EpisodeSpec, ...],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    feasible = np.asarray([spec.motion_class is not MotionClass.NEAR_LIMIT for spec in specs])
    train = feasible & np.asarray([spec.split is Split.TRAIN for spec in specs])
    validation = feasible & np.asarray([spec.split is Split.VALIDATION for spec in specs])
    train_rms = float(np.sqrt(np.mean(matrix[train] ** 2)))
    validation_rms = float(np.sqrt(np.mean(matrix[validation] ** 2)))
    relevant_indices = [
        index
        for index, feature in enumerate(feature_names)
        if _feature_family(feature) in {"tracking", "integral", "wrench"}
    ]
    by_split = {}
    for split, mask in ((Split.TRAIN, train), (Split.VALIDATION, validation)):
        relevant_peak = float(np.max(np.abs(matrix[mask][:, relevant_indices])))
        profiles = {
            spec.profile.value
            for row, spec, selected in zip(matrix, specs, mask, strict=True)
            if selected and np.max(np.abs(row)) >= 0.02
        }
        classes = {
            spec.motion_class.value
            for row, spec, selected in zip(matrix, specs, mask, strict=True)
            if selected and np.max(np.abs(row)) >= 0.02
        }
        column_rms = float(np.sqrt(np.mean(matrix[mask] ** 2)))
        by_split[split.value] = {
            "column_rms": column_rms,
            "tracking_integral_wrench_peak": relevant_peak,
            "profiles_at_or_above_0p02": sorted(profiles),
            "classes_at_or_above_0p02": sorted(classes),
            "passed": (
                column_rms >= 0.02
                and relevant_peak >= 0.05
                and len(profiles) >= 2
                and len(classes) >= 2
            ),
        }
    train_rows = matrix[train]
    validation_rows = matrix[validation]
    stability = _cosine(train_rows, validation_rows)
    detectable = all(item["passed"] for item in by_split.values())
    stable = stability >= 0.80
    objective_gradient = diagnostic.get("objective_gradient_contract")
    if (
        not isinstance(objective_gradient, dict)
        or type(objective_gradient.get("eligible")) is not bool
    ):
        raise G3ContractError(f"{name} has an invalid objective-gradient contract")
    technical_robustness = diagnostic.get(
        "technical_robustness", {"eligible": True, "reason": None, "nonfinite_effect_rollouts": []}
    )
    withholding_reasons = []
    if not objective_gradient["eligible"]:
        if objective_gradient.get(
            "withholding_reason"
        ) != WITHHELD_OBJECTIVE_GRADIENT_INVALID or not objective_gradient.get("finite_failures"):
            raise G3ContractError(f"{name} has an invalid objective-gradient withholding contract")
        withholding_reasons.append(WITHHELD_OBJECTIVE_GRADIENT_INVALID)
    if not technical_robustness["eligible"]:
        if (
            technical_robustness["reason"] != NONFINITE_NEAR_LIMIT_EFFECT_REASON
            or not technical_robustness["nonfinite_effect_rollouts"]
        ):
            raise G3ContractError(f"{name} has an invalid technical-withholding contract")
        withholding_reasons.append(WITHHELD_TECHNICAL_ROBUSTNESS)
    if withholding_reasons:
        category = WITHHELD_PARAMETER_CATEGORY
    else:
        category = (
            "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION"
            if detectable and stable
            else "DIAGNOSTIC_ONLY"
            if detectable
            else "NO_GO"
        )
    return {
        "parameter": name,
        "category": category,
        "withholding_reasons": withholding_reasons,
        "objective_gradient_eligible": objective_gradient["eligible"],
        "objective_gradient_withholding_reason": objective_gradient["withholding_reason"],
        "technical_robustness_eligible": technical_robustness["eligible"],
        "technical_robustness_reason": technical_robustness["reason"],
        "nonfinite_effect_rollout_count": len(technical_robustness["nonfinite_effect_rollouts"]),
        "detectability": {
            "passed": detectable,
            "train_column_rms": train_rms,
            "validation_column_rms": validation_rms,
            "by_split": by_split,
        },
        "stability": {"passed": stable, "train_validation_column_cosine": stability},
        "reverse_fd_objective_evidence": {
            "role": OPTIMIZER_OBJECTIVE_ROLE,
            "passed": objective_gradient["eligible"],
            "finite_failures": objective_gradient["finite_failures"],
        },
        "diagnostic_ad_evidence": {
            "role": DIAGNOSTIC_AD_ROLE,
            "optimizer_gradient_claim": False,
            "tolerance_mismatch_count": diagnostic["feature_tolerance_mismatch_count"],
            "tolerance_mismatch_strata": diagnostic["feature_tolerance_mismatch_strata"],
            "significant_sign_mismatch": diagnostic["feature_significant_sign_mismatch"],
            "false_zero": diagnostic["feature_false_zero"],
        },
        "scope_note": (
            "continuous simulation relaxation only; no integer or firmware GO"
            if name == "mass_thrust"
            else "effective controller parameter in this simulation chain"
            if name == "mass"
            else "experiment-local continuous diagnostic"
        ),
    }


def _group_conditioning(
    group: tuple[str, ...],
    matrices: dict[str, np.ndarray],
    specs: tuple[EpisodeSpec, ...],
    singles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    excluded_withheld = [
        name for name in group if singles[name]["category"] == WITHHELD_PARAMETER_CATEGORY
    ]
    result: dict[str, Any] = {
        "parameters": list(group),
        "excluded_withheld_parameters": excluded_withheld,
        "excluded_withholding_reasons": {
            name: singles[name]["withholding_reasons"] for name in excluded_withheld
        },
        "by_split": {},
    }
    joint = (
        all(singles[name]["category"] == "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION" for name in group)
        and not excluded_withheld
    )
    for split in (Split.TRAIN, Split.VALIDATION):
        mask = np.asarray(
            [
                spec.split is split and spec.motion_class is not MotionClass.NEAR_LIMIT
                for spec in specs
            ]
        )
        columns = [matrices[name][mask].ravel() for name in group]
        matrix = np.stack(columns, axis=1)
        singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix))
        sigma_max = float(singular[0]) if singular.size else 0.0
        sigma_min = float(singular[-1]) if singular.size else 0.0
        condition = None if sigma_min == 0.0 else sigma_max / sigma_min
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0)
        denominator = np.outer(norms, norms)
        covariance = centered.T @ centered
        correlation = np.divide(
            covariance,
            denominator,
            out=np.eye(len(group), dtype=np.float64),
            where=denominator > 0.0,
        )
        off_diagonal = correlation - np.eye(len(group))
        maximum_correlation = float(np.max(np.abs(off_diagonal))) if len(group) > 1 else 0.0
        sigma_ratio = 0.0 if sigma_max == 0.0 else sigma_min / sigma_max
        split_passed = (
            rank == len(group)
            and condition is not None
            and condition <= 30.0
            and sigma_ratio >= 1.0 / 30.0
            and maximum_correlation <= 0.95
        )
        joint = joint and split_passed
        result["by_split"][split.value] = {
            "singular_values": singular.tolist(),
            "rank": rank,
            "condition_number": condition,
            "condition_number_status": (
                "FINITE" if condition is not None else "INFINITE_ZERO_SIGMA_MIN"
            ),
            "sigma_min_over_sigma_max": sigma_ratio,
            "absolute_correlation": np.abs(correlation).tolist(),
            "maximum_off_diagonal_absolute_correlation": maximum_correlation,
            "passed": split_passed,
        }
    result["category"] = "GO_FOR_JOINT_OPTIMIZATION" if joint else "DIAGNOSTIC_ONLY"
    return result


def _conditioning_payload(
    diagnostics: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
    specs: tuple[EpisodeSpec, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    matrices = {
        name: _normalized_sensitivity(diagnostic, feature_names)
        for name, diagnostic in diagnostics.items()
    }
    singles = {
        name: _single_parameter_classification(
            name, matrices[name], feature_names, specs, diagnostics[name]
        )
        for name in PARAMETER_NAMES
    }
    groups = (
        ("mass", "mass_thrust"),
        ("kp_z", "kd_z", "ki_z", "mass", "mass_thrust"),
        PARAMETER_NAMES,
    )
    group_records = {
        "+".join(group): _group_conditioning(group, matrices, specs, singles) for group in groups
    }
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS_CONTINUOUS_SENSITIVITY_CONDITIONING",
            "feature_names": list(feature_names),
            "output_role_contract": _output_role_inventory(feature_names),
            "feature_families": {name: _feature_family(name) for name in feature_names},
            "coordinates": [
                {
                    "row": row,
                    "episode_id": spec.episode_id,
                    "split": spec.split.value,
                    "motion_class": spec.motion_class.value,
                    "profile": spec.profile.value,
                    "feasible_for_conditioning": spec.motion_class is not MotionClass.NEAR_LIMIT,
                }
                for row, spec in enumerate(specs)
            ],
            "feature_scaling": {
                feature: {
                    "rule": _feature_scale_rule(feature),
                    "values_by_coordinate": _feature_scale(
                        feature, diagnostics[PARAMETER_NAMES[0]]["baseline"][:, index]
                    ).tolist(),
                }
                for index, feature in enumerate(feature_names)
            },
            "single_parameters": singles,
            "groups": group_records,
            "thresholds": {
                "column_rms": 0.02,
                "relevant_peak": 0.05,
                "stability_cosine": 0.80,
                "condition_number_max": 30.0,
                "sigma_ratio_min": 1.0 / 30.0,
                "off_diagonal_correlation_max": 0.95,
            },
            "discrete_integer_response_included": False,
            "optimizer_initialization_count": 0,
            "optimizer_update_count": 0,
        },
        matrices,
    )


def _segment_summary(
    inputs: EvaluationInputs, outputs: Any, initial_data: SimData
) -> list[dict[str, Any]]:
    trace = outputs["trace"]
    integral = np.asarray(outputs["pos_err_i"])
    reference = inputs.reference
    stage_a = outputs["diagnostic"]["stage_a"]
    stage_b = outputs["diagnostic"]["stage_b"]
    limits = np.asarray(_state_params(initial_data)["int_err_max"])
    hover = hover_rotor_velocity(initial_data)
    loss_config = TrackingLossConfig()
    records = []
    for world, spec in enumerate(inputs.specs):
        segments = {
            "warmup_prefix": (0, spec.score_start),
            "scored_window": (spec.score_start, spec.score_stop),
            "full_rollout": (0, ROLLOUT_INTERVALS),
        }
        segment_records = {}
        for segment, (start, stop) in segments.items():
            segment_trace = jax.tree.map(lambda value: value[start:stop, world : world + 1], trace)
            segment_reference = jax.tree.map(
                lambda value: value[start:stop, world : world + 1], reference
            )
            loss_terms = tracking_loss_terms_per_case(
                segment_trace, segment_reference, hover[world : world + 1], loss_config
            )
            term_records = {}
            exact_loss_sum = np.float32(0.0)
            for term in LOSS_TERM_NAMES:
                contribution = np.float32(np.asarray(loss_terms.weighted[term])[0])
                exact_loss_sum = np.float32(exact_loss_sum + contribution)
                term_records[term] = {
                    "raw": float(np.asarray(loss_terms.raw[term])[0]),
                    "normalization_divisor": float(
                        np.asarray(loss_terms.normalization_divisor[term])
                    ),
                    "weight": float(np.asarray(loss_terms.weight[term])),
                    "contribution": float(contribution),
                }
            position_error = (
                np.asarray(trace.pos)[start:stop, world, 0]
                - np.asarray(reference.pos)[start:stop, world, 0]
            )
            velocity_error = (
                np.asarray(trace.vel)[start:stop, world, 0]
                - np.asarray(reference.vel)[start:stop, world, 0]
            )
            integral_value = integral[start:stop, world, 0]
            integral_contact = np.abs(integral_value) >= limits * (
                1.0 - robust.INTEGRAL_DETECTION_THRESHOLD
            )
            diagnostic_slice = slice(start, stop)
            stage_a_motor = np.asarray(stage_a["motor_pwm_any_clip_mask"])[
                diagnostic_slice, :, world, 0
            ]
            stage_a_torque = np.asarray(stage_a["torque_any_clip_mask"])[
                diagnostic_slice, :, world, 0
            ]
            stage_b_motor = np.asarray(stage_b["motor_force_any_clip_mask"])[
                diagnostic_slice, :, world, 0
            ]
            lower = np.asarray(stage_a["force_lower_reserve_normalized"])[
                diagnostic_slice, :, world, 0
            ]
            upper = np.asarray(stage_a["force_upper_reserve_normalized"])[
                diagnostic_slice, :, world, 0
            ]
            requested_wrench = np.asarray(stage_a["wrench_preclip"])[diagnostic_slice, :, world, 0]
            realized_wrench = np.asarray(stage_b["wrench_postclip"])[diagnostic_slice, :, world, 0]
            stage_a_distortion = np.asarray(stage_a["wrench_distortion_absolute"])[
                diagnostic_slice, :, world, 0
            ]
            stage_b_distortion = np.asarray(stage_b["wrench_distortion_absolute"])[
                diagnostic_slice, :, world, 0
            ]
            segment_records[segment] = {
                "interval_start": start,
                "interval_stop_exclusive": stop,
                "tracking": robust._error_metrics(position_error, velocity_error),
                "loss_v1": {
                    "terms": term_records,
                    "exact_contribution_sum": float(exact_loss_sum),
                    "loss_total": float(exact_loss_sum),
                },
                "integral_rms_by_axis": np.sqrt(np.mean(integral_value**2, axis=0)).tolist(),
                "integral_terminal_by_axis": integral_value[-1].tolist(),
                "integral_max_abs_by_axis": np.max(np.abs(integral_value), axis=0).tolist(),
                "integral_z_nearness_rms": float(
                    np.sqrt(np.mean((integral_value[:, 2] / limits[2]) ** 2))
                ),
                "integral_contact_count_total": int(np.count_nonzero(integral_contact)),
                "integral_contact_count_by_axis": np.count_nonzero(
                    integral_contact, axis=0
                ).tolist(),
                "minimum_normalized_motor_reserve": float(np.min(np.minimum(lower, upper))),
                "requested_wrench_rms_by_axis": np.sqrt(
                    np.mean(requested_wrench**2, axis=(0, 1))
                ).tolist(),
                "realized_wrench_rms_by_axis": np.sqrt(
                    np.mean(realized_wrench**2, axis=(0, 1))
                ).tolist(),
                "stage_a_wrench_distortion_abs_mean_by_axis": np.mean(
                    stage_a_distortion, axis=(0, 1)
                ).tolist(),
                "stage_b_wrench_distortion_abs_mean_by_axis": np.mean(
                    stage_b_distortion, axis=(0, 1)
                ).tolist(),
                "stage_a_wrench_distortion_platform_normalized_mean_by_axis": np.mean(
                    stage_a_distortion / np.asarray(WRENCH_PLATFORM_SCALE), axis=(0, 1)
                ).tolist(),
                "stage_b_wrench_distortion_platform_normalized_mean_by_axis": np.mean(
                    stage_b_distortion / np.asarray(WRENCH_PLATFORM_SCALE), axis=(0, 1)
                ).tolist(),
                "clip_counts": {
                    "stage_a_torque": int(stage_a_torque.sum()),
                    "stage_a_motor": int(stage_a_motor.sum()),
                    "stage_b_additional": int(stage_b_motor.sum()),
                },
            }
        records.append({"episode_id": spec.episode_id, "segments": segment_records})
    return records


def _science_batch_payload(
    batch_label: str, constructed: tuple[tuple[EpisodeSpec, ReferenceCandidate], ...]
) -> dict[str, Any]:
    """Execute exactly one complete science batch."""
    inputs = stack_evaluation_inputs(constructed)
    sim, initial_data = build_initial_data(inputs)
    parity, feature_names, baseline_features, baseline_outputs = _full_default_parity(
        batch_label, sim, initial_data, inputs
    )
    parameter_records = {
        name: _parameter_batch_diagnostic(
            batch_label, sim, initial_data, inputs, name, feature_names
        )
        for name in PARAMETER_NAMES
    }
    segment_features = _segment_summary(inputs, baseline_outputs, initial_data)
    return {
        "batch_label": batch_label,
        "parent_ids": [item.episode_id for item in inputs.specs],
        "parent_count": len(inputs.specs),
        "split_order": list(dict.fromkeys(item.split.value for item in inputs.specs)),
        "parity": parity,
        "feature_names": feature_names,
        "baseline_features": baseline_features,
        "segment_features": segment_features,
        "initial_data": initial_data,
        "parameter_records": parameter_records,
    }


def _science_batch(
    batch_label: str,
    constructed: tuple[tuple[EpisodeSpec, ReferenceCandidate], ...],
    *,
    primary: bool,
    batch_timeout_seconds: int,
) -> dict[str, Any]:
    """Execute one complete batch under an actual 900-second wall-clock deadline."""
    started = time.monotonic()
    try:
        with _batch_deadline(batch_timeout_seconds):
            payload = _science_batch_payload(batch_label, constructed)
    except _G3BatchDeadlineExceeded as error:
        message = f"{batch_label} exceeded the frozen {batch_timeout_seconds} s batch deadline"
        if primary:
            raise G3ResourceFallbackRequired(message) from error
        raise G3ContractError(message) from error
    except MemoryError as error:
        if primary:
            raise G3ResourceFallbackRequired(
                f"{batch_label} raised MemoryError inside the primary science batch"
            ) from error
        raise
    elapsed = time.monotonic() - started
    if elapsed > batch_timeout_seconds:
        message = (
            f"{batch_label} elapsed {elapsed:.3f} s exceeds "
            f"the frozen {batch_timeout_seconds} s batch limit"
        )
        if primary:
            raise G3ResourceFallbackRequired(message)
        raise G3ContractError(message)
    return payload | {
        "deadline_seconds": batch_timeout_seconds,
        "elapsed_seconds": elapsed,
        "deadline_passed": True,
    }


def _trajectory_contract_payload(
    constructed: tuple[tuple[EpisodeSpec, ReferenceCandidate], ...],
) -> dict[str, Any]:
    episodes = []
    for spec, candidate in constructed:
        episodes.append(
            {
                "episode_id": spec.episode_id,
                "pair_id": spec.pair_id,
                "split": spec.split.value,
                "motion_class": spec.motion_class.value,
                "profile": spec.profile.value,
                "seed": spec.seed,
                "warmup_s": spec.warmup_s,
                "score_start_interval": spec.score_start,
                "score_stop_interval_exclusive": spec.score_stop,
                "accepted_attempt": candidate.attempt,
                "rejection_log": list(candidate.attempts),
                "score_statistics": candidate.score_statistics,
                "parent_statistics": candidate.parent_statistics,
                "component_contracts": list(candidate.component_contracts),
                "array_digests": candidate.array_digests,
                "parent_digest": candidate.parent_digest,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_24_PARENT_TRAJECTORY_CONTRACT",
        "contract": {
            "platform": PLATFORM,
            "dynamics": "Dynamics.first_principles",
            "integrator": "Euler",
            "backend": "cpu",
            "jax_enable_x64": False,
            "simulation_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "state_control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "attitude_force_torque_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "drones_per_world": 1,
            "parent_intervals": PARENT_INTERVALS,
            "parent_samples": PARENT_INTERVALS + 1,
            "continuous_rollout_intervals": ROLLOUT_INTERVALS,
            "scored_window_intervals": SCORE_INTERVALS,
            "carry_reset_at_window": False,
            "dynamics_mass_kg": DYNAMICS_MASS_KG,
            "controller_defaults_modified": False,
            "initialization": {
                "position": "reference parent sample 0",
                "velocity": "reference parent sample 0",
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                "rotor_velocity": "physical hover from dynamics mass and rpm2thrust",
                "controller_carries": "all zero",
            },
            "disturbance_delay_noise_wind_ukf_initial_randomization": False,
            "test_constructed_loaded_listed_or_hashed": False,
            "workspace_m": {"minimum": [-0.60, -0.60, 0.35], "maximum": [0.60, 0.60, 1.20]},
            "global_envelopes": robust.GLOBAL_LIMITS,
            "class_bands": {
                motion_class.value: {
                    "lower_open": CLASS_CONTRACTS[motion_class].lower_open,
                    "upper_closed": CLASS_CONTRACTS[motion_class].upper_closed,
                    "target": CLASS_CONTRACTS[motion_class].target_usage,
                    "maximum_absolute_yaw_rad": CLASS_CONTRACTS[motion_class].max_abs_yaw_rad,
                    "feasible": motion_class is not MotionClass.NEAR_LIMIT,
                }
                for motion_class in MotionClass
            },
            "composition_order": [
                "bounded_fourier_translation",
                "class_scaling",
                "vertical_profile",
                "C3_product_rule",
                "z_after_fourier_scaling",
                "fixed_center_last",
                "separate_absolute_yaw",
            ],
            "profiles": {profile.value: PROFILE_RATES_M_S[profile] for profile in Profile},
            "seed_registry": {
                split.value: {
                    motion_class.value: list(EPISODE_SEEDS[split][motion_class])
                    for motion_class in MotionClass
                }
                for split in Split
            },
            "maximum_deterministic_attempts": MAX_ATTEMPTS,
        },
        "episodes": episodes,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def _manifest_payload(
    split: Split, constructed: tuple[tuple[EpisodeSpec, ReferenceCandidate], ...]
) -> dict[str, Any]:
    selected = [(spec, candidate) for spec, candidate in constructed if spec.split is split]
    if len(selected) != 12:
        raise G3ContractError(f"{split.value} manifest does not contain exactly 12 parents")
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split.value,
        "test_manifest": None,
        "parent_count": len(selected),
        "parents": [
            {
                "episode_id": spec.episode_id,
                "pair_id": spec.pair_id,
                "motion_class": spec.motion_class.value,
                "profile": spec.profile.value,
                "seed": spec.seed,
                "attempt": candidate.attempt,
                "attempt_identity": f"{spec.episode_id}/attempt-{candidate.attempt}",
                "component_key_digests": candidate.attempts[-1]["component_key_digests"],
                "parent_digest": candidate.parent_digest,
            }
            for spec, candidate in selected
        ],
        "split_disjointness_coordinates": [
            "episode_id",
            "seed",
            "attempt_identity",
            "component_key_digest",
            "parent_digest",
        ],
        "cross_split_leakage_count": 0,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def parameter_semantics_payload(
    initial_data: SimData, baseline_features: np.ndarray, feature_names: tuple[str, ...]
) -> dict[str, Any]:
    """Record source-path inference separately from measured and unresolved semantics."""
    state = _state_params(initial_data)
    defaults = {
        "kp_xy": float(np.asarray(state["kp"])[0]),
        "kp_z": float(np.asarray(state["kp"])[2]),
        "kd_xy": float(np.asarray(state["kd"])[0]),
        "kd_z": float(np.asarray(state["kd"])[2]),
        "ki_z": float(np.asarray(state["ki"])[2]),
        "mass": float(np.asarray(state["mass"])),
        "mass_thrust": int(np.asarray(state["mass_thrust"])),
    }
    parameter_records = {}
    for name in PARAMETER_NAMES:
        spec = PARAMETER_SPECS[name]
        parameter_records[name] = {
            "source_path": "crazyflow/control/mellinger/params.toml",
            "symbol": spec.parameter,
            "leaf_or_indices": list(spec.axes) if spec.axes else "scalar leaf",
            "default": spec.default if name != "mass_thrust" else CANONICAL_MASS_THRUST,
            "runtime_loaded_default": defaults[name],
            "unit": spec.unit,
            "load_merge_initialization_path": (
                "params.toml -> load_params(state2attitude) -> MellingerStateData.create -> "
                "experiment-local immutable SimData replacement"
            ),
            "hover_initialization": (
                "physical dynamics mass and force-torque rpm2thrust only; controller leaf remains "
                "an effective parameter"
            ),
            "feedforward_feedback_role": (
                "position/velocity/integral feedback gain"
                if name.startswith(("kp", "kd", "ki"))
                else "target-thrust feedforward multiplier"
                if name == "mass"
                else "legacy target-thrust-to-PWM scaling before linear pwm2force"
            ),
            "measured_behavior": "continuous local simulation sensitivity stored in report",
            "source_code_inference": "state2attitude and unchanged downstream allocation chain",
            "unresolved_provenance": (
                "physical unit and firmware/calibration transfer unresolved"
                if name == "mass_thrust"
                else "physical estimation/transfer not established"
                if name == "mass"
                else "no firmware/hardware transfer established"
            ),
            "continuous_relaxation": (
                "Float32 132000*exp(theta), experiment-local"
                if name == "mass_thrust"
                else "positive log coordinate, experiment-local"
            ),
            "discrete_integer_response": (
                "separate DISCRETE_INTEGER_RESPONSE; no AD validation"
                if name == "mass_thrust"
                else "not applicable"
            ),
            "firmware_or_calibration_transfer": "not established",
        }
    chain = [
        {
            "stage": "target_thrust",
            "source_path_symbol": (
                "crazyflow/control/mellinger/control.py::state2attitude.target_thrust"
            ),
            "measured_behavior": (
                "requested collective boundary, tracking error, and integral state are stored"
            ),
            "source_code_inference": (
                "mass*(setpoint_acc-gravity_vec)+kp*pos_err+kd*vel_err+ki*int_pos_err"
            ),
            "unresolved_provenance": "no physical parameter-estimation or firmware equivalence",
            "input_output_units": "SI state/setpoint to target vector nominally N",
            "bounds": "position integral clip at int_err_max",
        },
        {
            "stage": "body_z_then_mass_thrust",
            "source_path_symbol": (
                "crazyflow/control/mellinger/control.py::state2attitude.current_thrust"
            ),
            "measured_behavior": "continuous and rounded-integer responses are stored separately",
            "source_code_inference": ("body-z dot product followed by mass_thrust*current_thrust"),
            "unresolved_provenance": (
                "mass_thrust physical unit, calibration origin, quantization, and firmware "
                "transfer remain unresolved"
            ),
            "input_output_units": "nominal N times unresolved legacy factor to PWM-domain value",
            "bounds": "none at the multiplication",
        },
        {
            "stage": "pwm2force",
            "source_path_symbol": "crazyflow/control/transform.py::pwm2force",
            "measured_behavior": "downstream collective and wrench arrays are stored",
            "source_code_inference": "linear pwm/pwm_max*thrust_max scaling",
            "unresolved_provenance": "legacy factor calibration is not derived by this source",
            "input_output_units": "PWM-domain value to N under supplied platform scales",
            "bounds": "no clip inside pwm2force",
        },
        {
            "stage": "attitude_mixer_force2pwm_clips",
            "source_path_symbol": (
                "crazyflow/control/mellinger/control.py::attitude2force_torque and "
                "force_torque_pwms2pwms"
            ),
            "measured_behavior": "requested/realized wrench, PWM masks, reserve, and distortion",
            "source_code_inference": (
                "SO(3) feedback, linear force2pwm, legacy mixer, torque/motor PWM clips, "
                "then linear pwm2force"
            ),
            "unresolved_provenance": "no firmware numerical-equivalence or calibration claim",
            "input_output_units": "N plus legacy torque-PWM terms to SI wrench",
            "bounds": "torque_pwm_max and pwm_min/pwm_max",
        },
        {
            "stage": "stage_b_motor_force_to_rotor_velocity",
            "source_path_symbol": (
                "crazyflow/control/mellinger/control.py::force_torque2rotor_vel; "
                "crazyflow/control/transform.py::motor_force2rotor_vel"
            ),
            "measured_behavior": "Stage-B clips, realized wrench, motor commands, and reserve",
            "source_code_inference": (
                "X allocation, thrust_min/thrust_max clip, inverse quadratic motor model"
            ),
            "unresolved_provenance": "motor-model calibration and hardware transfer unresolved",
            "input_output_units": "SI wrench to motor force N to rotor-velocity convention",
            "bounds": "thrust_min/thrust_max",
        },
        {
            "stage": "first_principles_dynamics",
            "source_path_symbol": "Dynamics.first_principles with Euler integration",
            "measured_behavior": "complete Float32 CPU state rollout and technical gates",
            "source_code_inference": "quadratic rpm2thrust feeds rigid-body dynamics",
            "unresolved_provenance": "no physical validation, Sim2Real, or flight evidence",
            "input_output_units": "rotor velocity to SI rigid-body state",
            "bounds": "floor clip remains an external hard technical gate",
        },
    ]
    integral_indices = [feature_names.index(f"integral_rms_{axis}") for axis in ("x", "y")]
    xy_integral = baseline_features[:, integral_indices]
    int_limits = np.asarray(state["int_err_max"])[None, :2]
    normalized_xy = xy_integral / int_limits
    ki_xy_gate = float(np.max(normalized_xy)) >= 0.05
    x_energy = float(np.sum(baseline_features[:, feature_names.index("position_mse_xy")]))
    # The stored tied XY metric has no separable axis energy; this is explicitly a later gate.
    additional_groups = {
        "ki_xy": {
            "status": "LATER_GROUP",
            "members": ["ki[0]", "ki[1]"],
            "declared_defaults": [0.05, 0.05],
            "source_path_symbol": (
                "crazyflow/control/mellinger/params.toml::cf21B_500.state2attitude.ki; "
                "crazyflow/control/mellinger/control.py::state2attitude"
            ),
            "unit_contract": "current code contract N/(m s)",
            "pre_gate": {
                "maximum_integral_rms_fraction": float(np.max(normalized_xy)),
                "threshold": 0.05,
                "signal_threshold_met": ki_xy_gate,
                "two_classes_profiles_per_split_not_separately_quantified": True,
            },
            "reason": "requires a separately frozen two-class/two-profile gate before sensitivity",
        },
        "tied_vs_split_xy": {
            "status": "LATER_GROUP",
            "members": ["kp[0]", "kp[1]", "kd[0]", "kd[1]"],
            "declared_defaults": [0.4, 0.4, 0.2, 0.2],
            "source_path_symbol": (
                "crazyflow/control/mellinger/params.toml::cf21B_500.state2attitude; "
                "crazyflow/control/mellinger/control.py::state2attitude"
            ),
            "unit_contract": ["N/m", "N/m", "N s/m", "N s/m"],
            "pre_gate": {
                "combined_xy_tracking_energy": x_energy,
                "required_axis_energy_ratio": [0.8, 1.25],
                "axis_separated_energy_not_in_current_feature_inventory": True,
            },
            "reason": "current mandatory coordinate intentionally preserves tied x/y",
        },
        "attitude_and_rate_gains": {
            "status": "LATER_GROUP",
            "members": {
                "kR": [70000.0, 70000.0, 60000.0],
                "kw": [20000.0, 20000.0, 12000.0],
                "ki_m": [0.0, 0.0, 500.0],
                "kd_omega": [200.0, 200.0, 0.0],
            },
            "source_path_symbol": (
                "crazyflow/control/mellinger/params.toml::"
                "cf21B_500.attitude2force_torque; crazyflow/control/mellinger/control.py::"
                "attitude2force_torque"
            ),
            "unit_contract": "legacy PWM-domain semantics; physical units unresolved",
            "reason": "source units are legacy PWM-domain and no separate semantics gate is frozen",
        },
        "yaw_gains": {
            "status": "LATER_GROUP",
            "members": {"kR[2]": 60000.0, "kw[2]": 12000.0, "ki_m[2]": 500.0, "kd_omega[2]": 0.0},
            "source_path_symbol": (
                "crazyflow/control/mellinger/params.toml::"
                "cf21B_500.attitude2force_torque; crazyflow/control/mellinger/control.py::"
                "state2attitude/attitude2force_torque"
            ),
            "unit_contract": "legacy PWM-domain semantics; physical units unresolved",
            "reason": (
                "absolute yaw is excited and consumed, but yaw-rate feedforward and transfer "
                "semantics are not established"
            ),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_PARAMETER_SEMANTICS_AUDIT",
        "authorized_source_inventory": [
            "crazyflow/control/mellinger/params.toml",
            "crazyflow/control/core.py::load_params",
            "crazyflow/control/mellinger/control.py::MellingerStateData.create",
            "crazyflow/control/mellinger/control.py::state2attitude",
            "crazyflow/control/mellinger/control.py::attitude2force_torque",
            "crazyflow/control/mellinger/control.py::force_torque_pwms2pwms",
            "crazyflow/control/mellinger/control.py::force_torque2rotor_vel",
            "crazyflow/control/transform.py::force2pwm",
            "crazyflow/control/transform.py::pwm2force",
            "crazyflow/control/transform.py::motor_force2rotor_vel",
            "crazyflow/control/mellinger/research/runner.py::run_training_validation.train_step",
        ],
        "parameters": parameter_records,
        "conversion_chain": chain,
        "additional_parameter_groups": additional_groups,
        "canonical_defaults_modified": False,
        "shared_controller_modified": False,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def _loss_diagnostics_payload(
    diagnostics: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
    specs: tuple[EpisodeSpec, ...],
    baseline_features: np.ndarray,
    segment_features: list[dict[str, Any]],
) -> dict[str, Any]:
    feasible = np.asarray([spec.motion_class is not MotionClass.NEAR_LIMIT for spec in specs])
    candidate_index = feature_names.index("loss_v2_integral_z_candidate_v0")
    z_mse_index = feature_names.index("position_mse_z")
    loss_index = feature_names.index("loss_total")
    candidate_values = baseline_features[feasible, candidate_index]
    z_values = baseline_features[feasible, z_mse_index]
    correlation = _cosine(
        candidate_values - np.mean(candidate_values), z_values - np.mean(z_values)
    )
    candidate_gradient = np.asarray(
        [
            np.mean(diagnostics[name]["reverse"][feasible, candidate_index])
            for name in PARAMETER_NAMES
        ]
    )
    z_gradient = np.asarray(
        [np.mean(diagnostics[name]["reverse"][feasible, z_mse_index]) for name in PARAMETER_NAMES]
    )
    gradient_cosine = _cosine(candidate_gradient, z_gradient)
    per_parameter = {}
    reverse_fd_pass = True
    for name in PARAMETER_NAMES:
        reverse = diagnostics[name]["reverse"][feasible, candidate_index]
        fd = diagnostics[name]["fd"][feasible, candidate_index]
        comparison = _reverse_fd_comparison(reverse, fd)
        reverse_fd_pass = reverse_fd_pass and comparison["mismatch_count"] <= 1
        per_parameter[name] = comparison
    median_loss = float(np.median(baseline_features[feasible, loss_index]))
    median_candidate = float(np.median(candidate_values))
    proposed_weight = float(
        np.clip(0.05 * median_loss / max(median_candidate, 1.0e-6), 1.0e-4, 0.1)
    )
    scored_segments = {
        item["episode_id"]: item["segments"]["scored_window"] for item in segment_features
    }
    contact_indicator = np.asarray(
        [
            scored_segments[spec.episode_id]["integral_contact_count_total"] > 0
            for spec in specs
            if spec.motion_class is not MotionClass.NEAR_LIMIT
        ],
        dtype=np.float64,
    )
    contact_correlation = _cosine(
        candidate_values - np.mean(candidate_values), contact_indicator - np.mean(contact_indicator)
    )
    signal = bool(np.any(np.abs(candidate_gradient) > 1.0e-8))
    stability_records = {}
    stable = True
    for name in PARAMETER_NAMES:
        profile_records = {}
        for profile in Profile:
            train_mask = feasible & np.asarray(
                [s.split is Split.TRAIN and s.profile is profile for s in specs]
            )
            validation_mask = feasible & np.asarray(
                [s.split is Split.VALIDATION and s.profile is profile for s in specs]
            )
            cosine = _cosine(
                diagnostics[name]["reverse"][train_mask, candidate_index],
                diagnostics[name]["reverse"][validation_mask, candidate_index],
            )
            profile_records[profile.value] = cosine
            stable = stable and cosine >= 0.80
        stability_records[name] = profile_records
    recommendation = (
        "PROPOSE_VERSIONED_LOSS_V2_FOR_HAUPTLEITUNG_REVIEW"
        if reverse_fd_pass
        and signal
        and stable
        and abs(correlation) < 0.95
        and abs(gradient_cosine) < 0.95
        else "RETAIN_LOSS_V1_PLUS_EXTERNAL_GATES"
    )
    episode_loss_records = []
    episode_segment_records = []
    loss_total_index = feature_names.index("loss_total")
    for world, spec in enumerate(specs):
        terms = {}
        exact_sum = np.float32(0.0)
        for term in LOSS_TERM_NAMES:
            contribution = np.float32(
                baseline_features[world, feature_names.index(f"loss_contribution_{term}")]
            )
            exact_sum = np.float32(exact_sum + contribution)
            terms[term] = {
                "raw": float(baseline_features[world, feature_names.index(f"loss_raw_{term}")]),
                "normalization_divisor": float(
                    baseline_features[world, feature_names.index(f"loss_normalization_{term}")]
                ),
                "weight": float(
                    baseline_features[world, feature_names.index(f"loss_weight_{term}")]
                ),
                "contribution": float(contribution),
            }
        loss_total = np.float32(baseline_features[world, loss_total_index])
        if not np.array_equal(exact_sum, loss_total):
            raise G3ContractError(
                f"{spec.episode_id} Loss-v1 contribution sum changed: {exact_sum} != {loss_total}"
            )
        episode_loss_records.append(
            {
                "episode_id": spec.episode_id,
                "split": spec.split.value,
                "motion_class": spec.motion_class.value,
                "profile": spec.profile.value,
                "segment": "scored_window",
                "optimizer_objective_output_names": list(OPTIMIZER_OBJECTIVE_NAMES),
                "output_role": OPTIMIZER_OBJECTIVE_ROLE,
                "terms": terms,
                "exact_contribution_sum": float(exact_sum),
                "loss_total": float(loss_total),
                "reverse_gradient_by_parameter": {
                    parameter: {
                        "loss_total": float(
                            diagnostics[parameter]["reverse"][world, loss_total_index]
                        ),
                        "contributions": {
                            term: float(
                                diagnostics[parameter]["reverse"][
                                    world, feature_names.index(f"loss_contribution_{term}")
                                ]
                            )
                            for term in LOSS_TERM_NAMES
                        },
                    }
                    for parameter in PARAMETER_NAMES
                },
            }
        )
        segments = {}
        for segment in ("warmup_prefix", "scored_window", "full_rollout"):
            segment_terms = {}
            segment_sum = np.float32(0.0)
            for term in LOSS_TERM_NAMES:
                fields = {
                    quantity: feature_names.index(f"{segment}_loss_{quantity}_{term}")
                    for quantity in ("raw", "normalization", "weight", "contribution")
                }
                contribution = np.float32(baseline_features[world, fields["contribution"]])
                segment_sum = np.float32(segment_sum + contribution)
                segment_terms[term] = {
                    "raw": float(baseline_features[world, fields["raw"]]),
                    "normalization_divisor": float(
                        baseline_features[world, fields["normalization"]]
                    ),
                    "weight": float(baseline_features[world, fields["weight"]]),
                    "contribution": float(contribution),
                }
            segment_total_index = feature_names.index(f"{segment}_loss_total")
            segment_total = np.float32(baseline_features[world, segment_total_index])
            if not np.array_equal(segment_sum, segment_total):
                raise G3ContractError(
                    f"{spec.episode_id}/{segment} Loss-v1 contribution sum changed: "
                    f"{segment_sum} != {segment_total}"
                )
            segments[segment] = {
                "output_role": DIAGNOSTIC_AD_ROLE,
                "optimizer_gradient_claim": False,
                "terms": segment_terms,
                "exact_contribution_sum": float(segment_sum),
                "loss_total": float(segment_total),
                "reverse_gradient_by_parameter": {
                    parameter: {
                        "loss_total": float(
                            diagnostics[parameter]["reverse"][world, segment_total_index]
                        ),
                        "terms": {
                            term: {
                                output_name: float(
                                    diagnostics[parameter]["reverse"][
                                        world,
                                        feature_names.index(f"{segment}_loss_{quantity}_{term}"),
                                    ]
                                )
                                for output_name, quantity in (
                                    ("raw", "raw"),
                                    ("normalization_divisor", "normalization"),
                                    ("weight", "weight"),
                                    ("contribution", "contribution"),
                                )
                            }
                            for term in LOSS_TERM_NAMES
                        },
                    }
                    for parameter in PARAMETER_NAMES
                },
            }
        episode_segment_records.append(
            {
                "episode_id": spec.episode_id,
                "split": spec.split.value,
                "motion_class": spec.motion_class.value,
                "profile": spec.profile.value,
                "segments": segments,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_LOSS_V1_UNCHANGED_AND_INACTIVE_CANDIDATE_DIAGNOSED",
        "loss_v1": {
            "term_names": list(LOSS_TERM_NAMES),
            "optimizer_objective_output_names": list(OPTIMIZER_OBJECTIVE_NAMES),
            "optimizer_objective_role": OPTIMIZER_OBJECTIVE_ROLE,
            "supporting_output_role": DIAGNOSTIC_AD_ROLE,
            "active_and_numerically_unchanged": True,
            "raw_normalization_weight_contribution_and_total_stored": True,
            "xy_z_visibility": ["position_mse_xy", "position_mse_z"],
            "episode_scored_window_records": episode_loss_records,
            "episode_segment_records": episode_segment_records,
            "baseline_segment_records": segment_features,
        },
        "inactive_candidate": {
            "name": "loss_v2_integral_z_candidate_v0",
            "output_role": DIAGNOSTIC_AD_ROLE,
            "optimizer_gradient_claim": False,
            "formula": "mean((pos_err_i_z/int_err_max_z)^2) in scored window",
            "active_weight": 0.0,
            "added_to_rollout_objective": False,
            "per_parameter_reverse_fd": per_parameter,
            "correlation_with_z_mse": correlation,
            "gradient_cosine_with_z_tracking": gradient_cosine,
            "signal_nonzero": signal,
            "split_profile_stability_passed": stable,
            "train_validation_cosine_by_parameter_profile": stability_records,
            "episode_values": [
                {
                    "episode_id": spec.episode_id,
                    "split": spec.split.value,
                    "motion_class": spec.motion_class.value,
                    "profile": spec.profile.value,
                    "candidate": float(baseline_features[world, candidate_index]),
                    "z_mse": float(baseline_features[world, z_mse_index]),
                    "integral_contact_count": scored_segments[spec.episode_id][
                        "integral_contact_count_total"
                    ],
                }
                for world, spec in enumerate(specs)
            ],
            "correlation_with_any_integral_contact": contact_correlation,
            "proposed_weight_for_review_only": proposed_weight,
            "recommendation": recommendation,
        },
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def _g4_batch_schedule() -> dict[str, Any]:
    """Freeze the deterministic nine-by-32 prospective rotation without executing G4."""
    strata = [
        f"{motion_class.value}/{profile.value}"
        for motion_class in (MotionClass.SOFT, MotionClass.NOMINAL, MotionClass.DYNAMIC)
        for profile in Profile
    ]
    batches = []
    for batch_index in range(9):
        counts = {
            stratum: 4 if (index - batch_index) % 9 < 5 else 3
            for index, stratum in enumerate(strata)
        }
        if sum(counts.values()) != 32 or set(counts.values()) != {3, 4}:
            raise G3ContractError("prospective G4 3/4 rotation changed")
        batches.append({"batch_index": batch_index, "stratum_world_counts": counts})
    totals = {
        stratum: sum(batch["stratum_world_counts"][stratum] for batch in batches)
        for stratum in strata
    }
    if set(totals.values()) != {32}:
        raise G3ContractError("prospective G4 schedule is not exactly balanced")
    return {
        "batch_count": 9,
        "worlds_per_batch": 32,
        "feasible_stratum_order": strata,
        "per_batch_allocation_values": [3, 4],
        "batches": batches,
        "total_world_slots": 288,
        "total_slots_per_stratum": totals,
        "executed": False,
    }


def _freeze_proposal_payload(
    conditioning: dict[str, Any],
    matrices: dict[str, np.ndarray],
    specs: tuple[EpisodeSpec, ...],
    execution_mode: str,
) -> dict[str, Any]:
    singles = conditioning["single_parameters"]
    suitable = [
        name
        for name in PARAMETER_NAMES
        if singles[name]["category"] == "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION"
    ]
    feasible = np.asarray([spec.motion_class is not MotionClass.NEAR_LIMIT for spec in specs])
    total_energy = sum(float(np.sum(matrices[name][feasible] ** 2)) for name in suitable)
    xy_parameters = {"kp_xy", "kd_xy"}
    z_parameters = set(PARAMETER_NAMES) - xy_parameters
    candidates = []
    for size in range(2, len(suitable) + 1):
        for subset in itertools.combinations(suitable, size):
            if not (set(subset) & xy_parameters and set(subset) & z_parameters):
                continue
            group = _group_conditioning(tuple(subset), matrices, specs, singles)
            if group["category"] != "GO_FOR_JOINT_OPTIMIZATION":
                continue
            energy = sum(float(np.sum(matrices[name][feasible] ** 2)) for name in subset)
            coverage = 1.0 if total_energy == 0.0 else energy / total_energy
            if coverage < 0.90:
                continue
            condition = max(
                group["by_split"][split]["condition_number"] for split in ("train", "validation")
            )
            candidates.append(
                (
                    size,
                    condition,
                    tuple(PARAMETER_NAMES.index(name) for name in subset),
                    subset,
                    coverage,
                    group,
                )
            )
    candidates.sort(key=lambda item: item[:3])
    if candidates:
        _, _, _, subset, coverage, group = candidates[0]
        freeze_status = "PROPOSE_G4_PARAMETER_FREEZE_FOR_HAUPTLEITUNG_REVIEW"
        selected: list[str] = list(subset)
        group_evidence: dict[str, Any] | None = group
    else:
        freeze_status = "WITHHOLD_G4_PARAMETER_FREEZE"
        selected = []
        coverage = 0.0
        group_evidence = None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": freeze_status,
        "selected_parameters": selected,
        "selected_parameter_contract": {
            name: {
                "default": PARAMETER_SPECS[name].default,
                "lower_bound": PARAMETER_SPECS[name].lower,
                "upper_bound": PARAMETER_SPECS[name].upper,
                "unit": PARAMETER_SPECS[name].unit,
                "coordinate": (
                    "mass_thrust(theta)=float32(132000)*exp(theta)"
                    if name == "mass_thrust"
                    else "p(theta)=p_default*exp(theta)"
                ),
                "theta_zero_is_repository_default": True,
            }
            for name in selected
        },
        "parameter_disposition": {
            name: {
                "single_parameter_category": singles[name]["category"],
                "selected": name in selected,
                "exclusion_reasons": (
                    []
                    if name in selected
                    else singles[name]["withholding_reasons"]
                    if singles[name]["category"] == WITHHELD_PARAMETER_CATEGORY
                    else ["not individually GO-eligible"]
                    if singles[name]["category"] != "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION"
                    else ["not required by the smallest qualifying 90-percent-energy joint subset"]
                ),
                "withholding_reasons": singles[name]["withholding_reasons"],
                "objective_gradient_eligible": singles[name]["objective_gradient_eligible"],
                "objective_gradient_withholding_reason": singles[name][
                    "objective_gradient_withholding_reason"
                ],
                "technical_robustness_eligible": singles[name]["technical_robustness_eligible"],
                "technical_robustness_reason": singles[name]["technical_robustness_reason"],
            }
            for name in PARAMETER_NAMES
        },
        "normalized_sensitivity_energy_coverage": coverage,
        "selection_group_evidence": group_evidence,
        "selection_rule": {
            "members_individually_go_eligible": True,
            "requires_xy_and_z_effective_members": True,
            "minimum_energy_coverage": 0.90,
            "joint_gates_required": True,
            "tie_break": [
                "smallest_parameter_count",
                "lowest_condition_number",
                "mandatory_parameter_order",
            ],
        },
        "four_pd_gains_are_fallback_not_predecision": True,
        "g4_executed": False,
        "g5_executed": False,
        "training_contract": {
            "population": "Soft/Nominal/Dynamic balanced Hold/Climb/Descent; Near-limit separate",
            "batch_schedule": _g4_batch_schedule(),
            "resource_fallback_decision": "WITHHELD_PENDING_REBENCHMARK",
            "g4_batch_freeze_status": "WITHHELD_PENDING_REBENCHMARK",
            "g3_runtime_fallback_used": False,
            "g3_execution_mode": execution_mode,
            "g3_execution_selected_by": G3_004_SELECTION_DECISION,
            "development_seed": 9104001,
            "primary_objective": "mean feasible Train Loss v1",
            "secondary_metrics": [
                *(f"loss_v1_{term}" for term in LOSS_TERM_NAMES),
                "position_rmse_xy",
                "position_rmse_z",
                "velocity_rmse_xy",
                "velocity_rmse_z",
                "mean_signed_z_error",
                "terminal_signed_z_error",
                "integral_rms_max_terminal_by_axis",
                "integral_z_nearness_and_contact",
                "control_effort",
                "control_smoothness",
                "minimum_normalized_motor_reserve",
                "stage_a_torque_motor_clip_rate_duration_bound_channel",
                "stage_b_motor_clip_rate_duration_bound_channel",
                "requested_wrench_by_axis_and_class",
                "realized_wrench_by_axis_and_class",
                "absolute_and_platform_normalized_wrench_distortion",
            ],
            "resume_state": [
                "physical_and_raw_parameters",
                "optimizer_state",
                "step",
                "rng",
                "source_commit",
                "contract_manifest_loss_hashes",
            ],
            "resume_reproduction": "next ten steps",
            "plateau": "after >=1000 updates; three windows, median 250-vs-250 improvement <0.2%",
            "gradient_convergence": "median dimensionless L2 last 100 <=0.1 initial or <=1e-4",
            "validation": "fixed checkpoints; lowest valid mean, earlier when difference <=1e-6",
            "validation_updates_or_episode_tuning": False,
            "test": "closed",
        },
        "prospective_g5_gates": {
            "comparison_baseline": "repository-default parameters",
            "comparison_parent_identity": "identical fixed Validation parents",
            "relative_primary_improvement_formula": "(default-candidate)/default",
            "relative_regression_formula": "(candidate-default)/default",
            "relative_regression_grouping": [
                "feasible_motion_class_mean",
                "feasible_vertical_profile_mean",
            ],
            "candidate_and_default_use_identical_contract_and_inputs": True,
            "near_limit_decides_nominal_pass": False,
            "mean_primary_improvement_min": 0.05,
            "maximum_feasible_stratum_degradation": 0.02,
            "xy_z_rmse_max_degradation": 0.02,
            "at_least_one_rmse_improves": True,
            "integral_total_contact_increase_allowed": 0,
            "integral_nearness_absolute_degradation_max": 0.02,
            "soft_nominal_zero_clips": True,
            "dynamic_stage_b_zero": True,
            "dynamic_stage_a_motor_clip_fraction_max": 0.0025,
            "dynamic_clip_interval_max_s": 0.02,
            "minimum_feasible_motor_reserve": 0.05,
            "maximum_reserve_drop": 0.02,
            "wrench_degradation_bound": "max(1% platform scale, 5% default value)",
            "hard_gates": [
                "contract",
                "finite",
                "ground_floor",
                "zero_thrust",
                "reproduction",
                "checksum",
                "technical",
            ],
        },
        "claim_boundary": "proposal only; no G4 run, candidate selection, or approval",
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def _execution_batch_partition(
    execution_mode: str, constructed: tuple[tuple[EpisodeSpec, ReferenceCandidate], ...]
) -> tuple[tuple[str, tuple[tuple[EpisodeSpec, ReferenceCandidate], ...]], ...]:
    """Return a frozen partition without running a simulation."""
    if execution_mode == G3_004_EXECUTION_MODE:
        batches = tuple(
            (label, constructed[start:stop]) for label, start, stop, _split in G3_004_BATCH_SLICES
        )
        expected_counts = list(G3_004_BATCH_PARENT_COUNTS)
        for (_label, items), (_, _start, _stop, expected_split) in zip(
            batches, G3_004_BATCH_SLICES, strict=True
        ):
            if [spec.split.value for spec, _ in items] != [expected_split] * len(items):
                raise G3ContractError("six-batch partition changed Train/Validation separation")
    elif execution_mode == "primary-24":
        # Retained only for historical diagnostic comparison; no final path selects it.
        batches = (("primary-24", constructed),)
        expected_counts = [24]
    elif execution_mode == "split-12-plus-12":
        # Retained only for historical diagnostic comparison; no final path selects it.
        batches = (
            ("train-12", tuple(item for item in constructed if item[0].split is Split.TRAIN)),
            (
                "validation-12",
                tuple(item for item in constructed if item[0].split is Split.VALIDATION),
            ),
        )
        expected_counts = [12, 12]
    else:
        raise G3ContractError(f"unsupported execution mode {execution_mode}")
    expected_ids = tuple(spec.episode_id for spec, _ in constructed)
    observed_ids = tuple(spec.episode_id for _, items in batches for spec, _ in items)
    if (
        len(constructed) != 24
        or observed_ids != expected_ids
        or [len(items) for _, items in batches] != expected_counts
    ):
        raise G3ContractError("execution partition changed the exact 24-parent identity/order")
    return batches


def _batch_aggregation_contract(
    execution_mode: str,
    science_batches: tuple[dict[str, Any], ...],
    expected_ids: tuple[str, ...],
    batch_timeout_seconds: int,
    total_timeout_seconds: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Validate identity/inventory before any cross-batch scientific aggregation."""
    if execution_mode != G3_004_EXECUTION_MODE:
        raise G3ContractError("final aggregation requires the D-073 six-batch mode")
    expected_counts = list(G3_004_BATCH_PARENT_COUNTS)
    expected_labels = [item[0] for item in G3_004_BATCH_SLICES]
    observed_ids = tuple(
        parent_id for batch in science_batches for parent_id in batch["parent_ids"]
    )
    if observed_ids != expected_ids or [batch["parent_count"] for batch in science_batches] != (
        expected_counts
    ):
        raise G3ContractError("execution batching changed the exact 24-parent identity/order")
    if (
        [batch["batch_label"] for batch in science_batches] != expected_labels
        or [batch["deadline_seconds"] for batch in science_batches] != [900] * 6
        or not all(batch["deadline_passed"] for batch in science_batches)
    ):
        raise G3ContractError("execution labels or four-parent deadline contract changed")
    feature_names = science_batches[0]["feature_names"]
    if any(batch["feature_names"] != feature_names for batch in science_batches):
        raise G3ContractError("execution batching changed the feature inventory")
    contract = {
        "schema_version": G3_004_BATCH_EXECUTION_SCHEMA,
        "mode": execution_mode,
        "selection_decision": G3_004_SELECTION_DECISION,
        "selection_policy": G3_004_SELECTION_POLICY,
        "runtime_fallback_used": False,
        "runtime_fallback_reason": None,
        "selected_by_host_memory": False,
        "batch_count": len(science_batches),
        "batch_timeout_seconds": batch_timeout_seconds,
        "total_timeout_seconds": total_timeout_seconds,
        "batches": [
            {
                key: batch[key]
                for key in (
                    "batch_label",
                    "parent_ids",
                    "parent_count",
                    "split_order",
                    "deadline_seconds",
                    "deadline_passed",
                )
            }
            for batch in science_batches
        ],
        "aggregation": {
            "parent_order_equal_to_frozen_24_registry": True,
            "feature_inventory_identical_across_batches": True,
            "concatenation_axis": "world/parent axis 0",
            "global_gates_applied_only_after_24-parent_concatenation": True,
            "split_scalar_aggregation": (
                "sum(batch_mean_i * contributing_parent_count_i) / sum(contributing_parent_count_i)"
            ),
            "feasible_parent_counts_per_split_batch": [4, 4, 1],
            "whole_split_statistic_denominator": G3_004_FEASIBLE_SPLIT_DENOMINATOR,
        },
        "g4_batch_freeze_status": "WITHHELD_PENDING_REBENCHMARK",
        "G4": "WITHHELD_PENDING_REBENCHMARK",
    }
    return feature_names, contract


@lru_cache(maxsize=1)
def build_freeze_evidence(
    execution_mode: str = G3_004_EXECUTION_MODE,
    batch_timeout_seconds: int = 900,
    total_timeout_seconds: int = 2400,
) -> dict[str, Any]:
    """Run the prospectively fixed six contiguous four-parent batches."""
    if execution_mode != G3_004_EXECUTION_MODE:
        raise G3ContractError(f"unsupported execution mode {execution_mode}")
    if batch_timeout_seconds != 900 or total_timeout_seconds != 2400:
        raise G3ContractError("resource timeouts changed from the frozen 900/2400 s contract")
    total_started = time.monotonic()
    smoke = build_g3_004_smoke()
    if smoke["status"] != "PASS_ALL_SEVEN_PARAMETER_REVERSE_SMOKE":
        raise G3ContractError("all-seven reverse smoke did not pass")
    discrete = discrete_integer_response()
    if discrete["status"] != "DISCRETE_INTEGER_RESPONSE":
        raise G3ContractError("discrete response did not complete separately")
    constructed = construct_all_episodes()
    batch_inputs = _execution_batch_partition(execution_mode, constructed)
    science_batches = tuple(
        _science_batch(label, items, primary=False, batch_timeout_seconds=batch_timeout_seconds)
        for label, items in batch_inputs
    )
    if time.monotonic() - total_started > total_timeout_seconds:
        raise G3ContractError("complete diagnostic exceeded the frozen 2400 s total limit")
    expected_ids = tuple(spec.episode_id for spec, _ in constructed)
    expected_counts = list(G3_004_BATCH_PARENT_COUNTS)
    feature_names, batch_execution = _batch_aggregation_contract(
        execution_mode, science_batches, expected_ids, batch_timeout_seconds, total_timeout_seconds
    )
    baseline_features = np.concatenate(
        tuple(batch["baseline_features"] for batch in science_batches), axis=0
    )
    segment_features = [record for batch in science_batches for record in batch["segment_features"]]
    specs = tuple(spec for spec, _ in constructed)
    diagnostics = {
        name: _parameter_diagnostic(
            tuple(batch["parameter_records"][name] for batch in science_batches),
            specs,
            name,
            feature_names,
        )
        for name in PARAMETER_NAMES
    }
    parity = {
        "status": "PASS_24_PARENT_DEFAULT_PARITY",
        "execution_mode": execution_mode,
        "parent_count": 24,
        "parent_ids": list(expected_ids),
        "batch_parent_counts": expected_counts,
        "array_equal": True,
        "allowed_internal_exception": "mass_thrust integer-vs-Float32 dtype only",
        "batches": [batch["parity"] for batch in science_batches],
        "technical_gates": [batch["parity"]["technical_gates"] for batch in science_batches],
        "technical_details": [
            record for batch in science_batches for record in batch["parity"]["technical_details"]
        ],
    }
    conditioning, matrices = _conditioning_payload(diagnostics, feature_names, specs)
    trajectory_contract = _trajectory_contract_payload(constructed)
    train_manifest = _manifest_payload(Split.TRAIN, constructed)
    validation_manifest = _manifest_payload(Split.VALIDATION, constructed)
    semantics = parameter_semantics_payload(
        science_batches[0]["initial_data"], baseline_features, feature_names
    )
    loss_diagnostics = _loss_diagnostics_payload(
        diagnostics, feature_names, specs, baseline_features, segment_features
    )
    proposal = _freeze_proposal_payload(conditioning, matrices, specs, execution_mode)
    withheld_parameters = [
        name
        for name in PARAMETER_NAMES
        if conditioning["single_parameters"][name]["category"] == WITHHELD_PARAMETER_CATEGORY
    ]
    withheld_objective_parameters = [
        name
        for name in withheld_parameters
        if WITHHELD_OBJECTIVE_GRADIENT_INVALID
        in conditioning["single_parameters"][name]["withholding_reasons"]
    ]
    withheld_technical_parameters = [
        name
        for name in withheld_parameters
        if WITHHELD_TECHNICAL_ROBUSTNESS
        in conditioning["single_parameters"][name]["withholding_reasons"]
    ]
    report_diagnostics = {}
    for name, diagnostic in diagnostics.items():
        report_diagnostics[name] = {
            key: value
            for key, value in diagnostic.items()
            if key not in {"baseline", "reverse", "fd", "effect_plus", "effect_minus"}
        }
        report_diagnostics[name]["continuous_arrays"] = {
            "baseline": diagnostic["baseline"].tolist(),
            "reverse": diagnostic["reverse"].tolist(),
            "fd": diagnostic["fd"].tolist(),
            "effect_plus": diagnostic["effect_plus"].tolist(),
            "effect_minus": diagnostic["effect_minus"].tolist(),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": G3_004_COMPLETION_STATUS,
        "work_order": "WO-GR-G3-004",
        "platform": PLATFORM,
        "source_base_commit": SOURCE_BASE_COMMIT,
        "scope": {
            "simulation_only": True,
            "dynamics": "Dynamics.first_principles",
            "integrator": "Euler",
            "jax_dtype": "Float32/x64 disabled",
            "backend": "CPU",
            "mjx_impl": "Impl.JAX",
            "warp_installed": False,
            "execution_mode": execution_mode,
            "selection_decision": G3_004_SELECTION_DECISION,
            "selection_policy": G3_004_SELECTION_POLICY,
            "selected_by_host_memory": False,
            "runtime_fallback_used": False,
            "runtime_fallback_reason": None,
            "controller_defaults_modified": False,
            "canonical_integer_mass_thrust_modified": False,
            "test_access": False,
            "optimizer_initialization_count": 0,
            "optimizer_update_count": 0,
        },
        "focused_gates": {
            "default_parity": focused_default_parity(),
            "all_seven_parameter_reverse_smoke": smoke,
            "discrete_integer_response_reference": (
                "stored separately in sensitivity_conditioning.json"
            ),
        },
        "g3_003_forward_limitation": {
            "classification": "FORWARD_ONLY",
            "first_nonfinite_prefix": 1,
            "first_nonfinite_time_s": 0.01,
            "output_path": "controls/attitude/last_ang_vel",
            "split_interval_reproduced": False,
            "narrowest_claim_boundary": "full_rollout_carry_output",
            "individual_operation_or_singularity_claim": None,
            "parameter_classification_gate": False,
        },
        "full_default_parity": parity,
        "batch_execution": batch_execution,
        "feature_names": list(feature_names),
        "output_role_contract": _output_role_inventory(feature_names),
        "baseline_features": baseline_features.tolist(),
        "segment_features": segment_features,
        "continuous_parameter_diagnostics": report_diagnostics,
        "single_parameter_categories": conditioning["single_parameters"],
        "group_categories": conditioning["groups"],
        "withheld_parameters": withheld_parameters,
        "withheld_objective_gradient_parameters": withheld_objective_parameters,
        "withheld_technical_robustness_parameters": withheld_technical_parameters,
        "loss_v1_unchanged": True,
        "loss_v2_active_weight": 0.0,
        "g4_executed": False,
        "g5_executed": False,
        "discrete_integer_response_mixed_with_continuous_reverse": False,
        "forward_mode_executed_for_parameter_classification": False,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
        "claim_boundary": (
            "Pure simulation identifiability diagnostics; no optimized candidate, physical "
            "calibration, integer/firmware differentiability, hardware transfer, safety, flight, "
            "or Sim2Real claim."
        ),
    }
    return {
        "report": report,
        "trajectory_contract": trajectory_contract,
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "parameter_semantics": semantics,
        "sensitivity_conditioning": conditioning
        | {
            "focused_reverse_smoke": smoke,
            "continuous_parameter_records": {
                name: {
                    "coordinate": diagnostic["coordinate"],
                    "default": diagnostic["default"],
                    "bounds": diagnostic["bounds"],
                    "unit": diagnostic["unit"],
                    "isolation": diagnostic["isolation"],
                    "theta_zero_default_parity": diagnostic["theta_zero_default_parity"],
                    "output_role_contract": diagnostic["output_role_contract"],
                    "objective_gradient_contract": diagnostic["objective_gradient_contract"],
                    "diagnostic_ad_disclosure": diagnostic["diagnostic_ad_disclosure"],
                    "reverse_fd_loss_comparison": diagnostic["reverse_fd_loss_comparison"],
                    "reverse_fd_feature_comparison": diagnostic["reverse_fd_feature_comparison"],
                    "feature_tolerance_mismatch_count": diagnostic[
                        "feature_tolerance_mismatch_count"
                    ],
                    "feature_tolerance_mismatch_strata": diagnostic[
                        "feature_tolerance_mismatch_strata"
                    ],
                    "mismatch_strata": diagnostic["mismatch_strata"],
                    "allowed_mismatch_strata": diagnostic["allowed_mismatch_strata"],
                    "split_aggregates": diagnostic["split_aggregates"],
                    "baseline": diagnostic["baseline"].tolist(),
                    "reverse": diagnostic["reverse"].tolist(),
                    "central_fd": diagnostic["fd"].tolist(),
                    "effect_plus_0p05": diagnostic["effect_plus"].tolist(),
                    "effect_minus_0p05": diagnostic["effect_minus"].tolist(),
                    "forward_mode_executed": False,
                }
                for name, diagnostic in diagnostics.items()
            },
            "continuous_normalized_sensitivity": {
                name: matrix.tolist() for name, matrix in matrices.items()
            },
            "discrete_integer_response": discrete,
        },
        "loss_diagnostics": loss_diagnostics,
        "g4_g5_freeze_proposal": proposal,
        "feature_names": feature_names,
        "matrices": matrices,
        "specs": specs,
        "batch_execution": batch_execution,
        "runtime_batch_records": [
            {
                "batch_label": batch["batch_label"],
                "elapsed_seconds": batch["elapsed_seconds"],
                "deadline_seconds": batch["deadline_seconds"],
                "deadline_passed": batch["deadline_passed"],
            }
            for batch in science_batches
        ],
    }
