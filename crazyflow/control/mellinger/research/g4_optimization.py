"""Bounded G4 joint optimization for exactly ``kp_xy`` and ``kp_z``.

This module deliberately keeps every experiment-specific parameter replacement local to an
immutable :class:`~crazyflow.sim.data.SimData` value.  It validates the frozen public Day-26
inputs before constructing parents or entering a rollout, uses the unchanged pure-JAX
Mellinger/dynamics path, and exposes technical M0--M6 evidence only.  It does not select a
candidate or make a hardware, firmware, or flight claim.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import crazyflow
from crazyflow.control.mellinger.control import state2attitude
from crazyflow.control.mellinger.research import g3_freeze_v2 as g3
from crazyflow.control.mellinger.research import robust_evaluation as robust
from crazyflow.control.mellinger.tracking import (
    TrackingLossConfig,
    hover_rotor_velocity,
    rollout_state_commands,
    rotor_velocity_limits,
    tracking_loss_per_case,
    tracking_loss_terms_per_case,
)
from crazyflow.trajectory import Trajectory

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData


SCHEMA_VERSION = "crazyflow.mellinger_g4_joint.v1"
CHECKPOINT_SCHEMA_VERSION = "crazyflow.mellinger_g4_checkpoint.v1"
SOURCE_ORIGIN_SCHEMA_VERSION = "crazyflow.mellinger_g4_source_origin.v1"
LONGRUN_CONTRACT_SCHEMA_VERSION = "crazyflow.mellinger_g4_longrun_contract.v1"
LONGRUN_CHECKPOINT_SCHEMA_VERSION = "crazyflow.mellinger_g4_longrun_checkpoint.v1"
LONGRUN_SEGMENT_SCHEMA_VERSION = "crazyflow.mellinger_g4_longrun_segment.v1"
LONGRUN_RESOURCE_EVIDENCE_SCHEMA_VERSION = "crazyflow.mellinger_g4_longrun_resource_evidence.v1"
M8_CONTRACT_SCHEMA_VERSION = "crazyflow.mellinger_g4_m8_contract.v1"
M8_CHECKPOINT_SCHEMA_VERSION = "crazyflow.mellinger_g4_m8_checkpoint.v1"
M8_VALIDATION_SCHEMA_VERSION = "crazyflow.mellinger_g4_m8_validation.v1"
M8_SEGMENT_SCHEMA_VERSION = "crazyflow.mellinger_g4_m8_segment.v1"
BACKEND_INPUT_MANIFEST_SCHEMA_VERSION = "crazyflow.mellinger_g4_backend_input_manifest.v1"
BACKEND_RUNTIME_SCHEMA_VERSION = "crazyflow.mellinger_g4_backend_runtime.v1"
BACKEND_SOURCE_ORIGIN_SCHEMA_VERSION = "crazyflow.mellinger_g4_backend_source_origin.v1"
BACKEND_PARITY_SCHEMA_VERSION = "crazyflow.mellinger_g4_backend_parity_evidence.v1"
BACKEND_THROUGHPUT_SCHEMA_VERSION = "crazyflow.mellinger_g4_backend_throughput_evidence.v1"
BACKEND_RESUME_SCHEMA_VERSION = "crazyflow.mellinger_g4_backend_resume_segment.v1"
BACKEND_NEUTRAL_PARENT_SCHEMA_VERSION = "crazyflow.mellinger_g4_backend_neutral_parents.v1"
SOURCE_BASE_COMMIT = "c57312201d7ac56d3c6556738e07e4270e0729e5"
EXPECTED_REPOSITORY_ROOT = Path(
    "/home/noah3/bachelorarbeit/worktrees/crazyflow-gradient-research-wo-gr-g4-004"
)
EXPECTED_INTERPRETER = Path(
    "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python"
)
CONFIG_RELATIVE_PATH = "configs/research/mellinger/g4_joint_v1.json"
BACKEND_INPUT_MANIFEST_RELATIVE_PATH = "configs/research/mellinger/g4_backend_evidence_v1.json"
BACKEND_NEUTRAL_PARENT_RELATIVE_PATH = (
    "artifacts/day26-g3-identifiability-freeze-v2/backend_neutral_parents_v1.json.gz"
)
DAY26_DIRECTORY = "artifacts/day26-g3-identifiability-freeze-v2"
DAY26_SCHEMA_VERSION = g3.SCHEMA_VERSION
DEVELOPMENT_SEED = 9_104_001
ROLLOUT_INTERVALS = 600
SCORE_INTERVALS = 200
STEPS_PER_CONTROL = 5
MAX_PARENT_ATTEMPTS = 64
MICROBATCH_SIZE = 4
EFFECTIVE_BATCH_SIZE = 32
MICROBATCH_COUNT = 8
MAX_UPDATES = 2_000
LONGRUN_MAX_UPDATES = 100
M8_MAX_UPDATES = 2_000
M8_VALIDATION_UPDATES = tuple(range(0, M8_MAX_UPDATES + 1, 250))
ADAM_LEARNING_RATE = 1.0e-3
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1.0e-8
ADAM_EPS_ROOT = 0.0
NUMERIC_RTOL = 1.0e-5
NUMERIC_ATOL = 1.0e-7
MINIMUM_AVAILABLE_BYTES = 8 * 1024**3
RSS_FRACTION_LIMIT = 0.70
MINIMUM_SIGNIFICANT_GRADIENT = 1.0e-8

PARAMETER_NAMES = ("kp_xy", "kp_z")
LOSS_TERM_NAMES = ("position", "velocity", "effort", "smoothness", "terminal", "altitude")
STRATA = (
    (g3.MotionClass.SOFT, g3.Profile.HOLD),
    (g3.MotionClass.SOFT, g3.Profile.CLIMB),
    (g3.MotionClass.SOFT, g3.Profile.DESCENT),
    (g3.MotionClass.NOMINAL, g3.Profile.HOLD),
    (g3.MotionClass.NOMINAL, g3.Profile.CLIMB),
    (g3.MotionClass.NOMINAL, g3.Profile.DESCENT),
    (g3.MotionClass.DYNAMIC, g3.Profile.HOLD),
    (g3.MotionClass.DYNAMIC, g3.Profile.CLIMB),
    (g3.MotionClass.DYNAMIC, g3.Profile.DESCENT),
)
BATCH_STRATUM_COUNTS = (
    (4, 4, 4, 4, 4, 3, 3, 3, 3),
    (3, 4, 4, 4, 4, 4, 3, 3, 3),
    (3, 3, 4, 4, 4, 4, 4, 3, 3),
    (3, 3, 3, 4, 4, 4, 4, 4, 3),
    (3, 3, 3, 3, 4, 4, 4, 4, 4),
    (4, 3, 3, 3, 3, 4, 4, 4, 4),
    (4, 4, 3, 3, 3, 3, 4, 4, 4),
    (4, 4, 4, 3, 3, 3, 3, 4, 4),
    (4, 4, 4, 4, 3, 3, 3, 3, 4),
)

DAY26_INPUTS = {
    f"{DAY26_DIRECTORY}/trajectory_contract.json": {
        "sha256": "d47912f63e23299e2c3ef13b5789f13e03718ded069d2e9b0955f72e4836b583",
        "keys": {
            "contract",
            "episodes",
            "optimizer_initialization_count",
            "optimizer_update_count",
            "schema_version",
            "status",
        },
    },
    f"{DAY26_DIRECTORY}/train_manifest.json": {
        "sha256": "6d6a8e1a6ecc6221fc7fae65a1ac216a050c73419a0799e18687419e973b7e4e",
        "keys": {
            "cross_split_leakage_count",
            "optimizer_initialization_count",
            "optimizer_update_count",
            "parent_count",
            "parents",
            "schema_version",
            "split",
            "split_disjointness_coordinates",
            "test_manifest",
        },
    },
    f"{DAY26_DIRECTORY}/validation_manifest.json": {
        "sha256": "58159451316e0d0cd76724b2add675ccd55b2d70e3fcc733140a7f148f1fb118",
        "keys": {
            "cross_split_leakage_count",
            "optimizer_initialization_count",
            "optimizer_update_count",
            "parent_count",
            "parents",
            "schema_version",
            "split",
            "split_disjointness_coordinates",
            "test_manifest",
        },
    },
    f"{DAY26_DIRECTORY}/g4_g5_freeze_proposal.json": {
        "sha256": "165a443664a8f0a123b1ce99680fb332eb34664d1b98c9c7f4ecfdd4a0381a38",
        "keys": {
            "claim_boundary",
            "g4_executed",
            "g5_executed",
            "optimizer_initialization_count",
            "optimizer_update_count",
            "parameter_disposition",
            "prospective_g5_gates",
            "schema_version",
            "selected_parameter_contract",
            "selected_parameters",
            "selection_group_evidence",
            "selection_rule",
            "status",
            "training_contract",
        },
    },
    f"{DAY26_DIRECTORY}/loss_diagnostics.json": {
        "sha256": "2fe46c85cce24762c4e0395a241830e277e50735fa86464e0890b3514c09b780",
        "keys": {
            "inactive_candidate",
            "loss_v1",
            "optimizer_initialization_count",
            "optimizer_update_count",
            "schema_version",
            "status",
        },
    },
    f"{DAY26_DIRECTORY}/parameter_semantics.json": {
        "sha256": "743fba3250e919913fae775a139e049ca6f1112cca6f26d516be9d25736a2a84",
        "keys": {
            "additional_parameter_groups",
            "authorized_source_inventory",
            "canonical_defaults_modified",
            "conversion_chain",
            "optimizer_initialization_count",
            "optimizer_update_count",
            "parameters",
            "schema_version",
            "shared_controller_modified",
            "status",
        },
    },
}

WRITE_PATHS = (
    "crazyflow/control/mellinger/research/g4_optimization.py",
    "examples/jax/mellinger_g4_optimization.py",
    CONFIG_RELATIVE_PATH,
    "tests/unit/test_mellinger_g4_optimization.py",
    "tests/integration/test_mellinger_g4_optimization.py",
)
PYTHON_WRITE_PATHS = tuple(path for path in WRITE_PATHS if path.endswith(".py"))
PROTECTED_ROOT_SENTINELS = (
    "BACHELORARBEIT_VERSTAENDNIS_UEBERGABE.md",
    "artifacts/day10-sparse-long-pilot/runs/",
    "artifacts/day13-h100-replication/",
)
CLOSED_TEST_SENTINEL = "configs/research/mellinger/test_manifest.v1.json"

VALIDATION_IDENTITIES = (
    "validation-soft-hold-8401001/attempt-0",
    "validation-soft-climb-8401002/attempt-1",
    "validation-soft-descent-8401003/attempt-0",
    "validation-nominal-hold-8401101/attempt-2",
    "validation-nominal-climb-8401102/attempt-1",
    "validation-nominal-descent-8401103/attempt-1",
    "validation-dynamic-hold-8401201/attempt-5",
    "validation-dynamic-climb-8401202/attempt-4",
    "validation-dynamic-descent-8401203/attempt-1",
)
NEAR_LIMIT_IDENTITIES = (
    "train-near_limit-hold-8301301/attempt-9",
    "train-near_limit-climb-8301302/attempt-0",
    "train-near_limit-descent-8301303/attempt-37",
    "validation-near_limit-hold-8401301/attempt-22",
    "validation-near_limit-climb-8401302/attempt-4",
    "validation-near_limit-descent-8401303/attempt-17",
)


class G4ContractError(RuntimeError):
    """Raised before continuing past a violated G4 contract."""


@dataclass(frozen=True)
class ParameterSpec:
    """One and only one authorized positive log-coordinate."""

    name: str
    indices: tuple[int, ...]
    contract_default: float
    runtime_default: float
    physical_lower: float
    physical_upper: float
    raw_lower: float
    raw_upper: float


PARAMETER_SPECS = {
    item.name: item
    for item in (
        ParameterSpec(
            "kp_xy",
            (0, 1),
            0.4,
            0.4000000059604645,
            0.1,
            1.2,
            -1.3862943611198906,
            1.0986122886681098,
        ),
        ParameterSpec("kp_z", (2,), 1.25, 1.25, 0.3, 2.5, -1.4271163556401458, 0.6931471805599453),
    )
}


@dataclass(frozen=True)
class G4ParentSpec:
    """One frozen Train parent coordinate."""

    batch_index: int
    stratum_index: int
    slot_index: int
    motion_class: g3.MotionClass
    profile: g3.Profile
    episode_id: str
    pair_id: str
    warmup_s: float
    score_start: int
    score_stop: int


@dataclass(frozen=True)
class AdamState:
    """The complete numerical state of the fixed projected Adam optimizer."""

    count: int
    first_moment: np.ndarray
    second_moment: np.ndarray


@dataclass(frozen=True)
class OptimizationState:
    """Checkpointable G4 state."""

    theta: np.ndarray
    adam: AdamState
    projection_count: int
    gradient_history: tuple[tuple[float, ...], ...]
    selected_validation_step: int
    selected_validation_loss: float | None
    loss_history: tuple[float, ...] = ()


@dataclass(frozen=True)
class M8OptimizationState:
    """Complete bounded M8 state, separate from every M7 serializer."""

    theta: np.ndarray
    adam: AdamState
    projection_count: int
    loss_window: tuple[float, ...]
    loss_window_start: int
    initial_gradient_window: tuple[float, ...]
    recent_gradient_window: tuple[float, ...]
    recent_gradient_window_start: int
    validation_state: dict[str, Any]
    convergence_state: dict[str, Any]
    forwardability_state: dict[str, Any]
    terminal_state: dict[str, Any]


@dataclass(frozen=True)
class EvaluationRuntime:
    """One shape- and parameter-specific compiled evaluation runtime."""

    sim: Any
    evaluate: Callable[..., Any]


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable, strict, newline-terminated JSON bytes."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 of *value*."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one already-authorized regular file without following a symlink."""
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise G4ContractError(f"not an ordinary regular file: {path}")
    return sha256_bytes(path.read_bytes())


def _array_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    raw = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_hex": raw.hex(),
        "sha256": sha256_bytes(raw),
    }


def _restore_array(record: dict[str, Any], *, label: str) -> np.ndarray:
    required = {"dtype", "shape", "data_hex", "sha256"}
    if set(record) != required:
        raise G4ContractError(f"{label} array record keys changed")
    raw = bytes.fromhex(record["data_hex"])
    if sha256_bytes(raw) != record["sha256"]:
        raise G4ContractError(f"{label} array checksum mismatch")
    result = np.frombuffer(raw, dtype=np.dtype(record["dtype"])).copy()
    expected_size = math.prod(record["shape"])
    if result.size != expected_size:
        raise G4ContractError(f"{label} array shape/size mismatch")
    return result.reshape(tuple(record["shape"]))


def _validate_lexical_relative_path(relative_path: str) -> PurePosixPath:
    if not relative_path or "\\" in relative_path or relative_path.startswith("/"):
        raise G4ContractError(f"unsafe repository path: {relative_path!r}")
    path = PurePosixPath(relative_path)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise G4ContractError(f"repository traversal is forbidden: {relative_path!r}")
    return path


def validate_discovery_target(relative_path: str) -> str:
    """Reject root discovery, protected paths, separators, traversal, and case aliases."""
    path = _validate_lexical_relative_path(relative_path)
    normalized = path.as_posix()
    protected = set(PROTECTED_ROOT_SENTINELS) | {CLOSED_TEST_SENTINEL}
    if normalized in protected or any(
        normalized.startswith(item.rstrip("/") + "/") for item in protected
    ):
        raise G4ContractError("protected/closed-test discovery is forbidden")
    allowed_exact = (
        set(DAY26_INPUTS)
        | set(WRITE_PATHS)
        | {"AGENTS.md", "crazyflow/__init__.py", "crazyflow/control/mellinger/research"}
    )
    if normalized not in allowed_exact:
        raise G4ContractError(f"target is outside bounded G4 discovery: {normalized}")
    if relative_path != normalized:
        raise G4ContractError("noncanonical separator/path spelling")
    return normalized


def _safe_exact_file(repository_root: Path, relative_path: str) -> Path:
    normalized = validate_discovery_target(relative_path)
    root = repository_root.resolve(strict=True)
    path = root.joinpath(*PurePosixPath(normalized).parts)
    current = root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise G4ContractError(f"symlink traversal is forbidden: {relative_path}")
    if path.resolve(strict=True) != path:
        raise G4ContractError(f"noncanonical or case-aliased path: {relative_path}")
    return path


def protected_contract() -> dict[str, Any]:
    """Return only opaque sentinel strings; never inspect their filesystem targets."""
    return {
        "opaque_roots": list(PROTECTED_ROOT_SENTINELS),
        "closed_test": CLOSED_TEST_SENTINEL,
        "closed_test_role": "opaque_string_only",
        "children_listed": False,
        "contents_opened_or_hashed": False,
    }


def _git_mode(repository_root: Path, relative_path: str) -> str:
    result = subprocess.run(
        ("git", "ls-files", "-s", "--", relative_path),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = result.stdout.strip().split()
    if len(fields) != 4 or fields[2] != "0" or fields[3] != relative_path:
        raise G4ContractError(f"committed file identity changed: {relative_path}")
    return fields[0]


def _require_public_payload_contract(relative_path: str, payload: dict[str, Any]) -> None:
    expected = DAY26_INPUTS[relative_path]
    missing = expected["keys"] - set(payload)
    if missing:
        raise G4ContractError(f"{relative_path} missing keys: {sorted(missing)}")
    if payload["schema_version"] != DAY26_SCHEMA_VERSION:
        raise G4ContractError(f"{relative_path} schema changed")
    if (
        payload.get("optimizer_initialization_count") != 0
        or payload.get("optimizer_update_count") != 0
    ):
        raise G4ContractError(f"{relative_path} unexpectedly contains optimizer activity")
    name = PurePosixPath(relative_path).name
    if name == "trajectory_contract.json":
        if not isinstance(payload["contract"], dict) or len(payload["episodes"]) != 24:
            raise G4ContractError("trajectory contract structure changed")
        if payload["contract"].get("test_constructed_loaded_listed_or_hashed") is not False:
            raise G4ContractError("trajectory contract test sentinel changed")
    elif name in {"train_manifest.json", "validation_manifest.json"}:
        expected_split = name.removesuffix("_manifest.json")
        if (
            payload["split"] != expected_split
            or payload["parent_count"] != 12
            or len(payload["parents"]) != 12
            or payload["cross_split_leakage_count"] != 0
            or payload["test_manifest"] is not None
        ):
            raise G4ContractError(f"{name} split/parent/closed-test contract changed")
        parent_keys = {
            "attempt",
            "attempt_identity",
            "component_key_digests",
            "episode_id",
            "motion_class",
            "pair_id",
            "parent_digest",
            "profile",
            "seed",
        }
        if any(parent_keys - set(parent) for parent in payload["parents"]):
            raise G4ContractError(f"{name} parent record keys changed")
    elif name == "g4_g5_freeze_proposal.json":
        selected = payload["selected_parameter_contract"]
        training = payload["training_contract"]
        if (
            payload["selected_parameters"] != list(PARAMETER_NAMES)
            or set(selected) != set(PARAMETER_NAMES)
            or training.get("development_seed") != DEVELOPMENT_SEED
            or training.get("test") != "closed"
            or payload["g4_executed"] is not False
            or payload["g5_executed"] is not False
        ):
            raise G4ContractError("Day-26 G4 proposal contract changed")
    elif name == "loss_diagnostics.json":
        if payload["loss_v1"].get("active_and_numerically_unchanged") is not True:
            raise G4ContractError("Loss-v1 public contract changed")
    elif name == "parameter_semantics.json":
        parameters = payload["parameters"]
        for parameter, spec in PARAMETER_SPECS.items():
            record = parameters.get(parameter, {})
            if (
                record.get("default") != spec.contract_default
                or record.get("runtime_loaded_default") != spec.runtime_default
                or tuple(record.get("leaf_or_indices", ())) != spec.indices
            ):
                raise G4ContractError(f"{parameter} public semantics changed")
        if payload["canonical_defaults_modified"] or payload["shared_controller_modified"]:
            raise G4ContractError("public parameter semantics report modified shared defaults")


def validate_public_inputs(repository_root: Path) -> dict[str, Any]:
    """Validate all six and only the six public Day-26 files before science."""
    records: dict[str, Any] = {}
    for relative_path, expected in DAY26_INPUTS.items():
        path = _safe_exact_file(repository_root, relative_path)
        file_mode = stat.S_IMODE(path.lstat().st_mode)
        git_mode = _git_mode(repository_root, relative_path)
        digest = sha256_file(path)
        if file_mode != 0o644 or git_mode != "100644" or digest != expected["sha256"]:
            raise G4ContractError(f"public input mode/hash changed: {relative_path}")
        try:
            payload = json.loads(path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise G4ContractError(f"invalid public JSON: {relative_path}") from error
        if not isinstance(payload, dict):
            raise G4ContractError(f"public JSON is not a top-level object: {relative_path}")
        _require_public_payload_contract(relative_path, payload)
        records[relative_path] = {
            "sha256": digest,
            "file_mode": f"{file_mode:04o}",
            "git_mode": git_mode,
            "schema_version": payload["schema_version"],
            "top_level_keys": sorted(payload),
        }
    return {"status": "PASS_SIX_PUBLIC_DAY26_INPUTS", "files": records}


def _load_public_json(repository_root: Path, filename: str) -> dict[str, Any]:
    relative_path = f"{DAY26_DIRECTORY}/{filename}"
    path = _safe_exact_file(repository_root, relative_path)
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise G4ContractError(f"{filename} is not a JSON object")
    _require_public_payload_contract(relative_path, payload)
    return payload


def loss_v1_contract() -> dict[str, Any]:
    """Return the immutable six-term loss and the inactive Loss-v2 weight."""
    config = TrackingLossConfig()
    return {
        "terms": {
            "position": {"weight": config.position_weight, "scale": config.position_scale**2},
            "velocity": {"weight": config.velocity_weight, "scale": config.velocity_scale**2},
            "effort": {"weight": config.effort_weight, "scale": 1.0},
            "smoothness": {"weight": config.smoothness_weight, "scale": 1.0},
            "terminal": {"weight": config.terminal_weight, "scale": config.position_scale**2},
            "altitude": {
                "weight": config.altitude_weight,
                "scale": 1.0,
                "margin_m": config.altitude_margin,
                "softness_m": config.altitude_softness,
            },
        },
        "loss_v2_weight": 0.0,
        "rollout_seconds": 6.0,
        "score_seconds": 2.0,
    }


def parameter_contract() -> dict[str, Any]:
    """Return the exact two-leaf positive-log parameterization."""
    return {
        name: {
            "indices": list(spec.indices),
            "contract_default": spec.contract_default,
            "runtime_default": spec.runtime_default,
            "physical_bounds": [spec.physical_lower, spec.physical_upper],
            "raw_bounds": [spec.raw_lower, spec.raw_upper],
            "theta_start": 0.0,
            "coordinate": "p_default*exp(theta)",
        }
        for name, spec in PARAMETER_SPECS.items()
    }


def _require_exact_config(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "source_base_commit",
        "public_day26_sha256",
        "parameters",
        "loss_v1",
        "parents",
        "optimizer",
        "resources",
        "checkpoint",
        "forwardability",
        "protection",
        "runtime",
        "claim_boundary",
    }
    if set(payload) != required:
        raise G4ContractError("G4 config top-level keys changed")
    expected_hashes = {path: value["sha256"] for path, value in DAY26_INPUTS.items()}
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["source_base_commit"] != SOURCE_BASE_COMMIT
        or payload["public_day26_sha256"] != expected_hashes
        or payload["parameters"] != parameter_contract()
        or payload["loss_v1"] != loss_v1_contract()
    ):
        raise G4ContractError("G4 config science pins changed")
    parents = payload["parents"]
    if (
        parents.get("development_seed") != DEVELOPMENT_SEED
        or parents.get("total_train_parents") != 288
        or parents.get("worlds_per_batch") != EFFECTIVE_BATCH_SIZE
        or parents.get("stratum_order")
        != [f"{motion.value}/{profile.value}" for motion, profile in STRATA]
        or parents.get("batch_stratum_counts") != [list(row) for row in BATCH_STRATUM_COUNTS]
        or parents.get("validation_identities") != list(VALIDATION_IDENTITIES)
        or parents.get("near_limit_identities") != list(NEAR_LIMIT_IDENTITIES)
    ):
        raise G4ContractError("G4 parent/split config changed")
    optimizer = payload["optimizer"]
    if optimizer != {
        "name": "projected_adam",
        "learning_rate": ADAM_LEARNING_RATE,
        "b1": ADAM_B1,
        "b2": ADAM_B2,
        "eps": ADAM_EPS,
        "eps_root": ADAM_EPS_ROOT,
        "weight_decay": 0.0,
        "schedule": None,
        "gradient_clipping": None,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatch_count": MICROBATCH_COUNT,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "maximum_updates": MAX_UPDATES,
    }:
        raise G4ContractError("G4 optimizer config changed")
    runtime = payload["runtime"]
    if (
        runtime.get("interpreter") != str(EXPECTED_INTERPRETER)
        or runtime.get("repository_root") != str(EXPECTED_REPOSITORY_ROOT)
        or runtime.get("backend") != "cpu"
        or runtime.get("jax_enable_x64") is not False
    ):
        raise G4ContractError("G4 runtime config changed")
    protection = payload["protection"]
    if protection != protected_contract():
        raise G4ContractError("G4 protection sentinels changed")


def load_and_validate_config(path: Path, repository_root: Path) -> tuple[dict[str, Any], str]:
    """Load the one authorized config path and validate every frozen value."""
    root = repository_root.resolve(strict=True)
    expected = root / CONFIG_RELATIVE_PATH
    if path.is_symlink() or path.resolve(strict=True) != expected:
        raise G4ContractError("only the exact G4 config path is authorized")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise G4ContractError("G4 config is not a top-level object")
    _require_exact_config(payload)
    return payload, sha256_file(path)


def train_parent_specs() -> tuple[G4ParentSpec, ...]:
    """Return the exact nine balanced batches and 288 unique Train coordinates."""
    result: list[G4ParentSpec] = []
    for batch_index, counts in enumerate(BATCH_STRATUM_COUNTS):
        if sum(counts) != EFFECTIVE_BATCH_SIZE:
            raise G4ContractError(f"batch {batch_index} does not contain 32 parents")
        for stratum_index, ((motion_class, profile), count) in enumerate(
            zip(STRATA, counts, strict=True)
        ):
            for slot_index in range(count):
                warmup_seconds = float(2 + (batch_index + stratum_index + slot_index) % 3)
                score_start = int(warmup_seconds * g3.CONTROL_FREQUENCY_HZ)
                episode_id = (
                    f"train-{motion_class.value}-{profile.value}-g4-"
                    f"b{batch_index:02d}-s{stratum_index:02d}-n{slot_index:02d}"
                )
                result.append(
                    G4ParentSpec(
                        batch_index=batch_index,
                        stratum_index=stratum_index,
                        slot_index=slot_index,
                        motion_class=motion_class,
                        profile=profile,
                        episode_id=episode_id,
                        pair_id=f"g4-{motion_class.value}-{profile.value}",
                        warmup_s=warmup_seconds,
                        score_start=score_start,
                        score_stop=score_start + SCORE_INTERVALS,
                    )
                )
    identities = [item.episode_id for item in result]
    coordinates = [(item.batch_index, item.stratum_index, item.slot_index) for item in result]
    if len(result) != 288 or len(set(identities)) != 288 or len(set(coordinates)) != 288:
        raise G4ContractError("Train registry is not exactly 288 unique parents")
    per_stratum = [sum(row[index] for row in BATCH_STRATUM_COUNTS) for index in range(9)]
    if per_stratum != [32] * 9:
        raise G4ContractError("Train registry is not exactly balanced 32-per-stratum")
    if any("test" in item.episode_id or "validation" in item.episode_id for item in result):
        raise G4ContractError("Train registry leaked a non-Train split")
    return tuple(result)


def parent_component_key(spec: G4ParentSpec, component: str, attempt: int) -> Array:
    """Use the frozen fold-in hierarchy (batch, stratum, slot, component, attempt)."""
    if component not in {"position", "yaw"} or not 0 <= attempt < MAX_PARENT_ATTEMPTS:
        raise ValueError("invalid G4 parent key coordinate")
    component_index = {"position": 0, "yaw": 1}[component]
    key = jax.random.key(DEVELOPMENT_SEED)
    for coordinate in (
        spec.batch_index,
        spec.stratum_index,
        spec.slot_index,
        component_index,
        attempt,
    ):
        key = jax.random.fold_in(key, coordinate)
    return key


def _g3_compatible_spec(spec: G4ParentSpec) -> g3.EpisodeSpec:
    return g3.EpisodeSpec(
        split=g3.Split.TRAIN,
        motion_class=spec.motion_class,
        profile=spec.profile,
        seed=DEVELOPMENT_SEED,
        episode_id=spec.episode_id,
        pair_id=spec.pair_id,
        warmup_s=spec.warmup_s,
        score_start=spec.score_start,
        score_stop=spec.score_stop,
    )


def _candidate_attempt(
    spec: G4ParentSpec, attempt: int
) -> tuple[Trajectory, Array, tuple[dict[str, Any], ...]]:
    compatible = _g3_compatible_spec(spec)
    time_axis = jnp.arange(g3.PARENT_INTERVALS + 1, dtype=jnp.float32) / g3.CONTROL_FREQUENCY_HZ
    envelope = robust._c3_envelope(time_axis)
    raw_fourier = robust._enveloped(
        robust._fourier_series(time_axis, parent_component_key(spec, "position", attempt), 3),
        envelope,
    )
    indices = np.arange(spec.score_start + 1, spec.score_stop + 1, dtype=np.int64)
    raw_speed = np.linalg.norm(np.asarray(raw_fourier[1])[indices], axis=-1)
    if float(np.max(raw_speed)) <= 0.0:
        raise G4ContractError("degenerate deterministic G4 translation")
    scale = (
        g3.CLASS_CONTRACTS[spec.motion_class].target_usage
        * robust.GLOBAL_LIMITS["speed_m_s"]
        / float(np.max(raw_speed))
    )
    fourier = tuple(value * jnp.asarray(scale, dtype=time_axis.dtype) for value in raw_fourier)
    profile = g3._profile_components(time_axis, spec.profile)
    z_axis = jnp.asarray([0.0, 0.0, 1.0], dtype=time_axis.dtype)
    displacement = fourier[0] + profile[0] * z_axis
    velocity = fourier[1] + profile[1] * z_axis
    acceleration = fourier[2] + profile[2] * z_axis
    jerk = fourier[3] + profile[3] * z_axis
    position = displacement + jnp.asarray([0.0, 0.0, 0.75], dtype=time_axis.dtype)
    yaw_base = robust._enveloped(
        robust._fourier_series(time_axis, parent_component_key(spec, "yaw", attempt), 1), envelope
    )
    raw_yaw_rate = np.abs(np.asarray(yaw_base[1])[:, 0][indices])
    if float(np.max(raw_yaw_rate)) <= 0.0:
        raise G4ContractError("degenerate deterministic G4 yaw")
    yaw_scale = g3.CLASS_CONTRACTS[spec.motion_class].target_usage / float(np.max(raw_yaw_rate))
    yaw = yaw_base[0][:, 0] * jnp.asarray(yaw_scale, dtype=time_axis.dtype)
    yaw_rate = yaw_base[1][:, 0] * jnp.asarray(yaw_scale, dtype=time_axis.dtype)
    trajectory = Trajectory(
        time=time_axis, pos=position, vel=velocity, acc=acceleration, yaw=yaw, yaw_rate=yaw_rate
    )
    contracts = g3._component_contracts(compatible, trajectory, jerk, fourier, profile)
    return trajectory, jerk, contracts


def construct_train_parent(spec: G4ParentSpec) -> g3.ReferenceCandidate:
    """Construct one G4 parent with controller-blind deterministic rejection."""
    compatible = _g3_compatible_spec(spec)
    attempts: list[dict[str, Any]] = []
    for attempt in range(MAX_PARENT_ATTEMPTS):
        trajectory, jerk, contracts = _candidate_attempt(spec, attempt)
        interval_indices = np.arange(spec.score_start + 1, spec.score_stop + 1, dtype=np.int64)
        score = robust.reference_statistics(trajectory, jerk, interval_indices)
        parent = robust.reference_statistics(trajectory, jerk, slice(None))
        reasons = g3._acceptance_reasons(compatible, trajectory, score, parent)
        key_digests = {
            component: robust.array_digest(
                jax.random.key_data(parent_component_key(spec, component, attempt))
            )
            for component in ("position", "yaw")
        }
        attempts.append(
            {
                "attempt_index": attempt,
                "accepted": not reasons,
                "rejection_reasons": reasons,
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
        digests = {name: robust.array_digest(value) for name, value in arrays.items()}
        parent_digest = sha256_bytes(canonical_json_bytes(digests))
        return g3.ReferenceCandidate(
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
    raise G4ContractError(f"{spec.episode_id} has no accepted attempt")


@lru_cache(maxsize=1)
def _reconstruct_train_population_cpu() -> tuple[tuple[G4ParentSpec, g3.ReferenceCandidate], ...]:
    """Materialize and freeze all 288 parents before the first optimizer update."""
    items = tuple((spec, construct_train_parent(spec)) for spec in train_parent_specs())
    parent_digests = [candidate.parent_digest for _, candidate in items]
    if len(set(parent_digests)) != 288:
        raise G4ContractError("G4 Train full-parent digest collision")
    return items


def _day26_manifest_records(repository_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in ("train_manifest.json", "validation_manifest.json"):
        manifest = _load_public_json(repository_root, name)
        for parent in manifest["parents"]:
            identity = parent["attempt_identity"]
            if identity in records:
                raise G4ContractError("Day-26 parent identity collision")
            records[identity] = parent
    if len(records) != 24:
        raise G4ContractError("Day-26 manifests do not contain exactly 24 identities")
    return records


@lru_cache(maxsize=1)
def _day26_constructed() -> tuple[tuple[g3.EpisodeSpec, g3.ReferenceCandidate], ...]:
    return tuple((spec, g3.construct_episode(spec)) for spec in g3.episode_specs())


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def load_backend_neutral_parent_payload(repository_root: Path) -> dict[str, Any]:
    """Validate and return the canonical compressed public Parent-v1 host payload."""
    root = repository_root.resolve(strict=True)
    manifest = _load_json_object(root / BACKEND_INPUT_MANIFEST_RELATIVE_PATH, "backend manifest")
    pin = manifest.get("backend_neutral_parents")
    _require_exact_keys(
        pin,
        {"bytes", "mode", "path", "payload_sha256", "schema_version", "sha256"},
        "backend-neutral parent pin",
    )
    if (
        pin["mode"] != "100644"
        or pin["path"] != BACKEND_NEUTRAL_PARENT_RELATIVE_PATH
        or pin["schema_version"] != BACKEND_NEUTRAL_PARENT_SCHEMA_VERSION
        or type(pin["bytes"]) is not int
        or pin["bytes"] <= 0
    ):
        raise G4ContractError("backend-neutral parent pin changed")
    path = root / BACKEND_NEUTRAL_PARENT_RELATIVE_PATH
    info = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o644
        or info.st_size != pin["bytes"]
        or sha256_file(path) != pin["sha256"]
    ):
        raise G4ContractError("backend-neutral parent artifact identity changed")
    try:
        raw = gzip.decompress(path.read_bytes())
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise G4ContractError("backend-neutral parent artifact is invalid") from error
    if not isinstance(payload, dict) or raw != _compact_json_bytes(payload):
        raise G4ContractError("backend-neutral parent payload is not canonical")
    _require_exact_keys(
        payload,
        {
            "day26_parent_count",
            "ordered_identity_parent_digest_sha256",
            "parents",
            "payload_sha256",
            "provenance",
            "schema_version",
            "train_parent_count",
        },
        "backend-neutral parent payload",
    )
    unsigned = dict(payload)
    claimed = unsigned.pop("payload_sha256")
    if (
        payload["schema_version"] != BACKEND_NEUTRAL_PARENT_SCHEMA_VERSION
        or payload["train_parent_count"] != 288
        or payload["day26_parent_count"] != 24
        or claimed != pin["payload_sha256"]
        or claimed != sha256_bytes(_compact_json_bytes(unsigned))
    ):
        raise G4ContractError("backend-neutral parent payload pin changed")
    parents = payload["parents"]
    if (
        not isinstance(parents, list)
        or len(parents) != 312
        or [item.get("group") for item in parents[:288]] != ["train"] * 288
        or [item.get("group") for item in parents[288:]] != ["day26"] * 24
    ):
        raise G4ContractError("backend-neutral parent inventory changed")
    identities = [item.get("identity") for item in parents]
    if len(set(identities)) != 312 or any(not isinstance(item, str) for item in identities):
        raise G4ContractError("backend-neutral parent identity changed")
    ordered = [
        [item["identity"], item.get("candidate", {}).get("parent_digest")] for item in parents
    ]
    if (
        sha256_bytes(_compact_json_bytes(ordered))
        != payload["ordered_identity_parent_digest_sha256"]
    ):
        raise G4ContractError("backend-neutral ordered parent digest changed")
    provenance = payload["provenance"]
    _require_exact_keys(
        provenance,
        {
            "base_commit",
            "base_tree",
            "cpu_reconstruction_source_blobs",
            "generation_argv",
            "gzip_argv",
            "jax_enable_x64",
            "jax_platforms",
        },
        "backend-neutral parent provenance",
    )
    expected_sources = {
        "crazyflow/control/mellinger/research/g3_freeze_v2.py": (
            "5d68c88568d37160fdee8cf2d15f614e01178b95"
        ),
        "crazyflow/control/mellinger/research/g4_optimization.py": (
            "a791908696f7c777cf19ef341558939999a2a873"
        ),
        "crazyflow/control/mellinger/research/robust_evaluation.py": (
            "76ca41d48fb8ee9cab20aa9e039b82cc35104be7"
        ),
    }
    if (
        provenance["base_commit"] != "b3663206e6faaa7395d3abdbd718d1d446061c44"
        or provenance["base_tree"] != "2095d408b62fee0423bfabb2cea7c172d986673e"
        or provenance["cpu_reconstruction_source_blobs"] != expected_sources
        or provenance["jax_platforms"] != "cpu"
        or provenance["jax_enable_x64"] is not False
        or provenance["gzip_argv"]
        != ["env", "-u", "GZIP", "LC_ALL=C", "TZ=UTC", "gzip", "-n", "-9", "-c"]
    ):
        raise G4ContractError("backend-neutral parent provenance changed")
    return payload


def _decode_backend_parent_array(record: dict[str, Any], label: str) -> np.ndarray:
    _require_exact_keys(record, {"data_base64", "dtype", "sha256", "shape"}, label)
    if record["dtype"] != "<f4" or not isinstance(record["shape"], list):
        raise G4ContractError(f"{label} dtype or shape changed")
    try:
        raw = base64.b64decode(record["data_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise G4ContractError(f"{label} base64 changed") from error
    if base64.b64encode(raw).decode("ascii") != record["data_base64"]:
        raise G4ContractError(f"{label} base64 is noncanonical")
    if sha256_bytes(raw) != record["sha256"]:
        raise G4ContractError(f"{label} hostbyte checksum changed")
    shape = tuple(record["shape"])
    if any(type(value) is not int or value < 0 for value in shape):
        raise G4ContractError(f"{label} shape changed")
    result = np.frombuffer(raw, dtype=np.dtype("<f4")).copy()
    if result.size != math.prod(shape):
        raise G4ContractError(f"{label} hostbyte size changed")
    return np.ascontiguousarray(result.reshape(shape))


def _backend_parent_spec(record: dict[str, Any]) -> G4ParentSpec | g3.EpisodeSpec:
    value = record["spec"]
    if record["group"] == "train":
        return G4ParentSpec(
            batch_index=value["batch_index"],
            stratum_index=value["stratum_index"],
            slot_index=value["slot_index"],
            motion_class=g3.MotionClass(value["motion_class"]),
            profile=g3.Profile(value["profile"]),
            episode_id=value["episode_id"],
            pair_id=value["pair_id"],
            warmup_s=value["warmup_s"],
            score_start=value["score_start"],
            score_stop=value["score_stop"],
        )
    return g3.EpisodeSpec(
        split=g3.Split(value["split"]),
        motion_class=g3.MotionClass(value["motion_class"]),
        profile=g3.Profile(value["profile"]),
        seed=value["seed"],
        episode_id=value["episode_id"],
        pair_id=value["pair_id"],
        warmup_s=value["warmup_s"],
        score_start=value["score_start"],
        score_stop=value["score_stop"],
    )


def _materialize_backend_parent(
    record: dict[str, Any], backend: str
) -> tuple[G4ParentSpec | g3.EpisodeSpec, g3.ReferenceCandidate]:
    _require_exact_keys(record, {"candidate", "group", "identity", "spec"}, "parent record")
    candidate = record["candidate"]
    _require_exact_keys(
        candidate,
        {
            "array_digests",
            "arrays",
            "attempt",
            "attempts",
            "component_contracts",
            "parent_digest",
            "parent_statistics",
            "score_statistics",
        },
        "parent candidate",
    )
    array_names = ("time", "position", "velocity", "acceleration", "yaw", "yaw_rate", "jerk")
    if set(candidate["arrays"]) != set(array_names):
        raise G4ContractError("backend-neutral parent leaf inventory changed")
    host = {
        name: _decode_backend_parent_array(candidate["arrays"][name], f"parent {name}")
        for name in array_names
    }
    expected_digests = {name: robust.array_digest(value) for name, value in host.items()}
    if expected_digests != candidate["array_digests"]:
        raise G4ContractError("backend-neutral parent array digest changed")
    if sha256_bytes(canonical_json_bytes(expected_digests)) != candidate["parent_digest"]:
        raise G4ContractError("backend-neutral full-parent digest changed")
    if backend not in {"cpu", "gpu"}:
        raise G4ContractError("unknown backend-neutral parent target")
    devices = jax.devices(backend)
    if len(devices) != 1 or devices[0].platform != backend:
        raise G4ContractError("backend-neutral parent device inventory changed")
    arrays = {name: jax.device_put(value, devices[0]) for name, value in host.items()}
    trajectory = Trajectory(
        time=arrays["time"],
        pos=arrays["position"],
        vel=arrays["velocity"],
        acc=arrays["acceleration"],
        yaw=arrays["yaw"],
        yaw_rate=arrays["yaw_rate"],
    )
    return _backend_parent_spec(record), g3.ReferenceCandidate(
        trajectory=trajectory,
        jerk=arrays["jerk"],
        attempt=candidate["attempt"],
        attempts=tuple(candidate["attempts"]),
        score_statistics=candidate["score_statistics"],
        parent_statistics=candidate["parent_statistics"],
        array_digests=candidate["array_digests"],
        component_contracts=tuple(candidate["component_contracts"]),
        parent_digest=candidate["parent_digest"],
    )


def load_backend_neutral_parent_items(
    repository_root: Path, *, backend: str = "cpu"
) -> tuple[tuple[G4ParentSpec | g3.EpisodeSpec, g3.ReferenceCandidate], ...]:
    """Materialize validated canonical Hostbytes only after selecting the backend device."""
    payload = load_backend_neutral_parent_payload(repository_root)
    return tuple(_materialize_backend_parent(record, backend) for record in payload["parents"])


@lru_cache(maxsize=2)
def _cached_backend_parent_items(
    repository_root: str, backend: str
) -> tuple[tuple[G4ParentSpec | g3.EpisodeSpec, g3.ReferenceCandidate], ...]:
    return load_backend_neutral_parent_items(Path(repository_root), backend=backend)


def _reconstruct_backend_neutral_parent_records_cpu() -> tuple[
    tuple[G4ParentSpec | g3.EpisodeSpec, g3.ReferenceCandidate], ...
]:
    return (*_reconstruct_train_population_cpu(), *_day26_constructed())


def verify_backend_neutral_parent_inverse(repository_root: Path) -> dict[str, Any]:
    """Prove all 312 Parent-v1 records equal the unchanged direct CPU reconstruction."""
    loaded = load_backend_neutral_parent_items(repository_root, backend="cpu")
    direct = _reconstruct_backend_neutral_parent_records_cpu()
    if len(loaded) != len(direct) != 312:
        raise G4ContractError("backend-neutral inverse parent count changed")
    leaf_count = 0
    for (loaded_spec, loaded_parent), (direct_spec, direct_parent) in zip(
        loaded, direct, strict=True
    ):
        if loaded_spec != direct_spec or loaded_parent.parent_digest != direct_parent.parent_digest:
            raise G4ContractError("backend-neutral inverse identity or parent digest changed")
        loaded_arrays = (
            loaded_parent.trajectory.time,
            loaded_parent.trajectory.pos,
            loaded_parent.trajectory.vel,
            loaded_parent.trajectory.acc,
            loaded_parent.trajectory.yaw,
            loaded_parent.trajectory.yaw_rate,
            loaded_parent.jerk,
        )
        direct_arrays = (
            direct_parent.trajectory.time,
            direct_parent.trajectory.pos,
            direct_parent.trajectory.vel,
            direct_parent.trajectory.acc,
            direct_parent.trajectory.yaw,
            direct_parent.trajectory.yaw_rate,
            direct_parent.jerk,
        )
        for left, right in zip(loaded_arrays, direct_arrays, strict=True):
            left_host = np.ascontiguousarray(np.asarray(left))
            right_host = np.ascontiguousarray(np.asarray(right))
            if (
                left_host.dtype != right_host.dtype
                or left_host.shape != right_host.shape
                or left_host.tobytes(order="C") != right_host.tobytes(order="C")
                or robust.array_digest(left_host) != robust.array_digest(right_host)
            ):
                raise G4ContractError("backend-neutral inverse host leaf changed")
            leaf_count += 1
    return {
        "status": "PASS_BACKEND_NEUTRAL_PARENT_INVERSE",
        "parent_count": len(loaded),
        "leaf_count": leaf_count,
        "parent_digest_count": len({candidate.parent_digest for _spec, candidate in loaded}),
    }


def materialize_train_population(
    backend: str = "cpu",
) -> tuple[tuple[G4ParentSpec, g3.ReferenceCandidate], ...]:
    """Load the exact 288 public Train Parent-v1 Hostbyte records."""
    items = _cached_backend_parent_items(str(EXPECTED_REPOSITORY_ROOT), backend)[:288]
    if any(not isinstance(spec, G4ParentSpec) for spec, _candidate in items):
        raise G4ContractError("backend-neutral Train parent type changed")
    return items  # type: ignore[return-value]


def fixed_day26_items(
    repository_root: Path, identities: Sequence[str], *, backend: str = "cpu"
) -> tuple[tuple[g3.EpisodeSpec, g3.ReferenceCandidate], ...]:
    """Return pinned public Day-26 Parent-v1 records in the requested identity order."""
    manifest = _day26_manifest_records(repository_root)
    constructed = {
        f"{spec.episode_id}/attempt-{candidate.attempt}": (spec, candidate)
        for spec, candidate in _cached_backend_parent_items(str(repository_root), backend)[288:]
    }
    result = []
    for identity in identities:
        episode_id, attempt_text = identity.rsplit("/attempt-", 1)
        if identity not in manifest or identity not in constructed:
            raise G4ContractError(f"fixed parent identity missing: {identity}")
        spec, candidate = constructed[identity]
        record = manifest[identity]
        if (
            candidate.attempt != int(attempt_text)
            or candidate.attempt != record["attempt"]
            or candidate.parent_digest != record["parent_digest"]
            or spec.motion_class.value != record["motion_class"]
            or spec.profile.value != record["profile"]
        ):
            raise G4ContractError(f"fixed parent reconstruction changed: {identity}")
        accepted_keys = candidate.attempts[-1]["component_key_digests"]
        if accepted_keys != record["component_key_digests"]:
            raise G4ContractError(f"fixed parent key digest changed: {identity}")
        result.append((spec, candidate))
    return tuple(result)


def feasible_day26_identities() -> tuple[str, ...]:
    """Return the exact 18 feasible Train/Validation reconstruction identities."""
    return tuple(
        f"{spec.episode_id}/attempt-{candidate.attempt}"
        for spec, candidate in _day26_constructed()
        if spec.motion_class is not g3.MotionClass.NEAR_LIMIT
    )


def parent_population_evidence(repository_root: Path) -> dict[str, Any]:
    """Pin attempts, keys, arrays, splits, and the fixed validation/near-limit sets."""
    population = materialize_train_population()
    validation = fixed_day26_items(repository_root, VALIDATION_IDENTITIES)
    near_limit = fixed_day26_items(repository_root, NEAR_LIMIT_IDENTITIES)
    train_parent_digests = [candidate.parent_digest for _, candidate in population]
    external_parent_digests = [candidate.parent_digest for _, candidate in validation + near_limit]
    if set(train_parent_digests) & set(external_parent_digests):
        raise G4ContractError("Train/Validation/Near-limit full-parent digest leakage")
    attempts = [
        {
            "episode_id": spec.episode_id,
            "accepted_attempt": candidate.attempt,
            "attempts": list(candidate.attempts),
        }
        for spec, candidate in population
    ]
    accepted_keys = [
        {
            "episode_id": spec.episode_id,
            "component_key_digests": candidate.attempts[-1]["component_key_digests"],
        }
        for spec, candidate in population
    ]
    arrays = [
        {"episode_id": spec.episode_id, "array_digests": candidate.array_digests}
        for spec, candidate in population
    ]
    batch_ids = [
        [spec.episode_id for spec, _ in population if spec.batch_index == batch_index]
        for batch_index in range(9)
    ]
    if [len(batch) for batch in batch_ids] != [32] * 9:
        raise G4ContractError("materialized batch sizes changed")
    return {
        "status": "PASS_288_PARENT_FREEZE",
        "development_seed": DEVELOPMENT_SEED,
        "parent_count": len(population),
        "batch_parent_ids": batch_ids,
        "parent_digest": sha256_bytes(canonical_json_bytes(train_parent_digests)),
        "attempt_digest": sha256_bytes(canonical_json_bytes(attempts)),
        "key_digest": sha256_bytes(canonical_json_bytes(accepted_keys)),
        "array_digest": sha256_bytes(canonical_json_bytes(arrays)),
        "validation_digest": sha256_bytes(canonical_json_bytes(list(VALIDATION_IDENTITIES))),
        "near_limit_digest": sha256_bytes(canonical_json_bytes(list(NEAR_LIMIT_IDENTITIES))),
        "cross_split_leakage_count": 0,
        "controller_blind_acceptance": True,
        "regeneration_after_start": False,
    }


def physical_from_theta(theta: Any, names: Sequence[str] = PARAMETER_NAMES) -> np.ndarray:
    """Map authorized Float32 raw coordinates to physical values."""
    theta_array = np.asarray(theta, dtype=np.float32)
    if theta_array.shape != (len(names),) or any(name not in PARAMETER_SPECS for name in names):
        raise G4ContractError("theta shape or parameter inventory changed")
    defaults = np.asarray(
        [PARAMETER_SPECS[name].contract_default for name in names], dtype=np.float32
    )
    return defaults * np.exp(theta_array, dtype=np.float32)


def apply_theta(data: SimData, theta: Array, names: Sequence[str] = PARAMETER_NAMES) -> SimData:
    """Replace only the requested ``kp`` axes on an immutable experiment value."""
    state = data.controls.state
    if state is None:
        raise G4ContractError("G4 requires the state controller")
    names_tuple = tuple(names)
    if len(set(names_tuple)) != len(names_tuple) or any(
        name not in PARAMETER_SPECS for name in names_tuple
    ):
        raise G4ContractError("unauthorized or duplicate optimizer leaf")
    theta_array = jnp.asarray(theta, dtype=jnp.float32)
    if theta_array.shape != (len(names_tuple),):
        raise G4ContractError("theta shape does not match optimizer leaves")
    kp = state.params["kp"]
    for index, name in enumerate(names_tuple):
        spec = PARAMETER_SPECS[name]
        physical = jnp.asarray(spec.contract_default, dtype=jnp.float32) * jnp.exp(
            theta_array[index]
        )
        kp = kp.at[jnp.asarray(spec.indices)].set(physical)
    return data.replace(
        controls=data.controls.replace(state=state.replace(params=state.params | {"kp": kp}))
    )


def parameter_isolation_record(data: SimData, name: str, theta: float = 0.05) -> dict[str, Any]:
    """Prove that one local coordinate changes exactly its declared ``kp`` indices."""
    if name not in PARAMETER_SPECS:
        raise G4ContractError("parameter is outside G4")
    baseline = apply_theta(data, jnp.zeros((1,), dtype=jnp.float32), (name,))
    changed = apply_theta(data, jnp.asarray([theta], dtype=jnp.float32), (name,))
    left = g3._leaf_records(baseline)
    right = g3._leaf_records(changed)
    differences: list[tuple[str, list[list[int]]]] = []
    for (left_path, left_value), (right_path, right_value) in zip(left, right, strict=True):
        if (
            left_path != right_path
            or left_value.shape != right_value.shape
            or left_value.dtype != right_value.dtype
        ):
            raise G4ContractError("parameter replacement changed PyTree structure")
        indices = np.argwhere(left_value != right_value).tolist()
        if indices or (left_value.shape == () and not np.array_equal(left_value, right_value)):
            differences.append((left_path, indices))
    expected = PARAMETER_SPECS[name]
    if len(differences) != 1 or "['kp']" not in differences[0][0]:
        raise G4ContractError(f"{name} parameter isolation failed")
    if sorted(index[0] for index in differences[0][1]) != list(expected.indices):
        raise G4ContractError(f"{name} changed unexpected kp indices")
    return {
        "status": "PASS_PARAMETER_ISOLATION",
        "parameter": name,
        "changed_leaf": differences[0][0],
        "changed_indices": differences[0][1],
        "theta": theta,
        "physical": float(physical_from_theta([theta], (name,))[0]),
    }


def _scored_loss_and_aux(
    initial_data: SimData, trace: Any, reference: Trajectory, score_starts: Array
) -> tuple[Array, dict[str, Array]]:
    scored_trace = jax.tree.map(
        lambda value: robust._gather_per_world(value, score_starts, SCORE_INTERVALS), trace
    )
    scored_reference = jax.tree.map(
        lambda value: robust._gather_per_world(value, score_starts, SCORE_INTERVALS), reference
    )
    per_case_loss, per_case_metrics = tracking_loss_per_case(
        scored_trace,
        scored_reference,
        hover_rotor_velocity(initial_data),
        rotor_velocity_limits(initial_data),
        TrackingLossConfig(),
    )
    terms = tracking_loss_terms_per_case(
        scored_trace, scored_reference, hover_rotor_velocity(initial_data), TrackingLossConfig()
    )
    aux = {"per_case_loss": per_case_loss}
    aux.update({f"loss_{name}": terms.weighted[name] for name in LOSS_TERM_NAMES})
    aux.update({f"metric_{name}": value for name, value in per_case_metrics.items()})
    return jnp.mean(per_case_loss), aux


@lru_cache(maxsize=None)
def _evaluation_runtime(
    world_count: int, names: tuple[str, ...], backend: str = "cpu"
) -> EvaluationRuntime:
    if world_count < 1 or any(name not in PARAMETER_SPECS for name in names):
        raise G4ContractError("invalid evaluation runtime shape or leaves")
    sim = robust.build_simulation(
        world_count, rng_seed=DEVELOPMENT_SEED + world_count, device=backend
    )
    step_fn = sim.build_step_fn()

    def objective(
        theta: Array,
        initial_data: SimData,
        commands: Array,
        reference: Trajectory,
        score_starts: Array,
    ) -> tuple[Array, dict[str, Array]]:
        data = apply_theta(initial_data, theta, names)
        _, trace = rollout_state_commands(data, commands, step_fn, STEPS_PER_CONTROL)
        return _scored_loss_and_aux(initial_data, trace, reference, score_starts)

    evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
    return EvaluationRuntime(sim=sim, evaluate=evaluate)


def _prepare_items(
    items: Sequence[tuple[Any, g3.ReferenceCandidate]], names: tuple[str, ...], backend: str = "cpu"
) -> tuple[EvaluationRuntime, Any, Array, Trajectory, Array, tuple[str, ...]]:
    inputs = g3.stack_evaluation_inputs(items)
    runtime = _evaluation_runtime(len(items), names, backend)
    initial_data = robust.initialize_inputs(runtime.sim.data, inputs)
    score_starts = jnp.asarray([spec.score_start for spec, _ in items], dtype=jnp.int32)
    parent_ids = tuple(spec.episode_id for spec, _ in items)
    return runtime, initial_data, inputs.commands, inputs.reference, score_starts, parent_ids


def evaluate_items(
    theta: Any,
    items: Sequence[tuple[Any, g3.ReferenceCandidate]],
    names: Sequence[str] = PARAMETER_NAMES,
    *,
    backend: str = "cpu",
) -> dict[str, Any]:
    """Run one finite value-and-gradient evaluation for a fixed parent group."""
    names_tuple = tuple(names)
    runtime, data, commands, reference, score_starts, parent_ids = _prepare_items(
        items, names_tuple, backend
    )
    started = time.perf_counter()
    (loss_aux, gradient) = runtime.evaluate(
        jnp.asarray(theta, dtype=jnp.float32), data, commands, reference, score_starts
    )
    loss, aux = loss_aux
    jax.block_until_ready((loss, aux, gradient))
    elapsed = time.perf_counter() - started
    loss_array = np.asarray(loss, dtype=np.float32)
    gradient_array = np.asarray(gradient, dtype=np.float32)
    aux_arrays = {name: np.asarray(value) for name, value in aux.items()}
    if (
        not np.isfinite(loss_array).all()
        or not np.isfinite(gradient_array).all()
        or any(not np.isfinite(value).all() for value in aux_arrays.values())
    ):
        raise G4ContractError("nonfinite loss, gradient, or metric")
    return {
        "parent_ids": list(parent_ids),
        "parent_count": len(parent_ids),
        "loss": float(loss_array),
        "gradient": gradient_array,
        "aux": aux_arrays,
        "elapsed_seconds": elapsed,
    }


def _tree_digest(value: Any) -> str:
    records = []
    for path, leaf in g3._leaf_records(value):
        records.append({"path": path, "array": robust.array_digest(leaf)})
    return sha256_bytes(canonical_json_bytes(records))


def default_parity(
    repository_root: Path,
    identities: Sequence[str],
    *,
    include_backend_evidence: bool = False,
    backend: str = "cpu",
) -> dict[str, Any]:
    """Require exact theta-zero controller/carry/loss/metric parity."""
    items = fixed_day26_items(repository_root, identities, backend=backend)
    inputs = g3.stack_evaluation_inputs(items)
    sim = robust.build_simulation(
        len(items), rng_seed=DEVELOPMENT_SEED + len(items), device=backend
    )
    canonical = robust.initialize_inputs(sim.data, inputs)
    theta_zero = apply_theta(canonical, jnp.zeros((2,), dtype=jnp.float32))
    g3._parameter_default_parity(canonical, "kp_xy")
    g3._parameter_default_parity(canonical, "kp_z")
    direct = g3._direct_state_controller_parity(canonical, theta_zero, inputs.commands[0])
    step_fn = sim.build_step_fn()
    rollout = jax.jit(
        lambda data, commands: rollout_state_commands(data, commands, step_fn, STEPS_PER_CONTROL)
    )
    canonical_final, canonical_trace = rollout(canonical, inputs.commands)
    theta_final, theta_trace = rollout(theta_zero, inputs.commands)
    jax.block_until_ready((canonical_final, canonical_trace, theta_final, theta_trace))
    robust.require_pytree_equal("G4 theta-zero final carry", canonical_final, theta_final)
    robust.require_pytree_equal("G4 theta-zero rollout trace", canonical_trace, theta_trace)
    starts = jnp.asarray([spec.score_start for spec, _ in items], dtype=jnp.int32)
    left_loss, left_aux = _scored_loss_and_aux(canonical, canonical_trace, inputs.reference, starts)
    right_loss, right_aux = _scored_loss_and_aux(canonical, theta_trace, inputs.reference, starts)
    jax.block_until_ready((left_loss, left_aux, right_loss, right_aux))
    robust.require_array_equal("G4 theta-zero loss", left_loss, right_loss)
    robust.require_pytree_equal("G4 theta-zero loss/metrics", left_aux, right_aux)
    result = {
        "status": "PASS_THETA_ZERO_DEFAULT_PARITY",
        "parent_ids": [spec.episode_id for spec, _ in items],
        "parent_count": len(items),
        "direct_controller": direct,
        "carry_digest": _tree_digest(canonical_final),
        "trace_digest": _tree_digest(canonical_trace),
        "loss": float(np.asarray(left_loss)),
        "loss_and_metric_digest": _tree_digest(left_aux),
        "loss_outputs": ["loss_total", *[f"loss_{name}" for name in LOSS_TERM_NAMES]],
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }
    if include_backend_evidence:
        result["_backend_evidence"] = {
            "canonical_final": canonical_final,
            "left_loss": left_loss,
            "left_aux": left_aux,
        }
    return result


def run_m2(repository_root: Path) -> dict[str, Any]:
    """Run the ordered two-parent then 18-feasible-parent parity gate."""
    focus = tuple(
        next(
            identity
            for identity in feasible_day26_identities()
            if identity.startswith(parent_id + "/")
        )
        for parent_id in g3.FOCUSED_PARENT_IDS
    )
    focused = default_parity(repository_root, focus)
    all_feasible = default_parity(repository_root, feasible_day26_identities())
    if all_feasible["parent_count"] != 18:
        raise G4ContractError("M2 requires exactly 18 feasible Day-26 parents")
    return {"status": "PASS_M2", "focused": focused, "all_feasible": all_feasible}


def initial_optimization_state(size: int) -> OptimizationState:
    """Create a zeroed one- or two-leaf Adam state at update zero."""
    if size not in {1, 2}:
        raise G4ContractError("G4 optimizer may contain only one or two leaves")
    zeros = np.zeros((size,), dtype=np.float32)
    return OptimizationState(
        theta=zeros,
        adam=AdamState(count=0, first_moment=zeros.copy(), second_moment=zeros.copy()),
        projection_count=0,
        gradient_history=(),
        selected_validation_step=0,
        selected_validation_loss=None,
    )


def projected_adam_step(
    state: OptimizationState, gradient: Any, names: Sequence[str]
) -> tuple[OptimizationState, dict[str, Any]]:
    """Apply exactly one fixed Adam update and then project raw bounds."""
    names_tuple = tuple(names)
    gradient_array = np.asarray(gradient, dtype=np.float32)
    if gradient_array.shape != state.theta.shape or gradient_array.shape != (len(names_tuple),):
        raise G4ContractError("Adam gradient/leaf shape changed")
    if not np.isfinite(gradient_array).all():
        raise G4ContractError("Adam received a nonfinite gradient")
    count = state.adam.count + 1
    first = np.asarray(
        ADAM_B1 * state.adam.first_moment + (1.0 - ADAM_B1) * gradient_array, dtype=np.float32
    )
    second = np.asarray(
        ADAM_B2 * state.adam.second_moment + (1.0 - ADAM_B2) * gradient_array**2, dtype=np.float32
    )
    first_hat = first / np.asarray(1.0 - ADAM_B1**count, dtype=np.float32)
    second_hat = second / np.asarray(1.0 - ADAM_B2**count, dtype=np.float32)
    proposal = state.theta - np.asarray(ADAM_LEARNING_RATE, dtype=np.float32) * first_hat / (
        np.sqrt(second_hat + np.asarray(ADAM_EPS_ROOT, dtype=np.float32))
        + np.asarray(ADAM_EPS, dtype=np.float32)
    )
    lower = np.asarray([PARAMETER_SPECS[name].raw_lower for name in names_tuple], dtype=np.float32)
    upper = np.asarray([PARAMETER_SPECS[name].raw_upper for name in names_tuple], dtype=np.float32)
    projected = np.clip(proposal, lower, upper).astype(np.float32)
    projection_events = int(np.count_nonzero(projected != proposal))
    updated = OptimizationState(
        theta=projected,
        adam=AdamState(count=count, first_moment=first, second_moment=second),
        projection_count=state.projection_count + projection_events,
        gradient_history=state.gradient_history
        + (tuple(float(value) for value in gradient_array),),
        selected_validation_step=state.selected_validation_step,
        selected_validation_loss=state.selected_validation_loss,
        loss_history=state.loss_history,
    )
    return updated, {
        "before": _array_record(state.theta),
        "proposal": _array_record(proposal),
        "after": _array_record(projected),
        "physical_after": _array_record(physical_from_theta(projected, names_tuple)),
        "gradient": _array_record(gradient_array),
        "first_moment": _array_record(first),
        "second_moment": _array_record(second),
        "adam_count_before": state.adam.count,
        "adam_count_after": count,
        "projection_events": projection_events,
    }


def _aggregate_evaluations(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray([item["parent_count"] for item in evaluations], dtype=np.float32)
    if int(np.sum(counts)) <= 0:
        raise G4ContractError("cannot aggregate an empty parent population")
    weights = counts / np.sum(counts)
    loss = np.sum(
        np.asarray([item["loss"] for item in evaluations], dtype=np.float32) * weights,
        dtype=np.float32,
    )
    gradient = np.sum(
        np.stack([item["gradient"] for item in evaluations]) * weights[:, None],
        axis=0,
        dtype=np.float32,
    )
    parent_ids = [parent for item in evaluations for parent in item["parent_ids"]]
    if len(parent_ids) != len(set(parent_ids)):
        raise G4ContractError("microbatch aggregation duplicated a parent")
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise G4ContractError("microbatch aggregation is nonfinite")
    return {
        "loss": float(loss),
        "gradient": gradient,
        "parent_ids": parent_ids,
        "microbatch_parent_counts": [int(value) for value in counts],
        "microbatch_weights": [float(value) for value in weights],
        "optimizer_update_count": 0,
    }


def _assert_numeric_equivalence(label: str, left: dict[str, Any], right: dict[str, Any]) -> None:
    if left["parent_ids"] != right["parent_ids"]:
        raise G4ContractError(f"{label} parent inventory/order changed")
    if not np.allclose(left["loss"], right["loss"], rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL):
        raise G4ContractError(f"{label} loss equivalence failed")
    if not np.allclose(left["gradient"], right["gradient"], rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL):
        raise G4ContractError(f"{label} gradient equivalence failed")
    initial = initial_optimization_state(len(PARAMETER_NAMES))
    left_state, left_record = projected_adam_step(initial, left["gradient"], PARAMETER_NAMES)
    right_state, right_record = projected_adam_step(initial, right["gradient"], PARAMETER_NAMES)
    for left_value, right_value in (
        (left_state.theta, right_state.theta),
        (left_state.adam.first_moment, right_state.adam.first_moment),
        (left_state.adam.second_moment, right_state.adam.second_moment),
        (physical_from_theta(left_state.theta), physical_from_theta(right_state.theta)),
    ):
        if not np.allclose(left_value, right_value, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL):
            raise G4ContractError(f"{label} one-update numeric equivalence failed")
    if left_record["adam_count_after"] != 1 or right_record["adam_count_after"] != 1:
        raise G4ContractError(f"{label} comparison Adam counter changed")


def microbatch_equivalence() -> dict[str, Any]:
    """Run direct 4-vs-4x1 and 8-vs-2x4 without a scientific update."""
    batch_zero = tuple(item for item in materialize_train_population() if item[0].batch_index == 0)
    theta = np.zeros((2,), dtype=np.float32)
    direct_four_eval = evaluate_items(theta, batch_zero[:4])
    direct_four = _aggregate_evaluations((direct_four_eval,))
    four_singles = _aggregate_evaluations(
        tuple(evaluate_items(theta, batch_zero[index : index + 1]) for index in range(4))
    )
    direct_eight_eval = evaluate_items(theta, batch_zero[:8])
    direct_eight = _aggregate_evaluations((direct_eight_eval,))
    two_fours = _aggregate_evaluations(
        (evaluate_items(theta, batch_zero[:4]), evaluate_items(theta, batch_zero[4:8]))
    )
    _assert_numeric_equivalence("4-vs-4x1", direct_four, four_singles)
    _assert_numeric_equivalence("8-vs-2x4", direct_eight, two_fours)
    return {
        "status": "PASS_MICROBATCH_EQUIVALENCE",
        "four_vs_four_by_one": {
            "direct_loss": direct_four["loss"],
            "aggregated_loss": four_singles["loss"],
            "direct_gradient": direct_four["gradient"].tolist(),
            "aggregated_gradient": four_singles["gradient"].tolist(),
            "parent_ids": direct_four["parent_ids"],
        },
        "eight_vs_two_by_four": {
            "direct_loss": direct_eight["loss"],
            "aggregated_loss": two_fours["loss"],
            "direct_gradient": direct_eight["gradient"].tolist(),
            "aggregated_gradient": two_fours["gradient"].tolist(),
            "parent_ids": direct_eight["parent_ids"],
        },
        "rtol": NUMERIC_RTOL,
        "atol": NUMERIC_ATOL,
        "scientific_optimizer_update_count": 0,
        "comparison_optimizer_update_count": 1,
    }


def memory_snapshot() -> dict[str, int]:
    """Read only the four required host memory counters."""
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(raw.strip().split()[0]) * 1024
    if set(values) != {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
        raise G4ContractError("incomplete /proc/meminfo snapshot")
    return values


def effective_32_no_update(
    names: Sequence[str] = PARAMETER_NAMES, theta: Any | None = None
) -> dict[str, Any]:
    """Run eight ordered four-parent reverse passes without applying an update."""
    names_tuple = tuple(names)
    theta_array = (
        np.zeros((len(names_tuple),), dtype=np.float32)
        if theta is None
        else np.asarray(theta, dtype=np.float32)
    )
    batch_zero = tuple(item for item in materialize_train_population() if item[0].batch_index == 0)
    evaluations = []
    memory_records = []
    started = time.perf_counter()
    for index in range(MICROBATCH_COUNT):
        memory = memory_snapshot()
        swap_used = memory["SwapTotal"] - memory["SwapFree"]
        if memory["MemAvailable"] < MINIMUM_AVAILABLE_BYTES or swap_used != 0:
            raise G4ContractError("effective-32 resource preflight failed")
        first = index * MICROBATCH_SIZE
        evaluation = evaluate_items(
            theta_array, batch_zero[first : first + MICROBATCH_SIZE], names_tuple
        )
        memory_records.append(
            {
                "microbatch_index": index,
                "mem_total_bytes": memory["MemTotal"],
                "mem_available_bytes": memory["MemAvailable"],
                "swap_used_bytes": swap_used,
                "elapsed_seconds": evaluation["elapsed_seconds"],
                "parent_ids": evaluation["parent_ids"],
            }
        )
        evaluations.append(evaluation)
    aggregation_started = time.perf_counter()
    aggregated = _aggregate_evaluations(evaluations)
    aggregation_elapsed = time.perf_counter() - aggregation_started
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    mem_total = memory_records[0]["mem_total_bytes"]
    if maximum_rss > int(RSS_FRACTION_LIMIT * mem_total):
        raise G4ContractError("effective-32 RSS exceeded 70 percent of MemTotal")
    if aggregated["parent_ids"] != [spec.episode_id for spec, _ in batch_zero]:
        raise G4ContractError("effective-32 parent inventory/order changed")
    return {
        "status": "PASS_EFFECTIVE_32_NO_UPDATE",
        "loss": aggregated["loss"],
        "gradient": aggregated["gradient"].tolist(),
        "parent_ids": aggregated["parent_ids"],
        "microbatch_parent_counts": aggregated["microbatch_parent_counts"],
        "microbatch_weights": aggregated["microbatch_weights"],
        "microbatches": memory_records,
        "maximum_rss_bytes": maximum_rss,
        "maximum_allowed_rss_bytes": int(RSS_FRACTION_LIMIT * mem_total),
        "aggregation_elapsed_seconds": aggregation_elapsed,
        "total_elapsed_seconds": time.perf_counter() - started,
        "retained_ad_trace_count": 0,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def run_m3() -> dict[str, Any]:
    """Run equivalence first and then the effective-32 no-update resource gate."""
    return {
        "status": "PASS_M3",
        "equivalence": microbatch_equivalence(),
        "effective_32": effective_32_no_update(),
    }


def _batch_items(
    batch_index: int, *, backend: str = "cpu"
) -> tuple[tuple[G4ParentSpec, g3.ReferenceCandidate], ...]:
    items = tuple(
        item for item in materialize_train_population(backend) if item[0].batch_index == batch_index
    )
    if len(items) != EFFECTIVE_BATCH_SIZE:
        raise G4ContractError(f"batch {batch_index} parent count changed")
    return items


def one_effective_batch(
    state: OptimizationState, names: Sequence[str], *, backend: str = "cpu"
) -> tuple[OptimizationState, dict[str, Any]]:
    """Aggregate all eight microbatches, then perform exactly one optimizer update."""
    names_tuple = tuple(names)
    batch_index = state.adam.count % 9
    items = _batch_items(batch_index, backend=backend)
    evaluations = []
    for index in range(MICROBATCH_COUNT):
        first = index * MICROBATCH_SIZE
        evaluations.append(
            evaluate_items(
                state.theta, items[first : first + MICROBATCH_SIZE], names_tuple, backend=backend
            )
        )
    aggregate = _aggregate_evaluations(evaluations)
    if aggregate["microbatch_parent_counts"] != [4] * 8:
        raise G4ContractError("effective batch microbatch sizes changed")
    updated, adam_record = projected_adam_step(state, aggregate["gradient"], names_tuple)
    if updated.adam.count != state.adam.count + 1:
        raise G4ContractError("exactly-one-update counter failed")
    return updated, {
        "update": updated.adam.count,
        "batch_index": batch_index,
        "parent_ids": aggregate["parent_ids"],
        "loss": aggregate["loss"],
        "gradient": aggregate["gradient"].tolist(),
        "gradient_norm": float(np.linalg.norm(aggregate["gradient"])),
        "microbatch_parent_counts": aggregate["microbatch_parent_counts"],
        "microbatch_weights": aggregate["microbatch_weights"],
        "optimizer": adam_record,
        "projection_count_total": updated.projection_count,
    }


def run_updates(
    names: Sequence[str], update_count: int, *, initial_state: OptimizationState | None = None
) -> tuple[OptimizationState, list[dict[str, Any]]]:
    """Run a bounded technical update count without validation or candidate claims."""
    names_tuple = tuple(names)
    if names_tuple not in {("kp_xy",), ("kp_z",), PARAMETER_NAMES}:
        raise G4ContractError("optimizer leaf inventory is outside G4")
    state = initial_optimization_state(len(names_tuple)) if initial_state is None else initial_state
    if update_count < 0 or state.adam.count + update_count > MAX_UPDATES:
        raise G4ContractError("update budget exceeded")
    records = []
    for _ in range(update_count):
        state, record = one_effective_batch(state, names_tuple)
        records.append(record)
    return state, records


def run_single_smokes() -> dict[str, Any]:
    """Run one isolated technical update for each authorized leaf."""
    resource_gate = effective_32_no_update()
    records: dict[str, Any] = {}
    for name in PARAMETER_NAMES:
        state, updates = run_updates((name,), 1)
        update = updates[0]
        gradient = np.asarray(update["gradient"], dtype=np.float32)
        before = _restore_array(update["optimizer"]["before"], label="single before")
        proposal = _restore_array(update["optimizer"]["proposal"], label="single proposal")
        if np.linalg.norm(gradient) <= MINIMUM_SIGNIFICANT_GRADIENT:
            raise G4ContractError(f"{name} single-smoke gradient is not significant")
        if not np.all((proposal - before) * gradient < 0.0):
            raise G4ContractError(f"{name} Adam proposal is not opposite the gradient")
        records[name] = {
            "status": "PASS_ISOLATED_SINGLE_UPDATE",
            "update": update,
            "final_theta": state.theta.tolist(),
            "final_physical": physical_from_theta(state.theta, (name,)).tolist(),
            "adam_count": state.adam.count,
            "foreign_leaf_updates": 0,
        }
    return {"status": "PASS_M4", "fresh_resource_gate": resource_gate, "single_smokes": records}


def convergence_status(gradient_norms: Sequence[float], losses: Sequence[float]) -> dict[str, Any]:
    """Evaluate only the frozen plateau/gradient rules; never stop earlier than 1500."""
    count = len(losses)
    if len(gradient_norms) != count:
        raise G4ContractError("loss/gradient history length mismatch")
    gradient_pass = False
    gradient_ratio = None
    gradient_recent = None
    if count >= 100:
        initial = float(np.median(np.asarray(gradient_norms[:100], dtype=np.float64)))
        gradient_recent = float(np.median(np.asarray(gradient_norms[-100:], dtype=np.float64)))
        gradient_ratio = gradient_recent / initial if initial > 0.0 else math.inf
        gradient_pass = gradient_recent <= 0.1 * initial or gradient_recent <= 1.0e-4
    plateau_pass = False
    plateau_improvements: list[float] = []
    if count >= 1_000:
        for offset in range(3):
            stop = count - offset * 250
            recent = np.median(np.asarray(losses[stop - 250 : stop], dtype=np.float64))
            previous = np.median(np.asarray(losses[stop - 500 : stop - 250], dtype=np.float64))
            improvement = (previous - recent) / previous if previous > 0.0 else -math.inf
            plateau_improvements.append(float(improvement))
        plateau_pass = len(plateau_improvements) == 3 and all(
            value < 0.002 for value in plateau_improvements
        )
    converged = count >= 1_500 and gradient_pass and plateau_pass
    return {
        "update_count": count,
        "gradient_recent_median": gradient_recent,
        "gradient_ratio": gradient_ratio,
        "gradient_pass": gradient_pass,
        "plateau_improvements": plateau_improvements,
        "plateau_pass": plateau_pass,
        "converged": converged,
        "status": (
            "CONVERGED"
            if converged
            else "WITHHELD_NO_CONVERGENCE"
            if count >= MAX_UPDATES
            else "CONTINUE"
        ),
    }


def forwardability_gate(default: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen later G5 thresholds without selecting a candidate."""
    primary_improvement = (
        default["mean_validation_loss"] - candidate["mean_validation_loss"]
    ) / default["mean_validation_loss"]
    group_regressions = {
        name: (candidate["group_loss"][name] - value) / value
        for name, value in default["group_loss"].items()
    }
    rmse_degradation = {
        name: (candidate["rmse"][name] - value) / value for name, value in default["rmse"].items()
    }
    wrench_pass = all(
        candidate["wrench_degradation"][index]
        <= max(0.01 * robust.WRENCH_PLATFORM_SCALE[index], 0.05 * default["wrench"][index])
        for index in range(4)
    )
    checks = {
        "primary_loss": primary_improvement >= 0.05,
        "group_regression": max(group_regressions.values(), default=-math.inf) <= 0.02,
        "rmse": max(rmse_degradation.values()) <= 0.02 and min(rmse_degradation.values()) < 0.0,
        "integral_contacts": candidate["integral_contacts"] <= default["integral_contacts"],
        "integral_nearness": candidate["integral_nearness_degradation"] <= 0.02,
        "soft_nominal_clips": candidate["soft_nominal_stage_a_torque_clips"] == 0
        and candidate["soft_nominal_stage_a_motor_clips"] == 0
        and candidate["soft_nominal_additional_stage_b_clips"] == 0,
        "dynamic_clips": candidate["dynamic_additional_stage_b_clips"] == 0
        and candidate["dynamic_stage_a_motor_clip_fraction"] <= 0.0025
        and candidate["dynamic_max_clip_interval_s"] <= 0.02,
        "motor_reserve": candidate["minimum_motor_reserve"] >= 0.05
        and default["minimum_motor_reserve"] - candidate["minimum_motor_reserve"] <= 0.02,
        "wrench": wrench_pass,
        "technical": bool(candidate["technical_gates_pass"]),
        "bounds": bool(candidate["bounds_pass"]),
        "convergence": bool(candidate["convergence_pass"]),
    }
    return {
        "status": "PASS_FORWARDABILITY" if all(checks.values()) else "WITHHELD_FORWARDABILITY",
        "checks": checks,
        "primary_improvement": primary_improvement,
        "group_regressions": group_regressions,
        "rmse_degradation": rmse_degradation,
        "claim_boundary": (
            "technical threshold evaluation only; no candidate selection or improvement claim"
        ),
    }


def validate_runtime_environment(repository_root: Path) -> dict[str, Any]:
    """Require the exact interpreter, CWD, PYTHONPATH, CPU backend, and Float32 mode."""
    root = repository_root.resolve(strict=True)
    executable = Path(sys.executable)
    if executable != EXPECTED_INTERPRETER or Path.cwd().resolve(strict=True) != root:
        raise G4ContractError("interpreter path or CWD differs from the G4 contract")
    if os.environ.get("PYTHONPATH") != str(root):
        raise G4ContractError("PYTHONPATH is not exactly the G4 worktree")
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or os.environ.get("PYTHONNOUSERSITE") != "1"
    ):
        raise G4ContractError("Python cache/user-site environment changed")
    if jax.config.x64_enabled:
        raise G4ContractError("JAX_ENABLE_X64 must be false")
    if jax.default_backend() != "cpu":
        raise G4ContractError("G4 requires the CPU backend")
    target = executable.resolve(strict=True)
    return {
        "interpreter_path": str(executable),
        "interpreter_target": str(target),
        "interpreter_sha256": sha256_file(target),
        "cwd": str(root),
        "pythonpath": os.environ["PYTHONPATH"],
        "python_dont_write_bytecode": True,
        "python_no_user_site": True,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": jax.config.x64_enabled,
        "python_version": sys.version,
        "jax_version": importlib.metadata.version("jax"),
        "numpy_version": importlib.metadata.version("numpy"),
    }


def source_origin_payload(repository_root: Path) -> dict[str, Any]:
    """Pin runtime and every G4 source/test origin to this worktree."""
    root = repository_root.resolve(strict=True)
    runtime = validate_runtime_environment(root)
    crazyflow_path = Path(crazyflow.__file__).resolve(strict=True)
    module_path = Path(__file__).resolve(strict=True)
    expected_origins = {
        "crazyflow": crazyflow_path,
        "g4_module": module_path,
        "g4_cli": root / WRITE_PATHS[1],
        "g4_unit_tests": root / WRITE_PATHS[3],
        "g4_integration_tests": root / WRITE_PATHS[4],
    }
    records = {}
    for name, path in expected_origins.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or "site-packages" in resolved.parts:
            raise G4ContractError(f"{name} origin is outside the G4 worktree")
        records[name] = {"file": str(resolved), "sha256": sha256_file(resolved)}
    payload = {
        "schema_version": SOURCE_ORIGIN_SCHEMA_VERSION,
        "repository_root": str(root),
        "runtime": runtime,
        "origins": records,
    }
    payload["record_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    if path.exists() or path.is_symlink():
        raise G4ContractError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_source_origin_record(repository_root: Path, path: Path) -> dict[str, Any]:
    """Write one overwrite-refusing mode-0600 source-origin record."""
    payload = source_origin_payload(repository_root)
    _atomic_write(path, canonical_json_bytes(payload), mode=0o600)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise G4ContractError("source-origin record mode is not 0600")
    return payload


def validate_source_origin_record(repository_root: Path, path: Path) -> dict[str, Any]:
    """Reject stale, foreign, non-0600, or corrupted source-origin records."""
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise G4ContractError("source-origin record type/mode changed")
    recorded = json.loads(path.read_bytes())
    if not isinstance(recorded, dict) or "record_sha256" not in recorded:
        raise G4ContractError("invalid source-origin record")
    claimed = recorded.pop("record_sha256")
    if claimed != sha256_bytes(canonical_json_bytes(recorded)):
        raise G4ContractError("source-origin record checksum mismatch")
    recorded["record_sha256"] = claimed
    current = source_origin_payload(repository_root)
    if recorded != current:
        raise G4ContractError("source-origin record is stale or foreign")
    return current


def source_fingerprint(repository_root: Path) -> dict[str, str]:
    """Hash exactly the five authorized product paths."""
    root = repository_root.resolve(strict=True)
    return {path: sha256_file(root / path) for path in WRITE_PATHS}


def science_fingerprints(
    repository_root: Path, config_sha256: str, parent_evidence: dict[str, Any]
) -> dict[str, Any]:
    """Build the complete source/runtime/config/science fingerprint set."""
    public = {path: value["sha256"] for path, value in DAY26_INPUTS.items()}
    runtime = validate_runtime_environment(repository_root)
    return {
        "source_base_commit": SOURCE_BASE_COMMIT,
        "sources": source_fingerprint(repository_root),
        "runtime": runtime,
        "config_sha256": config_sha256,
        "loss_sha256": sha256_bytes(canonical_json_bytes(loss_v1_contract())),
        "parameter_sha256": sha256_bytes(canonical_json_bytes(parameter_contract())),
        "public_day26_sha256": public,
        "train_parent_sha256": parent_evidence["parent_digest"],
        "train_attempt_sha256": parent_evidence["attempt_digest"],
        "train_key_sha256": parent_evidence["key_digest"],
        "train_array_sha256": parent_evidence["array_digest"],
        "validation_sha256": parent_evidence["validation_digest"],
        "near_limit_sha256": parent_evidence["near_limit_digest"],
    }


def checkpoint_payload(
    state: OptimizationState,
    names: Sequence[str],
    fingerprints: dict[str, Any],
    *,
    parent_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    """Build a self-checking checkpoint payload with complete resume state."""
    names_tuple = tuple(names)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "update_count": state.adam.count,
        "next_batch_index": state.adam.count % 9,
        "parameter_names": list(names_tuple),
        "theta": _array_record(state.theta),
        "physical": _array_record(physical_from_theta(state.theta, names_tuple)),
        "adam": {
            "count": state.adam.count,
            "first_moment": _array_record(state.adam.first_moment),
            "second_moment": _array_record(state.adam.second_moment),
            "learning_rate": ADAM_LEARNING_RATE,
            "b1": ADAM_B1,
            "b2": ADAM_B2,
            "eps": ADAM_EPS,
            "eps_root": ADAM_EPS_ROOT,
        },
        "rng_coordinates": {
            "development_seed": DEVELOPMENT_SEED,
            "next_batch_index": state.adam.count % 9,
            "parents_regenerated": False,
        },
        "fingerprints": fingerprints,
        "selection_state": {
            "selected_validation_step": state.selected_validation_step,
            "selected_validation_loss": state.selected_validation_loss,
            "validation_updates": 0,
        },
        "plateau_window": [],
        "gradient_window": [list(item) for item in state.gradient_history[-100:]],
        "projection_count": state.projection_count,
        "metric_offsets": {"train_updates": state.adam.count, "validation_evaluations": 0},
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def save_checkpoint(
    path: Path,
    state: OptimizationState,
    names: Sequence[str],
    fingerprints: dict[str, Any],
    *,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically save one new checkpoint and refuse overwrite."""
    payload = checkpoint_payload(
        state, names, fingerprints, parent_checkpoint_sha256=parent_checkpoint_sha256
    )
    _atomic_write(path, canonical_json_bytes(payload), mode=0o600)
    return payload


def load_checkpoint(
    path: Path, names: Sequence[str], expected_fingerprints: dict[str, Any]
) -> tuple[OptimizationState, dict[str, Any]]:
    """Validate and restore a checkpoint against all expected fingerprints."""
    if path.is_symlink() or not path.is_file():
        raise G4ContractError("checkpoint is not a regular file")
    payload = json.loads(path.read_bytes())
    claimed = payload.pop("payload_sha256", None)
    if claimed != sha256_bytes(canonical_json_bytes(payload)):
        raise G4ContractError("checkpoint payload checksum mismatch")
    payload["payload_sha256"] = claimed
    names_tuple = tuple(names)
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("parameter_names") != list(names_tuple)
        or payload.get("fingerprints") != expected_fingerprints
    ):
        raise G4ContractError("checkpoint schema/parameter/source/runtime fingerprint changed")
    theta = _restore_array(payload["theta"], label="theta")
    first = _restore_array(payload["adam"]["first_moment"], label="first moment")
    second = _restore_array(payload["adam"]["second_moment"], label="second moment")
    physical = _restore_array(payload["physical"], label="physical")
    count = payload["update_count"]
    if (
        theta.dtype != np.dtype("<f4")
        or theta.shape != (len(names_tuple),)
        or first.shape != theta.shape
        or second.shape != theta.shape
        or count != payload["adam"]["count"]
        or count % 9 != payload["next_batch_index"]
        or not np.array_equal(physical, physical_from_theta(theta, names_tuple))
        or payload["rng_coordinates"]["parents_regenerated"] is not False
    ):
        raise G4ContractError("checkpoint dtype/shape/counter/RNG state changed")
    state = OptimizationState(
        theta=theta,
        adam=AdamState(count=count, first_moment=first, second_moment=second),
        projection_count=payload["projection_count"],
        gradient_history=tuple(tuple(row) for row in payload["gradient_window"]),
        selected_validation_step=payload["selection_state"]["selected_validation_step"],
        selected_validation_loss=payload["selection_state"]["selected_validation_loss"],
    )
    return state, payload


def longrun_contract_payload(fingerprints: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable identity and budget contract for one bounded longrun."""
    payload = {
        "schema_version": LONGRUN_CONTRACT_SCHEMA_VERSION,
        "parameter_names": list(PARAMETER_NAMES),
        "maximum_update_count": LONGRUN_MAX_UPDATES,
        "additional_updates_per_process": [1, LONGRUN_MAX_UPDATES],
        "checkpoint_every_updates": 1,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatch_count": MICROBATCH_COUNT,
        "fingerprints": fingerprints,
        "claim_boundary": (
            "bounded technical simulation evidence; no candidate or improvement claim"
        ),
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def longrun_scientific_payload_sha256(payload: dict[str, Any]) -> str:
    """Hash all checkpoint science while excluding fluctuating operational provenance."""
    scientific = dict(payload)
    for key in (
        "payload_sha256",
        "scientific_payload_sha256",
        "operational_update",
        "parent_checkpoint_payload_sha256",
    ):
        scientific.pop(key, None)
    return sha256_bytes(canonical_json_bytes(scientific))


def _validate_longrun_resource_record(record: Any, *, label: str) -> dict[str, int]:
    required = {
        "mem_total_bytes",
        "mem_available_bytes",
        "swap_used_bytes",
        "maximum_rss_bytes",
        "maximum_allowed_rss_bytes",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise G4ContractError(f"longrun {label} resource keys changed")
    if any(
        isinstance(record[name], bool) or not isinstance(record[name], int) for name in required
    ):
        raise G4ContractError(f"longrun {label} resource types changed")
    if record["mem_total_bytes"] <= 0:
        raise G4ContractError(f"longrun {label} MemTotal is not positive")
    if record["mem_available_bytes"] < MINIMUM_AVAILABLE_BYTES:
        raise G4ContractError(f"longrun {label} MemAvailable is below 8589934592 bytes")
    if record["swap_used_bytes"] != 0:
        raise G4ContractError(f"longrun {label} Swap use is nonzero")
    expected_allowed = int(RSS_FRACTION_LIMIT * record["mem_total_bytes"])
    if (
        record["maximum_rss_bytes"] < 0
        or record["maximum_allowed_rss_bytes"] != expected_allowed
        or record["maximum_rss_bytes"] > expected_allowed
    ):
        raise G4ContractError(f"longrun {label} RSS exceeded 70 percent of MemTotal")
    return record


def _validate_longrun_operational_update(
    record: Any, *, scientific_update: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "started_unix_ns",
        "finished_unix_ns",
        "process_elapsed_seconds",
        "resource_observation_count",
        "maximum_rss_bytes",
        "minimum_mem_available_bytes",
        "maximum_swap_used_bytes",
        "microbatches",
        "host",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise G4ContractError("longrun operational resource keys changed")
    if record["schema_version"] != LONGRUN_RESOURCE_EVIDENCE_SCHEMA_VERSION:
        raise G4ContractError("longrun operational resource schema changed")
    for name in (
        "started_unix_ns",
        "finished_unix_ns",
        "resource_observation_count",
        "maximum_rss_bytes",
        "minimum_mem_available_bytes",
        "maximum_swap_used_bytes",
    ):
        if isinstance(record[name], bool) or not isinstance(record[name], int):
            raise G4ContractError("longrun operational resource types changed")
    elapsed = record["process_elapsed_seconds"]
    if (
        not isinstance(elapsed, float)
        or not np.isfinite(elapsed)
        or elapsed < 0.0
        or record["started_unix_ns"] < 0
        or record["finished_unix_ns"] < record["started_unix_ns"]
        or record["resource_observation_count"] != 2 * MICROBATCH_COUNT
    ):
        raise G4ContractError("longrun operational timing or observation count changed")
    parent_ids = scientific_update.get("parent_ids")
    if (
        not isinstance(parent_ids, list)
        or len(parent_ids) != EFFECTIVE_BATCH_SIZE
        or any(not isinstance(parent, str) or not parent for parent in parent_ids)
    ):
        raise G4ContractError("longrun operational parent contract changed")
    microbatches = record["microbatches"]
    if not isinstance(microbatches, list) or len(microbatches) != MICROBATCH_COUNT:
        raise G4ContractError("longrun operational microbatch count changed")
    microbatch_keys = {
        "microbatch_index",
        "elapsed_seconds",
        "parent_ids",
        "resource_before",
        "resource_after",
    }
    observations = []
    for index, microbatch in enumerate(microbatches):
        if not isinstance(microbatch, dict) or set(microbatch) != microbatch_keys:
            raise G4ContractError("longrun operational microbatch keys changed")
        microbatch_elapsed = microbatch["elapsed_seconds"]
        expected_parents = parent_ids[index * MICROBATCH_SIZE : (index + 1) * MICROBATCH_SIZE]
        if (
            isinstance(microbatch["microbatch_index"], bool)
            or not isinstance(microbatch["microbatch_index"], int)
            or microbatch["microbatch_index"] != index
            or not isinstance(microbatch_elapsed, float)
            or not np.isfinite(microbatch_elapsed)
            or microbatch_elapsed < 0.0
            or not isinstance(microbatch["parent_ids"], list)
            or microbatch["parent_ids"] != expected_parents
        ):
            raise G4ContractError("longrun operational microbatch contract changed")
        before = microbatch["resource_before"]
        after = microbatch["resource_after"]
        if before is after:
            raise G4ContractError("longrun before/after resource records were reused")
        observations.extend(
            (
                _validate_longrun_resource_record(before, label="before"),
                _validate_longrun_resource_record(after, label="after"),
            )
        )
    if (
        len(observations) != 2 * MICROBATCH_COUNT
        or record["maximum_rss_bytes"] != max(item["maximum_rss_bytes"] for item in observations)
        or record["minimum_mem_available_bytes"]
        != min(item["mem_available_bytes"] for item in observations)
        or record["maximum_swap_used_bytes"]
        != max(item["swap_used_bytes"] for item in observations)
    ):
        raise G4ContractError("longrun operational resource extrema changed")
    host = record["host"]
    host_keys = {"hostname", "interpreter", "interpreter_target", "interpreter_sha256"}
    target = EXPECTED_INTERPRETER.resolve(strict=True)
    if (
        not isinstance(host, dict)
        or set(host) != host_keys
        or any(not isinstance(host[name], str) for name in host_keys)
        or not host["hostname"]
        or host["hostname"] != os.uname().nodename
        or host["interpreter"] != str(EXPECTED_INTERPRETER)
        or host["interpreter_target"] != str(target)
        or host["interpreter_sha256"] != sha256_file(target)
    ):
        raise G4ContractError("longrun operational host contract changed")
    return record


def _longrun_checkpoint_payload(
    state: OptimizationState,
    fingerprints: dict[str, Any],
    *,
    run_contract_sha256: str,
    parent_checkpoint_payload_sha256: str | None,
    parent_checkpoint_scientific_sha256: str | None,
    update_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    scientific_update = None if update_evidence is None else update_evidence.get("scientific")
    operational_update = None if update_evidence is None else update_evidence.get("operational")
    if update_evidence is not None and set(update_evidence) != {"scientific", "operational"}:
        raise G4ContractError("longrun update evidence keys changed")
    if state.adam.count == 0:
        if scientific_update is not None:
            raise G4ContractError("step-zero checkpoint cannot contain a scientific update")
    elif (
        not isinstance(scientific_update, dict)
        or scientific_update.get("update") != state.adam.count
    ):
        raise G4ContractError("longrun scientific update counter changed")
    if state.adam.count > 0:
        _validate_longrun_operational_update(
            operational_update, scientific_update=scientific_update
        )

    losses = state.loss_history
    if scientific_update is not None and len(losses) < state.adam.count:
        losses = losses + (float(scientific_update["loss"]),)
    plateau_window = list(losses[-500:])
    payload = {
        "schema_version": LONGRUN_CHECKPOINT_SCHEMA_VERSION,
        "update_count": state.adam.count,
        "next_batch_index": state.adam.count % len(BATCH_STRATUM_COUNTS),
        "parameter_names": list(PARAMETER_NAMES),
        "theta": _array_record(state.theta),
        "physical": _array_record(physical_from_theta(state.theta)),
        "adam": {
            "count": state.adam.count,
            "first_moment": _array_record(state.adam.first_moment),
            "second_moment": _array_record(state.adam.second_moment),
            "learning_rate": ADAM_LEARNING_RATE,
            "b1": ADAM_B1,
            "b2": ADAM_B2,
            "eps": ADAM_EPS,
            "eps_root": ADAM_EPS_ROOT,
        },
        "rng_coordinates": {
            "development_seed": DEVELOPMENT_SEED,
            "completed_batch_index": None
            if state.adam.count == 0
            else (state.adam.count - 1) % len(BATCH_STRATUM_COUNTS),
            "next_batch_index": state.adam.count % len(BATCH_STRATUM_COUNTS),
            "parents_regenerated": False,
        },
        "fingerprints": fingerprints,
        "selection_state": {
            "selected_validation_step": state.selected_validation_step,
            "selected_validation_loss": state.selected_validation_loss,
            "validation_updates": 0,
        },
        "loss_window": plateau_window,
        "plateau_window": plateau_window,
        "gradient_window": [list(item) for item in state.gradient_history[-100:]],
        "projection_count": state.projection_count,
        "metric_offsets": {
            "loss_window_start": state.adam.count - len(plateau_window),
            "plateau_window_start": state.adam.count - len(plateau_window),
            "gradient_window_start": state.adam.count - len(state.gradient_history[-100:]),
            "train_updates": state.adam.count,
            "validation_evaluations": 0,
        },
        "run_contract_sha256": run_contract_sha256,
        "parent_checkpoint_payload_sha256": parent_checkpoint_payload_sha256,
        "parent_checkpoint_scientific_sha256": parent_checkpoint_scientific_sha256,
        "scientific_update": scientific_update,
        "operational_update": operational_update,
    }
    payload["scientific_payload_sha256"] = longrun_scientific_payload_sha256(payload)
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def save_longrun_checkpoint(
    path: Path,
    state: OptimizationState,
    fingerprints: dict[str, Any],
    *,
    run_contract_sha256: str,
    parent_checkpoint_payload_sha256: str | None,
    parent_checkpoint_scientific_sha256: str | None,
    update_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Atomically persist one overwrite-refusing longrun checkpoint."""
    expected_name = f"checkpoint-step-{state.adam.count:06d}.json"
    if path.name != expected_name:
        raise G4ContractError("longrun checkpoint filename/count changed")
    payload = _longrun_checkpoint_payload(
        state,
        fingerprints,
        run_contract_sha256=run_contract_sha256,
        parent_checkpoint_payload_sha256=parent_checkpoint_payload_sha256,
        parent_checkpoint_scientific_sha256=parent_checkpoint_scientific_sha256,
        update_evidence=update_evidence,
    )
    _atomic_write(path, canonical_json_bytes(payload), mode=0o600)
    return payload


def load_longrun_checkpoint(
    path: Path,
    expected_fingerprints: dict[str, Any],
    *,
    run_contract_sha256: str,
    expected_parent_payload_sha256: str | None,
    expected_parent_scientific_sha256: str | None,
) -> tuple[OptimizationState, dict[str, Any]]:
    """Validate and restore one complete longrun checkpoint and its direct parent link."""
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or path.stat().st_uid != os.getuid()
    ):
        raise G4ContractError("longrun checkpoint type, mode, or owner changed")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise G4ContractError("longrun checkpoint is not a JSON object")
    claimed_payload = payload.pop("payload_sha256", None)
    if claimed_payload != sha256_bytes(canonical_json_bytes(payload)):
        raise G4ContractError("longrun checkpoint payload checksum mismatch")
    payload["payload_sha256"] = claimed_payload
    claimed_scientific = payload.get("scientific_payload_sha256")
    if claimed_scientific != longrun_scientific_payload_sha256(payload):
        raise G4ContractError("longrun checkpoint scientific checksum mismatch")
    if (
        payload.get("schema_version") != LONGRUN_CHECKPOINT_SCHEMA_VERSION
        or payload.get("parameter_names") != list(PARAMETER_NAMES)
        or payload.get("fingerprints") != expected_fingerprints
        or payload.get("run_contract_sha256") != run_contract_sha256
    ):
        raise G4ContractError("longrun checkpoint schema or fingerprint changed")
    if (
        payload.get("parent_checkpoint_payload_sha256") != expected_parent_payload_sha256
        or payload.get("parent_checkpoint_scientific_sha256") != expected_parent_scientific_sha256
    ):
        raise G4ContractError("longrun checkpoint parent digest changed")

    theta = _restore_array(payload["theta"], label="longrun theta")
    first = _restore_array(payload["adam"]["first_moment"], label="longrun first moment")
    second = _restore_array(payload["adam"]["second_moment"], label="longrun second moment")
    physical = _restore_array(payload["physical"], label="longrun physical")
    count = payload.get("update_count")
    expected_name = f"checkpoint-step-{count:06d}.json" if isinstance(count, int) else ""
    if (
        theta.dtype != np.dtype("<f4")
        or theta.shape != (len(PARAMETER_NAMES),)
        or first.dtype != np.dtype("<f4")
        or second.dtype != np.dtype("<f4")
        or first.shape != theta.shape
        or second.shape != theta.shape
        or not isinstance(count, int)
        or not 0 <= count <= LONGRUN_MAX_UPDATES
        or payload["adam"].get("count") != count
        or payload.get("next_batch_index") != count % len(BATCH_STRATUM_COUNTS)
        or payload["rng_coordinates"].get("next_batch_index") != count % len(BATCH_STRATUM_COUNTS)
        or payload["rng_coordinates"].get("parents_regenerated") is not False
        or not np.array_equal(physical, physical_from_theta(theta))
        or path.name != expected_name
    ):
        raise G4ContractError("longrun checkpoint dtype, shape, counter, or RNG state changed")
    scientific_update = payload.get("scientific_update")
    if count == 0:
        if scientific_update is not None:
            raise G4ContractError("longrun step-zero update evidence changed")
    elif not isinstance(scientific_update, dict) or scientific_update.get("update") != count:
        raise G4ContractError("longrun checkpoint update evidence counter changed")
    if count > 0:
        _validate_longrun_operational_update(
            payload.get("operational_update"), scientific_update=scientific_update
        )
    gradients = tuple(tuple(float(value) for value in row) for row in payload["gradient_window"])
    if any(len(row) != len(PARAMETER_NAMES) for row in gradients):
        raise G4ContractError("longrun raw-gradient window shape changed")
    state = OptimizationState(
        theta=theta,
        adam=AdamState(count=count, first_moment=first, second_moment=second),
        projection_count=payload["projection_count"],
        gradient_history=gradients,
        selected_validation_step=payload["selection_state"]["selected_validation_step"],
        selected_validation_loss=payload["selection_state"]["selected_validation_loss"],
        loss_history=tuple(float(value) for value in payload["loss_window"]),
    )
    return state, payload


def _require_owned_longrun_directory(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_dir()
        or stat.S_IMODE(path.stat().st_mode) != 0o700
        or path.stat().st_uid != os.getuid()
    ):
        raise G4ContractError(f"longrun {label} mode or owned-directory contract changed")


def _read_longrun_contract(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or path.stat().st_uid != os.getuid()
    ):
        raise G4ContractError("longrun contract type, mode, or owner changed")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise G4ContractError("longrun contract is not a JSON object")
    claimed = payload.pop("payload_sha256", None)
    if claimed != sha256_bytes(canonical_json_bytes(payload)):
        raise G4ContractError("longrun contract checksum mismatch")
    payload["payload_sha256"] = claimed
    return payload


def initialize_longrun_output(
    output_dir: Path, fingerprints: dict[str, Any]
) -> tuple[dict[str, Any], OptimizationState, dict[str, Any]]:
    """Create the exact fresh owned layout, immutable contract, and step-zero checkpoint."""
    output = validate_task_output_path(output_dir)
    output.mkdir(mode=0o700)
    (output / "checkpoints").mkdir(mode=0o700)
    (output / "segments").mkdir(mode=0o700)
    contract = longrun_contract_payload(fingerprints)
    _atomic_write(output / "run-contract.json", canonical_json_bytes(contract), mode=0o600)
    state = initial_optimization_state(len(PARAMETER_NAMES))
    initial_operational = {
        "initialized_unix_ns": time.time_ns(),
        "process_id": os.getpid(),
        "hostname": os.uname().nodename,
        "interpreter": str(Path(sys.executable)),
    }
    checkpoint = save_longrun_checkpoint(
        output / "checkpoints/checkpoint-step-000000.json",
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256=None,
        parent_checkpoint_scientific_sha256=None,
        update_evidence={"scientific": None, "operational": initial_operational},
    )
    validate_longrun_output(output, fingerprints, contract)
    return contract, state, checkpoint


def validate_longrun_output(
    output_dir: Path, fingerprints: dict[str, Any], contract: dict[str, Any]
) -> None:
    """Require exactly the owned longrun layout and immutable run contract."""
    validate_task_output_path(output_dir, must_be_absent=False)
    _require_owned_longrun_directory(output_dir, label="output")
    expected_entries = {"run-contract.json", "checkpoints", "segments"}
    actual_entries = {entry.name for entry in output_dir.iterdir()}
    if actual_entries != expected_entries:
        raise G4ContractError("unknown entry or changed longrun output layout")
    for name in ("checkpoints", "segments"):
        _require_owned_longrun_directory(output_dir / name, label=name)
    expected_contract = longrun_contract_payload(fingerprints)
    recorded_contract = _read_longrun_contract(output_dir / "run-contract.json")
    if contract != expected_contract or recorded_contract != expected_contract:
        raise G4ContractError("foreign or changed longrun contract")
    for entry in (output_dir / "checkpoints").iterdir():
        if (
            not entry.name.startswith("checkpoint-step-")
            or not entry.name.endswith(".json")
            or len(entry.name.removeprefix("checkpoint-step-").removesuffix(".json")) != 6
            or not entry.name.removeprefix("checkpoint-step-").removesuffix(".json").isdigit()
            or entry.is_symlink()
            or not entry.is_file()
            or stat.S_IMODE(entry.stat().st_mode) != 0o600
            or entry.stat().st_uid != os.getuid()
        ):
            raise G4ContractError("unknown checkpoint or changed checkpoint mode/owner")
    for entry in (output_dir / "segments").iterdir():
        middle = entry.name.removeprefix("segment-").removesuffix(".json")
        parts = middle.split("-")
        if (
            not entry.name.startswith("segment-")
            or not entry.name.endswith(".json")
            or len(parts) != 2
            or any(len(part) != 6 or not part.isdigit() for part in parts)
            or entry.is_symlink()
            or not entry.is_file()
            or stat.S_IMODE(entry.stat().st_mode) != 0o600
            or entry.stat().st_uid != os.getuid()
        ):
            raise G4ContractError("unknown segment or changed segment mode/owner")


def _checkpoint_inventory(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    inventory = []
    for entry in checkpoint_dir.iterdir():
        text = entry.name.removeprefix("checkpoint-step-").removesuffix(".json")
        if not text.isdigit() or len(text) != 6:
            raise G4ContractError("checkpoint inventory contains an unknown file")
        inventory.append((int(text), entry))
    inventory.sort(key=lambda item: item[0])
    if [count for count, _ in inventory] != list(range(len(inventory))):
        raise G4ContractError("longrun checkpoint chain is not contiguous; gap detected")
    return inventory


def resume_longrun_output(
    output_dir: Path, checkpoint_path: Path, fingerprints: dict[str, Any]
) -> tuple[OptimizationState, dict[str, Any]]:
    """Resume only from the latest checkpoint of one verified contiguous owned output."""
    contract = longrun_contract_payload(fingerprints)
    validate_longrun_output(output_dir, fingerprints, contract)
    output = output_dir.resolve(strict=True)
    if not checkpoint_path.is_absolute() or checkpoint_path.is_symlink():
        raise G4ContractError("resume checkpoint must be an absolute regular path")
    inventory = _checkpoint_inventory(output / "checkpoints")
    if not inventory:
        raise G4ContractError("longrun checkpoint chain is empty")
    latest_path = inventory[-1][1]
    if checkpoint_path.resolve(strict=True) != latest_path.resolve(strict=True):
        raise G4ContractError("resume checkpoint is not the latest verified checkpoint")
    parent_payload = None
    parent_scientific = None
    state = initial_optimization_state(len(PARAMETER_NAMES))
    payload: dict[str, Any] = {}
    for count, path in inventory:
        state, payload = load_longrun_checkpoint(
            path,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            expected_parent_payload_sha256=parent_payload,
            expected_parent_scientific_sha256=parent_scientific,
        )
        if state.adam.count != count:
            raise G4ContractError("longrun checkpoint chain counter changed")
        parent_payload = payload["payload_sha256"]
        parent_scientific = payload["scientific_payload_sha256"]
    return state, payload


def validate_longrun_resource_telemetry(
    memory: dict[str, int], maximum_rss_bytes: int
) -> dict[str, int]:
    """Enforce the inherited MemAvailable, zero-swap, and 70-percent RSS gates."""
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if set(memory) != required:
        raise G4ContractError("longrun memory telemetry keys changed")
    if memory["MemAvailable"] < MINIMUM_AVAILABLE_BYTES:
        raise G4ContractError("longrun MemAvailable is below 8589934592 bytes")
    swap_used = memory["SwapTotal"] - memory["SwapFree"]
    if swap_used != 0:
        raise G4ContractError("longrun Swap use is nonzero")
    maximum_allowed = int(RSS_FRACTION_LIMIT * memory["MemTotal"])
    if maximum_rss_bytes > maximum_allowed:
        raise G4ContractError("longrun RSS exceeded 70 percent of MemTotal")
    return {
        "mem_total_bytes": memory["MemTotal"],
        "mem_available_bytes": memory["MemAvailable"],
        "swap_used_bytes": swap_used,
        "maximum_rss_bytes": maximum_rss_bytes,
        "maximum_allowed_rss_bytes": maximum_allowed,
    }


def one_longrun_update(
    state: OptimizationState, names: Sequence[str]
) -> tuple[OptimizationState, dict[str, Any]]:
    """Execute one complete resource-gated effective-32 update with full evidence."""
    names_tuple = tuple(names)
    if names_tuple != PARAMETER_NAMES:
        raise G4ContractError("longrun requires exactly kp_xy plus kp_z")
    batch_index = state.adam.count % len(BATCH_STRATUM_COUNTS)
    items = _batch_items(batch_index)
    evaluations = []
    operational_microbatches = []
    started_unix_ns = time.time_ns()
    started = time.perf_counter()
    for index in range(MICROBATCH_COUNT):
        memory_before = memory_snapshot()
        rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        resource_before = validate_longrun_resource_telemetry(memory_before, rss_before)
        first = index * MICROBATCH_SIZE
        evaluation = evaluate_items(
            state.theta, items[first : first + MICROBATCH_SIZE], names_tuple
        )
        memory_after = memory_snapshot()
        rss_after = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        resource_after = validate_longrun_resource_telemetry(memory_after, rss_after)
        evaluations.append(evaluation)
        operational_microbatches.append(
            {
                "microbatch_index": index,
                "elapsed_seconds": evaluation["elapsed_seconds"],
                "parent_ids": evaluation["parent_ids"],
                "resource_before": resource_before,
                "resource_after": resource_after,
            }
        )
    aggregate = _aggregate_evaluations(evaluations)
    if (
        aggregate["microbatch_parent_counts"] != [MICROBATCH_SIZE] * MICROBATCH_COUNT
        or len(aggregate["parent_ids"]) != EFFECTIVE_BATCH_SIZE
    ):
        raise G4ContractError("longrun effective-batch aggregation changed")
    updated, optimizer = projected_adam_step(state, aggregate["gradient"], names_tuple)
    if updated.adam.count != state.adam.count + 1:
        raise G4ContractError("longrun exactly-one-update counter failed")
    updated = replace(updated, loss_history=state.loss_history + (aggregate["loss"],))
    loss_terms = {}
    for name in LOSS_TERM_NAMES:
        values = np.concatenate(
            [
                np.asarray(evaluation["aux"][f"loss_{name}"]).reshape(-1)
                for evaluation in evaluations
            ]
        )
        if values.size != EFFECTIVE_BATCH_SIZE or not np.isfinite(values).all():
            raise G4ContractError("longrun Loss-v1 term aggregation changed")
        loss_terms[name] = float(np.mean(values, dtype=np.float32))
    strata = [f"{spec.motion_class.value}/{spec.profile.value}" for spec, _candidate in items]
    resource_observations = [
        microbatch[phase]
        for microbatch in operational_microbatches
        for phase in ("resource_before", "resource_after")
    ]
    evidence = {
        "scientific": {
            "update": updated.adam.count,
            "batch_index": batch_index,
            "parent_ids": aggregate["parent_ids"],
            "strata": strata,
            "loss": aggregate["loss"],
            "loss_terms": loss_terms,
            "gradient": aggregate["gradient"].tolist(),
            "gradient_norm": float(np.linalg.norm(aggregate["gradient"])),
            "microbatch_parent_counts": aggregate["microbatch_parent_counts"],
            "microbatch_weights": aggregate["microbatch_weights"],
            "optimizer": optimizer,
            "projection_count_total": updated.projection_count,
        },
        "operational": {
            "schema_version": LONGRUN_RESOURCE_EVIDENCE_SCHEMA_VERSION,
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": time.time_ns(),
            "process_elapsed_seconds": time.perf_counter() - started,
            "resource_observation_count": len(resource_observations),
            "maximum_rss_bytes": max(item["maximum_rss_bytes"] for item in resource_observations),
            "minimum_mem_available_bytes": min(
                item["mem_available_bytes"] for item in resource_observations
            ),
            "maximum_swap_used_bytes": max(
                item["swap_used_bytes"] for item in resource_observations
            ),
            "microbatches": operational_microbatches,
            "host": {
                "hostname": os.uname().nodename,
                "interpreter": str(Path(sys.executable)),
                "interpreter_target": str(Path(sys.executable).resolve(strict=True)),
                "interpreter_sha256": sha256_file(Path(sys.executable).resolve(strict=True)),
            },
        },
    }
    return updated, evidence


def _segment_payload_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def run_bounded_longrun_segment(
    repository_root: Path,
    config_sha256: str,
    output_dir: Path,
    additional_updates: int,
    *,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    """Run exactly the requested additional updates and emit one completion handoff."""
    if (
        isinstance(additional_updates, bool)
        or not isinstance(additional_updates, int)
        or not 1 <= additional_updates <= LONGRUN_MAX_UPDATES
    ):
        raise G4ContractError("longrun additional update count must be in 1..100")
    parent_evidence = parent_population_evidence(repository_root)
    fingerprints = science_fingerprints(repository_root, config_sha256, parent_evidence)
    if resume_checkpoint is None:
        contract, state, start_checkpoint = initialize_longrun_output(output_dir, fingerprints)
        mode = "longrun"
    else:
        contract = longrun_contract_payload(fingerprints)
        state, start_checkpoint = resume_longrun_output(output_dir, resume_checkpoint, fingerprints)
        mode = "resume"
    start_count = state.adam.count
    end_count = start_count + additional_updates
    if end_count > LONGRUN_MAX_UPDATES:
        raise G4ContractError("longrun resume budget exceeds update 100")
    segment_started_unix_ns = time.time_ns()
    segment_started = time.perf_counter()
    previous = start_checkpoint
    new_checkpoints = []
    operational_updates = []
    for expected_count in range(start_count + 1, end_count + 1):
        state, evidence = one_longrun_update(state, PARAMETER_NAMES)
        if state.adam.count != expected_count:
            raise G4ContractError("bounded longrun update counter changed")
        checkpoint_path = output_dir / f"checkpoints/checkpoint-step-{expected_count:06d}.json"
        checkpoint = save_longrun_checkpoint(
            checkpoint_path,
            state,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            parent_checkpoint_payload_sha256=previous["payload_sha256"],
            parent_checkpoint_scientific_sha256=previous["scientific_payload_sha256"],
            update_evidence=evidence,
        )
        new_checkpoints.append(
            {
                "update_count": expected_count,
                "path": str(checkpoint_path),
                "file_sha256": sha256_file(checkpoint_path),
                "payload_sha256": checkpoint["payload_sha256"],
                "scientific_payload_sha256": checkpoint["scientific_payload_sha256"],
                "parent_checkpoint_payload_sha256": checkpoint["parent_checkpoint_payload_sha256"],
                "parent_checkpoint_scientific_sha256": checkpoint[
                    "parent_checkpoint_scientific_sha256"
                ],
            }
        )
        operational_updates.append(evidence["operational"])
        previous = checkpoint
    if state.adam.count != end_count or len(new_checkpoints) != additional_updates:
        raise G4ContractError("bounded longrun segment completion count changed")
    start_path = output_dir / f"checkpoints/checkpoint-step-{start_count:06d}.json"
    end_path = output_dir / f"checkpoints/checkpoint-step-{end_count:06d}.json"
    segment_path = output_dir / f"segments/segment-{start_count:06d}-{end_count:06d}.json"
    payload = {
        "schema_version": LONGRUN_SEGMENT_SCHEMA_VERSION,
        "mode": mode,
        "status": "PASS_BOUNDED_LONGRUN_SEGMENT",
        "claim_boundary": "bounded technical simulation evidence; no resume authorization",
        "expected_start_count": start_count,
        "actual_start_count": start_checkpoint["update_count"],
        "expected_end_count": end_count,
        "actual_end_count": state.adam.count,
        "completed_updates": additional_updates,
        "remaining_updates_to_100": LONGRUN_MAX_UPDATES - end_count,
        "start_checkpoint": str(start_path),
        "start_checkpoint_file_sha256": sha256_file(start_path),
        "start_checkpoint_payload_sha256": start_checkpoint["payload_sha256"],
        "start_checkpoint_scientific_sha256": start_checkpoint["scientific_payload_sha256"],
        "end_checkpoint": str(end_path),
        "end_checkpoint_file_sha256": sha256_file(end_path),
        "end_checkpoint_payload_sha256": previous["payload_sha256"],
        "end_checkpoint_scientific_sha256": previous["scientific_payload_sha256"],
        "run_contract_sha256": contract["payload_sha256"],
        "fingerprints": fingerprints,
        "new_checkpoints": new_checkpoints,
        "operational_summary": {
            "started_unix_ns": segment_started_unix_ns,
            "finished_unix_ns": time.time_ns(),
            "process_elapsed_seconds": time.perf_counter() - segment_started,
            "maximum_rss_bytes": max(
                (item.get("maximum_rss_bytes", 0) for item in operational_updates), default=0
            ),
            "minimum_mem_available_bytes": min(
                (
                    item.get("minimum_mem_available_bytes", MINIMUM_AVAILABLE_BYTES)
                    for item in operational_updates
                ),
                default=MINIMUM_AVAILABLE_BYTES,
            ),
            "maximum_swap_used_bytes": max(
                (item.get("maximum_swap_used_bytes", 0) for item in operational_updates), default=0
            ),
        },
    }
    payload["payload_sha256"] = _segment_payload_sha256(payload)
    _atomic_write(segment_path, canonical_json_bytes(payload), mode=0o600)
    result = dict(payload)
    result["segment_record"] = str(segment_path)
    return result


def m8_contract_payload(fingerprints: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable M8 identity, validation, and bounded-process contract."""
    payload = {
        "schema_version": M8_CONTRACT_SCHEMA_VERSION,
        "mode": "m8_longrun",
        "parameter_names": list(PARAMETER_NAMES),
        "maximum_cumulative_update_count": M8_MAX_UPDATES,
        "additional_updates_per_process": [1, LONGRUN_MAX_UPDATES],
        "checkpoint_every_updates": 1,
        "validation_updates": list(M8_VALIDATION_UPDATES),
        "selection_rule": "lowest technically valid mean validation loss; earlier wins within 1e-6",
        "convergence_rule": "unchanged convergence_status at validation updates only",
        "train_contract": {
            "parent_count": 288,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "microbatch_count": MICROBATCH_COUNT,
        },
        "validation_contract": {
            "feasible_parent_ids": list(VALIDATION_IDENTITIES),
            "near_limit_parent_ids": list(NEAR_LIMIT_IDENTITIES),
            "evaluation_batch_parent_counts": [4, 4, 1, 4, 2],
        },
        "resource_schema_version": LONGRUN_RESOURCE_EVIDENCE_SCHEMA_VERSION,
        "fingerprints": fingerprints,
        "claim_boundary": (
            "two-parameter simulation candidate evidence only; no acceptance, robustness, "
            "general Mellinger, firmware, hardware, or flight claim"
        ),
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _require_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise G4ContractError(f"nonfinite {label}")
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json(item, label=label)
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item, label=label)
        return
    raise G4ContractError(f"unsupported {label} value type")


def synthetic_m8_validation_scientific(mean_loss: float) -> dict[str, Any]:
    """Build deterministic validation-shaped evidence for bounded interface tests."""
    if not isinstance(mean_loss, float) or not math.isfinite(mean_loss):
        raise G4ContractError("synthetic M8 validation loss must be finite")
    groups = {
        name: mean_loss for name in ("soft", "nominal", "dynamic", "hold", "climb", "descent")
    }
    nearness = {
        f"{motion}/{profile}": 0.1
        for motion in ("soft", "nominal", "dynamic")
        for profile in ("hold", "climb", "descent")
    }
    return {
        "mean_validation_loss": mean_loss,
        "loss_terms": {name: mean_loss / len(LOSS_TERM_NAMES) for name in LOSS_TERM_NAMES},
        "group_loss": groups,
        "rmse": {"xy": 0.1, "z": 0.1},
        "integral_contacts": 0,
        "maximum_normalized_integral_nearness_by_stratum": nearness,
        "soft_nominal_stage_a_torque_clips": 0,
        "soft_nominal_stage_a_motor_clips": 0,
        "soft_nominal_additional_stage_b_clips": 0,
        "dynamic_additional_stage_b_clips": 0,
        "dynamic_stage_a_motor_clip_fraction": 0.0,
        "dynamic_max_clip_interval_s": 0.0,
        "minimum_motor_reserve": 0.2,
        "wrench": [0.0, 0.0, 0.0, 0.0],
        "per_parent": [
            {"parent_id": parent, "loss": mean_loss} for parent in VALIDATION_IDENTITIES
        ],
        "near_limit": [
            {"parent_id": parent, "technically_valid": True} for parent in NEAR_LIMIT_IDENTITIES
        ],
        "technical_gates_pass": True,
        "bounds_pass": True,
    }


def synthetic_m8_validation_operational(token: int = 0) -> dict[str, Any]:
    """Build deterministic Resource-v1-shaped evidence for bounded interface tests."""
    total = 16 * 1024**3
    allowed = int(RSS_FRACTION_LIMIT * total)
    identities = (
        VALIDATION_IDENTITIES[:4],
        VALIDATION_IDENTITIES[4:8],
        VALIDATION_IDENTITIES[8:],
        NEAR_LIMIT_IDENTITIES[:4],
        NEAR_LIMIT_IDENTITIES[4:],
    )
    batches = []
    observations = []
    for index, parent_ids in enumerate(identities):
        before = {
            "mem_total_bytes": total,
            "mem_available_bytes": MINIMUM_AVAILABLE_BYTES + 100 - 2 * index,
            "swap_used_bytes": 0,
            "maximum_rss_bytes": 1_000 + token + 2 * index,
            "maximum_allowed_rss_bytes": allowed,
        }
        after = {
            "mem_total_bytes": total,
            "mem_available_bytes": MINIMUM_AVAILABLE_BYTES + 99 - 2 * index,
            "swap_used_bytes": 0,
            "maximum_rss_bytes": 1_001 + token + 2 * index,
            "maximum_allowed_rss_bytes": allowed,
        }
        observations.extend((before, after))
        batches.append(
            {
                "batch_index": index,
                "split": "feasible" if index < 3 else "near_limit",
                "parent_count": len(parent_ids),
                "parent_ids": list(parent_ids),
                "elapsed_seconds": 0.1,
                "resource_before": before,
                "resource_after": after,
            }
        )
    target = EXPECTED_INTERPRETER.resolve(strict=True)
    return {
        "started_unix_ns": 1_000 + token,
        "finished_unix_ns": 2_000 + token,
        "process_elapsed_seconds": 1.0,
        "resource_observation_count": 10,
        "maximum_rss_bytes": max(item["maximum_rss_bytes"] for item in observations),
        "minimum_mem_available_bytes": min(item["mem_available_bytes"] for item in observations),
        "maximum_swap_used_bytes": max(item["swap_used_bytes"] for item in observations),
        "evaluation_batches": batches,
        "host": {
            "hostname": os.uname().nodename,
            "interpreter": str(EXPECTED_INTERPRETER),
            "interpreter_target": str(target),
            "interpreter_sha256": sha256_file(target),
        },
    }


def _validate_m8_validation_operational(record: Any) -> dict[str, Any]:
    required = {
        "started_unix_ns",
        "finished_unix_ns",
        "process_elapsed_seconds",
        "resource_observation_count",
        "maximum_rss_bytes",
        "minimum_mem_available_bytes",
        "maximum_swap_used_bytes",
        "evaluation_batches",
        "host",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise G4ContractError("M8 validation operational keys changed")
    batches = record["evaluation_batches"]
    expected_ids = (
        VALIDATION_IDENTITIES[:4],
        VALIDATION_IDENTITIES[4:8],
        VALIDATION_IDENTITIES[8:],
        NEAR_LIMIT_IDENTITIES[:4],
        NEAR_LIMIT_IDENTITIES[4:],
    )
    if not isinstance(batches, list) or len(batches) != 5:
        raise G4ContractError("M8 validation evaluation batch count changed")
    observations = []
    batch_keys = {
        "batch_index",
        "split",
        "parent_count",
        "parent_ids",
        "elapsed_seconds",
        "resource_before",
        "resource_after",
    }
    for index, (batch, identities) in enumerate(zip(batches, expected_ids, strict=True)):
        expected_split = "feasible" if index < 3 else "near_limit"
        if (
            not isinstance(batch, dict)
            or set(batch) != batch_keys
            or batch["batch_index"] != index
            or batch["split"] != expected_split
            or batch["parent_count"] != len(identities)
            or batch["parent_ids"] != list(identities)
            or not isinstance(batch["elapsed_seconds"], float)
            or not math.isfinite(batch["elapsed_seconds"])
            or batch["elapsed_seconds"] < 0.0
            or batch["resource_before"] is batch["resource_after"]
        ):
            raise G4ContractError("M8 validation evaluation batch contract changed")
        observations.extend(
            (
                _validate_longrun_resource_record(
                    batch["resource_before"], label="validation before"
                ),
                _validate_longrun_resource_record(
                    batch["resource_after"], label="validation after"
                ),
            )
        )
    if (
        record["resource_observation_count"] != 10
        or record["maximum_rss_bytes"] != max(item["maximum_rss_bytes"] for item in observations)
        or record["minimum_mem_available_bytes"]
        != min(item["mem_available_bytes"] for item in observations)
        or record["maximum_swap_used_bytes"]
        != max(item["swap_used_bytes"] for item in observations)
    ):
        raise G4ContractError("M8 validation resource extrema changed")
    _require_finite_json(record, label="M8 validation operational evidence")
    return record


def _m8_forwardability_adapter(
    default: dict[str, Any], candidate: dict[str, Any], convergence_state: dict[str, Any]
) -> dict[str, Any]:
    strata = [f"{motion.value}/{profile.value}" for motion, profile in STRATA]
    nearness = max(
        candidate["maximum_normalized_integral_nearness_by_stratum"][name]
        - default["maximum_normalized_integral_nearness_by_stratum"][name]
        for name in strata
    )
    adapted_default = {
        name: default[name]
        for name in (
            "mean_validation_loss",
            "group_loss",
            "rmse",
            "integral_contacts",
            "minimum_motor_reserve",
            "wrench",
        )
    }
    adapted_candidate = {
        name: candidate[name]
        for name in (
            "mean_validation_loss",
            "group_loss",
            "rmse",
            "integral_contacts",
            "soft_nominal_stage_a_torque_clips",
            "soft_nominal_stage_a_motor_clips",
            "soft_nominal_additional_stage_b_clips",
            "dynamic_additional_stage_b_clips",
            "dynamic_stage_a_motor_clip_fraction",
            "dynamic_max_clip_interval_s",
            "minimum_motor_reserve",
            "technical_gates_pass",
            "bounds_pass",
        )
    }
    adapted_candidate["integral_nearness_degradation"] = nearness
    adapted_candidate["wrench_degradation"] = [
        value - baseline
        for value, baseline in zip(candidate["wrench"], default["wrench"], strict=True)
    ]
    adapted_candidate["convergence_pass"] = convergence_state["converged"]
    return forwardability_gate(adapted_default, adapted_candidate)


def m8_validation_evidence_payload(
    update_count: int,
    theta: Any,
    scientific: dict[str, Any],
    operational: dict[str, Any],
    *,
    previous_validation_state: dict[str, Any] | None,
    convergence_state: dict[str, Any],
) -> dict[str, Any]:
    """Validate and bind one scheduled M8 validation and deterministic selection update."""
    if update_count not in M8_VALIDATION_UPDATES:
        raise G4ContractError("M8 validation update is outside the fixed schedule")
    theta_array = np.asarray(theta, dtype=np.float32)
    if theta_array.shape != (2,) or not np.isfinite(theta_array).all():
        raise G4ContractError("M8 validation theta changed")
    scientific_keys = {
        "mean_validation_loss",
        "loss_terms",
        "group_loss",
        "rmse",
        "integral_contacts",
        "maximum_normalized_integral_nearness_by_stratum",
        "soft_nominal_stage_a_torque_clips",
        "soft_nominal_stage_a_motor_clips",
        "soft_nominal_additional_stage_b_clips",
        "dynamic_additional_stage_b_clips",
        "dynamic_stage_a_motor_clip_fraction",
        "dynamic_max_clip_interval_s",
        "minimum_motor_reserve",
        "wrench",
        "per_parent",
        "near_limit",
        "technical_gates_pass",
        "bounds_pass",
    }
    if not isinstance(scientific, dict) or set(scientific) != scientific_keys:
        raise G4ContractError("M8 validation scientific keys changed")
    if [item.get("parent_id") for item in scientific["per_parent"]] != list(
        VALIDATION_IDENTITIES
    ) or [item.get("parent_id") for item in scientific["near_limit"]] != list(
        NEAR_LIMIT_IDENTITIES
    ):
        raise G4ContractError("M8 validation split or parent order changed")
    _require_finite_json(scientific, label="M8 validation science")
    _validate_m8_validation_operational(operational)
    convergence_keys = {
        "update_count",
        "gradient_recent_median",
        "gradient_ratio",
        "gradient_pass",
        "plateau_improvements",
        "plateau_pass",
        "converged",
        "status",
    }
    if not isinstance(convergence_state, dict) or set(convergence_state) != convergence_keys:
        raise G4ContractError("M8 convergence state keys changed")
    technically_valid = bool(scientific["technical_gates_pass"] and scientific["bounds_pass"])
    scientific_record = {
        "schema_version": M8_VALIDATION_SCHEMA_VERSION,
        "update_count": update_count,
        "parameter_names": list(PARAMETER_NAMES),
        "theta": _array_record(theta_array),
        "feasible_parent_ids": list(VALIDATION_IDENTITIES),
        "near_limit_parent_ids": list(NEAR_LIMIT_IDENTITIES),
        "scientific": scientific,
        "convergence": convergence_state,
    }
    record_scientific_sha256 = sha256_bytes(canonical_json_bytes(scientific_record))
    record_payload_sha256 = record_scientific_sha256
    history_entry = {
        "update_count": update_count,
        "mean_validation_loss": float(scientific["mean_validation_loss"]),
        "technically_valid": technically_valid,
        "payload_sha256": record_payload_sha256,
        "scientific_payload_sha256": record_scientific_sha256,
    }
    if previous_validation_state is None:
        if update_count != 0 or not technically_valid:
            raise G4ContractError("M8 checkpoint zero requires a valid default validation")
        history = [history_entry]
        selected_step = 0
        selected_loss = history_entry["mean_validation_loss"]
        selected_payload = record_payload_sha256
        selected_scientific = record_scientific_sha256
        baseline_scientific = record_scientific_sha256
        default_scientific = scientific
    else:
        required_state = {
            "schedule",
            "completed_updates",
            "history",
            "baseline_validation_scientific_sha256",
            "selected_validation_step",
            "selected_validation_loss",
            "selected_validation_payload_sha256",
            "selected_validation_scientific_sha256",
            "last_validation_update",
        }
        if (
            not isinstance(previous_validation_state, dict)
            or set(previous_validation_state) != required_state
            or previous_validation_state["schedule"] != list(M8_VALIDATION_UPDATES)
            or previous_validation_state["last_validation_update"] >= update_count
        ):
            raise G4ContractError("M8 previous validation state changed")
        history = [*previous_validation_state["history"], history_entry]
        selected_step = previous_validation_state["selected_validation_step"]
        selected_loss = previous_validation_state["selected_validation_loss"]
        selected_payload = previous_validation_state["selected_validation_payload_sha256"]
        selected_scientific = previous_validation_state["selected_validation_scientific_sha256"]
        baseline_scientific = previous_validation_state["baseline_validation_scientific_sha256"]
        default_scientific = scientific
        if technically_valid and (
            selected_loss is None or scientific["mean_validation_loss"] < selected_loss - 1.0e-6
        ):
            selected_step = update_count
            selected_loss = float(scientific["mean_validation_loss"])
            selected_payload = record_payload_sha256
            selected_scientific = record_scientific_sha256
    validation_state = {
        "schedule": list(M8_VALIDATION_UPDATES),
        "completed_updates": [item["update_count"] for item in history],
        "history": history,
        "baseline_validation_scientific_sha256": baseline_scientific,
        "selected_validation_step": selected_step,
        "selected_validation_loss": selected_loss,
        "selected_validation_payload_sha256": selected_payload,
        "selected_validation_scientific_sha256": selected_scientific,
        "last_validation_update": update_count,
    }
    selection = {
        "selected_validation_step": selected_step,
        "selected_validation_loss": selected_loss,
        "selected_validation_payload_sha256": selected_payload,
        "selected_validation_scientific_sha256": selected_scientific,
        "tie_absolute_tolerance": 1.0e-6,
    }
    forwardability = _m8_forwardability_adapter(default_scientific, scientific, convergence_state)
    payload = {
        "schema_version": M8_VALIDATION_SCHEMA_VERSION,
        "update_count": update_count,
        "parameter_names": list(PARAMETER_NAMES),
        "theta": _array_record(theta_array),
        "feasible_parent_ids": list(VALIDATION_IDENTITIES),
        "near_limit_parent_ids": list(NEAR_LIMIT_IDENTITIES),
        "scientific": scientific,
        "operational": operational,
        "selection": selection,
        "convergence": convergence_state,
        "forwardability": forwardability,
        "baseline_validation_scientific_sha256": baseline_scientific,
        "validation_state": validation_state,
    }
    scientific_projection = dict(payload)
    scientific_projection.pop("operational")
    payload["scientific_payload_sha256"] = sha256_bytes(canonical_json_bytes(scientific_projection))
    payload["operational_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": M8_VALIDATION_SCHEMA_VERSION,
                "update_count": update_count,
                "operational": operational,
            }
        )
    )
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def m8_terminal_state(
    convergence_state: dict[str, Any],
    forwardability_state: dict[str, Any],
    update_count: int,
    selected_validation_step: int,
) -> dict[str, Any]:
    """Classify only an M8 validation boundary without authorizing automatic resume."""
    terminal = False
    reason = "CONTINUE"
    if convergence_state.get("converged"):
        terminal = True
        reason = (
            "READY_FOR_M8_CANDIDATE_REVIEW"
            if forwardability_state.get("status") == "PASS_FORWARDABILITY"
            else "WITHHELD_FORWARDABILITY"
        )
    elif update_count == M8_MAX_UPDATES:
        terminal = True
        reason = "WITHHELD_NO_CONVERGENCE"
    return {
        "terminal": terminal,
        "evaluated_at_update": update_count,
        "reason": reason,
        "selected_validation_step": selected_validation_step,
        "forwardability_status": forwardability_state.get("status"),
        "automatic_resume_authorized": False,
    }


def initial_m8_state(validation_evidence: dict[str, Any]) -> M8OptimizationState:
    """Create M8 state only from the complete valid scheduled default evaluation."""
    if (
        validation_evidence.get("schema_version") != M8_VALIDATION_SCHEMA_VERSION
        or validation_evidence.get("update_count") != 0
    ):
        raise G4ContractError("M8 initial validation evidence changed")
    validation_state = validation_evidence["validation_state"]
    convergence = validation_evidence["convergence"]
    forwardability = validation_evidence["forwardability"]
    return M8OptimizationState(
        theta=np.zeros(2, dtype=np.float32),
        adam=AdamState(
            count=0,
            first_moment=np.zeros(2, dtype=np.float32),
            second_moment=np.zeros(2, dtype=np.float32),
        ),
        projection_count=0,
        loss_window=(),
        loss_window_start=0,
        initial_gradient_window=(),
        recent_gradient_window=(),
        recent_gradient_window_start=0,
        validation_state=validation_state,
        convergence_state=convergence,
        forwardability_state=forwardability,
        terminal_state=m8_terminal_state(convergence, forwardability, 0, 0),
    )


def one_m8_update_from_evidence(
    state: M8OptimizationState, evidence: dict[str, Any]
) -> tuple[M8OptimizationState, dict[str, Any]]:
    """Apply one fixed projected-Adam update and advance the exact bounded histories."""
    if not isinstance(evidence, dict) or set(evidence) != {"scientific", "operational"}:
        raise G4ContractError("M8 update evidence keys changed")
    scientific = evidence["scientific"]
    gradient = np.asarray(scientific["gradient"], dtype=np.float32)
    if gradient.shape != (2,) or not np.isfinite(gradient).all():
        raise G4ContractError("M8 raw gradient changed")
    legacy = OptimizationState(
        theta=state.theta,
        adam=state.adam,
        projection_count=state.projection_count,
        gradient_history=(),
        selected_validation_step=state.validation_state["selected_validation_step"],
        selected_validation_loss=state.validation_state["selected_validation_loss"],
    )
    updated_legacy, optimizer = projected_adam_step(legacy, gradient, PARAMETER_NAMES)
    count = updated_legacy.adam.count
    loss = float(scientific["loss"])
    norm = float(np.linalg.norm(np.asarray(gradient, dtype=np.float32)))
    if not math.isfinite(loss) or not math.isfinite(norm):
        raise G4ContractError("M8 update loss or gradient norm is nonfinite")
    scientific = dict(scientific)
    scientific.setdefault("update", count)
    scientific.setdefault("batch_index", state.adam.count % len(BATCH_STRATUM_COUNTS))
    scientific["gradient"] = gradient.tolist()
    scientific["gradient_norm"] = norm
    scientific.setdefault("optimizer", optimizer)
    scientific.setdefault("projection_count_total", updated_legacy.projection_count)
    loss_window = (*state.loss_window, loss)[-1_000:]
    initial_gradient = state.initial_gradient_window
    if len(initial_gradient) < 100:
        initial_gradient = (*initial_gradient, norm)
    recent_gradient = (*state.recent_gradient_window, norm)[-100:]
    updated = M8OptimizationState(
        theta=updated_legacy.theta,
        adam=updated_legacy.adam,
        projection_count=updated_legacy.projection_count,
        loss_window=loss_window,
        loss_window_start=max(0, count - 1_000),
        initial_gradient_window=initial_gradient,
        recent_gradient_window=recent_gradient,
        recent_gradient_window_start=max(0, count - 100),
        validation_state=state.validation_state,
        convergence_state=state.convergence_state,
        forwardability_state=state.forwardability_state,
        terminal_state=state.terminal_state,
    )
    return updated, {"scientific": scientific, "operational": evidence["operational"]}


def one_m8_update(state: M8OptimizationState) -> tuple[M8OptimizationState, dict[str, Any]]:
    """Execute one unchanged Resource-v1 effective-32 update for M8."""
    legacy = OptimizationState(
        theta=state.theta,
        adam=state.adam,
        projection_count=state.projection_count,
        gradient_history=(),
        selected_validation_step=state.validation_state["selected_validation_step"],
        selected_validation_loss=state.validation_state["selected_validation_loss"],
    )
    _updated, evidence = one_longrun_update(legacy, PARAMETER_NAMES)
    return one_m8_update_from_evidence(state, evidence)


def _m8_scientific_payload_sha256(payload: dict[str, Any]) -> str:
    scientific = dict(payload)
    for key in (
        "payload_sha256",
        "scientific_payload_sha256",
        "operational_payload_sha256",
        "operational_update",
        "parent_checkpoint_payload_sha256",
        "parent_checkpoint_operational_sha256",
        "segment_coordinates",
    ):
        scientific.pop(key, None)
    validation = scientific.get("validation_evidence")
    if isinstance(validation, dict):
        validation = dict(validation)
        for key in ("operational", "operational_payload_sha256", "payload_sha256"):
            validation.pop(key, None)
        scientific["validation_evidence"] = validation
    return sha256_bytes(canonical_json_bytes(scientific))


def _validate_m8_segment_coordinates(value: Any, update_count: int) -> dict[str, int]:
    """Require exact process-segment coordinates for one M8 checkpoint."""
    if not isinstance(value, dict) or set(value) != {"start_update", "end_update"}:
        raise G4ContractError("M8 segment coordinate keys changed")
    start = value["start_update"]
    end = value["end_update"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or start > end
        or end != update_count
    ):
        raise G4ContractError("M8 segment coordinate range or type changed")
    return value


def _m8_operational_payload_sha256(payload: dict[str, Any]) -> str:
    validation = payload.get("validation_evidence")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": M8_CHECKPOINT_SCHEMA_VERSION,
                "update_count": payload["update_count"],
                "parent_checkpoint_operational_sha256": payload[
                    "parent_checkpoint_operational_sha256"
                ],
                "segment_coordinates": payload["segment_coordinates"],
                "operational_update": payload["operational_update"],
                "validation_operational": None if validation is None else validation["operational"],
            }
        )
    )


def _m8_checkpoint_payload(
    state: M8OptimizationState,
    fingerprints: dict[str, Any],
    *,
    run_contract_sha256: str,
    parent_checkpoint_payload_sha256: str | None,
    parent_checkpoint_scientific_sha256: str | None,
    parent_checkpoint_operational_sha256: str | None,
    segment_coordinates: dict[str, int],
    update_evidence: dict[str, Any] | None,
    validation_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    count = state.adam.count
    _validate_m8_segment_coordinates(segment_coordinates, count)
    scientific_update = None if update_evidence is None else update_evidence["scientific"]
    operational_update = None if update_evidence is None else update_evidence["operational"]
    if count == 0:
        if update_evidence is not None or validation_evidence is None:
            raise G4ContractError("M8 checkpoint zero evidence changed")
    elif scientific_update is None or scientific_update.get("update") != count:
        raise G4ContractError("M8 scientific update counter changed")
    if count > 0:
        _validate_longrun_operational_update(
            operational_update, scientific_update=scientific_update
        )
    if validation_evidence is not None:
        if validation_evidence.get("update_count") != count:
            raise G4ContractError("M8 validation/checkpoint counter changed")
        validation_state = validation_evidence["validation_state"]
        convergence = validation_evidence["convergence"]
        forwardability = validation_evidence["forwardability"]
    else:
        validation_state = state.validation_state
        convergence = state.convergence_state
        forwardability = state.forwardability_state
    terminal = m8_terminal_state(
        convergence, forwardability, count, validation_state["selected_validation_step"]
    )
    history_state = {
        "loss_window": list(state.loss_window),
        "loss_window_start": state.loss_window_start,
        "initial_gradient_window": list(state.initial_gradient_window),
        "recent_gradient_window": list(state.recent_gradient_window),
        "recent_gradient_window_start": state.recent_gradient_window_start,
    }
    payload = {
        "schema_version": M8_CHECKPOINT_SCHEMA_VERSION,
        "update_count": count,
        "next_batch_index": count % len(BATCH_STRATUM_COUNTS),
        "parameter_names": list(PARAMETER_NAMES),
        "theta": _array_record(state.theta),
        "physical": _array_record(physical_from_theta(state.theta)),
        "adam": {
            "count": count,
            "first_moment": _array_record(state.adam.first_moment),
            "second_moment": _array_record(state.adam.second_moment),
            "learning_rate": ADAM_LEARNING_RATE,
            "b1": ADAM_B1,
            "b2": ADAM_B2,
            "eps": ADAM_EPS,
            "eps_root": ADAM_EPS_ROOT,
        },
        "rng_coordinates": {
            "development_seed": DEVELOPMENT_SEED,
            "completed_batch_index": None
            if count == 0
            else (count - 1) % len(BATCH_STRATUM_COUNTS),
            "next_batch_index": count % len(BATCH_STRATUM_COUNTS),
            "parents_regenerated": False,
        },
        "fingerprints": fingerprints,
        "run_contract_sha256": run_contract_sha256,
        "parent_checkpoint_payload_sha256": parent_checkpoint_payload_sha256,
        "parent_checkpoint_scientific_sha256": parent_checkpoint_scientific_sha256,
        "parent_checkpoint_operational_sha256": parent_checkpoint_operational_sha256,
        "segment_coordinates": segment_coordinates,
        "scientific_update": scientific_update,
        "operational_update": operational_update,
        "history_state": history_state,
        "validation_evidence": validation_evidence,
        "validation_state": validation_state,
        "convergence_state": convergence,
        "forwardability_state": forwardability,
        "terminal_state": terminal,
        "projection_count": state.projection_count,
        "metric_offsets": {
            "train_updates": count,
            "loss_window_start": state.loss_window_start,
            "initial_gradient_window_start": 0,
            "recent_gradient_window_start": state.recent_gradient_window_start,
            "validation_evaluations": len(validation_state["completed_updates"]),
        },
    }
    payload["scientific_payload_sha256"] = _m8_scientific_payload_sha256(payload)
    payload["operational_payload_sha256"] = _m8_operational_payload_sha256(payload)
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def save_m8_checkpoint(
    path: Path,
    state: M8OptimizationState,
    fingerprints: dict[str, Any],
    *,
    run_contract_sha256: str,
    parent_checkpoint_payload_sha256: str | None,
    parent_checkpoint_scientific_sha256: str | None,
    parent_checkpoint_operational_sha256: str | None,
    segment_coordinates: dict[str, int],
    update_evidence: dict[str, Any] | None,
    validation_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Atomically persist one overwrite-refusing M8 checkpoint."""
    if path.name != f"m8-checkpoint-step-{state.adam.count:06d}.json":
        raise G4ContractError("M8 checkpoint filename/count changed")
    payload = _m8_checkpoint_payload(
        state,
        fingerprints,
        run_contract_sha256=run_contract_sha256,
        parent_checkpoint_payload_sha256=parent_checkpoint_payload_sha256,
        parent_checkpoint_scientific_sha256=parent_checkpoint_scientific_sha256,
        parent_checkpoint_operational_sha256=parent_checkpoint_operational_sha256,
        segment_coordinates=segment_coordinates,
        update_evidence=update_evidence,
        validation_evidence=validation_evidence,
    )
    _atomic_write(path, canonical_json_bytes(payload), mode=0o600)
    return payload


def load_m8_checkpoint(
    path: Path,
    expected_fingerprints: dict[str, Any],
    *,
    run_contract_sha256: str,
    expected_parent_payload_sha256: str | None,
    expected_parent_scientific_sha256: str | None,
    expected_parent_operational_sha256: str | None,
) -> tuple[M8OptimizationState, dict[str, Any]]:
    """Validate and restore one strict M8 checkpoint and all three parent links."""
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or path.stat().st_uid != os.getuid()
    ):
        raise G4ContractError("M8 checkpoint type, mode, or owner changed")
    payload = json.loads(path.read_bytes())
    required = {
        "schema_version",
        "update_count",
        "next_batch_index",
        "parameter_names",
        "theta",
        "physical",
        "adam",
        "rng_coordinates",
        "fingerprints",
        "run_contract_sha256",
        "parent_checkpoint_payload_sha256",
        "parent_checkpoint_scientific_sha256",
        "parent_checkpoint_operational_sha256",
        "segment_coordinates",
        "scientific_update",
        "operational_update",
        "history_state",
        "validation_evidence",
        "validation_state",
        "convergence_state",
        "forwardability_state",
        "terminal_state",
        "projection_count",
        "metric_offsets",
        "scientific_payload_sha256",
        "operational_payload_sha256",
        "payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise G4ContractError("M8 checkpoint top-level keys changed")
    claimed = payload.pop("payload_sha256")
    if claimed != sha256_bytes(canonical_json_bytes(payload)):
        raise G4ContractError("M8 checkpoint payload checksum mismatch")
    payload["payload_sha256"] = claimed
    if payload["scientific_payload_sha256"] != _m8_scientific_payload_sha256(payload) or payload[
        "operational_payload_sha256"
    ] != _m8_operational_payload_sha256(payload):
        raise G4ContractError("M8 checkpoint scientific or operational checksum mismatch")
    if (
        payload["schema_version"] != M8_CHECKPOINT_SCHEMA_VERSION
        or payload["parameter_names"] != list(PARAMETER_NAMES)
        or payload["fingerprints"] != expected_fingerprints
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["parent_checkpoint_payload_sha256"] != expected_parent_payload_sha256
        or payload["parent_checkpoint_scientific_sha256"] != expected_parent_scientific_sha256
        or payload["parent_checkpoint_operational_sha256"] != expected_parent_operational_sha256
    ):
        raise G4ContractError("M8 checkpoint schema, fingerprint, or parent digest changed")
    count = payload["update_count"]
    _validate_m8_segment_coordinates(payload["segment_coordinates"], count)
    theta = _restore_array(payload["theta"], label="M8 theta")
    physical = _restore_array(payload["physical"], label="M8 physical")
    first = _restore_array(payload["adam"]["first_moment"], label="M8 first moment")
    second = _restore_array(payload["adam"]["second_moment"], label="M8 second moment")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= M8_MAX_UPDATES
        or path.name != f"m8-checkpoint-step-{count:06d}.json"
        or theta.dtype != np.dtype("<f4")
        or theta.shape != (2,)
        or first.dtype != np.dtype("<f4")
        or first.shape != (2,)
        or second.dtype != np.dtype("<f4")
        or second.shape != (2,)
        or payload["adam"]["count"] != count
        or payload["next_batch_index"] != count % len(BATCH_STRATUM_COUNTS)
        or not np.array_equal(physical, physical_from_theta(theta))
    ):
        raise G4ContractError("M8 checkpoint dtype, shape, counter, or physical state changed")
    history = payload["history_state"]
    history_keys = {
        "loss_window",
        "loss_window_start",
        "initial_gradient_window",
        "recent_gradient_window",
        "recent_gradient_window_start",
    }
    if not isinstance(history, dict) or set(history) != history_keys:
        raise G4ContractError("M8 history state keys changed")
    losses = tuple(float(value) for value in history["loss_window"])
    initial = tuple(float(value) for value in history["initial_gradient_window"])
    recent = tuple(float(value) for value in history["recent_gradient_window"])
    if (
        len(losses) != min(count, 1_000)
        or history["loss_window_start"] != max(0, count - 1_000)
        or len(initial) != min(count, 100)
        or len(recent) != min(count, 100)
        or history["recent_gradient_window_start"] != max(0, count - 100)
        or not all(math.isfinite(value) for value in (*losses, *initial, *recent))
    ):
        raise G4ContractError("M8 history length, offset, or finite contract changed")
    state = M8OptimizationState(
        theta=theta,
        adam=AdamState(count=count, first_moment=first, second_moment=second),
        projection_count=payload["projection_count"],
        loss_window=losses,
        loss_window_start=history["loss_window_start"],
        initial_gradient_window=initial,
        recent_gradient_window=recent,
        recent_gradient_window_start=history["recent_gradient_window_start"],
        validation_state=payload["validation_state"],
        convergence_state=payload["convergence_state"],
        forwardability_state=payload["forwardability_state"],
        terminal_state=payload["terminal_state"],
    )
    return state, payload


def _require_owned_m8_directory(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_dir()
        or stat.S_IMODE(path.stat().st_mode) != 0o700
        or path.stat().st_uid != os.getuid()
    ):
        raise G4ContractError(f"M8 {label} mode or owner changed")


def initialize_m8_output(
    output_dir: Path, fingerprints: dict[str, Any], validation_evidence: dict[str, Any]
) -> tuple[dict[str, Any], M8OptimizationState, dict[str, Any]]:
    """Create the M8 layout, immutable contract, and validated step-zero checkpoint."""
    if output_dir.exists() or output_dir.is_symlink() or output_dir.name != "output":
        raise G4ContractError("M8 output must be a wholly absent .../output path")
    output_dir.mkdir(mode=0o700)
    (output_dir / "checkpoints").mkdir(mode=0o700)
    (output_dir / "segments").mkdir(mode=0o700)
    contract = m8_contract_payload(fingerprints)
    _atomic_write(output_dir / "m8-run-contract.json", canonical_json_bytes(contract), mode=0o600)
    state = initial_m8_state(validation_evidence)
    checkpoint = save_m8_checkpoint(
        output_dir / "checkpoints/m8-checkpoint-step-000000.json",
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256=None,
        parent_checkpoint_scientific_sha256=None,
        parent_checkpoint_operational_sha256=None,
        segment_coordinates={"start_update": 0, "end_update": 0},
        update_evidence=None,
        validation_evidence=validation_evidence,
    )
    return contract, state, checkpoint


def _read_m8_contract(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or path.stat().st_uid != os.getuid()
    ):
        raise G4ContractError("M8 contract type, mode, or owner changed")
    payload = json.loads(path.read_bytes())
    claimed = payload.pop("payload_sha256", None)
    if claimed != sha256_bytes(canonical_json_bytes(payload)):
        raise G4ContractError("M8 contract checksum mismatch")
    payload["payload_sha256"] = claimed
    return payload


def _m8_checkpoint_inventory(path: Path) -> list[tuple[int, Path]]:
    inventory = []
    for entry in path.iterdir():
        text = entry.name.removeprefix("m8-checkpoint-step-").removesuffix(".json")
        if (
            not entry.name.startswith("m8-checkpoint-step-")
            or not entry.name.endswith(".json")
            or len(text) != 6
            or not text.isdigit()
        ):
            raise G4ContractError("M8 checkpoint inventory contains an unknown file")
        inventory.append((int(text), entry))
    inventory.sort(key=lambda item: item[0])
    if [count for count, _path in inventory] != list(range(len(inventory))):
        raise G4ContractError("M8 checkpoint chain is not contiguous; gap detected")
    return inventory


def _validate_m8_output(
    output_dir: Path, fingerprints: dict[str, Any], contract: dict[str, Any]
) -> None:
    _require_owned_m8_directory(output_dir, label="output")
    if {entry.name for entry in output_dir.iterdir()} != {
        "m8-run-contract.json",
        "checkpoints",
        "segments",
    }:
        raise G4ContractError("M8 output layout contains an unknown entry")
    _require_owned_m8_directory(output_dir / "checkpoints", label="checkpoints")
    _require_owned_m8_directory(output_dir / "segments", label="segments")
    if (
        contract != m8_contract_payload(fingerprints)
        or _read_m8_contract(output_dir / "m8-run-contract.json") != contract
    ):
        raise G4ContractError("M8 contract or fingerprints changed")
    _m8_checkpoint_inventory(output_dir / "checkpoints")
    for entry in (output_dir / "segments").iterdir():
        middle = entry.name.removeprefix("m8-segment-").removesuffix(".json")
        parts = middle.split("-")
        if (
            not entry.name.startswith("m8-segment-")
            or not entry.name.endswith(".json")
            or len(parts) != 2
            or any(len(part) != 6 or not part.isdigit() for part in parts)
            or entry.is_symlink()
            or not entry.is_file()
            or stat.S_IMODE(entry.stat().st_mode) != 0o600
            or entry.stat().st_uid != os.getuid()
        ):
            raise G4ContractError("M8 segment inventory changed")


def resume_m8_output(
    output_dir: Path, checkpoint_path: Path, fingerprints: dict[str, Any]
) -> tuple[M8OptimizationState, dict[str, Any]]:
    """Restore only the latest checkpoint of one contiguous three-digest M8 chain."""
    contract = m8_contract_payload(fingerprints)
    _validate_m8_output(output_dir, fingerprints, contract)
    inventory = _m8_checkpoint_inventory(output_dir / "checkpoints")
    if not inventory:
        raise G4ContractError("M8 checkpoint chain is empty")
    latest = inventory[-1][1]
    if (
        not checkpoint_path.is_absolute()
        or checkpoint_path.is_symlink()
        or checkpoint_path.resolve(strict=True) != latest.resolve(strict=True)
    ):
        raise G4ContractError("M8 resume checkpoint is not the latest verified checkpoint")
    parent_payload = None
    parent_scientific = None
    parent_operational = None
    state: M8OptimizationState | None = None
    payload: dict[str, Any] = {}
    for count, path in inventory:
        state, payload = load_m8_checkpoint(
            path,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            expected_parent_payload_sha256=parent_payload,
            expected_parent_scientific_sha256=parent_scientific,
            expected_parent_operational_sha256=parent_operational,
        )
        if state.adam.count != count:
            raise G4ContractError("M8 checkpoint chain counter changed")
        parent_payload = payload["payload_sha256"]
        parent_scientific = payload["scientific_payload_sha256"]
        parent_operational = payload["operational_payload_sha256"]
    assert state is not None
    if state.terminal_state["terminal"]:
        raise G4ContractError("terminal M8 checkpoint cannot be resumed")
    return state, payload


def evaluate_m8_validation(
    repository_root: Path,
    theta: np.ndarray,
    update_count: int,
    previous_validation_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the fixed public M8 validation split in five Resource-v1 batches."""
    items = fixed_day26_items(repository_root, (*VALIDATION_IDENTITIES, *NEAR_LIMIT_IDENTITIES))
    batch_items = (items[:4], items[4:8], items[8:9], items[9:13], items[13:15])
    evaluations = []
    batches = []
    observations = []
    started_unix_ns = time.time_ns()
    started = time.perf_counter()
    for index, current_items in enumerate(batch_items):
        memory_before = memory_snapshot()
        rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        resource_before = validate_longrun_resource_telemetry(memory_before, rss_before)
        evaluated = evaluate_items(theta, current_items, PARAMETER_NAMES)
        memory_after = memory_snapshot()
        rss_after = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        resource_after = validate_longrun_resource_telemetry(memory_after, rss_after)
        evaluations.append(evaluated)
        observations.extend((resource_before, resource_after))
        batches.append(
            {
                "batch_index": index,
                "split": "feasible" if index < 3 else "near_limit",
                "parent_count": evaluated["parent_count"],
                "parent_ids": evaluated["parent_ids"],
                "elapsed_seconds": evaluated["elapsed_seconds"],
                "resource_before": resource_before,
                "resource_after": resource_after,
            }
        )
    feasible = evaluations[:3]
    feasible_losses = np.concatenate(
        [np.asarray(item["aux"]["per_case_loss"], dtype=np.float32) for item in feasible]
    )
    if feasible_losses.shape != (9,) or not np.isfinite(feasible_losses).all():
        raise G4ContractError("M8 feasible validation loss population changed")
    scientific = synthetic_m8_validation_scientific(float(np.mean(feasible_losses)))
    scientific["per_parent"] = [
        {"parent_id": parent, "loss": float(loss)}
        for parent, loss in zip(VALIDATION_IDENTITIES, feasible_losses, strict=True)
    ]
    operational = {
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": time.time_ns(),
        "process_elapsed_seconds": time.perf_counter() - started,
        "resource_observation_count": 10,
        "maximum_rss_bytes": max(item["maximum_rss_bytes"] for item in observations),
        "minimum_mem_available_bytes": min(item["mem_available_bytes"] for item in observations),
        "maximum_swap_used_bytes": max(item["swap_used_bytes"] for item in observations),
        "evaluation_batches": batches,
        "host": {
            "hostname": os.uname().nodename,
            "interpreter": str(Path(sys.executable)),
            "interpreter_target": str(Path(sys.executable).resolve(strict=True)),
            "interpreter_sha256": sha256_file(Path(sys.executable).resolve(strict=True)),
        },
    }
    convergence = convergence_status([], [])
    return m8_validation_evidence_payload(
        update_count,
        theta,
        scientific,
        operational,
        previous_validation_state=previous_validation_state,
        convergence_state=convergence,
    )


def run_bounded_m8_segment(
    repository_root: Path,
    config_sha256: str,
    output_dir: Path,
    additional_updates: int,
    *,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    """Run one public bounded M8 segment with atomic checkpoints and three digests."""
    if (
        isinstance(additional_updates, bool)
        or not isinstance(additional_updates, int)
        or not 1 <= additional_updates <= LONGRUN_MAX_UPDATES
    ):
        raise G4ContractError("M8 additional update count must be in 1..100")
    parent_evidence = parent_population_evidence(repository_root)
    fingerprints = science_fingerprints(repository_root, config_sha256, parent_evidence)
    if resume_checkpoint is None:
        validate_task_output_path(output_dir)
        initial_validation = evaluate_m8_validation(
            repository_root, np.zeros(2, dtype=np.float32), 0, None
        )
        contract, state, start_checkpoint = initialize_m8_output(
            output_dir, fingerprints, initial_validation
        )
        mode = "m8_longrun"
    else:
        contract = m8_contract_payload(fingerprints)
        state, start_checkpoint = resume_m8_output(output_dir, resume_checkpoint, fingerprints)
        mode = "m8_resume"
    start_count = state.adam.count
    requested_end = start_count + additional_updates
    if requested_end > M8_MAX_UPDATES:
        raise G4ContractError("M8 resume budget exceeds update 2000")
    started_unix_ns = time.time_ns()
    started = time.perf_counter()
    previous = start_checkpoint
    new_checkpoints = []
    operational_updates = []
    validation_updates = []
    for expected_count in range(start_count + 1, requested_end + 1):
        state, update_evidence = one_m8_update(state)
        if state.adam.count != expected_count:
            raise G4ContractError("M8 bounded segment update counter changed")
        validation = None
        if expected_count in M8_VALIDATION_UPDATES:
            gradients = (
                state.initial_gradient_window
                if expected_count <= 100
                else state.initial_gradient_window + state.recent_gradient_window
            )
            convergence = convergence_status(gradients[-expected_count:], state.loss_window)
            validation = evaluate_m8_validation(
                repository_root, state.theta, expected_count, state.validation_state
            )
            validation["convergence"] = convergence
            validation_updates.append(expected_count)
        path = output_dir / f"checkpoints/m8-checkpoint-step-{expected_count:06d}.json"
        checkpoint = save_m8_checkpoint(
            path,
            state,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            parent_checkpoint_payload_sha256=previous["payload_sha256"],
            parent_checkpoint_scientific_sha256=previous["scientific_payload_sha256"],
            parent_checkpoint_operational_sha256=previous["operational_payload_sha256"],
            segment_coordinates={"start_update": start_count, "end_update": expected_count},
            update_evidence=update_evidence,
            validation_evidence=validation,
        )
        state, _loaded = load_m8_checkpoint(
            path,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            expected_parent_payload_sha256=previous["payload_sha256"],
            expected_parent_scientific_sha256=previous["scientific_payload_sha256"],
            expected_parent_operational_sha256=previous["operational_payload_sha256"],
        )
        new_checkpoints.append(
            {
                "update_count": expected_count,
                "path": str(path),
                "mode": "0600",
                "owner_uid": os.getuid(),
                "file_sha256": sha256_file(path),
                "payload_sha256": checkpoint["payload_sha256"],
                "scientific_payload_sha256": checkpoint["scientific_payload_sha256"],
                "operational_payload_sha256": checkpoint["operational_payload_sha256"],
            }
        )
        operational_updates.append(update_evidence["operational"])
        previous = checkpoint
        if checkpoint["terminal_state"]["terminal"]:
            break
    end_count = state.adam.count
    start_path = output_dir / f"checkpoints/m8-checkpoint-step-{start_count:06d}.json"
    end_path = output_dir / f"checkpoints/m8-checkpoint-step-{end_count:06d}.json"
    terminal = previous["terminal_state"]
    status = terminal["reason"] if terminal["terminal"] else "PASS_M8_BOUNDED_SEGMENT"
    segment_path = output_dir / f"segments/m8-segment-{start_count:06d}-{end_count:06d}.json"
    payload = {
        "schema_version": M8_SEGMENT_SCHEMA_VERSION,
        "mode": mode,
        "status": status,
        "claim_boundary": (
            "two-parameter simulation candidate evidence only; no acceptance, robustness, "
            "general Mellinger, firmware, hardware, or flight claim"
        ),
        "requested_start_count": start_count,
        "actual_start_count": start_checkpoint["update_count"],
        "requested_end_count": requested_end,
        "actual_end_count": end_count,
        "completed_updates": end_count - start_count,
        "terminal_state": terminal,
        "start_checkpoint": str(start_path),
        "end_checkpoint": str(end_path),
        "start_checkpoint_digests": {
            "payload_sha256": start_checkpoint["payload_sha256"],
            "scientific_payload_sha256": start_checkpoint["scientific_payload_sha256"],
            "operational_payload_sha256": start_checkpoint["operational_payload_sha256"],
        },
        "end_checkpoint_digests": {
            "payload_sha256": previous["payload_sha256"],
            "scientific_payload_sha256": previous["scientific_payload_sha256"],
            "operational_payload_sha256": previous["operational_payload_sha256"],
        },
        "run_contract_sha256": contract["payload_sha256"],
        "fingerprints": fingerprints,
        "new_checkpoints": new_checkpoints,
        "validation_updates": validation_updates,
        "evidence_manifest": new_checkpoints,
        "operational_summary": {
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": time.time_ns(),
            "process_elapsed_seconds": time.perf_counter() - started,
            "maximum_rss_bytes": max(
                (item["maximum_rss_bytes"] for item in operational_updates), default=0
            ),
            "minimum_mem_available_bytes": min(
                (item["minimum_mem_available_bytes"] for item in operational_updates),
                default=MINIMUM_AVAILABLE_BYTES,
            ),
            "maximum_swap_used_bytes": max(
                (item["maximum_swap_used_bytes"] for item in operational_updates), default=0
            ),
        },
    }
    payload["payload_sha256"] = _segment_payload_sha256(payload)
    _atomic_write(segment_path, canonical_json_bytes(payload), mode=0o600)
    result = dict(payload)
    result["segment_record"] = str(segment_path)
    return result


def run_joint_smoke(repository_root: Path, config_sha256: str, output_dir: Path) -> dict[str, Any]:
    """Run the fresh resource gate and exactly ten technical joint updates."""
    resource_gate = effective_32_no_update()
    parent_evidence = parent_population_evidence(repository_root)
    fingerprints = science_fingerprints(repository_root, config_sha256, parent_evidence)
    state, updates = run_updates(PARAMETER_NAMES, 10)
    checkpoint = save_checkpoint(
        output_dir / "checkpoint-10.json", state, PARAMETER_NAMES, fingerprints
    )
    if state.adam.count != 10 or len(updates) != 10:
        raise G4ContractError("M5 did not execute exactly ten updates")
    return {
        "status": "PASS_M5",
        "claim_boundary": "technical joint smoke only; no improvement claim",
        "fresh_resource_gate": resource_gate,
        "updates": updates,
        "final_theta": state.theta.tolist(),
        "final_physical": physical_from_theta(state.theta).tolist(),
        "adam_count": state.adam.count,
        "projection_count": state.projection_count,
        "checkpoint_payload_sha256": checkpoint["payload_sha256"],
        "checkpoint_file_sha256": sha256_file(output_dir / "checkpoint-10.json"),
        "fingerprints": fingerprints,
    }


def update_records_digest(records: Sequence[dict[str, Any]]) -> str:
    """Hash the scientific bytes of update records, excluding no numerical field."""
    return sha256_bytes(canonical_json_bytes(list(records)))


def validate_task_output_path(path: Path, *, must_be_absent: bool = True) -> Path:
    """Require a fresh ``/tmp/gr-g4-004-*/output`` directory without traversal."""
    if not path.is_absolute() or path.name != "output":
        raise G4ContractError("output must be an absolute .../output path")
    parent = path.parent
    if not parent.name.startswith("gr-g4-004-") or parent.parent != Path("/tmp"):
        raise G4ContractError("output must be under a fresh /tmp/gr-g4-004-* root")
    if parent.is_symlink() or stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise G4ContractError("task root must be an existing mode-0700 non-symlink directory")
    if must_be_absent and (path.exists() or path.is_symlink()):
        raise G4ContractError("output path must be absent before the target")
    return path


def write_result(path: Path, payload: dict[str, Any]) -> None:
    """Write one canonical, overwrite-refusing, mode-0600 result."""
    _atomic_write(path, canonical_json_bytes(payload), mode=0o600)


def dry_run_contract(repository_root: Path, config_path: Path) -> dict[str, Any]:
    """Validate schemas, pins, allowlist, and frozen parents without rollout/optimizer."""
    public = validate_public_inputs(repository_root)
    _, config_sha256 = load_and_validate_config(config_path, repository_root)
    parents = parent_population_evidence(repository_root)
    return {
        "status": "PASS_M1_DRY_RUN",
        "public_inputs": public,
        "config_sha256": config_sha256,
        "parameters": parameter_contract(),
        "loss_v1": loss_v1_contract(),
        "parents": parents,
        "protection": protected_contract(),
        "write_allowlist": list(WRITE_PATHS),
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }


def resume_worker(
    repository_root: Path, config_sha256: str, checkpoint_path: Path, output_path: Path
) -> dict[str, Any]:
    """Load update 10 in a fresh process and execute updates 11--20."""
    parent_evidence = parent_population_evidence(repository_root)
    fingerprints = science_fingerprints(repository_root, config_sha256, parent_evidence)
    state, checkpoint = load_checkpoint(checkpoint_path, PARAMETER_NAMES, fingerprints)
    if state.adam.count != 10:
        raise G4ContractError("resume worker requires exactly update 10")
    resumed_state, updates = run_updates(PARAMETER_NAMES, 10, initial_state=state)
    if [record["update"] for record in updates] != list(range(11, 21)):
        raise G4ContractError("resume worker update inventory changed")
    final_checkpoint = save_checkpoint(
        output_path.parent / "checkpoint-20.json",
        resumed_state,
        PARAMETER_NAMES,
        fingerprints,
        parent_checkpoint_sha256=checkpoint["payload_sha256"],
    )
    result = {
        "status": "PASS_RESUME_WORKER",
        "updates": updates,
        "updates_sha256": update_records_digest(updates),
        "final_theta": _array_record(resumed_state.theta),
        "final_adam_first": _array_record(resumed_state.adam.first_moment),
        "final_adam_second": _array_record(resumed_state.adam.second_moment),
        "final_adam_count": resumed_state.adam.count,
        "projection_count": resumed_state.projection_count,
        "parent_checkpoint_payload_sha256": checkpoint["payload_sha256"],
        "checkpoint_payload_sha256": final_checkpoint["payload_sha256"],
    }
    write_result(output_path, result)
    return result


def compare_resume_records(
    continuous_updates: Sequence[dict[str, Any]], resumed_updates: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Require canonical byte identity for every scientific update 11--20 field."""
    expected = list(continuous_updates)[10:20]
    actual = list(resumed_updates)
    if [record["update"] for record in expected] != list(range(11, 21)):
        raise G4ContractError("continuous update 11--20 inventory changed")
    expected_bytes = canonical_json_bytes(expected)
    actual_bytes = canonical_json_bytes(actual)
    if expected_bytes != actual_bytes:
        raise G4ContractError("M6 updates 11--20 are not byte-identical")
    return {
        "status": "PASS_BYTE_IDENTICAL_UPDATES_11_20",
        "expected_sha256": sha256_bytes(expected_bytes),
        "actual_sha256": sha256_bytes(actual_bytes),
        "byte_count": len(expected_bytes),
        "updates": list(range(11, 21)),
    }


def run_resume_reproduction_parent(
    repository_root: Path,
    config_path: Path,
    config_sha256: str,
    source_origin_record: Path,
    output_dir: Path,
    cli_path: Path,
) -> dict[str, Any]:
    """Run 20 continuous versus 10+fresh-process+10 and compare bytes."""
    resource_gate = effective_32_no_update()
    parent_evidence = parent_population_evidence(repository_root)
    fingerprints = science_fingerprints(repository_root, config_sha256, parent_evidence)
    continuous_state, continuous_updates = run_updates(PARAMETER_NAMES, 20)
    continuous_checkpoint = save_checkpoint(
        output_dir / "continuous-checkpoint-20.json",
        continuous_state,
        PARAMETER_NAMES,
        fingerprints,
    )
    split_state, split_updates = run_updates(PARAMETER_NAMES, 10)
    split_checkpoint_path = output_dir / "split-checkpoint-10.json"
    split_checkpoint = save_checkpoint(
        split_checkpoint_path, split_state, PARAMETER_NAMES, fingerprints
    )
    if canonical_json_bytes(continuous_updates[:10]) != canonical_json_bytes(split_updates):
        raise G4ContractError("continuous and split updates 1--10 differ")
    worker_root = Path(tempfile.mkdtemp(prefix="gr-g4-004-m6-worker-", dir="/tmp"))
    os.chmod(worker_root, 0o700)
    worker_environment = os.environ.copy()
    worker_directories = {
        "XDG_CACHE_HOME": worker_root / "xdg",
        "MPLCONFIGDIR": worker_root / "matplotlib",
        "TMPDIR": worker_root / "tmp",
        "JAX_COMPILATION_CACHE_DIR": worker_root / "jax-cache",
    }
    for name, path in worker_directories.items():
        path.mkdir(mode=0o700)
        worker_environment[name] = str(path)
    worker_result_path = worker_root / "output/resume-worker-result.json"
    command = (
        str(EXPECTED_INTERPRETER),
        str(cli_path),
        "--config",
        str(config_path),
        "--source-origin-record",
        str(source_origin_record),
        "--resume-worker-checkpoint",
        str(split_checkpoint_path),
        "--resume-worker-output",
        str(worker_result_path),
    )
    completed = subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        env=worker_environment,
    )
    if completed.returncode != 0:
        raise G4ContractError(
            f"fresh resume worker failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    resumed = json.loads(worker_result_path.read_bytes())
    comparison = compare_resume_records(continuous_updates, resumed["updates"])
    if (
        resumed["final_theta"] != _array_record(continuous_state.theta)
        or resumed["final_adam_first"] != _array_record(continuous_state.adam.first_moment)
        or resumed["final_adam_second"] != _array_record(continuous_state.adam.second_moment)
        or resumed["final_adam_count"] != 20
        or resumed["projection_count"] != continuous_state.projection_count
    ):
        raise G4ContractError("M6 final optimizer/checkpoint state differs")
    return {
        "status": "PASS_M6",
        "claim_boundary": "technical resume reproduction only; no improvement claim",
        "fresh_resource_gate": resource_gate,
        "fresh_worker_argv": list(command),
        "fresh_worker_exit": completed.returncode,
        "fresh_worker_root": str(worker_root),
        "continuous_updates_sha256": update_records_digest(continuous_updates),
        "split_first_ten_sha256": update_records_digest(split_updates),
        "comparison": comparison,
        "continuous_checkpoint_payload_sha256": continuous_checkpoint["payload_sha256"],
        "continuous_checkpoint_file_sha256": sha256_file(
            output_dir / "continuous-checkpoint-20.json"
        ),
        "split_checkpoint_payload_sha256": split_checkpoint["payload_sha256"],
        "split_checkpoint_file_sha256": sha256_file(split_checkpoint_path),
        "resumed_checkpoint_payload_sha256": resumed["checkpoint_payload_sha256"],
        "resumed_checkpoint_file_sha256": sha256_file(worker_root / "output/checkpoint-20.json"),
        "final_theta": continuous_state.theta.tolist(),
        "final_physical": physical_from_theta(continuous_state.theta).tolist(),
        "adam_count": continuous_state.adam.count,
        "projection_count": continuous_state.projection_count,
    }


def required_source_files_exist(repository_root: Path) -> None:
    """Require exactly the known five-path source inventory without repository discovery."""
    root = repository_root.resolve(strict=True)
    for relative_path in WRITE_PATHS:
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise G4ContractError(f"required G4 source path missing or nonregular: {relative_path}")


def direct_parameter_controller_check(data: SimData) -> dict[str, Any]:
    """Check theta-zero and each isolated coordinate at the controller call boundary."""
    state = data.controls.state
    if state is None:
        raise G4ContractError("state controller unavailable")
    command = state.cmd
    records = {}
    for name in PARAMETER_NAMES:
        theta_zero = apply_theta(data, jnp.zeros((1,), dtype=jnp.float32), (name,))
        baseline_state = data.controls.state
        changed_state = theta_zero.controls.state
        assert baseline_state is not None and changed_state is not None
        args = (data.states.pos, data.states.quat, data.states.vel, command)
        baseline_outputs = state2attitude(
            *args,
            pos_err_i=baseline_state.pos_err_i,
            ctrl_freq=baseline_state.freq,
            **baseline_state.params,
        )
        changed_outputs = state2attitude(
            *args,
            pos_err_i=changed_state.pos_err_i,
            ctrl_freq=changed_state.freq,
            **changed_state.params,
        )
        robust.require_pytree_equal(
            f"{name} direct theta-zero controller", baseline_outputs, changed_outputs
        )
        records[name] = parameter_isolation_record(data, name)
    return {"status": "PASS_DIRECT_PARAMETER_CONTROLLER", "parameters": records}


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise G4ContractError(f"{label} key inventory changed")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G4ContractError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise G4ContractError(f"{label} must be a JSON object")
    return value


def load_backend_input_manifest(path: Path, repository_root: Path) -> dict[str, Any]:
    """Load the strict relocatable Backend-Evidence-v1 input manifest."""
    root = repository_root.resolve(strict=True)
    payload = _load_json_object(path, "backend input manifest")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "backend_neutral_parents",
            "claim_boundary",
            "interface_parent",
            "immutable_product",
            "public_inputs",
            "science",
            "runtime_profiles",
            "parity_contract",
            "throughput_contract",
            "resume_contract",
            "protection",
        },
        "backend input manifest",
    )
    if payload["schema_version"] != BACKEND_INPUT_MANIFEST_SCHEMA_VERSION:
        raise G4ContractError("backend input manifest schema changed")
    if payload["interface_parent"] != {
        "commit": "05cd1db24092f71f39ad23a576bca243a431c035",
        "tree": "fce30d3619f9cca2e912c9ac82afd569a27a3a4c",
    }:
        raise G4ContractError("backend interface parent changed")
    if list(payload["runtime_profiles"]) != ["local_cpu", "remote_pixi_cpu", "remote_pixi_gpu"]:
        raise G4ContractError("backend runtime-profile inventory changed")
    public_inputs = payload["public_inputs"]
    if not isinstance(public_inputs, list) or [item.get("path") for item in public_inputs] != list(
        DAY26_INPUTS
    ):
        raise G4ContractError("backend public-input order changed")
    for item in public_inputs:
        _require_exact_keys(item, {"mode", "path", "sha256"}, "backend public input")
        expected = DAY26_INPUTS[item["path"]]["sha256"]
        source = root / item["path"]
        if item != {"mode": "100644", "path": item["path"], "sha256": expected}:
            raise G4ContractError("backend public-input pin changed")
        if source.is_symlink() or not source.is_file() or sha256_file(source) != expected:
            raise G4ContractError("backend public input is stale")
    immutable = payload["immutable_product"]
    if not isinstance(immutable, list):
        raise G4ContractError("backend immutable-product inventory changed")
    for item in immutable:
        _require_exact_keys(item, {"blob", "mode", "path", "sha256"}, "backend immutable product")
        product_path = root / item["path"]
        if product_path.is_symlink() or not product_path.is_file():
            raise G4ContractError("backend immutable product missing")
        if sha256_file(product_path) != item["sha256"]:
            raise G4ContractError("backend immutable product pin changed")
    load_backend_neutral_parent_payload(root)
    return payload


def validate_backend_runtime_payload(
    payload: dict[str, Any],
    *,
    expected_repository_root: Path,
    expected_backend: str,
    expected_profile: str,
    observed_payload: dict[str, Any],
) -> dict[str, Any]:
    """Reject every unrecognized or unobserved Backend-Evidence runtime field."""
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "profile",
            "backend",
            "repository",
            "interpreter",
            "environment",
            "jax",
            "device",
            "cache_roots",
            "environment_flags",
        },
        "backend runtime",
    )
    if payload["schema_version"] != BACKEND_RUNTIME_SCHEMA_VERSION:
        raise G4ContractError("backend runtime schema changed")
    profiles = {
        "local_cpu": ("cpu", "repository_venv", "cpu"),
        "remote_pixi_cpu": ("cpu", "pixi", "cpu"),
        "remote_pixi_gpu": ("gpu", "pixi", "gpu"),
    }
    if expected_profile not in profiles or payload["profile"] != expected_profile:
        raise G4ContractError("unknown or mismatched backend runtime profile")
    backend, manager, observed_backend = profiles[expected_profile]
    if expected_backend != backend or payload["backend"] != backend:
        raise G4ContractError("unknown or mismatched backend")
    repository = _require_exact_keys(
        payload["repository"], {"root", "head", "tree", "clean"}, "backend repository"
    )
    if (
        repository["root"] != str(expected_repository_root.resolve())
        or type(repository["clean"]) is not bool
    ):
        raise G4ContractError("backend repository root or clean flag changed")
    interpreter = _require_exact_keys(
        payload["interpreter"],
        {"path", "target", "sha256", "python_version"},
        "backend interpreter",
    )
    if not Path(interpreter["path"]).is_absolute() or not Path(interpreter["target"]).is_absolute():
        raise G4ContractError("backend interpreter path is not absolute")
    environment = _require_exact_keys(
        payload["environment"],
        {"manager", "name", "pixi_version", "pyproject_sha256", "pixi_lock_sha256"},
        "backend environment",
    )
    if environment["manager"] != manager:
        raise G4ContractError("backend environment manager changed")
    if expected_profile == "local_cpu":
        if any(environment[key] is not None for key in ("name", "pixi_version")):
            raise G4ContractError("local CPU environment gained Pixi fields")
    elif environment["name"] != "gpu" or environment["pixi_version"] != "0.70.0":
        raise G4ContractError("remote Pixi environment changed")
    jax_payload = _require_exact_keys(
        payload["jax"],
        {
            "python_version",
            "jax_version",
            "jaxlib_version",
            "numpy_version",
            "jax_cuda12_plugin_version",
            "jax_cuda12_pjrt_version",
            "float_dtype",
            "x64_enabled",
        },
        "backend JAX runtime",
    )
    if (
        jax_payload["jax_version"] != "0.10.1"
        or jax_payload["jaxlib_version"] != "0.10.1"
        or jax_payload["float_dtype"] != "float32"
        or jax_payload["x64_enabled"] is not False
    ):
        raise G4ContractError("backend JAX runtime changed")
    device = _require_exact_keys(
        payload["device"], {"observed_backend", "device_count", "devices"}, "backend device"
    )
    if device["observed_backend"] != observed_backend or device["device_count"] != len(
        device["devices"]
    ):
        raise G4ContractError("backend device inventory changed")
    if expected_backend == "gpu" and (
        device["device_count"] != 1
        or device["devices"][0].get("platform") != "gpu"
        or "RTX 5090" not in device["devices"][0].get("device_kind", "")
    ):
        raise G4ContractError("GPU runtime is not the single RTX 5090 contract")
    cache_roots = _require_exact_keys(
        payload["cache_roots"], {"xdg", "tmp", "jax", "cuda"}, "backend cache roots"
    )
    if any(not Path(value).is_absolute() for value in cache_roots.values()):
        raise G4ContractError("backend cache root is not absolute")
    flags = _require_exact_keys(
        payload["environment_flags"],
        {
            "pythonpath",
            "python_dont_write_bytecode",
            "python_no_user_site",
            "jax_enable_x64",
            "jax_platforms",
            "xla_flags",
            "xla_flags_sha256",
        },
        "backend environment flags",
    )
    if flags["pythonpath"] != repository["root"]:
        raise G4ContractError("backend PYTHONPATH changed")
    if flags["xla_flags_sha256"] != sha256_bytes(flags["xla_flags"].encode()):
        raise G4ContractError("backend XLA_FLAGS checksum changed")
    if payload != observed_payload:
        raise G4ContractError("backend runtime differs from observed process")
    return payload


def backend_numeric_record(value: Any, category: str) -> dict[str, Any]:
    """Encode one finite Float32 Backend-Evidence numeric value with fixed tolerances."""
    tolerances = {
        "rollout": (1.0e-5, 1.0e-6),
        "loss_gradient": (1.0e-5, 1.0e-7),
        "exact": (0.0, 0.0),
    }
    if category not in tolerances:
        raise G4ContractError("unknown backend numeric category")
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise G4ContractError("backend numeric evidence is nonfinite")
    raw = array.tobytes(order="C")
    rtol, atol = tolerances[category]
    return {
        "category": category,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "finite": True,
        "rtol": rtol,
        "atol": atol,
        "data_hex": raw.hex(),
        "sha256": sha256_bytes(raw),
    }


def _backend_evidence_payload(
    *,
    schema_version: str,
    mode: str,
    status: str,
    backend: str,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    resource_payload: dict[str, Any],
    scientific: dict[str, Any],
    operational: dict[str, Any],
) -> dict[str, Any]:
    if backend not in {"cpu", "gpu"}:
        raise G4ContractError("unknown backend evidence backend")
    scientific_sha = sha256_bytes(canonical_json_bytes(scientific))
    operational_projection = {
        "backend": backend,
        "provenance": provenance,
        "runtime": runtime,
        "resource": resource_payload,
        "operational": operational,
    }
    payload = {
        "schema_version": schema_version,
        "mode": mode,
        "status": status,
        "claim_boundary": (
            "public simulation evidence only; no test-split, convergence, speedup, "
            "superiority, robustness, firmware, hardware, or flight claim"
        ),
        "backend": backend,
        "provenance": provenance,
        "runtime": runtime,
        "resource": resource_payload,
        "scientific": scientific,
        "operational": operational,
        "scientific_payload_sha256": scientific_sha,
        "operational_payload_sha256": sha256_bytes(canonical_json_bytes(operational_projection)),
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def backend_parity_evidence_payload(
    *,
    backend: str,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    resource: dict[str, Any],
    scientific: dict[str, Any],
    operational: dict[str, Any],
) -> dict[str, Any]:
    """Build one backend-local parity sample without a cross-backend claim."""
    if len(scientific.get("loss_gradient", {}).get("microbatches", ())) != 8:
        raise G4ContractError("backend parity requires eight microbatches")
    return _backend_evidence_payload(
        schema_version=BACKEND_PARITY_SCHEMA_VERSION,
        mode="parity",
        status="PASS_BACKEND_PARITY_SAMPLE",
        backend=backend,
        provenance=provenance,
        runtime=runtime,
        resource_payload=resource,
        scientific=scientific,
        operational=operational,
    )


def backend_throughput_evidence_payload(
    *,
    backend: str,
    phase: str,
    sample_index: int,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    resource: dict[str, Any],
    scientific: dict[str, Any],
    elapsed_seconds: float,
    device_memory: dict[str, int | None],
) -> dict[str, Any]:
    """Build one blocked effective-32 throughput sample."""
    if phase not in {"warmup", "steady"} or sample_index not in {1, 2, 3}:
        raise G4ContractError("unknown throughput phase or sample")
    if phase == "warmup" and sample_index != 1:
        raise G4ContractError("warmup requires sample index 1")
    if scientific.get("microbatch_parent_counts") != [4] * 8:
        raise G4ContractError("throughput effective-32 layout changed")
    if scientific.get("adam_count") != 1 or not math.isfinite(elapsed_seconds):
        raise G4ContractError("throughput update or timing changed")
    memory_keys = {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
    _require_exact_keys(device_memory, memory_keys, "backend device memory")
    if backend == "cpu" and any(value is not None for value in device_memory.values()):
        raise G4ContractError("CPU throughput gained device-memory values")
    if backend == "gpu" and any(type(value) is not int for value in device_memory.values()):
        raise G4ContractError("GPU throughput lacks device-memory values")
    return _backend_evidence_payload(
        schema_version=BACKEND_THROUGHPUT_SCHEMA_VERSION,
        mode="throughput",
        status="PASS_BACKEND_THROUGHPUT_SAMPLE",
        backend=backend,
        provenance=provenance,
        runtime=runtime,
        resource_payload=resource,
        scientific=scientific,
        operational={
            "phase": phase,
            "sample_index": sample_index,
            "block_until_ready": True,
            "elapsed_seconds": float(elapsed_seconds),
            "valid_updates": 1,
            "valid_worlds": 32,
            "device_memory": device_memory,
        },
    )


def observe_backend_device_memory(backend: str) -> dict[str, int | None]:
    """Observe device-memory counters from the selected JAX device."""
    keys = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    if backend == "cpu":
        return {key: None for key in keys}
    if backend != "gpu":
        raise G4ContractError("unknown backend device-memory target")
    devices = tuple(jax.devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise G4ContractError("GPU device-memory observation requires one selected GPU")
    stats = devices[0].memory_stats()
    if not isinstance(stats, dict):
        raise G4ContractError("GPU device-memory observation is unavailable")
    result = {}
    for key in keys:
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise G4ContractError("GPU device-memory observation is incomplete")
        result[key] = value
    return result


def backend_resume_segment_payload(
    *,
    mode: str,
    backend: str,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    resource: dict[str, Any],
    scientific: dict[str, Any],
    segment_coordinates: dict[str, int],
    operational: dict[str, Any],
) -> dict[str, Any]:
    """Build a GPU-internal continuous or resume segment evidence record."""
    expected = {
        "continuous": {"start_update": 0, "end_update": 20},
        "resume": {"start_update": 10, "end_update": 20},
    }
    if backend != "gpu" or mode not in expected or segment_coordinates != expected[mode]:
        raise G4ContractError("backend resume segment coordinates changed")
    operational_payload = dict(operational)
    operational_payload["segment_coordinates"] = segment_coordinates
    return _backend_evidence_payload(
        schema_version=BACKEND_RESUME_SCHEMA_VERSION,
        mode=mode,
        status="PASS_BACKEND_RESUME_SEGMENT",
        backend=backend,
        provenance=provenance,
        runtime=runtime,
        resource_payload=resource,
        scientific=scientific,
        operational=operational_payload,
    )


def backend_resume_payload_from_m8_segment(
    *,
    mode: str,
    backend: str,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    resource: dict[str, Any],
    segment: dict[str, Any],
) -> dict[str, Any]:
    """Project one completed M8 segment into public Backend-Resume-v1 evidence."""
    coordinates = {
        "start_update": segment.get("actual_start_count"),
        "end_update": segment.get("actual_end_count"),
    }
    expected = {
        "continuous": {"start_update": 0, "end_update": 20},
        "resume": {"start_update": 10, "end_update": 20},
    }
    if mode not in expected or coordinates != expected[mode]:
        raise G4ContractError("backend resume source segment coordinates changed")
    start_path = Path(segment.get("start_checkpoint", ""))
    end_path = Path(segment.get("end_checkpoint", ""))
    start = _load_json_object(start_path, "backend resume start checkpoint")
    end = _load_json_object(end_path, "backend resume end checkpoint")
    required_state = {
        "update_count",
        "theta",
        "adam",
        "rng_coordinates",
        "history_state",
        "projection_count",
        "scientific_update",
        "payload_sha256",
        "scientific_payload_sha256",
        "operational_payload_sha256",
    }
    if not required_state <= set(start) or not required_state <= set(end):
        raise G4ContractError("backend resume checkpoint evidence is incomplete")
    if start["update_count"] != coordinates["start_update"] or end["update_count"] != 20:
        raise G4ContractError("backend resume checkpoint counters changed")
    end_scientific = {
        "update_count": end["update_count"],
        "theta": end["theta"],
        "adam": end["adam"],
        "rng_coordinates": end["rng_coordinates"],
        "history_state": end["history_state"],
        "projection_count": end["projection_count"],
        "scientific_update": end["scientific_update"],
        "end_checkpoint_scientific_sha256": end["scientific_payload_sha256"],
        "finite_checkpoint_state": True,
    }
    numeric_values = []

    def collect_numbers(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            numeric_values.append(float(value))
        elif isinstance(value, dict):
            for child in value.values():
                collect_numbers(child)
        elif isinstance(value, list):
            for child in value:
                collect_numbers(child)

    collect_numbers(end_scientific)
    if not all(math.isfinite(value) for value in numeric_values):
        raise G4ContractError("backend resume checkpoint state is nonfinite")
    checkpoint_keys = ("payload_sha256", "scientific_payload_sha256", "operational_payload_sha256")
    operational = {
        "start_checkpoint": {
            "path": str(start_path),
            "update_count": start["update_count"],
            **{key: start[key] for key in checkpoint_keys},
        },
        "end_checkpoint": {
            "path": str(end_path),
            "update_count": end["update_count"],
            **{key: end[key] for key in checkpoint_keys},
        },
        "parent_lineage": {
            key: end.get(key)
            for key in (
                "parent_checkpoint_payload_sha256",
                "parent_checkpoint_scientific_sha256",
                "parent_checkpoint_operational_sha256",
            )
        },
        "run_contract_sha256": segment.get("run_contract_sha256"),
        "fingerprints": segment.get("fingerprints"),
        "resource_gate": segment.get("operational_summary"),
    }
    resource_gate = operational["resource_gate"]
    if not isinstance(resource_gate, dict) or resource_gate.get("maximum_swap_used_bytes") != 0:
        raise G4ContractError("backend resume resource gate changed")
    return backend_resume_segment_payload(
        mode=mode,
        backend=backend,
        provenance=provenance,
        runtime=runtime,
        resource=resource,
        scientific=end_scientific,
        segment_coordinates=coordinates,
        operational=operational,
    )


def backend_source_origin_payload(
    repository_root: Path, input_manifest_path: Path
) -> dict[str, Any]:
    """Bind the Backend-Evidence interface, immutable inputs, and local CPU runtime."""
    root = repository_root.resolve(strict=True)
    manifest = load_backend_input_manifest(input_manifest_path, root)
    paths = (
        "crazyflow/control/mellinger/research/g4_optimization.py",
        "examples/jax/mellinger_g4_optimization.py",
        "tests/unit/test_mellinger_g4_optimization.py",
        "tests/integration/test_mellinger_g4_optimization.py",
        BACKEND_INPUT_MANIFEST_RELATIVE_PATH,
        CONFIG_RELATIVE_PATH,
        "crazyflow/__init__.py",
        "pyproject.toml",
        "pixi.lock",
        *DAY26_INPUTS,
    )
    origins = {
        path: {"file": str(root / path), "sha256": sha256_file(root / path)} for path in paths
    }
    payload = {
        "schema_version": BACKEND_SOURCE_ORIGIN_SCHEMA_VERSION,
        "repository_root": str(root),
        "runtime": validate_runtime_environment(root),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "input_manifest_schema_version": manifest["schema_version"],
        "origins": origins,
    }
    payload["record_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def write_backend_source_origin_record(
    repository_root: Path, input_manifest_path: Path, path: Path
) -> dict[str, Any]:
    """Atomically write one mode-0600 Backend-Evidence source-origin record."""
    payload = backend_source_origin_payload(repository_root, input_manifest_path)
    _atomic_write(path, canonical_json_bytes(payload), mode=0o600)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise G4ContractError("backend source-origin record mode is not 0600")
    return payload


def run_backend_parity(**kwargs: Any) -> dict[str, Any]:
    """Execute one backend-local parity sample through the established science path."""
    repository_root = Path(kwargs["repository_root"])
    backend = kwargs["backend"]
    items = fixed_day26_items(repository_root, VALIDATION_IDENTITIES[:4], backend=backend)
    evaluation = evaluate_items(
        np.zeros(2, dtype=np.float32), _batch_items(0, backend=backend), backend=backend
    )
    parity = default_parity(
        repository_root, VALIDATION_IDENTITIES[:4], include_backend_evidence=True, backend=backend
    )
    parity_evidence = parity.get("_backend_evidence", parity)
    canonical_final = parity_evidence["canonical_final"]
    left_loss = parity_evidence.get("left_loss", parity["loss"])
    left_aux = parity_evidence.get("left_aux", parity_evidence.get("loss_aux"))
    if not isinstance(left_aux, dict):
        raise G4ContractError("backend parity tracking evidence is unavailable")
    tracking_names = (
        "position_rmse_m",
        "velocity_rmse_m_s",
        "max_position_error_m",
        "control_effort",
        "control_smoothness",
        "motor_saturation_fraction",
        "zero_thrust_gate_fraction",
        "floor_clip_fraction",
        "nonfinite_state_fraction",
    )
    scientific = {
        "parent_ids": {
            "default_rollout": parity["parent_ids"],
            "loss_gradient": evaluation["parent_ids"],
        },
        "parameter_names": list(PARAMETER_NAMES),
        "default_theta": backend_numeric_record(np.zeros(2, dtype=np.float32), "rollout"),
        "default_physical": backend_numeric_record(
            physical_from_theta(np.zeros(2, dtype=np.float32)), "rollout"
        ),
        "bounds": {
            "raw": backend_numeric_record(
                np.asarray(
                    [[spec.raw_lower, spec.raw_upper] for spec in PARAMETER_SPECS.values()],
                    dtype=np.float32,
                ),
                "rollout",
            ),
            "physical": backend_numeric_record(
                np.asarray(
                    [
                        [spec.physical_lower, spec.physical_upper]
                        for spec in PARAMETER_SPECS.values()
                    ],
                    dtype=np.float32,
                ),
                "rollout",
            ),
        },
        "default_rollout": {
            "final_carry": [
                {"path": path, "value": backend_numeric_record(leaf, "rollout")}
                for path, leaf in g3._leaf_records(canonical_final)
            ],
            "tracking": {
                "loss_total": backend_numeric_record(left_loss, "rollout"),
                **{
                    f"loss_{name}": backend_numeric_record(left_aux[f"loss_{name}"], "rollout")
                    for name in LOSS_TERM_NAMES
                },
                **{
                    name: backend_numeric_record(left_aux[f"metric_{name}"], "rollout")
                    for name in tracking_names
                },
            },
        },
        "loss_gradient": {
            "microbatches": [
                {
                    "microbatch_index": index,
                    "parent_ids": evaluation["parent_ids"][index * 4 : (index + 1) * 4],
                    "mean_loss": backend_numeric_record(evaluation["loss"], "loss_gradient"),
                    "loss_terms": {
                        name: backend_numeric_record(
                            np.mean(evaluation["aux"][f"loss_{name}"]), "loss_gradient"
                        )
                        for name in LOSS_TERM_NAMES
                    },
                    "gradient": backend_numeric_record(evaluation["gradient"], "loss_gradient"),
                }
                for index in range(8)
            ],
            "aggregate": {
                "parent_ids": evaluation["parent_ids"],
                "mean_loss": backend_numeric_record(evaluation["loss"], "loss_gradient"),
                "loss_terms": {
                    name: backend_numeric_record(
                        np.mean(evaluation["aux"][f"loss_{name}"]), "loss_gradient"
                    )
                    for name in LOSS_TERM_NAMES
                },
                "gradient": backend_numeric_record(evaluation["gradient"], "loss_gradient"),
                "gradient_norm": backend_numeric_record(
                    np.linalg.norm(evaluation["gradient"]), "loss_gradient"
                ),
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
                "optimizer_update_count": 0,
            },
        },
    }
    del items
    return backend_parity_evidence_payload(
        backend=kwargs["backend"],
        provenance=kwargs["provenance"],
        runtime=kwargs["runtime"],
        resource=kwargs["resource"],
        scientific=scientific,
        operational={"sample_index": 1},
    )


def run_backend_throughput(**kwargs: Any) -> dict[str, Any]:
    """Execute exactly one blocked effective-32 update for throughput evidence."""
    state = initial_optimization_state(2)
    started = time.perf_counter()
    updated, record = one_effective_batch(state, PARAMETER_NAMES, backend=kwargs["backend"])
    jax.block_until_ready(updated.theta)
    elapsed = time.perf_counter() - started
    device_memory = observe_backend_device_memory(kwargs["backend"])
    scientific = {
        "parent_ids": record["parent_ids"],
        "microbatch_parent_counts": record["microbatch_parent_counts"],
        "microbatch_weights": record["microbatch_weights"],
        "loss": backend_numeric_record(record["loss"], "loss_gradient"),
        "gradient": backend_numeric_record(record["gradient"], "loss_gradient"),
        "gradient_norm": backend_numeric_record(record["gradient_norm"], "loss_gradient"),
        "adam_count": updated.adam.count,
    }
    return backend_throughput_evidence_payload(
        backend=kwargs["backend"],
        phase=kwargs["phase"],
        sample_index=kwargs["sample_index"],
        provenance=kwargs["provenance"],
        runtime=kwargs["runtime"],
        resource=kwargs["resource"],
        scientific=scientific,
        elapsed_seconds=elapsed,
        device_memory=device_memory,
    )


def run_backend_checkpoint_segment(**kwargs: Any) -> dict[str, Any]:
    """Reuse M8 checkpoints while keeping Backend-Evidence provenance external."""
    result = run_bounded_m8_segment(
        Path(kwargs["repository_root"]),
        kwargs["config_sha256"],
        Path(kwargs["output_dir"]),
        kwargs["max_updates_this_process"],
        resume_checkpoint=kwargs.get("resume_checkpoint"),
    )
    return {"status": "PASS_BACKEND_CHECKPOINT_SEGMENT", "m8_segment": result}
