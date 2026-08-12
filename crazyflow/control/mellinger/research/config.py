"""Validated configuration and seed contracts for Mellinger research runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

import jax

if TYPE_CHECKING:
    from pathlib import Path


class Split(StrEnum):
    """Scientifically distinct experiment splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


SPLIT_NAMESPACE = {Split.TRAIN: 101, Split.VALIDATION: 211, Split.TEST: 307}
COMPONENT_NAMESPACE = {"trajectory": 401, "mass": 503, "delay": 601, "wrench": 701}
LEGACY_TRAJECTORY_DISTRIBUTION = "legacy_normalized_time_v1"
FIXED_SUPPORT_PREFIX_DISTRIBUTION = "fixed_support_prefix_v2"
TRAJECTORY_DISTRIBUTIONS = frozenset(
    {LEGACY_TRAJECTORY_DISTRIBUTION, FIXED_SUPPORT_PREFIX_DISTRIBUTION}
)


@dataclass(frozen=True)
class SimulationConfig:
    n_worlds: int = 2
    n_drones: int = 1
    horizon: int = 20
    sim_freq_hz: int = 500
    control_freq_hz: int = 100
    drone: str = "cf2x_L250"

    def validate(self) -> None:
        if self.n_worlds < 1 or self.n_drones < 1 or self.horizon < 2:
            raise ValueError("worlds/drones must be positive and horizon must be at least two")
        if self.control_freq_hz < 1 or self.sim_freq_hz % self.control_freq_hz:
            raise ValueError("sim_freq_hz must be divisible by positive control_freq_hz")


@dataclass(frozen=True)
class TrajectoryConfig:
    harmonics: int = 3
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.75)
    amplitude_m: tuple[float, float, float] = (0.18, 0.14, 0.08)
    workspace_min_m: tuple[float, float, float] = (-0.6, -0.6, 0.35)
    workspace_max_m: tuple[float, float, float] = (0.6, 0.6, 1.2)
    max_speed_m_s: float = 1.5
    max_acceleration_m_s2: float = 5.0
    max_jerk_m_s3: float = 35.0
    max_yaw_rate_rad_s: float = 1.0
    max_tilt_rad: float = 0.7
    min_specific_force_m_s2: float = 2.0
    max_attempts: int = 16
    distribution_version: str = LEGACY_TRAJECTORY_DISTRIBUTION
    support_duration_s: float | None = None
    minimum_duration_s: float | None = None

    def validate(self) -> None:
        if self.harmonics < 1 or self.max_attempts < 1:
            raise ValueError("harmonics and max_attempts must be positive")
        if any(a < 0.0 for a in self.amplitude_m):
            raise ValueError("trajectory amplitudes must be nonnegative")
        if any(lo >= hi for lo, hi in zip(self.workspace_min_m, self.workspace_max_m, strict=True)):
            raise ValueError("workspace minima must be below maxima")
        limits = (
            self.max_speed_m_s,
            self.max_acceleration_m_s2,
            self.max_jerk_m_s3,
            self.max_yaw_rate_rad_s,
            self.max_tilt_rad,
            self.min_specific_force_m_s2,
        )
        if any(value <= 0.0 for value in limits):
            raise ValueError("trajectory limits must be positive")
        if self.distribution_version not in TRAJECTORY_DISTRIBUTIONS:
            raise ValueError(
                "unsupported trajectory distribution_version: "
                f"{self.distribution_version!r}; expected one of {sorted(TRAJECTORY_DISTRIBUTIONS)}"
            )
        duration_values = (self.support_duration_s, self.minimum_duration_s)
        if self.distribution_version == LEGACY_TRAJECTORY_DISTRIBUTION:
            if any(value is not None for value in duration_values):
                raise ValueError(
                    "legacy_normalized_time_v1 does not accept support/minimum durations"
                )
        elif any(
            value is None or not math.isfinite(value) or value <= 0.0 for value in duration_values
        ):
            raise ValueError(
                "fixed_support_prefix_v2 requires finite positive support_duration_s and "
                "minimum_duration_s"
            )
        elif self.minimum_duration_s > self.support_duration_s:
            raise ValueError("minimum_duration_s must not exceed support_duration_s")


@dataclass(frozen=True)
class DomainRandomizationConfig:
    mass_enabled: bool = True
    mass_half_width_kg: float = 0.0002
    delay_enabled: bool = False
    delay_max_control_steps: int = 0
    wrench_enabled: bool = False
    force_std_n: float = 0.0
    torque_std_nm: float = 0.0
    wrench_correlation_time_s: float = 0.1

    def validate(self) -> None:
        if self.mass_half_width_kg < 0.0:
            raise ValueError("mass_half_width_kg must be nonnegative")
        if self.delay_max_control_steps < 0:
            raise ValueError("delay_max_control_steps must be nonnegative")
        if self.force_std_n < 0.0 or self.torque_std_nm < 0.0:
            raise ValueError("wrench standard deviations must be nonnegative")
        if self.wrench_correlation_time_s <= 0.0:
            raise ValueError("wrench_correlation_time_s must be positive")


@dataclass(frozen=True)
class OptimizerConfig:
    steps: int = 1
    learning_rate: float = 0.001
    gain_stage: int = 1
    checkpoint_interval: int = field(default=1, metadata={"canonical_omit_default": True})

    def validate(self) -> None:
        if self.steps < 1 or self.learning_rate <= 0.0:
            raise ValueError("optimizer steps and learning_rate must be positive")
        if self.gain_stage not in {1, 2, 3, 4}:
            raise ValueError("gain_stage must be in {1, 2, 3, 4}")
        if (
            isinstance(self.checkpoint_interval, bool)
            or not isinstance(self.checkpoint_interval, int)
            or self.checkpoint_interval < 1
        ):
            raise ValueError("checkpoint_interval must be a positive integer")


@dataclass(frozen=True)
class ResearchConfig:
    schema_version: str
    run_id: str
    root_seed: int
    train: SimulationConfig
    validation: SimulationConfig
    trajectory: TrajectoryConfig
    domain_randomization: DomainRandomizationConfig
    optimizer: OptimizerConfig
    validation_manifest: str
    test_manifest: str

    def validate(self) -> None:
        if self.schema_version != "crazyflow.mellinger_research_config.v1":
            raise ValueError(f"unsupported config schema: {self.schema_version}")
        if not self.run_id or self.root_seed < 0:
            raise ValueError("run_id must be nonempty and root_seed nonnegative")
        self.train.validate()
        self.validation.validate()
        self.trajectory.validate()
        self.domain_randomization.validate()
        self.optimizer.validate()


@dataclass(frozen=True)
class ManifestEpisode:
    episode_id: str
    seed: int
    difficulty: str = "technical"


@dataclass(frozen=True)
class SplitManifest:
    schema_version: str
    manifest_id: str
    split: Split
    episodes: tuple[ManifestEpisode, ...]

    def validate(self) -> None:
        if self.schema_version != "crazyflow.mellinger_split_manifest.v1":
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")
        if self.split is Split.TRAIN:
            raise ValueError("fixed manifests are reserved for validation and test")
        ids = [episode.episode_id for episode in self.episodes]
        seeds = [episode.seed for episode in self.episodes]
        if not ids or len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
            raise ValueError("manifest episode IDs and seeds must be nonempty and unique")


T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], values: dict[str, Any]) -> T:
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: Path) -> ResearchConfig:
    """Load and eagerly validate a research configuration."""
    raw = json.loads(path.read_text())
    raw["train"] = _dataclass_from_dict(SimulationConfig, raw["train"])
    raw["validation"] = _dataclass_from_dict(SimulationConfig, raw["validation"])
    raw["trajectory"] = _dataclass_from_dict(TrajectoryConfig, raw["trajectory"])
    raw["domain_randomization"] = _dataclass_from_dict(
        DomainRandomizationConfig, raw["domain_randomization"]
    )
    raw["optimizer"] = _dataclass_from_dict(OptimizerConfig, raw["optimizer"])
    config = _dataclass_from_dict(ResearchConfig, raw)
    config.validate()
    return config


def load_manifest(path: Path, expected_split: Split | None = None) -> SplitManifest:
    """Load a fixed split manifest and reject split confusion."""
    raw = json.loads(path.read_text())
    episodes = tuple(_dataclass_from_dict(ManifestEpisode, item) for item in raw.pop("episodes"))
    raw["split"] = Split(raw["split"])
    manifest = _dataclass_from_dict(SplitManifest, raw | {"episodes": episodes})
    manifest.validate()
    if expected_split is not None and manifest.split is not expected_split:
        raise ValueError(f"expected {expected_split.value} manifest, got {manifest.split.value}")
    return manifest


def canonical_dict(value: Any) -> Any:
    """Convert dataclass/enums into a stable JSON-compatible structure."""
    if hasattr(value, "__dataclass_fields__"):
        result = {}
        for item in fields(value):
            item_value = getattr(value, item.name)
            if item.metadata.get("canonical_omit_default") and item_value == item.default:
                continue
            result[item.name] = canonical_dict(item_value)
        return result
    if isinstance(value, dict):
        return {str(key): canonical_dict(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical_dict(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for a config-like value."""
    payload = json.dumps(canonical_dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def seed_key(root_seed: int, split: Split, episode: int, world: int, component: str) -> jax.Array:
    """Derive one explicit split/episode/world/component PRNG key."""
    if episode < 0 or world < 0 or component not in COMPONENT_NAMESPACE:
        raise ValueError("invalid seed-tree coordinate")
    key = jax.random.key(root_seed)
    for value in (SPLIT_NAMESPACE[split], episode, world, COMPONENT_NAMESPACE[component]):
        key = jax.random.fold_in(key, value)
    return key
