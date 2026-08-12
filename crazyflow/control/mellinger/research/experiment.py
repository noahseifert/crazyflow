"""Leakage-resistant split pipelines and differentiable research rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass as pytree_dataclass
from flax.struct import field

import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.control.mellinger.research.config import (
    DomainRandomizationConfig,
    SimulationConfig,
    Split,
    SplitManifest,
    TrajectoryConfig,
    seed_key,
)
from crazyflow.control.mellinger.research.gains import apply_raw_gains
from crazyflow.control.mellinger.research.randomization import (
    apply_action_delay,
    apply_true_mass,
    sample_correlated_wrenches,
    sample_delay_steps,
    sample_world_masses,
)
from crazyflow.control.mellinger.research.trajectories import (
    generate_trajectory,
    stack_trajectories,
)
from crazyflow.control.mellinger.tracking import (
    RolloutTrace,
    TrackingLossConfig,
    hover_rotor_velocity,
    initialize_tracking_state,
    rotor_velocity_limits,
    tracking_loss_per_case,
)
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.trajectory import Trajectory, state_commands

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData


@dataclass(frozen=True)
class SplitPipeline:
    """One simulator and pure pipeline dedicated to exactly one split."""

    split: Split
    simulation_config: SimulationConfig
    sim: Sim
    step_fn: Callable[[SimData, int], SimData]


@dataclass(frozen=True)
class TrainingPipelines:
    """Training-visible pipelines; intentionally has no test attribute."""

    train: SplitPipeline
    validation: SplitPipeline


@pytree_dataclass
class EpisodeBatch:
    """Fully realized, replayable inputs for one split episode batch."""

    split: Split = field(pytree_node=False)
    episode_ids: tuple[str, ...] = field(pytree_node=False)
    initial_data: SimData
    commands: Array
    full_reference: Trajectory
    reference: Trajectory
    force_n: Array
    torque_nm: Array
    masses_kg: Array
    delay_steps: Array


def build_split_pipeline(
    split: Split, simulation_config: SimulationConfig, rng_seed: int
) -> SplitPipeline:
    """Construct a fresh simulator whose mutable wrapper is never shared."""
    simulation_config.validate()
    sim = Sim(
        n_worlds=simulation_config.n_worlds,
        n_drones=simulation_config.n_drones,
        drone=simulation_config.drone,
        dynamics=Dynamics.first_principles,
        control=Control.state,
        freq=simulation_config.sim_freq_hz,
        state_freq=simulation_config.control_freq_hz,
        attitude_freq=simulation_config.sim_freq_hz,
        force_torque_freq=simulation_config.sim_freq_hz,
        device="cpu",
        rng_key=rng_seed,
    )
    return SplitPipeline(split, simulation_config, sim, sim.build_step_fn())


def build_training_pipelines(
    train_config: SimulationConfig, validation_config: SimulationConfig, root_seed: int
) -> TrainingPipelines:
    """Build separate training and validation simulators without test access."""
    return TrainingPipelines(
        train=build_split_pipeline(Split.TRAIN, train_config, root_seed + 101),
        validation=build_split_pipeline(Split.VALIDATION, validation_config, root_seed + 211),
    )


def build_test_pipeline(test_config: SimulationConfig, root_seed: int) -> SplitPipeline:
    """Build the frozen test pipeline only for an explicit test-only entry point."""
    return build_split_pipeline(Split.TEST, test_config, root_seed + 307)


def _component_keys(
    root_seeds: tuple[int, ...], split: Split, episode_index: int, component: str
) -> Array:
    return jnp.stack(
        tuple(
            seed_key(root_seed, split, episode_index, world, component)
            for world, root_seed in enumerate(root_seeds)
        )
    )


def _episode_seed_contract(
    pipeline: SplitPipeline, root_seed: int, episode_index: int, manifest: SplitManifest | None
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    worlds = pipeline.simulation_config.n_worlds
    if pipeline.split is Split.TRAIN:
        if manifest is not None:
            raise ValueError("training must resample from its root seed, not a fixed manifest")
        return (root_seed,) * worlds, tuple(
            f"train-{episode_index:06d}-world-{world:03d}" for world in range(worlds)
        )
    if manifest is None or manifest.split is not pipeline.split:
        raise ValueError(f"{pipeline.split.value} requires its matching fixed manifest")
    if len(manifest.episodes) < worlds:
        raise ValueError("manifest has fewer episodes than configured worlds")
    selected = manifest.episodes[:worlds]
    return tuple(item.seed for item in selected), tuple(item.episode_id for item in selected)


def build_episode_batch(
    pipeline: SplitPipeline,
    root_seed: int,
    episode_index: int,
    trajectory_config: TrajectoryConfig,
    randomization_config: DomainRandomizationConfig,
    manifest: SplitManifest | None = None,
) -> EpisodeBatch:
    """Realize all per-world samples once before the differentiated rollout."""
    simulation = pipeline.simulation_config
    randomization_config.validate()
    root_seeds, episode_ids = _episode_seed_contract(pipeline, root_seed, episode_index, manifest)
    key_sets = {
        component: _component_keys(root_seeds, pipeline.split, episode_index, component)
        for component in ("trajectory", "mass", "delay", "wrench")
    }
    generated = tuple(
        generate_trajectory(
            key_sets["trajectory"][world],
            simulation.horizon,
            simulation.control_freq_hz,
            trajectory_config,
        )
        for world in range(simulation.n_worlds)
    )
    full_reference = stack_trajectories(tuple(item.trajectory for item in generated))
    reference = jax.tree.map(lambda value: value[1:], full_reference)
    commands = jnp.stack(tuple(state_commands(item.trajectory)[1:] for item in generated), axis=1)[
        :, :, None, :
    ]
    commands = jnp.broadcast_to(
        commands, (simulation.horizon, simulation.n_worlds, simulation.n_drones, 13)
    )

    data = pipeline.sim.data
    nominal_mass = float(data.params.mass[0, 0, 0])
    half_width = (
        randomization_config.mass_half_width_kg if randomization_config.mass_enabled else 0.0
    )
    masses = sample_world_masses(key_sets["mass"], nominal_mass, half_width, simulation.n_drones)
    data = apply_true_mass(data, masses)
    data = initialize_tracking_state(data, full_reference.pos[0], full_reference.vel[0])

    delay_maximum = (
        randomization_config.delay_max_control_steps if randomization_config.delay_enabled else 0
    )
    delays = sample_delay_steps(key_sets["delay"], delay_maximum)
    commands = apply_action_delay(commands, delays)
    force_std = randomization_config.force_std_n if randomization_config.wrench_enabled else 0.0
    torque_std = randomization_config.torque_std_nm if randomization_config.wrench_enabled else 0.0
    force, torque = sample_correlated_wrenches(
        key_sets["wrench"],
        simulation.horizon,
        simulation.n_drones,
        1.0 / simulation.control_freq_hz,
        force_std,
        torque_std,
        randomization_config.wrench_correlation_time_s,
    )
    return EpisodeBatch(
        split=pipeline.split,
        episode_ids=episode_ids,
        initial_data=data,
        commands=commands,
        full_reference=full_reference,
        reference=reference,
        force_n=force,
        torque_nm=torque,
        masses_kg=masses,
        delay_steps=delays,
    )


def rollout_episode(
    raw_gains: Array,
    batch: EpisodeBatch,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
    gain_stage: int,
) -> tuple[SimData, RolloutTrace]:
    """Roll out one realized batch with one shared gain vector."""
    if steps_per_command < 1:
        raise ValueError("steps_per_command must be positive")
    data = apply_raw_gains(batch.initial_data, raw_gains, gain_stage)

    def step(data: SimData, inputs: tuple[Array, Array, Array]) -> tuple[SimData, RolloutTrace]:
        command, force, torque = inputs
        data = data.replace(states=data.states.replace(force=force, torque=torque))
        data = F.state_control(data, command)
        data = step_fn(data, steps_per_command)
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
        return data, trace

    return jax.lax.scan(step, data, (batch.commands, batch.force_n, batch.torque_nm))


def extended_metrics(
    trace: RolloutTrace, reference: Trajectory, base_metrics: dict[str, Array]
) -> dict[str, Array]:
    """Add per-world p95, terminal velocity, and technical success diagnostics."""
    pos_error = jnp.linalg.vector_norm(trace.pos - reference.pos, axis=-1)
    per_world_errors = jnp.transpose(pos_error, (1, 0, 2)).reshape(pos_error.shape[1], -1)
    terminal_position = jnp.mean(pos_error[-1], axis=1)
    terminal_velocity = jnp.mean(
        jnp.linalg.vector_norm(trace.vel[-1] - reference.vel[-1], axis=-1), axis=1
    )
    metrics = dict(base_metrics)
    metrics.update(
        {
            "p95_position_error_m": jnp.quantile(per_world_errors, 0.95, axis=1),
            "terminal_position_error_m": terminal_position,
            "terminal_velocity_error_m_s": terminal_velocity,
        }
    )
    # Technical smoke definition only; it is not a hardware-safe success threshold.
    metrics["episode_success"] = (
        (metrics["nonfinite_state_fraction"] == 0.0)
        & (metrics["max_position_error_m"] <= 0.25)
        & (metrics["floor_clip_fraction"] == 0.0)
        & (metrics["motor_saturation_fraction"] < 0.5)
    )
    metrics["episode_failure"] = ~metrics["episode_success"]
    return metrics


def research_objective(
    raw_gains: Array,
    batch: EpisodeBatch,
    *,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
    gain_stage: int,
    loss_config: TrackingLossConfig | None = None,
) -> tuple[Array, dict[str, Any]]:
    """Return mean training/validation loss without accessing any other split."""
    loss_config = TrackingLossConfig() if loss_config is None else loss_config
    _, trace = rollout_episode(raw_gains, batch, step_fn, steps_per_command, gain_stage)
    per_case_loss, base_metrics = tracking_loss_per_case(
        trace,
        batch.reference,
        hover_rotor_velocity(batch.initial_data),
        rotor_velocity_limits(batch.initial_data),
        loss_config,
    )
    per_case_metrics = extended_metrics(trace, batch.reference, base_metrics)
    return jnp.mean(per_case_loss), {
        "per_case_loss": per_case_loss,
        "per_case_metrics": per_case_metrics,
        "mean_metrics": {
            name: jnp.mean(value.astype(jnp.float32)) for name, value in per_case_metrics.items()
        },
    }
