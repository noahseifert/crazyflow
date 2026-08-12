"""Train-only Optax updates for differentiable Mellinger gain research."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import jax.numpy as jnp
import optax

from crazyflow.control.mellinger.tracking import (
    GainVariables,
    TrackingLossConfig,
    aggregate_tracking_metrics,
    tracking_objective_per_case,
)

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData
    from crazyflow.trajectory import Trajectory


TRAIN_INDEX = 0
VALIDATION_INDEX = 1
TRAIN_MASK = jnp.asarray((1.0, 0.0), dtype=jnp.float32)
VALIDATION_MASK = jnp.asarray((0.0, 1.0), dtype=jnp.float32)
COMBINED_MASK = jnp.asarray((1.0, 1.0), dtype=jnp.float32)


def train_tracking_objective(
    variables: GainVariables,
    initial_data: SimData,
    commands: Array,
    reference: Trajectory,
    *,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
    config: TrackingLossConfig,
) -> tuple[Array, dict[str, Any]]:
    """Return the Figure-8/world-0 loss and diagnostic auxiliary values.

    Indexing the per-world loss before differentiation makes the train-only
    dependency explicit: circle/world 1 is returned as auxiliary evidence but
    cannot contribute to the scalar differentiated by ``value_and_grad``.
    """
    per_case_loss, per_case_metrics = tracking_objective_per_case(
        variables, initial_data, commands, reference, step_fn, steps_per_command, config
    )
    return per_case_loss[TRAIN_INDEX], {
        "per_case_loss": per_case_loss,
        "per_case_metrics": per_case_metrics,
        "train_metrics": aggregate_tracking_metrics(per_case_metrics, TRAIN_MASK),
    }


def evaluate_tracking_objectives(
    variables: GainVariables,
    initial_data: SimData,
    commands: Array,
    reference: Trajectory,
    *,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
    config: TrackingLossConfig,
) -> dict[str, Any]:
    """Evaluate train, validation, and equal-weight combined diagnostics."""
    per_case_loss, per_case_metrics = tracking_objective_per_case(
        variables, initial_data, commands, reference, step_fn, steps_per_command, config
    )
    case_count = per_case_loss.shape[0]
    if case_count == 1:
        single_mask = jnp.ones((1,), dtype=per_case_loss.dtype)
        single_metrics = aggregate_tracking_metrics(per_case_metrics, single_mask)
        return {
            "train_loss": per_case_loss[0],
            "validation_loss": per_case_loss[0],
            "combined_loss": per_case_loss[0],
            "per_case_loss": per_case_loss,
            "per_case_metrics": per_case_metrics,
            "train_metrics": single_metrics,
            "validation_metrics": single_metrics,
            "combined_metrics": single_metrics,
        }
    if case_count != 2:
        raise ValueError(f"Expected one or two evaluation cases, got {case_count}")
    return {
        "train_loss": per_case_loss[TRAIN_INDEX],
        "validation_loss": per_case_loss[VALIDATION_INDEX],
        "combined_loss": jnp.mean(per_case_loss),
        "per_case_loss": per_case_loss,
        "per_case_metrics": per_case_metrics,
        "train_metrics": aggregate_tracking_metrics(per_case_metrics, TRAIN_MASK),
        "validation_metrics": aggregate_tracking_metrics(per_case_metrics, VALIDATION_MASK),
        "combined_metrics": aggregate_tracking_metrics(per_case_metrics, COMBINED_MASK),
    }


def adam_optimization_step(
    variables: GainVariables,
    opt_state: optax.OptState,
    initial_data: SimData,
    commands: Array,
    reference: Trajectory,
    *,
    optimizer: optax.GradientTransformation,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
    config: TrackingLossConfig,
) -> tuple[GainVariables, optax.OptState, Array, dict[str, Any], GainVariables]:
    """Apply one Optax update using only the train objective and gradient."""

    def objective(candidate: GainVariables) -> tuple[Array, dict[str, Any]]:
        return train_tracking_objective(
            candidate,
            initial_data,
            commands,
            reference,
            step_fn=step_fn,
            steps_per_command=steps_per_command,
            config=config,
        )

    (train_loss, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(variables)
    updates, new_opt_state = optimizer.update(gradient, opt_state, variables)
    new_variables = optax.apply_updates(variables, updates)
    return new_variables, new_opt_state, train_loss, auxiliary, gradient


def with_controller_mass(data: SimData, mass_kg: float | Array) -> SimData:
    """Return experiment-local ``SimData`` with only controller mass replaced."""
    state_control = data.controls.state
    if state_control is None:
        raise ValueError("Controller-mass replacement requires Control.state")
    mass = jnp.asarray(mass_kg, dtype=state_control.params["mass"].dtype)
    params = state_control.params | {"mass": mass}
    controls = data.controls.replace(state=state_control.replace(params=params))
    return data.replace(controls=controls)


def tree_all_finite(tree: Any) -> bool:
    """Return whether every array leaf is finite."""
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(tree))


def tree_l2_norm(tree: Any) -> Array:
    """Return the Euclidean norm over all PyTree leaves."""
    return jnp.sqrt(sum(jnp.vdot(leaf, leaf) for leaf in jax.tree.leaves(tree)))


def select_best_train_checkpoint(history: Sequence[dict[str, Any]]) -> int:
    """Select minimum train loss, resolving exact ties toward the earliest step."""
    if not history:
        raise ValueError("Optimization history must not be empty")
    return min(
        range(len(history)),
        key=lambda index: (float(history[index]["train_loss"]), int(history[index]["step"])),
    )
