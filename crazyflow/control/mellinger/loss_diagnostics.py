"""Term-resolved gradients for the Stage-1 Mellinger tracking objective."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from crazyflow.control.mellinger.tracking import (
    TRACKING_LOSS_TERM_NAMES,
    GainVariables,
    TrackingLossConfig,
    TrackingLossTerms,
    apply_gain_variables,
    hover_rotor_velocity,
    rollout_state_commands,
    rotor_velocity_limits,
    tracking_loss_per_case,
    tracking_loss_terms_per_case,
)

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData
    from crazyflow.trajectory import Trajectory


GAIN_VARIABLE_NAMES = ("kp_xy", "kp_z", "kd_xy", "kd_z")
GAIN_VARIABLE_SPACE = "unconstrained_inputs_to_sigmoid_bounded_stage1_gains"


@dataclass
class TrackingLossDiagnostics:
    """Loss decomposition and term Jacobians for every rollout case."""

    total_loss: Array
    terms: TrackingLossTerms
    per_case_metrics: dict[str, Array]
    weighted_gradient_by_bound_variable: GainVariables


def tracking_loss_diagnostics_per_case(
    variables: GainVariables,
    initial_data: SimData,
    commands: Array,
    reference: Trajectory,
    step_fn: Callable[[SimData, int], SimData],
    steps_per_command: int,
    config: TrackingLossConfig,
) -> TrackingLossDiagnostics:
    """Differentiate each weighted loss contribution by each Stage-1 variable.

    Jacobian leaves have shape ``(n_cases, 6)``. Their last dimension follows
    ``TRACKING_LOSS_TERM_NAMES``. Gradients are with respect to the four
    dimensionless internal variables named by ``GAIN_VARIABLE_NAMES``; these
    variables feed the smooth bounded transform and are not physical gains.
    """
    hover_rpm = hover_rotor_velocity(initial_data)
    limits = rotor_velocity_limits(initial_data)

    def weighted_contributions_with_aux(
        candidate: GainVariables,
    ) -> tuple[Array, tuple[TrackingLossTerms, Array, dict[str, Array]]]:
        data = apply_gain_variables(initial_data, candidate)
        _, trace = rollout_state_commands(data, commands, step_fn, steps_per_command)
        terms = tracking_loss_terms_per_case(trace, reference, hover_rpm, config)
        total_loss, metrics = tracking_loss_per_case(trace, reference, hover_rpm, limits, config)
        contributions = jnp.stack(
            tuple(terms.weighted[name] for name in TRACKING_LOSS_TERM_NAMES), axis=-1
        )
        return contributions, (terms, total_loss, metrics)

    jacobian, (terms, total_loss, metrics) = jax.jacrev(
        weighted_contributions_with_aux, has_aux=True
    )(variables)
    return TrackingLossDiagnostics(total_loss, terms, metrics, jacobian)
