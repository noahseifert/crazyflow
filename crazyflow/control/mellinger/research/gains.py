"""Named, staged Mellinger gain registry for experiment-local candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import ControlData, SimData


@dataclass(frozen=True)
class GainSpec:
    name: str
    controller: Literal["state", "attitude"]
    parameter: str
    axes: tuple[int, ...]
    unit: str
    lower: float
    upper: float
    stage: int | None
    decision: Literal["optimize", "defer", "exclude"]
    rationale: str


GAIN_REGISTRY = (
    GainSpec(
        "kp_xy",
        "state",
        "kp",
        (0, 1),
        "N/m",
        0.10,
        1.20,
        1,
        "optimize",
        "proven gradient; preserve x/y symmetry",
    ),
    GainSpec("kp_z", "state", "kp", (2,), "N/m", 0.30, 2.50, 1, "optimize", "proven gradient"),
    GainSpec(
        "kd_xy",
        "state",
        "kd",
        (0, 1),
        "N s/m",
        0.05,
        0.80,
        1,
        "optimize",
        "proven gradient; preserve x/y symmetry",
    ),
    GainSpec("kd_z", "state", "kd", (2,), "N s/m", 0.10, 1.20, 1, "optimize", "proven gradient"),
    GainSpec(
        "ki_xy",
        "state",
        "ki",
        (0, 1),
        "N/(m s)",
        0.025,
        0.10,
        2,
        "optimize",
        "longer/bias episodes required",
    ),
    GainSpec(
        "ki_z",
        "state",
        "ki",
        (2,),
        "N/(m s)",
        0.025,
        0.10,
        2,
        "optimize",
        "longer/bias episodes required",
    ),
    GainSpec(
        "kR_xy",
        "attitude",
        "kR",
        (0, 1),
        "legacy PWM/rad",
        35000.0,
        140000.0,
        3,
        "optimize",
        "attitude stage; legacy firmware-domain units",
    ),
    GainSpec(
        "kR_z",
        "attitude",
        "kR",
        (2,),
        "legacy PWM/rad",
        30000.0,
        120000.0,
        4,
        "optimize",
        "yaw excitation and metric required",
    ),
    GainSpec(
        "kw_xy",
        "attitude",
        "kw",
        (0, 1),
        "legacy PWM s/rad",
        10000.0,
        40000.0,
        3,
        "optimize",
        "attitude-rate stage",
    ),
    GainSpec(
        "kw_z",
        "attitude",
        "kw",
        (2,),
        "legacy PWM s/rad",
        6000.0,
        24000.0,
        4,
        "optimize",
        "yaw excitation and metric required",
    ),
    GainSpec(
        "ki_m_z",
        "attitude",
        "ki_m",
        (2,),
        "legacy PWM/(rad s)",
        250.0,
        1000.0,
        4,
        "optimize",
        "nonzero default; yaw excitation required",
    ),
    GainSpec(
        "kd_omega_xy",
        "attitude",
        "kd_omega",
        (0, 1),
        "legacy PWM s2/rad",
        100.0,
        400.0,
        3,
        "optimize",
        "executed roll/pitch derivative path",
    ),
    GainSpec(
        "ki_m_xy",
        "attitude",
        "ki_m",
        (0, 1),
        "legacy PWM/(rad s)",
        0.0,
        1.0,
        None,
        "defer",
        "zero default and no evidenced nonzero initialization/bounds",
    ),
    GainSpec(
        "kd_omega_z",
        "attitude",
        "kd_omega",
        (2,),
        "legacy PWM s2/rad",
        0.0,
        1.0,
        None,
        "exclude",
        "yaw derivative error is explicitly zeroed in controller",
    ),
)


def specs_for_stage(stage: int) -> tuple[GainSpec, ...]:
    """Return all optimizable specs introduced up through ``stage``."""
    if stage not in {1, 2, 3, 4}:
        raise ValueError("stage must be in {1, 2, 3, 4}")
    return tuple(
        spec
        for spec in GAIN_REGISTRY
        if spec.decision == "optimize" and spec.stage is not None and spec.stage <= stage
    )


def registry_fingerprint() -> str:
    """Return a stable fingerprint of registry metadata."""
    from crazyflow.control.mellinger.research.config import fingerprint

    return fingerprint(GAIN_REGISTRY)


def _controller_data(data: SimData, spec: GainSpec) -> ControlData:
    controller = getattr(data.controls, spec.controller)
    if controller is None:
        raise ValueError(f"{spec.controller} controller is unavailable")
    return controller


def default_physical_vector(data: SimData, stage: int) -> Array:
    """Extract named defaults into a shared one-dimensional vector."""
    values = []
    for spec in specs_for_stage(stage):
        parameter = _controller_data(data, spec).params[spec.parameter]
        selected = parameter[jnp.asarray(spec.axes)]
        if not bool(jnp.allclose(selected, selected[0])):
            raise ValueError(f"{spec.name} axes are not symmetric")
        values.append(selected[0])
    return jnp.stack(values)


def _inverse_bounded(value: Array, spec: GainSpec) -> Array:
    unit = (value - spec.lower) / (spec.upper - spec.lower)
    unit = jnp.clip(unit, 1.0e-6, 1.0 - 1.0e-6)
    return jnp.log(unit) - jnp.log1p(-unit)


def raw_from_data(data: SimData, stage: int) -> Array:
    """Build a bounded-transform raw vector from current controller values."""
    physical = default_physical_vector(data, stage)
    return jnp.stack(
        tuple(
            _inverse_bounded(value, spec)
            for value, spec in zip(physical, specs_for_stage(stage), strict=True)
        )
    )


def physical_from_raw(raw: Array, stage: int) -> dict[str, Array]:
    """Map the shared raw vector to named physical gain scalars."""
    specs = specs_for_stage(stage)
    raw = jnp.asarray(raw)
    if raw.shape != (len(specs),):
        raise ValueError(f"expected raw gain vector shape {(len(specs),)}, got {raw.shape}")
    return {
        spec.name: spec.lower + (spec.upper - spec.lower) * jax.nn.sigmoid(value)
        for spec, value in zip(specs, raw, strict=True)
    }


def apply_raw_gains(data: SimData, raw: Array, stage: int) -> SimData:
    """Apply one shared gain vector without mutating Crazyflow defaults."""
    physical = physical_from_raw(raw, stage)
    controllers = data.controls
    for spec in specs_for_stage(stage):
        controller = getattr(controllers, spec.controller)
        if controller is None:
            raise ValueError(f"{spec.controller} controller is unavailable")
        parameter = controller.params[spec.parameter]
        parameter = parameter.at[jnp.asarray(spec.axes)].set(physical[spec.name])
        controller = controller.replace(params=controller.params | {spec.parameter: parameter})
        controllers = controllers.replace(**{spec.controller: controller})
    return data.replace(controls=controllers)
