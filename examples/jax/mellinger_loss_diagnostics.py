"""Generate Stage-2 loss-term diagnostics for the fixed H20 infrastructure smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crazyflow.control.mellinger import (
    GAIN_VARIABLE_NAMES,
    GAIN_VARIABLE_SPACE,
    TRACKING_LOSS_TERM_NAMES,
    TrackingLossConfig,
    physical_gains,
    tracking_loss_diagnostics_per_case,
)
from examples.jax.mellinger_batch_diagnostics import CASE_NAMES, build_experiment

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "crazyflow.mellinger_loss_diagnostics.v1"
SOURCE_BASE_COMMIT = "bfce9fcf04f97a609f569a6c9d95ea4f328a1cee"
WORK_ORDER = "WO-GR-S2-001"
CONTROL_FREQUENCY_HZ = 100
SIMULATION_FREQUENCY_HZ = 500
REQUIRED_HORIZON = 20
REQUIRED_SEED = 20260724
CASE_SPLITS = ("train", "validation")
REFERENCE_TOTAL_LOSSES = (0.0022911163978278637, 0.0011237584985792637)

LOSS_TERM_CONTRACT = {
    "position": {
        "raw_quantity": "mean_squared_position_error",
        "raw_unit": "m^2",
        "normalization": "divide_by_position_scale_squared",
    },
    "velocity": {
        "raw_quantity": "mean_squared_velocity_error",
        "raw_unit": "(m/s)^2",
        "normalization": "divide_by_velocity_scale_squared",
    },
    "effort": {
        "raw_quantity": "mean_squared_hover_normalized_commanded_rpm_deviation",
        "raw_unit": "1",
        "normalization": "identity",
    },
    "smoothness": {
        "raw_quantity": "mean_squared_hover_normalized_commanded_rpm_delta",
        "raw_unit": "1",
        "normalization": "identity",
    },
    "terminal": {
        "raw_quantity": "terminal_mean_squared_position_error",
        "raw_unit": "m^2",
        "normalization": "divide_by_position_scale_squared",
    },
    "altitude": {
        "raw_quantity": "mean_squared_softplus_altitude_margin_argument",
        "raw_unit": "1",
        "normalization": "identity",
    },
}


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse the deliberately narrow Stage-2 smoke CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=REQUIRED_HORIZON)
    parser.add_argument("--seed", type=int, default=REQUIRED_SEED)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/day20-stage2-loss-diagnostics")
    )
    return parser.parse_args(argv)


def _gain_dict(values: Any) -> dict[str, float]:
    return {name: float(getattr(values, name)) for name in GAIN_VARIABLE_NAMES}


def _physical_gain_dict(values: Any) -> dict[str, float]:
    gains = physical_gains(values)
    return {
        "kp_xy_n_per_m": float(gains.kp[0]),
        "kp_z_n_per_m": float(gains.kp[2]),
        "kd_xy_n_s_per_m": float(gains.kd[0]),
        "kd_z_n_s_per_m": float(gains.kd[2]),
    }


def _finite(value: Any) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree.leaves(value))


def _case_gradient_matrix(diagnostics: Any, case_index: int) -> list[list[float]]:
    return [
        [
            float(
                getattr(diagnostics.weighted_gradient_by_bound_variable, gain_name)[
                    case_index, term_index
                ]
            )
            for gain_name in GAIN_VARIABLE_NAMES
        ]
        for term_index in range(len(TRACKING_LOSS_TERM_NAMES))
    ]


def build_report(horizon: int, seed: int) -> dict[str, Any]:
    """Evaluate exactly the fixed H20 default-gain Figure-8/circle smoke."""
    if horizon != REQUIRED_HORIZON:
        raise ValueError(f"This diagnostic requires --horizon {REQUIRED_HORIZON}")
    if seed != REQUIRED_SEED:
        raise ValueError(f"This diagnostic requires --seed {REQUIRED_SEED}")

    batch = build_experiment(
        horizon, CONTROL_FREQUENCY_HZ, SIMULATION_FREQUENCY_HZ, seed, CASE_NAMES
    )
    config = TrackingLossConfig()
    evaluate = jax.jit(
        partial(
            tracking_loss_diagnostics_per_case,
            initial_data=batch.initial_data,
            commands=batch.commands,
            reference=batch.reference,
            step_fn=batch.step_fn,
            steps_per_command=batch.steps_per_command,
            config=config,
        )
    )
    diagnostics = evaluate(batch.variables)
    jax.block_until_ready(diagnostics)

    cases = []
    contribution_plot_series = []
    gradient_plot_panels = []
    contribution_differences = []
    for case_index, (trajectory_type, split) in enumerate(
        zip(CASE_NAMES, CASE_SPLITS, strict=True)
    ):
        terms = {}
        contributions = []
        for term_index, term_name in enumerate(TRACKING_LOSS_TERM_NAMES):
            gradient = {
                gain_name: float(
                    getattr(diagnostics.weighted_gradient_by_bound_variable, gain_name)[
                        case_index, term_index
                    ]
                )
                for gain_name in GAIN_VARIABLE_NAMES
            }
            contribution = float(diagnostics.terms.weighted[term_name][case_index])
            contributions.append(contribution)
            terms[term_name] = {
                **LOSS_TERM_CONTRACT[term_name],
                "raw_value": float(diagnostics.terms.raw[term_name][case_index]),
                "normalization_divisor": float(diagnostics.terms.normalization_divisor[term_name]),
                "normalized_value": float(diagnostics.terms.normalized[term_name][case_index]),
                "weight": float(diagnostics.terms.weight[term_name]),
                "weighted_contribution": contribution,
                "weighted_contribution_gradient_by_bound_variable": gradient,
            }
        total_loss = float(diagnostics.total_loss[case_index])
        contribution_sum = float(sum(contributions))
        contribution_difference = contribution_sum - total_loss
        contribution_differences.append(contribution_difference)
        technical_metrics = {
            name: float(values[case_index]) for name, values in diagnostics.per_case_metrics.items()
        }
        cases.append(
            {
                "case_index": case_index,
                "trajectory_type": trajectory_type,
                "split": split,
                "loss_total": total_loss,
                "weighted_contribution_sum": contribution_sum,
                "weighted_contribution_sum_minus_total": contribution_difference,
                "loss_terms": terms,
                "technical_metrics": technical_metrics,
            }
        )
        contribution_plot_series.append({"split": split, "values": contributions})
        gradient_plot_panels.append(
            {"split": split, "values_term_by_gain": _case_gradient_matrix(diagnostics, case_index)}
        )

    total_losses = np.asarray([case["loss_total"] for case in cases])
    technical_gate_keys = (
        "motor_saturation_fraction",
        "floor_clip_fraction",
        "zero_thrust_gate_fraction",
        "nonfinite_state_fraction",
    )
    validations = {
        "exactly_six_terms": len(TRACKING_LOSS_TERM_NAMES) == 6
        and all(len(case["loss_terms"]) == 6 for case in cases),
        "exactly_four_bound_gain_variables": len(GAIN_VARIABLE_NAMES) == 4,
        "finite_losses_terms_metrics_gradients": _finite(diagnostics),
        "weighted_contributions_reconstruct_total": all(
            np.isclose(difference, 0.0, rtol=1.0e-6, atol=1.0e-8)
            for difference in contribution_differences
        ),
        "unchanged_h20_reference_total_losses": bool(
            np.allclose(total_losses, REFERENCE_TOTAL_LOSSES, rtol=1.0e-6, atol=1.0e-8)
        ),
        "technical_smoke_gates_clear": all(
            case["technical_metrics"][name] == 0.0 for case in cases for name in technical_gate_keys
        ),
        "plot_data_complete": len(contribution_plot_series) == 2 and len(gradient_plot_panels) == 2,
        "zero_optimizer_updates": True,
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "work_order": WORK_ORDER,
            "source_base_commit": SOURCE_BASE_COMMIT,
            "generator": "examples/jax/mellinger_loss_diagnostics.py",
            "diagnostic_api": (
                "crazyflow.control.mellinger.loss_diagnostics.tracking_loss_diagnostics_per_case"
            ),
        },
        "claim_boundary": {
            "classification": "infrastructure_smoke",
            "optimizer_updates": 0,
            "evaluated_gain_state": "crazyflow_default",
            "supports_convergence_claim": False,
            "supports_generalization_claim": False,
            "supports_hardware_claim": False,
            "supports_controller_superiority_claim": False,
        },
        "experiment": {
            "platform": "cf2x_L250",
            "seed": seed,
            "simulation_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "steps_per_command": SIMULATION_FREQUENCY_HZ // CONTROL_FREQUENCY_HZ,
            "horizon_control_intervals": horizon,
            "duration_seconds": horizon / CONTROL_FREQUENCY_HZ,
            "case_order": [
                {"trajectory_type": name, "split": split}
                for name, split in zip(CASE_NAMES, CASE_SPLITS, strict=True)
            ],
        },
        "gains": {
            "gradient_variable_space": GAIN_VARIABLE_SPACE,
            "gradient_variable_unit": "1",
            "bound_variable_names": list(GAIN_VARIABLE_NAMES),
            "bound_variables": _gain_dict(batch.variables),
            "physical_default_gains": _physical_gain_dict(batch.variables),
        },
        "loss_contract": {
            "term_order": list(TRACKING_LOSS_TERM_NAMES),
            "weighted_contribution_unit": "1",
            "position_scale_m": config.position_scale,
            "velocity_scale_m_per_s": config.velocity_scale,
            "altitude_margin_m": config.altitude_margin,
            "altitude_softness_m": config.altitude_softness,
            "terms": {
                name: {
                    **LOSS_TERM_CONTRACT[name],
                    "normalization_divisor": float(diagnostics.terms.normalization_divisor[name]),
                    "weight": float(diagnostics.terms.weight[name]),
                }
                for name in TRACKING_LOSS_TERM_NAMES
            },
        },
        "cases": cases,
        "plot_data": {
            "loss_contributions": {
                "term_names": list(TRACKING_LOSS_TERM_NAMES),
                "series": contribution_plot_series,
            },
            "loss_term_gain_gradients": {
                "term_names": list(TRACKING_LOSS_TERM_NAMES),
                "gain_variable_names": list(GAIN_VARIABLE_NAMES),
                "variable_space": GAIN_VARIABLE_SPACE,
                "panels": gradient_plot_panels,
            },
        },
        "validations": validations,
        "all_validations_passed": all(validations.values()),
    }
    return report


def _save_loss_contributions(report: dict[str, Any], path: Path) -> None:
    plot_data = report["plot_data"]["loss_contributions"]
    names = plot_data["term_names"]
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    width = 0.36
    for index, series in enumerate(plot_data["series"]):
        offset = (index - 0.5) * width
        axis.bar(x + offset, series["values"], width=width, label=series["split"])
    axis.set(
        title="H20 default-gain loss contributions (infrastructure smoke)",
        xlabel="loss term",
        ylabel="weighted contribution [1]",
        xticks=x,
        xticklabels=names,
    )
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)


def _save_loss_term_gain_gradients(report: dict[str, Any], path: Path) -> None:
    plot_data = report["plot_data"]["loss_term_gain_gradients"]
    matrices = [np.asarray(panel["values_term_by_gain"]) for panel in plot_data["panels"]]
    maximum = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    maximum = maximum if maximum > 0.0 else 1.0
    figure, axes = plt.subplots(1, len(matrices), figsize=(12, 5.5), constrained_layout=True)
    image = None
    for axis, panel, matrix in zip(axes, plot_data["panels"], matrices, strict=True):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-maximum, vmax=maximum, aspect="auto")
        axis.set(
            title=panel["split"],
            xlabel="bound gain variable",
            ylabel="loss term",
            xticks=np.arange(len(plot_data["gain_variable_names"])),
            xticklabels=plot_data["gain_variable_names"],
            yticks=np.arange(len(plot_data["term_names"])),
            yticklabels=plot_data["term_names"],
        )
    assert image is not None
    figure.colorbar(image, ax=axes, label="d(weighted contribution)/d(bound variable)")
    figure.suptitle("H20 term-wise Stage-1 gain gradients")
    figure.savefig(path, dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifacts(report: dict[str, Any], output_dir: Path) -> tuple[Path, ...]:
    """Write the JSON, two JSON-driven plots, and their checksum index."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "loss_diagnostics.json"
    contribution_path = output_dir / "loss_contributions.png"
    gradient_path = output_dir / "loss_term_gain_gradients.png"
    checksum_path = output_dir / "SHA256SUMS"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    plot_report = json.loads(json_path.read_text())
    _save_loss_contributions(plot_report, contribution_path)
    _save_loss_term_gain_gradients(plot_report, gradient_path)
    indexed = (json_path, contribution_path, gradient_path)
    checksum_path.write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in indexed))
    return (*indexed, checksum_path)


def main(args: argparse.Namespace) -> None:
    """Run the fixed smoke and persist exactly four deterministic files."""
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    report = build_report(args.horizon, args.seed)
    if not report["all_validations_passed"]:
        raise RuntimeError(f"Stage-2 validation failed: {report['validations']}")
    paths = write_artifacts(report, output_dir)
    print(f"train_loss={report['cases'][0]['loss_total']:.10f}")
    print(f"validation_loss={report['cases'][1]['loss_total']:.10f}")
    print(f"validations={report['validations']}")
    print("artifacts=" + ",".join(str(path) for path in paths))


if __name__ == "__main__":
    main(parse_args())
