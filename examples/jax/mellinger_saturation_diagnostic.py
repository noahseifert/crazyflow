# ruff: noqa: E501 -- the required presentation caveat is intentionally one Markdown line.
"""Emit the frozen F0.1 rollout as a saturation-explicit diagnostic only.

This separate command preserves the unchanged F0.1 acceptance gate. It may
render the already frozen case only when every saturated motor command can be
attributed exactly to motor, sample, bound, and time interval.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crazyflow.control.mellinger import rotor_velocity_limits
from examples.jax import mellinger_friday_evidence as friday

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "97134b45cfce6efdf9915810ae01518a2f7b54c0"
BRANCH = "codex/wo-gr-f0-002"
WORKER = "WO-GR-F0-002-WORKER-01 /root/wo_gr_f0_002_worker"
WORKTREE = "/tmp/crazyflow-gradient-research-wo-gr-f0-002"
SCHEMA_VERSION = "crazyflow.saturation_diagnostic.v1"
STATUS = "DIAGNOSTIC_ONLY_SATURATION_PRESENT"
DEFAULT_STATUS = "NO_SATURATION_PRESENT"
GATE_FAILURE_STATUS = "FAIL_ZERO_MOTOR_SATURATION_GATE"
SATURATION_TOLERANCE = 1.0e-4
EXPECTED_CANDIDATE_FRACTION = 0.027499999850988388
DAY21_DIR = REPOSITORY_ROOT / "artifacts/day21-friday-evidence"
DAY21_INPUT_NAMES = (
    "PRESENTATION_HANDOFF.md",
    "friday_evidence.json",
    "h100_selected_end_gains.png",
    "h100_selected_gain_evolution.png",
    "provenance.json",
)
OUTPUT_NAMES = (
    "PRESENTATION_HANDOFF.md",
    "motor_saturation_detail.png",
    "provenance.json",
    "saturation_diagnostic.json",
    "trajectory_error_saturation.png",
)
INDEX_PATTERN = re.compile(r"([0-9a-f]{64})  (?:\./)?(.+)")
EXPECTED_CANDIDATE = {
    "label": friday.CANDIDATE_LABEL,
    "rule": "smallest existing selected validation loss among ten frozen H100 runs",
    "seed": 5,
    "selected_step": 5000,
    "selected_validation_loss": 0.002523899544030428,
    "selected_raw_gains": [
        -1.3688596487045288,
        2.3636748790740967,
        1.8979853391647339,
        2.1928365230560303,
    ],
    "physical_gains": {
        "kp_xy": 0.32308459281921387,
        "kp_z": 2.310833215713501,
        "kd_xy": 0.7022475004196167,
        "kd_z": 1.0895649194717407,
    },
}


class DiagnosticError(RuntimeError):
    """Raised before output when the frozen diagnostic contract is violated."""


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse the deliberately separate diagnostic-only command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-only-saturation-present",
        action="store_true",
        required=True,
        help="acknowledge that this path emits a failed-gate diagnostic, not acceptance evidence",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/day22-saturation-diagnostic")
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


def load_day21_candidate() -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    """Verify five accepted payload identities and the exact frozen candidate."""
    index_path = DAY21_DIR / "SHA256SUMS"
    index = _parse_checksum_index(index_path)
    identities = []
    for name in DAY21_INPUT_NAMES:
        expected = index.get(name)
        path = DAY21_DIR / name
        actual = _sha256_path(path)
        if expected is None or actual != expected:
            raise DiagnosticError(f"accepted Day-21 payload identity mismatch: {path}")
        identities.append({"path": path.relative_to(REPOSITORY_ROOT).as_posix(), "sha256": actual})

    report = _load_object(DAY21_DIR / "friday_evidence.json")
    if report.get("status") != "WITHHELD_TECHNICAL_GATE":
        raise DiagnosticError("Day-21 package does not preserve the withheld rollout status")
    if report.get("rollout") != {
        "status": "WITHHELD_TECHNICAL_GATE",
        "evidence_available": False,
        "missing_evidence": (
            "default-versus-provisional-candidate trajectory and tracking-error evidence"
        ),
        "acceptance_contract": "zero motor saturation for both compared controllers",
        "substitution_performed": False,
        "rollout_arrays_present": False,
        "rollout_figures_present": False,
    }:
        raise DiagnosticError("Day-21 normal withhold payload changed")
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise DiagnosticError("Day-21 candidate payload is absent")
    candidate_identity = {
        name: candidate.get(name)
        for name in (
            "label",
            "rule",
            "seed",
            "selected_step",
            "selected_validation_loss",
            "selected_raw_gains",
            "physical_gains",
        )
    }
    if candidate_identity != EXPECTED_CANDIDATE:
        raise DiagnosticError("Day-21 provisional candidate identity changed")
    provenance = _load_object(DAY21_DIR / "provenance.json")
    if (
        provenance.get("result_commit_generator_and_tests")
        != "2499aad364a44eadcc259005f4c18d8a21acb332"
    ):
        raise DiagnosticError("Day-21 generator identity changed")
    return candidate, identities, provenance


def saturation_decisions(
    commanded_rpm: np.ndarray, minimum_rpm: np.ndarray, maximum_rpm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the unchanged tracking-loss saturation formula."""
    commanded = np.asarray(commanded_rpm)
    minimum = np.broadcast_to(np.asarray(minimum_rpm), commanded.shape)
    maximum = np.broadcast_to(np.asarray(maximum_rpm), commanded.shape)
    lower = (commanded > 0.5 * minimum) & (commanded <= minimum * (1.0 + SATURATION_TOLERANCE))
    upper = commanded >= maximum * (1.0 - SATURATION_TOLERANCE)
    if np.any(lower & upper):
        raise DiagnosticError("one motor sample is attributed to both saturation bounds")
    return lower, upper


def _runs(mask: np.ndarray, bound: str) -> list[dict[str, Any]]:
    intervals = []
    index = 0
    while index < len(mask):
        if not bool(mask[index]):
            index += 1
            continue
        start = index
        while index + 1 < len(mask) and bool(mask[index + 1]):
            index += 1
        end = index
        count = end - start + 1
        intervals.append(
            {
                "bound": bound,
                "start_index_inclusive": start,
                "end_index_inclusive": end,
                "start_time_s": start / friday.CONTROL_FREQUENCY_HZ,
                "end_time_s": (end + 1) / friday.CONTROL_FREQUENCY_HZ,
                "sample_count": count,
                "duration_s": count / friday.CONTROL_FREQUENCY_HZ,
            }
        )
        index += 1
    return intervals


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _indices_and_times(mask: np.ndarray, state_times: np.ndarray) -> dict[str, Any]:
    indices = np.flatnonzero(mask).astype(int)
    return {"indices_zero_based": indices.tolist(), "state_times_s": state_times[indices].tolist()}


def attribute_saturation(
    commanded_rpm: np.ndarray,
    minimum_rpm: np.ndarray,
    maximum_rpm: np.ndarray,
    state_times: np.ndarray,
    metric_fraction: float,
) -> dict[str, Any]:
    """Attribute every saturation bool to one motor, bound, sample, and interval."""
    commanded = np.asarray(commanded_rpm)
    if commanded.shape != (friday.HORIZON, 1, 1, 4):
        raise DiagnosticError(f"unexpected commanded RPM shape: {commanded.shape}")
    if state_times.shape != (friday.HORIZON,):
        raise DiagnosticError(f"unexpected state-time shape: {state_times.shape}")
    minimum_full = np.broadcast_to(np.asarray(minimum_rpm), commanded.shape)
    maximum_full = np.broadcast_to(np.asarray(maximum_rpm), commanded.shape)
    lower_full, upper_full = saturation_decisions(commanded, minimum_full, maximum_full)
    commands = commanded[:, 0, 0, :]
    minimum = minimum_full[0, 0, 0, :]
    maximum = maximum_full[0, 0, 0, :]
    lower = lower_full[:, 0, 0, :]
    upper = upper_full[:, 0, 0, :]
    any_mask = lower | upper
    reconstructed_fraction = float(np.asarray(any_mask, dtype=np.float32).mean(dtype=np.float32))
    if not np.isclose(reconstructed_fraction, metric_fraction, rtol=0.0, atol=1.0e-9):
        raise DiagnosticError(
            "reconstructed saturation fraction differs from tracking_loss_per_case"
        )

    by_motor = []
    for motor in range(4):
        lower_mask = lower[:, motor]
        upper_mask = upper[:, motor]
        combined = any_mask[:, motor]
        intervals = _runs(lower_mask, "lower") + _runs(upper_mask, "upper")
        intervals.sort(key=lambda item: (item["start_index_inclusive"], item["bound"]))
        covered = np.zeros(friday.HORIZON, dtype=bool)
        covered_count = 0
        for interval in intervals:
            start = interval["start_index_inclusive"]
            end = interval["end_index_inclusive"]
            source = lower_mask if interval["bound"] == "lower" else upper_mask
            if np.any(covered[start : end + 1]) or not np.all(source[start : end + 1]):
                raise DiagnosticError("invalid or duplicate interval attribution")
            covered[start : end + 1] = True
            covered_count += int(interval["sample_count"])
        if covered_count != int(combined.sum()) or not np.array_equal(covered, combined):
            raise DiagnosticError("interval coverage does not equal saturation decisions")
        longest = _longest_true_run(combined)
        lower_count = int(lower_mask.sum())
        upper_count = int(upper_mask.sum())
        total_count = int(combined.sum())
        by_motor.append(
            {
                "motor_index": motor,
                "sample_count": friday.HORIZON,
                "lower_saturated_count": lower_count,
                "upper_saturated_count": upper_count,
                "saturated_count": total_count,
                "lower_saturated_fraction": lower_count / friday.HORIZON,
                "upper_saturated_fraction": upper_count / friday.HORIZON,
                "saturated_fraction": total_count / friday.HORIZON,
                "affected": {
                    "lower": _indices_and_times(lower_mask, state_times),
                    "upper": _indices_and_times(upper_mask, state_times),
                    "any": _indices_and_times(combined, state_times),
                },
                "intervals": intervals,
                "maximum_consecutive_saturated_samples": longest,
                "maximum_consecutive_saturated_duration_s": (longest / friday.CONTROL_FREQUENCY_HZ),
                "lower_upper_distinguished": True,
            }
        )

    total_count = int(any_mask.sum())
    if sum(record["saturated_count"] for record in by_motor) != total_count:
        raise DiagnosticError("per-motor saturation counts do not equal total count")
    return {
        "formula": {
            "tolerance": SATURATION_TOLERANCE,
            "lower": (
                "commanded_rpm > 0.5 * minimum_rpm and "
                "commanded_rpm <= minimum_rpm * (1 + tolerance)"
            ),
            "upper": "commanded_rpm >= maximum_rpm * (1 - tolerance)",
            "lower_upper_distinguished": True,
        },
        "limits_rpm": {
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "lower_positive_guard": (0.5 * minimum).tolist(),
            "lower_threshold": (minimum * (1.0 + SATURATION_TOLERANCE)).tolist(),
            "upper_threshold": (maximum * (1.0 - SATURATION_TOLERANCE)).tolist(),
        },
        "commanded_rotor_velocity_rpm": commands.tolist(),
        "decision_arrays": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "any": any_mask.tolist(),
        },
        "by_motor": by_motor,
        "total": {
            "sample_count_all_motors": int(any_mask.size),
            "lower_saturated_count": int(lower.sum()),
            "upper_saturated_count": int(upper.sum()),
            "saturated_count": total_count,
            "reconstructed_fraction_float32": reconstructed_fraction,
            "tracking_loss_metric_fraction": metric_fraction,
            "metric_match_atol_1e_9_rtol_0": True,
        },
    }


def _require_all_finite(label: str, value: Any) -> None:
    for leaf in jax.tree.leaves(value):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise DiagnosticError(f"nonfinite numeric value in {label}")


def build_diagnostic() -> dict[str, Any]:
    """Build the complete diagnostic in memory before any write."""
    candidate, day21_identities, day21_provenance = load_day21_candidate()
    rollout, traces, initial_data = friday.build_rollout_trace_bundle(candidate)
    _require_all_finite("rollout", rollout)
    _require_all_finite("default trace", traces["default"])
    _require_all_finite("Seed-05 trace", traces[friday.CANDIDATE_LABEL])

    expected_contract = {
        "platform": "cf2x_L250",
        "dynamics": "first_principles",
        "integrator": "explicit_euler",
        "simulation_frequency_hz": 500,
        "state_control_frequency_hz": 100,
        "simulation_steps_per_reference": 5,
        "horizon_control_intervals": 100,
        "duration_seconds": 1.0,
        "seed": 20260724,
        "trajectory": {
            "kind": "figure8",
            "center_m": [0.0, 0.0, 0.75],
            "amplitude_m": [0.3, 0.15],
            "period_seconds": 3.0,
            "ramp_duration_seconds": 0.5,
            "yaw_rad": 0.0,
        },
    }
    for key, expected in expected_contract.items():
        if rollout["contract"].get(key) != expected:
            raise DiagnosticError(f"frozen rollout contract changed: {key}")
    if rollout["candidate_label"] != friday.CANDIDATE_LABEL:
        raise DiagnosticError("frozen candidate label changed")

    state_times = np.asarray(rollout["state_time_s"])
    minimum_rpm, maximum_rpm = rotor_velocity_limits(initial_data)
    variants: dict[str, Any] = {}
    for label, status in (("default", DEFAULT_STATUS), (friday.CANDIDATE_LABEL, STATUS)):
        source = rollout["variants"][label]
        metric_fraction = float(source["metrics"]["motor_saturation_fraction"])
        saturation = attribute_saturation(
            np.asarray(traces[label].commanded_rotor_vel),
            np.asarray(minimum_rpm),
            np.asarray(maximum_rpm),
            state_times,
            metric_fraction,
        )
        variants[label] = {
            "status": status,
            "candidate_acceptance_gate_passed": metric_fraction == 0.0,
            "raw_gains": source["raw_gains"],
            "physical_gains": source["physical_gains"],
            "actual_position_m": source["actual_position_m"],
            "position_error_m": source["position_error_m"],
            "position_error_norm_m": source["position_error_norm_m"],
            "tracking_metrics": {
                "status": status,
                "position_rmse_m_direct": source["position_rmse_m_direct"],
                "max_position_error_m_direct": source["max_position_error_m_direct"],
                "unchanged_tracking_loss_metrics": source["metrics"],
                "claim_boundary": (
                    "descriptive values for this one saturated simulation case only; "
                    "not superiority, acceptance, flight readiness, or hardware transfer"
                ),
            },
            "technical_gates": source["technical_gates"],
            "saturation": saturation,
        }

    default_fraction = variants["default"]["saturation"]["total"]["tracking_loss_metric_fraction"]
    candidate_fraction = variants[friday.CANDIDATE_LABEL]["saturation"]["total"][
        "tracking_loss_metric_fraction"
    ]
    if default_fraction != 0.0:
        raise DiagnosticError("default no longer has zero motor saturation")
    if not np.isclose(candidate_fraction, EXPECTED_CANDIDATE_FRACTION, rtol=0.0, atol=1.0e-9):
        raise DiagnosticError("Seed-05 historical saturation fraction changed")
    if rollout["equality_and_technical_gates"]["candidate_all_technical_gates_pass"]:
        raise DiagnosticError("normal F0.1 zero-saturation gate unexpectedly passes")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "work_order": "WO-GR-F0-002",
        "purpose": (
            "diagnostic-only exposure of the unchanged F0.1 rollout after preserving its "
            "zero-saturation acceptance failure"
        ),
        "source_identity": {
            "accepted_day21_payloads_verified": day21_identities,
            "accepted_day21_checksum_index": {
                "path": "artifacts/day21-friday-evidence/SHA256SUMS",
                "sha256": _sha256_path(DAY21_DIR / "SHA256SUMS"),
            },
            "accepted_day21_generator_commit": day21_provenance[
                "result_commit_generator_and_tests"
            ],
            "candidate": EXPECTED_CANDIDATE,
            "five_payload_identities_verified": len(day21_identities) == 5,
        },
        "frozen_rollout_contract": rollout["contract"],
        "reference_time_s": rollout["reference_time_s"],
        "state_time_s": rollout["state_time_s"],
        "reference_position_m": rollout["reference_position_m"],
        "target_position_m": rollout["target_position_m"],
        "variants": variants,
        "normal_acceptance_gate": {
            "status": GATE_FAILURE_STATUS,
            "criterion": "motor_saturation_fraction == 0.0 for both variants",
            "criterion_unchanged": True,
            "default_motor_saturation_fraction": default_fraction,
            "seed05_motor_saturation_fraction": candidate_fraction,
            "seed05_status": STATUS,
            "accepted_candidate": False,
            "flight_ready": False,
        },
        "descriptive_comparison": {
            "status": STATUS,
            **rollout["comparison"],
            "claim_boundary": (
                "descriptive for the single frozen simulation only and always subordinate "
                "to the failed zero-saturation gate"
            ),
        },
        "validations": {
            "all_states_commands_metrics_and_plot_arrays_finite": True,
            "frozen_contract_match": True,
            "day21_candidate_match": True,
            "default_fraction_exactly_zero": True,
            "seed05_fraction_matches_historical_atol_1e_9_rtol_0": True,
            "every_saturation_attributed_once": True,
            "test_opened": False,
            "training_runs_started": 0,
        },
        "claim_boundary": (
            "simulation diagnostic only; the saturation is a hard FAIL for candidate and flight "
            "acceptance; no superiority, Test, cf21B_500, firmware, hardware, flight, or "
            "Sim2Real claim"
        ),
    }
    _require_all_finite("diagnostic report", report)
    return report


def _candidate_intervals(report: dict[str, Any]) -> list[tuple[float, float]]:
    intervals = []
    candidate = report["variants"][friday.CANDIDATE_LABEL]
    for motor in candidate["saturation"]["by_motor"]:
        for interval in motor["intervals"]:
            intervals.append((interval["start_time_s"], interval["end_time_s"]))
    return sorted(set(intervals))


def _figure_bytes(figure: Any) -> bytes:
    payload = io.BytesIO()
    figure.savefig(payload, format="png", dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)
    return payload.getvalue()


def render_trajectory_figure(report: dict[str, Any]) -> bytes:
    """Render the JSON-driven trajectory/error figure with saturation markers."""
    reference = np.asarray(report["reference_position_m"])
    target = np.asarray(report["target_position_m"])
    state_times = np.asarray(report["state_time_s"])
    default = report["variants"]["default"]
    candidate = report["variants"][friday.CANDIDATE_LABEL]
    default_pos = np.asarray(default["actual_position_m"])
    candidate_pos = np.asarray(candidate["actual_position_m"])
    saturated_indices = np.flatnonzero(
        np.asarray(candidate["saturation"]["decision_arrays"]["any"]).any(axis=1)
    )
    intervals = _candidate_intervals(report)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(reference[:, 0], reference[:, 1], "k--", label="reference")
    axes[0, 0].plot(default_pos[:, 0], default_pos[:, 1], label="default")
    axes[0, 0].plot(candidate_pos[:, 0], candidate_pos[:, 1], label="Seed 05 diagnostic")
    axes[0, 0].scatter(
        candidate_pos[saturated_indices, 0],
        candidate_pos[saturated_indices, 1],
        marker="x",
        color="red",
        s=55,
        linewidths=1.8,
        label="Seed 05: saturation sample",
        zorder=5,
    )
    axes[0, 0].set(title="XY trajectory", xlabel="x [m]", ylabel="y [m]", aspect="equal")

    axes[0, 1].plot(state_times, target[:, 2], "k--", label="reference")
    axes[0, 1].plot(state_times, default_pos[:, 2], label="default")
    axes[0, 1].plot(state_times, candidate_pos[:, 2], label="Seed 05 diagnostic")
    axes[0, 1].set(title="Altitude", xlabel="time [s]", ylabel="z [m]")

    axes[1, 0].plot(state_times, default["position_error_norm_m"], label="default")
    axes[1, 0].plot(state_times, candidate["position_error_norm_m"], label="Seed 05 diagnostic")
    axes[1, 0].set(title="Position-error norm", xlabel="time [s]", ylabel="error [m]")

    candidate_error = np.asarray(candidate["position_error_m"])
    for axis_index, axis_name in enumerate("xyz"):
        axes[1, 1].plot(state_times, candidate_error[:, axis_index], label=f"Seed 05: {axis_name}")
    axes[1, 1].set(
        title="Seed-05 signed tracking error", xlabel="time [s]", ylabel="actual - target [m]"
    )

    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        for start, end in intervals:
            axis.axvspan(start, end, color="red", alpha=0.16)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Frozen cf2x_L250 default versus Seed-05 rollout\n"
        "DIAGNOSTIC ONLY — SATURATION PRESENT — NOT FLIGHT READY",
        color="#A40000",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        f"Seed 05: {STATUS}; red spans/markers denote motor-saturation samples.",
        ha="center",
        color="#A40000",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    return _figure_bytes(figure)


def render_motor_figure(report: dict[str, Any]) -> bytes:
    """Render all motor commands and JSON-identical bound intervals."""
    state_times = np.asarray(report["state_time_s"])
    default = report["variants"]["default"]["saturation"]
    candidate = report["variants"][friday.CANDIDATE_LABEL]["saturation"]
    default_commands = np.asarray(default["commanded_rotor_velocity_rpm"])
    candidate_commands = np.asarray(candidate["commanded_rotor_velocity_rpm"])
    limits = candidate["limits_rpm"]
    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for motor, axis in enumerate(axes):
        axis.plot(state_times, default_commands[:, motor], label="default", color="#4C78A8")
        axis.plot(
            state_times, candidate_commands[:, motor], label="Seed 05 diagnostic", color="#F58518"
        )
        axis.axhline(
            limits["lower_threshold"][motor],
            color="#8B0000",
            linestyle=":",
            label="lower threshold" if motor == 0 else None,
        )
        axis.axhline(
            limits["upper_threshold"][motor],
            color="#5B005B",
            linestyle="--",
            label="upper threshold" if motor == 0 else None,
        )
        for interval in candidate["by_motor"][motor]["intervals"]:
            color = "#D62728" if interval["bound"] == "lower" else "#7A0177"
            axis.axvspan(interval["start_time_s"], interval["end_time_s"], color=color, alpha=0.22)
        count = candidate["by_motor"][motor]["saturated_count"]
        axis.set(title=f"Motor {motor}: Seed-05 saturated samples = {count}", ylabel="RPM")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("state time [s]")
    figure.suptitle(
        "Commanded rotor velocity and exact saturation intervals\n"
        "DIAGNOSTIC ONLY — SATURATION PRESENT — NOT FLIGHT READY",
        color="#A40000",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        f"Seed 05 status: {STATUS}. Default remains at zero saturation.",
        ha="center",
        color="#A40000",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    return _figure_bytes(figure)


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _generator_commit() -> str:
    records = _git_output("log", "--format=%H%x00%s", f"{BASE_COMMIT}..HEAD").splitlines()
    matches = [
        record.split("\0", maxsplit=1)[0]
        for record in records
        if record.split("\0", maxsplit=1)[1]
        == "research: add saturation-explicit rollout diagnostic"
    ]
    if len(matches) != 1 or not re.fullmatch(r"[0-9a-f]{40}", matches[0]):
        raise DiagnosticError("could not identify the unique committed diagnostic generator")
    return matches[0]


def _provenance(report: dict[str, Any]) -> dict[str, Any]:
    generator_commit = _generator_commit()
    return {
        "schema_version": "crazyflow.saturation_diagnostic_provenance.v1",
        "work_order": "WO-GR-F0-002",
        "base_commit": BASE_COMMIT,
        "result_commit_generator_and_tests": generator_commit,
        "package_commit": "recorded externally in worker handoff and independent review",
        "branch": BRANCH,
        "worker": WORKER,
        "worktree": WORKTREE,
        "inputs": {
            "accepted_day21_checksum_index": report["source_identity"][
                "accepted_day21_checksum_index"
            ],
            "five_verified_day21_payloads": report["source_identity"][
                "accepted_day21_payloads_verified"
            ],
            "accepted_day21_generator_commit": report["source_identity"][
                "accepted_day21_generator_commit"
            ],
            "candidate_identity": EXPECTED_CANDIDATE,
            "protected_day10_day13_read": False,
            "test_manifest_opened": False,
        },
        "configuration": {
            **report["frozen_rollout_contract"],
            "saturation_tolerance": SATURATION_TOLERANCE,
            "diagnostic_status": STATUS,
            "normal_zero_saturation_gate_unchanged": True,
        },
        "commands": {
            "generate": (
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-002 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-002-matplotlib "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "examples/jax/mellinger_saturation_diagnostic.py "
                "--diagnostic-only-saturation-present "
                "--output-dir artifacts/day22-saturation-diagnostic"
            ),
            "focused_pytest": (
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-002 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-002-matplotlib "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "-m pytest -q -p no:cacheprovider "
                "tests/unit/test_mellinger_saturation_diagnostic.py "
                "tests/unit/test_mellinger_friday_evidence.py "
                "tests/unit/test_mellinger_tracking.py "
                "tests/unit/test_mellinger_batch_diagnostics.py"
            ),
            "ruff_check": (
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-002 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "RUFF_CACHE_DIR=/tmp/crazyflow-gradient-f0-002-ruff "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "-m ruff check examples/jax/mellinger_friday_evidence.py "
                "examples/jax/mellinger_saturation_diagnostic.py "
                "tests/unit/test_mellinger_saturation_diagnostic.py"
            ),
            "ruff_format_check": (
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-002 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "-m ruff format --check examples/jax/mellinger_friday_evidence.py "
                "examples/jax/mellinger_saturation_diagnostic.py "
                "tests/unit/test_mellinger_saturation_diagnostic.py"
            ),
            "verify_checksums": (
                "cd artifacts/day22-saturation-diagnostic && sha256sum --quiet -c SHA256SUMS"
            ),
            "deterministic_reproduction": (
                "generate two fresh /tmp package directories with the exact diagnostic command; "
                "require diff -qr to be empty and SHA256SUMS 5/5 PASS in each"
            ),
            "f0_1_gate_regression_executed": (
                "load the checksum-verified accepted Day-21 candidate; call the real frozen "
                "build_rollout_trace_bundle; require Default motor_saturation_fraction 0.0 and "
                "Seed-05 motor_saturation_fraction 0.027499999850988388; pass that unchanged "
                "bundle through build_rollout and require EvidenceError at the unchanged "
                "technical gate with no output path supplied or created"
            ),
            "f0_1_source_and_test_regression_executed": (
                "verify by source diff that F0.1 parse_args, CLI, fallback branch, output paths, "
                "and fallback behavior are unchanged; run the existing F0.1 tests; verify the "
                "accepted Day-21 SHA256SUMS 9/9"
            ),
            "f0_1_full_normal_and_fallback_cli": (
                "NOT_EXECUTED_PROTECTED_DAY13_READ_PROHIBITED: both full F0.1 CLI modes would "
                "reread protected Day-13 inputs, so neither command was executed in WO-GR-F0-002"
            ),
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
        "output_checksum_rule": "SHA256SUMS covers the five other files, not itself",
    }


def _presentation_handoff(report: dict[str, Any]) -> str:
    default = report["variants"]["default"]
    candidate = report["variants"][friday.CANDIDATE_LABEL]
    default_total = default["saturation"]["total"]
    candidate_total = candidate["saturation"]["total"]
    per_motor = ", ".join(
        f"M{record['motor_index']}={record['saturated_count']}/100"
        for record in candidate["saturation"]["by_motor"]
    )
    return f"""# Saturation-explicit presentation handoff

## Fixed status and gate

Seed 05 is shown only with status `{STATUS}`. The unchanged zero-saturation acceptance gate is
`{GATE_FAILURE_STATUS}`. This package is neither candidate acceptance nor flight readiness.

| Figure | What it proves | What it does not prove | One-sentence slide takeaway |
|---|---|---|---|
| `trajectory_error_saturation.png` | Default and Seed 05 use the identical frozen cf2x_L250 H100 simulation contract; reference, actual trajectory and time-resolved error are shown, with every Seed-05 saturation sample marked. | Superiority, robustness, Test performance, convergence, flight readiness, `cf21B_500`, firmware or hardware transfer. | Seed 05 tracks this frozen simulation case descriptively, but `{candidate_total["saturated_count"]}` of `{candidate_total["sample_count_all_motors"]}` motor samples saturate, so it remains diagnostic only. |
| `motor_saturation_detail.png` | Commands, lower/upper thresholds and the exact per-motor saturation intervals are visible for both variants. | That brief saturation is acceptable, safe, or removable without a new scientific decision. | The unchanged gate fails because Seed-05 saturation is explicitly attributable per motor and time. |

## Exact saturation facts

- Default: `{default_total["saturated_count"]}/{default_total["sample_count_all_motors"]}` commands,
  fraction `{default_total["tracking_loss_metric_fraction"]}`.
- Seed 05: `{candidate_total["saturated_count"]}/{candidate_total["sample_count_all_motors"]}`
  commands, fraction `{candidate_total["tracking_loss_metric_fraction"]}`; `{per_motor}`.
- Lower and upper decisions are retained separately. Exact indices, state times and interval
  boundaries are in `saturation_diagnostic.json`.

## Exact verbal caveat for Noah

> Derselbe Simulationsfall wurde mit Default und Seed 05 ausgeführt. Seed 05 wird nur diagnostisch gezeigt, weil Motorsättigung das unveränderte Akzeptanzgate verletzt; der Satz ist weder finaler noch flugbereiter Kandidat und belegt nichts für `cf21B_500` oder Hardware. Gemessen wurden Default `{default_total["saturated_count"]}/{default_total["sample_count_all_motors"]}` und Seed 05 `{candidate_total["saturated_count"]}/{candidate_total["sample_count_all_motors"]}` sättigende Motor-Samples.

## Claim boundary

Simulation diagnostic only. Descriptive tracking metrics are inseparable from status `{STATUS}`.
No Test, tuning, candidate acceptance, superiority, safety, firmware, hardware, real-flight or
Sim2Real claim is made.
"""


def build_package_payloads(report: dict[str, Any]) -> dict[str, bytes]:
    """Build all five checksummed payloads in memory."""
    payloads = {
        "saturation_diagnostic.json": _json_bytes(report),
        "trajectory_error_saturation.png": render_trajectory_figure(report),
        "motor_saturation_detail.png": render_motor_figure(report),
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
    checksum_text = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n" for name in sorted(OUTPUT_NAMES)
    )
    (output_dir / "SHA256SUMS").write_text(checksum_text)


def main() -> None:
    """Generate the separate failed-gate diagnostic and report its exact status."""
    arguments = parse_args()
    if arguments.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {arguments.output_dir}")
    report = build_diagnostic()
    payloads = build_package_payloads(report)
    write_package(arguments.output_dir, payloads)
    candidate_total = report["variants"][friday.CANDIDATE_LABEL]["saturation"]["total"]
    print(f"diagnostic_status={STATUS}")
    print(f"normal_acceptance_gate={GATE_FAILURE_STATUS}")
    print(f"seed05_saturated_count={candidate_total['saturated_count']}")
    print(f"seed05_motor_saturation_fraction={candidate_total['tracking_loss_metric_fraction']}")
    print("test_opened=False")
    print("training_runs_started=0")


if __name__ == "__main__":
    main()
