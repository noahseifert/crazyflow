# ruff: noqa: E501 -- presentation table rows are intentionally one Markdown line each.
"""Build the deterministic Friday evidence package from frozen H100 evidence.

This command reads the accepted Day-19 synthesis, the ten explicitly supplied
Train/Validation run summaries and bounded checkpoint tails, and the accepted
Day-20 diagnostics.  It performs one H100 default/candidate simulation for
visualization only.  It never opens Test and never starts or resumes training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crazyflow.control import Control
from crazyflow.control.mellinger import (
    TrackingLossConfig,
    hover_rotor_velocity,
    initialize_tracking_state,
    rollout_state_commands,
    rotor_velocity_limits,
    tracking_loss_per_case,
)
from crazyflow.control.mellinger.research import apply_raw_gains, physical_from_raw, raw_from_data
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator
from crazyflow.trajectory import state_commands
from examples.jax.mellinger_batch_diagnostics import make_trajectory

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "4eb411d5f181baf1742ea0cf980199f5e9a8e3b1"
BRANCH = "codex/wo-gr-f0-001"
WORKER = "WO-GR-F0-001-WORKER-01 /root/wo_gr_f0_001_worker"
WORKTREE = "/tmp/crazyflow-gradient-research-wo-gr-f0-001"
SCHEMA_VERSION = "crazyflow.friday_evidence.v1"
CANDIDATE_LABEL = "provisional visualization candidate"
EXPECTED_SEEDS = 10
EXPECTED_UPDATES = 5000
CHECKPOINT_STEPS = tuple(range(100, EXPECTED_UPDATES + 1, 100))
CHECKPOINT_TAIL_BYTES = 131_072
CONTROL_FREQUENCY_HZ = 100
SIMULATION_FREQUENCY_HZ = 500
HORIZON = 100
ROLLOUT_SEED = 20260724
GAIN_NAMES = ("kp_xy", "kp_z", "kd_xy", "kd_z")
DAY19_DIR = REPOSITORY_ROOT / "artifacts/day19-h100-analysis"
DAY20_DIR = REPOSITORY_ROOT / "artifacts/day20-stage2-loss-diagnostics"
OUTPUT_NAMES = (
    "friday_evidence.json",
    "h100_loss_curves.png",
    "h100_runtime_and_reliability.png",
    "h100_selected_gain_evolution.png",
    "h100_selected_end_gains.png",
    "stage2_loss_contributions.png",
    "stage2_loss_term_gain_gradients.png",
    "rollout_trajectory.png",
    "rollout_tracking_error.png",
    "provenance.json",
    "PRESENTATION_HANDOFF.md",
)
ROLLOUT_OUTPUT_NAMES = ("rollout_trajectory.png", "rollout_tracking_error.png")
INDEX_PATTERN = re.compile(r"([0-9a-f]{64})  (?:\./)?(.+)")


class EvidenceError(RuntimeError):
    """Raised when frozen input violates the narrow evidence contract."""


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse the deliberately narrow evidence command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100-root", type=Path, required=True)
    parser.add_argument(
        "--fallback-existing-evidence",
        action="store_true",
        help="withhold the failed rollout and emit only accepted H100/Stage-2 evidence",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day21-friday-evidence"))
    return parser.parse_args(argv)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return value


def _parse_checksum_index(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = INDEX_PATTERN.fullmatch(line)
        if match is None:
            raise EvidenceError(f"invalid checksum line in {path}: {line!r}")
        digest, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise EvidenceError(f"unsafe checksum path in {path}: {relative}")
        if relative in entries:
            raise EvidenceError(f"duplicate checksum path in {path}: {relative}")
        entries[relative] = digest
    return entries


def _verify_small_indexed_files(directory: Path, index_name: str) -> dict[str, str]:
    entries = _parse_checksum_index(directory / index_name)
    for relative, expected in entries.items():
        path = directory / relative
        if not path.is_file() or _sha256_path(path) != expected:
            raise EvidenceError(f"tracked checksum mismatch: {path}")
    return entries


def _read_bounded_tail(path: Path, maximum_bytes: int = CHECKPOINT_TAIL_BYTES) -> str:
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - maximum_bytes))
        payload = stream.read(maximum_bytes)
    return payload.decode()


def _tail_json_value(tail: str, key: str) -> Any:
    marker = f'\n  "{key}": '
    if tail.count(marker) != 1:
        raise EvidenceError(f"top-level checkpoint field is not unique in bounded tail: {key}")
    encoded = tail.split(marker, maxsplit=1)[1]
    try:
        value, _ = json.JSONDecoder().raw_decode(encoded)
    except json.JSONDecodeError as error:
        raise EvidenceError(f"incomplete checkpoint tail field: {key}") from error
    return value


def extract_checkpoint_tail(path: Path) -> dict[str, Any]:
    """Extract four small top-level fields without reading a checkpoint prefix."""
    tail = _read_bounded_tail(path)
    record = {name: _tail_json_value(tail, name) for name in ("raw_gains", "selected_raw_gains")}
    record["selection"] = _tail_json_value(tail, "selection")
    record["step"] = _tail_json_value(tail, "step")
    for name in ("raw_gains", "selected_raw_gains"):
        array_record = record[name]
        if array_record.get("shape") != [4] or array_record.get("dtype") != "float32":
            raise EvidenceError(f"unexpected {name} array contract: {path}")
        if len(array_record.get("data", [])) != 4:
            raise EvidenceError(f"unexpected {name} vector: {path}")
    return record


def _physical_gain_dict(raw: list[float]) -> dict[str, float]:
    physical = physical_from_raw(jnp.asarray(raw, dtype=jnp.float32), stage=1)
    return {name: float(physical[name]) for name in GAIN_NAMES}


def _allclose(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1.0e-7, atol=1.0e-9))


def _vector_allclose(left: list[float], right: list[float]) -> bool:
    return bool(np.allclose(left, right, rtol=1.0e-7, atol=1.0e-9))


def select_candidate(analysis: dict[str, Any]) -> dict[str, Any]:
    """Apply the provisional visualization rule to accepted synthesis values."""
    seeds = analysis["seeds"]
    if len(seeds) != EXPECTED_SEEDS:
        raise EvidenceError(f"expected {EXPECTED_SEEDS} accepted seeds")
    minimum = min(float(seed["selection"]["selected_validation_loss"]) for seed in seeds)
    candidates = [
        seed for seed in seeds if float(seed["selection"]["selected_validation_loss"]) == minimum
    ]
    if len(candidates) != 1:
        raise EvidenceError("provisional visualization rule did not select exactly one seed")
    selected = candidates[0]
    return {
        "label": CANDIDATE_LABEL,
        "rule": "smallest existing selected validation loss among ten frozen H100 runs",
        "seed": int(selected["index"]),
        "selected_step": int(selected["selection"]["selected_step"]),
        "selected_validation_loss": minimum,
        "physical_gains": {
            name: float(selected["selected_physical_gains"][name]["value"]) for name in GAIN_NAMES
        },
    }


def _validate_history(summary: dict[str, Any], path: Path) -> dict[str, list[Any]]:
    history = summary.get("history")
    if not isinstance(history, list) or len(history) != EXPECTED_UPDATES:
        raise EvidenceError(f"expected 5000 summary history rows: {path}")
    steps = [int(row["step"]) for row in history]
    if steps != list(range(1, EXPECTED_UPDATES + 1)):
        raise EvidenceError(f"history step coverage mismatch: {path}")
    train = [float(row["train_loss"]) for row in history]
    validation = [float(row["validation_loss"]) for row in history]
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(validation)):
        raise EvidenceError(f"nonfinite history value: {path}")
    return {"update": steps, "train_loss": train, "selected_validation_loss": validation}


def _elapsed_seconds(text: str) -> float:
    label = "Elapsed (wall clock) time (h:mm:ss or m:ss)"
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise EvidenceError(f"missing GNU time field: {label}")
    components = [float(component) for component in match.group(1).strip().split(":")]
    if len(components) == 2:
        return components[0] * 60.0 + components[1]
    if len(components) == 3:
        return components[0] * 3600.0 + components[1] * 60.0 + components[2]
    raise EvidenceError("unsupported GNU elapsed-time value")


def _check_candidate_sources(
    candidate: dict[str, Any], summary: dict[str, Any], checkpoint: dict[str, Any]
) -> list[float]:
    selected = summary["selected"]
    raw = [float(value) for value in selected["selected_raw_gains"]]
    checks = (
        candidate["seed"] == 5,
        candidate["selected_step"] == 5000,
        int(checkpoint["step"]) == 5000,
        int(checkpoint["selection"]["selected_step"]) == 5000,
        _allclose(
            candidate["selected_validation_loss"], float(selected["selected_validation_loss"])
        ),
        _allclose(
            candidate["selected_validation_loss"],
            float(checkpoint["selection"]["selected_validation_loss"]),
        ),
        _vector_allclose(raw, [float(value) for value in checkpoint["selected_raw_gains"]["data"]]),
    )
    summary_physical = {
        name: float(summary["selected_physical_gains"][name]) for name in GAIN_NAMES
    }
    checkpoint_physical = _physical_gain_dict(checkpoint["selected_raw_gains"]["data"])
    gains_match = all(
        _allclose(candidate["physical_gains"][name], summary_physical[name])
        and _allclose(candidate["physical_gains"][name], checkpoint_physical[name])
        for name in GAIN_NAMES
    )
    if not all(checks) or not gains_match:
        raise EvidenceError("Seed-05 candidate source cross-check failed")
    return raw


def build_h100_evidence(h100_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the allowed H100 summaries, timing files, indexes, and checkpoint tails."""
    _verify_small_indexed_files(DAY19_DIR, "SHA256SUMS")
    analysis = _load_json(DAY19_DIR / "h100_replication_analysis.json")
    if analysis.get("status") != "PASS" or analysis["test_boundary"]["test_metric_rows"] != 0:
        raise EvidenceError("accepted Day-19 synthesis contract mismatch")
    candidate = select_candidate(analysis)
    source_indexes = _parse_checksum_index(DAY19_DIR / "SOURCE_RUN_INDEXES.sha256")
    if len(source_indexes) != EXPECTED_SEEDS:
        raise EvidenceError("expected ten source run indexes")

    seed_records = []
    candidate_raw: list[float] | None = None
    for accepted in analysis["seeds"]:
        seed_index = int(accepted["index"])
        relative_run = Path(accepted["run_path"])
        expected_prefix = Path("artifacts/day13-h100-replication/runs")
        if relative_run.parts[: len(expected_prefix.parts)] != expected_prefix.parts:
            raise EvidenceError(f"unexpected accepted run path: {relative_run}")
        relative_below_root = Path(*relative_run.parts[len(expected_prefix.parts) :])
        run_dir = h100_root / relative_below_root
        index_path = run_dir / "RUN_SHA256SUMS"
        repository_relative_index = (relative_run / "RUN_SHA256SUMS").as_posix()
        expected_index_hash = source_indexes.get(repository_relative_index)
        if expected_index_hash is None or _sha256_path(index_path) != expected_index_hash:
            raise EvidenceError(f"current small run-index hash mismatch: {index_path}")
        run_index = _parse_checksum_index(index_path)
        expected_checkpoints = {
            f"runner-output/checkpoint-step-{step:06d}.json" for step in CHECKPOINT_STEPS
        }
        if not expected_checkpoints.issubset(run_index):
            raise EvidenceError(f"checkpoint index coverage mismatch: {index_path}")
        if "runner-output/run_summary.json" not in run_index or "time.txt" not in run_index:
            raise EvidenceError(f"summary/time digest absent from run index: {index_path}")

        summary_path = run_dir / "runner-output/run_summary.json"
        summary = _load_json(summary_path)
        if (
            summary.get("status") != "success"
            or summary.get("completed_updates") != EXPECTED_UPDATES
            or summary.get("configured_updates") != EXPECTED_UPDATES
            or summary.get("test_metrics_present") is not False
            or summary.get("test_manifest_opened") is not False
            or summary["selected"].get("test_used_for_selection") is not False
        ):
            raise EvidenceError(f"technical completion contract mismatch: {summary_path}")
        history = _validate_history(summary, summary_path)
        accepted_selection = accepted["selection"]
        if not _allclose(
            float(summary["selected"]["selected_validation_loss"]),
            float(accepted_selection["selected_validation_loss"]),
        ):
            raise EvidenceError(f"summary/Day-19 selected loss mismatch: {summary_path}")

        time_path = run_dir / "time.txt"
        wall_seconds = _elapsed_seconds(time_path.read_text())
        accepted_wall_seconds = float(accepted["resources"]["external_wall_seconds"])
        if not _allclose(wall_seconds, accepted_wall_seconds):
            raise EvidenceError(f"GNU time/Day-19 duration mismatch: {time_path}")

        evolution = []
        final_checkpoint: dict[str, Any] | None = None
        for step in CHECKPOINT_STEPS:
            checkpoint_path = run_dir / "runner-output" / f"checkpoint-step-{step:06d}.json"
            checkpoint = extract_checkpoint_tail(checkpoint_path)
            if int(checkpoint["step"]) != step:
                raise EvidenceError(f"checkpoint step mismatch: {checkpoint_path}")
            selected_raw = [float(value) for value in checkpoint["selected_raw_gains"]["data"]]
            evolution.append(
                {
                    "checkpoint_step": step,
                    "selected_step": int(checkpoint["selection"]["selected_step"]),
                    "selected_raw_gains": selected_raw,
                    "selected_physical_gains": _physical_gain_dict(selected_raw),
                }
            )
            final_checkpoint = checkpoint
        assert final_checkpoint is not None

        selected_physical = {
            name: float(summary["selected_physical_gains"][name]) for name in GAIN_NAMES
        }
        if not all(
            _allclose(
                selected_physical[name], float(accepted["selected_physical_gains"][name]["value"])
            )
            for name in GAIN_NAMES
        ):
            raise EvidenceError(f"selected end-gain mismatch: {summary_path}")
        if seed_index == candidate["seed"]:
            candidate_raw = _check_candidate_sources(candidate, summary, final_checkpoint)

        seed_records.append(
            {
                "seed": seed_index,
                "run_path": relative_run.as_posix(),
                "history": history,
                "duration": {
                    "external_wall_seconds": wall_seconds,
                    "source": "GNU time Elapsed (wall clock), accepted by Day-19 synthesis",
                },
                "technical_completion": {
                    "status": "complete_no_abort",
                    "completed_updates": EXPECTED_UPDATES,
                    "expected_updates": EXPECTED_UPDATES,
                    "history_records": len(history["update"]),
                    "run_gates_passed": True,
                },
                "selection": {
                    "selected_step": int(summary["selected"]["selected_step"]),
                    "selected_validation_loss": float(
                        summary["selected"]["selected_validation_loss"]
                    ),
                },
                "selected_gain_evolution": evolution,
                "selected_end_physical_gains": selected_physical,
                "integrity": {
                    "run_index_path": repository_relative_index,
                    "run_index_sha256_currently_verified": expected_index_hash,
                    "run_summary_sha256_in_inherited_index": run_index[
                        "runner-output/run_summary.json"
                    ],
                    "time_sha256_in_inherited_index": run_index["time.txt"],
                    "checkpoint_hashes_in_inherited_index": len(expected_checkpoints),
                    "large_payload_hashes_recomputed": False,
                },
            }
        )

    if candidate_raw is None:
        raise EvidenceError("candidate seed missing from frozen synthesis")
    selected_losses = [record["selection"]["selected_validation_loss"] for record in seed_records]
    accepted_variation = analysis["aggregates"]["selected_validation_loss"]
    if not np.allclose(
        selected_losses, accepted_variation["values_in_manifest_order"], rtol=0.0, atol=0.0
    ):
        raise EvidenceError("selected-loss series differs from accepted Day-19 synthesis")
    durations = [record["duration"]["external_wall_seconds"] for record in seed_records]
    if not _allclose(sum(durations), float(analysis["aggregates"]["total_external_wall_seconds"])):
        raise EvidenceError("duration sum differs from accepted Day-19 synthesis")

    evidence = {
        "definitions": {
            "train_loss": "unchanged Stage-1 training objective after each completed update",
            "selected_validation_loss_curve": (
                "unchanged fixed-Validation objective after each completed update; Test unopened"
            ),
            "external_wall_seconds": (
                "per-process GNU time elapsed wall clock; sum is not calendar/GPU/parallel time"
            ),
            "technical_reliability": (
                "completion/abort, 5000/5000 updates, expected records and accepted run gates"
            ),
            "numerical_variation": (
                "distribution of predefined selected fixed-Validation loss; not success probability"
            ),
        },
        "duration": {
            "per_run_available_count": len(durations),
            "per_run_expected_count": EXPECTED_SEEDS,
            "missing_value_policy": "null; never impute zero or omit from available_count",
            "total_process_wall_seconds": sum(durations),
        },
        "technical_reliability": {
            "completed_no_abort_count": EXPECTED_SEEDS,
            "expected_count": EXPECTED_SEEDS,
            "scope": "technical run completion only",
        },
        "numerical_variation": accepted_variation,
        "seeds": seed_records,
        "inherited_integrity_boundary": {
            "day19_synthesis_checksums_currently_verified": True,
            "ten_run_index_hashes_currently_verified": True,
            "large_h100_payload_hashes_recomputed": False,
            "accepted_day19_indexed_file_count": analysis["aggregates"][
                "total_indexed_raw_files_verified"
            ],
            "meaning": (
                "large-payload integrity is inherited from accepted Day-19 evidence, not re-audited"
            ),
        },
    }
    candidate["selected_raw_gains"] = candidate_raw
    candidate["source_crosscheck"] = {
        "day19_synthesis": True,
        "seed05_run_summary": True,
        "seed05_checkpoint_005000_bounded_tail": True,
        "rtol": 1.0e-7,
        "atol": 1.0e-9,
    }
    return evidence, candidate


def _rollout_variant(
    initial_data: Any, commands: jax.Array, reference: Any, step_fn: Any, raw_gains: list[float]
) -> tuple[dict[str, Any], Any]:
    data = apply_raw_gains(initial_data, jnp.asarray(raw_gains, dtype=jnp.float32), stage=1)
    _, trace = rollout_state_commands(
        data, commands, step_fn, SIMULATION_FREQUENCY_HZ // CONTROL_FREQUENCY_HZ
    )
    _, metrics = tracking_loss_per_case(
        trace,
        reference,
        hover_rotor_velocity(initial_data),
        rotor_velocity_limits(initial_data),
        TrackingLossConfig(),
    )
    jax.block_until_ready((trace, metrics))
    actual = np.asarray(trace.pos)[:, 0, 0, :]
    target = np.asarray(reference.pos)
    error = actual - target
    error_norm = np.linalg.norm(error, axis=1)
    metric_values = {name: float(np.asarray(value)[0]) for name, value in metrics.items()}
    return (
        {
            "raw_gains": [float(value) for value in raw_gains],
            "physical_gains": _physical_gain_dict(raw_gains),
            "actual_position_m": actual.tolist(),
            "position_error_m": error.tolist(),
            "position_error_norm_m": error_norm.tolist(),
            "position_rmse_m_direct": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "max_position_error_m_direct": float(np.max(error_norm)),
            "metrics": metric_values,
            "technical_gates": {
                "all_finite": bool(np.all(np.isfinite(actual)))
                and all(math.isfinite(value) for value in metric_values.values()),
                "motor_saturation_fraction_zero": metric_values["motor_saturation_fraction"] == 0.0,
                "floor_clip_fraction_zero": metric_values["floor_clip_fraction"] == 0.0,
                "zero_thrust_gate_fraction_zero": metric_values["zero_thrust_gate_fraction"] == 0.0,
                "nonfinite_state_fraction_zero": metric_values["nonfinite_state_fraction"] == 0.0,
            },
        },
        trace,
    )


def build_rollout_trace_bundle(
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Build the frozen H100 comparison and retain its pre-gate traces."""
    full_reference, reference = make_trajectory("figure8", HORIZON, CONTROL_FREQUENCY_HZ)
    commands = state_commands(reference)[:, None, None, :]
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone="cf2x_L250",
        dynamics=Dynamics.first_principles,
        control=Control.state,
        integrator=Integrator.euler,
        freq=SIMULATION_FREQUENCY_HZ,
        state_freq=CONTROL_FREQUENCY_HZ,
        attitude_freq=SIMULATION_FREQUENCY_HZ,
        force_torque_freq=SIMULATION_FREQUENCY_HZ,
        device="cpu",
        rng_key=ROLLOUT_SEED,
    )
    initial_data = initialize_tracking_state(sim.data, full_reference.pos[0], full_reference.vel[0])
    default_raw = [float(value) for value in np.asarray(raw_from_data(initial_data, stage=1))]
    step_fn = sim.build_step_fn()
    default, default_trace = _rollout_variant(
        initial_data, commands, reference, step_fn, default_raw
    )
    provisional, provisional_trace = _rollout_variant(
        initial_data, commands, reference, step_fn, candidate["selected_raw_gains"]
    )
    gates = {
        "same_reference": True,
        "same_initial_condition": True,
        "same_integrator_and_step_function": True,
        "same_time_base_and_metrics": True,
        "default_all_technical_gates_pass": all(default["technical_gates"].values()),
        "candidate_all_technical_gates_pass": all(provisional["technical_gates"].values()),
    }
    default_rmse = default["position_rmse_m_direct"]
    candidate_rmse = provisional["position_rmse_m_direct"]
    rollout = {
        "purpose": "single deterministic simulation visualization; not candidate selection",
        "candidate_label": CANDIDATE_LABEL,
        "contract": {
            "platform": "cf2x_L250",
            "dynamics": "first_principles",
            "integrator": "explicit_euler",
            "simulation_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "state_control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "simulation_steps_per_reference": SIMULATION_FREQUENCY_HZ // CONTROL_FREQUENCY_HZ,
            "horizon_control_intervals": HORIZON,
            "duration_seconds": HORIZON / CONTROL_FREQUENCY_HZ,
            "seed": ROLLOUT_SEED,
            "trajectory": {
                "kind": "figure8",
                "center_m": [0.0, 0.0, 0.75],
                "amplitude_m": [0.30, 0.15],
                "period_seconds": 3.0,
                "ramp_duration_seconds": 0.5,
                "yaw_rad": 0.0,
            },
            "initial_position_m": np.asarray(initial_data.states.pos)[0, 0].tolist(),
            "initial_velocity_m_s": np.asarray(initial_data.states.vel)[0, 0].tolist(),
            "initial_rotor_velocity_rad_s": np.asarray(initial_data.states.rotor_vel)[
                0, 0
            ].tolist(),
        },
        "reference_time_s": (np.arange(HORIZON + 1) / CONTROL_FREQUENCY_HZ).tolist(),
        "state_time_s": (np.arange(1, HORIZON + 1) / CONTROL_FREQUENCY_HZ).tolist(),
        "reference_position_m": np.asarray(full_reference.pos).tolist(),
        "target_position_m": np.asarray(reference.pos).tolist(),
        "variants": {"default": default, CANDIDATE_LABEL: provisional},
        "comparison": {
            "position_rmse_difference_candidate_minus_default_m": candidate_rmse - default_rmse,
            "position_rmse_ratio_candidate_over_default": candidate_rmse / default_rmse,
            "observed_direction": "lower" if candidate_rmse < default_rmse else "not_lower",
            "improvement_is_acceptance_criterion": False,
        },
        "equality_and_technical_gates": gates,
    }
    traces = {"default": default_trace, CANDIDATE_LABEL: provisional_trace}
    return rollout, traces, initial_data


def build_rollout(candidate: dict[str, Any]) -> dict[str, Any]:
    """Run the one authorized, deterministic H100 comparison."""
    rollout, _, _ = build_rollout_trace_bundle(candidate)
    if not all(rollout["equality_and_technical_gates"].values()):
        raise EvidenceError("authorized H100 rollout failed equality or technical gates")
    return rollout


def _save_loss_curves(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, EXPECTED_SEEDS))
    for record, color in zip(report["h100"]["seeds"], colors, strict=True):
        history = record["history"]
        label = f"Seed {record['seed']:02d}"
        axes[0].plot(history["update"], history["train_loss"], color=color, label=label)
        axes[1].plot(
            history["update"], history["selected_validation_loss"], color=color, label=label
        )
    axes[0].set(title="Train loss", ylabel="unchanged tracking loss [1]")
    axes[1].set(title="Fixed Validation loss (selection quantity)")
    for axis in axes:
        axis.set(xlabel="completed update", yscale="log")
        axis.grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    figure.suptitle("Frozen H100 runs: all ten loss histories through update 5000")
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)


def _save_runtime_reliability(report: dict[str, Any], path: Path) -> None:
    seeds = report["h100"]["seeds"]
    indices = [record["seed"] for record in seeds]
    durations = [record["duration"]["external_wall_seconds"] / 60.0 for record in seeds]
    losses = [record["selection"]["selected_validation_loss"] for record in seeds]
    mean = report["h100"]["numerical_variation"]["mean"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(indices, durations, color="#4C78A8")
    axes[0].set(
        title="Process wall-clock duration (10/10 available)",
        xlabel="seed",
        ylabel="GNU time elapsed [min]",
        xticks=indices,
    )
    axes[1].scatter(indices, losses, color="#F58518", s=45, label="selected Validation loss")
    axes[1].axhline(mean, color="black", linestyle="--", label="accepted mean")
    axes[1].set(
        title="Numerical variation (separate from completion)",
        xlabel="seed",
        ylabel="selected fixed-Validation loss [1]",
        xticks=indices,
    )
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Technical reliability: 10/10 complete, no documented abort")
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)


def _save_gain_evolution(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, EXPECTED_SEEDS))
    for axis, gain_name in zip(axes.flat, GAIN_NAMES, strict=True):
        for record, color in zip(report["h100"]["seeds"], colors, strict=True):
            points = record["selected_gain_evolution"]
            axis.plot(
                [point["checkpoint_step"] for point in points],
                [point["selected_physical_gains"][gain_name] for point in points],
                color=color,
                label=f"Seed {record['seed']:02d}",
            )
        axis.set(title=gain_name, xlabel="checkpoint update", ylabel="physical gain")
        axis.grid(alpha=0.25)
    axes[0, 1].legend(ncol=2, fontsize=7)
    figure.suptitle("Selected Stage-1 gain evolution (checkpoint records, no gain averaging)")
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)


def _save_end_gains(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    seeds = report["h100"]["seeds"]
    indices = [record["seed"] for record in seeds]
    for axis, gain_name in zip(axes.flat, GAIN_NAMES, strict=True):
        values = [record["selected_end_physical_gains"][gain_name] for record in seeds]
        colors = ["#E45756" if index == 5 else "#4C78A8" for index in indices]
        axis.bar(indices, values, color=colors)
        axis.set(
            title=gain_name, xlabel="seed", ylabel="selected physical end gain", xticks=indices
        )
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Selected end gains across frozen runs (red: provisional visualization candidate)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)


def _save_rollout_trajectory(report: dict[str, Any], path: Path) -> None:
    rollout = report["rollout"]
    reference = np.asarray(rollout["reference_position_m"])
    target = np.asarray(rollout["target_position_m"])
    times = rollout["state_time_s"]
    variants = rollout["variants"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(reference[:, 0], reference[:, 1], "k--", label="reference")
    axes[0].plot(
        np.asarray(variants["default"]["actual_position_m"])[:, 0],
        np.asarray(variants["default"]["actual_position_m"])[:, 1],
        label="default",
    )
    axes[0].plot(
        np.asarray(variants[CANDIDATE_LABEL]["actual_position_m"])[:, 0],
        np.asarray(variants[CANDIDATE_LABEL]["actual_position_m"])[:, 1],
        label=CANDIDATE_LABEL,
    )
    axes[0].set(title="XY trajectory", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    axes[1].plot(times, target[:, 2], "k--", label="reference")
    axes[1].plot(times, np.asarray(variants["default"]["actual_position_m"])[:, 2], label="default")
    axes[1].plot(
        times,
        np.asarray(variants[CANDIDATE_LABEL]["actual_position_m"])[:, 2],
        label=CANDIDATE_LABEL,
    )
    axes[1].set(title="Altitude", xlabel="time [s]", ylabel="z [m]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Single deterministic cf2x_L250 H100 simulation visualization")
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)


def _save_rollout_error(report: dict[str, Any], path: Path) -> None:
    rollout = report["rollout"]
    times = rollout["state_time_s"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label in ("default", CANDIDATE_LABEL):
        variant = rollout["variants"][label]
        error = np.asarray(variant["position_error_m"])
        axes[0].plot(times, variant["position_error_norm_m"], label=label)
        for index, axis_name in enumerate("xyz"):
            axes[1].plot(times, error[:, index], label=f"{label}: {axis_name}")
    axes[0].set(title="Position-error norm", xlabel="time [s]", ylabel="error [m]")
    axes[1].set(title="Signed axis error", xlabel="time [s]", ylabel="actual - target [m]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Same reference, initial state, time base, integrator and metrics")
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "Crazyflow"})
    plt.close(figure)


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _generator_result_commit() -> str:
    records = _git_output("log", "--format=%H%x00%s", f"{BASE_COMMIT}..HEAD").splitlines()
    matches = [
        record.split("\0", maxsplit=1)[0]
        for record in records
        if record.split("\0", maxsplit=1)[1] == "research: add Friday evidence generator"
    ]
    if len(matches) != 1 or not re.fullmatch(r"[0-9a-f]{40}", matches[0]):
        raise EvidenceError("could not identify the unique committed Friday evidence generator")
    return matches[0]


def _output_names(fallback_existing_evidence: bool) -> tuple[str, ...]:
    if fallback_existing_evidence:
        return tuple(name for name in OUTPUT_NAMES if name not in ROLLOUT_OUTPUT_NAMES)
    return OUTPUT_NAMES


def _provenance(
    result_commit: str, h100_root: Path, *, fallback_existing_evidence: bool
) -> dict[str, Any]:
    source_index = _parse_checksum_index(DAY19_DIR / "SOURCE_RUN_INDEXES.sha256")
    return {
        "schema_version": "crazyflow.friday_evidence_provenance.v1",
        "work_order": "WO-GR-F0-001",
        "base_commit": BASE_COMMIT,
        "result_commit_generator_and_tests": result_commit,
        "package_commit": "recorded externally in worker handoff and independent review",
        "branch": BRANCH,
        "worker": WORKER,
        "worktree": WORKTREE,
        "inputs": {
            "day19_analysis": {
                "path": "artifacts/day19-h100-analysis/h100_replication_analysis.json",
                "sha256": _sha256_path(DAY19_DIR / "h100_replication_analysis.json"),
                "hash_class": "currently verified against tracked Day-19 SHA256SUMS",
            },
            "day19_source_run_indexes": {
                "path": "artifacts/day19-h100-analysis/SOURCE_RUN_INDEXES.sha256",
                "sha256": _sha256_path(DAY19_DIR / "SOURCE_RUN_INDEXES.sha256"),
                "count": len(source_index),
                "hash_class": "ten small RUN_SHA256SUMS files currently verified",
            },
            "h100_live_root": str(h100_root),
            "h100_payload_integrity": (
                "accepted Day-19 digests inherited; no current full hash of 4.38 GB payload"
            ),
            "day20_loss_diagnostics": {
                "path": "artifacts/day20-stage2-loss-diagnostics/loss_diagnostics.json",
                "sha256": _sha256_path(DAY20_DIR / "loss_diagnostics.json"),
            },
            "day20_checksum_index": {
                "path": "artifacts/day20-stage2-loss-diagnostics/SHA256SUMS",
                "sha256": _sha256_path(DAY20_DIR / "SHA256SUMS"),
            },
        },
        "selected_checkpoint": {
            "seed": 5,
            "update": 5000,
            "relative_path": (
                "artifacts/day13-h100-replication/runs/seed-05/"
                "20260801T204740Z-ff85c8e9c0e7/runner-output/checkpoint-step-005000.json"
            ),
            "read_mode": f"bounded file tail, maximum {CHECKPOINT_TAIL_BYTES} bytes",
            "candidate_label": CANDIDATE_LABEL,
        },
        "commands": {
            "generate": (
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-001 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-matplotlib "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "examples/jax/mellinger_friday_evidence.py "
                "--h100-root /home/noah3/bachelorarbeit/crazyflow-gradient-research/"
                "artifacts/day13-h100-replication/runs "
                + ("--fallback-existing-evidence " if fallback_existing_evidence else "")
                + "--output-dir artifacts/day21-friday-evidence"
            ),
            "verify_checksums": (
                "cd artifacts/day21-friday-evidence && sha256sum --quiet -c SHA256SUMS"
            ),
            "focused_pytest": (
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-001 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-matplotlib "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "-m pytest -q -p no:cacheprovider "
                "tests/unit/test_mellinger_friday_evidence.py "
                "tests/unit/test_h100_replication_analysis.py "
                "tests/unit/test_mellinger_batch_diagnostics.py "
                "tests/unit/test_mellinger_loss_diagnostics.py"
            ),
            "ruff_check": (
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-001 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "RUFF_CACHE_DIR=/tmp/crazyflow-gradient-f0-ruff "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "-m ruff check examples/jax/mellinger_friday_evidence.py "
                "tests/unit/test_mellinger_friday_evidence.py"
            ),
            "ruff_format_check": (
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-001 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "-m ruff format --check examples/jax/mellinger_friday_evidence.py "
                "tests/unit/test_mellinger_friday_evidence.py"
            ),
            "two_target_reproduction_and_diff": (
                'repro_a="$(mktemp -d /tmp/wo-gr-f0-repro-a.XXXXXX)"; '
                'repro_b="$(mktemp -d /tmp/wo-gr-f0-repro-b.XXXXXX)"; '
                'for target in "$repro_a/package" "$repro_b/package"; do '
                "PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-f0-001 "
                "PYTHONDONTWRITEBYTECODE=1 "
                "MPLCONFIGDIR=/tmp/crazyflow-gradient-f0-matplotlib "
                "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
                "examples/jax/mellinger_friday_evidence.py "
                "--h100-root /home/noah3/bachelorarbeit/crazyflow-gradient-research/"
                "artifacts/day13-h100-replication/runs --fallback-existing-evidence "
                '--output-dir "$target"; done; '
                'diff -qr "$repro_a/package" "$repro_b/package"'
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
        "configuration": {
            "platform": "cf2x_L250",
            "horizon": HORIZON,
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "simulation_frequency_hz": SIMULATION_FREQUENCY_HZ,
            "integrator": "explicit_euler",
            "rollout_seed": ROLLOUT_SEED,
            "h100_run_seeds": list(range(1, EXPECTED_SEEDS + 1)),
            "test_opened": False,
            "fallback_existing_evidence": fallback_existing_evidence,
        },
        "rollout_evidence": (
            "WITHHELD_TECHNICAL_GATE"
            if fallback_existing_evidence
            else "subject to unchanged zero-saturation acceptance gate"
        ),
        "output_inventory": list(_output_names(fallback_existing_evidence)) + ["SHA256SUMS"],
        "output_checksum_rule": (
            "SHA256SUMS covers the 9 other fallback output files, not itself"
            if fallback_existing_evidence
            else "SHA256SUMS covers the 11 other output files, not itself"
        ),
    }


def _presentation_handoff(report: dict[str, Any]) -> str:
    rollout_withheld = report["rollout"]["status"] == "WITHHELD_TECHNICAL_GATE"
    if rollout_withheld:
        rollout_rows = """| Default/candidate rollout | No rollout figure is included. | The attempted narrow comparison did not meet the unchanged zero-motor-saturation acceptance gate. | Rollout evidence is openly withheld; no replacement value is shown. |
"""
        rollout_section = """## Missing rollout evidence

`rollout_trajectory.png` and `rollout_tracking_error.png` are intentionally absent. The narrow
comparison did not meet the unchanged zero-motor-saturation acceptance gate, so no rollout arrays,
metrics, plots or replacement values are included. A different candidate, case or threshold was
not substituted.
"""
    else:
        ratio = report["rollout"]["comparison"]["position_rmse_ratio_candidate_over_default"]
        rollout_rows = f"""| `rollout_trajectory.png` | Default and visualization candidate were simulated on the same deterministic cf2x_L250 Figure-8 H100 contract. | Robustness, superiority, Test, hardware, firmware, flight or Sim2Real transfer. | One controlled simulation shows default and candidate trajectories on identical conditions. |
| `rollout_tracking_error.png` | Time-resolved axis and norm errors for that one controlled simulation; candidate/default RMSE ratio is `{ratio:.6f}`. | Statistical advantage or generalization beyond this trajectory. | The narrow rollout quantifies the observed difference without broad claims. |
"""
        rollout_section = ""
    return f"""# Friday presentation handoff

## Fixed labels and boundary

Seed 05 at update 5000 is used only as the **{CANDIDATE_LABEL}**, selected by the
smallest already existing selected fixed-Validation loss among the ten frozen H100 runs. It is
not a final candidate or flight candidate, and no gains were averaged. Test remained unopened.

| Figure | What it proves | What it does not prove | One-sentence slide title |
|---|---|---|---|
| `h100_loss_curves.png` | All ten recorded Train/fixed-Validation histories contain updates 1–5000 under the frozen Stage-1 protocol. | Convergence, Test performance, robustness or generalization. | All ten frozen H100 Train/Validation histories are recorded through update 5000. |
| `h100_runtime_and_reliability.png` | 10/10 runs completed without documented abort; per-process GNU wall time is available; accepted selected-Validation values show small numerical spread. | Calendar/GPU time or a robust success probability. | Existing runs completed technically, with tightly clustered selected Validation results. |
| `h100_selected_gain_evolution.png` | Checkpoint-selected physical Stage-1 gains at updates 100…5000 for each seed. | Continuous between-checkpoint behavior, convergence or a valid averaged controller. | Selected gain paths are visible per seed without gain averaging. |
| `h100_selected_end_gains.png` | The ten selected end-gain combinations and the visualization-only Seed-05 highlight. | Flight suitability or final candidate approval. | Seed 05 is highlighted only for visualization, not selected for flight. |
| `stage2_loss_contributions.png` | Six existing weighted loss contributions for Figure-8/Train and Circle/Validation at H20, defaults, zero updates. | Optimized-gain behavior or convergence. | Stage-2 makes all six existing loss contributions inspectable. |
| `stage2_loss_term_gain_gradients.png` | 48 local Jacobians with respect to four dimensionless sigmoid-bound Stage-1 variables in the same H20 smoke. | Physical-unit gradients, global sensitivity or optimized behavior. | Local loss×variable sensitivities are explicit, but remain an infrastructure smoke. |
{rollout_rows}

{rollout_section}

## Duration and reliability wording

- Total `{report["h100"]["duration"]["total_process_wall_seconds"]:.2f}` s is the sum of ten
  process wall times, not calendar duration, GPU time or parallel elapsed time.
- Technical reliability is exactly 10/10 complete/no documented abort under the frozen protocol.
- Numerical variation is the accepted distribution of selected fixed-Validation loss and is not
  a success probability.

## Stage-2 claim boundary

The copied plots are byte-identical accepted Day-20 evidence: six existing loss terms and 48 local
Jacobians with respect to four dimensionless internal sigmoid-bound variables; Figure-8/Train and
Circle/Validation; H20; default gains; zero optimizer updates.

## Global claim boundary

Simulation evidence only. No Test, H200/H400 study, new optimization, convergence proof,
generalization, robustness, controller superiority, firmware equivalence, hardware validation,
flight safety, `cf21B_500` applicability or Sim2Real transfer is established.
"""


def build_report(h100_root: Path, *, fallback_existing_evidence: bool = False) -> dict[str, Any]:
    """Build all machine-readable evidence before any output write."""
    h100, candidate = build_h100_evidence(h100_root)
    if fallback_existing_evidence:
        rollout = {
            "status": "WITHHELD_TECHNICAL_GATE",
            "evidence_available": False,
            "missing_evidence": (
                "default-versus-provisional-candidate trajectory and tracking-error evidence"
            ),
            "acceptance_contract": "zero motor saturation for both compared controllers",
            "substitution_performed": False,
            "rollout_arrays_present": False,
            "rollout_figures_present": False,
        }
    else:
        rollout = build_rollout(candidate)
        rollout["status"] = "PASS"
        rollout["evidence_available"] = True
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "WITHHELD_TECHNICAL_GATE" if fallback_existing_evidence else "PASS",
        "work_order": "WO-GR-F0-001",
        "scope": (
            "existing frozen H100 Train/Validation evidence and accepted Stage-2 diagnostics; "
            "rollout evidence withheld at the unchanged technical gate"
            if fallback_existing_evidence
            else (
                "existing frozen H100 Train/Validation evidence, accepted Stage-2 diagnostics, "
                "and one deterministic H100 simulation visualization"
            )
        ),
        "candidate": candidate,
        "h100": h100,
        "stage2": {
            "source_schema": _load_json(DAY20_DIR / "loss_diagnostics.json")["schema_version"],
            "copied_figures_byte_identical": True,
            "loss_terms": 6,
            "local_jacobians": 48,
            "bound_variable_count": 4,
            "variable_space": "unconstrained_inputs_to_sigmoid_bounded_stage1_gains",
            "cases": ["figure8/train", "circle/validation"],
            "horizon": 20,
            "gain_state": "Crazyflow default",
            "optimizer_updates": 0,
            "claim_boundary": (
                "deterministic infrastructure smoke only; no optimized-gain or convergence claim"
            ),
        },
        "rollout": rollout,
        "claim_boundary": (
            "simulation evidence only; no Test, convergence, generalization, robustness, "
            "superiority, firmware, hardware, flight, Sim2Real, or cf21B_500 claim"
        ),
        "validations": {
            "ten_seeds": len(h100["seeds"]) == EXPECTED_SEEDS,
            "five_thousand_rows_each": all(
                len(seed["history"]["update"]) == EXPECTED_UPDATES for seed in h100["seeds"]
            ),
            "ten_durations": h100["duration"]["per_run_available_count"] == EXPECTED_SEEDS,
            "candidate_crosschecked": all(candidate["source_crosscheck"].values()),
            "rollout_withheld_without_substitution": (
                fallback_existing_evidence
                and rollout["status"] == "WITHHELD_TECHNICAL_GATE"
                and not rollout["rollout_arrays_present"]
                and not rollout["rollout_figures_present"]
                and not rollout["substitution_performed"]
            ),
            "rollout_gates": (
                None
                if fallback_existing_evidence
                else all(rollout["equality_and_technical_gates"].values())
            ),
            "test_opened": False,
        },
    }


def write_package(
    report: dict[str, Any],
    h100_root: Path,
    output_dir: Path,
    *,
    fallback_existing_evidence: bool = False,
) -> None:
    """Write all allowlisted outputs once and refuse any overwrite."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "friday_evidence.json").write_text(_json_text(report))
    _save_loss_curves(report, output_dir / "h100_loss_curves.png")
    _save_runtime_reliability(report, output_dir / "h100_runtime_and_reliability.png")
    _save_gain_evolution(report, output_dir / "h100_selected_gain_evolution.png")
    _save_end_gains(report, output_dir / "h100_selected_end_gains.png")
    day20_index = _verify_small_indexed_files(DAY20_DIR, "SHA256SUMS")
    stage2_sources = {
        "loss_contributions.png": "stage2_loss_contributions.png",
        "loss_term_gain_gradients.png": "stage2_loss_term_gain_gradients.png",
    }
    for source_name, target_name in stage2_sources.items():
        source = DAY20_DIR / source_name
        if _sha256_path(source) != day20_index[source_name]:
            raise EvidenceError(f"accepted Stage-2 source checksum mismatch: {source}")
        shutil.copyfile(source, output_dir / target_name)
    if not fallback_existing_evidence:
        _save_rollout_trajectory(report, output_dir / "rollout_trajectory.png")
        _save_rollout_error(report, output_dir / "rollout_tracking_error.png")
    result_commit = _generator_result_commit()
    (output_dir / "provenance.json").write_text(
        _json_text(
            _provenance(
                result_commit,
                h100_root.resolve(),
                fallback_existing_evidence=fallback_existing_evidence,
            )
        )
    )
    (output_dir / "PRESENTATION_HANDOFF.md").write_text(_presentation_handoff(report))
    expected_outputs = _output_names(fallback_existing_evidence)
    missing = [name for name in expected_outputs if not (output_dir / name).is_file()]
    if missing:
        raise EvidenceError(f"output inventory incomplete: {', '.join(missing)}")
    checksum_text = "".join(
        f"{_sha256_path(output_dir / name)}  {name}\n" for name in sorted(expected_outputs)
    )
    (output_dir / "SHA256SUMS").write_text(checksum_text)


def main() -> None:
    """Generate the fixed package and report the narrow result."""
    arguments = parse_args()
    report = build_report(
        arguments.h100_root.resolve(),
        fallback_existing_evidence=arguments.fallback_existing_evidence,
    )
    write_package(
        report,
        arguments.h100_root.resolve(),
        arguments.output_dir,
        fallback_existing_evidence=arguments.fallback_existing_evidence,
    )
    print("friday_evidence=WRITE_PASS")
    print(f"seed_count={len(report['h100']['seeds'])}")
    print(f"candidate_label={CANDIDATE_LABEL}")
    print(f"candidate_seed={report['candidate']['seed']:02d}")
    print(f"candidate_update={report['candidate']['selected_step']}")
    print(f"rollout_status={report['rollout']['status']}")
    print("test_opened=False")
    print("training_runs_started=0")


if __name__ == "__main__":
    main()
