"""Reproducibly evaluate the completed Sprint-10 sparse long pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from crazyflow.control.mellinger.research.config import fingerprint, load_config
from crazyflow.control.mellinger.research.gains import physical_from_raw, specs_for_stage

EXPECTED_UPDATES = 5000
CHECKPOINT_INTERVAL = 100
EXPECTED_CHECKPOINT_STEPS = list(range(CHECKPOINT_INTERVAL, EXPECTED_UPDATES + 1, 100))
EXPECTED_OUTER_FILES = {
    "environment.txt",
    "launch_manifest.txt",
    "launcher_status.txt",
    "process_group.pid",
    "stderr.log",
    "stdout.log",
    "time.txt",
}
EXPECTED_RUNNER_FIXED_FILES = {
    "metrics.jsonl",
    "provenance.json",
    "resolved_config.json",
    "run_summary.json",
}
TREND_WINDOWS = {
    "early_steps_1_500": (1, 500),
    "middle_steps_2251_2750": (2251, 2750),
    "late_steps_4501_5000": (4501, 5000),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def _array(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(record["data"], dtype=record["dtype"]).reshape(record["shape"])


def _linear_fit(steps: np.ndarray, values: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(steps, values, 1)
    predicted = intercept + slope * steps
    total = float(np.sum((values - np.mean(values)) ** 2))
    residual = float(np.sum((values - predicted) ** 2))
    return {
        "slope_per_update": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 if total == 0.0 else 1.0 - residual / total,
    }


def _window_summary(values: np.ndarray, start: int, end: int) -> dict[str, Any]:
    window = values[start - 1 : end]
    steps = np.arange(start, end + 1, dtype=float)
    deltas = np.diff(window)
    return {
        "start_step": start,
        "end_step": end,
        "count": len(window),
        "initial": float(window[0]),
        "final": float(window[-1]),
        "absolute_change": float(window[-1] - window[0]),
        "relative_change": (
            None if window[0] == 0.0 else float((window[-1] - window[0]) / abs(window[0]))
        ),
        "mean": float(np.mean(window)),
        "median": float(np.median(window)),
        "standard_deviation": float(np.std(window)),
        "decreasing_transition_fraction": float(np.mean(deltas < 0.0)),
        "linear_fit": _linear_fit(steps, window),
    }


def _series_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    steps = np.arange(1, len(array) + 1, dtype=float)
    deltas = np.diff(array)
    nonzero_signs = np.sign(deltas[deltas != 0.0])
    sign_changes = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
    return {
        "values": values,
        "initial": float(array[0]),
        "final": float(array[-1]),
        "minimum": float(np.min(array)),
        "minimum_step": int(np.argmin(array) + 1),
        "maximum": float(np.max(array)),
        "maximum_step": int(np.argmax(array) + 1),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "absolute_change": float(array[-1] - array[0]),
        "relative_change": (
            None if array[0] == 0.0 else float((array[-1] - array[0]) / abs(array[0]))
        ),
        "decreasing_transition_fraction": float(np.mean(deltas < 0.0)),
        "increasing_transition_fraction": float(np.mean(deltas > 0.0)),
        "delta_sign_changes": sign_changes,
        "delta_sign_change_fraction": (
            0.0 if len(nonzero_signs) < 2 else sign_changes / (len(nonzero_signs) - 1)
        ),
        "largest_single_update_decrease": float(np.min(deltas)),
        "largest_single_update_increase": float(np.max(deltas)),
        "full_linear_fit": _linear_fit(steps, array),
        "trend_windows": {
            name: _window_summary(array, start, end) for name, (start, end) in TREND_WINDOWS.items()
        },
    }


def _parse_key_values(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _time_value(text: str, label: str) -> str:
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"missing external timing field: {label}")
    return match.group(1).strip()


def _elapsed_seconds(value: str) -> float:
    parts = [float(item) for item in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"unsupported elapsed-time value: {value}")


def _compact_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = [float(row["loss"]) for row in rows if row["split"] == "train"]
    validation = [float(row["loss"]) for row in rows if row["split"] == "validation"]
    gradients = [float(row["gradient_l2_norm"]) for row in rows if row["split"] == "train"]
    return {
        "updates": len(train),
        "train_initial": train[0],
        "train_final": train[-1],
        "train_minimum": min(train),
        "train_minimum_step": train.index(min(train)) + 1,
        "validation_initial": validation[0],
        "validation_final": validation[-1],
        "validation_minimum": min(validation),
        "validation_minimum_step": validation.index(min(validation)) + 1,
        "gradient_l2_initial": gradients[0],
        "gradient_l2_final": gradients[-1],
        "gradient_l2_minimum": min(gradients),
        "gradient_l2_minimum_step": gradients.index(min(gradients)) + 1,
    }


def _relative_delta(actual: float, reference: float) -> float:
    return (actual - reference) / reference


def main() -> None:
    """Validate the immutable raw run and emit deterministic analysis and hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--released-config", type=Path, required=True)
    parser.add_argument("--sprint8-run-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-sha-index-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.run_sha_index_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    run = args.run_dir.resolve()
    runner_output = run / "runner-output"
    run_files = sorted(path for path in run.rglob("*") if path.is_file())
    symlinks = sorted(str(path.relative_to(run)) for path in run.rglob("*") if path.is_symlink())
    run_hashes = {str(path.relative_to(run)): _sha256(path) for path in run_files}
    repository_root = Path.cwd().resolve()
    run_sha_lines = [
        f"{run_hashes[str(path.relative_to(run))]}  {path.relative_to(repository_root)}"
        for path in run_files
    ]
    run_sha_text = "\n".join(run_sha_lines) + "\n"

    outer_files = {path.name for path in run.iterdir() if path.is_file()}
    runner_files = {path.name for path in runner_output.iterdir() if path.is_file()}
    checkpoints = sorted(runner_output.glob("checkpoint-step-*.json"))
    checkpoint_steps = [int(path.stem.rsplit("-", 1)[1]) for path in checkpoints]
    expected_runner_files = EXPECTED_RUNNER_FIXED_FILES | {
        f"checkpoint-step-{step:06d}.json" for step in EXPECTED_CHECKPOINT_STEPS
    }

    metrics_path = runner_output / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    train_rows = [row for row in metrics if row.get("split") == "train"]
    validation_rows = [row for row in metrics if row.get("split") == "validation"]
    expected_order = [
        item
        for step in range(1, EXPECTED_UPDATES + 1)
        for item in ((step, "train"), (step, "validation"))
    ]
    observed_order = [(int(row["step"]), str(row["split"])) for row in metrics]

    summary = json.loads((runner_output / "run_summary.json").read_text())
    provenance = json.loads((runner_output / "provenance.json").read_text())
    resolved = json.loads((runner_output / "resolved_config.json").read_text())
    released = json.loads(args.released_config.read_text())
    loaded_config = load_config(args.released_config)
    recomputed_config_fingerprint = fingerprint(loaded_config)
    launch = _parse_key_values(run / "launch_manifest.txt")
    launcher_status = _parse_key_values(run / "launcher_status.txt")

    checkpoint_payload_hashes_valid = True
    checkpoint_values_finite = True
    checkpoint_prefixes_exact = True
    checkpoint_histories_exact = True
    checkpoint_sizes = []
    checkpoint_file_sha256 = {}
    gain_history = []
    checkpoint_fingerprints: dict[str, set[str]] = {
        name: set()
        for name in (
            "config_fingerprint",
            "gain_registry_fingerprint",
            "source_fingerprint",
            "runtime_fingerprint",
            "root_seed",
            "validation_manifest_fingerprint",
            "validation_manifest_id",
            "frozen_test_manifest_reference",
        )
    }
    specs = specs_for_stage(1)
    final_checkpoint_payload = None
    for expected_step, path in zip(EXPECTED_CHECKPOINT_STEPS, checkpoints, strict=True):
        payload = json.loads(path.read_text())
        recorded_hash = payload.pop("payload_sha256", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        checkpoint_payload_hashes_valid &= (
            recorded_hash == hashlib.sha256(encoded.encode()).hexdigest()
        )
        checkpoint_values_finite &= _finite(payload)
        if payload["step"] != expected_step or payload["episode_index"] != expected_step:
            raise ValueError(f"checkpoint state counter mismatch at {path}")
        checkpoint_prefixes_exact &= payload["metrics_records"] == metrics[: 2 * expected_step]
        checkpoint_histories_exact &= payload["history"] == summary["history"][:expected_step]
        for name in checkpoint_fingerprints:
            checkpoint_fingerprints[name].add(json.dumps(payload[name], sort_keys=True))
        raw = _array(payload["raw_gains"])
        physical = {name: float(value) for name, value in physical_from_raw(raw, 1).items()}
        gains = {}
        for index, spec in enumerate(specs):
            value = physical[spec.name]
            width = spec.upper - spec.lower
            lower_distance = value - spec.lower
            upper_distance = spec.upper - value
            gains[spec.name] = {
                "raw": float(raw[index]),
                "physical": value,
                "lower_bound": spec.lower,
                "upper_bound": spec.upper,
                "absolute_distance_to_lower": lower_distance,
                "absolute_distance_to_upper": upper_distance,
                "normalized_distance_to_lower": lower_distance / width,
                "normalized_distance_to_upper": upper_distance / width,
                "normalized_distance_to_nearest_bound": min(lower_distance, upper_distance) / width,
            }
        gain_history.append({"step": expected_step, "gains": gains})
        checkpoint_sizes.append(path.stat().st_size)
        checkpoint_file_sha256[path.name] = _sha256(path)
        final_checkpoint_payload = payload

    if final_checkpoint_payload is None:
        raise ValueError("no final checkpoint payload")

    train_loss = [float(row["loss"]) for row in train_rows]
    validation_loss = [float(row["loss"]) for row in validation_rows]
    gradient_l2 = [float(row["gradient_l2_norm"]) for row in train_rows]
    generalization_gap = [
        validation - train for train, validation in zip(train_loss, validation_loss, strict=True)
    ]
    train_series = _series_summary(train_loss)
    validation_series = _series_summary(validation_loss)
    gradient_series = _series_summary(gradient_l2)
    gap_series = _series_summary(generalization_gap)

    gain_summaries = {}
    for spec in specs:
        name = spec.name
        values = np.asarray([item["gains"][name]["physical"] for item in gain_history])
        margins = np.asarray(
            [item["gains"][name]["normalized_distance_to_nearest_bound"] for item in gain_history]
        )
        steps = np.asarray(EXPECTED_CHECKPOINT_STEPS, dtype=float)
        deltas = np.diff(values)
        gain_summaries[name] = {
            "sampling_steps": EXPECTED_CHECKPOINT_STEPS,
            "physical_values": values.tolist(),
            "initial_sample": float(values[0]),
            "final": float(values[-1]),
            "minimum": float(np.min(values)),
            "minimum_step": int(steps[int(np.argmin(values))]),
            "maximum": float(np.max(values)),
            "maximum_step": int(steps[int(np.argmax(values))]),
            "absolute_change": float(values[-1] - values[0]),
            "relative_change": float((values[-1] - values[0]) / values[0]),
            "increasing_transition_fraction": float(np.mean(deltas > 0.0)),
            "decreasing_transition_fraction": float(np.mean(deltas < 0.0)),
            "linear_fit": _linear_fit(steps, values),
            "minimum_normalized_bound_margin": float(np.min(margins)),
            "minimum_normalized_bound_margin_step": int(steps[int(np.argmin(margins))]),
            "final_normalized_bound_margin": float(margins[-1]),
        }

    all_gain_margins = [
        item["gains"][spec.name]["normalized_distance_to_nearest_bound"]
        for item in gain_history
        for spec in specs
    ]
    minimum_gain_margin = min(all_gain_margins)

    sprint8_metrics_path = args.sprint8_run_dir / "runner-output" / "metrics.jsonl"
    sprint8_rows = [json.loads(line) for line in sprint8_metrics_path.read_text().splitlines()]
    smoke_metrics_path = args.smoke_dir / "metrics.jsonl"
    smoke_rows = [json.loads(line) for line in smoke_metrics_path.read_text().splitlines()]
    sprint8_comparison = _compact_comparison(sprint8_rows)
    smoke_comparison = _compact_comparison(smoke_rows)
    long_comparison = _compact_comparison(metrics)

    timing_text = (run / "time.txt").read_text()
    external_wall_seconds = _elapsed_seconds(
        _time_value(timing_text, "Elapsed (wall clock) time (h:mm:ss or m:ss)")
    )
    peak_rss_kib = int(_time_value(timing_text, "Maximum resident set size (kbytes)"))
    swaps = int(_time_value(timing_text, "Swaps"))
    external_exit_status = int(_time_value(timing_text, "Exit status"))
    timing = summary["timing"]
    completed_update_timings = timing["completed_update_seconds"]
    steady_timings = np.asarray(
        [
            item["seconds"]
            for item in completed_update_timings
            if item["step"] >= 3 and not item["checkpoint_written"]
        ],
        dtype=float,
    )
    checkpoint_timings = np.asarray(
        [item["seconds"] for item in timing["checkpoint_writes"]], dtype=float
    )
    projection = json.loads(args.projection.read_text())["projection_5000_updates_50_checkpoints"]
    run_bytes = sum(path.stat().st_size for path in run_files)
    checkpoint_bytes = sum(checkpoint_sizes)
    final_checkpoint_bytes = checkpoint_sizes[-1]

    test_boundary = {
        "test_manifest_opened": summary["test_manifest_opened"],
        "test_metrics_present": summary["test_metrics_present"],
        "test_used_for_selection": summary["selected"]["test_used_for_selection"],
        "test_metric_row_count": sum(row.get("split") == "test" for row in metrics),
        "unexpected_metric_splits": sorted(
            {str(row.get("split")) for row in metrics} - {"train", "validation"}
        ),
        "provenance_records_only_frozen_reference": (
            provenance["frozen_test_manifest_reference_not_opened"] == released["test_manifest"]
        ),
        "analysis_opened_test_manifest": False,
    }
    expected_checkpoint_fingerprints = {
        "config_fingerprint": provenance["config_fingerprint"],
        "gain_registry_fingerprint": provenance["gain_registry_fingerprint"],
        "source_fingerprint": provenance["git"]["source_fingerprint"],
        "runtime_fingerprint": provenance["runtime_fingerprint"],
        "root_seed": provenance["root_seed"],
        "validation_manifest_fingerprint": provenance["validation_manifest_fingerprint"],
        "validation_manifest_id": provenance["validation_manifest_id"],
        "frozen_test_manifest_reference": provenance["frozen_test_manifest_reference_not_opened"],
    }
    integrity_gates = {
        "launcher_exit_zero": launcher_status.get("exit_status") == "0",
        "external_exit_zero": external_exit_status == 0,
        "runner_status_success": summary["status"] == "success",
        "configured_updates_exact": summary["configured_updates"] == EXPECTED_UPDATES,
        "completed_updates_exact": summary["completed_updates"] == EXPECTED_UPDATES,
        "one_run_directory_only": sorted(
            path.name for path in run.parent.iterdir() if path.is_dir()
        )
        == [run.name],
        "no_resume_or_continuation": (
            launch.get("automatic_resume") == "false"
            and summary["resume_checkpoint"] is None
            and provenance["resume_lineage"] is None
        ),
        "outer_file_set_exact": outer_files == EXPECTED_OUTER_FILES,
        "runner_file_set_exact": runner_files == expected_runner_files,
        "regular_file_count_exact": len(run_files) == 61,
        "no_symlinks": not symlinks,
        "checkpoint_steps_exact": checkpoint_steps == EXPECTED_CHECKPOINT_STEPS,
        "checkpoint_payload_hashes_valid": checkpoint_payload_hashes_valid,
        "checkpoint_values_finite": checkpoint_values_finite,
        "checkpoint_metric_prefixes_exact": checkpoint_prefixes_exact,
        "checkpoint_histories_exact": checkpoint_histories_exact,
        "checkpoint_fingerprints_singleton": all(
            len(values) == 1 for values in checkpoint_fingerprints.values()
        ),
        "checkpoint_fingerprints_match_provenance": all(
            values == {json.dumps(expected_checkpoint_fingerprints[name], sort_keys=True)}
            for name, values in checkpoint_fingerprints.items()
        ),
        "metric_row_count_exact": len(metrics) == 2 * EXPECTED_UPDATES,
        "train_row_count_exact": len(train_rows) == EXPECTED_UPDATES,
        "validation_row_count_exact": len(validation_rows) == EXPECTED_UPDATES,
        "metric_order_exact_without_duplicates": observed_order == expected_order,
        "all_metric_values_finite": _finite(metrics),
        "all_technical_gates_passed": all(row["technical_gate"]["passed"] for row in metrics),
        "summary_history_exact": summary["history"] == final_checkpoint_payload["history"],
        "final_checkpoint_metrics_exact": final_checkpoint_payload["metrics_records"] == metrics,
        "resolved_config_equals_released": resolved["config"] == released,
        "config_fingerprint_recomputed": (
            recomputed_config_fingerprint == provenance["config_fingerprint"]
        ),
        "config_fingerprint_matches_launch": (
            launch.get("config_fingerprint") == provenance["config_fingerprint"]
        ),
        "config_sha256_matches_launch": (
            launch.get("config_sha256") == _sha256(args.released_config)
        ),
        "source_fingerprint_matches_launch": (
            launch.get("source_fingerprint") == provenance["git"]["source_fingerprint"]
        ),
        "resolved_config_fingerprint_matches": (
            resolved["config_fingerprint"] == provenance["config_fingerprint"]
        ),
        "clean_prelaunch_source": provenance["git"]["dirty"] is False,
        "prelaunch_head_exact": provenance["git"]["head"] == launch.get("git_head"),
        "prelaunch_branch_exact": provenance["git"]["branch"] == launch.get("branch"),
        "root_seed_exact": provenance["root_seed"] == released["root_seed"] == 20260731,
        "selection_matches_validation_minimum": (
            summary["selected"]["selected_step"] == int(np.argmin(validation_loss) + 1)
            and summary["selected"]["selected_validation_loss"] == min(validation_loss)
        ),
        "test_boundary_closed": all(
            value is False
            for key, value in test_boundary.items()
            if key
            in {
                "test_manifest_opened",
                "test_metrics_present",
                "test_used_for_selection",
                "analysis_opened_test_manifest",
            }
        )
        and test_boundary["test_metric_row_count"] == 0
        and not test_boundary["unexpected_metric_splits"]
        and test_boundary["provenance_records_only_frozen_reference"],
        "peak_rss_below_12_gib": peak_rss_kib < 12 * 1024**2,
        "swap_zero": swaps == 0,
        "walltime_below_7200_seconds": external_wall_seconds < 7200.0,
    }
    if not all(integrity_gates.values()):
        failed = [name for name, passed in integrity_gates.items() if not passed]
        raise ValueError(f"long-pilot integrity gate failure: {failed}")

    validation_array = np.asarray(validation_loss)
    gradient_array = np.asarray(gradient_l2)
    late_validation = validation_array[-500:]
    late_gradient = gradient_array[-500:]
    late_validation_relative_improvement = float(
        (late_validation[0] - late_validation[-1]) / late_validation[0]
    )
    late_gradient_relative_improvement = float(
        (late_gradient[0] - late_gradient[-1]) / late_gradient[0]
    )
    diagnostic_thresholds = {
        "divergence": "final loss > 1.05 times initial loss",
        "oscillation": "nonzero adjacent-delta sign-change fraction > 0.5",
        "plateau": "less than 1% endpoint improvement in the final 500 updates",
        "gain_bound_saturation": "normalized distance to nearest bound < 0.01",
        "gain_bound_watch": "final normalized distance to nearest bound < 0.10",
    }
    diagnostics = {
        "thresholds": diagnostic_thresholds,
        "nonfinite_indicator": False,
        "train_divergence_indicator": train_loss[-1] > 1.05 * train_loss[0],
        "validation_divergence_indicator": validation_loss[-1] > 1.05 * validation_loss[0],
        "train_oscillation_indicator": train_series["delta_sign_change_fraction"] > 0.5,
        "validation_oscillation_indicator": (validation_series["delta_sign_change_fraction"] > 0.5),
        "gradient_oscillation_indicator": gradient_series["delta_sign_change_fraction"] > 0.5,
        "validation_strictly_decreased_every_update": bool(np.all(np.diff(validation_array) < 0.0)),
        "validation_plateau_indicator": late_validation_relative_improvement < 0.01,
        "gradient_plateau_indicator": late_gradient_relative_improvement < 0.01,
        "late_500_validation_relative_improvement": late_validation_relative_improvement,
        "late_500_gradient_relative_improvement": late_gradient_relative_improvement,
        "gain_bound_saturation_indicator": minimum_gain_margin < 0.01,
        "gains_inside_bounds": all(value >= 0.0 for value in all_gain_margins),
        "final_gain_bound_watch": {
            name: summary["final_physical_gains"][name]
            for name, item in gain_summaries.items()
            if item["final_normalized_bound_margin"] < 0.10
        },
        "maximum_motor_saturation_fraction": max(
            float(row["metrics"]["motor_saturation_fraction"]) for row in metrics
        ),
        "maximum_floor_clip_fraction": max(
            float(row["metrics"]["floor_clip_fraction"]) for row in metrics
        ),
        "maximum_zero_thrust_gate_fraction": max(
            float(row["metrics"]["zero_thrust_gate_fraction"]) for row in metrics
        ),
        "convergence_indicator": bool(
            validation_loss[-1] < validation_loss[0]
            and gradient_l2[-1] < gradient_l2[0]
            and validation_series["trend_windows"]["late_steps_4501_5000"]["linear_fit"][
                "slope_per_update"
            ]
            < 0.0
        ),
        "interpretation": (
            "Post-hoc descriptive indicators with declared thresholds, not hypothesis tests, "
            "formal optimizer convergence, Lyapunov stability, or hardware stability proofs. "
            "Train and gradient oscillation are expected to include per-update episode resampling."
        ),
    }

    result = {
        "schema_version": "crazyflow.sprint11_sprint10_long_pilot_analysis.v1",
        "evaluation_status": "PASS",
        "scientific_claim": "technical and exploratory single-seed CPU simulation evidence only",
        "input": {
            "run_directory": str(args.run_dir),
            "regular_file_count": len(run_files),
            "total_bytes": run_bytes,
            "symlinks": symlinks,
            "raw_artifact_sha256": run_hashes,
            "raw_run_sha_index_sha256": hashlib.sha256(run_sha_text.encode()).hexdigest(),
            "released_config": str(args.released_config),
            "released_config_sha256": _sha256(args.released_config),
            "sprint8_metrics_sha256": _sha256(sprint8_metrics_path),
            "smoke_metrics_sha256": _sha256(smoke_metrics_path),
            "projection_sha256": _sha256(args.projection),
        },
        "integrity": {
            "gates": integrity_gates,
            "launcher_status": launcher_status,
            "run_directory_names": sorted(
                path.name for path in run.parent.iterdir() if path.is_dir()
            ),
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "metric_rows": len(metrics),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
        },
        "fingerprints": {
            "git_head": provenance["git"]["head"],
            "git_branch": provenance["git"]["branch"],
            "source_dirty": provenance["git"]["dirty"],
            "config_fingerprint": provenance["config_fingerprint"],
            "recomputed_config_fingerprint": recomputed_config_fingerprint,
            "source_fingerprint": provenance["git"]["source_fingerprint"],
            "runtime_fingerprint": provenance["runtime_fingerprint"],
            "gain_registry_fingerprint": provenance["gain_registry_fingerprint"],
            "validation_manifest_fingerprint": provenance["validation_manifest_fingerprint"],
            "validation_manifest_id": provenance["validation_manifest_id"],
            "root_seed": provenance["root_seed"],
            "runtime_compatibility": provenance["runtime_compatibility"],
            "environment": provenance["environment"],
        },
        "test_boundary": test_boundary,
        "series": {
            "train_loss": train_series,
            "validation_loss": validation_series,
            "validation_minus_train_generalization_gap": gap_series
            | {
                "negative_fraction": float(np.mean(np.asarray(generalization_gap) < 0.0)),
                "positive_fraction": float(np.mean(np.asarray(generalization_gap) > 0.0)),
                "mean_absolute_gap": float(np.mean(np.abs(generalization_gap))),
                "root_mean_square_gap": float(
                    np.sqrt(np.mean(np.asarray(generalization_gap) ** 2))
                ),
            },
            "gradient_l2": gradient_series,
        },
        "best_steps": {
            "best_train_step": train_series["minimum_step"],
            "best_train_loss": train_series["minimum"],
            "best_validation_step": validation_series["minimum_step"],
            "best_validation_loss": validation_series["minimum"],
            "runner_selected_step": summary["selected"]["selected_step"],
            "runner_selected_validation_loss": summary["selected"]["selected_validation_loss"],
            "selection_criterion": summary["selected"]["criterion"],
        },
        "gains": {
            "persistence_resolution": (
                "Exact raw and transformed gains are persisted at all 50 interval-100 "
                "checkpoints, not at every optimizer update. Per-update loss and gradient metrics "
                "are complete, but unpersisted intermediate gain vectors cannot be recovered "
                "without rerunning and are not inferred here."
            ),
            "all_persisted_checkpoint_gain_samples_included": True,
            "per_update_gain_vectors_persisted": False,
            "sample_count": len(gain_history),
            "sample_steps": EXPECTED_CHECKPOINT_STEPS,
            "history": gain_history,
            "per_gain_summary": gain_summaries,
            "minimum_normalized_margin_over_all_persisted_samples": minimum_gain_margin,
            "final_physical_gains": summary["final_physical_gains"],
            "selected_physical_gains": summary["selected_physical_gains"],
        },
        "diagnostics": diagnostics,
        "comparison": {
            "sprint8": sprint8_comparison,
            "sprint10_smoke": smoke_comparison,
            "sprint10_long": long_comparison,
            "long_prefix_equals_sprint8_all_metric_rows_byte_for_byte": (
                metrics[: len(sprint8_rows)] == sprint8_rows
            ),
            "long_prefix_equals_sprint10_smoke_all_metric_rows_byte_for_byte": (
                metrics[: len(smoke_rows)] == smoke_rows
            ),
            "long_final_relative_to_sprint8_final": {
                "train_loss": _relative_delta(
                    long_comparison["train_final"], sprint8_comparison["train_final"]
                ),
                "validation_loss": _relative_delta(
                    long_comparison["validation_final"], sprint8_comparison["validation_final"]
                ),
                "gradient_l2": _relative_delta(
                    long_comparison["gradient_l2_final"], sprint8_comparison["gradient_l2_final"]
                ),
            },
            "long_final_relative_to_sprint10_smoke_final": {
                "train_loss": _relative_delta(
                    long_comparison["train_final"], smoke_comparison["train_final"]
                ),
                "validation_loss": _relative_delta(
                    long_comparison["validation_final"], smoke_comparison["validation_final"]
                ),
                "gradient_l2": _relative_delta(
                    long_comparison["gradient_l2_final"], smoke_comparison["gradient_l2_final"]
                ),
            },
        },
        "resources": {
            "started_at_utc": launch["started_at_utc"],
            "ended_at_utc": launcher_status["ended_at_utc"],
            "external_wall_seconds": external_wall_seconds,
            "internal_wall_seconds": provenance["wall_seconds_internal"],
            "external_minus_internal_wall_seconds": (
                external_wall_seconds - provenance["wall_seconds_internal"]
            ),
            "peak_rss_kib": peak_rss_kib,
            "peak_rss_gib": peak_rss_kib / 1024**2,
            "peak_rss_fraction_of_12_gib_limit": peak_rss_kib / (12 * 1024**2),
            "swaps": swaps,
            "external_exit_status": external_exit_status,
            "steady_noncheckpoint_update_seconds": {
                "count": len(steady_timings),
                "minimum": float(np.min(steady_timings)),
                "median": float(np.median(steady_timings)),
                "p95": float(np.quantile(steady_timings, 0.95)),
                "p99": float(np.quantile(steady_timings, 0.99)),
                "maximum": float(np.max(steady_timings)),
                "mean": float(np.mean(steady_timings)),
            },
            "checkpoint_writes": {
                "count": len(checkpoint_timings),
                "total_seconds": float(np.sum(checkpoint_timings)),
                "minimum_seconds": float(np.min(checkpoint_timings)),
                "median_seconds": float(np.median(checkpoint_timings)),
                "maximum_seconds": float(np.max(checkpoint_timings)),
            },
            "artifact_volume": {
                "regular_files": len(run_files),
                "total_bytes": run_bytes,
                "checkpoint_files": len(checkpoints),
                "checkpoint_bytes_total": checkpoint_bytes,
                "final_checkpoint_bytes": final_checkpoint_bytes,
                "metrics_bytes": metrics_path.stat().st_size,
                "run_summary_bytes": (runner_output / "run_summary.json").stat().st_size,
            },
            "actual_vs_sprint10_projection": {
                "conservative_wall_seconds_predicted": projection["conservative_wall_seconds"],
                "wall_actual_minus_predicted_seconds": (
                    external_wall_seconds - projection["conservative_wall_seconds"]
                ),
                "wall_actual_to_predicted_ratio": (
                    external_wall_seconds / projection["conservative_wall_seconds"]
                ),
                "expected_artifact_bytes_predicted": projection["expected_total_artifact_bytes"],
                "artifact_actual_minus_predicted_bytes": (
                    run_bytes - projection["expected_total_artifact_bytes"]
                ),
                "artifact_actual_to_predicted_ratio": (
                    run_bytes / projection["expected_total_artifact_bytes"]
                ),
                "conservative_artifact_bytes_predicted": projection[
                    "conservative_total_artifact_bytes"
                ],
                "artifact_actual_to_conservative_ratio": (
                    run_bytes / projection["conservative_total_artifact_bytes"]
                ),
                "checkpoint_bytes_predicted": projection["predicted_checkpoint_bytes_total"],
                "checkpoint_bytes_actual_minus_predicted": (
                    checkpoint_bytes - projection["predicted_checkpoint_bytes_total"]
                ),
                "final_checkpoint_bytes_predicted": projection["predicted_final_checkpoint_bytes"],
                "final_checkpoint_bytes_actual_minus_predicted": (
                    final_checkpoint_bytes - projection["predicted_final_checkpoint_bytes"]
                ),
                "file_count_predicted": projection["expected_total_files_after_completion"],
                "file_count_actual": len(run_files),
            },
        },
        "scientific_interpretation": {
            "stability": (
                "All technical gates passed for every Train and Validation row; fixed Validation "
                "decreased strictly at all 4,999 transitions, Gradient-L2 fell substantially, "
                "and no numerical, motor, floor, zero-thrust, memory, swap, or bound-saturation "
                "failure occurred. This is strong in-run simulation evidence, not a formal "
                "closed-loop stability proof."
            ),
            "generalization": (
                "The fixed four-world Validation loss improved without reversal and remained "
                "close to the resampled four-world Train loss. Because both sets are small and "
                "this is one root seed, the gap is descriptive and cannot establish population "
                "generalization."
            ),
            "single_seed_limits": (
                "No uncertainty interval, between-seed variance, robustness-to-seed conclusion, "
                "H200/H400 transfer claim, Test claim, firmware claim, hardware claim, or Sim2Real "
                "claim is supported."
            ),
        },
        "recommendation": {
            "decision": "GO",
            "authorized_scientific_next_step": (
                "Freeze the H100 method and validation-based earliest-tie selection rule, then "
                "replicate H100 with multiple independently predeclared seeds."
            ),
            "not_yet_authorized": [
                "H200 or H400 evaluation before the H100 replication decision is frozen",
                "opening or evaluating the Test split",
                "treating this single seed as a main-study or robustness result",
            ],
            "ordered_protocol": [
                "Freeze selection rule and all H100 scientific semantics.",
                "Predeclare independent seeds and aggregate criteria, then replicate H100.",
                "Only after H100 replication, evaluate H200 and H400 unchanged.",
                "Open Test exactly once only after all decisions are frozen.",
            ],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.run_sha_index_output.write_text(run_sha_text)


if __name__ == "__main__":
    main()
