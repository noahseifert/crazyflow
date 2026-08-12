"""Create the reproducible Sprint-9 analysis of the completed Sprint-8 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from crazyflow.control.mellinger.research.gains import physical_from_raw, specs_for_stage


def _array(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(record["data"], dtype=record["dtype"]).reshape(record["shape"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = y - predicted
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total == 0.0 else 1.0 - float(np.sum(residual**2)) / total
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": r_squared}


def _series_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    steps = np.arange(1, len(values) + 1, dtype=float)
    deltas = np.diff(array)
    nonzero_signs = np.sign(deltas[np.nonzero(deltas)])
    sign_changes = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
    tail_count = min(10, len(values))
    return {
        "values": values,
        "initial": float(array[0]),
        "final": float(array[-1]),
        "minimum": float(np.min(array)),
        "minimum_step": int(np.argmin(array) + 1),
        "maximum": float(np.max(array)),
        "maximum_step": int(np.argmax(array) + 1),
        "absolute_change": float(array[-1] - array[0]),
        "relative_change": float((array[-1] - array[0]) / array[0]),
        "decreasing_step_fraction": float(np.mean(deltas < 0.0)),
        "delta_sign_changes": sign_changes,
        "delta_sign_change_fraction": (
            0.0 if len(nonzero_signs) < 2 else sign_changes / (len(nonzero_signs) - 1)
        ),
        "full_linear_fit": _linear_fit(steps, array),
        "last_10_linear_fit": _linear_fit(steps[-tail_count:], array[-tail_count:]),
    }


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


def main() -> None:
    """Analyze one immutable Sprint-8 run and write a strict JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--released-config", type=Path, required=True)
    parser.add_argument("--run-sha-index", type=Path, required=True)
    parser.add_argument("--checkpoint-timing-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    run = args.run_dir.resolve()
    output = run / "runner-output"
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    train_rows = [row for row in metrics if row["split"] == "train"]
    validation_rows = [row for row in metrics if row["split"] == "validation"]
    checkpoints = sorted(output.glob("checkpoint-step-*.json"))
    if len(train_rows) != 50 or len(validation_rows) != 50 or len(checkpoints) != 50:
        raise ValueError("expected exactly 50 Train, Validation, and checkpoint records")

    specs = specs_for_stage(1)
    gain_history = []
    checkpoint_sizes = []
    checkpoint_payload_hashes_valid = True
    checkpoint_fingerprints: dict[str, set[str]] = {
        name: set()
        for name in (
            "config_fingerprint",
            "gain_registry_fingerprint",
            "source_fingerprint",
            "runtime_fingerprint",
            "root_seed",
            "frozen_test_manifest_reference",
        )
    }
    for expected_step, path in enumerate(checkpoints, start=1):
        payload_with_hash = json.loads(path.read_text())
        recorded_hash = payload_with_hash.pop("payload_sha256")
        encoded = json.dumps(
            payload_with_hash, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        checkpoint_payload_hashes_valid &= (
            recorded_hash == hashlib.sha256(encoded.encode()).hexdigest()
        )
        if payload_with_hash["step"] != expected_step:
            raise ValueError("checkpoint sequence mismatch")
        for name in checkpoint_fingerprints:
            checkpoint_fingerprints[name].add(json.dumps(payload_with_hash[name], sort_keys=True))
        raw = _array(payload_with_hash["raw_gains"])
        physical = {name: float(value) for name, value in physical_from_raw(raw, 1).items()}
        gains = {}
        for spec in specs:
            value = physical[spec.name]
            width = spec.upper - spec.lower
            lower_distance = value - spec.lower
            upper_distance = spec.upper - value
            gains[spec.name] = {
                "raw": float(raw[len(gains)]),
                "physical": value,
                "lower_bound": spec.lower,
                "upper_bound": spec.upper,
                "absolute_distance_to_lower": lower_distance,
                "absolute_distance_to_upper": upper_distance,
                "normalized_distance_to_nearest_bound": min(lower_distance, upper_distance) / width,
            }
        gain_history.append({"step": expected_step, "gains": gains})
        checkpoint_sizes.append(path.stat().st_size)

    train_loss = [float(row["loss"]) for row in train_rows]
    validation_loss = [float(row["loss"]) for row in validation_rows]
    gradient_l2 = [float(row["gradient_l2_norm"]) for row in train_rows]
    gap = [
        validation - train for train, validation in zip(train_loss, validation_loss, strict=True)
    ]
    train_summary = _series_summary(train_loss)
    validation_summary = _series_summary(validation_loss)
    gradient_summary = _series_summary(gradient_l2)
    gap_summary = _series_summary(gap)

    size_steps = np.arange(1, 51, dtype=float)
    size_fit = _linear_fit(size_steps, np.asarray(checkpoint_sizes, dtype=float))
    target_steps = np.arange(1, 5001, dtype=float)
    predicted_sizes = size_fit["intercept"] + size_fit["slope"] * target_steps
    predicted_checkpoint_bytes = int(round(float(np.sum(predicted_sizes))))
    predicted_final_checkpoint_bytes = int(round(float(predicted_sizes[-1])))

    timing_capture = json.loads(args.checkpoint_timing_capture.read_text())
    captured_checkpoints = timing_capture["checkpoints"]
    if [item["name"] for item in captured_checkpoints] != [path.name for path in checkpoints]:
        raise ValueError("checkpoint timing capture sequence mismatch")
    mtimes = np.asarray([int(item["mtime_ns"]) / 1.0e9 for item in captured_checkpoints])
    complete_update_intervals = np.diff(mtimes)
    interval_steps = np.arange(2, 51, dtype=float)
    runtime_fits = {}
    projected_totals = {}
    provenance = json.loads((output / "provenance.json").read_text())
    started_timestamp = datetime.fromisoformat(provenance["started_at_utc"]).timestamp()
    first_complete_seconds = float(mtimes[0] - started_timestamp)
    external_timing = (run / "time.txt").read_text()
    external_wall_seconds = _elapsed_seconds(
        _time_value(external_timing, "Elapsed (wall clock) time (h:mm:ss or m:ss)")
    )
    post_checkpoint_seconds = external_wall_seconds - float(mtimes[-1] - started_timestamp)
    for fit_start in (3, 10, 20):
        mask = interval_steps >= fit_start
        fit = _linear_fit(interval_steps[mask], complete_update_intervals[mask])
        projected_intervals = fit["intercept"] + fit["slope"] * np.arange(2, 5001)
        runtime_fits[str(fit_start)] = fit
        projected_totals[str(fit_start)] = float(
            first_complete_seconds + np.sum(projected_intervals) + post_checkpoint_seconds
        )

    final_normalized_margins = {
        name: gain_history[-1]["gains"][name]["normalized_distance_to_nearest_bound"]
        for name in (spec.name for spec in specs)
    }
    minimum_normalized_margin = min(
        record["gains"][spec.name]["normalized_distance_to_nearest_bound"]
        for record in gain_history
        for spec in specs
    )
    summary = json.loads((output / "run_summary.json").read_text())
    resolved = json.loads((output / "resolved_config.json").read_text())
    released = json.loads(args.released_config.read_text())
    run_files = sorted(path for path in run.rglob("*") if path.is_file())
    artifact_hashes = {str(path.relative_to(run)): _sha256(path) for path in run_files}
    estimated_metrics_bytes = round((output / "metrics.jsonl").stat().st_size / 50 * 5000)
    estimated_fixed_bytes = sum(
        path.stat().st_size
        for path in run_files
        if "checkpoint-step-" not in path.name and path.name != "metrics.jsonl"
    )
    estimated_total_bytes = (
        predicted_checkpoint_bytes + estimated_metrics_bytes + estimated_fixed_bytes
    )
    technical_metric_values = [row["metrics"] for row in metrics]
    diagnostics = {
        "nonfinite_detected": not all(
            math.isfinite(value)
            for series in (train_loss, validation_loss, gradient_l2, gap)
            for value in series
        ),
        "divergence_indicator": (
            train_loss[-1] > 1.05 * train_loss[0] or validation_loss[-1] > 1.05 * validation_loss[0]
        ),
        "gain_bound_saturation_indicator": minimum_normalized_margin < 0.01,
        "technical_saturation_max": max(
            float(item["motor_saturation_fraction"]) for item in technical_metric_values
        ),
        "technical_gate_failure_count": sum(not row["technical_gate"]["passed"] for row in metrics),
        "train_plateau_indicator": abs(train_summary["last_10_linear_fit"]["slope"])
        < 1.0e-4 * abs(train_loss[-1]),
        "validation_plateau_indicator": abs(validation_summary["last_10_linear_fit"]["slope"])
        < 1.0e-4 * abs(validation_loss[-1]),
        "train_oscillation_indicator": train_summary["delta_sign_change_fraction"] > 0.5,
        "validation_oscillation_indicator": (
            validation_summary["delta_sign_change_fraction"] > 0.5
        ),
        "interpretation": (
            "Post-hoc technical indicators with explicit thresholds; they are not scientific "
            "hypothesis tests or stability proofs."
        ),
    }
    result = {
        "schema_version": "crazyflow.sprint9_sprint8_pilot_analysis.v1",
        "scientific_claim": "technical and exploratory single-seed simulation evidence only",
        "input": {
            "run_directory": str(args.run_dir),
            "file_count": len(run_files),
            "file_bytes": sum(path.stat().st_size for path in run_files),
            "run_sha_index": str(args.run_sha_index),
            "run_sha_index_sha256": _sha256(args.run_sha_index),
            "checkpoint_timing_capture": str(args.checkpoint_timing_capture),
            "checkpoint_timing_capture_sha256": _sha256(args.checkpoint_timing_capture),
            "artifact_sha256": artifact_hashes,
        },
        "integrity": {
            "checkpoint_count": len(checkpoints),
            "checkpoint_payload_hashes_valid": checkpoint_payload_hashes_valid,
            "checkpoint_fingerprints_singleton": all(
                len(values) == 1 for values in checkpoint_fingerprints.values()
            ),
            "metric_rows": len(metrics),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "unexpected_splits": sorted(
                {str(row["split"]) for row in metrics} - {"train", "validation"}
            ),
            "all_technical_gates_passed": all(row["technical_gate"]["passed"] for row in metrics),
        },
        "fingerprints": {
            "git_head": provenance["git"]["head"],
            "git_branch": provenance["git"]["branch"],
            "source_dirty": provenance["git"]["dirty"],
            "config_fingerprint": provenance["config_fingerprint"],
            "source_fingerprint": provenance["git"]["source_fingerprint"],
            "runtime_fingerprint": provenance["runtime_fingerprint"],
            "gain_registry_fingerprint": provenance["gain_registry_fingerprint"],
            "root_seed": provenance["root_seed"],
            "released_config_sha256": _sha256(args.released_config),
            "resolved_config_equals_released_config": resolved["config"] == released,
        },
        "test_boundary": {
            "test_manifest_opened": summary["test_manifest_opened"],
            "test_metrics_present": summary["test_metrics_present"],
            "test_metric_row_count": sum(row["split"] == "test" for row in metrics),
            "frozen_reference_only": True,
        },
        "series": {
            "train_loss": train_summary,
            "validation_loss": validation_summary,
            "validation_minus_train_gap": gap_summary,
            "gradient_l2": gradient_summary,
        },
        "best_steps": {
            "best_train_step": train_summary["minimum_step"],
            "best_validation_step": validation_summary["minimum_step"],
            "runner_selected_validation_step": summary["selected"]["selected_step"],
        },
        "gain_history": gain_history,
        "gain_bound_summary": {
            "minimum_normalized_margin_over_all_steps_and_gains": minimum_normalized_margin,
            "final_normalized_margins": final_normalized_margins,
        },
        "diagnostics": diagnostics,
        "frequency": {
            "metrics_per_update": {"train": 1, "validation": 1},
            "checkpoints_per_update": 1,
            "runner_checkpoint_frequency_configurable": False,
            "first_complete_update_seconds": first_complete_seconds,
            "complete_update_intervals_seconds_steps_2_to_50": complete_update_intervals.tolist(),
            "steady_interval_median_seconds_steps_3_to_50": float(
                np.median(complete_update_intervals[1:])
            ),
            "external_wall_seconds": external_wall_seconds,
            "checkpoint_mtime_caveat": (
                "Original filesystem mtimes are operational observations captured verbatim in "
                "the separately hashed timing-capture input; they are not scientific metrics."
            ),
        },
        "checkpoint_scaling": {
            "observed_checkpoint_sizes_bytes": checkpoint_sizes,
            "size_linear_fit": size_fit,
            "predicted_step_5000_checkpoint_bytes": predicted_final_checkpoint_bytes,
            "predicted_5000_checkpoint_bytes_total": predicted_checkpoint_bytes,
            "predicted_metrics_bytes": estimated_metrics_bytes,
            "predicted_fixed_bytes": estimated_fixed_bytes,
            "predicted_total_artifact_bytes": estimated_total_bytes,
            "predicted_file_count": 5011,
            "runtime_linear_fits_by_first_included_step": runtime_fits,
            "predicted_total_wall_seconds_by_fit": projected_totals,
            "hard_wall_limit_seconds": 7200,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
