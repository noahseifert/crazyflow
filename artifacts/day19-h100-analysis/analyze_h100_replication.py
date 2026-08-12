"""Reproduce and verify the frozen ten-seed H100 validation synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_SEEDS = 10
EXPECTED_UPDATES = 5000
EXPECTED_METRIC_ROWS = 2 * EXPECTED_UPDATES
EXPECTED_CHECKPOINT_STEPS = tuple(range(100, EXPECTED_UPDATES + 1, 100))
EXPECTED_RUN_FILE_COUNT = 61
EXPECTED_INDEXED_FILE_COUNT = 60
EXPECTED_HEAD = "ff85c8e9c0e73bcf97dfd5f2a747aa374a834f91"
EXPECTED_BRANCH = "research/differentiable-mellinger"
T_CRITICAL_95_DF9 = 2.2621571627409915
CHECKPOINT_NAME = re.compile(r"checkpoint-step-(\d{6})\.json")
INDEX_LINE = re.compile(r"([0-9a-f]{64})  \./(.+)")
TECHNICAL_METRICS = (
    "motor_saturation_fraction",
    "floor_clip_fraction",
    "zero_thrust_gate_fraction",
    "nonfinite_state_fraction",
    "max_position_error_m",
)
EXPECTED_OUTER_FILES = {
    "environment.txt",
    "launch_manifest.txt",
    "launcher_status.txt",
    "stderr.log",
    "stdout.log",
    "time.txt",
    "RUN_SHA256SUMS",
}
EXPECTED_RUNNER_FIXED_FILES = {
    "metrics.jsonl",
    "provenance.json",
    "resolved_config.json",
    "run_summary.json",
}
OUTPUT_JSON = "h100_replication_analysis.json"
SOURCE_INDEX = "SOURCE_RUN_INDEXES.sha256"
OUTPUT_INDEX = "SHA256SUMS"


class AnalysisError(RuntimeError):
    """Raised when frozen input or derived evidence violates the analysis contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_document(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise AnalysisError(f"expected JSON object: {path}")
    return payload


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def _parse_key_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _external_time_value(text: str, label: str) -> str:
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise AnalysisError(f"missing external timing field: {label}")
    return match.group(1).strip()


def _elapsed_seconds(value: str) -> float:
    components = [float(component) for component in value.split(":")]
    if len(components) == 2:
        return components[0] * 60.0 + components[1]
    if len(components) == 3:
        return components[0] * 3600.0 + components[1] * 60.0 + components[2]
    raise AnalysisError(f"unsupported elapsed-time value: {value}")


def _aggregate(values: list[float]) -> dict[str, Any]:
    if len(values) != EXPECTED_SEEDS:
        raise AnalysisError(f"expected {EXPECTED_SEEDS} aggregate values, got {len(values)}")
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    half_width = T_CRITICAL_95_DF9 * sample_sd / math.sqrt(EXPECTED_SEEDS)
    return {
        "values_in_manifest_order": values,
        "count": EXPECTED_SEEDS,
        "mean": mean,
        "sample_standard_deviation_ddof_1": sample_sd,
        "student_t_critical_0_975_df_9": T_CRITICAL_95_DF9,
        "confidence_interval_95_percent_student_t": [mean - half_width, mean + half_width],
        "confidence_interval_half_width": half_width,
        "median": float(np.median(array)),
        "q1_linear": float(np.quantile(array, 0.25, method="linear")),
        "q3_linear": float(np.quantile(array, 0.75, method="linear")),
    }


def _analyze_metric_rows(
    metrics: list[dict[str, Any]], *, expected_updates: int = EXPECTED_UPDATES
) -> dict[str, Any]:
    if any(row.get("split") == "test" for row in metrics):
        raise AnalysisError("Test metric row encountered")
    expected_order = [
        (step, split)
        for step in range(1, expected_updates + 1)
        for split in ("train", "validation")
    ]
    observed_order = [(int(row["step"]), str(row["split"])) for row in metrics]
    if observed_order != expected_order:
        raise AnalysisError("metric order, step coverage, split coverage, or uniqueness mismatch")
    if not _finite(metrics):
        raise AnalysisError("nonfinite metric value encountered")
    if not all(bool(row["technical_gate"]["passed"]) for row in metrics):
        raise AnalysisError("failed Train/Validation technical gate encountered")

    train = [row for row in metrics if row["split"] == "train"]
    validation = [row for row in metrics if row["split"] == "validation"]
    validation_losses = np.asarray([float(row["loss"]) for row in validation], dtype=float)
    selected_offset = int(np.argmin(validation_losses))
    selected_step = selected_offset + 1
    selected_loss = float(validation_losses[selected_offset])
    technical_maxima = {
        name: max(float(row["metrics"][name]) for row in metrics) for name in TECHNICAL_METRICS
    }
    return {
        "metric_row_count": len(metrics),
        "train_row_count": len(train),
        "validation_row_count": len(validation),
        "test_row_count": 0,
        "selected_step": selected_step,
        "selected_validation_loss": selected_loss,
        "validation_loss_at_step_1": float(validation_losses[0]),
        "validation_loss_at_step_5000": float(validation_losses[-1]),
        "relative_validation_improvement": 1.0 - selected_loss / float(validation_losses[0]),
        "validation_strictly_decreased_every_update": bool(
            np.all(np.diff(validation_losses) < 0.0)
        ),
        "train_loss_at_selected_step": float(train[selected_offset]["loss"]),
        "train_loss_at_step_5000": float(train[-1]["loss"]),
        "gradient_l2_norm_at_selected_step": float(train[selected_offset]["gradient_l2_norm"]),
        "gradient_l2_norm_at_step_5000": float(train[-1]["gradient_l2_norm"]),
        "technical_metric_maxima": technical_maxima,
    }


def _validate_test_boundary(
    summary: dict[str, Any], provenance: dict[str, Any], released_config: dict[str, Any]
) -> dict[str, Any]:
    frozen_reference = str(released_config["test_manifest"])
    if summary["test_manifest_opened"] is not False:
        raise AnalysisError("run summary reports an opened Test manifest")
    if summary["test_metrics_present"] is not False:
        raise AnalysisError("run summary reports Test metrics")
    if summary["selected"]["test_used_for_selection"] is not False:
        raise AnalysisError("run summary reports Test use for selection")
    if provenance["frozen_test_manifest_reference_not_opened"] != frozen_reference:
        raise AnalysisError("opaque Test reference mismatch")
    return {
        "frozen_reference_recorded_but_not_opened": frozen_reference,
        "run_summary_test_manifest_opened": False,
        "run_summary_test_metrics_present": False,
        "test_used_for_selection": False,
        "analysis_opened_test_manifest": False,
    }


def _parse_run_index(index_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in index_path.read_text().splitlines():
        match = INDEX_LINE.fullmatch(line)
        if match is None:
            raise AnalysisError(f"invalid RUN_SHA256SUMS line in {index_path}: {line!r}")
        digest, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AnalysisError(f"unsafe indexed path in {index_path}: {relative}")
        if relative in entries:
            raise AnalysisError(f"duplicate indexed path in {index_path}: {relative}")
        entries[relative] = digest
    return entries


def _validate_checkpoint(
    payload_bytes: bytes, path: Path, expected_fingerprints: dict[str, Any]
) -> tuple[int, str | None, str | None]:
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict):
        raise AnalysisError(f"checkpoint is not a JSON object: {path}")
    recorded_hash = payload.pop("payload_sha256", None)
    if recorded_hash != _sha256_bytes(_canonical_json(payload).encode()):
        raise AnalysisError(f"checkpoint payload checksum mismatch: {path}")
    match = CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise AnalysisError(f"invalid checkpoint filename: {path}")
    step = int(match.group(1))
    if step not in EXPECTED_CHECKPOINT_STEPS:
        raise AnalysisError(f"unexpected checkpoint step: {path}")
    if payload["step"] != step or payload["episode_index"] != step:
        raise AnalysisError(f"checkpoint counter mismatch: {path}")
    for name, expected in expected_fingerprints.items():
        if payload[name] != expected:
            raise AnalysisError(f"checkpoint {name} mismatch: {path}")
    if len(payload["history"]) != step or len(payload["metrics_records"]) != 2 * step:
        raise AnalysisError(f"checkpoint history/metric prefix length mismatch: {path}")
    for offset, record in enumerate(payload["metrics_records"]):
        expected_step = offset // 2 + 1
        expected_split = "train" if offset % 2 == 0 else "validation"
        if record["step"] != expected_step or record["split"] != expected_split:
            raise AnalysisError(f"checkpoint metric order mismatch: {path}")
    if payload["history"] and payload["history"][-1]["step"] != step:
        raise AnalysisError(f"checkpoint history end-step mismatch: {path}")
    if not _finite(payload):
        raise AnalysisError(f"nonfinite checkpoint payload: {path}")
    if step != EXPECTED_UPDATES:
        return step, None, None
    return (
        step,
        _sha256_bytes(_canonical_json(payload["metrics_records"]).encode()),
        _sha256_bytes(_canonical_json(payload["history"]).encode()),
    )


def _verify_indexed_run(run_dir: Path, expected_fingerprints: dict[str, Any]) -> dict[str, Any]:
    index_path = run_dir / "RUN_SHA256SUMS"
    entries = _parse_run_index(index_path)
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    symlinks = sorted(path for path in run_dir.rglob("*") if path.is_symlink())
    if symlinks:
        raise AnalysisError(f"symlinks are forbidden in raw run: {run_dir}")
    if len(files) != EXPECTED_RUN_FILE_COUNT:
        raise AnalysisError(f"expected {EXPECTED_RUN_FILE_COUNT} run files: {run_dir}")
    expected_relatives = {
        path.relative_to(run_dir).as_posix() for path in files if path != index_path
    }
    if len(entries) != EXPECTED_INDEXED_FILE_COUNT or set(entries) != expected_relatives:
        raise AnalysisError(f"RUN_SHA256SUMS coverage mismatch: {index_path}")

    captures: dict[str, bytes] = {}
    checkpoint_steps: list[int] = []
    final_checkpoint_metrics_sha256: str | None = None
    final_checkpoint_history_sha256: str | None = None
    for relative in sorted(entries):
        path = run_dir / relative
        payload = path.read_bytes()
        if _sha256_bytes(payload) != entries[relative]:
            raise AnalysisError(f"raw-file SHA-256 mismatch: {path}")
        if path.parent.name == "runner-output" and CHECKPOINT_NAME.fullmatch(path.name):
            step, metrics_sha256, history_sha256 = _validate_checkpoint(
                payload, path, expected_fingerprints
            )
            checkpoint_steps.append(step)
            if metrics_sha256 is not None:
                final_checkpoint_metrics_sha256 = metrics_sha256
                final_checkpoint_history_sha256 = history_sha256
        elif relative in {
            "launch_manifest.txt",
            "launcher_status.txt",
            "time.txt",
            "runner-output/metrics.jsonl",
            "runner-output/provenance.json",
            "runner-output/resolved_config.json",
            "runner-output/run_summary.json",
        }:
            captures[relative] = payload
    if tuple(checkpoint_steps) != EXPECTED_CHECKPOINT_STEPS:
        raise AnalysisError(f"checkpoint sequence mismatch: {run_dir}")
    if final_checkpoint_metrics_sha256 is None or final_checkpoint_history_sha256 is None:
        raise AnalysisError(f"final checkpoint comparison payload missing: {run_dir}")
    return {
        "index_path": index_path,
        "index_sha256": _sha256_path(index_path),
        "indexed_file_count": len(entries),
        "run_file_count": len(files),
        "run_bytes": sum(path.stat().st_size for path in files),
        "checkpoint_payload_hashes_verified": len(checkpoint_steps),
        "final_checkpoint_metrics_sha256": final_checkpoint_metrics_sha256,
        "final_checkpoint_history_sha256": final_checkpoint_history_sha256,
        "captures": captures,
        "outer_files": {path.name for path in run_dir.iterdir() if path.is_file()},
        "runner_files": {
            path.name for path in (run_dir / "runner-output").iterdir() if path.is_file()
        },
    }


def _selected_gain_margins(
    selected_gains: dict[str, Any], gain_specs: list[dict[str, Any]]
) -> tuple[dict[str, Any], float]:
    specs = {str(spec["name"]): spec for spec in gain_specs}
    if set(selected_gains) != set(specs):
        raise AnalysisError("selected gain names differ from frozen gain registry")
    result: dict[str, Any] = {}
    minimum = math.inf
    for name in sorted(selected_gains):
        value = float(selected_gains[name])
        lower = float(specs[name]["lower"])
        upper = float(specs[name]["upper"])
        margin = min(value - lower, upper - value) / (upper - lower)
        if not 0.0 <= margin <= 0.5:
            raise AnalysisError(f"selected gain outside frozen bounds: {name}")
        result[name] = {
            "value": value,
            "lower_bound": lower,
            "upper_bound": upper,
            "normalized_nearest_bound_distance": margin,
        }
        minimum = min(minimum, margin)
    return result, minimum


def _decode_json_capture(captures: dict[str, bytes], name: str) -> dict[str, Any]:
    payload = json.loads(captures[name])
    if not isinstance(payload, dict):
        raise AnalysisError(f"expected JSON object: {name}")
    return payload


def _analyze_seed(
    repository_root: Path,
    protocol_dir: Path,
    runs_root: Path,
    seed_record: dict[str, Any],
    inventory_record: dict[str, Any],
    requirements: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    seed_index = int(seed_record["index"])
    root_seed = int(seed_record["root_seed"])
    seed_dir = runs_root / f"seed-{seed_index:02d}"
    run_dirs = sorted(path for path in seed_dir.iterdir() if path.is_dir())
    if len(run_dirs) != 1:
        raise AnalysisError(f"expected exactly one run directory for seed {seed_index:02d}")
    run_dir = run_dirs[0]
    released_path = protocol_dir / "configs" / f"seed-{seed_index:02d}.json"
    released = _load_json(released_path)
    expected_checkpoint_fingerprints = {
        "config_fingerprint": inventory_record["config_fingerprint"],
        "gain_registry_fingerprint": requirements["required_gain_registry_fingerprint"],
        "source_fingerprint": requirements["required_source_fingerprint"],
        "runtime_fingerprint": requirements["required_runtime_fingerprint"],
        "root_seed": root_seed,
        "validation_manifest_fingerprint": requirements["required_validation_manifest_fingerprint"],
        "validation_manifest_id": requirements["required_validation_manifest_id"],
        "frozen_test_manifest_reference": released["test_manifest"],
    }
    verified = _verify_indexed_run(run_dir, expected_checkpoint_fingerprints)
    expected_runner_files = EXPECTED_RUNNER_FIXED_FILES | {
        f"checkpoint-step-{step:06d}.json" for step in EXPECTED_CHECKPOINT_STEPS
    }
    if verified["outer_files"] != EXPECTED_OUTER_FILES:
        raise AnalysisError(f"outer file-set mismatch: {run_dir}")
    if verified["runner_files"] != expected_runner_files:
        raise AnalysisError(f"runner file-set mismatch: {run_dir}")

    captures = verified["captures"]
    metrics = [
        json.loads(line) for line in captures["runner-output/metrics.jsonl"].decode().splitlines()
    ]
    metric_analysis = _analyze_metric_rows(metrics)
    summary = _decode_json_capture(captures, "runner-output/run_summary.json")
    provenance = _decode_json_capture(captures, "runner-output/provenance.json")
    resolved = _decode_json_capture(captures, "runner-output/resolved_config.json")
    launch = _parse_key_values(captures["launch_manifest.txt"].decode())
    launcher_status = _parse_key_values(captures["launcher_status.txt"].decode())
    time_text = captures["time.txt"].decode()

    if summary["status"] != "success":
        raise AnalysisError(f"runner did not report success: {run_dir}")
    if summary["configured_updates"] != EXPECTED_UPDATES:
        raise AnalysisError(f"configured update mismatch: {run_dir}")
    if summary["completed_updates"] != EXPECTED_UPDATES:
        raise AnalysisError(f"incomplete update count: {run_dir}")
    if summary["selected"]["selected_step"] != metric_analysis["selected_step"]:
        raise AnalysisError(f"summary selected-step mismatch: {run_dir}")
    if (
        summary["selected"]["selected_validation_loss"]
        != metric_analysis["selected_validation_loss"]
    ):
        raise AnalysisError(f"summary selected-loss mismatch: {run_dir}")
    if verified["final_checkpoint_metrics_sha256"] != _sha256_bytes(
        _canonical_json(metrics).encode()
    ):
        raise AnalysisError(f"final checkpoint/metrics file mismatch: {run_dir}")
    if verified["final_checkpoint_history_sha256"] != _sha256_bytes(
        _canonical_json(summary["history"]).encode()
    ):
        raise AnalysisError(f"final checkpoint/summary history mismatch: {run_dir}")
    if resolved["config"] != released:
        raise AnalysisError(f"resolved/released config mismatch: {run_dir}")
    if _sha256_path(released_path) != inventory_record["sha256"]:
        raise AnalysisError(f"released config SHA-256 mismatch: {released_path}")
    if provenance["config_fingerprint"] != inventory_record["config_fingerprint"]:
        raise AnalysisError(f"config fingerprint mismatch: {run_dir}")
    if provenance["root_seed"] != root_seed or released["root_seed"] != root_seed:
        raise AnalysisError(f"root-seed mismatch: {run_dir}")
    git_provenance = provenance["git"]
    if (
        git_provenance["branch"] != EXPECTED_BRANCH
        or git_provenance["dirty"] is not False
        or git_provenance["head"] != EXPECTED_HEAD
        or git_provenance["source_fingerprint"] != requirements["required_source_fingerprint"]
        or git_provenance["status_short"] != []
        or git_provenance["repository"] != str(repository_root)
    ):
        raise AnalysisError(f"Git/source provenance mismatch: {run_dir}")
    if provenance["runtime_fingerprint"] != requirements["required_runtime_fingerprint"]:
        raise AnalysisError(f"runtime fingerprint mismatch: {run_dir}")
    if (
        provenance["gain_registry_fingerprint"]
        != requirements["required_gain_registry_fingerprint"]
    ):
        raise AnalysisError(f"gain-registry fingerprint mismatch: {run_dir}")
    if (
        provenance["validation_manifest_fingerprint"]
        != requirements["required_validation_manifest_fingerprint"]
    ):
        raise AnalysisError(f"Validation fingerprint mismatch: {run_dir}")
    if provenance["validation_manifest_id"] != requirements["required_validation_manifest_id"]:
        raise AnalysisError(f"Validation manifest ID mismatch: {run_dir}")
    if summary["resume_checkpoint"] is not None or provenance["resume_lineage"] is not None:
        raise AnalysisError(f"unexpected Resume lineage: {run_dir}")
    if launch.get("automatic_resume") != "false":
        raise AnalysisError(f"launch manifest does not prohibit automatic Resume: {run_dir}")
    if launcher_status.get("exit_status") != "0":
        raise AnalysisError(f"launcher exit status is nonzero: {run_dir}")
    if _external_time_value(time_text, "Exit status") != "0":
        raise AnalysisError(f"external process exit status is nonzero: {run_dir}")
    if launch.get("config_sha256") != _sha256_path(released_path):
        raise AnalysisError(f"launch/config SHA-256 mismatch: {run_dir}")

    test_boundary = _validate_test_boundary(summary, provenance, released)
    gains, minimum_margin = _selected_gain_margins(
        summary["selected_physical_gains"], requirements["scientific_semantics"]["gains"]
    )
    relative_run_path = run_dir.relative_to(repository_root).as_posix()
    relative_index_path = verified["index_path"].relative_to(repository_root).as_posix()
    source_index_line = f"{verified['index_sha256']}  {relative_index_path}\n"
    normalized_config = dict(released)
    normalized_config.pop("run_id")
    normalized_config.pop("root_seed")
    result = {
        "index": seed_index,
        "root_seed": root_seed,
        "run_id": run_dir.name,
        "run_path": relative_run_path,
        "run_file_count": verified["run_file_count"],
        "run_bytes": verified["run_bytes"],
        "run_sha256_index": {
            "path": relative_index_path,
            "sha256": verified["index_sha256"],
            "indexed_file_count": verified["indexed_file_count"],
            "all_indexed_hashes_verified": True,
        },
        "checkpoint_payload_hashes_verified": verified["checkpoint_payload_hashes_verified"],
        "selection": {
            "candidate_steps": [1, EXPECTED_UPDATES],
            "rule": "global fixed-Validation minimum; earliest exact tie",
            "validation_loss_at_step_1": metric_analysis["validation_loss_at_step_1"],
            "selected_step": metric_analysis["selected_step"],
            "selected_validation_loss": metric_analysis["selected_validation_loss"],
            "validation_loss_at_step_5000": metric_analysis["validation_loss_at_step_5000"],
            "relative_validation_improvement": metric_analysis["relative_validation_improvement"],
            "validation_strictly_decreased_every_update": metric_analysis[
                "validation_strictly_decreased_every_update"
            ],
        },
        "train": {
            "loss_at_selected_step": metric_analysis["train_loss_at_selected_step"],
            "loss_at_step_5000": metric_analysis["train_loss_at_step_5000"],
            "gradient_l2_norm_at_selected_step": metric_analysis[
                "gradient_l2_norm_at_selected_step"
            ],
            "gradient_l2_norm_at_step_5000": metric_analysis["gradient_l2_norm_at_step_5000"],
        },
        "selected_physical_gains": gains,
        "minimum_selected_gain_normalized_bound_distance": minimum_margin,
        "technical_metric_maxima": metric_analysis["technical_metric_maxima"],
        "resources": {
            "external_wall_seconds": _elapsed_seconds(
                _external_time_value(time_text, "Elapsed (wall clock) time (h:mm:ss or m:ss)")
            ),
            "peak_rss_kib": int(
                _external_time_value(time_text, "Maximum resident set size (kbytes)")
            ),
            "swaps": int(_external_time_value(time_text, "Swaps")),
        },
        "provenance": {
            "git_branch": provenance["git"]["branch"],
            "git_head": provenance["git"]["head"],
            "scientific_source_dirty": provenance["git"]["dirty"],
            "source_fingerprint": provenance["git"]["source_fingerprint"],
            "runtime_fingerprint": provenance["runtime_fingerprint"],
            "gain_registry_fingerprint": provenance["gain_registry_fingerprint"],
            "validation_manifest_fingerprint": provenance["validation_manifest_fingerprint"],
            "validation_manifest_id": provenance["validation_manifest_id"],
            "released_config_path": released_path.relative_to(repository_root).as_posix(),
            "released_config_sha256": _sha256_path(released_path),
        },
        "metric_rows": {
            "total": metric_analysis["metric_row_count"],
            "train": metric_analysis["train_row_count"],
            "validation": metric_analysis["validation_row_count"],
            "test": metric_analysis["test_row_count"],
        },
        "test_boundary": test_boundary,
    }
    return result, source_index_line, normalized_config


def _gate_summary(seeds: list[dict[str, Any]], aggregates: dict[str, Any]) -> dict[str, Any]:
    improvements = [seed["selection"]["relative_validation_improvement"] for seed in seeds]
    selected_losses = aggregates["selected_validation_loss"]
    improvement_summary = aggregates["relative_validation_improvement"]
    minimum_margin = min(seed["minimum_selected_gain_normalized_bound_distance"] for seed in seeds)
    data_gates = {
        "ten_of_ten_predeclared_seeds_complete_and_integrity_valid": len(seeds) == EXPECTED_SEEDS,
        "all_relative_validation_improvements_positive": all(value > 0.0 for value in improvements),
        "at_least_eight_improvements_at_least_0_50": sum(value >= 0.50 for value in improvements)
        >= 8,
        "improvement_ci_lower_bound_at_least_0_50": improvement_summary[
            "confidence_interval_95_percent_student_t"
        ][0]
        >= 0.50,
        "selected_loss_relative_ci_half_width_at_most_0_25": selected_losses[
            "confidence_interval_half_width"
        ]
        / selected_losses["mean"]
        <= 0.25,
        "all_final_validation_losses_at_most_1_10_times_selected": all(
            seed["selection"]["validation_loss_at_step_5000"]
            <= 1.10 * seed["selection"]["selected_validation_loss"]
            for seed in seeds
        ),
        "all_selected_gain_bound_distances_at_least_0_01": minimum_margin >= 0.01,
        "all_test_boundaries_closed": all(seed["metric_rows"]["test"] == 0 for seed in seeds),
        "all_raw_run_indexes_and_checkpoint_payload_hashes_verified": all(
            seed["run_sha256_index"]["all_indexed_hashes_verified"]
            and seed["checkpoint_payload_hashes_verified"] == len(EXPECTED_CHECKPOINT_STEPS)
            for seed in seeds
        ),
    }
    if not all(data_gates.values()):
        raise AnalysisError("one or more data-derived frozen H100 gates failed")
    return {
        "data_derived_gates": data_gates,
        "data_derived_gates_all_pass": True,
        "execution_protocol_conformity_gate": {
            "status": "NOT_DERIVABLE_FROM_RAW_RUNS_ALONE",
            "reason": (
                "The frozen gate also requires absence of unplanned execution-protocol change. "
                "That governance determination needs authorization/deviation evidence and is "
                "therefore kept outside this numerical synthesis."
            ),
        },
        "h200_h400_authorization": "NOT_GRANTED_BY_THIS_ANALYSIS",
    }


def _build_analysis(
    repository_root: Path, protocol_dir: Path, runs_root: Path
) -> tuple[dict[str, Any], str]:
    protocol = _load_json(protocol_dir / "protocol.json")
    analysis_plan = _load_json(protocol_dir / "analysis_plan.json")
    requirements = _load_json(protocol_dir / "requirements.json")
    seed_manifest = _load_json(protocol_dir / "seed_manifest.json")
    inventory = _load_json(protocol_dir / "run_inventory.json")
    if seed_manifest["seed_count"] != EXPECTED_SEEDS:
        raise AnalysisError("frozen seed count mismatch")
    if len(seed_manifest["seeds"]) != EXPECTED_SEEDS:
        raise AnalysisError("frozen seed manifest length mismatch")
    inventory_by_index = {int(record["index"]): record for record in inventory["configs"]}
    if set(inventory_by_index) != set(range(1, EXPECTED_SEEDS + 1)):
        raise AnalysisError("frozen config inventory mismatch")

    seeds: list[dict[str, Any]] = []
    source_index_lines: list[str] = []
    normalized_configs: list[dict[str, Any]] = []
    for seed_record in seed_manifest["seeds"]:
        seed_index = int(seed_record["index"])
        seed_result, source_index_line, normalized_config = _analyze_seed(
            repository_root,
            protocol_dir,
            runs_root,
            seed_record,
            inventory_by_index[seed_index],
            requirements,
        )
        seeds.append(seed_result)
        source_index_lines.append(source_index_line)
        normalized_configs.append(normalized_config)
    if any(config != normalized_configs[0] for config in normalized_configs[1:]):
        raise AnalysisError("seed configs differ beyond run_id and root_seed")

    selected_losses = [seed["selection"]["selected_validation_loss"] for seed in seeds]
    improvements = [seed["selection"]["relative_validation_improvement"] for seed in seeds]
    aggregates = {
        "selected_validation_loss": _aggregate(selected_losses),
        "relative_validation_improvement": _aggregate(improvements),
        "mean_train_loss_at_step_5000": float(
            np.mean([seed["train"]["loss_at_step_5000"] for seed in seeds])
        ),
        "mean_gradient_l2_norm_at_step_5000": float(
            np.mean([seed["train"]["gradient_l2_norm_at_step_5000"] for seed in seeds])
        ),
        "all_validation_curves_strictly_decreased_every_update": all(
            seed["selection"]["validation_strictly_decreased_every_update"] for seed in seeds
        ),
        "minimum_selected_gain_normalized_bound_distance": min(
            seed["minimum_selected_gain_normalized_bound_distance"] for seed in seeds
        ),
        "panel_technical_metric_maxima": {
            name: max(seed["technical_metric_maxima"][name] for seed in seeds)
            for name in TECHNICAL_METRICS
        },
        "total_external_wall_seconds": sum(
            seed["resources"]["external_wall_seconds"] for seed in seeds
        ),
        "peak_rss_kib_range": [
            min(seed["resources"]["peak_rss_kib"] for seed in seeds),
            max(seed["resources"]["peak_rss_kib"] for seed in seeds),
        ],
        "total_raw_run_files": sum(seed["run_file_count"] for seed in seeds),
        "total_raw_run_bytes": sum(seed["run_bytes"] for seed in seeds),
        "total_indexed_raw_files_verified": sum(
            seed["run_sha256_index"]["indexed_file_count"] for seed in seeds
        ),
        "total_checkpoint_payload_hashes_verified": sum(
            seed["checkpoint_payload_hashes_verified"] for seed in seeds
        ),
        "total_metric_rows": sum(seed["metric_rows"]["total"] for seed in seeds),
        "total_test_metric_rows": 0,
    }
    gate_summary = _gate_summary(seeds, aggregates)
    source_index_text = "".join(source_index_lines)
    analysis = {
        "schema_version": "crazyflow.h100_replication_synthesis.v1",
        "status": "PASS",
        "scope": (
            "Deterministic analysis of the ten existing H100 Train/Validation runs only; "
            "no simulation, optimization, new seed, or Test access."
        ),
        "input_contract": {
            "protocol_status": protocol["status"],
            "protocol_origin_head": protocol["protocol_origin_head"],
            "active_persistence_amendment": protocol["active_amendment"],
            "analysis_plan_schema": analysis_plan["schema_version"],
            "analysis_plan_sha256": _sha256_path(protocol_dir / "analysis_plan.json"),
            "seed_manifest_sha256": _sha256_path(protocol_dir / "seed_manifest.json"),
            "requirements_sha256": _sha256_path(protocol_dir / "requirements.json"),
            "selection_rule": analysis_plan["selection_rule"],
            "uncertainty": analysis_plan["primary_endpoint"]["uncertainty"],
            "software_runtime": requirements["runtime"],
            "required_source_fingerprint": requirements["required_source_fingerprint"],
            "required_runtime_fingerprint": requirements["required_runtime_fingerprint"],
            "required_gain_registry_fingerprint": requirements[
                "required_gain_registry_fingerprint"
            ],
            "required_validation_manifest_fingerprint": requirements[
                "required_validation_manifest_fingerprint"
            ],
            "normalized_seed_config_sha256": _sha256_bytes(
                _canonical_json(normalized_configs[0]).encode()
            ),
        },
        "source_run_index_set": {
            "file": "artifacts/day19-h100-analysis/SOURCE_RUN_INDEXES.sha256",
            "sha256": _sha256_bytes(source_index_text.encode()),
            "run_index_count": len(source_index_lines),
        },
        "seeds": seeds,
        "aggregates": aggregates,
        "horizon_gate": gate_summary,
        "test_boundary": {
            "locked_by_frozen_plan": analysis_plan["test_policy"]["locked"],
            "analysis_opened_test_manifest": False,
            "test_metric_rows": 0,
            "test_used_for_selection": False,
        },
        "claim_boundary": (
            "Positive simulation evidence under the frozen H100 configuration and documented "
            "Validation selection only; no Test, H200/H400, hardware, firmware, flight, or "
            "Sim2Real evidence."
        ),
    }
    return analysis, source_index_text


def _expected_outputs(
    analysis: dict[str, Any], source_index_text: str, script_path: Path
) -> dict[str, str]:
    analysis_text = _json_document(analysis)
    indexed_payloads = {
        script_path.name: script_path.read_bytes(),
        OUTPUT_JSON: analysis_text.encode(),
        SOURCE_INDEX: source_index_text.encode(),
    }
    checksum_text = "".join(
        f"{_sha256_bytes(payload)}  {name}\n" for name, payload in indexed_payloads.items()
    )
    return {
        OUTPUT_JSON: analysis_text,
        SOURCE_INDEX: source_index_text,
        OUTPUT_INDEX: checksum_text,
    }


def _write_outputs(output_dir: Path, outputs: dict[str, str]) -> None:
    existing = [output_dir / name for name in outputs if (output_dir / name).exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite versioned synthesis outputs: {paths}")
    for name, payload in outputs.items():
        with (output_dir / name).open("x") as stream:
            stream.write(payload)


def _check_outputs(output_dir: Path, outputs: dict[str, str]) -> None:
    mismatches = [
        name
        for name, expected in outputs.items()
        if not (output_dir / name).is_file() or (output_dir / name).read_text() != expected
    ]
    if mismatches:
        raise AnalysisError(f"versioned synthesis mismatch: {', '.join(mismatches)}")


def main() -> None:
    """Generate once or check the deterministic synthesis against immutable raw runs."""
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="create the three synthesis outputs")
    mode.add_argument("--check", action="store_true", help="compare with versioned outputs")
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=repository_root / "artifacts/day12-h100-replication-protocol",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=repository_root / "artifacts/day13-h100-replication/runs"
    )
    arguments = parser.parse_args()
    protocol_dir = arguments.protocol_dir.resolve()
    runs_root = arguments.runs_root.resolve()
    analysis, source_index_text = _build_analysis(repository_root, protocol_dir, runs_root)
    output_dir = Path(__file__).resolve().parent
    outputs = _expected_outputs(analysis, source_index_text, Path(__file__).resolve())
    if arguments.write:
        _write_outputs(output_dir, outputs)
        mode_name = "WRITE"
    else:
        _check_outputs(output_dir, outputs)
        mode_name = "CHECK"
    improvement = analysis["aggregates"]["relative_validation_improvement"]
    print(f"h100_replication_synthesis={mode_name}_PASS")
    print(f"seed_count={len(analysis['seeds'])}")
    print(
        "selected_steps="
        + ",".join(str(seed["selection"]["selected_step"]) for seed in analysis["seeds"])
    )
    print(f"mean_relative_validation_improvement={improvement['mean']:.12f}")
    print(
        "relative_validation_improvement_ci95="
        f"[{improvement['confidence_interval_95_percent_student_t'][0]:.12f},"
        f"{improvement['confidence_interval_95_percent_student_t'][1]:.12f}]"
    )
    print("test_manifest_opened=False")
    print("research_runs_started=0")


if __name__ == "__main__":
    main()
