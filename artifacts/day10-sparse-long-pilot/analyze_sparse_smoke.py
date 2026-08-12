"""Validate the Sprint-10 sparse smoke and project the exact 5,000-update pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


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


def _checkpoint_valid(path: Path) -> bool:
    payload = json.loads(path.read_text())
    recorded = payload.pop("payload_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return recorded == hashlib.sha256(encoded.encode()).hexdigest()


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - predicted) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 if total == 0.0 else 1.0 - residual / total,
    }


def _external_time(path: Path) -> dict[str, Any]:
    text = path.read_text()
    wall = "Elapsed (wall clock) time (h:mm:ss or m:ss)"
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "wall_seconds": _elapsed_seconds(_time_value(text, wall)),
        "peak_rss_kib": int(_time_value(text, "Maximum resident set size (kbytes)")),
        "swaps": int(_time_value(text, "Swaps")),
        "exit_status": int(_time_value(text, "Exit status")),
    }


def main() -> None:
    """Write a strict, deterministic validation and conservative projection record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path, required=True)
    parser.add_argument("--smoke-time", type=Path, required=True)
    parser.add_argument("--resume-time", type=Path, required=True)
    parser.add_argument("--smoke-config", type=Path, required=True)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--sprint9-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    smoke_files = sorted(path for path in args.smoke_dir.rglob("*") if path.is_file())
    resume_files = sorted(path for path in args.resume_dir.rglob("*") if path.is_file())
    checkpoints = sorted(args.smoke_dir.glob("checkpoint-step-*.json"))
    checkpoint_steps = [int(path.stem.rsplit("-", 1)[1]) for path in checkpoints]
    checkpoint_sizes = [path.stat().st_size for path in checkpoints]
    metrics = [
        json.loads(line) for line in (args.smoke_dir / "metrics.jsonl").read_text().splitlines()
    ]
    summary = json.loads((args.smoke_dir / "run_summary.json").read_text())
    provenance = json.loads((args.smoke_dir / "provenance.json").read_text())
    resumed_metrics = [
        json.loads(line) for line in (args.resume_dir / "metrics.jsonl").read_text().splitlines()
    ]
    resume_summary = json.loads((args.resume_dir / "run_summary.json").read_text())
    equivalence = json.loads((args.resume_dir / "resume_equivalence.json").read_text())

    expected_metric_steps = [step for step in range(1, 211) for _ in range(2)]
    expected_splits = ["train", "validation"] * 210
    integrity = {
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_sizes_bytes": checkpoint_sizes,
        "checkpoint_payload_hashes_valid": all(_checkpoint_valid(path) for path in checkpoints),
        "metric_rows": len(metrics),
        "metric_steps_complete_without_duplicates": (
            [row["step"] for row in metrics] == expected_metric_steps
        ),
        "metric_splits_complete_without_duplicates": (
            [row["split"] for row in metrics] == expected_splits
        ),
        "all_values_finite": _finite(metrics) and _finite(summary),
        "all_technical_gates_passed": all(row["technical_gate"]["passed"] for row in metrics),
        "summary_success_210_of_210": (
            summary["status"] == "success"
            and summary["completed_updates"] == 210
            and summary["configured_updates"] == 210
        ),
        "test_manifest_opened": summary["test_manifest_opened"],
        "test_metrics_present": summary["test_metrics_present"],
    }
    resume = {
        "status": resume_summary["status"],
        "parent_step": json.loads((args.resume_dir / "provenance.json").read_text())[
            "resume_lineage"
        ]["parent_step"],
        "checkpoint_steps": [
            int(path.stem.rsplit("-", 1)[1])
            for path in sorted(args.resume_dir.glob("checkpoint-step-*.json"))
        ],
        "metric_rows": len(resumed_metrics),
        "metrics_complete_without_duplicates": (
            [row["step"] for row in resumed_metrics] == expected_metric_steps
            and [row["split"] for row in resumed_metrics] == expected_splits
        ),
        "equivalence_status": equivalence["status"],
        "equivalent_exact_rtol_0_atol_0": equivalence["equivalent"],
        "all_exact_fields_match": all(equivalence["field_matches"].values()),
        "test_manifest_opened_or_test_episode_built": equivalence[
            "test_manifest_opened_or_test_episode_built"
        ],
    }

    update_timings = summary["timing"]["completed_update_seconds"]
    steady = np.asarray(
        [
            item["seconds"]
            for item in update_timings
            if item["step"] >= 3 and not item["checkpoint_written"]
        ],
        dtype=float,
    )
    checkpoint_writes = summary["timing"]["checkpoint_writes"]
    checkpoint_durations = np.asarray([item["seconds"] for item in checkpoint_writes], dtype=float)
    measured_checkpoint_sizes = np.asarray(
        [item["bytes"] for item in checkpoint_writes], dtype=float
    )
    measured_checkpoint_steps = np.asarray(
        [item["step"] for item in checkpoint_writes], dtype=float
    )
    size_fit = _linear_fit(measured_checkpoint_steps, measured_checkpoint_sizes)
    target_checkpoint_steps = np.arange(100, 5001, 100, dtype=float)
    projected_checkpoint_sizes = size_fit["slope"] * target_checkpoint_steps + size_fit["intercept"]
    projected_checkpoint_bytes = float(np.sum(projected_checkpoint_sizes))
    maximum_seconds_per_checkpoint_byte = float(
        np.max(checkpoint_durations / measured_checkpoint_sizes)
    )

    metrics_bytes = (args.smoke_dir / "metrics.jsonl").stat().st_size
    summary_bytes = (args.smoke_dir / "run_summary.json").stat().st_size
    fixed_runner_bytes = sum(
        (args.smoke_dir / name).stat().st_size
        for name in ("provenance.json", "resolved_config.json")
    )
    launcher_fixed_allowance_bytes = 1_048_576
    expected_total_bytes = int(
        round(
            projected_checkpoint_bytes
            + metrics_bytes / 210 * 5000
            + summary_bytes / 210 * 5000
            + fixed_runner_bytes
            + launcher_fixed_allowance_bytes
        )
    )
    conservative_total_bytes = 2 * expected_total_bytes
    startup_seconds = float(sum(item["seconds"] for item in update_timings[:2]))
    projected_checkpoint_seconds = maximum_seconds_per_checkpoint_byte * projected_checkpoint_bytes
    conservative_wall_seconds = float(
        np.max(steady) * 5000 + 2.0 * projected_checkpoint_seconds + 2.0 * startup_seconds + 30.0
    )

    smoke_config = json.loads(args.smoke_config.read_text())
    pilot_config = json.loads(args.pilot_config.read_text())
    sprint9_config = json.loads(args.sprint9_config.read_text())
    pilot_comparison = dict(pilot_config)
    sprint9_comparison = dict(sprint9_config)
    pilot_optimizer = dict(pilot_comparison["optimizer"])
    sprint9_optimizer = dict(sprint9_comparison["optimizer"])
    checkpoint_interval = pilot_optimizer.pop("checkpoint_interval")
    pilot_comparison["optimizer"] = pilot_optimizer
    sprint9_comparison["optimizer"] = sprint9_optimizer
    scientific_semantics_unchanged = (
        checkpoint_interval == 100
        and pilot_comparison | {"run_id": sprint9_comparison["run_id"]} == sprint9_comparison
    )

    smoke_external = _external_time(args.smoke_time)
    resume_external = _external_time(args.resume_time)
    result = {
        "schema_version": "crazyflow.sprint10_sparse_analysis.v1",
        "scientific_claim": "technical and exploratory single-seed simulation evidence only",
        "inputs": {
            "smoke_directory": str(args.smoke_dir),
            "smoke_file_count": len(smoke_files),
            "smoke_file_bytes": sum(path.stat().st_size for path in smoke_files),
            "smoke_artifact_sha256": {
                str(path.relative_to(args.smoke_dir)): _sha256(path) for path in smoke_files
            },
            "resume_directory": str(args.resume_dir),
            "resume_file_count": len(resume_files),
            "resume_file_bytes": sum(path.stat().st_size for path in resume_files),
            "resume_artifact_sha256": {
                str(path.relative_to(args.resume_dir)): _sha256(path) for path in resume_files
            },
            "smoke_config_sha256": _sha256(args.smoke_config),
            "pilot_config_sha256": _sha256(args.pilot_config),
            "sprint9_config_sha256": _sha256(args.sprint9_config),
        },
        "fingerprints": {
            "implementation_git_head": provenance["git"]["head"],
            "implementation_git_branch": provenance["git"]["branch"],
            "source_dirty": provenance["git"]["dirty"],
            "source_fingerprint": provenance["git"]["source_fingerprint"],
            "runtime_fingerprint": provenance["runtime_fingerprint"],
            "smoke_config_fingerprint": provenance["config_fingerprint"],
        },
        "integrity": integrity,
        "resume": resume,
        "resources": {"smoke": smoke_external, "resume": resume_external},
        "measurements": {
            "train_compile_seconds": summary["timing"]["train_compile_seconds"],
            "validation_compile_seconds": summary["timing"]["validation_compile_seconds"],
            "train_first_execution_seconds": summary["timing"]["train_first_execution_seconds"],
            "validation_first_execution_seconds": summary["timing"][
                "validation_first_execution_seconds"
            ],
            "steady_update_count": len(steady),
            "steady_update_seconds": {
                "minimum": float(np.min(steady)),
                "median": float(np.median(steady)),
                "p95": float(np.quantile(steady, 0.95)),
                "p99": float(np.quantile(steady, 0.99)),
                "maximum": float(np.max(steady)),
                "mean": float(np.mean(steady)),
            },
            "checkpoint_writes": checkpoint_writes,
        },
        "projection_5000_updates_50_checkpoints": {
            "method": (
                "Use the maximum measured non-checkpoint steady update for all 5000 updates; "
                "add twice the maximum measured checkpoint seconds/byte times linearly projected "
                "checkpoint bytes; add twice the first-two-update startup and 30 seconds fixed. "
                "Storage doubles the empirical linear payload projection and includes a 1 MiB "
                "launcher allowance."
            ),
            "checkpoint_size_linear_fit": size_fit,
            "predicted_final_checkpoint_bytes": int(round(projected_checkpoint_sizes[-1])),
            "predicted_checkpoint_bytes_total": int(round(projected_checkpoint_bytes)),
            "predicted_checkpoint_seconds_at_measured_max_rate": projected_checkpoint_seconds,
            "expected_total_artifact_bytes": expected_total_bytes,
            "conservative_total_artifact_bytes": conservative_total_bytes,
            "expected_total_files_after_completion": 61,
            "conservative_wall_seconds": conservative_wall_seconds,
            "hard_wall_limit_seconds": 7200,
            "artifact_limit_bytes": 2 * 1024**3,
            "peak_rss_limit_kib": 12 * 1024**2,
        },
        "scientific_semantics": {
            "pilot_differs_from_sprint9_only_by_run_id_and_checkpoint_interval": (
                scientific_semantics_unchanged
            ),
            "smoke_updates": smoke_config["optimizer"]["steps"],
            "pilot_updates": pilot_config["optimizer"]["steps"],
            "checkpoint_interval": checkpoint_interval,
            "test_split_opened": False,
        },
    }
    gates = {
        "smoke_exit_zero": smoke_external["exit_status"] == 0,
        "resume_exit_zero": resume_external["exit_status"] == 0,
        "integrity": all(
            value is True
            for key, value in integrity.items()
            if key
            not in {
                "checkpoint_steps",
                "checkpoint_sizes_bytes",
                "metric_rows",
                "test_manifest_opened",
                "test_metrics_present",
            }
        )
        and integrity["checkpoint_steps"] == [100, 200, 210]
        and integrity["test_manifest_opened"] is False
        and integrity["test_metrics_present"] is False,
        "exact_resume": (
            resume["equivalent_exact_rtol_0_atol_0"]
            and resume["metrics_complete_without_duplicates"]
            and resume["test_manifest_opened_or_test_episode_built"] is False
        ),
        "peak_rss_below_12_gib": smoke_external["peak_rss_kib"] < 12 * 1024**2,
        "swap_zero": smoke_external["swaps"] == 0 and resume_external["swaps"] == 0,
        "artifact_projection_at_most_2_gib": conservative_total_bytes <= 2 * 1024**3,
        "runtime_projection_below_7200_seconds": conservative_wall_seconds < 7200,
        "scientific_semantics_unchanged": scientific_semantics_unchanged,
    }
    result["gates"] = gates
    result["decision"] = "GO" if all(gates.values()) else "NO-GO"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
