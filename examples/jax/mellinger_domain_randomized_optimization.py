"""Run the train/validation-only domain-randomized Mellinger experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

from crazyflow.control.mellinger.research import (
    atomic_write_json,
    load_config,
    physical_from_raw,
    run_training_validation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_IN_INTEGRATION_TEST = False


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/mellinger/smoke.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day5-audit/local-smoke"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-this-process", type=int)
    parser.add_argument("--equivalence-reference-checkpoint", type=Path)
    return parser.parse_args(argv)


def _array_from_record(record: dict[str, object]) -> np.ndarray:
    return np.asarray(record["data"], dtype=str(record["dtype"])).reshape(record["shape"])


def _checkpoint_equivalence(
    reference_path: Path, actual_path: Path, gain_stage: int
) -> dict[str, object]:
    """Compare scientific continuation state exactly, excluding process-local metadata."""
    reference = json.loads(reference_path.read_text())
    actual = json.loads(actual_path.read_text())
    exact_fields = (
        "schema_version",
        "config_fingerprint",
        "gain_registry_fingerprint",
        "source_fingerprint",
        "runtime_fingerprint",
        "step",
        "episode_index",
        "root_seed",
        "raw_gains",
        "optimizer_state",
        "history",
        "metrics_records",
        "selection",
        "selected_raw_gains",
        "validation_manifest_id",
        "validation_manifest_fingerprint",
        "frozen_test_manifest_reference",
    )
    field_matches = {field: reference[field] == actual[field] for field in exact_fields}
    reference_raw = _array_from_record(reference["raw_gains"])
    actual_raw = _array_from_record(actual["raw_gains"])
    reference_selected = _array_from_record(reference["selected_raw_gains"])
    actual_selected = _array_from_record(actual["selected_raw_gains"])
    raw_max_abs_difference = float(np.max(np.abs(reference_raw - actual_raw)))
    selected_max_abs_difference = float(np.max(np.abs(reference_selected - actual_selected)))
    optimizer_leaf_max_abs_differences = []
    for reference_leaf, actual_leaf in zip(
        reference["optimizer_state"]["leaves"], actual["optimizer_state"]["leaves"], strict=True
    ):
        reference_value = _array_from_record(reference_leaf)
        actual_value = _array_from_record(actual_leaf)
        optimizer_leaf_max_abs_differences.append(
            float(np.max(np.abs(reference_value - actual_value)))
        )
    reference_physical = {
        name: float(value) for name, value in physical_from_raw(reference_raw, gain_stage).items()
    }
    actual_physical = {
        name: float(value) for name, value in physical_from_raw(actual_raw, gain_stage).items()
    }
    transformed_gain_max_abs_difference = max(
        abs(reference_physical[name] - actual_physical[name]) for name in reference_physical
    )
    equivalent = all(field_matches.values())
    return {
        "schema_version": "crazyflow.mellinger_resume_equivalence.v1",
        "status": "pass" if equivalent else "fail",
        "equivalent": equivalent,
        "comparison_policy": {
            "rtol": 0.0,
            "atol": 0.0,
            "justification": (
                "Identical source, runtime, backend, dtype, config, seed coordinates, and pure "
                "JAX update order require exact serialized equality; timing, command, paths, and "
                "UTC provenance are intentionally excluded."
            ),
        },
        "reference_checkpoint": str(reference_path),
        "actual_checkpoint": str(actual_path),
        "field_matches": field_matches,
        "raw_gain_max_abs_difference": raw_max_abs_difference,
        "selected_raw_gain_max_abs_difference": selected_max_abs_difference,
        "transformed_gain_max_abs_difference": transformed_gain_max_abs_difference,
        "optimizer_leaf_max_abs_differences": optimizer_leaf_max_abs_differences,
        "reference_physical_gains": reference_physical,
        "actual_physical_gains": actual_physical,
        "test_manifest_opened_or_test_episode_built": False,
    }


def main(args: argparse.Namespace | None = None) -> None:
    """Execute a validated config without exposing the frozen test pipeline."""
    args = parse_args(()) if args is None else args
    config_path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else REPOSITORY_ROOT / args.output_dir
    )
    resume_path = args.resume
    if resume_path is not None and not resume_path.is_absolute():
        resume_path = REPOSITORY_ROOT / resume_path
    equivalence_reference = args.equivalence_reference_checkpoint
    if equivalence_reference is not None and not equivalence_reference.is_absolute():
        equivalence_reference = REPOSITORY_ROOT / equivalence_reference
    config = load_config(config_path)
    summary = run_training_validation(
        config,
        config_path=config_path,
        output_dir=output_dir,
        repository_root=REPOSITORY_ROOT,
        command=shlex.join([sys.executable, *sys.argv]),
        resume_path=resume_path,
        max_updates_this_process=args.max_updates_this_process,
    )
    if equivalence_reference is not None:
        if summary["status"] != "success":
            raise ValueError("equivalence comparison requires a completed run")
        actual_checkpoint = output_dir / (f"checkpoint-step-{config.optimizer.steps:06d}.json")
        comparison = _checkpoint_equivalence(
            equivalence_reference, actual_checkpoint, config.optimizer.gain_stage
        )
        atomic_write_json(output_dir / "resume_equivalence.json", comparison)
        if not comparison["equivalent"]:
            raise RuntimeError("resumed state is not exactly equivalent to uninterrupted state")
        print("resume_equivalence=pass")
    print(f"status={summary['status']}")
    print(f"selected_step={summary['selected']['selected_step']}")
    print(f"test_metrics_present={summary['test_metrics_present']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main(parse_args())
