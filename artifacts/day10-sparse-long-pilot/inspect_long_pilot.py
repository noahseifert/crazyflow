"""Read-only status and integrity inspection for the Sprint-10 sparse long pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

EXPECTED_UPDATES = 5000
CHECKPOINT_INTERVAL = 100


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def _checkpoint_valid(path: Path) -> bool:
    payload = json.loads(path.read_text())
    recorded = payload.pop("payload_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return recorded == hashlib.sha256(encoded.encode()).hexdigest()


def _time_value(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    return None if match is None else match.group(1).strip()


def main() -> int:
    """Inspect an active, interrupted, or completed sparse long-pilot run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = run_root / "runner-output"
    checkpoints = sorted(output.glob("checkpoint-step-*.json"))
    process_path = run_root / "process_group.pid"
    process_active = (
        process_path.is_file() and Path("/proc").joinpath(process_path.read_text().strip()).is_dir()
    )
    checkpoint_steps = [int(path.stem.rsplit("-", 1)[1]) for path in checkpoints]
    expected_checkpoint_steps = list(
        range(
            CHECKPOINT_INTERVAL,
            (checkpoint_steps[-1] if checkpoint_steps else 0) + 1,
            CHECKPOINT_INTERVAL,
        )
    )
    checkpoint_valid = all(_checkpoint_valid(path) for path in checkpoints)
    metrics_path = output / "metrics.jsonl"
    metrics = []
    if metrics_path.is_file():
        lines = metrics_path.read_text().splitlines()
        try:
            metrics = [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            if not process_active:
                raise
            metrics = [json.loads(line) for line in lines[:-1]]
    train = [row for row in metrics if row.get("split") == "train"]
    validation = [row for row in metrics if row.get("split") == "validation"]
    completed_metrics = min(len(train), len(validation))
    metrics_complete = (
        len(train) == len(validation)
        and [row["step"] for row in train] == list(range(1, len(train) + 1))
        and [row["step"] for row in validation] == list(range(1, len(validation) + 1))
    )
    integrity = (
        checkpoint_steps == expected_checkpoint_steps
        and checkpoint_valid
        and metrics_complete
        and _finite(metrics)
        and {str(row.get("split")) for row in metrics} <= {"train", "validation"}
        and all(row.get("technical_gate", {}).get("passed") is True for row in metrics)
    )
    summary_path = output / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    successful = (
        summary is not None
        and summary.get("status") == "success"
        and summary.get("completed_updates") == EXPECTED_UPDATES
        and summary.get("configured_updates") == EXPECTED_UPDATES
        and summary.get("test_metrics_present") is False
        and summary.get("test_manifest_opened") is False
        and checkpoint_steps == list(range(100, 5001, 100))
        and completed_metrics == EXPECTED_UPDATES
    )
    if successful and integrity:
        status, exit_status = "PASS", 0
    elif process_active and integrity:
        status, exit_status = "ACTIVE", 0
    elif checkpoints and integrity:
        status, exit_status = "RESUMABLE", 2
    else:
        status, exit_status = "FAIL", 1

    print(f"evaluation_status={status}")
    print(f"run_root={run_root}")
    print(f"process_active={process_active}")
    print(f"completed_metric_updates={completed_metrics}")
    print(f"checkpoint_steps={checkpoint_steps}")
    print(f"checkpoint_payloads_valid={checkpoint_valid}")
    print(f"metrics_complete_without_duplicates={metrics_complete}")
    print(f"all_values_finite={_finite(metrics)}")
    if checkpoints:
        print(f"last_complete_checkpoint={checkpoints[-1]}")
    if train:
        print(f"train_loss_current={float(train[-1]['loss'])}")
        print(f"gradient_l2_current={float(train[-1]['gradient_l2_norm'])}")
    if validation:
        print(f"validation_loss_current={float(validation[-1]['loss'])}")
    time_path = run_root / "time.txt"
    if time_path.is_file():
        timing = time_path.read_text()
        wall_label = "Elapsed (wall clock) time (h:mm:ss or m:ss)"
        print(f"external_wall={_time_value(timing, wall_label)}")
        print(f"peak_rss_kib={_time_value(timing, 'Maximum resident set size (kbytes)')}")
        print(f"swaps={_time_value(timing, 'Swaps')}")
        print(f"external_exit_status={_time_value(timing, 'Exit status')}")
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
