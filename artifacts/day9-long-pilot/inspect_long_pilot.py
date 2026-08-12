"""Read-only status and integrity inspection for a Sprint-9 long-pilot run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

EXPECTED_UPDATES = 5000


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
    """Inspect an active, interrupted, or completed long-pilot run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = run_root / "runner-output"
    checkpoints = sorted(output.glob("checkpoint-step-*.json"))
    process_path = run_root / "process_group.pid"
    process_active = False
    if process_path.is_file():
        process_active = Path("/proc").joinpath(process_path.read_text().strip()).is_dir()
    if not checkpoints:
        print("evaluation_status=ACTIVE" if process_active else "evaluation_status=FAIL")
        print("completed_updates=0")
        print(f"process_active={process_active}")
        return 0 if process_active else 1

    sequence = [int(path.stem.rsplit("-", 1)[1]) for path in checkpoints]
    sequential = sequence == list(range(1, sequence[-1] + 1))
    checkpoint_valid = all(_checkpoint_valid(path) for path in checkpoints)
    metrics_path = output / "metrics.jsonl"
    metrics = (
        [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if metrics_path.is_file()
        else []
    )
    train = [row for row in metrics if row.get("split") == "train"]
    validation = [row for row in metrics if row.get("split") == "validation"]
    unexpected = sorted({str(row.get("split")) for row in metrics} - {"train", "validation"})
    integrity = (
        sequential
        and checkpoint_valid
        and _finite(metrics)
        and not unexpected
        and all(row.get("technical_gate", {}).get("passed") is True for row in metrics)
        and len(train) == sequence[-1]
        and len(validation) == sequence[-1]
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
        and sequence[-1] == EXPECTED_UPDATES
    )
    if successful and integrity:
        status, exit_status = "PASS", 0
    elif process_active and integrity:
        status, exit_status = "ACTIVE", 0
    elif integrity:
        status, exit_status = "RESUMABLE", 2
    else:
        status, exit_status = "FAIL", 1

    print(f"evaluation_status={status}")
    print(f"run_root={run_root}")
    print(f"process_active={process_active}")
    print(f"completed_updates={sequence[-1]}")
    print(f"configured_updates={EXPECTED_UPDATES}")
    print(f"checkpoint_count={len(checkpoints)}")
    print(f"checkpoint_sequence_complete={sequential}")
    print(f"checkpoint_payloads_valid={checkpoint_valid}")
    print(f"all_metrics_finite={_finite(metrics)}")
    print(f"unexpected_metric_splits={unexpected}")
    print(f"last_complete_checkpoint={checkpoints[-1]}")
    if train:
        train_losses = [float(row["loss"]) for row in train]
        gradients = [float(row["gradient_l2_norm"]) for row in train]
        print(f"train_loss_initial={train_losses[0]}")
        print(f"train_loss_current={train_losses[-1]}")
        print(f"train_loss_min={min(train_losses)}")
        print(f"gradient_l2_current={gradients[-1]}")
    if validation:
        losses = [float(row["loss"]) for row in validation]
        print(f"validation_loss_initial={losses[0]}")
        print(f"validation_loss_current={losses[-1]}")
        print(f"validation_loss_min={min(losses)}")
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
