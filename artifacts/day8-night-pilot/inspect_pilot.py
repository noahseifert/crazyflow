"""Read-only morning inspection for a Sprint-8 launcher run or runner output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


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
    """Inspect one run without modifying its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="launcher run root or direct runner output")
    args = parser.parse_args()
    supplied = args.path.resolve()
    output = supplied / "runner-output" if (supplied / "runner-output").is_dir() else supplied
    launch_root = supplied if output != supplied else None

    metrics_path = output / "metrics.jsonl"
    checkpoints = sorted(output.glob("checkpoint-step-*.json"))
    if not metrics_path.is_file() or not checkpoints:
        print("evaluation_status=FAIL")
        print(f"reason=missing metrics or complete checkpoint in {output}")
        return 1

    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    train = [row for row in metrics if row.get("split") == "train"]
    validation = [row for row in metrics if row.get("split") == "validation"]
    unexpected_splits = sorted({str(row.get("split")) for row in metrics} - {"train", "validation"})
    checkpoint_valid = all(_checkpoint_valid(path) for path in checkpoints)
    finite = _finite(metrics)
    gates = all(row.get("technical_gate", {}).get("passed") is True for row in metrics)
    gradients = [float(row["gradient_l2_norm"]) for row in train]
    sequence = [int(path.stem.rsplit("-", 1)[1]) for path in checkpoints]
    sequential = sequence == list(range(1, sequence[-1] + 1))

    summary_path = output / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    configured = None if summary is None else int(summary["configured_updates"])
    completed = sequence[-1]
    summary_safe = summary is not None and (
        summary.get("test_metrics_present") is False
        and summary.get("test_manifest_opened") is False
    )
    successful = (
        summary is not None
        and summary.get("status") == "success"
        and completed == configured
        and len(train) == completed
        and len(validation) == completed
    )
    resumable = checkpoint_valid and finite and gates and sequential and not successful
    integrity = (
        checkpoint_valid
        and finite
        and gates
        and sequential
        and not unexpected_splits
        and all(math.isfinite(value) for value in gradients)
    )
    if successful and integrity and summary_safe:
        status = "PASS"
        exit_status = 0
    elif resumable and integrity:
        status = "RESUMABLE"
        exit_status = 2
    else:
        status = "FAIL"
        exit_status = 1

    print(f"evaluation_status={status}")
    print(f"runner_output={output}")
    print(f"completed_updates={completed}")
    print(f"configured_updates={configured if configured is not None else 'unknown'}")
    print(f"checkpoint_count={len(checkpoints)}")
    print(f"last_complete_checkpoint={checkpoints[-1]}")
    print(f"checkpoint_payloads_valid={checkpoint_valid}")
    print(f"checkpoint_sequence_complete={sequential}")
    print(f"all_metrics_finite={finite}")
    print(f"all_technical_gates_pass={gates}")
    print(f"unexpected_metric_splits={unexpected_splits}")
    print(f"test_metrics_present={False if not unexpected_splits else 'unknown'}")
    if train:
        train_losses = [float(row["loss"]) for row in train]
        print(f"train_loss_initial={train_losses[0]}")
        print(f"train_loss_final={train_losses[-1]}")
        print(f"train_loss_min={min(train_losses)}")
        print(f"gradient_l2_initial={gradients[0]}")
        print(f"gradient_l2_final={gradients[-1]}")
        print(f"gradient_l2_min={min(gradients)}")
        print(f"gradient_l2_max={max(gradients)}")
    if validation:
        validation_losses = [float(row["loss"]) for row in validation]
        print(f"validation_loss_initial={validation_losses[0]}")
        print(f"validation_loss_final={validation_losses[-1]}")
        print(f"validation_loss_min={min(validation_losses)}")
    if summary is not None:
        print(f"runner_status={summary.get('status')}")
        print(f"selected_step={summary.get('selected', {}).get('selected_step')}")
        print(f"summary_test_metrics_present={summary.get('test_metrics_present')}")
        print(f"summary_test_manifest_opened={summary.get('test_manifest_opened')}")

    if launch_root is not None:
        launcher_status = launch_root / "launcher_status.txt"
        if launcher_status.is_file():
            print(launcher_status.read_text().strip())
        time_path = launch_root / "time.txt"
        if time_path.is_file():
            timing = time_path.read_text()
            wall_label = "Elapsed (wall clock) time (h:mm:ss or m:ss)"
            print(f"external_wall={_time_value(timing, wall_label)}")
            print(f"peak_rss_kib={_time_value(timing, 'Maximum resident set size (kbytes)')}")
            print(f"swaps={_time_value(timing, 'Swaps')}")
            print(f"external_exit_status={_time_value(timing, 'Exit status')}")
        print(
            "resume_command="
            f"{Path(__file__).with_name('launch_night_pilot.sh')} --resume {checkpoints[-1]}"
        )

    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
