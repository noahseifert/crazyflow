"""Generate the deterministic Day-26 G3.4 parameter-eligibility-audit package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from crazyflow.control.mellinger.research import g3_freeze_v2 as g3

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_COMMIT_SUBJECT = "research: complete g3.2 reverse-mode freeze diagnostics"
IMPLEMENTATION_COMMIT_PATHS = {
    "crazyflow/control/mellinger/research/g3_freeze_v2.py",
    "examples/jax/mellinger_g3_freeze_v2.py",
    "tests/unit/test_mellinger_g3_freeze_v2.py",
}
GENERATOR_COMMIT_SUBJECT = "research: support g3.4 recovery provenance"
GENERATOR_COMMIT_PARENT = "d8ad697fedf2cf124461c1df262ed149856ba30d"
GENERATOR_COMMIT_PATHS = {
    "examples/jax/mellinger_g3_freeze_v2.py",
    "tests/unit/test_mellinger_g3_freeze_v2.py",
}
INTEGRATION_TARGET_BRANCH = "research/differentiable-mellinger"
PACKAGE_NAMES = (
    "g3_freeze_report.json",
    "trajectory_contract.json",
    "train_manifest.json",
    "validation_manifest.json",
    "parameter_semantics.json",
    "sensitivity_conditioning.json",
    "loss_diagnostics.json",
    "g4_g5_freeze_proposal.json",
    "g3_freeze_overview.png",
    "parameter_sensitivity_matrix.png",
    "TECHNICAL_HANDOFF.md",
    "provenance.json",
)


class G3CliError(RuntimeError):
    """Raised before output if a package or resource contract fails."""


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate-package", action="store_true")
    mode.add_argument("--diagnose-g3-003", action="store_true")
    mode.add_argument("--diagnose-g3-004-smoke", action="store_true")
    mode.add_argument("--precommit-dry-run", action="store_true")
    parser.add_argument("--generator-commit")
    parser.add_argument("--execution-mode", choices=(g3.G3_004_EXECUTION_MODE,))
    parser.add_argument("--fallback-mode")
    parser.add_argument("--batch-timeout-seconds", type=int)
    parser.add_argument("--total-timeout-seconds", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--source-snapshot-sha256")
    args = parser.parse_args(argv)
    if args.generate_package:
        required = (
            args.generator_commit,
            args.execution_mode,
            args.batch_timeout_seconds,
            args.total_timeout_seconds,
            args.output_dir,
        )
        if (
            any(value is None for value in required)
            or args.fallback_mode is not None
            or args.output_file is not None
            or args.source_snapshot_sha256 is not None
        ):
            parser.error("package generation requires the fixed G3.4 mode without fallback")
        if args.batch_timeout_seconds != 900 or args.total_timeout_seconds != 2400:
            parser.error("timeouts must be exactly 900 s per batch and 2400 s total")
        if len(args.generator_commit) != 40:
            parser.error("--generator-commit must be a full 40-character hash")
        resolved_output = args.output_dir.resolve()
        final_output = (REPOSITORY_ROOT / "artifacts/day26-g3-identifiability-freeze-v2").resolve()
        temporary_output = str(resolved_output).startswith("/tmp/gr-g3-004-repro-") and (
            resolved_output.name == "package"
        )
        if resolved_output != final_output and not temporary_output:
            parser.error("package output must be the frozen Day-26 path or a repro runroot")
    elif args.diagnose_g3_003:
        if args.output_file is None or any(
            value is not None
            for value in (
                args.generator_commit,
                args.execution_mode,
                args.fallback_mode,
                args.batch_timeout_seconds,
                args.total_timeout_seconds,
                args.output_dir,
                args.source_snapshot_sha256,
            )
        ):
            parser.error("G3.3 diagnosis requires only --output-file")
        if not str(args.output_file).startswith("/tmp/gr-g3-003-"):
            parser.error("G3.3 diagnosis output must be under /tmp/gr-g3-003-*")
    elif args.diagnose_g3_004_smoke:
        if args.output_file is None or any(
            value is not None
            for value in (
                args.generator_commit,
                args.execution_mode,
                args.fallback_mode,
                args.batch_timeout_seconds,
                args.total_timeout_seconds,
                args.output_dir,
                args.source_snapshot_sha256,
            )
        ):
            parser.error("G3.4 smoke requires only --output-file")
        if not str(args.output_file.resolve()).startswith("/tmp/gr-g3-004-smoke-"):
            parser.error("G3.4 smoke output must be under /tmp/gr-g3-004-smoke-*")
    else:
        forbidden = (
            args.generator_commit,
            args.execution_mode,
            args.fallback_mode,
            args.batch_timeout_seconds,
            args.total_timeout_seconds,
            args.output_file,
        )
        if any(value is not None for value in forbidden):
            parser.error("precommit dry-run accepts only source snapshot and output directory")
        if args.source_snapshot_sha256 is None or args.output_dir is None:
            parser.error("precommit dry-run requires source snapshot and output directory")
        if len(args.source_snapshot_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in args.source_snapshot_sha256
        ):
            parser.error("--source-snapshot-sha256 must be 64 lowercase hex characters")
        if args.output_dir.resolve() != Path("/tmp/gr-g3-004-precommit-package"):
            parser.error("precommit dry-run output must be the frozen /tmp path")
    return args


def _json_bytes(value: Any) -> bytes:
    return g3.canonical_json_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_snapshot_sha256() -> str:
    """Hash the exact ordered Commit-1 source/test inventory."""
    digest = hashlib.sha256()
    for relative_path in sorted(IMPLEMENTATION_COMMIT_PATHS):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update((REPOSITORY_ROOT / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _validate_generator_commit(generator_commit: str) -> None:
    if _git("rev-parse", "HEAD") != generator_commit:
        raise G3CliError("generator_commit must be the checked-out provenance commit full hash")
    if _git("show", "-s", "--format=%s", generator_commit) != GENERATOR_COMMIT_SUBJECT:
        raise G3CliError("generator commit subject is not the prescribed provenance subject")
    paths = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", generator_commit).splitlines()
    )
    if paths != GENERATOR_COMMIT_PATHS:
        raise G3CliError("generator commit does not contain exactly the two provenance paths")
    if _git("rev-parse", f"{generator_commit}^") != GENERATOR_COMMIT_PARENT:
        raise G3CliError("generator commit parent is not the exact recovery commit")


def _memory_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(raw.strip().split()[0]) * 1024
    if set(values) != {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
        raise G3CliError("incomplete /proc/meminfo resource snapshot")
    return values


def _preflight_resources() -> dict[str, Any]:
    memory = _memory_snapshot()
    if memory["MemAvailable"] < 8 * 1024**3:
        raise G3CliError("G3.4 science requires MemAvailable >= 8 GiB before science")
    if memory["SwapTotal"] - memory["SwapFree"] != 0:
        raise G3CliError("G3.4 science requires zero active swap")
    return {
        "status": "PASS_SIX_BATCHES_OF_FOUR_PREFLIGHT",
        "mem_total_bytes": memory["MemTotal"],
        "mem_available_bytes": memory["MemAvailable"],
        "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
        "rss_limit_bytes": min(24 * 1024**3, int(0.70 * memory["MemTotal"])),
        "selected_execution_mode": g3.G3_004_EXECUTION_MODE,
        "selection_decision": g3.G3_004_SELECTION_DECISION,
        "selection_policy": g3.G3_004_SELECTION_POLICY,
        "selected_by_host_memory": False,
        "runtime_fallback_used": False,
        "runtime_fallback_reason": None,
    }


def _postflight_resources(
    preflight: dict[str, Any], elapsed_s: float, evidence: dict[str, Any]
) -> dict[str, Any]:
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    limit = min(24 * 1024**3, int(0.70 * preflight["mem_total_bytes"]))
    if maximum_rss > limit:
        raise G3CliError(f"Max RSS {maximum_rss} exceeds frozen limit {limit}")
    if elapsed_s > 2400.0:
        raise G3CliError(f"total diagnostic elapsed time {elapsed_s:.3f} s exceeds 2400 s")
    batch_execution = evidence["batch_execution"]
    expected_counts = list(g3.G3_004_BATCH_PARENT_COUNTS)
    expected_labels = [item[0] for item in g3.G3_004_BATCH_SLICES]
    runtime_batches = evidence["runtime_batch_records"]
    batch_parent_ids = [item["parent_ids"] for item in batch_execution["batches"]]
    expected_ids = [item.episode_id for item in g3.episode_specs()]
    if (
        batch_execution["mode"] != g3.G3_004_EXECUTION_MODE
        or [item["parent_count"] for item in batch_execution["batches"]] != expected_counts
        or [item["batch_label"] for item in batch_execution["batches"]] != expected_labels
        or [len(item) for item in batch_parent_ids] != expected_counts
        or [parent_id for batch in batch_parent_ids for parent_id in batch] != expected_ids
        or any(item["elapsed_seconds"] > 900.0 for item in runtime_batches)
        or [item["batch_label"] for item in runtime_batches] != expected_labels
        or [item["deadline_seconds"] for item in runtime_batches] != [900] * 6
        or not all(item["deadline_passed"] for item in runtime_batches)
    ):
        raise G3CliError("execution batch identity, count, or 900 s deadline changed")
    return {
        "schema_version": g3.G3_004_BATCH_EXECUTION_SCHEMA,
        "status": "PASS_SIX_BATCHES_OF_FOUR_RESOURCE_ENVELOPE",
        "elapsed_seconds": elapsed_s,
        "maximum_rss_bytes": maximum_rss,
        "maximum_allowed_rss_bytes": limit,
        "execution_mode": g3.G3_004_EXECUTION_MODE,
        "selection_decision": g3.G3_004_SELECTION_DECISION,
        "selection_policy": g3.G3_004_SELECTION_POLICY,
        "selected_by_host_memory": False,
        "runtime_fallback_used": False,
        "runtime_fallback_reason": None,
        "batch_records": runtime_batches,
        "batch_labels": expected_labels,
        "batch_parent_counts": expected_counts,
        "batch_parent_ids": batch_parent_ids,
        "batch_freeze_status": "WITHHELD_PENDING_REBENCHMARK",
        "G4": "WITHHELD_PENDING_REBENCHMARK",
    }


def _stable_resource_record(resources: dict[str, Any]) -> dict[str, Any]:
    """Keep volatile observations in /tmp process manifests, never in package bytes."""
    return {
        "schema_version": resources["schema_version"],
        "status": resources["status"],
        "maximum_rss_absolute_limit_bytes": 24 * 1024**3,
        "maximum_rss_fraction_of_mem_total": 0.70,
        "minimum_mem_available_bytes": 8 * 1024**3,
        "swap_used_required_bytes": 0,
        "batch_timeout_seconds": 900,
        "total_timeout_seconds": 2400,
        "execution_mode": resources["execution_mode"],
        "selection_decision": resources["selection_decision"],
        "selection_policy": resources["selection_policy"],
        "selected_by_host_memory": resources["selected_by_host_memory"],
        "runtime_fallback_used": resources["runtime_fallback_used"],
        "runtime_fallback_reason": resources["runtime_fallback_reason"],
        "batch_labels": resources["batch_labels"],
        "batch_parent_counts": resources["batch_parent_counts"],
        "batch_parent_ids": resources["batch_parent_ids"],
        "batch_freeze_status": resources["batch_freeze_status"],
        "G4": resources["G4"],
        "volatile_elapsed_rss_and_meminfo_in_package": False,
    }


def _run_fixed_evidence(
    preflight: dict[str, Any], batch_timeout_seconds: int, total_timeout_seconds: int
) -> dict[str, Any]:
    """Run exactly the prospectively selected D-073 mode without fallback."""
    if preflight["selected_execution_mode"] != g3.G3_004_EXECUTION_MODE:
        raise G3CliError("preflight execution mode changed from D-073")
    return g3.build_freeze_evidence(
        execution_mode=g3.G3_004_EXECUTION_MODE,
        batch_timeout_seconds=batch_timeout_seconds,
        total_timeout_seconds=total_timeout_seconds,
    )


def _figure_bytes(figure: Any) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Crazyflow deterministic G3.2 evidence generator"},
    )
    return buffer.getvalue()


def _overview_figure(evidence: dict[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = evidence["report"]
    specs = evidence["specs"]
    features = np.asarray(report["baseline_features"], dtype=np.float64)
    names = tuple(report["feature_names"])
    classes = [item.value for item in g3.MotionClass]
    profiles = [item.value for item in g3.Profile]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    x = np.arange(len(profiles))
    width = 0.18
    for class_index, motion_class in enumerate(classes):
        rmse = []
        reserve = []
        for profile in g3.Profile:
            mask = np.asarray(
                [
                    spec.motion_class.value == motion_class and spec.profile is profile
                    for spec in specs
                ]
            )
            rmse.append(float(np.mean(features[mask, names.index("position_rmse_z")])))
            reserve.append(
                float(np.min(features[mask, names.index("minimum_normalized_motor_reserve")]))
            )
        offset = (class_index - 1.5) * width
        axes[0, 0].bar(x + offset, rmse, width, label=motion_class)
        axes[0, 1].bar(x + offset, reserve, width, label=motion_class)
    categories = report["single_parameter_categories"]
    parameter_labels = list(g3.PARAMETER_NAMES)
    train_rms = [categories[name]["detectability"]["train_column_rms"] for name in parameter_labels]
    validation_rms = [
        categories[name]["detectability"]["validation_column_rms"] for name in parameter_labels
    ]
    px = np.arange(len(parameter_labels))
    axes[1, 0].bar(px - 0.18, train_rms, 0.36, label="Train")
    axes[1, 0].bar(px + 0.18, validation_rms, 0.36, label="Validation")
    category_counts = {
        category: sum(item["category"] == category for item in categories.values())
        for category in (
            "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION",
            "DIAGNOSTIC_ONLY",
            "NO_GO",
            g3.WITHHELD_PARAMETER_CATEGORY,
        )
    }
    category_labels = list(category_counts)
    category_values = [category_counts[label] for label in category_labels]
    axes[1, 1].bar(
        category_labels, category_values, color=("#2a9d8f", "#e9c46a", "#e76f51", "#6d597a")
    )
    axes[0, 0].set_title("Default scored Z-RMSE by class/profile")
    axes[0, 0].set_ylabel("m")
    axes[0, 1].set_title("Minimum scored normalized motor reserve")
    axes[0, 1].set_ylabel("fraction of force span")
    axes[1, 0].set_title("Continuous normalized sensitivity column RMS")
    axes[1, 0].axhline(0.02, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(px, parameter_labels, rotation=25)
    axes[1, 1].set_title("Continuous single-parameter categories")
    axes[1, 1].tick_params(axis="x", rotation=20)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_xticks(x, profiles)
    axes[0, 1].set_xticks(x, profiles)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend()
    figure.suptitle(
        "cf21B_500 G3.4 parameter eligibility audit\n"
        "Float32 mass_thrust relaxation is simulation-only; integer response is separate; "
        "zero optimizer updates"
    )
    payload = _figure_bytes(figure)
    plt.close(figure)
    return payload


def _sensitivity_figure(evidence: dict[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_names = evidence["feature_names"]
    matrices = evidence["matrices"]
    specs = evidence["specs"]
    families = ("tracking", "integral", "wrench")
    splits = (g3.Split.TRAIN, g3.Split.VALIDATION)
    rows = []
    labels = []
    for split in splits:
        mask = np.asarray(
            [
                spec.split is split and spec.motion_class is not g3.MotionClass.NEAR_LIMIT
                for spec in specs
            ]
        )
        for family in families:
            indices = [
                index
                for index, name in enumerate(feature_names)
                if g3._feature_family(name) == family
            ]
            rows.append(
                [
                    float(np.sqrt(np.mean(matrices[name][mask][:, indices] ** 2)))
                    for name in g3.PARAMETER_NAMES
                ]
            )
            labels.append(f"{split.value} / {family}")
    matrix = np.asarray(rows)
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(g3.PARAMETER_NAMES)), g3.PARAMETER_NAMES, rotation=25)
    axis.set_yticks(np.arange(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > 0.5 * np.max(matrix) else "black",
            )
    axis.set_title(
        "Normalized effect sensitivity RMS by split and feature family\n"
        "Diagnostic AD is not an optimizer-gradient contract; no firmware/calibration transfer"
    )
    figure.colorbar(image, ax=axis, label="dimensionless RMS")
    payload = _figure_bytes(figure)
    plt.close(figure)
    return payload


def _handoff(evidence: dict[str, Any], generator_commit: str, resources: dict[str, Any]) -> bytes:
    proposal = evidence["g4_g5_freeze_proposal"]
    report = evidence["report"]
    categories = evidence["sensitivity_conditioning"]["single_parameters"]
    category_lines = "\n".join(
        f"- `{name}`: `{categories[name]['category']}`" for name in g3.PARAMETER_NAMES
    )
    selected = proposal["selected_parameters"] or ["none; freeze withheld"]
    withheld = report["withheld_parameters"]
    withholding_sentence = (
        " Withheld parameter reasons: "
        + "; ".join(
            f"`{name}` = `{'+'.join(categories[name]['withholding_reasons'])}`" for name in withheld
        )
        + "."
        if withheld
        else " No parameter was withheld by an objective-gradient or technical gate."
    )
    execution_sentence = (
        "D-073 prospectively selected six contiguous four-parent batches after the observed "
        "local primary-24 OOM. This is not a runtime fallback; the prospective 32-world G4 "
        "batch freeze is WITHHELD_PENDING_REBENCHMARK."
    )
    return (
        "# Day 26 technical handoff: G3.4 parameter eligibility audit\n\n"
        "## Status\n\n"
        f"`{report['status']}` on `cf21B_500`. Exactly 24 "
        "Train/Validation parents, full six-second carry, unchanged Loss v1, and zero optimizer "
        f"initializations/updates. No G4 or G5 run occurred.{withholding_sentence}\n\n"
        "## Required focused sequence\n\n"
        "1. Static leaf/dtype and direct state-controller parity: PASS.\n"
        "2. Two-parent full default parity across controller, allocation, rollout, loss, features, "
        "and counters: PASS with exact array equality.\n"
        "3. All seven log-coordinates pass scalar/vector Reverse/FD at `h=0.01` on both "
        "frozen focus parents: finite, sign-consistent when significant, and within tolerance.\n"
        "4. Separate `DISCRETE_INTEGER_RESPONSE` at ±0.01 and ±0.05: reported without an AD or "
        "GO label. It neither replaces nor validates the continuous reverse-mode relaxation.\n\n"
        "## Continuous classifications\n\n"
        f"{category_lines}\n\n"
        "The seven Loss-v1 objective outputs alone carry the optimizer-gradient contract. The "
        "other 136 finite outputs retain complete AD comparisons as "
        "`DIAGNOSTIC_AD_NOT_OPTIMIZER_CONTRACT`; their mismatches are disclosed without an "
        "optimizer-gradient or causal claim. Objective-gradient and technical-withholding reasons "
        "remain separate and cumulative. A technical record identifies only nonfinite "
        "full-effect-rollout leaf and array-index metadata and makes no causal, "
        "controller-operation, singularity, firmware, or hardware claim.\n\n"
        "`mass` and `mass_thrust` remain effective controller parameters. The latter category "
        "applies only to its simulation relaxation and is not an integer, calibration, or firmware "
        "claim.\n\n"
        "## Inactive integral candidate\n\n"
        "Recommendation: "
        f"`{evidence['loss_diagnostics']['inactive_candidate']['recommendation']}`. "
        "Its active weight is exactly zero and it was not added to the objective.\n\n"
        "## Prospective freeze only\n\n"
        f"Status: `{proposal['status']}`; proposed parameters: `{', '.join(selected)}`. This is a "
        "Hauptleitung review proposal, not candidate selection or permission to start G4.\n\n"
        "## Reproduction\n\n"
        f"Generator commit: `{generator_commit}`. {execution_sentence} Volatile time/RSS "
        "observations remain only in `/tmp` process manifests. Required command:\n\n"
        "```bash\n"
        "PYTHONDONTWRITEBYTECODE=1 SCIPY_ARRAY_API=1 JAX_PLATFORM_NAME=cpu "
        "JAX_ENABLE_X64=false JAX_ENABLE_COMPILATION_CACHE=false "
        "/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python "
        "examples/jax/mellinger_g3_freeze_v2.py --generate-package "
        f"--generator-commit {generator_commit} --execution-mode six-batches-of-four "
        "--batch-timeout-seconds 900 "
        "--total-timeout-seconds 2400 --output-dir <NEW_OUTPUT_DIR>\n"
        "```\n\n"
        "## Claim boundary\n\n"
        "Pure JAX Float32 CPU simulation evidence. No optimized parameter, improvement, Test "
        "result, physical calibration, integer/firmware differentiability, hardware transfer, "
        "safety, flight readiness, or Sim2Real claim.\n"
    ).encode()


def _provenance(
    evidence: dict[str, Any], generator_commit: str, resources: dict[str, Any]
) -> dict[str, Any]:
    payload_hashes = {
        "trajectory_contract": _sha256_bytes(_json_bytes(evidence["trajectory_contract"])),
        "train_manifest": _sha256_bytes(_json_bytes(evidence["train_manifest"])),
        "validation_manifest": _sha256_bytes(_json_bytes(evidence["validation_manifest"])),
    }
    return {
        "schema_version": g3.SCHEMA_VERSION,
        "work_order": "WO-GR-G3-004",
        "source_base_commit": g3.SOURCE_BASE_COMMIT,
        "generator_commit": generator_commit,
        "generator_commit_resolution": {
            "subject": GENERATOR_COMMIT_SUBJECT,
            "paths": sorted(GENERATOR_COMMIT_PATHS),
            "stable_across_generator_and_artifact_commit": True,
        },
        "integration_target_branch": INTEGRATION_TARGET_BRANCH,
        "checkout_branch_included": False,
        "absolute_worktree_path_included": False,
        "timestamp_included": False,
        "platform": g3.PLATFORM,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "mjx_impl": evidence["report"]["scope"]["mjx_impl"],
        "warp_installed": evidence["report"]["scope"]["warp_installed"],
        "devices": [f"{device.platform}:{device.device_kind}" for device in jax.devices()],
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("flax", "matplotlib", "mujoco", "scipy")
        },
        "input_output_inventory": {
            "parents": 24,
            "package_payloads": list(PACKAGE_NAMES),
            "checksum_entries": 12,
        },
        "contract_and_manifest_hashes": payload_hashes,
        "execution": {
            "mode": resources["execution_mode"],
            "selection_decision": resources["selection_decision"],
            "selection_policy": resources["selection_policy"],
            "selected_by_host_memory": resources["selected_by_host_memory"],
            "runtime_fallback_used": resources["runtime_fallback_used"],
            "runtime_fallback_reason": resources["runtime_fallback_reason"],
            "batch_labels": resources["batch_labels"],
            "batch_parent_counts": resources["batch_parent_counts"],
            "batch_parent_ids": resources["batch_parent_ids"],
            "batch_timeout_seconds": 900,
            "total_timeout_seconds": 2400,
            "resource_contract_and_result": _stable_resource_record(resources),
            "G4": resources["G4"],
        },
        "protected_access": {
            "test_manifest_data": False,
            "day10_raw_runs": False,
            "day13_replication": False,
            "protected_primary_data": False,
        },
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
        "claim_boundary": evidence["report"]["claim_boundary"],
    }


def build_package_payloads(generator_commit: str, execution_mode: str) -> dict[str, bytes]:
    if execution_mode != g3.G3_004_EXECUTION_MODE:
        raise G3CliError("execution mode substitution is forbidden")
    if jax.default_backend() != "cpu" or jax.config.jax_enable_x64:
        raise G3CliError("package generation requires CPU with JAX x64 disabled")
    _validate_generator_commit(generator_commit)
    preflight = _preflight_resources()
    start = time.monotonic()
    evidence = _run_fixed_evidence(preflight, 900, 2400)
    overview_figure = _overview_figure(evidence)
    sensitivity_figure = _sensitivity_figure(evidence)
    elapsed = time.monotonic() - start
    resources = _postflight_resources(preflight, elapsed, evidence)
    report = dict(evidence["report"])
    report["generator_commit"] = generator_commit
    report["resource_execution"] = _stable_resource_record(resources)
    report["output_inventory"] = [*PACKAGE_NAMES, "SHA256SUMS"]
    payloads = {
        "g3_freeze_report.json": _json_bytes(report),
        "trajectory_contract.json": _json_bytes(evidence["trajectory_contract"]),
        "train_manifest.json": _json_bytes(evidence["train_manifest"]),
        "validation_manifest.json": _json_bytes(evidence["validation_manifest"]),
        "parameter_semantics.json": _json_bytes(evidence["parameter_semantics"]),
        "sensitivity_conditioning.json": _json_bytes(evidence["sensitivity_conditioning"]),
        "loss_diagnostics.json": _json_bytes(evidence["loss_diagnostics"]),
        "g4_g5_freeze_proposal.json": _json_bytes(evidence["g4_g5_freeze_proposal"]),
        "g3_freeze_overview.png": overview_figure,
        "parameter_sensitivity_matrix.png": sensitivity_figure,
        "TECHNICAL_HANDOFF.md": _handoff(evidence, generator_commit, resources),
        "provenance.json": _json_bytes(_provenance(evidence, generator_commit, resources)),
    }
    if tuple(payloads) != PACKAGE_NAMES:
        raise G3CliError("package payload inventory changed")
    return payloads


def _dry_run_png(label: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(2.0, 1.5), constrained_layout=True)
    axis.plot([0.0, 1.0], [0.0, 1.0])
    axis.set_title(label)
    payload = _figure_bytes(figure)
    plt.close(figure)
    return payload


def _dry_run_payloads(snapshot_sha256: str) -> dict[str, bytes]:
    """Return a small deterministic schema fixture; this is never scientific evidence."""
    common = {
        "schema_version": g3.SCHEMA_VERSION,
        "status": "PRECOMMIT_SCHEMA_FIXTURE_NOT_SCIENTIFIC_EVIDENCE",
        "work_order": "WO-GR-G3-004",
        "source_snapshot_sha256": snapshot_sha256,
        "scientific_parent_count": 0,
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }
    payloads = {
        "g3_freeze_report.json": _json_bytes(common | {"fixture_kind": "report"}),
        "trajectory_contract.json": _json_bytes(common | {"fixture_kind": "contract"}),
        "train_manifest.json": _json_bytes(common | {"fixture_kind": "train_manifest"}),
        "validation_manifest.json": _json_bytes(common | {"fixture_kind": "validation_manifest"}),
        "parameter_semantics.json": _json_bytes(common | {"fixture_kind": "semantics"}),
        "sensitivity_conditioning.json": _json_bytes(common | {"fixture_kind": "conditioning"}),
        "loss_diagnostics.json": _json_bytes(common | {"fixture_kind": "loss"}),
        "g4_g5_freeze_proposal.json": _json_bytes(common | {"fixture_kind": "proposal"}),
        "g3_freeze_overview.png": _dry_run_png("schema overview"),
        "parameter_sensitivity_matrix.png": _dry_run_png("schema matrix"),
        "TECHNICAL_HANDOFF.md": (
            "# PRECOMMIT DRY-RUN FIXTURE\n\n"
            "Not scientific evidence. No optimizer, G4, firmware, hardware, or flight action.\n"
        ).encode(),
        "provenance.json": _json_bytes(
            common
            | {
                "fixture_kind": "provenance",
                "generator_identity_included": False,
                "integration_target_branch": INTEGRATION_TARGET_BRANCH,
            }
        ),
    }
    if tuple(payloads) != PACKAGE_NAMES:
        raise G3CliError("precommit fixture inventory changed")
    return payloads


def run_precommit_dry_run(output_dir: Path, snapshot_sha256: str) -> None:
    """Exercise the complete package schema and overwrite/checksum gates with fixtures."""
    observed_snapshot = source_snapshot_sha256()
    if snapshot_sha256 != observed_snapshot:
        raise G3CliError(
            f"source snapshot changed: expected {snapshot_sha256}, observed {observed_snapshot}"
        )
    payloads = _dry_run_payloads(snapshot_sha256)
    for name, payload in payloads.items():
        if name.endswith(".json"):
            json.loads(payload)
        elif name.endswith(".png") and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise G3CliError(f"dry-run PNG signature failed for {name}")
    if b"generator_commit" in b"".join(payloads.values()):
        raise G3CliError("precommit fixture must not contain a generator commit")
    write_package(output_dir, payloads)
    checksum_lines = (output_dir / "SHA256SUMS").read_text().splitlines()
    if len(checksum_lines) != 12:
        raise G3CliError("precommit checksum index is not 12/12")
    names = [line.split("  ", 1)[1] for line in checksum_lines]
    if names != sorted(PACKAGE_NAMES):
        raise G3CliError("precommit checksum index order changed")
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        if digest != _sha256_bytes((output_dir / name).read_bytes()):
            raise G3CliError(f"precommit checksum mismatch for {name}")
    try:
        write_package(output_dir, payloads)
    except G3CliError as error:
        if "overwrite" not in str(error):
            raise
    else:
        raise G3CliError("precommit overwrite refusal did not fire")


def write_package(output_dir: Path, payloads: dict[str, bytes]) -> None:
    if output_dir.exists():
        raise G3CliError(f"refusing to overwrite output directory: {output_dir}")
    if tuple(payloads) != PACKAGE_NAMES:
        raise G3CliError("package payload inventory is incomplete or reordered")
    index = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n" for name in sorted(PACKAGE_NAMES)
    ).encode()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    (output_dir / "SHA256SUMS").write_bytes(index)


def generate_package(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise G3CliError(f"refusing to overwrite output directory: {args.output_dir}")
    payloads = build_package_payloads(args.generator_commit, args.execution_mode)
    write_package(args.output_dir, payloads)


def write_g3_003_diagnosis(output_file: Path) -> None:
    if output_file.exists():
        raise G3CliError(f"refusing to overwrite diagnosis output: {output_file}")
    if not str(output_file).startswith("/tmp/gr-g3-003-"):
        raise G3CliError("diagnosis output must remain under /tmp/gr-g3-003-*")
    start = time.monotonic()
    diagnosis = g3.build_g3_003_diagnosis()
    elapsed = time.monotonic() - start
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if elapsed > 600.0:
        raise G3CliError(f"G3.3 diagnosis exceeded 600 s: {elapsed:.3f} s")
    if maximum_rss > 24 * 1024**3:
        raise G3CliError(f"G3.3 diagnosis exceeded 24 GiB RSS: {maximum_rss}")
    output_file.write_bytes(_json_bytes(diagnosis))


def write_g3_004_smoke(output_file: Path) -> None:
    """Write raw values before assertions, then the validated canonical smoke record."""
    if output_file.exists():
        raise G3CliError(f"refusing to overwrite smoke output: {output_file}")
    if not str(output_file.resolve()).startswith("/tmp/gr-g3-004-smoke-"):
        raise G3CliError("G3.4 smoke output must remain under /tmp/gr-g3-004-smoke-*")
    if not output_file.parent.is_dir():
        raise G3CliError("G3.4 smoke runroot must already exist")
    raw_file = output_file.with_name(f"{output_file.stem}.raw.json")
    if raw_file.exists():
        raise G3CliError(f"refusing to overwrite smoke raw record: {raw_file}")
    raw = g3.build_g3_004_smoke(validate=False)
    raw_file.write_bytes(_json_bytes(raw))
    validated = g3.validate_g3_004_smoke(raw)
    output_file.write_bytes(_json_bytes(validated))


def main(argv: list[str] | tuple[str, ...] | None = None) -> None:
    args = parse_args(argv)
    if args.diagnose_g3_003:
        write_g3_003_diagnosis(args.output_file)
    elif args.diagnose_g3_004_smoke:
        write_g3_004_smoke(args.output_file)
    elif args.precommit_dry_run:
        run_precommit_dry_run(args.output_dir, args.source_snapshot_sha256)
    else:
        generate_package(args)


if __name__ == "__main__":
    main()
