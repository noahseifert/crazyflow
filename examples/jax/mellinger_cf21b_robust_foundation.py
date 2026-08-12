"""Measure and package the frozen cf21B_500 robust-baseline foundation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from crazyflow.control.mellinger.research import robust_evaluation as robust

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCHEMA = "crazyflow.cf21b_robust_benchmark.v1"
IMPLEMENTATION_COMMIT_SUBJECT = "research: add cf21b robust evaluator foundation"
INTEGRATION_TARGET_BRANCH = "research/differentiable-mellinger"
POST_INTEGRATION_METADATA_DECISION = "D-048"
CHECKOUT_BRANCH_PROVENANCE_REASON = (
    "checkout branch is a mutable execution-context reference; generator_commit is the "
    "canonical immutable science/code identity"
)
IMPLEMENTATION_COMMIT_PATHS = {
    "crazyflow/control/mellinger/research/robust_evaluation.py",
    "examples/jax/mellinger_cf21b_robust_foundation.py",
    "tests/unit/test_mellinger_robust_evaluation.py",
    "tests/unit/test_mellinger_cf21b_robust_foundation.py",
}
PRE_CORRECTION_PACKAGE_ROOT = "artifacts/day25-cf21b-robust-foundation"
PACKAGE_NAMES = (
    "robust_evaluation_report.json",
    "episode_contract.json",
    "train_manifest.json",
    "validation_manifest.json",
    "benchmark_measurements.json",
    "robust_foundation_overview.png",
    "batch_scaling.png",
    "TECHNICAL_HANDOFF.md",
    "provenance.json",
)


class FoundationCliError(RuntimeError):
    """Raised before output on a CLI, benchmark, or package contract failure."""


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse mutually exclusive benchmark, child, and package modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--measure-benchmark", action="store_true")
    modes.add_argument("--generate-package", action="store_true")
    modes.add_argument("--benchmark-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--benchmark-input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--world-counts", type=int, nargs="+", default=[4, 16, 32])
    parser.add_argument("--steady-repeats", type=int, default=3)
    parser.add_argument("--child-timeout-seconds", type=int, default=300)
    parser.add_argument("--world-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.measure_benchmark and args.benchmark_output is None:
        parser.error("--measure-benchmark requires --benchmark-output")
    if args.generate_package and (args.benchmark_input is None or args.output_dir is None):
        parser.error("--generate-package requires --benchmark-input and --output-dir")
    if args.benchmark_child and (args.world_count is None or args.child_output is None):
        parser.error("--benchmark-child requires --world-count and --child-output")
    return args


def _json_bytes(value: Any) -> bytes:
    return robust.canonical_json_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _generator_commit() -> str:
    """Resolve the truthful first source/test commit from the complete later history."""
    commits = _git("rev-list", "--reverse", f"{robust.SOURCE_BASE_COMMIT}..HEAD").splitlines()
    if not commits or any(
        len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit)
        for commit in commits
    ):
        raise FoundationCliError("expected nonempty well-formed 40-hex commit history")
    implementation = commits[0]
    if _git("show", "-s", "--format=%s", implementation) != IMPLEMENTATION_COMMIT_SUBJECT:
        raise FoundationCliError("first result commit is not the prescribed source/test commit")
    paths = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", implementation).splitlines()
    )
    if paths != IMPLEMENTATION_COMMIT_PATHS:
        raise FoundationCliError("source/test generator commit path inventory changed")
    return implementation


def _git_blob(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ("git", "show", f"{commit}:{path}"), cwd=REPOSITORY_ROOT, check=True, capture_output=True
    )
    return result.stdout


def _pre_correction_invariance(
    report: dict[str, Any], contract: dict[str, Any], benchmark_bytes: bytes
) -> dict[str, Any]:
    """Run the mandatory explicit c4-old-side comparator before payload creation."""
    old_commit = robust.PRE_CORRECTION_RESULT_COMMIT
    return robust.pre_correction_invariance_evidence(
        old_report_bytes=_git_blob(
            old_commit, f"{PRE_CORRECTION_PACKAGE_ROOT}/robust_evaluation_report.json"
        ),
        old_contract_bytes=_git_blob(
            old_commit, f"{PRE_CORRECTION_PACKAGE_ROOT}/episode_contract.json"
        ),
        old_benchmark_bytes=_git_blob(
            old_commit, f"{PRE_CORRECTION_PACKAGE_ROOT}/benchmark_measurements.json"
        ),
        new_report=report,
        new_contract=contract,
        benchmark_bytes=benchmark_bytes,
    )


def _parse_kib_file(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        fields = raw.strip().split()
        if fields and fields[0].isdigit():
            value = int(fields[0])
            if len(fields) > 1 and fields[1] == "kB":
                value *= 1024
            result[name] = value
    return result


def _memory_snapshot() -> dict[str, Any]:
    meminfo = _parse_kib_file(Path("/proc/meminfo"))
    status = _parse_kib_file(Path("/proc/self/status"))
    return {
        "mem_total_bytes": meminfo.get("MemTotal"),
        "mem_available_bytes": meminfo.get("MemAvailable"),
        "swap_total_bytes": meminfo.get("SwapTotal"),
        "swap_free_bytes": meminfo.get("SwapFree"),
        "vm_peak_bytes": status.get("VmPeak"),
        "vm_hwm_bytes": status.get("VmHWM"),
        "vm_swap_bytes": status.get("VmSwap", 0),
    }


def _block_until_ready(value: Any) -> Any:
    return jax.block_until_ready(value)


def _benchmark_child(world_count: int, steady_repeats: int, output: Path) -> None:
    """Measure exactly one world count in one fresh CPU process."""
    if steady_repeats != 3:
        raise FoundationCliError("the frozen benchmark requires exactly three steady repeats")
    if output.exists():
        raise FoundationCliError(f"refusing to overwrite child output: {output}")
    before = _memory_snapshot()
    sim, initial, inputs, raw, input_metadata = robust.benchmark_block_inputs(world_count)
    value_and_grad = robust.benchmark_value_and_grad(sim, initial, inputs)
    compiled_entry = jax.jit(value_and_grad)

    lower_start = time.perf_counter_ns()
    lowered = compiled_entry.lower(raw)
    lower_ns = time.perf_counter_ns() - lower_start
    compile_start = time.perf_counter_ns()
    compiled = lowered.compile()
    compile_ns = time.perf_counter_ns() - compile_start

    first_start = time.perf_counter_ns()
    first_value = compiled(raw)
    _block_until_ready(first_value)
    first_ns = time.perf_counter_ns() - first_start
    warmup_value = compiled(raw)
    _block_until_ready(warmup_value)
    steady_ns = []
    final_value = warmup_value
    for _ in range(steady_repeats):
        start = time.perf_counter_ns()
        final_value = compiled(raw)
        _block_until_ready(final_value)
        steady_ns.append(time.perf_counter_ns() - start)
    loss, gradient = final_value
    after = _memory_snapshot()
    ru_maxrss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    median_ns = int(statistics.median(steady_ns))
    if not np.isfinite(float(loss)) or not np.all(np.isfinite(np.asarray(gradient))):
        raise FoundationCliError("benchmark child produced nonfinite loss or gradient")
    result = {
        "world_count": world_count,
        "process_id": os.getpid(),
        "timing": {
            "timer": "time.perf_counter_ns",
            "lowering_ns": lower_ns,
            "compile_ns": compile_ns,
            "lowering_plus_compile_ns": lower_ns + compile_ns,
            "first_synchronized_execution_ns": first_ns,
            "unmeasured_warmup_count": 1,
            "steady_synchronized_execution_ns": steady_ns,
            "steady_median_ns": median_ns,
            "steady_worlds_per_s": world_count / (median_ns / 1.0e9),
            "steady_scored_episode_seconds_per_s": (world_count * robust.SCORE_DURATION_S)
            / (median_ns / 1.0e9),
        },
        "loss": float(loss),
        "gradient": np.asarray(gradient).tolist(),
        "memory": {
            "before": before,
            "after": after,
            "ru_maxrss_bytes": int(ru_maxrss_kib * 1024),
            "swap_activity_detected": bool(
                after.get("vm_swap_bytes", 0)
                or (
                    before.get("swap_free_bytes") is not None
                    and after.get("swap_free_bytes") is not None
                    and after["swap_free_bytes"] < before["swap_free_bytes"]
                )
            ),
        },
        "environment": {
            "backend": jax.default_backend(),
            "device": str(jax.devices("cpu")[0]),
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "jax_x64": bool(jax.config.jax_enable_x64),
        },
        "input_contract": input_metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(result))


def _validate_benchmark_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != BENCHMARK_SCHEMA:
        raise FoundationCliError("benchmark schema changed")
    checksum = payload.get("raw_measurement_sha256")
    core = dict(payload)
    core.pop("raw_measurement_sha256", None)
    if checksum != _sha256_bytes(_json_bytes(core)):
        raise FoundationCliError("benchmark raw-measurement checksum failed")
    if payload.get("status") != "PASS_4_16_32_BENCHMARK":
        raise FoundationCliError("benchmark status is not PASS")
    results = payload.get("results", [])
    if [item.get("world_count") for item in results] != [4, 16, 32]:
        raise FoundationCliError("benchmark result inventory is not exact 4/16/32")
    robust.benchmark_consistency_check(results)


def measure_benchmark(
    output: Path, world_counts: list[int], steady_repeats: int, child_timeout_seconds: int
) -> dict[str, Any]:
    """Measure 4/16/32 in fresh processes and write one checksum-pinned raw dataset."""
    if world_counts != [4, 16, 32]:
        raise FoundationCliError("world counts must be exactly 4 16 32")
    if steady_repeats != 3 or child_timeout_seconds != 300:
        raise FoundationCliError("benchmark repeats/child timeout changed from 3/300")
    if output.exists():
        raise FoundationCliError(f"refusing to overwrite benchmark output: {output}")
    overall_start = time.perf_counter()
    results = []
    with tempfile.TemporaryDirectory(prefix="gr-g2-001-benchmark-", dir="/tmp") as temporary:
        temporary_dir = Path(temporary)
        for world_count in world_counts:
            if world_count == 32:
                available = _memory_snapshot().get("mem_available_bytes")
                if available is None or available < 8 * 1024**3:
                    raise FoundationCliError("32-world MemAvailable guard failed")
                if results[-1]["memory"]["ru_maxrss_bytes"] >= 24 * 1024**3:
                    raise FoundationCliError("16-world Max-RSS guard blocks 32 worlds")
            remaining = 900.0 - (time.perf_counter() - overall_start)
            if remaining <= 0.0:
                raise FoundationCliError("overall 900-second benchmark timeout reached")
            child_output = temporary_dir / f"worlds-{world_count}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--benchmark-child",
                "--world-count",
                str(world_count),
                "--steady-repeats",
                str(steady_repeats),
                "--child-output",
                str(child_output),
            ]
            try:
                subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    timeout=min(float(child_timeout_seconds), remaining),
                )
            except subprocess.TimeoutExpired as error:
                raise FoundationCliError(f"{world_count}-world child exceeded timeout") from error
            result = json.loads(child_output.read_text())
            if result["world_count"] != world_count:
                raise FoundationCliError("benchmark child world count changed")
            results.append(result)
    consistency = robust.benchmark_consistency_check(results)
    allowed = [
        item
        for item in results
        if not item["memory"]["swap_activity_detected"]
        and item["memory"]["ru_maxrss_bytes"] <= 0.70 * item["memory"]["before"]["mem_total_bytes"]
    ]
    if not allowed:
        raise FoundationCliError("no benchmark count satisfies memory/swap recommendation guard")
    best_throughput = max(item["timing"]["steady_worlds_per_s"] for item in allowed)
    recommended = min(
        item["world_count"]
        for item in allowed
        if item["timing"]["steady_worlds_per_s"] >= 0.90 * best_throughput
    )
    core = {
        "schema_version": BENCHMARK_SCHEMA,
        "status": "PASS_4_16_32_BENCHMARK",
        "method": {
            "fresh_cpu_process_per_world_count": True,
            "world_counts": world_counts,
            "steady_repeats": steady_repeats,
            "child_timeout_seconds": child_timeout_seconds,
            "overall_timeout_seconds": 900,
            "representative_path": (
                "6-s rollout, per-world 2-s interior score, unchanged Loss v1, "
                "Stage-1 default raw-gain value_and_grad, repository mismatch"
            ),
            "raw_measurements_reused_for_package_reproduction": True,
        },
        "results": results,
        "consistency": consistency,
        "recommendation": {
            "world_count": recommended,
            "rule": (
                "smallest allowed count with >=90% of best allowed steady worlds/s; "
                "allowed requires zero child swap activity and Max-RSS <=70% MemTotal"
            ),
            "scope": "next CPU batch-size recommendation only; no global default or GPU claim",
        },
        "elapsed_seconds": time.perf_counter() - overall_start,
        "generator_commit": _generator_commit(),
    }
    payload = core | {"raw_measurement_sha256": _sha256_bytes(_json_bytes(core))}
    _validate_benchmark_payload(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(payload))
    return payload


def _figure_bytes(figure: Any) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "Crazyflow Day-25 deterministic evidence generator"},
    )
    return buffer.getvalue()


def _overview_figure(report: dict[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = ("repository_mismatch", "matched")
    labels = ("Mismatch 0.0393", "Physical-value match 0.04338")
    colors = ("#2474b5", "#d34f4f")
    classes = [item.value for item in robust.MotionClass]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(classes))
    width = 0.36
    for variant_index, (variant, label, color) in enumerate(
        zip(variants, labels, colors, strict=True)
    ):
        z_rmse = []
        position_rmse = []
        contacts = []
        reserve = []
        episodes = report["mass_variants"][variant]["episodes"]
        for motion_class in classes:
            selected = [item for item in episodes if item["motion_class"] == motion_class]
            z_rmse.append(
                np.mean([item["summary"]["tracking"]["position_rmse_z_m"] for item in selected])
            )
            position_rmse.append(
                np.mean([item["summary"]["tracking"]["position_rmse_total_m"] for item in selected])
            )
            contacts.append(
                sum(item["summary"]["integral"]["clip_detection"]["count"] for item in selected)
            )
            reserve.append(
                min(
                    item["summary"]["motor_reserve"]["minimum_normalized_force_reserve"]
                    for item in selected
                )
            )
        offset = (variant_index - 0.5) * width
        axes[0, 0].bar(x + offset, z_rmse, width, label=label, color=color)
        axes[0, 1].bar(x + offset, position_rmse, width, label=label, color=color)
        axes[1, 0].bar(x + offset, contacts, width, label=label, color=color)
        axes[1, 1].bar(x + offset, reserve, width, label=label, color=color)
    titles = (
        "Mean scored Z-RMSE (lower is descriptive, not a pass)",
        "Mean scored total position RMSE",
        "Scored-window integral-bound contact axis-samples",
        "Minimum scored-window normalized motor-force reserve",
    )
    ylabels = ("m", "m", "axis-samples", "fraction of force span")
    for axis, title, ylabel in zip(axes.flat, titles, ylabels, strict=True):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, classes, rotation=15)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    finding = report["mass_ablation"]["negative_physical_value_match_finding"]
    figure.suptitle(
        "cf21B_500 robust baseline — physical-value match worsens mean Z-RMSE and "
        f"negative-z contact (ΔRMSE={finding['matched_minus_mismatch']['mean_z_rmse_m']:.5f} m; "
        f"Δcontact={finding['matched_minus_mismatch']['negative_z_integral_contact_count']})\n"
        "Simulation-only baseline; no optimizer, candidate, firmware, hardware, or flight claim",
        fontsize=13,
    )
    payload = _figure_bytes(figure)
    plt.close(figure)
    return payload


def _scaling_figure(benchmark: dict[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = benchmark["results"]
    worlds = [item["world_count"] for item in results]
    throughput = [item["timing"]["steady_worlds_per_s"] for item in results]
    first_s = [item["timing"]["first_synchronized_execution_ns"] / 1.0e9 for item in results]
    steady_s = [item["timing"]["steady_median_ns"] / 1.0e9 for item in results]
    rss_gib = [item["memory"]["ru_maxrss_bytes"] / 1024**3 for item in results]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].plot(worlds, throughput, "o-", color="#2474b5")
    axes[0].set_ylabel("steady worlds/s")
    axes[1].plot(worlds, first_s, "o-", label="first synchronized")
    axes[1].plot(worlds, steady_s, "o-", label="steady median")
    axes[1].set_ylabel("seconds")
    axes[1].legend()
    axes[2].plot(worlds, rss_gib, "o-", color="#d34f4f")
    axes[2].set_ylabel("ru_maxrss GiB")
    for axis in axes:
        axis.set_xlabel("CPU worlds")
        axis.set_xticks(worlds)
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"Fresh-process CPU benchmark; recommended next batch: "
        f"{benchmark['recommendation']['world_count']} worlds\n"
        "Measured wallclock is pinned once and reused; no GPU/default claim"
    )
    payload = _figure_bytes(figure)
    plt.close(figure)
    return payload


def _handoff(report: dict[str, Any], benchmark: dict[str, Any]) -> bytes:
    finding = report["mass_ablation"]["negative_physical_value_match_finding"]
    mismatch = finding["variants"]["repository_mismatch"]
    matched = finding["variants"]["matched"]
    return (
        "# Day 25 technical handoff: cf21B_500 robust baseline foundation\n\n"
        "## Status\n\n"
        "`PASS_ROBUST_FOUNDATION_BASELINE` under prospective D-045. G2.1 and G2.2 remain "
        "historically blocked/rejected. Integral contact is measured baseline behavior, not a "
        "technical pass. All inherited nonintegral gates pass. The first independent review of "
        "result `c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` was NOT ACCEPTED; D-047 "
        "authorizes this sole transparency correction.\n\n"
        "## Disclosed retained trajectory component\n\n"
        "Every class contains the same fixed positive-z central climb: exact Float32 `0.06 m/s`, "
        "base displacement `0.06*(time-2 s)`, zero base acceleration/jerk, and exact product-rule "
        "derivatives through jerk under the same C3 smoothstep7 2 s entry / 4 s plateau / 2 s "
        "exit envelope as the Fourier parent. It is composed after Fourier class scaling and "
        "before the fixed center. D-047 changes no trajectory number. Absolute Z results apply "
        "only to trajectories containing this climb; neither climb nor controller mass is an "
        "isolated cause.\n\n"
        "## Duration and measurement scope\n\n"
        "References have 8 s parent support. Simulation is one continuous 6 s rollout from the "
        "parent start, with no reset at the half-open 2 s scored window after a 2/3/4 s warm-up. "
        "Finiteness, ground/floor and zero-thrust gates cover all 6 s. Torque, motor and Stage-B "
        "additional clipping, integral contact, tracking, Loss v1, reserve and wrench statements "
        "cover only the scored window.\n\n"
        "## Negative physical-value-match finding\n\n"
        f"Across 16 paired episodes, mismatch mean scored Z-RMSE is "
        f"`{mismatch['mean_z_rmse_m']:.9f} m` and physical-value-match mean is "
        f"`{matched['mean_z_rmse_m']:.9f} m`. Negative-z integral contact rises from "
        f"`{mismatch['negative_z_integral_contact_count']}` to "
        f"`{matched['negative_z_integral_contact_count']}` axis-samples. This is a negative "
        "finding, not improvement, candidate selection, or causal isolation.\n\n"
        "## Mechanism and transfer boundary\n\n"
        "Measured early-transient and scored arrays retain collective target-thrust-equivalent "
        "request, realized wrench, Z error/integral, motor command, and reserve. Separately, "
        "code inspection shows `state2attitude` uses `mass * (setpoint_acc - gravity_vec) + "
        "feedback`, followed by fixed `mass_thrust=132000` and nonlinear PWM-to-force mapping. "
        "That is code-path inference, not an isolated causal result. Controller mass is therefore "
        "a feedforward parameter in this chain, not a transferable physical-mass estimate or "
        "firmware/flight value.\n\n"
        "## D-047 invariance comparator\n\n"
        f"`{report['correction_invariance']['status']}` with explicit old side "
        f"`{report['correction_invariance']['old_side_result_commit']}`. All "
        f"`{report['correction_invariance']['numeric_report_projection']['old']['path_count']}` "
        "pre-existing numeric report paths, every retained parent digest/ID/seed/attempt/window, "
        "all old raw arrays/summaries/pair deltas/masses, and the pinned benchmark bytes are "
        "identical. Only new derived aggregates/metadata, corrected prose, stable generator "
        "provenance and checksums differ.\n\n"
        "## D-048 post-integration metadata closure\n\n"
        "The corrected scientific foundation review is `ACCEPTED`; D-048 changes metadata only. "
        f"The deterministic integration target is `{INTEGRATION_TARGET_BRANCH}`. The observed "
        "checkout branch is excluded from reproducible payload bytes because "
        f"{CHECKOUT_BRANCH_PROVENANCE_REASON}. The canonical immutable science/code identity "
        f"remains generator commit `{report['generator_commit']}`.\n\n"
        "## Benchmark\n\n"
        f"Fresh CPU processes completed 4/16/32 worlds with finite consistent objective and "
        f"gradient. The predeclared rule recommends `{benchmark['recommendation']['world_count']}` "
        "worlds for the next CPU batch only.\n\n"
        "## Claim boundary\n\n"
        "Pure Float32 Crazyflow simulation evidence. No optimizer update, Loss-v2, Test, GPU, "
        "candidate, superiority, safety, firmware, hardware, flight, or Sim2Real claim.\n"
    ).encode()


def _provenance(
    report: dict[str, Any], benchmark: dict[str, Any], benchmark_sha256: str, generator_commit: str
) -> dict[str, Any]:
    return {
        "schema_version": robust.SCHEMA_VERSION,
        "work_order": "WO-GR-G2-003",
        "decision": "D-045",
        "correction_decision": "D-047",
        "generator_commit": generator_commit,
        "generator_commit_resolution": {
            "rule": "first prescribed source/test commit after source_base_commit",
            "subject": IMPLEMENTATION_COMMIT_SUBJECT,
            "stable_from_implementation_or_evidence_commit": True,
        },
        "source_base_commit": robust.SOURCE_BASE_COMMIT,
        "integration_target_branch": INTEGRATION_TARGET_BRANCH,
        "checkout_branch_provenance": {
            "included_in_reproducible_payload": False,
            "reason": CHECKOUT_BRANCH_PROVENANCE_REASON,
        },
        "post_integration_metadata_decision": POST_INTEGRATION_METADATA_DECISION,
        "platform": robust.PLATFORM,
        "benchmark_input_sha256": benchmark_sha256,
        "benchmark_raw_measurement_sha256": benchmark["raw_measurement_sha256"],
        "report_input_identity": report["input_identity"],
        "vertical_excitation_contract": robust.vertical_excitation_machine_contract(),
        "duration_and_measurement_scope": robust.duration_scope_contract(),
        "pre_correction_invariance": report["correction_invariance"],
        "registry_fingerprint": report["configuration"]["registry_fingerprint"],
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("flax", "matplotlib", "mujoco", "scipy")
        },
        "reproduction_contract": (
            "benchmark wallclock measured once; complete packages regenerated byte-identically "
            "from the unchanged checksum-pinned benchmark input"
        ),
        "protected_access": {"test_manifest_or_split": False, "day10_or_day13_raw": False},
        "claim_boundary": (
            "Simulation-only baseline; no optimization, candidate, GPU, firmware, hardware, "
            "flight, safety, or transfer claim."
        ),
    }


def build_package_payloads(benchmark_path: Path) -> dict[str, bytes]:
    """Re-run every science gate and construct deterministic package payload bytes."""
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    _validate_benchmark_payload(benchmark)
    evidence = robust.build_foundation_evidence()
    report = evidence["report"]
    generator_commit = _generator_commit()
    benchmark_sha256 = _sha256_bytes(benchmark_bytes)
    report["status"] = "PASS_ROBUST_FOUNDATION_BASELINE"
    report["generator_commit"] = generator_commit
    report["benchmark"] = {
        "input_sha256": benchmark_sha256,
        "raw_measurement_sha256": benchmark["raw_measurement_sha256"],
        "status": benchmark["status"],
        "recommendation": benchmark["recommendation"],
    }
    report["output_inventory"] = [*PACKAGE_NAMES, "SHA256SUMS"]
    report["checksum_index"] = "SHA256SUMS"
    contract = evidence["episode_contract"]
    report["correction_invariance"] = _pre_correction_invariance(report, contract, benchmark_bytes)
    robust.validate_report_summaries(report)
    payloads = {
        "robust_evaluation_report.json": _json_bytes(report),
        "episode_contract.json": _json_bytes(contract),
        "train_manifest.json": _json_bytes(evidence["train_manifest"]),
        "validation_manifest.json": _json_bytes(evidence["validation_manifest"]),
        "benchmark_measurements.json": benchmark_bytes,
        "robust_foundation_overview.png": _overview_figure(report),
        "batch_scaling.png": _scaling_figure(benchmark),
        "TECHNICAL_HANDOFF.md": _handoff(report, benchmark),
    }
    payloads["provenance.json"] = _json_bytes(
        _provenance(report, benchmark, benchmark_sha256, generator_commit)
    )
    if tuple(payloads) != PACKAGE_NAMES:
        raise FoundationCliError("package payload inventory changed")
    return payloads


def write_package(output_dir: Path, payloads: dict[str, bytes]) -> None:
    """Write a new package atomically by inventory, refusing every overwrite."""
    if output_dir.exists():
        raise FoundationCliError(f"refusing to overwrite output directory: {output_dir}")
    if tuple(payloads) != PACKAGE_NAMES:
        raise FoundationCliError("package payload inventory is incomplete")
    index = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n" for name in sorted(PACKAGE_NAMES)
    ).encode()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    (output_dir / "SHA256SUMS").write_bytes(index)


def generate_package(benchmark_input: Path, output_dir: Path) -> None:
    """Validate the pinned benchmark, rerun science, and emit one deterministic package."""
    if output_dir.exists():
        raise FoundationCliError(f"refusing to overwrite output directory: {output_dir}")
    payloads = build_package_payloads(benchmark_input)
    write_package(output_dir, payloads)


def main(argv: list[str] | tuple[str, ...] | None = None) -> None:
    """Run the selected benchmark, child, or package mode."""
    args = parse_args(argv)
    if args.benchmark_child:
        _benchmark_child(args.world_count, args.steady_repeats, args.child_output)
    elif args.measure_benchmark:
        measure_benchmark(
            args.benchmark_output,
            args.world_counts,
            args.steady_repeats,
            args.child_timeout_seconds,
        )
    elif args.generate_package:
        generate_package(args.benchmark_input, args.output_dir)
    else:  # pragma: no cover - argparse enforces one mode
        raise AssertionError("unreachable CLI mode")


if __name__ == "__main__":
    main()
