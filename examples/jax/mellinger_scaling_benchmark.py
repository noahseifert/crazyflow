"""Measure compile, first execution, and steady-state gradient timing by world count."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np

from crazyflow.control.mellinger.research import (
    FIXED_SUPPORT_PREFIX_DISTRIBUTION,
    LEGACY_TRAJECTORY_DISTRIBUTION,
    ResearchConfig,
    Split,
    atomic_write_json,
    build_episode_batch,
    build_split_pipeline,
    canonical_dict,
    diagnose_trajectory_attempts,
    fingerprint,
    generate_trajectory,
    load_config,
    raw_from_data,
    research_objective,
    seed_key,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_IN_INTEGRATION_TEST = False
SOURCE_PATHS = ("crazyflow", "examples")


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/mellinger/workstation.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steady-repeats", type=int, default=5)
    parser.add_argument("--trajectory-diagnostics-only", action="store_true")
    parser.add_argument("--diagnostic-horizons", type=int, nargs="+")
    parser.add_argument("--diagnostic-seed-count", type=int, default=16)
    parser.add_argument("--diagnostic-episode-start", type=int, default=0)
    parser.add_argument("--diagnostic-max-attempts", type=int)
    return parser.parse_args(argv)


def _source_provenance() -> dict[str, object]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    source_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all", "--", *SOURCE_PATHS],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).splitlines()
    source_hash = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        source_hash.update(relative_path.encode())
        source_hash.update(
            subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{relative_path}"], cwd=REPOSITORY_ROOT, text=True
            )
            .strip()
            .encode()
        )
    source_hash.update(
        subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", *SOURCE_PATHS], cwd=REPOSITORY_ROOT
        )
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).splitlines()
    for relative_path in sorted(filter(None, untracked)):
        source_hash.update(relative_path.encode())
        source_hash.update((REPOSITORY_ROOT / relative_path).read_bytes())
    return {
        "git_head": head,
        "git_branch": branch,
        "source_dirty": bool(source_status),
        "source_status_short": source_status,
        "source_fingerprint": source_hash.hexdigest(),
    }


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_benchmark_failure(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    config_path: Path,
    failure_phase: str,
    error: Exception,
    process_start: int,
    config: ResearchConfig | None = None,
    effective_config: ResearchConfig | None = None,
) -> None:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    source = _source_provenance()
    failure = {
        "schema_version": "crazyflow.mellinger_scaling_benchmark.v3",
        "status": "failed",
        "failure_phase": failure_phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "worlds": args.worlds,
        "horizon_control_intervals": args.horizon,
        "config_path": str(config_path),
        "effective_config_fingerprint": (
            fingerprint(effective_config) if effective_config is not None else None
        ),
        "gain_stage": config.optimizer.gain_stage if config is not None else None,
        "root_seed": config.root_seed if config is not None else None,
        "episode_index": 0,
        "trajectory_distribution_version": (
            config.trajectory.distribution_version if config is not None else None
        ),
        "internal_wall_seconds_to_failure": (time.perf_counter_ns() - process_start) / 1.0e9,
        "process_max_rss_kib": usage.ru_maxrss,
        "process_max_rss_unit": "KiB on Linux",
        "command": shlex.join([sys.executable, *sys.argv]),
        **source,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "technical_gate": {"passed": False},
        "test_split_built_or_evaluated": False,
        "test_manifest_opened": False,
    }
    if config is not None:
        failure["domain_randomization"] = {
            "mass_enabled": config.domain_randomization.mass_enabled,
            "mass_half_width_kg": config.domain_randomization.mass_half_width_kg,
            "delay_enabled": config.domain_randomization.delay_enabled,
            "wrench_enabled": config.domain_randomization.wrench_enabled,
            "sensor_noise_enabled": False,
            "ukf_enabled": False,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "benchmark.json", failure)


def _numeric_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _accepted_diversity_summary(seed_records: list[dict]) -> dict[str, object]:
    accepted = [record["attempts"][-1] for record in seed_records if record["success"]]
    fields = {
        "position_displacement_vector_rms_m": ("position_displacement_m", "vector_norm_rms"),
        "position_displacement_vector_max_m": ("position_displacement_m", "vector_norm_max"),
        "speed_vector_rms_m_s": ("velocity_m_s", "vector_norm_rms"),
        "speed_vector_max_m_s": ("velocity_m_s", "vector_norm_max"),
        "acceleration_vector_rms_m_s2": ("acceleration_m_s2", "vector_norm_rms"),
        "acceleration_vector_max_m_s2": ("acceleration_m_s2", "vector_norm_max"),
        "jerk_vector_rms_m_s3": ("jerk_m_s3", "vector_norm_rms"),
        "jerk_vector_max_m_s3": ("jerk_m_s3", "vector_norm_max"),
    }
    summary = {
        name: _numeric_summary([float(item["statistics"][group][statistic]) for item in accepted])
        for name, (group, statistic) in fields.items()
    }
    for axis, axis_name in enumerate(("x", "y", "z")):
        summary[f"position_span_{axis_name}_m"] = _numeric_summary(
            [float(item["statistics"]["position_m"]["component_span"][axis]) for item in accepted]
        )
        summary[f"spectral_centroid_{axis_name}_hz"] = _numeric_summary(
            [
                float(item["statistics"]["spectrum"]["per_axis"][axis]["spectral_centroid_hz"])
                for item in accepted
                if item["statistics"]["spectrum"]["per_axis"][axis]["spectral_centroid_hz"]
                is not None
            ]
        )
    return summary


def _diagnostic_horizon_summary(horizon_record: dict) -> dict[str, object]:
    seeds = horizon_record["seeds"]
    successes = [record for record in seeds if record["success"]]
    rejection_counts = Counter()
    maximum_violations: dict[str, dict[str, object]] = {}
    for seed_record in seeds:
        for attempt in seed_record["attempts"]:
            for constraint in attempt["constraints"]:
                if not constraint["passed"]:
                    rejection_counts[constraint["name"]] += 1
                violation = constraint["violation_magnitude"]
                if violation is None or violation <= 0.0:
                    continue
                previous = maximum_violations.get(constraint["name"])
                if previous is None or violation > previous["violation_magnitude"]:
                    maximum_violations[constraint["name"]] = {
                        **constraint,
                        "diagnostic_episode_index": seed_record["episode_index"],
                        "attempt_index": attempt["attempt_index"],
                    }
    attempts_to_success = [record["attempt_count"] for record in successes]
    return {
        "horizon_control_intervals": horizon_record["horizon_control_intervals"],
        "duration_seconds": horizon_record["duration_seconds"],
        "seed_count": len(seeds),
        "success_count": len(successes),
        "failure_count": len(seeds) - len(successes),
        "success_rate_within_attempt_limit": len(successes) / len(seeds),
        "attempts_to_success": _numeric_summary([float(value) for value in attempts_to_success]),
        "attempted_candidate_count": sum(len(record["attempts"]) for record in seeds),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "maximum_constraint_violations": dict(sorted(maximum_violations.items())),
        "accepted_diversity": _accepted_diversity_summary(seeds),
        "construction_wall_seconds": horizon_record["construction_wall_seconds"],
    }


def _arrays_equal(left: object, right: object) -> bool:
    return all(
        np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _paired_distribution_consistency(
    config: ResearchConfig, horizons: list[int], seed_records: list[dict]
) -> dict[str, object] | None:
    required_horizons = (100, 200, 400)
    if config.trajectory.distribution_version != FIXED_SUPPORT_PREFIX_DISTRIBUTION or not set(
        required_horizons
    ).issubset(horizons):
        return None
    legacy_config = replace(
        config.trajectory,
        distribution_version=LEGACY_TRAJECTORY_DISTRIBUTION,
        support_duration_s=None,
        minimum_duration_s=None,
    )
    per_seed = []
    reference_h400_position = None
    distinct_from_first_count = 0
    for seed_record in seed_records:
        key = seed_key(
            seed_record["root_seed"],
            Split.TRAIN,
            seed_record["episode_index"],
            seed_record["world"],
            seed_record["component"],
        )
        generated = {
            horizon: generate_trajectory(
                key, horizon, config.train.control_freq_hz, config.trajectory
            )
            for horizon in required_horizons
        }
        legacy_h400 = generate_trajectory(key, 400, config.train.control_freq_hz, legacy_config)
        position_h400 = np.asarray(generated[400].trajectory.pos)
        if reference_h400_position is None:
            reference_h400_position = position_h400
        elif not np.array_equal(position_h400, reference_h400_position):
            distinct_from_first_count += 1
        finite = all(
            np.all(np.isfinite(np.asarray(leaf)))
            for item in generated.values()
            for leaf in jax.tree.leaves((item.trajectory, item.jerk))
        )
        per_seed.append(
            {
                "episode_index": seed_record["episode_index"],
                "accepted_attempt_indices": {
                    str(horizon): generated[horizon].attempt for horizon in required_horizons
                },
                "same_attempt_across_horizons": len(
                    {generated[horizon].attempt for horizon in required_horizons}
                )
                == 1,
                "h100_exact_prefix_of_h400": _arrays_equal(
                    (generated[100].trajectory, generated[100].jerk),
                    (
                        jax.tree.map(lambda value: value[:101], generated[400].trajectory),
                        generated[400].jerk[:101],
                    ),
                ),
                "h200_exact_prefix_of_h400": _arrays_equal(
                    (generated[200].trajectory, generated[200].jerk),
                    (
                        jax.tree.map(lambda value: value[:201], generated[400].trajectory),
                        generated[400].jerk[:201],
                    ),
                ),
                "h400_exact_legacy_v1": _arrays_equal(
                    (generated[400].trajectory, generated[400].jerk),
                    (legacy_h400.trajectory, legacy_h400.jerk),
                ),
                "h400_attempt_matches_legacy_v1": generated[400].attempt == legacy_h400.attempt,
                "all_returned_arrays_finite": bool(finite),
            }
        )
    required_checks = (
        "same_attempt_across_horizons",
        "h100_exact_prefix_of_h400",
        "h200_exact_prefix_of_h400",
        "h400_exact_legacy_v1",
        "h400_attempt_matches_legacy_v1",
        "all_returned_arrays_finite",
    )
    return {
        "required_horizons": list(required_horizons),
        "seed_count": len(seed_records),
        "per_seed": per_seed,
        "all_required_checks_passed": all(
            record[check] for record in per_seed for check in required_checks
        ),
        "different_seed_h400_positions_distinct_from_first_count": distinct_from_first_count,
        "different_seed_results_requirement_passed": distinct_from_first_count
        == max(0, len(seed_records) - 1),
    }


def _run_trajectory_diagnostics(
    *,
    args: argparse.Namespace,
    config_path: Path,
    config: ResearchConfig,
    output_dir: Path,
    process_start: int,
) -> None:
    horizons = args.diagnostic_horizons or [args.horizon]
    if any(horizon < 2 for horizon in horizons):
        raise ValueError("all diagnostic horizons must be at least two")
    if args.diagnostic_seed_count < 1 or args.diagnostic_episode_start < 0:
        raise ValueError("diagnostic seed count must be positive and episode start nonnegative")
    attempt_limit = args.diagnostic_max_attempts or config.trajectory.max_attempts
    if attempt_limit < 1:
        raise ValueError("diagnostic max attempts must be positive")
    episode_indices = list(
        range(
            args.diagnostic_episode_start,
            args.diagnostic_episode_start + args.diagnostic_seed_count,
        )
    )
    seed_records = []
    for episode_index in episode_indices:
        key = seed_key(config.root_seed, Split.TRAIN, episode_index, 0, "trajectory")
        seed_records.append(
            {
                "root_seed": config.root_seed,
                "split": Split.TRAIN.value,
                "episode_index": episode_index,
                "world": 0,
                "component": "trajectory",
                "jax_key_data": np.asarray(jax.random.key_data(key)).tolist(),
            }
        )
    seed_suite = {
        "suite_id": (
            f"train-trajectory-root-{config.root_seed}-episodes-"
            f"{episode_indices[0]}-{episode_indices[-1]}-world-0"
        ),
        "coordinates": seed_records,
    }
    seed_suite["sha256"] = _sha256_json(seed_suite)
    horizon_records = []
    for horizon in horizons:
        horizon_start = time.perf_counter_ns()
        per_seed = []
        for seed_record in seed_records:
            key = seed_key(
                seed_record["root_seed"],
                Split.TRAIN,
                seed_record["episode_index"],
                seed_record["world"],
                seed_record["component"],
            )
            seed_start = time.perf_counter_ns()
            reports = diagnose_trajectory_attempts(
                key,
                horizon,
                config.train.control_freq_hz,
                config.trajectory,
                max_attempts=attempt_limit,
            )
            per_seed.append(
                {
                    "episode_index": seed_record["episode_index"],
                    "jax_key_data": seed_record["jax_key_data"],
                    "success": reports[-1].accepted,
                    "attempt_count": len(reports),
                    "construction_wall_seconds": (time.perf_counter_ns() - seed_start) / 1.0e9,
                    "attempts": [asdict(report) for report in reports],
                }
            )
        horizon_records.append(
            {
                "horizon_control_intervals": horizon,
                "duration_seconds": horizon / config.train.control_freq_hz,
                "construction_wall_seconds": (time.perf_counter_ns() - horizon_start) / 1.0e9,
                "seeds": per_seed,
            }
        )
    deterministic_horizons = []
    for horizon_record in horizon_records:
        deterministic_horizons.append(
            {
                **{k: v for k, v in horizon_record.items() if k != "construction_wall_seconds"},
                "seeds": [
                    {k: v for k, v in record.items() if k != "construction_wall_seconds"}
                    for record in horizon_record["seeds"]
                ],
            }
        )
    scientific_payload = {
        "config": canonical_dict(config),
        "config_fingerprint": fingerprint(config),
        "seed_suite": seed_suite,
        "attempt_limit": attempt_limit,
        "horizons": deterministic_horizons,
    }
    paired_consistency = _paired_distribution_consistency(config, horizons, seed_records)
    scientific_payload["paired_distribution_consistency"] = paired_consistency
    source = _source_provenance()
    raw = {
        "schema_version": "crazyflow.trajectory_construction_diagnostics.v1",
        "scope": "construction-only; no simulation, training, validation, test, or jax.jit",
        "config_path": str(config_path),
        "config": canonical_dict(config),
        "config_fingerprint": fingerprint(config),
        "seed_suite": seed_suite,
        "configured_max_attempts": config.trajectory.max_attempts,
        "diagnostic_max_attempts": attempt_limit,
        "higher_attempt_limit_estimate_only": attempt_limit != config.trajectory.max_attempts,
        "horizons": horizon_records,
        "paired_distribution_consistency": paired_consistency,
        "scientific_payload_sha256": _sha256_json(scientific_payload),
        "command": shlex.join([sys.executable, *sys.argv]),
        **source,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "construction_only_contract": {
            "simulation_built": False,
            "training_run": False,
            "validation_run": False,
            "test_manifest_opened": False,
            "test_split_built_or_evaluated": False,
            "jax_jit_called": False,
        },
        "internal_total_seconds": (time.perf_counter_ns() - process_start) / 1.0e9,
    }
    summaries = [_diagnostic_horizon_summary(record) for record in horizon_records]
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "trajectory_diagnostics_raw.json"
    atomic_write_json(raw_path, raw)
    summary = {
        "schema_version": "crazyflow.trajectory_construction_diagnostics_summary.v1",
        "config_fingerprint": fingerprint(config),
        "seed_suite_id": seed_suite["suite_id"],
        "seed_suite_sha256": seed_suite["sha256"],
        "scientific_payload_sha256": raw["scientific_payload_sha256"],
        "raw_file_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "configured_max_attempts": config.trajectory.max_attempts,
        "diagnostic_max_attempts": attempt_limit,
        "higher_attempt_limit_estimate_only": attempt_limit != config.trajectory.max_attempts,
        "horizons": summaries,
        "paired_distribution_consistency": paired_consistency,
        "total_construction_wall_seconds": sum(
            item["construction_wall_seconds"] for item in horizon_records
        ),
        "construction_only_contract": raw["construction_only_contract"],
        **source,
    }
    atomic_write_json(output_dir / "trajectory_diagnostics_summary.json", summary)
    print(f"diagnostic_seed_suite={seed_suite['suite_id']}")
    for item in summaries:
        print(
            f"horizon={item['horizon_control_intervals']} "
            f"success_rate={item['success_rate_within_attempt_limit']:.6f}"
        )
    print("construction_only=True")
    print(f"output_dir={output_dir}")


def main(args: argparse.Namespace | None = None) -> None:
    """Run one bounded CPU benchmark in a fresh process."""
    process_start = time.perf_counter_ns()
    args = parse_args(()) if args is None else args
    if args.horizon < 2 or args.steady_repeats < 1:
        raise ValueError("horizon must be at least two and steady repeats positive")
    config_path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    output_dir = args.output_dir or Path(
        f"artifacts/day6-readiness/scaling-h{args.horizon}-w{args.worlds}"
    )
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    try:
        config = load_config(config_path)
    except Exception as error:
        _write_benchmark_failure(
            args=args,
            output_dir=output_dir,
            config_path=config_path,
            failure_phase="config_validation_before_pipeline",
            error=error,
            process_start=process_start,
        )
        raise
    if args.trajectory_diagnostics_only:
        _run_trajectory_diagnostics(
            args=args,
            config_path=config_path,
            config=config,
            output_dir=output_dir,
            process_start=process_start,
        )
        return
    simulation = replace(config.train, n_worlds=args.worlds, horizon=args.horizon)
    effective_config = replace(config, train=simulation)
    try:
        pipeline = build_split_pipeline(Split.TRAIN, simulation, config.root_seed)
    except Exception as error:
        _write_benchmark_failure(
            args=args,
            output_dir=output_dir,
            config_path=config_path,
            failure_phase="simulation_pipeline_construction_before_jit",
            error=error,
            process_start=process_start,
            config=config,
            effective_config=effective_config,
        )
        raise
    try:
        realized_batch = build_episode_batch(
            pipeline, config.root_seed, 0, config.trajectory, config.domain_randomization
        )
    except Exception as error:
        phase = (
            "trajectory_construction_before_jit"
            if isinstance(error, ValueError) and "no valid trajectory found" in str(error)
            else "episode_batch_realization_before_jit"
        )
        _write_benchmark_failure(
            args=args,
            output_dir=output_dir,
            config_path=config_path,
            failure_phase=phase,
            error=error,
            process_start=process_start,
            config=config,
            effective_config=effective_config,
        )
        raise
    try:
        episode_ids = realized_batch.episode_ids
        batch = realized_batch.replace(episode_ids=())
        raw = raw_from_data(batch.initial_data, config.optimizer.gain_stage)
    except Exception as error:
        _write_benchmark_failure(
            args=args,
            output_dir=output_dir,
            config_path=config_path,
            failure_phase="objective_preparation_before_jit",
            error=error,
            process_start=process_start,
            config=config,
            effective_config=effective_config,
        )
        raise

    def value_and_gradient(candidate: jax.Array) -> tuple[tuple[jax.Array, dict], jax.Array]:
        def objective(values: jax.Array) -> tuple[jax.Array, dict]:
            return research_objective(
                values,
                batch,
                step_fn=pipeline.step_fn,
                steps_per_command=simulation.sim_freq_hz // simulation.control_freq_hz,
                gain_stage=config.optimizer.gain_stage,
            )

        return jax.value_and_grad(objective, has_aux=True)(candidate)

    try:
        jitted = jax.jit(value_and_gradient)
        compile_start = time.perf_counter_ns()
        compiled = jitted.lower(raw).compile()
        compile_seconds = (time.perf_counter_ns() - compile_start) / 1.0e9
    except Exception as error:
        _write_benchmark_failure(
            args=args,
            output_dir=output_dir,
            config_path=config_path,
            failure_phase="jit_lowering_or_compilation",
            error=error,
            process_start=process_start,
            config=config,
            effective_config=effective_config,
        )
        raise
    try:
        first_start = time.perf_counter_ns()
        first = compiled(raw)
        jax.block_until_ready(first)
        first_seconds = (time.perf_counter_ns() - first_start) / 1.0e9
    except Exception as error:
        _write_benchmark_failure(
            args=args,
            output_dir=output_dir,
            config_path=config_path,
            failure_phase="first_compiled_execution",
            error=error,
            process_start=process_start,
            config=config,
            effective_config=effective_config,
        )
        raise
    try:
        # One explicit warm-up keeps steady samples separate from first execution.
        jax.block_until_ready(compiled(raw))
        steady_seconds = []
        for _ in range(args.steady_repeats):
            steady_start = time.perf_counter_ns()
            result = compiled(raw)
            jax.block_until_ready(result)
            steady_seconds.append((time.perf_counter_ns() - steady_start) / 1.0e9)
    except Exception as error:
        _write_benchmark_failure(
            args=args,
            output_dir=output_dir,
            config_path=config_path,
            failure_phase="warmup_or_steady_execution",
            error=error,
            process_start=process_start,
            config=config,
            effective_config=effective_config,
        )
        raise

    usage = resource.getrusage(resource.RUSAGE_SELF)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    source_hash = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        source_hash.update(relative_path.encode())
        source_hash.update(
            subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{relative_path}"], cwd=REPOSITORY_ROOT, text=True
            )
            .strip()
            .encode()
        )
    source_hash.update(
        subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", *SOURCE_PATHS], cwd=REPOSITORY_ROOT
        )
    )
    source_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all", "--", *SOURCE_PATHS],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).splitlines()
    for relative_path in sorted(filter(None, untracked)):
        source_hash.update(relative_path.encode())
        source_hash.update((REPOSITORY_ROOT / relative_path).read_bytes())
    (loss, auxiliary), gradient = first
    mean_metrics = auxiliary["mean_metrics"]
    finite = all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree.leaves((loss, auxiliary, gradient))
        if np.issubdtype(np.asarray(leaf).dtype, np.number)
    )
    gate_checks = {
        "all_outputs_finite": bool(finite),
        "nonfinite_state_fraction_zero": float(mean_metrics["nonfinite_state_fraction"]) == 0.0,
        "zero_thrust_gate_fraction_zero": float(mean_metrics["zero_thrust_gate_fraction"]) == 0.0,
        "floor_clip_fraction_zero": float(mean_metrics["floor_clip_fraction"]) == 0.0,
        "motor_saturation_fraction_below_half": (
            float(mean_metrics["motor_saturation_fraction"]) < 0.5
        ),
        "all_episodes_successful": float(mean_metrics["episode_success"]) == 1.0,
    }
    packages = ("jax", "jaxlib", "numpy", "optax")
    internal_total_seconds = (time.perf_counter_ns() - process_start) / 1.0e9
    result = {
        "schema_version": "crazyflow.mellinger_scaling_benchmark.v3",
        "status": "success" if all(gate_checks.values()) else "technical_gate_failed",
        "failure_phase": None if all(gate_checks.values()) else "technical_gate_evaluation",
        "scope": (
            "technical CPU train value-and-gradient scaling pilot; not an optimizer, "
            "validation, scientific-performance, hardware, or realism claim"
        ),
        "worlds": args.worlds,
        "drones_per_world": simulation.n_drones,
        "horizon_control_intervals": simulation.horizon,
        "gain_stage": config.optimizer.gain_stage,
        "root_seed": config.root_seed,
        "episode_index": 0,
        "episode_ids": episode_ids,
        "trajectory_distribution_version": config.trajectory.distribution_version,
        "config_path": str(config_path),
        "effective_config_fingerprint": fingerprint(effective_config),
        "domain_randomization": {
            "mass_enabled": config.domain_randomization.mass_enabled,
            "mass_half_width_kg": config.domain_randomization.mass_half_width_kg,
            "delay_enabled": config.domain_randomization.delay_enabled,
            "wrench_enabled": config.domain_randomization.wrench_enabled,
            "sensor_noise_enabled": False,
            "ukf_enabled": False,
        },
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_seconds,
        "steady_execution_seconds": steady_seconds,
        "steady_median_seconds": float(np.median(steady_seconds)),
        "internal_total_seconds": internal_total_seconds,
        "process_max_rss_kib": usage.ru_maxrss,
        "process_max_rss_unit": "KiB on Linux",
        "loss": float(loss),
        "gradient_l2_norm": float(np.linalg.norm(np.asarray(gradient))),
        "mean_metrics": mean_metrics,
        "technical_gate": gate_checks | {"passed": all(gate_checks.values())},
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_head": head,
        "git_branch": branch,
        "source_dirty": bool(source_status),
        "source_status_short": source_status,
        "source_fingerprint": source_hash.hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_devices": [str(device) for device in jax.devices()],
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        },
        "test_split_built_or_evaluated": False,
        "test_manifest_opened": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "benchmark.json", result)
    print(f"worlds={args.worlds}")
    print(f"compile_seconds={compile_seconds:.6f}")
    print(f"steady_median_seconds={result['steady_median_seconds']:.6f}")
    print(f"process_max_rss_kib={usage.ru_maxrss}")
    print(f"technical_gate_passed={result['technical_gate']['passed']}")
    print(f"output_dir={output_dir}")
    if not result["technical_gate"]["passed"]:
        raise RuntimeError("scaling pilot failed its technical gate; do not scale further")


if __name__ == "__main__":
    main(parse_args())
