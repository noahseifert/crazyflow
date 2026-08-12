"""Generate the frozen, run-free Sprint-12 H100 replication protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from crazyflow.control.mellinger.research.config import fingerprint, load_config
from crazyflow.control.mellinger.research.gains import specs_for_stage
from crazyflow.control.mellinger.tracking import TrackingLossConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/day12-h100-replication-protocol"
BASE_CONFIG_PATH = ROOT / "artifacts/day10-sparse-long-pilot/pilot_config.json"
PILOT_ANALYSIS_PATH = ROOT / "artifacts/day11-long-pilot-evaluation/long_pilot_analysis.json"
ORIGIN_HEAD = "a269046fb0e70cb5ecc8b4c1783b87aecd866b7f"
BASE_PROTOCOL_HEAD = "73868cfd84d25bec0c6612b31ee34f4e67acb307"
AMENDMENT_ID = "2026-08-01-local-ssd-persistence-v1"
AMENDMENT_EFFECTIVE_UTC = "2026-08-01T12:51:18Z"
PILOT_SEED = 20260731
SEED_NAMESPACE = "crazyflow-gradient-research|h100-confirmatory-replication|v1|" + ORIGIN_HEAD
SEED_COUNT = 10
EXPECTED_SOURCE_FINGERPRINT = "6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e"
EXPECTED_RUNTIME_FINGERPRINT = "e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4"
EXPECTED_GAIN_REGISTRY_FINGERPRINT = (
    "2e8dbc409dc994f8376d132757ad8c3b0a8744805865c71340e405579c5f3d99"
)
EXPECTED_VALIDATION_FINGERPRINT = "11e53e460bc93a64835af03bd5a6d22a2e4fb3f7c9d187324d23e7a3deac0c20"
EXPECTED_VALIDATION_ID = "mellinger-validation-v1-20260731"
EXPECTED_RUN_BYTES = 437_505_563
CONSERVATIVE_RUN_BYTES = 875_333_570
EXPECTED_RUN_SECONDS = 925.96
CONSERVATIVE_RUN_SECONDS = 2289.545259811676
EXPECTED_PEAK_RSS_KIB = 2_201_956
HARD_PEAK_RSS_KIB = 12 * 1024**2
HARD_WALL_SECONDS = 7200
STUDENT_T_975_DF9 = 2.2621571627409915
SEED_01_MINIMUM_LOCAL_FREE_BYTES = 20_000_000_000
LOCAL_FREE_RESERVE_BYTES = 10_000_000_000
OBSERVED_LOCAL_FREE_BYTES_AT_AMENDMENT = 1_006_592_950_272


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _derive_seed(index: int, used: set[int]) -> dict[str, Any]:
    retry = 0
    while True:
        derivation_input = f"{SEED_NAMESPACE}|seed-index={index:02d}|retry={retry}"
        digest = hashlib.sha256(derivation_input.encode()).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed != 0 and seed not in used:
            used.add(seed)
            return {
                "index": index,
                "root_seed": seed,
                "derivation_input": derivation_input,
                "derivation_sha256": hashlib.sha256(derivation_input.encode()).hexdigest(),
                "retry_counter": retry,
            }
        retry += 1


def _config_payload(base: dict[str, Any], seed_record: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    result["run_id"] = f"sprint13-h100-replication-seed-{seed_record['index']:02d}"
    result["root_seed"] = seed_record["root_seed"]
    return result


def _minimum_local_free_bytes(seed_index: int) -> int:
    if seed_index == 1:
        return SEED_01_MINIMUM_LOCAL_FREE_BYTES
    remaining_runs_including_seed = SEED_COUNT - seed_index + 1
    return LOCAL_FREE_RESERVE_BYTES + EXPECTED_RUN_BYTES * remaining_runs_including_seed


def _local_persistence_policy() -> dict[str, Any]:
    return {
        "mode": "internal_ssd_repository_artifact_area",
        "storage_medium": "user-declared internal 2-TB SSD",
        "repository_run_root": "artifacts/day13-h100-replication/runs/",
        "ordinary_git_tracks_raw_runs": False,
        "retain_complete_and_unchanged_throughout_confirmatory_series": True,
        "forbidden_operations_during_series": ["delete", "move", "rename", "compress", "modify"],
        "after_every_attempt": (
            "Generate a complete SHA-256 index of every run file and immediately verify the "
            "entire index locally with sha256sum -c. Preserve successful and failed attempts."
        ),
        "before_each_later_seed": (
            "Re-run sha256sum -c for the complete index of the previous seed before launch."
        ),
        "integrity_failure_action": (
            "Any missing file or SHA-256 mismatch blocks this and every later seed."
        ),
        "seed_01_minimum_local_free_bytes": SEED_01_MINIMUM_LOCAL_FREE_BYTES,
        "later_seed_minimum_local_free_bytes_formula": (
            "10000000000 + 437505563 * remaining_seed_count_including_seed_to_start"
        ),
        "local_free_reserve_bytes": LOCAL_FREE_RESERVE_BYTES,
        "expected_raw_bytes_per_remaining_seed": EXPECTED_RUN_BYTES,
        "external_storage_copy_required": False,
        "external_storage_capacity_check_required": False,
        "external_destination_hash_verification_required": False,
        "accepted_single_drive_risk": True,
        "limitation": (
            "SHA-256 detects unnoticed change but is not an independent backup; total loss or "
            "failure of the single internal SSD can destroy all raw replication data."
        ),
        "sprint10_pilot_must_remain_unchanged": True,
    }


def main() -> None:
    """Write deterministic protocol JSON and ten unexecuted configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Absent or empty destination for generated JSON (default: protocol directory)",
    )
    output = parser.parse_args().output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory {output}")

    base = json.loads(BASE_CONFIG_PATH.read_text())
    base_config = load_config(BASE_CONFIG_PATH)
    pilot_analysis = json.loads(PILOT_ANALYSIS_PATH.read_text())
    if fingerprint(base_config) != pilot_analysis["fingerprints"]["config_fingerprint"]:
        raise ValueError("Sprint-10 base config fingerprint mismatch")
    if pilot_analysis["fingerprints"]["source_fingerprint"] != EXPECTED_SOURCE_FINGERPRINT:
        raise ValueError("Sprint-10 source fingerprint mismatch")
    if pilot_analysis["fingerprints"]["runtime_fingerprint"] != EXPECTED_RUNTIME_FINGERPRINT:
        raise ValueError("Sprint-10 runtime fingerprint mismatch")

    used = {PILOT_SEED}
    seeds = [_derive_seed(index, used) for index in range(1, SEED_COUNT + 1)]
    seed_manifest = {
        "schema_version": "crazyflow.h100_replication_seed_manifest.v1",
        "status": "FROZEN_NOT_EXECUTED",
        "protocol_origin_head": ORIGIN_HEAD,
        "pilot_seed_excluded": PILOT_SEED,
        "seed_count": SEED_COUNT,
        "generation": {
            "algorithm": "SHA-256",
            "namespace": SEED_NAMESPACE,
            "encoding": "UTF-8",
            "derivation": (
                "SHA256(namespace + '|seed-index=NN|retry=K'); take the first four digest "
                "bytes as an unsigned big-endian integer and clear the high bit with "
                "0x7fffffff; increment K only for zero, collision, or the excluded pilot seed"
            ),
            "accepted_range": [1, 2**31 - 1],
            "independence_scope": (
                "Distinct predeclared root seeds create disjoint deterministic Train streams "
                "under the runner's split/episode/world/component fold-in contract. The fixed "
                "Validation manifest intentionally remains common across seeds."
            ),
        },
        "seeds": seeds,
    }
    _write_json(output / "seed_manifest.json", seed_manifest)

    specs = specs_for_stage(1)
    gains = [
        {
            "name": spec.name,
            "unit": spec.unit,
            "lower": spec.lower,
            "upper": spec.upper,
            "axes": list(spec.axes),
            "controller": spec.controller,
            "parameter": spec.parameter,
        }
        for spec in specs
    ]
    loss = asdict(TrackingLossConfig())
    requirements = {
        "schema_version": "crazyflow.h100_replication_requirements.v1",
        "status": "FROZEN_NOT_EXECUTED",
        "repository_branch": "research/differentiable-mellinger",
        "protocol_origin_head": ORIGIN_HEAD,
        "scientific_source_scope": ["crazyflow", "examples"],
        "required_source_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "required_runtime_fingerprint": EXPECTED_RUNTIME_FINGERPRINT,
        "required_gain_registry_fingerprint": EXPECTED_GAIN_REGISTRY_FINGERPRINT,
        "required_validation_manifest_id": EXPECTED_VALIDATION_ID,
        "required_validation_manifest_fingerprint": EXPECTED_VALIDATION_FINGERPRINT,
        "runtime": pilot_analysis["fingerprints"]["runtime_compatibility"],
        "environment_reference": pilot_analysis["fingerprints"]["environment"],
        "backend": "cpu",
        "jax_enable_x64": False,
        "config_base_path": str(BASE_CONFIG_PATH.relative_to(ROOT)),
        "config_base_sha256": _sha256(BASE_CONFIG_PATH),
        "config_base_fingerprint": fingerprint(base_config),
        "only_per_seed_config_differences": ["run_id", "root_seed"],
        "active_persistence_amendment": AMENDMENT_ID,
        "persistence_requirements": _local_persistence_policy(),
        "test_manifest_policy": {
            "reference": base["test_manifest"],
            "opaque_reference_only": True,
            "must_not_open_parse_validate_build_simulate_or_evaluate": True,
        },
        "validation_manifest": {
            "path": base["validation_manifest"],
            "manifest_id": EXPECTED_VALIDATION_ID,
            "fingerprint": EXPECTED_VALIDATION_FINGERPRINT,
            "fixed_worlds_per_seed": 4,
        },
        "scientific_semantics": {
            "trajectory": base["trajectory"],
            "train": base["train"],
            "validation": base["validation"],
            "domain_randomization": base["domain_randomization"],
            "optimizer": base["optimizer"],
            "optimizer_implementation": "optax.adam",
            "optimizer_version": pilot_analysis["fingerprints"]["environment"]["packages"]["optax"],
            "gains": gains,
            "shared_gain_vector_across_worlds": True,
            "loss_config": loss,
            "loss_aggregation": "equal arithmetic mean over the four cases in each split",
            "controller_mass_assumption_kg": 0.029,
            "nominal_dynamics_mass_kg": 0.0319,
            "true_dynamics_mass_variation_kg": [-0.0002, 0.0002],
            "controller_mass_is_not_randomized": True,
            "disabled": {
                "action_delay": True,
                "wrench_randomization": True,
                "sensor_noise": True,
                "ukf": True,
                "test_evaluation": True,
                "h200": True,
                "h400": True,
                "automatic_resume": True,
                "automatic_extension": True,
                "parallel_seed_execution": True,
                "non_stage1_gains": True,
            },
        },
    }
    _write_json(output / "requirements.json", requirements)

    configs = []
    inventory_runs = []
    for seed_record in seeds:
        index = seed_record["index"]
        relative_path = Path(
            f"artifacts/day12-h100-replication-protocol/configs/seed-{index:02d}.json"
        )
        config_path = output / "configs" / f"seed-{index:02d}.json"
        config_payload = _config_payload(base, seed_record)
        _write_json(config_path, config_payload)
        loaded = load_config(config_path)
        configs.append(
            {
                "index": index,
                "root_seed": seed_record["root_seed"],
                "run_id": config_payload["run_id"],
                "path": str(relative_path),
                "sha256": _sha256(config_path),
                "config_fingerprint": fingerprint(loaded),
            }
        )
        inventory_runs.append(
            {
                "execution_order": index,
                "seed_index": index,
                "root_seed": seed_record["root_seed"],
                "run_id": config_payload["run_id"],
                "config_path": str(relative_path),
                "config_sha256": _sha256(config_path),
                "config_fingerprint": fingerprint(loaded),
                "planned_run_root": (
                    f"artifacts/day13-h100-replication/runs/seed-{index:02d}/<UTC>-<GIT_HEAD>"
                ),
                "status": "PLANNED_NOT_STARTED",
                "allowed_successful_runs": 1,
                "allowed_replacement_seeds": 0,
                "maximum_exogenous_failure_retries": 1,
                "minimum_local_free_bytes_before_start": _minimum_local_free_bytes(index),
                "remaining_seed_count_including_this_run": SEED_COUNT - index + 1,
            }
        )

    run_inventory = {
        "schema_version": "crazyflow.h100_replication_run_inventory.v1",
        "status": "PLANNED_NOT_STARTED",
        "execution": {
            "mode": "strictly sequential in ascending seed index",
            "maximum_concurrent_research_processes": 1,
            "start_next_only_after_previous_local_sha256_reverification_pass": True,
            "automatic_seed_loop": False,
            "automatic_resume": False,
            "fresh_process_and_absent_run_root_required": True,
        },
        "persistence": _local_persistence_policy(),
        "resource_plan": {
            "per_run": {
                "expected_wall_seconds": EXPECTED_RUN_SECONDS,
                "conservative_wall_seconds": CONSERVATIVE_RUN_SECONDS,
                "hard_timeout_seconds": HARD_WALL_SECONDS,
                "expected_peak_rss_kib": EXPECTED_PEAK_RSS_KIB,
                "hard_peak_rss_kib": HARD_PEAK_RSS_KIB,
                "expected_raw_bytes": EXPECTED_RUN_BYTES,
                "factor_two_raw_bytes": CONSERVATIVE_RUN_BYTES,
            },
            "ten_run_panel": {
                "expected_compute_wall_seconds": EXPECTED_RUN_SECONDS * SEED_COUNT,
                "conservative_compute_wall_seconds": CONSERVATIVE_RUN_SECONDS * SEED_COUNT,
                "absolute_timeout_ceiling_seconds": HARD_WALL_SECONDS * SEED_COUNT,
                "expected_raw_bytes": EXPECTED_RUN_BYTES * SEED_COUNT,
                "factor_two_raw_bytes": CONSERVATIVE_RUN_BYTES * SEED_COUNT,
                "expected_raw_decimal_gb": EXPECTED_RUN_BYTES * SEED_COUNT / 1e9,
                "expected_raw_gib": EXPECTED_RUN_BYTES * SEED_COUNT / 1024**3,
                "seed_01_minimum_local_free_bytes": SEED_01_MINIMUM_LOCAL_FREE_BYTES,
                "later_seed_local_free_reserve_bytes": LOCAL_FREE_RESERVE_BYTES,
                "observed_local_free_bytes_at_amendment_preflight": (
                    OBSERVED_LOCAL_FREE_BYTES_AT_AMENDMENT
                ),
                "peak_rss_is_not_multiplied_because_execution_is_sequential": True,
            },
        },
        "configs": configs,
        "runs": inventory_runs,
    }
    _write_json(output / "run_inventory.json", run_inventory)

    analysis_plan = {
        "schema_version": "crazyflow.h100_replication_analysis_plan.v1",
        "status": "FROZEN_BEFORE_ANY_CONFIRMATORY_RUN",
        "analysis_population": {
            "planned_seed_count": 10,
            "pilot_seed_excluded": PILOT_SEED,
            "valid_panel_requires_all_predeclared_seeds": True,
            "replacement_seeds": 0,
            "imputation": "none",
            "exclude_after_viewing_scientific_result": False,
        },
        "selection_rule": {
            "candidate_steps": "every completed post-update step 1 through 5000",
            "quantity": "fixed Validation loss",
            "operation": "global minimum within each seed",
            "implementation": "update only when validation_loss < best_validation",
            "tie_breaking": (
                "an exact equality does not update, therefore the earliest exact tie wins"
            ),
            "step_zero_is_not_a_candidate": True,
            "test_used": False,
            "checkpoint_note": (
                "Checkpoint files occur every 100 updates, but each stores selected_step and "
                "selected_raw_gains even when that selected step lies between checkpoint files."
            ),
        },
        "primary_endpoint": {
            "name": "selected_validation_loss",
            "per_seed_source": "run_summary.selected.selected_validation_loss",
            "direction": "lower is better",
            "across_seed_point_estimate": "arithmetic mean over the ten manifest-ordered seeds",
            "uncertainty": {
                "sample_standard_deviation": "ddof=1",
                "confidence_interval": "two-sided 95% Student-t interval for the mean",
                "degrees_of_freedom": 9,
                "critical_value_t_0.975_df_9": STUDENT_T_975_DF9,
                "formula": "mean +/- critical_value * sample_sd / sqrt(10)",
            },
            "robust_descriptive_companion": {
                "median": True,
                "q1_q3": "numpy.quantile probabilities 0.25 and 0.75 with method='linear'",
                "all_ten_individual_values_in_manifest_order": True,
            },
        },
        "derived_primary_support": {
            "relative_validation_improvement_per_seed": (
                "1 - selected_validation_loss / validation_loss_at_step_1"
            ),
            "aggregation": "same mean, ddof=1 SD, 95% t interval, median, Q1/Q3, all values",
        },
        "secondary_endpoints": [
            "selected_step",
            "validation_loss_at_step_5000",
            "train_loss_at_selected_step",
            "train_loss_at_step_5000",
            "gradient_l2_norm_at_selected_step",
            "gradient_l2_norm_at_step_5000",
            "Validation-minus-Train loss gap at selected step and step 5000",
            "selected and final physical kp_xy, kp_z, kd_xy, kd_z",
            "selected and final normalized distance of each gain to its nearest bound",
            "runner-emitted loss components at selected step and step 5000",
            "runner-emitted tracking and technical metrics at selected step and step 5000",
            "per-run maxima of motor_saturation_fraction, floor_clip_fraction, "
            "zero_thrust_gate_fraction, nonfinite_state_fraction, and max_position_error_m",
            "external walltime, Peak-RSS, swap, artifact bytes, and checkpoint-write time",
        ],
        "runner_emitted_metric_names": [
            "loss",
            "gradient_l2_norm (Train only)",
            "loss_total",
            "loss_position",
            "loss_velocity",
            "loss_effort",
            "loss_smoothness",
            "loss_terminal",
            "loss_altitude",
            "position_rmse_m",
            "velocity_rmse_m_s",
            "control_effort",
            "control_smoothness",
            "max_position_error_m",
            "p95_position_error_m",
            "terminal_position_error_m",
            "terminal_velocity_error_m_s",
            "motor_saturation_fraction",
            "floor_clip_fraction",
            "zero_thrust_gate_fraction",
            "nonfinite_state_fraction",
            "episode_success",
            "episode_failure",
        ],
        "secondary_inference_policy": (
            "Descriptive only: all individual values, mean, ddof=1 SD, 95% t interval, median, "
            "and linear-method Q1/Q3 where numeric. No secondary p-values and no "
            "multiplicity claim."
        ),
        "technical_failure_policy": {
            "preflight_block": (
                "Branch, source/runtime/config/manifest fingerprint, capacity, sequentiality, or "
                "local-storage capacity/integrity failure before launch blocks launch and does "
                "not count as an attempt."
            ),
            "exogenous_retry": (
                "At most one full restart from update 0 with the same seed and byte-identical "
                "config is allowed only for a documented exogenous interruption or storage/host "
                "failure that is not caused by the runner's scientific or technical gates."
            ),
            "no_resume": True,
            "no_retry_for": [
                "nonfinite value",
                "failed Train or Validation technical gate",
                "trajectory construction failure",
                "7200-second timeout without documented exogenous host interruption",
                "Peak-RSS at or above 12 GiB",
                "any swap",
                "config/source/runtime/gain/Validation fingerprint mismatch after launch",
                "Test access or Test metric",
            ],
            "failed_attempt_preservation": (
                "Retain and checksum every failed attempt; classify it before any retry and before "
                "panel aggregation. Never delete, move, rename, compress, modify, or overwrite it "
                "during the confirmatory series."
            ),
            "successful_retry_inclusion": (
                "If the sole allowed exogenous retry passes, include that successful run for the "
                "same predeclared seed and retain the failed attempt as audit evidence."
            ),
            "second_failure_or_nonretryable_failure": (
                "Mark the seed invalid, perform no replacement, stop remaining launches, and set "
                "the panel decision to NO-GO."
            ),
        },
        "go_no_go_horizon_transition": {
            "decision_scope": "permission to prepare a later unchanged H200/H400 protocol only",
            "go_requires_all": [
                "10 of 10 predeclared seeds have one valid included run and no replacement seed",
                (
                    "all completion, checksum, finite, technical, fingerprint, resource, and "
                    "Test gates pass"
                ),
                "all ten relative_validation_improvement values are greater than zero",
                "at least 8 of 10 relative_validation_improvement values are at least 0.50",
                (
                    "lower bound of the frozen 95% t interval for mean relative improvement is "
                    "at least 0.50"
                ),
                (
                    "95% t-interval half-width for mean selected_validation_loss divided by its "
                    "mean is at most 0.25"
                ),
                (
                    "for every seed, validation_loss_at_step_5000 is at most 1.10 times "
                    "selected_validation_loss"
                ),
                "every selected gain's normalized nearest-bound distance is at least 0.01",
                (
                    "all raw runs remain unchanged in the repository artifact area and every "
                    "local SHA-256 index and required pre-seed reverification passes"
                ),
                (
                    "no unplanned scientific, statistical, runtime, or execution-protocol change "
                    "occurred"
                ),
            ],
            "no_go": (
                "Any failed GO condition is NO-GO for H200/H400. Preserve evidence and diagnose; "
                "do not tune, replace seeds, add runs, or open Test under this protocol."
            ),
        },
        "test_policy": {
            "locked": True,
            "open_count_during_replication": 0,
            "final_use": (
                "exactly once only after H100 replication, later horizons, and all decisions freeze"
            ),
        },
        "active_persistence_amendment": AMENDMENT_ID,
        "persistence_scope_only": True,
    }
    _write_json(output / "analysis_plan.json", analysis_plan)

    protocol = {
        "schema_version": "crazyflow.h100_replication_protocol.v1",
        "status": "FROZEN_PROTOCOL_ONLY_NO_RUN_AUTHORIZATION",
        "title": "Ten-seed H100 replication protocol",
        "protocol_origin_head": ORIGIN_HEAD,
        "active_amendment": {
            "id": AMENDMENT_ID,
            "effective_at_utc": AMENDMENT_EFFECTIVE_UTC,
            "base_protocol_commit": BASE_PROTOCOL_HEAD,
            "scope": "persistence and single-drive fault tolerance only",
        },
        "exploratory_pilot": {
            "run": ("artifacts/day10-sparse-long-pilot/runs/20260801T081232Z-1345032536fd"),
            "root_seed": PILOT_SEED,
            "role": "planning evidence only",
            "included_in_confirmatory_seed_count_or_aggregation": False,
        },
        "references": {
            "seed_manifest": "seed_manifest.json",
            "requirements": "requirements.json",
            "run_inventory": "run_inventory.json",
            "analysis_plan": "analysis_plan.json",
            "runbook_template": "sprint13_runbook_template.sh",
            "persistence_amendment": (f"amendments/{AMENDMENT_ID}.json"),
            "persistence_amendment_human_readable": (f"amendments/{AMENDMENT_ID}.md"),
        },
        "frozen_design": {
            "new_root_seeds": [item["root_seed"] for item in seeds],
            "seed_count": SEED_COUNT,
            "horizon": 100,
            "train_worlds": 4,
            "fixed_validation_worlds": 4,
            "updates_per_seed": 5000,
            "checkpoint_interval": 100,
            "gain_stage": 1,
            "gain_names": [item["name"] for item in gains],
            "execution": "strictly sequential, one fresh CPU process at a time",
            "test_locked": True,
        },
        "release": {
            "sprint12_research_runs_authorized": 0,
            "sprint12a_research_runs_authorized": 0,
            "sprint13_automatic_seed_loop_authorized": False,
            "later_execution_requires_new_preflight_and_explicit_authorization": True,
        },
    }
    _write_json(output / "protocol.json", protocol)

    release_gate = {
        "schema_version": "crazyflow.h100_replication_protocol_gate.v1",
        "decision": "NO-GO_RUNS_IN_SPRINT12A",
        "protocol_complete": True,
        "active_persistence_amendment": AMENDMENT_ID,
        "external_storage_gate_required": False,
        "local_persistence_gate_required": True,
        "seed_01_preparation_status": "GO_AFTER_SEPARATE_AUTHORIZATION_AND_EXACT_PREFLIGHT",
        "next_step": (
            "Sprint 13 may verify and execute exactly seed 01 only under separate explicit "
            "authorization and the amended local-persistence preflight"
        ),
        "research_processes_started": 0,
        "test_manifest_opened": False,
        "h200_h400_started": False,
        "pilot_in_confirmatory_panel": False,
    }
    _write_json(output / "release_gate.json", release_gate)

    amendment = {
        "schema_version": "crazyflow.h100_replication_protocol_amendment.v1",
        "amendment_id": AMENDMENT_ID,
        "status": "ADOPTED_PRE_SEED_01",
        "effective_at_utc": AMENDMENT_EFFECTIVE_UTC,
        "base_protocol_commit": BASE_PROTOCOL_HEAD,
        "protocol_origin_head": ORIGIN_HEAD,
        "reason": (
            "The user elected to retain all raw replication data on the internal 2-TB SSD and "
            "not to use external storage."
        ),
        "timing_and_bias_control": {
            "confirmatory_seeds_started_before_adoption": 0,
            "confirmatory_results_known_at_adoption": False,
            "adopted_before_seed_01": True,
            "result_dependent_change": False,
        },
        "scope": {
            "changed": ["storage location", "backup/fault-tolerance execution gates"],
            "unchanged": [
                "ten seeds and their order",
                "all ten seed configs and hashes",
                "scientific source and runtime requirements",
                "H100 scientific and optimizer semantics",
                "selection rule and tie-breaking",
                "primary and secondary endpoints and uncertainty",
                "technical failure and retry classification except storage preflight wording",
                "GO/NO-GO scientific thresholds",
                "strictly sequential execution",
                "locked Test split",
            ],
        },
        "superseded_rule": {
            "summary": (
                "After every attempt, copy the raw run and SHA-256 index byte-for-byte to "
                "approved backed-up external research storage and verify all hashes at the "
                "destination before the next seed."
            ),
            "external_capacity_copy_and_destination_verification_were_gates": True,
            "historical_authority": BASE_PROTOCOL_HEAD,
        },
        "replacement_rule": _local_persistence_policy(),
        "observed_local_capacity": {
            "probe_command": "df -B1 --output=avail,target .",
            "available_bytes_at_adoption_preflight": OBSERVED_LOCAL_FREE_BYTES_AT_AMENDMENT,
            "seed_01_threshold_passed": (
                OBSERVED_LOCAL_FREE_BYTES_AT_AMENDMENT >= SEED_01_MINIMUM_LOCAL_FREE_BYTES
            ),
        },
        "unchanged_seed_config_sha256": {
            f"seed-{item['index']:02d}": item["sha256"] for item in configs
        },
        "unchanged_fingerprints": {
            "source": EXPECTED_SOURCE_FINGERPRINT,
            "runtime": EXPECTED_RUNTIME_FINGERPRINT,
            "gain_registry": EXPECTED_GAIN_REGISTRY_FINGERPRINT,
            "validation_manifest": EXPECTED_VALIDATION_FINGERPRINT,
        },
        "accepted_limitation": (
            "The user knowingly accepts complete loss or failure of the single internal SSD. "
            "SHA-256 detects unnoticed modification but does not provide an independent backup."
        ),
        "sprint10_pilot_run_must_remain_unchanged": (
            "artifacts/day10-sparse-long-pilot/runs/20260801T081232Z-1345032536fd"
        ),
        "research_processes_started_by_amendment": 0,
        "test_manifest_opened_by_amendment": False,
    }
    _write_json(output / "amendments" / f"{AMENDMENT_ID}.json", amendment)


if __name__ == "__main__":
    main()
