"""Unit contracts for the bounded G4 kp_xy + kp_z implementation."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control.mellinger.research import g3_freeze_v2 as g3
from crazyflow.control.mellinger.research import g4_optimization as g4
from crazyflow.control.mellinger.research import robust_evaluation as robust

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / g4.CONFIG_RELATIVE_PATH
LONGRUN_RESOURCE_SCHEMA = "crazyflow.mellinger_g4_longrun_resource_evidence.v1"


def _longrun_operational_update(
    *, token: int = 0, parent_ids: list[str] | None = None
) -> dict[str, Any]:
    total = 16 * 1024**3
    allowed = int(g4.RSS_FRACTION_LIMIT * total)
    bound_parents = parent_ids or [f"p-{index}" for index in range(32)]
    microbatches = []
    observations = []
    for index in range(8):
        before = {
            "mem_total_bytes": total,
            "mem_available_bytes": g4.MINIMUM_AVAILABLE_BYTES + 4096 - 2 * index,
            "swap_used_bytes": 0,
            "maximum_rss_bytes": 1024 + token + 2 * index,
            "maximum_allowed_rss_bytes": allowed,
        }
        after = {
            "mem_total_bytes": total,
            "mem_available_bytes": g4.MINIMUM_AVAILABLE_BYTES + 4095 - 2 * index,
            "swap_used_bytes": 0,
            "maximum_rss_bytes": 1025 + token + 2 * index,
            "maximum_allowed_rss_bytes": allowed,
        }
        observations.extend((before, after))
        microbatches.append(
            {
                "microbatch_index": index,
                "elapsed_seconds": 0.1 + index / 1000,
                "parent_ids": bound_parents[4 * index : 4 * index + 4],
                "resource_before": before,
                "resource_after": after,
            }
        )
    target = g4.EXPECTED_INTERPRETER.resolve(strict=True)
    return {
        "schema_version": LONGRUN_RESOURCE_SCHEMA,
        "started_unix_ns": 1000 + token,
        "finished_unix_ns": 2000 + token,
        "process_elapsed_seconds": 1.0 + token / 1000,
        "resource_observation_count": 16,
        "maximum_rss_bytes": max(item["maximum_rss_bytes"] for item in observations),
        "minimum_mem_available_bytes": min(item["mem_available_bytes"] for item in observations),
        "maximum_swap_used_bytes": 0,
        "microbatches": microbatches,
        "host": {
            "hostname": os.uname().nodename,
            "interpreter": str(g4.EXPECTED_INTERPRETER),
            "interpreter_target": str(target),
            "interpreter_sha256": g4.sha256_file(target),
        },
    }


def test_exact_public_day26_hash_json_schema_key_and_mode_contracts() -> None:
    evidence = g4.validate_public_inputs(REPOSITORY_ROOT)

    assert evidence["status"] == "PASS_SIX_PUBLIC_DAY26_INPUTS"
    assert set(evidence["files"]) == set(g4.DAY26_INPUTS)
    for path, record in evidence["files"].items():
        assert record["sha256"] == g4.DAY26_INPUTS[path]["sha256"]
        assert record["file_mode"] == "0644"
        assert record["git_mode"] == "100644"
        assert record["schema_version"] == g4.DAY26_SCHEMA_VERSION
        assert set(record["top_level_keys"]) >= g4.DAY26_INPUTS[path]["keys"]


def test_public_payload_contract_rejects_schema_keys_optimizer_and_selection_drift() -> None:
    proposal_path = f"{g4.DAY26_DIRECTORY}/g4_g5_freeze_proposal.json"
    proposal = json.loads((REPOSITORY_ROOT / proposal_path).read_bytes())

    missing = dict(proposal)
    missing.pop("training_contract")
    with pytest.raises(g4.G4ContractError, match="missing keys"):
        g4._require_public_payload_contract(proposal_path, missing)

    wrong_schema = dict(proposal, schema_version="wrong")
    with pytest.raises(g4.G4ContractError, match="schema changed"):
        g4._require_public_payload_contract(proposal_path, wrong_schema)

    wrong_updates = dict(proposal, optimizer_update_count=1)
    with pytest.raises(g4.G4ContractError, match="optimizer activity"):
        g4._require_public_payload_contract(proposal_path, wrong_updates)

    wrong_selection = dict(proposal, selected_parameters=["kp_xy", "ki_z"])
    with pytest.raises(g4.G4ContractError, match="proposal contract"):
        g4._require_public_payload_contract(proposal_path, wrong_selection)


def test_config_pins_only_two_parameters_loss_seed_optimizer_and_protection() -> None:
    payload, digest = g4.load_and_validate_config(CONFIG_PATH, REPOSITORY_ROOT)

    assert len(digest) == 64
    assert tuple(payload["parameters"]) == g4.PARAMETER_NAMES
    assert payload["loss_v1"]["loss_v2_weight"] == 0.0
    assert tuple(payload["loss_v1"]["terms"]) == (
        "altitude",
        "effort",
        "position",
        "smoothness",
        "terminal",
        "velocity",
    )
    assert payload["parents"]["development_seed"] == 9_104_001
    assert payload["optimizer"]["microbatch_size"] == 4
    assert payload["optimizer"]["microbatch_count"] == 8
    assert payload["optimizer"]["effective_batch_size"] == 32
    assert payload["optimizer"]["maximum_updates"] == 2_000
    assert payload["protection"] == g4.protected_contract()


def test_parameter_contract_has_exact_defaults_bounds_and_deferred_inventory() -> None:
    contract = g4.parameter_contract()

    assert tuple(contract) == ("kp_xy", "kp_z")
    assert contract["kp_xy"]["contract_default"] == 0.4
    assert contract["kp_xy"]["runtime_default"] == 0.4000000059604645
    assert contract["kp_xy"]["physical_bounds"] == [0.1, 1.2]
    assert contract["kp_xy"]["raw_bounds"] == [-1.3862943611198906, 1.0986122886681098]
    assert contract["kp_z"]["contract_default"] == 1.25
    assert contract["kp_z"]["physical_bounds"] == [0.3, 2.5]
    assert contract["kp_z"]["raw_bounds"] == [-1.4271163556401458, 0.6931471805599453]
    for excluded in ("ki_z", "kd_xy", "kd_z", "mass", "mass_thrust"):
        assert excluded not in contract


def test_loss_v1_has_exact_six_terms_and_inactive_v2() -> None:
    contract = g4.loss_v1_contract()

    assert contract == {
        "terms": {
            "position": {"weight": 1.0, "scale": 0.25**2},
            "velocity": {"weight": 0.10, "scale": 1.0**2},
            "effort": {"weight": 1.0e-3, "scale": 1.0},
            "smoothness": {"weight": 1.0e-3, "scale": 1.0},
            "terminal": {"weight": 0.10, "scale": 0.25**2},
            "altitude": {"weight": 0.05, "scale": 1.0, "margin_m": 0.15, "softness_m": 0.05},
        },
        "loss_v2_weight": 0.0,
        "rollout_seconds": 6.0,
        "score_seconds": 2.0,
    }


def test_exact_288_parent_plan_is_balanced_unique_and_train_only() -> None:
    specs = g4.train_parent_specs()

    assert len(specs) == 288
    assert len({item.episode_id for item in specs}) == 288
    assert len({(item.batch_index, item.stratum_index, item.slot_index) for item in specs}) == 288
    assert [sum(item.batch_index == index for item in specs) for index in range(9)] == [32] * 9
    assert [sum(item.stratum_index == index for item in specs) for index in range(9)] == [32] * 9
    assert all(item.score_stop - item.score_start == 200 for item in specs)
    assert not any("validation" in item.episode_id or "test" in item.episode_id for item in specs)


def test_parent_key_hierarchy_is_stable_component_attempt_and_coordinate_disjoint() -> None:
    first, second = g4.train_parent_specs()[:2]
    key = np.asarray(jax.random.key_data(g4.parent_component_key(first, "position", 0)))

    assert np.array_equal(
        key, np.asarray(jax.random.key_data(g4.parent_component_key(first, "position", 0)))
    )
    variants = {
        np.asarray(jax.random.key_data(g4.parent_component_key(first, "yaw", 0))).tobytes(),
        np.asarray(jax.random.key_data(g4.parent_component_key(first, "position", 1))).tobytes(),
        np.asarray(jax.random.key_data(g4.parent_component_key(second, "position", 0))).tobytes(),
    }
    assert key.tobytes() not in variants
    assert len(variants) == 3
    with pytest.raises(ValueError):
        g4.parent_component_key(first, "closed-test", 0)


def test_one_controller_blind_parent_is_finite_accepted_and_digest_complete() -> None:
    spec = g4.train_parent_specs()[0]
    candidate = g4.construct_train_parent(spec)

    assert 0 <= candidate.attempt < 64
    assert candidate.attempts[-1]["accepted"] is True
    assert len(candidate.parent_digest) == 64
    assert set(candidate.array_digests) == {
        "time",
        "position",
        "velocity",
        "acceleration",
        "jerk",
        "yaw",
        "yaw_rate",
    }
    assert all(
        np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(candidate.trajectory)
    )


def test_fixed_validation_and_near_limit_pins_match_public_manifests() -> None:
    validation = g4.fixed_day26_items(REPOSITORY_ROOT, g4.VALIDATION_IDENTITIES)
    near_limit = g4.fixed_day26_items(REPOSITORY_ROOT, g4.NEAR_LIMIT_IDENTITIES)

    assert len(validation) == 9
    assert len(near_limit) == 6
    assert [
        f"{spec.episode_id}/attempt-{candidate.attempt}" for spec, candidate in validation
    ] == list(g4.VALIDATION_IDENTITIES)
    assert [
        f"{spec.episode_id}/attempt-{candidate.attempt}" for spec, candidate in near_limit
    ] == list(g4.NEAR_LIMIT_IDENTITIES)
    assert {candidate.parent_digest for _, candidate in validation}.isdisjoint(
        {candidate.parent_digest for _, candidate in near_limit}
    )


def _one_world_data() -> Any:
    item = g4.fixed_day26_items(REPOSITORY_ROOT, ("train-soft-hold-8301001/attempt-0",))
    inputs = g3.stack_evaluation_inputs(item)
    sim = robust.build_simulation(1, rng_seed=g4.DEVELOPMENT_SEED)
    return robust.initialize_inputs(sim.data, inputs)


def test_theta_zero_is_exact_and_each_parameter_changes_only_declared_kp_axes() -> None:
    data = _one_world_data()
    zero = g4.apply_theta(data, jnp.zeros((2,), dtype=jnp.float32))
    baseline_state = data.controls.state
    zero_state = zero.controls.state
    assert baseline_state is not None and zero_state is not None
    np.testing.assert_array_equal(
        np.asarray(baseline_state.params["kp"]), np.asarray(zero_state.params["kp"])
    )

    xy = g4.parameter_isolation_record(data, "kp_xy")
    z = g4.parameter_isolation_record(data, "kp_z")
    assert xy["changed_indices"] == [[0], [1]]
    assert z["changed_indices"] == [[2]]
    assert (
        g4.direct_parameter_controller_check(data)["status"] == "PASS_DIRECT_PARAMETER_CONTROLLER"
    )
    with pytest.raises(g4.G4ContractError, match="unauthorized"):
        g4.apply_theta(data, jnp.zeros((1,), dtype=jnp.float32), ("ki_z",))


def test_physical_log_coordinate_is_float32_and_exact_at_zero() -> None:
    physical = g4.physical_from_theta(np.zeros((2,), dtype=np.float32))

    assert physical.dtype == np.float32
    np.testing.assert_array_equal(
        physical, np.asarray([0.4000000059604645, 1.25], dtype=np.float32)
    )
    with pytest.raises(g4.G4ContractError, match="shape"):
        g4.physical_from_theta(np.zeros((3,), dtype=np.float32))


def test_microbatch_aggregation_uses_count_weights_and_never_updates() -> None:
    evaluations = (
        {"parent_count": 1, "loss": 2.0, "gradient": np.asarray([2.0, 4.0]), "parent_ids": ["a"]},
        {
            "parent_count": 3,
            "loss": 6.0,
            "gradient": np.asarray([6.0, 8.0]),
            "parent_ids": ["b", "c", "d"],
        },
    )
    record = g4._aggregate_evaluations(evaluations)

    assert record["loss"] == pytest.approx(5.0)
    np.testing.assert_allclose(record["gradient"], np.asarray([5.0, 7.0]))
    assert record["microbatch_weights"] == [0.25, 0.75]
    assert record["optimizer_update_count"] == 0
    with pytest.raises(g4.G4ContractError, match="duplicated"):
        g4._aggregate_evaluations((evaluations[0], dict(evaluations[0])))


def test_projected_adam_has_exact_hyperparameters_direction_counter_and_bounds() -> None:
    state = g4.initial_optimization_state(2)
    gradient = np.asarray([1.0, -2.0], dtype=np.float32)
    updated, record = g4.projected_adam_step(state, gradient, g4.PARAMETER_NAMES)

    assert updated.adam.count == 1
    assert record["adam_count_before"] == 0
    assert record["adam_count_after"] == 1
    assert record["projection_events"] == 0
    assert np.all((updated.theta - state.theta) * gradient < 0.0)
    assert updated.projection_count == 0

    extreme = g4.OptimizationState(
        theta=np.asarray(
            [g4.PARAMETER_SPECS["kp_xy"].raw_lower, g4.PARAMETER_SPECS["kp_z"].raw_upper],
            dtype=np.float32,
        ),
        adam=g4.AdamState(
            count=0,
            first_moment=np.zeros(2, dtype=np.float32),
            second_moment=np.zeros(2, dtype=np.float32),
        ),
        projection_count=0,
        gradient_history=(),
        selected_validation_step=0,
        selected_validation_loss=None,
    )
    projected, projected_record = g4.projected_adam_step(
        extreme, np.asarray([1.0, -1.0], dtype=np.float32), g4.PARAMETER_NAMES
    )
    assert projected_record["projection_events"] == 2
    assert projected.projection_count == 2


def test_checkpoint_roundtrip_rejects_overwrite_hash_fingerprint_dtype_and_counter(
    tmp_path: Path,
) -> None:
    fingerprints = {"source": "a" * 64, "runtime": {"cpu": True}}
    state, _ = g4.projected_adam_step(
        g4.initial_optimization_state(2), np.asarray([0.5, -0.25]), g4.PARAMETER_NAMES
    )
    path = tmp_path / "checkpoint.json"
    saved = g4.save_checkpoint(path, state, g4.PARAMETER_NAMES, fingerprints)
    restored, loaded = g4.load_checkpoint(path, g4.PARAMETER_NAMES, fingerprints)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert loaded["payload_sha256"] == saved["payload_sha256"]
    np.testing.assert_array_equal(restored.theta, state.theta)
    np.testing.assert_array_equal(restored.adam.first_moment, state.adam.first_moment)
    np.testing.assert_array_equal(restored.adam.second_moment, state.adam.second_moment)
    assert restored.adam.count == 1
    with pytest.raises(g4.G4ContractError, match="overwrite"):
        g4.save_checkpoint(path, state, g4.PARAMETER_NAMES, fingerprints)
    with pytest.raises(g4.G4ContractError, match="fingerprint"):
        g4.load_checkpoint(path, g4.PARAMETER_NAMES, {"source": "b" * 64})

    corrupted = json.loads(path.read_bytes())
    corrupted["update_count"] = 2
    path.unlink()
    path.write_bytes(g4.canonical_json_bytes(corrupted))
    with pytest.raises(g4.G4ContractError, match="checksum"):
        g4.load_checkpoint(path, g4.PARAMETER_NAMES, fingerprints)


def test_source_origin_runtime_and_freshness_roundtrip(tmp_path: Path) -> None:
    assert Path.cwd().resolve() == g4.EXPECTED_REPOSITORY_ROOT
    assert Path(os.environ["PYTHONPATH"]) == g4.EXPECTED_REPOSITORY_ROOT
    runtime = g4.validate_runtime_environment(REPOSITORY_ROOT)
    path = tmp_path / "source-origin.json"
    written = g4.write_source_origin_record(REPOSITORY_ROOT, path)
    loaded = g4.validate_source_origin_record(REPOSITORY_ROOT, path)

    assert runtime["jax_backend"] == "cpu"
    assert runtime["jax_enable_x64"] is False
    assert written == loaded
    assert set(loaded["origins"]) == {
        "crazyflow",
        "g4_module",
        "g4_cli",
        "g4_unit_tests",
        "g4_integration_tests",
    }
    assert all(
        Path(record["file"]).is_relative_to(REPOSITORY_ROOT)
        for record in loaded["origins"].values()
    )
    with pytest.raises(g4.G4ContractError, match="overwrite"):
        g4.write_source_origin_record(REPOSITORY_ROOT, path)


@pytest.mark.parametrize(
    "target",
    (
        ".",
        "..",
        "../crazyflow/control/mellinger/research",
        "/home/noah3/bachelorarbeit/crazyflow-gradient-research",
        "Crazyflow/control/mellinger/research",
        "crazyflow\\control\\mellinger\\research",
        "crazyflow/control/mellinger/research/../../../../configs/research/mellinger/g4_joint_v1.json",
        g4.CLOSED_TEST_SENTINEL,
        *g4.PROTECTED_ROOT_SENTINELS,
        "artifacts/day13-h100-replication/child.json",
    ),
)
def test_negative_discovery_traversal_case_separator_and_protection_fixtures(target: str) -> None:
    with pytest.raises(g4.G4ContractError):
        g4.validate_discovery_target(target)


def test_symlink_fixture_is_rejected_before_file_read(tmp_path: Path) -> None:
    relative = next(iter(g4.DAY26_INPUTS))
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    harmless = tmp_path / "harmless.json"
    harmless.write_text("{}")
    target.symlink_to(harmless)

    with pytest.raises(g4.G4ContractError, match="symlink"):
        g4._safe_exact_file(tmp_path, relative)


def test_output_freshness_requires_absent_mode_700_tmp_root(tmp_path: Path) -> None:
    root = Path("/tmp") / f"gr-g4-004-unit-{os.getpid()}"
    root.mkdir(mode=0o700)
    try:
        output = root / "output"
        assert g4.validate_task_output_path(output) == output
        output.mkdir(mode=0o700)
        with pytest.raises(g4.G4ContractError, match="absent"):
            g4.validate_task_output_path(output)
        with pytest.raises(g4.G4ContractError):
            g4.validate_task_output_path(tmp_path / "output")
    finally:
        output = root / "output"
        if output.exists():
            output.rmdir()
        root.rmdir()


def test_convergence_requires_both_frozen_rules_and_never_stops_before_1500() -> None:
    gradients = [1.0] * 100 + [0.05] * 1_400
    losses = [2.0] * 500 + [1.0] * 1_000
    record = g4.convergence_status(gradients, losses)

    assert record["gradient_pass"] is True
    assert record["plateau_pass"] is True
    assert record["converged"] is True
    early = g4.convergence_status(gradients[:1_499], losses[:1_499])
    assert early["converged"] is False
    assert early["status"] == "CONTINUE"


def _forwardability_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    default = {
        "mean_validation_loss": 1.0,
        "group_loss": {"soft": 1.0, "nominal": 1.0, "dynamic": 1.0, "hold": 1.0},
        "rmse": {"xy": 1.0, "z": 1.0},
        "integral_contacts": 3,
        "minimum_motor_reserve": 0.10,
        "wrench": [0.8, 0.028284, 0.028284, 0.004751147148794944],
    }
    candidate = {
        "mean_validation_loss": 0.94,
        "group_loss": {"soft": 0.99, "nominal": 1.01, "dynamic": 1.0, "hold": 1.0},
        "rmse": {"xy": 0.99, "z": 1.01},
        "integral_contacts": 3,
        "integral_nearness_degradation": 0.01,
        "soft_nominal_stage_a_torque_clips": 0,
        "soft_nominal_stage_a_motor_clips": 0,
        "soft_nominal_additional_stage_b_clips": 0,
        "dynamic_additional_stage_b_clips": 0,
        "dynamic_stage_a_motor_clip_fraction": 0.002,
        "dynamic_max_clip_interval_s": 0.02,
        "minimum_motor_reserve": 0.09,
        "wrench_degradation": [0.001, 0.0001, 0.0001, 0.00001],
        "technical_gates_pass": True,
        "bounds_pass": True,
        "convergence_pass": True,
    }
    return default, candidate


def test_quantitative_forwardability_gates_are_exact_and_external_to_loss() -> None:
    default, candidate = _forwardability_inputs()
    passing = g4.forwardability_gate(default, candidate)

    assert passing["status"] == "PASS_FORWARDABILITY"
    assert all(passing["checks"].values())
    assert "claim" in passing["claim_boundary"]

    failing_candidate = dict(candidate, mean_validation_loss=0.96)
    failing = g4.forwardability_gate(default, failing_candidate)
    assert failing["status"] == "WITHHELD_FORWARDABILITY"
    assert failing["checks"]["primary_loss"] is False


def test_update_record_digest_is_order_and_byte_sensitive() -> None:
    records = [{"update": 11, "loss": 1.0}, {"update": 12, "loss": 0.9}]

    assert g4.update_records_digest(records) == g4.update_records_digest(list(records))
    assert g4.update_records_digest(records) != g4.update_records_digest(list(reversed(records)))


def test_resume_comparison_requires_exact_updates_11_through_20() -> None:
    continuous = [{"update": index, "value": index} for index in range(1, 21)]
    resumed = continuous[10:]

    record = g4.compare_resume_records(continuous, resumed)
    assert record["updates"] == list(range(11, 21))
    assert record["expected_sha256"] == record["actual_sha256"]
    changed = [dict(item) for item in resumed]
    changed[-1]["value"] = -1
    with pytest.raises(g4.G4ContractError, match="byte-identical"):
        g4.compare_resume_records(continuous, changed)


def test_longrun_checkpoint_carries_complete_state_and_update_evidence(tmp_path: Path) -> None:
    fingerprints = {
        "sources": {"core": "a" * 64},
        "runtime": {"interpreter_sha256": "b" * 64, "jax_backend": "cpu"},
        "config_sha256": "c" * 64,
        "public_day26_sha256": {"input": "d" * 64},
    }
    contract = g4.longrun_contract_payload(fingerprints)
    state, optimizer = g4.projected_adam_step(
        g4.initial_optimization_state(2),
        np.asarray([0.25, -0.5], dtype=np.float32),
        g4.PARAMETER_NAMES,
    )
    evidence = {
        "scientific": {
            "update": 1,
            "batch_index": 0,
            "parent_ids": [f"parent-{index:02d}" for index in range(32)],
            "strata": [f"stratum-{index % 9}" for index in range(32)],
            "loss": 1.25,
            "loss_terms": {name: 0.1 for name in g4.LOSS_TERM_NAMES},
            "gradient": [0.25, -0.5],
            "gradient_norm": float(np.hypot(0.25, 0.5)),
            "microbatch_parent_counts": [4] * 8,
            "microbatch_weights": [0.125] * 8,
            "optimizer": optimizer,
            "projection_count_total": 0,
        },
        "operational": _longrun_operational_update(
            parent_ids=[f"parent-{index:02d}" for index in range(32)]
        ),
    }
    path = tmp_path / "checkpoint-step-000001.json"
    saved = g4.save_longrun_checkpoint(
        path,
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256="e" * 64,
        parent_checkpoint_scientific_sha256="f" * 64,
        update_evidence=evidence,
    )

    assert saved["schema_version"] == g4.LONGRUN_CHECKPOINT_SCHEMA_VERSION
    assert saved["update_count"] == saved["adam"]["count"] == 1
    assert saved["next_batch_index"] == 1
    assert saved["scientific_update"]["loss_terms"] == evidence["scientific"]["loss_terms"]
    assert len(saved["scientific_update"]["microbatch_parent_counts"]) == 8
    assert len(saved["operational_update"]["microbatches"]) == 8
    assert saved["metric_offsets"]["train_updates"] == 1
    assert len(saved["scientific_payload_sha256"]) == 64
    assert len(saved["payload_sha256"]) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_longrun_checkpoint_atomic_overwrite_rejects_and_chains_parent_digest(
    tmp_path: Path,
) -> None:
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    contract = g4.longrun_contract_payload(fingerprints)
    initial = g4.initial_optimization_state(2)
    first_path = tmp_path / "checkpoint-step-000000.json"
    first = g4.save_longrun_checkpoint(
        first_path,
        initial,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256=None,
        parent_checkpoint_scientific_sha256=None,
        update_evidence=None,
    )
    state, optimizer = g4.projected_adam_step(
        initial, np.asarray([0.1, -0.2], dtype=np.float32), g4.PARAMETER_NAMES
    )
    evidence = {
        "scientific": {
            "update": 1,
            "batch_index": 0,
            "loss": 1.0,
            "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
            "gradient": [0.1, -0.2],
            "gradient_norm": float(np.hypot(0.1, 0.2)),
            "parent_ids": [f"p-{index}" for index in range(32)],
            "strata": [f"s-{index % 9}" for index in range(32)],
            "microbatch_parent_counts": [4] * 8,
            "microbatch_weights": [0.125] * 8,
            "optimizer": optimizer,
            "projection_count_total": 0,
        },
        "operational": _longrun_operational_update(),
    }
    second_path = tmp_path / "checkpoint-step-000001.json"
    second = g4.save_longrun_checkpoint(
        second_path,
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256=first["payload_sha256"],
        parent_checkpoint_scientific_sha256=first["scientific_payload_sha256"],
        update_evidence=evidence,
    )

    assert second["parent_checkpoint_payload_sha256"] == first["payload_sha256"]
    assert second["parent_checkpoint_scientific_sha256"] == first["scientific_payload_sha256"]
    with pytest.raises(g4.G4ContractError, match="overwrite"):
        g4.save_longrun_checkpoint(
            second_path,
            state,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            parent_checkpoint_payload_sha256=first["payload_sha256"],
            parent_checkpoint_scientific_sha256=first["scientific_payload_sha256"],
            update_evidence=evidence,
        )


def test_longrun_checkpoint_rejects_fingerprint_dtype_counter_and_parent_drift(
    tmp_path: Path,
) -> None:
    fingerprints = {
        "sources": {"core": "a" * 64},
        "runtime": {"interpreter_sha256": "b" * 64},
        "config_sha256": "c" * 64,
        "public_day26_sha256": {"input": "d" * 64},
    }
    contract = g4.longrun_contract_payload(fingerprints)
    path = tmp_path / "checkpoint-step-000000.json"
    saved = g4.save_longrun_checkpoint(
        path,
        g4.initial_optimization_state(2),
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256=None,
        parent_checkpoint_scientific_sha256=None,
        update_evidence=None,
    )
    g4.load_longrun_checkpoint(
        path,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        expected_parent_payload_sha256=None,
        expected_parent_scientific_sha256=None,
    )

    for key, replacement in (
        ("sources", {"core": "0" * 64}),
        ("runtime", {"interpreter_sha256": "0" * 64}),
        ("config_sha256", "0" * 64),
        ("public_day26_sha256", {"input": "0" * 64}),
    ):
        drifted = dict(fingerprints, **{key: replacement})
        with pytest.raises(g4.G4ContractError, match="fingerprint"):
            g4.load_longrun_checkpoint(
                path,
                drifted,
                run_contract_sha256=contract["payload_sha256"],
                expected_parent_payload_sha256=None,
                expected_parent_scientific_sha256=None,
            )
    with pytest.raises(g4.G4ContractError, match="parent"):
        g4.load_longrun_checkpoint(
            path,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            expected_parent_payload_sha256="0" * 64,
            expected_parent_scientific_sha256=None,
        )

    corrupted = json.loads(path.read_bytes())
    corrupted["theta"]["dtype"] = ">f4"
    unsigned = dict(corrupted)
    unsigned.pop("payload_sha256")
    unsigned.pop("scientific_payload_sha256")
    corrupted["scientific_payload_sha256"] = g4.longrun_scientific_payload_sha256(unsigned)
    payload_unsigned = dict(corrupted)
    payload_unsigned.pop("payload_sha256")
    corrupted["payload_sha256"] = g4.sha256_bytes(g4.canonical_json_bytes(payload_unsigned))
    path.unlink()
    path.write_bytes(g4.canonical_json_bytes(corrupted))
    os.chmod(path, 0o600)
    with pytest.raises(g4.G4ContractError, match="dtype|counter"):
        g4.load_longrun_checkpoint(
            path,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            expected_parent_payload_sha256=None,
            expected_parent_scientific_sha256=None,
        )
    assert saved["update_count"] == 0


def test_longrun_output_requires_fresh_or_exact_owned_layout() -> None:
    root = Path("/tmp") / f"gr-g4-004-m7-output-unit-{os.getpid()}"
    output = root / "output"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700)
    try:
        contract, _, _ = g4.initialize_longrun_output(output, fingerprints)
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert stat.S_IMODE((output / "checkpoints").stat().st_mode) == 0o700
        assert stat.S_IMODE((output / "segments").stat().st_mode) == 0o700
        assert stat.S_IMODE((output / "run-contract.json").stat().st_mode) == 0o600
        g4.validate_longrun_output(output, fingerprints, contract)
        (output / "unknown.txt").write_text("foreign")
        with pytest.raises(g4.G4ContractError, match="unknown|layout"):
            g4.validate_longrun_output(output, fingerprints, contract)
        (output / "unknown.txt").unlink()
        os.chmod(output / "checkpoints", 0o755)
        with pytest.raises(g4.G4ContractError, match="mode|owned"):
            g4.validate_longrun_output(output, fingerprints, contract)
    finally:
        shutil.rmtree(root)


def test_resume_accepts_only_latest_verified_contiguous_checkpoint() -> None:
    root = Path("/tmp") / f"gr-g4-004-m7-resume-unit-{os.getpid()}"
    output = root / "output"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700)
    try:
        contract, state, previous = g4.initialize_longrun_output(output, fingerprints)
        paths = [output / "checkpoints/checkpoint-step-000000.json"]
        for count in (1, 2):
            state, optimizer = g4.projected_adam_step(
                state, np.asarray([0.1, -0.2], dtype=np.float32), g4.PARAMETER_NAMES
            )
            evidence = {
                "scientific": {
                    "update": count,
                    "batch_index": count - 1,
                    "loss": float(count),
                    "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
                    "gradient": [0.1, -0.2],
                    "gradient_norm": float(np.hypot(0.1, 0.2)),
                    "parent_ids": [f"p-{count}-{index}" for index in range(32)],
                    "strata": [f"s-{index % 9}" for index in range(32)],
                    "microbatch_parent_counts": [4] * 8,
                    "microbatch_weights": [0.125] * 8,
                    "optimizer": optimizer,
                    "projection_count_total": 0,
                },
                "operational": _longrun_operational_update(
                    token=count, parent_ids=[f"p-{count}-{index}" for index in range(32)]
                ),
            }
            path = output / f"checkpoints/checkpoint-step-{count:06d}.json"
            previous = g4.save_longrun_checkpoint(
                path,
                state,
                fingerprints,
                run_contract_sha256=contract["payload_sha256"],
                parent_checkpoint_payload_sha256=previous["payload_sha256"],
                parent_checkpoint_scientific_sha256=previous["scientific_payload_sha256"],
                update_evidence=evidence,
            )
            paths.append(path)
        resumed, latest = g4.resume_longrun_output(output, paths[-1], fingerprints)
        assert resumed.adam.count == latest["update_count"] == 2
        with pytest.raises(g4.G4ContractError, match="latest"):
            g4.resume_longrun_output(output, paths[1], fingerprints)
        paths[1].unlink()
        with pytest.raises(g4.G4ContractError, match="contiguous|gap"):
            g4.resume_longrun_output(output, paths[-1], fingerprints)
    finally:
        shutil.rmtree(root)


def test_bounded_segment_persists_each_update_and_exact_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp") / f"gr-g4-004-m7-segment-unit-{os.getpid()}"
    output = root / "output"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700)

    def fake_update(
        state: g4.OptimizationState, names: tuple[str, ...]
    ) -> tuple[g4.OptimizationState, dict[str, Any]]:
        updated, optimizer = g4.projected_adam_step(
            state, np.asarray([0.1, -0.2], dtype=np.float32), names
        )
        count = updated.adam.count
        return updated, {
            "scientific": {
                "update": count,
                "batch_index": state.adam.count % 9,
                "loss": float(count),
                "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
                "gradient": [0.1, -0.2],
                "gradient_norm": float(np.hypot(0.1, 0.2)),
                "parent_ids": [f"p-{count}-{index}" for index in range(32)],
                "strata": [f"s-{index % 9}" for index in range(32)],
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
                "optimizer": optimizer,
                "projection_count_total": 0,
            },
            "operational": _longrun_operational_update(
                token=count, parent_ids=[f"p-{count}-{index}" for index in range(32)]
            ),
        }

    monkeypatch.setattr(g4, "parent_population_evidence", lambda _root: {})
    monkeypatch.setattr(g4, "science_fingerprints", lambda *_args: fingerprints)
    monkeypatch.setattr(g4, "one_longrun_update", fake_update, raising=False)
    try:
        result = g4.run_bounded_longrun_segment(
            REPOSITORY_ROOT, "c" * 64, output, 3, resume_checkpoint=None
        )
        assert result["status"] == "PASS_BOUNDED_LONGRUN_SEGMENT"
        assert result["expected_start_count"] == result["actual_start_count"] == 0
        assert result["expected_end_count"] == result["actual_end_count"] == 3
        assert result["completed_updates"] == 3
        assert [path.name for path in sorted((output / "checkpoints").iterdir())] == [
            f"checkpoint-step-{count:06d}.json" for count in range(4)
        ]
        assert len(list((output / "segments").iterdir())) == 1
    finally:
        shutil.rmtree(root)


def test_partial_segment_preserves_only_last_complete_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp") / f"gr-g4-004-m7-partial-unit-{os.getpid()}"
    output = root / "output"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700)

    def fake_update(
        state: g4.OptimizationState, names: tuple[str, ...]
    ) -> tuple[g4.OptimizationState, dict[str, Any]]:
        if state.adam.count == 1:
            raise g4.G4ContractError("fixture interrupted partial update")
        updated, optimizer = g4.projected_adam_step(
            state, np.asarray([0.1, -0.2], dtype=np.float32), names
        )
        return updated, {
            "scientific": {
                "update": 1,
                "batch_index": 0,
                "loss": 1.0,
                "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
                "gradient": [0.1, -0.2],
                "gradient_norm": float(np.hypot(0.1, 0.2)),
                "parent_ids": [f"p-{index}" for index in range(32)],
                "strata": [f"s-{index % 9}" for index in range(32)],
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
                "optimizer": optimizer,
                "projection_count_total": 0,
            },
            "operational": _longrun_operational_update(),
        }

    monkeypatch.setattr(g4, "parent_population_evidence", lambda _root: {})
    monkeypatch.setattr(g4, "science_fingerprints", lambda *_args: fingerprints)
    monkeypatch.setattr(g4, "one_longrun_update", fake_update, raising=False)
    try:
        with pytest.raises(g4.G4ContractError, match="interrupted"):
            g4.run_bounded_longrun_segment(
                REPOSITORY_ROOT, "c" * 64, output, 2, resume_checkpoint=None
            )
        assert [path.name for path in sorted((output / "checkpoints").iterdir())] == [
            "checkpoint-step-000000.json",
            "checkpoint-step-000001.json",
        ]
        assert not list((output / "segments").iterdir())
    finally:
        shutil.rmtree(root)


def test_longrun_resource_telemetry_enforces_memavailable_swap_and_rss() -> None:
    valid = {
        "MemTotal": 16 * 1024**3,
        "MemAvailable": g4.MINIMUM_AVAILABLE_BYTES,
        "SwapTotal": 0,
        "SwapFree": 0,
    }
    assert g4.validate_longrun_resource_telemetry(valid, 1024)["swap_used_bytes"] == 0
    with pytest.raises(g4.G4ContractError, match="MemAvailable"):
        g4.validate_longrun_resource_telemetry(
            dict(valid, MemAvailable=g4.MINIMUM_AVAILABLE_BYTES - 1), 1024
        )
    with pytest.raises(g4.G4ContractError, match="Swap"):
        g4.validate_longrun_resource_telemetry(dict(valid, SwapTotal=1024, SwapFree=0), 1024)
    with pytest.raises(g4.G4ContractError, match="RSS"):
        g4.validate_longrun_resource_telemetry(valid, int(valid["MemTotal"] * 0.8))


def test_longrun_records_distinct_pre_post_resource_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [(spec, object()) for spec in g4.train_parent_specs()[:32]]
    memory_values = [
        {
            "MemTotal": 16 * 1024**3,
            "MemAvailable": g4.MINIMUM_AVAILABLE_BYTES + index,
            "SwapTotal": 0,
            "SwapFree": 0,
        }
        for index in range(16)
    ]
    rss_values = list(range(100, 116))
    memory_calls = []
    rss_calls = []

    def memory_snapshot() -> dict[str, int]:
        value = memory_values[len(memory_calls)]
        memory_calls.append(value)
        return value

    def getrusage(_who: int) -> object:
        value = rss_values[len(rss_calls)]
        rss_calls.append(value)
        return type("Usage", (), {"ru_maxrss": value})()

    evaluation_count = 0

    def evaluate_items(
        _theta: np.ndarray, _items: list[tuple[object, object]], _names: tuple[str, ...]
    ) -> dict[str, Any]:
        nonlocal evaluation_count
        first = 4 * evaluation_count
        evaluation_count += 1
        return {
            "elapsed_seconds": 0.1,
            "parent_ids": [f"parent-{index}" for index in range(first, first + 4)],
            "aux": {f"loss_{name}": np.zeros(4, dtype=np.float32) for name in g4.LOSS_TERM_NAMES},
        }

    monkeypatch.setattr(g4, "_batch_items", lambda _index: items)
    monkeypatch.setattr(g4, "memory_snapshot", memory_snapshot)
    monkeypatch.setattr(g4.resource, "getrusage", getrusage)
    monkeypatch.setattr(g4, "evaluate_items", evaluate_items)
    monkeypatch.setattr(
        g4,
        "_aggregate_evaluations",
        lambda evaluations: {
            "loss": 1.0,
            "gradient": np.asarray([0.1, -0.2], dtype=np.float32),
            "parent_ids": [item for value in evaluations for item in value["parent_ids"]],
            "microbatch_parent_counts": [4] * 8,
            "microbatch_weights": [0.125] * 8,
        },
    )

    updated, evidence = g4.one_longrun_update(g4.initial_optimization_state(2), g4.PARAMETER_NAMES)

    operational = evidence["operational"]
    assert updated.adam.count == 1
    assert len(memory_calls) == len(rss_calls) == 16
    assert operational["schema_version"] == LONGRUN_RESOURCE_SCHEMA
    assert operational["resource_observation_count"] == 16
    observations = []
    for index, microbatch in enumerate(operational["microbatches"]):
        before = microbatch["resource_before"]
        after = microbatch["resource_after"]
        assert before is not after
        assert before["mem_available_bytes"] == g4.MINIMUM_AVAILABLE_BYTES + 2 * index
        assert after["mem_available_bytes"] == g4.MINIMUM_AVAILABLE_BYTES + 2 * index + 1
        assert before["maximum_rss_bytes"] == rss_values[2 * index] * 1024
        assert after["maximum_rss_bytes"] == rss_values[2 * index + 1] * 1024
        observations.extend((before, after))
    assert len({id(item) for item in observations}) == 16
    assert operational["maximum_rss_bytes"] == max(
        item["maximum_rss_bytes"] for item in observations
    )
    assert operational["minimum_mem_available_bytes"] == min(
        item["mem_available_bytes"] for item in observations
    )
    assert operational["maximum_swap_used_bytes"] == 0


@pytest.mark.parametrize(
    ("phase", "field", "value", "message"),
    (
        ("resource_before", "mem_available_bytes", g4.MINIMUM_AVAILABLE_BYTES - 1, "MemAvailable"),
        ("resource_before", "swap_used_bytes", 1, "Swap"),
        ("resource_before", "maximum_rss_bytes", 16 * 1024**3, "RSS"),
        ("resource_after", "mem_available_bytes", g4.MINIMUM_AVAILABLE_BYTES - 1, "MemAvailable"),
        ("resource_after", "swap_used_bytes", 1, "Swap"),
        ("resource_after", "maximum_rss_bytes", 16 * 1024**3, "RSS"),
    ),
    ids=(
        "before-memavailable",
        "before-swap",
        "before-rss",
        "after-memavailable",
        "after-swap",
        "after-rss",
    ),
)
def test_longrun_pre_post_resource_failures_are_hard_stops(
    tmp_path: Path, phase: str, field: str, value: int, message: str
) -> None:
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    contract = g4.longrun_contract_payload(fingerprints)
    state, optimizer = g4.projected_adam_step(
        g4.initial_optimization_state(2),
        np.asarray([0.1, -0.2], dtype=np.float32),
        g4.PARAMETER_NAMES,
    )
    operational = _longrun_operational_update()
    operational["microbatches"][0][phase][field] = value
    evidence = {
        "scientific": {
            "update": 1,
            "batch_index": 0,
            "loss": 1.0,
            "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
            "gradient": [0.1, -0.2],
            "gradient_norm": float(np.hypot(0.1, 0.2)),
            "parent_ids": [f"p-{index}" for index in range(32)],
            "strata": [f"s-{index % 9}" for index in range(32)],
            "microbatch_parent_counts": [4] * 8,
            "microbatch_weights": [0.125] * 8,
            "optimizer": optimizer,
            "projection_count_total": 0,
        },
        "operational": operational,
    }
    with pytest.raises(g4.G4ContractError, match=message):
        g4.save_longrun_checkpoint(
            tmp_path / "checkpoint-step-000001.json",
            state,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            parent_checkpoint_payload_sha256="c" * 64,
            parent_checkpoint_scientific_sha256="d" * 64,
            update_evidence=evidence,
        )


def test_longrun_resource_schema_roundtrip_and_extrema(tmp_path: Path) -> None:
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    contract = g4.longrun_contract_payload(fingerprints)
    state, optimizer = g4.projected_adam_step(
        g4.initial_optimization_state(2),
        np.asarray([0.1, -0.2], dtype=np.float32),
        g4.PARAMETER_NAMES,
    )
    operational = _longrun_operational_update(token=7)
    evidence = {
        "scientific": {
            "update": 1,
            "batch_index": 0,
            "loss": 1.0,
            "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
            "gradient": [0.1, -0.2],
            "gradient_norm": float(np.hypot(0.1, 0.2)),
            "parent_ids": [f"p-{index}" for index in range(32)],
            "strata": [f"s-{index % 9}" for index in range(32)],
            "microbatch_parent_counts": [4] * 8,
            "microbatch_weights": [0.125] * 8,
            "optimizer": optimizer,
            "projection_count_total": 0,
        },
        "operational": operational,
    }
    path = tmp_path / "checkpoint-step-000001.json"
    saved = g4.save_longrun_checkpoint(
        path,
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256="c" * 64,
        parent_checkpoint_scientific_sha256="d" * 64,
        update_evidence=evidence,
    )
    _, loaded = g4.load_longrun_checkpoint(
        path,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        expected_parent_payload_sha256="c" * 64,
        expected_parent_scientific_sha256="d" * 64,
    )
    assert loaded["operational_update"] == saved["operational_update"] == operational
    assert operational["resource_observation_count"] == 16
    assert len(operational["microbatches"]) == 8

    corrupted = json.loads(path.read_bytes())
    corrupted["operational_update"]["microbatches"][0].pop("resource_after")
    unsigned = dict(corrupted)
    unsigned.pop("payload_sha256")
    corrupted["payload_sha256"] = g4.sha256_bytes(g4.canonical_json_bytes(unsigned))
    path.unlink()
    path.write_bytes(g4.canonical_json_bytes(corrupted))
    os.chmod(path, 0o600)
    with pytest.raises(g4.G4ContractError, match="resource|microbatch|keys"):
        g4.load_longrun_checkpoint(
            path,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            expected_parent_payload_sha256="c" * 64,
            expected_parent_scientific_sha256="d" * 64,
        )


def test_longrun_scientific_digest_excludes_operational_resources() -> None:
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    contract = g4.longrun_contract_payload(fingerprints)
    state, optimizer = g4.projected_adam_step(
        g4.initial_optimization_state(2),
        np.asarray([0.1, -0.2], dtype=np.float32),
        g4.PARAMETER_NAMES,
    )
    scientific = {
        "update": 1,
        "batch_index": 0,
        "loss": 1.0,
        "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
        "gradient": [0.1, -0.2],
        "gradient_norm": float(np.hypot(0.1, 0.2)),
        "parent_ids": [f"p-{index}" for index in range(32)],
        "strata": [f"s-{index % 9}" for index in range(32)],
        "microbatch_parent_counts": [4] * 8,
        "microbatch_weights": [0.125] * 8,
        "optimizer": optimizer,
        "projection_count_total": 0,
    }
    first = g4._longrun_checkpoint_payload(
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256="c" * 64,
        parent_checkpoint_scientific_sha256="d" * 64,
        update_evidence={
            "scientific": scientific,
            "operational": _longrun_operational_update(token=1),
        },
    )
    second = g4._longrun_checkpoint_payload(
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256="c" * 64,
        parent_checkpoint_scientific_sha256="d" * 64,
        update_evidence={
            "scientific": scientific,
            "operational": _longrun_operational_update(token=101),
        },
    )

    assert g4.LONGRUN_RESOURCE_EVIDENCE_SCHEMA_VERSION == LONGRUN_RESOURCE_SCHEMA
    assert first["scientific_payload_sha256"] == second["scientific_payload_sha256"]
    assert first["payload_sha256"] != second["payload_sha256"]


def _m8_scientific(mean_loss: float) -> dict[str, Any]:
    groups = {
        name: mean_loss for name in ("soft", "nominal", "dynamic", "hold", "climb", "descent")
    }
    nearness = {
        f"{motion}/{profile}": 0.1
        for motion in ("soft", "nominal", "dynamic")
        for profile in ("hold", "climb", "descent")
    }
    return {
        "mean_validation_loss": mean_loss,
        "loss_terms": {name: mean_loss / len(g4.LOSS_TERM_NAMES) for name in g4.LOSS_TERM_NAMES},
        "group_loss": groups,
        "rmse": {"xy": 0.1, "z": 0.1},
        "integral_contacts": 0,
        "maximum_normalized_integral_nearness_by_stratum": nearness,
        "soft_nominal_stage_a_torque_clips": 0,
        "soft_nominal_stage_a_motor_clips": 0,
        "soft_nominal_additional_stage_b_clips": 0,
        "dynamic_additional_stage_b_clips": 0,
        "dynamic_stage_a_motor_clip_fraction": 0.0,
        "dynamic_max_clip_interval_s": 0.0,
        "minimum_motor_reserve": 0.2,
        "wrench": [0.0, 0.0, 0.0, 0.0],
        "per_parent": [
            {"parent_id": parent, "loss": mean_loss} for parent in g4.VALIDATION_IDENTITIES
        ],
        "near_limit": [
            {"parent_id": parent, "technically_valid": True} for parent in g4.NEAR_LIMIT_IDENTITIES
        ],
        "technical_gates_pass": True,
        "bounds_pass": True,
    }


def _m8_operational(token: int = 0) -> dict[str, Any]:
    total = 16 * 1024**3
    allowed = int(g4.RSS_FRACTION_LIMIT * total)
    batches = []
    observations = []
    identities = (
        g4.VALIDATION_IDENTITIES[:4],
        g4.VALIDATION_IDENTITIES[4:8],
        g4.VALIDATION_IDENTITIES[8:],
        g4.NEAR_LIMIT_IDENTITIES[:4],
        g4.NEAR_LIMIT_IDENTITIES[4:],
    )
    for index, parent_ids in enumerate(identities):
        before = {
            "mem_total_bytes": total,
            "mem_available_bytes": g4.MINIMUM_AVAILABLE_BYTES + 100 - 2 * index,
            "swap_used_bytes": 0,
            "maximum_rss_bytes": 1000 + token + 2 * index,
            "maximum_allowed_rss_bytes": allowed,
        }
        after = {
            "mem_total_bytes": total,
            "mem_available_bytes": g4.MINIMUM_AVAILABLE_BYTES + 99 - 2 * index,
            "swap_used_bytes": 0,
            "maximum_rss_bytes": 1001 + token + 2 * index,
            "maximum_allowed_rss_bytes": allowed,
        }
        observations.extend((before, after))
        batches.append(
            {
                "batch_index": index,
                "split": "feasible" if index < 3 else "near_limit",
                "parent_count": len(parent_ids),
                "parent_ids": list(parent_ids),
                "elapsed_seconds": 0.1,
                "resource_before": before,
                "resource_after": after,
            }
        )
    target = g4.EXPECTED_INTERPRETER.resolve(strict=True)
    return {
        "started_unix_ns": 1000 + token,
        "finished_unix_ns": 2000 + token,
        "process_elapsed_seconds": 1.0,
        "resource_observation_count": 10,
        "maximum_rss_bytes": max(item["maximum_rss_bytes"] for item in observations),
        "minimum_mem_available_bytes": min(item["mem_available_bytes"] for item in observations),
        "maximum_swap_used_bytes": 0,
        "evaluation_batches": batches,
        "host": {
            "hostname": os.uname().nodename,
            "interpreter": str(g4.EXPECTED_INTERPRETER),
            "interpreter_target": str(target),
            "interpreter_sha256": g4.sha256_file(target),
        },
    }


def _m8_validation(
    update_count: int, mean_loss: float, previous: dict[str, Any] | None = None, token: int = 0
) -> dict[str, Any]:
    return g4.m8_validation_evidence_payload(
        update_count,
        np.zeros(2, dtype=np.float32),
        _m8_scientific(mean_loss),
        _m8_operational(token),
        previous_validation_state=previous,
        convergence_state=g4.convergence_status([], []),
    )


def test_m8_public_contract_is_versioned_and_m7_budget_remains_0_to_100() -> None:
    assert g4.M8_CONTRACT_SCHEMA_VERSION == "crazyflow.mellinger_g4_m8_contract.v1"
    contract = g4.m8_contract_payload({"sources": {"core": "a" * 64}})
    assert contract["maximum_cumulative_update_count"] == 2000
    assert contract["additional_updates_per_process"] == [1, 100]
    assert contract["validation_updates"] == list(range(0, 2001, 250))
    assert contract["resource_schema_version"] == g4.LONGRUN_RESOURCE_EVIDENCE_SCHEMA_VERSION
    assert g4.LONGRUN_MAX_UPDATES == 100
    assert g4.longrun_contract_payload({})["maximum_update_count"] == 100


def test_m8_validation_schedule_selection_tie_and_test_split_exclusion() -> None:
    assert g4.M8_VALIDATION_SCHEMA_VERSION == "crazyflow.mellinger_g4_m8_validation.v1"
    baseline = _m8_validation(0, 10.0)
    tied = _m8_validation(250, 10.0 - 5.0e-7, baseline["validation_state"], token=1)
    improved = _m8_validation(500, 9.0, tied["validation_state"], token=2)
    assert tied["selection"]["selected_validation_step"] == 0
    assert improved["selection"]["selected_validation_step"] == 500
    assert improved["validation_state"]["completed_updates"] == [0, 250, 500]
    all_ids = improved["feasible_parent_ids"] + improved["near_limit_parent_ids"]
    assert all("test" not in parent_id for parent_id in all_ids)


def test_m8_checkpoint_schema_roundtrip_strict_digests_and_atomic_overwrite(tmp_path: Path) -> None:
    assert g4.M8_CHECKPOINT_SCHEMA_VERSION == "crazyflow.mellinger_g4_m8_checkpoint.v1"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    contract = g4.m8_contract_payload(fingerprints)
    validation = _m8_validation(0, 10.0)
    state = g4.initial_m8_state(validation)
    path = tmp_path / "m8-checkpoint-step-000000.json"
    saved = g4.save_m8_checkpoint(
        path,
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256=None,
        parent_checkpoint_scientific_sha256=None,
        parent_checkpoint_operational_sha256=None,
        segment_coordinates={"start_update": 0, "end_update": 0},
        update_evidence=None,
        validation_evidence=validation,
    )
    restored, loaded = g4.load_m8_checkpoint(
        path,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        expected_parent_payload_sha256=None,
        expected_parent_scientific_sha256=None,
        expected_parent_operational_sha256=None,
    )
    assert restored.adam.count == loaded["update_count"] == 0
    assert saved["segment_coordinates"] == {"start_update": 0, "end_update": 0}
    assert all(
        len(saved[name]) == 64
        for name in ("scientific_payload_sha256", "operational_payload_sha256", "payload_sha256")
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(g4.G4ContractError, match="overwrite"):
        g4.save_m8_checkpoint(
            path,
            state,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            parent_checkpoint_payload_sha256=None,
            parent_checkpoint_scientific_sha256=None,
            parent_checkpoint_operational_sha256=None,
            segment_coordinates={"start_update": 0, "end_update": 0},
            update_evidence=None,
            validation_evidence=validation,
        )
    for invalid in (
        {},
        {"start_update": 0},
        {"start_update": 0, "end_update": 0, "extra": 0},
        {"start_update": True, "end_update": 0},
        {"start_update": 0.0, "end_update": 0},
        {"start_update": -1, "end_update": 0},
        {"start_update": 1, "end_update": 0},
        {"start_update": 0, "end_update": 1},
    ):
        with pytest.raises(g4.G4ContractError, match="segment coordinate"):
            g4._validate_m8_segment_coordinates(invalid, 0)
    extra = dict(saved, unexpected=True)
    extra_path = tmp_path / "extra/m8-checkpoint-step-000000.json"
    extra_path.parent.mkdir()
    extra_path.write_bytes(g4.canonical_json_bytes(extra))
    os.chmod(extra_path, 0o600)
    with pytest.raises(g4.G4ContractError, match="top-level keys"):
        g4.load_m8_checkpoint(
            extra_path,
            fingerprints,
            run_contract_sha256=contract["payload_sha256"],
            expected_parent_payload_sha256=None,
            expected_parent_scientific_sha256=None,
            expected_parent_operational_sha256=None,
        )


def test_m8_resume_restores_latest_lineage_windows_and_partial_handoff(tmp_path: Path) -> None:
    assert g4.M8_SEGMENT_SCHEMA_VERSION == "crazyflow.mellinger_g4_m8_segment.v1"
    output = tmp_path / "output"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    validation = _m8_validation(0, 10.0)
    contract, state, first = g4.initialize_m8_output(output, fingerprints, validation)
    parent_ids = [f"p-{index}" for index in range(32)]
    state, evidence = g4.one_m8_update_from_evidence(
        state,
        {
            "scientific": {
                "update": 1,
                "batch_index": 0,
                "parent_ids": parent_ids,
                "strata": [f"s-{index % 9}" for index in range(32)],
                "loss": 9.0,
                "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
                "gradient": [0.1, -0.2],
                "gradient_norm": float(np.hypot(0.1, 0.2)),
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
            },
            "operational": _longrun_operational_update(parent_ids=parent_ids),
        },
    )
    latest_path = output / "checkpoints/m8-checkpoint-step-000001.json"
    latest = g4.save_m8_checkpoint(
        latest_path,
        state,
        fingerprints,
        run_contract_sha256=contract["payload_sha256"],
        parent_checkpoint_payload_sha256=first["payload_sha256"],
        parent_checkpoint_scientific_sha256=first["scientific_payload_sha256"],
        parent_checkpoint_operational_sha256=first["operational_payload_sha256"],
        segment_coordinates={"start_update": 0, "end_update": 1},
        update_evidence=evidence,
        validation_evidence=None,
    )
    restored, payload = g4.resume_m8_output(output, latest_path, fingerprints)
    assert restored.adam.count == payload["update_count"] == 1
    assert restored.loss_window == (9.0,)
    assert payload["parent_checkpoint_payload_sha256"] == first["payload_sha256"]
    assert payload["segment_coordinates"] == {"start_update": 0, "end_update": 1}
    with pytest.raises(g4.G4ContractError, match="latest"):
        g4.resume_m8_output(
            output, output / "checkpoints/m8-checkpoint-step-000000.json", fingerprints
        )
    assert latest["validation_evidence"] is None


def test_m8_convergence_forwardability_and_2000_withhold_states() -> None:
    assert g4.M8_MAX_UPDATES == 2000
    continuing = g4.m8_terminal_state(
        g4.convergence_status([], []), {"status": "WITHHELD_FORWARDABILITY"}, 0, 0
    )
    converged = dict(
        g4.convergence_status([1.0] * 1500, [1.0] * 1500), converged=True, status="CONVERGED"
    )
    ready = g4.m8_terminal_state(converged, {"status": "PASS_FORWARDABILITY"}, 1500, 1250)
    withheld = g4.m8_terminal_state(
        dict(converged, converged=False, status="WITHHELD_NO_CONVERGENCE"),
        {"status": "WITHHELD_FORWARDABILITY"},
        2000,
        1750,
    )
    assert continuing["reason"] == "CONTINUE" and not continuing["terminal"]
    assert ready["reason"] == "READY_FOR_M8_CANDIDATE_REVIEW" and ready["terminal"]
    assert withheld["reason"] == "WITHHELD_NO_CONVERGENCE" and withheld["terminal"]
    assert not any(item["automatic_resume_authorized"] for item in (continuing, ready, withheld))


def test_m8_resource_v1_observations_and_scientific_operational_digest_separation() -> None:
    assert g4.M8_VALIDATION_SCHEMA_VERSION == "crazyflow.mellinger_g4_m8_validation.v1"
    first = _m8_validation(0, 10.0, token=1)
    second = _m8_validation(0, 10.0, token=101)
    assert first["operational"]["resource_observation_count"] == 10
    assert [batch["parent_count"] for batch in first["operational"]["evaluation_batches"]] == [
        4,
        4,
        1,
        4,
        2,
    ]
    assert first["scientific_payload_sha256"] == second["scientific_payload_sha256"]
    assert first["operational_payload_sha256"] != second["operational_payload_sha256"]
    assert first["payload_sha256"] != second["payload_sha256"]


def _backend_runtime_payload() -> dict[str, Any]:
    root = "/tmp/relocated-g4-product"
    cache = "/tmp/gr-g4-004-backend-runtime"
    return {
        "schema_version": g4.BACKEND_RUNTIME_SCHEMA_VERSION,
        "profile": "local_cpu",
        "backend": "cpu",
        "repository": {"root": root, "head": "a" * 40, "tree": "b" * 40, "clean": True},
        "interpreter": {
            "path": "/opt/g4/python",
            "target": "/usr/bin/python3.12",
            "sha256": "c" * 64,
            "python_version": "3.12.13",
        },
        "environment": {
            "manager": "repository_venv",
            "name": None,
            "pixi_version": None,
            "pyproject_sha256": "4ef3f08a5470a2d501881ff7b467397cdb3996631cad2999ee213cdc2e9ee882",
            "pixi_lock_sha256": "a38831c8560cdc3d04c9f74709bbf2e3050c477571f8101629f2d2149fa5c7ec",
        },
        "jax": {
            "python_version": "3.12.13",
            "jax_version": "0.10.1",
            "jaxlib_version": "0.10.1",
            "numpy_version": "2.5.1",
            "jax_cuda12_plugin_version": None,
            "jax_cuda12_pjrt_version": None,
            "float_dtype": "float32",
            "x64_enabled": False,
        },
        "device": {
            "observed_backend": "cpu",
            "device_count": 1,
            "devices": [{"id": 0, "platform": "cpu", "device_kind": "cpu"}],
        },
        "cache_roots": {
            "xdg": f"{cache}/xdg",
            "tmp": f"{cache}/tmp",
            "jax": f"{cache}/jax-cache",
            "cuda": f"{cache}/cuda-cache",
        },
        "environment_flags": {
            "pythonpath": root,
            "python_dont_write_bytecode": "1",
            "python_no_user_site": "1",
            "jax_enable_x64": "false",
            "jax_platforms": "cpu",
            "xla_flags": "",
            "xla_flags_sha256": g4.sha256_bytes(b""),
        },
    }


def _backend_resource_record(token: int = 0) -> dict[str, int]:
    total = 16 * 1024**3
    return {
        "mem_total_bytes": total,
        "mem_available_bytes": g4.MINIMUM_AVAILABLE_BYTES + token,
        "swap_used_bytes": 0,
        "maximum_rss_bytes": 1024 + token,
        "maximum_allowed_rss_bytes": int(g4.RSS_FRACTION_LIMIT * total),
    }


def _backend_parity_scientific() -> dict[str, Any]:
    validation = list(g4.VALIDATION_IDENTITIES[:4])
    train = [f"train-parent-{index:02d}" for index in range(32)]
    loss_terms = {
        name: g4.backend_numeric_record(np.float32(index + 1), "loss_gradient")
        for index, name in enumerate(g4.LOSS_TERM_NAMES)
    }
    microbatches = []
    for index in range(8):
        microbatches.append(
            {
                "microbatch_index": index,
                "parent_ids": train[4 * index : 4 * index + 4],
                "mean_loss": g4.backend_numeric_record(np.float32(index + 1), "loss_gradient"),
                "loss_terms": loss_terms,
                "gradient": g4.backend_numeric_record(
                    np.asarray([index, -index], dtype=np.float32), "loss_gradient"
                ),
            }
        )
    tracking_names = (
        "loss_total",
        *[f"loss_{name}" for name in g4.LOSS_TERM_NAMES],
        "position_rmse_m",
        "velocity_rmse_m_s",
        "max_position_error_m",
        "control_effort",
        "control_smoothness",
        "motor_saturation_fraction",
        "zero_thrust_gate_fraction",
        "floor_clip_fraction",
        "nonfinite_state_fraction",
    )
    return {
        "parent_ids": {"default_rollout": validation, "loss_gradient": train},
        "parameter_names": list(g4.PARAMETER_NAMES),
        "default_theta": g4.backend_numeric_record(np.zeros(2, dtype=np.float32), "rollout"),
        "default_physical": g4.backend_numeric_record(
            g4.physical_from_theta(np.zeros(2, dtype=np.float32)), "rollout"
        ),
        "bounds": {
            "raw": g4.backend_numeric_record(
                np.asarray([[-1.3862944, 1.0986123], [-1.4271164, 0.6931472]], dtype=np.float32),
                "rollout",
            ),
            "physical": g4.backend_numeric_record(
                np.asarray([[0.1, 1.2], [0.3, 2.5]], dtype=np.float32), "rollout"
            ),
        },
        "default_rollout": {
            "final_carry": [
                {
                    "path": "states.pos",
                    "value": g4.backend_numeric_record(
                        np.zeros((4, 3), dtype=np.float32), "rollout"
                    ),
                }
            ],
            "tracking": {
                name: g4.backend_numeric_record(np.float32(0.0), "rollout")
                for name in tracking_names
            },
        },
        "loss_gradient": {
            "microbatches": microbatches,
            "aggregate": {
                "parent_ids": train,
                "mean_loss": g4.backend_numeric_record(np.float32(4.5), "loss_gradient"),
                "loss_terms": loss_terms,
                "gradient": g4.backend_numeric_record(
                    np.asarray([3.5, -3.5], dtype=np.float32), "loss_gradient"
                ),
                "gradient_norm": g4.backend_numeric_record(
                    np.float32(np.hypot(3.5, 3.5)), "loss_gradient"
                ),
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
                "optimizer_update_count": 0,
            },
        },
    }


def test_backend_input_manifest_and_runtime_profiles_are_strict_and_relocatable(
    tmp_path: Path,
) -> None:
    assert (
        g4.BACKEND_INPUT_MANIFEST_SCHEMA_VERSION
        == "crazyflow.mellinger_g4_backend_input_manifest.v1"
    )
    manifest_path = REPOSITORY_ROOT / g4.BACKEND_INPUT_MANIFEST_RELATIVE_PATH
    manifest = g4.load_backend_input_manifest(manifest_path, REPOSITORY_ROOT)
    assert set(manifest) == {
        "schema_version",
        "claim_boundary",
        "interface_parent",
        "immutable_product",
        "public_inputs",
        "science",
        "runtime_profiles",
        "parity_contract",
        "throughput_contract",
        "resume_contract",
        "protection",
    }
    assert manifest["interface_parent"] == {
        "commit": "05cd1db24092f71f39ad23a576bca243a431c035",
        "tree": "fce30d3619f9cca2e912c9ac82afd569a27a3a4c",
    }
    assert list(manifest["runtime_profiles"]) == ["local_cpu", "remote_pixi_cpu", "remote_pixi_gpu"]
    assert [item["path"] for item in manifest["public_inputs"]] == list(g4.DAY26_INPUTS)
    relocated = tmp_path / "g4_backend_evidence_v1.json"
    shutil.copyfile(manifest_path, relocated)
    os.chmod(relocated, 0o644)
    assert g4.load_backend_input_manifest(relocated, REPOSITORY_ROOT) == manifest


def test_backend_runtime_rejects_unknown_profile_backend_interpreter_root_and_environment() -> None:
    payload = _backend_runtime_payload()
    validated = g4.validate_backend_runtime_payload(
        payload,
        expected_repository_root=Path(payload["repository"]["root"]),
        expected_backend="cpu",
        expected_profile="local_cpu",
        observed_payload=payload,
    )
    assert validated == payload
    mutations = (("profile", "unknown"), ("backend", "gpu"))
    for key, value in mutations:
        corrupted = json.loads(json.dumps(payload))
        corrupted[key] = value
        with pytest.raises(g4.G4ContractError):
            g4.validate_backend_runtime_payload(
                corrupted,
                expected_repository_root=Path(payload["repository"]["root"]),
                expected_backend="cpu",
                expected_profile="local_cpu",
                observed_payload=payload,
            )
    for group, key, value in (
        ("interpreter", "path", "/foreign/python"),
        ("repository", "root", "/foreign/repository"),
        ("environment", "manager", "pixi"),
    ):
        corrupted = json.loads(json.dumps(payload))
        corrupted[group][key] = value
        with pytest.raises(g4.G4ContractError):
            g4.validate_backend_runtime_payload(
                corrupted,
                expected_repository_root=Path(payload["repository"]["root"]),
                expected_backend="cpu",
                expected_profile="local_cpu",
                observed_payload=payload,
            )


def test_backend_parity_schema_maps_default_loss_v1_and_raw_gradient_fields() -> None:
    scientific = _backend_parity_scientific()
    resource_record = _backend_resource_record()
    first = g4.backend_parity_evidence_payload(
        backend="cpu",
        provenance={"input_manifest_sha256": "a" * 64, "runtime_contract_sha256": "b" * 64},
        runtime={"profile": "local_cpu", "token": 1},
        resource={"before": resource_record, "after": resource_record},
        scientific=scientific,
        operational={"sample_index": 1},
    )
    second = g4.backend_parity_evidence_payload(
        backend="cpu",
        provenance=first["provenance"],
        runtime={"profile": "local_cpu", "token": 2},
        resource=first["resource"],
        scientific=scientific,
        operational={"sample_index": 2},
    )
    assert first["schema_version"] == "crazyflow.mellinger_g4_backend_parity_evidence.v1"
    assert first["status"] == "PASS_BACKEND_PARITY_SAMPLE"
    assert len(first["scientific"]["loss_gradient"]["microbatches"]) == 8
    assert set(first["scientific"]["loss_gradient"]["aggregate"]["loss_terms"]) == set(
        g4.LOSS_TERM_NAMES
    )
    assert first["scientific_payload_sha256"] == second["scientific_payload_sha256"]
    assert first["operational_payload_sha256"] != second["operational_payload_sha256"]


def test_backend_parity_public_sample_contains_complete_carry_and_tracking_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_carry = {
        "state": {
            "position": np.arange(12, dtype=np.float32).reshape(4, 3),
            "velocity": np.full((4, 3), 0.25, dtype=np.float32),
        },
        "time": np.asarray(1.5, dtype=np.float32),
    }
    loss_aux = {
        "per_case_loss": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        **{
            f"loss_{name}": np.full(4, index + 1, dtype=np.float32)
            for index, name in enumerate(g4.LOSS_TERM_NAMES)
        },
        **{
            f"metric_{name}": np.full(4, index / 10, dtype=np.float32)
            for index, name in enumerate(
                (
                    "position_rmse_m",
                    "velocity_rmse_m_s",
                    "max_position_error_m",
                    "control_effort",
                    "control_smoothness",
                    "motor_saturation_fraction",
                    "zero_thrust_gate_fraction",
                    "floor_clip_fraction",
                    "nonfinite_state_fraction",
                ),
                1,
            )
        },
    }
    parent_ids = [f"train-parent-{index:02d}" for index in range(32)]
    evaluation = {
        "parent_ids": parent_ids,
        "loss": 2.5,
        "gradient": np.asarray([0.1, -0.2], dtype=np.float32),
        "aux": {
            f"loss_{name}": np.full(32, index + 1, dtype=np.float32)
            for index, name in enumerate(g4.LOSS_TERM_NAMES)
        },
    }
    monkeypatch.setattr(g4, "fixed_day26_items", lambda *_args: ())
    monkeypatch.setattr(g4, "_batch_items", lambda *_args: ())
    monkeypatch.setattr(g4, "evaluate_items", lambda *_args: evaluation)
    monkeypatch.setattr(
        g4,
        "default_parity",
        lambda *_args, **_kwargs: {
            "parent_ids": list(g4.VALIDATION_IDENTITIES[:4]),
            "loss": 2.5,
            "canonical_final": final_carry,
            "loss_aux": loss_aux,
            "carry_digest": "a" * 64,
        },
    )

    result = g4.run_backend_parity(
        backend="cpu",
        repository_root=REPOSITORY_ROOT,
        provenance={"input_manifest_sha256": "b" * 64, "runtime_contract_sha256": "c" * 64},
        runtime={"profile": "local_cpu"},
        resource={"before": _backend_resource_record(), "after": _backend_resource_record()},
    )

    rollout = result["scientific"]["default_rollout"]
    expected_paths = [path for path, _leaf in g3._leaf_records(final_carry)]
    assert [record["path"] for record in rollout["final_carry"]] == expected_paths
    assert all(
        set(record["value"])
        == {"category", "dtype", "shape", "finite", "rtol", "atol", "data_hex", "sha256"}
        for record in rollout["final_carry"]
    )
    assert set(rollout["tracking"]) == {
        "loss_total",
        *{f"loss_{name}" for name in g4.LOSS_TERM_NAMES},
        "position_rmse_m",
        "velocity_rmse_m_s",
        "max_position_error_m",
        "control_effort",
        "control_smoothness",
        "motor_saturation_fraction",
        "zero_thrust_gate_fraction",
        "floor_clip_fraction",
        "nonfinite_state_fraction",
    }


def test_backend_throughput_requires_blocked_warmup_and_three_steady_samples() -> None:
    resource_record = _backend_resource_record()
    scientific = {
        "parent_ids": [f"train-parent-{index:02d}" for index in range(32)],
        "microbatch_parent_counts": [4] * 8,
        "microbatch_weights": [0.125] * 8,
        "loss": g4.backend_numeric_record(np.float32(1.0), "loss_gradient"),
        "gradient": g4.backend_numeric_record(
            np.asarray([0.1, -0.2], dtype=np.float32), "loss_gradient"
        ),
        "gradient_norm": g4.backend_numeric_record(np.float32(np.hypot(0.1, 0.2)), "loss_gradient"),
        "adam_count": 1,
    }
    warmup = g4.backend_throughput_evidence_payload(
        backend="cpu",
        phase="warmup",
        sample_index=1,
        provenance={"input_manifest_sha256": "a" * 64, "runtime_contract_sha256": "b" * 64},
        runtime={"profile": "local_cpu"},
        resource={"before": resource_record, "after": resource_record},
        scientific=scientific,
        elapsed_seconds=1.0,
        device_memory={"bytes_in_use": None, "peak_bytes_in_use": None, "bytes_limit": None},
    )
    assert warmup["operational"]["block_until_ready"] is True
    assert warmup["scientific"]["adam_count"] == 1
    for sample_index in (1, 2, 3):
        assert (
            g4.backend_throughput_evidence_payload(
                backend="cpu",
                phase="steady",
                sample_index=sample_index,
                provenance=warmup["provenance"],
                runtime=warmup["runtime"],
                resource=warmup["resource"],
                scientific=scientific,
                elapsed_seconds=float(sample_index),
                device_memory=warmup["operational"]["device_memory"],
            )["status"]
            == "PASS_BACKEND_THROUGHPUT_SAMPLE"
        )
    with pytest.raises(g4.G4ContractError, match="warmup"):
        g4.backend_throughput_evidence_payload(
            backend="cpu",
            phase="warmup",
            sample_index=2,
            provenance=warmup["provenance"],
            runtime=warmup["runtime"],
            resource=warmup["resource"],
            scientific=scientific,
            elapsed_seconds=1.0,
            device_memory=warmup["operational"]["device_memory"],
        )


def test_backend_gpu_internal_resume_preserves_science_and_segment_digest_separation() -> None:
    resource_record = _backend_resource_record()
    scientific = {
        "end_checkpoint_scientific_sha256": "a" * 64,
        "theta": g4.backend_numeric_record(np.asarray([0.1, -0.2], dtype=np.float32), "exact"),
        "adam_count": 20,
        "rng": {"next_batch_index": 2},
        "history_sha256": "b" * 64,
    }
    common = {
        "backend": "gpu",
        "provenance": {"input_manifest_sha256": "c" * 64, "runtime_contract_sha256": "d" * 64},
        "runtime": {"profile": "remote_pixi_gpu"},
        "resource": {"before": resource_record, "after": resource_record},
        "scientific": scientific,
    }
    continuous = g4.backend_resume_segment_payload(
        mode="continuous",
        segment_coordinates={"start_update": 0, "end_update": 20},
        operational={"token": 1},
        **common,
    )
    resumed = g4.backend_resume_segment_payload(
        mode="resume",
        segment_coordinates={"start_update": 10, "end_update": 20},
        operational={"token": 2},
        **common,
    )
    assert continuous["scientific_payload_sha256"] == resumed["scientific_payload_sha256"]
    assert continuous["operational_payload_sha256"] != resumed["operational_payload_sha256"]
    assert continuous["payload_sha256"] != resumed["payload_sha256"]
    with pytest.raises(g4.G4ContractError, match="coordinates"):
        g4.backend_resume_segment_payload(
            mode="resume",
            segment_coordinates={"start_update": 0, "end_update": 20},
            operational={"token": 3},
            **common,
        )


def test_backend_contract_leaves_m7_m8_resource_and_protection_surfaces_unchanged() -> None:
    assert g4.BACKEND_INPUT_MANIFEST_RELATIVE_PATH not in g4.WRITE_PATHS
    assert g4.LONGRUN_MAX_UPDATES == 100
    assert g4.M8_MAX_UPDATES == 2000
    assert g4.LONGRUN_RESOURCE_EVIDENCE_SCHEMA_VERSION == LONGRUN_RESOURCE_SCHEMA
    assert g4.M8_CHECKPOINT_SCHEMA_VERSION == "crazyflow.mellinger_g4_m8_checkpoint.v1"
    assert g4.protected_contract() == {
        "opaque_roots": list(g4.PROTECTED_ROOT_SENTINELS),
        "closed_test": g4.CLOSED_TEST_SENTINEL,
        "closed_test_role": "opaque_string_only",
        "contents_opened_or_hashed": False,
        "children_listed": False,
    }


def test_backend_neutral_parent_v1_validates_all_public_records_and_manifest_pin() -> None:
    payload = g4.load_backend_neutral_parent_payload(REPOSITORY_ROOT)
    manifest = g4.load_backend_input_manifest(
        REPOSITORY_ROOT / g4.BACKEND_INPUT_MANIFEST_RELATIVE_PATH, REPOSITORY_ROOT
    )

    assert payload["schema_version"] == "crazyflow.mellinger_g4_backend_neutral_parents.v1"
    assert payload["train_parent_count"] == 288
    assert payload["day26_parent_count"] == 24
    assert len(payload["parents"]) == 312
    assert len({item["identity"] for item in payload["parents"]}) == 312
    assert manifest["backend_neutral_parents"] == {
        "mode": "100644",
        "path": g4.BACKEND_NEUTRAL_PARENT_RELATIVE_PATH,
        "schema_version": g4.BACKEND_NEUTRAL_PARENT_SCHEMA_VERSION,
        "sha256": g4.sha256_file(REPOSITORY_ROOT / g4.BACKEND_NEUTRAL_PARENT_RELATIVE_PATH),
        "bytes": (REPOSITORY_ROOT / g4.BACKEND_NEUTRAL_PARENT_RELATIVE_PATH).stat().st_size,
        "payload_sha256": payload["payload_sha256"],
    }


def test_backend_neutral_parent_v1_inverse_matches_all_312_cpu_parents() -> None:
    result = g4.verify_backend_neutral_parent_inverse(REPOSITORY_ROOT)

    assert result["status"] == "PASS_BACKEND_NEUTRAL_PARENT_INVERSE"
    assert result["parent_count"] == 312
    assert result["leaf_count"] == 312 * 7
    assert result["parent_digest_count"] == 312


def test_backend_neutral_parent_v1_materializes_same_hostbytes_without_cpu_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, bytes]] = []

    class FakeDevice:
        def __init__(self, platform: str) -> None:
            self.platform = platform

    devices = {name: FakeDevice(name) for name in ("cpu", "gpu")}

    def fake_materialize(raw: np.ndarray, device: FakeDevice) -> np.ndarray:
        observed.append((device.platform, raw.tobytes(order="C")))
        return raw.copy()

    monkeypatch.setattr(g4.jax, "devices", lambda name: [devices[name]])
    monkeypatch.setattr(g4.jax, "device_put", fake_materialize)
    monkeypatch.setattr(
        g4,
        "_reconstruct_backend_neutral_parent_records_cpu",
        lambda: pytest.fail("CPU reconstruction entered the artifact materialization path"),
    )

    cpu = g4.load_backend_neutral_parent_items(REPOSITORY_ROOT, backend="cpu")
    cpu_observed = list(observed)
    observed.clear()
    gpu = g4.load_backend_neutral_parent_items(REPOSITORY_ROOT, backend="gpu")

    assert [spec.episode_id for spec, _candidate in cpu] == [
        spec.episode_id for spec, _candidate in gpu
    ]
    assert [candidate.parent_digest for _spec, candidate in cpu] == [
        candidate.parent_digest for _spec, candidate in gpu
    ]
    assert cpu_observed == [("cpu", raw) for _platform, raw in observed]
    assert observed == [("gpu", raw) for _platform, raw in cpu_observed]


def test_build_simulation_preserves_cpu_default_and_requires_explicit_observed_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeSim:
        step_pipeline = robust.EXPECTED_PIPELINE

        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            self.device = type("Device", (), {"platform": kwargs["device"]})()

    monkeypatch.setattr(robust, "Sim", FakeSim)

    assert robust.build_simulation(4).device.platform == "cpu"
    assert robust.build_simulation(4, device="gpu").device.platform == "gpu"
    assert [call["device"] for call in calls] == ["cpu", "gpu"]

    class WrongSim(FakeSim):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.device = type("Device", (), {"platform": "cpu"})()

    monkeypatch.setattr(robust, "Sim", WrongSim)
    with pytest.raises(robust.RobustContractError, match="backend"):
        robust.build_simulation(4, device="gpu")
