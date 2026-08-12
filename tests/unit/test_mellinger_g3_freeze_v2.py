"""Frozen contracts for the prospective cf21B_500 G3.4 eligibility-audit package."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from crazyflow.control.mellinger.research import g3_freeze_v2 as g3
from examples.jax import mellinger_g3_freeze_v2 as cli


@pytest.fixture(scope="module")
def constructed() -> tuple[tuple[g3.EpisodeSpec, g3.ReferenceCandidate], ...]:
    return g3.construct_all_episodes()


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    return g3.build_freeze_evidence()


@pytest.fixture(scope="module")
def diagnosis() -> dict[str, Any]:
    return g3.build_g3_003_diagnosis()


@pytest.fixture(scope="module")
def smoke() -> dict[str, Any]:
    return g3.build_g3_004_smoke()


def _synthetic_effect_outputs(
    specs: tuple[g3.EpisodeSpec, ...], *, nonfinite_world: int | None = None
) -> dict[str, Any]:
    intervals = g3.ROLLOUT_INTERVALS
    substeps = g3.STEPS_PER_CONTROL
    worlds = len(specs)
    vector = np.zeros((intervals, worlds, 1, 3), dtype=np.float32)
    scalar = np.zeros((intervals, worlds, 1), dtype=np.float32)
    trace = g3.Trajectory(
        time=scalar,
        pos=vector,
        vel=vector.copy(),
        acc=vector.copy(),
        yaw=scalar.copy(),
        yaw_rate=scalar.copy(),
    )

    def diagnostic_array(channels: int, *, value: float = 0.0) -> np.ndarray:
        return np.full((intervals, substeps, worlds, 1, channels), value, dtype=np.float32)

    def diagnostic_mask(channels: int) -> np.ndarray:
        return np.zeros((intervals, substeps, worlds, 1, channels), dtype=bool)

    diagnostic = {
        "stage_a": {
            "collective_force_request_n": np.ones(
                (intervals, substeps, worlds, 1), dtype=np.float32
            ),
            "motor_pwm_any_clip_mask": diagnostic_mask(4),
            "motor_pwm_lower_clip_mask": diagnostic_mask(4),
            "motor_pwm_upper_clip_mask": diagnostic_mask(4),
            "torque_any_clip_mask": diagnostic_mask(3),
            "torque_lower_clip_mask": diagnostic_mask(3),
            "torque_upper_clip_mask": diagnostic_mask(3),
            "force_lower_reserve_normalized": diagnostic_array(4, value=1.0),
            "force_upper_reserve_normalized": diagnostic_array(4, value=1.0),
            "wrench_distortion_absolute": diagnostic_array(4),
        },
        "stage_b": {
            "motor_force_any_clip_mask": diagnostic_mask(4),
            "motor_force_lower_clip_mask": diagnostic_mask(4),
            "motor_force_upper_clip_mask": diagnostic_mask(4),
            "wrench_distortion_absolute": diagnostic_array(4),
        },
        "nonfinite_probe": diagnostic_array(1),
    }
    if nonfinite_world is not None:
        diagnostic["nonfinite_probe"][17, 2, nonfinite_world, 0, 0] = np.nan
    return {"trace": trace, "diagnostic": diagnostic, "pos_err_i": vector.copy()}


def _classification_diagnostic(
    *,
    withheld: bool,
    objective_valid: bool = True,
    diagnostic_mismatch_count: int = 0,
    diagnostic_false_zero: bool = False,
) -> dict[str, Any]:
    records = [{"reason": g3.NONFINITE_NEAR_LIMIT_EFFECT_REASON}] if withheld else []
    return {
        "feature_tolerance_mismatch_count": diagnostic_mismatch_count,
        "feature_tolerance_mismatch_strata": int(diagnostic_mismatch_count > 0),
        "feature_significant_sign_mismatch": False,
        "feature_false_zero": diagnostic_false_zero,
        "objective_gradient_contract": {
            "eligible": objective_valid,
            "withholding_reason": (
                None if objective_valid else g3.WITHHELD_OBJECTIVE_GRADIENT_INVALID
            ),
            "finite_failures": (
                []
                if objective_valid
                else [{"gate": "SYNTHETIC_FINITE_OBJECTIVE_REVERSE_FD_FAILURE"}]
            ),
        },
        "technical_robustness": {
            "eligible": not withheld,
            "reason": g3.NONFINITE_NEAR_LIMIT_EFFECT_REASON if withheld else None,
            "nonfinite_effect_rollouts": records,
        },
    }


def test_mass_thrust_relaxation_leaf_and_default_parity() -> None:
    result = g3.focused_default_parity()
    assert result["status"] == "PASS_FOCUSED_DEFAULT_PARITY"
    assert result["parents"] == list(g3.FOCUSED_PARENT_IDS)
    assert result["leaf_isolation"]["status"] == "PASS_ONLY_MASS_THRUST_DTYPE_RELAXED"
    difference = result["leaf_isolation"]["difference"]
    assert "['mass_thrust']" in difference["path"]
    assert difference["canonical_dtype"].startswith("int")
    assert difference["canonical_value"] == g3.CANONICAL_MASS_THRUST
    assert difference["relaxed_dtype"] == "float32"
    assert difference["relaxed_value"] == float(np.float32(g3.CANONICAL_MASS_THRUST))
    assert result["direct_state_controller"] == {
        "status": "PASS_DIRECT_STATE_CONTROLLER_DEFAULT_PARITY",
        "paths": ["command_rpyt", "pos_err_i"],
        "array_equal": True,
    }
    assert result["array_equal"] is True
    assert result["optimizer_initialization_count"] == 0
    assert result["optimizer_update_count"] == 0


def test_mass_thrust_focused_continuous_ad_fd_gate(diagnosis: dict[str, Any]) -> None:
    assert diagnosis["classification"] == "FORWARD_ONLY"
    assert [item["episode_id"] for item in diagnosis["loss_only"]] == list(g3.FOCUSED_PARENT_IDS)
    assert all(not item["jvp_all_rows_pass_fd"] for item in diagnosis["loss_only"])
    assert all(item["reverse_all_rows_pass_fd"] for item in diagnosis["loss_only"])


def test_mass_thrust_discrete_integer_response_is_separate() -> None:
    result = g3.discrete_integer_response()
    assert result["status"] == "DISCRETE_INTEGER_RESPONSE"
    assert result["category"] == "DISCRETE_DIAGNOSTIC_ONLY"
    assert result["continuous_ad_validation"] is False
    assert [item["theta_magnitude"] for item in result["records"]] == [0.01, 0.05]
    assert not ({"ad", "gradient", "go_label"} & result.keys())
    for item in result["records"]:
        assert item["q_minus"] < g3.CANONICAL_MASS_THRUST < item["q_plus"]
        assert len(item["symmetric_quotient_per_integer_unit"]) == 2
        assert len(item["theta_normalized_secant"]) == 2
    assert result["optimizer_update_count"] == 0


def test_g3_004_all_seven_parameter_reverse_smoke(smoke: dict[str, Any]) -> None:
    assert smoke["status"] == "PASS_ALL_SEVEN_PARAMETER_REVERSE_SMOKE"
    assert smoke["parameter_order"] == list(g3.PARAMETER_NAMES)
    assert smoke["parent_order"] == list(g3.FOCUSED_PARENT_IDS)
    assert smoke["loss_output_order"] == list(g3.LOSS_ONLY_NAMES)
    assert smoke["raw_recorded_before_assertions"] is True
    assert smoke["forward_mode_executed"] is False
    assert smoke["scalar_reverse_operator"] == "jax.value_and_grad(loss_total)"
    assert smoke["vector_reverse_operator"] == ("jax.jacrev(loss_v1_total_plus_six_contributions)")
    assert set(smoke["parameters"]) == set(g3.PARAMETER_NAMES)
    for parameter, record in smoke["parameters"].items():
        assert record["parameter"] == parameter
        assert record["isolation"]["status"] == "PASS_EXACT_PARAMETER_ISOLATION"
        assert record["theta_zero_default_parity"]["status"] == ("PASS_THETA_ZERO_DEFAULT_PARITY")
        assert [parent["parent_id"] for parent in record["parents"]] == list(g3.FOCUSED_PARENT_IDS)
        for parent in record["parents"]:
            assert parent["all_primal_values_finite"] is True
            assert parent["scalar_reverse_gate"]["passed"] is True
            assert [item["name"] for item in parent["loss_vector"]] == list(g3.LOSS_ONLY_NAMES)
            assert all(item["reverse_vs_fd"]["passed"] for item in parent["loss_vector"])
            assert all(item["false_zero"] is False for item in parent["loss_vector"])
    assert smoke["optimizer_initialization_count"] == 0
    assert smoke["optimizer_update_count"] == 0


def test_exact_24_parent_split_class_profile_registry_has_no_test() -> None:
    specs = g3.episode_specs()
    assert len(specs) == 24
    assert len({item.seed for item in specs}) == 24
    assert len({item.episode_id for item in specs}) == 24
    assert {item.split for item in specs} == {g3.Split.TRAIN, g3.Split.VALIDATION}
    assert set(g3.Profile) == {g3.Profile.HOLD, g3.Profile.CLIMB, g3.Profile.DESCENT}
    for split in g3.Split:
        for motion_class in g3.MotionClass:
            selected = [
                item for item in specs if item.split is split and item.motion_class is motion_class
            ]
            assert [item.profile for item in selected] == list(g3.Profile)
            assert [item.seed for item in selected] == list(g3.EPISODE_SEEDS[split][motion_class])
    assert all("test" not in item.episode_id for item in specs)
    focused = [item.episode_id for item in specs if item.episode_id in g3.FOCUSED_PARENT_IDS]
    assert tuple(focused) == g3.FOCUSED_PARENT_IDS


def test_six_batches_of_four_preserve_exact_24_parent_order_and_split() -> None:
    constructed = tuple((spec, None) for spec in g3.episode_specs())
    batches = g3._execution_batch_partition(g3.G3_004_EXECUTION_MODE, constructed)
    expected_ids = [spec.episode_id for spec in g3.episode_specs()]
    assert [(label, len(items)) for label, items in batches] == [
        (label, stop - start) for label, start, stop, _split in g3.G3_004_BATCH_SLICES
    ]
    assert [spec.episode_id for _, items in batches for spec, _ in items] == expected_ids
    assert [len(items) for _, items in batches] == list(g3.G3_004_BATCH_PARENT_COUNTS)
    assert all(
        spec.split.value == expected_split
        for (_, items), (_, _start, _stop, expected_split) in zip(
            batches, g3.G3_004_BATCH_SLICES, strict=True
        )
        for spec, _ in items
    )


def test_six_batch_aggregation_contract_is_ordered_global_and_count_explicit() -> None:
    ids = tuple(spec.episode_id for spec in g3.episode_specs())
    batches = tuple(
        {
            "batch_label": label,
            "parent_ids": list(ids[start:stop]),
            "parent_count": stop - start,
            "split_order": [split],
            "deadline_seconds": 900,
            "deadline_passed": True,
            "feature_names": ("loss_total", "position_rmse_z"),
        }
        for label, start, stop, split in g3.G3_004_BATCH_SLICES
    )
    feature_names, contract = g3._batch_aggregation_contract(
        g3.G3_004_EXECUTION_MODE, batches, ids, 900, 2400
    )
    assert feature_names == ("loss_total", "position_rmse_z")
    assert contract["schema_version"] == g3.G3_004_BATCH_EXECUTION_SCHEMA
    assert contract["selection_decision"] == "D-073"
    assert contract["runtime_fallback_used"] is False
    assert contract["runtime_fallback_reason"] is None
    assert contract["selected_by_host_memory"] is False
    assert [item["parent_count"] for item in contract["batches"]] == [4, 4, 4, 4, 4, 4]
    assert contract["aggregation"] == {
        "parent_order_equal_to_frozen_24_registry": True,
        "feature_inventory_identical_across_batches": True,
        "concatenation_axis": "world/parent axis 0",
        "global_gates_applied_only_after_24-parent_concatenation": True,
        "split_scalar_aggregation": (
            "sum(batch_mean_i * contributing_parent_count_i) / sum(contributing_parent_count_i)"
        ),
        "feasible_parent_counts_per_split_batch": [4, 4, 1],
        "whole_split_statistic_denominator": 9,
    }
    assert contract["g4_batch_freeze_status"] == "WITHHELD_PENDING_REBENCHMARK"


def test_count_weighted_scalar_aggregation_matches_unequal_whole_split_mean() -> None:
    values = (1.0, 3.0, 11.0)
    counts = (4, 4, 1)
    records = tuple(
        {"contributing_parent_count": count, "loss_total_scalar_value": value}
        for value, count in zip(values, counts, strict=True)
    )
    combined = g3._combine_count_weighted_scalar_aggregates(records, "loss_total_scalar_value", 9)
    direct_whole_split = float(np.mean([1.0] * 4 + [3.0] * 4 + [11.0]))
    forbidden_mean_of_means = float(np.mean(values))
    assert combined == direct_whole_split
    assert combined != forbidden_mean_of_means


def test_objective_01_exact_7_136_role_partition_and_mismatch_disclosure() -> None:
    specs = g3.episode_specs()
    diagnostic_names = tuple(
        f"mechanical_diagnostic_{index:03d}" for index in range(g3.G3_004_DIAGNOSTIC_FEATURE_COUNT)
    )
    feature_names = g3.OPTIMIZER_OBJECTIVE_NAMES + diagnostic_names
    reverse = np.zeros((24, g3.G3_004_FEATURE_COUNT), dtype=np.float64)
    fd = reverse.copy()
    fd[0, 0] = 0.4
    reverse[1, 1] = 0.2
    fd[1, 1] = -0.2
    fd[2, len(g3.OPTIMIZER_OBJECTIVE_NAMES)] = 0.1
    payload = g3._output_role_comparison_payload(reverse, fd, feature_names, specs)
    inventory = payload["inventory"]
    assert inventory["optimizer_objective"] == {
        "role": g3.OPTIMIZER_OBJECTIVE_ROLE,
        "feature_count": 7,
        "feature_names": list(g3.OPTIMIZER_OBJECTIVE_NAMES),
        "loss_contract": "Loss-v1 total and exact weighted scored-window decomposition",
    }
    assert inventory["diagnostic"]["feature_count"] == 136
    assert inventory["diagnostic"]["optimizer_gradient_claim"] is False
    assert inventory["partition_complete"] is True
    objective = payload["optimizer_objective"]
    assert objective["mismatch_count"] == 2
    assert objective["false_zero_count"] == 1
    assert objective["significant_sign_mismatch_count"] == 1
    assert [item["feature_name"] for item in objective["mismatch_records"]] == [
        "loss_total",
        "loss_contribution_position",
    ]
    diagnostic = payload["diagnostic"]
    assert diagnostic["role"] == g3.DIAGNOSTIC_AD_ROLE
    assert diagnostic["mismatch_count"] == 1
    assert diagnostic["false_zero_count"] == 1
    assert diagnostic["optimizer_gradient_claim"] is False
    assert diagnostic["finite_mismatch_effect_on_objective_validity"] == "NONE"
    record = diagnostic["mismatch_records"][0]
    assert record["parent_id"] == "train-soft-descent-8301003"
    assert record["feature_name"] == "mechanical_diagnostic_000"
    assert record["output_role"] == g3.DIAGNOSTIC_AD_ROLE
    assert record["causal_claim"] is None


def test_objective_01_frozen_gate_uses_real_comparisons_and_stratum_allowance() -> None:
    specs = g3.episode_specs()
    feature_names = g3.OPTIMIZER_OBJECTIVE_NAMES + tuple(
        f"mechanical_diagnostic_{index:03d}" for index in range(g3.G3_004_DIAGNOSTIC_FEATURE_COUNT)
    )

    def comparison(cells: tuple[tuple[int, int, float, float], ...]) -> dict[str, Any]:
        reverse = np.zeros((24, g3.G3_004_FEATURE_COUNT), dtype=np.float64)
        fd = reverse.copy()
        for row, column, reverse_value, fd_value in cells:
            reverse[row, column] = reverse_value
            fd[row, column] = fd_value
        return g3._output_role_comparison_payload(reverse, fd, feature_names, specs)[
            "optimizer_objective"
        ]

    one_allowed_stratum = comparison(((0, 0, 1.0, 1.06),))
    assert one_allowed_stratum["mismatch_count"] == 1
    assert one_allowed_stratum["mismatch_strata"] == 1
    assert one_allowed_stratum["significant_sign_mismatch_count"] == 0
    assert one_allowed_stratum["false_zero_count"] == 0
    allowed_contract = g3._objective_gradient_contract_payload(one_allowed_stratum, 1)
    assert allowed_contract["eligible"] is True
    assert allowed_contract["withholding_reason"] is None
    assert allowed_contract["finite_failures"] == []

    beyond_allowance = comparison(((0, 0, 1.0, 1.06), (1, 0, 1.0, 1.06)))
    sign_mismatch = comparison(((0, 0, 0.01, -0.01),))
    false_zero = comparison(((0, 0, 0.0, 0.01),))
    invalid_sections = (beyond_allowance, sign_mismatch, false_zero)
    for section in invalid_sections:
        contract = g3._objective_gradient_contract_payload(section, 1)
        assert contract["eligible"] is False
        assert contract["withholding_reason"] == g3.WITHHELD_OBJECTIVE_GRADIENT_INVALID
        assert len(contract["finite_failures"]) == 1
        failure = contract["finite_failures"][0]
        assert failure["gate"] == "PER_PARENT_OBJECTIVE_REVERSE_FD"
        assert failure["finite"] is True
        assert failure["comparison_disclosure"] == section
        assert section["mismatch_records"]
        assert all(
            np.isfinite(record[key])
            for record in section["mismatch_records"]
            for key in (
                "reverse",
                "fd",
                "absolute_reverse",
                "absolute_fd",
                "absolute_error",
                "tolerance",
            )
        )
    assert beyond_allowance["mismatch_strata"] == 2
    assert sign_mismatch["significant_sign_mismatch_count"] == 1
    assert false_zero["false_zero_count"] == 1

    matrix = np.ones((24, 1), dtype=np.float64)
    allowed_diagnostic = _classification_diagnostic(withheld=False)
    allowed_diagnostic["objective_gradient_contract"] = allowed_contract
    allowed_category = g3._single_parameter_classification(
        "mass", matrix, ("loss_total",), specs, allowed_diagnostic
    )
    assert allowed_category["category"] == "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION"
    for section in invalid_sections:
        invalid_diagnostic = _classification_diagnostic(withheld=False)
        invalid_diagnostic["objective_gradient_contract"] = g3._objective_gradient_contract_payload(
            section, 1
        )
        withheld = g3._single_parameter_classification(
            "mass", matrix, ("loss_total",), specs, invalid_diagnostic
        )
        assert withheld["category"] == g3.WITHHELD_PARAMETER_CATEGORY
        assert withheld["withholding_reasons"] == [g3.WITHHELD_OBJECTIVE_GRADIENT_INVALID]


def test_objective_01_objective_failure_withholds_but_diagnostic_mismatch_does_not() -> None:
    specs = g3.episode_specs()
    feature_names = ("loss_total",)
    matrix = np.ones((24, 1), dtype=np.float64)
    objective_withheld = g3._single_parameter_classification(
        "mass",
        matrix,
        feature_names,
        specs,
        _classification_diagnostic(withheld=False, objective_valid=False),
    )
    assert objective_withheld["category"] == g3.WITHHELD_PARAMETER_CATEGORY
    assert objective_withheld["withholding_reasons"] == [g3.WITHHELD_OBJECTIVE_GRADIENT_INVALID]
    diagnostic_only_mismatch = g3._single_parameter_classification(
        "mass",
        matrix,
        feature_names,
        specs,
        _classification_diagnostic(
            withheld=False, diagnostic_mismatch_count=12, diagnostic_false_zero=True
        ),
    )
    assert diagnostic_only_mismatch["category"] == "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION"
    assert diagnostic_only_mismatch["withholding_reasons"] == []
    assert diagnostic_only_mismatch["diagnostic_ad_evidence"] == {
        "role": g3.DIAGNOSTIC_AD_ROLE,
        "optimizer_gradient_claim": False,
        "tolerance_mismatch_count": 12,
        "tolerance_mismatch_strata": 1,
        "significant_sign_mismatch": False,
        "false_zero": True,
    }


def test_objective_01_multiple_withholding_reasons_are_stable_and_cumulative() -> None:
    specs = g3.episode_specs()
    matrix = np.ones((24, 1), dtype=np.float64)
    category = g3._single_parameter_classification(
        "kd_z",
        matrix,
        ("loss_total",),
        specs,
        _classification_diagnostic(withheld=True, objective_valid=False),
    )
    assert category["category"] == g3.WITHHELD_PARAMETER_CATEGORY
    assert category["withholding_reasons"] == [
        g3.WITHHELD_OBJECTIVE_GRADIENT_INVALID,
        g3.WITHHELD_TECHNICAL_ROBUSTNESS,
    ]
    matrices = {name: matrix.copy() for name in g3.PARAMETER_NAMES}
    singles = {
        name: (
            category
            if name == "kd_z"
            else g3._single_parameter_classification(
                name,
                matrices[name],
                ("loss_total",),
                specs,
                _classification_diagnostic(withheld=False),
            )
        )
        for name in g3.PARAMETER_NAMES
    }
    group = g3._group_conditioning(("kp_z", "kd_z"), matrices, specs, singles)
    assert group["category"] == "DIAGNOSTIC_ONLY"
    assert group["excluded_withholding_reasons"]["kd_z"] == category["withholding_reasons"]
    proposal = g3._freeze_proposal_payload(
        {"single_parameters": singles}, matrices, specs, g3.G3_004_EXECUTION_MODE
    )
    disposition = proposal["parameter_disposition"]["kd_z"]
    assert disposition["selected"] is False
    assert disposition["withholding_reasons"] == category["withholding_reasons"]
    assert disposition["exclusion_reasons"] == category["withholding_reasons"]


def test_nonfinite_01_finite_parent_adapter_matches_original_batch_gate() -> None:
    specs = g3.episode_specs()[:2]
    outputs = _synthetic_effect_outputs(specs)
    limits = np.ones(3, dtype=np.float32)
    original_gates = g3.robust._full_rollout_gate(specs, outputs, "finite", limits)
    original_details = g3._technical_detail_records(specs, outputs)
    adapted = g3._effect_technical_parent_adapter(
        "train-0-4",
        specs,
        outputs,
        "kp_xy/train-0-4/minus_0p05",
        limits,
        "kp_xy",
        "minus_0p05",
        -g3.EFFECT_STEP,
        g3.PARAMETER_SPECS["kp_xy"].default * np.exp(-g3.EFFECT_STEP),
    )
    assert adapted["gates"] == original_gates
    assert adapted["details"] == original_details
    assert adapted["nonfinite_effect_rollouts"] == []


def test_nonfinite_01_near_limit_effect_nonfinite_metadata_is_stable_and_withholds() -> None:
    spec = next(
        item for item in g3.episode_specs() if item.motion_class is g3.MotionClass.NEAR_LIMIT
    )
    outputs = _synthetic_effect_outputs((spec,), nonfinite_world=0)
    physical_value = g3.PARAMETER_SPECS["kd_z"].default * np.exp(-g3.EFFECT_STEP)
    result = g3._effect_technical_parent_adapter(
        "train-8-12",
        (spec,),
        outputs,
        "kd_z/train-8-12/minus_0p05",
        np.ones(3, dtype=np.float32),
        "kd_z",
        "minus_0p05",
        -g3.EFFECT_STEP,
        physical_value,
    )
    assert len(result["nonfinite_effect_rollouts"]) == 1
    record = result["nonfinite_effect_rollouts"][0]
    assert record["physical_parameter_value"] == physical_value
    assert record["technical_robustness_eligible"] is False
    assert record["reason"] == g3.NONFINITE_NEAR_LIMIT_EFFECT_REASON
    assert record["causal_claim"] is None
    assert record["individual_operation_claim"] is None
    assert [item["path"] for item in record["affected_leaves"]] == ["diagnostic['nonfinite_probe']"]
    leaf = record["affected_leaves"][0]
    assert leaf["nonfinite_count"] == 1
    assert leaf["first_nonfinite_full_array_index"] == [17, 2, 0, 0, 0]
    assert leaf["scored_window_relation"] == "BEFORE"
    assert leaf["physical_time_seconds"] == 87 / g3.SIMULATION_FREQUENCY_HZ
    assert leaf["physical_time_mapping_status"] == ("PROVEN_BY_EXISTING_ROLLOUT_AXIS_CONTRACT")
    assert result["gates"][0]["finite"] is False
    assert g3.canonical_json_bytes(result) == g3.canonical_json_bytes(copy.deepcopy(result))
    matrix = np.ones((24, 1), dtype=np.float64)
    category = g3._single_parameter_classification(
        "kd_z",
        matrix,
        ("loss_total",),
        g3.episode_specs(),
        _classification_diagnostic(withheld=True),
    )
    assert category["category"] == g3.WITHHELD_PARAMETER_CATEGORY
    assert category["withholding_reasons"] == [g3.WITHHELD_TECHNICAL_ROBUSTNESS]


def test_nonfinite_01_feasible_effect_nonfinite_still_raises() -> None:
    spec = next(item for item in g3.episode_specs() if item.motion_class is g3.MotionClass.SOFT)
    outputs = _synthetic_effect_outputs((spec,), nonfinite_world=0)
    with pytest.raises(g3.robust.RobustContractError, match="nonfinite rollout data"):
        g3._effect_technical_parent_adapter(
            "train-0-4",
            (spec,),
            outputs,
            "kp_xy/train-0-4/plus_0p05",
            np.ones(3, dtype=np.float32),
            "kp_xy",
            "plus_0p05",
            g3.EFFECT_STEP,
            g3.PARAMETER_SPECS["kp_xy"].default * np.exp(g3.EFFECT_STEP),
        )


def test_nonfinite_01_default_feature_reverse_and_fd_nonfinite_still_raise() -> None:
    finite = np.ones((2, 2), dtype=np.float64)
    for context in ("baseline", "effect_plus", "reverse", "fd"):
        arrays = {name: finite.copy() for name in ("baseline", "effect_plus", "reverse", "fd")}
        arrays[context][0, 0] = np.nan
        with pytest.raises(g3.G3ContractError, match=rf"/{context} .*nonfinite"):
            g3._require_parameter_diagnostic_arrays_finite("kd_z", "train-8-12", arrays)


def test_nonfinite_01_withheld_parameter_cannot_enter_single_joint_or_freeze_go() -> None:
    specs = g3.episode_specs()
    feature_names = ("loss_total",)
    matrices = {name: np.ones((24, 1), dtype=np.float64) for name in g3.PARAMETER_NAMES}
    singles = {
        name: g3._single_parameter_classification(
            name,
            matrices[name],
            feature_names,
            specs,
            _classification_diagnostic(withheld=name == "kd_z"),
        )
        for name in g3.PARAMETER_NAMES
    }
    group = g3._group_conditioning(("kp_z", "kd_z"), matrices, specs, singles)
    assert group["category"] != "GO_FOR_JOINT_OPTIMIZATION"
    assert group["excluded_withheld_parameters"] == ["kd_z"]
    assert group["excluded_withholding_reasons"] == {"kd_z": [g3.WITHHELD_TECHNICAL_ROBUSTNESS]}
    proposal = g3._freeze_proposal_payload(
        {"single_parameters": singles}, matrices, specs, g3.G3_004_EXECUTION_MODE
    )
    assert "kd_z" not in proposal["selected_parameters"]
    assert proposal["parameter_disposition"]["kd_z"] == {
        "single_parameter_category": g3.WITHHELD_PARAMETER_CATEGORY,
        "selected": False,
        "exclusion_reasons": [g3.WITHHELD_TECHNICAL_ROBUSTNESS],
        "withholding_reasons": [g3.WITHHELD_TECHNICAL_ROBUSTNESS],
        "objective_gradient_eligible": True,
        "objective_gradient_withholding_reason": None,
        "technical_robustness_eligible": False,
        "technical_robustness_reason": g3.NONFINITE_NEAR_LIMIT_EFFECT_REASON,
    }


def test_nonfinite_01_unrelated_parameter_and_batch_contracts_are_unchanged() -> None:
    specs = g3.episode_specs()
    matrix = np.ones((24, 1), dtype=np.float64)
    before = g3._single_parameter_classification(
        "kp_xy", matrix, ("loss_total",), specs, _classification_diagnostic(withheld=False)
    )
    _ = g3._single_parameter_classification(
        "kd_z", matrix, ("loss_total",), specs, _classification_diagnostic(withheld=True)
    )
    _ = g3._single_parameter_classification(
        "mass",
        matrix,
        ("loss_total",),
        specs,
        _classification_diagnostic(withheld=False, objective_valid=False),
    )
    after = g3._single_parameter_classification(
        "kp_xy", matrix, ("loss_total",), specs, _classification_diagnostic(withheld=False)
    )
    assert before == after
    assert len(specs) == 24
    assert g3.G3_004_BATCH_PARENT_COUNTS == (4, 4, 4, 4, 4, 4)
    assert g3.G3_004_FEASIBLE_SPLIT_DENOMINATOR == 9
    assert g3.G3_003_DIAGNOSIS_VERSION == "gr-g3-003-v1"


def test_key_namespaces_and_train_validation_identity_are_disjoint(
    constructed: tuple[tuple[g3.EpisodeSpec, g3.ReferenceCandidate], ...],
) -> None:
    first, second = g3.episode_specs()[:2]
    keys = {
        tuple(np.asarray(jax.random.key_data(g3.component_key(spec, attempt, component))))
        for spec, attempt, component in (
            (first, 0, "position"),
            (first, 0, "yaw"),
            (first, 1, "position"),
            (second, 0, "position"),
        )
    }
    assert len(keys) == 4
    train = [candidate for spec, candidate in constructed if spec.split is g3.Split.TRAIN]
    validation = [candidate for spec, candidate in constructed if spec.split is g3.Split.VALIDATION]
    assert not (
        {item.parent_digest for item in train} & {item.parent_digest for item in validation}
    )
    assert not (
        {item.attempts[-1]["component_key_digests"]["position"]["sha256"] for item in train}
        & {item.attempts[-1]["component_key_digests"]["position"]["sha256"] for item in validation}
    )


def test_reference_contract_arrays_profiles_and_c3_boundaries(
    constructed: tuple[tuple[g3.EpisodeSpec, g3.ReferenceCandidate], ...],
) -> None:
    profile_counts = {(split, profile): 0 for split in g3.Split for profile in g3.Profile}
    for spec, candidate in constructed:
        profile_counts[(spec.split, spec.profile)] += 1
        assert 0 <= candidate.attempt < g3.MAX_ATTEMPTS
        assert candidate.attempts[-1]["accepted"] is True
        assert candidate.trajectory.time.shape == (g3.PARENT_INTERVALS + 1,)
        assert spec.score_start in {200, 300, 400}
        assert spec.score_stop == spec.score_start + g3.SCORE_INTERVALS
        assert spec.score_stop < g3.PARENT_INTERVALS
        for value in (
            candidate.trajectory.vel,
            candidate.trajectory.acc,
            candidate.jerk,
            candidate.trajectory.yaw_rate,
        ):
            np.testing.assert_allclose(np.asarray(value)[[0, -1]], 0.0, atol=3.0e-5, rtol=0.0)
        names = [item["name"] for item in candidate.component_contracts]
        assert names == [
            "bounded_fourier_translation",
            "vertical_profile",
            "fixed_center",
            "bounded_absolute_yaw",
        ]
        assert candidate.component_contracts[1]["rate_m_s"] == g3.PROFILE_RATES_M_S[spec.profile]
        position = np.asarray(candidate.trajectory.pos)[g3._interval_indices(spec)]
        velocity = np.asarray(candidate.trajectory.vel)[g3._interval_indices(spec)]
        yaw = np.asarray(candidate.trajectory.yaw)[g3._interval_indices(spec)]
        yaw_rate = np.asarray(candidate.trajectory.yaw_rate)[g3._interval_indices(spec)]
        assert np.ptp(position[:, 0]) >= 0.01
        assert np.ptp(position[:, 1]) >= 0.01
        assert np.ptp(position[:, 2]) >= 0.005 or np.sqrt(np.mean(velocity[:, 2] ** 2)) >= 0.01
        assert np.ptp(yaw) >= 0.02
        assert np.sqrt(np.mean(yaw_rate**2)) >= 0.01
    assert set(profile_counts.values()) == {4}


def test_all_seven_log_coordinates_are_exactly_isolated_and_bounded() -> None:
    inputs = g3.focused_inputs()
    sim, initial_data = g3.build_initial_data(inputs)
    assert str(sim.mjx_model.impl) == "Impl.JAX"
    assert str(sim.mjx_data.impl) == "Impl.JAX"
    assert g3.mjx_warp.WARP_INSTALLED is False
    relaxed = g3.with_mass_thrust_relaxation(initial_data, 0.0)
    assert np.asarray(g3._state_params(initial_data)["mass_thrust"]).dtype.kind in "iu"
    assert np.asarray(g3._state_params(relaxed)["mass_thrust"]).dtype == np.dtype(np.float32)
    assert float(np.asarray(g3._state_params(relaxed)["mass_thrust"])) == float(
        np.float32(g3.CANONICAL_MASS_THRUST)
    )
    records = {name: g3._parameter_isolation(initial_data, name) for name in g3.PARAMETER_NAMES}
    assert set(records) == set(g3.PARAMETER_NAMES)
    assert all(item["status"] == "PASS_EXACT_PARAMETER_ISOLATION" for item in records.values())
    assert all(
        g3._parameter_default_parity(initial_data, name)["status"]
        == "PASS_THETA_ZERO_DEFAULT_PARITY"
        for name in g3.PARAMETER_NAMES
    )
    assert records["kp_xy"]["changed_indices"] == [[0], [1]]
    assert records["kd_xy"]["changed_indices"] == [[0], [1]]
    assert all(item["physical_value"] > 0.0 for item in records.values())


def test_full_parity_parameter_ad_fd_features_and_categories(evidence: dict[str, Any]) -> None:
    report = evidence["report"]
    assert report["status"] == g3.G3_004_COMPLETION_STATUS
    assert report["output_role_contract"]["optimizer_objective"]["feature_names"] == list(
        g3.OPTIMIZER_OBJECTIVE_NAMES
    )
    assert report["output_role_contract"]["diagnostic"]["feature_count"] == 136
    assert report["scope"]["execution_mode"] == "six-batches-of-four"
    assert report["scope"]["selection_decision"] == "D-073"
    assert report["scope"]["selected_by_host_memory"] is False
    assert report["scope"]["runtime_fallback_used"] is False
    assert report["batch_execution"]["mode"] == "six-batches-of-four"
    assert [item["parent_count"] for item in report["batch_execution"]["batches"]] == [
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    assert report["full_default_parity"]["status"] == "PASS_24_PARENT_DEFAULT_PARITY"
    assert report["full_default_parity"]["parent_count"] == 24
    assert report["full_default_parity"]["array_equal"] is True
    assert set(report["continuous_parameter_diagnostics"]) == set(g3.PARAMETER_NAMES)
    for name, diagnostic in report["continuous_parameter_diagnostics"].items():
        assert diagnostic["isolation"]["status"] == "PASS_EXACT_PARAMETER_ISOLATION"
        assert diagnostic["theta_zero_default_parity"]["status"] == (
            "PASS_THETA_ZERO_DEFAULT_PARITY"
        )
        assert diagnostic["output_role_contract"]["optimizer_objective"]["feature_count"] == 7
        assert diagnostic["output_role_contract"]["diagnostic"]["feature_count"] == 136
        assert diagnostic["objective_gradient_contract"]["role"] == g3.OPTIMIZER_OBJECTIVE_ROLE
        assert diagnostic["diagnostic_ad_disclosure"]["role"] == g3.DIAGNOSTIC_AD_ROLE
        assert diagnostic["diagnostic_ad_disclosure"]["optimizer_gradient_claim"] is False
        assert all(
            record["output_role"] == g3.DIAGNOSTIC_AD_ROLE and record["causal_claim"] is None
            for record in diagnostic["diagnostic_ad_disclosure"]["mismatch_records"]
        )
        arrays = diagnostic["continuous_arrays"]
        assert all(np.all(np.isfinite(value)) for value in arrays.values())
        assert diagnostic["forward_mode_executed"] is False
        assert diagnostic["scalar_reverse_operator"] == "jax.value_and_grad"
        assert diagnostic["vector_reverse_operator"] == "jax.jacrev"
        for split in ("train", "validation"):
            aggregate = diagnostic["split_aggregates"][split]
            assert aggregate["contributing_parent_counts"] == [4, 4, 1]
            assert aggregate["original_statistic_denominator"] == 9
            assert [
                item["contributing_parent_count"] for item in aggregate["batch_contributions"]
            ] == [4, 4, 1]
        frozen_gate = diagnostic["objective_gradient_contract"]["frozen_gate"]
        expected_objective_invalid = (
            frozen_gate["mismatch_strata"] > frozen_gate["allowed_mismatch_strata"]
            or frozen_gate["significant_sign_mismatch"]
            or frozen_gate["false_zero"]
            or frozen_gate["split_congruence_failure_count"] > 0
        )
        assert diagnostic["objective_gradient_contract"]["eligible"] is (
            not expected_objective_invalid
        )
        category = report["single_parameter_categories"][name]
        assert category["objective_gradient_eligible"] is (not expected_objective_invalid)
        assert (
            g3.WITHHELD_OBJECTIVE_GRADIENT_INVALID in category["withholding_reasons"]
        ) is expected_objective_invalid
        assert name in report["single_parameter_categories"]
        assert report["single_parameter_categories"][name]["category"] in {
            "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION",
            "DIAGNOSTIC_ONLY",
            "NO_GO",
            g3.WITHHELD_PARAMETER_CATEGORY,
        }
        if category["category"] == "GO_FOR_SINGLE_PARAMETER_OPTIMIZATION":
            assert category["reverse_fd_objective_evidence"]["passed"] is True
        assert category["diagnostic_ad_evidence"]["role"] == g3.DIAGNOSTIC_AD_ROLE
        assert category["diagnostic_ad_evidence"]["optimizer_gradient_claim"] is False
    assert "kd_z" in report["withheld_technical_robustness_parameters"]
    for name in report["withheld_technical_robustness_parameters"]:
        diagnostic = report["continuous_parameter_diagnostics"][name]
        category = report["single_parameter_categories"][name]
        assert diagnostic["technical_robustness"]["eligible"] is False
        assert diagnostic["technical_robustness"]["reason"] == (
            g3.NONFINITE_NEAR_LIMIT_EFFECT_REASON
        )
        assert diagnostic["technical_robustness"]["nonfinite_effect_rollouts"]
        assert category["category"] == g3.WITHHELD_PARAMETER_CATEGORY
        assert g3.WITHHELD_TECHNICAL_ROBUSTNESS in category["withholding_reasons"]
        assert category["technical_robustness_eligible"] is False
    kd_z_records = report["continuous_parameter_diagnostics"]["kd_z"]["technical_robustness"][
        "nonfinite_effect_rollouts"
    ]
    assert any(
        record["theta_label"] == "minus_0p05"
        and record["batch_label"] == "train-8-12"
        and record["parent_id"] == "train-near_limit-descent-8301303"
        and record["physical_parameter_value"]
        == report["continuous_parameter_diagnostics"]["kd_z"]["effect_physical_values"][
            "minus_0p05"
        ]
        for record in kd_z_records
    )
    conditioning = evidence["sensitivity_conditioning"]
    assert set(conditioning["groups"]) == {
        "mass+mass_thrust",
        "kp_z+kd_z+ki_z+mass+mass_thrust",
        "+".join(g3.PARAMETER_NAMES),
    }
    assert conditioning["discrete_integer_response"]["status"] == "DISCRETE_INTEGER_RESPONSE"
    assert conditioning["discrete_integer_response_included"] is False


def test_overview_figure_emits_nonempty_valid_png(evidence: dict[str, Any]) -> None:
    payload = cli._overview_figure(evidence)
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")


def test_loss_v1_invariance_and_inactive_integral_candidate(evidence: dict[str, Any]) -> None:
    loss = evidence["loss_diagnostics"]
    assert loss["status"] == "PASS_LOSS_V1_UNCHANGED_AND_INACTIVE_CANDIDATE_DIAGNOSED"
    assert loss["loss_v1"]["term_names"] == list(g3.LOSS_TERM_NAMES)
    assert loss["loss_v1"]["optimizer_objective_output_names"] == list(g3.OPTIMIZER_OBJECTIVE_NAMES)
    assert loss["loss_v1"]["optimizer_objective_role"] == g3.OPTIMIZER_OBJECTIVE_ROLE
    assert loss["loss_v1"]["supporting_output_role"] == g3.DIAGNOSTIC_AD_ROLE
    assert loss["loss_v1"]["active_and_numerically_unchanged"] is True
    segment_records = loss["loss_v1"]["episode_segment_records"]
    assert len(segment_records) == 24
    for episode in segment_records:
        assert set(episode["segments"]) == {"warmup_prefix", "scored_window", "full_rollout"}
        for segment in episode["segments"].values():
            assert segment["output_role"] == g3.DIAGNOSTIC_AD_ROLE
            assert segment["optimizer_gradient_claim"] is False
            assert set(segment["terms"]) == set(g3.LOSS_TERM_NAMES)
            assert segment["exact_contribution_sum"] == segment["loss_total"]
            assert set(segment["reverse_gradient_by_parameter"]) == set(g3.PARAMETER_NAMES)
            for gradient in segment["reverse_gradient_by_parameter"].values():
                assert np.isfinite(gradient["loss_total"])
                assert all(
                    np.isfinite(value)
                    for term in gradient["terms"].values()
                    for value in term.values()
                )
    candidate = loss["inactive_candidate"]
    assert candidate["output_role"] == g3.DIAGNOSTIC_AD_ROLE
    assert candidate["optimizer_gradient_claim"] is False
    assert candidate["active_weight"] == 0.0
    assert candidate["added_to_rollout_objective"] is False
    assert np.isfinite(candidate["correlation_with_z_mse"])
    assert np.isfinite(candidate["gradient_cosine_with_z_tracking"])
    assert 1.0e-4 <= candidate["proposed_weight_for_review_only"] <= 0.1
    assert candidate["recommendation"] in {
        "RETAIN_LOSS_V1_PLUS_EXTERNAL_GATES",
        "PROPOSE_VERSIONED_LOSS_V2_FOR_HAUPTLEITUNG_REVIEW",
    }


def test_semantics_manifests_proposal_and_optimizer_null_contract(evidence: dict[str, Any]) -> None:
    semantics = evidence["parameter_semantics"]
    assert semantics["status"] == "PASS_PARAMETER_SEMANTICS_AUDIT"
    assert set(semantics["parameters"]) == set(g3.PARAMETER_NAMES)
    assert semantics["parameters"]["mass_thrust"]["unit"] == "unresolved"
    assert (
        "DISCRETE_INTEGER_RESPONSE"
        in semantics["parameters"]["mass_thrust"]["discrete_integer_response"]
    )
    assert semantics["canonical_defaults_modified"] is False
    assert semantics["shared_controller_modified"] is False
    assert set(semantics["additional_parameter_groups"]) == {
        "ki_xy",
        "tied_vs_split_xy",
        "attitude_and_rate_gains",
        "yaw_gains",
    }
    assert evidence["train_manifest"]["parent_count"] == 12
    assert evidence["validation_manifest"]["parent_count"] == 12
    assert evidence["train_manifest"]["test_manifest"] is None
    assert evidence["validation_manifest"]["test_manifest"] is None
    proposal = evidence["g4_g5_freeze_proposal"]
    assert proposal["g4_executed"] is False
    assert proposal["g5_executed"] is False
    assert proposal["status"] in {
        "PROPOSE_G4_PARAMETER_FREEZE_FOR_HAUPTLEITUNG_REVIEW",
        "WITHHOLD_G4_PARAMETER_FREEZE",
    }
    assert proposal["training_contract"]["development_seed"] == 9104001
    assert proposal["training_contract"]["test"] == "closed"
    assert set(proposal["selected_parameter_contract"]) == set(proposal["selected_parameters"])
    for name, contract in proposal["selected_parameter_contract"].items():
        spec = g3.PARAMETER_SPECS[name]
        assert contract["default"] == spec.default
        assert [contract["lower_bound"], contract["upper_bound"]] == [spec.lower, spec.upper]
        assert contract["coordinate"] == (
            "mass_thrust(theta)=float32(132000)*exp(theta)"
            if name == "mass_thrust"
            else "p(theta)=p_default*exp(theta)"
        )
    schedule = proposal["training_contract"]["batch_schedule"]
    assert schedule["batch_count"] == 9
    assert schedule["worlds_per_batch"] == 32
    assert len(schedule["feasible_stratum_order"]) == 9
    assert len(schedule["batches"]) == 9
    assert all(
        sum(batch["stratum_world_counts"].values()) == 32
        and set(batch["stratum_world_counts"].values()) == {3, 4}
        for batch in schedule["batches"]
    )
    assert set(schedule["total_slots_per_stratum"].values()) == {32}
    secondary = set(proposal["training_contract"]["secondary_metrics"])
    assert {f"loss_v1_{term}" for term in g3.LOSS_TERM_NAMES} <= secondary
    assert {
        "position_rmse_xy",
        "position_rmse_z",
        "velocity_rmse_xy",
        "velocity_rmse_z",
        "mean_signed_z_error",
        "terminal_signed_z_error",
        "integral_rms_max_terminal_by_axis",
        "integral_z_nearness_and_contact",
        "control_effort",
        "control_smoothness",
        "minimum_normalized_motor_reserve",
        "requested_wrench_by_axis_and_class",
        "realized_wrench_by_axis_and_class",
    } <= secondary
    gates = proposal["prospective_g5_gates"]
    assert gates["comparison_baseline"] == "repository-default parameters"
    assert gates["comparison_parent_identity"] == "identical fixed Validation parents"
    assert gates["relative_regression_grouping"] == [
        "feasible_motion_class_mean",
        "feasible_vertical_profile_mean",
    ]
    for payload in (
        evidence["report"],
        semantics,
        evidence["sensitivity_conditioning"],
        evidence["loss_diagnostics"],
        proposal,
        evidence["train_manifest"],
        evidence["validation_manifest"],
    ):
        assert payload["optimizer_initialization_count"] == 0
        assert payload["optimizer_update_count"] == 0


def test_parse_exact_generation_contract() -> None:
    args = cli.parse_args(
        [
            "--generate-package",
            "--generator-commit",
            "1" * 40,
            "--execution-mode",
            "six-batches-of-four",
            "--batch-timeout-seconds",
            "900",
            "--total-timeout-seconds",
            "2400",
            "--output-dir",
            "/tmp/gr-g3-004-repro-test/package",
        ]
    )
    assert args.generate_package is True
    assert args.execution_mode == "six-batches-of-four"
    assert args.fallback_mode is None
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--generate-package",
                "--generator-commit",
                "1" * 40,
                "--execution-mode",
                "six-batches-of-four",
                "--batch-timeout-seconds",
                "899",
                "--total-timeout-seconds",
                "2400",
                "--output-dir",
                "/tmp/gr-g3-004-repro-test/package",
            ]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--generate-package",
                "--generator-commit",
                "1" * 40,
                "--execution-mode",
                "six-batches-of-four",
                "--fallback-mode",
                "split-12-plus-12",
                "--batch-timeout-seconds",
                "900",
                "--total-timeout-seconds",
                "2400",
                "--output-dir",
                "/tmp/gr-g3-004-repro-test/package",
            ]
        )

    smoke_args = cli.parse_args(
        ["--diagnose-g3-004-smoke", "--output-file", "/tmp/gr-g3-004-smoke-parser/scientific.json"]
    )
    assert smoke_args.diagnose_g3_004_smoke is True
    dry_run_args = cli.parse_args(
        [
            "--precommit-dry-run",
            "--source-snapshot-sha256",
            "a" * 64,
            "--output-dir",
            "/tmp/gr-g3-004-precommit-package",
        ]
    )
    assert dry_run_args.precommit_dry_run is True


def test_generator_commit_validation_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "1" * 40

    def fake_git(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return commit
        if arguments == ("show", "-s", "--format=%s", commit):
            return cli.GENERATOR_COMMIT_SUBJECT
        if arguments == ("diff-tree", "--no-commit-id", "--name-only", "-r", commit):
            return "\n".join(sorted(cli.GENERATOR_COMMIT_PATHS))
        if arguments == ("rev-parse", f"{commit}^"):
            return cli.GENERATOR_COMMIT_PARENT
        raise AssertionError(arguments)

    monkeypatch.setattr(cli, "_git", fake_git)
    cli._validate_generator_commit(commit)


@pytest.mark.parametrize(
    ("wrong_field", "wrong_value", "message"),
    (
        ("subject", "wrong subject", "subject"),
        ("paths", "examples/jax/mellinger_g3_freeze_v2.py", "two provenance paths"),
        ("parent", "2" * 40, "exact recovery commit"),
    ),
)
def test_generator_commit_validation_rejects_wrong_provenance_tuple(
    monkeypatch: pytest.MonkeyPatch, wrong_field: str, wrong_value: str, message: str
) -> None:
    commit = "1" * 40
    values = {
        "subject": cli.GENERATOR_COMMIT_SUBJECT,
        "paths": "\n".join(sorted(cli.GENERATOR_COMMIT_PATHS)),
        "parent": cli.GENERATOR_COMMIT_PARENT,
    }
    values[wrong_field] = wrong_value

    def fake_git(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return commit
        if arguments == ("show", "-s", "--format=%s", commit):
            return values["subject"]
        if arguments == ("diff-tree", "--no-commit-id", "--name-only", "-r", commit):
            return values["paths"]
        if arguments == ("rev-parse", f"{commit}^"):
            return values["parent"]
        raise AssertionError(arguments)

    monkeypatch.setattr(cli, "_git", fake_git)
    with pytest.raises(cli.G3CliError, match=message):
        cli._validate_generator_commit(commit)


def test_resource_preflight_keeps_fixed_mode_at_frozen_eight_gib_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    base = {"MemTotal": 32 * gib, "SwapTotal": 0, "SwapFree": 0}
    monkeypatch.setattr(cli, "_memory_snapshot", lambda: base | {"MemAvailable": 30 * gib})
    selected = cli._preflight_resources()
    assert selected["selected_execution_mode"] == "six-batches-of-four"
    assert selected["selection_decision"] == "D-073"
    assert selected["selected_by_host_memory"] is False
    assert selected["runtime_fallback_used"] is False
    assert selected["runtime_fallback_reason"] is None
    monkeypatch.setattr(cli, "_memory_snapshot", lambda: base | {"MemAvailable": 10 * gib})
    still_selected = cli._preflight_resources()
    assert still_selected["selected_execution_mode"] == "six-batches-of-four"
    assert still_selected["selected_by_host_memory"] is False
    monkeypatch.setattr(cli, "_memory_snapshot", lambda: base | {"MemAvailable": 7 * gib})
    with pytest.raises(cli.G3CliError, match="MemAvailable >= 8 GiB"):
        cli._preflight_resources()


def test_fixed_evidence_requests_only_six_batches_of_four(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"batch_execution": {"mode": g3.G3_004_EXECUTION_MODE}}

    monkeypatch.setattr(g3, "build_freeze_evidence", fake_builder)
    evidence = cli._run_fixed_evidence(
        {"selected_execution_mode": g3.G3_004_EXECUTION_MODE}, 900, 2400
    )
    assert evidence["batch_execution"]["mode"] == g3.G3_004_EXECUTION_MODE
    assert calls == [
        {
            "execution_mode": g3.G3_004_EXECUTION_MODE,
            "batch_timeout_seconds": 900,
            "total_timeout_seconds": 2400,
        }
    ]


def test_four_parent_memory_error_remains_a_hard_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_memory_error(_label: str, _constructed: Any) -> dict[str, Any]:
        raise MemoryError("fixture")

    monkeypatch.setattr(g3, "_science_batch_payload", raise_memory_error)
    with pytest.raises(MemoryError, match="fixture"):
        g3._science_batch("train-0-4", (), primary=False, batch_timeout_seconds=900)


def test_fixed_resource_provenance_records_six_deadlined_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Usage:
        ru_maxrss = 1024

    monkeypatch.setattr(cli.resource, "getrusage", lambda _kind: Usage())
    ids = [spec.episode_id for spec in g3.episode_specs()]
    evidence = {
        "batch_execution": {
            "mode": "six-batches-of-four",
            "batches": [
                {"batch_label": label, "parent_count": stop - start, "parent_ids": ids[start:stop]}
                for label, start, stop, _split in g3.G3_004_BATCH_SLICES
            ],
        },
        "runtime_batch_records": [
            {
                "batch_label": label,
                "elapsed_seconds": 100.0 + index,
                "deadline_seconds": 900,
                "deadline_passed": True,
            }
            for index, (label, _start, _stop, _split) in enumerate(g3.G3_004_BATCH_SLICES)
        ],
    }
    resources = cli._postflight_resources({"mem_total_bytes": 32 * 1024**3}, 620.0, evidence)
    assert resources["execution_mode"] == "six-batches-of-four"
    assert resources["schema_version"] == g3.G3_004_BATCH_EXECUTION_SCHEMA
    assert resources["runtime_fallback_used"] is False
    assert resources["runtime_fallback_reason"] is None
    assert resources["batch_labels"] == [item[0] for item in g3.G3_004_BATCH_SLICES]
    assert resources["batch_parent_counts"] == [4, 4, 4, 4, 4, 4]
    assert resources["batch_parent_ids"] == [
        ids[start:stop] for _label, start, stop, _split in g3.G3_004_BATCH_SLICES
    ]
    assert resources["batch_freeze_status"] == "WITHHELD_PENDING_REBENCHMARK"
    stable = cli._stable_resource_record(resources)
    assert stable["selection_decision"] == "D-073"
    assert stable["runtime_fallback_used"] is False
    assert stable["batch_parent_counts"] == [4, 4, 4, 4, 4, 4]


def test_package_writer_exact_inventory_checksums_and_refuses_overwrite(tmp_path: Path) -> None:
    payloads = {name: f"payload:{name}\n".encode() for name in cli.PACKAGE_NAMES}
    output = tmp_path / "package"
    cli.write_package(output, payloads)
    assert sorted(path.name for path in output.iterdir()) == sorted(
        (*cli.PACKAGE_NAMES, "SHA256SUMS")
    )
    observed = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (output / "SHA256SUMS").read_text().splitlines()
    }
    expected = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    assert observed == expected
    with pytest.raises(cli.G3CliError, match="overwrite"):
        cli.write_package(output, payloads)
    broken = copy.copy(payloads)
    broken.pop(cli.PACKAGE_NAMES[-1])
    with pytest.raises(cli.G3CliError, match="incomplete"):
        cli.write_package(tmp_path / "broken", broken)


def test_precommit_dry_run_has_full_schema_no_generator_commit_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = "b" * 64
    output = tmp_path / "precommit-package"
    monkeypatch.setattr(cli, "source_snapshot_sha256", lambda: snapshot)
    cli.run_precommit_dry_run(output, snapshot)
    assert sorted(path.name for path in output.iterdir()) == sorted(
        (*cli.PACKAGE_NAMES, "SHA256SUMS")
    )
    assert len((output / "SHA256SUMS").read_text().splitlines()) == 12
    assert b"generator_commit" not in b"".join(
        (output / name).read_bytes() for name in cli.PACKAGE_NAMES
    )
    with pytest.raises(cli.G3CliError, match="overwrite"):
        cli.run_precommit_dry_run(output, snapshot)


def test_provenance_excludes_branch_path_timestamp_and_records_resource_contract(
    evidence: dict[str, Any],
) -> None:
    ids = [spec.episode_id for spec in g3.episode_specs()]
    resources = {
        "status": "PASS_SIX_BATCHES_OF_FOUR_RESOURCE_ENVELOPE",
        "schema_version": g3.G3_004_BATCH_EXECUTION_SCHEMA,
        "elapsed_seconds": 1.0,
        "maximum_rss_bytes": 1024,
        "maximum_allowed_rss_bytes": 2048,
        "execution_mode": "six-batches-of-four",
        "selection_decision": "D-073",
        "selection_policy": "prospective_fixed_after_observed_local_primary_24_oom",
        "selected_by_host_memory": False,
        "runtime_fallback_used": False,
        "runtime_fallback_reason": None,
        "batch_labels": [item[0] for item in g3.G3_004_BATCH_SLICES],
        "batch_parent_counts": [4, 4, 4, 4, 4, 4],
        "batch_parent_ids": [
            ids[start:stop] for _label, start, stop, _split in g3.G3_004_BATCH_SLICES
        ],
        "batch_freeze_status": "WITHHELD_PENDING_REBENCHMARK",
        "G4": "WITHHELD_PENDING_REBENCHMARK",
    }
    provenance = cli._provenance(evidence, "1" * 40, resources)
    assert provenance["generator_commit"] == "1" * 40
    assert provenance["generator_commit_resolution"] == {
        "subject": cli.GENERATOR_COMMIT_SUBJECT,
        "paths": sorted(cli.GENERATOR_COMMIT_PATHS),
        "stable_across_generator_and_artifact_commit": True,
    }
    assert provenance["integration_target_branch"] == "research/differentiable-mellinger"
    assert provenance["checkout_branch_included"] is False
    assert provenance["absolute_worktree_path_included"] is False
    assert provenance["timestamp_included"] is False
    assert provenance["input_output_inventory"] == {
        "parents": 24,
        "package_payloads": list(cli.PACKAGE_NAMES),
        "checksum_entries": 12,
    }
    assert provenance["execution"]["mode"] == "six-batches-of-four"
    assert provenance["execution"]["selection_decision"] == "D-073"
    assert provenance["execution"]["selection_policy"] == (
        "prospective_fixed_after_observed_local_primary_24_oom"
    )
    assert provenance["execution"]["selected_by_host_memory"] is False
    assert provenance["execution"]["runtime_fallback_used"] is False
    assert provenance["execution"]["runtime_fallback_reason"] is None
    assert provenance["execution"]["batch_labels"] == [item[0] for item in g3.G3_004_BATCH_SLICES]
    assert provenance["execution"]["batch_parent_counts"] == [4, 4, 4, 4, 4, 4]
    assert provenance["execution"]["batch_parent_ids"] == [
        ids[start:stop] for _label, start, stop, _split in g3.G3_004_BATCH_SLICES
    ]
    assert provenance["execution"]["G4"] == "WITHHELD_PENDING_REBENCHMARK"
    assert provenance["optimizer_initialization_count"] == 0
    assert provenance["optimizer_update_count"] == 0


def test_g3_003_loss_only_modes_are_separated_and_reverse_is_runner_congruent(
    diagnosis: dict[str, Any],
) -> None:
    source = diagnosis["productive_reverse_source"]
    assert source == {
        "file": "crazyflow/control/mellinger/research/runner.py",
        "function": "run_training_validation.train_step.objective",
        "line": 255,
        "source": "jax.value_and_grad(objective, has_aux=True)(candidate)",
        "diagnostic_scalar_operator": "jax.value_and_grad(loss_total)",
        "diagnostic_vector_operator": "jax.jacrev(loss_only_vector)",
        "optimizer_library_imported_by_diagnostic": False,
        "optimizer_state_created_by_diagnostic": False,
    }
    for parent in diagnosis["loss_only"]:
        assert parent["feature_matrix_constructed"] is False
        assert parent["loss_output_order"] == list(g3.LOSS_ONLY_NAMES)
        assert [row["output_path"] for row in parent["rows"]] == list(g3.LOSS_ONLY_NAMES)
        assert all(row["primal"]["isfinite"] for row in parent["rows"])
        assert all(row["theta_plus_h"]["isfinite"] for row in parent["rows"])
        assert all(row["theta_minus_h"]["isfinite"] for row in parent["rows"])
        assert all(row["fd"]["isfinite"] for row in parent["rows"])
        assert all(not row["jvp"]["isfinite"] for row in parent["rows"])
        assert all(row["reverse"]["isfinite"] for row in parent["rows"])
        assert parent["vector_reverse_total_matches_scalar"] is True
        assert parent["scalar_value_and_grad"]["fd_comparison"]["passed"] is True
        assert parent["fd_total_significant"] is True


def test_g3_003_feature_controller_and_rollout_modes_are_separated(
    diagnosis: dict[str, Any],
) -> None:
    assert len(diagnosis["feature_matrix"]) == 2
    assert len(diagnosis["controller_pipeline"]) == 2
    assert len(diagnosis["rollout_carry"]) == 2
    for feature in diagnosis["feature_matrix"]:
        assert feature["rows"]
        assert all(
            {"primal", "theta_plus_h", "theta_minus_h", "fd", "jvp", "reverse"} <= row.keys()
            for row in feature["rows"]
        )
    for controller in diagnosis["controller_pipeline"]:
        assert controller["prefixes"]
        for interval in controller["prefixes"].values():
            paths = {row["output_path"] for row in interval["rows"]}
            assert any(path.startswith("direct_state_controller/") for path in paths)
            assert any(path.startswith("state_controller/") for path in paths)
            assert any(path.startswith("attitude_stage_a_diagnostic/") for path in paths)
            assert any(path.startswith("force_torque_stage_b_diagnostic/") for path in paths)
            assert any(path.startswith("dynamics_integration/") for path in paths)
            assert any(path.startswith("resulting_state_carry/") for path in paths)
            assert interval["operation_claim"] is None


def test_g3_003_first_nonfinite_localization_is_mode_specific_and_reproducible(
    diagnosis: dict[str, Any],
) -> None:
    pipelines = {item["episode_id"]: item for item in diagnosis["controller_pipeline"]}
    for parent in diagnosis["rollout_carry"]:
        forward = parent["forward_first_nonfinite"]
        assert forward is not None
        assert forward["ad_mode"] == "forward"
        assert (
            forward["last_fully_finite_prefix_index"] + 1 == forward["first_nonfinite_prefix_index"]
        )
        assert forward["first_nonfinite_time_s"] == (
            forward["first_nonfinite_prefix_index"] / g3.CONTROL_FREQUENCY_HZ
        )
        assert forward["output_path"] in parent["paths"]
        assert forward["derivative"]["isfinite"] is False
        assert forward["localization_status"] in {"LOCALIZED", "NOT_REPRODUCED_IN_SPLIT_INTERVAL"}
        assert (forward["pipeline_output_path"] is None) is (
            forward["localization_status"] == "NOT_REPRODUCED_IN_SPLIT_INTERVAL"
        )
        assert forward["individual_operation_claim"] is None
        assert forward["individual_operation_claim_permitted"] is False
        interval = pipelines[parent["episode_id"]]["prefixes"][
            str(forward["first_nonfinite_prefix_index"])
        ]
        if forward["localization_status"] == "LOCALIZED":
            pipeline_output_path = forward["pipeline_output_path"]
            assert isinstance(pipeline_output_path, str) and pipeline_output_path
            assert forward["narrowest_pipeline_section"] == pipeline_output_path.split("/", 1)[0]
            assert forward["split_interval_derivatives_all_finite"] is False
            matching_rows = [
                row for row in interval["rows"] if row["output_path"] == pipeline_output_path
            ]
            assert len(matching_rows) == 1
            assert matching_rows[0]["jvp"]["isfinite"] is False
            assert forward["localization_claim_boundary"] == "split_interval_pipeline_output"
        elif forward["localization_status"] == "NOT_REPRODUCED_IN_SPLIT_INTERVAL":
            assert forward["pipeline_output_path"] is None
            assert forward["narrowest_pipeline_section"] is None
            assert forward["split_interval_derivatives_all_finite"] is True
            assert all(row["jvp"]["isfinite"] for row in interval["rows"])
            assert forward["localization_claim_boundary"] == "full_rollout_carry_output"
        else:
            raise AssertionError(forward["localization_status"])
        digest_payload = {key: value for key, value in forward.items() if key != "run_digest"}
        assert forward["run_digest"] == g3.sha256_bytes(g3.canonical_json_bytes(digest_payload))
        assert parent["reverse_first_nonfinite"] is None
        assert parent["reverse_boundary"]["last_finite_prefix"] == g3.ROLLOUT_INTERVALS
        assert parent["reverse_boundary"]["first_nonfinite_prefix"] is None
        assert parent["optimizer_congruent_reverse_operator"] == "jax.value_and_grad"
        assert all(parent["reverse_prefix_finite_cache"].values())


def test_g3_003_diagnosis_schema_classification_and_zero_optimizer_contract(
    diagnosis: dict[str, Any],
) -> None:
    assert diagnosis["diagnosis_version"] == "gr-g3-003-v1"
    assert diagnosis["classification"] == "FORWARD_ONLY"
    assert diagnosis["classification_is_diagnostic_only"] is True
    assert diagnosis["optimizer_initialization_count"] == 0
    assert diagnosis["optimizer_update_count"] == 0
    assert diagnosis["source_identity"]["base_commit"] == g3.G3_003_SOURCE_BASE_COMMIT
    assert diagnosis["source_identity"]["input_draft_sha256"] == g3.G3_003_INPUT_DIGESTS
    assert set(diagnosis["source_identity"]["final_draft_sha256"]) == set(g3.G3_003_DRAFT_PATHS)
    assert [item["episode_id"] for item in diagnosis["parents"]] == list(g3.FOCUSED_PARENT_IDS)
    assert diagnosis["resource_contract"] == {
        "maximum_runtime_seconds": 600,
        "maximum_rss_bytes": 24 * 1024**3,
        "focus_parent_count": 2,
        "theta_values": [-0.01, 0.0, 0.01],
    }


def test_g3_003_cli_writes_canonical_diagnosis(
    diagnosis: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parsed = cli.parse_args(
        ["--diagnose-g3-003", "--output-file", "/tmp/gr-g3-003-parser-contract.json"]
    )
    assert parsed.diagnose_g3_003 is True
    output = Path(f"/tmp/gr-g3-003-{tmp_path.name}-diagnosis.json")
    monkeypatch.setattr(g3, "build_g3_003_diagnosis", lambda: diagnosis)
    try:
        cli.write_g3_003_diagnosis(output)
        assert output.read_bytes() == g3.canonical_json_bytes(diagnosis)
    finally:
        output.unlink(missing_ok=True)


def test_g3_004_cli_records_raw_before_validated_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runroot = Path(f"/tmp/gr-g3-004-smoke-{tmp_path.name}")
    runroot.mkdir()
    output = runroot / "scientific.json"
    raw = {
        "status": "RAW_VALUES_RECORDED_BEFORE_ASSERTIONS",
        "optimizer_initialization_count": 0,
        "optimizer_update_count": 0,
    }
    validated = raw | {
        "status": "PASS_ALL_SEVEN_PARAMETER_REVERSE_SMOKE",
        "raw_recorded_before_assertions": True,
    }
    monkeypatch.setattr(g3, "build_g3_004_smoke", lambda *, validate: raw)
    monkeypatch.setattr(g3, "validate_g3_004_smoke", lambda value: validated)
    try:
        cli.write_g3_004_smoke(output)
        assert (runroot / "scientific.raw.json").read_bytes() == g3.canonical_json_bytes(raw)
        assert output.read_bytes() == g3.canonical_json_bytes(validated)
        with pytest.raises(cli.G3CliError, match="overwrite"):
            cli.write_g3_004_smoke(output)
    finally:
        output.unlink(missing_ok=True)
        (runroot / "scientific.raw.json").unlink(missing_ok=True)
        runroot.rmdir()
