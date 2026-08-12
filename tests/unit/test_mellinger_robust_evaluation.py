"""Contracts for the cf21B_500 robust baseline evaluator."""

from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control.mellinger.research import robust_evaluation as robust

Constructed = tuple[tuple[robust.EpisodeSpec, robust.ReferenceCandidate], ...]
Foundation = dict[str, Any]


@pytest.fixture(scope="module")
def constructed() -> Constructed:
    """Construct the exact reference-only episode registry once."""
    return robust.construct_all_episodes()


@pytest.fixture(scope="module")
def foundation() -> Foundation:
    """Run the complete paired evaluator once for cross-contract assertions."""
    return robust.build_foundation_evidence()


def test_exact_seed_split_class_registry_has_no_test() -> None:
    specs = robust.episode_specs()
    assert len(specs) == 16
    assert len({item.seed for item in specs}) == 16
    assert len({item.episode_id for item in specs}) == 16
    assert {item.split for item in specs} == {robust.Split.TRAIN, robust.Split.VALIDATION}
    assert set(robust.Split) == {robust.Split.TRAIN, robust.Split.VALIDATION}
    for split in robust.Split:
        for motion_class in robust.MotionClass:
            selected = [
                item for item in specs if item.split is split and item.motion_class is motion_class
            ]
            assert [item.seed for item in selected] == list(
                robust.EPISODE_SEEDS[split][motion_class]
            )


def test_fold_in_keys_separate_every_coordinate() -> None:
    first, second = robust.episode_specs()[:2]
    keys = {
        tuple(np.asarray(jax.random.key_data(robust.component_key(spec, attempt, component))))
        for spec, attempt, component in (
            (first, 0, "position"),
            (first, 0, "yaw"),
            (first, 1, "position"),
            (second, 0, "position"),
        )
    }
    assert len(keys) == 4


def test_all_reference_only_attempts_meet_frozen_class_bands(constructed: Constructed) -> None:
    for spec, candidate in constructed:
        contract = robust.CLASS_CONTRACTS[spec.motion_class]
        score = candidate.score_statistics
        parent = candidate.parent_statistics
        assert 0 <= candidate.attempt < robust.MAX_ATTEMPTS
        assert len(candidate.attempts) == candidate.attempt + 1
        assert candidate.attempts[-1]["accepted"] is True
        assert contract.lower_open < score["u_trans"] <= contract.upper_closed
        assert contract.lower_open < score["u_yaw"] <= contract.upper_closed
        assert parent["u_trans"] <= 0.95
        assert parent["u_yaw"] <= 0.95
        assert score["max_abs_yaw_rate_rad_s"]["value"] > 0.0
        assert parent["max_abs_yaw_rad"]["value"] <= contract.max_abs_yaw_rad


def test_parent_is_c3_at_both_ends_and_yaw_is_continuous(constructed: Constructed) -> None:
    for _, candidate in constructed:
        trajectory = candidate.trajectory
        for value in (trajectory.vel, trajectory.acc, candidate.jerk, trajectory.yaw_rate):
            np.testing.assert_allclose(np.asarray(value)[[0, -1]], 0.0, atol=3.0e-5, rtol=0.0)
        assert np.all(np.isfinite(np.asarray(trajectory.yaw)))
        assert np.max(np.abs(np.diff(np.asarray(trajectory.yaw)))) < 0.02


def test_d047_vertical_excitation_metadata_and_exact_unchanged_semantics(
    constructed: Constructed,
) -> None:
    metadata = robust.vertical_excitation_machine_contract()
    assert metadata["axis"] == {"name": "z", "index": 2, "direction": "positive"}
    assert metadata["central_rate"] == {
        "literal_m_s": "0.06",
        "dtype": "float32",
        "float32_value_m_s": float(np.float32(0.06)),
    }
    assert metadata["base_before_envelope"]["time_origin_s"] == 2.0
    assert metadata["envelope"] == {
        "function": "smoothstep7",
        "continuity": "C3",
        "entry_duration_s": 2.0,
        "unit_plateau_duration_s": 4.0,
        "exit_duration_s": 2.0,
        "support_s": 8.0,
        "same_envelope_as_bounded_fourier_parent": True,
    }
    assert metadata["derivative_construction"]["highest_derivative_order"] == 3
    assert metadata["class_independent"] is True
    time = jnp.arange(robust.PARENT_INTERVALS + 1, dtype=jnp.float32) / robust.CONTROL_FREQUENCY_HZ
    envelope = robust._c3_envelope(time)
    rate = jnp.asarray(0.06, dtype=time.dtype)
    base = (
        (rate * (time - 2.0))[:, None],
        jnp.broadcast_to(rate, time.shape)[:, None],
        jnp.zeros_like(time)[:, None],
        jnp.zeros_like(time)[:, None],
    )
    climb = robust.vertical_excitation_components(time)
    for actual, expected in zip(climb, robust._enveloped(base, envelope), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    z_axis = jnp.asarray([0.0, 0.0, 1.0], dtype=time.dtype)
    center = jnp.asarray([0.0, 0.0, 0.75], dtype=time.dtype)
    for spec, candidate in constructed:
        fourier = robust._enveloped(
            robust._fourier_series(
                time, robust.component_key(spec, candidate.attempt, "position"), 3
            ),
            envelope,
        )
        score_indices = robust._interval_indices(spec)
        maximum_speed = float(
            np.max(np.linalg.norm(np.asarray(fourier[1])[score_indices], axis=-1))
        )
        scale = (
            robust.CLASS_CONTRACTS[spec.motion_class].target_usage
            * robust.GLOBAL_LIMITS["speed_m_s"]
            / maximum_speed
        )
        scaled = tuple(value * jnp.asarray(scale, dtype=time.dtype) for value in fourier)
        expected = (
            center + (scaled[0] + climb[0] * z_axis),
            scaled[1] + climb[1] * z_axis,
            scaled[2] + climb[2] * z_axis,
            scaled[3] + climb[3] * z_axis,
        )
        actual = (
            candidate.trajectory.pos,
            candidate.trajectory.vel,
            candidate.trajectory.acc,
            candidate.jerk,
        )
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(observed), np.asarray(reference))


def test_windows_are_true_interior_intervals_with_support_on_both_sides(
    constructed: Constructed,
) -> None:
    for spec, candidate in constructed:
        assert candidate.trajectory.time.shape == (robust.PARENT_INTERVALS + 1,)
        assert spec.score_start in {200, 300, 400}
        assert spec.score_stop == spec.score_start + robust.SCORE_INTERVALS
        assert spec.score_start > 0
        assert spec.score_stop < robust.PARENT_INTERVALS
        assert spec.warmup_s == 2.0 + spec.seed % 3


def test_complete_32_rollout_evidence_passes_nonintegral_gates(foundation: Foundation) -> None:
    report = foundation["report"]
    assert report["status"] == "PASS_ROBUST_FOUNDATION"
    assert set(report["mass_variants"]) == set(robust.MASS_VARIANTS)
    assert sum(len(value["episodes"]) for value in report["mass_variants"].values()) == 32
    for variant in report["mass_variants"].values():
        assert variant["differentiation"]["all_finite"] is True
        assert variant["differentiation"]["optimizer_updates"] == 0
        assert np.all(np.isfinite(variant["differentiation"]["gradient"]))
        for episode in variant["episodes"]:
            gates = episode["technical_gates"]
            assert gates["finite"] is True
            assert gates["ground_or_floor_activation"] is False
            assert gates["zero_thrust_activation"] is False
            clipping = episode["summary"]["clipping"]
            if episode["motion_class"] in {"soft", "nominal"}:
                assert clipping["stage_a_torque"]["count"] == 0
                assert clipping["stage_a_motor"]["count"] == 0
                assert clipping["stage_b_additional"]["count"] == 0
            if episode["motion_class"] == "dynamic":
                assert clipping["stage_b_additional"]["count"] == 0
                assert clipping["stage_a_motor"]["fraction"] <= 0.0025
                assert clipping["stage_a_motor"]["longest_duration_s"] <= 0.02


def test_complete_carry_replay_and_mass_only_diff_pass(foundation: Foundation) -> None:
    report = foundation["report"]
    isolation = report["mass_only_pytree_diff"]
    assert isolation["status"] == "PASS_ONLY_CONTROLS_STATE_PARAMS_MASS_DIFFERS"
    assert len(isolation["differences"]) == 1
    assert "['mass']" in isolation["differences"][0]["path"]
    for variant in report["mass_variants"].values():
        carry = variant["carry_continuity"]
        assert carry["status"] == "PASS_COMPLETE_SIMDATA_CARRY_CONTINUOUS"
        assert [item["boundary_interval"] for item in carry["comparisons"]] == [200, 300, 400]
        assert all(item["all_pytree_leaves_array_equal"] for item in carry["comparisons"])


def test_loss_v1_exact_arithmetic_and_summary_recomputation(foundation: Foundation) -> None:
    report = foundation["report"]
    for variant in report["mass_variants"].values():
        for episode in variant["episodes"]:
            loss = episode["summary"]["loss_v1"]
            assert set(loss["terms"]) == {
                "position",
                "velocity",
                "effort",
                "smoothness",
                "terminal",
                "altitude",
            }
            assert loss["arithmetic_absolute_error"] <= 2.0e-6
    robust.validate_report_summaries(report)
    broken = copy.deepcopy(report)
    broken["mass_variants"]["matched"]["episodes"][0]["summary"]["effort"] += 1.0
    with pytest.raises(robust.RobustContractError, match="summary recomputation"):
        robust.validate_report_summaries(broken)


def test_split_class_loss_and_trajectory_aggregates_recompute_from_episodes(
    foundation: Foundation,
) -> None:
    report = foundation["report"]
    expected_terms = {"position", "velocity", "effort", "smoothness", "terminal", "altitude"}
    for split, classes in report["mass_ablation"]["per_split_class"].items():
        for motion_class, record in classes.items():
            loss = record["loss_v1_term_aggregates"]
            assert set(loss["terms"]) == expected_terms
            assert loss["fields"] == list(robust.LOSS_TERM_FIELDS)
            tracking = record["trajectory_tracking_statistic_aggregates"]["statistics"]
            for variant in robust.MASS_VARIANTS:
                episodes = [
                    episode
                    for episode in report["mass_variants"][variant]["episodes"]
                    if episode["split"] == split and episode["motion_class"] == motion_class
                ]
                assert len(episodes) == 2
                for term in expected_terms:
                    for field in robust.LOSS_TERM_FIELDS:
                        values = [
                            episode["summary"]["loss_v1"]["terms"][term][field]
                            for episode in episodes
                        ]
                        aggregate = loss["terms"][term][variant][field]
                        assert aggregate == {
                            "mean": float(np.mean(values)),
                            "minimum": float(np.min(values)),
                            "maximum": float(np.max(values)),
                        }
                for name in episodes[0]["summary"]["tracking"]:
                    values = [episode["summary"]["tracking"][name] for episode in episodes]
                    assert tracking[name][variant] == {
                        "mean": float(np.mean(values)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
            for term in expected_terms:
                for field in robust.LOSS_TERM_FIELDS:
                    by_variant = loss["terms"][term]
                    assert by_variant["matched_minus_mismatch_mean"][field] == (
                        by_variant["matched"][field]["mean"]
                        - by_variant["repository_mismatch"][field]["mean"]
                    )


def test_d045_retains_contact_and_required_negative_finding(foundation: Foundation) -> None:
    report = foundation["report"]
    finding = report["mass_ablation"]["negative_physical_value_match_finding"]
    assert finding["status"] == "NEGATIVE_PHYSICAL_VALUE_MATCH_FINDING"
    mismatch = finding["variants"]["repository_mismatch"]
    matched = finding["variants"]["matched"]
    assert (
        matched["negative_z_integral_contact_count"] > mismatch["negative_z_integral_contact_count"]
    )
    assert matched["mean_z_rmse_m"] > mismatch["mean_z_rmse_m"]
    assert report["scope"]["decision_d045"].startswith("prospective integral contact")
    for variant in report["mass_variants"].values():
        assert len(variant["episodes"]) == 16
        for episode in variant["episodes"]:
            integral = episode["summary"]["integral"]
            assert len(integral["contact_by_axis"]) == 3
            for axis in integral["contact_by_axis"]:
                assert axis["total_duration_s"] == axis["hit_count"] / robust.CONTROL_FREQUENCY_HZ
                assert all(
                    interval["end_time_s"] > interval["start_time_s"]
                    for interval in axis["combined"]["intervals"]
                )


def test_reserve_clip_interval_and_wrench_contracts(foundation: Foundation) -> None:
    report = foundation["report"]
    for variant in report["mass_variants"].values():
        for episode in variant["episodes"]:
            summary = episode["summary"]
            assert np.isfinite(summary["motor_reserve"]["minimum_normalized_force_reserve"])
            assert np.isfinite(summary["motor_reserve"]["minimum_normalized_speed_reserve"])
            for stage in ("stage_a", "stage_b_additional"):
                axes = summary["wrench"][stage]["by_axis"]
                assert [item["axis"] for item in axes] == ["collective", "roll", "pitch", "yaw"]
                assert all(item["maximum_absolute_distortion"] >= 0.0 for item in axes)


def test_production_reconstruction_obeys_frozen_float32_bound(foundation: Foundation) -> None:
    for variant in foundation["report"]["mass_variants"].values():
        checks = variant["diagnostic_reconstruction"]
        assert all(item["passed"] for item in checks.values())
        assert all(
            item["maximum_absolute_error"] <= item["maximum_allowed_error"]
            for item in checks.values()
        )
    robust.reconstruction_check("within", np.asarray([1.0]), np.asarray([1.0]), 1.0)
    with pytest.raises(robust.RobustContractError, match="Float32 reconstruction"):
        robust.reconstruction_check("outside", np.asarray([0.0]), np.asarray([1.0]), 1.0)


def test_mechanism_audit_separates_measurement_inference_and_transfer(
    foundation: Foundation,
) -> None:
    report = foundation["report"]
    assert set(report["mechanism_audit"]) == {
        "measured",
        "code_path_inference",
        "transfer_boundary",
    }
    for variant in report["mass_variants"].values():
        for episode in variant["episodes"]:
            audit = episode["mechanism_audit"]
            assert set(audit) == {"measured", "code_path_inference", "transfer_boundary"}
            assert set(audit["measured"]) == {"early_transient", "scored_window", "boundary"}
            assert "not experimental causal proof" in audit["code_path_inference"]["status"]
            assert "not a physical-mass estimate" in audit["transfer_boundary"]


def test_benchmark_block_replication_is_bit_identical() -> None:
    for world_count in (4, 16, 32):
        _, _, inputs, raw, metadata = robust.benchmark_block_inputs(world_count)
        assert metadata["replication_array_equal"] is True
        assert metadata["near_limit_included"] is False
        assert inputs.commands.shape == (robust.ROLLOUT_INTERVALS, world_count, 1, 13)
        assert raw.shape == (4,)
        for start in range(0, world_count, 4):
            np.testing.assert_array_equal(
                np.asarray(inputs.commands)[:, start : start + 4],
                np.asarray(inputs.commands)[:, :4],
            )


def test_benchmark_consistency_uses_predeclared_tolerance() -> None:
    records = [
        {"world_count": count, "loss": 0.25, "gradient": [0.1, -0.2, 0.3, -0.4]}
        for count in (4, 16, 32)
    ]
    assert robust.benchmark_consistency_check(records)["status"].startswith("PASS")
    records[-1]["gradient"][0] = 0.2
    with pytest.raises(robust.RobustContractError, match="consistency failed"):
        robust.benchmark_consistency_check(records)


def test_episode_contract_and_manifests_retain_attempts_but_no_test(foundation: Foundation) -> None:
    contract = foundation["episode_contract"]
    assert len(contract["episodes"]) == 16
    assert contract["contract"]["test_constructed_or_loaded"] is False
    assert contract["contract"]["carry_reset_at_window"] is False
    assert (
        contract["contract"]["vertical_excitation"] == robust.vertical_excitation_machine_contract()
    )
    scope = contract["contract"]["duration_and_measurement_scope"]
    assert scope["parent_reference_support"]["duration_s"] == 8.0
    assert scope["continuous_rollout_from_parent_start"]["duration_s"] == 6.0
    assert scope["continuous_rollout_from_parent_start"]["reset_at_scored_window"] is False
    assert scope["scored_window"]["duration_s"] == 2.0
    assert scope["full_six_second_rollout_gates"] == [
        "all_arrays_finite",
        "ground_or_floor_activation",
        "zero_thrust_activation",
    ]
    pin = robust.parent_contract_pin_identity(contract)
    assert pin == {
        "record_count": 16,
        "serialized_bytes": 18_037,
        "sha256": robust.PRE_CORRECTION_PARENT_PIN_SHA256,
    }
    reference_aggregates = contract["per_split_class_reference_trajectory_statistics"]
    assert set(reference_aggregates) == {"train", "validation"}
    assert all(
        set(classes) == {item.value for item in robust.MotionClass}
        for classes in reference_aggregates.values()
    )
    for manifest_name, split in (
        ("train_manifest", "train"),
        ("validation_manifest", "validation"),
    ):
        manifest = foundation[manifest_name]
        assert manifest["split"] == split
        assert manifest["test_manifest"] is None
        assert len(manifest["episodes"]) == 8
        assert all(
            value["accepted_parent_count"] == 2
            for value in manifest["acceptance_by_class"].values()
        )
