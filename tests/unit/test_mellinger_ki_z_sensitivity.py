"""Regression tests for the frozen vertical ``ki_z`` sensitivity pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.jax import mellinger_ki_z_sensitivity as pilot


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    """Execute the small frozen pilot once for all scientific assertions."""
    return pilot.build_evidence()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.unit
def test_cli_requires_explicit_pilot_acknowledgement() -> None:
    with pytest.raises(SystemExit):
        pilot.parse_args(())
    parsed = pilot.parse_args(("--run-ki-z-identifiability-pilot",))
    assert parsed.output_dir == Path("artifacts/day23-ki-z-sensitivity")


@pytest.mark.unit
def test_reference_contract_is_exact_constant_hold() -> None:
    full, reference, commands = pilot.build_reference()
    assert full.time.shape == (501,)
    assert reference.time.shape == (500,)
    assert commands.shape == (500, 1, 1, 13)
    assert float(full.time[0]) == 0.0
    assert float(full.time[-1]) == pytest.approx(5.0)
    np.testing.assert_array_equal(np.asarray(full.pos), np.broadcast_to((0.0, 0.0, 0.75), (501, 3)))
    np.testing.assert_array_equal(np.asarray(full.vel), np.zeros((501, 3)))
    np.testing.assert_array_equal(np.asarray(full.acc), np.zeros((501, 3)))


@pytest.mark.unit
def test_mass_cases_are_only_mass_different_and_defaults_stay_unchanged() -> None:
    _, matched, mismatch, _, _ = pilot.build_simulation()
    assert float(matched.controls.state.params["mass"]) == pytest.approx(0.0319)
    assert float(mismatch.controls.state.params["mass"]) == pytest.approx(0.029)
    assert float(matched.params.mass[0, 0, 0]) == pytest.approx(0.0319)
    assert float(mismatch.params.mass[0, 0, 0]) == pytest.approx(0.0319)
    normalized_matched = pilot.with_controller_mass(matched, 0.0)
    normalized_mismatch = pilot.with_controller_mass(mismatch, 0.0)
    assert all(
        np.array_equal(pilot._numpy_leaf(left), pilot._numpy_leaf(right))
        for left, right in zip(
            pilot.jax.tree.leaves(normalized_matched),
            pilot.jax.tree.leaves(normalized_mismatch),
            strict=True,
        )
    )


@pytest.mark.unit
def test_registry_transform_and_raw_perturbation_are_exact() -> None:
    _, matched, _, _, _ = pilot.build_simulation()
    contract = pilot.stage2_gain_contract(matched)
    assert contract["names"] == ["kp_xy", "kp_z", "kd_xy", "kd_z", "ki_xy", "ki_z"]
    assert contract["index"] == 5
    assert contract["raw_delta"] == 0.25
    assert contract["only_ki_z_changes"]
    assert contract["raw_values"] == pytest.approx(
        {
            "lower": pilot.LOWER_RAW_KI_Z,
            "default": pilot.DEFAULT_RAW_KI_Z,
            "upper": pilot.UPPER_RAW_KI_Z,
        },
        abs=2.0e-7,
    )
    assert contract["physical_values_N_per_m_s"] == pytest.approx(
        {
            "lower": pilot.LOWER_PHYSICAL_KI_Z,
            "default": pilot.DEFAULT_KI_Z,
            "upper": pilot.UPPER_PHYSICAL_KI_Z,
        },
        abs=2.0e-7,
    )
    base = np.asarray(contract["base_raw_vector"])
    for name in ("lower", "upper"):
        changed = np.flatnonzero(np.asarray(contract["raw_vectors"][name]) != base)
        np.testing.assert_array_equal(changed, np.asarray((5,)))


@pytest.mark.unit
def test_evidence_has_six_complete_finite_rollouts(evidence: dict[str, Any]) -> None:
    assert tuple(evidence["cases"]) == pilot.CASE_NAMES
    for case in evidence["cases"].values():
        assert tuple(case["variants"]) == pilot.VARIANT_NAMES
        for variant in case["variants"].values():
            assert len(variant["state_time_s"]) == 500
            assert len(variant["actual_altitude_m"]) == 500
            assert len(variant["vertical_velocity_m_s"]) == 500
            assert len(variant["integral_state_z_m_s"]) == 500
            assert np.asarray(variant["commanded_rotor_rpm"]).shape == (500, 4)
            assert np.asarray(variant["lower_motor_margin_rpm"]).shape == (500, 4)
            assert np.asarray(variant["upper_motor_margin_rpm"]).shape == (500, 4)
            assert np.asarray(variant["nearest_motor_bound_margin_rpm"]).shape == (500, 4)
            assert np.asarray(variant["nearest_motor_bound_margin_normalized"]).shape == (500, 4)
            assert np.all(np.isfinite(np.asarray(variant["actual_altitude_m"])))
            assert np.all(np.isfinite(np.asarray(variant["integral_state_z_m_s"])))


@pytest.mark.unit
def test_loss_v1_terms_and_gradients_are_internally_consistent(evidence: dict[str, Any]) -> None:
    for case in evidence["cases"].values():
        for variant in case["variants"].values():
            weighted_sum = sum(term["weighted"] for term in variant["loss_v1_terms"].values())
            assert weighted_sum == pytest.approx(variant["metrics"]["loss_total"], abs=3.0e-7)
        local = case["local_sensitivity_at_default"]
        expected_fd = (local["losses"]["upper"] - local["losses"]["lower"]) / 0.5
        assert local["central_finite_difference"] == pytest.approx(expected_fd, abs=1.0e-12)
        assert local["ad_fd_consistency_pass"]
        assert np.isfinite(local["jax_ad_d_loss_v1_d_raw_ki_z"])


@pytest.mark.unit
def test_technical_gates_pass_and_recommendation_is_predeclared(evidence: dict[str, Any]) -> None:
    assert evidence["technical_gates"]["pass"]
    for case in evidence["cases"].values():
        for variant in case["variants"].values():
            descriptive = variant["descriptive_metrics"]
            assert descriptive["motor_saturated_sample_count"] == 0
            assert descriptive["max_abs_integral_z_m_s"] <= 0.36
            assert descriptive["minimum_motor_bound_margin_normalized"] >= 0.01
            assert descriptive["zero_thrust_fraction"] == 0.0
            assert descriptive["ground_floor_fraction"] == 0.0
            assert descriptive["nonfinite_fraction"] == 0.0
    failed = [name for name, gate in evidence["scientific_gates"].items() if not gate["pass"]]
    assert evidence["failed_scientific_criteria"] == failed
    expected = pilot.GO_STATUS if not failed else pilot.NO_GO_STATUS
    assert evidence["recommendation"] == expected


@pytest.mark.unit
def test_package_is_json_driven_and_byte_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: dict[str, Any]
) -> None:
    monkeypatch.setattr(pilot, "build_evidence", lambda: evidence)
    monkeypatch.setattr(pilot, "_generator_commit", lambda: "a" * 40)
    first = tmp_path / "first"
    second = tmp_path / "second"
    pilot.emit_package(first)
    pilot.emit_package(second)
    assert sorted(path.name for path in first.iterdir()) == sorted(
        (*pilot.OUTPUT_NAMES, "SHA256SUMS")
    )
    for name in (*pilot.OUTPUT_NAMES, "SHA256SUMS"):
        assert _sha256(first / name) == _sha256(second / name)
    persisted = json.loads((first / "ki_z_sensitivity.json").read_text())
    assert persisted == evidence
    checksum_lines = (first / "SHA256SUMS").read_text().splitlines()
    assert len(checksum_lines) == 4
    for line in checksum_lines:
        digest, name = line.split("  ", maxsplit=1)
        assert digest == _sha256(first / name)
    assert (first / "ki_z_sensitivity.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.unit
def test_nonempty_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("protected\n")
    with pytest.raises(pilot.PilotContractError, match="refusing to overwrite"):
        pilot.emit_package(output)
    assert (output / "keep.txt").read_text() == "protected\n"


@pytest.mark.unit
def test_technical_withhold_creates_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "withheld"

    def withhold() -> dict[str, Any]:
        raise pilot.PilotWithheldError("synthetic technical gate")

    monkeypatch.setattr(pilot, "build_evidence", withhold)
    with pytest.raises(pilot.PilotWithheldError, match="synthetic technical gate"):
        pilot.emit_package(output)
    assert not output.exists()
