from __future__ import annotations

import copy
import functools
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "examples/jax/mellinger_unclipped_allocation_diagnostic.py"


@functools.cache
def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mellinger_unclipped_allocation_diagnostic", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load unclipped allocation diagnostic module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return _load_module().build_diagnostic()


@pytest.mark.unit
def test_cli_requires_explicit_diagnostic_mode() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args([])
    arguments = module.parse_args(["--diagnostic-only-unclipped-allocation"])

    assert arguments.diagnostic_only_unclipped_allocation is True
    assert arguments.output_dir == Path("artifacts/day24-unclipped-allocation")


@pytest.mark.unit
def test_data_audit_uses_only_checksum_verified_derivative_and_source() -> None:
    audit = _load_module().load_data_audit()

    assert audit["status"] == "PASS_DERIVED_DATA_ONLY"
    assert audit["day19_checksum_index_sha256"] == (
        "fe607e7ee7b129e6e9741d2cc62e40625080b408388fcf4d584aca47789b8b17"
    )
    assert audit["seed_count"] == 10
    assert audit["rows_per_seed"] == {"total": 10000, "train": 5000, "validation": 5000, "test": 0}
    assert audit["total_train_validation_rows"] == 100000
    assert audit["total_test_rows"] == 0
    assert audit["per_seed_motor_saturation_fraction_maxima"] == [0.0] * 10
    assert audit["panel_motor_saturation_fraction_maximum"] == 0.0
    assert audit["analyzer_source_crosscheck"] == {
        "rejects_test_rows": True,
        "requires_complete_alternating_train_validation_order": True,
        "per_seed_maximum_over_all_metric_rows": True,
        "panel_maximum_over_per_seed_maxima": True,
        "analyzer_executed": False,
    }


@pytest.mark.unit
def test_actual_replay_is_exact_and_retains_failed_gate(report: dict[str, Any]) -> None:
    module = _load_module()
    default = report["variants"]["default"]
    candidate = report["variants"][module.friday.CANDIDATE_LABEL]

    assert report["status"] == "DIAGNOSTIC_ONLY_SATURATION_PRESENT"
    assert report["normal_acceptance_gate"]["status"] == "FAIL_ZERO_MOTOR_SATURATION_GATE"
    assert default["day22_replay"]["saturated_count"] == 0
    assert candidate["day22_replay"]["saturated_count"] == 11
    assert candidate["day22_replay"]["lower_saturated_count"] == 0
    assert candidate["day22_replay"]["upper_saturated_count"] == 11
    assert report["sampling_contract"]["stored_substep_one_based"] == 5
    for validation in report["validations"]["variants"].values():
        assert validation["all_diagnostic_arrays_finite"] is True
        assert validation["stage_a_preclip_speed_inverse_real_and_finite"] is True
        assert validation["custom_trace_exactly_equals_production_trace"] is True
        assert validation["production_trace_and_tracking_equal_accepted_day22"] is True
        assert validation["stage_a_local_postclip_wrench_matches_production"]["passed"] is True
        assert validation["stage_b_local_command_matches_production"]["passed"] is True


@pytest.mark.unit
def test_complete_stage_arrays_and_exact_seed05_stage_a_attribution(report: dict[str, Any]) -> None:
    module = _load_module()
    candidate = report["variants"][module.friday.CANDIDATE_LABEL]
    default = report["variants"]["default"]
    seed_stage_a = module._summary_arrays(candidate["stage_a_legacy_pwm"])
    seed_stage_b = module._summary_arrays(candidate["stage_b_si_force"])
    default_stage_a = module._summary_arrays(default["stage_a_legacy_pwm"])

    assert all(values.shape[0] == 100 for values in seed_stage_a.values())
    assert all(values.shape[0] == 100 for values in seed_stage_b.values())
    assert int(default_stage_a["motor_pwm_any_clip_mask"].sum()) == 0
    expected = np.zeros((100, 4), dtype=bool)
    expected[48:59, 0] = True
    assert np.array_equal(seed_stage_a["motor_pwm_any_clip_mask"], expected)
    assert np.all(np.isfinite(seed_stage_a["motor_speed_preclip"]))
    assert np.all(seed_stage_a["speed_inverse_discriminant"] >= 0)
    assert candidate["stage_b_si_force"]["summary"]["additional_clip_created"] == bool(
        seed_stage_b["motor_force_any_clip_mask"].any()
    )
    figure = module._figure_contract(report)
    assert figure["window_start_index"] == 44
    assert figure["window_end_index"] == 62
    assert figure["seed05_motor_pwm_preclip_window"].shape == (19, 4)
    assert figure["seed05_motor_pwm_postclip_window"].shape == (19, 4)
    assert np.array_equal(
        figure["signed_wrench_platform_normalized"],
        seed_stage_a["wrench_distortion_signed"]
        / np.asarray(report["numerical_contract"]["wrench_platform_scales"]),
    )
    text = figure["exact_text_panel"]
    assert figure["pwm_upper_limit"] == candidate["stage_a_legacy_pwm"]["limits"]["pwm_max"]
    for key in (
        "requested_pwm",
        "requested_motor_force_N",
        "requested_motor_speed",
        "upper_exceedance_pwm",
        "upper_exceedance_motor_force_N",
        "upper_exceedance_motor_speed",
    ):
        assert str(report["maximum_seed05_motor0_demand"][key]) in text
    assert "Seed 05 Stage A: torque 0/300 | motor 11/400" in text
    assert "Seed 05 Stage B added: motor 0/400" in text
    assert module.STATUS in text
    assert module.GATE_FAILURE_STATUS in text
    assert "NO HARDWARE OR FLIGHT CLAIM" in text


@pytest.mark.unit
def test_exceedance_is_raw_physical_amount_without_detection_threshold(
    report: dict[str, Any],
) -> None:
    module = _load_module()
    candidate = report["variants"][module.friday.CANDIDATE_LABEL]
    arrays = module._summary_arrays(candidate["stage_a_legacy_pwm"])
    limits = candidate["stage_a_legacy_pwm"]["limits"]

    assert np.array_equal(
        arrays["pwm_upper_exceedance"],
        np.maximum(arrays["motor_pwm_preclip"] - limits["pwm_max"], 0),
    )
    reconstructed_force_exceedance = np.maximum(
        arrays["motor_force_preclip_N"] - limits["motor_force_max_N"], 0
    )
    assert np.all(
        np.abs(arrays["force_upper_exceedance_N"] - reconstructed_force_exceedance)
        <= module.RECONSTRUCTION_FACTOR * module.FLOAT32_EPS * 0.12
    )
    assert report["numerical_contract"]["saturation_detection_threshold"] == 1.0e-4
    assert "detection threshold only" in report["numerical_contract"]["threshold_role"]


@pytest.mark.unit
def test_seed05_motor0_maximum_is_recomputed_from_arrays(report: dict[str, Any]) -> None:
    module = _load_module()
    candidate = report["variants"][module.friday.CANDIDATE_LABEL]
    arrays = module._summary_arrays(candidate["stage_a_legacy_pwm"])
    fact = report["maximum_seed05_motor0_demand"]
    index = int(np.argmax(arrays["motor_pwm_preclip"][:, 0]))

    assert fact["motor_index"] == 0
    assert fact["index_zero_based"] == index
    assert fact["requested_pwm"] == float(arrays["motor_pwm_preclip"][index, 0])
    assert fact["upper_exceedance_pwm"] == float(arrays["pwm_upper_exceedance"][index, 0])
    assert fact["requested_motor_force_N"] == float(arrays["motor_force_preclip_N"][index, 0])
    assert fact["upper_exceedance_motor_force_N"] == float(
        arrays["force_upper_exceedance_N"][index, 0]
    )
    assert fact["requested_motor_speed"] == float(arrays["motor_speed_preclip"][index, 0])
    assert fact["upper_exceedance_motor_speed"] == float(arrays["speed_upper_exceedance"][index, 0])
    assert candidate["stage_b_si_force"]["limits"]["configured_rpm2thrust"] == [
        0.0,
        -5.382196214637237e-7,
        2.4582929831265485e-10,
    ]


@pytest.mark.unit
def test_relative_wrench_values_only_exist_above_predeclared_floors(report: dict[str, Any]) -> None:
    module = _load_module()
    assert report["numerical_contract"]["wrench_relative_floors"] == pytest.approx(
        [5.859375e-05, 1.9060546875e-06, 1.9060546875e-06, 4.3071462050194093e-07],
        rel=0.0,
        abs=1.0e-18,
    )
    for variant in report["variants"].values():
        for name in ("stage_a_legacy_pwm", "stage_b_si_force"):
            section = variant[name]
            arrays = module._summary_arrays(section)
            payload = section["relative_wrench_distortion"]
            mask = np.asarray(payload["defined_mask"], dtype=bool)
            values = payload["values_or_null_when_undefined"]
            assert np.array_equal(
                mask, np.abs(arrays["wrench_preclip"]) >= np.asarray(payload["floors"])
            )
            for row, row_mask in zip(values, mask, strict=True):
                assert all(
                    (value is not None) == bool(active)
                    for value, active in zip(row, row_mask, strict=True)
                )


@pytest.mark.unit
def test_all_summaries_are_programmatically_array_derived(report: dict[str, Any]) -> None:
    module = _load_module()

    module.validate_report_summaries(report)
    tampered = copy.deepcopy(report)
    tampered["variants"]["default"]["stage_a_legacy_pwm"]["summary"]["motor_clipping"]["any"][
        "count_all_channels"
    ] = 1

    with pytest.raises(module.DiagnosticError, match="not array-derived"):
        module.validate_report_summaries(tampered)


@pytest.mark.unit
def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    module = _load_module()
    output_dir = tmp_path / "already-exists"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.write_package(output_dir, {})


@pytest.mark.unit
def test_checksum_inventory_covers_exactly_four_payloads(tmp_path: Path) -> None:
    module = _load_module()
    payloads = {name: f"payload:{name}\n".encode() for name in module.OUTPUT_NAMES}
    output_dir = tmp_path / "package"

    module.write_package(output_dir, payloads)

    checksum_lines = (output_dir / "SHA256SUMS").read_text().splitlines()
    assert len(checksum_lines) == 4
    assert [line.split("  ", maxsplit=1)[1] for line in checksum_lines] == sorted(
        module.OUTPUT_NAMES
    )
    assert {path.name for path in output_dir.iterdir()} == set(module.OUTPUT_NAMES) | {"SHA256SUMS"}
