from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "examples/jax/mellinger_saturation_diagnostic.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mellinger_saturation_diagnostic", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load saturation diagnostic module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_diagnostic_cli_requires_explicit_failed_gate_mode() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args([])
    arguments = module.parse_args(["--diagnostic-only-saturation-present"])

    assert arguments.diagnostic_only_saturation_present is True
    assert arguments.output_dir == Path("artifacts/day22-saturation-diagnostic")


@pytest.mark.unit
def test_saturation_formula_preserves_exact_boundaries() -> None:
    module = _load_module()
    minimum = np.full((1, 1, 1, 4), 100.0)
    maximum = np.full((1, 1, 1, 4), 1000.0)
    commanded = np.asarray([50.0, 50.0001, 100.01, 100.02, 999.8999, 999.9, 1000.0])
    commanded = np.broadcast_to(commanded[:, None, None, None], (7, 1, 1, 4))

    lower, upper = module.saturation_decisions(commanded, minimum, maximum)

    assert lower[:, 0, 0, 0].tolist() == [False, True, True, False, False, False, False]
    assert upper[:, 0, 0, 0].tolist() == [False, False, False, False, False, True, True]


@pytest.mark.unit
def test_attribution_is_exact_per_motor_bound_index_and_interval() -> None:
    module = _load_module()
    commands = np.full((module.friday.HORIZON, 1, 1, 4), 500.0, dtype=np.float32)
    commands[1:3, 0, 0, 0] = 100.0
    commands[3, 0, 0, 1] = 1000.0
    commands[4:6, 0, 0, 2] = 100.0
    minimum = np.full((1, 1, 1), 100.0, dtype=np.float32)
    maximum = np.full((1, 1, 1), 1000.0, dtype=np.float32)
    state_times = np.arange(1, module.friday.HORIZON + 1) / 100.0
    metric = float(np.asarray(5 / 400, dtype=np.float32))

    result = module.attribute_saturation(commands, minimum, maximum, state_times, metric)

    assert result["total"]["saturated_count"] == 5
    assert result["total"]["lower_saturated_count"] == 4
    assert result["total"]["upper_saturated_count"] == 1
    assert result["by_motor"][0]["affected"]["lower"] == {
        "indices_zero_based": [1, 2],
        "state_times_s": [0.02, 0.03],
    }
    assert result["by_motor"][0]["intervals"] == [
        {
            "bound": "lower",
            "start_index_inclusive": 1,
            "end_index_inclusive": 2,
            "start_time_s": 0.01,
            "end_time_s": 0.03,
            "sample_count": 2,
            "duration_s": 0.02,
        }
    ]
    assert result["by_motor"][1]["affected"]["upper"]["indices_zero_based"] == [3]
    assert result["by_motor"][2]["maximum_consecutive_saturated_samples"] == 2
    assert result["by_motor"][3]["saturated_count"] == 0


@pytest.mark.unit
def test_attribution_rejects_metric_disagreement() -> None:
    module = _load_module()
    commands = np.full((module.friday.HORIZON, 1, 1, 4), 500.0, dtype=np.float32)
    minimum = np.full((1, 1, 1), 100.0, dtype=np.float32)
    maximum = np.full((1, 1, 1), 1000.0, dtype=np.float32)
    state_times = np.arange(1, module.friday.HORIZON + 1) / 100.0

    with pytest.raises(module.DiagnosticError, match="differs"):
        module.attribute_saturation(commands, minimum, maximum, state_times, 0.1)


@pytest.mark.unit
def test_day21_candidate_and_five_payload_identities_are_frozen() -> None:
    module = _load_module()

    candidate, identities, provenance = module.load_day21_candidate()

    assert len(identities) == 5
    assert candidate["seed"] == 5
    assert candidate["selected_step"] == 5000
    assert candidate["selected_validation_loss"] == 0.002523899544030428
    assert candidate["physical_gains"] == module.EXPECTED_CANDIDATE["physical_gains"]
    assert provenance["rollout_evidence"] == "WITHHELD_TECHNICAL_GATE"


@pytest.mark.unit
def test_normal_f0_1_gate_still_rejects_candidate_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    synthetic_rollout = {
        "equality_and_technical_gates": {
            "same_reference": True,
            "same_initial_condition": True,
            "same_integrator_and_step_function": True,
            "same_time_base_and_metrics": True,
            "default_all_technical_gates_pass": True,
            "candidate_all_technical_gates_pass": False,
        }
    }
    monkeypatch.setattr(
        module.friday,
        "build_rollout_trace_bundle",
        lambda _candidate: (synthetic_rollout, {}, None),
    )

    with pytest.raises(module.friday.EvidenceError, match="technical gates"):
        module.friday.build_rollout({})


@pytest.mark.unit
def test_provenance_records_executed_f0_1_method_without_day13_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_generator_commit", lambda: "1" * 40)
    report = {
        "source_identity": {
            "accepted_day21_checksum_index": {"path": "SHA256SUMS", "sha256": "2" * 64},
            "accepted_day21_payloads_verified": [],
            "accepted_day21_generator_commit": "3" * 40,
        },
        "frozen_rollout_contract": {},
    }

    commands = module._provenance(report)["commands"]

    assert "build_rollout_trace_bundle" in commands["f0_1_gate_regression_executed"]
    assert "0.027499999850988388" in commands["f0_1_gate_regression_executed"]
    assert "existing F0.1 tests" in commands["f0_1_source_and_test_regression_executed"]
    assert commands["f0_1_full_normal_and_fallback_cli"].startswith("NOT_EXECUTED")
    assert "protected Day-13" in commands["f0_1_full_normal_and_fallback_cli"]
    assert "--h100-root" not in "\n".join(commands.values())


@pytest.mark.unit
def test_nonfinite_values_stop_before_output() -> None:
    module = _load_module()

    with pytest.raises(module.DiagnosticError, match="nonfinite"):
        module._require_all_finite("synthetic", {"bad": [0.0, float("nan")]})


@pytest.mark.unit
def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    module = _load_module()
    output_dir = tmp_path / "already-exists"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.write_package(output_dir, {})


@pytest.mark.unit
def test_checksum_inventory_covers_exactly_five_payloads(tmp_path: Path) -> None:
    module = _load_module()
    payloads = {name: f"payload:{name}\n".encode() for name in module.OUTPUT_NAMES}
    output_dir = tmp_path / "package"

    module.write_package(output_dir, payloads)

    checksum_lines = (output_dir / "SHA256SUMS").read_text().splitlines()
    assert len(checksum_lines) == 5
    assert [line.split("  ", maxsplit=1)[1] for line in checksum_lines] == sorted(
        module.OUTPUT_NAMES
    )
    assert {path.name for path in output_dir.iterdir()} == set(module.OUTPUT_NAMES) | {"SHA256SUMS"}


@pytest.mark.unit
def test_actual_frozen_diagnostic_has_expected_gate_and_complete_attribution() -> None:
    module = _load_module()

    report = module.build_diagnostic()
    default = report["variants"]["default"]
    candidate = report["variants"][module.friday.CANDIDATE_LABEL]

    assert report["status"] == module.STATUS
    assert report["normal_acceptance_gate"]["status"] == module.GATE_FAILURE_STATUS
    assert default["saturation"]["total"]["saturated_count"] == 0
    assert candidate["status"] == "DIAGNOSTIC_ONLY_SATURATION_PRESENT"
    assert candidate["saturation"]["total"]["saturated_count"] == 11
    assert candidate["saturation"]["total"]["tracking_loss_metric_fraction"] == pytest.approx(
        module.EXPECTED_CANDIDATE_FRACTION, abs=1.0e-9, rel=0.0
    )
    assert sum(record["saturated_count"] for record in candidate["saturation"]["by_motor"]) == 11
    assert report["validations"] == {
        "all_states_commands_metrics_and_plot_arrays_finite": True,
        "frozen_contract_match": True,
        "day21_candidate_match": True,
        "default_fraction_exactly_zero": True,
        "seed05_fraction_matches_historical_atol_1e_9_rtol_0": True,
        "every_saturation_attributed_once": True,
        "test_opened": False,
        "training_runs_started": 0,
    }
