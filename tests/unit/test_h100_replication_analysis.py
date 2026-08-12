from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "artifacts/day19-h100-analysis/analyze_h100_replication.py"


def _load_analysis_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("h100_replication_analysis", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load H100 replication analysis module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_row(step: int, split: str, loss: float) -> dict[str, object]:
    metrics = {
        "motor_saturation_fraction": 0.0,
        "floor_clip_fraction": 0.0,
        "zero_thrust_gate_fraction": 0.0,
        "nonfinite_state_fraction": 0.0,
        "max_position_error_m": 0.01,
    }
    result: dict[str, object] = {
        "step": step,
        "split": split,
        "loss": loss,
        "metrics": metrics,
        "technical_gate": {"passed": True},
    }
    if split == "train":
        result["gradient_l2_norm"] = 0.1 / step
    return result


@pytest.mark.unit
def test_frozen_aggregate_uses_ddof_one_student_t_and_linear_quantiles() -> None:
    module = _load_analysis_module()
    values = [float(value) for value in range(1, 11)]

    first = module._aggregate(values)
    second = module._aggregate(values)

    expected_sd = float(np.std(np.asarray(values), ddof=1))
    expected_half_width = module.T_CRITICAL_95_DF9 * expected_sd / math.sqrt(10)
    assert first == second
    assert first["mean"] == pytest.approx(5.5)
    assert first["sample_standard_deviation_ddof_1"] == pytest.approx(expected_sd)
    assert first["confidence_interval_half_width"] == pytest.approx(expected_half_width)
    assert first["q1_linear"] == pytest.approx(3.25)
    assert first["q3_linear"] == pytest.approx(7.75)


@pytest.mark.unit
def test_validation_selection_uses_all_updates_and_retains_earliest_exact_tie() -> None:
    module = _load_analysis_module()
    rows = [
        row
        for step, train_loss, validation_loss in ((1, 4.0, 3.0), (2, 3.0, 2.0), (3, 2.5, 2.0))
        for row in (
            _metric_row(step, "train", train_loss),
            _metric_row(step, "validation", validation_loss),
        )
    ]

    result = module._analyze_metric_rows(rows, expected_updates=3)

    assert result["selected_step"] == 2
    assert result["selected_validation_loss"] == 2.0
    assert result["validation_loss_at_step_1"] == 3.0
    assert result["validation_loss_at_step_5000"] == 2.0
    assert result["relative_validation_improvement"] == pytest.approx(1.0 / 3.0)


@pytest.mark.unit
def test_metric_analysis_hard_rejects_test_rows() -> None:
    module = _load_analysis_module()
    rows = [_metric_row(1, "train", 2.0), _metric_row(1, "test", 1.0)]

    with pytest.raises(module.AnalysisError, match="Test metric row"):
        module._analyze_metric_rows(rows, expected_updates=1)


@pytest.mark.unit
def test_test_manifest_path_remains_an_opaque_unopened_reference(tmp_path: Path) -> None:
    module = _load_analysis_module()
    forbidden_path = tmp_path / "must-not-open-test-manifest.json"
    summary = {
        "test_manifest_opened": False,
        "test_metrics_present": False,
        "selected": {"test_used_for_selection": False},
    }
    provenance = {"frozen_test_manifest_reference_not_opened": str(forbidden_path)}
    released_config = {"test_manifest": str(forbidden_path)}

    result = module._validate_test_boundary(summary, provenance, released_config)

    assert result["analysis_opened_test_manifest"] is False
    assert not forbidden_path.exists()


@pytest.mark.unit
def test_run_index_parser_rejects_parent_traversal(tmp_path: Path) -> None:
    module = _load_analysis_module()
    index = tmp_path / "RUN_SHA256SUMS"
    index.write_text(f"{'0' * 64}  ./../outside\n")

    with pytest.raises(module.AnalysisError, match="unsafe indexed path"):
        module._parse_run_index(index)
