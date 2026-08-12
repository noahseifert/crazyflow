from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "examples/jax/mellinger_friday_evidence.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mellinger_friday_evidence", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load Friday evidence module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_bounded_tail_extracts_only_top_level_checkpoint_fields(tmp_path: Path) -> None:
    module = _load_module()
    payload = {
        "history": [{"step": value} for value in range(20_000)],
        "raw_gains": {"data": [-1.0, 2.0, 1.5, 2.2], "dtype": "float32", "shape": [4]},
        "selected_raw_gains": {"data": [-1.1, 2.1, 1.6, 2.3], "dtype": "float32", "shape": [4]},
        "selection": {
            "selected_step": 5000,
            "selected_validation_loss": 0.0025,
            "test_used_for_selection": False,
        },
        "step": 5000,
        "validation_manifest_id": "opaque-id",
    }
    checkpoint = tmp_path / "checkpoint-step-005000.json"
    checkpoint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    assert checkpoint.stat().st_size > module.CHECKPOINT_TAIL_BYTES

    result = module.extract_checkpoint_tail(checkpoint)

    assert result["step"] == 5000
    assert result["selection"]["selected_step"] == 5000
    assert result["selected_raw_gains"]["data"] == [-1.1, 2.1, 1.6, 2.3]


@pytest.mark.unit
def test_candidate_rule_uses_unique_smallest_existing_selected_loss() -> None:
    module = _load_module()
    seeds = []
    for index in range(1, 11):
        loss = 0.0024 if index == 5 else 0.003 + index * 1.0e-5
        seeds.append(
            {
                "index": index,
                "selection": {"selected_step": 5000, "selected_validation_loss": loss},
                "selected_physical_gains": {
                    name: {"value": 0.1 * index} for name in module.GAIN_NAMES
                },
            }
        )

    result = module.select_candidate({"seeds": seeds})

    assert result["seed"] == 5
    assert result["label"] == module.CANDIDATE_LABEL
    assert result["selected_validation_loss"] == 0.0024


@pytest.mark.unit
def test_seed05_crosscheck_rejects_gain_average_or_source_mismatch() -> None:
    module = _load_module()
    raw = [-1.3688596487045288, 2.3636748790740967, 1.8979853391647339, 2.1928365230560303]
    physical = module._physical_gain_dict(raw)
    candidate = {
        "seed": 5,
        "selected_step": 5000,
        "selected_validation_loss": 0.002523899544030428,
        "physical_gains": physical,
    }
    summary = {
        "selected": {
            "selected_step": 5000,
            "selected_validation_loss": 0.002523899544030428,
            "selected_raw_gains": raw,
        },
        "selected_physical_gains": physical,
    }
    checkpoint = {
        "step": 5000,
        "selection": {"selected_step": 5000, "selected_validation_loss": 0.002523899544030428},
        "selected_raw_gains": {"data": raw},
    }

    assert np.allclose(module._check_candidate_sources(candidate, summary, checkpoint), raw)
    candidate["physical_gains"]["kp_xy"] += 0.01
    with pytest.raises(module.EvidenceError, match="source cross-check"):
        module._check_candidate_sources(candidate, summary, checkpoint)


@pytest.mark.unit
def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    module = _load_module()
    output_dir = tmp_path / "already-exists"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.write_package({}, tmp_path, output_dir)


@pytest.mark.unit
def test_fallback_withholds_rollout_without_calling_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    candidate = {
        "label": module.CANDIDATE_LABEL,
        "source_crosscheck": {"synthetic_unit_source": True},
    }
    h100 = {"seeds": [], "duration": {"per_run_available_count": module.EXPECTED_SEEDS}}
    monkeypatch.setattr(module, "build_h100_evidence", lambda _root: (h100, candidate))
    monkeypatch.setattr(
        module,
        "build_rollout",
        lambda _candidate: pytest.fail("fallback must not execute a rollout"),
    )
    monkeypatch.setattr(
        module,
        "_load_json",
        lambda _path: {"schema_version": "crazyflow.mellinger_loss_diagnostics.v1"},
    )

    report = module.build_report(Path("unused"), fallback_existing_evidence=True)

    assert report["status"] == "WITHHELD_TECHNICAL_GATE"
    assert "simulation visualization" not in report["scope"]
    assert "rollout evidence withheld" in report["scope"]
    assert report["rollout"] == {
        "status": "WITHHELD_TECHNICAL_GATE",
        "evidence_available": False,
        "missing_evidence": (
            "default-versus-provisional-candidate trajectory and tracking-error evidence"
        ),
        "acceptance_contract": "zero motor saturation for both compared controllers",
        "substitution_performed": False,
        "rollout_arrays_present": False,
        "rollout_figures_present": False,
    }
    assert module._output_names(True) == tuple(
        name for name in module.OUTPUT_NAMES if name not in module.ROLLOUT_OUTPUT_NAMES
    )
    assert len(module._output_names(True)) == 9


@pytest.mark.unit
def test_fallback_provenance_records_exact_verification_commands() -> None:
    module = _load_module()

    provenance = module._provenance(
        "a" * 40, Path("/read-only/h100-root"), fallback_existing_evidence=True
    )

    commands = provenance["commands"]
    assert "tests/unit/test_mellinger_friday_evidence.py" in commands["focused_pytest"]
    assert "tests/unit/test_mellinger_loss_diagnostics.py" in commands["focused_pytest"]
    assert "-m ruff check" in commands["ruff_check"]
    assert "-m ruff format --check" in commands["ruff_format_check"]
    assert "--fallback-existing-evidence" in commands["two_target_reproduction_and_diff"]
    assert "diff -qr" in commands["two_target_reproduction_and_diff"]
    assert provenance["output_checksum_rule"].startswith("SHA256SUMS covers the 9")


@pytest.mark.unit
def test_generator_commit_is_selected_by_unique_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    generator_commit = "1" * 40
    package_commit = "2" * 40
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda *_arguments: (
            f"{package_commit}\0research: record Friday evidence package\n"
            f"{generator_commit}\0research: add Friday evidence generator"
        ),
    )

    assert module._generator_result_commit() == generator_commit


@pytest.mark.unit
def test_normal_mode_propagates_unchanged_rollout_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    candidate = {"label": module.CANDIDATE_LABEL}
    monkeypatch.setattr(module, "build_h100_evidence", lambda _root: ({"seeds": []}, candidate))
    monkeypatch.setattr(
        module,
        "build_rollout",
        lambda _candidate: (_ for _ in ()).throw(module.EvidenceError("technical gates")),
    )

    with pytest.raises(module.EvidenceError, match="technical gates"):
        module.build_report(Path("unused"), fallback_existing_evidence=False)
