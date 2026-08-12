"""CLI, benchmark-envelope, and deterministic package tests for Day 25."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from crazyflow.control.mellinger.research import robust_evaluation as robust
from examples.jax import mellinger_cf21b_robust_foundation as cli

if TYPE_CHECKING:
    from pathlib import Path


def _benchmark_result(world_count: int) -> dict:
    return {
        "world_count": world_count,
        "loss": 0.25,
        "gradient": [0.1, -0.2, 0.3, -0.4],
        "timing": {
            "steady_worlds_per_s": float(world_count),
            "steady_scored_episode_seconds_per_s": float(2 * world_count),
            "steady_median_ns": 1_000_000,
            "first_synchronized_execution_ns": 2_000_000,
        },
        "memory": {
            "swap_activity_detected": False,
            "ru_maxrss_bytes": 1024,
            "before": {"mem_total_bytes": 16 * 1024**3},
        },
    }


def _benchmark_payload() -> dict:
    core = {
        "schema_version": cli.BENCHMARK_SCHEMA,
        "status": "PASS_4_16_32_BENCHMARK",
        "results": [_benchmark_result(count) for count in (4, 16, 32)],
        "recommendation": {"world_count": 32},
    }
    return core | {"raw_measurement_sha256": cli._sha256_bytes(cli._json_bytes(core))}


def test_parse_exact_public_modes() -> None:
    benchmark = cli.parse_args(
        [
            "--measure-benchmark",
            "--benchmark-output",
            "/tmp/benchmark.json",
            "--world-counts",
            "4",
            "16",
            "32",
            "--steady-repeats",
            "3",
            "--child-timeout-seconds",
            "300",
        ]
    )
    assert benchmark.measure_benchmark is True
    assert benchmark.world_counts == [4, 16, 32]
    package = cli.parse_args(
        [
            "--generate-package",
            "--benchmark-input",
            "/tmp/benchmark.json",
            "--output-dir",
            "/tmp/package",
        ]
    )
    assert package.generate_package is True


def test_parse_rejects_incomplete_modes() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--measure-benchmark"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--generate-package", "--benchmark-input", "/tmp/raw.json"])


def test_benchmark_payload_checksum_and_exact_inventory() -> None:
    payload = _benchmark_payload()
    cli._validate_benchmark_payload(payload)
    broken = copy.deepcopy(payload)
    broken["results"][0]["loss"] = 0.5
    with pytest.raises(cli.FoundationCliError, match="checksum"):
        cli._validate_benchmark_payload(broken)
    broken = copy.deepcopy(payload)
    broken["results"] = broken["results"][:2]
    core = dict(broken)
    core.pop("raw_measurement_sha256")
    broken["raw_measurement_sha256"] = cli._sha256_bytes(cli._json_bytes(core))
    with pytest.raises(cli.FoundationCliError, match="4/16/32"):
        cli._validate_benchmark_payload(broken)


def test_benchmark_rejects_substitution_before_child_process(tmp_path: Path) -> None:
    with pytest.raises(cli.FoundationCliError, match="exactly 4 16 32"):
        cli.measure_benchmark(tmp_path / "raw.json", [4, 16], 3, 300)
    with pytest.raises(cli.FoundationCliError, match="3/300"):
        cli.measure_benchmark(tmp_path / "raw.json", [4, 16, 32], 2, 300)


def test_child_rejects_nonfrozen_repeat_count_before_simulation(tmp_path: Path) -> None:
    with pytest.raises(cli.FoundationCliError, match="exactly three"):
        cli._benchmark_child(4, 2, tmp_path / "child.json")


def test_write_package_inventory_checksums_and_overwrite_refusal(tmp_path: Path) -> None:
    payloads = {name: f"payload:{name}\n".encode() for name in cli.PACKAGE_NAMES}
    output = tmp_path / "package"
    cli.write_package(output, payloads)
    assert sorted(path.name for path in output.iterdir()) == sorted(
        (*cli.PACKAGE_NAMES, "SHA256SUMS")
    )
    expected = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    observed = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (output / "SHA256SUMS").read_text().splitlines()
    }
    assert observed == expected
    with pytest.raises(cli.FoundationCliError, match="overwrite"):
        cli.write_package(output, payloads)


def test_write_package_rejects_missing_payload_before_output(tmp_path: Path) -> None:
    payloads = {name: b"x" for name in cli.PACKAGE_NAMES[:-1]}
    output = tmp_path / "package"
    with pytest.raises(cli.FoundationCliError, match="incomplete"):
        cli.write_package(output, payloads)
    assert not output.exists()


def test_generate_package_refuses_existing_directory_before_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    called = False

    def forbidden(_: Path):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(cli, "build_package_payloads", forbidden)
    with pytest.raises(cli.FoundationCliError, match="overwrite"):
        cli.generate_package(tmp_path / "benchmark.json", output)
    assert called is False


def test_generator_commit_is_stable_first_source_test_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = "1" * 40
    commits = (implementation, "2" * 40, "3" * 40, "4" * 40, "5" * 40, "6" * 40, "7" * 40)

    def fake_git(*arguments: str) -> str:
        if arguments[:2] == ("rev-list", "--reverse"):
            return "\n".join(commits)
        if arguments[:3] == ("show", "-s", "--format=%s"):
            return cli.IMPLEMENTATION_COMMIT_SUBJECT
        if arguments[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            return "\n".join(sorted(cli.IMPLEMENTATION_COMMIT_PATHS))
        raise AssertionError(arguments)

    monkeypatch.setattr(cli, "_git", fake_git)
    assert cli._generator_commit() == implementation


def test_provenance_is_independent_of_observed_checkout_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_git(*arguments: str) -> str:
        raise AssertionError(f"provenance observed mutable Git context: {arguments}")

    monkeypatch.setattr(cli, "_git", forbidden_git)
    provenance = cli._provenance(
        {
            "input_identity": {"frozen": True},
            "correction_invariance": {"status": "PASS_PRE_CORRECTION_C4_INVARIANCE"},
            "configuration": {"registry_fingerprint": "fixed"},
        },
        {"raw_measurement_sha256": "a" * 64},
        "b" * 64,
        "1" * 40,
    )

    assert "branch" not in provenance
    assert provenance["integration_target_branch"] == "research/differentiable-mellinger"
    assert provenance["checkout_branch_provenance"] == {
        "included_in_reproducible_payload": False,
        "reason": (
            "checkout branch is a mutable execution-context reference; generator_commit is the "
            "canonical immutable science/code identity"
        ),
    }
    assert provenance["post_integration_metadata_decision"] == "D-048"
    assert provenance["generator_commit"] == "1" * 40
    assert provenance["generator_commit_resolution"] == {
        "rule": "first prescribed source/test commit after source_base_commit",
        "subject": cli.IMPLEMENTATION_COMMIT_SUBJECT,
        "stable_from_implementation_or_evidence_commit": True,
    }


def test_pre_correction_review_pins_reproduce_from_explicit_c4() -> None:
    old = robust.PRE_CORRECTION_RESULT_COMMIT
    report_bytes = cli._git_blob(
        old, f"{cli.PRE_CORRECTION_PACKAGE_ROOT}/robust_evaluation_report.json"
    )
    contract_bytes = cli._git_blob(old, f"{cli.PRE_CORRECTION_PACKAGE_ROOT}/episode_contract.json")
    assert hashlib.sha256(report_bytes).hexdigest() == robust.PRE_CORRECTION_REPORT_SHA256
    assert hashlib.sha256(contract_bytes).hexdigest() == robust.PRE_CORRECTION_CONTRACT_SHA256
    numeric = robust.numeric_projection_identity(json.loads(report_bytes))
    assert numeric == {
        "path_count": 5_465_162,
        "serialized_bytes": 644_397_249,
        "sha256": robust.PRE_CORRECTION_NUMERIC_PROJECTION_SHA256,
    }
    assert robust.parent_contract_pin_identity(json.loads(contract_bytes)) == {
        "record_count": 16,
        "serialized_bytes": 18_037,
        "sha256": robust.PRE_CORRECTION_PARENT_PIN_SHA256,
    }


def test_full_package_runs_c4_invariance_and_discloses_d047(tmp_path: Path) -> None:
    benchmark = cli._git_blob(
        robust.PRE_CORRECTION_RESULT_COMMIT,
        f"{cli.PRE_CORRECTION_PACKAGE_ROOT}/benchmark_measurements.json",
    )
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_bytes(benchmark)
    payloads = cli.build_package_payloads(benchmark_path)
    report = json.loads(payloads["robust_evaluation_report.json"])
    contract = json.loads(payloads["episode_contract.json"])
    provenance = json.loads(payloads["provenance.json"])
    invariance = report["correction_invariance"]
    assert invariance["status"] == "PASS_PRE_CORRECTION_C4_INVARIANCE"
    assert invariance["old_side_result_commit"] == robust.PRE_CORRECTION_RESULT_COMMIT
    assert (
        invariance["numeric_report_projection"]["old"]
        == invariance["numeric_report_projection"]["new_values_at_all_old_paths"]
    )
    assert (
        invariance["parent_identity_projection"]["old"]
        == invariance["parent_identity_projection"]["new"]
    )
    assert invariance["benchmark_bytes_equal"] is True
    assert (
        contract["contract"]["vertical_excitation"] == robust.vertical_excitation_machine_contract()
    )
    assert report["trajectory_contract"]["numerical_construction_changed_by_d047"] is False
    assert provenance["correction_decision"] == "D-047"
    assert provenance["generator_commit"] == cli._generator_commit()
    assert "branch" not in provenance
    assert provenance["integration_target_branch"] == cli.INTEGRATION_TARGET_BRANCH
    assert provenance["checkout_branch_provenance"] == {
        "included_in_reproducible_payload": False,
        "reason": cli.CHECKOUT_BRANCH_PROVENANCE_REASON,
    }
    assert provenance["post_integration_metadata_decision"] == "D-048"
    assert provenance["pre_correction_invariance"] == invariance
    handoff = payloads["TECHNICAL_HANDOFF.md"].decode()
    assert "PASS_PRE_CORRECTION_C4_INVARIANCE" in handoff
    assert "continuous 6 s rollout" in handoff
    assert "Float32 `0.06 m/s`" in handoff
    assert "scientific foundation review is `ACCEPTED`" in handoff
    assert cli.CHECKOUT_BRANCH_PROVENANCE_REASON in handoff


def test_handoff_preserves_negative_finding_and_claim_boundary() -> None:
    report = {
        "generator_commit": "2288ab2fe3b55f585fb9b14a8fe9d87c07eb6c1a",
        "mass_ablation": {
            "negative_physical_value_match_finding": {
                "variants": {
                    "repository_mismatch": {
                        "mean_z_rmse_m": 0.08,
                        "negative_z_integral_contact_count": 500,
                    },
                    "matched": {"mean_z_rmse_m": 0.11, "negative_z_integral_contact_count": 1800},
                }
            }
        },
        "correction_invariance": {
            "status": "PASS_PRE_CORRECTION_C4_INVARIANCE",
            "old_side_result_commit": robust.PRE_CORRECTION_RESULT_COMMIT,
            "numeric_report_projection": {"old": {"path_count": 5_465_162}},
        },
    }
    benchmark = {"recommendation": {"world_count": 16}}
    handoff = cli._handoff(report, benchmark).decode()
    assert "negative finding" in handoff.lower()
    assert "not improvement" in handoff
    assert "code-path inference" in handoff
    assert "no optimizer update" in handoff.lower()
    assert "firmware" in handoff and "flight" in handoff


def test_checksum_json_is_deterministic() -> None:
    payload = {"b": [2, 1], "a": {"finite": True}}
    first = cli._json_bytes(payload)
    second = cli._json_bytes(copy.deepcopy(payload))
    assert first == second
    assert cli._sha256_bytes(first) == cli._sha256_bytes(second)
