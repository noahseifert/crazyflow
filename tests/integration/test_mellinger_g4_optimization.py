"""CLI and cross-component tests for bounded G4 technical gates."""

from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from crazyflow.control.mellinger.research import g4_optimization as g4
from examples.jax import mellinger_g4_optimization as cli

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / g4.CONFIG_RELATIVE_PATH
LONGRUN_RESOURCE_SCHEMA = "crazyflow.mellinger_g4_longrun_resource_evidence.v1"


def _longrun_operational_update(token: int, parent_ids: list[str]) -> dict[str, object]:
    total = 16 * 1024**3
    allowed = int(g4.RSS_FRACTION_LIMIT * total)
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
                "parent_ids": parent_ids[4 * index : 4 * index + 4],
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


def _task_root(label: str) -> Path:
    root = Path("/tmp") / f"gr-g4-004-{label}-{os.getpid()}"
    root.mkdir(mode=0o700)
    return root


def _remove_known_outputs(root: Path) -> None:
    output = root / "output"
    for name in (
        "result.json",
        "checkpoint-10.json",
        "checkpoint-20.json",
        "continuous-checkpoint-20.json",
        "split-checkpoint-10.json",
        "resume-worker-result.json",
    ):
        path = output / name
        if path.exists():
            path.unlink()
    if output.exists():
        output.rmdir()
    origin = root / "source-origin.json"
    if origin.exists():
        origin.unlink()
    root.rmdir()


def test_parse_requires_one_mode_exact_config_origin_output_and_joint_count() -> None:
    assert Path(cli.__file__).read_text().splitlines().count("RUN_IN_INTEGRATION_TEST = False") == 1
    with pytest.raises(SystemExit):
        cli.parse_args(())
    with pytest.raises(SystemExit):
        cli.parse_args(("--dry-run",))
    with pytest.raises(SystemExit):
        cli.parse_args(
            (
                "--dry-run",
                "--config",
                str(CONFIG_PATH),
                "--source-origin-record",
                "/tmp/gr-g4-004-x/source-origin.json",
            )
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            (
                "--joint-smoke",
                "--config",
                str(CONFIG_PATH),
                "--source-origin-record",
                "/tmp/gr-g4-004-x/source-origin.json",
                "--output-dir",
                "/tmp/gr-g4-004-x/output",
                "--max-updates-this-process",
                "9",
            )
        )

    args = cli.parse_args(
        (
            "--joint-smoke",
            "--config",
            str(CONFIG_PATH),
            "--source-origin-record",
            "/tmp/gr-g4-004-x/source-origin.json",
            "--output-dir",
            "/tmp/gr-g4-004-x/output",
            "--max-updates-this-process",
            "10",
        )
    )
    assert args.joint_smoke
    assert args.max_updates_this_process == 10


def test_source_origin_cli_writes_mode_600_and_refuses_overwrite() -> None:
    root = _task_root("origin-integration")
    try:
        path = root / "source-origin.json"
        args = cli.parse_args(("--write-source-origin-record", str(path)))
        result = cli.run(args)

        assert result["status"] == "PASS_SOURCE_ORIGIN_RECORD"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert (
            g4.validate_source_origin_record(REPOSITORY_ROOT, path)["record_sha256"]
            == result["record_sha256"]
        )
        with pytest.raises(g4.G4ContractError, match="absent"):
            cli.run(args)
    finally:
        _remove_known_outputs(root)


@pytest.fixture(scope="module")
def dry_run_result() -> dict[str, object]:
    root = _task_root("dry-run-integration")
    origin = root / "source-origin.json"
    cli.run(cli.parse_args(("--write-source-origin-record", str(origin))))
    args = cli.parse_args(
        (
            "--config",
            str(CONFIG_PATH),
            "--source-origin-record",
            str(origin),
            "--dry-run",
            "--output-dir",
            str(root / "output"),
        )
    )
    result = cli.run(args)
    yield result
    _remove_known_outputs(root)


def test_dry_run_enforces_public_config_parent_protection_allowlist_before_optimizer(
    dry_run_result: dict[str, object],
) -> None:
    assert dry_run_result["status"] == "PASS_M1_DRY_RUN"
    assert dry_run_result["mode"] == "dry_run"
    result = dry_run_result["result"]
    assert result["public_inputs"]["status"] == "PASS_SIX_PUBLIC_DAY26_INPUTS"
    assert result["parents"]["status"] == "PASS_288_PARENT_FREEZE"
    assert result["parents"]["parent_count"] == 288
    assert result["parents"]["cross_split_leakage_count"] == 0
    assert len(result["parents"]["parent_digest"]) == 64
    assert len(result["parents"]["attempt_digest"]) == 64
    assert len(result["parents"]["key_digest"]) == 64
    assert len(result["parents"]["array_digest"]) == 64
    assert result["write_allowlist"] == list(g4.WRITE_PATHS)
    assert result["optimizer_initialization_count"] == 0
    assert result["optimizer_update_count"] == 0
    assert result["protection"]["contents_opened_or_hashed"] is False


def test_dry_run_writes_canonical_mode_600_result_and_freshness_gate(
    dry_run_result: dict[str, object],
) -> None:
    root = Path("/tmp") / f"gr-g4-004-dry-run-integration-{os.getpid()}"
    result_path = root / "output/result.json"

    assert stat.S_IMODE((root / "output").stat().st_mode) == 0o700
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o600
    assert json.loads(result_path.read_bytes()) == dry_run_result
    assert result_path.read_bytes() == g4.canonical_json_bytes(dry_run_result)


def test_common_preflight_rejects_stale_source_origin_record() -> None:
    root = _task_root("stale-origin-integration")
    try:
        origin = root / "source-origin.json"
        cli.run(cli.parse_args(("--write-source-origin-record", str(origin))))
        payload = json.loads(origin.read_bytes())
        unsigned = dict(payload)
        unsigned.pop("record_sha256")
        unsigned["origins"]["g4_module"]["sha256"] = "0" * 64
        unsigned["record_sha256"] = g4.sha256_bytes(g4.canonical_json_bytes(unsigned))
        origin.write_bytes(g4.canonical_json_bytes(unsigned))
        os.chmod(origin, 0o600)
        args = cli.parse_args(
            (
                "--config",
                str(CONFIG_PATH),
                "--source-origin-record",
                str(origin),
                "--dry-run",
                "--output-dir",
                str(root / "output"),
            )
        )
        with pytest.raises(g4.G4ContractError, match="stale or foreign"):
            cli.run(args)
        assert not (root / "output").exists()
    finally:
        _remove_known_outputs(root)


@pytest.mark.parametrize(
    ("flag", "function_name", "status"),
    (
        ("--theta-zero-parity", "run_m2", "PASS_M2"),
        ("--resource-rebenchmark-only", "run_m3", "PASS_M3"),
        ("--single-smokes", "run_single_smokes", "PASS_M4"),
    ),
)
def test_cli_dispatches_bounded_modes_to_separate_targets_without_cross_mode_updates(
    monkeypatch: pytest.MonkeyPatch, flag: str, function_name: str, status: str
) -> None:
    root = _task_root(f"dispatch-{function_name}")
    origin = root / "source-origin.json"
    origin.write_text("placeholder")
    os.chmod(origin, 0o600)
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "_common_preflight",
        lambda _args: (
            {
                "public_inputs": {"status": "PASS_SIX_PUBLIC_DAY26_INPUTS"},
                "config": {"schema_version": g4.SCHEMA_VERSION},
            },
            "a" * 64,
        ),
    )

    def target(*_args: object) -> dict[str, str]:
        calls.append(function_name)
        return {"status": status}

    monkeypatch.setattr(g4, function_name, target)
    try:
        args = cli.parse_args(
            (
                flag,
                "--config",
                str(CONFIG_PATH),
                "--source-origin-record",
                str(origin),
                "--output-dir",
                str(root / "output"),
            )
        )
        result = cli.run(args)
        assert result["status"] == status
        assert calls == [function_name]
    finally:
        _remove_known_outputs(root)


def test_joint_smoke_dispatch_pins_exactly_ten_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _task_root("dispatch-joint")
    origin = root / "source-origin.json"
    origin.write_text("placeholder")
    os.chmod(origin, 0o600)
    observed: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        cli,
        "_common_preflight",
        lambda _args: (
            {
                "public_inputs": {"status": "PASS_SIX_PUBLIC_DAY26_INPUTS"},
                "config": {"schema_version": g4.SCHEMA_VERSION},
            },
            "b" * 64,
        ),
    )

    def target(_root: Path, config_sha256: str, output: Path) -> dict[str, str]:
        observed.append((config_sha256, output))
        return {"status": "PASS_M5"}

    monkeypatch.setattr(g4, "run_joint_smoke", target)
    try:
        args = cli.parse_args(
            (
                "--joint-smoke",
                "--max-updates-this-process",
                "10",
                "--config",
                str(CONFIG_PATH),
                "--source-origin-record",
                str(origin),
                "--output-dir",
                str(root / "output"),
            )
        )
        result = cli.run(args)
        assert result["status"] == "PASS_M5"
        assert observed == [("b" * 64, root / "output")]
    finally:
        _remove_known_outputs(root)


def test_private_resume_worker_requires_exact_paths_and_no_public_output_dir() -> None:
    valid = cli.parse_args(
        (
            "--resume-worker-checkpoint",
            "/tmp/gr-g4-004-x/output/split-checkpoint-10.json",
            "--resume-worker-output",
            "/tmp/gr-g4-004-x/output/resume-worker-result.json",
            "--config",
            str(CONFIG_PATH),
            "--source-origin-record",
            "/tmp/gr-g4-004-x/source-origin.json",
        )
    )
    assert valid.resume_worker_checkpoint.name == "split-checkpoint-10.json"
    with pytest.raises(SystemExit):
        cli.parse_args(
            (
                "--resume-worker-checkpoint",
                "/tmp/gr-g4-004-x/output/split-checkpoint-10.json",
                "--resume-worker-output",
                "/tmp/gr-g4-004-x/output/resume-worker-result.json",
                "--config",
                str(CONFIG_PATH),
                "--source-origin-record",
                "/tmp/gr-g4-004-x/source-origin.json",
                "--output-dir",
                "/tmp/gr-g4-004-x/output",
            )
        )


def test_cli_output_rejects_symlink_case_separator_and_existing_paths(tmp_path: Path) -> None:
    with pytest.raises(g4.G4ContractError):
        g4.validate_task_output_path(tmp_path / "output")
    with pytest.raises(g4.G4ContractError):
        g4.validate_task_output_path(Path("/tmp/gr-G4-004-case/output"))
    with pytest.raises(g4.G4ContractError):
        g4.validate_task_output_path(Path("/tmp/gr-g4-004-case/not-output"))

    root = _task_root("symlink-output-integration")
    try:
        target = root / "real"
        target.mkdir()
        output = root / "output"
        output.symlink_to(target, target_is_directory=True)
        with pytest.raises(g4.G4ContractError, match="absent"):
            g4.validate_task_output_path(output)
        output.unlink()
        target.rmdir()
    finally:
        root.rmdir()


def test_public_longrun_and_resume_parser_contract() -> None:
    common = (
        "--config",
        str(CONFIG_PATH),
        "--source-origin-record",
        "/tmp/gr-g4-004-x/source-origin.json",
        "--output-dir",
        "/tmp/gr-g4-004-x/output",
        "--max-updates-this-process",
        "3",
    )
    fresh = cli.parse_args(("--longrun", *common))
    resumed = cli.parse_args(
        (
            "--resume-checkpoint",
            "/tmp/gr-g4-004-x/output/checkpoints/checkpoint-step-000002.json",
            *common,
        )
    )

    assert fresh.longrun and fresh.max_updates_this_process == 3
    assert resumed.resume_checkpoint.name == "checkpoint-step-000002.json"
    for value in ("0", "101"):
        with pytest.raises(SystemExit):
            cli.parse_args(("--longrun", *common[:-1], value))
    with pytest.raises(SystemExit):
        cli.parse_args(("--longrun", *common[:-2]))
    with pytest.raises(SystemExit):
        cli.parse_args(
            (
                "--longrun",
                "--resume-checkpoint",
                "/tmp/gr-g4-004-x/output/checkpoints/checkpoint-step-000002.json",
                *common,
            )
        )
    with pytest.raises(SystemExit):
        cli.parse_args(("--dry-run", *common))


def test_public_longrun_cli_dispatches_bounded_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[Path, str, Path, int, None]] = []
    monkeypatch.setattr(
        cli,
        "_common_preflight",
        lambda _args: (
            {
                "public_inputs": {"status": "PASS_SIX_PUBLIC_DAY26_INPUTS"},
                "config": {"schema_version": g4.SCHEMA_VERSION},
            },
            "a" * 64,
        ),
    )

    def target(
        repository_root: Path,
        config_sha256: str,
        output: Path,
        additional_updates: int,
        *,
        resume_checkpoint: None,
    ) -> dict[str, object]:
        observed.append(
            (repository_root, config_sha256, output, additional_updates, resume_checkpoint)
        )
        return {
            "schema_version": "crazyflow.mellinger_g4_longrun_segment.v1",
            "status": "PASS_BOUNDED_LONGRUN_SEGMENT",
            "expected_start_count": 0,
            "actual_start_count": 0,
            "expected_end_count": 2,
            "actual_end_count": 2,
            "completed_updates": 2,
        }

    monkeypatch.setattr(g4, "run_bounded_longrun_segment", target, raising=False)
    args = cli.parse_args(
        (
            "--longrun",
            "--config",
            str(CONFIG_PATH),
            "--source-origin-record",
            "/tmp/gr-g4-004-x/source-origin.json",
            "--output-dir",
            "/tmp/gr-g4-004-x/output",
            "--max-updates-this-process",
            "2",
        )
    )
    result = cli.run(args)

    assert result["status"] == "PASS_BOUNDED_LONGRUN_SEGMENT"
    assert observed == [(REPOSITORY_ROOT, "a" * 64, Path("/tmp/gr-g4-004-x/output"), 2, None)]


def test_public_resume_cli_reuses_exact_owned_output_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = Path("/tmp/gr-g4-004-x/output/checkpoints/checkpoint-step-000002.json")
    observed: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        cli,
        "_common_preflight",
        lambda _args: (
            {
                "public_inputs": {"status": "PASS_SIX_PUBLIC_DAY26_INPUTS"},
                "config": {"schema_version": g4.SCHEMA_VERSION},
            },
            "b" * 64,
        ),
    )

    def target(
        _repository_root: Path,
        _config_sha256: str,
        output: Path,
        additional_updates: int,
        *,
        resume_checkpoint: Path,
    ) -> dict[str, object]:
        observed.append((resume_checkpoint, additional_updates))
        assert output == Path("/tmp/gr-g4-004-x/output")
        return {
            "schema_version": "crazyflow.mellinger_g4_longrun_segment.v1",
            "status": "PASS_BOUNDED_LONGRUN_SEGMENT",
            "expected_start_count": 2,
            "actual_start_count": 2,
            "expected_end_count": 3,
            "actual_end_count": 3,
            "completed_updates": 1,
        }

    monkeypatch.setattr(g4, "run_bounded_longrun_segment", target, raising=False)
    args = cli.parse_args(
        (
            "--resume-checkpoint",
            str(checkpoint),
            "--config",
            str(CONFIG_PATH),
            "--source-origin-record",
            "/tmp/gr-g4-004-x/source-origin.json",
            "--output-dir",
            "/tmp/gr-g4-004-x/output",
            "--max-updates-this-process",
            "1",
        )
    )
    result = cli.run(args)

    assert result["status"] == "PASS_BOUNDED_LONGRUN_SEGMENT"
    assert observed == [(checkpoint, 1)]


def test_tiny_continuous_and_resumed_segments_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuous_root = Path("/tmp") / f"gr-g4-004-m7-continuous-{os.getpid()}"
    resumed_root = Path("/tmp") / f"gr-g4-004-m7-resumed-{os.getpid()}"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    for root in (continuous_root, resumed_root):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(mode=0o700)

    def fake_update(
        state: g4.OptimizationState, names: tuple[str, ...]
    ) -> tuple[g4.OptimizationState, dict[str, object]]:
        updated, optimizer = g4.projected_adam_step(
            state, np.asarray([0.1, -0.2], dtype=np.float32), names
        )
        count = updated.adam.count
        updated = replace(updated, loss_history=state.loss_history + (float(count),))
        operational_token = count + len(os.environ)
        parent_ids = [f"p-{count}-{index}" for index in range(32)]
        return updated, {
            "scientific": {
                "update": count,
                "batch_index": state.adam.count % 9,
                "loss": float(count),
                "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
                "gradient": [0.1, -0.2],
                "gradient_norm": float(np.hypot(0.1, 0.2)),
                "parent_ids": parent_ids,
                "strata": [f"s-{index % 9}" for index in range(32)],
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
                "optimizer": optimizer,
                "projection_count_total": 0,
            },
            "operational": _longrun_operational_update(operational_token, parent_ids),
        }

    monkeypatch.setattr(g4, "parent_population_evidence", lambda _root: {})
    monkeypatch.setattr(g4, "science_fingerprints", lambda *_args: fingerprints)
    monkeypatch.setattr(g4, "one_longrun_update", fake_update, raising=False)
    try:
        continuous = g4.run_bounded_longrun_segment(
            REPOSITORY_ROOT, "c" * 64, continuous_root / "output", 2, resume_checkpoint=None
        )
        first = g4.run_bounded_longrun_segment(
            REPOSITORY_ROOT, "c" * 64, resumed_root / "output", 1, resume_checkpoint=None
        )
        resumed = g4.run_bounded_longrun_segment(
            REPOSITORY_ROOT,
            "c" * 64,
            resumed_root / "output",
            1,
            resume_checkpoint=Path(first["end_checkpoint"]),
        )
        continuous_payload = json.loads(Path(continuous["end_checkpoint"]).read_bytes())
        resumed_payload = json.loads(Path(resumed["end_checkpoint"]).read_bytes())

        assert (
            continuous_payload["scientific_payload_sha256"]
            == resumed_payload["scientific_payload_sha256"]
        )
        assert continuous_payload["theta"] == resumed_payload["theta"]
        assert continuous_payload["adam"] == resumed_payload["adam"]
        assert continuous_payload["payload_sha256"] != resumed_payload["payload_sha256"]
        assert g4.LONGRUN_RESOURCE_EVIDENCE_SCHEMA_VERSION == LONGRUN_RESOURCE_SCHEMA
        for payload in (continuous_payload, resumed_payload):
            operational = payload["operational_update"]
            assert operational["schema_version"] == LONGRUN_RESOURCE_SCHEMA
            assert operational["resource_observation_count"] == 16
            assert len(operational["microbatches"]) == 8
            assert all(
                set(microbatch)
                == {
                    "microbatch_index",
                    "elapsed_seconds",
                    "parent_ids",
                    "resource_before",
                    "resource_after",
                }
                for microbatch in operational["microbatches"]
            )
            restored, loaded = g4.load_longrun_checkpoint(
                Path(continuous["end_checkpoint"])
                if payload is continuous_payload
                else Path(resumed["end_checkpoint"]),
                fingerprints,
                run_contract_sha256=payload["run_contract_sha256"],
                expected_parent_payload_sha256=payload["parent_checkpoint_payload_sha256"],
                expected_parent_scientific_sha256=payload["parent_checkpoint_scientific_sha256"],
            )
            assert restored.adam.count == loaded["update_count"] == 2
        assert (
            continuous_payload["parent_checkpoint_scientific_sha256"]
            == resumed_payload["parent_checkpoint_scientific_sha256"]
        )
    finally:
        shutil.rmtree(continuous_root)
        shutil.rmtree(resumed_root)


def test_public_longrun_partial_completion_handoff_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp") / f"gr-g4-004-m7-handoff-{os.getpid()}"
    output = root / "output"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700)

    def fake_update(
        state: g4.OptimizationState, names: tuple[str, ...]
    ) -> tuple[g4.OptimizationState, dict[str, object]]:
        updated, optimizer = g4.projected_adam_step(
            state, np.asarray([0.1, -0.2], dtype=np.float32), names
        )
        count = updated.adam.count
        parent_ids = [f"p-{count}-{index}" for index in range(32)]
        return updated, {
            "scientific": {
                "update": count,
                "batch_index": state.adam.count % 9,
                "loss": float(count),
                "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
                "gradient": [0.1, -0.2],
                "gradient_norm": float(np.hypot(0.1, 0.2)),
                "parent_ids": parent_ids,
                "strata": [f"s-{index % 9}" for index in range(32)],
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
                "optimizer": optimizer,
                "projection_count_total": 0,
            },
            "operational": _longrun_operational_update(count, parent_ids),
        }

    monkeypatch.setattr(g4, "parent_population_evidence", lambda _root: {})
    monkeypatch.setattr(g4, "science_fingerprints", lambda *_args: fingerprints)
    monkeypatch.setattr(g4, "one_longrun_update", fake_update, raising=False)
    try:
        result = g4.run_bounded_longrun_segment(
            REPOSITORY_ROOT, "c" * 64, output, 2, resume_checkpoint=None
        )
        segment_path = Path(result["segment_record"])
        payload = json.loads(segment_path.read_bytes())

        assert payload["status"] == "PASS_BOUNDED_LONGRUN_SEGMENT"
        assert payload["expected_start_count"] == payload["actual_start_count"] == 0
        assert payload["expected_end_count"] == payload["actual_end_count"] == 2
        assert payload["completed_updates"] == 2
        assert payload["remaining_updates_to_100"] == 98
        assert len(payload["new_checkpoints"]) == 2
        assert payload["start_checkpoint"] == result["start_checkpoint"]
        assert payload["end_checkpoint"] == result["end_checkpoint"]
        assert stat.S_IMODE(segment_path.stat().st_mode) == 0o600
    finally:
        shutil.rmtree(root)


def test_public_m8_cli_modes_are_separate_and_m7_dispatch_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, Path | None]] = []
    monkeypatch.setattr(
        cli, "_common_preflight", lambda _args: ({"public_inputs": {}, "config": {}}, "a" * 64)
    )
    monkeypatch.setattr(
        g4,
        "run_bounded_m8_segment",
        lambda *_args, resume_checkpoint: (
            observed.append(("m8", resume_checkpoint)) or {"status": "PASS_M8_BOUNDED_SEGMENT"}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        g4,
        "run_bounded_longrun_segment",
        lambda *_args, resume_checkpoint: (
            observed.append(("m7", resume_checkpoint)) or {"status": "PASS_BOUNDED_LONGRUN_SEGMENT"}
        ),
    )
    common = (
        "--config",
        str(CONFIG_PATH),
        "--source-origin-record",
        "/tmp/gr-g4-004-x/source-origin.json",
        "--output-dir",
        "/tmp/gr-g4-004-x/output",
        "--max-updates-this-process",
        "1",
    )
    assert cli.run(cli.parse_args(("--m8-longrun", *common)))["status"] == "PASS_M8_BOUNDED_SEGMENT"
    checkpoint = "/tmp/gr-g4-004-x/output/checkpoints/m8-checkpoint-step-000001.json"
    assert (
        cli.run(cli.parse_args(("--m8-resume-checkpoint", checkpoint, *common)))["status"]
        == "PASS_M8_BOUNDED_SEGMENT"
    )
    assert (
        cli.run(cli.parse_args(("--longrun", *common)))["status"] == "PASS_BOUNDED_LONGRUN_SEGMENT"
    )
    assert [mode for mode, _path in observed] == ["m8", "m8", "m7"]
    assert observed[1][1] == Path(checkpoint)
    assert g4.LONGRUN_MAX_UPDATES == 100


def test_public_m8_tiny_continuous_and_resumed_segments_are_scientifically_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert g4.M8_MAX_UPDATES == 2000
    continuous_root = Path("/tmp") / f"gr-g4-004-m8-continuous-{os.getpid()}"
    resumed_root = Path("/tmp") / f"gr-g4-004-m8-resumed-{os.getpid()}"
    fingerprints = {"sources": {"core": "a" * 64}, "config_sha256": "b" * 64}
    for root in (continuous_root, resumed_root):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(mode=0o700)

    def validation(
        _root: Path, theta: np.ndarray, update_count: int, previous: dict[str, object] | None
    ) -> dict[str, object]:
        return g4.m8_validation_evidence_payload(
            update_count,
            theta,
            g4.synthetic_m8_validation_scientific(float(100 - update_count)),
            g4.synthetic_m8_validation_operational(update_count),
            previous_validation_state=previous,
            convergence_state=g4.convergence_status([], []),
        )

    def update(state: g4.M8OptimizationState) -> tuple[g4.M8OptimizationState, dict[str, object]]:
        legacy = g4.OptimizationState(state.theta, state.adam, state.projection_count, (), 0, None)
        updated, optimizer = g4.projected_adam_step(
            legacy, np.asarray([0.1, -0.2], dtype=np.float32), g4.PARAMETER_NAMES
        )
        parent_ids = [f"p-{updated.adam.count}-{index}" for index in range(32)]
        evidence = {
            "scientific": {
                "update": updated.adam.count,
                "batch_index": state.adam.count % 9,
                "loss": float(updated.adam.count),
                "loss_terms": {name: 0.0 for name in g4.LOSS_TERM_NAMES},
                "gradient": [0.1, -0.2],
                "gradient_norm": float(np.hypot(0.1, 0.2)),
                "parent_ids": parent_ids,
                "strata": [f"s-{index % 9}" for index in range(32)],
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
                "optimizer": optimizer,
                "projection_count_total": updated.projection_count,
            },
            "operational": _longrun_operational_update(
                updated.adam.count + len(os.environ), parent_ids
            ),
        }
        return g4.one_m8_update_from_evidence(state, evidence)

    monkeypatch.setattr(g4, "parent_population_evidence", lambda _root: {})
    monkeypatch.setattr(g4, "science_fingerprints", lambda *_args: fingerprints)
    monkeypatch.setattr(g4, "evaluate_m8_validation", validation, raising=False)
    monkeypatch.setattr(g4, "one_m8_update", update, raising=False)
    try:
        continuous = g4.run_bounded_m8_segment(
            REPOSITORY_ROOT, "c" * 64, continuous_root / "output", 2, resume_checkpoint=None
        )
        first = g4.run_bounded_m8_segment(
            REPOSITORY_ROOT, "c" * 64, resumed_root / "output", 1, resume_checkpoint=None
        )
        resumed = g4.run_bounded_m8_segment(
            REPOSITORY_ROOT,
            "c" * 64,
            resumed_root / "output",
            1,
            resume_checkpoint=Path(first["end_checkpoint"]),
        )
        continuous_payload = json.loads(Path(continuous["end_checkpoint"]).read_bytes())
        resumed_payload = json.loads(Path(resumed["end_checkpoint"]).read_bytes())
        assert (
            continuous_payload["scientific_payload_sha256"]
            == resumed_payload["scientific_payload_sha256"]
        )
        assert (
            continuous_payload["operational_payload_sha256"]
            != resumed_payload["operational_payload_sha256"]
        )
        assert continuous_payload["theta"] == resumed_payload["theta"]
        assert continuous_payload["history_state"] == resumed_payload["history_state"]
        assert continuous_payload["segment_coordinates"] == {"start_update": 0, "end_update": 2}
        assert resumed_payload["segment_coordinates"] == {"start_update": 1, "end_update": 2}
        assert (
            continuous_payload["parent_checkpoint_scientific_sha256"]
            == resumed_payload["parent_checkpoint_scientific_sha256"]
        )
        assert resumed["status"] == "PASS_M8_BOUNDED_SEGMENT"
        assert resumed["completed_updates"] == 1
    finally:
        shutil.rmtree(continuous_root)
        shutil.rmtree(resumed_root)


def test_public_backend_parity_and_throughput_argv_are_strict_and_eval_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        cli,
        "_backend_preflight",
        lambda _args: (
            {"schema_version": "manifest"},
            {"profile": "local_cpu"},
            "a" * 64,
            "b" * 64,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        g4,
        "run_backend_parity",
        lambda **kwargs: (
            observed.append(("parity", kwargs)) or {"status": "PASS_BACKEND_PARITY_SAMPLE"}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        g4,
        "run_backend_throughput",
        lambda **kwargs: (
            observed.append(("throughput", kwargs)) or {"status": "PASS_BACKEND_THROUGHPUT_SAMPLE"}
        ),
        raising=False,
    )
    common = (
        "--backend",
        "cpu",
        "--runtime-profile",
        "local_cpu",
        "--runtime-contract",
        str(tmp_path / "runtime.json"),
        "--input-manifest",
        str(REPOSITORY_ROOT / "configs/research/mellinger/g4_backend_evidence_v1.json"),
    )
    parity = cli.parse_args(
        ("--backend-parity", *common, "--output", str(tmp_path / "parity.json"))
    )
    assert cli.run(parity)["status"] == "PASS_BACKEND_PARITY_SAMPLE"
    throughput = cli.parse_args(
        (
            "--backend-throughput",
            *common,
            "--phase",
            "steady",
            "--sample-index",
            "3",
            "--effective-worlds",
            "32",
            "--microbatches",
            "8",
            "--worlds-per-microbatch",
            "4",
            "--output",
            str(tmp_path / "throughput.json"),
        )
    )
    assert cli.run(throughput)["status"] == "PASS_BACKEND_THROUGHPUT_SAMPLE"
    assert [mode for mode, _kwargs in observed] == ["parity", "throughput"]
    with pytest.raises(SystemExit):
        cli.parse_args(("--backend-parity", *common, "--phase", "warmup"))
    with pytest.raises(SystemExit):
        cli.parse_args(
            (
                "--backend-throughput",
                *common,
                "--phase",
                "warmup",
                "--sample-index",
                "2",
                "--effective-worlds",
                "32",
                "--microbatches",
                "8",
                "--worlds-per-microbatch",
                "4",
                "--output",
                str(tmp_path / "invalid.json"),
            )
        )
    assert "eval(" not in Path(cli.__file__).read_text()


def test_public_backend_gpu_checkpoint_resume_argv_is_strict_without_hardware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[Path | None] = []
    parent_ids = [f"train-parent-{index:02d}" for index in range(32)]
    validated_manifest = g4.load_backend_input_manifest(
        REPOSITORY_ROOT / "configs/research/mellinger/g4_backend_evidence_v1.json", REPOSITORY_ROOT
    )
    monkeypatch.setattr(
        cli,
        "_backend_preflight",
        lambda _args: (validated_manifest, {"profile": "remote_pixi_gpu"}, "a" * 64, "b" * 64),
        raising=False,
    )

    def fake_segment(**kwargs: object) -> dict[str, object]:
        observed.append(kwargs["resume_checkpoint"])
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        checkpoints = output / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        start = 10 if kwargs["resume_checkpoint"] is not None else 0
        end = start + int(kwargs["max_updates_this_process"])
        start_path = checkpoints / f"m8-checkpoint-step-{start:06d}.json"
        end_path = checkpoints / f"m8-checkpoint-step-{end:06d}.json"
        scientific_state = {
            "theta": {"dtype": "<f4", "shape": [2], "data_hex": "0000000000000000"},
            "adam": {"count": end},
            "rng_coordinates": {"next_batch_index": end % 9},
            "history_state": {"loss_window": [1.0], "recent_gradient_window": [0.5]},
            "projection_count": 0,
            "scientific_update": {"parent_ids": parent_ids, "gradient": [0.1, -0.2]},
        }
        digests = {
            "payload_sha256": "d" * 64,
            "scientific_payload_sha256": "e" * 64,
            "operational_payload_sha256": "f" * 64,
        }
        start_payload = {"update_count": start, **scientific_state, **digests}
        end_payload = {"update_count": end, **scientific_state, **digests}
        start_path.write_text(json.dumps(start_payload))
        end_path.write_text(json.dumps(end_payload))
        start_path.chmod(0o600)
        end_path.chmod(0o600)
        return {
            "status": "PASS_BACKEND_CHECKPOINT_SEGMENT",
            "m8_segment": {
                "actual_start_count": start,
                "actual_end_count": end,
                "start_checkpoint": str(start_path),
                "end_checkpoint": str(end_path),
                "start_checkpoint_digests": digests,
                "end_checkpoint_digests": digests,
                "run_contract_sha256": "1" * 64,
                "fingerprints": {"science": "2" * 64},
                "operational_summary": {
                    "maximum_rss_bytes": 1024,
                    "minimum_mem_available_bytes": 16 * 1024**3,
                    "maximum_swap_used_bytes": 0,
                },
            },
        }

    monkeypatch.setattr(g4, "run_backend_checkpoint_segment", fake_segment, raising=False)
    output = tmp_path / "output"
    common = (
        "--backend",
        "gpu",
        "--runtime-profile",
        "remote_pixi_gpu",
        "--runtime-contract",
        "/tmp/gr-g4-004-backend/runtime.json",
        "--input-manifest",
        str(REPOSITORY_ROOT / "configs/research/mellinger/g4_backend_evidence_v1.json"),
        "--output-dir",
        str(output),
    )
    fresh = cli.parse_args(
        ("--backend-checkpoint-run", *common, "--max-updates-this-process", "20")
    )
    assert cli.run(fresh)["status"] == "PASS_BACKEND_CHECKPOINT_SEGMENT"
    checkpoint = str(output / "checkpoints/m8-checkpoint-step-000010.json")
    resumed = cli.parse_args(
        ("--backend-checkpoint-resume", checkpoint, *common, "--max-updates-this-process", "10")
    )
    assert cli.run(resumed)["status"] == "PASS_BACKEND_CHECKPOINT_SEGMENT"
    assert observed == [None, Path(checkpoint)]
    with pytest.raises(SystemExit):
        cli.parse_args(
            (
                "--backend-checkpoint-resume",
                "relative-checkpoint.json",
                *common,
                "--max-updates-this-process",
                "10",
            )
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            ("--backend-checkpoint-resume", checkpoint, *common, "--max-updates-this-process", "20")
        )


def test_public_backend_throughput_observes_gpu_device_memory_v1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeGpuDevice:
        platform = "gpu"

        def memory_stats(self) -> dict[str, int]:
            return {"bytes_in_use": 1024, "peak_bytes_in_use": 2048, "bytes_limit": 4096}

    parent_ids = [f"train-parent-{index:02d}" for index in range(32)]
    updated = g4.initial_optimization_state(2)
    updated = replace(updated, adam=replace(updated.adam, count=1))
    monkeypatch.setattr(
        g4,
        "one_effective_batch",
        lambda *_args: (
            updated,
            {
                "parent_ids": parent_ids,
                "microbatch_parent_counts": [4] * 8,
                "microbatch_weights": [0.125] * 8,
                "loss": np.float32(1.0),
                "gradient": np.asarray([0.1, -0.2], dtype=np.float32),
                "gradient_norm": np.float32(np.hypot(0.1, 0.2)),
            },
        ),
    )
    monkeypatch.setattr(g4.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(g4.jax, "devices", lambda *_args, **_kwargs: [FakeGpuDevice()])
    monkeypatch.setattr(
        cli,
        "_backend_preflight",
        lambda _args: ({"immutable_product": [{"sha256": "a" * 64}]}, {}, "b" * 64, "c" * 64),
    )
    monkeypatch.setattr(
        g4,
        "memory_snapshot",
        lambda: _longrun_operational_update(0, parent_ids)["microbatches"][0]["resource_before"],
    )
    output = tmp_path / "throughput.json"
    args = cli.parse_args(
        (
            "--backend-throughput",
            "--backend",
            "gpu",
            "--runtime-profile",
            "remote_pixi_gpu",
            "--runtime-contract",
            str(tmp_path / "runtime.json"),
            "--input-manifest",
            str(tmp_path / "manifest.json"),
            "--phase",
            "steady",
            "--sample-index",
            "1",
            "--effective-worlds",
            "32",
            "--microbatches",
            "8",
            "--worlds-per-microbatch",
            "4",
            "--output",
            str(output),
        )
    )

    result = cli.run(args)

    assert result["operational"]["device_memory"] == {
        "bytes_in_use": 1024,
        "peak_bytes_in_use": 2048,
        "bytes_limit": 4096,
    }


def test_public_backend_preflight_compares_independent_runtime_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_path = tmp_path / "runtime.json"
    expected = {"schema_version": "expected", "repository": {"head": "a" * 40}}
    observed = {"schema_version": "observed", "repository": {"head": "b" * 40}}
    runtime_path.write_text(json.dumps(expected))
    runtime_path.chmod(0o600)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"{}")
    captured: dict[str, object] = {}
    monkeypatch.setattr(g4, "load_backend_input_manifest", lambda *_args: {"manifest": True})
    monkeypatch.setattr(cli, "_observe_backend_runtime", lambda _args: observed)

    def validate(payload: dict[str, object], **kwargs: object) -> dict[str, object]:
        captured["payload"] = payload
        captured.update(kwargs)
        return payload

    monkeypatch.setattr(g4, "validate_backend_runtime_payload", validate)
    args = cli.parse_args(
        (
            "--backend-parity",
            "--backend",
            "cpu",
            "--runtime-profile",
            "local_cpu",
            "--runtime-contract",
            str(runtime_path),
            "--input-manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "parity.json"),
        )
    )

    cli._backend_preflight(args)

    assert captured["payload"] == expected
    assert captured["observed_payload"] is observed
    assert captured["payload"] is not captured["observed_payload"]


def test_public_backend_checkpoint_modes_emit_resume_v1_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "_backend_preflight",
        lambda _args: (
            {"immutable_product": [{"sha256": "a" * 64}]},
            {"profile": "remote_pixi_gpu"},
            "b" * 64,
            "c" * 64,
        ),
    )

    def fake_segment(**kwargs: object) -> dict[str, object]:
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        checkpoints = output / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        start = 10 if kwargs["resume_checkpoint"] is not None else 0
        end = start + int(kwargs["max_updates_this_process"])
        start_path = checkpoints / f"m8-checkpoint-step-{start:06d}.json"
        end_path = checkpoints / f"m8-checkpoint-step-{end:06d}.json"
        scientific_state = {
            "theta": {"dtype": "<f4", "shape": [2], "data_hex": "0000000000000000"},
            "adam": {"count": end},
            "rng_coordinates": {"next_batch_index": end % 9},
            "history_state": {"loss_window": [1.0], "recent_gradient_window": [0.5]},
            "projection_count": 0,
            "scientific_update": {"parent_ids": parent_ids, "gradient": [0.1, -0.2]},
        }
        digests = {
            "payload_sha256": "d" * 64,
            "scientific_payload_sha256": "e" * 64,
            "operational_payload_sha256": "f" * 64,
        }
        start_payload = {"update_count": start, **scientific_state, **digests}
        end_payload = {"update_count": end, **scientific_state, **digests}
        start_path.write_text(json.dumps(start_payload))
        end_path.write_text(json.dumps(end_payload))
        start_path.chmod(0o600)
        end_path.chmod(0o600)
        return {
            "status": "PASS_BACKEND_CHECKPOINT_SEGMENT",
            "m8_segment": {
                "actual_start_count": start,
                "actual_end_count": end,
                "start_checkpoint": str(start_path),
                "end_checkpoint": str(end_path),
                "start_checkpoint_digests": digests,
                "end_checkpoint_digests": digests,
                "run_contract_sha256": "1" * 64,
                "fingerprints": {"science": "2" * 64},
                "operational_summary": {
                    "maximum_rss_bytes": 1024,
                    "minimum_mem_available_bytes": 16 * 1024**3,
                    "maximum_swap_used_bytes": 0,
                },
            },
        }

    parent_ids = [f"train-parent-{index:02d}" for index in range(32)]
    monkeypatch.setattr(g4, "run_backend_checkpoint_segment", fake_segment)
    common = (
        "--backend",
        "gpu",
        "--runtime-profile",
        "remote_pixi_gpu",
        "--runtime-contract",
        str(tmp_path / "runtime.json"),
        "--input-manifest",
        str(tmp_path / "manifest.json"),
    )
    continuous_output = tmp_path / "continuous-output"
    continuous = cli.parse_args(
        (
            "--backend-checkpoint-run",
            *common,
            "--output-dir",
            str(continuous_output),
            "--max-updates-this-process",
            "20",
        )
    )
    cli.run(continuous)
    continuous_record = json.loads(
        (continuous_output / "backend-resume-continuous-v1.json").read_bytes()
    )
    split_output = tmp_path / "split-output"
    first = cli.parse_args(
        (
            "--backend-checkpoint-run",
            *common,
            "--output-dir",
            str(split_output),
            "--max-updates-this-process",
            "10",
        )
    )
    cli.run(first)
    assert not (split_output / "backend-resume-resume-v1.json").exists()
    checkpoint = split_output / "checkpoints/m8-checkpoint-step-000010.json"
    resumed = cli.parse_args(
        (
            "--backend-checkpoint-resume",
            str(checkpoint),
            *common,
            "--output-dir",
            str(split_output),
            "--max-updates-this-process",
            "10",
        )
    )
    cli.run(resumed)
    resumed_record = json.loads((split_output / "backend-resume-resume-v1.json").read_bytes())
    assert continuous_record["schema_version"] == g4.BACKEND_RESUME_SCHEMA_VERSION
    assert continuous_record["mode"] == "continuous"
    assert resumed_record["mode"] == "resume"
    assert continuous_record["scientific"] == resumed_record["scientific"]
    assert (
        continuous_record["scientific_payload_sha256"]
        == resumed_record["scientific_payload_sha256"]
    )
    assert (
        continuous_record["operational_payload_sha256"]
        != resumed_record["operational_payload_sha256"]
    )


def test_public_backend_parity_propagates_gpu_to_parents_and_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str]] = []
    parent_ids = [f"parent-{index:02d}" for index in range(32)]
    evaluation = {
        "parent_ids": parent_ids,
        "loss": 1.0,
        "gradient": np.asarray([0.1, -0.2], dtype=np.float32),
        "aux": {f"loss_{name}": np.zeros(32, dtype=np.float32) for name in g4.LOSS_TERM_NAMES},
    }
    monkeypatch.setattr(
        g4,
        "fixed_day26_items",
        lambda _root, _ids, *, backend: observed.append(("parents", backend)) or (),
    )
    monkeypatch.setattr(
        g4,
        "_batch_items",
        lambda _index, *, backend, repository_root: observed.append(("batch", backend)) or (),
    )
    monkeypatch.setattr(
        g4,
        "evaluate_items",
        lambda _theta, _items, *, backend: observed.append(("evaluation", backend)) or evaluation,
    )
    monkeypatch.setattr(
        g4,
        "default_parity",
        lambda _root, identities, *, include_backend_evidence, backend: (
            observed.append(("simulation", backend))
            or {
                "parent_ids": list(identities),
                "loss": 1.0,
                "_backend_evidence": {
                    "canonical_final": {"state": np.zeros(1, dtype=np.float32)},
                    "left_loss": np.float32(1.0),
                    "left_aux": {
                        **{
                            f"loss_{name}": np.zeros(4, dtype=np.float32)
                            for name in g4.LOSS_TERM_NAMES
                        },
                        **{
                            f"metric_{name}": np.zeros(4, dtype=np.float32)
                            for name in (
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
                        },
                    },
                },
            }
        ),
    )

    result = g4.run_backend_parity(
        backend="gpu",
        repository_root=REPOSITORY_ROOT,
        provenance={"input_manifest_sha256": "a" * 64, "runtime_contract_sha256": "b" * 64},
        runtime={"profile": "remote_pixi_gpu"},
        resource={"before": {}, "after": {}},
    )

    assert result["backend"] == "gpu"
    assert observed == [
        ("parents", "gpu"),
        ("batch", "gpu"),
        ("evaluation", "gpu"),
        ("simulation", "gpu"),
    ]


def test_public_backend_throughput_propagates_gpu_to_one_effective_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    state = g4.initial_optimization_state(2)
    updated = replace(state, adam=replace(state.adam, count=1))
    record = {
        "parent_ids": [f"parent-{index:02d}" for index in range(32)],
        "microbatch_parent_counts": [4] * 8,
        "microbatch_weights": [0.125] * 8,
        "loss": 1.0,
        "gradient": [0.1, -0.2],
        "gradient_norm": float(np.hypot(0.1, 0.2)),
    }
    monkeypatch.setattr(
        g4,
        "one_effective_batch",
        lambda _state, _names, *, backend, repository_root: (
            observed.append(backend) or (updated, record)
        ),
    )
    monkeypatch.setattr(g4.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        g4,
        "observe_backend_device_memory",
        lambda backend: (
            {"bytes_in_use": 1, "peak_bytes_in_use": 2, "bytes_limit": 3}
            if backend == "gpu"
            else pytest.fail("backend changed")
        ),
    )

    result = g4.run_backend_throughput(
        backend="gpu",
        phase="steady",
        sample_index=1,
        provenance={"input_manifest_sha256": "a" * 64, "runtime_contract_sha256": "b" * 64},
        runtime={"profile": "remote_pixi_gpu"},
        resource={"before": {}, "after": {}},
    )

    assert observed == ["gpu"]
    assert result["backend"] == "gpu"


def test_public_backend_parent_payload_uses_supplied_relocated_repository_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    relocated_root = tmp_path / "relocated-public-root"
    relative_paths = (
        g4.BACKEND_NEUTRAL_PARENT_RELATIVE_PATH,
        g4.BACKEND_INPUT_MANIFEST_RELATIVE_PATH,
        f"{g4.DAY26_DIRECTORY}/train_manifest.json",
        f"{g4.DAY26_DIRECTORY}/validation_manifest.json",
    )
    for relative_path in relative_paths:
        source = REPOSITORY_ROOT / relative_path
        destination = relocated_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    expected_root = relocated_root.resolve(strict=True)
    original_cached_items = g4._cached_backend_parent_items
    observed_payload_roots: list[Path] = []

    def relocated_items(
        repository_root: str, backend: str
    ) -> tuple[tuple[g4.G4ParentSpec | object, object], ...]:
        actual_root = Path(repository_root).resolve(strict=True)
        observed_payload_roots.append(actual_root)
        if actual_root != expected_root:
            raise g4.G4ContractError("backend parent payload did not use supplied root")
        return original_cached_items(str(actual_root), backend)

    def fake_evaluate(
        _theta: object,
        items: tuple[tuple[object, object], ...],
        _names: object = g4.PARAMETER_NAMES,
        *,
        backend: str,
    ) -> dict[str, object]:
        parent_ids = [item[0].episode_id for item in items]
        return {
            "parent_count": len(items),
            "parent_ids": parent_ids,
            "loss": np.float32(1.0),
            "gradient": np.asarray([0.1, -0.2], dtype=np.float32),
            "aux": {
                f"loss_{name}": np.zeros(len(items), dtype=np.float32)
                for name in g4.LOSS_TERM_NAMES
            },
        }

    observed_parity_roots: list[Path] = []

    def fake_default_parity(
        repository_root: Path,
        identities: tuple[str, ...],
        *,
        include_backend_evidence: bool,
        backend: str,
    ) -> dict[str, object]:
        observed_parity_roots.append(repository_root.resolve(strict=True))
        assert include_backend_evidence is True
        assert backend == "cpu"
        return {
            "parent_ids": list(identities),
            "loss": 1.0,
            "_backend_evidence": {
                "canonical_final": {"state": np.zeros(1, dtype=np.float32)},
                "left_loss": np.float32(1.0),
                "left_aux": {
                    **{
                        f"loss_{name}": np.zeros(4, dtype=np.float32) for name in g4.LOSS_TERM_NAMES
                    },
                    **{
                        f"metric_{name}": np.zeros(4, dtype=np.float32)
                        for name in (
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
                    },
                },
            },
        }

    monkeypatch.setattr(g4, "_cached_backend_parent_items", relocated_items)
    monkeypatch.setattr(g4, "evaluate_items", fake_evaluate)
    monkeypatch.setattr(g4, "default_parity", fake_default_parity)
    monkeypatch.setattr(g4.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        g4,
        "observe_backend_device_memory",
        lambda backend: (
            {"bytes_in_use": None, "peak_bytes_in_use": None, "bytes_limit": None}
            if backend == "cpu"
            else pytest.fail("backend changed")
        ),
    )
    common = {
        "backend": "cpu",
        "repository_root": expected_root,
        "provenance": {"input_manifest_sha256": "a" * 64, "runtime_contract_sha256": "b" * 64},
        "runtime": {"profile": "local_cpu"},
        "resource": {"before": {}, "after": {}},
    }

    parity = g4.run_backend_parity(**common)
    throughput = g4.run_backend_throughput(**common, phase="steady", sample_index=1)

    assert parity["status"] == "PASS_BACKEND_PARITY_SAMPLE"
    assert throughput["status"] == "PASS_BACKEND_THROUGHPUT_SAMPLE"
    assert observed_payload_roots == [expected_root, expected_root, expected_root]
    assert observed_parity_roots == [expected_root]
