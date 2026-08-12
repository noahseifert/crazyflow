"""Read-only verification of the frozen Sprint-12 H100 replication protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax

from crazyflow.control.mellinger.research.config import (
    Split,
    fingerprint,
    load_config,
    load_manifest,
)
from crazyflow.control.mellinger.research.gains import registry_fingerprint, specs_for_stage
from crazyflow.control.mellinger.tracking import TrackingLossConfig

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = ROOT / "artifacts/day12-h100-replication-protocol"
BASE_CONFIG_PATH = ROOT / "artifacts/day10-sparse-long-pilot/pilot_config.json"
SMOKE_METRICS_PATH = ROOT / "artifacts/day10-sparse-long-pilot/smoke-210/metrics.jsonl"
PILOT_SEED = 20260731
EXPECTED_SEED_COUNT = 10
BASE_PROTOCOL_HEAD = "73868cfd84d25bec0c6612b31ee34f4e67acb307"
AMENDMENT_ID = "2026-08-01-local-ssd-persistence-v1"
SEED_01_MINIMUM_LOCAL_FREE_BYTES = 20_000_000_000
LOCAL_FREE_RESERVE_BYTES = 10_000_000_000
EXPECTED_RAW_BYTES_PER_SEED = 437_505_563
SOURCE_PATHS = ("crazyflow", "examples")
EXPECTED_PAYLOADS = {
    "README.md",
    "analysis_plan.json",
    "generate_protocol.py",
    "protocol.json",
    "release_gate.json",
    "requirements.json",
    "run_inventory.json",
    "seed_manifest.json",
    "sprint13_runbook_template.sh",
    "verify_protocol.py",
    f"amendments/{AMENDMENT_ID}.json",
    f"amendments/{AMENDMENT_ID}.md",
} | {f"configs/seed-{index:02d}.json" for index in range(1, 11)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _git_blob(commit: str, relative_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative_path}"], cwd=ROOT)


def _current_source_fingerprint() -> str:
    source_hash = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        source_hash.update(relative_path.encode())
        source_hash.update(_git("rev-parse", f"HEAD:{relative_path}").encode())
    source_hash.update(
        subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *SOURCE_PATHS], cwd=ROOT)
    )
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS).splitlines()
    for relative_path in sorted(filter(None, untracked)):
        source_hash.update(relative_path.encode())
        source_hash.update((ROOT / relative_path).read_bytes())
    return source_hash.hexdigest()


def _runtime_compatibility() -> dict[str, Any]:
    packages = ("jax", "jaxlib", "numpy", "optax")
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def _derive_seed(namespace: str, index: int, used: set[int]) -> dict[str, Any]:
    retry = 0
    while True:
        derivation_input = f"{namespace}|seed-index={index:02d}|retry={retry}"
        digest = hashlib.sha256(derivation_input.encode()).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed != 0 and seed not in used:
            used.add(seed)
            return {
                "index": index,
                "root_seed": seed,
                "derivation_input": derivation_input,
                "derivation_sha256": hashlib.sha256(derivation_input.encode()).hexdigest(),
                "retry_counter": retry,
            }
        retry += 1


def _without_seed_identity(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"run_id", "root_seed"}}


def _read_sha_index() -> dict[str, str]:
    result = {}
    for line in (PROTOCOL_ROOT / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        result[relative] = digest
    return result


def main() -> None:
    """Fail on any protocol drift without opening the Test manifest or starting a run."""
    protocol = json.loads((PROTOCOL_ROOT / "protocol.json").read_text())
    seeds = json.loads((PROTOCOL_ROOT / "seed_manifest.json").read_text())
    requirements = json.loads((PROTOCOL_ROOT / "requirements.json").read_text())
    inventory = json.loads((PROTOCOL_ROOT / "run_inventory.json").read_text())
    analysis = json.loads((PROTOCOL_ROOT / "analysis_plan.json").read_text())
    gate = json.loads((PROTOCOL_ROOT / "release_gate.json").read_text())
    amendment = json.loads((PROTOCOL_ROOT / "amendments" / f"{AMENDMENT_ID}.json").read_text())
    base = json.loads(BASE_CONFIG_PATH.read_text())
    base_requirements = json.loads(
        _git_blob(BASE_PROTOCOL_HEAD, "artifacts/day12-h100-replication-protocol/requirements.json")
    )
    base_inventory = json.loads(
        _git_blob(
            BASE_PROTOCOL_HEAD, "artifacts/day12-h100-replication-protocol/run_inventory.json"
        )
    )
    base_analysis = json.loads(
        _git_blob(
            BASE_PROTOCOL_HEAD, "artifacts/day12-h100-replication-protocol/analysis_plan.json"
        )
    )

    assert protocol["status"] == "FROZEN_PROTOCOL_ONLY_NO_RUN_AUTHORIZATION"
    assert protocol["release"]["sprint12_research_runs_authorized"] == 0
    assert protocol["release"]["sprint12a_research_runs_authorized"] == 0
    assert gate["decision"] == "NO-GO_RUNS_IN_SPRINT12A"
    assert gate["research_processes_started"] == 0
    assert gate["test_manifest_opened"] is False
    assert gate["h200_h400_started"] is False
    assert gate["pilot_in_confirmatory_panel"] is False
    assert protocol["exploratory_pilot"]["root_seed"] == PILOT_SEED
    assert (
        protocol["exploratory_pilot"]["included_in_confirmatory_seed_count_or_aggregation"] is False
    )
    assert protocol["active_amendment"]["id"] == AMENDMENT_ID
    assert protocol["active_amendment"]["base_protocol_commit"] == BASE_PROTOCOL_HEAD
    assert protocol["references"]["persistence_amendment"] == (f"amendments/{AMENDMENT_ID}.json")
    assert amendment["status"] == "ADOPTED_PRE_SEED_01"
    assert amendment["base_protocol_commit"] == BASE_PROTOCOL_HEAD
    assert amendment["timing_and_bias_control"] == {
        "adopted_before_seed_01": True,
        "confirmatory_results_known_at_adoption": False,
        "confirmatory_seeds_started_before_adoption": 0,
        "result_dependent_change": False,
    }
    assert amendment["research_processes_started_by_amendment"] == 0
    assert amendment["test_manifest_opened_by_amendment"] is False
    assert amendment["unchanged_fingerprints"] == {
        "gain_registry": requirements["required_gain_registry_fingerprint"],
        "runtime": requirements["required_runtime_fingerprint"],
        "source": requirements["required_source_fingerprint"],
        "validation_manifest": requirements["required_validation_manifest_fingerprint"],
    }
    assert requirements["scientific_semantics"] == base_requirements["scientific_semantics"]
    assert requirements["runtime"] == base_requirements["runtime"]
    assert requirements["config_base_fingerprint"] == base_requirements["config_base_fingerprint"]

    namespace = seeds["generation"]["namespace"]
    expected_seeds = []
    used = {PILOT_SEED}
    for index in range(1, EXPECTED_SEED_COUNT + 1):
        expected_seeds.append(_derive_seed(namespace, index, used))
    assert seeds["seed_count"] == EXPECTED_SEED_COUNT
    assert seeds["seeds"] == expected_seeds
    root_seeds = [item["root_seed"] for item in seeds["seeds"]]
    assert len(root_seeds) == len(set(root_seeds)) == EXPECTED_SEED_COUNT
    assert PILOT_SEED not in root_seeds
    assert protocol["frozen_design"]["new_root_seeds"] == root_seeds
    seed_manifest_path = "artifacts/day12-h100-replication-protocol/seed_manifest.json"
    assert (ROOT / seed_manifest_path).read_bytes() == _git_blob(
        BASE_PROTOCOL_HEAD, seed_manifest_path
    )

    base_loaded = load_config(BASE_CONFIG_PATH)
    assert _sha256(BASE_CONFIG_PATH) == requirements["config_base_sha256"]
    assert fingerprint(base_loaded) == requirements["config_base_fingerprint"]
    config_inventory = inventory["configs"]
    assert config_inventory == base_inventory["configs"]
    assert len(config_inventory) == EXPECTED_SEED_COUNT
    for seed, item in zip(seeds["seeds"], config_inventory, strict=True):
        config_path = ROOT / item["path"]
        config_dict = json.loads(config_path.read_text())
        config = load_config(config_path)
        assert config_dict["root_seed"] == seed["root_seed"] == item["root_seed"]
        assert config_dict["run_id"] == item["run_id"]
        assert _without_seed_identity(config_dict) == _without_seed_identity(base)
        assert _sha256(config_path) == item["sha256"]
        assert config_path.read_bytes() == _git_blob(BASE_PROTOCOL_HEAD, item["path"])
        assert (
            amendment["unchanged_seed_config_sha256"][f"seed-{seed['index']:02d}"] == item["sha256"]
        )
        assert fingerprint(config) == item["config_fingerprint"]
        assert config.train.horizon == config.validation.horizon == 100
        assert config.train.n_worlds == config.validation.n_worlds == 4
        assert config.optimizer.steps == 5000
        assert config.optimizer.checkpoint_interval == 100
        assert config.optimizer.gain_stage == 1
        assert config.optimizer.learning_rate == 0.001
        assert config.test_manifest == base["test_manifest"]

    assert inventory["execution"]["maximum_concurrent_research_processes"] == 1
    assert inventory["execution"]["automatic_seed_loop"] is False
    assert inventory["execution"]["automatic_resume"] is False
    assert [item["execution_order"] for item in inventory["runs"]] == list(range(1, 11))
    assert all(item["status"] == "PLANNED_NOT_STARTED" for item in inventory["runs"])
    assert all(item["allowed_replacement_seeds"] == 0 for item in inventory["runs"])
    for run, base_run in zip(inventory["runs"], base_inventory["runs"], strict=True):
        assert {
            key: value
            for key, value in run.items()
            if key
            not in {
                "minimum_local_free_bytes_before_start",
                "remaining_seed_count_including_this_run",
            }
        } == base_run
    assert (
        inventory["execution"]["start_next_only_after_previous_local_sha256_reverification_pass"]
        is True
    )
    persistence = requirements["persistence_requirements"]
    assert persistence == inventory["persistence"] == amendment["replacement_rule"]
    assert persistence["external_storage_copy_required"] is False
    assert persistence["external_storage_capacity_check_required"] is False
    assert persistence["external_destination_hash_verification_required"] is False
    assert persistence["accepted_single_drive_risk"] is True
    assert persistence["ordinary_git_tracks_raw_runs"] is False
    assert persistence["seed_01_minimum_local_free_bytes"] == SEED_01_MINIMUM_LOCAL_FREE_BYTES
    assert persistence["local_free_reserve_bytes"] == LOCAL_FREE_RESERVE_BYTES
    for run in inventory["runs"]:
        index = run["seed_index"]
        remaining = EXPECTED_SEED_COUNT - index + 1
        expected_minimum = (
            SEED_01_MINIMUM_LOCAL_FREE_BYTES
            if index == 1
            else LOCAL_FREE_RESERVE_BYTES + EXPECTED_RAW_BYTES_PER_SEED * remaining
        )
        assert run["remaining_seed_count_including_this_run"] == remaining
        assert run["minimum_local_free_bytes_before_start"] == expected_minimum
    assert (
        "minimum_external_free_bytes_before_start"
        not in inventory["resource_plan"]["ten_run_panel"]
    )

    assert _current_source_fingerprint() == requirements["required_source_fingerprint"]
    runtime = _runtime_compatibility()
    assert runtime == requirements["runtime"]
    assert fingerprint(runtime) == requirements["required_runtime_fingerprint"]
    assert registry_fingerprint() == requirements["required_gain_registry_fingerprint"]
    validation = load_manifest(ROOT / base["validation_manifest"], expected_split=Split.VALIDATION)
    assert validation.manifest_id == requirements["required_validation_manifest_id"]
    assert fingerprint(validation) == requirements["required_validation_manifest_fingerprint"]
    assert requirements["test_manifest_policy"]["opaque_reference_only"] is True
    assert (
        requirements["test_manifest_policy"][
            "must_not_open_parse_validate_build_simulate_or_evaluate"
        ]
        is True
    )
    assert requirements["scientific_semantics"]["loss_config"] == asdict(TrackingLossConfig())
    assert [item["name"] for item in requirements["scientific_semantics"]["gains"]] == [
        spec.name for spec in specs_for_stage(1)
    ]

    runner_source = (ROOT / "crazyflow/control/mellinger/research/runner.py").read_text()
    assert "if float(validation_loss) < best_validation:" in runner_source
    assert '"criterion": "minimum validation loss; earliest exact tie"' in runner_source
    assert analysis["selection_rule"]["candidate_steps"] == (
        "every completed post-update step 1 through 5000"
    )
    assert analysis["selection_rule"]["step_zero_is_not_a_candidate"] is True
    assert analysis["selection_rule"]["test_used"] is False

    smoke_rows = [json.loads(line) for line in SMOKE_METRICS_PATH.read_text().splitlines()[:2]]
    train_row, validation_row = smoke_rows
    emitted = {"loss", "gradient_l2_norm (Train only)"} | set(train_row["metrics"])
    assert set(train_row["metrics"]) == set(validation_row["metrics"])
    assert set(analysis["runner_emitted_metric_names"]) == emitted
    assert analysis["primary_endpoint"]["name"] == "selected_validation_loss"
    assert analysis["primary_endpoint"]["uncertainty"]["degrees_of_freedom"] == 9
    assert analysis["analysis_population"]["replacement_seeds"] == 0
    assert analysis["technical_failure_policy"]["no_resume"] is True
    assert len(analysis["go_no_go_horizon_transition"]["go_requires_all"]) == 10
    assert analysis["test_policy"]["locked"] is True
    assert analysis["test_policy"]["open_count_during_replication"] == 0
    assert analysis["active_persistence_amendment"] == AMENDMENT_ID
    assert analysis["persistence_scope_only"] is True
    for unchanged_key in (
        "analysis_population",
        "derived_primary_support",
        "primary_endpoint",
        "runner_emitted_metric_names",
        "secondary_endpoints",
        "secondary_inference_policy",
        "selection_rule",
        "test_policy",
    ):
        assert analysis[unchanged_key] == base_analysis[unchanged_key]
    amended_go = analysis["go_no_go_horizon_transition"]
    base_go = base_analysis["go_no_go_horizon_transition"]
    assert amended_go["go_requires_all"][:8] == base_go["go_requires_all"][:8]
    assert amended_go["go_requires_all"][9:] == base_go["go_requires_all"][9:]
    assert amended_go["decision_scope"] == base_go["decision_scope"]
    assert amended_go["no_go"] == base_go["no_go"]
    assert (
        "external storage"
        not in " ".join(analysis["go_no_go_horizon_transition"]["go_requires_all"]).lower()
    )
    assert gate["active_persistence_amendment"] == AMENDMENT_ID
    assert gate["external_storage_gate_required"] is False
    assert gate["local_persistence_gate_required"] is True
    assert gate["seed_01_preparation_status"] == (
        "GO_AFTER_SEPARATE_AUTHORIZATION_AND_EXACT_PREFLIGHT"
    )

    sha_index = _read_sha_index()
    assert set(sha_index) == EXPECTED_PAYLOADS
    for relative_path, digest in sha_index.items():
        assert _sha256(PROTOCOL_ROOT / relative_path) == digest
    actual_files = {
        str(path.relative_to(PROTOCOL_ROOT)) for path in PROTOCOL_ROOT.rglob("*") if path.is_file()
    }
    assert actual_files == EXPECTED_PAYLOADS | {"SHA256SUMS"}

    print("protocol_verification=PASS")
    print(f"seed_count={len(root_seeds)}")
    print("seeds=" + ",".join(str(seed) for seed in root_seeds))
    print(f"config_count={len(config_inventory)}")
    print(f"source_fingerprint={requirements['required_source_fingerprint']}")
    print(f"runtime_fingerprint={requirements['required_runtime_fingerprint']}")
    local_free_bytes = shutil.disk_usage(ROOT).free
    print("base_seed_manifest_unchanged=True")
    print("base_seed_configs_unchanged=True")
    print(f"local_free_bytes={local_free_bytes}")
    print(f"seed01_minimum_local_free_bytes={SEED_01_MINIMUM_LOCAL_FREE_BYTES}")
    print(
        f"seed01_local_capacity_gate_passed={local_free_bytes >= SEED_01_MINIMUM_LOCAL_FREE_BYTES}"
    )
    print("external_storage_required=False")
    print("accepted_single_drive_risk=True")
    print("test_manifest_opened=False")
    print("research_runs_started=0")


if __name__ == "__main__":
    main()
