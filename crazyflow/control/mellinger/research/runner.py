"""Train/validation-only runner for small or workstation Mellinger experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import optax

if TYPE_CHECKING:
    from pathlib import Path

from crazyflow.control.mellinger.research.checkpointing import (
    atomic_write_json,
    atomic_write_text,
    load_checkpoint,
    save_checkpoint,
)
from crazyflow.control.mellinger.research.config import (
    ResearchConfig,
    Split,
    canonical_dict,
    fingerprint,
    load_manifest,
)
from crazyflow.control.mellinger.research.experiment import (
    EpisodeBatch,
    build_episode_batch,
    build_training_pipelines,
    research_objective,
)
from crazyflow.control.mellinger.research.gains import (
    physical_from_raw,
    raw_from_data,
    registry_fingerprint,
)

RUN_SUMMARY_SCHEMA = "crazyflow.mellinger_research_run_summary.v1"
SCIENTIFIC_SOURCE_PATHS = ("crazyflow", "examples")


def _git_provenance(repository_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=repository_root, text=True).strip()

    status = git(
        "status", "--short", "--untracked-files=all", "--", *SCIENTIFIC_SOURCE_PATHS
    ).splitlines()
    head = git("rev-parse", "HEAD")
    source_hash = hashlib.sha256()
    for relative_path in SCIENTIFIC_SOURCE_PATHS:
        source_hash.update(relative_path.encode())
        source_hash.update(git("rev-parse", f"HEAD:{relative_path}").encode())
    source_hash.update(
        subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", *SCIENTIFIC_SOURCE_PATHS], cwd=repository_root
        )
    )
    untracked = git(
        "ls-files", "--others", "--exclude-standard", "--", *SCIENTIFIC_SOURCE_PATHS
    ).splitlines()
    for relative_path in sorted(filter(None, untracked)):
        source_hash.update(relative_path.encode())
        source_hash.update((repository_root / relative_path).read_bytes())
    return {
        "repository": str(repository_root),
        "branch": git("branch", "--show-current"),
        "head": head,
        "dirty": bool(status),
        "status_short": status,
        "source_fingerprint": source_hash.hexdigest(),
        "source_fingerprint_scope": list(SCIENTIFIC_SOURCE_PATHS),
    }


def _environment() -> dict[str, Any]:
    packages = ("jax", "jaxlib", "numpy", "optax", "flax", "pytest")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def _runtime_compatibility() -> dict[str, Any]:
    packages = ("jax", "jaxlib", "numpy", "optax")
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def _json_metrics(value: Any) -> Any:
    if isinstance(value, dict):
        return {name: _json_metrics(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metrics(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    return value


def _write_metrics(path: Path, records: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    payload = "".join(
        json.dumps(_json_metrics(record), sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    atomic_write_text(path, payload, overwrite=overwrite)


def _append_metrics(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    """Durably append only new completed-update metrics without rewriting history."""
    payload = "".join(
        json.dumps(_json_metrics(record), sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _require_finite(label: str, value: Any) -> None:
    for leaf in jax.tree.leaves(value):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise FloatingPointError(f"nonfinite value in {label}")


def _technical_gate(metrics: dict[str, Any]) -> dict[str, bool]:
    values = {name: float(value) for name, value in metrics.items()}
    checks = {
        "nonfinite_state_fraction_zero": values["nonfinite_state_fraction"] == 0.0,
        "zero_thrust_gate_fraction_zero": values["zero_thrust_gate_fraction"] == 0.0,
        "floor_clip_fraction_zero": values["floor_clip_fraction"] == 0.0,
        "motor_saturation_fraction_below_half": values["motor_saturation_fraction"] < 0.5,
        "all_episodes_successful": values["episode_success"] == 1.0,
    }
    return checks | {"passed": all(checks.values())}


def _without_static_metadata(batch: EpisodeBatch) -> EpisodeBatch:
    """Keep JIT PyTree metadata constant while IDs remain in host provenance."""
    return batch.replace(episode_ids=())


def run_training_validation(
    config: ResearchConfig,
    *,
    config_path: Path,
    output_dir: Path,
    repository_root: Path,
    command: str,
    resume_path: Path | None = None,
    max_updates_this_process: int | None = None,
) -> dict[str, Any]:
    """Run optimization and validation; never build or evaluate the test pipeline."""
    config.validate()
    if max_updates_this_process is not None and max_updates_this_process < 1:
        raise ValueError("max_updates_this_process must be positive")
    started_at = datetime.now(UTC)
    start_ns = time.perf_counter_ns()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"run output directory must be absent or empty: {output_dir}")
    if resume_path is not None and resume_path.parent.resolve() == output_dir.resolve():
        raise ValueError("resume checkpoint and new output directory must be separate")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = repository_root / config.validation_manifest
    validation_manifest = load_manifest(validation_path, expected_split=Split.VALIDATION)
    validation_manifest_hash = fingerprint(validation_manifest)
    frozen_test_reference = config.test_manifest
    pipelines = build_training_pipelines(config.train, config.validation, config.root_seed)
    validation_batch = build_episode_batch(
        pipelines.validation,
        config.root_seed,
        0,
        config.trajectory,
        config.domain_randomization,
        validation_manifest,
    )
    validation_jit_batch = _without_static_metadata(validation_batch)
    raw = raw_from_data(pipelines.train.sim.data, config.optimizer.gain_stage)
    optimizer = optax.adam(config.optimizer.learning_rate)
    optimizer_state = optimizer.init(raw)
    config_hash = fingerprint(config)
    registry_hash = registry_fingerprint()
    git_provenance = _git_provenance(repository_root)
    source_hash = git_provenance["source_fingerprint"]
    runtime_compatibility = _runtime_compatibility()
    runtime_hash = fingerprint(runtime_compatibility)
    history: list[dict[str, Any]] = []
    metrics_records: list[dict[str, Any]] = []
    start_step = 0
    episode_index = 0
    restored_selection: dict[str, Any] = {}
    if resume_path is not None:
        restored = load_checkpoint(
            resume_path,
            optimizer_template=optimizer_state,
            raw_template=raw,
            expected_config_fingerprint=config_hash,
            expected_gain_registry_fingerprint=registry_hash,
            expected_source_fingerprint=source_hash,
            expected_runtime_fingerprint=runtime_hash,
        )
        if restored.validation_manifest_id != validation_manifest.manifest_id:
            raise ValueError("checkpoint validation manifest mismatch")
        if restored.validation_manifest_fingerprint != validation_manifest_hash:
            raise ValueError("checkpoint validation manifest fingerprint mismatch")
        if restored.frozen_test_manifest_reference != frozen_test_reference:
            raise ValueError("checkpoint frozen test manifest reference mismatch")
        if restored.root_seed != config.root_seed:
            raise ValueError("checkpoint root seed mismatch")
        raw = restored.raw_gains
        optimizer_state = restored.optimizer_state
        history = list(restored.history)
        metrics_records = list(restored.metrics_records)
        restored_selection = restored.selection
        start_step = restored.step
        episode_index = restored.episode_index
    if start_step >= config.optimizer.steps:
        raise ValueError("resume checkpoint already completed all configured updates")

    steps_per_command = config.train.sim_freq_hz // config.train.control_freq_hz

    def train_step(
        candidate: jax.Array, state: Any, batch: EpisodeBatch
    ) -> tuple[jax.Array, Any, jax.Array, dict[str, Any], jax.Array]:
        def objective(values: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
            return research_objective(
                values,
                batch,
                step_fn=pipelines.train.step_fn,
                steps_per_command=steps_per_command,
                gain_stage=config.optimizer.gain_stage,
            )

        (loss, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(candidate)
        updates, new_state = optimizer.update(gradient, state, candidate)
        return optax.apply_updates(candidate, updates), new_state, loss, auxiliary, gradient

    validation_steps = config.validation.sim_freq_hz // config.validation.control_freq_hz

    def validate(candidate: jax.Array, batch: EpisodeBatch) -> tuple[jax.Array, dict[str, Any]]:
        return research_objective(
            candidate,
            batch,
            step_fn=pipelines.validation.step_fn,
            steps_per_command=validation_steps,
            gain_stage=config.optimizer.gain_stage,
        )

    jitted_train_step = jax.jit(train_step)
    jitted_validate = jax.jit(validate)
    timing: dict[str, Any] = {}
    best_raw = raw
    best_validation = float("inf")
    best_step = start_step
    if restored_selection:
        best_validation = float(restored_selection["selected_validation_loss"])
        best_step = int(restored_selection["selected_step"])
        best_raw = jax.numpy.asarray(restored_selection["selected_raw_gains"], dtype=raw.dtype)
        if best_raw.shape != raw.shape:
            raise ValueError("checkpoint selected raw gains shape mismatch")
    resume_lineage = None
    if resume_path is not None:
        resume_lineage = {
            "checkpoint_path": str(resume_path),
            "checkpoint_sha256": hashlib.sha256(resume_path.read_bytes()).hexdigest(),
            "parent_step": start_step,
            "parent_source_fingerprint": restored.source_fingerprint,
            "parent_command": restored.provenance.get("command"),
        }
    provenance = {
        "schema_version": "crazyflow.mellinger_research_provenance.v1",
        "command": command,
        "config_path": str(config_path),
        "config_fingerprint": config_hash,
        "gain_registry_fingerprint": registry_hash,
        "validation_manifest_id": validation_manifest.manifest_id,
        "validation_manifest_fingerprint": validation_manifest_hash,
        "frozen_test_manifest_reference_not_opened": frozen_test_reference,
        "root_seed": config.root_seed,
        "seed_coordinate_contract": "split -> episode -> world -> component",
        "started_at_utc": started_at.isoformat(),
        "git": git_provenance,
        "runtime_compatibility": runtime_compatibility,
        "runtime_fingerprint": runtime_hash,
        "environment": _environment(),
        "resume_lineage": resume_lineage,
        "checkpoint_policy": {
            "interval_completed_updates": config.optimizer.checkpoint_interval,
            "configured_final_step_always_checkpointed": True,
            "planned_process_stop_always_checkpointed": True,
            "metrics_persistence": "durable append per completed update; full history at end",
        },
    }

    stop_step = config.optimizer.steps
    if max_updates_this_process is not None:
        stop_step = min(stop_step, start_step + max_updates_this_process)
    checkpoint_writes: list[dict[str, Any]] = []
    completed_update_seconds: list[dict[str, Any]] = []
    if metrics_records:
        _write_metrics(output_dir / "metrics.jsonl", metrics_records)
    for step_index in range(start_step, stop_step):
        update_start_ns = time.perf_counter_ns()
        train_batch = build_episode_batch(
            pipelines.train,
            config.root_seed,
            episode_index,
            config.trajectory,
            config.domain_randomization,
        )
        train_ids = train_batch.episode_ids
        train_jit_batch = _without_static_metadata(train_batch)
        if step_index == start_step:
            compile_start = time.perf_counter_ns()
            compiled_train = jitted_train_step.lower(
                raw, optimizer_state, train_jit_batch
            ).compile()
            timing["train_compile_seconds"] = (time.perf_counter_ns() - compile_start) / 1.0e9
            execution_start = time.perf_counter_ns()
            raw, optimizer_state, train_loss, train_aux, gradient = compiled_train(
                raw, optimizer_state, train_jit_batch
            )
            jax.block_until_ready((raw, train_loss, gradient))
            timing["train_first_execution_seconds"] = (
                time.perf_counter_ns() - execution_start
            ) / 1.0e9
        else:
            raw, optimizer_state, train_loss, train_aux, gradient = jitted_train_step(
                raw, optimizer_state, train_jit_batch
            )
            jax.block_until_ready((raw, train_loss, gradient))

        if step_index == start_step:
            compile_start = time.perf_counter_ns()
            compiled_validation = jitted_validate.lower(raw, validation_jit_batch).compile()
            timing["validation_compile_seconds"] = (time.perf_counter_ns() - compile_start) / 1.0e9
            validation_start = time.perf_counter_ns()
            validation_loss, validation_aux = compiled_validation(raw, validation_jit_batch)
            jax.block_until_ready((validation_loss, validation_aux))
            timing["validation_first_execution_seconds"] = (
                time.perf_counter_ns() - validation_start
            ) / 1.0e9
            steady_samples = []
            for _ in range(3):
                steady_start = time.perf_counter_ns()
                steady_result = compiled_validation(raw, validation_jit_batch)
                jax.block_until_ready(steady_result)
                steady_samples.append((time.perf_counter_ns() - steady_start) / 1.0e9)
            timing["validation_steady_seconds"] = steady_samples
        else:
            validation_loss, validation_aux = jitted_validate(raw, validation_jit_batch)
            jax.block_until_ready((validation_loss, validation_aux))

        _require_finite("training update", (raw, optimizer_state, train_loss, train_aux, gradient))
        _require_finite("validation", (validation_loss, validation_aux))

        step_number = step_index + 1
        train_record = {
            "step": step_number,
            "split": Split.TRAIN.value,
            "episode_ids": train_ids,
            "loss": train_loss,
            "metrics": train_aux["mean_metrics"],
            "gradient_l2_norm": np.linalg.norm(np.asarray(gradient)),
            "technical_gate": _technical_gate(train_aux["mean_metrics"]),
        }
        validation_record = {
            "step": step_number,
            "split": Split.VALIDATION.value,
            "manifest_id": validation_manifest.manifest_id,
            "episode_ids": validation_batch.episode_ids,
            "loss": validation_loss,
            "metrics": validation_aux["mean_metrics"],
            "technical_gate": _technical_gate(validation_aux["mean_metrics"]),
        }
        metrics_records.extend((train_record, validation_record))
        history.append(
            {
                "step": step_number,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
            }
        )
        if float(validation_loss) < best_validation:
            best_validation = float(validation_loss)
            best_step = step_number
            best_raw = raw
        episode_index += 1
        current_selection = {
            "criterion": "minimum validation loss; earliest exact tie",
            "selected_step": best_step,
            "selected_validation_loss": best_validation,
            "selected_raw_gains": best_raw,
            "test_used_for_selection": False,
        }
        checkpoint_due = (
            step_number % config.optimizer.checkpoint_interval == 0
            or step_number == config.optimizer.steps
            or step_number == stop_step
        )
        if checkpoint_due:
            checkpoint_path = output_dir / f"checkpoint-step-{step_number:06d}.json"
            checkpoint_start_ns = time.perf_counter_ns()
            save_checkpoint(
                checkpoint_path,
                config_fingerprint=config_hash,
                gain_registry_fingerprint=registry_hash,
                step=step_number,
                episode_index=episode_index,
                root_seed=config.root_seed,
                raw_gains=raw,
                optimizer_state=optimizer_state,
                history=history,
                metrics_records=metrics_records,
                selection=current_selection,
                validation_manifest_id=validation_manifest.manifest_id,
                validation_manifest_fingerprint=validation_manifest_hash,
                frozen_test_manifest_reference=frozen_test_reference,
                source_fingerprint=source_hash,
                runtime_fingerprint=runtime_hash,
                provenance=provenance,
            )
            checkpoint_writes.append(
                {
                    "step": step_number,
                    "seconds": (time.perf_counter_ns() - checkpoint_start_ns) / 1.0e9,
                    "bytes": checkpoint_path.stat().st_size,
                }
            )
        _append_metrics(output_dir / "metrics.jsonl", (train_record, validation_record))
        completed_update_seconds.append(
            {
                "step": step_number,
                "seconds": (time.perf_counter_ns() - update_start_ns) / 1.0e9,
                "checkpoint_written": checkpoint_due,
            }
        )

    if not history:
        raise ValueError("configuration/resume state leaves no optimization step to execute")

    selection = {
        "criterion": "minimum validation loss; earliest exact tie",
        "selected_step": best_step,
        "selected_validation_loss": best_validation,
        "selected_raw_gains": best_raw,
        "test_used_for_selection": False,
    }
    ended_at = datetime.now(UTC)
    total_seconds = (time.perf_counter_ns() - start_ns) / 1.0e9
    timing["completed_update_seconds"] = completed_update_seconds
    timing["checkpoint_writes"] = checkpoint_writes
    provenance |= {"ended_at_utc": ended_at.isoformat(), "wall_seconds_internal": total_seconds}
    completed = stop_step == config.optimizer.steps
    summary = {
        "schema_version": RUN_SUMMARY_SCHEMA,
        "status": "success" if completed else "paused",
        "scientific_claim": "technical train/validation infrastructure evidence only",
        "selected": selection,
        "selected_physical_gains": physical_from_raw(best_raw, config.optimizer.gain_stage),
        "final_raw_gains": raw,
        "final_physical_gains": physical_from_raw(raw, config.optimizer.gain_stage),
        "history": history,
        "completed_updates": stop_step,
        "configured_updates": config.optimizer.steps,
        "resume_checkpoint": (None if completed else f"checkpoint-step-{stop_step:06d}.json"),
        "timing": timing,
        "warnings": [
            "The frozen test manifest was not opened and no test episode was built or evaluated.",
            "Simulation evidence is not hardware or firmware validation.",
        ],
        "test_metrics_present": False,
        "test_manifest_opened": False,
    }
    atomic_write_json(
        output_dir / "resolved_config.json",
        {
            "config": canonical_dict(config),
            "config_fingerprint": config_hash,
            "source": str(config_path),
        },
    )
    atomic_write_json(output_dir / "provenance.json", provenance)
    if not (output_dir / "metrics.jsonl").exists():
        _write_metrics(output_dir / "metrics.jsonl", metrics_records)
    atomic_write_json(output_dir / "run_summary.json", summary)
    return _json_metrics(summary)
