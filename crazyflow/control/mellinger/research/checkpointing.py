"""Small atomic JSON checkpoints with strict compatibility fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

CHECKPOINT_SCHEMA = "crazyflow.mellinger_research_checkpoint.v2"


@dataclass(frozen=True)
class RestoredCheckpoint:
    step: int
    episode_index: int
    root_seed: int
    raw_gains: jax.Array
    optimizer_state: Any
    history: tuple[dict[str, Any], ...]
    metrics_records: tuple[dict[str, Any], ...]
    selection: dict[str, Any]
    validation_manifest_id: str
    validation_manifest_fingerprint: str
    frozen_test_manifest_reference: str
    source_fingerprint: str
    runtime_fingerprint: str
    provenance: dict[str, Any]


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    return value


def optimizer_state_record(optimizer_state: Any) -> dict[str, Any]:
    """Serialize optimizer leaves while preserving dtype and shape metadata."""
    leaves, tree = jax.tree.flatten(optimizer_state)
    return {
        "tree_repr": str(tree),
        "leaves": [
            {
                "dtype": str(np.asarray(leaf).dtype),
                "shape": list(np.asarray(leaf).shape),
                "data": np.asarray(leaf).tolist(),
            }
            for leaf in leaves
        ],
    }


def array_record(value: Any) -> dict[str, Any]:
    """Serialize one array without losing dtype or shape information."""
    array = np.asarray(value)
    return {"dtype": str(array.dtype), "shape": list(array.shape), "data": array.tolist()}


def restore_array(record: dict[str, Any], template: Any, *, label: str) -> jax.Array:
    """Restore an array against a trusted dtype/shape template."""
    expected = np.asarray(template)
    array = np.asarray(record["data"], dtype=record["dtype"]).reshape(record["shape"])
    if array.shape != expected.shape or array.dtype != expected.dtype:
        raise ValueError(f"{label} dtype/shape mismatch")
    return jnp.asarray(array)


def restore_optimizer_state(record: dict[str, Any], template: Any) -> Any:
    """Restore leaves against a caller-provided trusted optimizer structure."""
    template_leaves, tree = jax.tree.flatten(template)
    saved_leaves = record.get("leaves", [])
    if record.get("tree_repr") != str(tree) or len(saved_leaves) != len(template_leaves):
        raise ValueError("optimizer-state structure mismatch")
    restored = []
    for saved, expected in zip(saved_leaves, template_leaves, strict=True):
        array = np.asarray(saved["data"], dtype=saved["dtype"]).reshape(saved["shape"])
        expected_array = np.asarray(expected)
        if array.shape != expected_array.shape or array.dtype != expected_array.dtype:
            raise ValueError("optimizer-state leaf mismatch")
        restored.append(jnp.asarray(array))
    return jax.tree.unflatten(tree, restored)


def atomic_write_text(path: Path, payload: str, *, overwrite: bool = False) -> None:
    """Durably write text by same-directory replace and reject accidental overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    try:
        with temporary.open("x") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    """Write strict finite JSON atomically and reject accidental overwrites."""
    encoded = json.dumps(_json_native(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, encoded, overwrite=overwrite)


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _json_native(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    config_fingerprint: str,
    gain_registry_fingerprint: str,
    step: int,
    episode_index: int,
    root_seed: int,
    raw_gains: jax.Array,
    optimizer_state: Any,
    history: list[dict[str, Any]],
    metrics_records: list[dict[str, Any]],
    selection: dict[str, Any],
    validation_manifest_id: str,
    validation_manifest_fingerprint: str,
    frozen_test_manifest_reference: str,
    source_fingerprint: str,
    runtime_fingerprint: str,
    provenance: dict[str, Any],
) -> None:
    """Persist all state required to continue a compatible optimization."""
    if step < 0 or episode_index < 0 or root_seed < 0:
        raise ValueError("checkpoint counters and root seed must be nonnegative")
    selection_payload = dict(selection)
    selected_raw = selection_payload.pop("selected_raw_gains")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config_fingerprint": config_fingerprint,
        "gain_registry_fingerprint": gain_registry_fingerprint,
        "source_fingerprint": source_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "step": step,
        "episode_index": episode_index,
        "root_seed": root_seed,
        "raw_gains": array_record(raw_gains),
        "optimizer_state": optimizer_state_record(optimizer_state),
        "history": history,
        "metrics_records": metrics_records,
        "selection": selection_payload,
        "selected_raw_gains": array_record(selected_raw),
        "validation_manifest_id": validation_manifest_id,
        "validation_manifest_fingerprint": validation_manifest_fingerprint,
        "frozen_test_manifest_reference": frozen_test_manifest_reference,
        "provenance": provenance,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    atomic_write_json(path, payload)


def load_checkpoint(
    path: Path,
    *,
    optimizer_template: Any,
    raw_template: Any,
    expected_config_fingerprint: str,
    expected_gain_registry_fingerprint: str,
    expected_source_fingerprint: str,
    expected_runtime_fingerprint: str,
) -> RestoredCheckpoint:
    """Load a checkpoint and hard-reject config/registry incompatibility."""
    payload = json.loads(path.read_text())
    recorded_checksum = payload.pop("payload_sha256", None)
    if recorded_checksum is None or recorded_checksum != _payload_sha256(payload):
        raise ValueError("checkpoint payload checksum mismatch")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("config_fingerprint") != expected_config_fingerprint:
        raise ValueError("checkpoint config fingerprint mismatch")
    if payload.get("gain_registry_fingerprint") != expected_gain_registry_fingerprint:
        raise ValueError("checkpoint gain registry fingerprint mismatch")
    if payload.get("source_fingerprint") != expected_source_fingerprint:
        raise ValueError("checkpoint source fingerprint mismatch")
    if payload.get("runtime_fingerprint") != expected_runtime_fingerprint:
        raise ValueError("checkpoint runtime fingerprint mismatch")
    step = int(payload["step"])
    episode_index = int(payload["episode_index"])
    if step < 0 or episode_index != step:
        raise ValueError("checkpoint step/episode counters are inconsistent")
    history = tuple(payload["history"])
    metrics_records = tuple(payload["metrics_records"])
    if history and int(history[-1]["step"]) != step:
        raise ValueError("checkpoint history does not end at checkpoint step")
    if metrics_records and int(metrics_records[-1]["step"]) != step:
        raise ValueError("checkpoint metrics do not end at checkpoint step")
    raw_gains = restore_array(payload["raw_gains"], raw_template, label="raw gains")
    selected_raw = restore_array(
        payload["selected_raw_gains"], raw_template, label="selected raw gains"
    )
    selection = payload["selection"] | {"selected_raw_gains": selected_raw}
    return RestoredCheckpoint(
        step=step,
        episode_index=episode_index,
        root_seed=int(payload["root_seed"]),
        raw_gains=raw_gains,
        optimizer_state=restore_optimizer_state(payload["optimizer_state"], optimizer_template),
        history=history,
        metrics_records=metrics_records,
        selection=selection,
        validation_manifest_id=payload["validation_manifest_id"],
        validation_manifest_fingerprint=payload["validation_manifest_fingerprint"],
        frozen_test_manifest_reference=payload["frozen_test_manifest_reference"],
        source_fingerprint=payload["source_fingerprint"],
        runtime_fingerprint=payload["runtime_fingerprint"],
        provenance=payload["provenance"],
    )
