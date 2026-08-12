from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from crazyflow.control.mellinger.research.checkpointing import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_checkpoint_roundtrip_restores_optimizer_and_resume_counters(tmp_path: Path) -> None:
    raw = jnp.array([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    optimizer = optax.adam(0.001)
    initial_state = optimizer.init(raw)
    updates, updated_state = optimizer.update(jnp.ones_like(raw), initial_state, raw)
    updated_raw = optax.apply_updates(raw, updates)
    path = tmp_path / "checkpoint-step-000001.json"

    save_checkpoint(
        path,
        config_fingerprint="a" * 64,
        gain_registry_fingerprint="b" * 64,
        step=1,
        episode_index=1,
        root_seed=17,
        raw_gains=updated_raw,
        optimizer_state=updated_state,
        history=[{"step": 1, "validation_loss": 0.5}],
        metrics_records=[{"step": 1, "split": "validation", "loss": 0.5}],
        selection={
            "selected_step": 1,
            "criterion": "validation_loss",
            "selected_raw_gains": updated_raw,
        },
        validation_manifest_id="validation-fixture",
        validation_manifest_fingerprint="c" * 64,
        frozen_test_manifest_reference="test-frozen-unopened.json",
        source_fingerprint="d" * 64,
        runtime_fingerprint="e" * 64,
        provenance={"command": "fixture"},
    )
    restored = load_checkpoint(
        path,
        optimizer_template=optimizer.init(raw),
        raw_template=raw,
        expected_config_fingerprint="a" * 64,
        expected_gain_registry_fingerprint="b" * 64,
        expected_source_fingerprint="d" * 64,
        expected_runtime_fingerprint="e" * 64,
    )

    assert restored.step == restored.episode_index == 1
    assert restored.root_seed == 17
    assert np.array_equal(restored.raw_gains, updated_raw)
    assert restored.history[0]["validation_loss"] == pytest.approx(0.5)
    assert restored.metrics_records[0]["split"] == "validation"
    assert restored.raw_gains.dtype == raw.dtype
    assert restored.selection["selected_raw_gains"].dtype == raw.dtype
    for actual, expected in zip(
        jax.tree.leaves(restored.optimizer_state), jax.tree.leaves(updated_state), strict=True
    ):
        assert np.array_equal(actual, expected)


@pytest.mark.unit
def test_checkpoint_rejects_overwrite_and_fingerprint_mismatch(tmp_path: Path) -> None:
    raw = jnp.zeros(1)
    optimizer = optax.adam(0.001)
    state = optimizer.init(raw)
    path = tmp_path / "checkpoint.json"
    arguments = {
        "config_fingerprint": "c" * 64,
        "gain_registry_fingerprint": "d" * 64,
        "step": 1,
        "episode_index": 1,
        "root_seed": 1,
        "raw_gains": raw,
        "optimizer_state": state,
        "history": [{"step": 1}],
        "metrics_records": [{"step": 1}],
        "selection": {"selected_step": 1, "selected_raw_gains": raw},
        "validation_manifest_id": "validation",
        "validation_manifest_fingerprint": "f" * 64,
        "frozen_test_manifest_reference": "test-unopened.json",
        "source_fingerprint": "1" * 64,
        "runtime_fingerprint": "2" * 64,
        "provenance": {},
    }
    save_checkpoint(path, **arguments)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_checkpoint(path, **arguments)
    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        load_checkpoint(
            path,
            optimizer_template=state,
            raw_template=raw,
            expected_config_fingerprint="e" * 64,
            expected_gain_registry_fingerprint="d" * 64,
            expected_source_fingerprint="1" * 64,
            expected_runtime_fingerprint="2" * 64,
        )
    with pytest.raises(ValueError, match="gain registry fingerprint mismatch"):
        load_checkpoint(
            path,
            optimizer_template=state,
            raw_template=raw,
            expected_config_fingerprint="c" * 64,
            expected_gain_registry_fingerprint="e" * 64,
            expected_source_fingerprint="1" * 64,
            expected_runtime_fingerprint="2" * 64,
        )
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        load_checkpoint(
            path,
            optimizer_template=state,
            raw_template=raw,
            expected_config_fingerprint="c" * 64,
            expected_gain_registry_fingerprint="d" * 64,
            expected_source_fingerprint="3" * 64,
            expected_runtime_fingerprint="2" * 64,
        )
    with pytest.raises(ValueError, match="runtime fingerprint mismatch"):
        load_checkpoint(
            path,
            optimizer_template=state,
            raw_template=raw,
            expected_config_fingerprint="c" * 64,
            expected_gain_registry_fingerprint="d" * 64,
            expected_source_fingerprint="1" * 64,
            expected_runtime_fingerprint="3" * 64,
        )


@pytest.mark.unit
def test_checkpoint_rejects_nonfinite_payload_and_checksum_corruption(tmp_path: Path) -> None:
    raw = jnp.zeros(1, dtype=jnp.float32)
    optimizer = optax.adam(0.001)
    state = optimizer.init(raw)
    arguments = {
        "config_fingerprint": "a" * 64,
        "gain_registry_fingerprint": "b" * 64,
        "step": 1,
        "episode_index": 1,
        "root_seed": 1,
        "raw_gains": raw,
        "optimizer_state": state,
        "history": [{"step": 1}],
        "metrics_records": [{"step": 1, "loss": 0.0}],
        "selection": {"selected_step": 1, "selected_raw_gains": raw},
        "validation_manifest_id": "validation",
        "validation_manifest_fingerprint": "c" * 64,
        "frozen_test_manifest_reference": "test-unopened.json",
        "source_fingerprint": "d" * 64,
        "runtime_fingerprint": "e" * 64,
        "provenance": {},
    }
    with pytest.raises(ValueError, match="Out of range float values"):
        save_checkpoint(
            tmp_path / "nonfinite.json", **(arguments | {"raw_gains": jnp.array([jnp.nan])})
        )

    path = tmp_path / "corrupt.json"
    save_checkpoint(path, **arguments)
    path.write_text(path.read_text().replace('"root_seed": 1', '"root_seed": 2'))
    with pytest.raises(ValueError, match="payload checksum mismatch"):
        load_checkpoint(
            path,
            optimizer_template=state,
            raw_template=raw,
            expected_config_fingerprint="a" * 64,
            expected_gain_registry_fingerprint="b" * 64,
            expected_source_fingerprint="d" * 64,
            expected_runtime_fingerprint="e" * 64,
        )
