# Sprint 11 evaluation of the Sprint-10 sparse long pilot

## Result

The sole run
`artifacts/day10-sparse-long-pilot/runs/20260801T081232Z-1345032536fd` is a complete
`PASS`: exit 0, 5,000/5,000 updates, exactly 50 interval-100 checkpoints, 10,000 ordered
Train/Validation metric rows, valid payload checksums, finite values, passing technical gates,
and no Test access. No process, automatic continuation, Resume, second run, H200/H400 run, seed
loop, or Test evaluation was started during Sprint 11.

The machine-readable analysis is `long_pilot_analysis.json`. It contains all 5,000 Train,
Validation, generalization-gap, and Gradient-L2 values; early/middle/late window statistics; all
50 persisted gain vectors and bound distances; resource/projection comparisons; provenance; and
SHA-256 for every raw run file. `SPRINT10_RUN_SHA256SUMS` is the independently checkable 61-file
raw-run index.

The sparse policy persists exact gain vectors only at checkpoint steps 100, 200, ..., 5,000.
Therefore the analysis includes every persisted gain sample but does not invent the unavailable
intermediate per-update gain vectors. Per-update loss and gradient metrics are complete.

## Reproduction

Run from the repository root. The generator reads no Test manifest and refuses to overwrite its
outputs.

```bash
.venv/bin/python artifacts/day11-long-pilot-evaluation/analyze_sprint10_long_pilot.py \
  --run-dir artifacts/day10-sparse-long-pilot/runs/20260801T081232Z-1345032536fd \
  --released-config artifacts/day10-sparse-long-pilot/pilot_config.json \
  --sprint8-run-dir artifacts/day8-night-pilot/runs/20260731T213826Z-2e16f365ecfd \
  --smoke-dir artifacts/day10-sparse-long-pilot/smoke-210 \
  --projection artifacts/day10-sparse-long-pilot/sparse_analysis.json \
  --output /tmp/sprint11-long-pilot-analysis.json \
  --run-sha-index-output /tmp/sprint11-sprint10-run-sha256sums
cmp artifacts/day11-long-pilot-evaluation/long_pilot_analysis.json \
  /tmp/sprint11-long-pilot-analysis.json
cmp artifacts/day11-long-pilot-evaluation/SPRINT10_RUN_SHA256SUMS \
  /tmp/sprint11-sprint10-run-sha256sums
sha256sum -c artifacts/day11-long-pilot-evaluation/SPRINT10_RUN_SHA256SUMS
(cd artifacts/day11-long-pilot-evaluation && sha256sum -c SHA256SUMS)
```

## Persistence boundary

The immutable raw run is 437,505,563 bytes, while the repository has no Git-LFS policy and its
existing packed history is only about 21 MiB. The 50 checkpoints also duplicate growing complete
histories. The raw directory is consequently integrity-indexed but deliberately not committed to
ordinary Git. Preserve it unchanged locally and mirror the directory plus
`SPRINT10_RUN_SHA256SUMS` byte-for-byte to backed-up institutional research storage. Verify the
index after copying. Do not delete, rewrite, compress, move, ignore, or partially stage it as part
of this evaluation.

## Recommendation

GO to freeze the validation-based earliest-tie selection rule and all H100 semantics, then
predeclare and execute a multiple-independent-seed H100 replication in a later authorized
sprint. H200/H400 remains after that replication, and Test remains closed until every decision is
frozen for the final one-time protocol.
