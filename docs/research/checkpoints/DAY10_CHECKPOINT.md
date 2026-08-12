# Day 10 checkpoint: sparse checkpoint implementation and long-pilot GO

## Boundary and commits

- Branch: `research/differentiable-mellinger`
- Initial HEAD: `156128505b85012385f480e0503c8eaddd188d09`
- Initial message: `Evaluate Sprint 8 and gate Sprint 9 pilot`
- Implementation commit: `a58b78861cd15e904f7883668a689f297524f4a1`
- Implementation message: `Add sparse research checkpoints`
- Artifact/documentation commit: the commit containing this checkpoint
- No dependency, lockfile, environment, controller/dynamics default, Test manifest, loss, gain,
  optimizer update, trajectory, seed, split, domain-randomization, or Validation change.

## Implemented contract

`OptimizerConfig.checkpoint_interval` is a positive integer with default 1. The default is omitted
from canonical serialization, preserving old configs' canonical dictionaries, fingerprints, and
per-update behavior. Explicit intervals write at exact completed-update multiples. The configured
final update and a planned process stop always receive a checkpoint; coincident conditions produce
one file. An interval greater than the total still writes the final update.

Checkpoints retain the existing full atomic/`fsync`/checksum state and strict compatibility
checks. Metrics use durable append of only the new Train/Validation pair after every completed
update. Resume initializes a new metrics file once from checkpoint history and appends subsequent
updates without missing or duplicate rows. Final summary and each checkpoint retain complete
history.

## Tests

```text
pip check: PASS; No broken requirements found
Ruff check crazyflow examples tests: PASS
Ruff format check crazyflow examples tests: PASS; 113 files
Focused Config/Checkpoint/Runner/Research set:
  PASS; 26 passed, 5 skipped, 22 deselected in 52.18 s
Sparse schedule parameterization:
  PASS; 4 passed in 24.88 s
```

The tests explicitly cover legacy interval-one behavior and old Sprint-9 fingerprint, valid 100,
invalid zero/negative/float/bool/string intervals, exact divisible/nondivisible/greater-than-total
schedules, no duplicate final, atomic/checksum/overwrite/nonfinite protection, config/registry/
source/runtime mismatches, complete provenance, exact optimizer/gain/history/metric Resume, no
missing or duplicate Resume metrics, and an intentionally nonexistent Test path that the runner
does not open.

## Real H100×4 sparse smoke

- Config: 210 updates, interval 100, otherwise exact Sprint-9 scientific semantics
- Config fingerprint: `27b0b8ebb9989077b560d5eb5fa7ace57e910ea81c1fa984d5c02c40f2fc13c5`
- Source fingerprint: `6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e`
- Exit: 0; status success; selected step 210
- External/internal wall: 55.99 / 53.772237384 seconds
- Train/Validation compile: 7.997607284 / 0.798102518 seconds
- Steady noncheckpoint update median/p95/p99/max:
  0.170630664 / 0.1792951942 / 0.1898798258 / 0.428558212 seconds
- Peak-RSS: 1,309,284 KiB; swaps: 0
- Checkpoint 100: 335,925 bytes / 0.036024238 seconds
- Checkpoint 200: 667,310 bytes / 0.069061382 seconds
- Checkpoint 210: 700,428 bytes / 0.063584925 seconds
- All three internal payload checksums pass.
- Exactly 420 ordered rows cover Train and Validation for every step 1..210; all values finite and
  all technical gates pass.
- Test opened/metrics: false/false.

Fresh-process Resume from step 200 to 210 exited 0 in 20.44 seconds, Peak-RSS 1,217,916 KiB, zero
swap, and exact `rtol=0`, `atol=0` equivalence for every contract field. The resumed metrics file
also contains exactly the same 420 rows without missing or duplicate updates.

## Conservative 5,000-update projection

The projection deliberately uses the maximum measured noncheckpoint steady update for all 5,000
updates, twice the maximum measured checkpoint seconds/byte over the linearly projected 50
checkpoints, twice the first-two-update startup, and 30 seconds fixed reserve.

| Quantity | Projection/gate |
|---|---:|
| Conservative walltime | 2,289.5453 s / 38.16 min |
| Hard wall limit | 7,200 s |
| Final checkpoint | 16,573,248 bytes |
| All 50 checkpoints | 422,729,347 bytes |
| Expected total artifacts | 437,666,785 bytes |
| Doubled conservative artifacts | 875,333,570 bytes |
| Artifact gate | 2,147,483,648 bytes |
| Expected files after completion | 61 |

The old 41.4-GB per-update checkpoint projection is eliminated without changing scientific
semantics. The byte-reproducible analysis is `sparse_analysis.json`.

## Release

Decision: **GO** for exactly one H100×4, seed-20260731, 5,000-update CPU pilot with interval 100
in detached tmux. Pilot config SHA-256 is
`6c6c166a19d4f9fb306597cdd111ff9fd78d7b176068635a7cd21045480864f1`; canonical fingerprint is
`859ca650fbe598c24172cba233b9ed15c7f51d23063b7dd6210847ead8910769`.

The launcher pins GO, config and source fingerprints, clean worktree, unique root,
`/usr/bin/time -v`, `timeout --signal=TERM --kill-after=30s 7200s`, full logs, and process-group
termination. It has no Resume/second-run entry point. The long run may start only after the final
artifact/documentation commit and a clean worktree. No Test, H200/H400, seed loop, second run, or
automatic extension is authorized.

## Artifact integrity

`artifacts/day10-sparse-long-pilot/SHA256SUMS` lists 24 preparation payloads, is 2,328 bytes, and
hashes to `182f30383d8b92bd0115de8258c9bd142fb91725e4d05dfe5f6d78e48d5daf2a`.
All entries pass. Future active/completed `runs/` output is deliberately outside this preparation
index.
