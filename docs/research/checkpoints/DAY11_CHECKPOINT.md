# Day 11 checkpoint: completed Sprint-10 long-pilot evaluation

## Boundary

- Branch: `research/differentiable-mellinger`
- Pre-launch and run-provenance HEAD: `1345032536fd18b7d69d7bc29a7819f1e57e58e0`
- Evaluated run: `artifacts/day10-sparse-long-pilot/runs/20260801T081232Z-1345032536fd`
- Exactly one existing run; no new run, Resume, continuation, seed loop, H200/H400, or Test action
- Raw run unchanged and deliberately not staged in ordinary Git

## Integrity result

Status: **PASS**. Launcher and external exit status are zero. The runner completed exactly
5,000/5,000 updates and selected step 5,000. Exactly 50 atomic checkpoints cover
100/200/.../5,000. Every payload checksum passes; every checkpoint prefix, history, counter, and
fingerprint matches. Metrics contain exactly 10,000 ordered rows, one Train and one Validation
record for each update, with no duplicate or missing step. All metric, gradient, gain, optimizer,
and checkpoint values are finite; all technical gates pass.

Config/source/runtime/gain-registry/Validation-manifest fingerprints are respectively
`859ca650fbe598c24172cba233b9ed15c7f51d23063b7dd6210847ead8910769`,
`6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e`,
`e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4`,
`2e8dbc409dc994f8376d132757ad8c3b0a8744805865c71340e405579c5f3d99`, and
`11e53e460bc93a64835af03bd5a6d22a2e4fb3f7c9d187324d23e7a3deac0c20`.
Root seed is `20260731`; source was clean. Summary, selection, metrics, provenance, and every
checkpoint agree that Test was not opened, evaluated, or used.

## Scientific trajectory

| Quantity | Initial | Final | Minimum / step | Relative final change |
|---|---:|---:|---:|---:|
| Train loss | 0.0102038635 | 0.0025606975 | 0.0022851648 / 4,931 | -74.90% |
| Validation loss | 0.0100101354 | 0.0025241498 | 0.0025241498 / 5,000 | -74.78% |
| Gradient-L2 | 0.0062514464 | 0.0002818444 | 0.0002585518 / 4,931 | -95.49% |

Validation decreases at every transition. Early/middle/late slopes over 500-update windows are
`-6.19994e-6`, `-6.43868e-7`, and `-1.84670e-7` loss/update. Late Validation and Gradient-L2
endpoints improve another 3.53% and 16.28%, so the declared plateau indicators are false. Train
and Gradient-L2 have post-hoc oscillation indicators from resampled Train episodes; fixed
Validation does not. There is no divergence, technical saturation, or nonfinite result.

Mean and final Validation-minus-Train gaps are `-0.0000293970` and `-0.0000365477`; the sign is
negative for 57.18% and positive for 42.82% of updates. This small fluctuating gap is descriptive,
not a population-generalization estimate.

## Gains and bounds

Every persisted checkpoint gain is finite and inside bounds. Final physical gains and normalized
nearest-bound margins are:

| Gain | Final | Margin |
|---|---:|---:|
| `kp_xy` | 0.3351839483 | 0.21380 |
| `kp_z` | 2.3107547760 | 0.08602 |
| `kd_xy` | 0.7023677230 | 0.13018 |
| `kd_z` | 1.0894526243 | 0.10050 |

`kp_z` crosses the declared 0.10 watch threshold but remains well above the 0.01 saturation
criterion. Sparse checkpoints persist gains at 100-update intervals only. All 50 exact samples
are analyzed; unavailable per-update intermediate vectors are not inferred.

## Continuity and resources

The first 50 metric updates are byte-identical to Sprint 8; the first 210 are byte-identical to
the Sprint-10 smoke. Final long-run Train/Validation/Gradient-L2 are 71.80%/73.66%/94.96% below
Sprint-8 final and 70.07%/69.98%/94.31% below smoke final.

- External/internal wall: 925.96 / 923.170801088 seconds
- Peak-RSS: 2,201,956 KiB = 2.100 GiB; swap: 0
- Checkpoint writes: 50 files, 423,588,483 bytes, 40.660681623 seconds total
- Complete run: 61 files, 437,505,563 bytes
- Actual/conservative projected wall ratio: 0.40443
- Actual/expected projected artifact ratio: 0.99963

## Evidence and persistence

The byte-reproducible analysis is
`artifacts/day11-long-pilot-evaluation/long_pilot_analysis.json`; SHA-256
`b2c517f8e41b3fca89da05cc6595979a5640c1639656acd2c5596b26c6ebee14`. The 61-file raw index
hashes to `ed3c8b7c59b96393cde13fb98663096bec21ca2b149dfc19a4a85cc1d9161f5e` and passes in full.
The four-file evaluation inventory hashes to
`ae1ef6d2dae3c285b8d65c23f84a366998e2d1649d1ead529e3ce63672ff6ebc` and passes.

The 437.5-MB raw run is not suitable for the repository's ordinary non-LFS Git history. Preserve
it unchanged and mirror it with the raw index to backed-up institutional research storage, then
verify every hash. This sprint performed no external copy and did not delete, move, compress,
ignore, or partially stage the raw files.

## Recommendation

**GO** to a separate selection-rule/H100-protocol freeze, then a predeclared multiple-independent-
seed H100 replication. Only afterward consider H200/H400 with unchanged methodology. Keep Test
closed until all choices are frozen for the final one-time protocol. The current evidence remains
one exploratory CPU simulation seed, not a main study, uncertainty interval, formal stability
proof, hardware result, or Sim2Real validation.
