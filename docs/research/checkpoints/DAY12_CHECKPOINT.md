# Day 12 checkpoint: H100 replication protocol freeze

## Boundary

- Branch: `research/differentiable-mellinger`
- Initial HEAD: `a269046fb0e70cb5ecc8b4c1783b87aecd866b7f`
- Initial and final permitted untracked path: `artifacts/day10-sparse-long-pilot/runs/`
- Protocol artifact: `artifacts/day12-h100-replication-protocol/`
- No research run, seed execution, tmux session, Resume, H200/H400, Test access, push, or amend
- Sprint-10 seed `20260731` remains exploratory and excluded from the confirmatory panel

## Ten frozen seeds

The versioned SHA-256 namespace is
`crazyflow-gradient-research|h100-confirmatory-replication|v1|a269046fb0e70cb5ecc8b4c1783b87aecd866b7f`.
For index `NN`, hash `namespace|seed-index=NN|retry=K`, interpret the first four bytes unsigned
big-endian, clear the high bit, and reject zero/collision/pilot seed. Every seed used `K=0`.

```text
01  1432116264
02   366692846
03   235438753
04  1369406745
05  1081462774
06  1276309202
07   987511476
08  2125653507
09    31735934
10   986925065
```

## Frozen semantics

The ten validated configs differ from the Sprint-10 pilot config only in `run_id` and
`root_seed`. They retain CPU, H100, four Train/four fixed Validation worlds, 5,000 updates,
checkpoint interval 100, `fixed_support_prefix_v2`, Stage-1 `kp_xy/kp_z/kd_xy/kd_z`, bounds,
Adam `0.001`, all loss weights/scales, mass-only ±0.0002-kg randomization, controller/dynamics
mass assumptions, seed tree, and manifests. Delay, Wrench, sensor noise, UKF, non-Stage-1 gains,
H200/H400, Test, parallel runs, seed loops, Resume, and extension remain off.

Source/runtime/gain/Validation fingerprints are pinned to
`6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e`,
`e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4`,
`2e8dbc409dc994f8376d132757ad8c3b0a8744805865c71340e405579c5f3d99`, and
`11e53e460bc93a64835af03bd5a6d22a2e4fb3f7c9d187324d23e7a3deac0c20`.

## Selection and analysis

Selection considers every completed post-update step 1..5,000. The strict code condition
`validation_loss < best_validation` chooses the fixed-Validation global minimum and retains the
earliest exact tie; step 0 and Test are excluded. Sparse file checkpoints do not restrict the
candidate steps because every checkpoint stores the current selected step and raw gains.

Primary is selected Validation loss. Report all ten individual values, mean, `ddof=1` SD,
two-sided 95% Student-t interval (`df=9`, `t=2.2621571627409915`), median, and linear-method
Q1/Q3. Relative step-1-to-selected Validation improvement uses the same display. Secondary
runner-emitted loss, gradient, gain, bound, tracking, technical, and resource values are
descriptive with no secondary p-values or multiplicity claim.

No result-dependent exclusion, replacement seed, or imputation exists. A single complete update-0
retry is possible only after a documented exogenous failure; scientific/technical failures are
not retried and Resume is forbidden. Any second/nonretryable failure invalidates the seed, stops
further starts, and makes H200/H400 NO-GO.

## Horizon gate

All ten `analysis_plan.json` conditions must pass: 10/10 valid and fully persisted seeds, complete
integrity/Test gates, positive improvement for all, at least 50% improvement for 8/10, lower 95%-
CI bound for mean improvement at least 50%, primary relative CI half-width at most 25%, every
final Validation loss at most 110% of its selected minimum, every selected normalized gain margin
at least 0.01, and no protocol drift. Any miss is NO-GO and authorizes no tuning, added seed,
H200/H400, or Test.

## Resources and persistence

- Expected/conservative per seed: 925.96 / 2,289.55 seconds
- Hard per seed: 7,200 seconds, 12 GiB Peak-RSS, zero swap
- Expected sequential Peak-RSS: 2,201,956 KiB
- Expected/conservative ten-seed compute: 2:34:19.60 / 6:21:35.45
- Expected/factor-two raw volume: 4,375,055,630 / 8,753,335,700 bytes
- Required external free space before start: at least 10,000,000,000 bytes

Every attempt must be indexed with SHA-256, copied byte-for-byte to approved backed-up external
research storage, and fully verified at destination before the next seed. Raw output remains
outside ordinary Git.

## Verification

```text
Protocol SHA256SUMS: 20/20 PASS
Protocol verifier: PASS
Seed count/config count: 10/10
Source/runtime fingerprints: PASS
Selection/emitted-metric/failure/GO-gate cross-checks: PASS
Test manifest opened: false
Research runs started: 0
Sprint-13 dry-run-only template: bash syntax PASS; safe dry-run PASS
Generator clean-target replay: byte-identical generated JSON
Ruff check/format: PASS
pip check: PASS
Focused config/seed tests: 3 passed
```

`SHA256SUMS` is 1,716 bytes, hashes to
`e7fb98455029bff42f7cfec562dc541a1f9c7df1cfecc6156f279b3d7e63b3d1`, and covers all 20
payload files. The protocol directory contains 21 files totaling 93,704 bytes. A generator replay
into a new empty temporary directory is byte-identical for every generated JSON file.

## Next step

Sprint 13 may revalidate the committed protocol, implement/review a one-seed launcher, and only
under new explicit authorization execute sequentially. Sprint 12's release decision remains
`NO-GO_RUNS_IN_SPRINT12`.
