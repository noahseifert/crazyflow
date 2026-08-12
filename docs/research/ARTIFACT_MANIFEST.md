# Artifact manifest

Status: 2026-08-08. Listed files are generated scientific evidence. Day-2 artifacts were
untracked when executed but were later included in commit `32cb8ffa`; Day-3 artifacts were
preserved in checkpoint commit `edc57ba`. The repository must not ignore the entire `artifacts/` tree.
Result JSON files are explicitly unignored because they contain the machine-readable scientific
record.

| Relative path | Purpose | Source or generation command | Required? | Checked in or generated | Reproducible or manual | Sprint |
|---|---|---|---|---|---|---|
| `artifacts/environment-pip-freeze.txt` | Historical dependency snapshot | `.venv/bin/python -m pip freeze` captured to the file | Optional context | Generated, not checked in | Manual snapshot | Day 1 |
| `artifacts/system-info.txt` | Historical commit, status, Python, and WSL system snapshot | `git rev-parse HEAD`, `git status`, `.venv/bin/python --version`, `uname -a` captured to the file | Optional context | Generated, not checked in | Manual snapshot | Day 1 |
| `artifacts/smoke/figure8_result.json` | Earlier H20 Figure-8 validation record, seed 7 | Main example with `--trajectory figure8 --horizon 20 --seed 7 --output-dir artifacts/smoke` | Optional historical proof | Generated, not checked in | Reproducible from recorded metadata | Day 1 |
| `artifacts/smoke/figure8_tracking.png` | Earlier H20 Figure-8 plot, seed 7 | Same command as preceding JSON | Optional historical plot | Generated, not checked in | Reproducible | Day 1 |
| `artifacts/figure8-h200/figure8_result.json` | Earlier H200 Figure-8 validation record, seed 7 | Main example with `--trajectory figure8 --horizon 200 --seed 7 --output-dir artifacts/figure8-h200` | Required historical proof | Generated, not checked in | Reproducible from recorded metadata | Day 1 |
| `artifacts/figure8-h200/figure8_tracking.png` | Earlier H200 Figure-8 plot, seed 7 | Same command as preceding JSON | Required historical plot | Generated, not checked in | Reproducible | Day 1 |
| `artifacts/circle-h200/circle_result.json` | Earlier H200 circle validation record, seed 7 | Main example with `--trajectory circle --horizon 200 --seed 7 --output-dir artifacts/circle-h200` | Required historical proof | Generated, not checked in | Reproducible from recorded metadata | Day 1 |
| `artifacts/circle-h200/circle_tracking.png` | Earlier H200 circle plot, seed 7 | Same command as preceding JSON | Required historical plot | Generated, not checked in | Reproducible | Day 1 |
| `artifacts/mellinger_tracking/figure8_result.json` | Earlier default H200 Figure-8 record, seed 20260724 | Main example defaults with `--output-dir artifacts/mellinger_tracking` | Optional duplicate proof | Generated, not checked in | Reproducible from recorded metadata | Day 1 |
| `artifacts/mellinger_tracking/figure8_tracking.png` | Earlier default H200 Figure-8 plot | Same command as preceding JSON | Optional duplicate plot | Generated, not checked in | Reproducible | Day 1 |
| `artifacts/day1-audit/smoke-run-1/figure8_result.json` | Audited H20 record and all validation flags | `RUNBOOK.md` smoke command | Required audit proof | Generated, not checked in | Reproducible | Day-1 audit |
| `artifacts/day1-audit/smoke-run-1/figure8_tracking.png` | Audited H20 plot | `RUNBOOK.md` smoke command | Required audit plot | Generated, not checked in | Reproducible | Day-1 audit |
| `artifacts/day1-audit/smoke-run-2/figure8_result.json` | Independent fixed-seed replay record | `RUNBOOK.md` reproducibility command | Required audit proof | Generated, not checked in | Reproducible | Day-1 audit |
| `artifacts/day1-audit/smoke-run-2/figure8_tracking.png` | Independent fixed-seed replay plot | `RUNBOOK.md` reproducibility command | Required audit plot | Generated, not checked in | Reproducible, byte-identical to run 1 | Day-1 audit |
| `artifacts/day1-audit/figure8-h200/figure8_result.json` | Audited full H200 Figure-8 record | `RUNBOOK.md` full Figure-8 command | Required audit proof | Generated, not checked in | Reproducible | Day-1 audit |
| `artifacts/day1-audit/figure8-h200/figure8_tracking.png` | Audited full H200 Figure-8 plot | `RUNBOOK.md` full Figure-8 command | Required audit plot | Generated, not checked in | Reproducible | Day-1 audit |
| `artifacts/day1-audit/circle-h200/circle_result.json` | Audited full H200 circle record | `RUNBOOK.md` full circle command | Required audit proof | Generated, not checked in | Reproducible | Day-1 audit |
| `artifacts/day1-audit/circle-h200/circle_tracking.png` | Audited full H200 circle plot | `RUNBOOK.md` full circle command | Required audit plot | Generated, not checked in | Reproducible | Day-1 audit |
| `artifacts/day2-audit/smoke-run-1/batch_diagnostics.json` | First H20 two-world scientific record, gradients, sensitivity, profiling, and validations | `RUNBOOK.md` Day-2 H20 smoke run 1 command | Required; passed | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Reproducible except named profiling-time blocks | Day 2 |
| `artifacts/day2-audit/smoke-run-1/batch_tracking.png` | First H20 Figure-8/train and circle/validation tracking plot | Same H20 smoke run 1 command | Required; visually checked | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Byte-reproducible; SHA-256 `578fa448...f001` | Day 2 |
| `artifacts/day2-audit/smoke-run-1/loss_and_sensitivity.png` | First H20 loss-component and controlled-sensitivity plot | Same H20 smoke run 1 command | Required; visually checked | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Byte-reproducible; SHA-256 `00868b6a...9cbc` | Day 2 |
| `artifacts/day2-audit/smoke-run-1/gradient_diagnostics.png` | First H20 gradient and directional-derivative plot | Same H20 smoke run 1 command | Required; visually checked | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Byte-reproducible; SHA-256 `d05fe557...ec2e` | Day 2 |
| `artifacts/day2-audit/smoke-run-2/batch_diagnostics.json` | Independent H20 fixed-seed replay record | `RUNBOOK.md` Day-2 H20 smoke run 2 command | Required; passed | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Scientific payload equals run 1 after only named profiling-time exclusions | Day 2 |
| `artifacts/day2-audit/smoke-run-2/batch_tracking.png` | Independent H20 tracking-plot replay | Same H20 smoke run 2 command | Required; byte comparison passed | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Byte-identical to run 1; SHA-256 `578fa448...f001` | Day 2 |
| `artifacts/day2-audit/smoke-run-2/loss_and_sensitivity.png` | Independent H20 loss/sensitivity-plot replay | Same H20 smoke run 2 command | Required; byte comparison passed | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Byte-identical to run 1; SHA-256 `00868b6a...9cbc` | Day 2 |
| `artifacts/day2-audit/smoke-run-2/gradient_diagnostics.png` | Independent H20 gradient-plot replay | Same H20 smoke run 2 command | Required; byte comparison passed | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Byte-identical to run 1; SHA-256 `d05fe557...ec2e` | Day 2 |
| `artifacts/day2-audit/batch-h200/batch_diagnostics.json` | Full H200 two-world record and Day-1 regression evidence | `RUNBOOK.md` Day-2 H200 full-batch command | Required; all validations passed | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Reproducible from recorded source state; timing varies | Day 2 |
| `artifacts/day2-audit/batch-h200/batch_tracking.png` | Full H200 per-case tracking plot | Same H200 full-batch command | Required; visually checked | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Deterministic; SHA-256 `c18195f5...c80a` | Day 2 |
| `artifacts/day2-audit/batch-h200/loss_and_sensitivity.png` | Full H200 components and physical ±10% sensitivity plot | Same H200 full-batch command | Required; visually checked | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Deterministic; SHA-256 `9a55e7bc...5584` | Day 2 |
| `artifacts/day2-audit/batch-h200/gradient_diagnostics.png` | Full H200 train/validation/combined gradient and finite-difference plot | Same H200 full-batch command | Required; visually checked | Generated; tracked in later Sprint-2 commit `32cb8ffa` | Deterministic; SHA-256 `a5f380ad...c63` | Day 2 |
| `artifacts/day3-audit/smoke-run-1/optimization_result.json` | H20 three-update optimization, gradients, validations, and profiling | `RUNBOOK.md` Day-3 smoke run 1 | Required; passed | Generated working-tree evidence | Scientific payload replayed after six timing exclusions | Day 3 |
| `artifacts/day3-audit/smoke-run-1/optimized_gains.json` | Compact selected H20 research gains | Same smoke command | Required | Generated working-tree evidence | Byte-identical to smoke run 2 | Day 3 |
| `artifacts/day3-audit/smoke-run-1/robustness_results.json` | H20 trajectory/mass robustness cells and summaries | Same smoke command | Required | Generated working-tree evidence | Byte-identical to smoke run 2 | Day 3 |
| `artifacts/day3-audit/smoke-run-1/optimization_history.png` | Train/validation/combined smoke history | Same smoke command | Required; visually checked | Generated working-tree evidence | Byte-identical; SHA-256 `f2ae0470...82ce` | Day 3 |
| `artifacts/day3-audit/smoke-run-1/gain_history.png` | Bounded physical gain smoke history | Same smoke command | Required; visually checked | Generated working-tree evidence | Byte-identical; SHA-256 `e0f8a295...35bc` | Day 3 |
| `artifacts/day3-audit/smoke-run-1/robustness_matrix.png` | H20 baseline/best trajectory/mass comparison | Same smoke command | Required; visually checked | Generated working-tree evidence | Byte-identical; SHA-256 `5f1a4635...4539` | Day 3 |
| `artifacts/day3-audit/smoke-run-2/optimization_result.json` | Independent H20 three-update replay | `RUNBOOK.md` Day-3 smoke run 2 | Required; passed | Generated working-tree evidence | Scientific payload equals run 1 after six timing exclusions | Day 3 |
| `artifacts/day3-audit/smoke-run-2/optimized_gains.json` | Independent compact selected H20 gains | Same smoke command | Required | Generated working-tree evidence | Byte-identical to smoke run 1 | Day 3 |
| `artifacts/day3-audit/smoke-run-2/robustness_results.json` | Independent H20 robustness replay | Same smoke command | Required | Generated working-tree evidence | Byte-identical to smoke run 1 | Day 3 |
| `artifacts/day3-audit/smoke-run-2/optimization_history.png` | Independent optimization-history replay | Same smoke command | Required; byte comparison passed | Generated working-tree evidence | SHA-256 `f2ae0470...82ce` | Day 3 |
| `artifacts/day3-audit/smoke-run-2/gain_history.png` | Independent gain-history replay | Same smoke command | Required; byte comparison passed | Generated working-tree evidence | SHA-256 `e0f8a295...35bc` | Day 3 |
| `artifacts/day3-audit/smoke-run-2/robustness_matrix.png` | Independent robustness-plot replay | Same smoke command | Required; byte comparison passed | Generated working-tree evidence | SHA-256 `5f1a4635...4539` | Day 3 |
| `artifacts/day3-audit/optimize-h200/optimization_result.json` | Full 50-update record, selected-checkpoint gradients, validations, and profiling | `RUNBOOK.md` Day-3 H200 command | Required; all validations passed | Generated working-tree evidence | Reproducible except named timing fields | Day 3 |
| `artifacts/day3-audit/optimize-h200/optimized_gains.json` | Bestes getestetes Train-Checkpoint des festgelegten H200-Laufs; simulationsoptimierter Gain-Kandidat | Same H200 command | Required | Generated working-tree evidence | Deterministic; defaults unchanged | Day 3 |
| `artifacts/day3-audit/optimize-h200/robustness_results.json` | Complete 32-cell H20/H100/H200/H400 matrix | Same H200 command | Required; complete and finite | Generated working-tree evidence | Deterministic scientific payload | Day 3 |
| `artifacts/day3-audit/optimize-h200/optimization_history.png` | Full train/validation/combined loss history | Same H200 command | Required; visually checked | Generated working-tree evidence | SHA-256 `99888e47...06f5` | Day 3 |
| `artifacts/day3-audit/optimize-h200/gain_history.png` | Full bounded physical gain history | Same H200 command | Required; visually checked | Generated working-tree evidence | SHA-256 `6899c713...10e` | Day 3 |
| `artifacts/day3-audit/optimize-h200/robustness_matrix.png` | Full baseline/best horizon/trajectory/mass plot | Same H200 command | Required; visually checked | Generated working-tree evidence | SHA-256 `bf54eb82...fdf6` | Day 3 |

The historical files record commit `800a4e4835c3e5e8e47db82422cddb39a474ab40` plus a source-state
hash from the then-uncommitted Day-1 implementation. The new audit records run the committed
implementation at `1aa648b15e9e9ed9bada8aaa7408cdff22d417fd`. Timing fields are expected to vary;
the fixed-seed scientific payload and plots are the reproducibility targets.

Day-2 records use base/end HEAD `9eac04ea28997e76b9bc8c6e97ca53ef93398d55` plus a source-state
hash covering the uncommitted Day-2 source/test state and excluding generated artifacts. The two
H20 records have identical source metadata because they were generated before the later
documentation-only edits. Their scientific payloads compare equal after excluding only the six
profiling-time blocks named in `RUNBOOK.md`; all three plot pairs are byte-identical. Their
execution metadata remains unchanged, but all twelve files are now tracked because the later
Sprint-2 commit `32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed` included them.

Day-3 records start from that clean Sprint-2 commit and use scientific source-state
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`. The two smoke runs have
equal scientific optimization payloads after removing only the six timing keys named in
`RUNBOOK.md`; both compact-gain and robustness JSON pairs and all three PNG pairs are
byte-identical. The H200 record reports all validations true and a complete 32-cell matrix.

## Sprint-4 documentation and artifact freeze

Sprint 4 created no scientific artifact and changed or regenerated none of the files listed
above. Read-only verification reconfirmed all eighteen Tag-3 paths, their exact directory
membership, and every SHA-256 recorded in
[`checkpoints/DAY3_CHECKPOINT.md`](checkpoints/DAY3_CHECKPOINT.md). The actual audit is recorded in
[`checkpoints/DAY4_CHECKPOINT.md`](checkpoints/DAY4_CHECKPOINT.md).

The following new files are explanatory and handover documentation, not scientific result
artifacts:

- [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md)
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)
- [`SUPERVISOR_MEETING_BRIEF.md`](SUPERVISOR_MEETING_BRIEF.md)

Because Sprint 4 started from the documented uncommitted Sprint-3 state rather than a later
checkpoint commit, the Day-3 rows correctly remain `Generated working-tree evidence`. Historical
JSON execution metadata was not rewritten.

## Sprint-5 technical smoke and scaling evidence

The table's older Day-3 “generated working-tree” wording records execution-time status. All those
files are now tracked by `edc57ba0bb884b69305b2a6af979e29f2de4566e`. Sprint-5 records were
generated from a clean worktree at implementation HEAD
`c4492b187d336d75757c560fecced53c7ec47b7a`. They are technical verification, not final scientific
test results.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day5-audit/local-smoke/resolved_config.json` | Complete resolved smoke config, source, fingerprint | 1802 | `e77f41cc2c7fe55fbad79ca5f6890c5d433b5e27aef1fd71cc37efad9802bffa` |
| `artifacts/day5-audit/local-smoke/provenance.json` | Clean HEAD, command, versions, UTC interval, internal walltime | 1449 | `6be4f80191d0ae0c9b8b3b2a88174e9af0b12ad56a711f180981690bfa9cdee0` |
| `artifacts/day5-audit/local-smoke/metrics.jsonl` | Separate one-step Train and fixed-Validation metrics | 2012 | `0e9c818bd24d9b22e4786886fe1ced0cbee8ef369c7a4d55749a22677b76faee` |
| `artifacts/day5-audit/local-smoke/checkpoint-step-000001.json` | Raw gains, complete Optax state, counters, selection, manifests, fingerprints | 3234 | `e7edd06ed3c1556841b5de3f88c25b020bff9806df2bb5ca5b87106424c0b44b` |
| `artifacts/day5-audit/local-smoke/run_summary.json` | Technical status, selected/final gains, timing, warnings, no test metrics | 1522 | `55d99ee60aaacf10167eb513baea4d2a44c748af2075b4476e9bd90601641af6` |
| `artifacts/day5-audit/scaling-w1/benchmark.json` | Fresh-process H20 one-world gradient benchmark | 852 | `93e9c4432c74be767584e60b540546de55f70f8a3cc3677bf06004da184a2581` |
| `artifacts/day5-audit/scaling-w2/benchmark.json` | Fresh-process H20 two-world gradient benchmark | 852 | `933c71add523d8eb7f4a3a1c35a90444bad9ec857c686acb9066dfeddc683552` |
| `artifacts/day5-audit/scaling-w4/benchmark.json` | Fresh-process H20 four-world gradient benchmark | 850 | `2fccfdf72d45b2c9e93fd067439f2db24baab30b768a0a7d981166105f5c62f3` |

The local smoke command and all benchmark commands are exact in `RUNBOOK.md`. External
`/usr/bin/time -v` values are in `DAY5_CHECKPOINT.md`; the JSON records deliberately avoid
claiming external Peak-RSS when only the process-internal `resource.ru_maxrss` was available.
There are no Day-5 plots because two scalar split rows and three scaling points do not gain
scientific clarity from a generated figure.

## Sprint-6 Resume and scaling-readiness evidence

All Sprint-6 evidence is under `artifacts/day6-readiness/`. The complete relative-path SHA-256
inventory is `artifacts/day6-readiness/SHA256SUMS`: 46 listed files, 4,848 bytes, SHA-256
`7580bd41d41314b9d13a9fa9ea052b67fcb715930d2ddbb53a9759023c85316d`.
`sha256sum -c SHA256SUMS` reported `OK` for every entry.

The `resume-final-*` and `pilot-final-*` directories are authoritative. They were generated from a
clean executable-source worktree at `182d0cc9068c4920ac25146ff0016260f802e5dd` with scoped source
fingerprint `aae2c6b0b2d4962eb801ddb22947be57af149fc20c1a8d5f08a3dcc9ac14d6da`.
The older `resume-*`, `pilot-h100-w1`, and attempt timing records preserve the diagnosis and
pre-final reruns; they are not the authoritative final comparison.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day6-readiness/resume-final-uninterrupted/checkpoint-step-000002.json` | Final uninterrupted two-update state | 10903 | `e97f03182d60a42dd13a408f003c146adade96c84e4ca1b367c16c49d837b421` |
| `artifacts/day6-readiness/resume-final-resumed/checkpoint-step-000002.json` | Final one-update plus fresh-process-resume state | 11773 | `896eeaf3035de85125c1ce542f43a75e2a0f13556af0ce477d9734d56cf5a740` |
| `artifacts/day6-readiness/resume-final-resumed/resume_equivalence.json` | Exact scientific-state comparison, all required fields pass | 1840 | `859d3403fa5e07d6c9596761aaa9c28144fbdf2ead2ac5921e2ec1b673c72383` |
| `artifacts/day6-readiness/resume-final-resumed/metrics.jsonl` | Complete two-step Train/Validation metric history after resume | 4841 | `ce10e971029cb5d33fa575468e8a0061c38fbf80e8277def8933f150cbd3717f` |
| `artifacts/day6-readiness/resume-final-resumed/provenance.json` | Config/source/runtime/Validation fingerprints and parent checkpoint hash | 3180 | `351064384f00407f6d3f9465f567594c8292a0031fc8a1317d123185789df7b0` |
| `artifacts/day6-readiness/pilot-final-h100-w1/benchmark.json` | Machine-readable pre-JIT H100 trajectory-rejection/NO-GO record | 1776 | `bedee2cccf753a08cafb325f38b5539fec914edd2193920190ea2c1362e6c4a2` |
| `artifacts/day6-readiness/timing/resume-final-uninterrupted-h20.time.txt` | External time/RSS/swap/exit evidence | 966 | `7a8395e3cb000badf7e881af9ff42ae7c03f69899dc54a5cb556751a1d0b6584` |
| `artifacts/day6-readiness/timing/resume-final-paused-h20.time.txt` | External planned-pause time/RSS/swap/exit evidence | 988 | `5da082617b970f598c98789fad305c8aeecdf91b0f0e3d82863deea136538567` |
| `artifacts/day6-readiness/timing/resume-final-resumed-h20.time.txt` | External resumed time/RSS/swap/exit evidence | 1157 | `0a1c95c0bcf08c09aa6ee677c7eb4524883282fe26ed1cae47d0122bf48ac159` |
| `artifacts/day6-readiness/timing/pilot-final-h100-w1.time.txt` | External failed-pilot time/RSS/swap/exit evidence | 1020 | `709db237959ce9c7d8bffaa9445785c0e792ddd6ff5c14abc9698a93caf17e05` |

Resolved configs, summaries, every intermediate checkpoint, complete metrics, full provenance,
diagnostic attempts, and timing files are individually listed in `SHA256SUMS`. No plot was created:
the decisive result is an exact state equality plus a pre-JIT gate failure, for which a figure
would add no evidentiary value. No Test episode or Test metric artifact exists.

## Sprint-7 trajectory diagnosis, construction gate, Resume, and CPU pilots

All Sprint-7 evidence is under `artifacts/day7-trajectory/`. The complete inventory is
`artifacts/day7-trajectory/SHA256SUMS`: 46 listed files, all verified with `sha256sum -c`; the
index hashes to `2340d25908c737f05581dd1d9766f3ff8c1f190301008059e828da48b0730bd8`.

The unchanged Phase-A failure was generated at initial HEAD `3286b5dc...`; construction-gate,
pilot, and Resume evidence used clean final executable-source HEAD
`1080aadf65a6df56528efcaa5da34f4dfc812e7d`, source fingerprint
`8ff8317b01de2aafda276bd13af3f2ee920a9610b764344c05d0fc620ae87320` where recorded, and explicit
`sprint7_pilot.json`. No Test manifest, Test episode, or Test metric was opened or generated.

| Relative path | Purpose | SHA-256 |
|---|---|---|
| `artifacts/day7-trajectory/phase-a/reproduce-h100-w1/benchmark.json` | Unchanged Sprint-6 H100×1 rejection reproduction | `135baa6a141411a1596477e68a2237874fa01d49b6cb8f4920c1a127cc09d50f` |
| `artifacts/day7-trajectory/phase-a/h100-attempt-diagnostics/trajectory_diagnostics_raw.json` | Exact 16-attempt signed H100 constraint evidence | `12e5c1fb89aafb7e6d921d7fa052d5d119201f340237e0954733a5e3e83eec04` |
| `artifacts/day7-trajectory/phase-b/legacy-horizons-32-seeds/trajectory_diagnostics_raw.json` | Legacy H20/H50/H100/H200/H400 raw construction audit | `89acfaa4ba3b064b57cd95d4843800d060349d4126ccb04fb2afa29d90e04c99` |
| `artifacts/day7-trajectory/phase-b/legacy-horizons-32-seeds/trajectory_diagnostics_summary.json` | Acceptance, rejection, diversity, spectral, and runtime summary | `623e5bc4a17dfe63d3bdb13b359d94af0515dcc2396e9b68bcaecd2376a6b643` |
| `artifacts/day7-trajectory/phase-b/legacy-h100-16-seeds-128-attempt-estimate/trajectory_diagnostics_raw.json` | Estimate-only 2,048-candidate H100 evidence | `0acd815754db85cdaccd0ec5e6eeef3a015532a0d394ea008d12dacdbfc0cf3e` |
| `artifacts/day7-trajectory/decision/pre_gate_criteria.json` | Options and thresholds frozen before v2 measurement | `50dac4fa8e76fa26b79f30e16de050a8649e06533d794892f30eb097e0aff2b4` |
| `artifacts/day7-trajectory/decision/gate128_evaluation.json` | Final technical/diversity/compatibility gate evaluation | `5f51b22d80cbe25c7b1f0edceec48f700d2911a48b9a7d67a0ce869f4ea04e55` |
| `artifacts/day7-trajectory/gate128/fixed-prefix-run1/trajectory_diagnostics_raw.json` | First 128-seed H100/H200/H400 v2 construction gate | `5289b405ccee37e9193aa773d0413d9a37220e4487e9b01089681369fbf1b63b` |
| `artifacts/day7-trajectory/gate128/fixed-prefix-run2/trajectory_diagnostics_raw.json` | Independent deterministic construction replay | `6aae502495674263492edb1abbc9f34aa3b4c8f57ed507bc45f2fb5708fb4219` |
| `artifacts/day7-trajectory/pilots/h100-w1/benchmark.json` | Successful real CPU H100×1 value/gradient pilot | `a2946cb23f3de34a03c65f866928896df4e46618aff5bfeb661223677a800888` |
| `artifacts/day7-trajectory/pilots/h100-w4/benchmark.json` | Successful real CPU H100×4 value/gradient pilot | `a6f20eebfffc47e76fed6fdb4695349be16033cb038230e32fdeceae37b771db` |
| `artifacts/day7-trajectory/pilots/h200-w4/benchmark.json` | Successful real CPU H200×4 value/gradient pilot | `22a99ee6381daf66f60d0e2a7acc972f4accd6ed621b3051590fc1d08cef2d26` |
| `artifacts/day7-trajectory/resume/resumed/resume_equivalence.json` | Exact H100 uninterrupted-versus-resumed equality | `c1de79c713506c8643802c8ade131c4040b3b4d3269703cb9a626a6040a8aa7b` |

Raw construction JSON includes explicit seed coordinates/list hash, config, Git/source
fingerprints, every attempted candidate's constraint margins and extrema, kinematic and spectral
statistics, array digests, and timing. The duplicated gate raw files intentionally retain both
independent process records; timing differs, while their internal timing-excluded scientific hash
is identical. External `/usr/bin/time -v` files cover every Phase-A/Phase-B/gate/pilot/Resume
process. No plot was needed for exact prefix/hash and constraint evidence.

## Sprint-8 manual night-pilot preparation

All preparation and handover files are under `artifacts/day8-night-pilot/`. `SHA256SUMS` indexes
13 payload files; all pass `sha256sum -c`. The index is 1,235 bytes and hashes to
`7b50c9bb1461cc4ce4b24e8ed9a3e11f682c061c50e12629360d769ffa88ab93`.
The eventual user-created `runs/` tree is deliberately absent from this preparation inventory and
must be hashed after execution.

| Relative path | Purpose | SHA-256 |
|---|---|---|
| `artifacts/day8-night-pilot/preflight_config.json` | Exact H100×4/two-update preparation config | `d4aea5ca02b75ec50967459190ce3cf5a463a761a6430257948caf1b63fb9c8a` |
| `artifacts/day8-night-pilot/preflight-h100-w4/run_summary.json` | Successful finite two-update gate and internal timings | `597a236682cb1e61874c818933e1d274b99672ca1fbbfd8eea763be1f25d8cb4` |
| `artifacts/day8-night-pilot/preflight-h100-w4/metrics.jsonl` | Complete Train/Validation metrics and technical gates | `0e5bbe186510a289d291bd042a1ece49ac694c15ca6f670c8357441377040faa` |
| `artifacts/day8-night-pilot/preflight-h100-w4/checkpoint-step-000001.json` | First atomic complete-update state | `453cea678b22caafdc3a541df27c4b2089c28a9c34261d2fe7411f490f4433f3` |
| `artifacts/day8-night-pilot/preflight-h100-w4/checkpoint-step-000002.json` | Second atomic complete-update state | `6281bbe40c70e52febba7703ada98210f91d6c215a2e6b5beb0e65ff14436d9a` |
| `artifacts/day8-night-pilot/timing/preflight-h100-w4.time.txt` | Exit, external walltime, Peak-RSS, and swap evidence | `f296633bad052c9fa5e7be2ff5e72ddc945568b31cd78df656681f8e1015aa97` |
| `artifacts/day8-night-pilot/preparation_gate.json` | Machine-readable GO and 10/25/50-update budget derivation | `b92faa313b61ef069b0bdd4a16498a484895c32f1d0963d644df00a388d792c2` |
| `artifacts/day8-night-pilot/pilot_config.json` | Released H100×4/50-update night-pilot config | `43a2a75be839ff0ec75c462d3d502f1b5f8330747c5e9b94f71d57ed9f02bd07` |
| `artifacts/day8-night-pilot/launch_night_pilot.sh` | Manual unique-run launcher with timeout, logging, and Resume | `0aa301e7a3912c3278e13caecf1db374e7e8ef03c5399214a0d086a0397d53c9` |
| `artifacts/day8-night-pilot/inspect_pilot.py` | Read-only morning integrity/trend/resource inspection | `809ed7ac12af6851df7f6fc946e3b9b48dbf61de3f4f037b80297df16f079a1d` |
| `artifacts/day8-night-pilot/README.md` | Exact manual start, status, stop, Resume, and handover guide | `ae7a108004f449ac9c2b4a6f164304bad9ece5cbcd57240b13f9c585ea7fa532` |

The other indexed preparation files are the resolved config, provenance, and full run output.
No Test manifest, Test episode, Test metric, H200/H400 process, seed loop, or night-pilot process
was opened or executed in Sprint 8.

### Completed Sprint-8 50-update pilot evidence

The user-started run at
`artifacts/day8-night-pilot/runs/20260731T213826Z-2e16f365ecfd/` contains exactly 61 regular
files totaling 4,602,398 bytes: 50 sequential atomic checkpoints, 100-row Train/Validation
metrics, resolved config, provenance, summary, launch/environment/process records, complete
stdout/stderr, and external timing. There are no links, devices, binary payloads, unexpected
files, sensitive credential patterns, Test metrics, or Test artifacts.

Every checkpoint payload checksum and every technical gate passes. The run recorded clean source
HEAD `2e16f365ecfd60357d846c159549fe910a4fd220`, source fingerprint
`8ff8317b01de2aafda276bd13af3f2ee920a9610b764344c05d0fc620ae87320`, config fingerprint
`6d0465b5776fe1f09b0f5b1ab8777d02b9b3bd01386d8f00cc6209fb97d2efaa`, runtime fingerprint
`e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4`, and root seed
`20260731`. The resolved config equals the released Sprint-8 config. The Test reference remained
opaque; summary fields record `test_metrics_present=false` and `test_manifest_opened=false`.

`artifacts/day8-night-pilot/SPRINT8_RUN_SHA256SUMS` lists all 61 files and passes in full. The
10,137-byte index hashes to
`ff711180460939be60af0052e4ea8449900f5374859608553eb3661f66688a6e`. The run evidence is
generated, tracked scientific evidence; its raw files were not edited or overwritten during
preservation.

## Sprint-9 analysis and blocked long-pilot preparation

All Sprint-9 payloads are under `artifacts/day9-long-pilot/`. `SHA256SUMS` lists nine files; every
entry passes. The 795-byte index hashes to
`80f5c4dde4ddec1e0a2835c2591e1a0737c67289c953586271eb169d471de7af`.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day9-long-pilot/sprint8_analysis.json` | Full reproducible 50-update trajectory, gain, bound, integrity, fingerprint, and scaling analysis | 97,632 | `4d5d501bca1d336478637bd0971ca285e0fde704e491ff6f9eb03602352f9018` |
| `artifacts/day9-long-pilot/sprint8_checkpoint_timing_capture.json` | Verbatim hashed operational checkpoint mtimes used by runtime fits | 4,311 | `2178611f54bea31d42f748ac3b7a6ead7db8d47bd01757e71847262868e7b21d` |
| `artifacts/day9-long-pilot/analyze_sprint8_pilot.py` | Deterministic analysis generator; independent reproduction is byte-identical | 15,848 | `d16f316587510d2f93f61b2e0bfcbaa46f358c4f1108266afd534dbf96775169` |
| `artifacts/day9-long-pilot/pilot_config.json` | Exact requested H100×4/5,000-update scientific config; prepared but not released | 1,487 | `265e5ee3a0669d464eb07fe26e7ec147ef4c26ee0d6ba2a095e309e8362aad1e` |
| `artifacts/day9-long-pilot/release_gate.json` | Machine-readable NO-GO and checkpoint/runtime projections | 2,451 | `2a5b974a465eeb6989c7f6dd31c1f42dd24572620bef70092e6b80a06da5cc28` |
| `artifacts/day9-long-pilot/verification_summary.json` | Config, launcher, tmux, quality-test, and no-start record | 1,970 | `da6344f4715e5fc7f68474180cef4f50ae2138130b42c9c5240d979bf7dce150` |
| `artifacts/day9-long-pilot/launch_long_pilot.sh` | Checked bounded launcher; exits before run creation while gate is NO-GO | 5,989 | `7cc37094749c6764926ed11d96a77cfecd98e1ed496d4a9f7bdb32f3b3d3221b` |
| `artifacts/day9-long-pilot/inspect_long_pilot.py` | Read-only active/interrupted/completed integrity inspector | 5,331 | `1244d9ff6e1c012a1c781fb530099df2667965ccbe27fe46ecc0874abbd94f39` |
| `artifacts/day9-long-pilot/README.md` | NO-GO rationale, reproducibility commands, and disabled operating handover | 4,311 | `969a43239300b1b26af30598f81782bec3b9e0e725b9e79804ccf8edd6084ec4` |

No `runs/` directory, training output, Test artifact, plot, H200/H400 result, or seed-loop result
was generated in Sprint 9. The prepared config is not a released scientific run configuration
while its machine-readable gate remains `NO-GO`.

## Sprint-10 sparse checkpoint smoke, Resume, analysis, and release

All preparation evidence is below `artifacts/day10-sparse-long-pilot/`. `SHA256SUMS` lists 24
payload files; every entry passes. The 2,328-byte index hashes to
`182f30383d8b92bd0115de8258c9bd142fb91725e4d05dfe5f6d78e48d5daf2a`. Future `runs/` output is
excluded and must be indexed only after completion.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day10-sparse-long-pilot/pilot_config.json` | Released H100×4/5,000-update interval-100 config | 1,527 | `6c6c166a19d4f9fb306597cdd111ff9fd78d7b176068635a7cd21045480864f1` |
| `artifacts/day10-sparse-long-pilot/smoke_config.json` | Exact H100×4/210-update interval-100 smoke config | 1,520 | `352e17287a88661e0b32cc60bffc2f9b05118d4d9eaf6c4bd6a6709768d7e304` |
| `artifacts/day10-sparse-long-pilot/smoke-210/checkpoint-step-000100.json` | First complete sparse checkpoint | 335,925 | `942c3f0844dbf5d28c47936c2a56a6521e5fc0c12327d4878561b57460e4e9bf` |
| `artifacts/day10-sparse-long-pilot/smoke-210/checkpoint-step-000200.json` | Second complete sparse/Resume-parent checkpoint | 667,310 | `3b7713f520fa6c94f22165a95b636d85934055b5b2890fe25dcceca60108aca5` |
| `artifacts/day10-sparse-long-pilot/smoke-210/checkpoint-step-000210.json` | Mandatory nondivisible final checkpoint | 700,428 | `2dfbb9a5e7932f6d5ab63b4e5f2d01aa7a15303a5e8d52a9b29870f2ed877774` |
| `artifacts/day10-sparse-long-pilot/smoke-210/metrics.jsonl` | All 420 ordered Train/Validation smoke metrics | 534,014 | `f4043a69c1953b9457b814eb97fc1446cd07cca1b99b4fd775d106aa0780e86a` |
| `artifacts/day10-sparse-long-pilot/smoke-resumed-200-to-210/resume_equivalence.json` | Exact zero-tolerance continuation comparison | 1,849 | `7dd29d5050bd18459c9c8dfa124fceb56cf13a82ac5c3bbe3aa908e356366a0f` |
| `artifacts/day10-sparse-long-pilot/sparse_analysis.json` | Integrity, resource, timing, Resume, and conservative projection | 6,455 | `7499f55e95bea1a06823b5ee666b39c11ce8f931647019c72b0782d525a12ef0` |
| `artifacts/day10-sparse-long-pilot/release_gate.json` | Machine-readable GO and immutable launch constraints | 2,311 | `d71a81fff647d31a52384fe390fe6018fbbcbb8c9ff606c9dc41667d82915205` |
| `artifacts/day10-sparse-long-pilot/launch_long_pilot.sh` | Exact clean-worktree, fingerprint-pinned detached-run launcher | 6,962 | `f9035be550c8ff04c5e974d477f94589b5018df60341896ab1cd0c50c98ab218` |

The index also covers analysis/inspection code, both external timing records, full smoke and
resumed summaries/provenance/configs, the exact resumed checkpoint and complete handover. No Test
manifest, Test episode, Test metric, H200/H400 result, seed-loop result, or long-run result is part
of the preparation evidence.

## Sprint-11 completed long-pilot evaluation

The immutable source run remains at
`artifacts/day10-sparse-long-pilot/runs/20260801T081232Z-1345032536fd/`. It contains exactly 61
regular files totaling 437,505,563 bytes. All 50 checkpoint payload checksums and the independent
raw-file hashes pass; metrics contain exactly 5,000 ordered Train and 5,000 ordered Validation
rows, with no duplicates, nonfinite values, failed technical gate, unexpected split, Test metric,
or Test access. Execution provenance records clean HEAD `1345032536fd18b7d69d7bc29a7819f1e57e58e0`,
config/source/runtime fingerprints `859ca650...0769` / `6b3fed4a...1c5e` /
`e3c408d5...d2a4`, and root seed `20260731`.

The raw run is generated, verified, locally preserved evidence but is not checked in. At 437.5 MB
of primarily growing-history checkpoint JSON, it is unsuitable for this repository's ordinary
Git history, which has no LFS policy. Preserve it unchanged and mirror it with the full hash index
to backed-up institutional research storage. No file was deleted, moved, compressed, ignored, or
partially staged.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day11-long-pilot-evaluation/analyze_sprint10_long_pilot.py` | Deterministic full integrity, trend, gain, resource, comparison, and recommendation generator | 37,531 | `0b5282818422d46030cddc526b4ab83a3dc60ac10627a3d1a93d8aea37a7b1f8` |
| `artifacts/day11-long-pilot-evaluation/long_pilot_analysis.json` | Complete 5,000-update analysis with every loss/gradient value, all persisted gains, gates, and raw hashes | 774,106 | `b2c517f8e41b3fca89da05cc6595979a5640c1639656acd2c5596b26c6ebee14` |
| `artifacts/day11-long-pilot-evaluation/SPRINT10_RUN_SHA256SUMS` | Independently checkable SHA-256 index for all 61 immutable raw-run files | 10,564 | `ed3c8b7c59b96393cde13fb98663096bec21ca2b149dfc19a4a85cc1d9161f5e` |
| `artifacts/day11-long-pilot-evaluation/README.md` | Result, exact reproduction, persistence boundary, and next-step handover | 3,196 | `9771cd1c8f2c5a54111ff16979e6f20e7360e859e486da3b73fdea5e15ec35f0` |

`artifacts/day11-long-pilot-evaluation/SHA256SUMS` indexes the four payloads above. It is 354
bytes, hashes to `ae1ef6d2dae3c285b8d65c23f84a366998e2d1649d1ead529e3ce63672ff6ebc`,
and passes in full. Independent reproduction is byte-identical. No new research run, Resume,
seed loop, H200/H400 evaluation, or Test action was performed.

## Sprint-12 H100 replication protocol freeze

`artifacts/day12-h100-replication-protocol/` contains 21 tracked protocol files totaling 93,704
bytes. It contains no run directory or scientific output. `SHA256SUMS` indexes the other 20 files,
is 1,716 bytes, hashes to
`e7fb98455029bff42f7cfec562dc541a1f9c7df1cfecc6156f279b3d7e63b3d1`, and passes in full.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day12-h100-replication-protocol/protocol.json` | Top-level frozen design, pilot exclusion, references, and zero-run release status | 1,520 | `f9fb568c82f37320bd4ef6d396d6aecbe1b88f6b1c075373810e4edc56ec5c79` |
| `artifacts/day12-h100-replication-protocol/seed_manifest.json` | Reproducible SHA-256 derivation and ten predeclared root seeds | 4,330 | `60da60dc3bbd75d886464ef9831e52405d1f4ece69921cfac81ba186b3038ce6` |
| `artifacts/day12-h100-replication-protocol/requirements.json` | Exact base config, source/runtime/manifest fingerprints, gains, loss, mass, and disabled components | 5,742 | `695a1f02c897eb96f54cd3871dd1d6c58b9424f9f2e1cdbb6fbf25446d3c7bb7` |
| `artifacts/day12-h100-replication-protocol/run_inventory.json` | Ten configs/runs in sequential order plus time, RSS, storage, and retry plan | 11,755 | `bd65e206fb3877c242a8195e6cdca123d978777d8c86401bb35af59a4e116773` |
| `artifacts/day12-h100-replication-protocol/analysis_plan.json` | Frozen selection, endpoints, t-interval, secondary analysis, failures, and ten GO gates | 6,483 | `a50111f6217cd011591d703ecb574aa2f4358b57938bd22e058b9d14c5d7fcee` |
| `artifacts/day12-h100-replication-protocol/release_gate.json` | Explicit NO-GO for Sprint-12 runs and confirmation of no Test/H200/H400 | 371 | `946bc226641ffcf90a35874d1a4523fda0f5097c12e38e37cd63067a4311703d` |
| `artifacts/day12-h100-replication-protocol/generate_protocol.py` | Deterministic generator for manifests, plans, inventory, and ten configs | 23,058 | `9bcdd769b5ead1c941e40883bbb6441ea32499f17e78b810737aec8245b821f9` |
| `artifacts/day12-h100-replication-protocol/verify_protocol.py` | Read-only config/source/runtime/selection/metric/hash verifier; never opens Test | 10,556 | `8297479b9ccaec09c3190a47c54c9dad08a584c6e10264398676a16535b32bb3` |
| `artifacts/day12-h100-replication-protocol/sprint13_runbook_template.sh` | Executable dry-run-only one-seed handover template with no execution mode | 1,689 | `b7bac4914fffaacba70878a637395401fa904f386ed56b763ecab7bcd355e49c` |
| `artifacts/day12-h100-replication-protocol/README.md` | Complete human-readable protocol and later handover | 10,370 | `1cefeeeff029725a30af120af8f740712570cc640021ff8d3d9d61727bafed16` |

The index additionally covers all ten `configs/seed-01.json` through `seed-10.json`; their exact
SHA-256 and canonical config fingerprints are in `run_inventory.json`. All configs validate and
differ from Sprint 10 only in `run_id` and `root_seed`. A clean-target generator replay is
byte-identical for all generated JSON. The read-only verifier reproduces every
seed and config fingerprint, source/runtime/gain/Validation requirements, selection rule, emitted
metrics, protocol cross-reference, and file hash. Verification reports `PASS`, Test unopened, and
zero research runs. The untracked Sprint-10 raw run was neither changed nor staged.

## Sprint-12A local-SSD persistence amendment

The preceding Sprint-12 section records the immutable base freeze at commit
`73868cfd84d25bec0c6612b31ee34f4e67acb307`. Sprint 12A changes only persistence and
fault-tolerance gates before Seed 01. The current protocol directory contains 23 tracked files
totaling 125,699 bytes and no run output. `SHA256SUMS` indexes the other 22 files, is 1,950 bytes,
hashes to `8c7702e84621a1df78328c53d95d336d0e2b0adf81b0f7d0304b7631356d2675`, and passes completely.

| Relative path | Sprint-12A purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day12-h100-replication-protocol/amendments/2026-08-01-local-ssd-persistence-v1.json` | Machine-readable dated amendment, old/new rules, invariant hashes, timing, capacity, and accepted risk | 5,084 | `74f61ef167ca03bdde77642306cde8197e57cfe90120e78b423ad3a865a1dd4c` |
| `artifacts/day12-h100-replication-protocol/amendments/2026-08-01-local-ssd-persistence-v1.md` | Human-readable amendment and transparent single-drive limitation | 3,756 | `f6ea13801b226c82c8510770ce83fcec0c1d3b711b98616dc08ebd5ea5bc3578` |
| `artifacts/day12-h100-replication-protocol/protocol.json` | Active amendment reference and zero-run Sprint-12A release boundary | 2,009 | `68c1ee09a6988d7166d2ff9d7930b9e5ade0b870e2a4190b1cc390034c340a16` |
| `artifacts/day12-h100-replication-protocol/requirements.json` | Amended local persistence requirements beside unchanged science/runtime requirements | 7,391 | `18b7a19f1458cdf197fc21a3eb9f1be61f42c21ee2e618fedf4c321e70c22816` |
| `artifacts/day12-h100-replication-protocol/run_inventory.json` | Per-seed local capacity thresholds and previous-index reverification gate | 14,555 | `330a96e194492fff4d597c7e6c4eeafad67ced9dbcbec4940c81728eef755afe` |
| `artifacts/day12-h100-replication-protocol/analysis_plan.json` | Persistence-only GO/failure wording; scientific thresholds unchanged | 6,744 | `e5bfdd14525f40d97814437f828de0dda3e2de57571a23002a1b3dff4b04d488` |
| `artifacts/day12-h100-replication-protocol/release_gate.json` | Sprint-12A zero-run gate and conditional Seed-01 preparation GO | 678 | `52e8a9134e5c3918cb040bb8ca4c31176acff9ff8975dea4bbf7a34b7ace699a` |
| `artifacts/day12-h100-replication-protocol/generate_protocol.py` | Deterministic amended JSON generator with unchanged config generation | 30,559 | `642022dbd4aa660ce24f5875d19a59210fe0703b94474f56a34b4480ec2264dc` |
| `artifacts/day12-h100-replication-protocol/verify_protocol.py` | Verifies base-byte-identical seeds/configs, local rules, unchanged science, hashes, and Test lock | 17,088 | `f276186cef0770e8af8060ee82ddfa92d40cc5f1635ee8105b341436b40a37ab` |
| `artifacts/day12-h100-replication-protocol/sprint13_runbook_template.sh` | Nonexecuting dry-run with real local capacity and previous-index gates | 3,590 | `c6e1cf77ff64e1cb378d581fa1721d7dc545816e2eb4d0f544b38eb45408c655` |
| `artifacts/day12-h100-replication-protocol/README.md` | Current human-readable protocol and local persistence procedure | 11,851 | `b72ac57916350ad2448c1029d20a93bbaa55c1b72d2ffd58b94af82086ec0f5d` |

The seed manifest and all ten config bytes/hashes remain identical to the base commit. A clean
generator replay is byte-identical for every generator-owned JSON. The verifier reports all 22
payload hashes valid, Test unopened, no research run, external storage not required, accepted
single-drive risk, and the 20,000,000,000-byte Seed-01 local-capacity gate passing. The existing
Sprint-10 raw run remains unchanged and outside ordinary Git.

## Ten-seed H100 synthesis and repository closure

The closure uses the existing untracked primary data without modifying or staging them:

| Primary-data path | Role | Files | Bytes | Git policy |
|---|---|---:|---:|---|
| `artifacts/day10-sparse-long-pilot/runs/` | Exploratory long-pilot source retained for chronology | 61 | 437,505,563 | Immutable, untracked, outside ordinary Git |
| `artifacts/day13-h100-replication/` | Ten predeclared H100 confirmatory runs | 610 | 4,375,130,360 | Immutable, untracked, outside ordinary Git |

Each H100 seed has exactly one run directory with 61 files: 50 checkpoints, four fixed runner
files, six outer process/provenance files, and one `RUN_SHA256SUMS`. The ten indexes cover and
verify 600/600 other run files; the deterministic synthesis separately validates 500/500 internal
checkpoint payload hashes and 100,000/100,000 ordered Train/Validation metric rows. No Test metric
or Test-manifest access occurs.

| Versioned closure path | Purpose | Reproduction / integrity |
|---|---|---|
| `artifacts/day19-h100-analysis/analyze_h100_replication.py` | Read-only ten-seed integrity, selection, aggregation, provenance, technical-gate, resource, and Test-boundary verifier | `.venv/bin/python -B artifacts/day19-h100-analysis/analyze_h100_replication.py --check` |
| `artifacts/day19-h100-analysis/h100_replication_analysis.json` | Manifest-ordered per-seed start/selected/final values, frozen statistics, data-derived gates, provenance, and claim boundary | Rebuilt in memory and compared byte-for-byte by `--check` |
| `artifacts/day19-h100-analysis/SOURCE_RUN_INDEXES.sha256` | SHA-256 identity of the ten immutable source `RUN_SHA256SUMS` files | Rebuilt and compared by `--check`; source indexes themselves verify 600 raw files |
| `artifacts/day19-h100-analysis/SHA256SUMS` | Hash index for generator, synthesis JSON, and source-index set | `(cd artifacts/day19-h100-analysis && sha256sum -c SHA256SUMS)` |
| `tests/unit/test_h100_replication_analysis.py` | Regression coverage for frozen t-statistics, earliest-tie selection, hard Test-row rejection, opaque Test reference, and safe index paths | `.venv/bin/python -m pytest -q tests/unit/test_h100_replication_analysis.py` |
| `docs/research/PRIMARY_DATA_ARCHIVE.md` | Dated independent TAR receipt and exact evidence boundary | Documentary receipt; not a new archive action |
| `docs/research/checkpoints/GRADIENT_RESEARCH_CLOSURE_2026-08-02.md` | Formal closure checkpoint and verification record | Commands and results recorded in that checkpoint |

The synthesis input provenance is the frozen protocol in
`artifacts/day12-h100-replication-protocol/`, source HEAD
`ff85c8e9c0e73bcf97dfd5f2a747aa374a834f91`, CPU runtime Python 3.12.13 / JAX-JAXlib 0.10.1 /
NumPy 2.5.1 / Optax 0.2.8, and the ten released seed configs. Selection uses every fixed-
Validation post-update step 1..5,000 with earliest exact-tie retention. Test remains closed.

The independent TAR receipt is `crazyflow-gradient-primary-data_2026-08-02.tar`,
4,813,772,800 bytes, SHA-256
`8766b98f3a8a231f4bd377ecb9b4fc66ae0fa4f26d02fc3c117dbdb595f244ce`, placed in Uni-OneDrive
`Bachelorarbeit/Archive/Gradient-Research/`. It is backup evidence, not ordinary Git evidence and
not a new semantic validation. No post-sync cloud redownload/hash check is claimed.

## Stage-2 H20 loss-diagnostic evidence

These four files were generated from default gains with zero optimizer updates. They are a
reproducible infrastructure smoke, not new convergence, robustness, hardware, or controller-
superiority evidence. `SHA256SUMS` covers the JSON and both PNGs and itself hashes to
`3ca5dab192030c12245945152d775bc1bf2b9a77c49da5cd7ecd68eede4c254a`.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day20-stage2-loss-diagnostics/loss_diagnostics.json` | Full six-term raw/normalization/weight/contribution record, term×gain Jacobians, provenance, plot data, technical metrics, and validations | 17,765 | `9f4821b804b928e5c95cc3fc7aef6cfa6f96e826de8f109374321cc3abee5151` |
| `artifacts/day20-stage2-loss-diagnostics/loss_contributions.png` | Train/Validation weighted contributions rendered only from JSON plot data | 48,750 | `8f54c8a83168d49d3810aeffc0862d027b45f6655a70c62651c8f43d8d18dfd3` |
| `artifacts/day20-stage2-loss-diagnostics/loss_term_gain_gradients.png` | Train/Validation loss-term×bound-variable Jacobian heatmaps rendered only from JSON plot data | 81,909 | `8c9f465ed83801783e0b10f7876306dc6f49263f822f78693e4da4ddc5d949ea` |
| `artifacts/day20-stage2-loss-diagnostics/SHA256SUMS` | Integrity index for the three generated payloads | 272 | `3ca5dab192030c12245945152d775bc1bf2b9a77c49da5cd7ecd68eede4c254a` |

Generation and byte-level reproduction are documented in
[`RUNBOOK.md`](RUNBOOK.md#stage-2-h20-loss-term-diagnostics). The generator refuses overwrite,
and focused tests reproduce the report schema, historical H20 totals and metrics, contribution
sum, term Jacobians, clear technical gates, and byte-identical JSON-driven artifacts.

## Day-21 Friday existing-evidence fallback

The directory contains ten files totaling 5,377,622 bytes. `SHA256SUMS` indexes the other nine
files, is 836 bytes, hashes to
`b64e0523600b5a24c9dcd58ec7fa45b42143f5ccdbf5b7a49cc676f21a4c8a21`, and passes in full.
The two rollout PNGs are intentionally absent because the narrow comparison did not meet the
unchanged zero-motor-saturation gate; no rollout array, metric, or replacement value is included.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day21-friday-evidence/friday_evidence.json` | Complete 10×5,000 Train/Validation plot data, durations, separated reliability/variation, 50 selected gain points per seed, candidate proof, fallback status and claim boundary | 4,596,533 | `6a533f4b7ba892bda31aadf040dce875132e15cff582e7f96c00530ce0ec3a14` |
| `artifacts/day21-friday-evidence/h100_loss_curves.png` | All ten Train and fixed-Validation histories | 155,501 | `921874691455d6710bacba7743caeabe1b5db6f87e8f4bb5fdd19cdb11df8f67` |
| `artifacts/day21-friday-evidence/h100_runtime_and_reliability.png` | Per-process duration and separately labelled technical completion/numerical variation | 92,681 | `e2a50c3d3c72cad307d73f38e07075896a73049a647012bf984b541d6ef79a2a` |
| `artifacts/day21-friday-evidence/h100_selected_gain_evolution.png` | 50 checkpoint-selected physical-gain records for every seed, without gain averaging | 277,655 | `ec0ee54e3e88d310d6c0f44a3b7dd9e122af5051bcd89e095b7ab0d60c62b316` |
| `artifacts/day21-friday-evidence/h100_selected_end_gains.png` | Selected end gains with Seed 05 marked only as provisional visualization candidate | 114,537 | `a55f7bb2a7ace966eab0668481269dcf5b2990bfed622b2e34a514a02bd5d192` |
| `artifacts/day21-friday-evidence/stage2_loss_contributions.png` | Byte-identical accepted Day-20 loss-contribution figure | 48,750 | `8f54c8a83168d49d3810aeffc0862d027b45f6655a70c62651c8f43d8d18dfd3` |
| `artifacts/day21-friday-evidence/stage2_loss_term_gain_gradients.png` | Byte-identical accepted Day-20 local-Jacobian figure | 81,909 | `8c9f465ed83801783e0b10f7876306dc6f49263f822f78693e4da4ddc5d949ea` |
| `artifacts/day21-friday-evidence/provenance.json` | Base/generator commits, environment, inputs/hash classes, selected checkpoint, exact generation/test/lint/two-target reproduction commands, fallback and output inventory | 5,538 | `805b7042f1223345ff581ddd410e4fab2033955848b8564df3d204a87c651ca8` |
| `artifacts/day21-friday-evidence/PRESENTATION_HANDOFF.md` | Figure-by-figure claims, nonclaims, slide titles and open missing-rollout statement | 3,682 | `9c98a835855e5d5c7b351be08d33c47ce3e3102cb52df0012275a64dc6eec1b7` |
| `artifacts/day21-friday-evidence/SHA256SUMS` | Integrity index for the nine fallback payloads | 836 | `b64e0523600b5a24c9dcd58ec7fa45b42143f5ccdbf5b7a49cc676f21a4c8a21` |

Two fresh generations from generator/test commit
`2499aad364a44eadcc259005f4c18d8a21acb332` were byte-identical to each other and the final
payload. Current verification covers the tracked Day-19/Day-20 checksums and the ten small source
run-index identities. The accepted Day-19 full-payload integrity is inherited; this tranche did
not rehash the 4.38-GB H100 primary data.

## Day-22 saturation-explicit diagnostic review candidate

The directory contains six files totaling 782,991 bytes. `SHA256SUMS` indexes the other five
files, is 457 bytes, hashes to
`8fa0b309cea5d4903675527625e8a71cb92b831ba6d6155ab37b967adaee3c68`, and passes in full.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day22-saturation-diagnostic/saturation_diagnostic.json` | Complete frozen reference/actual/error arrays, 100×4 Default/Seed-05 RPM commands, thresholds, lower/upper/any decisions, exact per-motor indices/times/intervals, tracking metrics, gate status and claim boundary | 184,928 | `b6c94fdb2535f165912fdf188e49cdfdb8105c25e962dcc6d35bcab249afcfc4` |
| `artifacts/day22-saturation-diagnostic/trajectory_error_saturation.png` | JSON-driven trajectory/error figure with every Seed-05 saturation sample and interval marked and diagnostic-only label | 315,079 | `e71e756fa84aa904dd2188d9fe20d855ec1589935ff0178288605199803fba8f` |
| `artifacts/day22-saturation-diagnostic/motor_saturation_detail.png` | JSON-driven four-motor command/threshold figure with exact upper-bound interval on motor 0 | 273,612 | `b9be6317b5fa8441d87c27924d5c673ff955f48cae32003ccf523d290b80bb4c` |
| `artifacts/day22-saturation-diagnostic/provenance.json` | Base/generator identities, five accepted Day-21 input hashes, exact executed F0.1 regression method/nonexecuted protected-data CLI boundary, frozen configuration, commands, environment and inventory | 6,720 | `2b7298c643903291bd7e9a72e8acdbf87c52dcfd73c72ccc2d35245d1babed90` |
| `artifacts/day22-saturation-diagnostic/PRESENTATION_HANDOFF.md` | Figure claims/nonclaims, exact slide takeaways, saturation facts and required verbal caveat | 2,195 | `c5e695893922e492273ed434c6c72ad0aa77d387451eed88a4912875a411ea98` |
| `artifacts/day22-saturation-diagnostic/SHA256SUMS` | Integrity index for the five diagnostic payloads | 457 | `8fa0b309cea5d4903675527625e8a71cb92b831ba6d6155ab37b967adaee3c68` |

The generator/test commit is `2d90302f883de2ab2b03c60706314d1097f75d25`. Two fresh
generations are byte-identical to each other and this final worker-result payload. The exact gate
result remains Default `0/400`, Seed 05 `11/400` (`0.027499999850988388`), all eleven on motor 0
at the upper bound. Status is `DIAGNOSTIC_ONLY_SATURATION_PRESENT`, not acceptance or flight
readiness. Five accepted Day-21 payload identities were verified; Day 10, Day 13 and Test were
not read. Neither full F0.1 CLI was executed because doing so would reread protected Day 13; the
provenance records the actual accepted-Day21-to-real-trace-builder gate regression instead.

## Day-23 vertical ki_z sensitivity review candidate

The directory contains five files totaling 4,935,078 bytes. `SHA256SUMS` indexes the other four
files, is 347 bytes, hashes to
`2898301f8668aed1cfe52ca79787aaa59957235bea6b922f5908ab7e61a40738`, and passes in full.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day23-ki-z-sensitivity/ki_z_sensitivity.json` | Complete frozen contracts, six 500-sample traces, height/error/velocity/integral/RPM/raw-and-normalized margins, Loss-v1 terms, local AD/FD gradients, predeclared gates and recommendation | 4,703,110 | `10abd53a9ddd71ff51af10e3885c1cbf9cb9dd0f52feb3c0d217b5a13853298a` |
| `artifacts/day23-ki-z-sensitivity/ki_z_sensitivity.png` | Presentation-quality JSON-driven altitude, error, integral-excitation and local-loss-sensitivity diagnostic | 226,180 | `5edde35c45e4d0aef7917e2850a1e6e8079f39de3759b9b8afcafa064e8eac4c` |
| `artifacts/day23-ki-z-sensitivity/provenance.json` | Base/generator identities, input-source hashes, environment, exact commands, inventory, zero optimizer/protected-data record and deterministic-reproduction rule | 3,730 | `239e4bea5e03637d7641fcc8b0caae70b3a2adf888f7eb4fcb68ac678a0c4d9b` |
| `artifacts/day23-ki-z-sensitivity/PRESENTATION_HANDOFF.md` | Exact evidence, claim boundary, backup-slide takeaway and verbal caveat | 1,711 | `2eb73a74cecd425c984b4cf7cbff9cdc262ba26dde0a2a837baac219fb20b3f4` |
| `artifacts/day23-ki-z-sensitivity/SHA256SUMS` | Integrity index for the four generated payloads | 347 | `2898301f8668aed1cfe52ca79787aaa59957235bea6b922f5908ab7e61a40738` |

The generator/test commit is `beb59393280b6b515c328edce13c430ebfbaa826`. Two fresh
generations are byte-identical to each other and the final worker-result package. Every technical
gate and every predeclared scientific identifiability threshold passes, giving
`GO_FOR_FUTURE_KI_Z_OPTIMIZATION`. This is only a future-Work-Order recommendation, not an
optimized result or transfer claim. Day 10, Day 13, Test and accepted Day-20/21/22 packages were
not read or changed.

## Day-24 unclipped two-stage allocation diagnostic review candidate

The directory contains five files totaling 1,943,983 bytes. `SHA256SUMS` indexes the other four
files, is 377 bytes, hashes to
`f75e4d17bfcef14a56235eab91ac395904e340ead24936338007ce88dbc2f51a`, and passes in full.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day24-unclipped-allocation/unclipped_allocation_diagnostic.json` | Complete 100×4 Stage-A/Stage-B raw/postclip/mask/exceedance arrays, wrench reconstruction/distortion, exact production/Day-22 identities, summaries, Float32 contract, hover point, DATA-AUDIT-01 and failed-gate boundary | 1,573,468 | `5a7900398c177ef15b533e8dd40cb36ab709b4450671db2af8c574a0b064ef52` |
| `artifacts/day24-unclipped-allocation/unclipped_allocation_diagnostic.png` | Sole 2240×1600 JSON-driven figure with motor-0 unclipped/clipped demand and true limit, all four Stage-A pre/post motor commands in the fixed 44–62 window, signed platform-normalized wrench distortion, and compact exact maximum/count/status/nonclaim text | 360,461 | `21ae033bece6b507d39d42cadccc18b40948481d09bb683cad5759461d571990` |
| `artifacts/day24-unclipped-allocation/provenance.json` | Base/generator identity, accepted Day-22 and permitted Day-19-derived inputs, frozen sampling/configuration, exact commands, environment, inventory and protected-data boundary | 7,207 | `297636564b50b1ae6aed7bc61c1a5373113f45baec89113aa695954bf3307597` |
| `artifacts/day24-unclipped-allocation/PRESENTATION_HANDOFF.md` | Exact two-stage facts, corrected four-panel figure use, DATA-AUDIT boundary, German verbal caveat and nonclaims | 2,470 | `ae63ef7a22eb707c008b0f14821c050a4acfabbe6b20940d58dc03da54e5eafb` |
| `artifacts/day24-unclipped-allocation/SHA256SUMS` | Integrity index for the four diagnostic payloads | 377 | `f75e4d17bfcef14a56235eab91ac395904e340ead24936338007ce88dbc2f51a` |

The generator/test commit is `728e8b9f89e4f92c7fa4eefee9808a82904975c2`. Two fresh
generations pass 4/4 checksums and are byte-identical to each other and this final worker-result
package. The corrected figure cycle left `unclipped_allocation_diagnostic.json` byte-identical at
`5a7900398c177ef15b533e8dd40cb36ab709b4450671db2af8c574a0b064ef52`; provenance changed only
with the amended generator identity, while PNG, presentation handoff and checksum index reflect
the corrected figure contract. The production replay, accepted Day-22 arrays/metrics/
classification, and local Stage-A/
Stage-B production identities pass. Stage-A motor clipping is Default `0`, Seed 05 `11`; torque-
axis clipping and additional Stage-B clipping are both zero. The unchanged status remains
`DIAGNOSTIC_ONLY_SATURATION_PRESENT` / `FAIL_ZERO_MOTOR_SATURATION_GATE`. Day 10, Day 13, Test,
training and optimization were not accessed or started; no controller, candidate, hardware,
safety or flight claim is made.

## Day-25 cf21B_500 robust baseline foundation corrected review candidate

The directory contains ten files totaling 294,798,314 bytes. `SHA256SUMS` indexes the other nine
files, is 805 bytes, hashes to
`819d2ec725b5b8b045a6b6acfaae6443a6151df01d59d9d7c5a178d787b70437`, and passes in full.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day25-cf21b-robust-foundation/robust_evaluation_report.json` | Complete 32-rollout paired mass-ablation evidence: state-continuous trajectories, tracking/Loss-v1/integral/reserve/clip/wrench arrays, all six Loss-v1 aggregate fields, trajectory-statistic aggregates, production identity, corrected scope, D-047 invariance evidence, mechanism layers, technical gates and explicit negative finding | 285,845,306 | `1b5dc30d6bbcd6274947fba5ab7237d7188183d04b4deac240607aab2c8b1078` |
| `artifacts/day25-cf21b-robust-foundation/episode_contract.json` | Complete deterministic 16-parent episode contract, analytic references, retained fixed positive-z component and derivative construction, 8-s support/6-s rollout/2-s scored-window scope, classes, splits, seeds, indices and input identities | 8,564,229 | `24dd47d14b5c871228e90cb46d1583d5e61ff4425c38bd1beffb4083e6461535` |
| `artifacts/day25-cf21b-robust-foundation/train_manifest.json` | Eight Train episode records and identities without Test access | 15,596 | `f5e2f533bd65fb1adc50072e61a1445dc072d12c8b099720d6ad3642b76daddd` |
| `artifacts/day25-cf21b-robust-foundation/validation_manifest.json` | Eight Validation episode records and identities without Test access | 15,741 | `8c0077f2d4c9dbb797f8adb7bd028433d6cf75763aa415ef6e025870327f40b5` |
| `artifacts/day25-cf21b-robust-foundation/benchmark_measurements.json` | Checksum-pinned fresh-process 4/16/32 CPU benchmark with compile/first/steady timing, throughput, memory/swap/timeout guards and objective/gradient consistency | 17,365 | `9ca5d2e07da2e41b1144117c744cd36a6a908e078bd4f7e21c06d44323b2e8d0` |
| `artifacts/day25-cf21b-robust-foundation/robust_foundation_overview.png` | JSON-driven paired scored-window Z-RMSE/contact/clip overview with explicit negative-result and simulation-only claim boundary | 189,619 | `5c1504b9683ddf28e9337ffaedf49c95603d7c92e22a4484e6e0a305cab517de` |
| `artifacts/day25-cf21b-robust-foundation/batch_scaling.png` | JSON-driven 4/16/32 CPU compile/first/steady/throughput/RSS figure with rule-based next-batch recommendation and no GPU/default claim | 138,643 | `5cedd25b5e69dd8b18516c2a3faf57b3b4d8e6a6c61fb73cbb9aa26388c6ea09` |
| `artifacts/day25-cf21b-robust-foundation/TECHNICAL_HANDOFF.md` | Compact D-047 disclosure, corrected duration/measurement scope, old-c4 invariance PASS, negative result, mechanism separation, benchmark recommendation and nonclaims | 3,138 | `97ce483f4a331ad5dd84805876a11acde665dc0f4a817dee619646b936fed27e` |
| `artifacts/day25-cf21b-robust-foundation/provenance.json` | Base/stable generator identity, D-047 old-c4 comparator evidence, retained vertical-excitation and duration machine contracts, environment, input hashes, exact commands, benchmark pin, package inventory and protected/transfer boundary | 7,872 | `239c6fcea6a941b0a0a48a9e63dc5d5af06f4a54caab06213b4943235d3ebd39` |
| `artifacts/day25-cf21b-robust-foundation/SHA256SUMS` | Integrity index for the nine generated payloads | 805 | `819d2ec725b5b8b045a6b6acfaae6443a6151df01d59d9d7c5a178d787b70437` |

The first independent review of result commit
`c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` was `NOT ACCEPTED`. D-047 permits only this
transparency correction (`CORR-01`), not a numeric rerun or acceptance. The rewritten stable
generator/CLI/test commit is `2288ab2fe3b55f585fb9b14a8fe9d87c07eb6c1a`. The unchanged
raw benchmark was supplied to two fresh `/tmp` packages and the final package. All three are
byte-identical and every checksum set passes. Both PNGs were inspected at original resolution
and make the negative physical-value-match finding, scored-window scope and CPU-only/simulation-
only boundaries legible.

The automatic comparator always loads the old side from
`c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` and records
`PASS_PRE_CORRECTION_C4_INVARIANCE`. Old and corrected numeric report projections are identical:
5,465,162 paths, 644,397,249 serialized bytes, SHA-256
`fd1195f9d10c88a2d550090631f6b21d0d4f80fd47a2bbe09a52e7c9a0975a61`. The parent projection
is also identical: 16 records, 18,037 bytes, SHA-256
`bcb40e132f4b85e99f814f0a9db4d90244523b488697868ece8597d3654de82c`. The comparator also
pins every retained old scalar outside the declared prose/provenance replacements, raw array,
summary, pair delta, ID/seed/attempt/window/mass/digest and the benchmark bytes. Only permitted
aggregate/metadata/prose/provenance/checksum additions differ.

References have 8-s support; simulation is a continuous 6-s rollout from the parent start; and
the 2-s half-open scored window follows the fixed 2/3/4-s warm-up without reset. Full-rollout
gates are finiteness, ground/floor and zero thrust. Integral, torque/motor/Stage-B clipping,
tracking, Loss v1, reserve and wrench are scored-window-only. Every class retains the same exact
Float32 positive-z component: `0.06 m/s`, `0.06 * (time - 2 s)` base displacement, zero base
acceleration/jerk and product derivatives through jerk under the C3 smoothstep7 2-s entry /
4-s plateau / 2-s exit envelope.

Repository mismatch records mean Z-RMSE `0.08494756871345588 m` and 593 negative-z
integral-boundary samples; physical-value match records `0.11452380346939901 m` and 1,844. Thus
matching worsens mean Z-RMSE by `0.029576234755943134 m` and adds 1,251 contacts in this fixed
conversion chain. All nonintegral gates pass, with zero scored-window Stage-A torque, Stage-A
motor and Stage-B additional clips across all 32 rollouts. D-045 makes integral contact
descriptive, not a pass.
This corrected package is a worker review candidate pending independent Gradient-Lead acceptance.
It proves
neither improvement, causal isolation, optimization, candidate acceptance, physical-mass
estimation, firmware/hardware transfer, safety, flight readiness nor Sim2Real performance. Day
10, Day 13 and Test were not opened or changed.

## Day-26 G3.4 parameter eligibility package

The accepted directory `artifacts/day26-g3-identifiability-freeze-v2/` contains exactly 13
regular mode-`600` files totaling `31,722,155` bytes. `SHA256SUMS` indexes the other 12 files and
passes `12/12`. The canonical complete 13-file SHA-256 list has SHA-256
`57934a75f9fcbfec5480184405170fd87b6ff82b8c0cf942e1b55af82f87dbfe`.

| Relative path | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `artifacts/day26-g3-identifiability-freeze-v2/SHA256SUMS` | Integrity index for all 12 package payloads | 1,081 | `325c6a9d06e74b23f12ba647ccbc2ab2c70b101ebafdef80d4b55cc6f0a86e07` |
| `artifacts/day26-g3-identifiability-freeze-v2/TECHNICAL_HANDOFF.md` | Compact status, parameter dispositions, Loss decision, reproduction and claim boundary | 3,443 | `d4a8459ca02dfc12a95ff99c5d1ae165c2eb651fe812d835c9f11bf5dea0bf72` |
| `artifacts/day26-g3-identifiability-freeze-v2/g3_freeze_overview.png` | JSON-driven overview of parameter categories and package result | 192,034 | `8e2b308ba1418b1518e955533c60a7ad774951d71b52c7796fb85845dff5120a` |
| `artifacts/day26-g3-identifiability-freeze-v2/g3_freeze_report.json` | Complete 24-parent, seven-parameter G3.4 report and finite metadata | 18,194,379 | `53a881b268a77511aaa87f5f457c154359e9949597be2bba4934d5f4d86efebe` |
| `artifacts/day26-g3-identifiability-freeze-v2/g4_g5_freeze_proposal.json` | Hauptleitung-only joint proposal, parameter exclusions and unchanged prospective gates | 13,128 | `165a443664a8f0a123b1ce99680fb332eb34664d1b98c9c7f4ecfdd4a0381a38` |
| `artifacts/day26-g3-identifiability-freeze-v2/loss_diagnostics.json` | Loss-v1 objective-output and diagnostic-output AD/FD disclosures | 1,450,546 | `2fe46c85cce24762c4e0395a241830e277e50735fa86464e0890b3514c09b780` |
| `artifacts/day26-g3-identifiability-freeze-v2/parameter_semantics.json` | Exact seven-parameter log-coordinate and role contract | 14,997 | `743fba3250e919913fae775a139e049ca6f1112cca6f26d516be9d25736a2a84` |
| `artifacts/day26-g3-identifiability-freeze-v2/parameter_sensitivity_matrix.png` | JSON-driven sensitivity and conditioning figure | 149,055 | `79dffb46e445c06584d459bfb9e5e885aefa74efd02089f1ab622df7d7aa5a8d` |
| `artifacts/day26-g3-identifiability-freeze-v2/provenance.json` | Branch-independent generator, environment, execution mode, inventory and protection provenance | 6,436 | `48707cc9810bd839c19c40feeb30ebd72a5087dc4032229bf1f9c27a8d7fb026` |
| `artifacts/day26-g3-identifiability-freeze-v2/sensitivity_conditioning.json` | Complete Train/Validation sensitivity and joint-conditioning evidence | 10,857,525 | `f1f907faf440cd361c5e13fdd082a5fc91b6261f5fcd9356d7f47e921c32a582` |
| `artifacts/day26-g3-identifiability-freeze-v2/train_manifest.json` | Twelve ordered Train parents and identities | 9,925 | `6d6a8e1a6ecc6221fc7fae65a1ac216a050c73419a0799e18687419e973b7e4e` |
| `artifacts/day26-g3-identifiability-freeze-v2/trajectory_contract.json` | Frozen parent, rollout, scored-window, batch and technical contracts | 819,554 | `d47912f63e23299e2c3ef13b5789f13e03718ded069d2e9b0955f72e4836b583` |
| `artifacts/day26-g3-identifiability-freeze-v2/validation_manifest.json` | Twelve ordered Validation parents and identities | 10,052 | `58159451316e0d0cd76724b2add675ccd55b2d70e3fcc733140a7f148f1fb118` |

The package status is `COMPLETE_G3_4_PARAMETER_ELIGIBILITY_AUDIT`. Individual GO parameters are
`kp_xy`, `kp_z`, and `ki_z`; `kd_xy` is `DIAGNOSTIC_ONLY`; and `kd_z`, `mass`, and
`mass_thrust` are withheld for `WITHHELD_TECHNICAL_ROBUSTNESS`. The joint `kp_xy + kp_z`
proposal is for Hauptleitung review only. Loss remains
`RETAIN_LOSS_V1_PLUS_EXTERNAL_GATES`; G4 remains `WITHHELD_PENDING_REBENCHMARK` and was not
executed; optimizer counts are `0/0`.

The stable generator is exactly
`5c28db1aae6bee67f06a2764f535f5c38924495e` with subject
`research: support g3.4 recovery provenance`. The separate compatibility commit
`5994f49a342b14e3b0fa47117f26488a0c11a0b4` is not the generator. Package A, Package B and
Final are byte-identical. The exact CPU reproduction and the narrow external telemetry exception
are documented in
[`RUNBOOK.md`](RUNBOOK.md#day-26-g34-freeze-package-verification-and-reproduction).

This is pure JAX Float32 CPU simulation evidence. It proves no optimized candidate, improvement,
Test result, physical calibration, integer or firmware differentiability, hardware transfer,
safety, flight readiness, or Sim2Real performance.
