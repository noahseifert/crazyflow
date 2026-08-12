# Gradient Research closure checkpoint: ten-seed H100 synthesis

## Boundary

- Closure date: 2026-08-02
- Repository: `/home/noah3/bachelorarbeit/crazyflow-gradient-research`
- Branch: `research/differentiable-mellinger`
- Input HEAD: `ff85c8e9c0e73bcf97dfd5f2a747aa374a834f91`
- Initial tracked worktree and index: clean
- Protected untracked primary data: Day-10 `runs/` and Day-13 H100 replication tree
- Pre-existing untracked understanding handoff: preserved, outside this closure
- At this historical pre-commit checkpoint: no research run, new seed, H200/H400, Test access,
  hardware action, dependency change, commit, amend, push, merge, rebase, tag, reset, clean, or
  raw-data staging

This checkpoint was prepared to close the existing Gradient simulation evidence for pre-commit
review. It did not open a new research phase or authorize a later horizon or Test.

## Post-checkpoint chronology

The boundary above records the state when this checkpoint was first written on 2 August 2026.
Pre-commit approval was subsequently granted, and the twelve allowlisted closure paths were
committed on 3 August 2026 as `d386148edad9242720177180183940e74c081fde` (`research: close
ten-seed H100 gradient study`).

The first independent read-only closure audit after that commit passed all Git, primary-data,
deterministic-reproduction, test, protection, and scientific-claim checks. Its formal result was
nevertheless `GRADIENT-SCHLUSSAUDIT: FAIL`, exclusively because `PROJECT_STATE.md`, this
checkpoint, `ARTIFACT_MANIFEST.md`, and `DECISIONS.md` retained mutually inconsistent closure
status statements. This documentation-only correction records that audit result; a repeated
read-only closure audit has not yet been performed or passed.

## Source classification

The current Git/code/artifact state is authoritative for current technical facts. The canonical
dossier dated 2 August 2026 supplied the independent archive receipt. The normalized 286-line
repository-audit attachment hashes to
`6cec6d7ebe0893ebc0b6a55b6a578a41a0c44a1f007320fe997d6bc84e2d09d0`, exactly matching the
dossier register; its raw attachment hash differs only because the delivered copy uses CRLF line
endings. The historical Gradient lead handoff supplies chronology and the Sprint-17 wrapper/
authorization account, but does not override current raw evidence.

The sources agree on the numerical H100 reference result. Their material boundary is that the
historical source reports a Sprint-17 procedural deviation while the repository itself contains
no authorization ID, deviation record, waiver, or later horizon authorization. This checkpoint
preserves that distinction rather than converting historical reporting into repository proof.

## Allowed closure paths

Only these paths belong to the closure change:

- `artifacts/day19-h100-analysis/analyze_h100_replication.py`
- `artifacts/day19-h100-analysis/h100_replication_analysis.json`
- `artifacts/day19-h100-analysis/SOURCE_RUN_INDEXES.sha256`
- `artifacts/day19-h100-analysis/SHA256SUMS`
- `tests/unit/test_h100_replication_analysis.py`
- `docs/research/PRIMARY_DATA_ARCHIVE.md`
- `docs/research/PROJECT_STATE.md`
- `docs/research/RUNBOOK.md`
- `docs/research/ARTIFACT_MANIFEST.md`
- `docs/research/RESULTS_SUMMARY.md`
- `docs/research/DECISIONS.md`
- `docs/research/checkpoints/GRADIENT_RESEARCH_CLOSURE_2026-08-02.md`

No primary-data, controller, dynamics, runner, config, manifest, dependency, lockfile, environment,
historical checkpoint, image, or understanding path is changed.

## Deterministic synthesis contract

The synthesis consumes only:

1. the committed frozen protocol under `artifacts/day12-h100-replication-protocol/`;
2. exactly one existing run directory for each manifest seed 01 through 10;
3. each run's local `RUN_SHA256SUMS`, metrics, summary, resolved config, provenance, checkpoints,
   launcher status, launch manifest, and external timing record.

It verifies, before aggregation:

- exact 10/10 manifest seeds and one run per seed;
- 61 files per run and exact outer/runner file sets;
- 60/60 indexed files per run, 600/600 total;
- 50 internal checkpoint-payload hashes per run, 500/500 total;
- 5,000 Train and 5,000 Validation metric rows per run in exact interleaved order;
- finite values and passing technical gates;
- successful 5,000-update summaries and external/launcher exit status zero;
- released/resolved config equality and frozen config SHA/fingerprint identity;
- branch, source HEAD, clean scientific source scope, source/runtime/gain/Validation fingerprints;
- no Resume lineage, no retry directory, and no replacement seed;
- closed Test flags, zero Test rows, and an opaque Test reference that is never opened;
- selection recomputed from every Validation step rather than copied from the summary.

Selection is the global fixed-Validation minimum over post-update steps 1..5,000. NumPy's first
minimum index and the runner's strict `<` replacement both retain the earliest exact tie. Step 0
and Test are excluded. Aggregation follows the frozen plan: arithmetic mean, `ddof=1` sample SD,
two-sided 95% Student-t interval with `df=9` and `t=2.2621571627409915`, median, and linear-method
Q1/Q3.

## Per-seed result

| Index | Root seed | Step-1 Validation | Selected step | Selected/final Validation | Relative improvement |
|---:|---:|---:|---:|---:|---:|
| 01 | 1432116264 | 0.010010135360062122 | 5000 | 0.002524877665564418 | 0.7477678797793251 |
| 02 | 366692846 | 0.010010110214352608 | 5000 | 0.002525355899706483 | 0.7477194710518172 |
| 03 | 235438753 | 0.010010110214352608 | 5000 | 0.0025249135214835405 | 0.7477636641938976 |
| 04 | 1369406745 | 0.010010110214352608 | 5000 | 0.0025246848817914724 | 0.7477865050704886 |
| 05 | 1081462774 | 0.010010127909481525 | 5000 | 0.002523899544030428 | 0.7478654052322541 |
| 06 | 1276309202 | 0.010010110214352608 | 5000 | 0.002524987794458866 | 0.7477562443979378 |
| 07 | 987511476 | 0.0100101288408041 | 5000 | 0.0025246264412999153 | 0.7477928125151768 |
| 08 | 2125653507 | 0.010010136291384697 | 5000 | 0.0025241717230528593 | 0.7478384260137090 |
| 09 | 31735934 | 0.010010110214352608 | 5000 | 0.0025242639239877462 | 0.7478285583341102 |
| 10 | 986925065 | 0.010010110214352608 | 5000 | 0.002524841111153364 | 0.7477708979134696 |

All ten curves decrease strictly over all 4,999 transitions, so selected and final Validation
loss coincide at step 5,000 for every seed.

## Aggregate result

Selected Validation loss:

- mean: `0.0025246622506529095`;
- sample SD: `4.360755287449738e-07`;
- 95% Student-t interval: `[0.002524350301011905; 0.002524974200293914]`;
- median: `0.0025247629964724183`;
- linear Q1/Q3: `0.0025243545533157885 / 0.00252490455750376`.

Relative Validation improvement:

- mean: `0.7477889864502185` = `74.7788986450 %`;
- sample SD: `4.370248745099389e-05` = `0.0043702487451` percentage points;
- 95% Student-t interval:
  `[0.7477577235740132; 0.7478202493264239]` =
  `[74.7757723574 %; 74.7820249326 %]`;
- median: `0.7477787014919791`;
- linear Q1/Q3: `0.7477647180902545 / 0.7478196218793769`.

Supporting descriptive results:

- mean final Train loss: `0.00247700703330338`;
- mean final gradient L2 norm: `0.00027522838208824394`;
- minimum selected-gain normalized bound distance: `0.08598490194840865`;
- panel maximum position error: `0.03437680006027222 m`;
- panel maxima for motor saturation, floor clip, zero-thrust gate, and nonfinite state: `0.0`;
- external walltime total: `9,286.5 s`;
- Peak-RSS range: `2,188,252..2,211,196 KiB`;
- swaps: zero for every run.

## Horizon and Sprint-17 classification

All data-derived frozen H100 gates pass. That does not settle the independent requirement that no
unplanned execution-protocol change occurred.

Repository-verified:

- Seed 09 and 10 are complete, hash-valid, finite, technically passing H100 runs;
- their launch manifests identify the Sprint-17 execution context;
- the repository provides no `authorization_id`, deviation record, waiver, or explicit later
  H200/H400 authorization;
- the frozen protocol requires explicit later authorization and treats any missed GO condition as
  horizon NO-GO.

Historically reported, not repository-proved:

- two just-in-time Sprint-17 wrapper/preflight invocations exited nonzero;
- execution continued after the first event despite the literal stop rule;
- later analysis retained all ten runs but used the deviation as the formal horizon NO-GO reason.

Closure classification: include Seed 09 and 10 in the predeclared scientific panel and preserve
their raw evidence, while keeping H200/H400 **NO-GO / not authorized**. No new scientific horizon
decision is made. Any future change needs an explicit, separately documented decision; numerical
H100 success is not a waiver.

## Independent primary-data archive receipt

On 2 August 2026 the unchanged Day-10 and H100 directories were archived as
`crazyflow-gradient-primary-data_2026-08-02.tar`:

- size: `4,813,772,800` bytes;
- SHA-256: `8766b98f3a8a231f4bd377ecb9b4fc66ae0fa4f26d02fc3c117dbdb595f244ce`;
- local staging: `/home/noah3/backup-staging/crazyflow-gradient-primary-data_2026-08-02.tar`;
- independent target: Uni-OneDrive, `Bachelorarbeit/Archive/Gradient-Research/`;
- local target before sync: exact size and `sha256sum -c` `OK`;
- cloud evidence: client “Aktuell”, TAR plus `.sha256` online, no `.uploading` file, web size
  4,700,950 KB.

No cloud redownload followed by SHA-256 was performed. The receipt proves the documented local
source/target byte identity and subsequent cloud presence, not renewed semantic validation, Git
versioning, or an independently rehashed downloaded cloud object. Full wording is in
[`PRIMARY_DATA_ARCHIVE.md`](../PRIMARY_DATA_ARCHIVE.md).

## Verification performed

Protocol and primary data:

```text
(cd artifacts/day12-h100-replication-protocol && sha256sum -c SHA256SUMS)
  PASS, 22/22
.venv/bin/python -B artifacts/day12-h100-replication-protocol/verify_protocol.py
  PASS; seed/config 10/10; Test unopened; research runs started 0
sha256sum --quiet -c artifacts/day11-long-pilot-evaluation/SPRINT10_RUN_SHA256SUMS
  PASS, 61/61
ten per-run sha256sum --quiet -c RUN_SHA256SUMS checks
  PASS, 600/600
```

Synthesis and repeated determinism:

```text
.venv/bin/python -B artifacts/day19-h100-analysis/analyze_h100_replication.py --write
  WRITE_PASS; initial deterministic creation
.venv/bin/python -B artifacts/day19-h100-analysis/analyze_h100_replication.py --check
  CHECK_PASS
.venv/bin/python -B artifacts/day19-h100-analysis/analyze_h100_replication.py --check
  CHECK_PASS with identical compact values
(cd artifacts/day19-h100-analysis && sha256sum -c SHA256SUMS)
  PASS, 3/3
```

Focused and higher-level non-research tests:

```text
.venv/bin/python -m pytest -q \
  tests/unit/test_h100_replication_analysis.py \
  tests/unit/test_mellinger_checkpointing.py \
  tests/unit/test_mellinger_research.py::test_configs_and_validation_manifest_validate_while_test_stays_opaque \
  tests/unit/test_mellinger_research.py::test_checkpoint_interval_validation_and_legacy_fingerprint \
  tests/unit/test_mellinger_research.py::test_seed_tree_is_deterministic_and_split_separated
  11 passed
.venv/bin/python -m ruff check \
  artifacts/day19-h100-analysis/analyze_h100_replication.py \
  tests/unit/test_h100_replication_analysis.py
  PASS
.venv/bin/python -m ruff format --check \
  artifacts/day19-h100-analysis/analyze_h100_replication.py \
  tests/unit/test_h100_replication_analysis.py
  PASS
```

No integration test that executes a new simulation, optimization runner, later horizon, or Test
path was run; those actions are outside the authorized closure scope.

## Claim boundary and final gate

Supported: the ten existing H100 seeds provide positive simulation evidence under the frozen
H100 configuration and documented Validation selection.

Unsupported: Test performance, H200/H400 performance, general superiority outside the frozen
configuration, convergence/global optimality, firmware or STM32 equivalence, physical fidelity,
hardware/HIL/flight validation, Sim2Real evidence, or complete external reproducibility without
the documented runtime and untracked primary data.

At the time this checkpoint was first written, the closure remained uncommitted pending review;
no commit or push was part of that historical pre-commit state. Pre-commit approval and closure
commit `d386148edad9242720177180183940e74c081fde` followed as recorded in the chronology above.
No primary-data path belongs in the index or ordinary Git.

**Historical marker at checkpoint creation:** `PRE-COMMIT-ABNAHME ERFORDERLICH`
