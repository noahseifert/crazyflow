# Day-6 checkpoint: deterministic Resume proof and Workstation pilot gate

Date: 2026-07-31
Branch: `research/differentiable-mellinger`
Initial HEAD: `aca84429bf8924bd41e14ffd04a79029a5f2e922`
Final executable-source HEAD: `182d0cc9068c4920ac25146ff0016260f802e5dd`
Final documentation/artifact commit: the commit containing this checkpoint

## Scope and final decision

Sprint 6 accepts the Sprint-5 stack, implements only the missing safe Resume/monitoring functions,
proves exact end-to-end continuation in fresh processes, and attempts the explicitly bounded local
scaling sequence. It runs no main optimization, seed study, ablation, H400 local job, or Test
evaluation.

**Final decision: NO-GO for incremental Workstation pilots and the later main run.** Exact Resume
readiness passes. The required first H100 × one-world pilot fails before JIT because the unchanged
Workstation trajectory distribution cannot construct a valid H100 episode in 16 deterministic
attempts. The gate therefore forbids H100 × 4 and H200 × 4 locally and all Workstation stages.

This is not a hardware, firmware, safety, realism, Sim2Real, convergence, generalization, or
scientific-performance result.

## Initial state and audit

Before reading repository instructions or changing files, the required commands confirmed:

```text
branch: research/differentiable-mellinger
HEAD: aca84429bf8924bd41e14ffd04a79029a5f2e922
status: clean
```

The Day-5 checkpoint, `PROJECT_STATE.md`, `RUNBOOK.md`, `DECISIONS.md`,
`ARTIFACT_MANIFEST.md`, `USER_UNDERSTANDING_AND_OPEN_QUESTIONS.md`, Workstation config, and both
fixed manifests were read. The main agent then audited only the Sprint-5 Runner, checkpoint,
benchmark, entry point, focused tests, and relevant artifacts.

Two requested read-only subaudits were used and rechecked by the main agent:

- Compute/Resume audit: found no real runner-level equivalence test, raw dtype loss, partial resume
  metrics, weak source/runtime/Validation compatibility, non-atomic JSONL, missing lineage and
  no strict empty-target rule.
- Scientific-separation audit: confirmed structural Train/Validation/Test separation, shared gains,
  mass-only Workstation semantics and controller-mass preservation; found that the runner still
  opened the Test manifest for ID provenance and that the default benchmark used heuristic smoke
  delay/wrench unless an explicit Workstation config was supplied.

Neither subagent changed files, ran tests, or started compute. The main agent owned every edit,
test, process, artifact, decision, and commit.

## Immutable scientific boundary

- Stage 1 only: `kp_xy`, `kp_z`, `kd_xy`, `kd_z`;
- one shared raw gain vector for every world;
- true dynamics mass uniform ±`0.0002 kg`; controller mass unchanged;
- Delay, Wrench, sensor noise, and UKF off in Sprint-6 acceptance and pilot configs;
- fixed Validation selects/diagnoses only;
- the experiment runner does not open the frozen Test manifest, build Test, simulate Test, create
  Test metrics, or use Test for a decision;
- no loss, bounds, Workstation trajectory distribution, Validation manifest, controller, dynamics,
  dependencies, lockfiles, or environment metadata changed.

The initial user-directed audit and a pre-existing structural manifest test may read the static
Test JSON, but no acceptance or pilot process opened it. The file remained unchanged at SHA-256
`6ba8ff313189fcf7062b0ee029f6d2a41a2ec20b2a3116cec2f57bb02615477d`.

## Implemented safety/readiness functions

Checkpoint schema v2 now provides:

- dtype/shape/data records for current and selected raw gains;
- complete Optax tree/leaves with dtype and shape;
- complete Train/Validation metrics and history across resume;
- config, gain-registry, scoped executable-source, runtime/backend/x64, and Validation-content
  fingerprints;
- root seed, step/episode counters, selection, frozen opaque Test reference, parent checkpoint
  path/hash, and command lineage;
- canonical payload SHA-256 and strict nonfinite JSON rejection;
- counter/history/metric consistency checks;
- atomic JSON and JSONL writes with flush, file `fsync`, same-directory replace, and directory
  `fsync`;
- absent-or-empty target enforcement, distinct Resume target, and refusal to overwrite.

`--max-updates-this-process` creates a planned safe pause after a completed update. No emergency
signal checkpoint was implemented: compiled JAX work does not expose a trustworthy partial-update
boundary. A hard interrupt discards the current uncommitted step and resumes from the previous
atomic checkpoint.

The scaling benchmark now accepts 8/16-world Workstation protocol values, defaults to the
mass-only Workstation config, records config/seed/source/environment/gates, and writes a
machine-readable failure artifact even when episode construction fails before JIT.

## Exact end-to-end Resume acceptance

The authoritative commands are copied exactly in `RUNBOOK.md`. Each used a fresh process,
`timeout --signal=TERM --kill-after=30s 300s`, and external `/usr/bin/time -v`:

1. H20 mass-only technical fixture, two uninterrupted updates;
2. same config, one update, planned pause;
3. same config, checkpoint-1 Resume in a new directory, second update, automatic comparison with
   uninterrupted checkpoint 2.

Final source/run identities:

```text
execution HEAD: 182d0cc9068c4920ac25146ff0016260f802e5dd
source fingerprint: aae2c6b0b2d4962eb801ddb22947be57af149fc20c1a8d5f08a3dcc9ac14d6da
config fingerprint: 69cb19645493c7905f89ce0e914add62e06bd5faf07d632e12d4de7e9240aa87
runtime fingerprint: e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4
Validation fingerprint: 11e53e460bc93a64835af03bd5a6d22a2e4fb3f7c9d187324d23e7a3deac0c20
root seed: 20260731
backend/x64: CPU / false
```

Exact comparison policy was `rtol=0`, `atol=0`. This is justified because source, runtime,
backend, dtype, config, seeds, and pure JAX update order were identical. Timing, path, command, and
UTC provenance were correctly excluded.

All required field matches were true:

- current and selected raw vectors including dtype/shape;
- transformed Stage-1 gains;
- Optax tree and all three serialized leaves;
- step 2 and episode index 2;
- selected step/raw/loss and earliest-tie rule;
- both Train and both Validation history/metric rows, episode IDs, and gradients;
- root seed, frozen Test reference, config/registry/source/runtime fingerprints, and Validation
  manifest ID/content fingerprint.

Numeric result:

| Quantity | Uninterrupted | Resumed | Max abs difference |
|---|---:|---:|---:|
| raw `kp_xy` coordinate | -0.9791043401 | -0.9791043401 | 0 |
| raw `kp_z` coordinate | -0.2724381685 | -0.2724381685 | 0 |
| raw `kd_xy` coordinate | -1.3882366419 | -1.3882366419 | 0 |
| raw `kd_z` coordinate | -0.5576145649 | -0.5576145649 | 0 |
| transformed `kp_xy` | 0.4003765285 | 0.4003765285 | 0 |
| transformed `kp_z` | 1.2510789633 | 1.2510789633 | 0 |
| transformed `kd_xy` | 0.1997670680 | 0.1997670680 | 0 |
| transformed `kd_z` | 0.5005095601 | 0.5005095601 | 0 |
| Optax leaves | exact | exact | `[0, 0, 0]` |

History was exactly:

| Step | Train loss | Validation loss | Technical gates |
|---:|---:|---:|---|
| 1 | 0.0005476958468 | 0.0005446682335 | all pass |
| 2 | 0.0006267180433 | 0.0005445563002 | all pass |

Step 2 was selected by minimum fixed-Validation loss. This two-update technical fixture is a
continuation proof, not evidence that two updates are scientifically useful.

| Process | External wall | Peak-RSS | Swaps | Exit |
|---|---:|---:|---:|---:|
| uninterrupted two updates | 19.13 s | 1,151,416 KiB | 0 | 0 |
| planned one-update pause | 14.53 s | 1,051,448 KiB | 0 | 0 |
| fresh-process Resume update 2 | 14.43 s | 1,062,300 KiB | 0 | 0 |

## Local scaling sequence and gate outcome

The first and only authorized scaling command used the unchanged Workstation trajectory settings,
Stage 1, H100, one world, root seed `20260731`, CPU, and mass-only randomization. The final process
produced:

```text
status: failed
failure phase: deterministic_episode_construction_before_jit
error: no valid trajectory found in 16 deterministic attempts
external wall: 4.41 s
internal wall to failure: 2.54305152 s
Peak-RSS: 591,776 KiB
swaps: 0
exit: 1
compile/first/steady: not reached
test manifest opened: false
```

The earlier diagnostic attempts reported the same trajectory rejection. One initial H100 attempt
used the first technical Resume fixture before it was corrected back to the existing H20 smoke
distribution; it also failed before any update. These attempt files are preserved and explicitly
non-authoritative.

Gate matrix:

| Stage | Result | Wall / Peak-RSS | Decision |
|---|---|---|---|
| local H100 × 1 | FAIL before JIT | 4.41 s / 591,776 KiB | stop |
| local H100 × 4 | not run | n/a | forbidden after prior fail |
| local H200 × 4 | not run | n/a | forbidden after prior fail |
| local H400 / 8 / 16 worlds | not run | n/a | explicitly forbidden |
| Workstation H100 × 4 | not run | n/a | blocked by current NO-GO |
| Workstation H200 × 8 | not run | n/a | blocked by W1 |
| Workstation H400 × 16 | not run | n/a | blocked by W2 |
| H400 × 16 × 200 main run | not run | n/a | forbidden |

No scaling law or linear extrapolation is made. H20 and pre-JIT H100 observations cannot predict
H200/H400 compilation, steady time, or RAM; JAX compilation, horizon, world count, backend, and
cache state interact.

## Workstation protocol and required user inputs

`RUNBOOK.md` contains exact blocked commands for H100 × 4, H200 × 8, and H400 × 16; each is a
fresh process with external time/RSS, hard timeout, new output directory, environment/source/config
provenance, technical gates, and hashes. The next stage requires exit 0, no swap/critical RAM,
Peak-RSS well below RAM and the user limit, plausible compile/first/steady timing, exact Resume on
that machine, expected artifacts, finite values, and no gate/floor/saturation failure.

Before unblocking, the user must provide or confirm: OS, CPU/cores, RAM, GPU/VRAM or none, JAX
backend, CUDA/driver or N/A, repository path, same-filesystem artifact path, maximum runtime, and
maximum RSS. No credentials belong in these records. Current code is CPU-only; GPU portability is
not claimed.

## Tests and checks

```text
.venv/bin/python -m pip check
No broken requirements found (nonfunctional cache-permission warning only)

.venv/bin/python -m ruff check crazyflow examples tests
All checks passed!

.venv/bin/python -m ruff format --check crazyflow examples tests
113 files already formatted

/usr/bin/time -v .venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py tests/integration/test_examples.py \
  -k mellinger
14 passed, 5 skipped, 22 deselected in 29.14s; external 31.27s;
Peak-RSS 2,103,000 KiB; swaps 0

/usr/bin/time -v .venv/bin/python -m pytest -q \
  tests/integration/test_mellinger_research_runner.py::test_runner_resume_is_exact_and_never_opens_test_manifest
1 passed in 17.30s; external 18.87s; Peak-RSS 1,259,320 KiB; swaps 0

.venv/bin/python -m pytest -q tests/integration/test_examples.py -k mellinger_scaling_benchmark
1 skipped, 26 deselected in 0.01s
```

The generic research example remains deliberately skipped; its exact bounded processes above are
the authoritative integration evidence. The known MuJoCo viewer/EGL/X11 issue was not provoked by
an unfiltered suite.

Local focused tests, final Resume processes, failed pilot/diagnostic processes, and short checks
consumed only a few minutes, comfortably below the approximately 15-minute Sprint-6 local budget.
The longest successful Sprint-6 command was the 31.27 s focused set; the largest observed Peak-RSS
was 2,103,000 KiB. No measured command swapped or timed out.

## Artifacts and hashes

`artifacts/day6-readiness/SHA256SUMS` lists 46 files. `sha256sum -c SHA256SUMS` reported `OK` for
every entry. The index is 4,848 bytes and hashes to
`7580bd41d41314b9d13a9fa9ea052b67fcb715930d2ddbb53a9759023c85316d`.
Principal hashes are listed in `ARTIFACT_MANIFEST.md`; every resolved config, provenance, metric
history, checkpoint, summary, equivalence record, failure record, diagnostic attempt, and external
timing record is covered. No Test metric or plot exists.

## Commits and next sprint boundary

Sprint-6 executable commits:

```text
9a748c696e2ff6c6cc840bef19d1bda236692516  Add deterministic workstation resume readiness
efc3da4d3ae70e4de74c4ee3bdceaf9fa4f628ab  Use bounded technical config for resume proof
06681a20584ee3fc73c82ead61614988a71450fe  Record pre-JIT scaling pilot failures
182d0cc9068c4920ac25146ff0016260f802e5dd  Scope resume fingerprints to executable source
```

The next sprint must first resolve the horizon-consistent H100 pilot distribution through an
explicit scientific/configuration decision without silently changing the prepared H400
distribution. Then rerun from H100 × one world, preserve the same gates, and only proceed to the
Workstation stages if every predecessor passes. Do not start the main optimization or open Test.
