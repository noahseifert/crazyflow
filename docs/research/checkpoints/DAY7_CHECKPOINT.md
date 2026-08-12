# Sprint-7 checkpoint: horizon-consistent trajectory diagnosis and CPU pilot gate

Date: 2026-07-31
Branch: `research/differentiable-mellinger`
Initial HEAD: `3286b5dc03ed110fe7166f5f12e5f4f31ecf8e68`
Final executable-source HEAD: `1080aadf65a6df56528efcaa5da34f4dfc812e7d`
Evidence commit: `a22bea2` (`Record Sprint 7 trajectory evidence`)
Final documentation commit: the commit containing this checkpoint

## Scope and decision

Sprint 7 reproduces and explains the Sprint-6 H100 rejection, adds a construction-only diagnostic
path, measures the unchanged distribution across horizons, makes a pre-result scientific design
decision, implements one explicitly versioned trajectory distribution, and repeats bounded local
construction, scaling, and Resume gates.

**Decision: GO only for a later incremental Workstation pilot on the explicitly checked CPU
backend and only with a new explicit v2 Workstation config.** The unchanged `workstation.json`
remains legacy-v1 and was not edited. No main optimization, Test evaluation, H400 simulation,
Workstation job, GPU work, seed training study, controller/dynamics/gain/loss change, or manifest
change occurred.

This GO is not evidence of useful optimization, generalization, GPU readiness, Sim2Real,
firmware, hardware, physical safety, or full-distribution validity.

## Initial state and sprint boundary

The required preflight was exact:

```text
branch: research/differentiable-mellinger
HEAD: 3286b5dc03ed110fe7166f5f12e5f4f31ecf8e68
worktree: clean
```

`git log -10 --oneline`, the repository `AGENTS.md`, all required Sprint-6/current research
documents, both requested configs, the final Sprint-6 H100 failure record, and its checksum index
were read. The frozen Test manifest was never opened; its configured path remained an opaque
reference. The Day-6 acceptance evidence was complete and explicitly named this trajectory
question as the next sprint boundary.

## File allowlist and actual changes

The exact allowlist recorded before the first generated file or edit was:

- `crazyflow/control/mellinger/research/config.py`
- `crazyflow/control/mellinger/research/trajectories.py`
- `crazyflow/control/mellinger/research/__init__.py`
- `examples/jax/mellinger_scaling_benchmark.py`
- new `configs/research/mellinger/sprint7_pilot.json`
- `tests/unit/test_mellinger_research.py`
- `tests/integration/test_examples.py`
- `docs/research/PROJECT_STATE.md`
- `docs/research/RUNBOOK.md`
- `docs/research/DECISIONS.md`
- `docs/research/ARTIFACT_MANIFEST.md`
- `docs/research/USER_UNDERSTANDING_AND_OPEN_QUESTIONS.md`
- new `docs/research/checkpoints/DAY7_CHECKPOINT.md`
- new files only below `artifacts/day7-trajectory/`

Every actual change stayed inside this list. `runner.py`, `checkpointing.py`, `experiment.py`,
controller/dynamics files, `workstation.json`, `sprint6_resume.json`, both manifests,
dependencies, lockfiles, environment metadata, and global Crazyflow defaults remained unchanged.

## Read-only subaudits

Two requested read-only subagents were used. Neither changed a file, ran a test, constructed a
trajectory, started simulation/JIT, or opened the Test manifest.

Trajectory-semantics findings, rechecked by the main agent:

- legacy candidates are random Fourier geometry in normalized episode time;
- `H` is the control-interval count, there are `H+1` samples, and duration is
  `D=H/control_freq_hz`;
- the current carrier frequencies are `k/D`, so shorter horizons time-compress the same random
  geometry;
- velocity, acceleration, and jerk therefore scale as `D^-1`, `D^-2`, and `D^-3`;
- rejection induces a strongly horizon-dependent low-dynamics selection bias;
- a fixed 4-second parent validated before prefix extraction is the most backward-compatible
  horizon-consistent option, with an explicit non-hover terminal trade-off for short prefixes.

Experiment/test/backend findings, rechecked by the main agent:

- the fold-in seed tree and Train/Validation/Test object separation are deterministic and sound;
- the existing unit test opened the real Test manifest and was changed to opaque-reference
  coverage before Sprint-7 test execution;
- the old benchmark collapsed every episode-construction exception into one phase and had no real
  construction-only integration test;
- the Research pipeline is code-restricted by `device="cpu"`, although Crazyflow `Sim` accepts
  CPU or GPU and MJX can place its data on the selected device;
- this local WSL `.venv` exposes only `cpu:0` and has no JAX CUDA plugin.

## Phase A: unchanged reproduction and exact rejection cause

The unchanged Sprint-6 command was repeated before changing source semantics:

```bash
/usr/bin/time -v \
  -o artifacts/day7-trajectory/phase-a/timing/reproduce-h100-w1.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/workstation.json \
  --worlds 1 --horizon 100 --steady-repeats 5 \
  --output-dir artifacts/day7-trajectory/phase-a/reproduce-h100-w1
```

Result: the same `no valid trajectory found in 16 deterministic attempts`, exit 1, external
walltime 6.31 s, Peak-RSS 591,036 KiB, swaps 0. The run used root seed `20260731`, Train episode
0/world 0/trajectory, H100, one world, 100 Hz control, 500 Hz dynamics, `dt_control=0.01 s`, and
legacy source fingerprint `aae2c6b0b2d4962eb801ddb22947be57af149fc20c1a8d5f08a3dcc9ac14d6da`.

The old Boolean validator did not retain attempt reasons, so a diagnostic-only report was added
without changing candidate keys, formulas, bounds, ordering, or the 16-attempt limit. Exact
H100/episode-0 results:

| Constraint | Rejected attempts | Limit | Largest observed | Signed violation | Sample/time |
|---|---:|---:|---:|---:|---:|
| speed | 5/16 | 1.5 m/s | 2.411716 m/s | +0.911716 m/s | 41 / 0.410 s |
| acceleration | 16/16 | 5 m/s² | 44.704662 m/s² | +39.704662 m/s² | 48 / 0.480 s |
| jerk | 16/16 | 35 m/s³ | 945.269775 m/s³ | +910.269775 m/s³ | 41 / 0.410 s |
| tilt | 16/16 | 0.7 rad | 1.868721 rad | +1.168721 rad | 53 / 0.530 s |
| minimum specific force | 1/16 | 2 m/s² | 1.808247 m/s² | -0.191753 m/s² | 55 / 0.550 s |

Every attempt passed finite, workspace, and yaw-rate checks. Full per-attempt signed margins and
indices are in `phase-a/h100-attempt-diagnostics/trajectory_diagnostics_raw.json`.

Classification: not an implementation, SI-unit, seed, shape, `H/H+1`, or validation-consistency
bug. The exact cause is a scientifically unsuitable horizon semantic: the same normalized random
path is compressed from the prepared 4 seconds at H400 into 1 second at H100 while physical
derivative limits remain fixed. Relative to H400, identical coefficients have H100 speed ×4,
acceleration ×16, and jerk ×64.

## Phase B: unchanged legacy distribution across horizons

Construction-only command (no Sim, training, Validation, Test, or `jax.jit`):

```bash
/usr/bin/time -v \
  -o artifacts/day7-trajectory/phase-b/timing/legacy-horizons-32-seeds.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_scaling_benchmark.py \
  --config configs/research/mellinger/workstation.json \
  --trajectory-diagnostics-only \
  --diagnostic-horizons 20 50 100 200 400 \
  --diagnostic-seed-count 32 --diagnostic-max-attempts 16 \
  --output-dir artifacts/day7-trajectory/phase-b/legacy-horizons-32-seeds
```

Seed suite: root `20260731`, Train episodes 0..31, world 0, trajectory component; suite SHA-256
`6e9f4661e908bf055841185f53547e3f4a999250a89b3ddf06d52b1b0eb9a898`.

| Horizon / duration | Success within 16 | Attempt median / max | Candidates examined | Dominant rejection evidence |
|---|---:|---:|---:|---|
| H20 / 0.2 s | 0/32 | n/a | 512 | speed/acceleration/jerk/tilt: 512 each |
| H50 / 0.5 s | 0/32 | n/a | 512 | acceleration/jerk/tilt: 512 each; speed 503 |
| H100 / 1.0 s | 0/32 | n/a | 512 | acceleration/jerk: 512 each; tilt 511; speed 292 |
| H200 / 2.0 s | 12/32 | 6 / 15 | 415 | jerk 403; acceleration 353; tilt 210 |
| H400 / 4.0 s | 32/32 | 1 / 1 | 32 | none |

The separate, explicitly estimate-only H100 run raised the diagnostic cap to 128: 0 acceptances
in 2,048 candidates; every candidate violated acceleration and jerk. The one-sided Bernoulli 95%
candidate-acceptance upper bound is approximately 0.1462%, implying at most approximately 2.314%
success within 16 under that model. It was not used as a solution or pilot input.

## Scientific option decision

The option decision and all diversity thresholds were saved before v2 implementation or
measurement in `decision/pre_gate_criteria.json`.

| Option | Decision | Main reason |
|---|---|---|
| increase attempts | reject | does not remove time compression/conditioning; 0/2,048 H100 candidates |
| horizon-dependent amplitude/frequency bounds | reject | no single scale preserves position, velocity, acceleration, and jerk; degeneration risk |
| coefficient scaling/projection | reject for Sprint 7 | new bound-concentrated pushforward distribution and horizon-dependent scaling bias |
| fixed parent plus horizon prefixes | select | fixed physical spectrum/support, common rejection distribution, exact paired prefixes, H400 legacy preservation |
| unrelated stationary/spline family | defer | plausible but less backward-compatible and larger than the blocker requires |

## Versioned trajectory semantics

`TrajectoryConfig.distribution_version` now accepts exactly:

- `legacy_normalized_time_v1`: the default for old configs that omit the field; candidate RNG,
  equations, per-horizon envelope, rejection, and first-valid semantics are unchanged;
- `fixed_support_prefix_v2`: constructs and validates a complete configured parent (4.0 s in the
  Sprint-7 pilot), then returns an exact prefix. The pilot explicitly permits 1.0..4.0 s only.

Unknown versions, missing/nonfinite duration fields, non-integral support/control-time grids, and
requested durations outside the configured range fail eagerly. There is no fallback. Paired H400
v2 candidates and accepted attempt indices are exactly legacy-v1. Short v2 prefixes start at the
same C3 hover endpoint but are not forced to end at hover; this is an explicit new task/terminal
semantic. New Validation losses must not be compared numerically with legacy-v1 as if the dataset
were unchanged.

`configs/research/mellinger/sprint7_pilot.json` is a separate H100/one-world/two-update,
mass-only, Stage-1 local config. `workstation.json` and all old configs remain unchanged.

## 128-seed construction gate and diversity

Two fresh processes used Train episodes 0..127 with seed-suite SHA-256
`ff0b8b3534af1e2e5db5aa566e64f0e51c03bcd46fc988a8c33e15ed3852ce5f`.
Both produced timing-excluded scientific SHA-256
`4118a23a1f562fc27fbe99cefc6dfd8c6a02966f973929d68ce57d32217c8069`.

For H100, H200, and H400:

- 128/128 succeeded, every one on attempt 1;
- no rejection, nonfinite, or unexplained failure occurred;
- each seed selected the same attempt across horizons;
- H100/H200 arrays were exact prefixes of paired H400;
- H400 arrays and attempts were exact paired legacy-v1;
- all 127 other seeds differed from the first seed.

Predeclared diversity outcomes:

| Quantity | H100 | H200 | H400 |
|---|---:|---:|---:|
| median displacement RMS | 0.008244 m | 0.048517 m | 0.048330 m |
| median speed RMS | 0.044125 m/s | 0.170066 m/s | 0.174593 m/s |
| median x/y/z span | 0.01547/0.01041/0.00722 m | 0.09586/0.07961/0.04978 m | 0.12730/0.10762/0.05764 m |
| maximum returned acceleration | 1.12955 m/s² | 3.24937 m/s² | 3.58214 m/s² |
| maximum returned jerk | 6.64300 m/s³ | 19.32616 m/s³ | 19.32616 m/s³ |
| near-stationary count | 0/128 | 0/128 | 0/128 |

H200 v2 median displacement RMS was 1.428× the accepted legacy-H200 median; speed RMS was 0.954×.
Every predeclared configuration-derived and legacy-relative diversity threshold passed. This is an
empirical technical gate, not a proof of the full distribution.

## Local CPU pilots

Each pilot was a fresh process with `/usr/bin/time -v`, the 300-second hard timeout, five steady
samples, mass-only randomization, and no Test access.

| Stage | Status | Compile | First | Steady median | External wall | Peak-RSS | Swap / exit |
|---|---|---:|---:|---:|---:|---:|---:|
| H100 × 1 | PASS | 6.9485 s | 0.04160 s | 0.04137 s | 14.13 s | 1,006,864 KiB | 0 / 0 |
| H100 × 4 | PASS | 6.5943 s | 0.15372 s | 0.12626 s | 14.13 s | 1,033,476 KiB | 0 / 0 |
| H200 × 4 | PASS | 6.9792 s | 0.27633 s | 0.25525 s | 15.23 s | 1,038,940 KiB | 0 / 0 |

All finite, episode-success, zero-thrust, floor, and saturation technical checks passed. H100 × 4
was below 120 seconds and 8 GiB, so H200 × 4 was permitted. No H400 simulation, 8/16-world job, or
more-than-two-update job ran.

## Exact H100 Resume proof

Because trajectory source and the final pilot config changed, three fresh H100/one-world processes
repeated the exact two-update proof:

| Process | External wall | Peak-RSS | Swap / exit |
|---|---:|---:|---:|
| uninterrupted two updates | 17.03 s | 1,159,432 KiB | 0 / 0 |
| planned one-update pause | 13.43 s | 1,077,380 KiB | 0 / 0 |
| fresh-process resumed update 2 | 13.72 s | 1,082,724 KiB | 0 / 0 |

`resume_equivalence=pass` at `rtol=0`, `atol=0`: raw/selected/transformed gains, all Optax leaves,
counters, selection, full Train/Validation history and metrics, seeds, manifests, and fingerprints
matched exactly. Every maximum difference was zero. Source fingerprint was
`8ff8317b01de2aafda276bd13af3f2ee920a9610b764344c05d0fc620ae87320`; config fingerprint
`0de39663348f6908ad35989addc72e60b3cbd4b0d55d158a9482295d0383a47a`; runtime fingerprint
`e3c408d547653ebbb87107ca3c31b844c12acc43448378a5a48266306ee3d2a4`. No Test metric exists.

An attempted Resume from the old Sprint-6 step-1 checkpoint exited 1 after 5.91 s with
`checkpoint config fingerprint mismatch`, proving controlled incompatibility rather than silent
continuation.

## CPU/backend clarification

The present restriction has two independent layers:

1. Research code: `build_split_pipeline(...)` explicitly passes `device="cpu"`; `SimulationConfig`
   and the JSON configs expose no device switch. The present GO is therefore code-limited to CPU.
2. Local environment: JAX reports only `default_backend=cpu`, `devices=[cpu:0]`; installed related
   distributions are `jax`/`jaxlib` only, with no CUDA plugin, visible `nvidia-smi`, `/dev/dxg`, or
   `/usr/local/cuda`.

Crazyflow itself accepts `device="cpu"` or `"gpu"`, calls `jax.devices(device)[0]`, and places MJX
model/data on that device. The MuJoCo/MJX interface is not itself the source of this Sprint-7 CPU
restriction. No GPU support or portability test was implemented. CPU success is not GPU readiness.

## Checks and tests

```text
.venv/bin/python -m pip check
exit 0, 0.39 s; No broken requirements found (cache warning only)

.venv/bin/python -m ruff check crazyflow examples tests
exit 0, 0.03 s; All checks passed

.venv/bin/python -m ruff format --check crazyflow examples tests
exit 0, 0.04 s; 113 files already formatted

.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py tests/integration/test_examples.py \
  -k mellinger
exit 0; 21 passed, 5 skipped, 22 deselected in 27.63 s; external 29.89 s;
Peak-RSS 2,148,836 KiB; swaps 0

.venv/bin/python -m pytest -q tests/unit/control/test_mellinger.py
exit 0; 52 passed in 0.98 s; external 2.14 s; Peak-RSS 281,936 KiB; swaps 0

.venv/bin/python -m pytest -q \
  tests/integration/test_mellinger_research_runner.py::test_runner_resume_is_exact_and_never_opens_test_manifest
exit 0; 1 passed in 17.01 s; external 18.40 s; Peak-RSS 1,247,956 KiB; swaps 0
```

The documented broad headless-compatible regression was attempted once with the required 300 s
timeout. It reached 88% without a displayed failure, then exited 124 at 300.47 s; Peak-RSS was
6,772,496 KiB and swaps 0. It is incomplete and is not counted as passed. The known Viewer/EGL/X11
gap was not provoked.

## Artifacts, hashes, and measured effort

All Sprint-7 evidence is below `artifacts/day7-trajectory/`. `SHA256SUMS` lists 46 files; every
entry passes `sha256sum -c`. The index hash is
`2340d25908c737f05581dd1d9766f3ff8c1f190301008059e828da48b0730bd8`.
Principal evidence hashes include:

- Phase-A unchanged failure: `135baa6a...9d50f`;
- legacy 32-seed raw/summary: `89acfaa4...e04c99` / `623e5bc4...6b643`;
- gate run-1 raw/summary: `5289b405...1b63b` / `e0933158...cd1e9`;
- gate run-2 raw/summary: `6aae5024...4219` / `0b758985...4d3`;
- gate evaluation: `5f51b22d...04e55`;
- H100×1/H100×4/H200×4 benchmarks: `a2946cb2...0888`, `a6f20eeb...1db`,
  `22a99ee6...2d26`;
- Resume equivalence: `c1de79c7...aa7b`.

Approximate total local test/diagnostic/pilot process walltime was 8.5 minutes, including the
five-minute incomplete broad regression. It stayed below the approximately 15-minute budget. The
largest observed Peak-RSS was 6,772,496 KiB during that broad suite; no process swapped.

## Commits and next sprint boundary

```text
e224b5c Add construction-only trajectory diagnostics
1080aad Version horizon-consistent prefix trajectories
a22bea2 Record Sprint 7 trajectory evidence
<documentation commit> Document Sprint 7 trajectory go
```

The recommended next sprint is a CPU-only Workstation-pilot preparation sprint: collect the
required machine inventory and limits, create a new explicit v2 Workstation config without editing
legacy `workstation.json`, repeat exact Resume on that machine, and execute only the incremental
H100×4 then H200×8 gates. H400×16 and the main run remain blocked until every predecessor passes
within user-approved runtime/RSS limits. Test remains frozen.
