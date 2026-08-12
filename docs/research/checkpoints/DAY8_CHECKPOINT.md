# Sprint-8 checkpoint: bounded manual CPU night-pilot preparation

Date: 2026-07-31
Branch: `research/differentiable-mellinger`
Initial/end-of-Sprint-7 HEAD: `1990b2292b849fa735b3b9926e31c5df3f14d609`
Final executable-source HEAD: unchanged from Sprint 7 at
`1080aadf65a6df56528efcaa5da34f4dfc812e7d`
Artifact/launcher commit: `5b14228f59b6b258e1902f1f5d0982f173d60f8b`
Final documentation commit: the commit containing this checkpoint

## Scope and decision

Sprint 8 prepares, but does not start, one bounded, manually launched CPU night pilot. The exact
release is H100×4, 50 optimizer updates, one seed, shared Stage-1 gains, mass-only randomization,
existing Validation, and no Test access.

**Preparation decision: GO.** The sole fresh two-update gate exited 0, remained finite, passed all
technical gates, produced complete checksum-valid atomic checkpoints, used 1.143 GiB Peak-RSS and
no swap, and had plausible timing. Fifty updates are released because the measured estimate is
about 3.6 minutes, or 7.2 minutes with an explicit factor-two allowance, well within the six-hour
hard cap.

This GO is only authorization for the user to start the exact checked config. The night pilot was
not started in Sprint 8. It is technical and exploratory evidence, not a main run, scientific
seed study, Test result, generalization result, GPU result, Sim2Real result, firmware/hardware
validation, or physical-safety evidence.

## Preflight and sprint boundary

Before any change, the required checks were exact:

```text
branch: research/differentiable-mellinger
HEAD: 1990b2292b849fa735b3b9926e31c5df3f14d609
worktree: clean
latest commit: 1990b22 Document Sprint 7 trajectory go
```

The repository `AGENTS.md`, complete Day-7 checkpoint, Project State, Runbook, Decisions, and
Artifact Manifest were read. The Sprint-7 checkpoint identifies its documentation commit by
content; current HEAD has the exact recorded message and is that commit. No later sprint was
active. The frozen Test manifest was never opened; its configured path remained an opaque string.

## File allowlist and unchanged boundaries

The declared allowlist was:

- new files only under `artifacts/day8-night-pilot/`;
- `docs/research/PROJECT_STATE.md`;
- `docs/research/RUNBOOK.md`;
- `docs/research/DECISIONS.md`;
- `docs/research/ARTIFACT_MANIFEST.md`;
- new `docs/research/checkpoints/DAY8_CHECKPOINT.md`.

No file in `crazyflow/`, `examples/`, `tests/`, existing configs, Validation/Test manifests,
dependencies, lockfiles, environment metadata, controller/dynamics parameters, or historical
artifacts changed. The runner, checkpointing, loss, gain bounds, gain registry, and Crazyflow
defaults remain unchanged.

## Immutable pilot semantics

Both preparation and released configs use:

- `fixed_support_prefix_v2`, H100, 500 Hz simulation, 100 Hz control;
- four Train and four existing-Validation worlds, one drone per world;
- root/optimizer seed `20260731` only;
- Stage 1 names `kp_xy`, `kp_z`, `kd_xy`, `kd_z` and one shared raw/physical vector;
- true dynamics mass sampled uniformly within ±0.0002 kg of nominal;
- unchanged controller-mass assumption;
- Action Delay and Wrench/Disturbance disabled;
- no sensor-noise or UKF path;
- unchanged learning rate `0.001`, loss weights, bounds, trajectory limits, and manifests.

The two configs differ only in `run_id` and update count: two for the preparation gate and 50 for
the released pilot. Their canonical fingerprints are
`273203c2d4256df35d902b3e2dbe9336352a5cb4464de4c58aec48f986e1798c` and
`6d0465b5776fe1f09b0f5b1ab8777d02b9b3bd01386d8f00cc6209fb97d2efaa`.

## Sole fresh H100×4 measurement

Exactly one fresh process ran, with no more than two optimizer updates:

```bash
/usr/bin/time -v \
  -o artifacts/day8-night-pilot/timing/preflight-h100-w4.time.txt \
  timeout --signal=TERM --kill-after=30s 300s \
  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config artifacts/day8-night-pilot/preflight_config.json \
  --output-dir artifacts/day8-night-pilot/preflight-h100-w4
```

| Measurement | Result |
|---|---:|
| Exit status | 0 |
| External walltime | 19.24 s |
| Peak-RSS | 1,198,476 KiB / 1.143 GiB |
| Swap | 0 |
| Train compile | 7.432617069 s |
| Validation compile | 0.725118265 s |
| Total reported compile | 8.157735334 s |
| First Train execution | 0.155692356 s |
| Runner start to first complete checkpoint | 13.097996712 s |
| Complete checkpoint 1 to complete checkpoint 2 | 4.096372366 s |
| Validation steady samples | 0.002953741 / 0.003233002 / 0.004259748 s |

The operational complete-update measurements use `provenance.started_at_utc` and the nanosecond
filesystem modification times of the atomically completed checkpoints. They include the real
episode/Validation/checkpoint workflow and therefore define the transparent night-budget
estimate; they are not presented as pure kernel timings.

Step 1/2 Train losses were `0.010203863494098186` / `0.010006280615925789`; gradient norms were
`0.006251446437090635` / `0.006109291687607765`. Validation losses were
`0.010010135360062122` / `0.01000117789953947`. Every value was finite and every Train/Validation
technical gate passed. Both checkpoint payload SHA-256 fields were independently recomputed and
matched. The directory contained exactly resolved config, provenance, metrics, checkpoints 1/2,
and summary. `test_metrics_present=False` and `test_manifest_opened=False`.

## Gate and 10/25/50-update decision

| Required preparation gate | Result |
|---|---|
| Exitcode 0 | PASS |
| No nonfinite loss/gradient/state/metric | PASS |
| Complete artifacts | PASS |
| Functional atomic checkpoints | PASS, both payload hashes valid |
| Peak-RSS below 12 GiB | PASS, 1.143 GiB |
| No relevant swap | PASS, 0 |
| Plausible runtime | PASS |

Using `13.098 + (N-1) × 4.096 + 2.046` seconds gives:

| Updates | Estimate | Decision |
|---:|---:|---|
| 10 | 52.0 s | valid but shortest observation |
| 25 | 113.5 s | valid intermediate option |
| 50 | 215.9 s / 3.6 min | selected |

Fifty is the hard maximum and supplies the most repeated atomic checkpoints and Train/Validation
trend samples. A factor-two allowance is about 7.2 minutes. The launcher nevertheless fixes a
21,600-second timeout, performs no automatic extension, and ends as soon as update 50 completes.
There is no second config, H200/H400 continuation, seed loop, or Test action.

## Launcher, stop, and real-runner Resume

`launch_night_pilot.sh`:

- validates branch, tracked/unmodified released config, and clean `crazyflow`/`examples` source;
- creates a UTC/HEAD-named absent run root and refuses collisions;
- writes launch Git HEAD, config SHA/fingerprint, environment, command, and six-hour budget;
- keeps complete runner stdout/stderr in separate log files;
- executes the real Train/Validation runner under `/usr/bin/time -v` and GNU `timeout`;
- delegates nonfinite checks, atomic per-update checkpoints, metrics, selection, and Resume to the
  existing runner;
- records a process group for controlled SIGTERM and retains the last complete checkpoint;
- accepts `--resume CHECKPOINT`, while still creating a new run root.

There is intentionally no partial/signal checkpoint. SIGTERM may discard the in-flight update and
Resume begins from the preceding complete atomic state. Config/source/runtime/registry/seed/
Validation/Test-reference/checksum incompatibilities remain hard errors.

`tmux` was already installed at `/usr/bin/tmux`; nothing was installed. The exact recommended
start, foreground fallback, status, controlled stop, Resume, and morning commands are in
`artifacts/day8-night-pilot/README.md`. Both shell syntax and fresh/Resume dry-run commands passed
without starting the night pilot. The read-only inspector reported `evaluation_status=PASS` for
the preparation output.

## Verification

```text
.venv/bin/python config-validation command
pass; preflight fingerprint 273203c2...e1798c; pilot fingerprint 6d0465b5...d2efaa

.venv/bin/python -m pip check
exit 0; No broken requirements found (nonfunctional cache warning only)

.venv/bin/python -m ruff check crazyflow examples tests
exit 0; All checks passed

.venv/bin/python -m ruff format --check crazyflow examples tests
exit 0; 113 files already formatted

.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py tests/integration/test_examples.py \
  -k mellinger
exit 0; 21 passed, 5 skipped, 22 deselected in 35.69 s

bash -n artifacts/day8-night-pilot/launch_night_pilot.sh
exit 0

.venv/bin/python -m ruff check artifacts/day8-night-pilot/inspect_pilot.py
exit 0; All checks passed

.venv/bin/python -m ruff format --check artifacts/day8-night-pilot/inspect_pilot.py
exit 0; 1 file already formatted
```

`shellcheck` was not installed and was not added. The known Viewer/EGL/X11 suite was not
provoked. The bounded automated preparation processes and checks were well within the approximate
15-minute execution budget.

## Artifacts and hashes

All Sprint-8 payload is below `artifacts/day8-night-pilot/`. `SHA256SUMS` contains 13 entries;
every entry passes. The 1,235-byte index hashes to
`7b50c9bb1461cc4ce4b24e8ed9a3e11f682c061c50e12629360d769ffa88ab93`.

Principal payload hashes:

- released config: `43a2a75b...bd07`;
- launcher: `0aa301e7...53c9`;
- morning inspector: `809ed7ac...9a1d`;
- machine-readable gate: `b92faa31...92c2`;
- step-1/step-2 checkpoints: `453cea67...33f3` / `6281bbe4...6d9a`;
- timing: `f296633b...aa97`;
- handover: `ae7a1080...a532`.

No `runs/` directory was produced because the night pilot remained unstarted. User-created run
evidence must be hashed after execution and supplied with the inspector output, launch manifest,
external timing, and log tails to the project-management chat.

## Final handover boundary

The only authorized next action is the user's manual execution of the exact 50-update launcher
after confirming mains power, disabled Windows/WSL sleep, cooling, no `wsl --shutdown`, and a
stable foreground terminal or tmux session. If the run is interrupted, inspect and Resume the last
complete checkpoint. If it fails integrity/nonfinite/resource gates, stop; do not substitute a
new configuration or advance to H200/H400, more updates, seeds, Test, or a main run.
