# Day-5 checkpoint: split-safe domain-randomization infrastructure

Date: 2026-07-31
Branch: `research/differentiable-mellinger`
Initial HEAD: `32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed`
Preserved Sprint-3/4 checkpoint: `edc57ba0bb884b69305b2a6af979e29f2de4566e`
Implementation HEAD used for runtime artifacts: `c4492b187d336d75757c560fecced53c7ec47b7a`
Final documentation/artifact commit: the commit containing this checkpoint

## Scope and claim boundary

This sprint implements local research infrastructure for structurally separate Train,
Validation, and Test splits, shared-gain multiworld episodes, lightweight mass/delay/wrench
randomization, staged gain registration, smooth random references, checkpoint/resume, metrics,
and provenance. It executes exactly one tiny one-update smoke and H20 scaling at 1/2/4 worlds.

It does not execute the prepared H400/16-world/200-update run, a seed study, an ablation matrix,
or a true test evaluation. Delay and wrench distributions are not empirically calibrated. Nothing
here is physical-flight, firmware-equivalence, hardware-safety, Sim2Real, or generalization proof.

## Phase-0 preservation and environment audit

The documented dirty Sprint-3/4 state was exactly attributable to the prior checkpoints and was
preserved without reset, stash, cleanup, or rewriting. Recomputed scientific source-state was
`8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`. All 18 Day-3 checkpoint
hashes matched, 20 JSON documents parsed, and all 14 result-level validation flags were true. No
secret candidate or Day-3 file above 1 MiB was found.

Hardware/environment observed read-only:

- AMD Ryzen 7 PRO 7840U, 8 cores/16 threads;
- WSL memory approximately 14 GiB plus 4 GiB swap;
- CPU-only JAX;
- Python 3.12.13, JAX/jaxlib 0.10.1, NumPy 2.5.1, Optax 0.2.8, pytest 9.1.1;
- `.venv/bin/python -m pip check`: no broken requirements;
- previous focused 36-test verification: `36 passed in 120.86s`, external walltime `2:04.07`,
  Peak-RSS `3,122,340 KiB`, swaps `0`.

Initial estimate was a large development sprint; 10–25 minutes local test/smoke compute, longest
single job 1–5 minutes, and 2–8 GiB RAM with high uncertainty. Main optimization, seed study,
ablation, and true test evaluation were explicitly deferred.

## Read-only audit conclusions

Three requested read-only subaudits were consolidated and rechecked by the main agent:

- Crazyflow API audit: reuse `Sim`, functional `step_fn`, world batching, dynamics mass, external
  force/torque channels, and `Trajectory`; fix two global zero-output reductions; do not reuse the
  existing example delay buffer because its zero/off-by-one/reset semantics are unsuitable.
- Leakage/experiment audit: replace the shared two-world mask design with separate simulator
  instances, explicit seed namespaces, fixed manifests, validation-only selection, and a training
  API without Test.
- Test/compute audit: make heavy research examples opt out before dynamic import, serialize all
  optimizer leaves, reject fingerprint mismatch, benchmark fresh processes, and measure external
  Peak-RSS with `/usr/bin/time -v`.

Confirmed source facts:

- state controller mass is approximately `0.029 kg`; first-principles dynamics mass is
  `0.0319 kg` for `cf2x_L250`;
- `SimCore.rng_key` is a typed scalar key despite an outdated `(N,1)` comment;
- `states.force` and `states.torque` are both world-frame inputs; dynamics transforms torque to
  the body frame internally;
- `kd_omega_z` cannot affect the current moment because yaw derivative error is explicitly set to
  zero;
- scheduler frequencies must divide exactly, otherwise controller integrator time and execution
  schedule disagree.

## Architecture and invariants

`build_training_pipelines(...)` constructs independent Train and Validation `Sim`/`SimData` and
pure step functions. `TrainingPipelines` deliberately has no Test. `build_test_pipeline(...)` is a
separate explicit entry point. The runner loads only the test manifest schema/ID for provenance;
it never realizes a test batch.

Research keys are `fold_in(root, split namespace, episode, world, component namespace)`. Train
episode index advances per update. Fixed Validation/Test roots come from their manifests. Tests
show that changing the Train root does not change fixed Validation samples.

`EpisodeBatch` is a JAX PyTree. Commands, references, masses, delays, force, and torque are dynamic
leaves; host episode IDs are removed before JIT so resampling does not change the PyTree metadata
and cause recompilation. The gain vector has shape `(number_of_registered_stage_gains,)`, never a
world axis, so all worlds share it.

## Domain-randomization decisions

| Component | Decision | Implemented semantics | Scientific status |
|---|---|---|---|
| Multiple worlds | reuse + wrap | independent samples/state on world axis; shared gains | verified locally |
| Mass | reuse dynamics field | one uniform true mass per world/episode, ±`0.0002 kg`; controller fixed | provisional engineering interval |
| Action delay | implement | integer control intervals, first-command fill, exact zero identity | uncalibrated heuristic; workstation disabled |
| Dynamic wrench | wrap existing state channels | stationary AR(1), force N and torque N m, both world frame | uncalibrated heuristic; workstation disabled |
| Sensor noise | defer | no audited estimator-in-loop interface or values | not implemented |
| UKF | defer | not in differentiated simulation path | not implemented |
| Random trajectory | implement over reused `Trajectory` | bounded Fourier with analytic position/velocity/acceleration/jerk | simulation filters only |

The smoke's `0.0001 N`, `0.000001 N m`, one-step maximum delay, and `0.05 s` correlation time are
only technical values. D-028 records the reported oral supervisor permission without inventing a
written source and keeps D-025 as the calibration gate.

## Gain inventory

Defaults are read from the current `cf2x_L250` parameter dictionaries:

| Registry name | Underlying parameter/axes | Default | Unit/status | Stage/decision |
|---|---|---:|---|---|
| `kp_xy` | state `kp[0:2]` | 0.4 | N/m | 1 optimize |
| `kp_z` | state `kp[2]` | 1.25 | N/m | 1 optimize |
| `kd_xy` | state `kd[0:2]` | 0.2 | N s/m | 1 optimize |
| `kd_z` | state `kd[2]` | 0.5 | N s/m | 1 optimize |
| `ki_xy` | state `ki[0:2]` | 0.05 | N/(m s) | 2; longer/bias excitation |
| `ki_z` | state `ki[2]` | 0.05 | N/(m s) | 2; longer/bias excitation |
| `kR_xy` | attitude `kR[0:2]` | 70000 | legacy PWM/rad | 3 optimize |
| `kR_z` | attitude `kR[2]` | 60000 | legacy PWM/rad | 4; yaw metric required |
| `kw_xy` | attitude `kw[0:2]` | 20000 | legacy PWM s/rad | 3 optimize |
| `kw_z` | attitude `kw[2]` | 12000 | legacy PWM s/rad | 4; yaw metric required |
| `ki_m_z` | attitude `ki_m[2]` | 500 | legacy PWM/(rad s) | 4; yaw metric required |
| `kd_omega_xy` | attitude `kd_omega[0:2]` | 200 | legacy PWM s²/rad | 3 optimize |
| `ki_m_xy` | attitude `ki_m[0:2]` | 0 | legacy PWM/(rad s) | defer; no supported nonzero bounds/init |
| `kd_omega_z` | attitude `kd_omega[2]` | 0 | dead executed derivative path | exclude |

The older four gain bounds remain unchanged. Other positive nonzero defaults use 0.5×–2×
configurable engineering bounds. The logistic transform enforces the interval but does not prove
closed-loop stability. Stage 1 has a real two-world finite, nonzero gradient test. Later stages
are registry/path-tested but not claimed identified by the H6/H20 technical maneuvers.

## Trajectory and metric contracts

For normalized time `u`, the envelope is `4*s7(u)*s7(1-u)`, where
`s7(u)=35u^4-84u^5+70u^6-20u^7`. Product-rule derivatives through third order combine with the
Fourier base. Deterministic finite rejection checks finite values, workspace/minimum/maximum
height, speed, acceleration, jerk, yaw rate, tilt, and minimum specific force. The prepared limits
are not hardware-certified.

Metrics retain the historical normalized loss components and add position p95, terminal position,
terminal velocity, per-episode technical success/failure, RMSE/max error, normalized motor effort
and smoothness, saturation, zero-gate, floor, and nonfinite fractions. The success threshold is a
technical smoke diagnostic, not a flight-safety criterion.

## Tests and measurements

Commands and actual outcomes:

```text
.venv/bin/python -m pytest -q tests/unit/control/test_mellinger.py
52 passed in 1.74s

/usr/bin/time -v .venv/bin/python -m pytest -q \
  tests/integration/test_mellinger_research_runner.py::test_real_multiworld_objective_has_shared_finite_gain_gradient_and_metrics
1 passed in 17.35s; external 19.27s; Peak-RSS 1,331,724 KiB; swaps 0

/usr/bin/time -v .venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_research.py tests/unit/test_mellinger_checkpointing.py \
  tests/integration/test_mellinger_research_runner.py tests/integration/test_examples.py -k mellinger
12 passed, 5 skipped, 22 deselected in 11.47s; external 13.47s;
Peak-RSS 1,537,224 KiB; swaps 0
```

The five skips are deliberate research entry points. They have bounded dedicated tests/runs and
must not be executed by the generic `main()` harness. Ruff lint/format and `git diff --check`
passed on the changed code/test files before runtime artifacts.

Broader regression:

```text
/usr/bin/time -v .venv/bin/python -m pytest -q \
  tests/unit/control/test_mellinger.py tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py tests/unit/test_mellinger_gain_optimization.py
88 passed in 107.16s; external 1:50.15; Peak-RSS 3,050,736 KiB; swaps 0

/usr/bin/time -v .venv/bin/python -m pytest -q
reached 89%, then aborted in tests/unit/test_visualizations.py::test_draw_capsule;
MuJoCo viewer initialization received SIGABRT; external 7:28.54;
Peak-RSS 6,977,152 KiB; swaps 0

/usr/bin/time -v .venv/bin/python -m pytest -q \
  --ignore tests/unit/test_render.py --ignore tests/unit/test_visualizations.py \
  -k 'not test_render_rgb_array'
523 passed, 36 skipped, 2 deselected, 2 warnings in 289.06s;
external 4:59.02; Peak-RSS 6,568,928 KiB; swaps 0
```

`DISPLAY=:0` existed, so the repository's `skip_if_headless` condition did not skip rendering, but
the session could not create a functioning MuJoCo viewer. The earlier regular `F` at approximately
89% aligns with `test_sim.py::test_render_rgb_array`; the later visualizations test aborted the
process before pytest could print a normal summary. No renderer or test harness outside the
sprint allowlist was changed. The successful broad command excludes exactly the two render test
modules and that unmarked RGB-render test.

The two warnings in the successful broad run are pre-existing JAX overflow-on-cast warnings from
`examples/contacts/contacts.py` and `test_contacts_between_drones[True]`; both tests passed.

End-to-end smoke:

```text
/usr/bin/time -v .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/smoke.json \
  --output-dir artifacts/day5-audit/local-smoke
external wall 17.11s; Peak-RSS 1,116,596 KiB; swaps 0; exit 0
train loss 0.0005378891947; validation loss 0.0005391469458;
gradient L2 0.0001059675851; test_metrics_present false
```

Internal synchronized timing: train compile `8.204728028 s`, first train execution
`0.013780410 s`; validation compile `0.882622932 s`, first validation execution
`0.002459971 s`; three validation steady samples `0.000977934/0.000800173/0.000802878 s`.

Fresh H20 scaling:

| Worlds | External wall | External Peak-RSS | Compile | First execution | Steady median |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.35 s | 1,024,200 KiB | 6.396366 s | see artifact | 0.009443 s |
| 2 | 14.87 s | 1,036,532 KiB | 7.139935 s | see artifact | 0.008859 s |
| 4 | 15.06 s | 1,045,444 KiB | 7.218194 s | see artifact | 0.027623 s |

All three used one gradient evaluation, warm-up, and five steady repeats. No 8-world step was
needed because the specified 1/2/4 benchmark already bounded the local smoke question. This is
not a model for H400 scaling.

Across the explicitly measured verification/smoke commands, local compute was approximately
18–20 minutes excluding development, documentation, and very short lint/JSON checks. This matches
the initial 10–25 minute test/smoke estimate. The longest attempted command was the unfiltered
suite at `7:28.54` before its renderer abort; the longest successful command was the broad
headless-compatible suite at `4:59.02`. Successful focused scientific commands stayed below
approximately 3.0 GiB, while the broad repository suite peaked near 6.27 GiB.

For the next sprint, expect medium development effort if it only calibrates configs and prepares
the Workstation execution protocol, but large effort if it adds yaw/attitude objectives or
system identification. Local regression remains about 5–8 minutes when render tests are excluded.
H100/H200/H400 world-scaling pilots should each be budgeted below 5 minutes initially and stopped
before increasing size if Peak-RSS or compile time becomes nonlinear. The prepared
`16 worlds × H400 × 200 updates` run cannot be estimated reliably from H20; plan a Workstation with
at least 32 GiB RAM and expect a minutes-to-hours CPU job until incremental pilots narrow the
uncertainty. A GPU is optional only after verifying that this Crazyflow/MJX path supports it and
re-running numerical equivalence; no GPU claim was tested here.

## Artifact hashes

All eight Day-5 files and their exact SHA-256/size are listed in `ARTIFACT_MANIFEST.md`. The local
smoke directory has exactly five expected files; each scaling directory has exactly one
`benchmark.json`. JSON parsing, finite loss checks, split-label checks, and absence of test metrics
passed after generation.

## Remaining gates

- Calibrate or supervisor-approve delay/wrench values before scientific interpretation.
- Add yaw/attitude metrics and exciting references before Stage 4; add longer bias episodes before
  interpreting integral-gain gradients.
- Benchmark horizon scaling and 4/8/16 worlds before the prepared workstation run.
- Freeze controller, config, selected checkpoint, and analysis before one-way true test evaluation.
- Decide whether/how simulation gains map to actual firmware units and establish a separate
  hardware safety process before any flight.
- Investigate the recurring XLA persistent-cache CPU feature warning; it did not cause a test
  failure here, but cross-machine cache portability must not be assumed.
- Optional `warp`/`mujoco_warp` imports report missing modules; current CPU first-principles runs
  do not require them.
