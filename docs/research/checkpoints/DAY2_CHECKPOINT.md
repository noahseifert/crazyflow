# Day-2 checkpoint

Date: 2026-07-24 (Europe/Berlin)  
Repository: `/home/noah3/bachelorarbeit/crazyflow-gradient-research`  
Branch: `research/differentiable-mellinger`  
Base/end HEAD: `9eac04ea28997e76b9bc8c6e97ca53ef93398d55`
(`Checkpoint differentiable Mellinger rollout day 1`; no Day-2 commit)

## Starting state and preflight

Initial `git status --short --branch`:

```text
## research/differentiable-mellinger
```

The requested audited implementation commit
`1aa648b15e9e9ed9bada8aaa7408cdff22d417fd` is an ancestor of HEAD. Current HEAD is its Day-1
checkpoint/documentation commit. The repository and branch were therefore consistent with the
authoritative transition.

Executed before editing:

```bash
git status --short --branch
git rev-parse HEAD
git branch --show-current
test -x .venv/bin/python
.venv/bin/python --version
.venv/bin/python -m pip --version
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q tests/unit/test_mellinger_tracking.py
```

Actual preflight:

- Python `3.12.13`;
- pip `26.1.2` from the repository-local `.venv`;
- `No broken requirements found`;
- unchanged Day-1 test: `7 passed in 33.42s`.

The pip cache ownership warning and optional `warp`/`mujoco_warp` import warnings were nonfatal.
No package was installed or updated.

The complete source/document audit covered every file required by the Day-2 task, including root
`AGENTS.md`. The actual constructor is
`Sim(n_worlds: int = 1, n_drones: int = 1, ..., rng_key: int = 0)`. `SimData` remains the
`lax.scan` carry; `Sim.build_step_fn()` supplies the pure inner step function.

## Implemented data flow

```text
Figure-8(T+1)  circle(T+1)
       |             |
       +-- stack on world axis --+
                                  v
reference position/velocity (T, 2, 1, 3)
state commands             (T, 2, 1, 13)
per-world initial state    (2, 1, ...)
                                  |
shared raw GainVariables (4 scalar float32 leaves)
                                  |
pure JAX controller + first-principles dynamics
outer lax.scan over T, static inner step_fn ticks
                                  |
RolloutTrace (T, 2, 1, ...)
                                  |
per-case loss/metrics (2,)
  | train mask [1,0]
  | validation mask [0,1]
  + combined mask [1,1] -> equal arithmetic mean
```

The Day-1 loss components, normalizations, and weights are unchanged. `tracking_loss_per_case`
retains the world axis and reduces time/drones. `tracking_loss` still returns the mean over cases;
for `N=M=1` it preserves the former semantics. Train loss contains only Figure-8/world 0.
Circle/world 1 is a diagnostic validation case and not a broad generalization claim.

The controller mass remains `0.029 kg`; `cf2x_L250` first-principles dynamics mass remains
`0.0319 kg`. Mass was not varied. Only `kp_xy`, `kp_z`, `kd_xy`, and `kd_z` use the existing
logistic transform and ranges.

## Changed and new files

Modified:

- `crazyflow/control/mellinger/tracking.py`
- `crazyflow/control/mellinger/__init__.py`
- `docs/user-guide/mellinger-gradient-rollout.md`
- `docs/research/PROJECT_STATE.md`
- `docs/research/RUNBOOK.md`
- `docs/research/ARTIFACT_MANIFEST.md`
- `docs/research/DECISIONS.md`

New:

- `examples/jax/mellinger_batch_diagnostics.py`
- `tests/unit/test_mellinger_batch_diagnostics.py`
- `docs/research/checkpoints/DAY2_CHECKPOINT.md`

Unchanged:

- `examples/jax/mellinger_tracking.py`
- `tests/unit/test_mellinger_tracking.py`
- all dependency, lock, environment, controller-default, and dynamics-default files.

## Verification commands and results

The prescribed combined checks were executed:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py
.venv/bin/python -m ruff check \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_tracking.py \
  examples/jax/mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py
.venv/bin/python -m ruff format --check \
  crazyflow/control/mellinger/tracking.py \
  crazyflow/control/mellinger/__init__.py \
  examples/jax/mellinger_tracking.py \
  examples/jax/mellinger_batch_diagnostics.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_batch_diagnostics.py
```

Actual results:

- environment: `No broken requirements found`;
- combined focused tests: `21 passed in 43.74s`;
- new Day-2 test alone: `14 passed in 60.12s`;
- Ruff lint: `All checks passed!`;
- Ruff format: `6 files already formatted`.

The new tests cover all fourteen requested relation groups: command/world layout; trace axes and
finiteness; split masks; per-case loss; batch/single loss and gradient; finite/nontrivial
gradients; JIT/eager/determinism; `scan` JAXPR; three directional derivatives; train/combined
descent; order invariance; physical bounds; and JSON schema/finite serialization.

## Executed experiment commands

Three commands were run exactly as recorded in `RUNBOOK.md`:

1. H20, seed `20260724`, three profiling repeats,
   `--output-dir artifacts/day2-audit/smoke-run-1`;
2. identical H20 replay,
   `--output-dir artifacts/day2-audit/smoke-run-2`;
3. H200, seed `20260724`, five profiling repeats,
   `--output-dir artifacts/day2-audit/batch-h200`.

All used 500 Hz simulation, 100 Hz control, finite-difference epsilon `1e-2`, physical gain
perturbation fraction `0.10`, and
`MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib`.

## H20 results

Losses and diagnostics:

| Objective/case | Loss | Position RMSE [m] | Max position error [m] |
|---|---:|---:|---:|
| Figure-8/train | `0.0022911166306585073` | `0.006479553878` | `0.015639800578` |
| Circle/validation | `0.0011237590806558728` | `0.004677512683` | `0.011013435200` |
| Combined | `0.001707437913864851` | `0.005650829058` | `0.015639800578` |

All saturation, nonpositive-thrust gate, floor-clip, and nonfinite fractions were zero.

Raw-space gradients:

| Objective | `kp_xy` | `kp_z` | `kd_xy` | `kd_z` | L2 norm |
|---|---:|---:|---:|---:|---:|
| Train | `-3.96287305e-6` | `-7.61165211e-6` | `-7.96433014e-5` | `-9.85476581e-5` | `1.26997387e-4` |
| Validation | `-1.41582791e-6` | `-7.62742775e-6` | `-2.88078281e-5` | `-1.01870079e-4` | `1.06148887e-4` |
| Combined | `-2.68935059e-6` | `-7.61954061e-6` | `-5.42255621e-5` | `-1.00208876e-4` | `1.14225746e-4` |

Train/validation gradient cosine was `0.9196212888`. Maximum relative directional errors were
`0.0139947375` for train and `0.00984670315` for combined.

The normalized raw-space `1e-2` negative-gradient trials reduced:

- train `0.0022911166307 -> 0.0022898591124` (difference `-1.2575183e-6`);
- combined `0.0017074379139 -> 0.0017063210253` (difference `-1.1168886e-6`).

Batch/separate maximum per-case loss difference was `2.3283064e-10`; combined loss minus separate
mean was `1.7462298e-10`; maximum combined-gradient leaf difference was `7.2759576e-12`.
Reordering differences were exactly zero.

Smoke-run-1 profiling medians:

- cached batch forward: `0.001432337 s`;
- cached batch forward/backward: `0.018772579 s`;
- cached sequential two-case forward: `0.002804433 s`;
- cached sequential two-case forward/backward: `0.030699458 s`;
- sequential/batch ratios: `1.95794216` forward, `1.63533513` forward/backward;
- compile plus first batch forward/backward: `12.971827052 s`.

These measurements are observations, not a speedup claim.

## H200 results

Loss components:

| Component | Train | Validation | Combined |
|---|---:|---:|---:|
| position | `0.060949910432` | `0.021152760834` | `0.041051335633` |
| velocity | `0.007938757539` | `0.001192477415` | `0.004565617535` |
| effort | `1.94378617e-6` | `4.77656897e-7` | `1.21072151e-6` |
| smoothness | `2.45957885e-8` | `6.19159435e-9` | `1.53936917e-8` |
| terminal | `0.004213325214` | `0.001747127506` | `0.002980226418` |
| altitude | `5.96811402e-12` | `5.48788228e-12` | `5.72799794e-12` |
| total | `0.073103964329` | `0.024092847481` | `0.048598404974` |

H200 position RMSE/max error:

- Figure-8/train: `0.061720088124 m` / `0.091990016401 m`;
- circle/validation: `0.036359973252 m` / `0.048079244792 m`;
- combined: `0.050652824342 m` / `0.091990016401 m`.

All H200 saturation, nonpositive-thrust gate, floor-clip, and nonfinite fractions were zero.

Raw-space gradients:

| Objective | `kp_xy` | `kp_z` | `kd_xy` | `kd_z` | L2 norm |
|---|---:|---:|---:|---:|---:|
| Train | `0.020742760971` | `-0.010740003549` | `-0.050698254257` | `-0.004768092185` | `0.056023720652` |
| Validation | `0.000481965661` | `-0.009044516832` | `-0.011416674592` | `-0.002544825664` | `0.014793653041` |
| Combined | `0.010612358339` | `-0.009892259724` | `-0.031057460234` | `-0.003656459507` | `0.034473389387` |

Gradient cosine was `0.8422763944`; the documented exact-zero-norm behavior is a null cosine and
status `undefined_zero_norm`.

Maximum relative directional errors:

- train: `0.0002238699672`;
- combined: `0.0004414202122`.

The local trials reduced:

- train `0.073103964329 -> 0.072545759380` (difference `-0.000558204949`);
- combined `0.048598404974 -> 0.048254765570` (difference `-0.000343639404`).

Aggregation and regression:

- batch per-case versus separate `N=1` loss maximum absolute difference: `4.4703484e-8`;
- combined loss minus separate-case mean: `-4.2840838e-8`;
- combined gradient versus mean separate gradient maximum leaf difference: `7.2643161e-8`;
- reversed case-order loss and gradient differences: exactly zero;
- JIT/eager loss difference: `1.4901161e-8`;
- JIT/eager maximum gradient leaf difference: `1.6763806e-8`;
- Day-1 per-case loss and all four gradient leaves passed `rtol=1e-5`, `atol=1e-7`.

## Controlled H200 gain sensitivity

Central loss sensitivity per physical gain unit:

| Varied gain | Train | Validation | Combined |
|---|---:|---:|---:|
| `kp_xy` | `0.0951752984` | `0.00221156512` | `0.0486934550` |
| `kp_z` | `-0.0201236010` | `-0.0169485435` | `-0.0185360759` |
| `kd_xy` | `-0.4261085700` | `-0.0972732421` | `-0.2616908362` |
| `kd_z` | `-0.0187491626` | `-0.0100689568` | `-0.0144090503` |

Every perturbed value remained strictly inside its existing logistic bounds. In this controlled
local matrix `kd_xy` had the largest absolute central sensitivity for all three objectives.
`kp_xy` was positive at H200 while the other three sensitivities were negative. These signs are
reported without an expected direction and do not imply causality, global behavior, or an
optimization prescription.

## H200 profiling

JAX `0.10.1`, backend/platform/device `cpu`, synchronized by `jax.block_until_ready`:

| Measurement | Minimum [s] | Median [s] | Maximum [s] |
|---|---:|---:|---:|
| cached batch forward | `0.006217759` | `0.006383277` | `0.006772743` |
| cached batch forward/backward | `0.148388891` | `0.151880844` | `0.158523609` |
| cached sequential two-case forward | `0.010207519` | `0.011121430` | `0.011817156` |
| cached sequential two-case forward/backward | `0.270861473` | `0.274278655` | `0.281013448` |

Compile plus first batch forward/backward: `13.463255712 s`. Median sequential/batch ratios were
`1.7422759501` forward and `1.8058805033` forward/backward. No speedup claim is made. Plotting and
JSON writes were outside the timed regions. Peak memory was not measured; no rematerialization was
introduced.

## Reproducibility and artifacts

The two H20 scientific JSON payloads were equal after excluding only the six named profiling-time
blocks in `RUNBOOK.md`. All corresponding plot pairs were byte-identical:

- `batch_tracking.png`:
  `578fa448fc85862d7f9afcfac4b3d966c43716057ad589616e78d8f13752f001`;
- `loss_and_sensitivity.png`:
  `00868b6aabefc1ed900737d207f46b80d3209726a70a59531bb2272e50dd9cbc`;
- `gradient_diagnostics.png`:
  `d05fe557fadc5cd518779af7e0ad3c3553c015207d23399e8fa95a13d523ec2e`.

Generated files:

- `artifacts/day2-audit/smoke-run-1/batch_diagnostics.json`
- `artifacts/day2-audit/smoke-run-1/batch_tracking.png`
- `artifacts/day2-audit/smoke-run-1/loss_and_sensitivity.png`
- `artifacts/day2-audit/smoke-run-1/gradient_diagnostics.png`
- `artifacts/day2-audit/smoke-run-2/batch_diagnostics.json`
- `artifacts/day2-audit/smoke-run-2/batch_tracking.png`
- `artifacts/day2-audit/smoke-run-2/loss_and_sensitivity.png`
- `artifacts/day2-audit/smoke-run-2/gradient_diagnostics.png`
- `artifacts/day2-audit/batch-h200/batch_diagnostics.json`
- `artifacts/day2-audit/batch-h200/batch_tracking.png`
- `artifacts/day2-audit/batch-h200/loss_and_sensitivity.png`
- `artifacts/day2-audit/batch-h200/gradient_diagnostics.png`

Each directory contains exactly these four fixed filenames. Prior Day-1 artifacts were preserved.

## Acceptance result

All Day-2 acceptance criteria passed:

- unchanged Day-1 test remains at 7 tests;
- all 14 new tests, combined 21-test run, Ruff lint/format, and `pip check` pass;
- true `N=2`, `M=1` distinct-world rollout and disjoint numeric split are verified;
- all states, losses, metrics, gradients, and relevant norms are finite/valid;
- batch/single aggregation, gradient mean, JIT/eager, determinism, and order invariance pass;
- train and combined directional errors are below `5e-2`;
- both local negative-gradient trials reduce their own objective;
- H200 per-case loss/gradients reproduce Day 1 within fixed tolerances;
- all profiling times are positive and finite;
- both H20 scientific payload and byte-level plot reproducibility pass;
- H200 has all four artifacts and `all_validations_passed: true`.

## Warnings, risks, and Day-3 gate

- The fixed two-case split is limited evidence, not a comprehensive generalization result.
- The controller/dynamics mass mismatch remains `0.029 kg` versus `0.0319 kg`.
- Tested runs did not exercise motor saturation, the nonpositive-thrust gate, or floor clipping.
- Long-horizon reverse-mode residual memory remains an open risk; no peak-memory measurement or
  rematerialization was performed.
- Sensitivity is local, one-gain-at-a-time, and diagnostic.
- CPU timing ratios are descriptive and are not an acceptance speedup claim.

No optimization loop, Optax import/dependency, persistent gain update, native C/C++ bridge,
BetaFlight, or TinyMPC was introduced. No dependency, lockfile, environment metadata, checked-in
controller default, or dynamics default changed.

Optax remains gated. A separately scoped Day-3 sprint may consider it only because this checkpoint
records batch, aggregation, reproducibility, and gradient criteria as passed. Day 3 must still
define its own experiment design and preserve the current evidence.

## Closing repository state

Final audit:

```text
forbidden_code_paths_absent=true
No broken requirements found.
7 passed in 18.00s
21 passed in 43.72s
All checks passed!
6 files already formatted
```

`git diff --check` produced no output. All three JSON files had
`all_validations_passed: true`; all three result directories contained exactly four files
(twelve total).

Final `git status --short --branch`:

```text
## research/differentiable-mellinger
 M crazyflow/control/mellinger/__init__.py
 M crazyflow/control/mellinger/tracking.py
 M docs/research/ARTIFACT_MANIFEST.md
 M docs/research/DECISIONS.md
 M docs/research/PROJECT_STATE.md
 M docs/research/RUNBOOK.md
 M docs/user-guide/mellinger-gradient-rollout.md
?? artifacts/day2-audit/
?? docs/research/checkpoints/DAY2_CHECKPOINT.md
?? examples/jax/mellinger_batch_diagnostics.py
?? tests/unit/test_mellinger_batch_diagnostics.py
```

Tracked `git diff --stat`:

```text
 crazyflow/control/mellinger/__init__.py       |   8 ++
 crazyflow/control/mellinger/tracking.py       | 114 ++++++++++++++---
 docs/research/ARTIFACT_MANIFEST.md            |  18 +++
 docs/research/DECISIONS.md                    |  72 +++++++++++
 docs/research/PROJECT_STATE.md                | 121 +++++++++++++++---
 docs/research/RUNBOOK.md                      | 177 +++++++++++++++++++++++++-
 docs/user-guide/mellinger-gradient-rollout.md |  57 ++++++++-
 7 files changed, 523 insertions(+), 44 deletions(-)
```

Git omits the three new source/test/checkpoint files and generated artifact tree from that
statistic because they are untracked. Initial and final HEAD are both
`9eac04ea28997e76b9bc8c6e97ca53ef93398d55`; no commit was made.
