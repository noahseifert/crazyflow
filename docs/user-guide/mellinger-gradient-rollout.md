# Differentiable Mellinger tracking rollout

This prototype differentiates trajectory-tracking loss through Crazyflow's
pure JAX Mellinger controller and first-principles JAX dynamics. The native
Crazyflie bridge used to validate the controller is deliberately absent from
the rollout graph.

## Run

From the repository root:

```bash
.venv/bin/python examples/jax/mellinger_tracking.py \
  --trajectory figure8 \
  --horizon 200 \
  --output-dir artifacts/mellinger_tracking
```

The script writes a PNG with reference/actual tracking and weighted loss
components, plus a JSON record containing the Git commit, seed, horizon,
loss, metrics, gains, gradients, timing, and numerical validation results.
JAX timings call `block_until_ready()` so asynchronous execution is included.

## Functional pipeline

The object-oriented `Sim` instance is only a builder. The differentiated path
uses immutable values:

1. `SimData` from `Sim`, initialized at the trajectory's position and velocity;
2. `F.state_control(data, command)` to stage `(N, M, 13)` commands;
3. the pure function returned by `Sim.build_step_fn()`;
4. `jax.lax.scan` over commands;
5. `tracking_loss` over the resulting `RolloutTrace`.

`SimData` is the scan carry. It contains rigid-body and rotor states,
Mellinger position and attitude integrators, previous angular velocity,
staging buffers, controller step counters, physical parameters, and the RNG
key. Its PyTree structure, shapes, and dtypes stay constant over the horizon.
The default example uses one existing Crazyflow world rather than adding an
unnecessary `vmap`; commands already use `(T, N, M, 13)`, so later experiments
can put independent cases in the world dimension.

A Python loop inside `jax.jit` is generally unrolled during tracing. That
makes compile time and generated program size grow with a fixed horizon.
`lax.scan` represents the recurrence as a loop primitive. Reverse-mode
autodiff can still store residuals for all steps; for longer horizons,
`jax.checkpoint`/`jax.remat` can trade recomputation for lower backward-pass
memory.

## Analytic trajectories

`crazyflow.trajectory` provides circle and figure-eight references. Both use a
quintic ramp of phase rate. At `t=0`, phase rate and acceleration are zero;
position, velocity, and acceleration are continuous through the ramp.
Derivatives are analytic, not finite differences.

The reference fields and units are:

| Field | Shape | Unit |
|---|---:|---|
| `pos` | `(T, 3)` | m |
| `vel` | `(T, 3)` | m/s |
| `acc` | `(T, 3)` | m/s² |
| `yaw` | `(T,)` | rad |
| `yaw_rate` | `(T,)` | rad/s |

The initial simulator position and velocity equal the `t=0` reference. Rotor
state starts at the first-principles hover RPM, avoiding an artificial
zero-RPM launch transient.

## Optimization variables

Only four unconstrained float scalars are differentiated. A smooth logistic
transform maps them to physical gains and preserves common x/y gains:

| Physical gain | Default | Smooth range | Unit |
|---|---:|---:|---|
| `kp_xy` | 0.4 | [0.10, 1.20] | N/m |
| `kp_z` | 1.25 | [0.30, 2.50] | N/m |
| `kd_xy` | 0.2 | [0.05, 0.80] | N·s/m |
| `kd_z` | 0.5 | [0.10, 1.20] | N·s/m |

The transformed arrays replace `data.controls.state.params["kp"]` and
`["kd"]` only in the experiment's immutable `SimData`. Checked-in controller
defaults are not changed. The PyTree passed to `value_and_grad` contains no
integers, booleans, mode enums, or other nondifferentiable leaves.

## Loss

Position and velocity MSE are normalized by 0.25 m and 1 m/s. Default
dimensionless weights are 1.0 for position, 0.10 for velocity, 0.10 for
terminal position, and `1e-3` each for normalized rotor effort and rotor
smoothness. A weight of 0.05 multiplies a smooth softplus altitude-margin
penalty.

`has_aux=True` returns weighted components plus position/velocity RMSE,
maximum position error, effort, smoothness, final motor-saturation fraction,
nonpositive-thrust gate fraction, floor-clip fraction, and nonfinite-state
fraction. Boolean saturation/contact indicators are diagnostics only and do
not contribute gradients. The scalar safety term is smooth.

## Non-smooth operations

The real controller and dynamics are retained:

- position and attitude integrators use `clip`; gradients are zero outside
  their clamp intervals;
- torque PWM and motor force are saturated with `clip`; gradients are zero in
  saturated directions;
- nonpositive thrust uses `where` to gate torque and reset integrators; the
  selected branch supplies the gradient and the switching boundary is
  nondifferentiable;
- desired-axis normalization and quaternion rotations are smooth away from
  zero norm and representation singularities;
- motor spool-up/down dynamics use a `where` branch at equal RPM;
- quaternion integration has a tiny-rotation `where` guard;
- floor clipping uses boolean gates and kills position/velocity gradients
  below the floor;
- controller and dynamics modes are static Python configuration, not
  optimization variables.

No native bridge, host callback, integer PWM conversion, or firmware
quantization is present in the differentiated path. Motor saturation is
measured from commanded RPM at the controller's per-motor thrust limits.
The diagnostic does not separately identify torque-PWM clipping before the
motor mixer.

## Validation

The runnable example checks:

- finite loss and gradient leaves;
- a nonzero, bounded gradient norm;
- deterministic replay with the same seed;
- eager/JIT forward agreement;
- three central directional differences against autodiff;
- local loss decrease after a small normalized negative-gradient step.

The CI-oriented unit test uses a short actual first-principles rollout. A
longer local run should be used for scientific results and for assessing
saturation, memory, and trajectory-dependent sensitivity.

The `cf2x_L250` controller configuration uses its checked-in 0.029 kg mass,
while the first-principles drone model uses 0.0319 kg. The audited baseline
does not synchronize or overwrite those defaults. Sprint 3 evaluates one
additional matched condition by replacing controller mass only in immutable
experiment-local `SimData`; it does not change either checked-in default.

## Two-world batch diagnostics

The Day-2 diagnostic runs Figure-8 and circle references in one simulation:

```text
world 0 -> Figure-8 -> train mask      [1, 0]
world 1 -> circle   -> validation mask [0, 1]
combined mask                         [1, 1]
```

Commands have shape `(T, 2, 1, 13)`. Reference position and velocity use
`(T, 2, 1, 3)`, and the initial position/velocity for each world comes from
that world's own `t=0` sample. The four raw gains are shared across worlds.
`tracking_loss_per_case` reduces time and drone axes but retains the world
axis. `aggregate_tracking_metrics` then applies numeric masks. The train loss
therefore contains only world 0; the combined loss is the equal arithmetic
mean of both worlds. The circle case is a small held-out diagnostic, not a
comprehensive generalization result.

Run the full fixed design from the repository root:

```bash
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_batch_diagnostics.py \
  --horizon 200 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --fd-epsilon 1e-2 \
  --gain-perturbation-fraction 0.10 \
  --profile-repeats 5 \
  --output-dir artifacts/day2-audit/batch-h200
```

The output directory must not already exist. It receives exactly:

- `batch_diagnostics.json`
- `batch_tracking.png`
- `loss_and_sensitivity.png`
- `gradient_diagnostics.png`

The JSON records per-case and masked losses/metrics, train/validation/combined
gradients, gradient cosine, three train and combined directional checks, two
local negative-gradient trials, independent physical ±10% gain perturbations,
batch-versus-single checks, case-order checks, and synchronized profiling.
Profiling covers computation only. The script does not measure peak memory,
introduce rematerialization, import Optax, run an optimizer, or persist a gain
update.

The controlled physical perturbations are local diagnostics: one gain changes
at a time while the other three remain at baseline. Their signs and magnitudes
must not be interpreted as causal effects or as evidence that repeated updates
would improve either split. The controller/dynamics mass difference remains
fixed at 0.029 kg versus 0.0319 kg and is not a Day-2 variable.

## Train-only Optax gain optimization

Sprint 3 adds a fixed Optax Adam experiment:

```bash
MPLCONFIGDIR=/tmp/crazyflow-gradient-research-matplotlib \
  .venv/bin/python examples/jax/mellinger_gain_optimization.py \
  --horizon 200 \
  --steps 50 \
  --learning-rate 1e-2 \
  --control-freq 100 \
  --sim-freq 500 \
  --seed 20260724 \
  --fd-epsilon 1e-2 \
  --profile-repeats 3 \
  --robustness-horizons 20 100 200 400 \
  --output-dir artifacts/day3-audit/optimize-h200
```

The optimizer differentiates exactly `per_case_loss[0]`, the Figure-8/world-0
train objective. Only that gradient reaches `optax.adam`. Circle/world 1 and
the equal combined mean are recorded after every step but never affect an
update, the Adam state, hyperparameters, or checkpoint choice. The selected
research checkpoint is the minimum train loss, with the earliest step winning
an exact tie.

Only the four existing raw variables are updated. Physical gains always come
from the same logistic transform and bounds. The selected gains are written to
`optimized_gains.json` as a reusable research artifact; the program does not
write controller parameter files.

Each output directory contains exactly:

- `optimization_result.json`
- `optimized_gains.json`
- `robustness_results.json`
- `optimization_history.png`
- `gain_history.png`
- `robustness_matrix.png`

The full robustness matrix evaluates baseline and selected gains without
further optimization across H20/H100/H200/H400, Figure-8/circle, and two
controller/dynamics mass conditions: the audited 0.029/0.0319 kg mismatch and
an experiment-local 0.0319/0.0319 kg match. Mass is not differentiated.

The selected checkpoint is checked with finite train/validation/combined
gradients, three fixed normalized central directions at epsilon `1e-2`,
batch-versus-separate execution, world-order invariance, JIT/eager agreement,
and a normalized negative train-gradient step. Saturation, nonpositive-thrust
gate, floor, and nonfinite fractions remain explicit diagnostics. The program
does not introduce rematerialization, host callbacks, NumPy conversion inside
the objective, native code, noise, disturbances, or hardware integration.

Two analytic trajectories and two mass conditions are controlled simulation
evidence, not hardware, disturbance, stochastic-seed, sim-to-real, or broad
generalization validation. Profiling uses `jax.block_until_ready`; it reports
observed timings and ratios without a speedup or peak-memory claim.

## Split-safe domain-randomized experiments

The experimental `crazyflow.control.mellinger.research` package extends the historical examples
without changing Crazyflow defaults. It provides:

```python
build_training_pipelines(train_config, validation_config, root_seed) -> TrainingPipelines
build_test_pipeline(test_config, root_seed) -> SplitPipeline
build_episode_batch(pipeline, root_seed, episode_index, trajectory_config,
                    randomization_config, manifest=None) -> EpisodeBatch
research_objective(raw_gains, batch, *, step_fn, steps_per_command,
                   gain_stage, loss_config=None) -> (loss, auxiliary)
```

`TrainingPipelines` contains only Train and Validation. Test construction is explicit and is not
available to the training runner. Train episodes are resampled by episode index. Validation and
Test require fixed, versioned manifests. PRNG keys are derived through split, episode, world, and
component `fold_in` namespaces.

All worlds consume one shared raw gain vector. The named registry groups x/y axes and supports
four stages: position/proportional damping, position integrals, roll/pitch attitude terms, and
later yaw terms. Yaw derivative gain is excluded because the executed controller zeros that error;
zero-default roll/pitch moment-integral gains are deferred. New nonzero-gain bounds are engineering
bounds and do not guarantee stability.

True dynamics mass can be sampled once per world/episode while controller mass stays fixed.
Integer action delay operates on control intervals and has exact zero-delay identity. External
world-frame force `[N]` and torque `[N m]` can follow a stationary AR(1) process. Delay and wrench
parameters are not calibrated by this package; the prepared workstation preset disables both.

Random references are finite Fourier sums with analytic position, velocity, acceleration, and
jerk. A seventh-order start/end envelope creates rest at both endpoints. Bounded host-side
rejection checks workspace and configured kinematic limits and fails after `max_attempts`; those
limits are simulation filters, not certified hardware limits.

Run the bounded technical smoke from the repository root:

```bash
.venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \
  --config configs/research/mellinger/smoke.json \
  --output-dir artifacts/day5-audit/local-smoke
```

The runner writes resolved config, provenance, split-labelled JSONL metrics, an atomic checkpoint,
and a run summary. Validation selects the earliest minimum-validation checkpoint. No test metrics
are produced. Checkpoints include the entire Optax PyTree and reject config/registry/manifest
mismatches. See `docs/research/RUNBOOK.md` for resume, measurement, and the prepared—but not
executed—workstation command.
