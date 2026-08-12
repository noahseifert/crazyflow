# Project state

Status date: 2026-08-08
Repository: `/home/noah3/bachelorarbeit/crazyflow-gradient-research`  
Branch: `research/differentiable-mellinger`  
Gradient-closure input HEAD: `ff85c8e9c0e73bcf97dfd5f2a747aa374a834f91`
Gradient-closure commit: `d386148edad9242720177180183940e74c081fde`
Sprint-3 base/end HEAD: `32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed` (no Sprint-3 commit)
Sprint-3 source state: `8559648a3c52ca262f8eb5b4160540cc8e138c42c85d748f13daefe996d268f7`
Sprint-4 base/end HEAD: `32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed` (then uncommitted)
Sprint-3/4 preservation checkpoint: `edc57ba0bb884b69305b2a6af979e29f2de4566e`
Sprint-5 implementation HEAD before documentation: `c4492b187d336d757c560fecced53c7ec47b7a`
Sprint-6 final executable-source HEAD: `182d0cc9068c4920ac25146ff0016260f802e5dd`
Sprint-7 final executable-source HEAD: `1080aadf65a6df56528efcaa5da34f4dfc812e7d`
Sprint-7 final/end HEAD and Sprint-8 preparation base:
`1990b2292b849fa735b3b9926e31c5df3f14d609`
Sprint-8 artifact/launcher commit: `5b14228f59b6b258e1902f1f5d0982f173d60f8b`
Stage-2 Tranche 1 base HEAD: `bfce9fcf04f97a609f569a6c9d95ea4f328a1cee`
Stage-2 Tranche 1 Work Order: `WO-GR-S2-001`
Friday-evidence base HEAD: `4eb411d5f181baf1742ea0cf980199f5e9a8e3b1`
Friday-evidence generator/test commit: `2499aad364a44eadcc259005f4c18d8a21acb332`
Friday-evidence Work Order: `WO-GR-F0-001`
Day-25 robust-foundation base: `a4e4136f8312851790555e3de6cdf23065252edd`
Day-25 corrected evaluator/CLI/test commit: `2288ab2fe3b55f585fb9b14a8fe9d87c07eb6c1a`
Day-25 Work Order: `WO-GR-G2-003` / prospective decisions `D-045`, `D-047` (`CORR-01`)

## Research goal

The project evaluates differentiable trajectory tracking through Crazyflow's pure JAX Mellinger
controller and pure JAX first-principles dynamics. Analytic reference trajectories provide smooth
state setpoints. Controller and dynamics state are rolled out functionally, and the resulting
scalar tracking loss is differentiated with respect to selected controller gains.

The native C/C++ Mellinger bridge is outside this repository and outside the gradient path. This
work does not claim complete physical, firmware, sim-to-real, or hardware validation.

## Current closure state

The formal Gradient-repository closure deterministically evaluates the ten existing H100 runs;
it starts no simulation, optimization, seed, later horizon, or Test action. The versioned
synthesis and its generator are in
[`artifacts/day19-h100-analysis/`](../../artifacts/day19-h100-analysis/), with the exact
reproduction procedure in [`RUNBOOK.md`](RUNBOOK.md#gradient-repository-closure-reproduction) and
the closure record in
[`GRADIENT_RESEARCH_CLOSURE_2026-08-02.md`](checkpoints/GRADIENT_RESEARCH_CLOSURE_2026-08-02.md).
Pre-commit approval was granted after that record was prepared, and the allowlisted closure was
committed on 3 August 2026 as `d386148edad9242720177180183940e74c081fde` (`research: close
ten-seed H100 gradient study`). The first subsequent read-only closure audit passed every Git,
data, reproduction, test, protection, and claim check, but its formal result was
`GRADIENT-SCHLUSSAUDIT: FAIL` solely because four documents still described the earlier
pre-commit state inconsistently. At that point, a repeated read-only closure audit had not yet been performed.

All ten predeclared root seeds have exactly one complete 5,000-update H100 run. The synthesis
verifies 610 raw files, all 600 files covered by the ten `RUN_SHA256SUMS` indexes, 500 internal
checkpoint-payload hashes, 100,000 ordered Train/Validation metric rows, released/resolved
configs, source/runtime/gain/Validation fingerprints, successful process status, absence of
Resume/retry directories, and the locked Test boundary. All ten fixed-Validation curves decrease
strictly over all 4,999 transitions and select update 5,000 as the earliest global minimum.

The mean relative Validation improvement from update 1 to the selected update is
`74.7788986450 %`; its frozen two-sided 95% Student-t interval is
`[74.7757723574 %; 74.7820249326 %]`. Mean selected Validation loss is
`0.0025246622506529095` with interval
`[0.002524350301011905; 0.002524974200293914]`. These are reproduced results from existing data,
not new experimental output. They are positive simulation evidence only under the frozen H100
configuration and documented Validation selection.

The numerical, integrity, resource, and Test-boundary gates derivable from the raw runs pass. The
separate execution-protocol-conformity gate does not become true from those data: the provided
historical lead handoff reports two failed just-in-time Sprint-17 wrapper/preflight attempts and
continued Seed-09/10 execution, while the repository contains no authorization ID, deviation
record, waiver, or later explicit horizon authorization. Seed 09 and 10 remain technically valid
and included in the predeclared primary panel; deleting, repeating, or selectively excluding them
would damage the audit trail. H200/H400 nevertheless remains formally **NO-GO / not authorized**.
This is a governance/protocol classification, not a negative H100 result. A future change would
require a new, explicit scientific decision and is outside this closure.

The Test split remains closed. The analysis records its opaque reference only, finds zero Test
metric rows, and does not open, parse, validate, build, simulate, or evaluate the Test manifest.
No H200/H400 result, hardware/HIL/flight evidence, firmware/STM32 equivalence, Sim2Real evidence,
global optimality, or cross-configuration superiority is established.

The Day-10 and H100 primary data remain untracked and unchanged. On 2 August 2026 an independent
TAR copy was verified locally and placed in Uni-OneDrive; exact size, SHA-256, destination, and
the no-cloud-redownload limitation are recorded in
[`PRIMARY_DATA_ARCHIVE.md`](PRIMARY_DATA_ARCHIVE.md). That receipt is backup evidence, not Git
versioning or renewed semantic validation.

**Current formal sequence:** the four-file documentation correction was completed, accepted, and
committed as `10ea7d6dd8c46f8273aab0961ab0dc33025c51fe`. The subsequent read-only repeat
audit reconfirmed all continuing technical, scientific, and protection findings but formally ended
`GRADIENT-SCHLUSSAUDIT: FAIL` solely because this passage still treated that correction commit as
pending. After this minimal `PROJECT_STATE.md` correction is committed, only one further
read-only repeat audit remains as the formal closure step; it has not yet been performed or
passed. H200/H400, Test, hardware, and new research remain outside the phase.

## Historical technical progression through Sprint 12A

Sprint 8 prepares, but does not start, one bounded manual CPU night pilot. A single fresh H100×4
Train/Validation process executed the maximum permitted two preparation updates with
`fixed_support_prefix_v2`, one optimizer seed, shared Stage-1 gains, and mass-only ±0.0002 kg
randomization. It exited 0 with finite losses/gradients, four passing Train/Validation technical
gates, two checksum-valid atomic checkpoints, 19.24 s external walltime, 1,198,476 KiB Peak-RSS,
and zero swap. Measured operational time was 13.098 s to the first complete update and 4.096 s
between the first and second complete updates.

The resulting decision is GO for exactly 50 manually started updates, estimated at about 3.6
minutes or 7.2 minutes with a transparent factor-two allowance, with a hard six-hour timeout and
no automatic extension. The checked launcher creates a unique run root, logs stdout/stderr and
environment, records Git/config/command provenance, uses the real runner's per-update atomic
checkpoints, and resumes only into another new run root. The frozen Test reference remained
opaque; no Test manifest, episode, simulation, or metric was opened or produced. The pilot remains
unstarted. Exact handover and evidence are in
[`artifacts/day8-night-pilot/README.md`](../../artifacts/day8-night-pilot/README.md) and
[`checkpoints/DAY8_CHECKPOINT.md`](checkpoints/DAY8_CHECKPOINT.md).

Sprint 7 resolves the H100 trajectory blocker without changing legacy semantics. The exact cause
is duration-dependent time compression in the legacy normalized-time Fourier distribution:
identical geometry has H100 versus H400 velocity ×4, acceleration ×16, and jerk ×64. A 32-seed
construction-only audit measured success rates of 0%/0%/0%/37.5%/100% at
H20/H50/H100/H200/H400. Even an estimate-only 2,048-candidate H100 sample produced zero accepted
candidates.

The new explicit `fixed_support_prefix_v2` distribution validates one 4-second parent and returns
H100/H200/H400 prefixes, while old configs default to unchanged
`legacy_normalized_time_v1`. A repeated 128-seed gate passed at all three horizons with exact
prefix consistency, exact paired H400 legacy equality, deterministic scientific hashes, and no
near-stationary samples under predeclared criteria. Local CPU H100×1, H100×4, and H200×4 pilots
all passed; exact H100 two-update Resume also passed at zero tolerance. Sprint-7 details and claim
limits are in [`checkpoints/DAY7_CHECKPOINT.md`](checkpoints/DAY7_CHECKPOINT.md).

The resulting GO is only readiness for a later incremental CPU Workstation pilot using a separate
explicit v2 config. `workstation.json` remains legacy-v1 and unchanged. No main run, Test access,
H400 simulation, GPU validation, or scientific optimization result exists.

Sprint 6 proves exact two-update checkpoint/resume equivalence in fresh CPU processes and adds
strict compatibility, atomic metric, lineage, nonfinite, checksum, and no-overwrite safeguards.
It does not authorize a Workstation main run: the mandatory first H100/one-world scaling pilot
fails before JIT because the unchanged H400 Workstation trajectory distribution cannot realize a
valid H100 trajectory in 16 deterministic attempts. H100/four-world and H200/four-world local
stages were therefore not run. Exact evidence is in
[`checkpoints/DAY6_CHECKPOINT.md`](checkpoints/DAY6_CHECKPOINT.md).

Sprint 5 supersedes the old feature freeze after a user-reported oral supervisor decision. It
implements a train/validation/test architecture for later domain-randomized optimization, but runs
only one tiny technical smoke and three H20 scaling measurements. The exact implementation,
verification, artifact hashes, and remaining gates are recorded in
[`checkpoints/DAY5_CHECKPOINT.md`](checkpoints/DAY5_CHECKPOINT.md). The test manifest was validated
but no test episode was built or evaluated.

Day 1, Day 2, and Day 3 are implemented and functionally verified. Day-1 evidence is recorded in
[`checkpoints/DAY1_CHECKPOINT.md`](checkpoints/DAY1_CHECKPOINT.md); actual Day-2 commands, measured
values, and acceptance evidence are in
[`checkpoints/DAY2_CHECKPOINT.md`](checkpoints/DAY2_CHECKPOINT.md). Sprint-3 optimization,
robustness, profiling, and acceptance evidence are in
[`checkpoints/DAY3_CHECKPOINT.md`](checkpoints/DAY3_CHECKPOINT.md).

Sprint 4 froze the scientific implementation and performed only read-only verification,
documentation audit, and supervisor handover. It changed no Python, tests, dependencies, lockfiles,
controller/dynamics defaults, or result artifacts and generated no new scientific artifacts. The
audit and actual verification results are in
[`checkpoints/DAY4_CHECKPOINT.md`](checkpoints/DAY4_CHECKPOINT.md). Explanatory handover documents:

- [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md)
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)
- [`SUPERVISOR_MEETING_BRIEF.md`](SUPERVISOR_MEETING_BRIEF.md)

Day-2 execution provenance remains historical and unchanged: the records were generated at
`9eac04ea28997e76b9bc8c6e97ca53ef93398d55` with `source_dirty: true` and source-state
`a1d7f1dce6ed3b1aa9cbb7cc53abb4a3bcada379129dfbde1787c1a5d1564229`. Commit
`32cb8ffae8ec3b522c589dffc8cc6d9b97d8f1ed` is the later clean, authorized Sprint-2 checkpoint
from which Sprint 3 started.

The actual data flow is:

```text
analytic circle/figure-eight trajectory
  -> Trajectory fields at T+1 sample times
  -> rollout-aligned slice [1:]
  -> state setpoints (T, 13)
  -> broadcast commands (T, N, M, 13)
  -> pure JAX Mellinger state/attitude/force-torque stages
  -> commanded rotor RPM (T, N, M, 4)
  -> pure JAX first-principles dynamics
  -> jax.lax.scan over T control intervals
  -> scalar normalized tracking loss plus auxiliary metrics
  -> jax.value_and_grad(..., has_aux=True)
```

`Sim` is used to construct immutable `SimData` and the pure `step_fn`; object-oriented mutation is
not part of the differentiated objective. The outer tracking recurrence is a `lax.scan`. Each
command also calls the step function for a static number of dynamics ticks; Crazyflow implements
those ticks with an inner `lax.scan`.

Day 2 uses the audited constructor
`Sim(n_worlds=2, n_drones=1, drone="cf2x_L250", dynamics=Dynamics.first_principles,
control=Control.state, ...)`. Figure-8 and circle commands are stacked as `(T, 2, 1, 13)`;
reference position/velocity have `(T, 2, 1, 3)`. Each world is initialized from its own `t=0`
reference. `tracking_loss_per_case(...)` returns `(2,)`, and numeric masks define:

- train: Figure-8/world 0 with `[1, 0]`;
- validation: circle/world 1 with `[0, 1]`;
- combined: equal arithmetic mean with `[1, 1]`.

The validation case never contributes to train loss. It is a small held-out diagnostic, not a
comprehensive generalization result. The same four raw gains are shared across both worlds.

## Structures, shapes, and timing

`T` is the control horizon, `N` the world count, and `M` the drones per world. The verified example
uses `N=M=1` on Day 1. Day 2 and Day 3 use `N=2, M=1` for the joint train/validation batch;
separate equivalence and individual robustness cells use `N=1, M=1`.

| Structure | Relevant fields | Shape |
|---|---|---:|
| `Trajectory` | `time`, `yaw`, `yaw_rate` | `(T,)` |
| `Trajectory` | `pos`, `vel`, `acc` | `(T, 3)` |
| state setpoint | `[pos(3), vel(3), acc(3), yaw, roll_rate, pitch_rate, yaw_rate]` | `(T, 13)` |
| batched setpoint | time, world, drone, command | `(T, N, M, 13)` |
| `SimState` | `pos`, `vel`, `ang_vel`, `force`, `torque` | `(N, M, 3)` |
| `SimState` | `quat` | `(N, M, 4)` |
| `SimState`/control | actual and commanded rotor RPM | `(N, M, 4)` |
| controller state | position/attitude integrators and prior angular velocity | `(N, M, 3)` |
| `RolloutTrace` vector fields | leading time dimension plus corresponding state shape | `(T, N, M, ...)` |

The trajectory generator creates `T+1` samples so simulator position and velocity can be
initialized from `t=0`; commands and loss references use samples `1..T`. Defaults are 500 Hz
dynamics, 100 Hz state control, `dt=0.002 s`, five dynamics steps per command, and `0.01 s` per
control interval. The smoke horizon is 20 (`0.2 s`); the full Day-1 horizon is 200 (`2.0 s`).

The reference acceleration is used by `state2attitude` in the force target. Roll and pitch rate
setpoints are zero and the supplied yaw rate is currently not consumed by the controller.

## Differentiated parameters

Only four unconstrained scalar `float32` leaves are passed to `value_and_grad`:

| Variable | Physical value at the checked-in default | Logistic range | Unit |
|---|---:|---:|---|
| `kp_xy` | 0.4 | `[0.10, 1.20]` | N/m |
| `kp_z` | 1.25 | `[0.30, 2.50]` | N/m |
| `kd_xy` | 0.2 | `[0.05, 0.80]` | N·s/m |
| `kd_z` | 0.5 | `[0.10, 1.20]` | N·s/m |

The smooth transform is `lower + (upper-lower) * sigmoid(raw)`. Shared x/y scalars preserve gain
symmetry. Initialization uses the inverse logistic transform with a protective `clip`; that clip is
outside the differentiated objective. Only the experiment-local `SimData` parameter dictionary is
replaced. Checked-in Crazyflow defaults are unchanged.

## Loss and gradient checks

The scalar loss is the sum of:

| Component | Definition/normalization | Weight |
|---|---|---:|
| position | mean squared 3-D position error / `(0.25 m)^2` | 1.0 |
| velocity | mean squared 3-D velocity error / `(1 m/s)^2` | 0.10 |
| effort | squared commanded RPM deviation from hover, normalized by hover RPM | `1e-3` |
| smoothness | squared time difference of normalized commanded RPM | `1e-3` |
| terminal | final position MSE / `(0.25 m)^2` | 0.10 |
| altitude | squared softplus margin penalty; margin `0.15 m`, softness `0.05 m` | 0.05 |

Auxiliary diagnostics report RMSE, maximum position error, saturation/gate/floor fractions, and
nonfinite state fraction. Boolean diagnostics do not contribute to the scalar objective.

The runnable validation requires finite loss and gradient leaves, gradient norm in
`(1e-8, 1e4)`, deterministic replay, eager/JIT agreement within `1e-5`, three central directional
derivatives with maximum relative error at most `5e-2`, and a decrease after a normalized
`1e-2` negative-gradient step. All checks passed in the 2026-07-24 audit.

## Non-smooth and piecewise operations

- Trajectory phase construction uses `clip` and `where`; its quintic construction makes phase,
  rate, and acceleration continuous through the ramp boundaries.
- Position and attitude integral errors use `clip`; saturated directions have zero gradient.
- Torque PWM, mixed PWM, and per-motor thrust use `clip`.
- Nonpositive collective thrust uses `where` to gate torque, and zero collective force gates motor
  forces. Contrary to one pre-existing prose statement, the inspected implementation does not
  explicitly reset the integrator leaves at this gate.
- Controller cadence is selected with boolean masks and `where`; cadence/mode values are static,
  not optimization variables.
- Rotor spin-up versus spin-down dynamics use `where` at commanded RPM equal to actual RPM.
- Quaternion integration uses a tiny-rotation `where` guard.
- Floor handling uses boolean gates to clamp `z` to `-0.001 m` and set velocity to zero below the
  floor, removing useful gradients in the clamped region.
- `Sim.reset` is a separate masked PyTree replacement and is not called inside the audited
  tracking objective.

## Implemented scope

The actual Day-1 implementation consists of:

- `crazyflow/trajectory.py`
- `crazyflow/control/mellinger/tracking.py`
- `crazyflow/control/mellinger/__init__.py`
- `examples/jax/mellinger_tracking.py`
- `tests/unit/test_mellinger_tracking.py`
- `docs/user-guide/mellinger-gradient-rollout.md`

The audit added only research/setup documentation, targeted ignore rules, and new generated audit
artifacts. It did not modify these six files.

Day 2 additionally implements:

- per-world loss/diagnostic reductions and numeric aggregation in
  `crazyflow/control/mellinger/tracking.py`;
- exports in `crazyflow/control/mellinger/__init__.py`;
- the deterministic CLI `examples/jax/mellinger_batch_diagnostics.py`;
- 14 focused relation tests in `tests/unit/test_mellinger_batch_diagnostics.py`;
- three H20/H200 evidence directories with twelve generated files.

The Day-1 example and Day-1 tests remain unchanged.

## Day-2 numerical result

| Horizon | Figure-8/train | Circle/validation | Combined |
|---:|---:|---:|---:|
| H20 | `0.0022911166306585073` | `0.0011237590806558728` | `0.001707437913864851` |
| H200 | `0.07310396432876587` | `0.024092847481369972` | `0.048598404973745346` |

H200 position RMSE was `0.0617200881 m` for Figure-8 and `0.0363599733 m` for circle; the combined
RMSE was `0.0506528243 m`. Maximum errors were `0.0919900164 m`, `0.0480792448 m`, and
`0.0919900164 m`, respectively. Motor saturation, nonpositive-thrust gate, floor clip, and
nonfinite fractions were all zero.

H200 raw-space gradients `(kp_xy, kp_z, kd_xy, kd_z)` were:

- train: `(0.02074276097, -0.01074000355, -0.05069825426, -0.00476809219)`, norm
  `0.05602372065`;
- validation: `(0.00048196566, -0.00904451683, -0.01141667459, -0.00254482566)`, norm
  `0.01479365304`;
- combined: `(0.01061235834, -0.00989225972, -0.03105746023, -0.00365645951)`, norm
  `0.03447338939`.

Train/validation gradient cosine was `0.8422763944`. If either exact L2 norm were zero, the JSON
would report a null cosine with status `undefined_zero_norm`.

The maximum H200 relative directional errors were `2.2386997e-4` for train and `4.4142021e-4` for
combined. Normalized raw-space negative-gradient steps of length `1e-2` changed train loss from
`0.07310396433` to `0.07254575938` and combined loss from `0.04859840497` to `0.04825476557`.

Batch per-case loss differed from separate `N=1` execution by at most `4.4703484e-8`. Combined
loss minus the separate-case mean was `-4.2840838e-8`; maximum leafwise gradient difference was
`7.2643161e-8`. Reversing world order changed neither combined loss nor gradient. Separate H200
losses and all four gradient leaves reproduced the Day-1 evidence within `rtol=1e-5`,
`atol=1e-7`.

The controlled H200 physical-gain central sensitivities per gain unit were:

| Gain | Train | Validation | Combined |
|---|---:|---:|---:|
| `kp_xy` | `0.0951753` | `0.00221157` | `0.0486935` |
| `kp_z` | `-0.0201236` | `-0.0169485` | `-0.0185361` |
| `kd_xy` | `-0.426109` | `-0.0972732` | `-0.261691` |
| `kd_z` | `-0.0187492` | `-0.0100690` | `-0.0144091` |

These are local one-gain-at-a-time diagnostics, not causal estimates or optimization updates.

H200 synchronized profiling on JAX `0.10.1`, CPU, gave medians of `0.006383277 s` for cached batch
forward, `0.151880844 s` for cached batch forward/backward, `0.011121430 s` for sequential two-case
forward, and `0.274278655 s` for sequential two-case forward/backward. Sequential/batch median
ratios were `1.74227595` and `1.80588050`. These observations are not a speedup claim or acceptance
criterion. Compile plus first batch forward/backward took `13.463255712 s`.

## Historical Day-2 acceptance criteria

- Focused test file passes.
- Loss and all four gradient leaves are finite and the gradient is nontrivial.
- The JAXPR contains `scan`.
- Analytic trajectory derivatives match autodiff.
- Directional finite differences agree with autodiff.
- JIT/eager results agree and repeated fixed-seed runs reproduce the scientific payload.
- JSON records and PNG plots are produced at documented paths.
- Relevant Ruff lint and format checks pass.
- Existing controller/dynamics defaults and all Day-1 scientific results remain unchanged.
- A real `N=2`, `M=1` batch contains distinct Figure-8/train and circle/validation worlds.
- Train/validation masks are disjoint, nonempty, complete, and implemented numerically.
- Batch loss/gradient match separate cases, and case order does not change the combined objective.
- Train and combined directional checks and local descent checks pass.
- Physical ±10% perturbations stay within the existing logistic bounds.
- Profiling times are synchronized, positive, and finite.
- Two H20 runs have equal scientific JSON payloads after excluding only named profiling-time
  blocks and have byte-identical plots.
- H200 reproduces the Day-1 per-case losses and gradients and reports all validations true.

## Limits identified through Day 2

- Validation is CPU-only, with two worlds and one drone per world, for two 2-second trajectories.
- Through Day 2, no noise, disturbances, broad parameter sweep, optimization loop, hardware flight,
  or full physical validation had been performed. Sprint 3 subsequently added the fixed
  train-only Adam loop documented below; the remaining exclusions still apply.
- The controller config uses mass `0.029 kg`; the first-principles `cf2x_L250` model uses
  `0.0319 kg`. The audit intentionally preserves this default mismatch.
- Passing runs did not exercise motor saturation, the nonpositive-thrust gate, or floor clipping.
- Reverse-mode differentiation through longer scans may retain substantial residual memory; Day 2
  did not measure peak memory or introduce rematerialization/checkpointing.
- Existing controller prose says some acceleration/rate fields are ignored, but the current
  `state2attitude` implementation consumes acceleration. Body-rate setpoints remain ignored.
- Missing optional `warp`/`mujoco_warp` imports emit warnings in this CPU environment but did not
  affect the verified path.

## Day-3 optimization result

Optax 0.2.8 was already importable and exactly resolved in `pixi.lock`; no dependency or lockfile
changed. The H200 run used `optax.adam`, learning rate `1e-2`, 50 updates, seed `20260724`, and the
unchanged logistic gain bounds and loss. Figure-8/world 0 was the only differentiated and updated
objective. Circle/world 1 and the combined mean were evaluation-only, including during checkpoint
selection.

Step 0 reproduced the Day-2 train baseline within `rtol=1e-5`, `atol=1e-7`. The minimum train loss
occurred at the final step 50:

| State | Step | Train | Validation | Combined |
|---|---:|---:|---:|---:|
| Initial | 0 | `0.0731039271` | `0.0240929127` | `0.0485984199` |
| Final | 50 | `0.0437690690` | `0.0157986116` | `0.0297838412` |
| Bestes getestetes Train-Checkpoint des festgelegten H200-Laufs | 50 | `0.0437690690` | `0.0157986116` | `0.0297838412` |

Train changed by `-0.0293348581` (`-40.1276%`), validation by `-0.0082943011`
(`-34.4263%`), and combined by `-0.0188145787` (`-38.7144%`). Validation improvement is an
observed held-out result, not an optimization condition.

Selected physical gains are `kp_xy=0.3156390190`, `kp_z=1.5052711964`,
`kd_xy=0.2637814879`, and `kd_z=0.6184355021`. Corresponding raw values are
`(-1.4112596512, 0.1919898838, -0.9195872545, -0.1149065346)` in
`(kp_xy, kp_z, kd_xy, kd_z)` order. All remain strictly inside the unchanged bounds.

At the selected point, raw-space gradient norms were `0.0312928669` train,
`0.0083939498` validation, and `0.0193920601` combined. Train/validation cosine was
`0.8651632667`. Maximum relative directional errors were `0.0023284324` train and
`0.0015936176` combined. A normalized negative train-gradient step of length `1e-2` changed
train loss from `0.0437690727` to `0.0434579439`. Batch/separate, world-order, and JIT/eager
checks all passed. Saturation, nonpositive-thrust gate, floor, and nonfinite fractions were zero.

## Day-3 robustness and profiling

The 32-cell forward matrix crossed baseline and the bestes getestetes Train-Checkpoint des
festgelegten H200-Laufs, H20/H100/H200/H400,
Figure-8/circle, and the audited/matched mass conditions. Every cell was finite; no cell activated
saturation, gate, or floor diagnostics.

| Parameter set | Matrix mean | Matrix maximum | Worst cell |
|---|---:|---:|---|
| Baseline | `0.0319434039` | `0.0791174546` | H400 Figure-8, audited mismatch |
| Bestes getestetes Train-Checkpoint des festgelegten H200-Laufs | `0.0198831878` | `0.0506328344` | H100 Figure-8, audited mismatch |

Audited-mismatch means were `0.0373575335` baseline and `0.0234195167` for the selected
simulation-optimized gain candidate. Matched-mass means were `0.0265292743` and `0.0163468588`;
matched minus mismatch was `-0.0108282591` baseline and `-0.0070726578` for that candidate. At
H200, the candidate changes versus baseline were
`-40.1275%` Figure-8 and `-34.4264%` circle under mismatch, and `-41.0682%` and `-38.3999%`
under matched mass. These directions were not acceptance criteria.

H400 completed. Cached forward medians for H20/H100/H200/H400 were
`0.002343864/0.004290380/0.006714594/0.013119240 s`; cached forward/backward medians were
`0.021593010/0.084925729/0.175491063/0.328635742 s`. The H400/H200 forward/backward ratio was
`1.8726636923`, below the factor-10 scaling indicator. Compile plus first optimization step was
`15.743286041 s`, cached Optax-step median `0.153715902 s`, and the 50-update loop
`32.805345890 s`. These are synchronized observations, not speedup or peak-memory claims.

## Current limits and supervisor gate

- The evidence covers two analytic trajectories, four horizons, and two controller-mass
  conditions on CPU; it is not broad generalization.
- The fixed seed is a reproducibility input, not stochastic robustness evidence, because no noise
  or disturbance model was introduced.
- No hardware, firmware, native bridge, BetaFlight, TinyMPC, sim-to-real, or physical-flight
  validation was added.
- H400 completed without rematerialization, but reverse-mode memory at longer horizons remains an
  open risk; peak memory was not measured.
- The selected gains are the „bestes getestetes Train-Checkpoint des festgelegten H200-Laufs“ and
  a „simulationsoptimierter Gain-Kandidat“ only. They are not a hardware release artifact.
  Checked-in controller and dynamics defaults remain unchanged.
- No sensor-noise, delay, disturbance, battery, aerodynamics-variation, estimator, firmware,
  hardware, or sim-to-real validation exists. All tested runs used perfect simulation states.
- No documented upload, arm, kill, limit, or staged flight-test process exists.
- BetaFlight and TinyMPC were neither integrated nor compared. Peak memory was not measured.
- The two controller-mass conditions are a controlled two-point comparison, not domain
  randomization.
- An artificial noise or domain-randomization model without flight-log/Mocap evaluation would not
  currently be empirically calibrated.
- Further scientific implementation is frozen until supervisors decide the binding bachelor
  scope, target controller, parameter/unit mapping, and hardware path. See
  [`SUPERVISOR_MEETING_BRIEF.md`](SUPERVISOR_MEETING_BRIEF.md).

## Sprint-5 domain-randomized research infrastructure

The final current architecture is intentionally different from the historical two-world
train/validation diagnostic above:

```text
train Sim/pure step_fn         validation Sim/pure step_fn       explicit test-only builder
  resampled episode batches     fixed manifest episodes            frozen test manifest
             \                    /
              shared named gain vector
                       |
              validation-only selection
                       |
               atomic step checkpoint
```

- `TrainingPipelines` contains only `train` and `validation`; it has no test field.
- `build_test_pipeline(...)` is a separate explicit API and is never called by the training
  runner.
- Split, episode, world, and component namespaces are derived with repeated
  `jax.random.fold_in`. Trajectory, mass, delay, and wrench receive separate component keys.
- Every training update realizes a new episode batch. Validation consumes the versioned fixed
  manifest. The test manifest is schema-checked and its ID recorded, but its episode records are
  not turned into a batch by the training runner.
- One raw gain vector is transformed and applied to every world. There is no world axis on the
  gain vector.
- True dynamics mass is sampled once per world/episode in `[m_nominal-0.0002,
  m_nominal+0.0002] kg`; the controller mass remains unchanged. The current audited defaults are
  approximately `0.0319 kg` dynamics and `0.029 kg` controller.
- Action delay is an integer number of control intervals, with `output[t] = input[max(t-d,0)]`.
  Zero delay is exactly identity and initial filling uses the first command.
- External force `[N]` and torque `[N m]` are stationary AR(1) sequences applied through
  `SimState.force` and `SimState.torque`; both input channels use world-frame coordinates. These
  heuristic models are disabled in `workstation.json` because no empirical calibration exists.
- Random trajectories are three-axis Fourier sums multiplied by a seventh-order start/end
  envelope. Position, velocity, acceleration, and jerk are analytic. Finite host-side rejection
  checks workspace, height, speed, acceleration, jerk, yaw rate, tilt, and specific force with a
  hard `max_attempts` bound.

The gain registry contains fourteen named grouped candidates. Stage 1 exposes `kp_xy`, `kp_z`,
`kd_xy`, and `kd_z`. Stage 2 adds `ki_xy` and `ki_z`. Stage 3 adds `kR_xy`, `kw_xy`, and
`kd_omega_xy`. Stage 4 registers `kR_z`, `kw_z`, and `ki_m_z`, but requires yaw excitation and a
yaw/attitude metric before scientific use. `ki_m_xy` is deferred because its default is zero and
no evidenced nonzero initialization/bounds exist. `kd_omega_z` is excluded because the executed
controller explicitly sets the yaw derivative error to zero. Old four-gain bounds are preserved;
new positive bounds are configurable 0.5×–2× engineering bounds, not stability guarantees.

The technical H20 smoke at implementation HEAD `c4492b1` used two train and two validation
worlds, one Stage-1 Adam update, and no test evaluation. Train loss was `0.0005378892`, validation
loss `0.0005391469`, and the gradient L2 norm `0.0001059676`; all were finite. External walltime
was `17.11 s`, Peak-RSS `1,116,596 KiB`, and swaps `0`. The small trajectory amplitude makes these
numbers a data-flow proof only, not evidence of useful optimization or generalization.

Fresh-process H20 gradient benchmarks for 1/2/4 worlds measured external walltimes
`13.35/14.87/15.06 s`, Peak-RSS `1,024,200/1,036,532/1,045,444 KiB`, compile times
`6.396/7.140/7.218 s`, and steady medians `0.00944/0.00886/0.02762 s`. These do not justify a
blind extrapolation to the prepared `16 worlds × H400 × 200 updates` workstation configuration.
That main run, seed study, ablation matrix, and true test evaluation remain unexecuted.

## Sprint-6 checkpoint/resume and Workstation readiness

The runner now supports a planned process budget through `--max-updates-this-process`. A paused
process writes the last completed update, complete metric history, selection state, Optax state,
config/registry/source/runtime/Validation fingerprints, and resume lineage. Checkpoint arrays
retain dtype and shape. Checkpoints and JSONL metrics use same-directory atomic replacement with
file and directory `fsync`; nonfinite JSON is rejected. Run targets must be absent or empty, and a
resume target must differ from the checkpoint directory.

The training runner no longer loads the frozen Test manifest. It records only the configured
opaque reference. Validation is still loaded, fingerprinted by content, built, and used for
selection. Initial source audit and the pre-existing manifest unit test may read the static Test
JSON, but no Sprint-6 acceptance process opens it, no Test batch is built, and no Test metric or
decision exists.

At executable-source HEAD `182d0cc9068c4920ac25146ff0016260f802e5dd`, fresh H20 processes
ran two updates uninterrupted and one update followed by a new-process resume for the second
update. Exact serialized comparison at `rtol=0`, `atol=0` passed for all required fields. Final raw
Stage-1 gains were
`[-0.9791043401, -0.2724381685, -1.3882366419, -0.5576145649]`; transformed gains were
`kp_xy=0.4003765285`, `kp_z=1.2510789633`, `kd_xy=0.1997670680`, and
`kd_z=0.5005095601`. All raw, selected, transformed, and Optax-leaf maximum absolute differences
were zero. Step and episode index were both 2, selected step was 2, and the full two-step
Train/Validation history matched.

Final fresh-process timings were 19.13 s / 1,151,416 KiB for uninterrupted execution,
14.53 s / 1,051,448 KiB for planned pause, and 14.43 s / 1,062,300 KiB for resume; all had zero
swaps and exit status 0.

The allowed scaling sequence then stopped at its first gate. H100 × one world, mass-only
randomization, root seed `20260731`, CPU backend, and unchanged Workstation trajectory settings
failed in deterministic episode construction before JIT, compilation, first execution, or steady
samples. The final attempt took 4.41 s, peaked at 591,776 KiB, used no swap, and exited 1. Its JSON
records `technical_gate.passed=false`. This is a configuration/readiness incompatibility, not a
resource-limit result, and it must not be repaired by silently changing the scientific trajectory
distribution.

**Readiness decision: NO-GO.** The stack is technically capable of exact, deterministic resume,
but it is not ready for incremental Workstation pilots with the currently specified H100 entry
gate. Consequently H100 × 4, H200 × 4 locally and H100 × 4, H200 × 8, H400 × 16 on the
Workstation remain unexecuted; the H400 × 16 × 200 main optimization and Test evaluation remain
strictly forbidden.

## Sprint-7 horizon-consistent trajectory readiness

The Sprint-6 NO-GO remains historically correct for `legacy_normalized_time_v1`. Sprint 7 adds an
opt-in distribution rather than redefining it. `fixed_support_prefix_v2` uses a configured
physical support and minimum duration; the Sprint-7 pilot uses 4.0 s support and accepts only
1.0..4.0 s returned prefixes. The complete parent is validated before slicing, so acceptance and
attempt selection do not depend on the requested H100/H200/H400 prefix.

The construction-only CLI mode records every attempted candidate's signed constraint margins,
extremal sample/time, kinematic/diversity/spectral statistics, array digests, seed-suite hash,
source/config fingerprints, raw-file hash, and timing. It builds no `Sim`, calls no `jax.jit`, and
does not run Train, Validation, or Test. Benchmark failures now distinguish config, simulator
pipeline, trajectory, episode realization, objective preparation, compilation, first execution,
steady execution, and technical-gate phases.

The current Research pipeline is explicitly CPU-bound in code (`device="cpu"`) and the local
environment exposes only `cpu:0`. Crazyflow's general `Sim`/MJX interface can select a GPU, but that
path has not been enabled or tested here. Any later GO remains CPU-only until a separate GPU scope
is implemented and validated.

## Sprint-8 bounded manual pilot preparation

`artifacts/day8-night-pilot/pilot_config.json` is the sole released pilot configuration. It fixes
H100, four worlds for both Train and existing Validation, seed `20260731`, 50 updates at learning
rate `0.001`, Stage 1, `fixed_support_prefix_v2`, and mass-only randomization. It changes no loss,
bounds, manifest, controller assumption, Crazyflow default, dependency, or executable source.
Action Delay and Wrench are disabled; sensor noise and UKF do not exist in this path and remain
off. The config fingerprint is
`6d0465b5776fe1f09b0f5b1ab8777d02b9b3bd01386d8f00cc6209fb97d2efaa`.

The launcher enforces the research branch, tracked/unmodified pilot config, clean executable
source scope, unique absent run roots, a 21,600-second timeout, and separate full logs. Every
completed runner update is a same-directory atomic, checksummed restart boundary. SIGTERM may
discard only the current incomplete update; Resume uses the last complete checkpoint and a fresh
directory. This does not establish long-run success until the user actually executes and reports
the pilot. It also does not authorize H200/H400, another configuration, more than 50 updates,
seeds, Test, a main run, GPU, firmware, hardware, physical-safety, or Sim2Real claims.

## Sprint-9 evaluation and long-pilot gate

The completed Sprint-8 run is now immutable tracked evidence. Commit
`d1d0cc0a400b8af81da9f435226a6b5797dc2c97` preserves its 61 unchanged raw files plus the
complete run hash index and manifest update. The raw run contains 50 valid sequential
checkpoints, 100 finite Train/Validation rows, a successful summary, complete provenance, and no
Test access or metric.

The reproducible Sprint-9 analysis confirms best Train step 31 and best Validation/selected step
50. Train loss changed from `0.0102038635` to `0.0090798726`, Validation from `0.0100101354` to
`0.0095826006`, and Gradient-L2 from `0.0062514464` to `0.0055943080`. Validation decreased at
every transition. Train has a post-hoc oscillation indicator, but neither split has a plateau or
divergence indicator. The minimum normalized distance to a gain bound is `0.2001600`; no
technical saturation or nonfinite state occurred. This remains exploratory single-seed
simulation evidence.

**Sprint-9 long-pilot decision: NO-GO.** The valid prepared config changes only the run ID and
update count to 5,000; its canonical fingerprint is
`23ee0f46f581b6453ec51915a8cd5410e7cd77d48c9d6a07caadee74a98c5fb5`. The existing runner has
no configurable checkpoint interval and writes a complete-history atomic checkpoint after every
update. The observed 50-step size fit projects 5,000 checkpoints totaling `41,408,829,822` bytes
and 5,011 run files. All three measured interval-growth fits project `8,032.99` to `13,438.55`
seconds, exceeding the hard 7,200-second limit. The config, launcher, inspector, tests, disk, and
observed RSS gates pass, but exact completion within the wall limit does not. No Sprint-9
training process, tmux session, run directory, Test evaluation, H200/H400 run, or seed loop was
started. A later explicitly authorized code-change sprint must decide and verify a sparse atomic
checkpoint strategy before reconsidering 5,000 updates.

## Sprint-10 sparse checkpoint release

Sprint 10 implements the authorized minimal sparse-checkpoint change at implementation commit
`a58b78861cd15e904f7883668a689f297524f4a1`. `OptimizerConfig.checkpoint_interval` is a strictly
positive integer with default 1. Configs that omit it keep their old per-update checkpoint
behavior and canonical fingerprints. An explicit interval writes complete atomic, checksummed
checkpoints at interval boundaries, the configured final update, and a planned process stop,
without double-writing coincident boundaries. Metrics are durably appended once per completed
update rather than rewriting all prior rows; checkpoints and the final summary retain complete
history for exact Resume.

The real H100×4, 210-update, interval-100 smoke exited 0 in 55.99 seconds, peaked at 1,309,284 KiB,
used no swap, and wrote exactly steps 100/200/210 at 335,925/667,310/700,428 bytes in
0.0360/0.0691/0.0636 seconds. All 420 ordered Train/Validation rows were finite and passed every
technical gate. A new process resumed step 200 through 210 and matched the uninterrupted final
checkpoint exactly at `rtol=0`, `atol=0`, including optimizer, gains, selection, history, metrics,
fingerprints, and counters. Neither process opened or evaluated Test.

**Sprint-10 decision: GO for exactly one 5,000-update detached CPU pilot.** The released config
keeps every Sprint-9 scientific field and changes only run ID and the technical checkpoint
interval from implicit 1 to explicit 100. Config fingerprint is
`859ca650fbe598c24172cba233b9ed15c7f51d23063b7dd6210847ead8910769`; executable-source
fingerprint is `6b3fed4a445b82b2a1354115edc982a15e0d93cd4aff91fc8f51475e60f01c5e`.
The conservative projection uses the slowest measured steady update for all 5,000 updates,
doubles measured checkpoint I/O cost, doubles startup, and adds fixed reserve: 2,289.55 seconds.
Expected artifacts are 437,666,785 bytes and 61 files; the doubled storage bound is 875,333,570
bytes, below 2 GiB. The launch remains limited to 7,200 seconds, 50 checkpoints, one seed, no
automatic Resume/extension, no second run, no H200/H400, and no Test.

## Sprint-11 completed long-pilot evaluation

The sole Sprint-10 long-pilot run at
`artifacts/day10-sparse-long-pilot/runs/20260801T081232Z-1345032536fd` completed successfully
without Resume or continuation. Read-only validation confirms launcher and external exit 0,
exactly 5,000/5,000 updates, checkpoints 100/200/.../5,000, 10,000 ordered nonduplicate
Train/Validation rows, valid internal payload checksums for all 50 checkpoints, finite metric and
checkpoint values, passing technical gates, consistent config/source/runtime/gain/Validation
fingerprints, and a closed Test boundary. There is one run directory only; the completed process
and tmux server are absent.

Train loss changed from `0.0102038635` to `0.0025606975` (-74.90%) with its sampled minimum
`0.0022851648` at step 4,931. Fixed Validation changed monotonically at all 4,999 transitions from
`0.0100101354` to its minimum `0.0025241498` at step 5,000, which is also the runner-selected
step. Gradient-L2 changed from `0.0062514464` to `0.0002818444` (-95.49%) and reached its minimum
at the same step 4,931 as Train. The mean Validation-minus-Train gap is `-0.0000293970`, the final
gap is `-0.0000365477`, and its sign varies because Train episodes are resampled while Validation
is fixed.

Early/middle/late 500-update Validation slopes are respectively `-6.19994e-6`, `-6.43868e-7`,
and `-1.84670e-7` loss/update. The late window still improves 3.53%; Gradient-L2 improves 16.28%
over the same endpoints. Thus improvement is slowing but the declared 1% plateau indicator is
false. Train and Gradient-L2 have delta-sign oscillation indicators, consistent with stochastic
Train episode resampling; fixed Validation has none. There is no divergence, nonfinite value,
motor/floor/zero-thrust technical saturation, or gain-bound saturation.

All 50 persisted interval-100 gain vectors are finite and inside their bounds. At step 5,000 the
physical vector is `kp_xy=0.3351839483`, `kp_z=2.3107547760`, `kd_xy=0.7023677230`, and
`kd_z=1.0894526243`. The final normalized nearest-bound margins are respectively 0.21380,
0.08602, 0.13018, and 0.10050. `kp_z` therefore merits monitoring under later replications but is
not saturated under the declared 0.01 criterion. Sparse checkpoints do not contain intermediate
per-update gain vectors; the analysis includes every persisted gain sample and does not infer the
unrecorded vectors.

The first 50 metric updates are byte-identical to Sprint 8 and the first 210 to the Sprint-10
smoke. Relative to Sprint-8 final values, the long-run final Train/Validation/Gradient-L2 values
are lower by 71.80%/73.66%/94.96%; relative to the smoke final they are lower by
70.07%/69.98%/94.31%. This establishes deterministic prefix continuity under the unchanged
single-seed semantics, not between-seed robustness.

External/internal walltime is 925.96/923.17 seconds, Peak-RSS is 2,201,956 KiB (2.100 GiB), swap
is zero, and total checkpoint-write time is 40.66 seconds. The 61 raw files total 437,505,563
bytes, including 423,588,483 checkpoint bytes. Actual walltime is 40.44% of the conservative
2,289.55-second projection; artifact bytes are 99.963% of the 437,666,785-byte expectation and
49.982% of the doubled bound.

**Sprint-11 recommendation: GO to a protocol-freeze step and then multiple independent H100
seeds.** Freeze the validation-based minimum/earliest-exact-tie selection rule and all H100
scientific semantics before predeclaring seeds and aggregation criteria. Only after that
replication should unchanged H200/H400 be considered. Test remains closed until every decision
is frozen for the final one-time protocol. This single seed is exploratory simulation evidence,
not a main study, uncertainty estimate, formal stability proof, hardware result, or Sim2Real
claim.

The raw run is integrity-indexed but remains untracked: 437.5 MB of growing-history checkpoint
JSON is disproportionate for ordinary Git, and the repository has no Git-LFS policy. Preserve it
unchanged and mirror it together with the 61-file SHA-256 index to backed-up institutional
research storage. The compact deterministic analysis and hash inventories are the versioned
repository evidence.

## Sprint-12 H100 replication protocol freeze

This section records the base freeze. Its external-persistence paragraphs are historical and
superseded only by the Sprint-12A amendment below; all scientific/statistical content remains
active.

Sprint 12 freezes the complete ten-seed H100 replication procedure before any confirmatory run.
The machine- and human-readable authority is
`artifacts/day12-h100-replication-protocol/`. The successful Sprint-10 run and its root seed
`20260731` remain exploratory planning evidence and are explicitly excluded from the ten-seed
panel and every confirmatory aggregate.

The ten new root seeds are, in immutable execution order: `1432116264`, `366692846`, `235438753`,
`1369406745`, `1081462774`, `1276309202`, `987511476`, `2125653507`, `31735934`, and
`986925065`. They are derived reproducibly from a versioned SHA-256 namespace plus their ordinal;
the first 31 digest bits are accepted after deterministic zero/collision/pilot-seed rejection.
Distinct roots drive the existing split/episode/world/component fold-in tree. Validation remains
the same fixed four-world manifest for every seed.

All ten generated configs change Sprint 10 only in `run_id` and `root_seed`. H100, four Train and
four Validation worlds, 5,000 updates, Stage-1 `kp_xy/kp_z/kd_xy/kd_z`, Adam `0.001`, bounds,
loss, fixed-support trajectory, mass-only ±0.0002-kg randomization, controller-mass assumption,
checkpoint interval 100, CPU backend, manifests, and disabled delay/wrench/noise/UKF remain
unchanged. Required source/runtime/gain/Validation fingerprints are pinned. Test stays an opaque
unopened reference.

The exact selection population is every completed post-update state 1 through 5,000. The runner
updates selection only for strict `validation_loss < best_validation`; therefore it chooses the
global fixed-Validation minimum and retains the earliest exact tie. Step 0 is not eligible. The
older phrase “earliest checkpoint” is now clarified: sparse checkpoint files occur every 100
updates, but selection is evaluated every update and the interval checkpoint stores selected
step and raw gains even for an intermediate step.

The primary per-seed endpoint is selected Validation loss. The panel reports all ten values,
mean, `ddof=1` standard deviation, a two-sided 95% Student-t interval with nine degrees of freedom
and fixed critical value `2.2621571627409915`, plus median and linear-method Q1/Q3. Relative
Validation improvement from step 1 is analyzed identically. Secondary runner-emitted losses,
gradients, gains, bound margins, tracking/technical diagnostics and resources are descriptive;
there are no secondary p-values, multiplicity claim, imputation, replacement seeds, or
post-result exclusions.

Technical execution is strictly sequential in manifest order. At most one full restart from
update 0 is allowed for a documented exogenous host/storage interruption, with the same seed and
byte-identical config; Resume is forbidden. Scientific/technical failures are not retried. A
second or nonretryable failure invalidates the seed, stops further starts, and makes the panel
NO-GO. Every attempt remains immutable, checksummed evidence.

H100 permits preparation of a later H200/H400 protocol only if all ten frozen GO conditions in
`analysis_plan.json` pass. They require 10/10 valid seeds, complete integrity/Test/persistence
gates, positive improvement for all seeds, at least 50% improvement for at least 8/10, a lower
95%-CI bound of at least 50% for mean improvement, at most 25% relative CI half-width for the
primary loss, no more than 10% final-versus-selected Validation degradation, all selected gain
margins at least 0.01, and no protocol drift. Any failed condition is NO-GO; it does not authorize
tuning, added seeds, H200/H400, or Test.

Planning uses 925.96 seconds and 437,505,563 bytes per successful seed, with conservative values
2,289.55 seconds and 875,333,570 bytes. Ten runs project to 2:34:19.60 observed-rate or 6:21:35.45
conservative compute time and 4,375,055,630 bytes (about 4.4 GB) expected raw data. Sequential
Peak-RSS remains approximately 2,201,956 KiB rather than multiplying by ten; each process retains
the 7,200-second and 12-GiB caps. At least 10 GB external capacity must be available.

After each attempt, raw data and a complete SHA-256 index must be copied byte-for-byte to approved
backed-up external research storage and fully verified there before the next seed. Ordinary Git
does not store raw runs. Sprint 12 started no run, seed, H200/H400, Resume, Test action, or tmux
session.

**Historical next step at the Sprint-12 freeze:** Sprint 13 could perform a new exact preflight
and implement/review the single-seed launcher. Execution still required explicit authorization
and had to proceed one seed at a time with persistence between seeds; the Sprint-12 release gate
itself was `NO-GO_RUNS_IN_SPRINT12`.

## Sprint-12A persistence amendment before Seed 01

At `2026-08-01T12:51:18Z`, before any confirmatory seed or result, the user replaced only the
Sprint-12 external-persistence requirement. The versioned authority is
`artifacts/day12-h100-replication-protocol/amendments/2026-08-01-local-ssd-persistence-v1.json`
with a matching human-readable amendment. Base commit
`73868cfd84d25bec0c6612b31ee34f4e67acb307` preserves the original rule and rationale.

Every successful or failed replication attempt now remains complete and unchanged under the
repository artifact area on the user-declared internal 2-TB SSD. Raw runs remain outside
ordinary Git. After each attempt, a full `RUN_SHA256SUMS` is created over all other run files and
immediately checked locally with `sha256sum -c`; before every later seed, the previous seed's
complete index is checked again. Any missing file or mismatch blocks all later seeds. During the
series, runs must not be deleted, moved, renamed, compressed, changed, or overwritten. The
Sprint-10 pilot also remains unchanged.

Seed 01 requires at least 20,000,000,000 local free bytes. Each later seed requires
10,000,000,000 bytes plus 437,505,563 bytes for every remaining seed including the seed about to
start. Amendment preflight measured 1,006,592,950,272 free bytes, so the Seed-01 capacity gate has
large headroom. External capacity, copy, destination verification, and storage receipt are no
longer execution gates.

This change affects storage location and fault tolerance only. The seed manifest and all ten
config files are byte-identical to the base protocol. H100 semantics, run order, source/runtime/
gain/Validation fingerprints, optimizer, gains, bounds, worlds, loss, selection and tie-breaking,
statistical endpoints and uncertainty, scientific GO/NO-GO thresholds, failure classification,
and locked Test boundary remain unchanged. It is not result-dependent.

The user knowingly accepts that complete failure or loss of the single internal SSD can destroy
all raw replication evidence. SHA-256 detects missing or modified data but is not an independent
backup; this is an explicit scientific and operational limitation.

Sprint 12A authorized zero runs and left the Test split unopened. **Historical next step at the
Sprint-12A checkpoint:** the amended protocol was GO for a separately authorized Sprint 13 to
perform a fresh exact preflight and then launch exactly Seed 01. It does not authorize a run in
Sprint 12A, another seed, a loop, Resume, H200/H400, or Test.

## Stage-2 Tranche 1: term-resolved loss diagnostics

The approved first Stage-2 tranche adds a JAX-compatible decomposition of the unchanged tracking
loss. For each of the six existing terms it exposes raw value, normalization divisor, normalized
value, weight, weighted contribution, and a Jacobian column for each of the four current
Stage-1 variables `kp_xy`, `kp_z`, `kd_xy`, and `kd_z`. Those Jacobians are explicitly labelled as
derivatives with respect to the dimensionless internal variables that feed the sigmoid-bounded
transform; they are not derivatives per physical gain unit.

The deterministic CLI evaluates only the existing H20 Figure-8/train and circle/validation case
on `cf2x_L250`, with seed `20260724`, 500-Hz simulation, 100-Hz control, default gains, and zero
optimizer updates. Weighted contributions reconstruct the existing total within `rtol=1e-6` and
`atol=1e-8`. The observed totals are `0.0022911166306585073` for Train and
`0.0011237586149945855` for Validation. Motor saturation, zero-thrust gate, floor clip, and
nonfinite-state fractions are all zero for both cases.

The machine-readable record and two JSON-driven plots are in
`artifacts/day20-stage2-loss-diagnostics/`. All focused tracking, diagnostics, and gain-
optimization regressions pass (`26 passed`). This is infrastructure-smoke evidence only. It does
not establish convergence, generalization, robustness, a better gain set, controller superiority,
firmware equivalence, hardware safety, or transfer to `cf21B_500`. It starts no new H100/H200/H400
study, does not access the closed Test split, and does not read or alter protected primary data.

The next scientific tranche remains separately gated: additional gain groups must be assessed in
controlled groups, tied-vs-split x/y gains require an explicit ablation, and trajectory severity,
plateau/convergence metrics, continuation points, platform parameters, candidate selection, and
firmware transfer each require their own bounded decision and Work Order.

## Friday evidence package: existing-evidence fallback

The Day-21 package collects presentation-ready evidence from the existing frozen ten-seed H100
panel and the accepted Day-20 diagnostics. It contains every Train and fixed-Validation history
row through update 5,000, all ten GNU process wall times, technical completion separately from
the accepted numerical variation, 50 checkpoint-selected gain records per seed, selected end
gains, four new H100 figures, and byte-identical copies of the two accepted Stage-2 figures.
Seed 05 at update 5,000 is labelled only **provisional visualization candidate**; it was selected
by the smallest already existing selected Validation loss and is not a final or flight candidate.
No gains were averaged.

The initially requested narrow default/candidate visualization did not meet the unchanged
zero-motor-saturation acceptance gate. The pre-authorized fallback therefore records
`WITHHELD_TECHNICAL_GATE` and deliberately contains no rollout arrays, metrics, replacement
values, `rollout_trajectory.png`, or `rollout_tracking_error.png`. No alternative candidate,
trajectory, platform, or threshold was substituted. This leaves the supervisor's trajectory-
before/after request openly unanswered while preserving the technical and claim boundary.

The package is under `artifacts/day21-friday-evidence/`. Its nine-entry `SHA256SUMS` covers the
other nine fallback files; two fresh generations from the committed generator were byte-identical.
The generator rechecked the tracked Day-19/Day-20 checksum indexes and the ten small live
`RUN_SHA256SUMS` identities but did not rehash the 4.38-GB H100 payload. Test remained unopened,
no training/optimization was started, and the protected Day-10/Day-13 inputs were not changed.
This package establishes presentation and provenance readiness for existing simulation evidence
only, not convergence, generalization, robustness, superiority, firmware/hardware validity,
flight safety, `cf21B_500` applicability, or Sim2Real transfer.

## Day-22 saturation-explicit diagnostic review candidate

The separately named Day-22 package exposes exactly the frozen Day-21 default-versus-Seed-05
`cf2x_L250` Figure-8 H100 rollout as a diagnostic. It does not rewrite or relabel Day 21. The
normal zero-motor-saturation gate remains unchanged and still fails: Default has `0/400`
saturated motor samples (`0.0`), while Seed 05 has `11/400`
(`0.027499999850988388`). Seed 05 therefore carries the exact status
`DIAGNOSTIC_ONLY_SATURATION_PRESENT`; it is not accepted and is not flight ready.

All eleven decisions are upper-bound saturation on motor 0 at zero-based state indices 48–58.
They form one contiguous interval with boundaries `0.48–0.59 s`, eleven samples, and duration
`0.11 s`; the corresponding recorded state times are `0.49–0.59 s`. Motors 1–3 and all Default
motors have zero lower- and upper-bound decisions. The complete 100×4 commanded-RPM arrays,
lower/upper/combined bool arrays, per-motor counts, indices, state times, intervals and limits are
in `artifacts/day22-saturation-diagnostic/saturation_diagnostic.json`. The reconstructed
float32 fraction equals the unchanged tracking-loss metric exactly.

The two JSON-driven figures visibly mark the same interval and state samples and carry the
unmissable subtitle `DIAGNOSTIC ONLY — SATURATION PRESENT — NOT FLIGHT READY`. Descriptive RMSE
is `0.05751212686300278 m` for Default and `0.030816158279776573 m` for Seed 05 in this one case,
but the latter value is inseparable from the failed saturation gate and establishes no
superiority or candidate acceptance.

The package is a worker review candidate pending independent Lead acceptance. It uses five
checksum-verified accepted Day-21 payload identities and reads neither Day-10 nor Day-13 primary
data nor Test. It starts no optimization or training and changes no platform default. Two fresh
generations are byte-identical to each other and the final candidate package; all five payload
checksums pass. The scientific boundary remains simulation diagnostic only: no robustness,
convergence, Test, `cf21B_500`, firmware, hardware, flight, safety, or Sim2Real claim.

The Day-22 F0.1 regression loads the accepted Day-21 candidate, executes the real frozen trace
builder, verifies both historical saturation fractions, and requires the unchanged
`build_rollout` gate to raise without an output path. The full normal and fallback F0.1 CLIs were
deliberately not executed because both would reread protected Day-13 inputs. Source diff plus the
existing F0.1 tests establish that CLI, fallback branch and accepted Day-21 behavior are
unchanged.

## Day-23 vertical ki_z excitation and sensitivity review candidate

The Day-23 pilot evaluates only the Stage-2 registry member `ki_z`; it performs no optimizer
update. The frozen `cf2x_L250` case is a 5.0-s constant hold at 0.75 m with explicit Euler,
500-Hz dynamics, 100-Hz state control, default gains and six Lower/Default/Upper rollouts. The
matched reference uses controller and dynamics mass 0.0319 kg. The audited-mismatch case changes
only the experiment-local controller mass to 0.029 kg while the dynamics remain 0.0319 kg.

All six rollouts are finite, have zero motor saturation, zero floor/ground and zero-thrust
fractions, at least 0.01 normalized motor-bound margin and at least 10% margin to the Z-integrator
clip. The mismatch Default reaches `0.1340937316417694 m s` maximum absolute integral state,
versus `0.03675120696425438 m s` for the matched reference. The unchanged bounded Registry
transform maps raw Lower/Default/Upper to physical `ki_z` values approximately
0.04601988/0.05000000/0.05432435 N/(m s); all other gains remain identical.

At Default, JAX gives `d loss_v1 / d raw_ki_z = -0.0007489713025279343` in the mismatch case and
`-0.000056265325838467106` in the matched case. Central finite differences are respectively
`-0.0007464457303285599` and `-0.00005596131086349487`; both satisfy the predeclared scale-aware
AD/FD rule. Every predeclared excitation, nontrivial-sensitivity, bias-contrast and transform-
conditioning criterion passes, so the worker result is
`GO_FOR_FUTURE_KI_Z_OPTIMIZATION`.

This status is only a binary signal for considering a later, separately authorized `ki_z`
optimization Work Order. The package in `artifacts/day23-ki-z-sensitivity/` is a worker review
candidate pending independent Gradient-Lead acceptance. It is backup/next-step evidence only and
does not prove optimization, improvement, superiority, convergence, robustness, hardware
transfer or flight readiness. The Friday presentation does not depend on it. Day 10, Day 13,
Test, and the accepted Day-20/21/22 packages were not read or modified.

## Day-24 unclipped two-stage allocation diagnostic review candidate

The Day-24 diagnostic reproduces the exact accepted Day-22 Default/Seed-05 `cf2x_L250` Figure-8
case without changing controller, allocator, gains, bounds, thresholds, defaults, trajectory or
production source. A diagnostic-local nested scan executes the unchanged six-stage simulation
pipeline for five 500-Hz substeps per 100-Hz interval and stores only substep five. Its complete
production trace is array-identical to a fresh production replay, and production commands,
tracking arrays, tracking metrics and saturation decisions are array-identical to accepted Day 22.

Stage A has no torque-axis clipping in either variant. Default has zero motor clips. Seed 05 has
exactly eleven Stage-A upper motor clips, only motor 0 at indices 48–58 and interval
`[0.48,0.59) s`. Stage B creates no additional clip and no additional wrench distortion. The
maximum Seed-05 motor-0 request occurs at index 54: `74753.53125 PWM`, `0.13687989115715027 N`
and `24716.875` in the existing Crazyflow motor-speed convention. Its upper exceedance is
`9218.53125 PWM`, `0.016879891976714134 N` and `1501.0977783203125` respectively.

At that same sample, maximum absolute Stage-A wrench distortion is
`0.016879886388778687 N` collective, `0.0005491026677191257 N m` roll,
`0.0005491029005497694 N m` pitch and `0.0001240818528458476 N m` yaw. Every preclip quadratic
speed inverse is real and finite. The configured calibration
`[0,-5.382196214637237e-7,2.4582929831265485e-10]` is preserved alongside its runtime Float32
representation. Hover is `18967.771484375`, `0.8170207356628263` of the configured maximum speed,
and `0.6519561779191235` of maximum calibrated thrust; these are pure-simulation quantities only.

DATA-AUDIT-01 uses only the checksum-verified Day-19 derivative and analyzer source; the analyzer
was not executed. It confirms ten seeds with 5,000 Train, 5,000 Validation and zero Test rows each,
100,000 Train/Validation rows total, and per-seed/panel registered saturation maxima `0.0`. This
proves no stored-objective saturation registration only, not motor reserve or allocation
distortion. Protected Day-10/Day-13/Test data were not opened.

The five-file package in `artifacts/day24-unclipped-allocation/` is a worker review candidate.
Two fresh packages from generator/test commit `728e8b9f89e4f92c7fa4eefee9808a82904975c2` pass 4/4
checksums and are byte-identical to each other and the final package. The sole JSON-driven PNG was
inspected at 2240×1600 after its narrow figure-contract correction: it shows motor-0 unclipped/
clipped demand and the true limit, all four motors' Stage-A pre/post commands around indices
48–58, signed platform-normalized wrench distortion, and exact maximum/count/status/nonclaim text.
The unchanged zero-saturation gate remains
`FAIL_ZERO_MOTOR_SATURATION_GATE`; Seed 05 remains `DIAGNOSTIC_ONLY_SATURATION_PRESENT`, not an
accepted candidate and not flight ready. No Test, optimization, superiority, safety, firmware,
hardware, flight or Sim2Real claim is made.

## Day-25 cf21B_500 robust baseline foundation corrected review candidate

`WO-GR-G2-003` prospectively applies D-045 without changing the historical disposition of
`WO-GR-G2-001` or `WO-GR-G2-002`: those attempts remain `BLOCKED/WITHHELD` and their reviews
remain `REJECTED`. Integral-boundary contact is now a required measured baseline property for
both fixed mass variants, not a stop/pass criterion. It remains neither a controller pass nor a
candidate or flight-readiness signal. Every nonintegral scientific, numerical, technical and
protection gate remains hard.

The first independent review of result commit
`c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` was explicitly `NOT ACCEPTED`. D-047 authorizes
exactly one transparency-only correction cycle, `CORR-01`; it does not retroactively accept that
result. The correction rewrites the two worker commits from the same base, discloses the retained
trajectory component and measurement scope, adds required aggregates and machine-readable
invariance evidence, and stabilizes generator provenance. It changes no trajectory or result
number.

The foundation contains 16 deterministic parent episodes: two Train and two Validation episodes
for each of Soft, Nominal, Dynamic and Near-limit. References have 8-s parent support. Each
simulation is one continuous 6-s rollout from the parent start, without a reset, and only the
half-open 2-s window after the fixed class-specific 2/3/4-s warm-up is scored. Position and
analytic nonzero-yaw reference derivatives are smooth. After class-scaled Fourier construction
and before addition of the fixed center, every class retains the same positive-z component:
exact Float32 `0.06 m/s`, base displacement `0.06 * (time - 2 s)`, base velocity `0.06 m/s`,
zero base acceleration/jerk, and product derivatives through jerk under the same C3
smoothstep7 2-s entry / 4-s plateau / 2-s exit envelope. Absolute z results therefore apply to
trajectories containing this fixed climb; the climb itself is not experimentally isolated.

The fixed paired ablation changes only the experiment-local controller mass: repository mismatch
is 0.0393 kg controller versus 0.04338 kg dynamics; physical-value match uses 0.04338 kg for
both. It performs 32 production rollouts and no optimizer update.

All 32 objectives and gradients are finite. Full-carry replay, production/diagnostic identity,
mass-only PyTree isolation, deterministic split/class/seed contracts and report recomputation
pass. Finiteness, ground/floor and zero-thrust gates cover the full continuous 6-s rollout.
Integral contact, torque/motor/Stage-B clipping, tracking, unchanged Loss v1, reserve and wrench
statements cover only the half-open scored window. Every scored-window Stage-A torque-axis clip
count, Stage-A motor clip count and Stage-B additional clip count is zero, including the Dynamic
and Near-limit episodes. Thus every unchanged nonintegral gate passes.

The measured result is negative for physical-value matching in this fixed simulation and
controller conversion chain. Repository mismatch has mean Z-RMSE `0.08494756871345588 m` and
593 negative-z integral-boundary axis-samples; physical-value match has mean Z-RMSE
`0.11452380346939901 m` and 1,844 negative-z samples. Matched minus mismatch is therefore
`+0.029576234755943134 m` mean Z-RMSE and `+1,251` negative-z contacts. Positive-z contact is
zero in both variants. This is explicitly not improvement, a candidate pass, causal isolation,
physical-mass estimation, firmware transfer, hardware evidence or flight readiness.

The mechanism audit keeps three layers separate. Measured early-transient and scored-window
arrays report requested collective/target-thrust-equivalent quantities at the available
diagnostic boundary, realized wrench, z error, z integral, motors and reserve. Code-path inference
records that `state2attitude` uses
`mass * (setpoint_acc - gravity_vec) + feedback`, followed by the existing `mass_thrust=132000`
and nonlinear PWM-to-force conversion; this explains a path but does not isolate causality. The
transfer boundary treats controller `mass` as a feedforward parameter within this fixed chain,
not as a physical estimate or deployable firmware/flight value.

Fresh-process CPU measurements pass at 4, 16 and 32 worlds with no timeout, memory-guard or swap
failure and consistent objective/gradient values. Steady throughput is respectively
`5.341502820040405`, `17.768144107209054` and `29.869865657590918 worlds/s`; Max RSS is
`1.132907867`, `1.186634064` and `1.263717651 GiB`. The predeclared rule recommends 32 worlds
for the next CPU batch only. This is not a repository default and makes no GPU claim.

The ten-file package in `artifacts/day25-cf21b-robust-foundation/` is a corrected worker review
candidate, not an acceptance. It was generated on the stable first rewritten commit
`2288ab2fe3b55f585fb9b14a8fe9d87c07eb6c1a` from the unchanged checksum-pinned benchmark,
then reproduced twice into fresh `/tmp` destinations; both reproductions and the final package
are byte-identical. Its nine payload checksums pass, and the checksum-index SHA-256 is
`819d2ec725b5b8b045a6b6acfaae6443a6151df01d59d9d7c5a178d787b70437`.

The automatic D-047 comparator explicitly uses old result
`c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` and records
`PASS_PRE_CORRECTION_C4_INVARIANCE`. Old and corrected reports have the identical numeric
projection SHA-256
`fd1195f9d10c88a2d550090631f6b21d0d4f80fd47a2bbe09a52e7c9a0975a61`
(5,465,162 paths; 644,397,249 serialized bytes) and identical parent projection SHA-256
`bcb40e132f4b85e99f814f0a9db4d90244523b488697868ece8597d3654de82c`
(16 records; 18,037 serialized bytes). It also validates every retained pre-existing scalar leaf
outside the declared prose/provenance replacements, every old raw array, summary, pair delta,
ID/seed/attempt/window/mass/digest and the pinned benchmark bytes.
Only the permitted derived aggregates, metadata, corrected prose, stable generator provenance
and checksums differ.

Both figures were inspected at original resolution and visibly preserve the negative finding,
corrected scored-window labels and CPU-only claim boundary. A new, non-package 4/16/32
fresh-process run also passed the unchanged resource and replicated-objective/gradient gates.
Day-19 through Day-25 checksum gates pass. Protected Day-10/Day-13/Test data were not opened;
shared controller/default/dependency/lockfile paths and the live checkout were not changed.
Foundation completion is simulation evidence only and does not automatically authorize G3,
Loss v2, optimization, desaturation, transfer or flight work.

## Day-26 G3.4 parameter eligibility audit

The reviewed Day-26 package at
[`artifacts/day26-g3-identifiability-freeze-v2/`](../../artifacts/day26-g3-identifiability-freeze-v2/)
has status `COMPLETE_G3_4_PARAMETER_ELIGIBILITY_AUDIT`. It covers the unchanged registry of 24
ordered Train/Validation parents in six prospectively fixed batches of four, all seven declared
continuous parameters, full six-second carry, the unchanged Loss v1, and zero optimizer
initializations or updates. The six-by-four mode was selected prospectively by D-073 after the
local primary-24 OOM; it is not a runtime fallback and was not selected from host memory.

The parameter dispositions are:

| Parameter | G3.4 disposition | Binding reason or boundary |
|---|---|---|
| `kp_xy` | `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION` | Passed the objective-gradient, detectability, stability, conditioning and technical gates |
| `kp_z` | `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION` | Passed the objective-gradient, detectability, stability, conditioning and technical gates |
| `ki_z` | `GO_FOR_SINGLE_PARAMETER_OPTIMIZATION` | Passed individually; not needed in the smallest qualifying joint subset |
| `kd_xy` | `DIAGNOSTIC_ONLY` | Not individually GO-eligible |
| `kd_z` | `WITHHELD_TECHNICAL_ROBUSTNESS` | `NONFINITE_NEAR_LIMIT_EFFECT_ROLLOUT` |
| `mass` | `WITHHELD_TECHNICAL_ROBUSTNESS` | `NONFINITE_NEAR_LIMIT_EFFECT_ROLLOUT` |
| `mass_thrust` | `WITHHELD_TECHNICAL_ROBUSTNESS` | `NONFINITE_NEAR_LIMIT_EFFECT_ROLLOUT`; simulation relaxation only |

The withholding records preserve finite metadata for the affected Near-limit effect rollouts and
make no causal claim about a controller operation, mathematical singularity, calibration,
firmware, or hardware. The seven scored-window Loss-v1 outputs alone carry the optimizer-gradient
contract. The other 136 finite features remain mandatory diagnostics and are disclosed as
`DIAGNOSTIC_AD_NOT_OPTIMIZER_CONTRACT`; their AD comparisons are not optimizer gradients.

The inactive integral candidate remains excluded from the objective. The accepted Loss decision
is `RETAIN_LOSS_V1_PLUS_EXTERNAL_GATES`. The package proposes `kp_xy + kp_z` only for a separate
Hauptleitung review and records `PROPOSE_G4_PARAMETER_FREEZE_FOR_HAUPTLEITUNG_REVIEW`.
`G4` is `WITHHELD_PENDING_REBENCHMARK` and was not executed; no G5, optimizer, candidate
selection, firmware, hardware, or flight action occurred.

The package contains exactly 13 regular mode-`600` files totaling `31,722,155` bytes. Its
`SHA256SUMS` indexes the other 12 files and passes `12/12`; the canonical complete 13-file hash
list has SHA-256
`57934a75f9fcbfec5480184405170fd87b6ff82b8c0cf942e1b55af82f87dbfe`.
Generator provenance is exactly
`5c28db1aae6bee67f06a2764f535f5c38924495e` (`research: support g3.4 recovery provenance`).
The later compatibility commit
`5994f49a342b14e3b0fa47117f26488a0c11a0b4`
(`research: stabilize robust generator on extended history`) is recorded separately and is never
called the generator. Exact verification, reproduction, CPU telemetry handling and claim limits
are in [`RUNBOOK.md`](RUNBOOK.md#day-26-g34-freeze-package-verification-and-reproduction) and
[`DAY26_G3_2_CHECKPOINT.md`](checkpoints/DAY26_G3_2_CHECKPOINT.md).

This is pure JAX Float32 CPU simulation evidence. It establishes neither an optimized parameter,
improvement, Test result, physical calibration, integer or firmware differentiability, hardware
transfer, safety, flight readiness, nor Sim2Real performance.
