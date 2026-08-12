# Architecture and research decisions

Status date: 2026-08-03. Decisions D-001 through D-054 record the scientific implementation,
governance, and closure history through the H100 conclusion. The latest scientific decision,
D-054, remains dated 2026-08-02; this status synchronization adds no new research decision.

## D-001: Pure JAX differentiated pipeline

**Decision:** Reference generation, Mellinger control, first-principles dynamics, rollout, and loss
remain JAX-native and functional.

**Reason:** JAX transformations need a traceable PyTree computation without host callbacks or
mutable simulator state in the objective.

**Consequence:** `Sim` may construct `SimData` and pure functions, but the gradient path uses
immutable values and pure calls.

## D-002: Native bridge excluded

**Decision:** The native C/C++ Mellinger bridge is a completed validation proof and is not included
in this repository or gradient path.

**Reason:** Native callbacks, firmware quantization, and integer PWM behavior would break or obscure
the intended pure JAX gradient experiment.

**Consequence:** Native equivalence evidence is not the same as gradient, physical, or hardware
validation and must not be presented as such.

## D-003: `lax.scan` for temporal recursion

**Decision:** Carry full `SimData` through `jax.lax.scan` over state setpoints.

**Reason:** A Python loop under `jit` tends to unroll with horizon-dependent compile/program size;
`scan` represents a fixed-shape recurrence directly and supports reverse-mode autodiff.

**Consequence:** The carry PyTree structure, shapes, and dtypes must stay static. Long-horizon
backward memory remains a risk; rematerialization is deferred until measured.

## D-004: `value_and_grad(..., has_aux=True)`

**Decision:** Compute scalar loss and its gradient in one transformation while returning
interpretable diagnostics as auxiliary data.

**Reason:** This avoids duplicated differentiated forward logic and keeps boolean diagnostics out
of the scalar objective.

**Consequence:** Only floating scalar gain variables are differentiated; modes, cadence, seeds, and
diagnostic indicators are not optimization variables.

## D-005: Smooth bounded gain parameterization

**Decision:** Optimize unconstrained scalar variables and map them through a logistic transform to
conservative physical `kp`/`kd` ranges, sharing x/y values.

**Reason:** This preserves positivity, finite ranges, and controller symmetry without hard clipping
inside the differentiated objective.

**Consequence:** The inverse transform clips only during initialization. Extreme raw values can
still suffer sigmoid saturation.

## D-006: Numerical gradient control is required

**Decision:** Every scientific rollout must compare autodiff against central directional finite
differences and check a small negative-gradient step.

**Reason:** Finite loss and finite gradients alone do not show that gradients are locally correct
or useful.

**Consequence:** The current acceptance threshold is maximum relative directional error `<= 5e-2`
at epsilon `1e-2`, plus a decreasing trial loss.

## D-007: Preserve Crazyflow defaults during Day 1

**Decision:** Replace only experiment-local `kp`/`kd` values in immutable `SimData`; do not alter
checked-in parameters or silently synchronize controller and dynamics mass.

**Reason:** Day 1 validates the current repository configuration. Changing defaults would confound
the baseline.

**Consequence:** The `cf2x_L250` controller/dynamics mass mismatch remains an explicit later
experimental factor.

## D-008: Deterministic, machine-readable evidence

**Decision:** Record commit, source-state hash, seed, frequencies, gains, gradients, metrics,
validations, and versions in JSON, accompanied by a plot.

**Reason:** Scientific evidence needs both machine-readable values and visual inspection, with
enough metadata to reproduce it.

**Consequence:** Timing fields may vary. Result JSON under `artifacts/` is explicitly unignored;
cache and binary outputs remain excluded from routine source analysis.

## D-009: No overclaim of validation

**Decision:** Describe current evidence as differentiable simulation validation only.

**Reason:** The audited cases do not cover hardware, complete physics, disturbances, broad
parameters, saturation regimes, or sim-to-real behavior.

**Consequence:** Reports must list these limitations and must not claim full physical or hardware
validation.

## D-010: Explicit repository-local Python

**Decision:** Automated commands use `.venv/bin/python` and its `-m` entry points from the WSL
repository root.

**Reason:** Shell activation is stateful and unreliable across automation calls; an explicit
interpreter makes environment selection observable and reproducible.

**Consequence:** Do not use unqualified `python`, `pytest`, or `pip`, and do not change dependencies
unless a later scoped task requires it.

## D-011: Batch independent cases on the Crazyflow world axis

**Decision:** Day 2 uses one `Sim(n_worlds=2, n_drones=1, ...)` and commands shaped
`(T, 2, 1, 13)`. World 0 receives Figure-8 and world 1 receives circle. Each world is initialized
from its own trajectory at `t=0`; the raw gain PyTree is shared.

**Reason:** Crazyflow already represents independent simulation cases on the world axis, so no
additional time loop or `vmap` is needed. The outer recurrence remains a single `jax.lax.scan`.

**Consequence:** Per-case reductions retain the world axis. Batch equivalence is checked against
two separately built `N=1` simulations, and reversing world order must leave the combined result
unchanged.

## D-012: Fixed Figure-8 train and circle validation split

**Decision:** Numeric masks `[1, 0]` and `[0, 1]` select Figure-8 as train and circle as validation.
The validation mask does not contribute to `train_loss`.

**Reason:** This creates one deterministic held-out case without putting strings or Python
selection logic in the jitted objective.

**Consequence:** Validation gradients are diagnostic only. This two-case split is not described as
a broad generalization result.

## D-013: Equal arithmetic aggregation after per-case loss

**Decision:** The unchanged Day-1 loss is computed per world before the final reduction. The
combined objective is the equal arithmetic mean of the two case losses and of each additive loss
component.

**Reason:** Equal weighting makes the aggregation explicit and auditable while preserving the
existing `N=M=1` semantics.

**Consequence:** Combined loss and gradient must equal the mean of the separate `N=1` values within
`rtol=1e-5`, `atol=1e-7`. Aggregated RMSE is the square root of mean case MSE, and aggregate maximum
position error is the maximum over selected cases.

## D-014: Preserve the controller/dynamics mass difference during Day 2

**Decision:** Keep controller mass `0.029 kg` and `cf2x_L250` first-principles dynamics mass
`0.0319 kg` unchanged. Mass is not a Day-2 experiment variable.

**Reason:** Changing mass would confound the Day-1 regression and the requested gain-only
diagnostics.

**Consequence:** The mismatch is recorded in every result and remains a limitation. No controller,
dynamics, or checked-in default is synchronized or overwritten.

## D-015: Physical ±10% gain sensitivity is diagnostic only

**Decision:** Perturb exactly one physical gain at a time by `-10%` and `+10%`, map those values
back through the existing logistic parameterization, and evaluate train, validation, and combined
loss. Do not enforce a sign.

**Reason:** This measures a controlled local response around the audited baseline without turning
the sprint into parameter optimization.

**Consequence:** JSON stores physical/raw values, loss differences, and central sensitivities.
Results support local comparison only and do not establish causality or a multi-step update rule.

## D-016: Optax and persistent optimization remain gated

**Decision:** Day 2 contains no optimization loop, Optax import/dependency, or persistent gain
update. The normalized negative-gradient step of raw-space length `1e-2` remains a local validation.

**Reason:** Batch equivalence, split semantics, reproducibility, and gradient controls had to be
established before any optimizer is introduced.

**Consequence:** A later sprint may consider Optax only after the Day-2 checkpoint records all
batch, aggregation, reproducibility, and gradient acceptance criteria as passed. BetaFlight,
TinyMPC, and the native bridge remain outside scope.

## D-017: Fixed Optax Adam optimization

**Decision:** Sprint 3 uses `optax.adam` 0.2.8 with a fixed learning rate of `1e-2`, 50 updates for
the H200 scientific run, and three updates for each H20 smoke run.

**Reason:** Optax 0.2.8 was already importable and exactly resolved in `pixi.lock` as a transitive
dependency of Flax. A single fixed optimizer design avoids an unrecorded replacement
implementation or post-result hyperparameter tuning.

**Consequence:** No dependency or lockfile changed. Learning rate, step count, finite-difference
epsilon `1e-2`, and directional-error threshold `5e-2` were not changed after observing the main
result.

## D-018: Train-only gradient and update

**Decision:** `jax.value_and_grad(..., has_aux=True)` differentiates exactly
`per_case_loss[0]`, the Figure-8/world-0 train loss. Only that gradient is passed to Optax.

**Reason:** The circle case is held out for diagnostic validation and must not affect the optimizer.
Indexing world 0 before differentiation makes this separation explicit rather than relying on a
zero-valued validation weight.

**Consequence:** Validation and combined values are recorded after every step but cannot influence
the gradient, Adam state, parameter update, learning rate, or step count.

## D-019: Train-only checkpoint selection

**Decision:** The research artifact selects the checkpoint with minimum train loss; an exact tie
selects the earliest step.

**Reason:** Validation-based selection would leak the held-out circle case into the reported
optimized parameter set.

**Consequence:** Circle/validation and combined loss are evaluation results only. Step 0, every
post-update step, the final step, and the selected step remain in the history.

## D-020: Optimized gains and matched mass remain experiment-local

**Decision:** Optimized gains are stored only in Day-3 research artifacts. The matched-mass
condition replaces controller `mass` only in immutable experiment-local `SimData`.

**Reason:** Changing checked-in controller or dynamics defaults would destroy the audited baseline
and conflate a research factor with library configuration.

**Consequence:** Controller defaults remain at 0.029 kg and dynamics at 0.0319 kg. Mass is neither
an optimization variable nor differentiated; no default parameter file changed.

## D-021: Two-level controller/dynamics mass matrix

**Decision:** Evaluate both the audited `0.029/0.0319 kg` controller/dynamics mismatch and an
experiment-local `0.0319/0.0319 kg` matched condition.

**Reason:** Day 2 identified the mass mismatch as a controlled open factor. Two explicit levels
isolate its observed association without introducing a broad parameter sweep.

**Consequence:** Both baseline and the selected simulation-optimized gain candidate are evaluated
under both conditions. No
improvement direction is required, and the two levels do not constitute broad physical
robustness.

## D-022: Fixed horizon and trajectory robustness matrix

**Decision:** The full matrix crosses baseline and the selected simulation-optimized gain
candidate, H20/H100/H200/H400, Figure-8/circle, and both mass conditions for 32 cells without
further optimization.

**Reason:** These fixed factors test horizon, held-out trajectory, and the audited mass issue while
keeping Sprint 3 bounded and reproducible.

**Consequence:** H400 is a completion gate. No rematerialization or gradient checkpointing is
introduced; a failed H400 run would leave Sprint 3 incomplete.

## D-023: No stochastic seed-robustness claim

**Decision:** Keep seed `20260724` fixed and make no stochastic robustness claim.

**Reason:** The audited first-principles path has no introduced noise or disturbance model, so a
multi-seed sweep would not establish stochastic robustness.

**Consequence:** Evidence is deterministic simulation evidence for two analytic trajectories, four
horizons, and two mass conditions. It is not hardware, disturbance, sim-to-real, or broad
generalization validation.

## D-024: Feature freeze before supervisor alignment

**Decision:** After Sprint 3, freeze scientific functionality. Sprint 4 may audit, explain, and
prepare handover documentation only.

**Reason:** The prototype already exceeds the original minimal gradient proof, while bachelor
scope, target controller, parameter mapping, and hardware path remain undecided.

**Consequence:** No new simulation feature, controller integration, optimizer experiment, test,
script, notebook, dependency, configuration, or result artifact is added before supervisor
alignment. The scientific Sprint-3 source-state and evidence remain unchanged.

## D-025: No artificial noise or delay without system identification

**Decision:** Do not introduce an arbitrary noise, delay, latency, or domain-randomization model
without evaluating flight/Mocap logs and documenting an identification or calibration method.

**Reason:** Uncalibrated stochastic or latency assumptions would add apparent realism without an
empirical basis and could weaken rather than strengthen the scientific claim.

**Consequence:** A later Sim2Real extension must first identify available logs, observables,
sampling/timestamp semantics, target parameterization, train/validation/test split, and acceptance
criterion. Exactly one empirically calibrated sensing component is the preferred
bachelor-realistic extension.

## D-026: Simulation gains are not a hardware release artifact

**Decision:** Describe the selected Tag-3 parameters only as „bestes getestetes Train-Checkpoint
des festgelegten H200-Laufs“ or as „simulationsoptimierter Gain-Kandidat“.

**Reason:** Perfect simulation states, one train trajectory, one analytic validation trajectory,
missing firmware/unit equivalence, unexercised safety branches, and the absence of a laboratory
safety process do not support direct flight.

**Consequence:** Do not upload, arm, or fly these gains based on the current evidence. A future
hardware path needs implementation mapping, unit checks, limits, safety review, staged tests, kill
procedure, and explicit authorization.

## D-027: Bachelor scope, target controller, and hardware path require supervisor decision

**Decision:** Treat the binding bachelor scope, the controller that will actually run on the
Crazyflie, UKF role, test trajectories, loss/gain boundaries, mass handling, and any
BetaFlight/TinyMPC comparison as supervisor decisions.

**Reason:** These choices materially change the research question, implementation work, evidence
requirements, and safety responsibility and cannot be inferred from the simulation prototype.

**Consequence:** Use
[`SUPERVISOR_MEETING_BRIEF.md`](SUPERVISOR_MEETING_BRIEF.md) as the decision agenda. Until those
questions are resolved, Option A—the cleanly documented simulation prototype—is the safe
completion boundary; broader options remain conditional.

## D-028: Heuristic domain randomization is allowed but is not calibration

**Date:** 2026-07-31

**Source:** User-reported oral supervisor decision; no written supervisor statement was supplied.

**Decision:** Refine, rather than rewrite, D-025. Configurable simulation-only domain
randomization may be implemented and technically exercised without prior system identification,
provided every uncalibrated component is called heuristic, its values remain explicit, and the
research/workstation preset keeps uncalibrated delay and wrench disabled.

**Reason:** The current sprint needs the experiment architecture before logs and lab calibration
exist. Hiding this engineering capability would block the agreed simulation research, while
calling it realistic would overstate evidence.

**Consequence:** The smoke may use tiny technical delay/wrench values. No Sim2Real, realism,
hardware-safety, or empirical-robustness claim follows. D-025 remains the gate for interpreting a
model as calibrated.

## D-029: Train, validation, and test use structural API separation

**Decision:** Create separate `Sim`/`step_fn` instances. The training-visible container contains
only Train and Validation. The test pipeline requires the explicit `build_test_pipeline(...)`
API, and the training runner must never call it.

**Reason:** Masks in a shared two-world batch prevent numerical contribution but do not create a
strong boundary against state, PRNG, checkpoint-selection, or logging leakage.

**Consequence:** Train resamples per update; Validation uses a fixed versioned manifest and may
select a checkpoint; Test is frozen and unavailable to the training loop. The test manifest ID may
be recorded for provenance without realizing any test episode.

## D-030: Explicit fold-in seed tree owns research randomness

**Decision:** Derive keys by the stable namespaces `split → episode → world → component`, with
components `trajectory`, `mass`, `delay`, and `wrench`.

**Reason:** `SimCore.rng_key` is a single typed key despite an outdated shape comment and its split
order is pipeline-dependent. It is unsuitable as the scientific experiment manifest.

**Consequence:** Same coordinates reproduce bit-identical samples; worlds and components are
independent by construction; changes to a Train root seed do not alter manifest-seeded Validation
or Test.

## D-031: Gains are named, bounded, shared, and staged

**Decision:** Register grouped x/y and z gains by name and metadata. Stage 1 is `kp/kd`, Stage 2
adds position integrals, Stage 3 adds roll/pitch attitude/rate terms, and Stage 4 registers yaw
terms only for a later yaw-exciting objective.

**Reason:** A flat vector with undocumented positions cannot support audit, resume compatibility,
or principled exclusion. Some gains are structurally inactive or unidentified in current tasks.

**Consequence:** All worlds share the same raw vector. `ki_m_xy` is deferred because the zero
default gives no supported nonzero initialization/bounds. `kd_omega_z` is excluded because
`attitude2force_torque` explicitly zeros yaw derivative error. New 0.5×–2× bounds are configurable
engineering bounds only and provide no stability guarantee.

## D-032: Random Fourier references use analytic C3 endpoint envelopes

**Decision:** Generate finite-harmonic three-axis Fourier curves multiplied by the product of two
seventh-order smoothsteps, and compute derivatives through jerk analytically.

**Reason:** Existing circle/figure-eight references remain useful baselines but do not define a
random distribution. The Fourier representation is compact, deterministic, JAX-compatible, and
does not require a spline solver.

**Consequence:** Start/end position equals the configured hover center, with zero velocity,
acceleration, and jerk up to numerical tolerance. Host rejection is capped by `max_attempts`; an
invalid distribution fails explicitly instead of looping under JIT. Limits are conservative
simulation filters, not hardware certification.

## D-033: Validation selects; test never selects

**Decision:** Choose the earliest checkpoint with minimum fixed-validation loss. Do not generate
test metrics during training.

**Reason:** Train-only selection used in Sprint 3 could overfit training; test-based selection
would destroy the test split.

**Consequence:** Each checkpoint records the criterion, selected step/raw gains, and
`test_used_for_selection: false`. True test evaluation is a later, one-way procedure after all
design choices are frozen.

## D-034: Zero-output reductions are per world

**Decision:** In both legacy PWM clipping and force/torque-to-rotor conversion, reduce the motor
axis only (`axis=-1, keepdims=True`).

**Reason:** A scalar `all(...)` coupled every world: one active world caused a null world to receive
minimum PWM/thrust.

**Consequence:** Mixed null/active batches now equal sequential evaluation. Dedicated regression
tests cover both conversion layers. This is a general Crazyflow controller correctness fix, not an
experiment-specific workaround.

## D-035: Resume requires exact scientific-state equivalence and strict compatibility

**Date:** 2026-07-31

**Decision:** Accept checkpoint/resume only when an uninterrupted two-update run and a planned
one-update pause plus fresh-process second update are exactly equal for raw/selected/transformed
gains, all Optax leaves and tree structure, counters, selection, full Train/Validation history and
metrics, seeds, manifests, and fingerprints. With identical CPU backend, dtype, executable source,
runtime, config, and update order, the tolerance is `rtol=0`, `atol=0`.

**Reason:** A JSON roundtrip alone does not prove deterministic continuation. Raw-array dtype,
prior metric records, Validation-manifest content, code state, runtime/backend, and resume lineage
were previously not enforced.

**Consequence:** Checkpoint schema v2 stores typed arrays, complete metrics, checksummed payload,
Validation content fingerprint, opaque frozen-Test reference, executable-source fingerprint,
runtime fingerprint, and lineage. Incompatible or corrupt state is rejected. Timings, paths,
commands, and UTC differ across processes and are excluded from mathematical equivalence.

## D-036: The training runner does not open the frozen Test manifest

**Date:** 2026-07-31

**Decision:** The Train/Validation runner records `config.test_manifest` only as an opaque frozen
reference. It must not open, parse, validate, build, simulate, metrically evaluate, or use Test for
selection.

**Reason:** Recording the actual manifest ID required file access and weakened the strict Sprint-6
boundary. Config fingerprint plus clean repository state and the separately audited static
Test-file SHA-256 preserve the reference without exposing episodes to the runtime.

**Consequence:** Validation remains content-fingerprinted and may select. The initial source audit
and structural manifest tests may inspect static JSON, but experiment processes do not. A future
one-way Test entry point remains separate and requires an explicit later decision.

## D-037: Only completed updates are checkpointable; planned pause is preferred to signals

**Date:** 2026-07-31

**Decision:** Use `--max-updates-this-process` for planned stops. Atomically checkpoint only after
Train update, Validation, history, and selection are complete. Do not attempt signal/emergency
serialization of a partially executing JAX update.

**Reason:** Python signal delivery during compiled JAX work cannot establish a trustworthy atomic
optimizer boundary. Saving partial or ambiguous state is worse than replaying one deterministic
uncommitted step.

**Consequence:** A timeout or termination resumes from the last complete atomic checkpoint in a
new empty directory. JSON and JSONL are flushed, `fsync`ed, and same-directory replaced. Existing
outputs are never overwritten. A missing success summary means interrupted, not successful.

## D-038: H100 trajectory rejection is a Workstation-pilot NO-GO

**Date:** 2026-07-31

**Decision:** Do not advance beyond the mandatory H100 × one-world local gate and do not execute
Workstation H100 × 4, H200 × 8, H400 × 16, or the main run while the unchanged Workstation
trajectory distribution fails deterministic H100 episode construction.

**Reason:** The final H100 × one-world process failed before JIT after all 16 deterministic
trajectory attempts. Runtime was 4.41 s, Peak-RSS 591,776 KiB, swaps 0. This is neither a memory
nor compile-performance result, and H20 evidence cannot justify bypassing it.

**Consequence:** H100 × 4 and H200 × 4 local pilots were skipped. No bounds, loss, Validation
manifest, trajectory distribution, or generator limit is changed silently. A later
scientific/configuration decision must define a horizon-consistent pilot and repeat the sequence
from H100. The only positive claim is exact Resume readiness, not Workstation, scientific,
hardware, firmware, safety, realism, or Sim2Real readiness.

## D-039: Legacy normalized-time trajectories remain immutable and explicitly named

**Date:** 2026-07-31

**Decision:** Old configs that omit a trajectory distribution version resolve to
`legacy_normalized_time_v1`. Preserve its PRNG order, coefficient/phase sampling, per-horizon
frequency/envelope equations, validation comparisons, attempt order, and first-valid behavior.
Unknown distribution names are hard errors with no fallback.

**Reason:** Sprint 7 proves that the Sprint-6 failure is not a coding/units defect but an inherent
property of the original normalized-time definition. Reinterpreting old configs would invalidate
historical reproducibility and hide a scientific distribution change.

**Consequence:** A new default field changes newly resolved config fingerprints, and new source
changes the executable-source fingerprint; old checkpoints are therefore correctly incompatible.
Historical files remain unchanged. Golden H400 array digests protect exact legacy candidate
semantics.

## D-040: Horizon-consistent pilots use a versioned fixed parent and exact prefixes

**Date:** 2026-07-31

**Decision:** Add opt-in `fixed_support_prefix_v2`. Construct and validate one complete parent on
an explicit physical support before returning a horizon prefix. The Sprint-7 pilot uses 4.0 s
support and explicitly permits only 1.0..4.0 s returned durations.

**Reason:** With legacy duration `D=H/f_control`, fixed random geometry has velocity `D^-1`,
acceleration `D^-2`, and jerk `D^-3`. The measured success rate was 0% at H20/H50/H100, 37.5% at
H200, and 100% at H400 over 32 seeds; 0 of 2,048 estimate-only H100 candidates were valid. More
attempts, amplitude schedules, or coefficient projection do not preserve one physical
distribution as directly as a common validated parent.

**Consequence:** Paired H100/H200/H400 seeds share coefficients, accepted attempt, and exact array
prefixes; paired H400 is exactly legacy-v1. H100/H200 no longer have a forced-hover terminal
endpoint, which is an explicit new task semantic. Validation losses across v1/v2 are not directly
comparable. `workstation.json` remains v1; later v2 Workstation work requires a new config.

## D-041: Sprint-7 GO is CPU-only incremental-pilot readiness

**Date:** 2026-07-31

**Decision:** Accept Sprint 7 only as GO for later incremental Workstation pilots on the checked
CPU path after machine inventory, explicit v2 config creation, and exact same-machine Resume.

**Reason:** Two fresh 128-seed construction processes passed H100/H200/H400 deterministically and
all predeclared diversity criteria. Local H100×1, H100×4, and H200×4 real value/gradient pilots
passed within time/RSS gates, and exact H100 Resume passed. But `build_split_pipeline` still
hard-codes `device="cpu"`, the local JAX environment exposes only `cpu:0`, and no GPU path was
enabled or tested.

**Consequence:** CPU pilot success is not GPU readiness. H400 simulation, larger local world
counts, Workstation execution, main optimization, Test, scientific performance, generalization,
Sim2Real, firmware, hardware, and safety claims remain outside the Sprint-7 result.

## D-042: Sprint-8 night pilot is fixed at H100×4 and 50 updates

**Date:** 2026-07-31

**Decision:** Release exactly one manually started CPU pilot config with H100, four Train worlds,
four existing-Validation worlds, one seed, shared Stage-1 gains, `fixed_support_prefix_v2`, and 50
updates. Keep mass-only uniform ±0.0002 kg randomization and every existing loss, bound, manifest,
and controller-mass assumption unchanged. Do not start it during preparation.

**Reason:** The sole fresh two-update H100×4 gate exited 0, produced complete finite evidence and
two valid atomic checkpoints, peaked at 1.143 GiB without swap, and measured 4.096 s between
complete updates after the first. Estimated walltimes for 10/25/50 updates are approximately
52.0/113.5/215.9 s. Fifty updates maximize bounded repeated-checkpoint and trend observation while
remaining far below the six-hour cap, even under a factor-two allowance.

**Consequence:** The job stops at update 50 or 21,600 seconds, whichever comes first. It has no
automatic extension, second configuration, H200/H400 follow-on, seed loop, or Test entry point.
Its result is technical/exploratory evidence only, not a main optimization or scientific seed,
generalization, GPU, firmware, hardware, physical-safety, or Sim2Real result.

## D-043: Manual launcher owns process boundaries; runner owns scientific state

**Date:** 2026-07-31

**Decision:** Use a checked shell launcher only for unique run-directory creation, full stream and
environment logging, provenance preflight, a six-hour external timeout, process-group control,
and dispatch. Keep optimization, finite checks, metric writes, atomic checkpoints, and Resume in
the existing real Python runner.

**Reason:** Duplicating optimizer or serialization logic in an overnight wrapper would create a
second scientific path. The existing runner already fingerprints config/source/runtime, rejects
nonfinite state, atomically checkpoints every complete update, refuses overwrites, and strictly
loads Resume state. A signal cannot make a partial JAX update trustworthy.

**Consequence:** Every invocation uses a new absent run root; Resume also writes a new root and
retains the parent checkpoint. SIGTERM can lose only the in-flight update. The read-only morning
inspector verifies sequential checkpoint checksums, finite Train/Validation metrics, technical
gates, completion, resource records, and absence of Test metrics without opening the Test
manifest.

## D-044: Preserve the completed Sprint-8 run as immutable scientific evidence

**Date:** 2026-08-01

**Decision:** Track exactly the audited 61-file run
`artifacts/day8-night-pilot/runs/20260731T213826Z-2e16f365ecfd`, a complete relative-path SHA-256
index, and the required artifact-manifest entry. Do not normalize or rewrite any raw run file.

**Reason:** Read-only validation tied every file to clean execution HEAD `2e16f365…`, the released
config, seed `20260731`, and the expected source/runtime fingerprints. All 50 atomic checkpoint
payload hashes, 100 Train/Validation metrics, finite and technical gates, external exit/resource
records, and Test-boundary fields pass. The 4,602,398-byte raw evidence is complete, text/JSON,
free of detected sensitive material, and suitable for the repository's existing evidence model.

**Consequence:** Commit `d1d0cc0a400b8af81da9f435226a6b5797dc2c97` is the authorized
Sprint-9 base. It contains 63 files: 61 unchanged raw files, the run hash index, and the manifest
update. Historical execution metadata remains unchanged.

## D-045: Do not release Sprint-9's 5,000-update run with per-update full-history checkpoints

**Date:** 2026-08-01

**Decision:** Prepare and fingerprint the exact requested 5,000-update H100×4 Train/Validation
config and safe launcher, but set the release gate to `NO-GO` and start no training process.

**Reason:** Checkpoint frequency is not configurable. A complete-history atomic checkpoint is
written at every update, and observed sizes are almost exactly linear in the update number. The
50-update evidence projects 5,000 checkpoint files totaling 41,408,829,822 bytes. Fits to
measured completed-update interval growth project 8,032.99, 8,340.84, and 13,438.55 seconds; all
exceed the mandatory 7,200-second cap. Filesystem capacity and observed memory headroom pass, but
a timeout-safe interrupted job is not the required result of exactly 5,000 completed updates.

**Consequence:** No tmux session, run root, PID, log, checkpoint, Test access, seed loop,
H200/H400 process, second configuration, or automatic continuation exists for Sprint 9. Do not
silently change checkpoint semantics. A later explicitly authorized sprint may implement and
prove configurable sparse atomic checkpoints while preserving every-update metrics, exact
Resume, mismatch rejection, a mandatory final checkpoint, and the scientific config.

## D-046: Sparse checkpoint interval is backward-compatible technical configuration

**Date:** 2026-08-01

**Decision:** Add positive integer `optimizer.checkpoint_interval` with default 1. Omit the default
from canonical serialization so existing configs retain their exact historical fingerprints and
per-update checkpoint semantics. Explicit intervals write at multiples, the configured final
step, and planned process stops; coincident conditions result in one write.

**Reason:** Checkpoints must remain complete atomic Resume states, but writing complete growing
history after every update caused quadratic aggregate output. A technical persistence interval
does not change trajectory, split, seed, optimizer update, loss, gain, or selection semantics.
Durably append only the two new metric rows per update so complete metrics remain available
without another growing-history rewrite.

**Consequence:** Config/source/runtime mismatches, checksum validation, nonfinite rejection,
overwrite refusal, and new-directory Resume remain hard requirements. Missing intervals behave as
before. Explicit sparse configs receive distinct fingerprints. Interrupted work may fall back to
the latest complete interval checkpoint; partial JAX state is never serialized.

## D-047: Release exactly one Sprint-10 H100×4/5,000 sparse CPU pilot

**Date:** 2026-08-01

**Decision:** Set the Sprint-10 gate to GO for exactly one detached tmux run with interval 100,
50 checkpoints, hard 7,200-second timeout, 12-GiB RSS limit, and 2-GiB artifact limit.

**Reason:** The real 210-update smoke wrote exact steps 100/200/210, exited 0 in 55.99 seconds,
peaked at 1,309,284 KiB without swap, preserved 420 finite and passing metric rows, and never
opened Test. Fresh-process Resume 200→210 matched uninterrupted state exactly. A deliberately
conservative model gives 2,289.55 seconds and 875,333,570 bytes after doubled checkpoint/storage
allowances, below both hard limits.

**Consequence:** The released pilot changes Sprint-9 semantics only by run ID and technical
checkpoint interval. No automatic Resume, extension, second run/config, H200/H400, seed loop, or
Test evaluation is authorized. Results remain technical/exploratory single-seed simulation
evidence, not a main study, hardware result, or Sim2Real claim.

## D-048: Accept the long-pilot PASS and replicate H100 before horizon expansion

**Date:** 2026-08-01

**Decision:** Accept the sole Sprint-10 5,000-update run as healthy exploratory H100 evidence.
Recommend a separate protocol-freeze step followed by multiple independently predeclared H100
seeds. Preserve the validation-based minimum/earliest-exact-tie selection rule and all scientific
semantics. Do not proceed to H200/H400 until H100 replication is evaluated, and do not open Test
until every decision is frozen for its final one-time protocol.

**Reason:** All completion, checksum, finite-value, fingerprint, resource, and Test-boundary gates
pass. Validation decreases strictly across all 4,999 transitions, final Train and Validation loss
are about 75% below their initial values, and Gradient-L2 is 95.49% lower. Late Validation and
gradient endpoints still improve by 3.53% and 16.28%, so the declared plateau criteria do not
fire. All persisted gains remain inside bounds; `kp_z` is closest at a normalized margin of
0.08602. The deterministic prefixes exactly reproduce Sprint 8 and the Sprint-10 smoke.

**Consequence:** The result supports method freezing and replication, not a seed-robustness,
population-generalization, H200/H400, Test, firmware, hardware, formal-stability, or Sim2Real
claim. Train/gradient oscillation is interpreted alongside per-update Train resampling. Sparse
gain history is resolved at the 50 checkpoint steps only; unavailable intermediate vectors are
not reconstructed or inferred.

## D-049: Keep the large raw Sprint-10 run outside ordinary Git

**Date:** 2026-08-01

**Decision:** Version the deterministic Sprint-11 analysis, generator, documentation, and full
61-file SHA-256 index, but do not stage the unchanged 437,505,563-byte raw run in ordinary Git.
Preserve it locally and mirror it byte-for-byte with the index to backed-up institutional
research storage.

**Reason:** The repository has no Git-LFS policy, its existing packed history is about 21 MiB,
and the raw run would add roughly 437.5 MB, predominantly 50 JSON checkpoints containing growing
duplicate histories. This is disproportionate to the existing evidence model, while the compact
analysis and full hash inventory retain reproducible verification and exact identity.

**Consequence:** The raw run remains the immutable scientific source and is neither edited,
deleted, moved, compressed, ignored, nor partially staged. Git alone is not its durability layer;
the external mirror must be hash-verified after copying. No external copy was performed by this
sprint.

## D-050: Freeze ten-seed H100 replication before any confirmatory execution

**Date:** 2026-08-01

**Decision:** Exclude the Sprint-10 seed `20260731` as exploratory and freeze exactly ten new
SHA-256-derived root seeds, ten configs, sequential execution order, H100 scientific semantics,
selection, endpoints, aggregation, uncertainty, failure handling, resources, persistence, and
H200/H400 gates before starting any confirmatory run.

**Reason:** The single healthy pilot informs feasibility but cannot estimate between-seed
variation. Predeclaring the complete panel prevents seed selection, hyperparameter drift,
post-result endpoint changes, replacement-seed bias, and premature horizon/Test decisions. A
versioned deterministic SHA-256 namespace yields distinct auditable root seeds while the existing
fold-in contract separates Train randomness by split, episode, world, and component. The same
fixed Validation worlds remain deliberate common comparators.

**Consequence:** Configs differ from Sprint 10 only by `run_id` and `root_seed`. Run order is
strictly sequential and the next seed waits for integrity plus externally verified persistence.
There is no automatic loop, parallel execution, Resume, replacement seed, H200/H400, or Test.
Sprint 12 authorizes zero research processes; Sprint 13 requires a new explicit preflight and
authorization.

## D-051: Selection is over every post-update step, with strict earliest-tie retention

**Date:** 2026-08-01

**Decision:** Define candidate steps as every completed post-update state 1 through 5,000. Choose
the fixed-Validation global minimum. The implementation updates only on strict
`validation_loss < best_validation`; exact equality therefore retains the earliest step. Step 0
and Test are ineligible.

**Reason:** This is the exact current runner behavior and summary criterion. Earlier wording
“earliest checkpoint” could be read as restricting selection to sparse checkpoint-file steps,
although the runner validates and compares every update. Interval checkpoints store the current
selected step and selected raw gains even when the selected step lies between files.

**Consequence:** The clarification changes no runner or Sprint-10 semantics. Per-seed primary
output is `run_summary.selected.selected_validation_loss`; selected gains remain recoverable from
the final checkpoint/summary. Sparse gain trajectory samples are still only interval-persisted.

## D-052: Freeze confirmatory aggregation, failures, horizon gates, and raw persistence

**Date:** 2026-08-01

**Decision:** Aggregate the ten selected Validation losses with all individual values, mean,
`ddof=1` standard deviation, a two-sided 95% Student-t interval (`df=9`, critical value
`2.2621571627409915`), median, and linear-method Q1/Q3. Analyze relative step-1-to-selected
Validation improvement identically. Secondary runner-emitted quantities are descriptive only.
Allow at most one full update-0 retry for a documented exogenous failure; never Resume or replace
a seed. Require all ten frozen GO conditions before preparing H200/H400.

**Reason:** Ten seeds support a transparent uncertainty display but not unrestricted model
selection or broad asymptotic claims. Exact failure categories prevent result-dependent retries.
The GO gates demand complete technical validity, consistent improvement, bounded uncertainty,
late-run health, non-saturated selected gains, and unchanged methodology. Roughly 4.4 GB expected
raw output is too large for ordinary Git and needs a real durability layer.

**Consequence:** Any nonretryable/second failure or failed GO condition is NO-GO, stops new
launches, and does not authorize tuning or added seeds. Each attempt is checksummed, copied
byte-for-byte to approved backed-up external research storage, and fully verified at destination
before the next seed. Missing capacity or persistence blocks launch. Test remains closed until
H100, later horizons, and all decisions are frozen for the eventual one-time protocol.

## D-053: Amend replication persistence to immutable local-SSD retention before Seed 01

**Date:** 2026-08-01

**Decision:** Before any confirmatory seed or result, replace only the mandatory external-storage
capacity/copy/destination-verification gates with immutable local retention under the repository
artifact area on the user-declared internal 2-TB SSD. Create and immediately verify a complete
local SHA-256 index after every attempt; reverify the previous seed's complete index before every
later seed. Require 20,000,000,000 free local bytes before Seed 01 and, before later seeds,
10,000,000,000 bytes plus 437,505,563 bytes for every remaining seed including the one to start.

**Reason:** The user explicitly chose not to use an external data carrier or research-storage
service and accepts the durability tradeoff. The amendment was adopted at
`2026-08-01T12:51:18Z`, with zero confirmatory seeds started, no confirmatory results known, Test
unopened, and 1,006,592,950,272 local bytes available. It is a storage/fault-tolerance decision,
not result-dependent model or analysis adaptation. The original external rule remains auditable
in base commit `73868cfd84d25bec0c6612b31ee34f4e67acb307`.

**Consequence:** Missing files, a SHA mismatch, or insufficient local capacity blocks all later
seeds. Successful and failed attempts, plus the Sprint-10 pilot, remain unchanged; no run may be
deleted, moved, renamed, compressed, modified, overwritten, or added to ordinary Git during the
series. External capacity, copy, destination verification, and receipt are no longer gates. All
seeds/configs, scientific semantics, fingerprints, selection, statistical endpoints and
uncertainty, scientific GO/NO-GO thresholds, sequential order, and Test lock stay frozen.
SHA-256 detects alteration but is not a backup: complete loss or failure of the single SSD can
destroy all raw evidence, and the user knowingly accepts that limitation. Sprint 12A starts no
run; a separately authorized Sprint 13 may perform a fresh preflight for exactly Seed 01.

## D-054: Close H100 as positive Validation evidence while retaining horizon NO-GO

**Date:** 2026-08-02

**Decision:** Version a deterministic ten-seed synthesis of the existing H100 primary data and
accept its result as positive simulation evidence under the frozen H100 configuration. Include all
ten predeclared, technically valid seeds. Keep H200/H400 formally NO-GO / not authorized and keep
Test closed. This closure makes no new horizon, Test, hardware, or controller decision.

**Reason:** Every data-derived frozen gate passes: ten complete runs, 600/600 indexed raw-file
hashes, 500/500 checkpoint-payload hashes, exact config/fingerprint provenance, 100,000 ordered
Train/Validation rows, positive improvement for every seed, the frozen uncertainty thresholds,
gain-bound margins, resource limits, and Test-boundary checks. All seeds select update 5,000. The
mean relative Validation improvement is `74.7788986450 %` with the frozen 95% Student-t interval
`[74.7757723574 %; 74.7820249326 %]`.

The execution-protocol condition is different. The provided historical lead handoff reports that
two Sprint-17 just-in-time wrapper/preflight attempts exited nonzero before Seed 09 and Seed 10 and
that execution continued after the first event despite the literal stop rule. The current raw runs
prove that Seed 09 and 10 completed successfully and are internally valid, but the repository has
no authorization ID, deviation record, waiver, or explicit later horizon authorization. The exact
wording of a user decision that would cure the governance gap is not available. Numerical success
cannot silently substitute for that missing evidence.

**Consequence:** Seed 09 and 10 stay in the primary ten-seed analysis; they are not deleted,
repeated, or post-hoc excluded. The H100 scientific result is positive, while formal permission to
prepare or run H200/H400 is absent. Any future horizon change requires a separately documented,
explicit scientific/governance decision. The versioned synthesis reports the data-derived gate
subset and deliberately leaves execution-protocol conformity outside numerical inference.
