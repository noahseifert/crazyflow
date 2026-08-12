# Day 25 checkpoint: cf21B_500 robust baseline foundation

## Boundary

- Work Order: `WO-GR-G2-003`; prospective decisions: `D-045`, D-047 correction `CORR-01`.
- Base: `a4e4136f8312851790555e3de6cdf23065252edd`.
- Corrected evaluator/CLI/test commit: `2288ab2fe3b55f585fb9b14a8fe9d87c07eb6c1a`.
- Platform: `cf21B_500`; JAX Float32; CPU; first-principles dynamics; Euler integration.
- Sampling: 500-Hz simulation/attitude/force-torque, 100-Hz state control, five substeps.
- Population: 16 deterministic parents, 32 fixed paired rollouts, Train/Validation only.
- Scope: robust baseline evaluation and resource measurement; zero optimizer updates.
- Shared controller/default/config/dependency/lockfile changes: zero.
- Protected Day-10 / Day-13 / Test reads: zero.

The first independent review of result commit
`c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` was explicitly `NOT ACCEPTED`. D-047 authorizes
one transparency-only correction cycle, not a numeric change and not acceptance. This is the
corrected worker review candidate pending independent Gradient-Lead acceptance. Historical
`WO-GR-G2-001` and `WO-GR-G2-002` results remain `BLOCKED/WITHHELD` and their reviews remain
`REJECTED`; this checkpoint does not edit or reclassify them.

## Episode, carry and ablation contract

Soft, Nominal, Dynamic and Near-limit each have two Train and two Validation parents. Every
analytic position/nonzero-yaw reference has 8-s parent support. Each simulation is one continuous
6-s rollout from the parent start. A half-open 2-s scored window follows the fixed class-specific
2/3/4-s warm-up with no reset. Fixed seeds, initial states, reference derivatives, window indices
and split/class membership pass their deterministic contracts.

Every class retains the same fixed positive-z central climb, added after class-scaled Fourier
construction and before the fixed center: exact Float32 `0.06 m/s`, base displacement
`0.06 * (time - 2 s)`, base velocity `0.06 m/s`, zero base acceleration and jerk, and product-
rule derivatives through jerk under the same C3 smoothstep7 2-s entry / 4-s plateau / 2-s exit
envelope. No trajectory number changed in `CORR-01`. Absolute z results apply only to references
containing this climb; it is not an isolated causal factor.

Each episode is paired between repository mismatch (0.0393-kg controller, 0.04338-kg dynamics)
and physical-value match (0.04338 kg for both). Exact PyTree/input comparison proves that only
`controls.state.params["mass"]` differs. Gains, `int_err_max`, controller semantics, canonical
TOMLs, dynamics, trajectory, seeds, initial state, warm-up and scoring windows remain identical.

## Technical result

- All 32 complete production rollouts, objectives, gradients and stored arrays are finite.
- Full-carry replay and production/diagnostic trace identity pass under the fixed Float32 rule.
- Full-6-s ground/floor and zero-thrust activation are zero.
- Scored-window Stage-A torque-axis, Stage-A motor and Stage-B additional clip counts are zero in
  every episode and both mass variants.
- Soft/Nominal and Dynamic hard clip gates pass; Near-limit clipping is fully described and is
  also zero in this run.
- Tracking, unchanged Loss v1, integral contacts, reserve, requested/realized wrench,
  trajectory arrays, per-episode pairs and per-split/per-class aggregates are present and
  recomputable. The Loss-v1 aggregates explicitly retain raw term, normalization, weight,
  normalized term, weighted term and total; trajectory-statistic aggregates are explicit.
- The complete nonintegral gate result is PASS.

Finiteness, ground/floor and zero thrust are full-rollout gates. Integral contact, torque/motor/
Stage-B clipping, tracking, Loss v1, reserve and wrench statements are scored-window-only.
Integral contact uses the unchanged detector
`abs(pos_err_i) >= int_err_max * (1 - 1e-4)`. D-045 classifies it as measured baseline evidence,
not a technical stop and not a controller/candidate pass.

## D-047 pre-correction invariance

The automatic comparator always loads its old side explicitly from
`c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` and records
`PASS_PRE_CORRECTION_C4_INVARIANCE`.

- Numeric report projection: old and corrected SHA-256
  `fd1195f9d10c88a2d550090631f6b21d0d4f80fd47a2bbe09a52e7c9a0975a61`, 5,465,162 paths and
  644,397,249 serialized bytes.
- Parent projection: old and corrected SHA-256
  `bcb40e132f4b85e99f814f0a9db4d90244523b488697868ece8597d3654de82c`, 16 records and 18,037
  serialized bytes.
- Old report, contract and benchmark pins validate; every retained pre-existing scalar leaf
  outside declared prose/provenance replacements, raw array, summary, pair delta, ID, seed,
  attempt, window, mass and digest is identical.
- Only D-047-permitted aggregates, metadata, corrected prose, stable generator provenance and
  checksums differ.

## Negative physical-value-match finding

| Variant | Mean Z-RMSE | Negative-z contact samples | Positive-z contact samples |
|---|---:|---:|---:|
| Repository mismatch | 0.08494756871345588 m | 593 | 0 |
| Physical-value match | 0.11452380346939901 m | 1,844 | 0 |
| Matched minus mismatch | +0.029576234755943134 m | +1,251 | 0 |

Physical-value matching increases negative-z integral-boundary contact and worsens mean Z-RMSE
for these fixed episodes in this fixed simulation/controller conversion chain. This is a
negative result, not improvement, candidate acceptance, causal isolation, physical-mass
estimation, transfer or flight readiness.

## Mechanism audit and transfer boundary

Measured evidence stores the available requested collective/target-thrust-equivalent boundary,
realized wrench, z error, z integral, motor commands and reserve separately for the early
transient and scored window. Code-path inference separately records
`mass * (setpoint_acc - gravity_vec) + feedback`, followed by existing `mass_thrust=132000` and
nonlinear PWM-to-force conversion. This path explains why physical equality need not be output-
neutral, but it does not experimentally isolate a cause.

The controller `mass` is therefore a thrust-feedforward parameter inside this fixed conversion
chain for this evidence. It is not a physical-mass estimate or a deployable firmware/real-flight
parameter until a separate semantics/transfer gate exists and passes.

## Fresh-process benchmark

| Worlds | Compile (s) | First call (s) | Steady median (s) | Steady worlds/s | Max RSS (GiB) |
|---:|---:|---:|---:|---:|---:|
| 4 | 3.489196014 | 0.812371716 | 0.748852923 | 5.341502820040405 | 1.132907867 |
| 16 | 3.672480289 | 0.922251176 | 0.900487969 | 17.768144107209054 | 1.186634064 |
| 32 | 3.658886151 | 1.096963572 | 1.071313824 | 29.869865657590918 | 1.263717651 |

Every 4/16/32 child has zero swap, no timeout or memory-guard failure, finite objective/gradient
and consistent replicated-block values. The predeclared smallest-within-90%-of-best allowed rule
recommends 32 worlds for the next CPU batch only. It is not a global default and makes no GPU
claim. The benchmark file SHA-256 is
`9ca5d2e07da2e41b1144117c744cd36a6a908e078bd4f7e21c06d44323b2e8d0`; its canonical raw
measurement hash is `282dc6e652a560152efac0f7dce62dd9d9bbc845fc4c05f5dda4fc48d2e564e5`.

## Verification

- Corrected focused suite after the first rewritten commit: `30 passed in 109.08s`.
- Prescribed eight-file suite: `126 passed in 128.54s`.
- Ruff lint: PASS; Ruff format check: PASS.
- Complete 32-rollout generation and every nonintegral gate: PASS.
- Two fresh `/tmp` packages: byte-identical to each other and the final package.
- Day-25 checksum index: 9/9 PASS; SHA-256
  `819d2ec725b5b8b045a6b6acfaae6443a6151df01d59d9d7c5a178d787b70437`.
- New non-package fresh-process 4/16/32 benchmark: `PASS_4_16_32_BENCHMARK`, replicated-block
  objective/gradient consistency PASS, zero swap, all resource/timeout guards PASS, unchanged
  recommendation 32 worlds; result-file SHA-256
  `e1096e1d896ffeb784ffda3c4df0b939b7bf3291f185665e71babbd49c2d9b16` and canonical raw-
  measurement SHA-256 `a91e0dfefaf008ca9c9b9b031319888855af87f1f3c8912f93b2ed5fe88b06aa`.
- Day-19 through Day-24 checksum sets: PASS; historical payloads were checksum-only.
- Both PNGs: inspected at original resolution; values, negative status, recommendation and claim
  boundaries are legible; overview measurement labels correctly say scored window.
- Exact 18-path allowlist, diff/protected-path/live-status checks: PASS. Source/test and
  evidence/documentation are split into exactly two linear worker commits; no push or integration
  by the worker.

The stable generator identity is the first rewritten commit
`2288ab2fe3b55f585fb9b14a8fe9d87c07eb6c1a`, so generation from either worker commit must be
byte-identical. The ten package files total 294,798,314 bytes.

## Claim boundary

The result is `NEGATIVE_PHYSICAL_VALUE_MATCH_FINDING` with all nonintegral foundation gates
passing under D-045. It is pure simulation baseline evidence only. It establishes no optimized
gain or mass, improvement, superiority, robustness outside the frozen population, Test result,
candidate, causal mechanism, physical estimate, controller/default change, firmware equivalence,
hardware transfer, safety, flight readiness or Sim2Real evidence. It does not authorize G3,
Loss v2, optimization, desaturation or real-flight work.
