# Day 25 technical handoff: cf21B_500 robust baseline foundation

## Status

`PASS_ROBUST_FOUNDATION_BASELINE` under prospective D-045. G2.1 and G2.2 remain historically blocked/rejected. Integral contact is measured baseline behavior, not a technical pass. All inherited nonintegral gates pass. The first independent review of result `c4bbb57443836aecba52d72cb73a11ee4cbdb5dd` was NOT ACCEPTED; D-047 authorizes this sole transparency correction.

## Disclosed retained trajectory component

Every class contains the same fixed positive-z central climb: exact Float32 `0.06 m/s`, base displacement `0.06*(time-2 s)`, zero base acceleration/jerk, and exact product-rule derivatives through jerk under the same C3 smoothstep7 2 s entry / 4 s plateau / 2 s exit envelope as the Fourier parent. It is composed after Fourier class scaling and before the fixed center. D-047 changes no trajectory number. Absolute Z results apply only to trajectories containing this climb; neither climb nor controller mass is an isolated cause.

## Duration and measurement scope

References have 8 s parent support. Simulation is one continuous 6 s rollout from the parent start, with no reset at the half-open 2 s scored window after a 2/3/4 s warm-up. Finiteness, ground/floor and zero-thrust gates cover all 6 s. Torque, motor and Stage-B additional clipping, integral contact, tracking, Loss v1, reserve and wrench statements cover only the scored window.

## Negative physical-value-match finding

Across 16 paired episodes, mismatch mean scored Z-RMSE is `0.084947569 m` and physical-value-match mean is `0.114523803 m`. Negative-z integral contact rises from `593` to `1844` axis-samples. This is a negative finding, not improvement, candidate selection, or causal isolation.

## Mechanism and transfer boundary

Measured early-transient and scored arrays retain collective target-thrust-equivalent request, realized wrench, Z error/integral, motor command, and reserve. Separately, code inspection shows `state2attitude` uses `mass * (setpoint_acc - gravity_vec) + feedback`, followed by fixed `mass_thrust=132000` and nonlinear PWM-to-force mapping. That is code-path inference, not an isolated causal result. Controller mass is therefore a feedforward parameter in this chain, not a transferable physical-mass estimate or firmware/flight value.

## D-047 invariance comparator

`PASS_PRE_CORRECTION_C4_INVARIANCE` with explicit old side `c4bbb57443836aecba52d72cb73a11ee4cbdb5dd`. All `5465162` pre-existing numeric report paths, every retained parent digest/ID/seed/attempt/window, all old raw arrays/summaries/pair deltas/masses, and the pinned benchmark bytes are identical. Only new derived aggregates/metadata, corrected prose, stable generator provenance and checksums differ.

## D-048 post-integration metadata closure

The corrected scientific foundation review is `ACCEPTED`; D-048 changes metadata only. The deterministic integration target is `research/differentiable-mellinger`. The observed checkout branch is excluded from reproducible payload bytes because checkout branch is a mutable execution-context reference; generator_commit is the canonical immutable science/code identity. The canonical immutable science/code identity remains generator commit `2288ab2fe3b55f585fb9b14a8fe9d87c07eb6c1a`.

## Benchmark

Fresh CPU processes completed 4/16/32 worlds with finite consistent objective and gradient. The predeclared rule recommends `32` worlds for the next CPU batch only.

## Claim boundary

Pure Float32 Crazyflow simulation evidence. No optimizer update, Loss-v2, Test, GPU, candidate, superiority, safety, firmware, hardware, flight, or Sim2Real claim.
