# Day 22 checkpoint: saturation-explicit rollout diagnostic

## Boundary

- Work Order: `WO-GR-F0-002`
- Base: `97134b45cfce6efdf9915810ae01518a2f7b54c0`
- Generator/test commit: `2d90302f883de2ab2b03c60706314d1097f75d25`
- Platform: `cf2x_L250`
- Case: unchanged F0.1 Figure-8, H100, seed `20260724`, 500-Hz simulation, 100-Hz
  state-control, explicit Euler
- Candidate: unchanged Seed 05 update 5,000, only **provisional visualization candidate**
- Inputs: five checksum-verified accepted Day-21 payloads
- Day 10 / Day 13 / Test reads: zero
- New optimizer/training updates: zero

This is a worker review candidate pending independent Gradient-Lead acceptance. It is a separate
diagnostic path and does not modify the accepted Day-21 fallback.

## Unchanged gate and diagnostic result

The normal F0.1 zero-motor-saturation gate remains unchanged. A real rollout built from the
accepted Day-21 candidate again produced:

- Default: `0/400`, fraction `0.0`, status `NO_SATURATION_PRESENT`;
- Seed 05: `11/400`, fraction `0.027499999850988388`, status
  `DIAGNOSTIC_ONLY_SATURATION_PRESENT`;
- normal candidate acceptance: `FAIL_ZERO_MOTOR_SATURATION_GATE`;
- accepted candidate: false; flight ready: false.

All eleven Seed-05 decisions are upper-bound saturation on motor 0, indices 48 through 58. They
form one contiguous interval with boundary times `0.48–0.59 s`, state times `0.49–0.59 s`, and
duration `0.11 s`. There are no lower-bound events and no events on motors 1–3. Every bool is
attributed exactly once; per-motor counts sum to the total and the reconstructed float32 fraction
equals `tracking_loss_per_case` exactly.

The complete finite reference, actual trajectory, signed/norm error, tracking metrics, commands,
thresholds, decisions and intervals are in `saturation_diagnostic.json`. The two figures render
only those final JSON arrays, visibly mark the saturation, and carry
`DIAGNOSTIC ONLY — SATURATION PRESENT — NOT FLIGHT READY`.

## Verification

- Focused Pytest: 40 tests passed.
- Ruff lint: PASS.
- Ruff format check: PASS.
- `git diff --check`: PASS.
- Accepted Day-21 checksum index: 9/9 PASS.
- Day-22 checksum index: 5/5 PASS.
- Two fresh Day-22 generations: byte-identical to each other and the final package.
- Normal F0.1 regression: accepted Day-21 candidate enters the real frozen trace builder,
  reproduces Default `0.0` and Seed 05 `0.027499999850988388`, then stops at the unchanged
  technical gate without output.
- Existing F0.1 fallback branch and CLI are unchanged by diff; existing F0.1 tests pass.
- Neither full normal nor fallback F0.1 CLI was executed, because both would reread protected
  Day-13 inputs. Provenance records this nonexecution and the actual replacement method exactly.
- Both figures were inspected at original resolution and are legible; labels, red interval/point
  marks, commands and thresholds are visible.

The six files total 782,991 bytes. `SHA256SUMS` covers the other five and hashes to
`8fa0b309cea5d4903675527625e8a71cb92b831ba6d6155ab37b967adaee3c68`.

## Claim boundary

The descriptive one-case RMSE is lower for Seed 05, but it is inseparable from status
`DIAGNOSTIC_ONLY_SATURATION_PRESENT` and the failed gate. This package proves only the exact
trajectory/error/command/saturation behavior of the frozen `cf2x_L250` simulation case. It does
not prove superiority, robustness, convergence, Test performance, candidate acceptance, safety,
flight readiness, firmware or hardware validity, `cf21B_500` applicability, or Sim2Real transfer.
