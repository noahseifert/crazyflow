# Unclipped two-stage allocation presentation handoff

## Fixed status and gate

This is a review-candidate worker result with status `DIAGNOSTIC_ONLY_SATURATION_PRESENT`. The unchanged gate remains
`FAIL_ZERO_MOTOR_SATURATION_GATE`: Default is 0/400 and Seed 05 is
11/400, only motor 0 upper at indices 48–58 and
interval [0.48, 0.59) s. Nothing here accepts a candidate or changes the controller.

## Exact two-stage result

- Stage A motor clipping: 11 Seed-05 samples.
- Stage B created additional clipping: False with
  0 samples.
- Maximum absolute Stage-A wrench distortion by axis: collective_N=0.0168798863888, roll_Nm=0.000549102667719, pitch_Nm=0.00054910290055, yaw_Nm=0.000124081852846.
- The maximum Seed-05 motor-0 request is 74753.53125 PWM,
  0.136879891157 N and 24716.875 in the existing
  Crazyflow motor-speed convention. It exceeds the corresponding upper limits by
  9218.53125 PWM, 0.0168798919767 N and
  1501.09777832, at index 54.

## Figure use

`unclipped_allocation_diagnostic.png` is generated only from the validated JSON. It shows the raw
and clipped Seed-05 motor-0 request against the true limit, all four motors' Stage-A pre/postclip
commands around indices 48–58, signed platform-normalized Stage-A wrench distortion, and a compact
panel containing exact maximum request/exceedance, clip counts, failed status and claim boundary.
Use it only with the failed-gate status visible.

## DATA-AUDIT-01 boundary

The checksum-verified Day-19 derivative has ten seeds with 5000 Train, 5000 Validation and zero
Test rows each (100000 Train/Validation total), and per-seed/panel registered saturation maxima of
0.0. The analyzer source rejects Test rows, requires complete alternating order and aggregates
per-seed then panel maxima. The analyzer was not executed and protected Day-13/Test data were not
opened. This does not provide motor traces, reserve or allocation distortion for those runs.

## Exact verbal caveat

> Der unveränderte Day-22-Simulationsfall wurde nur diagnostisch bis vor beide Clippingstufen zurückverfolgt. Seed 05 verlangt zeitweise mehr als die konfigurierte Motor-0-Grenze; das Null-Sättigungs-Gate bleibt deshalb FAIL. Die Zahlen sind weder Controllerfreigabe noch Sicherheits-, Hardware- oder Flugnachweis.

## Claim boundary

Pure `cf2x_L250` simulation diagnostic only. No Test, tuning, optimization, candidate acceptance,
superiority, safety, firmware, hardware, real-flight or Sim2Real claim is made.
