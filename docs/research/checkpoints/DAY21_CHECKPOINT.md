# Day 21 checkpoint: Friday existing-evidence fallback

## Boundary

- Work Order: `WO-GR-F0-001`
- Base: `4eb411d5f181baf1742ea0cf980199f5e9a8e3b1`
- Generator/test commit: `2499aad364a44eadcc259005f4c18d8a21acb332`
- Platform of accepted existing evidence: `cf2x_L250`
- Inputs: accepted Day-19 ten-seed H100 synthesis, ten protected live run indexes/summaries/
  times/bounded checkpoint tails, and accepted Day-20 loss diagnostics
- Test: unopened; no test metric or manifest read
- New optimizer/training updates: zero

The protected live checkout and Day-10/Day-13 roots remained unchanged. Current checks covered
the tracked Day-19/Day-20 checksum indexes and the ten small `RUN_SHA256SUMS` identities. The
accepted Day-19 full-payload integrity was inherited; the 4.38-GB primary payload was not rehashed.

## Candidate proof and package result

The provisional visualization rule selected the unique smallest existing selected fixed-
Validation loss: Seed 05, update 5,000, loss `0.002523899544030428`. Day-19 synthesis, the Seed-05
run summary and the bounded tail of checkpoint 005000 agree within `rtol=1e-7`, `atol=1e-9` on:

- `kp_xy=0.32308459281921387`
- `kp_z=2.310833215713501`
- `kd_xy=0.7022475004196167`
- `kd_z=1.0895649194717407`

It is labelled only **provisional visualization candidate**. It is not a final or flight
candidate, and gains were not averaged.

The strict normal generator path stopped before output because the requested comparison did not
meet the unchanged zero-motor-saturation acceptance gate. The one pre-authorized correction cycle
added an explicit tested fallback without changing the gate, candidate, trajectory, platform, or
controller. The worker-result package status is `WITHHELD_TECHNICAL_GATE`: no rollout arrays,
replacement values, `rollout_trajectory.png`, or `rollout_tracking_error.png` are present.

## Existing evidence delivered

- ten complete Train and fixed-Validation histories with 5,000 rows each;
- ten available GNU process wall times totaling `9286.5` seconds, explicitly not calendar/GPU/
  parallel time;
- 10/10 technical completion/no documented abort, kept separate from the accepted numerical
  distribution of selected fixed-Validation loss;
- 50 checkpoint-selected physical-gain records per seed and ten selected end-gain vectors;
- four visually checked H100 figures and two byte-identical accepted Stage-2 figures;
- machine-readable evidence, exact provenance, presentation handoff and nine payload checksums.

The new H100 figures were visually checked and have legible titles, axes, legends and scope
labels. The copied Stage-2 figures retain their accepted H20/default/zero-update boundary.

## Verification

The focused suite completed with `31 passed in 26.58s`. Ruff lint and format checks passed. The
strict normal command exited `1` at the unchanged rollout gate and produced no output. Two fresh
fallback generations in distinct `/tmp` targets exited `0`, `diff -qr` reported no difference,
both nine-entry checksum indexes passed, and the final package is byte-identical to them.

The final output contains ten files totaling 5,377,622 bytes. Its checksum index hashes to
`b64e0523600b5a24c9dcd58ec7fa45b42143f5ccdbf5b7a49cc676f21a4c8a21`.

## Claim boundary and open item

This package proves only the recorded ten-run H100 histories, process durations, technical
completion, accepted numerical variation, selected gain records, and accepted Stage-2
infrastructure diagnostics under their documented simulation contracts. It does not prove
convergence, generalization, robustness, superiority, Test performance, firmware/hardware
validity, flight safety, `cf21B_500` applicability, or Sim2Real transfer.

The supervisor's requested trajectory-before/after figure remains openly missing. Any other
candidate, case, threshold, platform or rollout requires a new main-lead decision and Work Order.
