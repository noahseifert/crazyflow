# Day 20 checkpoint: Stage-2 loss-term diagnostics

## Boundary

- Work Order: `WO-GR-S2-001`
- Branch/worktree: `codex/wo-gr-s2-001` at
  `/tmp/crazyflow-gradient-research-wo-gr-s2-001`
- Base/parent: `bfce9fcf04f97a609f569a6c9d95ea4f328a1cee`
- Intended enclosing commit subject: `research: instrument stage-2 loss diagnostics`
- Platform/cases: `cf2x_L250`; Figure-8/train and circle/validation
- Simulation/control: 500 Hz / 100 Hz; H20; seed `20260724`
- Gain state/updates: Crazyflow default / zero optimizer updates

The tranche did not open Test, read protected H100/Day-10 primary data, run an optimizer, change a
gain bound/default/loss weight, add a gain group, switch platform, or perform firmware/hardware
work.

## Implementation result

`tracking_loss_terms_per_case` exposes the six existing position, velocity, effort, smoothness,
terminal, and altitude terms as raw value, normalization divisor, normalized value, weight, and
weighted contribution. `tracking_loss_diagnostics_per_case` returns one term Jacobian for each
of `kp_xy`, `kp_z`, `kd_xy`, and `kd_z`. The recorded variable space is
`unconstrained_inputs_to_sigmoid_bounded_stage1_gains`; no physical-unit gradient is implied.

The existing total arithmetic, metric keys, default gains, and bounds remain unchanged. The new
H20 record reports:

| Split/trajectory | Total loss | Contribution sum minus total |
|---|---:|---:|
| Train/Figure-8 | `0.0022911166306585073` | `-7.628058774911128e-12` |
| Validation/circle | `0.0011237586149945855` | `2.7859561662832433e-11` |

All loss values, technical metrics, and 48 term/gain derivatives are finite. Motor saturation,
zero-thrust gate, floor clip, and nonfinite-state fractions are zero in both cases. The plots were
visually checked and are legible.

## Artifact record

| Path | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/day20-stage2-loss-diagnostics/loss_diagnostics.json` | 17,765 | `9f4821b804b928e5c95cc3fc7aef6cfa6f96e826de8f109374321cc3abee5151` |
| `artifacts/day20-stage2-loss-diagnostics/loss_contributions.png` | 48,750 | `8f54c8a83168d49d3810aeffc0862d027b45f6655a70c62651c8f43d8d18dfd3` |
| `artifacts/day20-stage2-loss-diagnostics/loss_term_gain_gradients.png` | 81,909 | `8c9f465ed83801783e0b10f7876306dc6f49263f822f78693e4da4ddc5d949ea` |
| `artifacts/day20-stage2-loss-diagnostics/SHA256SUMS` | 272 | `3ca5dab192030c12245945152d775bc1bf2b9a77c49da5cd7ecd68eede4c254a` |

The two PNGs are generated exclusively from the plot data re-read from
`loss_diagnostics.json`. Two independent writes in the unit test produce byte-identical JSON,
PNGs, and checksum indexes. A second fresh CLI evaluation in a separate temporary directory also
matched all four final artifacts byte-for-byte (`diff -qr`: PASS).

## Verification

The worker used the repository-local environment from the live checkout while pinning imports to
the isolated worktree:

```text
PYTHONPATH=/tmp/crazyflow-gradient-research-wo-gr-s2-001 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/crazyflow-gradient-stage2-matplotlib \
/home/noah3/bachelorarbeit/crazyflow-gradient-research/.venv/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/unit/test_mellinger_loss_diagnostics.py \
  tests/unit/test_mellinger_tracking.py \
  tests/unit/test_mellinger_gain_optimization.py
```

Result: `26 passed in 118.28s`.

The generation command reported Train loss `0.0022911166`, Validation loss `0.0011237586`, and
all eight machine-readable validations true. Ruff lint/format, checksum, diff, allowlist, commit,
and live-checkout protection results are recorded in the Work Order handoff.

## Claim boundary and next gate

This checkpoint proves only that term-resolved diagnostics and presentation plots can be generated
deterministically for the fixed H20 default-gain infrastructure smoke. It does not prove
convergence, plateau behavior, generalization, robustness, improved gains, controller
superiority, firmware equivalence, hardware safety, Sim2Real transfer, or `cf21B_500`
applicability.

Any next tranche requires a new bounded Work Order. Additional gain groups, tied-vs-split x/y
ablation, moderate/dynamic/Near-Limit trajectories, long-run continuation and convergence
metrics, platform-parameter reconciliation, candidate export, firmware transfer, and real flight
remain outside this checkpoint.
