# Sprint 9: 5,000-update long-pilot preparation

## Decision

**NO-GO. The 5,000-update pilot was not started.** The scientific config is valid and unchanged
from Sprint 8 except for its run ID and `optimizer.steps=5000`, but the existing runner writes one
atomic checkpoint after every update and every checkpoint repeats the complete history. No
checkpoint-frequency setting exists.

The immutable Sprint-8 sample grows from 8,055 bytes at step 1 to 170,242 bytes at step 50. Its
near-exact linear size fit projects 5,000 checkpoint files totaling 41,408,829,822 bytes, a final
checkpoint of 16,555,498 bytes, and about 41,421,555,609 bytes over 5,011 total run files. Fits to
the observed complete-update interval growth project 8,032.99, 8,340.84, or 13,438.55 seconds,
depending on the retained measured range. These runtime projections are operational estimates,
not precise forecasts, but every measured-growth fit exceeds the hard 7,200-second limit.

The Sprint-8 run itself peaked at 1,252,988 KiB with zero swap and the filesystem had
1,007,043,264,512 available bytes at final preflight. Capacity and observed-memory headroom are
therefore not the blocker. The blocker is simultaneous compliance with exactly 5,000 completed
updates and at most two hours.

## Reproducible evaluation

The complete analysis, including every loss, gradient, gain, bound margin, gap, integrity field,
fingerprint, checkpoint size, and artifact SHA-256, is in `sprint8_analysis.json`. Reproduce it
without relying on checkout mtimes:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
.venv/bin/python artifacts/day9-long-pilot/analyze_sprint8_pilot.py \
  --run-dir artifacts/day8-night-pilot/runs/20260731T213826Z-2e16f365ecfd \
  --released-config artifacts/day8-night-pilot/pilot_config.json \
  --run-sha-index artifacts/day8-night-pilot/SPRINT8_RUN_SHA256SUMS \
  --checkpoint-timing-capture \
    artifacts/day9-long-pilot/sprint8_checkpoint_timing_capture.json \
  --output /tmp/sprint9-sprint8-analysis-reproduction.json
cmp artifacts/day9-long-pilot/sprint8_analysis.json \
  /tmp/sprint9-sprint8-analysis-reproduction.json
```

The checked 50-update result has best Train step 31 and best Validation/selected step 50. Train
loss changed from 0.0102038635 to 0.0090798726 (minimum 0.0090447003), Validation from
0.0100101354 to 0.0095826006, and gradient L2 from 0.0062514464 to 0.0055943080. Validation
decreases at every transition. Train has a post-hoc oscillation indicator but no plateau or
divergence indicator. No gain approaches a bound: the minimum normalized bound margin across all
updates and gains is 0.2001600. The final Validation-minus-Train gap is +0.0005027279. Every
checkpoint checksum, metric finite check, and technical gate passes; the Test split was not
opened or evaluated.

## Prepared but disabled operation

`pilot_config.json`, `launch_long_pilot.sh`, and `inspect_long_pilot.py` are complete. The launcher
contains the required `/usr/bin/time -v` and
`timeout --signal=TERM --kill-after=30s 7200s` boundary, unique run roots, full logs, process-group
termination, and the unchanged real runner. It reads `release_gate.json` and currently exits 3
before creating a run directory. Do not edit the gate manually and do not run this command:

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
tmux new-session -d -s crazyflow-s9-long \
  'cd /home/noah3/bachelorarbeit/crazyflow-gradient-research && exec artifacts/day9-long-pilot/launch_long_pilot.sh'
```

There is consequently no Sprint-9 session, PID, run path, log path, start time, morning
evaluation, or controlled-abort target. Read-only confirmation of that state is:

```bash
tmux has-session -t crazyflow-s9-long 2>/dev/null; echo "tmux_status=$? (1 means absent)"
pgrep -af 'mellinger_domain_randomized_optimization.py.*day9-long-pilot' || true
find artifacts/day9-long-pilot/runs -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null
```

A later explicitly authorized sprint may implement and prove configurable sparse atomic
checkpoints while retaining per-update metrics, complete Resume state, and a mandatory final
checkpoint. Sprint 9 does not authorize that runner change, a second config, a shorter substitute
run, a seed loop, H200/H400, Test evaluation, or automatic continuation.
