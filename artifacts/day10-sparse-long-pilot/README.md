# Sprint 10 sparse 5,000-update long pilot

## Release

**GO for exactly one detached H100×4 Train/Validation CPU run with 5,000 updates and checkpoint
interval 100.** No second run, automatic Resume, extension, H200/H400, seed loop, or Test action is
authorized.

The new config differs from the Sprint-9 5,000-update config only by its run ID and explicit
`optimizer.checkpoint_interval=100`. Missing intervals retain the historical value 1 and their
historical canonical config fingerprints. Metrics are durably appended for every completed
update. Checkpoints remain atomic, checksummed, complete Resume states and occur at 100, 200, …,
5,000; the configured final step is always written exactly once.

The real 210-update H100×4 smoke completed in 55.99 seconds with checkpoints at 100/200/210,
Peak-RSS 1,309,284 KiB, zero swap, finite metrics, passing technical gates, and no Test access.
Exact fresh-process Resume 200→210 passed at zero tolerance. The conservative projection is
2,289.55 seconds (38.16 minutes), 437,666,785 expected bytes, and 875,333,570 bytes under the
declared factor-two storage bound, well below 7,200 seconds and 2 GiB.

## Exact start

```bash
cd /home/noah3/bachelorarbeit/crazyflow-gradient-research
tmux new-session -d -s crazyflow-s10-sparse-long \
  'cd /home/noah3/bachelorarbeit/crazyflow-gradient-research && exec artifacts/day10-sparse-long-pilot/launch_long_pilot.sh'
```

The launcher requires a completely clean worktree and tracked, unmodified GO gate/config. It
uses `/usr/bin/time -v`, `timeout --signal=TERM --kill-after=30s 7200s`, a unique UTC/HEAD run
root, separate stdout/stderr, and a controlled process group.

## Status, evaluation, and controlled stop

```bash
tmux capture-pane -pt crazyflow-s10-sparse-long -S -40
RUN_ROOT="$(find artifacts/day10-sparse-long-pilot/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
cat "$RUN_ROOT/launch_manifest.txt"
ps -o pid,ppid,pgid,etimes,rss,pcpu,pmem,args -p "$(cat "$RUN_ROOT/process_group.pid")"
tail -n 30 "$RUN_ROOT/stdout.log" "$RUN_ROOT/stderr.log"
.venv/bin/python artifacts/day10-sparse-long-pilot/inspect_long_pilot.py "$RUN_ROOT"
```

Controlled termination, only if required:

```bash
kill -TERM -- "-$(cat "$RUN_ROOT/process_group.pid")"
```

Morning/final evaluation uses the same inspector. `PASS` requires all 50 checkpoint payloads,
10,000 ordered Train/Validation metric rows, 5,000 completed updates, finite values, passing
technical gates, successful summary/resource exit, and no Test metrics/opening. `ACTIVE` means the
process is live. `RESUMABLE` preserves a valid last complete checkpoint but does not authorize an
automatic continuation in this sprint.
