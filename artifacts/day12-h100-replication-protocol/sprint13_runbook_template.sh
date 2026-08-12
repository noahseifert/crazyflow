#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=/home/noah3/bachelorarbeit/crazyflow-gradient-research
PROTOCOL_ROOT=artifacts/day12-h100-replication-protocol
EXPECTED_RAW_BYTES_PER_SEED=437505563
SEED_01_MINIMUM_LOCAL_FREE_BYTES=20000000000
LOCAL_FREE_RESERVE_BYTES=10000000000

if [[ $# -lt 2 || $# -gt 3 || "$1" != "--dry-run" || ! "$2" =~ ^([1-9]|10|0[1-9])$ ]]; then
  echo "usage: $0 --dry-run SEED_INDEX(1..10) [PREVIOUS_RUN_ROOT]" >&2
  echo "This Sprint-12 template deliberately has no execution mode." >&2
  exit 3
fi

seed_number=$((10#$2))
printf -v seed_index '%02d' "$seed_number"
config_path="$PROTOCOL_ROOT/configs/seed-$seed_index.json"
planned_root="artifacts/day13-h100-replication/runs/seed-$seed_index/<UTC>-<GIT_HEAD>"
previous_run_root=${3-}

cd "$REPOSITORY_ROOT"
.venv/bin/python "$PROTOCOL_ROOT/verify_protocol.py"

if [[ $seed_number -eq 1 ]]; then
  minimum_local_free_bytes=$SEED_01_MINIMUM_LOCAL_FREE_BYTES
  if [[ -n $previous_run_root ]]; then
    echo "Seed 01 must not receive a previous run root." >&2
    exit 5
  fi
else
  remaining_seed_count=$((11 - seed_number))
  minimum_local_free_bytes=$((
    LOCAL_FREE_RESERVE_BYTES + EXPECTED_RAW_BYTES_PER_SEED * remaining_seed_count
  ))
  if [[ -z $previous_run_root ]]; then
    echo "Seed $seed_index requires PREVIOUS_RUN_ROOT for local SHA-256 reverification." >&2
    exit 5
  fi
  previous_run_root=$(realpath -e "$previous_run_root")
  if [[ $previous_run_root != "$REPOSITORY_ROOT/artifacts/day13-h100-replication/runs/"* ]]; then
    echo "Previous run root must remain inside the repository artifact area." >&2
    exit 5
  fi
  if [[ ! -f $previous_run_root/RUN_SHA256SUMS ]]; then
    echo "Missing previous local index: $previous_run_root/RUN_SHA256SUMS" >&2
    exit 5
  fi
  (cd "$previous_run_root" && sha256sum -c RUN_SHA256SUMS)
fi

available_local_bytes=$(df -B1 --output=avail . | awk 'NR == 2 {print $1}')
if [[ ! $available_local_bytes =~ ^[0-9]+$ ]]; then
  echo "Could not determine local free bytes." >&2
  exit 4
fi
if ((available_local_bytes < minimum_local_free_bytes)); then
  echo "Local capacity gate failed: available=$available_local_bytes required=$minimum_local_free_bytes" >&2
  exit 4
fi

echo "SPRINT-13 DRY-RUN TEMPLATE ONLY"
echo "seed_index=$seed_index"
echo "config_path=$config_path"
echo "planned_unique_run_root=$planned_root"
echo "execution_order=ascending, strictly sequential, maximum one research process"
echo "local_available_bytes=$available_local_bytes"
echo "local_required_bytes=$minimum_local_free_bytes"
echo "local_capacity_gate=PASS"
echo "external_storage_gate_required=false"
echo "precondition=previous local run/index immutable and SHA-256 reverified"
echo
echo "A separately reviewed Sprint-13 launcher must wrap exactly this runner shape:"
echo "/usr/bin/time -v -o <RUN_ROOT>/time.txt timeout --signal=TERM --kill-after=30s 7200s \\"
echo "  .venv/bin/python examples/jax/mellinger_domain_randomized_optimization.py \\"
echo "  --config $config_path --output-dir <RUN_ROOT>/runner-output"
echo
echo "Required before the next seed: validate 5,000 updates and 50 checkpoints; generate"
echo "<RUN_ROOT>/RUN_SHA256SUMS over every other run file; immediately run sha256sum -c"
echo "locally; retain the run unchanged in the repository artifact area; rerun the full"
echo "local index before the next seed. A missing file or mismatch blocks all later seeds."
echo "SHA-256 is not a backup; the accepted single internal-SSD loss risk remains."
echo "No Resume, seed loop, parallel launch, H200/H400, Test access, or automatic extension."
