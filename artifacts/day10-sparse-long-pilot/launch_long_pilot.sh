#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIRECTORY}/pilot_config.json"
RELEASE_GATE_PATH="${SCRIPT_DIRECTORY}/release_gate.json"
RUNS_DIRECTORY="${SCRIPT_DIRECTORY}/runs"
WALL_BUDGET_SECONDS=7200
RUN_ROOT_ARGUMENT=""
DRY_RUN=false

usage() {
  printf 'Usage: %s [--run-root ABSOLUTE_PATH] [--dry-run]\n' "$0"
}

while (($#)); do
  case "$1" in
    --run-root)
      (($# >= 2)) || { usage >&2; exit 2; }
      RUN_ROOT_ARGUMENT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd -- "${REPOSITORY_ROOT}"
test -x .venv/bin/python
test -f "${CONFIG_PATH}"
test -f "${RELEASE_GATE_PATH}"
test -x /usr/bin/time
command -v timeout >/dev/null
command -v setsid >/dev/null

CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_HEAD="$(git rev-parse HEAD)"
if [[ "${CURRENT_BRANCH}" != "research/differentiable-mellinger" ]]; then
  printf 'Refusing branch %s; expected research/differentiable-mellinger\n' "${CURRENT_BRANCH}" >&2
  exit 1
fi

readarray -t RELEASE_VALUES < <(
  .venv/bin/python -c \
    'import json, sys; from pathlib import Path; from crazyflow.control.mellinger.research import fingerprint, load_config; from crazyflow.control.mellinger.research.runner import _git_provenance; c=load_config(Path(sys.argv[1])); g=json.loads(Path(sys.argv[2]).read_text()); print(g["decision"]); print(fingerprint(c)); print(c.optimizer.steps); print(c.optimizer.checkpoint_interval); print(g["prepared_config"]["canonical_fingerprint"]); print(_git_provenance(Path(sys.argv[3]))["source_fingerprint"]); print(g["implementation"]["source_fingerprint"])' \
    "${CONFIG_PATH}" "${RELEASE_GATE_PATH}" "${REPOSITORY_ROOT}" | tail -n 7
)
RELEASE_DECISION="${RELEASE_VALUES[0]}"
CONFIG_FINGERPRINT="${RELEASE_VALUES[1]}"
CONFIGURED_UPDATES="${RELEASE_VALUES[2]}"
CHECKPOINT_INTERVAL="${RELEASE_VALUES[3]}"
GATED_CONFIG_FINGERPRINT="${RELEASE_VALUES[4]}"
CURRENT_SOURCE_FINGERPRINT="${RELEASE_VALUES[5]}"
GATED_SOURCE_FINGERPRINT="${RELEASE_VALUES[6]}"
CONFIG_SHA256="$(sha256sum "${CONFIG_PATH}" | cut -d ' ' -f 1)"

if [[ "${RELEASE_DECISION}" != "GO" ]]; then
  printf 'Refusing start: Sprint-10 release gate is %s\n' "${RELEASE_DECISION}" >&2
  exit 3
fi
if [[ "${CONFIGURED_UPDATES}" != "5000" || "${CHECKPOINT_INTERVAL}" != "100" ]]; then
  printf 'Refusing config: expected 5000 updates and checkpoint interval 100\n' >&2
  exit 3
fi
if [[ "${CONFIG_FINGERPRINT}" != "${GATED_CONFIG_FINGERPRINT}" ]]; then
  printf 'Refusing config fingerprint outside release gate\n' >&2
  exit 3
fi
if [[ "${CURRENT_SOURCE_FINGERPRINT}" != "${GATED_SOURCE_FINGERPRINT}" ]]; then
  printf 'Refusing source fingerprint outside release gate\n' >&2
  exit 3
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "${RUN_ROOT_ARGUMENT}" ]]; then
  [[ "${RUN_ROOT_ARGUMENT}" = /* ]] || {
    printf '%s\n' '--run-root must be an absolute path' >&2
    exit 2
  }
  RUN_ROOT="${RUN_ROOT_ARGUMENT}"
else
  RUN_ROOT="${RUNS_DIRECTORY}/${TIMESTAMP}-${CURRENT_HEAD:0:12}"
fi
RUNNER_OUTPUT="${RUN_ROOT}/runner-output"
TIMED_COMMAND=(
  /usr/bin/time -v -o "${RUN_ROOT}/time.txt"
  timeout --signal=TERM --kill-after=30s "${WALL_BUDGET_SECONDS}s"
  .venv/bin/python
  examples/jax/mellinger_domain_randomized_optimization.py
  --config "${CONFIG_PATH}"
  --output-dir "${RUNNER_OUTPUT}"
)

printf 'release_decision=%s\n' "${RELEASE_DECISION}"
printf 'git_head=%s\n' "${CURRENT_HEAD}"
printf 'config_fingerprint=%s\n' "${CONFIG_FINGERPRINT}"
printf 'source_fingerprint=%s\n' "${CURRENT_SOURCE_FINGERPRINT}"
printf 'checkpoint_interval=%s\n' "${CHECKPOINT_INTERVAL}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'command='
printf '%q ' "${TIMED_COMMAND[@]}"
printf '\n'

if "${DRY_RUN}"; then
  printf 'dry_run=pass\n'
  exit 0
fi

WORKTREE_STATUS="$(git status --porcelain --untracked-files=all)"
if [[ -n "${WORKTREE_STATUS}" ]]; then
  printf 'Refusing non-clean pre-launch worktree:\n%s\n' "${WORKTREE_STATUS}" >&2
  exit 1
fi
for tracked_path in "${CONFIG_PATH}" "${RELEASE_GATE_PATH}"; do
  relative_path="${tracked_path#${REPOSITORY_ROOT}/}"
  git ls-files --error-unmatch -- "${relative_path}" >/dev/null
  git diff --quiet HEAD -- "${relative_path}" || {
    printf 'Refusing modified release input: %s\n' "${tracked_path}" >&2
    exit 1
  }
done
if [[ -e "${RUN_ROOT}" ]]; then
  printf 'Refusing existing run root: %s\n' "${RUN_ROOT}" >&2
  exit 1
fi

mkdir -p -- "${RUN_ROOT}"
{
  printf 'schema_version=crazyflow.sprint10_sparse_launch_manifest.v1\n'
  printf 'started_at_utc=%s\n' "$(date -u --iso-8601=seconds)"
  printf 'repository_root=%s\n' "${REPOSITORY_ROOT}"
  printf 'branch=%s\n' "${CURRENT_BRANCH}"
  printf 'git_head=%s\n' "${CURRENT_HEAD}"
  printf 'config_path=%s\n' "${CONFIG_PATH}"
  printf 'config_sha256=%s\n' "${CONFIG_SHA256}"
  printf 'config_fingerprint=%s\n' "${CONFIG_FINGERPRINT}"
  printf 'source_fingerprint=%s\n' "${CURRENT_SOURCE_FINGERPRINT}"
  printf 'release_gate=%s\n' "${RELEASE_DECISION}"
  printf 'configured_updates=%s\n' "${CONFIGURED_UPDATES}"
  printf 'checkpoint_interval=%s\n' "${CHECKPOINT_INTERVAL}"
  printf 'wall_budget_seconds=%s\n' "${WALL_BUDGET_SECONDS}"
  printf 'automatic_resume=false\n'
  printf 'command='
  printf '%q ' "${TIMED_COMMAND[@]}"
  printf '\n'
} >"${RUN_ROOT}/launch_manifest.txt"

{
  uname -a
  lscpu
  free -h
  .venv/bin/python --version
  .venv/bin/python -c \
    'import importlib.metadata as m, jax, platform; print("platform=" + platform.platform()); print("jax_backend=" + jax.default_backend()); print("jax_devices=" + repr([str(d) for d in jax.devices()])); print("packages=" + repr({n:m.version(n) for n in ("jax","jaxlib","numpy","optax")}))'
} >"${RUN_ROOT}/environment.txt" 2>&1

printf 'stdout_log=%s\n' "${RUN_ROOT}/stdout.log"
printf 'stderr_log=%s\n' "${RUN_ROOT}/stderr.log"
printf 'runner_output=%s\n' "${RUNNER_OUTPUT}"

PROCESS_GROUP_ID=""
forward_termination() {
  if [[ -n "${PROCESS_GROUP_ID}" ]]; then
    kill -TERM -- "-${PROCESS_GROUP_ID}" 2>/dev/null || true
  fi
}
trap forward_termination INT TERM

setsid "${TIMED_COMMAND[@]}" >"${RUN_ROOT}/stdout.log" 2>"${RUN_ROOT}/stderr.log" &
PROCESS_GROUP_ID=$!
printf '%s\n' "${PROCESS_GROUP_ID}" >"${RUN_ROOT}/process_group.pid"

set +e
wait "${PROCESS_GROUP_ID}"
PROCESS_STATUS=$?
set -e
trap - INT TERM

{
  printf 'ended_at_utc=%s\n' "$(date -u --iso-8601=seconds)"
  printf 'exit_status=%s\n' "${PROCESS_STATUS}"
} >"${RUN_ROOT}/launcher_status.txt"

if ((PROCESS_STATUS != 0)); then
  printf 'pilot_status=failed_or_interrupted exit_status=%s run_root=%s\n' \
    "${PROCESS_STATUS}" "${RUN_ROOT}" >&2
  exit "${PROCESS_STATUS}"
fi
printf 'pilot_status=completed run_root=%s\n' "${RUN_ROOT}"
