#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIRECTORY}/pilot_config.json"
RUNS_DIRECTORY="${SCRIPT_DIRECTORY}/runs"
WALL_BUDGET_SECONDS=21600
RESUME_ARGUMENT=""
RUN_ROOT_ARGUMENT=""
DRY_RUN=false

usage() {
  printf 'Usage: %s [--resume CHECKPOINT] [--run-root ABSOLUTE_PATH] [--dry-run]\n' "$0"
}

while (($#)); do
  case "$1" in
    --resume)
      (($# >= 2)) || { usage >&2; exit 2; }
      RESUME_ARGUMENT="$2"
      shift 2
      ;;
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
test -x /usr/bin/time
command -v timeout >/dev/null
command -v setsid >/dev/null

CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_HEAD="$(git rev-parse HEAD)"
if [[ "${CURRENT_BRANCH}" != "research/differentiable-mellinger" ]]; then
  printf 'Refusing branch %s; expected research/differentiable-mellinger\n' "${CURRENT_BRANCH}" >&2
  exit 1
fi

CONFIG_FINGERPRINT="$(
  .venv/bin/python -c \
    'import sys; from pathlib import Path; from crazyflow.control.mellinger.research import fingerprint, load_config; print(fingerprint(load_config(Path(sys.argv[1]))))' \
    "${CONFIG_PATH}" | tail -n 1
)"
CONFIG_SHA256="$(sha256sum "${CONFIG_PATH}" | cut -d ' ' -f 1)"

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

RUNNER_COMMAND=(
  .venv/bin/python
  examples/jax/mellinger_domain_randomized_optimization.py
  --config "${CONFIG_PATH}"
  --output-dir "${RUNNER_OUTPUT}"
)
if [[ -n "${RESUME_ARGUMENT}" ]]; then
  RESUME_PATH="$(realpath -- "${RESUME_ARGUMENT}")"
  test -f "${RESUME_PATH}"
  RUNNER_COMMAND+=(--resume "${RESUME_PATH}")
fi
TIMED_COMMAND=(
  /usr/bin/time -v -o "${RUN_ROOT}/time.txt"
  timeout --signal=TERM --kill-after=30s "${WALL_BUDGET_SECONDS}s"
  "${RUNNER_COMMAND[@]}"
)

if "${DRY_RUN}"; then
  printf 'dry_run=pass\n'
  printf 'git_head=%s\n' "${CURRENT_HEAD}"
  printf 'config_fingerprint=%s\n' "${CONFIG_FINGERPRINT}"
  printf 'run_root=%s\n' "${RUN_ROOT}"
  printf 'command='
  printf '%q ' "${TIMED_COMMAND[@]}"
  printf '\n'
  exit 0
fi

SOURCE_STATUS="$(git status --porcelain --untracked-files=all -- crazyflow examples)"
if [[ -n "${SOURCE_STATUS}" ]]; then
  printf 'Refusing dirty executable-source scope:\n%s\n' "${SOURCE_STATUS}" >&2
  exit 1
fi
git ls-files --error-unmatch -- "${CONFIG_PATH#${REPOSITORY_ROOT}/}" >/dev/null
git diff --quiet HEAD -- "${CONFIG_PATH#${REPOSITORY_ROOT}/}" || {
  printf 'Refusing modified pilot config: %s\n' "${CONFIG_PATH}" >&2
  exit 1
}

if [[ -e "${RUN_ROOT}" ]]; then
  printf 'Refusing existing run root: %s\n' "${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p -- "${RUN_ROOT}"
{
  printf 'schema_version=crazyflow.sprint8_launch_manifest.v1\n'
  printf 'started_at_utc=%s\n' "$(date -u --iso-8601=seconds)"
  printf 'repository_root=%s\n' "${REPOSITORY_ROOT}"
  printf 'branch=%s\n' "${CURRENT_BRANCH}"
  printf 'git_head=%s\n' "${CURRENT_HEAD}"
  printf 'config_path=%s\n' "${CONFIG_PATH}"
  printf 'config_sha256=%s\n' "${CONFIG_SHA256}"
  printf 'config_fingerprint=%s\n' "${CONFIG_FINGERPRINT}"
  printf 'wall_budget_seconds=%s\n' "${WALL_BUDGET_SECONDS}"
  printf 'resume_checkpoint=%s\n' "${RESUME_ARGUMENT:-none}"
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

printf 'run_root=%s\n' "${RUN_ROOT}"
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
