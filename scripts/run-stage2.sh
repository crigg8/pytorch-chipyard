#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

log() {
  printf '[stage2] %s\n' "$*"
}

die() {
  printf '[stage2][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-stage2.sh [options]

Run the Stage 2 host workflow in order:
  1. Build model-<N>core.elf files from Stage 1 artifacts.
  2. Package FireMarshal and FireSim workload files.
  3. Build/install FireMarshal images.
  4. Run FireSim workloads and collect results.
  5. Generate figures.

Options:
  --chipyard-env=PATH   Local Chipyard env.sh. Passed to build-chipyard-elves.sh.
  --workload=LIST       Run selected FireSim workload(s). May be repeated.
  --skip-elves          Skip model-<N>core.elf generation.
  --skip-package        Skip FireMarshal workload/package generation.
  --skip-images         Skip FireMarshal image build/install.
  --skip-firesim        Skip FireSim execution/collection.
  --skip-plot           Skip figure generation.
  -h, --help            Show this help.

Environment:
  CHIPYARD_DIR must point to the local Chipyard checkout before sourcing
  scripts/env.sh, unless all derived paths are provided explicitly.
EOF
}

chipyard_env_arg=""
workload_args=()
skip_elves=0
skip_package=0
skip_images=0
skip_firesim=0
skip_plot=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --chipyard-env)
      [[ "$#" -ge 2 ]] || die "--chipyard-env requires a value"
      chipyard_env_arg="$2"
      shift 2
      ;;
    --chipyard-env=*)
      chipyard_env_arg="${1#--chipyard-env=}"
      shift
      ;;
    --workload)
      [[ "$#" -ge 2 ]] || die "--workload requires a value"
      workload_args+=("--workload=$2")
      shift 2
      ;;
    --workload=*)
      workload_args+=("$1")
      shift
      ;;
    --skip-elves)
      skip_elves=1
      shift
      ;;
    --skip-package)
      skip_package=1
      shift
      ;;
    --skip-images)
      skip_images=1
      shift
      ;;
    --skip-firesim)
      skip_firesim=1
      shift
      ;;
    --skip-plot)
      skip_plot=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument '$1'; pass --help for usage"
      ;;
  esac
done

if [[ -n "${chipyard_env_arg}" ]]; then
  export CHIPYARD_ENV_PATH="${chipyard_env_arg}"
fi

set +u
source "${SCRIPT_DIR}/env.sh"
set -u

cd "${REPO_ROOT}"

chipyard_env_args=()
if [[ -n "${chipyard_env_arg}" ]]; then
  chipyard_env_args=(--chipyard-env="${chipyard_env_arg}")
fi

if [[ "${skip_elves}" -eq 0 ]]; then
  log "building ELFs from examples"
  bash "${SCRIPT_DIR}/build-chipyard-elves.sh" \
    "${chipyard_env_args[@]}"
fi

if [[ "${skip_package}" -eq 0 ]]; then
  log "packaging FireMarshal/FireSim workloads from examples"
  bash "${SCRIPT_DIR}/package-firemarshal-workload.sh"
fi

if [[ "${skip_images}" -eq 0 ]]; then
  log "building and installing FireMarshal images"
  bash "${SCRIPT_DIR}/build-firemarshal-images.sh"
fi

if [[ "${skip_firesim}" -eq 0 ]]; then
  firesim_args=("${workload_args[@]}")
  log "running FireSim workloads"
  bash "${SCRIPT_DIR}/run-firesim-workloads.sh" "${firesim_args[@]}"
fi

if [[ "${skip_plot}" -eq 0 ]]; then
  log "generating figures"
  bash "${SCRIPT_DIR}/figure/plot_results.sh"
fi

log "done"
