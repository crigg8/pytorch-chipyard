#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_gemmini_autotune.sh --artifact-dir=<path>

Options:
  --artifact-dir=PATH  Output directory. Default:
                       examples/artifact-gemmini-max-autotune/gemmini
                       This script only generates compiler artifacts. Build ELF
                       files later with scripts/build-chipyard-elves.sh on the
                       local Chipyard/FireSim host.
  -h, --help           Show this help.
EOF
}

artifact_dir=""
backend=gemmini

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      [[ "$#" -ge 2 ]] || pc_usage_error "--artifact-dir requires a value"
      artifact_dir="$2"
      shift 2
      ;;
    --artifact-dir=*)
      artifact_dir="${1#--artifact-dir=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      pc_usage_error "unknown argument '$1'"
      ;;
  esac
done

pc_prepare_environment

export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=1

script_path="${PC_REPO_ROOT}/examples/gemmini-max-autotune.py"
suffix="gemmini-max-autotune/gemmini"
storage_suffix="$(pc_artifact_storage_suffix "${suffix}")"
output_dir="${artifact_dir:-${PC_REPO_ROOT}/examples/${storage_suffix}}"
pc_write_artifact_workload_hint "${output_dir}" "${suffix}"
pc_write_artifact_build_plan "${output_dir}" "${backend}" 4

pc_run_compile_once "${backend}" "${output_dir}" "gemmini-max-autotune" "${script_path}"

pc_log "done; build ELF files on the local Chipyard host with scripts/build-chipyard-elves.sh"
