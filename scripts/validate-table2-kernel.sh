#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

kernel=""
artifact_dir=""
spec="${REPO_ROOT}/benchmarks/table2-kernels.json"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --kernel=*) kernel="${1#*=}"; shift ;;
    --kernel) [[ "$#" -ge 2 ]] || pc_usage_error "--kernel requires a value"; kernel="$2"; shift 2 ;;
    --artifact-dir=*) artifact_dir="${1#*=}"; shift ;;
    --artifact-dir) [[ "$#" -ge 2 ]] || pc_usage_error "--artifact-dir requires a value"; artifact_dir="$2"; shift 2 ;;
    --spec=*) spec="${1#*=}"; shift ;;
    --spec) [[ "$#" -ge 2 ]] || pc_usage_error "--spec requires a value"; spec="$2"; shift 2 ;;
    *) pc_usage_error "unknown argument '$1'" ;;
  esac
done

[[ -n "${kernel}" && -n "${artifact_dir}" ]] || \
  pc_usage_error "--kernel and --artifact-dir are required"
pc_prepare_environment
PYTORCH_CHIPYARD_DUMP_PATH="${artifact_dir}" \
  python "${REPO_ROOT}/examples/table2-kernel.py" --validate \
  --kernel "${kernel}" --spec "${spec}"
