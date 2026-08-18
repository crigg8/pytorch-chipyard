#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
TVM_AE_ROOT="${TABLE2_TVM_AE_ROOT:-/home/ae/tvm-gemmini-ae}"
source "${TVM_AE_ROOT}/scripts/env.sh"
if [[ -n "${TABLE2_TVM_BUILD_DIR:-}" ]]; then
  TVM_BUILD_DIR="${TABLE2_TVM_BUILD_DIR}"
  export TVM_BUILD_DIR
fi
export TVM_LIBRARY_PATH="${TVM_BUILD_DIR}"
export LD_LIBRARY_PATH="${TVM_BUILD_DIR}:${LD_LIBRARY_PATH:-}"
export TABLE2_TVM_GEMMINI_INCLUDE="${TABLE2_TVM_GEMMINI_INCLUDE:-${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests/include}"
export TABLE2_TVM_CHIPYARD_COMMIT="$(git -C "${CHIPYARD_DIR}" rev-parse HEAD)"

kernel=""
spec=""
output_dir=""
force=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --kernel=*) kernel="${1#*=}"; shift ;;
    --kernel) [[ "$#" -ge 2 ]] || exit 2; kernel="$2"; shift 2 ;;
    --spec=*) spec="${1#*=}"; shift ;;
    --spec) [[ "$#" -ge 2 ]] || exit 2; spec="$2"; shift 2 ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --output-dir) [[ "$#" -ge 2 ]] || exit 2; output_dir="$2"; shift 2 ;;
    --force) force=1; shift ;;
    -h | --help)
      printf '%s\n' 'Usage: compile-kernel.sh --kernel=ID --spec=PATH --output-dir=PATH'
      exit 0
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n "${kernel}" && -n "${spec}" && -n "${output_dir}" ]] || exit 2
[[ -f "${spec}" ]] || { printf 'kernel specification not found: %s\n' "${spec}" >&2; exit 1; }
[[ -f "${TABLE2_TVM_GEMMINI_INCLUDE}/gemmini_params.h" ]] || {
  printf 'Chipyard Gemmini headers not found: %s\n' "${TABLE2_TVM_GEMMINI_INCLUDE}" >&2
  exit 1
}
[[ -f "${TVM_BUILD_DIR}/libtvm.so" ]] || {
  printf 'LLVM-enabled TVM build not found: %s/libtvm.so\n' "${TVM_BUILD_DIR}" >&2
  printf 'Run scripts/tvm-gemmini-table2/prepare-tvm.sh first.\n' >&2
  exit 1
}
args=(--kernel "${kernel}" --spec "${spec}" --output-dir "${output_dir}")
[[ "${force}" -eq 0 ]] || args+=(--force)
exec "${TVM_ENV}/bin/python" "${SCRIPT_DIR}/compile-kernel.py" "${args[@]}"
