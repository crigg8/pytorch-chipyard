#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd -P)"

CHIPYARD_DIR="${CHIPYARD_DIR:?CHIPYARD_DIR must point to the target Chipyard checkout}"
LLVM_PROJECT_PATH="${LLVM_PROJECT_PATH:-${REPO_ROOT}/llvm-project}"
GEMMINI_PARAMS_H="${GEMMINI_PARAMS_H:-${CHIPYARD_DIR}/sims/firesim/deploy/results-build/gemmini_params.h}"
GEMMINI_TEST_ROOT="${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests"

find_riscv_gxx() {
  if [[ -n "${PYTORCH_CHIPYARD_RISCV_GXX:-}" ]]; then
    printf '%s\n' "${PYTORCH_CHIPYARD_RISCV_GXX}"
    return
  fi
  if [[ -n "${PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR:-}" ]]; then
    local root="${PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR%/}"
    local candidate
    for candidate in \
      "${root}/bin/riscv64-unknown-linux-gnu-g++" \
      "${root}/riscv64-unknown-linux-gnu-g++"; do
      if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return
      fi
    done
  fi
  command -v riscv64-unknown-linux-gnu-g++ || true
}

RISCV_GXX="$(find_riscv_gxx)"
[[ -n "${RISCV_GXX}" && -x "${RISCV_GXX}" ]] || {
  echo "could not find riscv64-unknown-linux-gnu-g++" >&2
  exit 1
}
RISCV_AR="${RISCV_GXX%g++}ar"
[[ -x "${RISCV_AR}" ]] || {
  echo "could not find matching RISC-V ar: ${RISCV_AR}" >&2
  exit 1
}
[[ -f "${GEMMINI_PARAMS_H}" ]] || {
  echo "Gemmini parameter header not found: ${GEMMINI_PARAMS_H}" >&2
  exit 1
}
[[ -f "${GEMMINI_TEST_ROOT}/include/gemmini.h" ]] || {
  echo "gemmini.h not found under ${GEMMINI_TEST_ROOT}/include" >&2
  exit 1
}

BUILD_DIR="${SCRIPT_DIR}/build"
OBJECT_PATH="${BUILD_DIR}/gemmini_mv.o"
LIBRARY_PATH="${SCRIPT_DIR}/libpytorch_chipyard.a"
mkdir -p "${BUILD_DIR}"
cp "${GEMMINI_TEST_ROOT}/include/gemmini.h" "${BUILD_DIR}/gemmini.h"
(
  cd "${BUILD_DIR}"
  patch --silent gemmini.h "${SCRIPT_DIR}/gemmini_fp32.patch"
)

"${RISCV_GXX}" \
  -include "${GEMMINI_PARAMS_H}" \
  -I"${BUILD_DIR}" \
  -I"${GEMMINI_TEST_ROOT}" \
  -I"${LLVM_PROJECT_PATH}/mlir/include" \
  -I"${LLVM_PROJECT_PATH}/llvm/include" \
  -march=rv64imafdc \
  -mabi=lp64d \
  -O2 \
  -std=gnu++17 \
  -c "${SCRIPT_DIR}/gemmini_mv.cpp" \
  -o "${OBJECT_PATH}"

"${RISCV_AR}" rcs "${LIBRARY_PATH}" "${OBJECT_PATH}"
echo "[custom-mv] wrote ${LIBRARY_PATH}"
