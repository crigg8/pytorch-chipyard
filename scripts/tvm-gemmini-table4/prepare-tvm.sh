#!/usr/bin/env bash
set -euo pipefail

account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
fi
TVM_AE_ROOT="${TABLE4_TVM_AE_ROOT:-${HOME}/tvm-gemmini-ae}"
account_chipyard_dir="${CHIPYARD_DIR:-}"
source "${TVM_AE_ROOT}/scripts/env.sh"
if [[ -n "${account_chipyard_dir}" ]]; then
  export CHIPYARD_DIR="${account_chipyard_dir}"
  export FIRESIM_DIR="${CHIPYARD_DIR}/sims/firesim"
  export DTC_BIN="${CHIPYARD_DIR}/.conda-env/bin/dtc"
  export PK_BIN="${CHIPYARD_DIR}/.conda-env/riscv-tools/riscv64-unknown-elf/bin/pk"
  export LD_LIBRARY_PATH="${CHIPYARD_DIR}/.conda-env/riscv-tools/lib:${LD_LIBRARY_PATH:-}"
fi

build_dir="${TABLE4_TVM_BUILD_DIR:-${PWD}/results/table4/tvm-build-llvm}"
jobs="${TABLE4_TVM_BUILD_JOBS:-${TVM_BUILD_JOBS:-8}}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --build-dir=*) build_dir="${1#*=}"; shift ;;
    --build-dir) [[ "$#" -ge 2 ]] || exit 2; build_dir="$2"; shift 2 ;;
    -j) [[ "$#" -ge 2 ]] || exit 2; jobs="$2"; shift 2 ;;
    -h | --help)
      printf '%s\n' 'Usage: prepare-tvm.sh [--build-dir=PATH] [-j N]'
      exit 0
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid job count: %s\n' "${jobs}" >&2; exit 2; }
mkdir -p "${build_dir}"
cmake -S "${TVM_DIR}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_GEMMINI=ON -DUSE_MICRO=ON \
  -DUSE_LLVM="${LLVM_CONFIG}" -DUSE_OPENMP=none -DUSE_CUDA=OFF \
  -DUSE_VULKAN=OFF -DUSE_OPENCL=OFF
cmake --build "${build_dir}" --target tvm gemmini standalone_crt -j"${jobs}"
printf '[tvm-table4] LLVM-enabled build=%s\n' "${build_dir}"
