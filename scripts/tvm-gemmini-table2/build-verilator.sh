#!/usr/bin/env bash
set -euo pipefail

TVM_AE_ROOT="${TABLE2_TVM_AE_ROOT:-/home/ae/tvm-gemmini-ae}"
source "${TVM_AE_ROOT}/scripts/env.sh"

config="${TABLE2_VERILATOR_CONFIG:-OriginalGemminiRocketConfig}"
jobs="${TABLE2_VERILATOR_BUILD_JOBS:-8}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config=*) config="${1#*=}"; shift ;;
    --config) [[ "$#" -ge 2 ]] || exit 2; config="$2"; shift 2 ;;
    -j) [[ "$#" -ge 2 ]] || exit 2; jobs="$2"; shift 2 ;;
    -h | --help)
      printf '%s\n' \
        'Usage: build-verilator.sh [--config=CONFIG] [-j N]' \
        'Default: OriginalGemminiRocketConfig (INT8, DIM=16, one Rocket core).'
      exit 0
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid job count: %s\n' "${jobs}" >&2; exit 2; }
sim_dir="${CHIPYARD_DIR}/sims/verilator"
[[ -f "${sim_dir}/Makefile" ]] || { printf 'Chipyard Verilator directory not found: %s\n' "${sim_dir}" >&2; exit 1; }
printf '[tvm-verilator] building CONFIG=%s with %s jobs\n' "${config}" "${jobs}"
make -C "${sim_dir}" -j"${jobs}" CONFIG="${config}"
simulator="${sim_dir}/simulator-chipyard.harness-${config}"
[[ -x "${simulator}" ]] || { printf 'simulator was not produced: %s\n' "${simulator}" >&2; exit 1; }
printf '[tvm-verilator] simulator=%s\n' "${simulator}"
printf '[tvm-verilator] chipyard-commit=%s\n' "$(git -C "${CHIPYARD_DIR}" rev-parse HEAD)"
printf '[tvm-verilator] gemmini-params=%s\n' \
  "$(sha256sum "${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h" | awk '{print $1}')"
