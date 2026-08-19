#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

log() { printf '[smoke] %s\n' "$*"; }
pass() { printf '[smoke][PASS] %s\n' "$*"; }
die() { printf '[smoke][error] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-smoke-test.sh [options]

Run the bounded host smoke tests:
  1. Compile and execute one 32x32x32 GEMM with TVM-Gemmini/Verilator.
  2. Build and execute the same GEMM with PyTorch-Chipyard on FireSim using
     RVV 4 cores, Gemmini 4 cores, and scalar Rocket 16 cores.

Options:
  --artifact-root=PATH  Docker Stage 1 artifact root. Default:
                        results/smoke-test/artifacts/pytorch-chipyard
  --output-dir=PATH     Per-run logs/results. Default:
                        results/smoke-test/runs/<UTC run id>
  --build-verilator    Build the TVM-Gemmini Verilator target if necessary.
  --skip-tvm           Skip the TVM-Gemmini/Verilator check.
  --skip-firesim       Skip the three PyTorch-Chipyard/FireSim checks.
  -h, --help           Show this help.

Environment:
  TABLE4_TVM_AE_ROOT       Prepared TVM-Gemmini tree.
  TABLE4_TVM_BUILD_DIR     Reusable LLVM-enabled TVM build directory.
  TABLE4_VERILATOR_BIN     Prebuilt Verilator simulator override.
  PYTORCH_CHIPYARD_ACCOUNT_ENV
                           Preconfigured host account environment.
  PYTORCH_CHIPYARD_SMOKE_FIRESIM_HW_CONFIG_OVERRIDES
                           Smoke-specific workload=hardware overrides.
EOF
}

run_logged() {
  local log_path="$1"
  shift
  local rc
  mkdir -p "$(dirname -- "${log_path}")"
  set +e
  "$@" 2>&1 | tee "${log_path}"
  rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

require_nonempty() {
  local path="$1"
  [[ -s "${path}" ]] || die "expected non-empty smoke-test output: ${path}"
}

artifact_root="${REPO_ROOT}/results/smoke-test/artifacts/pytorch-chipyard"
output_dir=""
build_verilator=0
skip_tvm=0
skip_firesim=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-root=*) artifact_root="${1#*=}"; shift ;;
    --artifact-root)
      [[ "$#" -ge 2 ]] || die "--artifact-root requires a value"
      artifact_root="$2"
      shift 2
      ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --output-dir)
      [[ "$#" -ge 2 ]] || die "--output-dir requires a value"
      output_dir="$2"
      shift 2
      ;;
    --build-verilator) build_verilator=1; shift ;;
    --skip-tvm) skip_tvm=1; shift ;;
    --skip-firesim) skip_firesim=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown argument '$1'; pass --help for usage" ;;
  esac
done

if [[ "${skip_tvm}" -eq 1 && "${skip_firesim}" -eq 1 ]]; then
  die "--skip-tvm and --skip-firesim cannot be used together"
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ -z "${output_dir}" ]]; then
  output_dir="${REPO_ROOT}/results/smoke-test/runs/${run_id}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
if [[ "${artifact_root}" != /* ]]; then
  artifact_root="${PWD}/${artifact_root}"
fi
mkdir -p "${output_dir}/logs" "${output_dir}/setup"
output_dir="$(cd -- "${output_dir}" >/dev/null 2>&1 && pwd -P)"

account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
  log "loaded account environment: ${account_env}"
fi
set +u
source "${SCRIPT_DIR}/env.sh"
set -u

run_tvm_smoke() {
  local tvm_root tvm_build_dir config simulator target_header
  local project elf compile_log verilator_log

  tvm_root="${TABLE4_TVM_AE_ROOT:-${HOME}/tvm-gemmini-ae}"
  require_file "${tvm_root}/scripts/env.sh"
  [[ -n "${CHIPYARD_DIR:-}" ]] || die "CHIPYARD_DIR is required for the Verilator smoke test"

  config="${TABLE4_VERILATOR_CONFIG:-OriginalGemminiRocketConfig}"
  simulator="${TABLE4_VERILATOR_BIN:-${CHIPYARD_DIR}/sims/verilator/simulator-chipyard.harness-${config}}"
  target_header="${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h"

  if [[ "${build_verilator}" -eq 1 ]]; then
    log "building/checking the TVM-Gemmini Verilator target"
    run_logged "${output_dir}/logs/tvm-setup-verilator.log" \
      env TABLE4_TVM_AE_ROOT="${tvm_root}" \
      bash "${SCRIPT_DIR}/tvm-gemmini-table4/build-verilator.sh" \
      || die "Verilator target build failed; see ${output_dir}/logs/tvm-setup-verilator.log"
  fi

  [[ -x "${simulator}" ]] || \
    die "Verilator simulator not found at ${simulator}; rerun with --build-verilator"
  require_file "${target_header}"
  grep -Eq '^#define DIM 16$' "${target_header}" || \
    die "${target_header} is not the DIM=16 TVM-Gemmini target; rerun with --build-verilator"
  grep -Eq '^typedef int8_t elem_t;$' "${target_header}" || \
    die "${target_header} is not the INT8 TVM-Gemmini target; rerun with --build-verilator"

  tvm_build_dir="${TABLE4_TVM_BUILD_DIR:-${REPO_ROOT}/results/smoke-test/setup/tvm-build-llvm}"
  log "preparing the reusable LLVM-enabled TVM build"
  run_logged "${output_dir}/logs/tvm-prepare.log" \
    env TABLE4_TVM_AE_ROOT="${tvm_root}" TABLE4_TVM_BUILD_DIR="${tvm_build_dir}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table4/prepare-tvm.sh" \
      --build-dir "${tvm_build_dir}" \
    || die "TVM preparation failed; see ${output_dir}/logs/tvm-prepare.log"

  project="${output_dir}/artifacts/tvm-gemmini/smoke-gemm-32/generated-project"
  compile_log="${output_dir}/logs/tvm-smoke-gemm-compile.log"
  log "compiling the TVM-Gemmini 32x32x32 GEMM"
  run_logged "${compile_log}" \
    env TABLE4_TVM_AE_ROOT="${tvm_root}" \
      TABLE4_TVM_BUILD_DIR="${tvm_build_dir}" \
      TABLE4_VERILATOR_BIN="${simulator}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table4/compile-kernel.sh" \
      --kernel smoke_gemm_32 --output-dir "${project}" \
    || die "TVM-Gemmini smoke compile failed; see ${compile_log}"

  elf="${project}/src/build/dense-baremetal"
  require_file "${elf}"
  verilator_log="${output_dir}/logs/tvm-smoke-gemm-verilator.log"
  log "executing the TVM-Gemmini 32x32x32 GEMM on Verilator"
  run_logged "${verilator_log}" \
    env TABLE4_TVM_AE_ROOT="${tvm_root}" \
      TABLE4_TVM_BUILD_DIR="${tvm_build_dir}" \
      TABLE4_VERILATOR_BIN="${simulator}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table4/run-verilator.sh" --elf "${elf}" \
    || die "TVM-Gemmini Verilator smoke test failed; see ${verilator_log}"

  grep -Fq 'KERNEL_SHAPE=32x32x32' "${verilator_log}" || \
    die "32x32x32 runtime marker missing from ${verilator_log}"
  grep -Fq 'KERNEL_EXECUTION=PASS' "${verilator_log}" || \
    die "TVM runtime did not report PASS in ${verilator_log}"
  grep -Fq 'VERILATOR_STATUS=PASS' "${verilator_log}" || \
    die "Verilator wrapper did not report PASS in ${verilator_log}"
  pass "TVM-Gemmini Verilator shape=32x32x32 ELF=${elf} log=${verilator_log}"
}

run_firesim_smoke() {
  local workload_root collected_root firesim_log_root runtime_root
  local smoke_hw_overrides
  local backend core artifact workload result_dir label output_file
  local artifact_args=() workload_args=()
  local backends=(rvv gemmini scalar)
  local cores=(4 4 16)
  local workloads=(
    smoke-gemm-rvv-4core
    smoke-gemm-gemmini-4core
    smoke-gemm-scalar-16core
  )
  local labels=("RVV 4-core" "Gemmini 4-core" "scalar 16-core")

  [[ -n "${CHIPYARD_DIR:-}" ]] || die "CHIPYARD_DIR is required for the FireSim smoke tests"
  [[ -d "${artifact_root}" ]] || \
    die "smoke artifacts not found at ${artifact_root}; run scripts/run-smoke-test-stage1.sh in Docker first"

  for index in "${!backends[@]}"; do
    backend="${backends[${index}]}"
    artifact="${artifact_root}/${backend}"
    require_file "${artifact}/model_spec.json"
    require_file "${artifact}/input.bin"
    require_file "${artifact}/weights.bin"
    artifact_args+=(--artifact-dir "${artifact}")
    workload_args+=(--workload "${workloads[${index}]}")
  done

  workload_root="${output_dir}/setup/firemarshal-workloads"
  collected_root="${output_dir}/firesim-results"
  firesim_log_root="${output_dir}/logs/firesim"
  runtime_root="${output_dir}/setup/firesim-runtime"
  smoke_hw_overrides="${PYTORCH_CHIPYARD_SMOKE_FIRESIM_HW_CONFIG_OVERRIDES:-smoke-gemm-rvv-4core=alveo_u250_firesim_minv128d64_rocket_4core_no_nic}"
  mkdir -p "${workload_root}" "${collected_root}" "${firesim_log_root}" "${runtime_root}"

  log "building the RVV 4-core, Gemmini 4-core, and scalar 16-core ELFs"
  run_logged "${output_dir}/logs/pytorch-build-elves.log" \
    bash "${SCRIPT_DIR}/build-chipyard-elves.sh" "${artifact_args[@]}" \
    || die "smoke ELF construction failed; see ${output_dir}/logs/pytorch-build-elves.log"

  for index in "${!backends[@]}"; do
    require_file "${artifact_root}/${backends[${index}]}/model-${cores[${index}]}core.elf"
  done

  log "packaging the three FireSim smoke workloads"
  run_logged "${output_dir}/logs/pytorch-package.log" \
    env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
      PYTORCH_CHIPYARD_OMP_FIRST_CPU_SMOKE_GEMM_RVV_4CORE=0 \
    bash "${SCRIPT_DIR}/package-firemarshal-workload.sh" "${artifact_args[@]}" \
    || die "smoke workload packaging failed; see ${output_dir}/logs/pytorch-package.log"

  log "building and installing the three FireMarshal images"
  run_logged "${output_dir}/logs/pytorch-firemarshal.log" \
    env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
    bash "${SCRIPT_DIR}/build-firemarshal-images.sh" "${workload_args[@]}" \
    || die "smoke FireMarshal image build failed; see ${output_dir}/logs/pytorch-firemarshal.log"

  log "running the three PyTorch-Chipyard FireSim smoke workloads"
  run_logged "${output_dir}/logs/pytorch-firesim.log" \
    env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
      PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${collected_root}" \
      PYTORCH_CHIPYARD_LOG_DIR="${firesim_log_root}" \
      PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${runtime_root}" \
      PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS=1 \
      PYTORCH_CHIPYARD_FIRESIM_HW_CONFIG_OVERRIDES="${smoke_hw_overrides}" \
    bash "${SCRIPT_DIR}/run-firesim-workloads.sh" \
      "${workload_args[@]}" --rvv-panic-retries=0 \
    || die "PyTorch-Chipyard FireSim smoke test failed; see ${output_dir}/logs/pytorch-firesim.log"

  for index in "${!workloads[@]}"; do
    workload="${workloads[${index}]}"
    label="${labels[${index}]}"
    result_dir="${collected_root}/${workload}"
    for output_file in uartlog model.log autotune.log output.bin; do
      require_nonempty "${result_dir}/${output_file}"
    done
    pass "PyTorch-Chipyard ${label} shape=32x32x32 results=${result_dir}"
  done
}

log "output: ${output_dir}"
if [[ "${skip_tvm}" -eq 0 ]]; then
  run_tvm_smoke
fi
if [[ "${skip_firesim}" -eq 0 ]]; then
  run_firesim_smoke
fi

mkdir -p "${REPO_ROOT}/results/smoke-test"
ln -sfn "${output_dir}" \
  "${REPO_ROOT}/results/smoke-test/latest"
printf 'SMOKE_TEST_RESULTS=%s\n' "${output_dir}"
printf 'SMOKE_TEST_STATUS=PASS\n'
