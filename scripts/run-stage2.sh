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
  5. Complete Table 4's sampled-kernel FireSim and Verilator measurements
     when running the complete, non-selective workflow.

Options:
  --riscv-toolchain-dir=PATH
                        RISC-V toolchain root or bin dir containing
                        riscv64-unknown-linux-gnu-g++.
  --riscv-gxx=PATH      Exact riscv64-unknown-linux-gnu-g++ path.
  --workload=LIST       Run selected FireSim workload(s). May be repeated.
  --only-alias-first    Build the Figure 9 ablation artifacts and limit
                        image generation and FireSim execution to 4 CNN
                        Gemmini 4-core off workloads and 8 LLM seq=256 SDPA
                        Gemmini 4-core on/off workloads.
  --only-alias-first-cnn-off
                        Build and run only the four CNN Gemmini 4-core
                        alias-first off workloads.
  --resume, --resume-firesim
                        Skip ELF/package/image stages, keep collected results,
                        and run only FireSim workloads without complete outputs.
  --resume-from=NAME    Skip ELF/package/image stages, keep collected results,
                        and restart FireSim at workload NAME.
  --rvv-panic-retries=N|unlimited
                        Retry an RVV workload after a detected guest kernel
                        panic. Default: unlimited; 0 disables retries.
  --skip-elves          Skip model-<N>core.elf generation.
  --skip-package        Skip FireMarshal workload/package generation.
  --skip-images         Skip FireMarshal image build/install.
  --skip-firesim        Skip FireSim execution/collection.
  --skip-table4         Skip the sampled-kernel Table 4 workflow.
  --only-table4         Skip ordinary Stage 2 workloads and complete only the
                        Table 4 run started by Docker Stage 1.
  --table4-kernels=LIST Limit Table 4 to selected built-in kernel IDs.
                        Default: all three.
  --table4-repeats=N    Must match the Docker Stage 1 trial count. Default: 1.
  -h, --help            Show this help.

Environment:
  The script automatically sources PYTORCH_CHIPYARD_ACCOUNT_ENV (or the legacy
  TABLE4_ACCOUNT_ENV) when it exists. It defaults to $HOME/.ae-env.sh.
  CHIPYARD_DIR must point to the local Chipyard checkout. scripts/env.sh
  derives CHIPYARD_ENV_PATH as $CHIPYARD_DIR/env.sh.
  PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS defaults to 0 so existing collected
  results and logs are preserved. Set it to 1 to clean them before FireSim runs.
  TABLE4_TVM_AE_ROOT points to the prepared TVM-Gemmini AE tree. It defaults to
  $HOME/tvm-gemmini-ae on the author review server.
EOF
}

chipyard_env_arg=""
riscv_toolchain_dir_arg=""
riscv_gxx_arg=""
workload_args=()
only_alias_first=0
only_alias_first_cnn_off=0
resume_firesim=0
resume_from_arg=""
rvv_panic_retries_arg=""
skip_elves=0
skip_package=0
skip_images=0
skip_firesim=0
skip_table4=0
only_table4=0
table4_kernels="squeezenet_fire2_squeeze,resnet50_classifier,mobilenetv2_classifier"
table4_repeats="${TABLE4_REPEATS:-1}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --chipyard-env)
      [[ "$#" -ge 2 ]] || die "--chipyard-env requires a value"
      log "using deprecated --chipyard-env; prefer CHIPYARD_DIR with optional --riscv-toolchain-dir"
      chipyard_env_arg="$2"
      shift 2
      ;;
    --chipyard-env=*)
      log "using deprecated --chipyard-env; prefer CHIPYARD_DIR with optional --riscv-toolchain-dir"
      chipyard_env_arg="${1#--chipyard-env=}"
      shift
      ;;
    --riscv-toolchain-dir)
      [[ "$#" -ge 2 ]] || die "--riscv-toolchain-dir requires a value"
      riscv_toolchain_dir_arg="$2"
      shift 2
      ;;
    --riscv-toolchain-dir=*)
      riscv_toolchain_dir_arg="${1#--riscv-toolchain-dir=}"
      shift
      ;;
    --riscv-gxx)
      [[ "$#" -ge 2 ]] || die "--riscv-gxx requires a value"
      riscv_gxx_arg="$2"
      shift 2
      ;;
    --riscv-gxx=*)
      riscv_gxx_arg="${1#--riscv-gxx=}"
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
    --only-alias-first | --only-alias-first-ablation)
      only_alias_first=1
      shift
      ;;
    --only-alias-first-cnn-off)
      only_alias_first_cnn_off=1
      shift
      ;;
    --resume | --resume-firesim)
      resume_firesim=1
      skip_elves=1
      skip_package=1
      skip_images=1
      shift
      ;;
    --resume-from)
      [[ "$#" -ge 2 ]] || die "--resume-from requires a value"
      resume_from_arg="$2"
      skip_elves=1
      skip_package=1
      skip_images=1
      shift 2
      ;;
    --resume-from=*)
      resume_from_arg="${1#--resume-from=}"
      skip_elves=1
      skip_package=1
      skip_images=1
      shift
      ;;
    --rvv-panic-retries)
      [[ "$#" -ge 2 ]] || die "--rvv-panic-retries requires a value"
      rvv_panic_retries_arg="$2"
      shift 2
      ;;
    --rvv-panic-retries=*)
      rvv_panic_retries_arg="${1#--rvv-panic-retries=}"
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
    --skip-table4)
      skip_table4=1
      shift
      ;;
    --only-table4)
      only_table4=1
      shift
      ;;
    --table4-kernels=*)
      table4_kernels="${1#*=}"
      shift
      ;;
    --table4-kernels)
      [[ "$#" -ge 2 ]] || die "--table4-kernels requires a value"
      table4_kernels="$2"
      shift 2
      ;;
    --table4-repeats=*)
      table4_repeats="${1#*=}"
      shift
      ;;
    --table4-repeats)
      [[ "$#" -ge 2 ]] || die "--table4-repeats requires a value"
      table4_repeats="$2"
      shift 2
      ;;
    --skip-plot)
      log "ignoring deprecated --skip-plot; plotting now lives in scripts/run-plot.sh"
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

alias_first_workloads=()
alias_first_artifact_dirs=()
alias_first_selected=0
if [[ "${only_alias_first}" -eq 1 && "${only_alias_first_cnn_off}" -eq 1 ]]; then
  die "--only-alias-first and --only-alias-first-cnn-off are mutually exclusive"
fi
if [[ "${only_table4}" -eq 1 && "${skip_table4}" -eq 1 ]]; then
  die "--only-table4 cannot be combined with --skip-table4"
fi
if [[ "${only_table4}" -eq 1 && \
      ( "${only_alias_first}" -eq 1 || "${only_alias_first_cnn_off}" -eq 1 || \
        "${#workload_args[@]}" -gt 0 || "${resume_firesim}" -eq 1 || \
        -n "${resume_from_arg}" ) ]]; then
  die "--only-table4 cannot be combined with workload or ordinary FireSim selection options"
fi
[[ "${table4_repeats}" =~ ^[1-9][0-9]*$ ]] || \
  die "--table4-repeats must be a positive integer"
if [[ "${only_alias_first}" -eq 1 || "${only_alias_first_cnn_off}" -eq 1 ]]; then
  alias_first_selected=1
  [[ "${#workload_args[@]}" -eq 0 ]] || \
    die "an alias-first only option cannot be combined with --workload"
  if [[ "${only_alias_first}" -eq 1 ]]; then
    for model in gpt2 gpt-neo opt pythia; do
      for mode in on off; do
        alias_first_workloads+=(
          "${model}-gemmini-sdpa-256tok-alias-first-${mode}-4core"
        )
        alias_first_artifact_dirs+=(
          "${REPO_ROOT}/examples/artifact-${model}/gemmini/sdpa/seq256/alias-first-${mode}"
        )
      done
    done
  fi
  for model in alexnet mobilenetv2 resnet50 squeezenet; do
    alias_first_workloads+=(
      "${model}-gemmini-alias-first-off-4core"
    )
    alias_first_artifact_dirs+=(
      "${REPO_ROOT}/examples/artifact-${model}/gemmini-alias-first-off"
    )
  done
  for workload in "${alias_first_workloads[@]}"; do
    workload_args+=("--workload=${workload}")
  done
  log "selected Figure 9 alias-first ablation: ${#alias_first_workloads[@]} workloads"
fi

# Table 4 is a complete cross-toolchain experiment, not per-workload
# post-processing. Do not unexpectedly launch it after a selective Stage 2 run
# or when the caller explicitly disabled FireSim.
if [[ "${#workload_args[@]}" -gt 0 || \
      ( "${skip_firesim}" -eq 1 && "${only_table4}" -eq 0 ) ]]; then
  if [[ "${skip_table4}" -eq 0 ]]; then
    log "skipping Table 4 after a selective or --skip-firesim Stage 2 run"
  fi
  skip_table4=1
fi

if [[ "${only_table4}" -eq 1 ]]; then
  skip_elves=1
  skip_package=1
  skip_images=1
  skip_firesim=1
fi

if [[ -n "${chipyard_env_arg}" ]]; then
  export CHIPYARD_ENV_PATH="${chipyard_env_arg}"
fi
if [[ -n "${riscv_toolchain_dir_arg}" ]]; then
  export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR="${riscv_toolchain_dir_arg}"
fi
if [[ -n "${riscv_gxx_arg}" ]]; then
  export PYTORCH_CHIPYARD_RISCV_GXX="${riscv_gxx_arg}"
fi

# Stage 2 runs may be invoked one workload at a time. Preserve results collected
# by earlier invocations unless cleanup is explicitly requested by the caller.
export PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS="${PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS:-0}"

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

cd "${REPO_ROOT}"

if [[ "${skip_table4}" -eq 0 ]]; then
  table4_script="${SCRIPT_DIR}/run_table4.sh"
  table4_results_tool="${SCRIPT_DIR}/table4_results.py"
  table4_tvm_ae_root="${TABLE4_TVM_AE_ROOT:-${HOME}/tvm-gemmini-ae}"
  table4_stage1_output="${TABLE4_STAGE1_OUTPUT_DIR:-${REPO_ROOT}/results/table4/stage1-latest}"
  [[ -f "${table4_script}" ]] || die "missing Table 4 runner: ${table4_script}"
  [[ -f "${table4_results_tool}" ]] || die "missing Table 4 results tool: ${table4_results_tool}"
  [[ -f "${table4_tvm_ae_root}/scripts/env.sh" ]] || \
    die "missing TVM-Gemmini AE environment: ${table4_tvm_ae_root}/scripts/env.sh; set TABLE4_TVM_AE_ROOT or pass --skip-table4"
  [[ -s "${table4_stage1_output}/raw.csv" ]] || \
    die "missing Docker Stage 1 Table 4 results: ${table4_stage1_output}/raw.csv; rerun Stage 1 without --skip-table4"
  [[ -w "${table4_stage1_output}" && -w "${table4_stage1_output}/raw.csv" ]] || \
    die "Docker Stage 1 Table 4 results are not writable by $(id -un): ${table4_stage1_output}; repair bind-mount ownership or permissions before Stage 2"
  IFS=',' read -r -a table4_kernel_list <<<"${table4_kernels}"
  for table4_kernel in "${table4_kernel_list[@]}"; do
    table4_kernel="${table4_kernel//[[:space:]]/}"
    [[ -n "${table4_kernel}" ]] || continue
    for ((table4_trial = 1; table4_trial <= table4_repeats; table4_trial++)); do
      table4_kernel_spec="${table4_stage1_output}/artifacts/pytorch-chipyard/${table4_kernel}/trial-${table4_trial}/gemmini/model_spec.json"
      [[ -s "${table4_kernel_spec}" ]] || \
        die "missing Docker Stage 1 Table 4 artifact: ${table4_kernel_spec}"
      table4_artifact_dir="$(dirname -- "${table4_kernel_spec}")"
      [[ -w "${table4_artifact_dir}" ]] || \
        die "Docker Stage 1 artifact directory is not writable by $(id -un): ${table4_artifact_dir}"
      for table4_blob in input.bin weights.bin; do
        [[ -r "${table4_artifact_dir}/${table4_blob}" ]] || \
          die "Docker Stage 1 artifact is not readable by $(id -un): ${table4_artifact_dir}/${table4_blob}; repair bind-mount ownership or permissions before Stage 2"
      done
    done
  done
fi

build_elves_args=()
if [[ -n "${chipyard_env_arg}" ]]; then
  build_elves_args+=(--chipyard-env="${chipyard_env_arg}")
fi
if [[ -n "${riscv_toolchain_dir_arg}" ]]; then
  build_elves_args+=(--riscv-toolchain-dir="${riscv_toolchain_dir_arg}")
fi
if [[ -n "${riscv_gxx_arg}" ]]; then
  build_elves_args+=(--riscv-gxx="${riscv_gxx_arg}")
fi
if [[ "${alias_first_selected}" -eq 1 ]]; then
  for artifact_dir in "${alias_first_artifact_dirs[@]}"; do
    build_elves_args+=(--artifact-dir="${artifact_dir}")
  done
fi

if [[ "${skip_elves}" -eq 0 ]]; then
  log "building ELFs from examples"
  bash "${SCRIPT_DIR}/build-chipyard-elves.sh" \
    "${build_elves_args[@]}"
  printf '[stage2][PASS] ELF generation root=%s\n' "${REPO_ROOT}/examples"
fi

if [[ "${skip_package}" -eq 0 ]]; then
  log "packaging FireMarshal/FireSim workloads from examples"
  package_args=()
  if [[ "${alias_first_selected}" -eq 1 ]]; then
    for artifact_dir in "${alias_first_artifact_dirs[@]}"; do
      package_args+=(--artifact-dir="${artifact_dir}")
    done
  fi
  bash "${SCRIPT_DIR}/package-firemarshal-workload.sh" "${package_args[@]}"
  printf '[stage2][PASS] FireMarshal workload directory=%s\n' "${PYTORCH_CHIPYARD_WORKLOAD_DIR}"
fi

if [[ "${skip_images}" -eq 0 ]]; then
  log "building and installing FireMarshal images"
  image_args=()
  if [[ "${alias_first_selected}" -eq 1 ]]; then
    for workload in "${alias_first_workloads[@]}"; do
      image_args+=(--workload="${workload}")
    done
  fi
  bash "${SCRIPT_DIR}/build-firemarshal-images.sh" "${image_args[@]}"
  printf '[stage2][PASS] FireMarshal images=%s FireSim workloads=%s\n' \
    "${FIREMARSHAL_IMAGE_DIR}" "${FIRESIM_WORKLOAD_DIR}"
fi

if [[ "${skip_firesim}" -eq 0 ]]; then
  firesim_args=("${workload_args[@]}")
  if [[ "${resume_firesim}" -eq 1 ]]; then
    firesim_args+=(--resume)
  fi
  if [[ -n "${resume_from_arg}" ]]; then
    firesim_args+=(--resume-from="${resume_from_arg}")
  fi
  if [[ -n "${rvv_panic_retries_arg}" ]]; then
    firesim_args+=(--rvv-panic-retries="${rvv_panic_retries_arg}")
  fi
  log "running FireSim workloads"
  bash "${SCRIPT_DIR}/run-firesim-workloads.sh" "${firesim_args[@]}"
  printf '[stage2][PASS] FireSim results=%s\n' "${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
fi

if [[ "${skip_table4}" -eq 0 ]]; then
  log "completing Table 4 from Docker Stage 1 compile measurements"
  bash "${SCRIPT_DIR}/run_table4.sh" \
    --resume \
    --build-verilator \
    --kernels="${table4_kernels}" \
    --repeats="${table4_repeats}" \
    --output-dir="${table4_stage1_output}"
  printf '[stage2][PASS] Table 4 results=%s\n' "${table4_stage1_output}"
fi

log "done; run bash scripts/run-plot.sh to generate figures"
printf 'STAGE2_RESULTS_DIR=%s\n' "${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
printf 'STAGE2_STATUS=PASS\n'
