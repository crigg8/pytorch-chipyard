#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
RESULTS_TOOL="${SCRIPT_DIR}/table2_results.py"

log() { printf '[table2] %s\n' "$*"; }
warn() { printf '[table2][warn] %s\n' "$*" >&2; }
die() { printf '[table2][error] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_table2.sh [options]

Measure native compile and cycle-accurate RTL turnaround for three bounded,
model-derived GEMM kernels. PyTorch-Chipyard uses its FP32 DIM=8 four-core
FireSim target; TVM-Gemmini uses its INT8 DIM=16 single-core Verilator target.
The measurements characterize each tool's own workflow and are not a kernel
performance comparison.

Options:
  --kernels=LIST       Built-in Table 2 kernel IDs (default: all)
  --toolchains=LIST    pytorch,tvm (default: both)
  --phases=LIST        compile,rtl (default: both)
  --repeats=N          Independent trials; report successful medians (default: 1)
  --output-dir=PATH    Result directory (default: results/table2/<UTC run id>)
  --resume             Run only missing/failed measurements in --output-dir
  --build-verilator    Build the TVM-Gemmini Verilator simulator if absent
  --dry-run            Print the experiment matrix without running commands
  -h, --help           Show this help

Outputs:
  raw.csv              Provenance and status for every trial and phase
  table2.csv           One summary row per sampled kernel
  table2_rows.tex      LaTeX rows generated from table2.csv
  logs/                Full compiler and simulator logs
  artifacts/           Per-toolchain compiler artifacts

Only a complete successful matrix is copied to scripts/figures/table2.csv and
scripts/figures/table2_rows.tex. Simulator construction, ELF construction,
FireMarshal image construction, and FireSim setup are deliberately untimed.

Environment:
  TABLE2_TVM_AE_ROOT              TVM-Gemmini AE tree
                                  (default: /home/ae/tvm-gemmini-ae)
  TABLE2_TVM_BUILD_DIR            Private LLVM-enabled TVM build
                                  (default: <output-dir>/setup/tvm-build-llvm)
  TABLE2_TVM_GEMMINI_INCLUDE      Gemmini headers for the Verilator RTL target
                                  (default: current Chipyard target headers)
  TABLE2_VERILATOR_BIN            Prebuilt Verilator simulator override
  TABLE2_BUILD_VERILATOR=1        Same as --build-verilator
  TABLE2_PYTORCH_FIRESIM_HW_CONFIG
                                  FireSim HW config override
  TABLE2_KEEP_GOING=0             Stop after the first failed measurement
  TABLE2_RESULTS_PYTHON           Python for the CSV helper
EOF
}

if [[ -n "${TABLE2_RESULTS_PYTHON:-}" ]]; then
  RESULTS_PYTHON="${TABLE2_RESULTS_PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  RESULTS_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  RESULTS_PYTHON="$(command -v python)"
else
  die "python is required for ${RESULTS_TOOL}"
fi

mapfile -t ALL_KERNELS < <("${RESULTS_PYTHON}" "${RESULTS_TOOL}" list-kernels)
[[ "${#ALL_KERNELS[@]}" -gt 0 ]] || die "no built-in Table 2 kernels found"
kernels_arg="$(IFS=,; printf '%s' "${ALL_KERNELS[*]}")"
toolchains_arg="pytorch,tvm"
phases_arg="compile,rtl"
repeats="${TABLE2_REPEATS:-1}"
output_dir=""
resume=0
dry_run=0
build_verilator="${TABLE2_BUILD_VERILATOR:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --kernels=*) kernels_arg="${1#*=}"; shift ;;
    --kernels) [[ "$#" -ge 2 ]] || die "--kernels requires a value"; kernels_arg="$2"; shift 2 ;;
    --toolchains=*) toolchains_arg="${1#*=}"; shift ;;
    --toolchains) [[ "$#" -ge 2 ]] || die "--toolchains requires a value"; toolchains_arg="$2"; shift 2 ;;
    --phases=*) phases_arg="${1#*=}"; shift ;;
    --phases) [[ "$#" -ge 2 ]] || die "--phases requires a value"; phases_arg="$2"; shift 2 ;;
    --repeats=*) repeats="${1#*=}"; shift ;;
    --repeats) [[ "$#" -ge 2 ]] || die "--repeats requires a value"; repeats="$2"; shift 2 ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --output-dir) [[ "$#" -ge 2 ]] || die "--output-dir requires a value"; output_dir="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    --build-verilator) build_verilator=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown argument '$1'; pass --help for usage" ;;
  esac
done

[[ "${repeats}" =~ ^[1-9][0-9]*$ ]] || die "--repeats must be a positive integer"
[[ "${build_verilator}" == 0 || "${build_verilator}" == 1 ]] || \
  die "TABLE2_BUILD_VERILATOR must be 0 or 1"

contains() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    [[ "${value}" == "${needle}" ]] && return 0
  done
  return 1
}

split_unique() {
  local raw="$1"
  local out_name="$2"
  local -n out_ref="${out_name}"
  local value seen
  local parts=()
  out_ref=()
  IFS=',' read -r -a parts <<<"${raw}"
  for value in "${parts[@]}"; do
    value="${value//[[:space:]]/}"
    [[ -n "${value}" ]] || continue
    for seen in "${out_ref[@]}"; do
      [[ "${seen}" == "${value}" ]] && continue 2
    done
    out_ref+=("${value}")
  done
  [[ "${#out_ref[@]}" -gt 0 ]] || die "empty list"
}

kernels=()
toolchains=()
phases=()
split_unique "${kernels_arg}" kernels
split_unique "${toolchains_arg}" toolchains
split_unique "${phases_arg}" phases
for kernel in "${kernels[@]}"; do
  contains "${kernel}" "${ALL_KERNELS[@]}" || die "unknown kernel '${kernel}'"
done
for toolchain in "${toolchains[@]}"; do
  contains "${toolchain}" pytorch tvm || die "unknown toolchain '${toolchain}'"
done
for phase in "${phases[@]}"; do
  contains "${phase}" compile rtl || die "unknown phase '${phase}'"
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${resume}" -eq 1 && -z "${output_dir}" ]]; then
  die "--resume requires --output-dir"
fi
if [[ -z "${output_dir}" ]]; then
  output_dir="${REPO_ROOT}/results/table2/${run_id}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
mkdir -p "${output_dir}/logs" "${output_dir}/artifacts" "${output_dir}/setup"
raw_csv="${output_dir}/raw.csv"
summary_csv="${output_dir}/table2.csv"
latex_rows="${output_dir}/table2_rows.tex"
if [[ "${resume}" -eq 1 && ! -s "${raw_csv}" ]]; then
  die "cannot resume without ${raw_csv}"
fi
"${RESULTS_PYTHON}" "${RESULTS_TOOL}" init --csv "${raw_csv}"

kernel_field() {
  "${RESULTS_PYTHON}" "${RESULTS_TOOL}" kernel-field \
    --kernel "$1" --field "$2"
}

already_passed() {
  local trial="$1" kernel="$2" toolchain="$3" phase="$4" simulator="$5"
  [[ "${resume}" -eq 1 ]] || return 1
  "${RESULTS_PYTHON}" "${RESULTS_TOOL}" has-pass --csv "${raw_csv}" \
    --trial "${trial}" --kernel-id "${kernel}" --toolchain "${toolchain}" \
    --phase "${phase}" --simulator "${simulator}"
}

append_result() {
  local trial="$1" kernel="$2" toolchain="$3" phase="$4" simulator="$5"
  local wall="$6" status="$7" rc="$8" artifact="$9" log_path="${10}"
  local target="${11}" notes="${12}"
  "${RESULTS_PYTHON}" "${RESULTS_TOOL}" append --csv "${raw_csv}" \
    --run-id "${run_id}" --trial "${trial}" --kernel-id "${kernel}" \
    --kernel-label "$(kernel_field "${kernel}" label)" \
    --source-model "$(kernel_field "${kernel}" source_model)" \
    --shape "$(kernel_field "${kernel}" m)x$(kernel_field "${kernel}" n)x$(kernel_field "${kernel}" k)" \
    --macs "$(kernel_field "${kernel}" macs)" --toolchain "${toolchain}" \
    --phase "${phase}" --simulator "${simulator}" --total-wall-s "${wall}" \
    --status "${status}" --exit-code "${rc}" --artifact-path "${artifact}" \
    --log-path "${log_path}" --target-metadata "${target}" --notes "${notes}"
}

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname -- "${log_path}")"
  set +e
  "$@" 2>&1 | tee "${log_path}"
  RUN_RC=${PIPESTATUS[0]}
  set -e
}

elapsed_seconds() {
  awk -v start="$1" -v end="$2" 'BEGIN { printf "%.6f", end - start }'
}

failures=0
keep_going="${TABLE2_KEEP_GOING:-1}"
record_failure() {
  failures=$((failures + 1))
  [[ "${keep_going}" != 0 ]] || return 1
  return 0
}

pytorch_artifact() {
  printf '%s/artifacts/pytorch-chipyard/%s/trial-%s/gemmini\n' "${output_dir}" "$1" "$2"
}

tvm_artifact() {
  printf '%s/artifacts/tvm-gemmini/%s/trial-%s/generated-project\n' "${output_dir}" "$1" "$2"
}

run_pytorch_compile() {
  local trial="$1" kernel="$2" artifact log_path wall status
  artifact="$(pytorch_artifact "${kernel}" "${trial}")"
  log_path="${output_dir}/logs/pytorch-${kernel}-trial-${trial}-compile.log"
  if already_passed "${trial}" "${kernel}" PyTorch-Chipyard compile ""; then
    log "resume: PyTorch compile ${kernel} trial ${trial} already passed"
    return
  fi
  if [[ "${dry_run}" -eq 1 ]]; then
    log "DRY RUN PyTorch compile ${kernel} trial ${trial}"
    return
  fi
  run_logged "${log_path}" bash "${SCRIPT_DIR}/run-table2-kernel-compile.sh" \
    --kernel "${kernel}" --artifact-dir "${artifact}" --force-recompile
  if [[ "${RUN_RC}" -eq 0 ]] && wall="$("${RESULTS_PYTHON}" "${RESULTS_TOOL}" extract --log "${log_path}" --key COMPILE_WALL_S)"; then
    status=PASS
  else
    status=FAIL
    wall=""
  fi
  append_result "${trial}" "${kernel}" PyTorch-Chipyard compile "" "${wall}" \
    "${status}" "${RUN_RC}" "${artifact}" "${log_path}" \
    "fp32,gemmini-dim8,rocket-4core,firesim" \
    "normal bounded TorchInductor candidate set; context=${TABLE2_PYTORCH_COMPILE_CONTEXT:-host}"
  [[ "${status}" == PASS ]] || record_failure
}

tvm_build_dir="${TABLE2_TVM_BUILD_DIR:-${output_dir}/setup/tvm-build-llvm}"
tvm_prepared=0
prepare_tvm() {
  [[ "${tvm_prepared}" -eq 0 ]] || return
  run_logged "${output_dir}/logs/setup-tvm.log" \
    env TABLE2_TVM_AE_ROOT="${TABLE2_TVM_AE_ROOT:-/home/ae/tvm-gemmini-ae}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table2/prepare-tvm.sh" --build-dir "${tvm_build_dir}"
  [[ "${RUN_RC}" -eq 0 ]] || die "failed to prepare the LLVM-enabled TVM build; see ${output_dir}/logs/setup-tvm.log"
  tvm_prepared=1
}

run_tvm_compile() {
  local trial="$1" kernel="$2" artifact log_path wall status force=()
  artifact="$(tvm_artifact "${kernel}" "${trial}")"
  log_path="${output_dir}/logs/tvm-${kernel}-trial-${trial}-compile.log"
  if already_passed "${trial}" "${kernel}" TVM-Gemmini compile ""; then
    log "resume: TVM compile ${kernel} trial ${trial} already passed"
    return
  fi
  if [[ "${dry_run}" -eq 1 ]]; then
    log "DRY RUN TVM compile ${kernel} trial ${trial}"
    return
  fi
  prepare_tvm
  [[ ! -e "${artifact}" ]] || force=(--force)
  run_logged "${log_path}" env \
    TABLE2_TVM_AE_ROOT="${TABLE2_TVM_AE_ROOT:-/home/ae/tvm-gemmini-ae}" \
    TABLE2_TVM_BUILD_DIR="${tvm_build_dir}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table2/compile-kernel.sh" \
    --kernel "${kernel}" --output-dir "${artifact}" "${force[@]}"
  if [[ "${RUN_RC}" -eq 0 ]] && wall="$("${RESULTS_PYTHON}" "${RESULTS_TOOL}" extract --log "${log_path}" --key COMPILE_WALL_S)"; then
    status=PASS
  else
    status=FAIL
    wall=""
  fi
  append_result "${trial}" "${kernel}" TVM-Gemmini compile "" "${wall}" \
    "${status}" "${RUN_RC}" "${artifact}" "${log_path}" \
    "int8,gemmini-dim16,rocket-singlecore,verilator" \
    "Relay/Gemmini compilation; TFLite conversion and generated-project C build excluded"
  [[ "${status}" == PASS ]] || record_failure
}

verilator_prepared=0
prepare_verilator() {
  local simulator="${TABLE2_VERILATOR_BIN:-}"
  [[ "${verilator_prepared}" -eq 0 ]] || return
  if [[ -z "${simulator}" ]]; then
    local tvm_root="${TABLE2_TVM_AE_ROOT:-/home/ae/tvm-gemmini-ae}"
    [[ -f "${tvm_root}/scripts/env.sh" ]] || die "TVM-Gemmini environment not found: ${tvm_root}/scripts/env.sh"
    simulator="$(
      set +u
      source "${tvm_root}/scripts/env.sh"
      set -u
      printf '%s/sims/verilator/simulator-chipyard.harness-%s' \
        "${CHIPYARD_DIR}" "${TABLE2_VERILATOR_CONFIG:-OriginalGemminiRocketConfig}"
    )"
  fi
  if [[ ! -x "${simulator}" && "${build_verilator}" -eq 1 ]]; then
    run_logged "${output_dir}/logs/setup-verilator.log" \
      env TABLE2_TVM_AE_ROOT="${TABLE2_TVM_AE_ROOT:-/home/ae/tvm-gemmini-ae}" \
      bash "${SCRIPT_DIR}/tvm-gemmini-table2/build-verilator.sh"
    [[ "${RUN_RC}" -eq 0 ]] || die "failed to build Verilator; see ${output_dir}/logs/setup-verilator.log"
  fi
  [[ -x "${simulator}" ]] || \
    die "Verilator simulator not found at ${simulator}; pass --build-verilator or set TABLE2_VERILATOR_BIN"
  export TABLE2_VERILATOR_BIN="${simulator}"
  verilator_prepared=1
}

run_tvm_rtl() {
  local trial="$1" kernel="$2" artifact elf log_path start end wall status
  artifact="$(tvm_artifact "${kernel}" "${trial}")"
  elf="${artifact}/src/build/dense-baremetal"
  log_path="${output_dir}/logs/tvm-${kernel}-trial-${trial}-verilator.log"
  if already_passed "${trial}" "${kernel}" TVM-Gemmini rtl verilator; then
    log "resume: TVM Verilator ${kernel} trial ${trial} already passed"
    return
  fi
  if [[ "${dry_run}" -eq 1 ]]; then
    log "DRY RUN TVM Verilator ${kernel} trial ${trial}"
    return
  fi
  [[ -f "${elf}" ]] || {
    append_result "${trial}" "${kernel}" TVM-Gemmini rtl verilator "" FAIL 1 \
      "${artifact}" "${log_path}" "int8,gemmini-dim16,rocket-singlecore,verilator" \
      "missing compiled ELF; run the compile phase first"
    record_failure
    return
  }
  prepare_verilator
  start="$(date +%s.%N)"
  run_logged "${log_path}" env \
    TABLE2_TVM_AE_ROOT="${TABLE2_TVM_AE_ROOT:-/home/ae/tvm-gemmini-ae}" \
    TABLE2_VERILATOR_BIN="${TABLE2_VERILATOR_BIN}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table2/run-verilator.sh" --elf "${elf}"
  end="$(date +%s.%N)"
  if [[ "${RUN_RC}" -eq 0 ]]; then
    wall="$(elapsed_seconds "${start}" "${end}")"
    status=PASS
  else
    wall=""
    status=FAIL
  fi
  append_result "${trial}" "${kernel}" TVM-Gemmini rtl verilator "${wall}" \
    "${status}" "${RUN_RC}" "${artifact}" "${log_path}" \
    "int8,gemmini-dim16,rocket-singlecore,verilator" \
    "host wall time of the correctness-checked Verilator process"
  [[ "${status}" == PASS ]] || record_failure
}

load_pytorch_host_environment() {
  if [[ -f "${TABLE2_ACCOUNT_ENV:-${HOME}/.ae-env.sh}" ]]; then
    set +u
    source "${TABLE2_ACCOUNT_ENV:-${HOME}/.ae-env.sh}"
    set -u
  fi
  set +u
  source "${SCRIPT_DIR}/env.sh"
  set -u
  [[ -n "${CHIPYARD_DIR:-}" ]] || die "CHIPYARD_DIR is required for FireSim"
}

run_pytorch_rtl() {
  local trial="$1" kernel="$2" artifact workload_root collected_root workload
  local log_path uart output_bin wall status hw_override
  artifact="$(pytorch_artifact "${kernel}" "${trial}")"
  log_path="${output_dir}/logs/pytorch-${kernel}-trial-${trial}-firesim.log"
  if already_passed "${trial}" "${kernel}" PyTorch-Chipyard rtl firesim; then
    log "resume: PyTorch FireSim ${kernel} trial ${trial} already passed"
    return
  fi
  if [[ "${dry_run}" -eq 1 ]]; then
    log "DRY RUN PyTorch FireSim ${kernel} trial ${trial}"
    return
  fi
  [[ -s "${artifact}/model_spec.json" && -s "${artifact}/input.bin" && -s "${artifact}/weights.bin" ]] || {
    append_result "${trial}" "${kernel}" PyTorch-Chipyard rtl firesim "" FAIL 1 \
      "${artifact}" "${log_path}" "fp32,gemmini-dim8,rocket-4core,firesim" \
      "missing Stage 1 artifact; run the compile phase first"
    record_failure
    return
  }
  load_pytorch_host_environment
  workload_root="${output_dir}/setup/firemarshal-workloads"
  collected_root="${output_dir}/firesim-results"
  mkdir -p "${workload_root}" "${collected_root}" "${output_dir}/logs/firesim"
  workload="table2-${run_id}-${kernel}-trial-${trial}-gemmini-4core"
  printf '%s\n' "table2-${run_id}-${kernel}-trial-${trial}/gemmini" > \
    "${artifact}/.pytorch-chipyard-workload-rel"

  run_logged "${log_path}.elf" bash "${SCRIPT_DIR}/build-chipyard-elves.sh" \
    --artifact-dir "${artifact}" --backend gemmini --core 4
  if [[ "${RUN_RC}" -eq 0 ]]; then
    run_logged "${log_path}.package" env \
      PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
      bash "${SCRIPT_DIR}/package-firemarshal-workload.sh" --no-clean --artifact-dir "${artifact}"
  fi
  if [[ "${RUN_RC}" -eq 0 ]]; then
    run_logged "${log_path}.image" env \
      PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
      bash "${SCRIPT_DIR}/build-firemarshal-images.sh" --workload "${workload}"
  fi
  if [[ "${RUN_RC}" -eq 0 ]]; then
    hw_override="${workload}=${TABLE2_PYTORCH_FIRESIM_HW_CONFIG:-alveo_u250_firesim_fp8x8_gemmini_rocket_4core_no_nic}"
    run_logged "${log_path}" env \
      PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
      PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${collected_root}" \
      PYTORCH_CHIPYARD_LOG_DIR="${output_dir}/logs/firesim" \
      PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${output_dir}/setup/firesim-runtime" \
      PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS=0 \
      PYTORCH_CHIPYARD_FIRESIM_HW_CONFIG_OVERRIDES="${hw_override}" \
      bash "${SCRIPT_DIR}/run-firesim-workloads.sh" --workload "${workload}"
  fi
  uart="${collected_root}/${workload}/uartlog"
  output_bin="${collected_root}/${workload}/output.bin"
  if [[ "${RUN_RC}" -eq 0 && -s "${output_bin}" ]]; then
    cp -f "${output_bin}" "${artifact}/output.bin"
    run_logged "${log_path}.validate" bash "${SCRIPT_DIR}/validate-table2-kernel.sh" \
      --kernel "${kernel}" --artifact-dir "${artifact}"
  elif [[ "${RUN_RC}" -eq 0 ]]; then
    warn "FireSim did not collect ${output_bin}"
    RUN_RC=1
  fi
  if [[ "${RUN_RC}" -eq 0 ]] && wall="$("${RESULTS_PYTHON}" "${RESULTS_TOOL}" extract-firesim-wall --log "${uart}")"; then
    status=PASS
  else
    wall=""
    status=FAIL
  fi
  append_result "${trial}" "${kernel}" PyTorch-Chipyard rtl firesim "${wall}" \
    "${status}" "${RUN_RC}" "${artifact}" "${log_path}" \
    "fp32,gemmini-dim8,rocket-4core,firesim" \
    "FireSim UART wall time; includes native runtime autotuning and launches; excludes ELF/image/FireSim setup"
  [[ "${status}" == PASS ]] || record_failure
}

log "output: ${output_dir}"
log "kernels: ${kernels[*]}"
log "toolchains: ${toolchains[*]}; phases: ${phases[*]}; repeats: ${repeats}"

# Build or resolve the RTL target before TVM compilation. The compiler copies
# that target's Gemmini headers into each project, preventing a stale vendored
# TVM header from being paired with a different Verilator configuration.
if contains tvm "${toolchains[@]}" && contains rtl "${phases[@]}" && \
   [[ "${dry_run}" -eq 0 ]]; then
  prepare_verilator
fi

for ((trial = 1; trial <= repeats; trial++)); do
  for kernel in "${kernels[@]}"; do
    if contains pytorch "${toolchains[@]}" && contains compile "${phases[@]}"; then
      run_pytorch_compile "${trial}" "${kernel}" || break 2
    fi
    if contains tvm "${toolchains[@]}" && contains compile "${phases[@]}"; then
      run_tvm_compile "${trial}" "${kernel}" || break 2
    fi
    if contains tvm "${toolchains[@]}" && contains rtl "${phases[@]}"; then
      run_tvm_rtl "${trial}" "${kernel}" || break 2
    fi
    if contains pytorch "${toolchains[@]}" && contains rtl "${phases[@]}"; then
      run_pytorch_rtl "${trial}" "${kernel}" || break 2
    fi
  done
done

"${RESULTS_PYTHON}" "${RESULTS_TOOL}" summarize \
  --csv "${raw_csv}" --output "${summary_csv}"
"${RESULTS_PYTHON}" "${RESULTS_TOOL}" latex \
  --summary "${summary_csv}" --output "${latex_rows}"

complete_matrix=0
if [[ "${dry_run}" -eq 0 && "${failures}" -eq 0 ]] && \
   [[ "${#kernels[@]}" -eq "${#ALL_KERNELS[@]}" ]] && \
   contains pytorch "${toolchains[@]}" && contains tvm "${toolchains[@]}" && \
   contains compile "${phases[@]}" && contains rtl "${phases[@]}"; then
  complete_matrix=1
fi
if [[ "${complete_matrix}" -eq 1 ]]; then
  awk -F, 'NR > 1 && $NF != "PASS" { exit 1 }' "${summary_csv}" || \
    die "internal summary validation failed"
  cp -f "${summary_csv}" "${SCRIPT_DIR}/figures/table2.csv"
  cp -f "${latex_rows}" "${SCRIPT_DIR}/figures/table2_rows.tex"
  log "updated stable Table 2 artifacts under ${SCRIPT_DIR}/figures"
fi

log "summary: ${summary_csv}"
[[ "${failures}" -eq 0 ]] || die "${failures} measurement(s) failed; resume with --resume --output-dir=${output_dir}"
