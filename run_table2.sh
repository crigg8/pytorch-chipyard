#!/usr/bin/env bash
set -euo pipefail

TABLE2_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
RESULTS_TOOL="${TABLE2_REPO_ROOT}/scripts/table2_results.py"

log() {
  printf '[table2] %s\n' "$*"
}

warn() {
  printf '[table2][warn] %s\n' "$*" >&2
}

die() {
  printf '[table2][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash run_table2.sh [options]

Measure the four CNN rows in Table 2 for PyTorch-Chipyard and TVM-Gemmini.
Compilation is reported both as total host wall time and wall time per generated
kernel. Simulation host wall time is reported in separate Spike and FireSim
columns.

Options:
  --models=LIST       alexnet,squeezenet,mobilenetv2,resnet50 (default: all)
  --toolchains=LIST   pytorch,tvm (default: both)
  --phases=LIST       compile,spike,firesim (default: all)
  --repeats=N         Independent trials; summary uses successful medians (default: 1)
  --output-dir=PATH   Result directory (default: results/table2/<UTC run id>)
  --dry-run           Print the experiment matrix without compiling or simulating
  -h, --help          Show this help

Outputs:
  raw.csv             One row per trial and phase, including artifact/log paths
  table2.csv          Paper-facing medians with compile/kernel, Spike, FireSim columns
  logs/               Full command logs
  artifacts/          Per-trial compiler artifacts

On a successful non-dry run, table2.csv is also copied to the stable paper
artifact path scripts/figures/table2.csv.

Environment:
  TABLE2_ACCOUNT_ENV                Account setup script. Default: $HOME/.ae-env.sh
  TABLE2_TVM_AE_ROOT                TVM AE tree. Default: /home/ae/tvm-gemmini-ae
  TABLE2_PYTORCH_SPIKE_BIN          Gemmini Spike compatible with the PyTorch-
                                    Chipyard FP32 DIM=8 configuration
  TABLE2_PYTORCH_PK_BIN             Proxy kernel used for the static Linux ELF
  TABLE2_TVM_SPIKE_BIN              TVM-Gemmini Spike override
  TABLE2_PYTORCH_FIRESIM_HW_CONFIG  Default: FP32 DIM=8 Gemmini Rocket, 4 cores
  TABLE2_PYTORCH_FIRESIM_RESULT_DIR Collected 4-core result root. Default:
                                    scripts/figures/results-workload
  TABLE2_TVM_FIRESIM_HW_CONFIG      Default: default INT8 Gemmini Rocket, 1 core
  TABLE2_TVM_MANAGE_LLVM=0          Do not temporarily rebuild TVM with LLVM
  TABLE2_KEEP_GOING=0               Stop scheduling work after the first failure
  PYTORCH_CHIPYARD_CONDA_ENV        PyTorch-Chipyard conda env (default:
                                    pytorch-chipyard)

Notes:
  - PyTorch kernel_count is len(model_spec["kernels"]), including generated
    TorchInductor autotuning candidates.
  - TVM kernel_count is the number of generated tvmgen_default compute-function
    definitions, excluding __tvm_main__.
  - PyTorch-Chipyard FireSim first reuses a successful collected
    <model>-gemmini-4core/uartlog. If it is absent or invalid, the script runs
    the 4-core workload and caches its timing log alongside that result.
  - FireSim image preparation and infrasetup are not timed. A reused UART uses
    FireSim's `Wallclock Time Elapsed`; a fallback run times runworkload only.
EOF
}

models_arg="alexnet,squeezenet,mobilenetv2,resnet50"
toolchains_arg="pytorch,tvm"
phases_arg="compile,spike,firesim"
repeats="${TABLE2_REPEATS:-1}"
output_dir=""
dry_run=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --models=*) models_arg="${1#*=}"; shift ;;
    --models) [[ "$#" -ge 2 ]] || die "--models requires a value"; models_arg="$2"; shift 2 ;;
    --toolchains=*) toolchains_arg="${1#*=}"; shift ;;
    --toolchains) [[ "$#" -ge 2 ]] || die "--toolchains requires a value"; toolchains_arg="$2"; shift 2 ;;
    --phases=*) phases_arg="${1#*=}"; shift ;;
    --phases) [[ "$#" -ge 2 ]] || die "--phases requires a value"; phases_arg="$2"; shift 2 ;;
    --repeats=*) repeats="${1#*=}"; shift ;;
    --repeats) [[ "$#" -ge 2 ]] || die "--repeats requires a value"; repeats="$2"; shift 2 ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --output-dir) [[ "$#" -ge 2 ]] || die "--output-dir requires a value"; output_dir="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown argument '$1'; pass --help for usage" ;;
  esac
done

[[ "${repeats}" =~ ^[1-9][0-9]*$ ]] || die "--repeats must be a positive integer"

split_and_validate() {
  local raw="$1"
  local kind="$2"
  local out_name="$3"
  local -n out_ref="${out_name}"
  local value existing
  local parts=()
  out_ref=()
  IFS=',' read -r -a parts <<<"${raw}"
  for value in "${parts[@]}"; do
    value="${value//[[:space:]]/}"
    [[ -n "${value}" ]] || continue
    case "${kind}:${value}" in
      model:alexnet | model:squeezenet | model:mobilenetv2 | model:resnet50 | \
      toolchain:pytorch | toolchain:tvm | \
      phase:compile | phase:spike | phase:firesim) ;;
      *) die "invalid ${kind} '${value}'" ;;
    esac
    for existing in "${out_ref[@]}"; do
      [[ "${existing}" == "${value}" ]] && continue 2
    done
    out_ref+=("${value}")
  done
  [[ "${#out_ref[@]}" -gt 0 ]] || die "empty ${kind} list"
}

models=()
toolchains=()
phases=()
split_and_validate "${models_arg}" model models
split_and_validate "${toolchains_arg}" toolchain toolchains
split_and_validate "${phases_arg}" phase phases

contains() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    [[ "${value}" == "${needle}" ]] && return 0
  done
  return 1
}

if ! contains compile "${phases[@]}" && \
    { contains spike "${phases[@]}" || contains firesim "${phases[@]}"; }; then
  die "simulation phases currently require compile in --phases so each run uses a fresh artifact"
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${output_dir}" ]]; then
  output_dir="${TABLE2_REPO_ROOT}/results/table2/${run_id}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
mkdir -p "${output_dir}/logs" "${output_dir}/artifacts" "${output_dir}/setup"
raw_csv="${output_dir}/raw.csv"
table_csv="${output_dir}/table2.csv"
paper_table_csv="${TABLE2_REPO_ROOT}/scripts/figures/table2.csv"
python "${RESULTS_TOOL}" init --csv "${raw_csv}"

account_env="${TABLE2_ACCOUNT_ENV:-${HOME}/.ae-env.sh}"
if [[ -f "${account_env}" ]]; then
  log "sourcing account environment: ${account_env}"
  set +u
  source "${account_env}"
  set -u
fi
set +u
source "${TABLE2_REPO_ROOT}/scripts/env.sh"
set -u

tvm_ae_root="${TABLE2_TVM_AE_ROOT:-/home/ae/tvm-gemmini-ae}"
tvm_source_root=""
tvm_environment_loaded=0
tvm_restore_llvm=0
firesim_active=0
firesim_runtime=""
failures=0
keep_going="${TABLE2_KEEP_GOING:-1}"

load_tvm_environment() {
  [[ "${tvm_environment_loaded}" -eq 0 ]] || return 0
  [[ -f "${tvm_ae_root}/scripts/env.sh" ]] || die "TVM AE environment not found: ${tvm_ae_root}/scripts/env.sh"
  set +u
  source "${tvm_ae_root}/scripts/env.sh"
  set -u
  tvm_source_root="${AE_ROOT}"
  tvm_environment_loaded=1
}

configure_tvm() {
  local llvm_value="$1"
  local log_path="$2"
  cmake -S "${TVM_DIR}" -B "${TVM_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DUSE_GEMMINI=ON -DUSE_MICRO=ON \
    -DUSE_LLVM="${llvm_value}" -DUSE_OPENMP=none -DUSE_CUDA=OFF \
    -DUSE_VULKAN=OFF -DUSE_OPENCL=OFF >>"${log_path}" 2>&1
  cmake --build "${TVM_BUILD_DIR}" --target tvm gemmini standalone_crt \
    -j"${TVM_BUILD_JOBS}" >>"${log_path}" 2>&1
}

cleanup() {
  local status=$?
  if [[ "${firesim_active}" -eq 1 && -n "${firesim_runtime}" ]]; then
    warn "terminating active FireSim run farm"
    (
      set +u
      source "${FIRESIM_DIR}/sourceme-manager.sh" >/dev/null 2>&1
      set -u
      cd "${FIRESIM_DEPLOY_DIR}"
      firesim kill -c "${firesim_runtime}" -a "${FIRESIM_HWDB_PATH}" \
        -r "${FIRESIM_BUILD_RECIPES_PATH}" >/dev/null 2>&1 || true
      firesim terminaterunfarm -c "${firesim_runtime}" -a "${FIRESIM_HWDB_PATH}" \
        -r "${FIRESIM_BUILD_RECIPES_PATH}" --forceterminate >/dev/null 2>&1 || true
    )
  fi
  if [[ "${tvm_restore_llvm}" -eq 1 ]]; then
    log "restoring TVM persistent build to USE_LLVM=OFF"
    configure_tvm OFF "${output_dir}/setup/tvm-restore-llvm-off.log" || \
      warn "failed to restore TVM USE_LLVM=OFF; see setup log"
  fi
  return "${status}"
}
trap cleanup EXIT INT TERM

record() {
  python "${RESULTS_TOOL}" append --csv "${raw_csv}" --run-id "${run_id}" "$@"
  python "${RESULTS_TOOL}" summarize --csv "${raw_csv}" --output "${table_csv}"
}

LAST_RC=0
LAST_WALL_S=""
LAST_COMPILE_OK=0
run_logged_timed() {
  local log_path="$1"
  shift
  local start_ns end_ns
  mkdir -p "$(dirname -- "${log_path}")"
  start_ns="$(date +%s%N)"
  set +e
  "$@" > >(tee "${log_path}") 2>&1
  LAST_RC=$?
  set -e
  end_ns="$(date +%s%N)"
  LAST_WALL_S="$(awk -v start="${start_ns}" -v end="${end_ns}" 'BEGIN { printf "%.6f", (end-start)/1000000000 }')"
}

display_model() {
  case "$1" in
    alexnet) printf '%s\n' AlexNet ;;
    squeezenet) printf '%s\n' SqueezeNet ;;
    mobilenetv2) printf '%s\n' MobileNetV2 ;;
    resnet50) printf '%s\n' ResNet50 ;;
  esac
}

mark_failure() {
  failures=$((failures + 1))
  if [[ "${keep_going}" == "0" ]]; then
    return 1
  fi
  return 0
}

pc_compile() {
  local model="$1"
  local trial="$2"
  local artifact="$3"
  local log_path="$4"
  local wall_s kernel_count status
  if [[ "${dry_run}" -eq 1 ]]; then
    log "DRY RUN compile pytorch ${model} trial=${trial} -> ${artifact}"
    LAST_COMPILE_OK=1
    return 0
  fi
  run_logged_timed "${log_path}" bash "${TABLE2_REPO_ROOT}/scripts/run-cnn.sh" \
    --backend=gemmini --model="${model}" --artifact-dir="${artifact}"
  if [[ "${LAST_RC}" -eq 0 ]]; then
    if wall_s="$(python "${RESULTS_TOOL}" extract --log "${log_path}" --key seconds 2>/dev/null)" && \
        kernel_count="$(python "${RESULTS_TOOL}" count-pytorch "${artifact}/model_spec.json" 2>/dev/null)"; then
      status=PASS
    else
      wall_s="${LAST_WALL_S}"
      kernel_count=""
      status=FAIL
    fi
  else
    wall_s="${LAST_WALL_S}"
    kernel_count=""
    status=FAIL
  fi
  record --trial "${trial}" --workload "$(display_model "${model}")" \
    --toolchain PyTorch-Chipyard --phase compile --total-wall-s "${wall_s}" \
    --kernel-count "${kernel_count}" --status "${status}" --exit-code "${LAST_RC}" \
    --artifact-path "${artifact}" --log-path "${log_path}" \
    --notes 'TorchInductor model compile; autotuning candidates included'
  LAST_COMPILE_OK=0
  if [[ "${status}" == PASS ]]; then
    LAST_COMPILE_OK=1
  fi
  [[ "${status}" == PASS ]] || mark_failure
}

tvm_compile_command() {
  local model="$1"
  local trial_root="$2"
  export AE_ROOT="${trial_root}"
  export TEST_DATA_ROOT_PATH="${tvm_source_root}/cache/tvm-test-data"
  mkdir -p "${AE_ROOT}/work"
  if [[ "${model}" == mobilenetv2 ]]; then
    local work="${AE_ROOT}/work/mobilenet"
    mkdir -p "${work}"
    cp -f "${tvm_source_root}/work/mobilenet/micro_gemmini_mobilenet-instrumented.py" \
      "${work}/table2_mobilenet.py"
    (cd "${work}" && "${TVM_ENV}/bin/python" ./table2_mobilenet.py)
  else
    "${TVM_ENV}/bin/python" "${tvm_source_root}/scripts/port-vision-model.py" "${model}"
  fi
}

tvm_model_dir() {
  local model="$1"
  local trial_root="$2"
  if [[ "${model}" == mobilenetv2 ]]; then
    printf '%s\n' "${trial_root}/work/mobilenet/generated-project/src/model"
  else
    printf '%s\n' "${trial_root}/work/${model}/generated-project/src/model"
  fi
}

tvm_elf_path() {
  local model="$1"
  local trial_root="$2"
  if [[ "${model}" == mobilenetv2 ]]; then
    printf '%s\n' "${trial_root}/work/mobilenet/generated-project/src/build/mobilenet-baremetal"
  else
    printf '%s\n' "${trial_root}/work/${model}/generated-project/src/build/mobilenet-baremetal"
  fi
}

tvm_compile() {
  local model="$1"
  local trial="$2"
  local trial_root="$3"
  local log_path="$4"
  local wall_s kernel_count status model_dir
  if [[ "${dry_run}" -eq 1 ]]; then
    log "DRY RUN compile tvm ${model} trial=${trial} -> ${trial_root}"
    LAST_COMPILE_OK=1
    return 0
  fi
  run_logged_timed "${log_path}" tvm_compile_command "${model}" "${trial_root}"
  if [[ "${LAST_RC}" -eq 0 ]]; then
    model_dir="$(tvm_model_dir "${model}" "${trial_root}")"
    if wall_s="$(python "${RESULTS_TOOL}" extract --log "${log_path}" --key RELAY_COMPILE_WALL_S 2>/dev/null)" && \
        kernel_count="$(python "${RESULTS_TOOL}" count-tvm "${model_dir}" 2>/dev/null)"; then
      status=PASS
    else
      wall_s="${LAST_WALL_S}"
      kernel_count=""
      status=FAIL
    fi
  else
    wall_s="${LAST_WALL_S}"
    kernel_count=""
    status=FAIL
  fi
  record --trial "${trial}" --workload "$(display_model "${model}")" \
    --toolchain TVM-Gemmini --phase compile --total-wall-s "${wall_s}" \
    --kernel-count "${kernel_count}" --status "${status}" --exit-code "${LAST_RC}" \
    --artifact-path "${trial_root}" --log-path "${log_path}" \
    --notes 'Relay build time; generated compute functions excluding __tvm_main__'
  LAST_COMPILE_OK=0
  if [[ "${status}" == PASS ]]; then
    LAST_COMPILE_OK=1
  fi
  [[ "${status}" == PASS ]] || mark_failure
}

build_pc_elf() {
  local artifact="$1"
  local core="$2"
  local spike_mode="$3"
  local log_path="$4"
  local env_value=0
  [[ "${spike_mode}" == 1 ]] && env_value=1
  run_logged_timed "${log_path}" env PYTORCH_CHIPYARD_SPIKE_EXECUTABLE="${env_value}" \
    bash "${TABLE2_REPO_ROOT}/scripts/build-chipyard-elves.sh" --artifact-dir="${artifact}" \
    --backend=gemmini --cores="${core}"
  return "${LAST_RC}"
}

pc_spike_command() {
  local artifact="$1"
  local spike_bin="$2"
  local pk_bin="$3"
  (cd "${artifact}" && "${spike_bin}" -p1 --extension=gemmini "${pk_bin}" ./model-1core.elf)
}

tvm_spike_command() {
  local elf="$1"
  local spike_bin="$2"
  "${spike_bin}" -p1 --extension=gemmini "${elf}"
}

run_spike_measurement() {
  local toolchain="$1"
  local model="$2"
  local trial="$3"
  local artifact="$4"
  local log_path="${output_dir}/logs/${toolchain}-${model}-trial${trial}-spike.log"
  local build_log="${output_dir}/logs/${toolchain}-${model}-trial${trial}-spike-build.log"
  local status spike_bin pk_bin elf
  if [[ "${dry_run}" -eq 1 ]]; then
    log "DRY RUN spike ${toolchain} ${model} trial=${trial}"
    return 0
  fi

  if [[ "${toolchain}" == pytorch ]]; then
    spike_bin="${TABLE2_PYTORCH_SPIKE_BIN:-${CHIPYARD_DIR}/.conda-env/riscv-tools/bin/spike}"
    pk_bin="${TABLE2_PYTORCH_PK_BIN:-${CHIPYARD_DIR}/.conda-env/riscv-tools/riscv64-unknown-elf/bin/pk}"
    [[ -x "${spike_bin}" ]] || die "PyTorch Spike not executable: ${spike_bin}"
    [[ -f "${pk_bin}" ]] || die "proxy kernel not found: ${pk_bin}"
    if ! build_pc_elf "${artifact}" 1 1 "${build_log}"; then
      record --trial "${trial}" --workload "$(display_model "${model}")" \
        --toolchain PyTorch-Chipyard --phase simulation --simulator spike \
        --status FAIL --exit-code "${LAST_RC}" --artifact-path "${artifact}" \
        --log-path "${build_log}" --notes 'static ELF build failed'
      mark_failure
      return
    fi
    run_logged_timed "${log_path}" pc_spike_command "${artifact}" "${spike_bin}" "${pk_bin}"
    elf="${artifact}/model-1core.elf"
  else
    load_tvm_environment
    spike_bin="${TABLE2_TVM_SPIKE_BIN:-${SPIKE_BIN}}"
    elf="$(tvm_elf_path "${model}" "${artifact}")"
    [[ -x "${spike_bin}" ]] || die "TVM Spike not executable: ${spike_bin}"
    [[ -x "${elf}" ]] || die "TVM ELF not found: ${elf}"
    run_logged_timed "${log_path}" tvm_spike_command "${elf}" "${spike_bin}"
  fi
  status=PASS
  [[ "${LAST_RC}" -eq 0 ]] || status=FAIL
  record --trial "${trial}" --workload "$(display_model "${model}")" \
    --toolchain "$([[ "${toolchain}" == pytorch ]] && printf PyTorch-Chipyard || printf TVM-Gemmini)" \
    --phase simulation --simulator spike --total-wall-s "${LAST_WALL_S}" \
    --status "${status}" --exit-code "${LAST_RC}" --artifact-path "${elf}" \
    --log-path "${log_path}" --notes 'host wall time; one Spike hart'
  [[ "${status}" == PASS ]] || mark_failure
}

marshal_build_install() {
  local workdir="$1"
  local config="$2"
  local log_path="$3"
  (
    cd "${FIREMARSHAL_DIR}"
    ./marshal --workdir "${workdir}" build "${config}"
    ./marshal --workdir "${workdir}" install "${config}"
  ) >>"${log_path}" 2>&1
}

prepare_pc_firesim() {
  local model="$1"
  local trial="$2"
  local artifact="$3"
  local workload_name="$4"
  local log_path="$5"
  local hint="${workload_name%-gemmini-4core}/gemmini"
  if ! build_pc_elf "${artifact}" 4 0 "${log_path}"; then
    return 1
  fi
  printf '%s\n' "${hint}" >"${artifact}/.pytorch-chipyard-workload-rel"
  PYTORCH_CHIPYARD_WORKLOAD_DIR="${output_dir}/firemarshal/pytorch" \
    FIRESIM_WORKLOAD_DIR="${FIRESIM_WORKLOAD_DIR}" \
    bash "${TABLE2_REPO_ROOT}/scripts/package-firemarshal-workload.sh" \
      --artifact-dir="${artifact}" --no-clean >>"${log_path}" 2>&1
  marshal_build_install "${output_dir}/firemarshal/pytorch" \
    "${output_dir}/firemarshal/pytorch/${workload_name}.json" "${log_path}"
}

pc_firesim_workload_name() {
  printf '%s-gemmini-4core\n' "$1"
}

pc_firesim_result_root() {
  printf '%s\n' "${TABLE2_PYTORCH_FIRESIM_RESULT_DIR:-${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}}"
}

find_pc_firesim_log() {
  local model="$1"
  local workload result_dir candidate
  workload="$(pc_firesim_workload_name "${model}")"
  result_dir="$(pc_firesim_result_root)/${workload}"

  # Prefer FireSim's UART summary. The sidecar is produced only when this
  # script had to execute a missing 4-core result itself.
  for candidate in "${result_dir}/uartlog" "${result_dir}/table2-firesim.log"; do
    [[ -f "${candidate}" ]] || continue
    if python "${RESULTS_TOOL}" extract-firesim-wall --log "${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    warn "ignoring incomplete FireSim cache: ${candidate}"
  done
  return 1
}

cache_pc_firesim_log() {
  local model="$1"
  local source_log="$2"
  local result_dir
  result_dir="$(pc_firesim_result_root)/$(pc_firesim_workload_name "${model}")"
  mkdir -p "${result_dir}"
  cp -f "${source_log}" "${result_dir}/table2-firesim.log"
}

prepare_tvm_firesim() {
  local model="$1"
  local artifact="$2"
  local workload_name="$3"
  local log_path="$4"
  local workdir="${output_dir}/firemarshal/tvm/${workload_name}"
  local elf="$(tvm_elf_path "${model}" "${artifact}")"
  mkdir -p "${workdir}"
  cp -f "${elf}" "${workdir}/${workload_name}.elf"
  cat >"${workdir}/${workload_name}.json" <<EOF
{
  "name": "${workload_name}",
  "base": "bare-base.json",
  "bin": "${workload_name}.elf"
}
EOF
  marshal_build_install "${workdir}" "${workdir}/${workload_name}.json" "${log_path}"
}

generate_firesim_runtime() {
  local workload_name="$1"
  local hw_config="$2"
  local runtime="$3"
  local simulation_dir="${output_dir}/firesim-runs/${workload_name}"
  mkdir -p "${simulation_dir}" "$(dirname -- "${runtime}")"
  cat >"${runtime}" <<EOF
run_farm:
  base_recipe: run-farm-recipes/externally_provisioned.yaml
  recipe_arg_overrides:
    default_platform: XilinxAlveoU250InstanceDeployManager
    default_simulation_dir: "${simulation_dir}"
    default_fpga_db: "${PYTORCH_CHIPYARD_FPGA_DB}"
    run_farm_hosts_to_use:
      - "${PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_HOST:-localhost}": ${PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_SPEC:-one_fpgas_spec}
metasimulation:
  metasimulation_enabled: false
  metasimulation_host_simulator: verilator
  metasimulation_only_plusargs: "+fesvr-step-size=128 +max-cycles=100000000"
  metasimulation_only_vcs_plusargs: "+vcs+initreg+0 +vcs+initmem+0"
target_config:
  topology: no_net_config
  no_net_num_nodes: 1
  link_latency: 6405
  switching_latency: 10
  net_bandwidth: 200
  profile_interval: -1
  default_hw_config: ${hw_config}
  plusarg_passthrough: ""
tracing:
  enable: no
  output_format: 0
  selector: 1
  start: 0
  end: -1
autocounter:
  read_rate: 0
workload:
  workload_name: ${workload_name}.json
  terminate_on_completion: yes
  suffix_tag: ${workload_name}
host_debug:
  zero_out_dram: no
  disable_synth_asserts: no
synth_print:
  start: 0
  end: -1
  cycle_prefix: yes
EOF
}

firesim_command() {
  local runtime="$1"
  set +u
  source "${FIRESIM_DIR}/sourceme-manager.sh"
  set -u
  cd "${FIRESIM_DEPLOY_DIR}"
  firesim_active=1
  firesim_runtime="${runtime}"
  firesim launchrunfarm -c "${runtime}" -a "${FIRESIM_HWDB_PATH}" -r "${FIRESIM_BUILD_RECIPES_PATH}"
  firesim infrasetup -c "${runtime}" -a "${FIRESIM_HWDB_PATH}" -r "${FIRESIM_BUILD_RECIPES_PATH}"
  local start_ns end_ns status=0
  start_ns="$(date +%s%N)"
  firesim runworkload -c "${runtime}" -a "${FIRESIM_HWDB_PATH}" -r "${FIRESIM_BUILD_RECIPES_PATH}" || status=$?
  end_ns="$(date +%s%N)"
  awk -v start="${start_ns}" -v end="${end_ns}" \
    'BEGIN { printf "TABLE2_FIRESIM_WALL_S=%.6f\n", (end-start)/1000000000 }'
  if [[ "${status}" -ne 0 ]]; then
    firesim kill -c "${runtime}" -a "${FIRESIM_HWDB_PATH}" -r "${FIRESIM_BUILD_RECIPES_PATH}" || true
  fi
  firesim terminaterunfarm -c "${runtime}" -a "${FIRESIM_HWDB_PATH}" \
    -r "${FIRESIM_BUILD_RECIPES_PATH}" --forceterminate || true
  firesim_active=0
  firesim_runtime=""
  return "${status}"
}

run_firesim_measurement() {
  local toolchain="$1"
  local model="$2"
  local trial="$3"
  local artifact="$4"
  local unique_model="table2-${run_id,,}-${toolchain}-${model}-trial${trial}"
  local workload_name hw_config cached_log=""
  local prepare_log="${output_dir}/logs/${toolchain}-${model}-trial${trial}-firesim-prepare.log"
  local run_log="${output_dir}/logs/${toolchain}-${model}-trial${trial}-firesim.log"
  local runtime="${output_dir}/firesim-runtime/${unique_model}.yaml"
  local status wall_s
  if [[ "${toolchain}" == pytorch ]]; then
    hw_config="${TABLE2_PYTORCH_FIRESIM_HW_CONFIG:-alveo_u250_firesim_fp8x8_gemmini_rocket_4core_no_nic}"
    cached_log="$(find_pc_firesim_log "${model}" || true)"
    if [[ -n "${cached_log}" ]]; then
      wall_s="$(python "${RESULTS_TOOL}" extract-firesim-wall --log "${cached_log}")"
      log "reusing 4-core FireSim result: ${cached_log} (${wall_s}s)"
      if [[ "${dry_run}" -eq 0 ]]; then
        record --trial "${trial}" --workload "$(display_model "${model}")" \
          --toolchain PyTorch-Chipyard --phase simulation --simulator firesim \
          --total-wall-s "${wall_s}" --status PASS --exit-code 0 \
          --artifact-path "$(dirname -- "${cached_log}")" --log-path "${cached_log}" \
          --notes "reused successful 4-core Gemmini FireSim result; hw=${hw_config}"
      fi
      return 0
    fi
    log "no valid 4-core FireSim result for ${model}; scheduling fallback execution"
  fi
  if [[ "${dry_run}" -eq 1 ]]; then
    log "DRY RUN firesim ${toolchain} ${model} trial=${trial}"
    return 0
  fi
  [[ -f "${FIREMARSHAL_DIR}/marshal" ]] || die "FireMarshal not found under ${FIREMARSHAL_DIR}"
  [[ -f "${FIRESIM_DIR}/sourceme-manager.sh" ]] || die "FireSim manager environment not found"
  [[ -f "${PYTORCH_CHIPYARD_FPGA_DB}" ]] || die "FPGA DB not found: ${PYTORCH_CHIPYARD_FPGA_DB}"

  : >"${prepare_log}"
  if [[ "${toolchain}" == pytorch ]]; then
    workload_name="${unique_model}-gemmini-4core"
    hw_config="${TABLE2_PYTORCH_FIRESIM_HW_CONFIG:-alveo_u250_firesim_fp8x8_gemmini_rocket_4core_no_nic}"
    if ! prepare_pc_firesim "${model}" "${trial}" "${artifact}" "${workload_name}" "${prepare_log}"; then
      status=FAIL
    else
      status=PASS
    fi
  else
    workload_name="${unique_model}"
    hw_config="${TABLE2_TVM_FIRESIM_HW_CONFIG:-alveo_u250_firesim_gemmini_rocket_singlecore_no_nic}"
    if ! prepare_tvm_firesim "${model}" "${artifact}" "${workload_name}" "${prepare_log}"; then
      status=FAIL
    else
      status=PASS
    fi
  fi
  if [[ "${status}" == FAIL ]]; then
    record --trial "${trial}" --workload "$(display_model "${model}")" \
      --toolchain "$([[ "${toolchain}" == pytorch ]] && printf PyTorch-Chipyard || printf TVM-Gemmini)" \
      --phase simulation --simulator firesim --status FAIL --exit-code 1 \
      --artifact-path "${artifact}" --log-path "${prepare_log}" \
      --notes 'FireMarshal/ELF preparation failed; preparation is not timed'
    mark_failure
    return
  fi

  generate_firesim_runtime "${workload_name}" "${hw_config}" "${runtime}"
  run_logged_timed "${run_log}" firesim_command "${runtime}"
  if ! wall_s="$(python "${RESULTS_TOOL}" extract --log "${run_log}" --key TABLE2_FIRESIM_WALL_S 2>/dev/null)"; then
    wall_s="${LAST_WALL_S}"
  fi
  status=PASS
  [[ "${LAST_RC}" -eq 0 ]] || status=FAIL
  if [[ "${status}" == PASS && "${toolchain}" == pytorch ]]; then
    cache_pc_firesim_log "${model}" "${run_log}"
  fi
  record --trial "${trial}" --workload "$(display_model "${model}")" \
    --toolchain "$([[ "${toolchain}" == pytorch ]] && printf PyTorch-Chipyard || printf TVM-Gemmini)" \
    --phase simulation --simulator firesim --total-wall-s "${wall_s}" \
    --status "${status}" --exit-code "${LAST_RC}" --artifact-path "${artifact}" \
    --log-path "${run_log}" --notes "firesim runworkload wall time only; hw=${hw_config}"
  [[ "${status}" == PASS ]] || mark_failure
}

log "run id      : ${run_id}"
log "output      : ${output_dir}"
log "models      : ${models[*]}"
log "toolchains  : ${toolchains[*]}"
log "phases      : ${phases[*]}"
log "repeats     : ${repeats}"

if contains tvm "${toolchains[@]}" && contains compile "${phases[@]}" && [[ "${dry_run}" -eq 0 ]]; then
  load_tvm_environment
  if [[ "${TABLE2_TVM_MANAGE_LLVM:-1}" == 1 ]]; then
    log "temporarily configuring TVM with LLVM for Relay compilation"
    tvm_restore_llvm=1
    configure_tvm "${LLVM_CONFIG}" "${output_dir}/setup/tvm-enable-llvm.log"
  fi
fi

stop_requested=0
for toolchain in "${toolchains[@]}"; do
  for model in "${models[@]}"; do
    for ((trial = 1; trial <= repeats; trial++)); do
      LAST_COMPILE_OK=0
      if [[ "${toolchain}" == pytorch ]]; then
        artifact="${output_dir}/artifacts/pytorch-chipyard/${model}/trial-${trial}/gemmini"
        compile_log="${output_dir}/logs/pytorch-${model}-trial${trial}-compile.log"
        if contains compile "${phases[@]}"; then
          pc_compile "${model}" "${trial}" "${artifact}" "${compile_log}" || stop_requested=1
        fi
      else
        load_tvm_environment
        artifact="${output_dir}/artifacts/tvm-gemmini/${model}/trial-${trial}"
        compile_log="${output_dir}/logs/tvm-${model}-trial${trial}-compile.log"
        if contains compile "${phases[@]}"; then
          tvm_compile "${model}" "${trial}" "${artifact}" "${compile_log}" || stop_requested=1
        fi
      fi
      [[ "${stop_requested}" -eq 0 ]] || break 3
      if contains compile "${phases[@]}" && [[ "${LAST_COMPILE_OK}" -eq 0 ]]; then
        warn "skipping simulations after failed compile: ${toolchain}/${model}/trial-${trial}"
        continue
      fi
      if contains spike "${phases[@]}"; then
        run_spike_measurement "${toolchain}" "${model}" "${trial}" "${artifact}" || stop_requested=1
      fi
      [[ "${stop_requested}" -eq 0 ]] || break 3
      if contains firesim "${phases[@]}"; then
        run_firesim_measurement "${toolchain}" "${model}" "${trial}" "${artifact}" || stop_requested=1
      fi
      [[ "${stop_requested}" -eq 0 ]] || break 3
    done
  done
done

python "${RESULTS_TOOL}" summarize --csv "${raw_csv}" --output "${table_csv}"
log "raw results : ${raw_csv}"
log "Table 2    : ${table_csv}"

if [[ "${failures}" -ne 0 ]]; then
  die "${failures} measurement phase(s) failed; inspect raw.csv and logs"
fi

if [[ "${dry_run}" -eq 0 ]]; then
  mkdir -p "$(dirname -- "${paper_table_csv}")"
  cp -f "${table_csv}" "${paper_table_csv}"
  log "paper table: ${paper_table_csv}"
fi
