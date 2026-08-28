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
  --experiment=NAME     Run one resumable paper experiment unit. NAME is one of:
                          figures-6-7-8-9-table5
                          figure-10
                          figure-11
                          figure-13
                          table-4
                          simple
                        Completed ordinary workloads are detected under the
                        collected results directory and skipped automatically.
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
  --resume-rebuild-images
                        Skip ELF/package stages, deep-clean Linux/OpenSBI,
                        rebuild images only for workloads without complete
                        outputs, then run only those FireSim workloads.
  --resume-from=NAME    Skip ELF/package/image stages, keep collected results,
                        and restart FireSim at workload NAME.
  --rvv-panic-retries=N|unlimited
                        Retry an RVV workload after a detected guest kernel
                        panic. Default: 0; unlimited must be explicit.
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
  PYTORCH_CHIPYARD_SIMPLE_STAGE2=1 selects the quick partial Figure
  6/8/9/11/13 experiment. Prefer scripts/simple-stage2.sh to set it.
  TABLE4_TVM_AE_ROOT points to the prepared TVM-Gemmini AE tree. It defaults to
  $HOME/tvm-gemmini-ae on the author review server.
EOF
}

chipyard_env_arg=""
riscv_toolchain_dir_arg=""
riscv_gxx_arg=""
workload_args=()
experiment_arg=""
experiment_name=""
experiment_selected=0
experiment_artifact_dirs=()
experiment_pending_artifact_dirs=()
experiment_workloads=()
experiment_pending_workloads=()
only_alias_first=0
only_alias_first_cnn_off=0
resume_firesim=0
rebuild_pending_images=0
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
    --experiment)
      [[ "$#" -ge 2 ]] || die "--experiment requires a value"
      experiment_arg="$2"
      shift 2
      ;;
    --experiment=*)
      experiment_arg="${1#--experiment=}"
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
    --resume-rebuild-images)
      resume_firesim=1
      rebuild_pending_images=1
      skip_elves=1
      skip_package=1
      skip_images=0
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

if [[ -n "${experiment_arg}" ]]; then
  if [[ "${#workload_args[@]}" -gt 0 || "${only_alias_first}" -eq 1 || \
        "${only_alias_first_cnn_off}" -eq 1 || "${only_table4}" -eq 1 || \
        "${resume_firesim}" -eq 1 || -n "${resume_from_arg}" ]]; then
    die "--experiment cannot be combined with another workload, resume, or only-selection option"
  fi

  case "${experiment_arg}" in
    figures-6-7-8-9-table5 | figures-7-8-9-table5 | fig6-7-8-9-table5 | fig7-8-9-table5 | fig7-9-table5)
      experiment_name="figures-6-7-8-9-table5"
      ;;
    figure-10 | fig10)
      experiment_name="figure-10"
      ;;
    figure-11 | fig11)
      experiment_name="figure-11"
      ;;
    figure-13 | fig13)
      experiment_name="figure-13"
      ;;
    table-4 | table4)
      experiment_name="table-4"
      ;;
    simple | simple-stage2)
      experiment_name="simple"
      export PYTORCH_CHIPYARD_SIMPLE_STAGE2=1
      ;;
    *)
      die "unknown experiment '${experiment_arg}'; expected figures-6-7-8-9-table5, figure-10, figure-11, figure-13, table-4, or simple"
      ;;
  esac

  if [[ "${experiment_name}" == "table-4" ]]; then
    only_table4=1
  else
    experiment_selected=1
    resume_firesim=1
    skip_table4=1

    case "${experiment_name}" in
      figures-6-7-8-9-table5)
        for model in alexnet mobilenetv2 resnet50 squeezenet; do
          for backend in gemmini rvv scalar; do
            experiment_artifact_dirs+=(
              "${REPO_ROOT}/examples/artifact-${model}/${backend}"
            )
            case "${backend}" in
              gemmini | rvv) cores=(2 4) ;;
              scalar) cores=(4 8 16) ;;
            esac
            for core in "${cores[@]}"; do
              experiment_workloads+=("${model}-${backend}-${core}core")
            done
          done

          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-${model}/gemmini-alias-first-off"
          )
          experiment_workloads+=(
            "${model}-gemmini-alias-first-off-4core"
          )
        done

        for model in gpt2 gpt-neo opt pythia; do
          for mode in on off; do
            experiment_artifact_dirs+=(
              "${REPO_ROOT}/examples/artifact-${model}/gemmini/sdpa/seq256/alias-first-${mode}"
            )
            experiment_workloads+=(
              "${model}-gemmini-sdpa-256tok-alias-first-${mode}-4core"
            )
          done
        done

        for model in gpt2 gpt-neo; do
          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-${model}/gemmini"
          )
          for core in 2 4; do
            experiment_workloads+=("${model}-gemmini-${core}core")
          done
        done
        for model in opt pythia; do
          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-${model}/gemmini/sdpa/seq256"
          )
          for core in 2 4; do
            experiment_workloads+=(
              "${model}-rocket-gemmini-sdpa-256tok-${core}core"
            )
          done
        done
        ;;
      figure-10)
        # Figure 10 compares im2col against the same four-core direct Gemmini
        # runs already used by Figures 7--9.
        for model in alexnet mobilenetv2 resnet50 squeezenet; do
          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-${model}/gemmini"
            "${REPO_ROOT}/examples/artifact-${model}/gemmini-im2col"
          )
          experiment_workloads+=(
            "${model}-gemmini-4core"
            "${model}-gemmini-im2col-4core"
          )
        done
        ;;
      figure-11)
        experiment_artifact_dirs+=(
          "${REPO_ROOT}/examples/artifact-gemmini-max-autotune/gemmini"
        )
        experiment_workloads+=("gemmini-max-autotune-gemmini-4core")
        ;;
      figure-13)
        # Figure 13 reuses the OPT/Pythia seq=256 SDPA runs from Table 5 and
        # adds the remaining SDPA, FlashAttention, and windowed-attention runs.
        for model in opt pythia; do
          for attention in sdpa flash window; do
            for seq_len in 256 512 768 1024; do
              experiment_artifact_dirs+=(
                "${REPO_ROOT}/examples/artifact-${model}/gemmini/${attention}/seq${seq_len}"
              )
              experiment_workloads+=(
                "${model}-rocket-gemmini-${attention}-${seq_len}tok-4core"
              )
              if [[ "${model}" == "opt" && "${attention}" != "sdpa" ]]; then
                experiment_workloads+=(
                  "${model}-boom-gemmini-${attention}-${seq_len}tok-1core"
                )
              fi
            done
          done
        done
        ;;
      simple)
        experiment_artifact_dirs+=(
          "${REPO_ROOT}/examples/artifact-squeezenet/scalar"
          "${REPO_ROOT}/examples/artifact-squeezenet/rvv"
          "${REPO_ROOT}/examples/artifact-squeezenet/gemmini"
          "${REPO_ROOT}/examples/artifact-squeezenet/gemmini-alias-first-off"
          "${REPO_ROOT}/examples/artifact-opt/gemmini/sdpa/seq256"
          "${REPO_ROOT}/examples/artifact-opt/gemmini/sdpa/seq256/alias-first-on"
          "${REPO_ROOT}/examples/artifact-opt/gemmini/sdpa/seq256/alias-first-off"
        )
        experiment_workloads+=(
          "squeezenet-scalar-4core"
          "squeezenet-rvv-2core"
          "squeezenet-gemmini-2core"
          "squeezenet-gemmini-4core"
          "squeezenet-gemmini-alias-first-off-4core"
          "opt-rocket-gemmini-sdpa-256tok-2core"
          "opt-rocket-gemmini-sdpa-256tok-4core"
          "opt-gemmini-sdpa-256tok-alias-first-on-4core"
          "opt-gemmini-sdpa-256tok-alias-first-off-4core"
        )

        for attention in flash window; do
          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-opt/gemmini/${attention}/seq256"
          )
          experiment_workloads+=(
            "opt-rocket-gemmini-${attention}-256tok-4core"
            "opt-boom-gemmini-${attention}-256tok-1core"
          )
        done

        experiment_artifact_dirs+=(
          "${REPO_ROOT}/examples/artifact-gemmini-max-autotune-simple/gemmini"
        )
        experiment_workloads+=(
          "gemmini-max-autotune-simple-gemmini-4core"
        )
        ;;
    esac
    log "selected ${experiment_name}: ${#experiment_workloads[@]} workload(s)"
  fi
fi

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
# Experiment units are always resumable, so their collected results must remain
# available for the per-workload completion check below.
if [[ "${experiment_selected}" -eq 1 ]]; then
  export PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS=0
else
  export PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS="${PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS:-0}"
fi

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

if [[ "${experiment_name}" == "simple" ]]; then
  export PYTORCH_CHIPYARD_SIMPLE_STAGE2=1
fi

cd "${REPO_ROOT}"

should_collect_output_bin() {
  local workload="$1"

  case "${workload}" in
    gpt2-* | gpt-neo-* | opt-* | pythia-* | *tok*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

collected_result_complete() {
  local workload="$1"
  local result_dir="${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}/${workload}"

  [[ -f "${result_dir}/.completed" ]] || return 1
  [[ -s "${result_dir}/model.log" ]] || return 1
  [[ -s "${result_dir}/autotune.log" ]] || return 1
  if should_collect_output_bin "${workload}"; then
    [[ -s "${result_dir}/output.bin" ]] || return 1
  fi
  if [[ -f "${result_dir}/run.log" ]]; then
    grep -Fq '[runner] model_ret=0' "${result_dir}/run.log" || return 1
  fi
}

reusable_completed_workload() {
  local workload="$1"

  if collected_result_complete "${workload}"; then
    printf '%s\n' "${workload}"
    return 0
  fi
  if [[ "${workload}" == "gemmini-max-autotune-simple-gemmini-4core" ]] && \
      collected_result_complete "gemmini-max-autotune-gemmini-4core"; then
    printf '%s\n' "gemmini-max-autotune-gemmini-4core"
    return 0
  fi
  return 1
}

artifact_dir_for_workload() {
  local workload="$1"

  if [[ "${workload}" =~ ^(alexnet|mobilenetv2|resnet50|squeezenet)-(gemmini|rvv|scalar)-[0-9]+core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
  elif [[ "${workload}" =~ ^(alexnet|mobilenetv2|resnet50|squeezenet)-gemmini-(alias-first-off|im2col)-[0-9]+core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/gemmini-${BASH_REMATCH[2]}"
  elif [[ "${workload}" =~ ^(gpt2|gpt-neo)-gemmini-[0-9]+core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/gemmini"
  elif [[ "${workload}" =~ ^(gpt2|gpt-neo|opt|pythia)-gemmini-sdpa-256tok-alias-first-(on|off)-4core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/gemmini/sdpa/seq256/alias-first-${BASH_REMATCH[2]}"
  elif [[ "${workload}" =~ ^(opt|pythia)-(rocket|boom)-gemmini-(sdpa|flash|window)-([0-9]+)tok-[0-9]+core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/gemmini/${BASH_REMATCH[3]}/seq${BASH_REMATCH[4]}"
  elif [[ "${workload}" == "gemmini-max-autotune-gemmini-4core" ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-gemmini-max-autotune/gemmini"
  elif [[ "${workload}" == "gemmini-max-autotune-simple-gemmini-4core" ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-gemmini-max-autotune-simple/gemmini"
  else
    die "no artifact mapping for experiment workload ${workload}"
  fi
}

append_pending_artifact_once() {
  local artifact_dir="$1"
  local existing

  for existing in "${experiment_pending_artifact_dirs[@]}"; do
    [[ "${existing}" == "${artifact_dir}" ]] && return 0
  done
  experiment_pending_artifact_dirs+=("${artifact_dir}")
}

if [[ "${experiment_selected}" -eq 1 ]]; then
  completed_workloads=0
  for workload in "${experiment_workloads[@]}"; do
    if reused_workload="$(reusable_completed_workload "${workload}")"; then
      completed_workloads=$((completed_workloads + 1))
      if [[ "${reused_workload}" == "${workload}" ]]; then
        log "${experiment_name}: skipping completed workload ${workload}"
      else
        log "${experiment_name}: reusing ${reused_workload} for ${workload}"
      fi
    else
      experiment_pending_workloads+=("${workload}")
      append_pending_artifact_once "$(artifact_dir_for_workload "${workload}")"
      workload_args+=("--workload=${workload}")
    fi
  done

  log "${experiment_name}: ${completed_workloads} complete, ${#experiment_pending_workloads[@]} pending"
  if [[ "${#experiment_pending_workloads[@]}" -eq 0 ]]; then
    skip_elves=1
    skip_package=1
    skip_images=1
    skip_firesim=1
    log "${experiment_name}: every required workload is already complete"
  fi
fi

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
if [[ "${experiment_selected}" -eq 1 ]]; then
  for artifact_dir in "${experiment_pending_artifact_dirs[@]}"; do
    build_elves_args+=(--artifact-dir="${artifact_dir}")
  done
elif [[ "${alias_first_selected}" -eq 1 ]]; then
  for artifact_dir in "${alias_first_artifact_dirs[@]}"; do
    build_elves_args+=(--artifact-dir="${artifact_dir}")
  done
fi

# Validate selected bitstream and driver archives before expensive ELF/image work.
if [[ "${skip_firesim}" -eq 0 && "${#workload_args[@]}" -gt 0 ]]; then
  log "validating selected FireSim hardware artifacts"
  bash "${SCRIPT_DIR}/run-firesim-workloads.sh" \
    --preflight-only \
    "${workload_args[@]}"
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
  if [[ "${experiment_selected}" -eq 1 ]]; then
    for artifact_dir in "${experiment_pending_artifact_dirs[@]}"; do
      package_args+=(--artifact-dir="${artifact_dir}")
    done
  elif [[ "${alias_first_selected}" -eq 1 ]]; then
    for artifact_dir in "${alias_first_artifact_dirs[@]}"; do
      package_args+=(--artifact-dir="${artifact_dir}")
    done
  fi
  bash "${SCRIPT_DIR}/package-firemarshal-workload.sh" "${package_args[@]}"
  printf '[stage2][PASS] FireMarshal workload directory=%s\n' "${PYTORCH_CHIPYARD_WORKLOAD_DIR}"
fi

# A non-selective run can only discover its workload set after packaging.
if [[ "${skip_firesim}" -eq 0 && "${#workload_args[@]}" -eq 0 ]]; then
  firesim_preflight_args=(--preflight-only)
  if [[ "${resume_firesim}" -eq 1 ]]; then
    firesim_preflight_args+=(--resume)
  fi
  if [[ -n "${resume_from_arg}" ]]; then
    firesim_preflight_args+=(--resume-from="${resume_from_arg}")
  fi
  log "validating discovered FireSim hardware artifacts"
  bash "${SCRIPT_DIR}/run-firesim-workloads.sh" \
    "${firesim_preflight_args[@]}"
fi

if [[ "${skip_images}" -eq 0 ]]; then
  log "building and installing FireMarshal images"
  image_args=()
  if [[ "${rebuild_pending_images}" -eq 1 ]]; then
    image_args+=(--pending-only)
  fi
  if [[ "${experiment_selected}" -eq 1 ]]; then
    for workload in "${experiment_pending_workloads[@]}"; do
      image_args+=(--workload="${workload}")
    done
  elif [[ "${alias_first_selected}" -eq 1 ]]; then
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
if [[ -n "${experiment_name}" ]]; then
  printf 'STAGE2_EXPERIMENT=%s\n' "${experiment_name}"
fi
printf 'STAGE2_RESULTS_DIR=%s\n' "${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
printf 'STAGE2_STATUS=PASS\n'
