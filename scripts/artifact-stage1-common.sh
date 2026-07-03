#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "artifact-stage1-common.sh is a helper and must be sourced" >&2
  exit 1
fi

PC_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PC_REPO_ROOT="$(cd -- "${PC_SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

pc_log() {
  printf '[artifact-stage1] %s\n' "$*"
}

pc_die() {
  printf '[artifact-stage1][error] %s\n' "$*" >&2
  exit 1
}

pc_usage_error() {
  pc_die "$1; pass --help for usage"
}

pc_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

pc_lower() {
  printf '%s\n' "${1,,}"
}

pc_split_csv() {
  local raw="$1"
  local out_name="$2"
  shift 2
  local -n out_ref="${out_name}"
  local defaults=("$@")
  local part normalized

  out_ref=()
  raw="$(pc_trim "${raw:-default}")"
  if [[ -z "${raw}" || "$(pc_lower "${raw}")" == "default" ]]; then
    out_ref=("${defaults[@]}")
    return
  fi

  IFS=',' read -r -a parts <<< "${raw}"
  for part in "${parts[@]}"; do
    normalized="$(pc_lower "$(pc_trim "${part}")")"
    [[ -n "${normalized}" ]] || continue
    if [[ "${normalized}" == "default" ]]; then
      out_ref+=("${defaults[@]}")
    else
      out_ref+=("${normalized}")
    fi
  done

  [[ "${#out_ref[@]}" -gt 0 ]] || out_ref=("${defaults[@]}")
}

pc_append_unique() {
  local out_name="$1"
  local value="$2"
  local -n out_ref="${out_name}"
  local existing
  for existing in "${out_ref[@]}"; do
    [[ "${existing}" == "${value}" ]] && return
  done
  out_ref+=("${value}")
}

pc_normalize_backend() {
  case "$(pc_lower "$1")" in
    gemmini) printf '%s\n' gemmini ;;
    rvv | saturn) printf '%s\n' rvv ;;
    scalar | rocket) printf '%s\n' scalar ;;
    *) pc_die "unknown backend '$1'; expected gemmini, rvv, scalar, or default" ;;
  esac
}

pc_normalize_cnn_model() {
  case "$(pc_lower "$1")" in
    alexnet) printf '%s\n' alexnet ;;
    mobilenet | mobilenetv2 | mobilenet-v2 | mobile_net_v2) printf '%s\n' mobilenetv2 ;;
    resnet | resnet50 | resnet-50) printf '%s\n' resnet50 ;;
    squeezenet | squeezenet1_1 | squeezenet-1.1) printf '%s\n' squeezenet ;;
    *) pc_die "unknown CNN model '$1'; expected alexnet, mobilenetv2, resnet50, squeezenet, or default" ;;
  esac
}

pc_normalize_llm_model() {
  case "$(pc_lower "$1")" in
    gpt2 | gpt-2 | gpt2-124m) printf '%s\n' gpt2 ;;
    gpt-neo | gptneo | gpt-neo-125m) printf '%s\n' gpt-neo ;;
    opt | opt-125m) printf '%s\n' opt ;;
    pythia | pythia-160m) printf '%s\n' pythia ;;
    *) pc_die "unknown LLM model '$1'; expected gpt2, gpt-neo, opt, pythia, or default" ;;
  esac
}

pc_normalize_attention() {
  case "$(pc_lower "$1")" in
    sdpa | flash | window) printf '%s\n' "$(pc_lower "$1")" ;;
    *) pc_die "unknown attention mode '$1'; expected sdpa, flash, window, or default" ;;
  esac
}

pc_normalize_host() {
  case "$(pc_lower "$1")" in
    rocket | boom) printf '%s\n' "$(pc_lower "$1")" ;;
    *) pc_die "unknown host '$1'; expected rocket, boom, or default" ;;
  esac
}

pc_validate_seq_len() {
  local seq_len="$1"
  case "${seq_len}" in
    256 | 512 | 768 | 1024) ;;
    *) pc_die "invalid sequence length '${seq_len}'; expected 256, 512, 768, or 1024" ;;
  esac
}

pc_cnn_script() {
  case "$1" in
    alexnet) printf '%s\n' "${PC_REPO_ROOT}/examples/AlexNet.py" ;;
    mobilenetv2) printf '%s\n' "${PC_REPO_ROOT}/examples/MobileNetV2.py" ;;
    resnet50) printf '%s\n' "${PC_REPO_ROOT}/examples/ResNet50.py" ;;
    squeezenet) printf '%s\n' "${PC_REPO_ROOT}/examples/SqueezeNet.py" ;;
    *) pc_die "unknown CNN model '$1'" ;;
  esac
}

pc_llm_script() {
  case "$1" in
    gpt2) printf '%s\n' "${PC_REPO_ROOT}/examples/GPT2.py" ;;
    gpt-neo) printf '%s\n' "${PC_REPO_ROOT}/examples/GPT-Neo-125m.py" ;;
    opt) printf '%s\n' "${PC_REPO_ROOT}/examples/Opt-125m.py" ;;
    pythia) printf '%s\n' "${PC_REPO_ROOT}/examples/Pythia-160m.py" ;;
    *) pc_die "unknown LLM model '$1'" ;;
  esac
}

pc_conda_env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -qx "${env_name}"
}

pc_activate_conda_if_available() {
  local env_name="$1"
  if pc_conda_env_exists "${env_name}"; then
    conda activate "${env_name}"
  else
    pc_log "conda env '${env_name}' was not found; continuing with the current shell environment"
  fi
}

pc_prepare_environment() {
  local conda_env="${CONDA_ENV_NAME:-${CONDA_ENV:-llapi}}"

  if [[ "${PYTORCH_CHIPYARD_SKIP_CONDA:-0}" != "1" ]]; then
    if command -v conda >/dev/null 2>&1; then
      set +u
      source "$(conda info --base)/etc/profile.d/conda.sh"
      pc_activate_conda_if_available "${conda_env}"
      set -u
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
      set +u
      source "${HOME}/anaconda3/etc/profile.d/conda.sh"
      pc_activate_conda_if_available "${conda_env}"
      set -u
    else
      pc_log "conda was not found; continuing with the current shell environment"
    fi
  fi

  # This file defines WORKSPACE, CHIPYARD_ENV_PATH, LLVM_PROJECT_PATH, and the
  # default TorchInductor/Triton-Chipyard environment.
  set +u
  source "${PC_REPO_ROOT}/scripts/env.sh"
  set -u

  [[ -f "${CHIPYARD_ENV_PATH}" ]] || pc_die "missing CHIPYARD_ENV_PATH: ${CHIPYARD_ENV_PATH}"
  [[ -d "${LLVM_PROJECT_PATH}" ]] || pc_die "missing LLVM_PROJECT_PATH: ${LLVM_PROJECT_PATH}"
}

pc_common_env() {
  printf '%s\0' \
    "TRITON_ALWAYS_COMPILE=1" \
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1" \
    "TORCHINDUCTOR_MAX_AUTOTUNE=1" \
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON" \
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=TRITON" \
    "TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1" \
    "TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS=1" \
    "TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER=${TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER:-0}" \
    "TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=${TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE:-0}"
}

pc_backend_env_array() {
  local backend="$1"
  local out_name="$2"
  local -n out_ref="${out_name}"
  case "${backend}" in
    gemmini)
      out_ref=(
        TRITON_CHIPYARD_USE_GEMMINI=1
        TRITON_CHIPYARD_USE_RVV=0
        TRITON_CHIPYARD_GEMMINI_ADDR_LEN=32
        TRITON_CHIPYARD_GEMMINI_DIM=8
        TRITON_CHIPYARD_GEMMINI_BANK_ROWS=2048
        TRITON_CHIPYARD_GEMMINI_ACC_ROWS=2048
        TRITON_CHIPYARD_GEMMINI_ELEM_T=f32
        TRITON_CHIPYARD_GEMMINI_ACC_T=f32
        TRITON_CHIPYARD_RISCV_MARCH=rv64imafdc
        TRITON_CHIPYARD_RISCV_MABI=lp64d
      )
      ;;
    rvv)
      out_ref=(
        TRITON_CHIPYARD_USE_GEMMINI=0
        TRITON_CHIPYARD_USE_RVV=1
        TRITON_CHIPYARD_RISCV_MARCH=rv64imafdcv_zicsr_zifencei_zvl128b
        TRITON_CHIPYARD_RISCV_MABI=lp64d
        TRITON_CHIPYARD_RISCV_VARCH=vlen:128,elen:64
      )
      ;;
    scalar)
      out_ref=(
        TRITON_CHIPYARD_USE_GEMMINI=0
        TRITON_CHIPYARD_USE_RVV=0
        TRITON_CHIPYARD_RISCV_MARCH=rv64imafdc
        TRITON_CHIPYARD_RISCV_MABI=lp64d
      )
      ;;
    *) pc_die "unknown backend '${backend}'" ;;
  esac
}

pc_run_backend_env() {
  local backend="$1"
  shift
  local backend_env=()
  pc_backend_env_array "${backend}" backend_env

  if [[ "${backend}" == "gemmini" || "${backend}" == "scalar" ]]; then
    env \
      -u TRITON_CHIPYARD_RISCV_VARCH \
      "${backend_env[@]}" \
      "$@"
  else
    env "${backend_env[@]}" "$@"
  fi
}

pc_cores_for_backend() {
  case "$1" in
    gemmini) printf '%s\n' 2 4 ;;
    rvv) printf '%s\n' 2 4 ;;
    scalar) printf '%s\n' 4 8 16 ;;
    *) pc_die "unknown backend '$1'" ;;
  esac
}

pc_core_for_host() {
  case "$1" in
    rocket) printf '%s\n' 4 ;;
    boom) printf '%s\n' 1 ;;
    *) pc_die "unknown host '$1'" ;;
  esac
}

pc_combo_artifact_dir() {
  local artifact_root="$1"
  local combo_count="$2"
  local default_dir="$3"
  local suffix="$4"

  if [[ -z "${artifact_root}" ]]; then
    printf '%s\n' "${default_dir}"
  elif [[ "${combo_count}" -eq 1 ]]; then
    printf '%s\n' "${artifact_root}"
  else
    printf '%s\n' "${artifact_root%/}/${suffix}"
  fi
}

pc_artifact_storage_suffix() {
  local logical_suffix="$1"
  local first rest

  logical_suffix="${logical_suffix#/}"
  first="${logical_suffix%%/*}"
  if [[ "${logical_suffix}" == "${first}" ]]; then
    printf 'artifact-%s\n' "${first}"
  else
    rest="${logical_suffix#*/}"
    printf 'artifact-%s/%s\n' "${first}" "${rest}"
  fi
}

pc_require_file() {
  local path="$1"
  [[ -f "${path}" ]] || pc_die "required file not found: ${path}"
}

pc_require_artifacts() {
  local artifact_dir="$1"
  pc_require_file "${artifact_dir}/build.sh"
  pc_require_file "${artifact_dir}/model_spec.json"
  pc_require_file "${artifact_dir}/input.bin"
  pc_require_file "${artifact_dir}/weights.bin"
}

pc_write_artifact_workload_hint() {
  local artifact_dir="$1"
  local logical_rel="$2"

  mkdir -p "${artifact_dir}"
  printf '%s\n' "${logical_rel}" > "${artifact_dir}/.pytorch-chipyard-workload-rel"
}

pc_compile_fingerprint() {
  local backend="$1"
  local cache_key="$2"
  local script_path="$3"
  shift 3

  printf 'backend=%s\n' "${backend}"
  printf 'cache_key=%s\n' "${cache_key}"
  printf 'script_path=%s\n' "${script_path}"
  printf 'LLM_TOKEN_LENGTH=%s\n' "${LLM_TOKEN_LENGTH:-}"
  printf 'TORCHINDUCTOR_IM2COL_MM=%s\n' "${TORCHINDUCTOR_IM2COL_MM:-}"
  printf 'TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=%s\n' "${TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE:-0}"
  local arg
  for arg in "$@"; do
    printf 'arg=%q\n' "${arg}"
  done
}

pc_compile_stamp_path() {
  local artifact_dir="$1"
  printf '%s\n' "${artifact_dir}/.pytorch-chipyard-compile-fingerprint"
}

pc_artifacts_match_fingerprint() {
  local artifact_dir="$1"
  local fingerprint="$2"
  local stamp

  stamp="$(pc_compile_stamp_path "${artifact_dir}")"
  [[ -f "${stamp}" ]] || return 1
  pc_require_artifacts "${artifact_dir}"
  [[ "$(cat "${stamp}")" == "${fingerprint}" ]]
}

pc_build_core_elf() {
  local backend="$1"
  local artifact_dir="$2"
  local core="$3"

  pc_require_file "${artifact_dir}/build.sh"
  pc_log "building ${artifact_dir}/model-${core}core.elf"
  (
    cd "${artifact_dir}"
    pc_run_backend_env "${backend}" CHIPYARD_OMP_NUM_THREADS="${core}" bash ./build.sh
    pc_require_file "${artifact_dir}/model.elf"
    cp -f model.elf "model-${core}core.elf"
  )
}

pc_build_stamp_path() {
  local artifact_dir="$1"
  local core="$2"
  printf '%s\n' "${artifact_dir}/.model-${core}core.elf-fingerprint"
}

pc_build_fingerprint() {
  local backend="$1"
  local artifact_dir="$2"
  local core="$3"
  local compile_stamp

  compile_stamp="$(pc_compile_stamp_path "${artifact_dir}")"
  printf 'backend=%s\n' "${backend}"
  printf 'core=%s\n' "${core}"
  if [[ -f "${compile_stamp}" ]]; then
    cat "${compile_stamp}"
  else
    printf 'compile_stamp=\n'
  fi
}

pc_core_elf_matches_fingerprint() {
  local artifact_dir="$1"
  local core="$2"
  local fingerprint="$3"
  local elf stamp

  elf="${artifact_dir}/model-${core}core.elf"
  stamp="$(pc_build_stamp_path "${artifact_dir}" "${core}")"
  [[ -f "${elf}" && -f "${stamp}" ]] || return 1
  [[ "$(cat "${stamp}")" == "${fingerprint}" ]]
}

pc_build_core_elf_once() {
  local backend="$1"
  local artifact_dir="$2"
  local core="$3"
  local fingerprint stamp

  fingerprint="$(pc_build_fingerprint "${backend}" "${artifact_dir}" "${core}")"
  if pc_core_elf_matches_fingerprint "${artifact_dir}" "${core}" "${fingerprint}"; then
    pc_log "reusing ${artifact_dir}/model-${core}core.elf"
    return
  fi

  pc_build_core_elf "${backend}" "${artifact_dir}" "${core}"
  stamp="$(pc_build_stamp_path "${artifact_dir}" "${core}")"
  printf '%s\n' "${fingerprint}" > "${stamp}"
}

pc_run_compile() {
  local backend="$1"
  local artifact_dir="$2"
  local cache_key="$3"
  local script_path="$4"
  shift 4

  pc_require_file "${script_path}"
  mkdir -p "${artifact_dir}"
  pc_log "compiling ${script_path} -> ${artifact_dir}"

  local common_env=()
  while IFS= read -r -d '' item; do
    common_env+=("${item}")
  done < <(pc_common_env)

  pc_run_backend_env "${backend}" \
    "${common_env[@]}" \
    "PYTORCH_CHIPYARD_DUMP_PATH=${artifact_dir}" \
    "TRITON_CHIPYARD_DUMP_PATH=${artifact_dir}" \
    "TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/${cache_key}" \
    python "${script_path}" --compile "$@"

  pc_require_artifacts "${artifact_dir}"
}

pc_run_compile_once() {
  local backend="$1"
  local artifact_dir="$2"
  local cache_key="$3"
  local script_path="$4"
  shift 4

  local fingerprint stamp
  fingerprint="$(pc_compile_fingerprint "${backend}" "${cache_key}" "${script_path}" "$@")"
  if pc_artifacts_match_fingerprint "${artifact_dir}" "${fingerprint}"; then
    pc_log "reusing ${script_path} -> ${artifact_dir}"
    return
  fi

  pc_run_compile "${backend}" "${artifact_dir}" "${cache_key}" "${script_path}" "$@"
  stamp="$(pc_compile_stamp_path "${artifact_dir}")"
  printf '%s\n' "${fingerprint}" > "${stamp}"
}
