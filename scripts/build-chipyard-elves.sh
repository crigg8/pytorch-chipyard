#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PC_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

pc_log() {
  printf '[artifact-stage2] %s\n' "$*"
}

pc_die() {
  printf '[artifact-stage2][error] %s\n' "$*" >&2
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

pc_normalize_backend() {
  case "$(pc_lower "$1")" in
    gemmini) printf '%s\n' gemmini ;;
    rvv | saturn) printf '%s\n' rvv ;;
    scalar | rocket) printf '%s\n' scalar ;;
    *) pc_die "unknown backend '$1'; expected gemmini, rvv, scalar, or default" ;;
  esac
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

pc_compile_stamp_path() {
  local artifact_dir="$1"
  printf '%s\n' "${artifact_dir}/.pytorch-chipyard-compile-fingerprint"
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

pc_prepare_stage2_environment() {
  set +u
  source "${SCRIPT_DIR}/env.sh"
  set -u

  [[ -d "${LLVM_PROJECT_PATH}" ]] || pc_die "missing LLVM_PROJECT_PATH: ${LLVM_PROJECT_PATH}"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/build-chipyard-elves.sh [options]

Build model-<N>core.elf files from compiler artifacts on a local Chipyard host.
This script is intended to run outside the compiler-only Docker container.

Options:
  --artifact-dir=PATH   Build one artifact directory. May be passed multiple times.
  --chipyard-env=PATH   Local Chipyard env.sh. Overrides CHIPYARD_ENV_PATH.
  --backend=BACKEND     Override backend for every artifact: gemmini, rvv, or scalar.
                        Default: read .pytorch-chipyard-backend or infer from hint.
  --core=LIST           Override core-count list for every artifact.
                        Default: read .pytorch-chipyard-build-cores or backend default.
  -h, --help            Show this help.

Examples:
  CHIPYARD_ENV_PATH=/home/hongjun/hk_chipyard/chipyard/env.sh \
    bash scripts/build-chipyard-elves.sh

  bash scripts/build-chipyard-elves.sh \
    --chipyard-env=/home/hongjun/hk_chipyard/chipyard/env.sh \
    --artifact-dir=examples/artifact-resnet50/gemmini --core=2,4
EOF
}

artifact_dir_args=()
chipyard_env_arg=""
backend_arg="auto"
core_arg="auto"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      [[ "$#" -ge 2 ]] || pc_usage_error "--artifact-dir requires a value"
      artifact_dir_args+=("$2")
      shift 2
      ;;
    --artifact-dir=*)
      artifact_dir_args+=("${1#--artifact-dir=}")
      shift
      ;;
    --chipyard-env)
      [[ "$#" -ge 2 ]] || pc_usage_error "--chipyard-env requires a value"
      chipyard_env_arg="$2"
      shift 2
      ;;
    --chipyard-env=*)
      chipyard_env_arg="${1#--chipyard-env=}"
      shift
      ;;
    --backend)
      [[ "$#" -ge 2 ]] || pc_usage_error "--backend requires a value"
      backend_arg="$2"
      shift 2
      ;;
    --backend=*)
      backend_arg="${1#--backend=}"
      shift
      ;;
    --core | --cores)
      [[ "$#" -ge 2 ]] || pc_usage_error "$1 requires a value"
      core_arg="$2"
      shift 2
      ;;
    --core=* | --cores=*)
      core_arg="${1#*=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      pc_usage_error "unknown argument '$1'"
      ;;
  esac
done

if [[ -n "${chipyard_env_arg}" ]]; then
  export CHIPYARD_ENV_PATH="${chipyard_env_arg}"
fi

pc_prepare_stage2_environment

pc_abs_dir() {
  local path="$1"
  [[ -d "${path}" ]] || pc_die "directory not found: ${path}"
  (cd "${path}" && pwd -P)
}

pc_split_core_list() {
  local raw="$1"
  local out_name="$2"
  local -n out_ref="${out_name}"
  local part core existing

  out_ref=()
  IFS=',' read -r -a parts <<< "${raw}"
  for part in "${parts[@]}"; do
    core="$(pc_trim "${part}")"
    [[ -n "${core}" ]] || continue
    [[ "${core}" =~ ^[0-9]+$ && "${core}" != "0" ]] || pc_die "invalid core count '${core}'"
    for existing in "${out_ref[@]}"; do
      [[ "${existing}" == "${core}" ]] && continue 2
    done
    out_ref+=("${core}")
  done
  [[ "${#out_ref[@]}" -gt 0 ]] || pc_die "empty core list"
}

pc_backend_from_workload_rel() {
  local rel="$1"
  local part candidate

  IFS='/' read -r -a parts <<< "${rel}"
  for part in "${parts[@]}"; do
    candidate="${part%-im2col}"
    case "${candidate}" in
      gemmini | rvv | scalar)
        printf '%s\n' "${candidate}"
        return
        ;;
    esac
  done

  return 1
}

pc_infer_artifact_backend() {
  local artifact_dir="$1"
  local hint_file="${artifact_dir}/.pytorch-chipyard-backend"
  local rel_file="${artifact_dir}/.pytorch-chipyard-workload-rel"
  local backend rel

  if [[ "$(pc_lower "${backend_arg}")" != "auto" ]]; then
    pc_normalize_backend "${backend_arg}"
    return
  fi

  if [[ -f "${hint_file}" ]]; then
    backend="$(pc_trim "$(head -n 1 "${hint_file}")")"
    if [[ -n "${backend}" ]]; then
      pc_normalize_backend "${backend}"
      return
    fi
  fi

  if [[ -f "${rel_file}" ]]; then
    rel="$(pc_trim "$(head -n 1 "${rel_file}")")"
    if backend="$(pc_backend_from_workload_rel "${rel}")"; then
      pc_normalize_backend "${backend}"
      return
    fi
  fi

  pc_die "could not infer backend for ${artifact_dir}; pass --backend or rerun artifact generation"
}

pc_read_artifact_cores() {
  local artifact_dir="$1"
  local backend="$2"
  local out_name="$3"
  local -n out_ref="${out_name}"
  local core_file="${artifact_dir}/.pytorch-chipyard-build-cores"
  local core existing parsed_cores=()

  out_ref=()
  if [[ "$(pc_lower "${core_arg}")" != "auto" ]]; then
    pc_split_core_list "${core_arg}" parsed_cores
    out_ref=("${parsed_cores[@]}")
    return
  fi

  if [[ -f "${core_file}" ]]; then
    while IFS= read -r core; do
      core="$(pc_trim "${core}")"
      [[ -n "${core}" ]] || continue
      [[ "${core}" =~ ^[0-9]+$ && "${core}" != "0" ]] || pc_die "invalid core count '${core}' in ${core_file}"
      for existing in "${out_ref[@]}"; do
        [[ "${existing}" == "${core}" ]] && continue 2
      done
      out_ref+=("${core}")
    done < "${core_file}"
  fi

  if [[ "${#out_ref[@]}" -eq 0 ]]; then
    while IFS= read -r core; do
      for existing in "${out_ref[@]}"; do
        [[ "${existing}" == "${core}" ]] && continue 2
      done
      out_ref+=("${core}")
    done < <(pc_cores_for_backend "${backend}")
  fi
}

artifact_dirs=()
if [[ "${#artifact_dir_args[@]}" -gt 0 ]]; then
  for artifact_dir in "${artifact_dir_args[@]}"; do
    artifact_dirs+=("$(pc_abs_dir "${artifact_dir}")")
  done
else
  artifact_root="${PC_REPO_ROOT}/examples"
  artifact_root="$(pc_abs_dir "${artifact_root}")"
  while IFS= read -r -d '' build_script; do
    artifact_dirs+=("$(dirname "${build_script}")")
  done < <(find "${artifact_root}" -type f -name build.sh -print0 | sort -z)
fi

[[ "${#artifact_dirs[@]}" -gt 0 ]] || pc_die "no artifact directories found"
if [[ -z "${CHIPYARD_ENV_PATH:-}" ]]; then
  pc_die "CHIPYARD_ENV_PATH is not set; set CHIPYARD_DIR to your Chipyard checkout and source scripts/env.sh, or pass --chipyard-env=<chipyard-env.sh>"
fi
pc_require_file "${CHIPYARD_ENV_PATH}"

pc_log "Chipyard env: ${CHIPYARD_ENV_PATH}"
pc_log "artifact directories: ${#artifact_dirs[@]}"

for artifact_dir in "${artifact_dirs[@]}"; do
  backend="$(pc_infer_artifact_backend "${artifact_dir}")"
  cores=()
  pc_read_artifact_cores "${artifact_dir}" "${backend}" cores

  pc_require_artifacts "${artifact_dir}"
  pc_log "${artifact_dir}: backend=${backend}, cores=${cores[*]}"
  for core in "${cores[@]}"; do
    pc_build_core_elf_once "${backend}" "${artifact_dir}" "${core}"
  done
done

pc_log "done"
