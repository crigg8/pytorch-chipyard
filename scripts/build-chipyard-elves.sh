#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/artifact-stage1-common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/build-chipyard-elves.sh [options]

Build model-<N>core.elf files from compiler artifacts on a local Chipyard host.
This script is intended to run outside the compiler-only Docker container.

Options:
  --artifact-root=PATH  Root to scan for artifact directories containing build.sh.
                        Default: $PYTORCH_CHIPYARD_ARTIFACT_ROOT or examples/.
  --artifact-dir=PATH   Build one artifact directory. May be passed multiple times.
  --chipyard-env=PATH   Local Chipyard env.sh. Overrides CHIPYARD_ENV_PATH.
  --backend=BACKEND     Override backend for every artifact: gemmini, rvv, or scalar.
                        Default: read .pytorch-chipyard-backend or infer from hint.
  --core=LIST           Override core-count list for every artifact.
                        Default: read .pytorch-chipyard-build-cores or backend default.
  -h, --help            Show this help.

Examples:
  CHIPYARD_ENV_PATH=/home/hongjun/hk_chipyard/chipyard/env.sh \
    bash scripts/build-chipyard-elves.sh --artifact-root=examples

  bash scripts/build-chipyard-elves.sh \
    --chipyard-env=/home/hongjun/hk_chipyard/chipyard/env.sh \
    --artifact-dir=examples/artifact-resnet50/gemmini --core=2,4
EOF
}

artifact_root_arg=""
artifact_dir_args=()
chipyard_env_arg=""
backend_arg="auto"
core_arg="auto"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-root)
      [[ "$#" -ge 2 ]] || pc_usage_error "--artifact-root requires a value"
      artifact_root_arg="$2"
      shift 2
      ;;
    --artifact-root=*)
      artifact_root_arg="${1#--artifact-root=}"
      shift
      ;;
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

pc_prepare_environment

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
  artifact_root="${artifact_root_arg:-${PYTORCH_CHIPYARD_ARTIFACT_ROOT:-${PC_REPO_ROOT}/examples}}"
  artifact_root="$(pc_abs_dir "${artifact_root}")"
  while IFS= read -r -d '' build_script; do
    artifact_dirs+=("$(dirname "${build_script}")")
  done < <(find "${artifact_root}" -type f -name build.sh -print0 | sort -z)
fi

[[ "${#artifact_dirs[@]}" -gt 0 ]] || pc_die "no artifact directories found"
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
