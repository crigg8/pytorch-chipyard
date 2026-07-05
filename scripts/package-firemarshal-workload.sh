#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

log() {
  printf '[artifact-package] %s\n' "$*"
}

warn() {
  printf '[artifact-package][warn] %s\n' "$*" >&2
}

die() {
  printf '[artifact-package][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/package-firemarshal-workload.sh [--no-clean]

Default:
  Scan artifact-* directories under examples/ for generated Stage 1 artifacts
  and package every model-<N>core.elf into FireMarshal and FireSim workload
  files.

Generated files:
  $PYTORCH_CHIPYARD_WORKLOAD_DIR/<workload>.json
  $PYTORCH_CHIPYARD_WORKLOAD_DIR/overlay-<workload>/
  $FIRESIM_WORKLOAD_DIR/<workload>.json

Options:
  --no-clean            Keep previously generated workload JSONs/overlays.
  --no-chipyard-check   Allow file generation without initialized FireMarshal/FireSim.
  -h, --help            Show this help.

Environment:
  PYTORCH_CHIPYARD_FIREMARSHAL_ROOTFS_SIZE  FireMarshal rootfs-size. Default: 8GiB.
  PYTORCH_CHIPYARD_WORKLOAD_DIR             FireMarshal workload output directory.
  FIRESIM_WORKLOAD_DIR                      FireSim deploy workload output directory.
EOF
}

abs_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "directory not found: ${path}"
  (cd -- "${path}" >/dev/null 2>&1 && pwd -P)
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

validate_name() {
  local label="$1"
  local value="$2"
  [[ "${value}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "invalid ${label} '${value}'; use only letters, numbers, '.', '_', and '-'"
}

validate_size() {
  local value="$1"
  [[ "${value}" =~ ^[A-Za-z0-9._+-]+$ ]] || die "invalid rootfs size: ${value}"
}

shell_quote() {
  printf '%q' "$1"
}

artifact_storage_rel() {
  local logical_rel="$1"
  local first rest

  logical_rel="${logical_rel#/}"
  first="${logical_rel%%/*}"
  if [[ "${logical_rel}" == "${first}" ]]; then
    printf 'artifact-%s\n' "${first}"
  else
    rest="${logical_rel#*/}"
    printf 'artifact-%s/%s\n' "${first}" "${rest}"
  fi
}

normalize_artifact_rel() {
  local rel="$1"
  local first rest

  rel="${rel#./}"
  first="${rel%%/*}"
  if [[ "${first}" == artifact-* ]]; then
    first="${first#artifact-}"
  fi

  if [[ "${rel}" == */* ]]; then
    rest="${rel#*/}"
    printf '%s/%s\n' "${first}" "${rest}"
  else
    printf '%s\n' "${first}"
  fi
}

existing_artifact_dir_for_rel() {
  local logical_rel="$1"
  local storage_dir legacy_dir

  storage_dir="${ARTIFACT_ROOT}/$(artifact_storage_rel "${logical_rel}")"
  if [[ -d "${storage_dir}" ]]; then
    printf '%s\n' "${storage_dir}"
    return 0
  fi

  legacy_dir="${ARTIFACT_ROOT}/${logical_rel}"
  if [[ -d "${legacy_dir}" ]]; then
    printf '%s\n' "${legacy_dir}"
    return 0
  fi

  return 1
}

infer_core_from_elf() {
  local elf_name="$1"
  if [[ "${elf_name}" =~ ([0-9]+)core\.elf$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

select_input_bin() {
  local artifact_dir="$1"
  local matches=()
  local candidate

  if [[ -f "${artifact_dir}/input.bin" ]]; then
    printf '%s\n' "${artifact_dir}/input.bin"
    return 0
  fi

  for candidate in "${artifact_dir}"/*input*.bin; do
    [[ -f "${candidate}" ]] || continue
    matches+=("${candidate}")
  done

  case "${#matches[@]}" in
    0)
      return 1
      ;;
    1)
      printf '%s\n' "${matches[0]}"
      return 0
      ;;
    *)
      printf '[artifact-package][error] multiple input bin candidates under %s:\n' "${artifact_dir}" >&2
      printf '  %s\n' "${matches[@]}" >&2
      return 2
      ;;
  esac
}

workload_kind_for() {
  local artifact_dir="$1"
  local rel first_component

  rel="$(artifact_rel_path "${artifact_dir}")"
  first_component="${rel%%/*}"

  case "${first_component}" in
    gpt2 | gpt-neo | opt | pythia)
      printf '%s\n' llm
      ;;
    *)
      printf '%s\n' cnn
      ;;
  esac
}

llm_seq_len_for() {
  local text="$1"

  if [[ "${text}" =~ (^|[/._-])seq([0-9]+)([/._-]|$) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
  elif [[ "${text}" =~ (^|[/._-])([0-9]+)tok([/._-]|$) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
  elif [[ "${text}" =~ (^|[/._-])(256|512|768|1024)([/._-]|$) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
  else
    printf '%s\n' 256
  fi
}

guest_root_for() {
  local kind="$1"
  case "${kind}" in
    llm) printf '%s\n' /root/llm ;;
    *) printf '%s\n' /root/cnn ;;
  esac
}

artifact_rel_path() {
  local artifact_dir="$1"
  local rel hint_path

  hint_path="${artifact_dir}/.pytorch-chipyard-workload-rel"
  if [[ -f "${hint_path}" ]]; then
    rel="$(head -n 1 "${hint_path}")"
    [[ "${rel}" =~ ^[A-Za-z0-9._/-]+$ ]] || die "invalid workload hint in ${hint_path}: ${rel}"
    normalize_artifact_rel "${rel}"
    return
  fi

  case "${artifact_dir}" in
    "${ARTIFACT_ROOT}/"*)
      rel="${artifact_dir#"${ARTIFACT_ROOT}/"}"
      ;;
    "${WORKSPACE}/examples/"*)
      rel="${artifact_dir#"${WORKSPACE}/examples/"}"
      ;;
    *)
      rel="$(basename "${artifact_dir}")"
      ;;
  esac

  normalize_artifact_rel "${rel}"
}

derive_workload_name() {
  local artifact_dir="$1"
  local core="$2"
  local rel

  rel="$(artifact_rel_path "${artifact_dir}")"

  rel="${rel//\//-}"
  rel="${rel//_/-}"
  rel="$(printf '%s\n' "${rel}" | sed -E 's/(^|-)seq([0-9]+)(-|$)/\1\2tok\3/g')"
  rel="${rel//--/-}"

  if [[ "${rel}" =~ [0-9]+core$ ]]; then
    printf '%s\n' "${rel}"
  else
    printf '%s-%score\n' "${rel}" "${core}"
  fi
}

derive_packaged_workload_name() {
  local artifact_dir="$1"
  local core="$2"
  local rel model attention seq_len

  rel="$(artifact_rel_path "${artifact_dir}")"
  if [[ "${rel}" =~ ^(opt|pythia)/gemmini/(sdpa|flash|window)/seq([0-9]+)$ ]]; then
    model="${BASH_REMATCH[1]}"
    attention="${BASH_REMATCH[2]}"
    seq_len="${BASH_REMATCH[3]}"
    if [[ "${core}" == "1" ]]; then
      printf '%s-boom-gemmini-%s-%stok-%score\n' "${model}" "${attention}" "${seq_len}" "${core}"
    else
      printf '%s-rocket-gemmini-%s-%stok-%score\n' "${model}" "${attention}" "${seq_len}" "${core}"
    fi
    return
  fi

  derive_workload_name "${artifact_dir}" "${core}"
}

is_shadowed_elf() {
  local artifact_dir="$1"
  local core="$2"
  local rel model host attention seq_len canonical_dir

  rel="$(artifact_rel_path "${artifact_dir}")"
  if [[ "${rel}" =~ ^(opt|pythia)/gemmini$ ]]; then
    model="${BASH_REMATCH[1]}"
    if canonical_dir="$(existing_artifact_dir_for_rel "${model}/gemmini/sdpa/seq256")"; then
      warn "skipping shadowed SDPA artifact ${artifact_dir}/model-${core}core.elf; using ${canonical_dir}"
      return 0
    fi
    return
  fi

  if [[ "${rel}" =~ ^(opt|pythia)/(rocket|boom)-gemmini/(sdpa|flash|window)/seq([0-9]+)$ ]]; then
    model="${BASH_REMATCH[1]}"
    host="${BASH_REMATCH[2]}"
    attention="${BASH_REMATCH[3]}"
    seq_len="${BASH_REMATCH[4]}"
    if canonical_dir="$(existing_artifact_dir_for_rel "${model}/gemmini/${attention}/seq${seq_len}")"; then
      warn "skipping shadowed ${host}-gemmini artifact ${artifact_dir}/model-${core}core.elf; using ${canonical_dir}"
      return 0
    fi
  fi

  return 1
}

clean_generated_workloads() {
  [[ -d "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" ]] || return 0

  find "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" \
    -maxdepth 1 \
    -type f \
    -name '*.json' \
    ! -name '*base.json' \
    -delete
  find "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" \
    -maxdepth 1 \
    -type d \
    -name 'overlay-*' \
    -exec rm -rf {} +
}

discover_artifact_elf_paths() {
  local root="$1"
  local artifact_dirs=()
  local dir

  if [[ "$(basename "${root}")" == artifact-* ]]; then
    find "${root}" -type f -name 'model-*core.elf' -print0 | sort -z
    return
  fi

  while IFS= read -r -d '' dir; do
    artifact_dirs+=("${dir}")
  done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'artifact-*' -print0 | sort -z)

  if [[ "${#artifact_dirs[@]}" -gt 0 ]]; then
    find "${artifact_dirs[@]}" -type f -name 'model-*core.elf' -print0 | sort -z
  else
    find "${root}" -type f -name 'model-*core.elf' -print0 | sort -z
  fi
}

omp_places_for() {
  local core="$1"
  local places=""
  local i

  for ((i = 0; i < core; i++)); do
    places+="${places:+,}{${i}}"
  done
  printf '%s\n' "${places}"
}

omp_cpu_affinity_for() {
  local core="$1"
  local affinity=""
  local i

  for ((i = 0; i < core; i++)); do
    affinity+="${affinity:+ }${i}"
  done
  printf '%s\n' "${affinity}"
}

write_workload() {
  local artifact_dir="$1"
  local elf_path="$2"
  local rootfs_size="$3"
  local workload_name="$4"

  local elf_base core kind guest_root guest_root_rel
  local input_path input_base weights_path
  local workload_dir deploy_workload_dir overlay_dir guest_dir runner_path hook_path
  local workload_json deploy_json places cpu_affinity model_command
  local seq_len expect_output_bin runner_check_files output_entries

  elf_base="$(basename "${elf_path}")"
  core="$(infer_core_from_elf "${elf_base}")"
  [[ -n "${core}" ]] || die "could not infer core count from ${elf_path}"
  [[ "${core}" =~ ^[0-9]+$ && "${core}" != "0" ]] || die "invalid core count in ${elf_path}"

  input_path="$(select_input_bin "${artifact_dir}")" || \
    die "could not find input.bin or a unique *input*.bin under ${artifact_dir}"
  input_base="$(basename "${input_path}")"

  weights_path="${artifact_dir}/weights.bin"
  require_file "${weights_path}"

  kind="$(workload_kind_for "${artifact_dir}")"
  guest_root="$(guest_root_for "${kind}")"
  guest_root_rel="${guest_root#/}"
  validate_name "workload name" "${workload_name}"

  if [[ -n "${PACKAGED_WORKLOADS[${workload_name}]:-}" ]]; then
    die "duplicate workload name '${workload_name}' from ${artifact_dir} and ${PACKAGED_WORKLOADS[${workload_name}]}"
  fi
  PACKAGED_WORKLOADS["${workload_name}"]="${artifact_dir}"

  workload_dir="${PYTORCH_CHIPYARD_WORKLOAD_DIR}"
  deploy_workload_dir="${FIRESIM_WORKLOAD_DIR}"
  overlay_dir="${workload_dir}/overlay-${workload_name}"
  guest_dir="${overlay_dir}/${guest_root_rel}/${workload_name}"
  runner_path="${overlay_dir}/${guest_root_rel}/run_${workload_name}.sh"
  hook_path="${overlay_dir}/firemarshal.sh"
  workload_json="${workload_dir}/${workload_name}.json"
  deploy_json="${deploy_workload_dir}/${workload_name}.json"

  mkdir -p "${guest_dir}" "${deploy_workload_dir}" "$(dirname "${runner_path}")"

  cp -f "${elf_path}" "${guest_dir}/${elf_base}"
  cp -f "${input_path}" "${guest_dir}/${input_base}"
  cp -f "${weights_path}" "${guest_dir}/weights.bin"
  chmod +x "${guest_dir}/${elf_base}" 2>/dev/null || true

  places="$(omp_places_for "${core}")"
  cpu_affinity="$(omp_cpu_affinity_for "${core}")"
  model_command="$(shell_quote "./${elf_base}") $(shell_quote "${input_base}") weights.bin output.bin"
  expect_output_bin=1
  if [[ "${kind}" == "llm" ]]; then
    seq_len="$(llm_seq_len_for "${artifact_dir}")"
    if [[ "${seq_len}" -gt 256 ]]; then
      model_command+=" --no-output"
      expect_output_bin=0
    fi
  fi

  if [[ "${expect_output_bin}" -eq 1 ]]; then
    runner_check_files="output.bin run.log model.log autotune.log"
    output_entries="    \"${guest_root}/${workload_name}/output.bin\",
    \"${guest_root}/${workload_name}/run.log\",
    \"${guest_root}/${workload_name}/model.log\",
    \"${guest_root}/${workload_name}/autotune.log\""
  else
    runner_check_files="run.log model.log autotune.log"
    output_entries="    \"${guest_root}/${workload_name}/run.log\",
    \"${guest_root}/${workload_name}/model.log\",
    \"${guest_root}/${workload_name}/autotune.log\""
  fi

  cat >"${runner_path}" <<EOF
#!/bin/bash
set -u
set -o pipefail

ulimit -s unlimited
cd ${guest_root}/${workload_name}

if [ -b /dev/iceblk ]; then
  mount -o remount,rw,noatime / 2>/dev/null || \\
    mount -o remount,rw,noatime /dev/iceblk / 2>/dev/null || true
else
  mount -o remount,rw,noatime / 2>/dev/null || true
fi

export OMP_NUM_THREADS=${core}
export OMP_THREAD_LIMIT=${core}
export OMP_STACKSIZE=64M
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=false
export OMP_PROC_BIND=close
export OMP_PLACES="${places}"
export OMP_SCHEDULE=static
export OMP_WAIT_POLICY=PASSIVE
export OMP_DISPLAY_ENV=FALSE
export OMP_DISPLAY_AFFINITY=FALSE
export GOMP_CPU_AFFINITY="${cpu_affinity}"
export GOMP_SPINCOUNT=0
export MALLOC_ARENA_MAX=1

chmod +x ./${elf_base} 2>/dev/null || true
rm -f output.bin run.log model.log autotune.log

exec >./run.log 2>&1

echo "[runner] start ${workload_name}"
echo "[runner] OMP_NUM_THREADS=\${OMP_NUM_THREADS} OMP_THREAD_LIMIT=\${OMP_THREAD_LIMIT} OMP_STACKSIZE=\${OMP_STACKSIZE} OMP_DYNAMIC=\${OMP_DYNAMIC} OMP_MAX_ACTIVE_LEVELS=\${OMP_MAX_ACTIVE_LEVELS} OMP_NESTED=\${OMP_NESTED} OMP_PROC_BIND=\${OMP_PROC_BIND} OMP_PLACES=\${OMP_PLACES} OMP_SCHEDULE=\${OMP_SCHEDULE} OMP_WAIT_POLICY=\${OMP_WAIT_POLICY} GOMP_CPU_AFFINITY=\${GOMP_CPU_AFFINITY} GOMP_SPINCOUNT=\${GOMP_SPINCOUNT} MALLOC_ARENA_MAX=\${MALLOC_ARENA_MAX}"
${model_command}
rc=\$?
echo "[runner] model_ret=\${rc}"
for log_file in ${runner_check_files}; do
  if [ -f "\${log_file}" ]; then
    echo "[runner] captured \${log_file}"
  else
    echo "[runner] missing \${log_file}"
  fi
done
sync || true
sleep 5
exit "\${rc}"
EOF
  chmod +x "${runner_path}"

  cat >"${hook_path}" <<EOF
#!/bin/sh
bash ${guest_root}/run_${workload_name}.sh
EOF
  chmod +x "${hook_path}"

  cat >"${workload_json}" <<EOF
{
  "name": "${workload_name}",
  "workdir": ".",
  "base": "br-base.json",
  "rootfs-size": "${rootfs_size}",
  "overlay": "overlay-${workload_name}",
  "command": "bash ${guest_root}/run_${workload_name}.sh",
  "outputs": [
${output_entries}
  ]
}
EOF

  cat >"${deploy_json}" <<EOF
{
  "benchmark_name": "${workload_name}",
  "common_simulation_outputs": [
    "uartlog"
  ],
  "common_bootbinary": "../../../../../software/firemarshal/images/firechip/${workload_name}/${workload_name}-bin",
  "common_rootfs": "../../../../../software/firemarshal/images/firechip/${workload_name}/${workload_name}.img",
  "common_outputs": [
${output_entries}
  ]
}
EOF

  log "packaged ${workload_name}"
}

artifact_root="${WORKSPACE}/examples"
check_chipyard=1
clean_workloads=1
rootfs_size="${PYTORCH_CHIPYARD_FIREMARSHAL_ROOTFS_SIZE:-8GiB}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --no-clean)
      clean_workloads=0
      shift
      ;;
    --no-chipyard-check)
      check_chipyard=0
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

artifact_root="$(abs_dir "${artifact_root}")"
ARTIFACT_ROOT="${artifact_root}"
validate_size "${rootfs_size}"

if [[ "${check_chipyard}" -eq 1 ]]; then
  [[ -f "${FIREMARSHAL_DIR}/marshal" ]] || \
    die "FireMarshal is not initialized at ${FIREMARSHAL_DIR}; check CHIPYARD_DIR or FIREMARSHAL_DIR"
  [[ -d "${FIRESIM_DEPLOY_DIR}" ]] || \
    die "FireSim deploy directory not found: ${FIRESIM_DEPLOY_DIR}; check CHIPYARD_DIR or FIRESIM_DEPLOY_DIR"
fi

mkdir -p "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" "${FIRESIM_WORKLOAD_DIR}"
if [[ "${clean_workloads}" -eq 1 ]]; then
  log "cleaning generated FireMarshal workload JSONs/overlays"
  clean_generated_workloads
fi

shopt -s nullglob

elf_paths=()
while IFS= read -r -d '' elf_path; do
  elf_paths+=("${elf_path}")
done < <(discover_artifact_elf_paths "${artifact_root}")

if [[ "${#elf_paths[@]}" -eq 0 ]]; then
  warn "no model-*core.elf files found under ${artifact_root}"
  exit 0
fi

declare -A PACKAGED_WORKLOADS=()
packaged_count=0

log "artifact root: ${artifact_root}"
log "FireMarshal workload dir: ${PYTORCH_CHIPYARD_WORKLOAD_DIR}"
log "FireSim workload dir: ${FIRESIM_WORKLOAD_DIR}"

for elf_path in "${elf_paths[@]}"; do
  artifact_dir="$(dirname "${elf_path}")"
  elf_base="$(basename "${elf_path}")"
  core="$(infer_core_from_elf "${elf_base}")"
  [[ -n "${core}" ]] || die "could not infer core count from ${elf_path}"
  if is_shadowed_elf "${artifact_dir}" "${core}"; then
    continue
  fi
  workload_name="$(derive_packaged_workload_name "${artifact_dir}" "${core}")"
  write_workload "${artifact_dir}" "${elf_path}" "${rootfs_size}" "${workload_name}"
  packaged_count=$((packaged_count + 1))
done

log "done: packaged ${packaged_count} workload(s)"
log "next: bash ${SCRIPT_DIR}/build-firemarshal-images.sh"
