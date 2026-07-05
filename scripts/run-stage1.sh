#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

log() {
  printf '[stage1] %s\n' "$*"
}

die() {
  printf '[stage1][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-stage1.sh [options]

Run the Stage 1 compiler workflow and write artifacts under examples/.

Options:
  --skip-cnn                Skip default CNN workloads.
  --skip-im2col             Skip im2col CNN workloads.
  --skip-sdpa               Skip default SDPA LLM workloads.
  --skip-flex-attn          Skip flash/window attention LLM workloads.
  --skip-gemmini-autotune   Skip gemmini-max-autotune.
  -h, --help                Show this help.
EOF
}

skip_cnn=0
skip_im2col=0
skip_sdpa=0
skip_flex_attn=0
skip_gemmini_autotune=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --skip-cnn)
      skip_cnn=1
      shift
      ;;
    --skip-im2col)
      skip_im2col=1
      shift
      ;;
    --skip-sdpa)
      skip_sdpa=1
      shift
      ;;
    --skip-flex-attn)
      skip_flex_attn=1
      shift
      ;;
    --skip-gemmini-autotune)
      skip_gemmini_autotune=1
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

cd "${REPO_ROOT}"
mkdir -p examples

if [[ "${skip_cnn}" -eq 0 ]]; then
  log "running default CNN workloads"
  bash "${SCRIPT_DIR}/run-cnn.sh"
fi

if [[ "${skip_im2col}" -eq 0 ]]; then
  log "running im2col CNN workloads"
  bash "${SCRIPT_DIR}/run-im2col.sh"
fi

if [[ "${skip_sdpa}" -eq 0 ]]; then
  log "running SDPA LLM workloads"
  bash "${SCRIPT_DIR}/run-sdpa.sh"
fi

if [[ "${skip_flex_attn}" -eq 0 ]]; then
  log "running flash/window attention LLM workloads"
  bash "${SCRIPT_DIR}/run-flex-attn.sh"
fi

if [[ "${skip_gemmini_autotune}" -eq 0 ]]; then
  log "running gemmini-max-autotune workload"
  bash "${SCRIPT_DIR}/run_gemmini_autotune.sh"
fi

log "done; artifacts are under ${REPO_ROOT}/examples"
