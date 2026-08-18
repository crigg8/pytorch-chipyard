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
  --skip-alias-first        Skip the alias-first ablation workloads.
  --skip-im2col             Skip im2col CNN workloads.
  --skip-sdpa               Skip default SDPA LLM workloads.
  --skip-flex-attn          Skip flash/window attention LLM workloads.
  --skip-gemmini-autotune   Skip gemmini-max-autotune.
  --skip-table2             Skip Docker-side Table 2 PyTorch compile timing.
  --only-table2             Run only the Docker-side Table 2 PyTorch compile
                            measurements and save artifacts for Stage 2.
  --table2-models=LIST      Limit --only-table2 testing to selected CNNs.
                            Default: alexnet,squeezenet,mobilenetv2,resnet50.
  --table2-repeats=N        Table 2 compile trials. Default: 1.
  --only-alias-first        Compile only the Figure 6(c) alias-first ablation:
                            CNN alias-first off plus LLM alias-first on/off,
                            all using Gemmini and 4 cores.
  --only-alias-first-cnn-off
                            Compile only the four CNN alias-first off cases.
  -h, --help                Show this help.
EOF
}

skip_cnn=0
skip_alias_first=0
skip_im2col=0
skip_sdpa=0
skip_flex_attn=0
skip_gemmini_autotune=0
skip_table2=0
only_table2=0
table2_models="alexnet,squeezenet,mobilenetv2,resnet50"
table2_repeats="${TABLE2_REPEATS:-1}"
only_alias_first=0
only_alias_first_cnn_off=0
skip_alias_first_requested=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --skip-cnn)
      skip_cnn=1
      shift
      ;;
    --skip-alias-first | --skip-alias-first-ablation)
      skip_alias_first=1
      skip_alias_first_requested=1
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
    --skip-table2)
      skip_table2=1
      shift
      ;;
    --only-table2)
      only_table2=1
      shift
      ;;
    --table2-models=*)
      table2_models="${1#*=}"
      shift
      ;;
    --table2-models)
      [[ "$#" -ge 2 ]] || die "--table2-models requires a value"
      table2_models="$2"
      shift 2
      ;;
    --table2-repeats=*)
      table2_repeats="${1#*=}"
      shift
      ;;
    --table2-repeats)
      [[ "$#" -ge 2 ]] || die "--table2-repeats requires a value"
      table2_repeats="$2"
      shift 2
      ;;
    --only-alias-first | --only-alias-first-ablation)
      only_alias_first=1
      shift
      ;;
    --only-alias-first-cnn-off)
      only_alias_first_cnn_off=1
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

if [[ "${only_alias_first}" -eq 1 && "${only_alias_first_cnn_off}" -eq 1 ]]; then
  die "--only-alias-first and --only-alias-first-cnn-off are mutually exclusive"
fi
if [[ "${only_table2}" -eq 1 && \
      ( "${only_alias_first}" -eq 1 || "${only_alias_first_cnn_off}" -eq 1 ) ]]; then
  die "--only-table2 cannot be combined with an alias-first only option"
fi
if [[ "${only_table2}" -eq 1 && "${skip_table2}" -eq 1 ]]; then
  die "--only-table2 cannot be combined with --skip-table2"
fi

[[ "${table2_repeats}" =~ ^[1-9][0-9]*$ ]] || \
  die "--table2-repeats must be a positive integer"

if [[ "${only_table2}" -eq 1 ]]; then
  skip_cnn=1
  skip_alias_first=1
  skip_im2col=1
  skip_sdpa=1
  skip_flex_attn=1
  skip_gemmini_autotune=1
fi

if [[ "${only_alias_first}" -eq 1 || "${only_alias_first_cnn_off}" -eq 1 ]]; then
  [[ "${skip_alias_first_requested}" -eq 0 ]] || \
    die "an alias-first only option cannot be combined with --skip-alias-first"
  skip_cnn=1
  skip_im2col=1
  skip_sdpa=1
  skip_flex_attn=1
  skip_gemmini_autotune=1
  skip_table2=1
fi

cd "${REPO_ROOT}"
mkdir -p examples

if [[ "${skip_cnn}" -eq 0 ]]; then
  log "running default CNN workloads"
  bash "${SCRIPT_DIR}/run-cnn.sh"
fi

if [[ "${skip_alias_first}" -eq 0 ]]; then
  log "running CNN Gemmini 4-core alias-first off ablation"
  bash "${SCRIPT_DIR}/run-alias-first-cnn.sh" --mode=off

  if [[ "${only_alias_first_cnn_off}" -eq 0 ]]; then
    log "running LLM SDPA seq=256 Gemmini 4-core alias-first on/off ablation"
    bash "${SCRIPT_DIR}/run-alias-first-sdpa.sh"
  fi
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

if [[ "${skip_table2}" -eq 0 ]]; then
  table2_results_root="${TABLE2_RESULTS_ROOT:-${REPO_ROOT}/results/table2}"
  table2_run_id="$(date -u +%Y%m%dT%H%M%SZ)-stage1"
  table2_output_dir="${table2_results_root}/${table2_run_id}"
  log "measuring Table 2 PyTorch compile times inside the Stage 1 container"
  TABLE2_PYTORCH_COMPILE_CONTEXT=docker-stage1 \
    bash "${REPO_ROOT}/run_table2.sh" \
    --models="${table2_models}" \
    --toolchains=pytorch \
    --phases=compile \
    --repeats="${table2_repeats}" \
    --output-dir="${table2_output_dir}"
  # Docker root is mapped to nobody on root-squashed bind mounts, and Torch's
  # atomic weight-file write leaves weights.bin at mode 0600. Stage 2 runs as
  # the host AE account and must be able to read artifacts and append results.
  chmod -R a+rwX "${table2_output_dir}"
  mkdir -p "${table2_results_root}"
  ln -sfn "${table2_run_id}" "${table2_results_root}/stage1-latest"
  log "Table 2 Stage 1 results: ${table2_output_dir}"
fi

log "done; artifacts are under ${REPO_ROOT}/examples and ${REPO_ROOT}/results"
