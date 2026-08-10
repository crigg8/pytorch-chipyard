#!/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
export TRITON_PLUGIN_DIRS=$WORKSPACE/triton_chipyard
export LLVM_BUILD_DIR=$WORKSPACE/llvm-project/build
export LLVM_BINARY_DIR=$WORKSPACE/llvm-project/build/bin
export TRITON_CHIPYARD_OPT_PATH=$WORKSPACE/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_chipyard/tools/triton-chipyard-opt/triton-chipyard-opt


############################################################
## conda
############################################################

conda create -n pytorch-chipyard python=3.12 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-chipyard
conda install -n pytorch-chipyard conda-forge::conda-lock=1.4.0 -y
conda install -n pytorch-chipyard -c conda-forge gcc_linux-64=13 gxx_linux-64=13 zlib libzlib -y

python -m pip install matplotlib pandas
python -m pip install torch==2.8.0
python -m pip install torchvision==0.23.0
python -m pip uninstall -y triton

# triton-chipyard deps
python -m pip install ninja cmake wheel pytest pybind11 setuptools

# triton deps
pushd triton
python -m pip install -r python/requirements.txt 
popd

# buddy-mlir
pushd buddy-mlir
python -m pip install -r requirements.txt
popd

############################################################
## Build LLVM
############################################################

pushd llvm-project
mkdir -p build
cmake -G Ninja -B build llvm \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU;RISCV"
cmake --build build
popd

############################################################
## Build Triton + Triton_Shared
############################################################

# git clone https://github.com/microsoft/triton-shared.git triton_shared
# git clone https://github.com/triton-lang/triton.git
pushd triton
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  python -m pip install --no-build-isolation -e .
popd 


############################################################
## Build Buddy-Mlir
############################################################

pushd buddy-mlir
mkdir -p build
pushd build
cmake -G Ninja .. \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DLLVM_DIR=$WORKSPACE/llvm-project/build/lib/cmake/llvm \
    -DMLIR_DIR=$WORKSPACE/llvm-project/build/lib/cmake/mlir \
    -DLLVM_MAIN_SRC_DIR=$WORKSPACE/llvm-project/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
ninja
popd
popd

############################################################
## Build PyTorch
############################################################

BLUE="\033[0;34m"; GREEN="\033[0;32m"; RED="\033[0;31m"; NC="\033[0m"
info(){ echo -e "${BLUE}[INFO]${NC} $*"; }
ok(){ echo -e "${GREEN}[OK]${NC} $*"; }
die(){ echo -e "${RED}[ERR]${NC} $*"; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-python3}"

# 1) source _inductor location
SOURCE="${PATCHED_INDUCTOR_DIR:-}"
if [[ -z "$SOURCE" ]]; then
  if [[ -d "$WORKSPACE/pytorch/torch/_inductor" ]]; then
    SOURCE="$WORKSPACE/pytorch/torch/_inductor"
  elif [[ -d "$WORKSPACE/torch/_inductor" ]]; then
    SOURCE="$WORKSPACE/torch/_inductor"
  elif [[ -d "$WORKSPACE/LLAPI-PyTorch/torch/_inductor" ]]; then
    SOURCE="$WORKSPACE/LLAPI-PyTorch/torch/_inductor"
  else
    die "patched _inductor not found. Set PATCHED_INDUCTOR_DIR or place it under ./pytorch/torch/_inductor"
  fi
fi
[[ -d "$SOURCE" ]] || die "source not a dir: $SOURCE"

# 2) installed torch location
TORCH_DIR="$("$PYTHON_BIN" - <<'PY'
import os, sys
import torch
v = torch.__version__
if not str(v).startswith("2.8.0"):
    print(f"ERROR: expected torch 2.8.0.*, got {v}", file=sys.stderr)
    sys.exit(2)
print(os.path.dirname(torch.__file__))  # .../site-packages/torch
PY
)" || die "cannot locate torch (or version mismatch)"

TARGET="$TORCH_DIR/_inductor"
[[ -w "$TORCH_DIR" ]] || die "no write permission: $TORCH_DIR (wrong env?)"

info "python : $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
info "source : $SOURCE"
info "target : $TARGET"

# 3) backup -> replace
info "Removing existing: $TARGET"
rm -rf "$TARGET"
mkdir -p "$TARGET"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$SOURCE/" "$TARGET/"
else
  cp -a "$SOURCE/." "$TARGET/"
fi

# 4) verify
"$PYTHON_BIN" - <<'PY'
import torch, torch._inductor, os
print("torch version  :", torch.__version__)
print("torch._inductor:", torch._inductor.__file__)
PY

ok "patched torch/_inductor"
