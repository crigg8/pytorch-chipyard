# `linalg.matmul` custom archive test

This example keeps `aten.mm` on the normal TorchInductor/Triton path, then
replaces the bufferized `linalg.matmul` inside the generated Triton kernel with
a call to `triton_chipyard_matmul_f32` from `libtriton_chipyard.a`.

The call lives inside the kernel object. `runner.cpp` still launches the Triton
kernel normally, and the generated `build.sh` links the archive after all kernel
objects so the external symbol is resolved in `model.elf`.

Docker, FireMarshal, and FireSim commands are intentionally left for manual
execution because they may require host sudo access.

## 1. Build `libtriton_chipyard.a`

```bash
cd /home/hongjun/pytorch-chipyard

export CHIPYARD_DIR=/home/hongjun/hk_chipyard/chipyard
export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR=/home/hongjun/miniforge3/pkgs/riscv-tools-1.0.3-0_h1234567_ga1b1b14/riscv-tools
export GEMMINI_PARAMS_H="$CHIPYARD_DIR/sims/firesim/deploy/results-build/gemmini_params.h"

bash examples/custom-linalg-matmul/build_lib.sh
```

## 2. Rebuild Stage 1 and generate artifacts

The C++ `triton-chipyard-opt` binary must be rebuilt because this feature adds a
new MLIR pass.

```bash
sudo docker build \
  -f docker/stage1.Dockerfile \
  -t pytorch-chipyard:stage1 .
```

```bash
sudo docker run --rm \
  -v "$PWD/examples:/opt/pytorch-chipyard/examples" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 \
  bash scripts/run_custom_linalg_matmul.sh
```

`model.py` registers the GEMV-priority and ordinary matmul rules with
`@register_triton_chipyard_extern_call`. `mapping.json` records the equivalent
serialized schema for compatibility testing; the normal run does not read it.

The final TTShared IR must contain the external call and no remaining
`linalg.matmul`:

```bash
ART=examples/artifact-custom-linalg-matmul/gemmini
rg -n 'triton_chipyard_matmul_f32|linalg.matmul' "$ART/ttshared.mlir"
```

Check archive staging and link order:

```bash
python - <<'PY'
import json
from pathlib import Path

artifact = Path("examples/artifact-custom-linalg-matmul/gemmini")
spec = json.loads((artifact / "model_spec.json").read_text())
assert len(spec["triton_custom_libraries"]) == 1
assert spec["triton_custom_libraries"][0]["basename"] == "libtriton_chipyard.a"
assert any(step["kind"] == "launch" for step in spec["steps"])
assert all(step["kind"] != "custom_call" for step in spec["steps"])
print("libtriton archive plan OK")
PY

rg -n 'KERNEL_OBJECTS|TRITON_CUSTOM_LIBRARIES|libtriton_chipyard' "$ART/build.sh"
```

The example finalizes the kernel object and custom archive after
`async_compile.wait()` completes.  To repair an artifact produced by an older
image without recompiling it, run:

```bash
python examples/custom-linalg-matmul/finalize_artifact.py --artifact-dir examples/artifact-custom-linalg-matmul/gemmini --library examples/custom-linalg-matmul/libtriton_chipyard.a
```

Optional pass-level checks inside the rebuilt image:

```bash
sudo docker run --rm pytorch-chipyard:stage1 bash -lc '
  source scripts/env.sh
  OPT="$TRITON_CHIPYARD_OPT_PATH"
  FILECHECK="$LLVM_PROJECT_PATH/build/bin/FileCheck"
  "$OPT" triton_chipyard/test/Conversion/bufferize-and-deallocate.mlir \
    --one-shot-bufferize=allow-return-allocs-from-loops \
    --buffer-deallocation-pipeline \
    | "$FILECHECK" triton_chipyard/test/Conversion/bufferize-and-deallocate.mlir
  "$OPT" triton_chipyard/test/Conversion/linalg-to-function-call.mlir \
    --linalg-to-function-call="target-op=linalg.matmul function=custom_matmul" \
    | "$FILECHECK" triton_chipyard/test/Conversion/linalg-to-function-call.mlir
  "$OPT" triton_chipyard/test/Conversion/linalg-to-function-call-types.mlir \
    --linalg-to-function-call="target-op=linalg.matmul function=custom_gemv operand-types=f32:1|2:1:squeeze_static:1;f32:2:2:preserve:0;f32:1|2:1:squeeze_static:1" \
    | "$FILECHECK" triton_chipyard/test/Conversion/linalg-to-function-call-types.mlir
  python triton_chipyard/test/test_custom_linalg.py
'
```

## 3. Build and run the Gemmini 2-core workload

```bash
sudo chown -R "$USER:$USER" examples/artifact-custom-linalg-matmul

export CHIPYARD_DIR=/home/hongjun/hk_chipyard/chipyard
export PYTORCH_CHIPYARD_FPGA_DB=/opt/firesim-db0.json
export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR=/home/hongjun/miniforge3/pkgs/riscv-tools-1.0.3-0_h1234567_ga1b1b14/riscv-tools
source scripts/env.sh

bash scripts/build-chipyard-elves.sh \
  --artifact-dir=examples/artifact-custom-linalg-matmul/gemmini \
  --backend=gemmini \
  --core=2

bash scripts/package-firemarshal-workload.sh
bash scripts/build-firemarshal-images.sh
bash scripts/run-firesim-workloads.sh \
  --workload=custom-linalg-matmul-gemmini-2core
```

## 4. Validate the result

```bash
PYTORCH_CHIPYARD_DUMP_PATH="$PWD/examples/artifact-custom-linalg-matmul/gemmini" \
python examples/custom-linalg-matmul/model.py \
  --validate \
  --output-bin="$PWD/scripts/figures/results-workload/custom-linalg-matmul-gemmini-2core/output.bin"
```

Validation succeeds when it prints `[validate] match=True`.
