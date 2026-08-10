# `aten.mv` custom archive test

This example routes `aten.mv.default` to a `torch.library.custom_op`, records
the resulting extern allocation as a Chipyard runner custom call, and links a
Gemmini implementation from `libpytorch_chipyard.a`. The custom call is made
once by `runner.cpp`; it is not wrapped in an OpenMP or grid loop.

The commands below are intentionally not run by the repository automation.
Docker, FireMarshal, and FireSim may require host sudo access.

## 1. Build the RISC-V archive

Use the parameter header generated for the FP32 DIM=8 Gemmini bitstream. The
default Gemmini test header in an unconfigured Chipyard checkout is commonly
an int8 DIM=16 configuration and must not be used for this test.

```bash
cd /home/hongjun/pytorch-chipyard

export CHIPYARD_DIR=/home/hongjun/hk_chipyard/chipyard
export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR=/path/to/riscv-gcc-12.2.0
export GEMMINI_PARAMS_H="$CHIPYARD_DIR/sims/firesim/deploy/results-build/gemmini_params.h"

bash examples/custom-mv/build_lib.sh
```

## 2. Build the Stage 1 image and generate artifacts

```bash
sudo docker build \
  -f docker/stage1.Dockerfile \
  -t pytorch-chipyard:stage1 .

sudo docker run --rm -it \
  -v "$PWD/examples:/opt/pytorch-chipyard/examples" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  -e PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY=/opt/pytorch-chipyard/examples/custom-mv/libpytorch_chipyard.a \
  pytorch-chipyard:stage1 \
  bash scripts/run_custom_mv.sh

sudo chown -R "$USER:$USER" examples/artifact-custom-mv
```

Optional source-level tests inside the rebuilt image:

```bash
sudo docker run --rm \
  pytorch-chipyard:stage1 \
  bash -lc 'python pytorch/test/inductor/test_chipyard_custom_ops.py && \
    python pytorch/test/inductor/test_chipyard_model_builder.py'
```

The Stage 1 result should have one custom call and no Triton launch:

```bash
python - <<'PY'
import json
from pathlib import Path

artifact = Path("examples/artifact-custom-mv/gemmini")
spec = json.loads((artifact / "model_spec.json").read_text())
assert [step["kind"] for step in spec["steps"]] == ["custom_call"]
assert spec["steps"][0]["symbol"] == "pytorch_chipyard_mv_f32"
assert len(spec["custom_op_libraries"]) == 1
print("custom call plan OK")
PY

rg -n 'pytorch_chipyard_mv_f32|#pragma omp' \
  examples/artifact-custom-mv/gemmini/runner.cpp
```

`pytorch_chipyard_mv_f32` should have one generated invocation. There should be
no `#pragma omp` line because this model has no Triton launch.

## 3. Build and run the Gemmini 2-core workload

```bash
export CHIPYARD_DIR=/home/hongjun/hk_chipyard/chipyard
export PYTORCH_CHIPYARD_FPGA_DB=/opt/firesim-db0.json
export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR=/path/to/riscv-gcc-12.2.0
source scripts/env.sh

bash scripts/build-chipyard-elves.sh \
  --artifact-dir=examples/artifact-custom-mv/gemmini \
  --backend=gemmini \
  --core=2

bash scripts/package-firemarshal-workload.sh
bash scripts/build-firemarshal-images.sh
bash scripts/run-firesim-workloads.sh \
  --workload=custom-mv-gemmini-2core
```

The package and image scripts operate on generated artifacts already present
under `examples/`; use a clean artifact directory when only this workload
should be packaged.

## 4. Validate the collected output

```bash
PYTORCH_CHIPYARD_DUMP_PATH="$PWD/examples/artifact-custom-mv/gemmini" \
python examples/custom-mv/model.py \
  --validate \
  --output-bin="$PWD/scripts/figures/results-workload/custom-mv-gemmini-2core/output.bin"
```

Validation passes when the FireSim output matches eager PyTorch with
`atol=rtol=1e-3`.
