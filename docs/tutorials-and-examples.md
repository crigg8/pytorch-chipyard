# Section 2: Tutorials

This section contains two tutorials:

- Standalone Triton-Chipyard matmul example.
- PyTorch-Chipyard ResNet50 compilation and artifact execution flow.

To test Triton-Chipyard, install Chipyard and build the target hardware with
Verilator. To use PyTorch-Chipyard beyond compilation and run generated model
artifacts, the additional FireSim local setup described in the installation
section is also required.

## 2.1 Triton-Chipyard Matmul

The main goal of
[pytorch-chipyard](https://github.com/JongseoKang/pytorch-chipyard) is PyTorch
model execution, but Triton-Chipyard can also be used as a standalone Triton
backend.

To run a standalone Triton kernel, set the Chipyard environment path and the
Verilator simulator path with `CHIPYARD_ENV_PATH` and
`CHIPYARD_SIM_VERILATOR_PATH`.

This tutorial uses the Triton-Chipyard matmul example:

```
triton_chipyard/example/test_matmul.py
```

Triton-Chipyard follows Triton-Shared's driver activation model. After
`ChipyardDriver` is set as the active driver, Triton compilation is routed
through `triton_chipyard` instead of the NVIDIA or AMD backend.

```python
import torch
import triton
import triton.language as tl
from triton.backends.triton_chipyard.driver import ChipyardDriver

triton.runtime.driver.set_active(ChipyardDriver())
```

The rest of the kernel authoring flow is almost identical to Triton. Write the
JIT function with the `triton.jit` decorator. Newer Triton APIs that target
NVIDIA or AMD features, such as `tma.load`, are not available on the Chipyard
path; use the standard `tl.load` and `tl.store` memory operations.

```python
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_mask = k_start + offs_k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_mask[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_mask[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
```

The host function also follows normal Triton conventions, with additional
Gemmini metadata. The `gemmini_tile_*` metadata is consumed by Buddy-MLIR to
choose Gemmini matmul tiling. If all values are set to `0`, Buddy-MLIR uses its
own heuristic.

```python
def matmul(a, b, block_m, block_n, block_k,
    gemmini_tile_i, gemmini_tile_j, gemmini_tile_k):
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0),
        b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        gemmini_tile_i=gemmini_tile_i,
        gemmini_tile_j=gemmini_tile_j,
        gemmini_tile_k=gemmini_tile_k,
    )
    return c
```

Call the host function with `torch.Tensor` inputs and metadata matching the
function signature above. The Verilator result is returned as a `torch.Tensor`,
which can be compared against eager PyTorch.

```python
triton.runtime.driver.set_active(ChipyardDriver())

torch.manual_seed(0)
a = torch.randint(-4, 4, (128, 128), device="cpu", dtype=torch.float32)
b = torch.randint(-4, 4, (128, 128), device="cpu", dtype=torch.float32)


triton_output = matmul(a, b, 32, 32, 32, 0, 0, 0)
torch_output = torch.matmul(a, b)
torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=1e-2)

print(f"A shape: {tuple(a.shape)}")
print(f"B shape: {tuple(b.shape)}")
print(f"Output shape: {tuple(triton_output.shape)}")
print("Triton output:")
print(triton_output)
print("Torch eager output:")
print(torch_output)
```

Run the example with:

```bash
cd pytorch-chipyard

# Edit scripts/env.sh if your local checkout or build paths differ.
source scripts/env.sh

export CHIPYARD_ENV_PATH=/path/to/chipyard/conda-env
export CHIPYARD_SIM_VERILATOR_PATH=/path/to/chipyard/verilator/simulator
# Optional IR dump directory. Unset it to disable dump output.
export TRITON_CHIPYARD_DUMP_PATH=$PWD/IR/triton-matmul

python triton_chipyard/example/test_matmul.py \
  --block-size-m 128 \
  --block-size-n 128 \
  --block-size-k 128

# Expected output
A shape: (128, 128)
B shape: (128, 128)
Output shape: (128, 128)
Triton output:
tensor([...])
Torch eager output:
tensor([...])
```

The example finishes without an assertion failure when
`torch.testing.assert_close` passes. The exact tensor values are omitted above.

Verilator is slow, so the run may take minutes to tens of minutes depending on
the host CPU. `CHIPYARD_SIM_VERILATOR_PATH` is required for this standalone
example to execute the kernel through the Verilator simulator. If the variable
is empty, Triton-Chipyard still compiles the kernel into a RISC-V binary, but
the standalone simulator launch is skipped.

### Triton-Chipyard Environment Variables

`scripts/env.sh` defines the default environment for both standalone
Triton-Chipyard examples and PyTorch-Chipyard model compilation. Update local
paths in that file if your checkout, LLVM build, Triton build, Buddy-MLIR build,
or Chipyard tree differs from the default repository layout.

Core path variables:

| Variable | Purpose |
| --- | --- |
| `WORKSPACE` | Repository root. Computed from `scripts/env.sh`. |
| `BUDDY_BINARY_DIR` | Buddy-MLIR build binary directory. |
| `TRITON_CHIPYARD_OPT_PATH` | `triton-chipyard-opt` pass driver built with Triton. |
| `LLVM_PROJECT_PATH` | LLVM/MLIR source tree used for runtime headers and support sources. |
| `CHIPYARD_DIR` | Chipyard checkout. |
| `CHIPYARD_ENV_PATH` | Chipyard environment script sourced by generated build paths. |
| `CHIPYARD_SIM_VERILATOR_PATH` | Verilator simulator binary used by standalone kernel execution. |
| `TRITON_CHIPYARD_DUMP_PATH` | Optional dump directory for `tt.mlir`, `ttshared.mlir`, and related lowering files. |

Gemmini and RVV target variables select the hardware target for
Triton-Chipyard lowering, so they are also used by the full PyTorch model
example below. Set only one target at a time. If neither Gemmini nor RVV is
enabled, Triton-Chipyard lowers to the scalar CPU path, such as a Rocket core.

Gemmini target variables:

```bash
# Enable Gemmini lowering.
export TRITON_CHIPYARD_USE_GEMMINI=1
# Disable RVV lowering for the Gemmini target.
export TRITON_CHIPYARD_USE_RVV=0
# Gemmini address length for the default FP32 configuration.
export TRITON_CHIPYARD_GEMMINI_ADDR_LEN=32
# Gemmini systolic array dimension.
export TRITON_CHIPYARD_GEMMINI_DIM=8
# Number of rows in each Gemmini scratchpad bank.
export TRITON_CHIPYARD_GEMMINI_BANK_ROWS=2048
# Number of rows in the Gemmini accumulator.
export TRITON_CHIPYARD_GEMMINI_ACC_ROWS=2048
# Gemmini element type.
export TRITON_CHIPYARD_GEMMINI_ELEM_T=f32
# Gemmini accumulator type.
export TRITON_CHIPYARD_GEMMINI_ACC_T=f32
# RISC-V ISA string used by generated builds.
export TRITON_CHIPYARD_RISCV_MARCH=rv64imafdc
# RISC-V ABI string used by generated builds.
export TRITON_CHIPYARD_RISCV_MABI=lp64d
```

RVV target variables:
Use these variables to compile through the RVV path, such as a Saturn target.

```bash
# Disable Gemmini lowering for the RVV target.
export TRITON_CHIPYARD_USE_GEMMINI=0
# Enable RVV lowering.
export TRITON_CHIPYARD_USE_RVV=1
# RISC-V ISA string for the RVV target.
export TRITON_CHIPYARD_RISCV_MARCH=rv64imafdcv_zicsr_zifencei_zvl128b
# RISC-V ABI string used by generated builds.
export TRITON_CHIPYARD_RISCV_MABI=lp64d
# RVV vector architecture string consumed by the backend.
export TRITON_CHIPYARD_RISCV_VARCH=vlen:128,elen:64
```

FireMarshal and FireSim path variables are described in the ResNet50 flow below.

## 2.2 PyTorch-Chipyard ResNet50

The PyTorch-Chipyard path starts from a regular PyTorch model example under
`examples/`. The ResNet50 example configures Triton-Chipyard as the Inductor CPU
backend, runs `torch.compile`, and writes model-level artifacts.

The flow is mostly the same as using a normal PyTorch model, with a few
PyTorch-Chipyard-specific settings. This example runs the `torchvision`
ResNet50 model. First, set the TorchInductor CPU backend to `triton_chipyard`
and activate the Triton Chipyard driver so Triton kernels can lower to RISC-V.

The model weights are then made contiguous. This is usually not strictly
required because most weights are already contiguous, and Triton kernels can
lower tensor views using PyTorch tensor metadata, but keeping weights contiguous
makes the generated model artifact simpler. The example image is then loaded
and preprocessed.

```python
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

import triton
from triton.backends.triton_chipyard.driver import ChipyardDriver

triton.runtime.driver.set_active(ChipyardDriver())
inductor_config.cpu_backend = "triton_chipyard"

ARTIFACT_DIR = Path(os.environ["PYTORCH_CHIPYARD_DUMP_PATH"]).resolve()
IMAGE_PATH = Path("examples/img/bus.jpg")
WEIGHTS = ResNet50_Weights.DEFAULT
DTYPE = torch.float32
ATOL = 1e-3


def build_model() -> torch.nn.Module:
    torch.manual_seed(0)
    model = resnet50(weights=WEIGHTS).to(device="cpu", dtype=DTYPE).eval()
    for parameter in model.parameters():
        if not parameter.is_contiguous():
            parameter.data = parameter.data.contiguous()
    for buffer in model.buffers():
        if not buffer.is_contiguous():
            buffer.data = buffer.data.contiguous()
    return model

def make_image_input() -> torch.Tensor:
    image = Image.open(IMAGE_PATH).convert("RGB")
    tensor = WEIGHTS.transforms()(image).to(dtype=DTYPE)
    batch = tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
    return torch.as_strided(
        batch, size=batch.shape, stride=(batch.stride(-1), *batch.stride()[1:])
    )
```

Model compilation happens when the compiled model is called with an input
tensor. `torch.compile()` prepares the model for compilation, but the actual
compile is triggered by `compiled_model(inputs)`. The generated model is
specialized to the static shape of that input tensor, so a different batch size
or input shape requires another compiled-model call and may trigger another
compile.

The compiled model call normally returns an output tensor. In this
PyTorch-Chipyard artifact-generation flow, do not set the standalone Verilator
simulator path: without a simulator launch, the Triton-Chipyard JIT path may
return a placeholder value rather than a meaningful numerical result. This is
expected, and it keeps model artifact generation within a reasonable time.

Use the following environment variables for the compile step. Keep the
Triton-Chipyard target hardware variables from the previous section set as
well.

```bash
# Force Inductor to regenerate artifacts instead of reusing cached code.
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
# Restrict Inductor kernel generation and backend selection to Triton.
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=TRITON
# Enable generation of runner.cpp, model_spec.json, util.py, and related files.
export TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1
# Copy kernel object artifacts under PYTORCH_CHIPYARD_DUMP_PATH/kernels.
export TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS=1
# Leave model.elf compilation to the explicit build.sh step below.
export TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER=0
# Disable the larger Gemmini autotune search space for full-model runs.
export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=0
```

The generated artifacts are described after the compile command.

Compile:

```python
model = build_model()
inputs = make_image_input()

compiled_model = torch.compile(model, backend="inductor")
with torch.inference_mode():
    compiled_model(inputs)

util = import_artifact_util()
util.write_inputs_bin(inputs)
```

util.py:

PyTorch-Chipyard generates `util.py` for convenience. It has two main roles.
First, it packs the input tensor into `input.bin` for the Chipyard run; the
compile snippet above dynamically imports `util.py` and uses it to create
`input.bin`. Second, it converts the Chipyard result `output.bin` back into a
`torch.Tensor` so it can be validated against the golden eager-mode value.

```python
def import_artifact_util():
    import importlib.util

    util_path = ARTIFACT_DIR / "util.py"
    if not util_path.exists():
        raise FileNotFoundError(f"generated util.py not found: {util_path}")
    spec = importlib.util.spec_from_file_location("resnet50_artifact_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import artifact util: {util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Validate:

This snippet uses `util.py` to compare the target output with the PyTorch
eager-mode golden value. `read_inputs_bin` loads the `input.bin` used by the
Chipyard run as a `torch.Tensor`, and `read_outputs_bin` loads the generated
`output.bin`. `torch.testing.assert_close` requires matching shapes, so the
observed tensor is reshaped when it has the same number of elements as the
golden tensor but a different shape.

The scalar and Gemmini paths have shown error below `1e-3`, while the RVV path
has shown larger error below `1e-1`. In both cases the semantic result, such as
top-k classification behavior, remains the same; the difference comes from
running the same model through different hardware paths.

```python
util = import_artifact_util()
inputs = util.read_inputs_bin(ARTIFACT_DIR / "input.bin")
observed = util.read_outputs_bin(ARTIFACT_DIR / "output.bin")

model = build_model()
with torch.inference_mode():
    golden = model(inputs)

if tuple(golden.shape) != tuple(observed.shape):
    observed = observed.reshape_as(golden)

max_abs_err = (observed.float() - golden.float()).abs().max().item()
print(f"[validate] max_abs_err={max_abs_err:.6e}")
torch.testing.assert_close(observed, golden, atol=ATOL, rtol=0)
```

Compile ResNet50 with the default FP32 Gemmini configuration:

```bash
cd pytorch-chipyard
conda activate pytorch-chipyard

export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=TRITON
export TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1
export TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS=1
export TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER=0
export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=0

export TRITON_CHIPYARD_USE_GEMMINI=1
export TRITON_CHIPYARD_USE_RVV=0
export TRITON_CHIPYARD_GEMMINI_ADDR_LEN=32
export TRITON_CHIPYARD_GEMMINI_DIM=8
export TRITON_CHIPYARD_GEMMINI_BANK_ROWS=2048
export TRITON_CHIPYARD_GEMMINI_ACC_ROWS=2048
export TRITON_CHIPYARD_GEMMINI_ELEM_T=f32
export TRITON_CHIPYARD_GEMMINI_ACC_T=f32
export TRITON_CHIPYARD_RISCV_MARCH=rv64imafdc
export TRITON_CHIPYARD_RISCV_MABI=lp64d

export PYTORCH_CHIPYARD_DUMP_PATH=$PWD/examples/resnet50/gemmini
export TRITON_CHIPYARD_DUMP_PATH=$PYTORCH_CHIPYARD_DUMP_PATH
export TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/resnet50-gemmini

python examples/ResNet50.py --compile
```

After the run, the artifact directory has this structure:

```text
$PYTORCH_CHIPYARD_DUMP_PATH/
  runner.cpp
  model_spec.json
  weights.bin
  weights.manifest.json
  input.bin
  build.sh
  util.py
  kernels/
    <kernel-cache-key>/
      <kernel-name>.obj
  tt.mlir
  ttshared.mlir
```

`build.sh` is generated but is not called automatically in the command sequence
above. Build the RISC-V executable explicitly:

```bash
cd "$PYTORCH_CHIPYARD_DUMP_PATH"
CHIPYARD_OMP_NUM_THREADS=4 bash ./build.sh
```

The generated executable expects input, weight, and output file paths:

```bash
./model.elf input.bin weights.bin output.bin
```

### FireMarshal Packaging and FireSim Execution

To run the generated ELF through FireSim, first package it as a FireMarshal
workload. The example below uses an RVV workload name,
`resnet50-rvv-4core`. For the Gemmini artifact compiled above, use the same
layout with `resnet50-gemmini-4core` instead.

The FireMarshal workload directory should contain the workload JSON, overlay
directory, and guest-side runner script. The structure is roughly:

```bash
export WORKLOAD=resnet50-rvv-4core
export GUEST_DIR="$PYTORCH_CHIPYARD_WORKLOAD_DIR/overlay-$WORKLOAD/root/cnn/$WORKLOAD"

mkdir -p "$GUEST_DIR"
cp "$PYTORCH_CHIPYARD_DUMP_PATH/model.elf" "$GUEST_DIR/model.elf"
cp "$PYTORCH_CHIPYARD_DUMP_PATH/input.bin" "$GUEST_DIR/input.bin"
cp "$PYTORCH_CHIPYARD_DUMP_PATH/weights.bin" "$GUEST_DIR/weights.bin"
```

```text
$PYTORCH_CHIPYARD_WORKLOAD_DIR/
  resnet50-rvv-4core.json
  overlay-resnet50-rvv-4core/
    firemarshal.sh
    root/
      cnn/
        run-resnet50.sh
        resnet50-rvv-4core/
          model.elf
          input.bin
          weights.bin
```

`run-resnet50.sh` is executed inside the guest Linux image. Both the Saturn/RVV
and Gemmini paths may involve asynchronous target-side work, so the runner sets
OpenMP scheduling and affinity variables explicitly:

```bash
#!/bin/bash
set -u
set -o pipefail

ulimit -s unlimited
cd /root/cnn/resnet50-rvv-4core

mount -o remount,rw,noatime / 2>/dev/null || true

export OMP_NUM_THREADS=4
export OMP_THREAD_LIMIT=4
export OMP_STACKSIZE=64M
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=false
export OMP_PROC_BIND=close
export OMP_PLACES="{0},{1},{2},{3}"
export OMP_SCHEDULE=static
export OMP_WAIT_POLICY=PASSIVE
export GOMP_CPU_AFFINITY="0-3"
export GOMP_SPINCOUNT=0
export MALLOC_ARENA_MAX=1

chmod +x ./model.elf 2>/dev/null || true
rm -f output.bin run.log model.log autotune.log

exec >./run.log 2>&1
echo "[runner] start resnet50-rvv-4core"
./model.elf input.bin weights.bin output.bin
rc=$?
echo "[runner] model_ret=${rc}"
sync || true
sleep 5
exit "${rc}"
```

The overlay hook can simply call the guest runner:

```bash
#!/bin/sh
bash /root/cnn/run-resnet50.sh
```

Create the FireMarshal workload JSON under
`$PYTORCH_CHIPYARD_WORKLOAD_DIR/resnet50-rvv-4core.json`.
FireSim reads the workload JSON from `$FIRESIM_WORKLOAD_DIR`, so write these
fields as paths relative to `$FIRESIM_WORKLOAD_DIR`. Do not hardcode a relative
path that only works on one machine; compute the relative path that matches your
Chipyard/FireMarshal layout.

```json
{
  "benchmark_name": "resnet50-rvv-4core",
  "common_simulation_outputs": [
    "uartlog"
  ],
  "common_bootbinary": "<relative path from $FIRESIM_WORKLOAD_DIR to $FIREMARSHAL_IMAGE_DIR/resnet50-rvv-4core/resnet50-rvv-4core-bin>",
  "common_rootfs": "<relative path from $FIRESIM_WORKLOAD_DIR to $FIREMARSHAL_IMAGE_DIR/resnet50-rvv-4core/resnet50-rvv-4core.img>",
  "common_outputs": [
    "/root/cnn/resnet50-rvv-4core/output.bin",
    "/root/cnn/resnet50-rvv-4core/run.log",
    "/root/cnn/resnet50-rvv-4core/model.log",
    "/root/cnn/resnet50-rvv-4core/autotune.log"
  ]
}
```

Then build and install the FireMarshal image:

```bash
cd "$FIREMARSHAL_DIR"
./marshal --workdir "$PYTORCH_CHIPYARD_WORKLOAD_DIR" \
  build "$PYTORCH_CHIPYARD_WORKLOAD_DIR/resnet50-rvv-4core.json"
./marshal --workdir "$PYTORCH_CHIPYARD_WORKLOAD_DIR" \
  install "$PYTORCH_CHIPYARD_WORKLOAD_DIR/resnet50-rvv-4core.json"
```

This produces the FireMarshal image files referenced by the FireSim workload:

```text
$FIREMARSHAL_IMAGE_DIR/resnet50-rvv-4core/resnet50-rvv-4core-bin
$FIREMARSHAL_IMAGE_DIR/resnet50-rvv-4core/resnet50-rvv-4core.img
$FIRESIM_WORKLOAD_DIR/resnet50-rvv-4core.json
```

To run the workload through FireSim, prepare a runtime YAML and invoke the
FireSim manager commands directly. 
You can start from the default `config_runtime.yaml` and modify only the
required fields. The important fields are:

```yaml
target_config:
    default_hw_config: <FireSim hardware config name for the target bitstream>

workload:
    workload_name: resnet50-rvv-4core.json
    terminate_on_completion: yes
    suffix_tag: resnet50-rvv-4core

run_farm:
  recipe_arg_overrides:
    default_simulation_dir: <FireSim run directory>
    default_fpga_db: <FPGA DB json path>
    run_farm_hosts_to_use:
        - <fpga-host-name>: <run-farm-host-spec>
```

Run the FireSim manager commands with the modified runtime YAML:

```bash
cd "$FIRESIM_DIR"
source ./sourceme-manager.sh --skip-ssh-setup

firesim launchrunfarm \
  -c "$RUNTIME" \
  -a "$FIRESIM_HWDB_PATH" \
  -r "$FIRESIM_BUILD_RECIPES_PATH"
firesim infrasetup \
  -c "$RUNTIME" \
  -a "$FIRESIM_HWDB_PATH" \
  -r "$FIRESIM_BUILD_RECIPES_PATH"
firesim runworkload \
  -c "$RUNTIME" \
  -a "$FIRESIM_HWDB_PATH" \
  -r "$FIRESIM_BUILD_RECIPES_PATH"
firesim terminaterunfarm \
  -c "$RUNTIME" \
  -a "$FIRESIM_HWDB_PATH" \
  -r "$FIRESIM_BUILD_RECIPES_PATH" \
  --forceterminate
```

After `runworkload`, the logs and output files are generated under the FireSim
deploy `results-workload` directory. If no custom path is configured, this is
usually `$FIRESIM_DEPLOY_DIR/results-workload`. Copy the result files back to
the original PyTorch-Chipyard artifact directory before validation.

```bash
mkdir -p "$COLLECT_DIR"
cp "$RESULT_DIR/output.bin" "$COLLECT_DIR/output.bin"
cp "$RESULT_DIR/model.log" "$COLLECT_DIR/model.log"
cp "$RESULT_DIR/autotune.log" "$COLLECT_DIR/autotune.log"

cp "$COLLECT_DIR/output.bin" "$PYTORCH_CHIPYARD_DUMP_PATH/output.bin"

cd pytorch-chipyard
# Use the same artifact directory that was used for compilation, such as
# $PWD/examples/resnet50/gemmini or $PWD/examples/resnet50/rvv.
PYTORCH_CHIPYARD_DUMP_PATH=$PWD/examples/resnet50/gemmini \
  python examples/ResNet50.py --validate
```
