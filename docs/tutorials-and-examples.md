# Section 2: Tutorials and Examples

This section contains two tutorials:

- Standalone Triton-Chipyard matmul example.
- PyTorch-Chipyard ResNet50 compilation and artifact execution flow.

Start with [Prerequisites and Installation](getting-started.md). The standalone
matmul tutorial uses Verilator and needs no FPGA. ResNet50 compilation needs
the compiler stack; its FireMarshal/FireSim execution steps use the separately
configured FPGA environment.

## 2.1 Triton-Chipyard Matmul

The main goal of
[pytorch-chipyard](https://github.com/crigg8/pytorch-chipyard) is PyTorch
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
    M, K = a.shape
    assert b.shape[0] == K
    N = b.shape[1]
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

Before running, select the Gemmini target using the
[target variables below](#triton-chipyard-environment-variables), with values
matching the Verilator hardware configuration. Then run:

```bash
cd /path/to/pytorch-chipyard

# Set the external hardware checkout before loading derived paths.
export CHIPYARD_DIR=/path/to/chipyard
source scripts/env.sh

export CHIPYARD_ENV_PATH="$CHIPYARD_DIR/env.sh"
export CHIPYARD_SIM_VERILATOR_PATH=/path/to/built-verilator-simulator
# Optional IR dump directory. Unset it to disable dump output.
export TRITON_CHIPYARD_DUMP_PATH=$PWD/IR/triton-matmul

python triton_chipyard/example/test_matmul.py \
  --block-size-m 128 \
  --block-size-n 128 \
  --block-size-k 128
```

Expected output:

```text
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

To customize this example, change the input shapes and the three
`BLOCK_SIZE_*` launch parameters together. The grid launches one program per
output tile; each program loops over K, and the masks handle edge tiles.
Gemmini's `gemmini_tile_i/j/k` values control accelerator tiling within those
Triton tiles. Keep the dtype, ISA, and Gemmini configuration consistent with
the simulator, and compare each changed case with `torch.matmul`.

Verilator is slow, so the run may take minutes to tens of minutes depending on
the host CPU. `CHIPYARD_SIM_VERILATOR_PATH` is required for this standalone
example to execute the kernel through the Verilator simulator. If the variable
is empty, Triton-Chipyard still compiles the kernel into a RISC-V binary, but
the standalone simulator launch is skipped.

### Triton-Chipyard Environment Variables

`scripts/env.sh` defines the default paths for standalone Triton-Chipyard
examples and PyTorch-Chipyard model compilation. Export `CHIPYARD_DIR` before
sourcing it to select an external hardware checkout. If the LLVM, Triton, or
Buddy-MLIR build layout differs, override the compiler paths after sourcing it.
Target-selection examples in `env.sh` are comments: set the target variables
explicitly as below. Sourcing the file also clears
`CHIPYARD_SIM_VERILATOR_PATH`, so set the simulator path afterwards when needed.

Core path variables:

| Variable | Purpose |
| --- | --- |
| `WORKSPACE` | Repository root. Computed from `scripts/env.sh`. |
| `BUDDY_BINARY_DIR` | Buddy-MLIR build binary directory. |
| `TRITON_CHIPYARD_OPT_PATH` | `triton-chipyard-opt` pass driver built with Triton. |
| `LLVM_PROJECT_PATH` | LLVM/MLIR source tree used for runtime headers and support sources. |
| `CHIPYARD_DIR` | Chipyard checkout. |
| `CHIPYARD_ENV_PATH` | Chipyard's `env.sh` script sourced by generated build paths; this is a file, not the Conda environment directory. |
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

For a scalar target, set both `TRITON_CHIPYARD_USE_GEMMINI=0` and
`TRITON_CHIPYARD_USE_RVV=0`, and use `rv64imafdc` / `lp64d` for the ISA/ABI.
When switching targets, use a separate artifact directory and a matching
simulator or bitstream.

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
    image = Image.new("RGB", (224, 224), (73, 109, 137))
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

Run validation after collecting the target's `output.bin` from FireSim.
`read_inputs_bin` loads the exact input used by the target and
`read_outputs_bin` loads its result. Running `examples/ResNet50.py --validate`
checks that the mean absolute error (MAE) is at most `1e-3`.
The snippet below uses the same criterion. A nonzero exit or `match=False`
indicates a failed numerical check; compilation or simulator completion alone
does not establish numerical correctness.

```python
util = import_artifact_util()
inputs = util.read_inputs_bin(ARTIFACT_DIR / "input.bin")
observed = util.read_outputs_bin(ARTIFACT_DIR / "output.bin")

model = build_model()
with torch.inference_mode():
    golden = model(inputs)

if tuple(golden.shape) != tuple(observed.shape):
    if golden.numel() != observed.numel():
        raise ValueError("target output size differs from the eager output")
    observed = observed.reshape_as(golden)

mae = (observed.float() - golden.float()).abs().mean().item()
match = mae <= ATOL
print(f"[validate] mae={mae:.6e}")
print(f"[validate] match={match}")
if not match:
    raise SystemExit(1)
```

Compile ResNet50 with the default FP32 Gemmini configuration:

```bash
cd /path/to/pytorch-chipyard
conda activate pytorch-chipyard
# Select the external checkout for the later ELF build and FPGA steps.
export CHIPYARD_DIR=/path/to/chipyard
source scripts/env.sh
# Compile model artifacts without launching the standalone simulator.
export CHIPYARD_SIM_VERILATOR_PATH=

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

To customize the workload, `--batch-size=N` changes the input batch and
`--seed=N` controls the seed. Recompile into a separate artifact directory
after changing the input shape or target. Keep the generated `input.bin`,
`weights.bin`, kernel objects, and runner from the same compile, and use the
same model, weights, and seed for eager validation.

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

The generated executable expects input, weight, and output file paths. This
command runs inside the target RISC-V Linux guest:

```bash
./model.elf input.bin weights.bin output.bin
```

### FireSim and FireMarshal Environment Variables

Configure the [FPGA host prerequisites](getting-started.md#prerequisites)
once before packaging or running a workload. These variables locate existing
host tools, configurations, and writable output directories; they are separate
from the compiler target variables above.

| Variable | Role and default from `scripts/env.sh` |
| --- | --- |
| `CHIPYARD_DIR` | External hardware checkout; set before sourcing `env.sh`. |
| `CHIPYARD_ENV_PATH` | `$CHIPYARD_DIR/env.sh`, used for the RISC-V build environment. |
| `PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR` / `PYTORCH_CHIPYARD_RISCV_GXX` | Optional Linux cross-compiler override for model ELF builds. |
| `FIREMARSHAL_DIR` | `$CHIPYARD_DIR/software/firemarshal`. |
| `FIREMARSHAL_CONFIG_PATH` | `$FIREMARSHAL_DIR/marshal-config.yaml`. |
| `MARSHAL_IMAGE_DIR` / `FIREMARSHAL_IMAGE_DIR` | FireMarshal image root and its `firechip` subdirectory; keep producer and consumer paths consistent. |
| `PYTORCH_CHIPYARD_WORKLOAD_DIR` | Workload JSONs and overlays; defaults to `$FIREMARSHAL_DIR/custom_application/pytorch-chipyard-workloads`. |
| `FIRESIM_DIR` / `FIRESIM_DEPLOY_DIR` | `$CHIPYARD_DIR/sims/firesim` and its `deploy` directory. |
| `FIRESIM_HWDB_PATH` / `FIRESIM_BUILD_RECIPES_PATH` | Hardware database and build-recipe YAMLs in the deploy directory. |
| `FIRESIM_WORKLOAD_DIR` | Installed workload JSONs in `$FIRESIM_DEPLOY_DIR/workloads`. |
| `MARSHAL_FIRESIM_DIR` | FireSim root used by FireMarshal's install target; derived from the selected deploy directory. |
| `PYTORCH_CHIPYARD_FPGA_DB` | FPGA device database; defaults to `/opt/firesim-db.json`. |
| `FIRESIM_RUNS_DIR` / `PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR` | Simulation work directory and generated runtime YAML directory under the repository. |
| `PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_HOST` / `PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_SPEC` | Run host and host specification; defaults are `localhost` and `one_fpgas_spec`. |
| `PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR` | Raw FireSim results; defaults to `$FIRESIM_DEPLOY_DIR/results-workload`. |

Export host-path overrides before sourcing `scripts/env.sh` when the local
layout differs. For example, in the compiler environment:

```bash
cd "$WORKSPACE"
export CHIPYARD_DIR=/path/to/chipyard
source scripts/env.sh
```

### FireMarshal Packaging and FireSim Execution

Package the Gemmini artifact compiled above as `resnet50-gemmini-4core` and
select a four-core FP32 Gemmini bitstream with the same hardware parameters.
To use RVV instead, compile a separate RVV artifact, change the workload name,
and select a matching Saturn bitstream.

The FireMarshal workload directory should contain the workload JSON, overlay
directory, and guest-side runner script. The structure is roughly:

```bash
export WORKLOAD=resnet50-gemmini-4core
export GUEST_DIR="$PYTORCH_CHIPYARD_WORKLOAD_DIR/overlay-$WORKLOAD/root/cnn/$WORKLOAD"

mkdir -p "$GUEST_DIR"
cp "$PYTORCH_CHIPYARD_DUMP_PATH/model.elf" "$GUEST_DIR/model.elf"
cp "$PYTORCH_CHIPYARD_DUMP_PATH/input.bin" "$GUEST_DIR/input.bin"
cp "$PYTORCH_CHIPYARD_DUMP_PATH/weights.bin" "$GUEST_DIR/weights.bin"
```

```text
$PYTORCH_CHIPYARD_WORKLOAD_DIR/
  resnet50-gemmini-4core.json
  overlay-resnet50-gemmini-4core/
    root/
      cnn/
        run-resnet50.sh
        resnet50-gemmini-4core/
          model.elf
          input.bin
          weights.bin
```

Save the following as
`$PYTORCH_CHIPYARD_WORKLOAD_DIR/overlay-$WORKLOAD/root/cnn/run-resnet50.sh`.
It is executed inside the guest Linux image. Both the Saturn/RVV
and Gemmini paths may involve asynchronous target-side work, so the runner sets
OpenMP scheduling and affinity variables explicitly:

```bash
#!/bin/bash
set -u
set -o pipefail

ulimit -s unlimited
cd /root/cnn/resnet50-gemmini-4core

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
echo "[runner] start resnet50-gemmini-4core"
./model.elf input.bin weights.bin output.bin
rc=$?
echo "[runner] model_ret=${rc}"
sync || true
sleep 5
exit "${rc}"
```

Create the FireMarshal workload JSON under
`$PYTORCH_CHIPYARD_WORKLOAD_DIR/resnet50-gemmini-4core.json`.
It describes the base Linux image, overlay, guest command, and collected files:

```json
{
  "name": "resnet50-gemmini-4core",
  "workdir": ".",
  "base": "br-base.json",
  "rootfs-size": "8GiB",
  "overlay": "overlay-resnet50-gemmini-4core",
  "command": "bash /root/cnn/run-resnet50.sh",
  "outputs": [
    "/root/cnn/resnet50-gemmini-4core/output.bin",
    "/root/cnn/resnet50-gemmini-4core/run.log",
    "/root/cnn/resnet50-gemmini-4core/model.log",
    "/root/cnn/resnet50-gemmini-4core/autotune.log"
  ]
}
```

Then build and install the FireMarshal image:

```bash
source "$CHIPYARD_ENV_PATH"
cd "$FIREMARSHAL_DIR"
./marshal --workdir "$PYTORCH_CHIPYARD_WORKLOAD_DIR" \
  build "$PYTORCH_CHIPYARD_WORKLOAD_DIR/resnet50-gemmini-4core.json"
./marshal --workdir "$PYTORCH_CHIPYARD_WORKLOAD_DIR" \
  install "$PYTORCH_CHIPYARD_WORKLOAD_DIR/resnet50-gemmini-4core.json"
```

This produces the FireMarshal image files referenced by the FireSim workload:

```text
$FIREMARSHAL_IMAGE_DIR/resnet50-gemmini-4core/resnet50-gemmini-4core-bin
$FIREMARSHAL_IMAGE_DIR/resnet50-gemmini-4core/resnet50-gemmini-4core.img
$FIRESIM_WORKLOAD_DIR/resnet50-gemmini-4core.json
```

The installed JSON is a **FireSim** workload description. It uses
`benchmark_name`, `common_bootbinary`, `common_rootfs`, and `common_outputs`.
FireMarshal's `install` step creates it from the **FireMarshal** input above.
If editing the installed JSON, resolve relative boot-binary and rootfs paths
from `$FIRESIM_WORKLOAD_DIR` to the generated image files.

To run the workload through FireSim, prepare a runtime YAML and invoke the
FireSim manager commands directly. Copy the existing host runtime configuration:

```bash
mkdir -p "$PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR"
export RUNTIME="$PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR/resnet50-gemmini-4core.yaml"
cp "$FIRESIM_DEPLOY_DIR/config_runtime.yaml" "$RUNTIME"
```

Edit these fields in `$RUNTIME`, preserving the other fields from the host's
configuration. Replace the placeholders with the matching Gemmini hardware
entry and configured run-farm paths:

```yaml
target_config:
    default_hw_config: <FireSim hardware config name for the target bitstream>

workload:
    workload_name: resnet50-gemmini-4core.json
    terminate_on_completion: yes
    suffix_tag: resnet50-gemmini-4core

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

Use the job directory reported by this `runworkload` invocation as
`RESULT_DIR`; it must contain this run's output and logs.

```bash
export RESULT_DIR=/path/to/this-run/job-output-directory
cp "$RESULT_DIR/output.bin" "$PYTORCH_CHIPYARD_DUMP_PATH/output.bin"
cp "$RESULT_DIR/model.log" "$PYTORCH_CHIPYARD_DUMP_PATH/model.log"
cp "$RESULT_DIR/autotune.log" "$PYTORCH_CHIPYARD_DUMP_PATH/autotune.log"

cd "$WORKSPACE"
conda activate pytorch-chipyard
python examples/ResNet50.py --validate
```

Successful numerical validation prints `[validate] match=True` and exits with
status zero. Inspect `model.log` for model and per-kernel cycle measurements
and `autotune.log` for target-side candidate selection. The AE smoke test and
Stage 2 completion markers check execution and collected artifacts; this
explicit validation step compares numerical results with eager PyTorch.
