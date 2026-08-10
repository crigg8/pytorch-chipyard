# Section 3. Adding New Hardware

```{image} figures/custom-hw-extension.png
:alt: Workflow for adding a custom hardware library and registering custom operations
:align: center
:width: 100%
```

PyTorch-Chipyard provides two static-library paths for adding hardware
operations.

- `libpytorch_chipyard.a` replaces an Aten operator with a custom call in the
  runner.
- `libtriton_chipyard.a` replaces a bufferized Linalg operation inside a Triton
  kernel with an external function call.


## Common environment setup

First, specify the absolute paths to the PyTorch-Chipyard repository, the
Chipyard checkout, and the RISC-V compiler. `RISCV_GXX` and `RISCV_AR` must
point to the executables themselves.

```bash
cd /path/to/pytorch-chipyard
conda activate pytorch-chipyard

export PYTORCH_CHIPYARD_ROOT="$PWD"
export CHIPYARD_DIR=/path/to/chipyard
export RISCV_GXX=/path/to/riscv-toolchain/bin/riscv64-unknown-linux-gnu-g++
export RISCV_AR=/path/to/riscv-toolchain/bin/riscv64-unknown-linux-gnu-ar
export RISCV_NM=/path/to/riscv-toolchain/bin/riscv64-unknown-linux-gnu-nm

export LLVM_PROJECT_PATH="$PYTORCH_CHIPYARD_ROOT/llvm-project"
export GEMMINI_TEST_ROOT="$CHIPYARD_DIR/generators/gemmini/software/gemmini-rocc-tests"
export GEMMINI_PARAMS_H="$CHIPYARD_DIR/sims/firesim/deploy/results-build/gemmini_params.h"
export CHIPYARD_ENV_PATH="$CHIPYARD_DIR/env.sh"
export BUDDY_BINARY_DIR="$PYTORCH_CHIPYARD_ROOT/buddy-mlir/build/bin"
export TRITON_CHIPYARD_OPT_PATH="$PYTORCH_CHIPYARD_ROOT/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_chipyard/tools/triton-chipyard-opt/triton-chipyard-opt"

export PATH="$(dirname "$RISCV_GXX"):$PATH"
```

Verify that all required files are available. These commands must finish
without producing any output before you proceed.

```bash
test -d "$PYTORCH_CHIPYARD_ROOT"
test -d "$LLVM_PROJECT_PATH/mlir/include"
test -f "$GEMMINI_TEST_ROOT/include/gemmini.h"
test -f "$GEMMINI_PARAMS_H"
test -x "$RISCV_GXX"
test -x "$RISCV_AR"
test -x "$RISCV_NM"
test -x "$TRITON_CHIPYARD_OPT_PATH"
```

This tutorial uses a Gemmini bitstream with FP32, `DIM=8`, `BANK_ROWS=2048`,
and `ACC_ROWS=2048`. Configure PyTorch-Chipyard and Triton-Chipyard with the
same values.

```bash
export CHIPYARD_SIM_VERILATOR_PATH=

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

export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1
export TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS=1
export TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER=0
export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=0
```

`gemmini.h` must provide the hardware API for the configuration above.
Registering a GEMV or matmul does not itself require modifying `gemmini.h`.
Target-specific values such as `DIM`, element type, and scratchpad size are
injected with the compiler option `-include "$GEMMINI_PARAMS_H"`.

## `libpytorch_chipyard.a`: Replacing an Aten operator

`libpytorch_chipyard.a` serves a role similar to a PyTorch custom operator
library. When Inductor lowers the FX graph, it lowers an Aten node to a custom
kernel instead of a Triton kernel. This approach is effective when only a
small number of operators need to run on the new hardware.

The following tutorial lowers `aten.mv.default` to a custom kernel. The
overall flow is to write a hardware kernel using the `gemmini.h` API provided
by Gemmini, put its C symbol in `libpytorch_chipyard.a`, and then associate the
Aten operator with that symbol in Python. This example uses an FP32, `DIM=8`
Gemmini bitstream.

The registered `aten.mv.default` node becomes a runner custom call named
`pytorch_chipyard_mv_f32`. The custom call runs instead of the Triton kernel
that would otherwise be generated.

### 1. Create the working directories

```bash
export CUSTOM_MV_DIR="$PYTORCH_CHIPYARD_ROOT/tutorial/custom-mv"
export CUSTOM_MV_ARTIFACT="$PYTORCH_CHIPYARD_ROOT/tutorial-artifacts/custom-mv"

mkdir -p "$CUSTOM_MV_DIR/build"
mkdir -p "$CUSTOM_MV_ARTIFACT"
```

### 2. Write the Gemmini GEMV kernel

Copy the following code block to `$CUSTOM_MV_DIR/gemmini_mv.cpp`. The
`extern "C"` function name must match the `symbol` used later in the Python
registration.

```{literalinclude} ../examples/custom-mv/gemmini_mv.cpp
:language: cpp
:caption: tutorial/custom-mv/gemmini_mv.cpp
```

Each Tensor argument is passed as a `(rank, descriptor)` pair in schema order,
followed by the output's `(rank, descriptor)` pair. Casting a descriptor to
`StridedMemRefType<T, rank>` provides access to `data`, `offset`, `sizes`, and
`strides`. This custom call does not use a Triton grid; the generated runner
invokes it once.

### 3. Compile the RISC-V object

The following command first includes the target-generated `gemmini_params.h`
and then compiles `gemmini_mv.cpp` into a RISC-V object.

```bash
"$RISCV_GXX" \
  -include "$GEMMINI_PARAMS_H" \
  -I"$GEMMINI_TEST_ROOT/include" \
  -I"$GEMMINI_TEST_ROOT" \
  -I"$LLVM_PROJECT_PATH/mlir/include" \
  -I"$LLVM_PROJECT_PATH/llvm/include" \
  -march=rv64imafdc \
  -mabi=lp64d \
  -O2 \
  -std=gnu++17 \
  -c "$CUSTOM_MV_DIR/gemmini_mv.cpp" \
  -o "$CUSTOM_MV_DIR/build/gemmini_mv.o"
```

Verify that the object was created.

```bash
test -f "$CUSTOM_MV_DIR/build/gemmini_mv.o"
```

### 4. Create `libpytorch_chipyard.a`

```bash
"$RISCV_AR" rcs \
  "$CUSTOM_MV_DIR/libpytorch_chipyard.a" \
  "$CUSTOM_MV_DIR/build/gemmini_mv.o"
```

Verify that the archive contains the C symbol.

```bash
"$RISCV_AR" t "$CUSTOM_MV_DIR/libpytorch_chipyard.a"
"$RISCV_NM" \
  "$CUSTOM_MV_DIR/libpytorch_chipyard.a" \
  | grep 'pytorch_chipyard_mv_f32'
```

The second command must display `pytorch_chipyard_mv_f32`. If your toolchain's
`nm` is installed elsewhere, replace the path with its absolute path.

### 5. Write the Python model with the Aten mapping

Save the complete code below as `$CUSTOM_MV_DIR/compile_custom_mv.py`. This
file contains the custom-op definition, fake implementation, support
predicate, Aten mapping, test model, and artifact generation.

```python
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
import triton
from triton.backends.triton_chipyard.driver import ChipyardDriver
from torch._inductor.codegen.chipyard.custom_ops import (
    register_chipyard_custom_op,
)


ARTIFACT_DIR = Path(os.environ["PYTORCH_CHIPYARD_DUMP_PATH"]).resolve()
DTYPE = torch.float32
M = 32
K = 32


@torch.library.custom_op("pytorch_chipyard_tutorial::mv", mutates_args=())
def custom_mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.mv(matrix, vector)


@custom_mv.register_fake
def _(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    torch._check(matrix.dim() == 2)
    torch._check(vector.dim() == 1)
    torch._check(matrix.shape[1] == vector.shape[0])
    return matrix.new_empty((matrix.shape[0],))


def supports_custom_mv(matrix: torch.Tensor, vector: torch.Tensor) -> bool:
    return bool(
        matrix.device.type == "cpu"
        and vector.device.type == "cpu"
        and matrix.dtype == DTYPE
        and vector.dtype == DTYPE
        and matrix.dim() == 2
        and vector.dim() == 1
        and matrix.is_contiguous()
        and vector.is_contiguous()
        and matrix.shape[1] == vector.shape[0]
    )


register_chipyard_custom_op(
    torch.ops.aten.mv.default,
    custom_mv,
    symbol="pytorch_chipyard_mv_f32",
    supports=supports_custom_mv,
)


class AtenMvModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(0)
        weight = torch.randn(M, K, generator=generator, dtype=DTYPE)
        self.register_buffer("weight", weight.contiguous())

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return torch.mv(self.weight, vector)


def import_artifact_util():
    util_path = ARTIFACT_DIR / "util.py"
    spec = importlib.util.spec_from_file_location("custom_mv_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CHIPYARD_DUMP_PATH"] = str(ARTIFACT_DIR)
    os.environ["TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER"] = "1"

    triton.runtime.driver.set_active(ChipyardDriver())
    inductor_config.cpu_backend = "triton_chipyard"

    generator = torch.Generator(device="cpu").manual_seed(1)
    vector = torch.randn(K, generator=generator, dtype=DTYPE).contiguous()
    model = AtenMvModule().eval()
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)

    with torch.inference_mode():
        compiled_model(vector)

    util = import_artifact_util()
    input_path = util.write_inputs_bin(vector)
    print(f"artifact: {ARTIFACT_DIR}")
    print(f"input: {input_path}")


if __name__ == "__main__":
    main()
```

The `custom_op` schema must have the same number and types of arguments as the
original Aten overload. Inputs may currently be `Tensor`, `bool`, `int`, or
`float`, and they must not be mutated or aliased. The output must be a single
Tensor that does not alias an input. Inputs for which `supports` returns
`False` follow the existing decomposition path instead of the custom kernel.
Therefore, keep the Python `supports` predicate consistent with the C++
kernel's dtype, rank, shape, and stride constraints.

### 6. Generate the model artifact

The registered path is active only when the CPU backend is `triton_chipyard`
and `PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY` points to an existing `.a` file. Set
the archive path and the artifact output path, and then run the Python file
directly.

```bash
export PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY="$CUSTOM_MV_DIR/libpytorch_chipyard.a"
export PYTORCH_CHIPYARD_DUMP_PATH="$CUSTOM_MV_ARTIFACT"
export TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/tutorial-custom-mv

python "$CUSTOM_MV_DIR/compile_custom_mv.py"
```

Verify that the generated plan contains a runner custom call.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

artifact = Path(os.environ["PYTORCH_CHIPYARD_DUMP_PATH"])
spec = json.loads((artifact / "model_spec.json").read_text())
assert [step["kind"] for step in spec["steps"]] == ["custom_call"]
assert spec["steps"][0]["symbol"] == "pytorch_chipyard_mv_f32"
print("custom call plan OK")
PY
```

Instead of lowering a supported `aten.mv.default` node to a Triton kernel,
Inductor records it as a `pytorch_chipyard_mv_f32` custom call. The generated
`model_spec.json` contains a `custom_call` step, and
`libpytorch_chipyard.a` is linked with the runner when the final model ELF is
built.

`PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY` is not read when `build.sh` runs. Inductor
reads this path while running the Python compilation above and records the
archive path and SHA-256 digest in `model_spec.json` under
`custom_op_libraries`. Because the common setup sets
`TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS=1`, Inductor copies the archive
into the artifact and adds its path to the generated `build.sh` file's
`CUSTOM_OP_LIBRARIES` array. Once the artifact has been generated, `build.sh`
therefore links the archive automatically without requiring the same
environment variable to be set again.

### 7. Link the model ELF

This single-custom-call model has no Triton kernel object. First, verify that
the custom archive appears in the `build.sh` generated by Inductor.

```bash
grep -nE \
  '^(KERNEL_OBJECTS|TRITON_CUSTOM_LIBRARIES|CUSTOM_OP_LIBRARIES)' \
  "$CUSTOM_MV_ARTIFACT/build.sh"
```

The output must show the staged `libpytorch_chipyard.a` path under
`CUSTOM_OP_LIBRARIES`. Build the ELF by running the generated script as is.
The script combines the runner, custom archive, MLIR runtime sources, and
required compiler and linker options in the correct order.

```bash
CHIPYARD_OMP_NUM_THREADS=1 bash "$CUSTOM_MV_ARTIFACT/build.sh"
```

```bash
test -x "$CUSTOM_MV_ARTIFACT/model.elf"
```

## `libtriton_chipyard.a`: Replacing a Linalg operation

`libtriton_chipyard.a` does not replace an Aten operator directly. It preserves
the existing TorchInductor/Triton lowering path, then replaces a specific
Linalg operation with a call to a C function in the static library after
bufferizing the TTShared IR. A single rule can be reused when multiple Aten
operators lower to the same Linalg operation.

The following tutorial replaces `linalg.matmul` with a Gemmini kernel inside
the normal Triton path for `aten.mm`.

```text
aten.mm
  -> TorchInductor/Triton kernel
  -> bufferized linalg.matmul
  -> func.call @triton_chipyard_matmul_f32
  -> Triton kernel object + libtriton_chipyard.a
```

### 1. Create the working directories

```bash
export CUSTOM_LINALG_DIR="$PYTORCH_CHIPYARD_ROOT/tutorial/custom-linalg-matmul"
export CUSTOM_LINALG_ARTIFACT="$PYTORCH_CHIPYARD_ROOT/tutorial-artifacts/custom-linalg-matmul"

mkdir -p "$CUSTOM_LINALG_DIR/build"
mkdir -p "$CUSTOM_LINALG_ARTIFACT"
```

### 2. Write the Gemmini GEMV/matmul kernel

Copy the following code block to
`$CUSTOM_LINALG_DIR/gemmini_matmul.cpp`. The archive exports C symbols for
both GEMV and general matmul.

```{literalinclude} ../examples/custom-linalg-matmul/gemmini_matmul.cpp
:language: cpp
:caption: tutorial/custom-linalg-matmul/gemmini_matmul.cpp
```

The bufferized Linalg operation's DPS inputs are passed first, followed by its
init/output operand. Each ranked memref is cast to an unranked memref
immediately before the external call, so every memref becomes a `(rank,
descriptor)` pair in the C ABI. The C++ kernel must accept
`StridedMemRefType<T, rank>` descriptors, validate their shapes and strides,
and then call the hardware API. This example uses Gemmini's transpose flag
when the RHS has a non-row-major stride layout. Because the `tl.dot`
accumulator is initialized to zero, the bias/D input is passed as `nullptr`.

### 3. Compile the RISC-V object

```bash
"$RISCV_GXX" \
  -include "$GEMMINI_PARAMS_H" \
  -I"$GEMMINI_TEST_ROOT/include" \
  -I"$GEMMINI_TEST_ROOT" \
  -I"$LLVM_PROJECT_PATH/mlir/include" \
  -I"$LLVM_PROJECT_PATH/llvm/include" \
  -march=rv64imafdc \
  -mabi=lp64d \
  -O2 \
  -std=gnu++17 \
  -c "$CUSTOM_LINALG_DIR/gemmini_matmul.cpp" \
  -o "$CUSTOM_LINALG_DIR/build/gemmini_matmul.o"
```

```bash
test -f "$CUSTOM_LINALG_DIR/build/gemmini_matmul.o"
```

### 4. Create `libtriton_chipyard.a`

```bash
"$RISCV_AR" rcs \
  "$CUSTOM_LINALG_DIR/libtriton_chipyard.a" \
  "$CUSTOM_LINALG_DIR/build/gemmini_matmul.o"
```

```bash
"$RISCV_NM" \
  "$CUSTOM_LINALG_DIR/libtriton_chipyard.a" \
  | grep -E 'triton_chipyard_(gemv|matmul)_f32'
```

The output must contain both `triton_chipyard_gemv_f32` and
`triton_chipyard_matmul_f32`.

The archive's ISA, ABI, element type, and Gemmini parameters must match the
final model runner and bitstream. The C++ `static_assert` declarations verify
these conditions when the archive is compiled.

### 5. Write the Python model with the Linalg rule

Save the complete code below as
`$CUSTOM_LINALG_DIR/compile_custom_linalg.py`. The decorated empty Python
functions are never executed; each function name becomes a C symbol in the
archive. To use a different C symbol, specify the decorator's `symbol=`
argument.

```python
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
import triton
from triton.backends.triton_chipyard.driver import ChipyardDriver
from triton.backends.triton_chipyard.extern_calls import (
    MemRefPattern,
    register_triton_chipyard_extern_call,
)


ARTIFACT_DIR = Path(os.environ["PYTORCH_CHIPYARD_DUMP_PATH"]).resolve()
CUSTOM_LIBRARY_PATH = Path(
    os.environ["TRITON_CHIPYARD_EXTERN_CALL_LIBRARY"]
).resolve()
DTYPE = torch.float32
M = 32
K = 32
N = 32


@register_triton_chipyard_extern_call(
    library=CUSTOM_LIBRARY_PATH,
    target_op="linalg.matmul",
    operands=(
        MemRefPattern(
            element_type="f32",
            physical_ranks=(1, 2),
            logical_rank=1,
            unit_dims="squeeze_static",
            canonicalize_to_logical_rank=True,
        ),
        MemRefPattern(
            element_type="f32",
            physical_ranks=(2,),
            logical_rank=2,
        ),
        MemRefPattern(
            element_type="f32",
            physical_ranks=(1, 2),
            logical_rank=1,
            unit_dims="squeeze_static",
            canonicalize_to_logical_rank=True,
        ),
    ),
)
def triton_chipyard_gemv_f32() -> None:
    pass


@register_triton_chipyard_extern_call(
    library=CUSTOM_LIBRARY_PATH,
    target_op="linalg.matmul",
    operands=(
        MemRefPattern(element_type="f32", physical_ranks=(2,), logical_rank=2),
        MemRefPattern(element_type="f32", physical_ranks=(2,), logical_rank=2),
        MemRefPattern(element_type="f32", physical_ranks=(2,), logical_rank=2),
    ),
)
def triton_chipyard_matmul_f32() -> None:
    pass


class AtenMmModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(7)
        weight = torch.randn(K, N, generator=generator, dtype=DTYPE)
        self.register_buffer("weight", weight.contiguous())

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.mm(matrix, self.weight)


def import_artifact_util():
    util_path = ARTIFACT_DIR / "util.py"
    spec = importlib.util.spec_from_file_location("custom_linalg_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CHIPYARD_DUMP_PATH"] = str(ARTIFACT_DIR)
    os.environ["TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER"] = "1"

    triton.runtime.driver.set_active(ChipyardDriver())
    inductor_config.cpu_backend = "triton_chipyard"
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm_backends = "TRITON"

    generator = torch.Generator(device="cpu").manual_seed(8)
    matrix = torch.randn(M, K, generator=generator, dtype=DTYPE).contiguous()
    model = AtenMmModule().eval()
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)

    with torch.inference_mode():
        compiled_model(matrix)

    util = import_artifact_util()
    input_path = util.write_inputs_bin(matrix)
    print(f"artifact: {ARTIFACT_DIR}")
    print(f"input: {input_path}")


if __name__ == "__main__":
    main()
```

The main `MemRefPattern` fields have the following meanings.

| Field | Meaning |
| --- | --- |
| `element_type` | The MLIR element type to match. This example accepts only `f32`. |
| `physical_ranks` | The physical memref ranks allowed after bufferization. |
| `logical_rank` | The rank received by the custom C ABI. |
| `unit_dims` | `preserve` retains the rank, while `squeeze_static` removes static unit dimensions for matching. |
| `canonicalize_to_logical_rank` | Whether to rank-reduce the memref to `logical_rank` after matching. |

Rule registration order determines match priority. Therefore, register the
GEMV rule with logical ranks `(1, 2, 1)` before the general matmul rule with
ranks `(2, 2, 2)`. All rules must be registered before the first
`torch.compile()` call.

The first rule in this example matches `(vector, matrix, output)` with logical
ranks `(1, 2, 1)`, while the second rule matches `(lhs, rhs, output)` with rank
2 for every operand. `target_op` must name an exact named Linalg operation such
as `linalg.matmul`. `linalg.generic`, which requires additional indexing-map
and body constraints, is not supported.

The registry is frozen when the first `torch.compile()` starts, so all rules
must be registered beforehand. Decorated Python functions are never executed;
each function name becomes a C symbol in the archive. `library` must point to
an existing `.a` file at registration time, and one registry may refer to only
one archive. The registered rules and archive SHA-256 digest are also included
in the Triton cache key. The existing
`TRITON_CHIPYARD_LINALG_TO_FUNC_CONFIG` JSON mechanism remains available as a
compatibility path, but this tutorial uses the decorator registry.

### 6. Generate the model artifact

The model's `aten.mm` is compiled through Inductor/Triton as usual. Restrict
the max-autotune GEMM backend to `TRITON` so that the Gemmini backend selects a
Triton GEMM. `triton-chipyard-opt` must already have been built with the
`linalg-to-function-call` pass. If you have just added this pass or changed its
source, rebuild the compiler stack and the Stage 1 image.

```bash
docker build \
  -f "$PYTORCH_CHIPYARD_ROOT/docker/stage1.Dockerfile" \
  -t pytorch-chipyard:stage1 \
  "$PYTORCH_CHIPYARD_ROOT"
```

```bash
export TRITON_CHIPYARD_EXTERN_CALL_LIBRARY="$CUSTOM_LINALG_DIR/libtriton_chipyard.a"
export PYTORCH_CHIPYARD_DUMP_PATH="$CUSTOM_LINALG_ARTIFACT"
export TRITON_CACHE_DIR=/tmp/triton-chipyard-cache/tutorial-custom-linalg

export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON

python "$CUSTOM_LINALG_DIR/compile_custom_linalg.py"
```

Inductor records the Triton custom-library path and SHA-256 digest in
`model_spec.json` under `triton_custom_libraries`. If kernel-object metadata
becomes available late on a path that uses asynchronous compilation,
post-compile finalization stages `log-ttshared.o` and the archive inside the
artifact and updates the `TRITON_CUSTOM_LIBRARIES` link list. The generated
Python wrapper performs this update automatically immediately after
`async_compile.wait()`, so no separate finalization command is required.

Here, too, the decorator reads `TRITON_CHIPYARD_EXTERN_CALL_LIBRARY` during
Python compilation. The registered archive information flows through the
extern-call registry into `model_spec.json` and `build.sh`. `build.sh` does not
read this environment variable again or require the archive path to be passed
as a direct argument.

Verify that the final TTShared IR contains the external call and no longer
contains the replaced `linalg.matmul`.

```bash
grep -n 'triton_chipyard_matmul_f32' "$CUSTOM_LINALG_ARTIFACT/ttshared.mlir"

if grep -q 'linalg.matmul' "$CUSTOM_LINALG_ARTIFACT/ttshared.mlir"; then
  echo 'unconverted linalg.matmul remains' >&2
  exit 1
fi
```

The model plan for this path remains a normal Triton launch rather than a
runner custom call.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

artifact = Path(os.environ["PYTORCH_CHIPYARD_DUMP_PATH"])
spec = json.loads((artifact / "model_spec.json").read_text())
assert any(step["kind"] == "launch" for step in spec["steps"])
assert all(step["kind"] != "custom_call" for step in spec["steps"])
print("Triton launch plan OK")
PY
```

Unlike the `libpytorch_chipyard.a` path, the external call resides inside the
Triton kernel object. Therefore, the `model_spec.json` step remains a `launch`
rather than becoming a `custom_call`.

### 7. Link the model ELF

Because the external call is inside the Triton kernel object, the archive must
appear after the kernel object so that the linker can resolve the C symbol.
The generated `build.sh` links `KERNEL_OBJECTS`,
`TRITON_CUSTOM_LIBRARIES`, and `CUSTOM_OP_LIBRARIES` in that order. The object
generated by this single-kernel tutorial is `log-ttshared.o`; verify that both
paths appear in the script.

```bash
test -f "$CUSTOM_LINALG_ARTIFACT/log-ttshared.o"
```

```bash
grep -nE \
  '^(KERNEL_OBJECTS|TRITON_CUSTOM_LIBRARIES|CUSTOM_OP_LIBRARIES)' \
  "$CUSTOM_LINALG_ARTIFACT/build.sh"
```

The output must show the staged `log-ttshared.o` path under `KERNEL_OBJECTS`
and the staged `libtriton_chipyard.a` path under
`TRITON_CUSTOM_LIBRARIES`. Run the generated script instead of recreating the
linker command manually.

```bash
CHIPYARD_OMP_NUM_THREADS=1 bash "$CUSTOM_LINALG_ARTIFACT/build.sh"
```

```bash
test -x "$CUSTOM_LINALG_ARTIFACT/model.elf"
```
