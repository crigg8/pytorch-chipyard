#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from triton.backends.triton_chipyard.extern_calls import (
    MemRefPattern,
    register_triton_chipyard_extern_call,
)

from finalize_artifact import finalize_artifact


TASK_NAME = "custom-linalg-matmul"
MODEL_NAME = "aten_mm_32x32"
M = 32
K = 32
N = 32
DTYPE = torch.float32
SEED = 7
ATOL = 1e-3
RTOL = 1e-3
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = (
    SCRIPT_DIR.parent / "artifact-custom-linalg-matmul" / "gemmini"
)
CUSTOM_LIBRARY_PATH = Path(
    os.environ.get(
        "TRITON_CHIPYARD_EXTERN_CALL_LIBRARY",
        SCRIPT_DIR / "libtriton_chipyard.a",
    )
)


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
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        weight = torch.randn(K, N, generator=generator, dtype=DTYPE) * 0.125
        self.register_buffer("weight", weight.contiguous())

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.mm(matrix, self.weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile or validate the libtriton_chipyard matmul example."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--output-bin",
        type=Path,
        help="FireSim output.bin. Defaults to the artifact directory output.bin.",
    )
    return parser.parse_args()


def artifact_dir() -> Path:
    return Path(
        os.environ.get("PYTORCH_CHIPYARD_DUMP_PATH", DEFAULT_ARTIFACT_DIR)
    ).resolve()


def configure_artifact_env(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ["PYTORCH_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TRITON_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER"] = "1"


def configure_triton_chipyard() -> None:
    import triton
    from triton.backends.triton_chipyard.driver import ChipyardDriver

    cache_dir = Path(
        os.environ.setdefault(
            "TRITON_CACHE_DIR", f"/tmp/triton-chipyard-cache/{TASK_NAME}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    triton.runtime.driver.set_active(ChipyardDriver())
    inductor_config.cpu_backend = "triton_chipyard"
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm_backends = "TRITON"


def build_model() -> torch.nn.Module:
    return AtenMmModule().eval()


def make_input() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    return (torch.randn(M, K, generator=generator, dtype=DTYPE) * 0.125).contiguous()


def import_artifact_util(path: Path):
    util_path = path / "util.py"
    if not util_path.is_file():
        raise FileNotFoundError(f"generated util.py not found: {util_path}")
    spec = importlib.util.spec_from_file_location(f"{TASK_NAME}_artifact_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_compile() -> None:
    path = artifact_dir()
    configure_artifact_env(path)
    configure_triton_chipyard()
    model = build_model()
    matrix = make_input()

    started_at = time.perf_counter()
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        _ = compiled_model(matrix)
    compile_time = time.perf_counter() - started_at

    finalize_artifact(path, library_path=CUSTOM_LIBRARY_PATH)

    util = import_artifact_util(path)
    input_path = util.write_inputs_bin(matrix)
    print(f"[compile] model={MODEL_NAME} seconds={compile_time:.3f}")
    print(f"[artifact] directory={path}")
    print(f"[artifact] input_bin={input_path}")


def run_validate(output_bin: Path | None) -> None:
    path = artifact_dir()
    util = import_artifact_util(path)
    matrix = util.read_inputs_bin(path / "input.bin")
    output_path = (output_bin or path / "output.bin").resolve()
    observed = util.read_outputs_bin(output_path)
    if not isinstance(matrix, torch.Tensor) or not isinstance(observed, torch.Tensor):
        raise TypeError("custom-linalg-matmul expects one Tensor input and output")

    with torch.inference_mode():
        expected = build_model()(matrix)
    if observed.numel() == expected.numel() and observed.shape != expected.shape:
        observed = observed.reshape_as(expected)
    torch.testing.assert_close(observed, expected, atol=ATOL, rtol=RTOL)
    max_abs_err = float((observed - expected).abs().max())
    print(f"[validate] output_bin={output_path}")
    print(f"[validate] max_abs_err={max_abs_err:.6e}")
    print("[validate] match=True")


def main() -> None:
    args = parse_args()
    if args.compile:
        run_compile()
    else:
        run_validate(args.output_bin)


if __name__ == "__main__":
    main()
