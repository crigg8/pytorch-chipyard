#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

from torch._inductor.codegen.chipyard.custom_ops import (
    register_chipyard_custom_op,
)


TASK_NAME = "custom-mv"
MODEL_NAME = "aten_mv_32x32"
M = 32
K = 32
DTYPE = torch.float32
SEED = 0
ATOL = 1e-3
RTOL = 1e-3
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR.parent / "artifact-custom-mv" / "gemmini"


@torch.library.custom_op("pytorch_chipyard_test::mv", mutates_args=())
def custom_mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    # The normal Python wrapper executes once while producing Stage 1 artifacts.
    # Keep a real CPU semantic even though runner.cpp calls the archive symbol.
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
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        weight = torch.randn(M, K, generator=generator, dtype=DTYPE) * 0.125
        self.register_buffer("weight", weight.contiguous())

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return torch.mv(self.weight, vector)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile or validate the libpytorch_chipyard aten.mv example."
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


def build_model() -> torch.nn.Module:
    return AtenMvModule().eval()


def make_input() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    return (torch.randn(K, generator=generator, dtype=DTYPE) * 0.125).contiguous()


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
    vector = make_input()

    started_at = time.perf_counter()
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        _ = compiled_model(vector)
    compile_time = time.perf_counter() - started_at

    util = import_artifact_util(path)
    input_path = util.write_inputs_bin(vector)
    print(f"[compile] model={MODEL_NAME} seconds={compile_time:.3f}")
    print(f"[artifact] directory={path}")
    print(f"[artifact] input_bin={input_path}")


def run_validate(output_bin: Path | None) -> None:
    path = artifact_dir()
    util = import_artifact_util(path)
    vector = util.read_inputs_bin(path / "input.bin")
    output_path = (output_bin or path / "output.bin").resolve()
    observed = util.read_outputs_bin(output_path)
    if not isinstance(vector, torch.Tensor) or not isinstance(observed, torch.Tensor):
        raise TypeError("custom-mv expects one Tensor input and one Tensor output")

    with torch.inference_mode():
        expected = build_model()(vector)
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
