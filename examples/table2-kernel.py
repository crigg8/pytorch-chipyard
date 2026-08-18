#!/usr/bin/env python3
"""Compile or validate one model-derived GEMM for the RTL-turnaround table."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch._inductor.config as inductor_config


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SPEC = REPO_ROOT / "benchmarks" / "table2-kernels.json"
DEFAULT_ARTIFACT_ROOT = SCRIPT_DIR / "artifact-table2-kernels"
DTYPE = torch.float32
SEED = 2027
VALIDATE_ATOL = 1e-3


class SingleMatmul(torch.nn.Module):
    def __init__(self, k: int, n: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weight = torch.randn(k, n, generator=generator, dtype=DTYPE) * 0.01
        self.register_buffer("weight", weight.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    return parser.parse_args()


def load_kernel(path: Path, kernel_id: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    matches = [entry for entry in data.get("kernels", []) if entry.get("id") == kernel_id]
    if len(matches) != 1:
        choices = ", ".join(entry.get("id", "?") for entry in data.get("kernels", []))
        raise SystemExit(f"unknown kernel {kernel_id!r}; expected one of: {choices}")
    kernel = matches[0]
    expected_macs = int(kernel["m"]) * int(kernel["n"]) * int(kernel["k"])
    if int(kernel["macs"]) != expected_macs:
        raise SystemExit(f"invalid MAC count for {kernel_id}: {kernel['macs']} != {expected_macs}")
    return kernel


def artifact_dir(kernel_id: str) -> Path:
    configured = os.environ.get("PYTORCH_CHIPYARD_DUMP_PATH")
    return Path(configured).resolve() if configured else (DEFAULT_ARTIFACT_ROOT / kernel_id).resolve()


def configure_backend(kernel_id: str, path: Path) -> None:
    import triton
    from triton.backends.triton_chipyard.driver import ChipyardDriver

    cache_dir = Path(
        os.environ.setdefault(
            "TRITON_CACHE_DIR", f"/tmp/triton-chipyard-cache/table2-{kernel_id}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    triton.runtime.driver.set_active(ChipyardDriver())
    inductor_config.cpu_backend = "triton_chipyard"
    # Keep the normal, bounded TorchInductor candidate set. The separate
    # Section 4.3 experiment is the only one that enables Gemmini max autotune.
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm_backends = "TRITON"
    os.environ["PYTORCH_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TRITON_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER"] = "1"
    os.environ["TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE"] = "0"


def make_model_and_input(kernel: dict[str, Any]) -> tuple[SingleMatmul, torch.Tensor]:
    kernel_index = sum(ord(char) for char in str(kernel["id"]))
    model = SingleMatmul(int(kernel["k"]), int(kernel["n"]), SEED + kernel_index).eval()
    generator = torch.Generator(device="cpu").manual_seed(SEED + kernel_index + 1)
    inputs = torch.randn(
        int(kernel["m"]), int(kernel["k"]), generator=generator, dtype=DTYPE
    ).contiguous()
    return model, inputs


def import_artifact_util(path: Path):
    util_path = path / "util.py"
    if not util_path.is_file():
        raise FileNotFoundError(f"generated util.py not found: {util_path}")
    spec = importlib.util.spec_from_file_location("table2_kernel_artifact_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_metadata(kernel: dict[str, Any], path: Path) -> None:
    print(f"KERNEL_ID={kernel['id']}")
    print(f"KERNEL_LABEL={kernel['label']}")
    print(f"KERNEL_SHAPE={kernel['m']}x{kernel['n']}x{kernel['k']}")
    print(f"KERNEL_MACS={kernel['macs']}")
    print("TARGET_METADATA=fp32,gemmini-dim8,rocket-4core,firesim")
    print(f"ARTIFACT_DIR={path}")


def compile_kernel(kernel: dict[str, Any], path: Path) -> None:
    configure_backend(str(kernel["id"]), path)
    model, inputs = make_model_and_input(kernel)
    print_metadata(kernel, path)
    started = time.perf_counter()
    compiled = torch.compile(model, backend="inductor")
    with torch.inference_mode():
        compiled(inputs)
    compile_wall_s = time.perf_counter() - started
    util = import_artifact_util(path)
    input_path = util.write_inputs_bin(inputs)
    print(f"COMPILE_WALL_S={compile_wall_s:.6f}")
    print(f"INPUT_BIN={input_path}")


def validate_kernel(kernel: dict[str, Any], path: Path) -> None:
    util = import_artifact_util(path)
    inputs = util.read_inputs_bin(path / "input.bin")
    observed = util.read_outputs_bin(path / "output.bin")
    if not isinstance(inputs, torch.Tensor) or not isinstance(observed, torch.Tensor):
        raise TypeError("expected one input tensor and one output tensor")
    model, _ = make_model_and_input(kernel)
    with torch.inference_mode():
        golden = model(inputs)
    if observed.numel() != golden.numel():
        raise SystemExit(f"output size mismatch: {observed.numel()} != {golden.numel()}")
    observed = observed.reshape_as(golden).to(torch.float32)
    max_abs_err = float((observed - golden.to(torch.float32)).abs().max())
    print_metadata(kernel, path)
    print(f"MAX_ABS_ERR={max_abs_err:.6e}")
    print(f"KERNEL_RESULT={'PASS' if max_abs_err <= VALIDATE_ATOL else 'FAIL'}")
    if max_abs_err > VALIDATE_ATOL:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    kernel = load_kernel(args.spec.resolve(), args.kernel)
    path = artifact_dir(str(kernel["id"]))
    if args.compile:
        compile_kernel(kernel, path)
    else:
        validate_kernel(kernel, path)


if __name__ == "__main__":
    main()
